"""Fine-tune the SafeSpace model on labelled clips in data/safe and data/unsafe.

Matches the live pipeline exactly:
  * frames go through frame_prep.prepare_for_model(run_person_detector=False)
    (3-value return, HOG off, no person-box cache)
  * 16 consecutive frames at a stride matching ~15 fps (stride = round(fps/15))
  * /255 scaling only (NO Kinetics mean/std) - inference_core.preprocess_clip

Training recipe: start from the currently selected model (load_model:
SAFESPACE_MODEL or the auto-picked file), freeze everything up to layer3, train
layer4 at 1e-5 and fc at 1e-3, sqrt-inverse class weights, a seeded 15%
video-level validation hold-out.  The result is saved as a NEW timestamped file
plus a JSON sidecar; existing weights are never overwritten.  To use it:

    SAFESPACE_MODEL=<printed path> uvicorn main:app

Then re-run video_calibration.py on some footage: a fine-tuned model's score
distribution shifts, so thresholds in the sidecar may need adjusting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from frame_prep import prepare_for_model
from inference_core import CAMERA_TARGET_FPS, CLIP_LENGTH, load_model, preprocess_clip

log = logging.getLogger("safespace.train")

DATA_DIR = "data"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
LR_LAYER4 = 1e-5
LR_FC = 1e-3


def list_videos(root: str) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for cls_name, cls_idx in (("safe", 0), ("unsafe", 1)):
        d = os.path.join(root, cls_name)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith(VIDEO_EXTS):
                paths.append(os.path.join(d, fname))
                labels.append(cls_idx)
    return paths, labels


def split_videos(paths, labels, val_frac: float, seed: int):
    """Stratified, video-level hold-out so clips of one video never straddle the split."""
    rng = random.Random(seed)
    train, val = [], []
    for cls in (0, 1):
        items = [(p, l) for p, l in zip(paths, labels) if l == cls]
        rng.shuffle(items)
        n_val = int(round(len(items) * val_frac)) if len(items) >= 4 else 0
        val += items[:n_val]
        train += items[n_val:]
    return train, val


class ClipDataset(Dataset):
    def __init__(self, items, clips_per_video: int, train: bool, seed: int):
        self.items = items
        self.clips_per_video = max(1, clips_per_video)
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.items) * self.clips_per_video

    def __getitem__(self, idx):
        vid = idx // self.clips_per_video
        k = idx % self.clips_per_video
        path, label = self.items[vid]
        frames = _read_clip(path, k, self.clips_per_video, self.train, self.seed + idx)
        return preprocess_clip(frames), torch.tensor(label, dtype=torch.long)


def _read_clip(path: str, k: int, n_clips: int, train: bool, seed: int) -> List[np.ndarray]:
    """16 consecutive (stride-spaced) frames, prepared with the live frame_prep."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or float(CAMERA_TARGET_FPS)
    if not math.isfinite(fps) or fps <= 1e-3:
        fps = float(CAMERA_TARGET_FPS)
    stride = max(1, int(round(fps / CAMERA_TARGET_FPS)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    need = (CLIP_LENGTH - 1) * stride + 1  # source frames spanned by one clip

    if total <= 0:  # unknown length: decode everything (rare)
        all_frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            all_frames.append(f)
        total = len(all_frames)
    else:
        all_frames = None

    max_start = max(0, total - need - stride)  # leave one pre-frame for motion diff
    if train:
        start = random.Random(seed).randint(0, max_start) if max_start > 0 else 0
    else:
        start = int(round(max_start * (k + 0.5) / n_clips)) if max_start > 0 else 0

    pre = start - stride if start >= stride else None  # extra frame to prime prev_gray
    positions = ([pre] if pre is not None else []) + [start + i * stride for i in range(CLIP_LENGTH)]

    prepared: List[np.ndarray] = []
    prev_gray = None
    last_rgb = None
    if all_frames is None and positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, positions[0])
    cursor = positions[0] if positions else 0
    for pos in positions:
        frame = None
        if all_frames is not None:
            if all_frames:
                frame = all_frames[min(pos, len(all_frames) - 1)]
        else:
            while cursor < pos:
                cap.grab()
                cursor += 1
            ok, frame = cap.read()
            cursor += 1
            if not ok:
                frame = None
        if frame is None:
            if last_rgb is None:
                continue
            prepared.append(last_rgb)
            continue
        rgb, _meta, prev_gray = prepare_for_model(
            frame, prev_gray, person_box_cache=None, run_person_detector=False
        )
        last_rgb = rgb
        prepared.append(rgb)
    cap.release()

    if pre is not None and prepared:
        prepared = prepared[1:]  # drop the priming frame
    if not prepared:
        prepared = [np.zeros((112, 112, 3), np.uint8)]
    while len(prepared) < CLIP_LENGTH:
        prepared.append(prepared[-1])
    return prepared[:CLIP_LENGTH]


def sqrt_class_weights(labels: List[int], clips_per_video: int) -> torch.Tensor:
    counts = np.array([max(1, sum(1 for l in labels if l == c)) for c in (0, 1)], dtype=np.float64)
    counts *= clips_per_video
    w = np.sqrt(counts.sum() / (2.0 * counts))
    return torch.tensor(w, dtype=torch.float32)


def freeze_for_finetune(model: nn.Module) -> List[nn.Module]:
    """Freeze stem..layer3 (and their BN stats). Returns the frozen modules."""
    frozen = [model.stem, model.layer1, model.layer2, model.layer3]
    for m in frozen:
        for p in m.parameters():
            p.requires_grad = False
    for p in list(model.layer4.parameters()) + list(model.fc.parameters()):
        p.requires_grad = True
    return frozen


def set_train_mode(model: nn.Module, frozen: List[nn.Module]) -> None:
    model.train()
    for m in frozen:
        m.eval()  # keep BatchNorm running stats of frozen blocks untouched


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return (loss_sum / max(1, total)), (100.0 * correct / max(1, total))


def unique_output_path(base_dir: str = ".") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"safespace_bullying_model_ft_{stamp}.pth")
    n = 1
    while os.path.exists(path) or os.path.exists(os.path.splitext(path)[0] + ".json"):
        path = os.path.join(base_dir, f"safespace_bullying_model_ft_{stamp}_{n}.pth")
        n += 1
    return path


def train(args: Optional[argparse.Namespace] = None) -> Optional[str]:
    if args is None:
        args = parse_args([])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.data_dir, "safe"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "unsafe"), exist_ok=True)
    paths, labels = list_videos(args.data_dir)
    n_safe, n_unsafe = labels.count(0), labels.count(1)
    log.info("Found %d videos (safe=%d, unsafe=%d)", len(paths), n_safe, n_unsafe)
    if n_safe == 0 or n_unsafe == 0:
        log.error("Need at least one video in BOTH %s/safe and %s/unsafe.", args.data_dir, args.data_dir)
        return None

    train_items, val_items = split_videos(paths, labels, args.val_frac, args.seed)
    log.info("Train videos: %d, validation videos: %d", len(train_items), len(val_items))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_info = load_model(args.base_model, device)
    frozen = freeze_for_finetune(model)

    train_ds = ClipDataset(train_items, args.clips_per_video, True, args.seed)
    val_ds = ClipDataset(val_items, 2, False, args.seed) if val_items else None
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    weights = sqrt_class_weights([l for _, l in train_items], args.clips_per_video).to(device)
    log.info("Class weights (sqrt inverse freq): safe=%.3f unsafe=%.3f", *weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.fc.parameters(), "lr": LR_FC},
        ]
    )

    best_state, best_val, best_epoch = None, float("inf"), -1
    history = []
    log.info("Fine-tuning on %s ...", device)
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, frozen)
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        rec = {"epoch": epoch, "train_loss": run_loss / max(1, total), "train_acc": 100.0 * correct / max(1, total)}
        if val_loader is not None:
            vl, va = evaluate(model, val_loader, val_criterion, device)
            rec.update(val_loss=vl, val_acc=va)
            if vl < best_val:
                best_val, best_epoch = vl, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        log.info("Epoch %d/%d %s", epoch, args.epochs, json.dumps({k: round(v, 4) for k, v in rec.items()}))

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Using best validation epoch %d (val_loss %.4f)", best_epoch, best_val)

    out_path = unique_output_path(args.out_dir)
    torch.save(model.state_dict(), out_path)
    sidecar = {
        "thresholds": base_info["thresholds"],
        "base_model": base_info["version"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": {"layer4": LR_LAYER4, "fc": LR_FC},
        "videos": {"safe": n_safe, "unsafe": n_unsafe, "val": len(val_items)},
        "history": history,
        "note": "Thresholds were inherited from the base model; re-run video_calibration.py and edit them here if needed.",
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)
    log.info("Saved %s (+ .json sidecar). Use it with: SAFESPACE_MODEL=%s", out_path, out_path)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--base-model", default=None, help="Defaults to SAFESPACE_MODEL / auto-selected weights.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--clips-per-video", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
