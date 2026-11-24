"""Fine-tune the SafeSpace model on labelled clips in data/safe and data/unsafe.

Matches the live pipeline exactly:
  * every video is prepared ONCE into the frame cache (frame_cache.py) by the
    same front-end /analyze/* uses (prepared_frames.PreparedFrameReader):
    15 fps timeline resampling, frame_prep.prepare_for_model with HOG off and
    no person-box cache.  A window cut from the cache is bit-identical to
    preparing it on the fly, it is just not re-decoded every epoch.
  * 16-frame windows sampled like ClipWindow.sample (clip_sampler)
  * /255 scaling only (NO Kinetics mean/std) - inference_core.preprocess_batch

Training recipe: start from the currently selected model (load_model:
SAFESPACE_MODEL or the auto-picked file), freeze everything up to layer3, train
layer4 at 1e-5 and fc at 1e-3, sqrt-inverse class weights, a seeded 15%
video-level validation hold-out.  Each epoch draws a fresh random window per
video (--windows-per-video of them).

The best epoch is chosen on the validation videos the way the model is used
live (clip_eval): every INFERENCE_STRIDE window is scored, replayed through
IncidentTracker, and the epoch with the best --selection-metric (default
pr_auc_max) wins.

The result is saved as a NEW timestamped file plus a JSON sidecar (with the
validation report); existing weights are never overwritten.  To use it:

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cached_dataset import CacheItem, CachedClipDataset, iter_eval_videos, seed_worker
from clip_eval import (
    DEFAULT_SELECTION_METRIC,
    SELECTION_METRICS,
    evaluate_model,
    format_summary,
    selection_value,
)
from frame_cache import FORMATS, FrameCache
from inference_core import load_model

log = logging.getLogger("safespace.train")

DATA_DIR = "data"
CACHE_DIR = ".frame_cache"
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


def build_cache_items(
    cache: FrameCache, items: List[Tuple[str, int]], workers: int
) -> List[CacheItem]:
    """Cache every video (reusing existing entries) and drop unreadable ones."""
    report = cache.build_many([p for p, _ in items], workers=workers)
    log.info("Frame cache %s: %s", cache.root, report)
    for path, err in report.failed.items():
        log.warning("Skipping unreadable video %s (%s)", path, err)
    names = {0: "safe", 1: "unsafe"}
    return [
        CacheItem(key=report.keys[p], label=l, path=p, cls=names[l])
        for p, l in items
        if p in report.keys
    ]


def _json_safe(obj: Any) -> Any:
    """Replace nan/inf (not valid JSON) with None, recursively."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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

    train_split, val_split = split_videos(paths, labels, args.val_frac, args.seed)
    cache = FrameCache(args.cache_dir, fmt=args.cache_format)
    train_items = build_cache_items(cache, train_split, args.cache_workers)
    val_items = build_cache_items(cache, val_split, args.cache_workers)
    log.info("Train videos: %d, validation videos: %d", len(train_items), len(val_items))
    if not {0, 1} <= {it.label for it in train_items}:
        log.error("After caching, the training split no longer has both classes.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_info = load_model(args.base_model, device)
    thresholds = base_info["thresholds"]
    frozen = freeze_for_finetune(model)

    train_ds = CachedClipDataset(cache, train_items, windows_per_video=args.windows_per_video)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
    )

    weights = sqrt_class_weights([it.label for it in train_items], args.windows_per_video).to(device)
    log.info("Class weights (sqrt inverse freq): safe=%.3f unsafe=%.3f", *weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.fc.parameters(), "lr": LR_FC},
        ]
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_value, best_epoch = -math.inf, -1
    best_summary: Optional[Dict[str, Any]] = None
    history = []
    log.info("Fine-tuning on %s (selecting by %s) ...", device, args.selection_metric)
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
        rec: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": run_loss / max(1, total),
            "train_acc": 100.0 * correct / max(1, total),
        }
        if val_items:
            res = evaluate_model(
                model, iter_eval_videos(cache, val_items), device,
                thresholds=thresholds, batch_size=args.eval_batch_size,
            )
            value = selection_value(res.summary, args.selection_metric)
            rec.update(
                selection_value=value,
                val_pr_auc_max=res.summary["scores"]["max_p"]["pr_auc"],
                val_live_normal_recall=res.summary["live"]["normal_recall"],
                val_live_dangerous_recall=res.summary["live"]["dangerous_recall"],
            )
            log.info("Epoch %d validation (deployment schedule):\n%s", epoch, format_summary(res.summary))
            if value > best_value:
                best_value, best_epoch, best_summary = value, epoch, res.summary
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        log.info("Epoch %d/%d %s", epoch, args.epochs,
                 json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Using best validation epoch %d (%s %.4f)", best_epoch, args.selection_metric, best_value)
    else:
        log.info("No validation videos: keeping the last epoch.")

    out_path = unique_output_path(args.out_dir)
    torch.save(model.state_dict(), out_path)
    sidecar = {
        "thresholds": thresholds,
        "base_model": base_info["version"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": {"layer4": LR_LAYER4, "fc": LR_FC},
        "videos": {"safe": n_safe, "unsafe": n_unsafe, "train": len(train_items), "val": len(val_items)},
        "windows_per_video": args.windows_per_video,
        "selection": {
            "metric": args.selection_metric,
            "value": best_value if best_state is not None else None,
            "validation": best_summary,
        },
        "frame_cache": {"format": cache.fmt, "fingerprint": cache.fingerprint},
        "history": history,
        "note": "Thresholds were inherited from the base model; re-run video_calibration.py and edit them here if needed.",
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(_json_safe(sidecar), f, indent=2)
    log.info("Saved %s (+ .json sidecar). Use it with: SAFESPACE_MODEL=%s", out_path, out_path)
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--base-model", default=None, help="Defaults to SAFESPACE_MODEL / auto-selected weights.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--windows-per-video", "--clips-per-video", dest="windows_per_video", type=int, default=4,
                   help="Random training windows drawn per video per epoch.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=CACHE_DIR, help="Frame cache directory (safe to delete).")
    p.add_argument("--cache-format", choices=FORMATS, default="npy",
                   help="npy = exact (default); jpg = ~10-25x smaller but lossy.")
    p.add_argument("--cache-workers", type=int, default=max(1, min(4, (os.cpu_count() or 1) - 1)),
                   help="Processes used to build the frame cache.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (cache reads are cheap).")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--selection-metric", choices=sorted(SELECTION_METRICS), default=DEFAULT_SELECTION_METRIC)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
