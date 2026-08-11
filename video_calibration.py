"""
video_calibration.py

Offline calibration tool for the SafeSpace violence-detection model.

Feed it any video file (an action movie clip, a UFC fight, a fighting
compilation) and it will:
  1. Run the exact same frame prep (frame_prep.prepare_for_model) and
     windowing/threshold logic used live in main.py / camera.py.
  2. Log the raw model score for every inference window to a CSV so you
     can plot/inspect where the model's confidence actually sits.
  3. Cut out an .mp4 clip for every segment where the model would have
     triggered an alert (spike or sustained path), padded with a few
     seconds of context on each side.

This does NOT touch your live system, thresholds, or model weights — it's
read-only calibration. Point it at footage you already have the rights to
use, review the CSV + clips, and decide whether SPIKE_THRESHOLD /
ALERT_THRESHOLD in main.py need to move.

Usage:
    python video_calibration.py path/to/video.mp4
    python video_calibration.py path/to/video.mp4 --out calibration_out \
        --spike-threshold 0.85 --pad-seconds 2.0

Requires: the same safespace_bullying_model.pth weights file as main.py,
in the working directory (or pass --model-path).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models

from frame_prep import prepare_for_model

# --- Keep these in sync with main.py / camera.py. They are duplicated
# here (not imported) so this script can run standalone against a video
# file without booting the FastAPI app, camera threads, etc. ---
CLIP_LENGTH = 16
CAMERA_TARGET_FPS = 15
INFERENCE_STRIDE = 8
SPIKE_THRESHOLD = 0.90
ALERT_THRESHOLD = 0.75
SCORE_HISTORY_LEN = 4
PERSON_DETECT_EVERY = 10  # matches camera.py's HOG cadence


def load_model(model_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Copy safespace_bullying_model.pth "
            f"into the working directory or pass --model-path."
        )
    model = video_models.r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def score_window(model: nn.Module, frames_rgb_112: List[np.ndarray], device: torch.device) -> float:
    """Same tensor prep as main.py's _run_inference_batch: scale to [0,1]
    only, no Kinetics normalize (must match training — see main.py notes)."""
    arr = np.stack(frames_rgb_112)  # (T, H, W, C) uint8
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    tensor.div_(255.0)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
    return probs[0][1].item()  # class 1 = violent, matches main.py


def process_video(
    path: str,
    model: nn.Module,
    device: torch.device,
    target_fps: float,
    spike_threshold: float,
    alert_threshold: float,
    history_len: int,
    pad_seconds: float,
    merge_gap_seconds: float,
) -> Tuple[List[Tuple[float, float, float, str]], List[dict], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    # Live cameras get paced to target_fps by sleeping between reads. A video
    # file has no real-time pacing, so we approximate the same sampling rate
    # by taking every Nth native frame.
    step = max(1, round(native_fps / target_fps))
    effective_fps = native_fps / step

    frame_buffer: deque = deque(maxlen=CLIP_LENGTH)
    score_history: deque = deque(maxlen=history_len)
    prev_gray = None
    person_box = None
    detect_counter = 0

    events: List[list] = []  # [start_sec, end_sec, peak_score, trigger]
    active_start: Optional[float] = None
    active_peak = 0.0
    active_trigger: Optional[str] = None
    consecutive_above = 0

    per_sample_log: List[dict] = []

    frames_read = 0
    sampled_index = 0
    since_last_infer = 0
    last_t_sec = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frames_read % step != 0:
            frames_read += 1
            continue
        frames_read += 1

        detect_counter = (detect_counter + 1) % PERSON_DETECT_EVERY
        run_hog = detect_counter == 1
        rgb112, scene_meta, prev_gray, person_box = prepare_for_model(
            frame, prev_gray, person_box_cache=person_box, run_person_detector=run_hog
        )
        frame_buffer.append(rgb112)
        sampled_index += 1
        since_last_infer += 1
        last_t_sec = sampled_index / effective_fps

        if len(frame_buffer) < CLIP_LENGTH or since_last_infer < INFERENCE_STRIDE:
            continue
        since_last_infer = 0

        raw_score = score_window(model, list(frame_buffer), device)
        score_history.append(raw_score)
        avg_score = sum(score_history) / len(score_history)

        is_spike = raw_score >= spike_threshold
        if avg_score >= alert_threshold:
            consecutive_above += 1
        else:
            consecutive_above = 0
        is_sustained = consecutive_above >= 2
        triggered = is_spike or is_sustained
        trigger_reason = "spike" if is_spike else ("sustained" if is_sustained else "")

        per_sample_log.append(
            {
                "time_sec": round(last_t_sec, 2),
                "raw_score": round(raw_score, 4),
                "smoothed_avg": round(avg_score, 4),
                "triggered": triggered,
                "trigger_reason": trigger_reason,
                "scene_method": scene_meta.get("method"),
                "subject_area_frac": scene_meta.get("subject_area_frac"),
                "suitable_for_model": scene_meta.get("suitable_for_model"),
            }
        )

        if triggered and scene_meta.get("suitable_for_model", True):
            if active_start is None:
                active_start = last_t_sec
                active_trigger = trigger_reason
            active_peak = max(active_peak, raw_score)
        else:
            if active_start is not None:
                events.append([active_start, last_t_sec, active_peak, active_trigger])
                active_start = None
                active_peak = 0.0
                active_trigger = None

    if active_start is not None:
        events.append([active_start, last_t_sec, active_peak, active_trigger])

    cap.release()

    # Merge events that are close together so one fight doesn't become 20 clips.
    merged: List[list] = []
    for ev in events:
        if merged and ev[0] - merged[-1][1] <= merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], ev[1])
            merged[-1][2] = max(merged[-1][2], ev[2])
        else:
            merged.append(ev)

    duration_sec = frames_read / native_fps if native_fps else last_t_sec
    padded = []
    for s, e, peak, trig in merged:
        s2 = max(0.0, s - pad_seconds)
        e2 = min(duration_sec, e + pad_seconds)
        padded.append((s2, e2, peak, trig))

    return padded, per_sample_log, native_fps


def cut_clip_opencv(src_path: str, out_path: str, start_sec: float, end_sec: float) -> None:
    """Fallback clip extraction using OpenCV when ffmpeg is not installed."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for clipping: {src_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    start_frame = int(start_sec * native_fps)
    end_frame = int(end_sec * native_fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, native_fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()


def get_ffmpeg_binary() -> str:
    """Returns the path to ffmpeg executable (from imageio_ffmpeg in virtualenv or system PATH)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return "ffmpeg"


def cut_clip(src_path: str, out_path: str, start_sec: float, end_sec: float) -> None:
    """Uses ffmpeg to cut [start_sec, end_sec] out of src_path into out_path.
    Falls back to OpenCV if ffmpeg is not available on the system."""
    duration = max(0.1, end_sec - start_sec)
    ffmpeg_exe = get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", f"{start_sec:.2f}",
        "-i", src_path,
        "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        cut_clip_opencv(src_path, out_path, start_sec, end_sec)


def main():
    parser = argparse.ArgumentParser(description="Clip out violent segments per the SafeSpace model.")
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument("--out", default="calibration_out", help="Output directory.")
    parser.add_argument("--model-path", default="safespace_bullying_model.pth")
    parser.add_argument("--target-fps", type=float, default=CAMERA_TARGET_FPS)
    parser.add_argument("--spike-threshold", type=float, default=SPIKE_THRESHOLD)
    parser.add_argument("--alert-threshold", type=float, default=ALERT_THRESHOLD)
    parser.add_argument("--history-len", type=int, default=SCORE_HISTORY_LEN)
    parser.add_argument("--pad-seconds", type=float, default=2.0, help="Context padding added to each side of a clipped event.")
    parser.add_argument("--merge-gap-seconds", type=float, default=1.5, help="Merge events separated by less than this many seconds.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Loading model on {device}...")
    model = load_model(args.model_path, device)

    print(f"[SYSTEM] Processing {args.video} ...")
    events, per_sample_log, native_fps = process_video(
        args.video,
        model,
        device,
        target_fps=args.target_fps,
        spike_threshold=args.spike_threshold,
        alert_threshold=args.alert_threshold,
        history_len=args.history_len,
        pad_seconds=args.pad_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
    )

    # Write the full raw-score log — this is the important part for
    # calibration. Plot time_sec vs raw_score to see the model's confidence
    # curve across the whole video, not just where it happened to trigger.
    log_path = os.path.join(args.out, "score_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_log[0].keys()) if per_sample_log else [])
        if per_sample_log:
            writer.writeheader()
            writer.writerows(per_sample_log)
    print(f"[SYSTEM] Wrote raw score log: {log_path} ({len(per_sample_log)} samples)")

    if not events:
        print("[SYSTEM] No segments crossed SPIKE_THRESHOLD/ALERT_THRESHOLD. "
              "Either the video has nothing the model reads as violent, or "
              "the thresholds are too strict for this footage — check score_log.csv.")
        return

    print(f"[SYSTEM] {len(events)} triggered segment(s) found. Cutting clips...")
    manifest = []
    base = os.path.splitext(os.path.basename(args.video))[0]
    for i, (s, e, peak, trig) in enumerate(events, 1):
        out_name = f"{base}_event{i:02d}_{trig}_{s:.1f}s-{e:.1f}s_peak{peak:.2f}.mp4"
        out_path = os.path.join(args.out, out_name)
        try:
            cut_clip(args.video, out_path, s, e)
            print(f"  [{i}/{len(events)}] {s:.1f}s-{e:.1f}s (peak={peak:.2f}, trigger={trig}) -> {out_name}")
            manifest.append({"start_sec": round(s, 2), "end_sec": round(e, 2), "peak_score": round(peak, 4), "trigger": trig, "file": out_name})
        except subprocess.CalledProcessError as exc:
            print(f"  [{i}/{len(events)}] ffmpeg failed for {s:.1f}s-{e:.1f}s: {exc}")

    manifest_path = os.path.join(args.out, "events_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_sec", "end_sec", "peak_score", "trigger", "file"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[SYSTEM] Wrote event manifest: {manifest_path}")
    print(f"[SYSTEM] Done. Review the clips in {args.out}/ against the raw scores in score_log.csv "
          f"to see whether SPIKE_THRESHOLD ({args.spike_threshold}) / ALERT_THRESHOLD "
          f"({args.alert_threshold}) are catching what you'd visually call violent.")


if __name__ == "__main__":
    main()