"""
video_calibration.py

Offline calibration tool for the SafeSpace violence-detection model.

Feed it any video file (an action movie clip, a UFC fight, a fighting
compilation) and it will:
  1. Run the exact same frame prep (prepared_frames.PreparedFrameReader ->
     frame_prep.prepare_for_model) and
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

Requires: the same weights as main.py (env SAFESPACE_MODEL, else the first
existing of the v3 / smart_crop / base .pth in the working directory, or pass
--model-path).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from clip_sampler import ClipWindow
from inference_core import (
    CAMERA_TARGET_FPS,
    CLIP_LENGTH,
    INFERENCE_STRIDE,
    SCORE_HISTORY_LEN,
    SPIKE_THRESHOLD,
    ALERT_THRESHOLD,
    load_model,
    make_tracker,
    predict_violence,
    preprocess_clip,
)

from prepared_frames import PreparedFrameReader

log = logging.getLogger("safespace.calibration")

# Constants (CLIP_LENGTH, CAMERA_TARGET_FPS, INFERENCE_STRIDE, thresholds) come
# from inference_core so offline tools can never drift from live mode.


def score_window(model: nn.Module, frames_rgb_112: List[np.ndarray], device: torch.device) -> float:
    """Same tensor prep as live mode (inference_core.preprocess_clip: /255 only)."""
    batch = preprocess_clip(frames_rgb_112).unsqueeze(0).to(device)
    return float(predict_violence(model, batch)[0].item())


def analyze_windows(
    path: str,
    model: nn.Module,
    device: torch.device,
    target_fps: float = CAMERA_TARGET_FPS,
    thresholds: Optional[Dict[str, Any]] = None,
    spike_threshold: Optional[float] = None,
    alert_threshold: Optional[float] = None,
    history_len: Optional[int] = None,
    max_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Run a video through exactly the live pipeline: frame_prep (HOG off) ->
    15 fps timeline resample -> timestamp clip sampling -> /255 -> model ->
    IncidentTracker.  Returns raw per-window samples plus incident records."""
    reader = PreparedFrameReader(path, target_fps=target_fps, max_seconds=max_seconds)
    stride_s = INFERENCE_STRIDE / float(target_fps)

    window = ClipWindow(n=CLIP_LENGTH, fps=target_fps)
    tracker = make_tracker(
        thresholds,
        spike_threshold=spike_threshold,
        alert_threshold=alert_threshold,
        score_history_len=history_len,
    )

    samples: List[Dict[str, Any]] = []
    incidents: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    last_scored: Optional[float] = None
    last_meta: Dict[str, Any] = {}
    suitable_count = unsuitable_count = global_count = 0
    peak_raw = peak_smoothed = 0.0
    last_t = 0.0
    model.eval()

    def handle(events: List[dict]) -> None:
        nonlocal current
        for ev in events:
            if ev["type"] == "open":
                current = {
                    "start_t": ev["start_t"], "trigger": ev["trigger"], "peak": ev["peak"],
                    "n_triggers": ev["n_triggers"], "end_t": None,
                }
                incidents.append(current)
            elif ev["type"] == "extend" and current is not None:
                current["peak"] = ev["peak"]
                current["n_triggers"] = ev["n_triggers"]
            elif ev["type"] == "close" and current is not None:
                current["peak"] = ev["peak"]
                current["n_triggers"] = ev["n_triggers"]
                current["end_t"] = ev["end_t"]
                current["close_reason"] = ev["reason"]
                current = None

    for pf in reader:  # closes the capture when exhausted
        t = pf.t
        meta = pf.meta
        last_meta = meta
        last_t = t
        if pf.suitable:
            suitable_count += 1
        else:
            unsuitable_count += 1
        if pf.global_motion:
            global_count += 1
        if window.add(t, pf.rgb, pf.suitable, pf.global_motion):
            handle(tracker.reset(t, reason="gap"))
            last_scored = None

        sample = window.sample()
        if sample is None:
            continue
        if last_scored is not None and (t - last_scored) < stride_s - 1e-6:
            continue
        last_scored = t

        raw = score_window(model, sample["items"], device)
        events = tracker.update(
            t, raw, suitable=sample["suitable"], degraded=False,
            global_motion=sample["global_motion"],
        )
        handle(events)
        avg = tracker.average() or 0.0
        peak_raw = max(peak_raw, raw)
        peak_smoothed = max(peak_smoothed, avg)
        trig = next((e for e in events if e["type"] in ("open", "extend")), None)
        samples.append({
            "t": t,
            "frame": pf.src_index,
            "raw": raw,
            "smoothed": avg,
            "triggered": trig is not None,
            "trigger_reason": (trig or {}).get("trigger", ""),
            "suitable": bool(sample["suitable"]),
            "global_motion": bool(sample["global_motion"]),
            "in_incident": tracker.active,
            "meta": meta,
        })

    reader.close()
    handle(tracker.finish(last_t))
    for inc in incidents:
        if inc["end_t"] is None:
            inc["end_t"] = last_t

    return {
        "native_fps": reader.native_fps,
        "total_frames": reader.total_frames,
        "frames_read": reader.frames_read,
        "duration_sec": reader.frames_read / reader.native_fps,
        "samples": samples,
        "incidents": incidents,
        "suitable_count": suitable_count,
        "unsuitable_count": unsuitable_count,
        "global_motion_count": global_count,
        "last_meta": last_meta,
        "peak_raw": peak_raw,
        "peak_smoothed": peak_smoothed,
        "alert_threshold": tracker.alert_threshold,
        "spike_threshold": tracker.spike_threshold,
    }


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
    thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[float, float, float, str]], List[dict], float]:
    """Returns (events, per_sample_log, native_fps) - same shape as before, but
    incidents now come from the shared IncidentTracker."""
    res = analyze_windows(
        path, model, device, target_fps=target_fps, thresholds=thresholds,
        spike_threshold=spike_threshold, alert_threshold=alert_threshold,
        history_len=history_len,
    )
    per_sample_log: List[dict] = []
    for s in res["samples"]:
        meta = s["meta"]
        per_sample_log.append(
            {
                "time_sec": round(s["t"], 2),
                "raw_score": round(s["raw"], 4),
                "smoothed_avg": round(s["smoothed"], 4),
                "triggered": s["triggered"],
                "trigger_reason": s["trigger_reason"],
                "scene_method": meta.get("method"),
                "subject_area_frac": meta.get("subject_area_frac"),
                "suitable_for_model": s["suitable"],
                "global_motion": s["global_motion"],
            }
        )

    window_s = (CLIP_LENGTH - 1) / float(target_fps)
    events: List[list] = [
        [max(0.0, i["start_t"] - window_s), i["end_t"], i["peak"], i["trigger"]]
        for i in res["incidents"]
    ]
    merged: List[list] = []
    for ev in events:
        if merged and ev[0] - merged[-1][1] <= merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], ev[1])
            merged[-1][2] = max(merged[-1][2], ev[2])
        else:
            merged.append(ev)

    duration_sec = res["duration_sec"]
    padded = []
    for s, e, peak, trig in merged:
        padded.append((max(0.0, s - pad_seconds), min(duration_sec, e + pad_seconds), peak, trig))
    return padded, per_sample_log, res["native_fps"]


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
        subprocess.run(cmd, check=True, timeout=300)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        cut_clip_opencv(src_path, out_path, start_sec, end_sec)


def main():
    parser = argparse.ArgumentParser(description="Clip out violent segments per the SafeSpace model.")
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument("--out", default="calibration_out", help="Output directory.")
    parser.add_argument("--model-path", default=None, help="Defaults to SAFESPACE_MODEL / auto-selected weights.")
    parser.add_argument("--target-fps", type=float, default=CAMERA_TARGET_FPS)
    parser.add_argument("--spike-threshold", type=float, default=None, help="Default: model sidecar / inference_core.")
    parser.add_argument("--alert-threshold", type=float, default=None)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--pad-seconds", type=float, default=2.0, help="Context padding added to each side of a clipped event.")
    parser.add_argument("--merge-gap-seconds", type=float, default=1.5, help="Merge events separated by less than this many seconds.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Loading model on {device}...")
    model, info = load_model(args.model_path, device)
    print(f"[SYSTEM] Model {info['version']}")
    th = info["thresholds"]
    spike_t = args.spike_threshold if args.spike_threshold is not None else th["spike_threshold"]
    alert_t = args.alert_threshold if args.alert_threshold is not None else th["alert_threshold"]
    hist_n = args.history_len if args.history_len is not None else th["score_history_len"]

    print(f"[SYSTEM] Processing {args.video} ...")
    events, per_sample_log, native_fps = process_video(
        args.video,
        model,
        device,
        target_fps=args.target_fps,
        spike_threshold=spike_t,
        alert_threshold=alert_t,
        history_len=hist_n,
        thresholds=th,
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
          f"to see whether SPIKE_THRESHOLD ({spike_t}) / ALERT_THRESHOLD "
          f"({alert_t}) are catching what you'd visually call violent.")


if __name__ == "__main__":
    main()