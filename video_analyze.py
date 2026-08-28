"""Offline video analysis using the same clip + smoothing logic as live mode."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from camera import CLIP_LENGTH
from frame_prep import prepare_for_model

INFERENCE_STRIDE = 8
SCORE_HISTORY_LEN = 10
def _preprocess_clip(frames_rgb_112: List[np.ndarray], device: torch.device) -> torch.Tensor:
    frames_array = np.stack(frames_rgb_112)
    tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
    tensor.div_(255.0)
    tensor = tensor.permute(1, 0, 2, 3)
    return tensor.unsqueeze(0).to(device)


def analyze_video_file(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
    alert_threshold: float = 0.60,
    max_seconds: float = 180.0,
) -> Dict[str, Any]:
    """
    Run the violence model over a recorded video.
    Uses the same subject-crop framing as live mode.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if fps <= 1e-3:
        fps = 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seconds * fps) if max_seconds > 0 else 0

    clip: Deque[np.ndarray] = deque(maxlen=CLIP_LENGTH)
    score_history: Deque[float] = deque(maxlen=SCORE_HISTORY_LEN)
    samples: List[Dict[str, Any]] = []
    would_alert = False
    alert_active = False
    peak_raw = 0.0
    peak_smoothed = 0.0
    frames_seen = 0
    since_infer = 0
    prev_gray = None
    suitable_count = 0
    unsuitable_count = 0
    last_meta: Dict[str, Any] = {}

    model.eval()
    with torch.no_grad():
        while True:
            if max_frames and frames_seen >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames_seen += 1
            since_infer += 1

            rgb, meta, prev_gray = prepare_for_model(frame, prev_gray)
            last_meta = meta
            if meta.get("suitable_for_model"):
                suitable_count += 1
            else:
                unsuitable_count += 1
            clip.append(rgb)

            if len(clip) < CLIP_LENGTH:
                continue
            if since_infer < INFERENCE_STRIDE:
                continue
            since_infer = 0

            inputs = _preprocess_clip(list(clip), device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            raw = float(probs[0][1].item())
            score_history.append(raw)

            smoothed = None
            # Adaptive smoothing: for offline analysis of short evidence clips,
            # we can't wait for the full SCORE_HISTORY_LEN window (a 3-second
            # clip only yields ~5 predictions). Start computing smoothed as
            # soon as we have enough predictions for a meaningful average.
            min_for_smoothed = min(3, SCORE_HISTORY_LEN)
            if len(score_history) >= min_for_smoothed:
                smoothed = float(sum(score_history) / len(score_history))
                peak_smoothed = max(peak_smoothed, smoothed)
                # Only count alerts when framing is suitable
                if smoothed >= alert_threshold and meta.get("suitable_for_model"):
                    if not alert_active:
                        would_alert = True
                        alert_active = True
                else:
                    alert_active = False

            peak_raw = max(peak_raw, raw)
            t_sec = round(frames_seen / fps, 2)
            samples.append(
                {
                    "t_sec": t_sec,
                    "frame": frames_seen,
                    "raw_pct": round(raw * 100, 1),
                    "smoothed_pct": round(smoothed * 100, 1) if smoothed is not None else None,
                    "above_threshold": bool(
                        smoothed is not None and smoothed >= alert_threshold
                    ),
                    "suitable_for_model": bool(meta.get("suitable_for_model")),
                    "subject_area_frac": meta.get("subject_area_frac"),
                    "method": meta.get("method"),
                }
            )

    cap.release()

    if not samples:
        raise RuntimeError(
            "Video too short or unreadable — need enough frames for one 16-frame clip."
        )

    framing_ok_ratio = suitable_count / max(1, suitable_count + unsuitable_count)
    verdict = "would_alert" if would_alert else "safe"
    if framing_ok_ratio < 0.25 and not would_alert:
        explanation = (
            f"Most of this video is overview CCTV (subjects too small for 112×112). "
            f"Peak smoothed {peak_smoothed*100:.1f}% — high scores here are unreliable. "
            f"Native resolution does not matter; camera distance/framing does. "
            f"Use a nearer view or fine-tune with labeled data."
        )
    elif would_alert:
        explanation = (
            f"Model would raise an alert (smoothed score reached {peak_smoothed*100:.1f}% "
            f"≥ threshold {alert_threshold*100:.0f}%) on frames with suitable subject scale."
        )
    else:
        explanation = (
            f"Model stayed below alert threshold (or scores were suppressed due to "
            f"unsuitable framing). Peak smoothed {peak_smoothed*100:.1f}% "
            f"(threshold {alert_threshold*100:.0f}%)."
        )

    return {
        "ok": True,
        "path": path,
        "verdict": verdict,
        "would_alert": would_alert,
        "peak_raw_pct": round(peak_raw * 100, 1),
        "peak_smoothed_pct": round(peak_smoothed * 100, 1),
        "alert_threshold_pct": round(alert_threshold * 100, 1),
        "fps": round(fps, 2),
        "frames_analyzed": frames_seen,
        "total_frames": total_frames or frames_seen,
        "duration_sec": round(frames_seen / fps, 2),
        "framing_suitable_ratio": round(framing_ok_ratio, 3),
        "scene": last_meta,
        "samples": samples,
        "explanation": explanation,
    }
