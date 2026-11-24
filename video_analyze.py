"""Offline video analysis using exactly the live pipeline.

Built on video_calibration.analyze_windows, which in turn uses inference_core
(/255-only preprocessing, 15 fps timeline resampling, timestamp clip sampling,
frame_prep with HOG off) and the shared IncidentTracker, so a verdict here
means "live mode would have raised an incident on this footage".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from inference_core import CAMERA_TARGET_FPS
from video_calibration import analyze_windows


def analyze_video_file(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
    alert_threshold: Optional[float] = None,
    max_seconds: float = 180.0,
    thresholds: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    res = analyze_windows(
        path,
        model,
        device,
        target_fps=CAMERA_TARGET_FPS,
        thresholds=thresholds,
        alert_threshold=alert_threshold,
        max_seconds=max_seconds,
    )
    if not res["samples"]:
        raise RuntimeError(
            "Video too short or unreadable - need enough frames for one 16-frame clip."
        )

    thr = res["alert_threshold"]
    incidents = res["incidents"]
    would_alert = bool(incidents)
    n_kept = max(1, res["suitable_count"] + res["unsuitable_count"])
    framing_ok_ratio = res["suitable_count"] / n_kept
    peak_smoothed = res["peak_smoothed"]

    samples = []
    for s in res["samples"]:
        meta = s["meta"]
        samples.append(
            {
                "t_sec": round(s["t"], 2),
                "frame": s["frame"],
                "raw_pct": round(s["raw"] * 100, 1),
                "smoothed_pct": round(s["smoothed"] * 100, 1),
                "above_threshold": bool(s["smoothed"] >= thr),
                "suitable_for_model": bool(s["suitable"]),
                "subject_area_frac": meta.get("subject_area_frac"),
                "method": meta.get("method"),
                "global_motion": s["global_motion"],
                "in_incident": s["in_incident"],
            }
        )

    if framing_ok_ratio < 0.25 and not would_alert:
        explanation = (
            f"Most of this video is overview CCTV (subjects too small for 112x112). "
            f"Peak smoothed {peak_smoothed*100:.1f}% - high scores here are unreliable. "
            f"Native resolution does not matter; camera distance/framing does. "
            f"Use a nearer view or fine-tune with labeled data."
        )
    elif would_alert:
        first = incidents[0]
        explanation = (
            f"Model would raise {len(incidents)} incident(s) (first: {first['trigger']} at "
            f"{first['start_t']:.1f}s, peak raw {max(i['peak'] for i in incidents)*100:.1f}%, "
            f"threshold {thr*100:.0f}%), judged with the same preprocessing, 15 fps "
            f"resampling, thresholds and incident rules as live mode."
        )
    else:
        explanation = (
            f"Model stayed below the incident rules (or scores were suppressed due to "
            f"unsuitable framing / global motion). Peak smoothed {peak_smoothed*100:.1f}% "
            f"(threshold {thr*100:.0f}%)."
        )

    return {
        "ok": True,
        "path": path,
        "verdict": "would_alert" if would_alert else "safe",
        "would_alert": would_alert,
        "peak_raw_pct": round(res["peak_raw"] * 100, 1),
        "peak_smoothed_pct": round(peak_smoothed * 100, 1),
        "alert_threshold_pct": round(thr * 100, 1),
        "fps": round(res["native_fps"], 2),
        "frames_analyzed": res["frames_read"],
        "total_frames": res["total_frames"] or res["frames_read"],
        "duration_sec": round(res["duration_sec"], 2),
        "framing_suitable_ratio": round(framing_ok_ratio, 3),
        "global_motion_ratio": round(res["global_motion_count"] / n_kept, 3),
        "scene": res["last_meta"],
        "samples": samples,
        "incidents": [
            {
                "start_sec": round(i["start_t"], 2),
                "end_sec": round(i["end_t"], 2),
                "peak_pct": round(i["peak"] * 100, 1),
                "trigger": i["trigger"],
                "n_triggers": i["n_triggers"],
            }
            for i in incidents
        ],
        "model_version": model_version,
        "explanation": explanation,
    }
