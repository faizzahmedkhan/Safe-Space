"""
Framing prep for the 112x112 model.

Honest limit: native video resolution (1080p vs 720p) does NOT matter — we always
end at 112x112. What matters is subject scale (how large people are in the frame).

This module:
1) Tries to crop/zoom onto people or motion so 112x112 is less empty hallway.
2) Reports whether the scene is suitable for this model (subjects large enough).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

MODEL_SIZE = 112
# Subject bounding box must cover at least this fraction of the full frame
# before we treat the scene as "near-field enough" for reliable alerts.
MIN_SUBJECT_AREA_FRAC = 0.05
# When zooming, never crop tighter than this fraction of the shorter side
# (avoids blowing up pure compression noise).
MIN_CROP_SIDE_FRAC = 0.35


_hog = None


def _get_hog():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


def _square_crop(frame: np.ndarray, x: int, y: int, bw: int, bh: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = max(bw, bh)
    min_side = int(min(h, w) * MIN_CROP_SIDE_FRAC)
    side = max(side, min_side)
    side = min(side, min(h, w))

    cx = x + bw // 2
    cy = y + bh // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return frame[y0 : y0 + side, x0 : x0 + side]


def _detect_person_box(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return largest person box (x, y, w, h) or None."""
    h, w = frame.shape[:2]
    # HOG is slow on 1080p — run on a downscaled copy
    scale = 400.0 / max(h, w)
    if scale > 1.0:
        scale = 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    try:
        boxes, weights = _get_hog().detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
    except Exception:
        return None
    if boxes is None or len(boxes) == 0:
        return None
    # map back to full resolution; pick highest weight / largest
    best_i = int(np.argmax([b[2] * b[3] for b in boxes]))
    x, y, bw, bh = boxes[best_i]
    inv = 1.0 / scale if scale > 1e-6 else 1.0
    return (
        int(x * inv),
        int(y * inv),
        int(bw * inv),
        int(bh * inv),
    )


def _detect_motion_box(
    frame: np.ndarray, prev_gray: Optional[np.ndarray]
) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return None, gray

    diff = cv2.absdiff(prev_gray, gray)
    _, th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, gray
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = frame.shape[:2]
    if area < 0.002 * (h * w):
        return None, gray
    x, y, bw, bh = cv2.boundingRect(c)
    return (x, y, bw, bh), gray


def prepare_for_model(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
    person_box_cache: Optional[Tuple[int, int, int, int]] = None,
    run_person_detector: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Build a 112x112 RGB frame for the model + suitability metadata.

    Returns:
      rgb_112, meta, gray_for_next_frame, person_box_cache
    """
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)

    person = person_box_cache
    if run_person_detector:
        detected = _detect_person_box(frame_bgr)
        if detected is not None:
            person = detected
        # if detector finds nothing, keep previous cache briefly (caller may clear)

    motion, gray = _detect_motion_box(frame_bgr, prev_gray)

    box = None
    method = "full_frame"
    if person is not None:
        box = person
        method = "person_crop"
    elif motion is not None:
        box = motion
        method = "motion_crop"

    if box is not None:
        x, y, bw, bh = box
        # expand a bit so limbs aren't clipped
        pad = int(0.15 * max(bw, bh))
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2 * pad)
        bh = min(h - y, bh + 2 * pad)
        subject_frac = (bw * bh) / frame_area
        crop = _square_crop(frame_bgr, x, y, bw, bh)
    else:
        # Mild center crop — still better than letterboxing empty edges sometimes
        side = int(min(h, w) * 0.7)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = frame_bgr[y0 : y0 + side, x0 : x0 + side]
        subject_frac = 0.0
        method = "center_crop"

    suitable = subject_frac >= MIN_SUBJECT_AREA_FRAC and method in (
        "person_crop",
        "motion_crop",
    )

    resized = cv2.resize(crop, (MODEL_SIZE, MODEL_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    meta = {
        "input_width": w,
        "input_height": h,
        "method": method,
        "subject_area_frac": round(float(subject_frac), 4),
        "suitable_for_model": bool(suitable),
        "note": (
            "Subjects large enough for 112x112 analysis."
            if suitable
            else (
                "Subjects too small / far for reliable 112x112 analysis "
                "(overview CCTV). Alerts suppressed; use nearer camera or fine-tune."
            )
        ),
    }
    return rgb, meta, gray, person if method == "person_crop" else None
