"""The single offline frame front-end: video file -> model-ready 112x112 frames.

Everything that turns a recorded video into model input goes through
PreparedFrameReader: offline analysis (video_calibration.analyze_windows and
therefore /analyze/*), the training frame cache (frame_cache.py) and through
it train.py and the training notebook.  It mirrors what camera.py does live:

  1. timestamps are taken from the file (frame_index / native_fps);
  2. frames are thinned onto the 15 fps camera timeline (FpsResampler), so a
     30 fps file keeps every 2nd frame and a 25 fps file keeps the frame
     nearest each 1/15 s tick;
  3. each kept frame goes through the per-frame half of the model's crop
     mode (preprocessing.py), with prev_gray chained over the KEPT frames
     (camera.py chains it over its 15 fps-paced frames):
       crop="frame": frame_prep.prepare_for_model (HOG OFF, no person-box
                     cache) -> image is the final 112x112 crop;
       crop="clip":  frame_prep.observe_frame -> image is the working frame
                     and box its motion box; frame_prep.build_clip crops a
                     whole clip at once later;
  4. a timestamp gap > MAX_FRAME_GAP_S resets prev_gray before the next frame
     is prepared, exactly like CameraStream._reset_continuity.

Both per-frame functions depend only on the current frame and the previous
kept frame, so preparing a whole file once in order and later cutting any
16-frame window out of it (clip_from_items) is bit-identical to preparing
that window on the fly (the reason the frame cache is exact, not
approximate).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from clip_sampler import FpsResampler, detect_gap
from frame_prep import build_clip, observe_frame, prepare_for_model
from preprocessing import CROP_MODES
from thresholds import CAMERA_TARGET_FPS, MAX_FRAME_GAP_S


class PreparedFrame(NamedTuple):
    t: float  # seconds from the start of the file
    src_index: int  # 1-based index of the frame in the source file
    image: np.ndarray  # uint8 RGB: the 112x112 crop (crop="frame") or the working frame (crop="clip")
    box: Optional[Tuple[int, int, int, int]]  # motion box in `image` pixels (crop="clip" only)
    suitable: bool  # frame_prep "suitable_for_model"
    global_motion: bool  # frame_prep "global_motion" (shake / lighting change)
    meta: Dict[str, Any]  # full frame_prep metadata


class PreparedFrameReader:
    """Iterate a video file as PreparedFrame objects (see module docstring).

    Use as a context manager or call close().  After iteration the
    attributes native_fps, total_frames (container header, may be 0) and
    frames_read (source frames decoded, including any that were skipped by
    resampling) describe the file.
    """

    def __init__(
        self,
        path: str,
        target_fps: float = CAMERA_TARGET_FPS,
        max_seconds: float = 0.0,
        max_gap: float = MAX_FRAME_GAP_S,
        crop_mode: str = "frame",
    ):
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
        self.crop_mode = crop_mode
        self.path = path
        self.target_fps = float(target_fps)
        self.max_seconds = float(max_seconds or 0.0)
        self.max_gap = max_gap
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 1e-3:
            fps = self.target_fps
        self.native_fps = fps
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frames_read = 0
        self._consumed = False

    # -------------------------------------------------------------- iteration
    def __iter__(self) -> Iterator[PreparedFrame]:
        if self._consumed:
            raise RuntimeError("PreparedFrameReader can only be iterated once")
        self._consumed = True
        cap = self._cap
        native_dt = 1.0 / self.native_fps
        resampler = FpsResampler(self.target_fps)
        prev_gray: Optional[np.ndarray] = None
        last_t: Optional[float] = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = self.frames_read / self.native_fps
                self.frames_read += 1
                if self.max_seconds and t > self.max_seconds:
                    break
                if not resampler.keep(t, native_dt):
                    continue
                if detect_gap(last_t, t, self.max_gap):
                    prev_gray = None  # live: CameraStream._reset_continuity
                last_t = t
                if self.crop_mode == "frame":
                    image, meta, prev_gray = prepare_for_model(
                        frame, prev_gray, person_box_cache=None, run_person_detector=False
                    )
                else:
                    image, meta, prev_gray = observe_frame(frame, prev_gray)
                yield PreparedFrame(
                    t=t,
                    src_index=self.frames_read,
                    image=image,
                    box=meta.get("motion_box"),
                    suitable=bool(meta.get("suitable_for_model", True)),
                    global_motion=bool(meta.get("global_motion", False)),
                    meta=meta,
                )
        finally:
            self.close()

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "PreparedFrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def window_item(pf: PreparedFrame) -> Tuple[np.ndarray, Any]:
    """What a ClipWindow stores per frame: (image, box)."""
    return pf.image, pf.box


def clip_from_items(items: Sequence[Tuple[np.ndarray, Any]], crop_mode: str) -> np.ndarray:
    """ClipWindow.sample()["items"] (or cached frames) -> (T, 112, 112, 3)."""
    return build_clip([it[0] for it in items], [it[1] for it in items], crop_mode)
