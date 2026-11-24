"""Pure-Python timestamp-based clip sampling (no numpy/cv2/torch).

The model was trained on clips at an effective 15 fps, so a 16-frame clip
must always span ~1 second of *real time* regardless of how fast the camera
actually delivers frames.  We therefore pick, for each of the 16 target times
t_end - k/15 (k = 15..0), the stored frame whose timestamp is closest.  A slow
camera yields repeated frames (time is preserved); a fast one skips frames.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from typing import Any, Deque, List, NamedTuple, Optional, Sequence, Tuple

from thresholds import (
    CAMERA_TARGET_FPS,
    CLIP_FRACTION,
    CLIP_LENGTH,
    INFERENCE_STRIDE,
    MAX_FRAME_GAP_S,
)

# A clip is "ready" once the stored history reaches back to its first target
# time, give or take this much (ClipWindow and scoring_schedule share it).
START_TOLERANCE_S = 0.1


def select_clip_indices(
    timestamps: Sequence[float],
    n: int = CLIP_LENGTH,
    fps: float = CAMERA_TARGET_FPS,
    t_end: Optional[float] = None,
) -> Optional[List[int]]:
    """Indices (len n, non-decreasing, may repeat) of the frames nearest to
    t_end - k/fps for k = n-1..0.  `timestamps` must be ascending."""
    if not timestamps:
        return None
    if t_end is None:
        t_end = timestamps[-1]
    out: List[int] = []
    last = len(timestamps) - 1
    for k in range(n - 1, -1, -1):
        target = t_end - k / fps
        i = bisect_left(timestamps, target)
        if i <= 0:
            best = 0
        elif i > last:
            best = last
        else:
            best = i if (timestamps[i] - target) < (target - timestamps[i - 1]) else i - 1
        out.append(best)
    return out


def detect_gap(prev_ts: Optional[float], ts: float, max_gap: float = MAX_FRAME_GAP_S) -> bool:
    """True when the interval between consecutive frames is a discontinuity."""
    return prev_ts is not None and (ts - prev_ts) > max_gap


def window_coverage(
    t_ends: Sequence[float],
    window_s: float = (CLIP_LENGTH - 1) / CAMERA_TARGET_FPS,
    horizon_s: float = 20.0,
) -> Optional[float]:
    """Percent (0-100) of the last `horizon_s` of wall time that lies inside at
    least one scored window [t_end - window_s, t_end].  None if <2 windows."""
    if len(t_ends) < 2:
        return None
    ends = sorted(t_ends)
    hi = ends[-1]
    lo = max(ends[0] - window_s, hi - horizon_s)
    total = hi - lo
    if total <= 0:
        return None
    covered = 0.0
    cur_s: Optional[float] = None
    cur_e = 0.0
    for e in ends:
        s = max(e - window_s, lo)
        if e <= lo:
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        covered += cur_e - cur_s
    return round(min(100.0, 100.0 * covered / total), 1)


def aggregate_clip_flags(
    suitable: Sequence[bool], global_motion: Sequence[bool]
) -> Tuple[bool, bool]:
    """Clip-level (suitable, global_motion) from the per-frame flags of the
    clip's UNIQUE frames: each is True when at least CLIP_FRACTION of them are.
    Shared by ClipWindow.sample (live) and frame_cache (training/eval)."""
    n_s = len(suitable)
    n_g = len(global_motion)
    return (
        sum(bool(x) for x in suitable) >= CLIP_FRACTION * n_s,
        sum(bool(x) for x in global_motion) >= CLIP_FRACTION * n_g,
    )


class ScheduledWindow(NamedTuple):
    """One window that live mode would score on a recorded timeline."""

    end: int  # index of the newest frame of the clip
    seg_start: int  # first frame index of the continuous segment it lies in
    segment: int  # 0-based segment number (bumps after every frame gap)


def scoring_schedule(
    timestamps: Sequence[float],
    n: int = CLIP_LENGTH,
    fps: float = CAMERA_TARGET_FPS,
    stride_frames: int = INFERENCE_STRIDE,
    max_gap: float = MAX_FRAME_GAP_S,
    start_tolerance_s: float = START_TOLERANCE_S,
) -> List[ScheduledWindow]:
    """Which windows of an already-recorded (ascending) frame timeline live
    mode would score, replicating video_calibration.analyze_windows exactly:

      * a frame gap > max_gap starts a new segment (ClipWindow is cleared and
        the scoring clock restarts);
      * a window becomes available once the segment reaches back one clip span
        (ClipWindow.sample's readiness rule);
      * after a scored window, the next one is scored once at least
        stride_frames / fps seconds have passed.
    """
    span = (n - 1) / float(fps)
    stride_s = stride_frames / float(fps)
    out: List[ScheduledWindow] = []
    seg_start = 0
    segment = 0
    last_scored: Optional[float] = None
    prev: Optional[float] = None
    for i, t in enumerate(timestamps):
        if detect_gap(prev, t, max_gap):
            seg_start = i
            segment += 1
            last_scored = None
        prev = t
        if timestamps[seg_start] > t - span + start_tolerance_s:
            continue  # not enough history yet
        if last_scored is not None and (t - last_scored) < stride_s - 1e-6:
            continue
        last_scored = t
        out.append(ScheduledWindow(i, seg_start, segment))
    return out


class FpsResampler:
    """Decide which frames of a fixed-fps file to keep so the kept stream looks
    like a live camera paced at `target_fps` (used by the offline tools)."""

    def __init__(self, target_fps: float = CAMERA_TARGET_FPS):
        self.dt = 1.0 / float(target_fps)
        self.next_t: Optional[float] = None

    def keep(self, t: float, native_dt: float) -> bool:
        half = 0.5 * native_dt
        if self.next_t is None:
            self.next_t = t + self.dt
            return True
        if t >= self.next_t - half:
            self.next_t = max(self.next_t + self.dt, t + 0.5 * self.dt)
            return True
        return False


class ClipWindow:
    """Rolling timestamped frame store + clip-level flag aggregation."""

    def __init__(
        self,
        n: int = CLIP_LENGTH,
        fps: float = CAMERA_TARGET_FPS,
        retain_s: float = 3.0,
        max_items: int = 128,
        max_gap: float = MAX_FRAME_GAP_S,
        start_tolerance_s: float = START_TOLERANCE_S,
    ):
        self.n = n
        self.fps = float(fps)
        self.retain_s = retain_s
        self.max_gap = max_gap
        self.tol = start_tolerance_s
        self.ts: Deque[float] = deque(maxlen=max_items)
        self.items: Deque[Any] = deque(maxlen=max_items)
        self.suitable: Deque[bool] = deque(maxlen=max_items)
        self.global_motion: Deque[bool] = deque(maxlen=max_items)

    def clear(self) -> None:
        self.ts.clear()
        self.items.clear()
        self.suitable.clear()
        self.global_motion.clear()

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def last_ts(self) -> Optional[float]:
        return self.ts[-1] if self.ts else None

    def add(self, ts: float, item: Any, suitable: bool = True, global_motion: bool = False) -> bool:
        """Append a frame.  Returns True if a gap was detected, in which case
        the window was cleared first (the new frame starts a fresh clip)."""
        gap = detect_gap(self.last_ts, ts, self.max_gap)
        if gap:
            self.clear()
        self.ts.append(ts)
        self.items.append(item)
        self.suitable.append(bool(suitable))
        self.global_motion.append(bool(global_motion))
        while self.ts and self.ts[0] < ts - self.retain_s:
            self.ts.popleft()
            self.items.popleft()
            self.suitable.popleft()
            self.global_motion.popleft()
        return gap

    def available_in_span(self) -> int:
        """Frames inside the last clip span (for UI 'buffer_fill')."""
        if not self.ts:
            return 0
        lo = self.ts[-1] - (self.n - 1) / self.fps - self.tol
        return sum(1 for t in self.ts if t >= lo)

    def sample(self) -> Optional[dict]:
        if not self.ts:
            return None
        ts = list(self.ts)
        t_end = ts[-1]
        span = (self.n - 1) / self.fps
        if ts[0] > t_end - span + self.tol:
            return None
        idx = select_clip_indices(ts, self.n, self.fps, t_end)
        if idx is None:
            return None
        uniq = sorted(set(idx))
        suitable, global_motion = aggregate_clip_flags(
            [self.suitable[i] for i in uniq], [self.global_motion[i] for i in uniq]
        )
        picked = [ts[i] for i in idx]
        return {
            "indices": idx,
            "items": [self.items[i] for i in idx],
            "t_start": picked[0],
            "t_end": t_end,
            "span_s": picked[-1] - picked[0],
            "unique_frames": len(uniq),
            "suitable": suitable,
            "global_motion": global_motion,
        }
