"""On-disk cache of model-ready frames for training and evaluation.

Why: a training sample used to decode up to 340 raw frames and run frame_prep
on them to use 16, on every access, every epoch.  That CPU work bounded every
experiment.  This module prepares each video ONCE, in order, with the shared
front-end (prepared_frames.PreparedFrameReader, i.e. exactly what
/analyze/* and camera.py do) and stores the resulting 112x112 RGB frames plus
their timestamps and frame_prep flags.  Any window cut from the cache is
bit-identical to preparing it on the fly (see prepared_frames' docstring), so
training and evaluation numbers do not move because of caching.

Cache entries are keyed by
  * the SHA-1 of the video file's bytes (paths may differ between Colab, the
    team's laptops and Drive; content does not), and
  * a fingerprint of everything that determines the frames: the source of
    frame_prep / prepared_frames / clip_sampler / thresholds, the OpenCV
    version (decoders differ), target fps, max_seconds and storage format.
Changing any of those silently creates new entries instead of serving stale
frames.  Old entries are never deleted automatically; remove the directory to
reclaim space.

What is stored depends on the crop mode of the model's preprocessing
contract (preprocessing.py):
  crop="frame": the final 112x112 crop of every frame (prepare_for_model);
  crop="clip":  every frame's working image (shorter side <= 224) plus its
                motion box; CachedVideo.window crops each requested clip with
                frame_prep.build_clip, exactly as the live camera does.

Storage formats ("auto" = npy for crop="frame", jpg for crop="clip"):
  "npy" - raw uint8, loaded with mmap: exact and fastest.  About 37 KB per
          kept frame in frame mode (roughly 6 MB for a 10 s clip), ~270 KB
          in clip mode: keep it on the local disk (e.g. /content/frame_cache
          on Colab).
  "jpg" - JPEG at `jpeg_quality` (default 95): roughly 10-25x smaller, small
          enough for Google Drive, but LOSSY (pixels differ from what live
          mode sees, by a few grey levels on average and up to ~30 at hard
          edges).  Use for persistence across sessions when
          disk is the constraint; prefer "npy" for final runs.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from clip_sampler import (
    ScheduledWindow,
    detect_gap,
    aggregate_clip_flags,
    scoring_schedule,
    select_clip_indices,
)
from frame_prep import build_clip
from prepared_frames import PreparedFrameReader
from preprocessing import CROP_MODES
from thresholds import CAMERA_TARGET_FPS, CLIP_LENGTH, INFERENCE_STRIDE

log = logging.getLogger("safespace.frame_cache")

# Bump when the on-disk layout or the meaning of a stored field changes.
FORMAT_VERSION = 2
FORMATS = ("npy", "jpg")
FORMAT_CHOICES = ("auto",) + FORMATS
NO_BOX = (-1, -1, -1, -1)
# Modules whose code decides which frames are kept and what they look like.
FINGERPRINT_MODULES = ("frame_prep", "prepared_frames", "clip_sampler", "thresholds")


# ---------------------------------------------------------------- fingerprint
def _module_source_digest(name: str) -> str:
    mod = importlib.import_module(name)
    with open(mod.__file__, "rb") as f:
        src = f.read().replace(b"\r\n", b"\n")  # Windows checkouts hash the same
    return hashlib.sha1(src).hexdigest()


def prep_fingerprint(
    target_fps: float = CAMERA_TARGET_FPS,
    max_seconds: float = 0.0,
    fmt: str = "npy",
    jpeg_quality: int = 95,
    crop_mode: str = "frame",
) -> str:
    """Hex digest of every input that determines cached frames."""
    parts = [
        f"format_version={FORMAT_VERSION}",
        f"crop_mode={crop_mode}",
        f"cv2={cv2.__version__}",
        f"target_fps={float(target_fps)!r}",
        f"max_seconds={float(max_seconds or 0.0)!r}",
        f"fmt={fmt}",
        f"jpeg_quality={int(jpeg_quality) if fmt == 'jpg' else 0}",
    ]
    parts += [f"{m}={_module_source_digest(m)}" for m in FINGERPRINT_MODULES]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


_CONTENT_MEMO: Dict[Tuple[str, int, int], str] = {}


def content_sha1(path: str) -> str:
    """SHA-1 of a file's bytes, memoised per (path, size, mtime) in-process."""
    st = os.stat(path)
    memo_key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    hit = _CONTENT_MEMO.get(memo_key)
    if hit is not None:
        return hit
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _CONTENT_MEMO[memo_key] = digest
    return digest


# ---------------------------------------------------------------- cached video
class ClipSample(NamedTuple):
    frames: np.ndarray  # (CLIP_LENGTH, 112, 112, 3) uint8 RGB
    suitable: bool  # clip-level flag, same rule as ClipWindow.sample
    global_motion: bool  # clip-level flag, same rule as ClipWindow.sample
    t_end: float  # timestamp of the newest frame


class _LazyJpegFrames:
    """Array-like view over JPEG-encoded frames; decodes only what is indexed."""

    def __init__(self, blob: np.ndarray, offsets: np.ndarray, frame_shape: Tuple[int, ...]):
        self._blob = blob
        self._offsets = offsets  # len N+1
        self.shape = (len(offsets) - 1,) + tuple(frame_shape)
        self.dtype = np.dtype(np.uint8)

    def __len__(self) -> int:
        return self.shape[0]

    def _decode(self, i: int) -> np.ndarray:
        buf = self._blob[self._offsets[i] : self._offsets[i + 1]]
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"corrupt cached JPEG frame {i}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: Any) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            return self._decode(int(idx) % len(self))
        sel = np.arange(len(self))[idx]
        if sel.ndim == 0:
            return self._decode(int(sel))
        out = np.empty((len(sel),) + self.shape[1:], np.uint8)
        decoded: Dict[int, np.ndarray] = {}
        for k, i in enumerate(sel.tolist()):
            if i not in decoded:  # windows repeat frames on short clips
                decoded[i] = self._decode(i)
            out[k] = decoded[i]
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self[np.arange(len(self))]
        return arr if dtype is None else arr.astype(dtype)


FramesLike = Union[np.ndarray, _LazyJpegFrames]


@dataclass
class CachedVideo:
    """A whole video as prepared frames on the 15 fps camera timeline."""

    frames: FramesLike  # (N, H, W, 3) uint8 RGB: 112x112 crops (frame mode) or working frames (clip mode)
    t: np.ndarray  # (N,) float64 seconds, ascending
    suitable: np.ndarray  # (N,) bool, frame_prep suitable_for_model
    global_motion: np.ndarray  # (N,) bool, frame_prep global_motion
    native_fps: float
    source: str = ""
    fps: float = CAMERA_TARGET_FPS
    crop_mode: str = "frame"
    boxes: Optional[np.ndarray] = None  # (N, 4) int32 motion boxes, -1 rows = none (clip mode)
    _schedule: Optional[List[ScheduledWindow]] = field(default=None, repr=False)
    _ready: Optional[List[ScheduledWindow]] = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(len(self.t))

    @classmethod
    def from_reader(cls, reader: PreparedFrameReader) -> "CachedVideo":
        frames, ts, suit, glob, boxes = [], [], [], [], []
        for pf in reader:
            frames.append(pf.image)
            boxes.append(pf.box if pf.box is not None else NO_BOX)
            ts.append(pf.t)
            suit.append(pf.suitable)
            glob.append(pf.global_motion)
        if not frames:
            raise ValueError(f"no frames could be prepared from {reader.path}")
        return cls(
            frames=np.stack(frames),
            t=np.asarray(ts, np.float64),
            suitable=np.asarray(suit, bool),
            global_motion=np.asarray(glob, bool),
            native_fps=float(reader.native_fps),
            source=reader.path,
            fps=reader.target_fps,
            crop_mode=reader.crop_mode,
            boxes=np.asarray(boxes, np.int32).reshape(-1, 4),
        )

    # ---------------------------------------------------------- windows
    def schedule(self, stride_frames: int = INFERENCE_STRIDE) -> List[ScheduledWindow]:
        """Windows live mode would score on this video (clip_sampler.scoring_schedule)."""
        if stride_frames != INFERENCE_STRIDE:
            return scoring_schedule(self.t.tolist(), fps=self.fps, stride_frames=stride_frames)
        if self._schedule is None:
            self._schedule = scoring_schedule(self.t.tolist(), fps=self.fps)
        return self._schedule

    def ready_windows(self) -> List[ScheduledWindow]:
        """Every window end with a full clip of history (stride 0): the pool
        training samples from."""
        if self._ready is None:
            self._ready = scoring_schedule(self.t.tolist(), fps=self.fps, stride_frames=0)
        return self._ready

    def fallback_window(self) -> ScheduledWindow:
        """For a video shorter than one clip: the newest frame, padded by
        repeating the earliest frames (select_clip_indices clamps)."""
        seg_start, segment = 0, 0
        ts = self.t.tolist()
        for i in range(1, len(ts)):
            if detect_gap(ts[i - 1], ts[i]):
                seg_start, segment = i, segment + 1
        return ScheduledWindow(len(ts) - 1, seg_start, segment)

    def random_window(self, rng) -> ScheduledWindow:
        """Uniformly random ready window (rng: random.Random / module)."""
        pool = self.ready_windows()
        if not pool:
            return self.fallback_window()
        return pool[rng.randrange(len(pool))]

    def window(self, w: Union[ScheduledWindow, int], n: int = CLIP_LENGTH) -> ClipSample:
        """The clip ending at `w` exactly as ClipWindow.sample + clip assembly
        (prepared_frames.clip_from_items) would build it live."""
        if isinstance(w, ScheduledWindow):
            end, seg_start = w.end, w.seg_start
        else:
            end, seg_start = int(w), 0
        if not 0 <= end < len(self):
            raise IndexError(f"window end {end} out of range for {len(self)} frames")
        t_end = float(self.t[end])
        span = (n - 1) / self.fps
        # Frames older than one span (+1 s margin) can never be the nearest
        # match for a target time, so only that tail is searched.
        lo = max(seg_start, int(np.searchsorted(self.t, t_end - span - 1.0, side="left")))
        local = select_clip_indices(self.t[lo : end + 1].tolist(), n, self.fps, t_end)
        idx = np.asarray(local, np.int64) + lo
        uniq = np.unique(idx)
        suitable, global_motion = aggregate_clip_flags(
            self.suitable[uniq].tolist(), self.global_motion[uniq].tolist()
        )
        boxes = self.boxes[idx] if self.boxes is not None else [None] * len(idx)
        frames = build_clip(np.asarray(self.frames[idx]), boxes, self.crop_mode)
        return ClipSample(frames, suitable, global_motion, t_end)


def extract_video(
    path: str,
    target_fps: float = CAMERA_TARGET_FPS,
    max_seconds: float = 0.0,
    crop_mode: str = "frame",
) -> CachedVideo:
    """Prepare a whole video in memory (no disk cache)."""
    return CachedVideo.from_reader(
        PreparedFrameReader(path, target_fps=target_fps, max_seconds=max_seconds,
                            crop_mode=crop_mode)
    )


# ---------------------------------------------------------------- disk cache
@dataclass
class BuildReport:
    keys: Dict[str, str]  # video path -> cache key
    failed: Dict[str, str]  # video path -> error message
    built: int
    reused: int
    seconds: float

    def __str__(self) -> str:
        return (
            f"{len(self.keys)} cached ({self.built} built, {self.reused} reused), "
            f"{len(self.failed)} failed, {self.seconds:.0f}s"
        )


class FrameCache:
    """Directory of prepared videos.  Safe to share between processes
    (entries are written atomically; the meta file is the completion marker)
    and picklable, so DataLoader workers can each hold a copy."""

    def __init__(
        self,
        root: str,
        fmt: str = "auto",
        jpeg_quality: int = 95,
        target_fps: float = CAMERA_TARGET_FPS,
        max_seconds: float = 0.0,
        crop_mode: str = "frame",
    ):
        if crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
        if fmt == "auto":
            fmt = "npy" if crop_mode == "frame" else "jpg"
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMAT_CHOICES}, got {fmt!r}")
        self.crop_mode = crop_mode
        self.root = root
        self.fmt = fmt
        self.jpeg_quality = int(jpeg_quality)
        self.target_fps = float(target_fps)
        self.max_seconds = float(max_seconds or 0.0)
        self._fingerprint: Optional[str] = None
        os.makedirs(root, exist_ok=True)

    # ---------------------------------------------------------- naming
    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = prep_fingerprint(
                self.target_fps, self.max_seconds, self.fmt, self.jpeg_quality, self.crop_mode
            )
        return self._fingerprint

    def key_for(self, video_path: str) -> str:
        return f"{content_sha1(video_path)[:20]}-{self.fingerprint[:12]}"

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.meta.npz")

    def _frames_path(self, key: str) -> str:
        ext = "frames.npy" if self.fmt == "npy" else "jpg.npz"
        return os.path.join(self.root, f"{key}.{ext}")

    def has_key(self, key: str) -> bool:
        return os.path.isfile(self._meta_path(key)) and os.path.isfile(self._frames_path(key))

    def contains(self, video_path: str) -> bool:
        return self.has_key(self.key_for(video_path))

    # ---------------------------------------------------------- build / load
    def ensure(self, video_path: str) -> Tuple[str, bool]:
        """Cache `video_path` if needed.  Returns (key, built_now)."""
        key = self.key_for(video_path)
        if self.has_key(key):
            return key, False
        video = extract_video(video_path, self.target_fps, self.max_seconds, self.crop_mode)
        self._write(key, video)
        return key, True

    def get(self, video_path: str, mmap: bool = True) -> CachedVideo:
        key, _ = self.ensure(video_path)
        return self.load(key, mmap=mmap)

    def load(self, key: str, mmap: bool = True) -> CachedVideo:
        with np.load(self._meta_path(key), allow_pickle=False) as m:
            meta = {k: m[k] for k in m.files}
        if int(meta["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"cache entry {key} has format {int(meta['format_version'])}")
        if str(meta["crop_mode"]) != self.crop_mode:
            raise ValueError(f"cache entry {key} is crop_mode={meta['crop_mode']}, cache is {self.crop_mode}")
        if self.fmt == "npy":
            frames: FramesLike = np.load(
                self._frames_path(key), mmap_mode="r" if mmap else None, allow_pickle=False
            )
        else:
            with np.load(self._frames_path(key), allow_pickle=False) as z:
                frames = _LazyJpegFrames(z["blob"], z["offsets"], tuple(z["frame_shape"]))
        return CachedVideo(
            frames=frames,
            t=meta["t"],
            suitable=meta["suitable"],
            global_motion=meta["global_motion"],
            native_fps=float(meta["native_fps"]),
            source=str(meta["source"]),
            fps=float(meta["fps"]),
            crop_mode=str(meta["crop_mode"]),
            boxes=meta["boxes"],
        )

    def _write(self, key: str, video: CachedVideo) -> None:
        frames = np.asarray(video.frames)
        if self.fmt == "npy":
            _atomic_write(self._frames_path(key), lambda f: np.save(f, frames, allow_pickle=False))
        else:
            chunks, offsets = [], [0]
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            for rgb in frames:
                ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                chunks.append(enc.reshape(-1))
                offsets.append(offsets[-1] + enc.size)
            blob = np.concatenate(chunks).astype(np.uint8)
            _atomic_write(
                self._frames_path(key),
                lambda f: np.savez(
                    f,
                    blob=blob,
                    offsets=np.asarray(offsets, np.int64),
                    frame_shape=np.asarray(frames.shape[1:], np.int64),
                ),
            )
        # Meta last: its presence marks the entry complete.
        _atomic_write(
            self._meta_path(key),
            lambda f: np.savez(
                f,
                format_version=np.int64(FORMAT_VERSION),
                t=video.t.astype(np.float64),
                suitable=video.suitable.astype(bool),
                global_motion=video.global_motion.astype(bool),
                native_fps=np.float64(video.native_fps),
                fps=np.float64(video.fps),
                source=np.str_(video.source),
                crop_mode=np.str_(video.crop_mode),
                boxes=(video.boxes if video.boxes is not None
                       else np.full((len(video), 4), -1, np.int32)).astype(np.int32),
            ),
        )

    def build_many(
        self, paths: Sequence[str], workers: int = 0, log_every: int = 100
    ) -> BuildReport:
        """Cache every path (idempotent).  workers > 1 uses processes started
        with "spawn": forking a process that has already used OpenCV can
        deadlock the child (OpenCV's thread pool does not survive fork), and
        notebooks always have.  Spawned workers re-import this module, so
        scripts must call this under `if __name__ == "__main__":`."""
        t0 = time.time()
        keys: Dict[str, str] = {}
        failed: Dict[str, str] = {}
        built = reused = 0
        unique = list(dict.fromkeys(paths))

        def record(path: str, key: Optional[str], was_built: bool, err: Optional[str]) -> None:
            nonlocal built, reused
            if err is not None:
                failed[path] = err
                log.warning("frame cache: %s failed: %s", path, err)
            else:
                keys[path] = key  # type: ignore[assignment]
                built += int(was_built)
                reused += int(not was_built)
            done = len(keys) + len(failed)
            if log_every and (done % log_every == 0 or done == len(unique)):
                log.info("frame cache: %d/%d videos (%d built, %d reused, %d failed) %.0fs",
                         done, len(unique), built, reused, len(failed), time.time() - t0)

        if workers and workers > 1 and len(unique) > 1:
            cfg = (self.root, self.fmt, self.jpeg_quality, self.target_fps, self.max_seconds,
                   self.crop_mode)
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_worker_init) as ex:
                futs = {ex.submit(_build_one, cfg, p): p for p in unique}
                for fut in as_completed(futs):
                    record(*fut.result())
        else:
            for p in unique:
                record(*_build_one_with(self, p))
        return BuildReport(keys, failed, built, reused, time.time() - t0)


# ---------------------------------------------------------------- helpers
def _atomic_write(final_path: str, write_fn) -> None:
    tmp = f"{final_path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
        os.replace(tmp, final_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _worker_init() -> None:
    cv2.setNumThreads(1)  # one process per core; avoid oversubscription


def _build_one_with(cache: FrameCache, path: str) -> Tuple[str, Optional[str], bool, Optional[str]]:
    try:
        key, was_built = cache.ensure(path)
        return path, key, was_built, None
    except Exception as e:  # one bad file must not abort a 3k-video build
        return path, None, False, f"{type(e).__name__}: {e}"


def _build_one(cfg: Tuple[Any, ...], path: str) -> Tuple[str, Optional[str], bool, Optional[str]]:
    root, fmt, quality, fps, max_s, crop_mode = cfg
    return _build_one_with(FrameCache(root, fmt, quality, fps, max_s, crop_mode), path)


__all__ = [
    "BuildReport",
    "CachedVideo",
    "ClipSample",
    "FrameCache",
    "content_sha1",
    "extract_video",
    "prep_fingerprint",
]
