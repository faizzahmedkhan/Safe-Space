import cv2
import logging
import queue
import time
import threading
import socket
import http.client
import urllib.request
import urllib.error
import numpy as np
from collections import deque
from datetime import datetime
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from frame_prep import prepare_for_model
from clip_sampler import ClipWindow, detect_gap
from thresholds import (
  CAMERA_TARGET_FPS,
  CLIP_LENGTH,
  DEGRADED_FPS,
  MAX_FRAME_GAP_S,
)

log = logging.getLogger("safespace.camera")

EVIDENCE_BUFFER_LEN = 60
# Frozen-stream detection: mean abs diff of a tiny grey frame below this for
# FROZEN_SECONDS means the camera is repeating the same image.
FROZEN_DIFF = 0.5
FROZEN_SECONDS = 3.0
# Writer queue bound (frames).  If the disk/encoder cannot keep up, frames are
# dropped from the evidence file rather than slowing capture (which would
# also corrupt the model's time base).
EVIDENCE_QUEUE_FRAMES = 120
HTTP_USER_AGENT = "SafeSpace/1.0 (IP-Webcam client)"


def _is_http_url(src) -> bool:
  return isinstance(src, str) and src.lower().startswith(("http://", "https://"))


def _origin(url: str) -> str:
  parsed = urlparse(url)
  return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def diagnose_http_camera(url: str, timeout: float = 4.0) -> dict:
  """
  Probe whether a phone IP Webcam URL is reachable.
  Returns {ok, stage, message, detail}.
  """
  parsed = urlparse(url)
  host = parsed.hostname
  port = parsed.port or (443 if parsed.scheme == "https" else 80)
  if not host:
    return {
      "ok": False,
      "stage": "parse",
      "message": "Invalid camera URL.",
      "detail": url,
    }

  try:
    with socket.create_connection((host, port), timeout=timeout):
      pass
  except OSError as e:
    return {
      "ok": False,
      "stage": "tcp",
      "message": (
        f"Cannot reach {host}:{port} ({e}). "
        "Confirm the phone IP in IP Webcam, that Start server is on, "
        "and both devices are on the same Wi‑Fi (not guest/VPN)."
      ),
      "detail": str(e),
    }

  origin = _origin(url)
  candidates = [url]
  shot = f"{origin}/shot.jpg"
  if shot not in candidates:
    candidates.append(shot)

  last_error = None
  for candidate in candidates:
    try:
      req = urllib.request.Request(
        candidate,
        headers={"User-Agent": HTTP_USER_AGENT, "Connection": "close"},
      )
      with urllib.request.urlopen(req, timeout=timeout) as resp:
        chunk = resp.read(2048)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if chunk or "image" in ctype or "multipart" in ctype or "octet" in ctype:
          return {
            "ok": True,
            "stage": "http",
            "message": f"HTTP OK via {candidate}",
            "detail": ctype,
            "working_url": candidate,
          }
        last_error = f"{candidate} returned empty body"
    except Exception as e:
      last_error = f"{candidate}: {type(e).__name__}: {e}"

  return {
    "ok": False,
    "stage": "http",
    "message": (
      f"Port {port} is open on {host}, but this Python process cannot complete HTTP "
      f"({last_error}). On macOS this usually means Local Network is blocked for Cursor. "
      "FIX: quit the backend in Cursor, then double-click start_backend.command "
      "(opens Terminal.app) — Safari can reach the phone, but Cursor often cannot. "
      f"Also close tabs viewing {origin}, then Connect again from the dashboard."
    ),
    "detail": last_error,
  }


class _HttpFrameSource:
  """Pull frames from IP Webcam via /shot.jpg (reliable) or MJPEG /video."""

  def __init__(self, url: str, target_fps: int = 15):
    self.url = url
    self.target_fps = target_fps
    self._opened = False
    self._mode = None  # "shot" | "mjpeg"
    self._mjpeg_resp = None
    self._mjpeg_fp = None
    self._boundary = None
    self._lock = threading.Lock()
    # Persistent keep-alive connection for "shot" mode. Without this, every
    # single frame paid a fresh TCP+HTTP handshake (urllib defaulted to
    # "Connection: close"), which is the dominant source of per-frame
    # latency for phone cameras on Wi-Fi, and gets worse linearly with more
    # phone cameras attached. One connection is reused for the life of this
    # source and only rebuilt if the server drops it.
    self._conn = None
    self._conn_host = None
    self._conn_port = None

  def open(self) -> Tuple[bool, Optional[str]]:
    origin = _origin(self.url)
    shot_url = f"{origin}/shot.jpg"
    # Prefer single JPEG snapshots — far more reliable than OpenCV+MJPEG on macOS
    err = self._try_shot(shot_url)
    if err is None:
      self.url = shot_url
      self._mode = "shot"
      self._opened = True
      # Warmup/drain stale frames from phone buffer
      for _ in range(5):
        try:
          self._download_jpeg(shot_url)
        except Exception:
          pass
      return True, None

    err2 = self._try_mjpeg(self.url)
    if err2 is None:
      self._mode = "mjpeg"
      self._opened = True
      # Warmup/drain stale frames
      for _ in range(5):
        try:
          self._read_mjpeg_frame()
        except Exception:
          pass
      return True, None

    return False, (
      f"HTTP camera open failed. shot.jpg: {err}; mjpeg: {err2}"
    )

  def _try_shot(self, shot_url: str) -> Optional[str]:
    try:
      frame = self._download_jpeg(shot_url)
      if frame is None:
        return "empty/invalid JPEG"
      return None
    except Exception as e:
      return f"{type(e).__name__}: {e}"

  def _try_mjpeg(self, url: str) -> Optional[str]:
    try:
      req = urllib.request.Request(
        url, headers={"User-Agent": HTTP_USER_AGENT}
      )
      resp = urllib.request.urlopen(req, timeout=8)
      ctype = (resp.headers.get("Content-Type") or "").lower()
      if "multipart" not in ctype and "image" not in ctype:
        # Still try — some servers omit content-type
        pass
      boundary = None
      if "boundary=" in ctype:
        boundary = ctype.split("boundary=", 1)[1].strip().strip('"')
      self._mjpeg_resp = resp
      self._mjpeg_fp = resp
      self._boundary = boundary
      # Warm up one frame
      frame = self._read_mjpeg_frame()
      if frame is None:
        self.close()
        return "could not decode first MJPEG frame"
      return None
    except Exception as e:
      self.close()
      return f"{type(e).__name__}: {e}"

  def _ensure_conn(self, url: str):
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if self._conn is None or self._conn_host != host or self._conn_port != port:
      self._close_conn()
      cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
      self._conn = cls(host, port, timeout=5)
      self._conn_host = host
      self._conn_port = port
    return parsed

  def _close_conn(self):
    if self._conn is not None:
      try:
        self._conn.close()
      except Exception:
        pass
    self._conn = None
    self._conn_host = None
    self._conn_port = None

  def _request_path(self, parsed) -> str:
    path = parsed.path or "/"
    if parsed.query:
      path += "?" + parsed.query
    return path

  def _download_jpeg(self, url: str):
    parsed = self._ensure_conn(url)
    path = self._request_path(parsed)
    headers = {"User-Agent": HTTP_USER_AGENT, "Connection": "keep-alive"}
    try:
      self._conn.request("GET", path, headers=headers)
      resp = self._conn.getresponse()
      data = resp.read()
    except Exception:
      # Server likely closed the keep-alive connection (idle timeout, etc).
      # Reopen once and retry before giving up — this keeps the fast path
      # (reused connection) as the common case without failing hard on the
      # occasional server-side close.
      self._close_conn()
      parsed = self._ensure_conn(url)
      path = self._request_path(parsed)
      self._conn.request("GET", path, headers=headers)
      resp = self._conn.getresponse()
      data = resp.read()
    if not data:
      return None
    arr = cv2.imdecode(
      np.frombuffer(data, dtype=np.uint8),
      cv2.IMREAD_COLOR,
    )
    return arr

  def _read_mjpeg_frame(self):
    fp = self._mjpeg_fp
    if fp is None:
      return None

    # Read until JPEG SOI/EOI markers
    buf = b""
    while True:
      chunk = fp.read(4096)
      if not chunk:
        return None
      buf += chunk
      start = buf.find(b"\xff\xd8")
      if start < 0:
        if len(buf) > 256_000:
          buf = buf[-64_000:]
        continue
      end = buf.find(b"\xff\xd9", start + 2)
      if end < 0:
        if len(buf) > 2_000_000:
          buf = buf[start:]
        continue
      jpg = buf[start : end + 2]
      buf = buf[end + 2 :]
      frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
      return frame

  def read(self):
    """Returns (grabbed, frame) like VideoCapture.read()."""
    if not self._opened:
      return False, None
    try:
      if self._mode == "shot":
        frame = self._download_jpeg(self.url)
        return (frame is not None), frame
      frame = self._read_mjpeg_frame()
      return (frame is not None), frame
    except Exception as e:
      log.warning("HTTP frame read error: %s", e)
      return False, None

  def isOpened(self) -> bool:
    return self._opened

  def grab(self):
    # No internal buffer to drain for shot mode
    return True

  def release(self):
    self.close()

  def close(self):
    self._opened = False
    if self._mjpeg_resp is not None:
      try:
        self._mjpeg_resp.close()
      except Exception:
        pass
    self._mjpeg_resp = None
    self._mjpeg_fp = None
    self._close_conn()


class _LatestFrameReader:
  """A2: reads an OpenCV network stream (RTSP/MJPEG) continuously on its own
  thread and only keeps the newest frame.  The old code did grab() x5 + read()
  per loop, which discarded 4 of every 5 frames and made a clip span ~2.7 s."""

  def __init__(self, cap):
    self._cap = cap
    self._cond = threading.Condition()
    self._frame = None
    self._ts = None
    self._seq = 0
    self._last_seq = 0
    self._fail = 0
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    while not self._stop.is_set():
      try:
        ok, frame = self._cap.read()
      except Exception:
        ok, frame = False, None
      if ok and frame is not None:
        ts = time.monotonic()
        with self._cond:
          self._frame, self._ts = frame, ts
          self._seq += 1
          self._fail = 0
          self._cond.notify_all()
      else:
        with self._cond:
          self._fail += 1
        time.sleep(0.05)

  def read(self, timeout: float = 1.0):
    """Returns (grabbed, frame, ts).  grabbed=False if no NEW frame arrived."""
    with self._cond:
      self._cond.wait_for(
        lambda: self._seq != self._last_seq or self._stop.is_set(), timeout=timeout
      )
      if self._seq == self._last_seq:
        return False, None, None
      self._last_seq = self._seq
      return True, self._frame, self._ts

  def isOpened(self) -> bool:
    return (not self._stop.is_set()) and self._cap.isOpened()

  def release(self):
    self._stop.set()
    with self._cond:
      self._cond.notify_all()
    self._thread.join(timeout=2.0)
    try:
      self._cap.release()
    except Exception:
      pass


class _EvidenceRecorder:
  """A4: evidence is encoded on its own thread, fed through a bounded queue,
  so VideoWriter latency never slows the capture loop."""

  def __init__(self, path, size, fps, preroll, on_done):
    self.path = path
    self.size = size
    self.fps = fps
    self._preroll = preroll
    self._on_done = on_done
    self._q = queue.Queue(maxsize=EVIDENCE_QUEUE_FRAMES)
    self._finish = threading.Event()
    self.dropped = 0
    self.written = 0
    self.opened = threading.Event()
    self._ok = False
    self._thread = threading.Thread(target=self._run, daemon=True)

  def start(self):
    self._thread.start()

  def push(self, frame) -> bool:
    try:
      self._q.put_nowait(frame)
      return True
    except queue.Full:
      self.dropped += 1
      return False

  def finish(self):
    self._finish.set()

  def _run(self):
    writer = None
    ok = False
    try:
      # mp4v is reliable in OpenCV; it is transcoded to H.264 afterwards.
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
      if not writer.isOpened():
        log.error("Failed to open VideoWriter for %s", self.path)
        writer = None
        return
      for f in self._preroll:
        writer.write(f)
        self.written += 1
      self._preroll = None
      while True:
        try:
          frame = self._q.get(timeout=0.25)
        except queue.Empty:
          if self._finish.is_set():
            break
          continue
        writer.write(frame)
        self.written += 1
      ok = self.written > 0
    except Exception as e:
      log.error("Evidence writer error for %s: %s", self.path, e)
    finally:
      if writer is not None:
        try:
          writer.release()
        except Exception:
          pass
      if self.dropped:
        log.warning("Evidence %s: dropped %d frame(s) (encoder too slow)", self.path, self.dropped)
      try:
        self._on_done(self.path, ok)
      except Exception as e:
        log.error("Evidence completion callback failed: %s", e)


def _clamp(v, lo, hi):
  return max(lo, min(hi, v))


class CameraStream:
  def __init__(self, src=0, target_fps=CAMERA_TARGET_FPS, show_preview=True, soft_fail=True):
    """
    src=0 opens the default laptop webcam.
    src="http://..." opens a network camera stream (HTTP IP Webcam preferred).
    soft_fail=True: failed open leaves camera inactive instead of raising.
    """
    self.target_fps = target_fps
    self.show_preview = show_preview
    self.preview_enabled = show_preview
    self.soft_fail = soft_fail

    # Timestamped inference window (A1).  Each entry is one prepared 112x112
    # RGB frame with its real arrival time (time.monotonic()).
    self.window = ClipWindow(n=CLIP_LENGTH, fps=CAMERA_TARGET_FPS)
    self.evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LEN)
    self.max_evidence_mb = 150
    self.evidence_buffer_initialized = False

    # Rolling arrival timestamps -> real fps.
    self._loop_timestamps = deque(maxlen=30)
    self._fps_warn_counter = 0
    self.degraded = False

    self._lock = threading.Lock()
    self._thread = None

    self.frame_count = 0
    self._recorder: Optional[_EvidenceRecorder] = None
    self._recording_path = None
    # Called from the writer thread as cb(path, ok) once the file is closed.
    self.on_recording_finished: Optional[Callable[[str, bool], None]] = None
    self.last_frame_time = None      # time.monotonic() of newest frame
    self.open_error = None
    self._prev_gray = None
    self._crop_state: dict = {}
    # A3/C7: bumped on every discontinuity (grab failure, frame gap, restart).
    self.generation = 0
    self.last_reset_reason = "init"
    # E3: frozen-stream detection
    self._prev_small = None
    self._frozen_since = None
    self.frozen = False
    # E2/E3: wall-clock time (epoch) the camera stopped delivering frames.
    self.offline_since: Optional[float] = None
    self.scene_meta = {
        "suitable_for_model": True,
        "subject_area_frac": 0.0,
        "method": "full_frame",
        "note": "",
        "global_motion": False,
      }

    self.stream = None
    self.src = src
    self.is_network_stream = False
    self.stopped = True

    self._open_source(src)

  # ------------------------------------------------------------------ open
  def _open_source(self, src):
    """Open capture for src. Returns True on success."""
    if src is None:
      self.src = None
      self.is_network_stream = False
      self.stream = None
      self.stopped = True
      self.open_error = None
      return False

    self.src = src
    self.is_network_stream = _is_http_url(src) or (
      isinstance(src, str) and "://" in src
    )
    self.open_error = None

    if _is_http_url(src):
      diag = diagnose_http_camera(src)
      if not diag["ok"]:
        self.stream = None
        self.stopped = True
        self.open_error = diag["message"]
        log.warning("%s", self.open_error)
        if not self.soft_fail:
          raise RuntimeError(self.open_error)
        return False

      http_src = _HttpFrameSource(src, target_fps=self.target_fps)
      ok, err = http_src.open()
      if not ok:
        self.stream = None
        self.stopped = True
        self.open_error = err or f"Failed to open HTTP camera {src}"
        log.warning("%s", self.open_error)
        if not self.soft_fail:
          raise RuntimeError(self.open_error)
        return False

      self.stream = http_src
      log.info("HTTP camera opened (%s) via %s", http_src._mode, http_src.url)
      self.stopped = False
      return True

    # Local webcam / RTSP via OpenCV
    stream = cv2.VideoCapture(src)
    if not stream.isOpened():
      stream.release()
      self.stream = None
      self.stopped = True
      self.open_error = f"Failed to open camera with src={src}"
      log.warning("%s", self.open_error)
      if not self.soft_fail:
        raise RuntimeError(self.open_error)
      return False

    if self.is_network_stream:
      try:
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      except Exception:
        pass
      # A2: dedicated latest-frame reader thread.
      self.stream = _LatestFrameReader(stream)
      log.info("Network stream detected (OpenCV). Using latest-frame reader thread.")
    else:
      self.stream = stream

    self.stopped = False
    return True

  def start(self):
    """Starts the background thread to continuously read frames."""
    if self.stream is None or not self.stream.isOpened():
      log.info("Camera not open - start() skipped.")
      return self
    if self._thread is not None and self._thread.is_alive():
      return self
    self.stopped = False
    log.info("Starting camera thread...")
    self._thread = threading.Thread(target=self.update, args=(), daemon=True)
    self._thread.start()
    return self

  def _reset_continuity(self, reason: str):
    """A3: forget everything that spans a discontinuity so old and new
    footage are never glued into one clip."""
    with self._lock:
      self.window.clear()
      self._prev_gray = None
      self._crop_state = {}
      self._prev_small = None
      self._frozen_since = None
      self.frozen = False
      self._loop_timestamps.clear()
      self.generation += 1
      self.last_reset_reason = reason

  def restart(self, src):
    """
    Stop the current stream, open a new source, clear buffers, and start capturing.
    Returns (success: bool, error: str | None).
    """
    log.info("Restarting camera with src=%s", src)
    old_thread = self._thread
    self.stop()
    if old_thread is not None and old_thread.is_alive():
      old_thread.join(timeout=2.0)

    self._reset_continuity("restart")
    with self._lock:
      self.evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LEN)
      self.evidence_buffer_initialized = False
      self.frame_count = 0
      self.last_frame_time = None
      self._fps_warn_counter = 0
      self.degraded = False
      self.offline_since = None
      self.scene_meta = {
        "suitable_for_model": True,
        "subject_area_frac": 0.0,
        "method": "full_frame",
        "note": "",
        "global_motion": False,
      }

    if not self._open_source(src):
      return False, self.open_error

    self.start()
    return True, None

  # --------------------------------------------------------------- capture
  def _read_frame(self, stream):
    """Returns (grabbed, frame, arrival_ts_monotonic)."""
    if isinstance(stream, _LatestFrameReader):
      return stream.read(timeout=1.0)
    grabbed, frame = stream.read()
    return grabbed, frame, time.monotonic()

  def _update_frozen(self, frame, ts):
    small = cv2.cvtColor(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    prev = self._prev_small
    self._prev_small = small
    if prev is None:
      return
    diff = float(np.mean(cv2.absdiff(prev, small)))
    if diff < FROZEN_DIFF:
      if self._frozen_since is None:
        self._frozen_since = ts
      elif ts - self._frozen_since >= FROZEN_SECONDS:
        self.frozen = True
    else:
      self._frozen_since = None
      self.frozen = False

  def update(self):
    """This function runs on an infinite loop in the background."""
    sleep_time = 1.0 / self.target_fps
    consecutive_failures = 0
    max_consecutive_failures = 30
    next_deadline = time.perf_counter()
    last_ts = None

    while not self.stopped:
      stream = self.stream
      if stream is None or not stream.isOpened():
        log.warning("Stream closed unexpectedly.")
        if self.offline_since is None:
          self.offline_since = time.time()
        self.stop()
        break

      grabbed, frame, ts = self._read_frame(stream)

      if not grabbed or frame is None:
        consecutive_failures += 1
        if consecutive_failures == 1:
          # A3: first failure of a streak invalidates the clip window.
          self._reset_continuity("grab_failure")
          if self.offline_since is None:
            self.offline_since = time.time()
          last_ts = None
        wait_time = min(2 ** (consecutive_failures - 1), 5)
        log.warning(
          "Frame grab failed (%d/%d). Retrying in %ss...",
          consecutive_failures, max_consecutive_failures, wait_time,
        )
        if consecutive_failures >= max_consecutive_failures:
          log.error("Camera disconnected. Too many failures. Giving up.")
          self.stop()
          break
        time.sleep(wait_time)
        next_deadline = time.perf_counter()
        continue

      consecutive_failures = 0
      self.offline_since = None

      # A3: a long gap between frames (slow network, stall) must not be glued
      # into one clip, nor produce a whole-frame motion diff.
      if detect_gap(last_ts, ts, MAX_FRAME_GAP_S):
        self._reset_continuity("gap")
      last_ts = ts

      if not self.evidence_buffer_initialized:
        frame_bytes = frame.nbytes
        max_frames = int((self.max_evidence_mb * 1024 * 1024) / frame_bytes)
        max_frames = max(max_frames, CLIP_LENGTH)
        self.evidence_buffer = deque(maxlen=max_frames)
        self.evidence_buffer_initialized = True
        log.info("Evidence buffer capped at %d frames (%s MB)", max_frames, self.max_evidence_mb)

      # B1: live matches training - HOG person detector OFF, no person-box
      # cache (the old never-expiring cache pinned the crop to a stale spot).
      frame_rgb, scene_meta, self._prev_gray = prepare_for_model(
        frame,
        self._prev_gray,
        person_box_cache=None,
        run_person_detector=False,
        crop_state=self._crop_state,
      )
      self._update_frozen(frame, ts)

      with self._lock:
        self.window.add(
          ts,
          frame_rgb,
          suitable=scene_meta.get("suitable_for_model", True),
          global_motion=scene_meta.get("global_motion", False),
        )
        self.evidence_buffer.append(frame)
        self.frame_count += 1
        self.last_frame_time = ts
        self._loop_timestamps.append(ts)
        self.scene_meta = scene_meta
        recorder = self._recorder

      if recorder is not None:
        recorder.push(frame)

      # Real fps -> degraded flag (A1), with a little hysteresis.
      self._fps_warn_counter += 1
      actual = self.get_actual_fps()
      if actual is not None and len(self._loop_timestamps) >= 10:
        if actual < DEGRADED_FPS:
          if not self.degraded:
            log.warning(
              "Camera degraded: %.1f fps < %.0f fps. Alerts cannot open until it recovers.",
              actual, DEGRADED_FPS,
            )
          self.degraded = True
        elif actual >= DEGRADED_FPS + 1.0:
          self.degraded = False
      if self._fps_warn_counter >= 30:
        self._fps_warn_counter = 0
        if actual is not None and actual < 0.8 * self.target_fps:
          log.warning(
            "fps drift: sustaining %.1f fps vs target %s fps (clips are time-resampled)",
            actual, self.target_fps,
          )

      # Deadline-based pacing (no cumulative drift).
      next_deadline += sleep_time
      now = time.perf_counter()
      remaining = next_deadline - now
      if remaining > 0:
        time.sleep(remaining)
      else:
        next_deadline = now

  # ---------------------------------------------------------------- queries
  def get_buffer(self):
    """RGB frames of the latest clip window (<= CLIP_LENGTH, oldest first)."""
    with self._lock:
      return list(self.window.items)[-CLIP_LENGTH:]

  def get_buffer_fill(self):
    with self._lock:
      n = min(CLIP_LENGTH, self.window.available_in_span())
      ready = self.window.sample() is not None
    return (CLIP_LENGTH if ready else min(n, CLIP_LENGTH - 1)), ready

  def get_evidence_buffer(self):
    with self._lock:
      return list(self.evidence_buffer)

  def get_inference_snapshot(self):
    """Everything the watcher needs, taken atomically.  `ready` is True when a
    full 15 fps-timeline clip can be sampled from timestamped frames."""
    with self._lock:
      sample = self.window.sample()
      meta = dict(self.scene_meta)
      gen = self.generation
      reason = self.last_reset_reason
      count = self.frame_count
      last = self.last_frame_time
    age_s = None if last is None else time.monotonic() - last
    snap = {
      "ready": sample is not None,
      "generation": gen,
      "reset_reason": reason,
      "frame_count": count,
      "age_s": age_s,
      "degraded": self.degraded,
      "frames": None,
      "t_end": None,
      "suitable": True,
      "global_motion": False,
      "scene": meta,
    }
    if sample is not None:
      snap.update(
        frames=sample["items"],
        t_end=sample["t_end"],
        suitable=sample["suitable"],
        global_motion=sample["global_motion"],
        span_s=sample["span_s"],
      )
      # Clip-level (not last-frame) suitability, B4.
      snap["scene"] = {**meta, "suitable_for_model": bool(sample["suitable"]),
                       "global_motion": bool(sample["global_motion"])}
    return snap

  def get_scene_meta(self):
    with self._lock:
      meta = dict(self.scene_meta)
      sample = self.window.sample()
    if sample is not None:
      meta["suitable_for_model"] = bool(sample["suitable"])
      meta["global_motion"] = bool(sample["global_motion"])
    return meta

  def get_generation(self):
    with self._lock:
      return self.generation

  def get_last_frame_age_ms(self):
    with self._lock:
      if self.last_frame_time is None:
        return None
      return round((time.monotonic() - self.last_frame_time) * 1000, 1)

  def get_actual_fps(self):
    """Rolling fps measured from real frame-arrival timestamps (last ~30 frames)."""
    with self._lock:
      ts = list(self._loop_timestamps)
    if len(ts) < 2:
      return None
    span = ts[-1] - ts[0]
    if span <= 0:
      return None
    return round((len(ts) - 1) / span, 2)

  def get_offline_since_iso(self):
    t = self.offline_since
    if t is None:
      return None
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

  def set_evidence_budget_mb(self, max_evidence_mb: float):
    """
    Update the memory ceiling for the evidence buffer, resizing it in place
    if it's already running. Used to keep TOTAL evidence memory roughly
    constant regardless of how many cameras are attached (each camera's
    share shrinks as more cameras join), rather than every camera
    independently claiming up to max_evidence_mb of its own.
    """
    with self._lock:
      self.max_evidence_mb = max_evidence_mb
      if self.evidence_buffer_initialized and len(self.evidence_buffer) > 0:
        frame_bytes = self.evidence_buffer[0].nbytes
        max_frames = int((max_evidence_mb * 1024 * 1024) / frame_bytes)
        max_frames = max(max_frames, CLIP_LENGTH)
        trimmed = list(self.evidence_buffer)[-max_frames:]
        self.evidence_buffer = deque(trimmed, maxlen=max_frames)

  def get_latest_frame_jpeg(self):
    with self._lock:
      if len(self.evidence_buffer) == 0:
        return None
      frame = self.evidence_buffer[-1]
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()

  # -------------------------------------------------------------- recording
  def is_recording(self):
    with self._lock:
      return self._recorder is not None

  def start_recording(self, filename) -> bool:
    """Open an evidence file on a writer thread.  C3: the file is written at
    the MEASURED fps (clamped 1-30) so playback matches real time."""
    with self._lock:
      if self._recorder is not None:
        return True
      if not self.evidence_buffer:
        return False
      height, width = self.evidence_buffer[0].shape[:2]
      fps = self.get_actual_fps_unlocked()
      fps = _clamp(fps if fps else float(self.target_fps), 1.0, 30.0)
      rec = _EvidenceRecorder(
        filename, (width, height), fps, list(self.evidence_buffer), self._recording_done
      )
      self._recorder = rec
      self._recording_path = filename
    rec.start()
    return True

  def get_actual_fps_unlocked(self):
    ts = list(self._loop_timestamps)
    if len(ts) < 2:
      return None
    span = ts[-1] - ts[0]
    if span <= 0:
      return None
    return (len(ts) - 1) / span

  def _recording_done(self, path, ok):
    cb = self.on_recording_finished
    if cb is not None:
      cb(path, ok)

  def stop_recording(self):
    """Non-blocking: the writer thread drains its queue, closes the file and
    then reports through on_recording_finished(path, ok)."""
    with self._lock:
      rec = self._recorder
      path = self._recording_path
      self._recorder = None
      self._recording_path = None
    if rec is not None:
      rec.finish()
    return path

  def get_frame_count(self):
    with self._lock:
      return self.frame_count

  def is_active(self):
    """True when producer thread should be running and stream is open."""
    return (
      not self.stopped
      and self.stream is not None
      and self.stream.isOpened()
    )

  def stop(self):
    """Stops the producer thread and releases the camera hardware."""
    self.stopped = True
    self.stop_recording()
    if getattr(self, "stream", None) is not None:
      try:
        if hasattr(self.stream, "release"):
          self.stream.release()
        elif hasattr(self.stream, "close"):
          self.stream.close()
      except Exception:
        pass
      self.stream = None
    self._thread = None


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  cam = CameraStream(src=0, target_fps=CAMERA_TARGET_FPS, show_preview=True, soft_fail=False).start()

  time.sleep(1)

  try:
    while True:
      fill, ready = cam.get_buffer_fill()
      print(
        f"clip ready={ready} fill={fill}/{CLIP_LENGTH} "
        f"fps={cam.get_actual_fps()} total={cam.get_frame_count()}"
      )
      time.sleep(1)

  except KeyboardInterrupt:
    print("\n[SYSTEM] Shutting down...")
    cam.stop()
