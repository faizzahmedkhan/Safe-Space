import cv2
import time
import threading
import socket
import http.client
import urllib.request
import urllib.error
import numpy as np
from collections import deque
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse
from frame_prep import prepare_for_model

CLIP_LENGTH = 16
EVIDENCE_BUFFER_LEN = 60
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
      print(f"[PRODUCER] HTTP frame read error: {e}")
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


import subprocess
class _FFmpegStream:
    def __init__(self, url: str, target_fps: int):
        self.url = url
        self.target_fps = target_fps
        self.process = None
        self._opened = False
        self._buf = b""

    def open(self):
        try:
            from video_calibration import get_ffmpeg_binary
            ffmpeg_exe = get_ffmpeg_binary()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
            
        actual_url = self.url
        if "youtube.com" in actual_url or "youtu.be" in actual_url:
            try:
                import yt_dlp
                print(f"[PRODUCER] Resolving YouTube URL dynamically...")
                ydl_opts = {'format': 'best[protocol^=m3u8]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(actual_url, download=False)
                    actual_url = info['url']
            except Exception as e:
                return False, f"yt-dlp failed: {e}"

        cmd = [
            ffmpeg_exe,
            "-hide_banner", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-reconnect_on_network_error", "1", "-reconnect_on_http_error", "4xx,5xx",
            "-i", actual_url,
            "-r", str(self.target_fps),
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-"
        ]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._opened = True
            return True, None
        except Exception as e:
            return False, str(e)

    def read(self):
        if not self._opened or not self.process:
            return False, None
        
        while True:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                return False, None
            self._buf += chunk
            
            start = self._buf.find(b"\xff\xd8")
            if start < 0:
                if len(self._buf) > 256000:
                    self._buf = self._buf[-64000:]
                continue
            
            end = self._buf.find(b"\xff\xd9", start + 2)
            if end < 0:
                if len(self._buf) > 2000000:
                    self._buf = self._buf[start:]
                continue
            
            jpg = self._buf[start : end + 2]
            self._buf = self._buf[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return True, frame

    def isOpened(self):
        return self._opened
        
    def release(self):
        self._opened = False
        if self.process:
            try:
                self.process.kill()
                self.process.wait(timeout=1)
            except Exception:
                pass
            self.process = None
            
    def grab(self):
        return True


class CameraStream:
  def __init__(self, src=0, target_fps=15, show_preview=True, soft_fail=True):
    """
    src=0 opens the default laptop webcam.
    src="http://..." opens a network camera stream (HTTP IP Webcam preferred).
    soft_fail=True: failed open leaves camera inactive instead of raising.
    """
    self.target_fps = target_fps
    self.show_preview = show_preview
    self.preview_enabled = show_preview
    self.soft_fail = soft_fail

    # Stores (timestamp, frame_rgb) pairs, not bare frames. Deeper than
    # CLIP_LENGTH on purpose: get_snapshot() resamples down to exactly
    # CLIP_LENGTH frames at even 1/target_fps spacing (see _resample_window),
    # rather than just handing over "whatever's in the deque" -- that keeps
    # clip timing matched to what the model was trained on even when a
    # camera's real capture rate is irregular or below target_fps. 8x
    # headroom comfortably covers sources running well above target_fps too
    # (storage is cheap: 112x112x3 uint8 frames are ~37KB each).
    self.frame_buffer = deque(maxlen=CLIP_LENGTH * 8)
    self.evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LEN)
    self.max_evidence_mb = 150
    self.evidence_buffer_initialized = False

    # Rolling actual-fps measurement. target_fps is what we *ask* for; this
    # tracks what we're *actually* sustaining. The capture loop's own pacing
    # (time.sleep(remaining)) silently degrades to whatever fps a struggling
    # camera can sustain with no correction and no signal anywhere — this
    # makes that visible instead of silent, since a camera running well
    # under its target fps feeds the model clips that span more real-world
    # time than the model was trained on (quiet accuracy drift, not a crash).
    self._loop_timestamps = deque(maxlen=30)
    self._fps_warn_counter = 0

    self._lock = threading.Lock()
    self._thread = None

    self.frame_count = 0
    self.video_writer = None
    self._recording_path = None
    self.last_frame_time = None
    self.open_error = None
    self._prev_gray = None
    self.scene_meta = {
        "suitable_for_model": True,
        "subject_area_frac": 0.0,
        "method": "full_frame",
        "note": "",
      }

    self.stream = None
    self.src = src
    self.is_network_stream = False
    self.stopped = True

    self._open_source(src)

  def _open_source(self, src):
    """Open capture for src. Returns True on success."""
    if src is None:
      self.src = None
      self.is_network_stream = False
      self.is_raw_stream = False
      self.stream = None
      self.stopped = True
      self.open_error = None
      return False

    self.src = src
    self.is_network_stream = _is_http_url(src) or (
      isinstance(src, str) and "://" in src
    )
    self.is_raw_stream = isinstance(src, str) and (
      ".m3u8" in src or "manifest" in src or "googlevideo.com" in src or 
      "youtube.com" in src or "youtu.be" in src
    )
    self.open_error = None

    if _is_http_url(src) and not self.is_raw_stream:
      diag = diagnose_http_camera(src)
      if not diag["ok"]:
        self.stream = None
        self.stopped = True
        self.open_error = diag["message"]
        print(f"[PRODUCER] {self.open_error}")
        if not self.soft_fail:
          raise RuntimeError(self.open_error)
        return False

      http_src = _HttpFrameSource(src, target_fps=self.target_fps)
      ok, err = http_src.open()
      if not ok:
        self.stream = None
        self.stopped = True
        self.open_error = err or f"Failed to open HTTP camera {src}"
        print(f"[PRODUCER] {self.open_error}")
        if not self.soft_fail:
          raise RuntimeError(self.open_error)
        return False

      self.stream = http_src
      print(f"[PRODUCER] HTTP camera opened ({http_src._mode}) via {http_src.url}")
      self.stopped = False
      return True

    if self.is_raw_stream:
      ff_src = _FFmpegStream(src, target_fps=self.target_fps)
      ok, err = ff_src.open()
      if not ok:
        self.stream = None
        self.stopped = True
        self.open_error = err or f"Failed to open FFmpeg stream for {src}"
        print(f"[PRODUCER] {self.open_error}")
        if not self.soft_fail:
            raise RuntimeError(self.open_error)
        return False
      self.stream = ff_src
      print(f"[PRODUCER] FFmpeg stream opened for raw HLS/YouTube via {src[:50]}...")
      self.stopped = False
      return True

    # Local webcam / RTSP via OpenCV
    stream = cv2.VideoCapture(src)
    if not stream.isOpened():
      stream.release()
      self.stream = None
      self.stopped = True
      self.open_error = f"Failed to open camera with src={src}"
      print(f"[PRODUCER] {self.open_error}")
      if not self.soft_fail:
        raise RuntimeError(self.open_error)
      return False

    self.stream = stream
    if self.is_network_stream and not getattr(self, 'is_raw_stream', False):
      try:
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      except Exception:
        pass
      print("[PRODUCER] Network stream detected (OpenCV). Buffer minimized.")

    self.stopped = False
    return True

  def start(self):
    """Starts the background thread to continuously read frames."""
    if self.stream is None or not self.stream.isOpened():
      print("[PRODUCER] Camera not open — start() skipped.")
      return self
    if self._thread is not None and self._thread.is_alive():
      return self
    self.stopped = False
    print("[PRODUCER] Starting camera thread...")
    self._thread = threading.Thread(target=self.update, args=(), daemon=True)
    self._thread.start()
    return self

  def restart(self, src):
    """
    Stop the current stream, open a new source, clear buffers, and start capturing.
    Returns (success: bool, error: str | None).
    """
    print(f"[PRODUCER] Restarting camera with src={src}")
    old_thread = self._thread
    self.stop()
    if old_thread is not None and old_thread.is_alive():
      old_thread.join(timeout=2.0)

    with self._lock:
      self.frame_buffer.clear()
      self.evidence_buffer.clear()
      self.evidence_buffer = deque(maxlen=EVIDENCE_BUFFER_LEN)
      self.evidence_buffer_initialized = False
      self.frame_count = 0
      self.last_frame_time = None
      self._loop_timestamps.clear()
      self._fps_warn_counter = 0
      self.video_writer = None
      self._recording_path = None
      self._prev_gray = None
      self.scene_meta = {
        "suitable_for_model": True,
        "subject_area_frac": 0.0,
        "method": "full_frame",
        "note": "",
      }

    if not self._open_source(src):
      return False, self.open_error

    self.start()
    return True, None

  def update(self):
    """This function runs on an infinite loop in the background."""
    sleep_time = 1.0 / self.target_fps
    consecutive_failures = 0
    max_consecutive_failures = 30
    use_http = isinstance(self.stream, _HttpFrameSource)

    while not self.stopped:
      loop_start = time.perf_counter()

      if self.stream is None or not self.stream.isOpened():
        print("[PRODUCER] Stream closed unexpectedly.")
        self.stop()
        break

      if use_http:
        grabbed, frame = self.stream.read()
      elif self.is_network_stream and not getattr(self, 'is_raw_stream', False):
        for _ in range(5):
          try:
            self.stream.grab()
          except Exception:
            pass
        grabbed, frame = self.stream.read()
      else:
        grabbed, frame = self.stream.read()

      if not grabbed or frame is None:
        consecutive_failures += 1
        
        # If a raw stream drops, reconnect quickly
        if getattr(self, 'is_raw_stream', False) and consecutive_failures == 2:
            print("[PRODUCER] Raw stream stalled. Fast reconnect...")
            try:
                self.stream.release()
            except Exception:
                pass
            ff_src = _FFmpegStream(self.src, target_fps=self.target_fps)
            ok, err = ff_src.open()
            if not ok:
                print(f"[PRODUCER] Fast reconnect failed: {err}")
                time.sleep(2)
            self.stream = ff_src
            consecutive_failures = 0
            continue

        wait_time = min(2 ** (consecutive_failures - 1), 30)
        print(
          f"[PRODUCER] Frame grab failed ({consecutive_failures}/"
          f"{max_consecutive_failures}). Retrying in {wait_time}s..."
        )
        if consecutive_failures >= max_consecutive_failures:
          print("[PRODUCER] Camera disconnected. Too many failures. Giving up.")
          self.stop()
          break
        time.sleep(wait_time)
        continue

      consecutive_failures = 0

      if not self.evidence_buffer_initialized:
        frame_bytes = frame.nbytes
        max_frames = int((self.max_evidence_mb * 1024 * 1024) / frame_bytes)
        max_frames = max(max_frames, CLIP_LENGTH)
        self.evidence_buffer = deque(maxlen=max_frames)
        self.evidence_buffer_initialized = True
        print(
          f"[PRODUCER] Evidence buffer capped at {max_frames} frames "
          f"({self.max_evidence_mb} MB)"
        )

      # Crop/zoom toward motion so 112x112 isn't empty hallway,
      # and track whether subjects are large enough for this model.
      frame_rgb, scene_meta, self._prev_gray = prepare_for_model(
        frame,
        self._prev_gray,
      )

      with self._lock:
        now = time.time()
        self.frame_buffer.append((now, frame_rgb))
        self.evidence_buffer.append(frame)
        self.frame_count += 1
        self.last_frame_time = now
        self._loop_timestamps.append(now)
        self.scene_meta = scene_meta

        if self.video_writer is not None:
          self.video_writer.write(frame)

      # Every ~30 frames, check whether we're actually keeping pace with
      # target_fps. This never blocks or corrects anything — pacing stays
      # as-is — it just surfaces drift that was previously invisible.
      self._fps_warn_counter += 1
      if self._fps_warn_counter >= 30:
        self._fps_warn_counter = 0
        actual = self.get_actual_fps()
        if actual is not None and actual < 0.8 * self.target_fps:
          print(
            f"[PRODUCER] fps drift: sustaining {actual:.1f}fps vs "
            f"target {self.target_fps}fps — clips will span more real time "
            f"than expected."
          )

      # Pace both local and HTTP snapshot mode
      elapsed = time.perf_counter() - loop_start
      remaining = sleep_time - elapsed
      if remaining > 0:
        time.sleep(remaining)

  def _resample_window(self, num_frames=None, target_interval=None):
    """
    Nearest-timestamp resampling: returns `num_frames` frames spaced at
    even `target_interval` seconds apart, ending at the most recently
    captured frame -- regardless of whether the camera's actual capture
    rate is irregular, above, or below target_fps.

    Why this exists: the old design just handed over "whatever's in a
    fixed-size deque," so a camera sustaining less than target_fps
    silently produced windows spanning MORE real time than the model was
    trained on (CLIP_LENGTH/target_fps seconds) -- every frame-to-frame
    step then represented more real motion than the model's spatiotemporal
    kernels were calibrated for. This resamples by wall-clock time instead
    of by raw arrival order, so timing stays matched to training even when
    the source can't sustain target_fps. It degrades honestly: if the
    source truly lacks a frame close to some target timestamp, the nearest
    available one gets reused for that slot -- which is a smaller, more
    predictable distribution shift (a static/duplicate frame) than
    "everything happened twice as fast as training assumed."

    Returns (frames, temporal_quality). `frames` is [] if there isn't yet
    enough real history to responsibly fill the window (same spirit as
    the old cold-start gate, just measured by real elapsed time covered
    rather than raw frame count, which is harder to fool with a bursty
    source).
    """
    if num_frames is None:
      num_frames = CLIP_LENGTH
    if target_interval is None:
      target_interval = 1.0 / self.target_fps

    buffered = list(self.frame_buffer)  # [(timestamp, frame_rgb), ...] oldest -> newest

    if len(buffered) < 2:
      return [], {
        "unique_frames_used": len(buffered),
        "duplication_ratio": 1.0,
        "suitable_for_model": False,
        "note": "Not enough history yet.",
      }

    span = buffered[-1][0] - buffered[0][0]
    ideal_span = (num_frames - 1) * target_interval
    # Require at least half the ideal window's real time to actually be
    # covered before treating a resampled window as trustworthy. Below
    # that, we'd be filling most slots with duplicates of very few real
    # frames -- better to report "not ready" (matches the old behavior of
    # waiting for the buffer to genuinely fill) than hand the model a
    # near-static window and let it guess.
    if span < 0.5 * ideal_span:
      return [], {
        "unique_frames_used": len(buffered),
        "duplication_ratio": 1.0,
        "suitable_for_model": False,
        "note": f"Only {span:.2f}s of real history covered (need ~{ideal_span:.2f}s).",
      }

    now = buffered[-1][0]
    target_times = [now - (num_frames - 1 - i) * target_interval for i in range(num_frames)]

    frames = []
    used_indices = set()
    j = 0  # both buffered timestamps and target_times are non-decreasing,
           # so a single forward-walking pointer finds each nearest match in O(n) total
    for t in target_times:
      while j + 1 < len(buffered) and abs(buffered[j + 1][0] - t) <= abs(buffered[j][0] - t):
        j += 1
      frames.append(buffered[j][1])
      used_indices.add(j)

    unique_used = len(used_indices)
    duplication_ratio = round(1.0 - (unique_used / num_frames), 3)
    temporal_quality = {
      "unique_frames_used": unique_used,
      "duplication_ratio": duplication_ratio,
      # More than half the window had to reuse a frame because nothing
      # closer in time existed -- the camera's real capture rate is too
      # far below target for this window to faithfully represent
      # training-matched timing. Mirrors frame_prep's spatial
      # suitable_for_model check, just for the temporal axis.
      "suitable_for_model": duplication_ratio < 0.5,
      "note": "",
    }
    return frames, temporal_quality

  def get_buffer(self):
    """Bare frames, most-recent CLIP_LENGTH resampled window. Used for
    status/display purposes (e.g. buffer-fill indicators), not inference --
    see get_snapshot() for that."""
    with self._lock:
      frames, _ = self._resample_window()
      return frames

  def get_evidence_buffer(self):
    with self._lock:
      return list(self.evidence_buffer)

  def get_snapshot(self):
    """Returns (inference_frames, evidence_frames, frame_count,
    temporal_quality). inference_frames is a resampled CLIP_LENGTH window
    at even target_fps spacing (see _resample_window) -- empty if there
    isn't enough real history yet, same as the old fixed-deque behavior."""
    with self._lock:
      frames, temporal_quality = self._resample_window()
      return frames, list(self.evidence_buffer), self.frame_count, temporal_quality

  def get_scene_meta(self):
    with self._lock:
      return dict(self.scene_meta)

  def get_last_frame_age_ms(self):
    with self._lock:
      if self.last_frame_time is None:
        return None
      return round((time.time() - self.last_frame_time) * 1000, 1)

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

  def start_recording(self, filename):
    with self._lock:
      if self.video_writer is not None:
        return

      if not self.evidence_buffer:
        return

      height, width = self.evidence_buffer[0].shape[:2]
      # Record with mp4v (reliable in OpenCV); transcode to H.264 on stop for browsers.
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      self.video_writer = cv2.VideoWriter(
        filename, fourcc, self.target_fps, (width, height)
      )
      if not self.video_writer.isOpened():
        self.video_writer = None
        print(f"[PRODUCER] Failed to open VideoWriter for {filename}")
        return

      self._recording_path = filename
      for f in self.evidence_buffer:
        self.video_writer.write(f)

  def stop_recording(self):
    path = None
    with self._lock:
      if self.video_writer is not None:
        self.video_writer.release()
        self.video_writer = None
        path = self._recording_path
        self._recording_path = None

    if path:
      try:
        from evidence_video import make_browser_compatible

        make_browser_compatible(path)
      except Exception as e:
        print(f"[PRODUCER] Evidence browser transcode skipped: {e}")

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
  cam = CameraStream(src=0, target_fps=6, show_preview=True, soft_fail=False).start()

  time.sleep(1)

  try:
    while True:
      current_buffer = cam.get_buffer()

      if len(current_buffer) == CLIP_LENGTH:
        print(
          f"✅ Buffer Full: {len(current_buffer)} frames ready. "
          f"Shape of one frame: {current_buffer[0].shape} | "
          f"Total captured: {cam.get_frame_count()}"
        )
      else:
        print(f"⏳ Buffering... {len(current_buffer)}/{CLIP_LENGTH} frames")

      time.sleep(1)

  except KeyboardInterrupt:
    print("\n[SYSTEM] Shutting down...")
    cam.stop()
