"""Make evidence MP4s playable in Safari/Chrome (H.264 + yuv420p + faststart)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Optional

log = logging.getLogger("safespace.evidence")

TRANSCODE_TIMEOUT_S = 300


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def make_browser_compatible(path: str, timeout: float = TRANSCODE_TIMEOUT_S) -> bool:
    """
    Re-encode path in-place to H.264 MP4 browsers can play.
    Returns True if the file was converted (or already fine after rewrite).
    """
    if not path or not os.path.isfile(path):
        return False

    ffmpeg = _ffmpeg_exe()
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=dir_name)
    os.close(fd)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        tmp_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        if proc.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 1000:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"ffmpeg failed with code {proc.returncode}")
        os.replace(tmp_path, path)
        log.info("Transcoded for browser playback: %s", path)
        return True
    except Exception as e:
        log.error("Transcode failed for %s: %s", path, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


# NOTE: ensure_browser_compatible() used to live here and was called by
# GET /evidence/{filename}. It has been removed: make_browser_compatible()
# above already converts each clip to H.264 in place when recording stops, so
# that second pass only ever re-encoded an already-playable file into a
# redundant .web.mp4 sidecar — doubling storage for every clip ever viewed.
# It also wrote straight to its output path with no temp-then-rename, so two
# concurrent requests (a HEAD plus a ranged GET, which browsers routinely
# issue for <video>) could interleave writes and permanently cache a corrupt
# file. Serve the original directly instead.
