"""Make evidence MP4s playable in Safari/Chrome (H.264 + yuv420p + faststart)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def make_browser_compatible(path: str) -> bool:
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 1000:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"ffmpeg failed with code {proc.returncode}")
        os.replace(tmp_path, path)
        print(f"[EVIDENCE] Transcoded for browser playback: {path}")
        return True
    except Exception as e:
        print(f"[EVIDENCE] Transcode failed for {path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def ensure_browser_compatible(path: str, marker_dir: Optional[str] = None) -> str:
    """
    Ensure path is browser-playable. Uses a sidecar .web.mp4 cache when possible
    so originals can remain, but prefers in-place conversion for simplicity.
    """
    if not os.path.isfile(path):
        return path

    # Sidecar next to file: incident_x.web.mp4
    root, ext = os.path.splitext(path)
    web_path = f"{root}.web{ext}"

    if os.path.isfile(web_path) and os.path.getsize(web_path) > 1000:
        # Refresh if source is newer
        if os.path.getmtime(web_path) >= os.path.getmtime(path):
            return web_path

    # Convert original into sidecar (keep original for download/debug)
    ffmpeg = _ffmpeg_exe()
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
        web_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and os.path.isfile(web_path) and os.path.getsize(web_path) > 1000:
        print(f"[EVIDENCE] Browser copy ready: {web_path}")
        return web_path

    # Fall back to in-place conversion
    make_browser_compatible(path)
    return path
