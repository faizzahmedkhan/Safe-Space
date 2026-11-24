"""Synthetic test videos (needs numpy + cv2)."""
import cv2
import numpy as np


def write_video(path, fps=30.0, seconds=6.0, size=(320, 240), bursts=((2.0, 3.5),), seed=0, drift=0.6):
    """A textured background with a block that drifts slowly, and moves fast
    and jitters during `bursts` (start_s, end_s): enough motion structure for
    frame_prep's motion crop and for a motion-based stand-in scorer to spike.
    drift=0 with no bursts gives a static scene.  Written as MJPG/AVI, which every OpenCV build can encode."""
    w, h = size
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 90, (h, w, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open an MJPG writer")
    x = 20.0
    for i in range(int(round(seconds * fps))):
        t = i / fps
        fast = any(a <= t < b for a, b in bursts)
        x += (9.0 if fast else drift) * (30.0 / fps)
        jitter = int(rng.integers(-12, 13)) if fast else 0
        frame = bg.copy()
        x0 = int(x) % (w - 70)
        y0 = 70 + jitter
        cv2.rectangle(frame, (x0, y0), (x0 + 60, y0 + 110), (200, 170, 90), -1)
        writer.write(frame)
    writer.release()
    return path
