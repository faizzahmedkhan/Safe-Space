"""End-to-end: the frame cache + clip_eval must reach the SAME per-window
scores and the SAME incident verdict as video_calibration.analyze_windows
(the /analyze/video path).

The model is a deterministic numpy stand-in (score = sigmoid of the clip's
mean frame-to-frame change), and torch / inference_core are replaced by small
stubs inside a SUBPROCESS, so this test runs without torch and cannot leak
stubs into other tests.  test_core_torch.py repeats the comparison with the
real R3D-18 wherever torch is installed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests import _helpers

try:
    import numpy  # noqa: F401
    import cv2  # noqa: F401

    from tests._video import write_video

    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

STUB_TORCH = '''
"""Minimal torch stand-in for the equivalence subprocess."""
from . import nn  # noqa: F401
def device(name="cpu"):
    return name
'''

STUB_TORCH_NN = '''
class Module:
    pass
'''

STUB_INFERENCE_CORE = '''
"""numpy stand-in for inference_core (only what analyze_windows / clip_eval use)."""
import numpy as np
from thresholds import *  # noqa: F401,F403
from incident import IncidentTracker, default_thresholds, make_tracker  # noqa: F401
from preprocessing import KINETICS_MEAN, KINETICS_STD

class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.arr, dim))
    def to(self, device):
        return self
    def __getitem__(self, i):
        return FakeTensor(self.arr[i])
    def item(self):
        return float(self.arr)
    def numpy(self):
        return self.arr

def preprocess_batch(clips_uint8, normalize="none"):
    arr = np.ascontiguousarray(clips_uint8).astype(np.float32) / 255.0
    if normalize == "kinetics":
        arr = (arr - np.float32(KINETICS_MEAN)) / np.float32(KINETICS_STD)
    return FakeTensor(arr.transpose(0, 4, 1, 2, 3))

def preprocess_clip(frames, normalize="none"):
    return preprocess_batch(np.stack(frames)[None], normalize=normalize)[0]

class FakeModel:
    def eval(self):
        return self

def load_model(*a, **k):
    raise RuntimeError("not available in the equivalence stub")

def predict_violence(model, batch):
    x = batch.arr.astype(np.float64)  # (B, C, T, H, W)
    motion = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2, 3, 4))
    return FakeTensor(1.0 / (1.0 + np.exp(-(motion * 400.0 - 4.0))))
'''

DRIVER = '''
import json, sys
import numpy as np
import preprocessing
from inference_core import FakeModel
from video_calibration import analyze_windows
from frame_cache import FrameCache
import clip_eval

video, cache_dir, crop, normalize = sys.argv[1:5]
spec = preprocessing.PreprocSpec(crop, normalize)
model = preprocessing.attach(FakeModel(), spec)
res = analyze_windows(video, model, "cpu")
cache = FrameCache(cache_dir, fmt="npy", crop_mode=crop)
cache.get(video)  # build
ev = clip_eval.evaluate_model(model, [(video, 1, "Violence", cache.get(video))], "cpu", batch_size=5)
vs = ev.scores[0]
print(json.dumps({
    "a_preprocessing": res["preprocessing"],
    "a_t": [s["t"] for s in res["samples"]],
    "a_raw": [s["raw"] for s in res["samples"]],
    "a_suitable": [s["suitable"] for s in res["samples"]],
    "a_global": [s["global_motion"] for s in res["samples"]],
    "a_incidents": len(res["incidents"]),
    "a_peak_smoothed": res["peak_smoothed"],
    "b_t": vs.t_end.tolist(),
    "b_raw": vs.raw.tolist(),
    "b_suitable": vs.suitable.tolist(),
    "b_global": vs.global_motion.tolist(),
    "b_incidents": ev.rows[0]["n_incidents"],
    "b_peak_smoothed": ev.rows[0]["peak_smoothed"],
}))
'''


@unittest.skipUnless(HAVE, "numpy/cv2 not installed")
class PipelineEquivalenceTests(unittest.TestCase):
    SPECS = (("frame", "none"), ("clip", "kinetics"), ("clip", "none"), ("frame", "kinetics"))

    def run_driver(self, video_path, crop, normalize):
        stub = os.path.join(self.tmp, "stubs")
        os.makedirs(os.path.join(stub, "torch"), exist_ok=True)
        with open(os.path.join(stub, "torch", "__init__.py"), "w") as f:
            f.write(STUB_TORCH)
        with open(os.path.join(stub, "torch", "nn.py"), "w") as f:
            f.write(STUB_TORCH_NN)
        with open(os.path.join(stub, "inference_core.py"), "w") as f:
            f.write(STUB_INFERENCE_CORE)
        driver = os.path.join(self.tmp, "driver.py")
        with open(driver, "w") as f:
            f.write(DRIVER)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([stub, _helpers.ROOT]))
        out = subprocess.run(
            [sys.executable, driver, video_path, os.path.join(self.tmp, f"cache_{crop}"), crop, normalize],
            capture_output=True, text=True, env=env, cwd=self.tmp, timeout=300,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ss_equiv_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def check(self, video_path, expect_alert):
        for crop, normalize in self.SPECS:
            with self.subTest(crop=crop, normalize=normalize):
                self._check_one(video_path, expect_alert, crop, normalize)

    def _check_one(self, video_path, expect_alert, crop, normalize):
        r = self.run_driver(video_path, crop, normalize)
        self.assertEqual(r["a_preprocessing"],
                         {"frame,none": "v1", "clip,kinetics": "v2"}.get(f"{crop},{normalize}",
                                                                         f"crop={crop},normalize={normalize}"))
        self.assertGreater(len(r["a_t"]), 5)
        self.assertEqual(len(r["a_t"]), len(r["b_t"]))
        for a, b in zip(r["a_t"], r["b_t"]):
            self.assertAlmostEqual(a, b, places=9)
        for a, b in zip(r["a_raw"], r["b_raw"]):
            self.assertAlmostEqual(a, b, places=6)  # float32 vs float64 order of ops
        self.assertEqual(r["a_suitable"], r["b_suitable"])
        self.assertEqual(r["a_global"], r["b_global"])
        self.assertEqual(r["a_incidents"], r["b_incidents"])
        self.assertAlmostEqual(r["a_peak_smoothed"], r["b_peak_smoothed"], places=6)
        self.assertEqual(r["a_incidents"] > 0, expect_alert, textwrap.shorten(str(r["a_raw"]), 300))

    def test_30fps_with_violent_burst(self):
        path = write_video(os.path.join(self.tmp, "burst.avi"), fps=30.0, seconds=8.0,
                           bursts=((2.0, 5.0),))
        self.check(path, expect_alert=True)

    def test_25fps_static_scene(self):
        path = write_video(os.path.join(self.tmp, "static.avi"), fps=25.0, seconds=6.0,
                           bursts=(), drift=0.0)
        self.check(path, expect_alert=False)


if __name__ == "__main__":
    unittest.main()
