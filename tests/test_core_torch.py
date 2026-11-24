"""Tests that need torch/numpy/cv2; skipped automatically where they are missing."""
import json
import os
import tempfile
import unittest

from tests import _helpers  # noqa: F401

try:
    import numpy as np
    import torch
    import inference_core as core
    import frame_prep
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "torch/numpy/cv2 not installed")
class CoreTests(unittest.TestCase):
    def test_preprocess_is_div255_only_and_layout(self):
        frames = [np.full((112, 112, 3), v, np.uint8) for v in range(16)]
        frames[0][0, 0] = (255, 128, 0)
        t = core.preprocess_clip(frames)
        self.assertEqual(tuple(t.shape), (3, 16, 112, 112))
        self.assertEqual(t.dtype, torch.float32)
        self.assertAlmostEqual(t[0, 0, 0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(t[1, 0, 0, 0].item(), 128 / 255.0, places=6)
        self.assertAlmostEqual(t[2, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(t[0, 5, 50, 50].item(), 5 / 255.0, places=6)  # no mean/std

    def test_constants(self):
        self.assertEqual(core.CLIP_LENGTH, 16)
        self.assertEqual(core.CAMERA_TARGET_FPS, 15)
        self.assertEqual(core.INFERENCE_STRIDE, 8)
        self.assertEqual(core.SCORE_HISTORY_LEN, 4)
        self.assertEqual(core.ALERT_THRESHOLD, 0.67)
        self.assertEqual(core.SPIKE_THRESHOLD, 0.90)

    def test_sidecar_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            pth = os.path.join(d, "m.pth")
            with open(os.path.join(d, "m.json"), "w") as f:
                json.dump({"thresholds": {"alert_threshold": 0.8, "history_len": 6, "bogus": 1}}, f)
            vals, path = core._load_sidecar(pth)
            self.assertEqual(vals, {"alert_threshold": 0.8, "score_history_len": 6})
            self.assertTrue(path.endswith("m.json"))

    def test_model_path_env(self):
        old = os.environ.get("SAFESPACE_MODEL")
        os.environ["SAFESPACE_MODEL"] = "/tmp/x.pth"
        try:
            self.assertEqual(core.resolve_model_path(), "/tmp/x.pth")
        finally:
            if old is None:
                del os.environ["SAFESPACE_MODEL"]
            else:
                os.environ["SAFESPACE_MODEL"] = old

    def test_prepare_for_model_three_values_and_global_motion(self):
        a = np.random.RandomState(0).randint(0, 255, (240, 320, 3), np.uint8)
        rgb, meta, gray = frame_prep.prepare_for_model(a, None, run_person_detector=False)
        self.assertEqual(rgb.shape, (112, 112, 3))
        self.assertIn("global_motion", meta)
        self.assertFalse(meta["global_motion"])
        # a large brightness jump is flagged
        b = np.clip(a.astype(int) + 60, 0, 255).astype(np.uint8)
        _, meta2, _ = frame_prep.prepare_for_model(b, gray, run_person_detector=False)
        self.assertTrue(meta2["global_motion"])

    def test_tracker_factory_uses_thresholds(self):
        tr = core.make_tracker({"alert_threshold": 0.8, "score_history_len": 6})
        self.assertEqual(tr.alert_threshold, 0.8)
        self.assertEqual(tr.history.maxlen, 6)


if __name__ == "__main__":
    unittest.main()
