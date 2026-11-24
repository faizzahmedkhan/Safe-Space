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


@unittest.skipUnless(HAVE, "torch/numpy/cv2 not installed")
class CachedPipelineTorchTests(unittest.TestCase):
    """Real-torch counterparts of test_pipeline_equivalence / test_frame_cache."""

    @classmethod
    def setUpClass(cls):
        import shutil  # noqa: F401
        from tests._video import write_video

        cls.tmp = tempfile.mkdtemp(prefix="ss_torch_cache_")
        cls.video = write_video(os.path.join(cls.tmp, "v.avi"), fps=30.0, seconds=4.0)
        torch.manual_seed(0)
        cls.model = core.build_model().eval()  # random weights: equality, not accuracy

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_preprocess_batch_matches_preprocess_clip(self):
        rng = np.random.RandomState(1)
        clips = rng.randint(0, 256, (3, 16, 112, 112, 3)).astype(np.uint8)
        batch = core.preprocess_batch(clips)
        self.assertEqual(tuple(batch.shape), (3, 3, 16, 112, 112))
        for i in range(3):
            self.assertTrue(torch.equal(batch[i], core.preprocess_clip(list(clips[i]))))
        flipped = clips[:, :, :, ::-1, :]  # negative-stride view must work
        self.assertTrue(torch.equal(core.preprocess_batch(flipped), core.preprocess_batch(flipped.copy())))
        with self.assertRaises(ValueError):
            core.preprocess_batch(clips[0])

    def test_cached_scores_match_analyze_windows(self):
        import clip_eval
        import preprocessing
        from frame_cache import FrameCache
        from video_calibration import analyze_windows

        dev = torch.device("cpu")
        for spec in (preprocessing.PREPROC_V1, preprocessing.PREPROC_V2):
            with self.subTest(preprocessing=spec.name):
                preprocessing.attach(self.model, spec)
                ref = analyze_windows(self.video, self.model, dev)
                self.assertEqual(ref["preprocessing"], spec.name)
                cache = FrameCache(os.path.join(self.tmp, f"cache_{spec.name}"), fmt="npy",
                                   crop_mode=spec.crop)
                res = clip_eval.evaluate_model(
                    self.model, [(self.video, 1, "Violence", cache.get(self.video))], dev, batch_size=3
                )
                vs = res.scores[0]
                self.assertEqual(len(ref["samples"]), vs.n_windows)
                for s, raw, t in zip(ref["samples"], vs.raw, vs.t_end):
                    self.assertAlmostEqual(s["t"], t, places=9)
                    self.assertAlmostEqual(s["raw"], float(raw), places=5)  # batched vs single forward
                self.assertEqual(len(ref["incidents"]), res.rows[0]["n_incidents"])
        preprocessing.attach(self.model, preprocessing.PREPROC_V1)

    def test_eval_rejects_cache_with_other_crop_mode(self):
        import clip_eval
        import preprocessing
        from frame_cache import FrameCache

        cache = FrameCache(os.path.join(self.tmp, "cache_mismatch"), crop_mode="frame")
        preprocessing.attach(self.model, preprocessing.PREPROC_V2)
        try:
            with self.assertRaises(ValueError):
                clip_eval.evaluate_model(self.model, [(self.video, 1, "V", cache.get(self.video))],
                                         torch.device("cpu"))
        finally:
            preprocessing.attach(self.model, preprocessing.PREPROC_V1)

    def test_kinetics_normalization(self):
        import preprocessing

        clips = np.zeros((1, 16, 4, 4, 3), np.uint8)
        clips[..., 0], clips[..., 1], clips[..., 2] = 255, 128, 0
        x = core.preprocess_batch(clips, normalize="kinetics")
        for c, v in enumerate((255, 128, 0)):
            want = (v / 255.0 - preprocessing.KINETICS_MEAN[c]) / preprocessing.KINETICS_STD[c]
            self.assertAlmostEqual(x[0, c, 0, 0, 0].item(), want, places=5)
        self.assertTrue(torch.equal(core.preprocess_clip(list(clips[0]), normalize="kinetics"), x[0]))
        with self.assertRaises(ValueError):
            core.preprocess_batch(clips, normalize="imagenet")

    def test_load_model_preprocessing_contract(self):
        import preprocessing

        state = self.model.state_dict()
        d = os.path.join(self.tmp, "contract")
        os.makedirs(d, exist_ok=True)
        cpu = torch.device("cpu")

        legacy = os.path.join(d, "legacy.pth")
        torch.save(state, legacy)
        m, info = core.load_model(legacy, cpu)
        self.assertEqual(info["preprocessing"], preprocessing.PREPROC_V1)
        self.assertEqual(info["preprocessing_source"], "legacy default")
        self.assertEqual(preprocessing.of_model(m), preprocessing.PREPROC_V1)

        side = os.path.join(d, "side.pth")
        torch.save(state, side)
        with open(os.path.join(d, "side.json"), "w") as f:
            json.dump({"thresholds": {"alert_threshold": 0.7}, "preprocessing": "v2"}, f)
        m, info = core.load_model(side, cpu)
        self.assertEqual(preprocessing.of_model(m), preprocessing.PREPROC_V2)
        self.assertEqual(info["preprocessing_source"], "sidecar")
        self.assertEqual(info["thresholds"]["alert_threshold"], 0.7)

        full = os.path.join(d, "full.pth")
        torch.save({"state_dict": state, "config": {"preprocessing": {"crop": "clip", "normalize": "kinetics"}}}, full)
        m, info = core.load_model(full, cpu)
        self.assertEqual(info["preprocessing"], preprocessing.PREPROC_V2)
        self.assertEqual(info["preprocessing_source"], "checkpoint")

        bad = os.path.join(d, "bad.pth")
        torch.save(state, bad)
        with open(os.path.join(d, "bad.json"), "w") as f:
            json.dump({"preprocessing": "v9"}, f)
        with self.assertRaises(ValueError):
            core.load_model(bad, cpu)

    def test_cached_dataset_items(self):
        import random

        from cached_dataset import CacheItem, CachedClipDataset
        from frame_cache import FrameCache

        cache = FrameCache(os.path.join(self.tmp, "ds"))
        key, _ = cache.ensure(self.video)
        ds = CachedClipDataset(cache, [CacheItem(key, 1, self.video, "Violence")],
                               windows_per_video=3, hflip=True)
        self.assertEqual(len(ds), 3)
        random.seed(0)
        x, y = ds[2]
        self.assertEqual(tuple(x.shape), (3, 16, 112, 112))
        import preprocessing

        clip_cache = FrameCache(os.path.join(self.tmp, "ds_clip"), crop_mode="clip")
        ckey, _ = clip_cache.ensure(self.video)
        v2 = CachedClipDataset(clip_cache, [CacheItem(ckey, 1, self.video, "Violence")],
                               preprocessing=preprocessing.PREPROC_V2)
        x2, _ = v2[0]
        self.assertEqual(tuple(x2.shape), (3, 16, 112, 112))
        self.assertLess(float(x2.min()), 0.0)  # normalized, not [0, 1]
        with self.assertRaises(ValueError):  # cache crop mode must match the contract
            CachedClipDataset(cache, [], preprocessing=preprocessing.PREPROC_V2)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(int(y), 1)
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertEqual(ds.labels, [1])
        self.assertEqual(ds.classes, ["Violence"])

    def test_train_script_smoke(self):
        import shutil

        import train
        from tests._video import write_video

        data = os.path.join(self.tmp, "data")
        for cls, n, bursts in (("safe", 4, ()), ("unsafe", 4, ((0.5, 2.5),))):
            os.makedirs(os.path.join(data, cls), exist_ok=True)
            for i in range(n):
                write_video(os.path.join(data, cls, f"{cls}{i}.avi"), fps=30.0, seconds=3.0,
                            bursts=bursts, seed=i)
        base = os.path.join(self.tmp, "base.pth")
        torch.save(self.model.state_dict(), base)
        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        args = train.parse_args([
            "--data-dir", data, "--out-dir", out_dir, "--base-model", base,
            "--epochs", "1", "--batch-size", "2", "--windows-per-video", "1",
            "--val-frac", "0.25", "--cache-dir", os.path.join(self.tmp, "train_cache"),
            "--cache-workers", "1", "--preprocessing", "v2",
        ])
        out = train.train(args)
        self.assertTrue(out and os.path.isfile(out))
        with open(os.path.splitext(out)[0] + ".json") as f:
            side = json.load(f)
        self.assertEqual(side["selection"]["metric"], "pr_auc_max")
        self.assertEqual(side["best_epoch"], 1)
        self.assertIn("live", side["selection"]["validation"])
        self.assertEqual(side["preprocessing"], {"name": "v2", "crop": "clip", "normalize": "kinetics"})
        import preprocessing

        m, info = core.load_model(out, torch.device("cpu"))  # the backend picks the contract up
        self.assertEqual(preprocessing.of_model(m), preprocessing.PREPROC_V2)
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
