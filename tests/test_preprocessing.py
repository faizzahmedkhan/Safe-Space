"""Preprocessing contract, clip-consistent cropping, and the live camera path."""
import os
import shutil
import tempfile
import unittest

from tests import _helpers  # noqa: F401

import preprocessing as pp

try:
    import cv2
    import numpy as np

    import frame_prep as fp
    from camera import CameraStream
    from frame_cache import extract_video
    from tests._video import write_video

    HAVE_CV = True
except Exception:  # pragma: no cover
    HAVE_CV = False


class SpecTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(pp.PREPROC_V1, pp.PreprocSpec("frame", "none"))
        self.assertEqual(pp.PREPROC_V2, pp.PreprocSpec("clip", "kinetics"))
        self.assertEqual(pp.PREPROC_V1.name, "v1")
        self.assertEqual(pp.PREPROC_V2.name, "v2")
        self.assertEqual(pp.PreprocSpec("clip", "none").name, "crop=clip,normalize=none")

    def test_parse(self):
        self.assertEqual(pp.parse_preprocessing(None), pp.LEGACY_PREPROC)
        self.assertEqual(pp.LEGACY_PREPROC, pp.PREPROC_V1)  # old weights keep working
        self.assertEqual(pp.parse_preprocessing("V2"), pp.PREPROC_V2)
        self.assertEqual(pp.parse_preprocessing({"crop": "clip"}), pp.PreprocSpec("clip", "none"))
        self.assertEqual(pp.parse_preprocessing(pp.sidecar_value(pp.PREPROC_V2)), pp.PREPROC_V2)
        for bad in ("v3", {"crop": "tile"}, {"normalize": "imagenet"}, {"crop": "clip", "extra": 1}):
            with self.assertRaises(ValueError):
                pp.parse_preprocessing(bad)
        with self.assertRaises(TypeError):
            pp.parse_preprocessing(2)

    def test_model_attachment(self):
        class M:
            pass

        m = M()
        self.assertEqual(pp.of_model(m), pp.LEGACY_PREPROC)
        self.assertIs(pp.attach(m, pp.PREPROC_V2), m)
        self.assertEqual(pp.of_model(m), pp.PREPROC_V2)

    def test_kinetics_constants(self):
        # torchvision R3D_18_Weights.KINETICS400_V1
        self.assertEqual(pp.KINETICS_MEAN, (0.43216, 0.394666, 0.37645))
        self.assertEqual(pp.KINETICS_STD, (0.22803, 0.22145, 0.216989))


@unittest.skipUnless(HAVE_CV, "numpy/cv2 not installed")
class ClipCropTests(unittest.TestCase):
    def test_working_frame_size(self):
        self.assertEqual(fp.to_working_frame(np.zeros((1080, 1920, 3), np.uint8)).shape, (224, 398, 3))
        self.assertEqual(fp.to_working_frame(np.zeros((480, 360, 3), np.uint8)).shape, (299, 224, 3))
        small = np.zeros((120, 160, 3), np.uint8)
        self.assertIs(fp.to_working_frame(small), small)  # never upscaled

    def test_clip_box_needs_enough_frames(self):
        b = (40, 30, 50, 60)
        self.assertIsNone(fp.clip_box([b] * 3 + [None] * 13))  # 3/16 < 25%
        self.assertEqual(fp.clip_box([b] * 4 + [None] * 12), b)
        self.assertIsNone(fp.clip_box([]))

    def test_clip_box_ignores_one_outlier(self):
        boxes = [(100, 50, 40, 80)] * 15 + [(0, 0, 398, 224)]
        x, y, w, h = fp.clip_box(boxes)
        self.assertGreaterEqual(x, 90)
        self.assertLessEqual(x + w, 150)

    def test_clip_box_cache_rows(self):
        rows = np.array([[10, 10, 20, 20]] * 4 + [[-1, -1, -1, -1]] * 4, np.int32)
        self.assertEqual(fp.clip_box(rows), (10, 10, 20, 20))

    def test_build_clip_uses_one_rect_for_every_frame(self):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (224, 398, 3), dtype=np.uint8) for _ in range(16)]
        # boxes that wander: per-frame cropping would differ frame to frame
        boxes = [(60 + 8 * i, 40, 50, 90) for i in range(16)]
        clip = fp.build_clip(frames, boxes, "clip")
        self.assertEqual(clip.shape, (16, 112, 112, 3))
        x0, y0, side, method = fp.clip_crop_rect(boxes, (224, 398))
        self.assertEqual(method, "motion_crop")
        for f, out in zip(frames, clip):
            ref = cv2.resize(f[y0:y0 + side, x0:x0 + side], (112, 112),
                             interpolation=cv2.INTER_AREA if side > 112 else cv2.INTER_LINEAR)
            np.testing.assert_array_equal(out, ref)

    def test_build_clip_center_fallback(self):
        x0, y0, side, method = fp.clip_crop_rect([None] * 16, (224, 398))
        self.assertEqual(method, "center_crop")
        self.assertEqual(side, int(224 * 0.7))
        self.assertEqual((x0, y0), ((398 - side) // 2, (224 - side) // 2))

    def test_build_clip_frame_mode_stacks(self):
        frames = [np.full((112, 112, 3), i, np.uint8) for i in range(16)]
        np.testing.assert_array_equal(fp.build_clip(frames, [None] * 16, "frame"), np.stack(frames))
        with self.assertRaises(ValueError):
            fp.build_clip(frames, [None] * 16, "tile")

    def test_observe_frame_flags_match_prepare_for_model_rule(self):
        a = np.random.RandomState(0).randint(0, 255, (480, 640, 3), np.uint8)
        rgb, meta, gray = fp.observe_frame(a, None)
        self.assertEqual(rgb.shape, (224, 299, 3))
        self.assertIsNone(meta["motion_box"])
        self.assertEqual(meta["method"], "center_crop")
        self.assertFalse(meta["suitable_for_model"])
        b = a.copy()
        b[100:300, 200:350] = (250, 240, 30)  # a large solid object appears (pure-noise
        # inversion would not do: area-downscaling averages noise to flat grey)
        _, meta2, _ = fp.observe_frame(b, gray)
        self.assertIsNotNone(meta2["motion_box"])
        self.assertEqual(meta2["method"], "motion_crop")
        self.assertTrue(meta2["suitable_for_model"])
        bright = np.clip(a.astype(int) + 60, 0, 255).astype(np.uint8)
        _, meta3, _ = fp.observe_frame(bright, gray)
        self.assertTrue(meta3["global_motion"])


@unittest.skipUnless(HAVE_CV, "numpy/cv2 not installed")
class LiveCameraPathTests(unittest.TestCase):
    """The live camera must hand the watcher the same 16 frames training sees."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ss_cam_test_")
        cls.video = write_video(os.path.join(cls.tmp, "v.avi"), fps=30.0, seconds=4.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def feed_camera(self, crop_mode):
        """Feed the 15 fps-kept frames of the file into a CameraStream the way
        its capture loop does (prepare, then add to the window)."""
        cam = CameraStream(src=None, soft_fail=True, crop_mode=crop_mode)
        cap = cv2.VideoCapture(self.video)
        i = 0
        snaps = {}
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 2 == 0:  # 30 fps -> 15 fps, like FpsResampler
                item, meta = cam._prepare_frame(frame)
                cam.window.add(i / 30.0, item,
                               suitable=meta.get("suitable_for_model", True),
                               global_motion=meta.get("global_motion", False))
                snap = cam.get_inference_snapshot()
                if snap["ready"]:
                    snaps[kept] = snap
                kept += 1
            i += 1
        cap.release()
        return snaps

    def test_snapshots_equal_training_windows(self):
        for crop in ("frame", "clip"):
            with self.subTest(crop=crop):
                snaps = self.feed_camera(crop)
                video = extract_video(self.video, crop_mode=crop)
                ready = {w.end: w for w in video.ready_windows()}
                self.assertEqual(set(snaps), set(ready))
                for end, snap in snaps.items():
                    clip = video.window(ready[end])
                    np.testing.assert_array_equal(np.stack(list(snap["frames"])), clip.frames)
                    self.assertEqual(snap["suitable"], clip.suitable)
                    self.assertEqual(snap["global_motion"], clip.global_motion)

    def test_invalid_crop_mode(self):
        with self.assertRaises(ValueError):
            CameraStream(src=None, soft_fail=True, crop_mode="tile")

    def test_get_buffer_returns_images(self):
        cam = CameraStream(src=None, soft_fail=True, crop_mode="clip")
        frame = np.zeros((240, 320, 3), np.uint8)
        item, _ = cam._prepare_frame(frame)
        cam.window.add(0.0, item)
        buf = cam.get_buffer()
        self.assertEqual(buf[0].shape, (224, 299, 3))


if __name__ == "__main__":
    unittest.main()
