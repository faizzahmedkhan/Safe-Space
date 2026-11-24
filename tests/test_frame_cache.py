"""Frame cache + shared front-end: exactness against the live ClipWindow path."""
import os
import random
import shutil
import tempfile
import unittest

from tests import _helpers  # noqa: F401

try:
    import numpy as np
    import cv2  # noqa: F401

    from clip_sampler import ClipWindow
    from frame_cache import CachedVideo, FrameCache, content_sha1, extract_video, prep_fingerprint
    from prepared_frames import PreparedFrameReader, clip_from_items, window_item
    from tests._video import write_video

    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


def stream_like_analyze_windows(frames, stride_frames=8, fps=15.0):
    """The windows analyze_windows / live mode would score, built by streaming
    PreparedFrames into a real ClipWindow: [(frame_index, sample_dict)]."""
    cw = ClipWindow()
    out, last = [], None
    for i, pf in enumerate(frames):
        if cw.add(pf.t, window_item(pf), pf.suitable, pf.global_motion):
            last = None
        s = cw.sample()
        if s is None:
            continue
        if last is not None and (pf.t - last) < stride_frames / fps - 1e-6:
            continue
        last = pf.t
        out.append((i, s))
    return out


@unittest.skipUnless(HAVE, "numpy/cv2 not installed")
class FrameCacheTests(unittest.TestCase):
    CROP = "frame"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ss_cache_test_")
        cls.v30 = write_video(os.path.join(cls.tmp, "v30.avi"), fps=30.0, seconds=5.0)
        cls.v25 = write_video(os.path.join(cls.tmp, "v25.avi"), fps=25.0, seconds=4.0, seed=1)
        cls.short = write_video(os.path.join(cls.tmp, "short.avi"), fps=30.0, seconds=0.5, bursts=())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cache(self, name, **kw):
        kw.setdefault("crop_mode", self.CROP)
        kw.setdefault("fmt", "npy")  # exactness tests need the lossless format
        return FrameCache(os.path.join(self.tmp, f"{self.CROP}_{name}"), **kw)

    def _reader(self, path):
        return PreparedFrameReader(path, crop_mode=self.CROP)

    # ------------------------------------------------------------ reader
    def test_reader_resamples_to_15fps(self):
        r = self._reader(self.v30)
        frames = list(r)
        self.assertEqual(r.native_fps, 30.0)
        self.assertEqual(r.frames_read, 150)
        self.assertEqual(len(frames), 75)  # every 2nd frame of a 30 fps file
        self.assertEqual([f.src_index for f in frames[:3]], [1, 3, 5])
        r25 = self._reader(self.v25)
        n25 = len(list(r25))
        self.assertAlmostEqual(n25, 4.0 * 15, delta=2)  # ~15 fps from 25 fps
        expected = (112, 112, 3) if self.CROP == "frame" else (224, 299, 3)  # 320x240 -> short side 224
        self.assertEqual(frames[0].image.shape, expected)
        self.assertEqual(frames[0].image.dtype, np.uint8)
        if self.CROP == "frame":
            self.assertTrue(all(f.box is None for f in frames))
        else:
            self.assertTrue(any(f.box is not None for f in frames))

    def test_reader_bad_path(self):
        with self.assertRaises(RuntimeError):
            PreparedFrameReader(os.path.join(self.tmp, "missing.mp4"))

    # ------------------------------------------------------------ exactness
    def test_cached_windows_identical_to_live_clipwindow(self):
        for path in (self.v30, self.v25):
            frames = list(self._reader(path))
            live = stream_like_analyze_windows(frames)
            cache = self._cache("exact")
            video = cache.get(path)  # built, then reloaded with mmap
            sched = video.schedule()
            self.assertEqual([w.end for w in sched], [i for i, _ in live])
            for w, (_, s) in zip(sched, live):
                c = video.window(w)
                np.testing.assert_array_equal(c.frames, clip_from_items(s["items"], self.CROP))
                self.assertEqual(c.frames.shape, (16, 112, 112, 3))
                self.assertEqual(c.suitable, s["suitable"])
                self.assertEqual(c.global_motion, s["global_motion"])
                self.assertAlmostEqual(c.t_end, s["t_end"])

    def test_every_ready_window_identical(self):
        # Training samples from ready_windows(); each must equal what
        # ClipWindow would have produced at that frame.
        frames = list(self._reader(self.v30))
        video = self._cache("ready").get(self.v30)
        cw = ClipWindow()
        ready = {w.end: w for w in video.ready_windows()}
        n_checked = 0
        for i, pf in enumerate(frames):
            cw.add(pf.t, window_item(pf), pf.suitable, pf.global_motion)
            s = cw.sample()
            self.assertEqual(s is not None, i in ready, f"frame {i}")
            if s is not None:
                np.testing.assert_array_equal(video.window(ready[i]).frames,
                                              clip_from_items(s["items"], self.CROP))
                n_checked += 1
        self.assertGreater(n_checked, 50)

    def test_npy_roundtrip_exact_and_reused(self):
        cache = self._cache("rt")
        mem = extract_video(self.v30, crop_mode=self.CROP)
        key, built = cache.ensure(self.v30)
        self.assertTrue(built)
        key2, built2 = cache.ensure(self.v30)
        self.assertEqual(key, key2)
        self.assertFalse(built2)
        disk = cache.load(key)
        self.assertIsInstance(disk.frames, np.memmap)
        np.testing.assert_array_equal(np.asarray(disk.frames), mem.frames)
        np.testing.assert_array_equal(disk.t, mem.t)
        np.testing.assert_array_equal(disk.suitable, mem.suitable)
        np.testing.assert_array_equal(disk.global_motion, mem.global_motion)
        np.testing.assert_array_equal(disk.boxes, mem.boxes)
        self.assertEqual(disk.crop_mode, self.CROP)
        self.assertEqual(disk.native_fps, 30.0)
        no_mmap = cache.load(key, mmap=False)
        self.assertNotIsInstance(no_mmap.frames, np.memmap)

    def test_jpg_is_close_and_smaller(self):
        ecache = self._cache("cmp_npy")
        exact = ecache.get(self.v30)
        jcache = self._cache("cmp_jpg", fmt="jpg")
        lossy = jcache.get(self.v30)
        w = exact.schedule()[0]
        a = exact.window(w).frames.astype(np.int16)
        b = lossy.window(w).frames.astype(np.int16)
        self.assertEqual(a.shape, b.shape)
        self.assertLess(np.abs(a - b).mean(), 4.0)
        self.assertEqual(jcache.fmt, "jpg")
        size = lambda d: sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))  # noqa: E731
        self.assertLess(size(jcache.root) * 4, size(ecache.root))

    # ------------------------------------------------------------ keys
    def test_key_is_content_based(self):
        copy = os.path.join(self.tmp, "renamed_copy.avi")
        shutil.copyfile(self.v30, copy)
        cache = self._cache("keys")
        self.assertEqual(cache.key_for(self.v30), cache.key_for(copy))
        self.assertNotEqual(cache.key_for(self.v30), cache.key_for(self.v25))
        self.assertEqual(content_sha1(copy), content_sha1(self.v30))

    def test_fingerprint_tracks_settings(self):
        base = prep_fingerprint()
        self.assertEqual(base, prep_fingerprint())
        self.assertNotEqual(base, prep_fingerprint(target_fps=10))
        self.assertNotEqual(base, prep_fingerprint(max_seconds=5))
        self.assertNotEqual(base, prep_fingerprint(fmt="jpg"))
        self.assertNotEqual(prep_fingerprint(fmt="jpg", jpeg_quality=90),
                            prep_fingerprint(fmt="jpg", jpeg_quality=95))
        self.assertNotEqual(base, prep_fingerprint(crop_mode="clip"))
        # Two caches with different formats never share entries.
        self.assertNotEqual(self._cache("fa").key_for(self.v30),
                            self._cache("fb", fmt="jpg").key_for(self.v30))

    def test_bad_format_rejected(self):
        with self.assertRaises(ValueError):
            self._cache("bad", fmt="png")
        with self.assertRaises(ValueError):
            self._cache("bad2", crop_mode="tile")

    def test_auto_format_and_crop_mismatch(self):
        self.assertEqual(FrameCache(os.path.join(self.tmp, "af"), crop_mode="frame").fmt, "npy")
        self.assertEqual(FrameCache(os.path.join(self.tmp, "ac"), crop_mode="clip").fmt, "jpg")
        a = FrameCache(os.path.join(self.tmp, "shared"), fmt="npy", crop_mode="frame")
        key, _ = a.ensure(self.v30)
        b = FrameCache(os.path.join(self.tmp, "shared"), fmt="npy", crop_mode="clip")
        self.assertNotEqual(b.key_for(self.v30), key)  # never served each other's entries
        with self.assertRaises(ValueError):
            b.load(key)

    # ------------------------------------------------------------ short / random
    def test_short_video_uses_fallback_window(self):
        video = self._cache("short").get(self.short)
        self.assertLess(len(video), 16)
        self.assertEqual(video.schedule(), [])
        self.assertEqual(video.ready_windows(), [])
        w = video.random_window(random.Random(0))
        self.assertEqual(w.end, len(video) - 1)
        clip = video.window(w)
        self.assertEqual(clip.frames.shape, (16, 112, 112, 3))
        if self.CROP == "frame":
            np.testing.assert_array_equal(clip.frames[-1], np.asarray(video.frames[-1]))

    def test_random_window_covers_clip(self):
        video = self._cache("rand").get(self.v30)
        rng = random.Random(123)
        ends = {video.random_window(rng).end for _ in range(400)}
        ready = {w.end for w in video.ready_windows()}
        self.assertTrue(ends <= ready)
        self.assertGreater(len(ends), 0.8 * len(ready))

    def test_window_out_of_range(self):
        video = self._cache("oor").get(self.v30)
        with self.assertRaises(IndexError):
            video.window(len(video))

    # ------------------------------------------------------------ build_many
    def test_build_many_parallel_and_failures(self):
        bogus = os.path.join(self.tmp, "not_a_video.mp4")
        with open(bogus, "wb") as f:
            f.write(b"definitely not a video")
        cache = self._cache("many")
        rep = cache.build_many([self.v30, self.v25, bogus, self.v30], workers=2, log_every=0)
        self.assertEqual(set(rep.keys), {self.v30, self.v25})
        self.assertIn(bogus, rep.failed)
        self.assertEqual(rep.built, 2)
        rep2 = cache.build_many([self.v30, self.v25], workers=1, log_every=0)
        self.assertEqual((rep2.built, rep2.reused), (0, 2))
        self.assertEqual(rep2.keys, {k: v for k, v in rep.keys.items()})
        leftovers = [f for f in os.listdir(cache.root) if ".tmp-" in f]
        self.assertEqual(leftovers, [])

    def test_cached_video_requires_frames(self):
        class Empty:
            path = "x"
            native_fps = 30.0
            target_fps = 15.0

            def __iter__(self):
                return iter(())

        with self.assertRaises(ValueError):
            CachedVideo.from_reader(Empty())



class ClipModeFrameCacheTests(FrameCacheTests):
    """Every test above, with one crop per clip (preprocessing crop="clip")."""

    CROP = "clip"

if __name__ == "__main__":
    unittest.main()
