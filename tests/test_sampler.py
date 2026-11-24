"""Timestamp clip sampler, gap detection, coverage, resampler (pure Python)."""
import unittest

from tests import _helpers  # noqa: F401
from clip_sampler import (
    ClipWindow,
    FpsResampler,
    aggregate_clip_flags,
    detect_gap,
    scoring_schedule,
    select_clip_indices,
    window_coverage,
)


def stamps(fps, seconds=3.0, t0=50.0):
    n = int(seconds * fps)
    return [t0 + i / fps for i in range(n)]


class SamplerTests(unittest.TestCase):
    def _check(self, fps):
        ts = stamps(fps)
        idx = select_clip_indices(ts)
        self.assertEqual(len(idx), 16)
        self.assertEqual(idx, sorted(idx))
        picked = [ts[i] for i in idx]
        span = picked[-1] - picked[0]
        self.assertAlmostEqual(span, 1.0, delta=1.0 / fps + 1e-9, msg=f"fps={fps}")
        return idx

    def test_span_8fps(self):
        idx = self._check(8)
        self.assertLess(len(set(idx)), 16)  # slow camera -> repeated frames

    def test_span_15fps(self):
        idx = self._check(15)
        self.assertEqual(len(set(idx)), 16)
        self.assertEqual(idx[-1] - idx[0], 15)

    def test_span_30fps(self):
        idx = self._check(30)
        self.assertEqual(len(set(idx)), 16)
        self.assertTrue(all(b - a == 2 for a, b in zip(idx, idx[1:])))

    def test_last_index_is_newest(self):
        for fps in (8, 15, 30):
            ts = stamps(fps)
            self.assertEqual(select_clip_indices(ts)[-1], len(ts) - 1)

    def test_empty(self):
        self.assertIsNone(select_clip_indices([]))

    def test_clip_window_ready_only_with_enough_history(self):
        w = ClipWindow()
        for t in stamps(15, 0.5):
            w.add(t, "x")
        self.assertIsNone(w.sample())
        for t in stamps(15, 3.0, t0=50.5):
            w.add(t, "x")
        s = w.sample()
        self.assertIsNotNone(s)
        self.assertEqual(len(s["items"]), 16)
        self.assertAlmostEqual(s["span_s"], 1.0, delta=1 / 15 + 1e-9)

    def test_clip_window_slow_camera_still_one_second(self):
        w = ClipWindow()
        for t in stamps(8, 3.0):
            w.add(t, "x")
        s = w.sample()
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s["span_s"], 1.0, delta=1 / 8 + 1e-9)
        self.assertLess(s["unique_frames"], 16)

    def test_clip_level_flags_use_half_of_frames(self):
        w = ClipWindow()
        for i, t in enumerate(stamps(15, 3.0)):
            # only the oldest ~third of frames are unsuitable / global
            bad = i < 30
            w.add(t, i, suitable=not bad, global_motion=bad)
        s = w.sample()
        self.assertTrue(s["suitable"])
        self.assertFalse(s["global_motion"])
        w = ClipWindow()
        for i, t in enumerate(stamps(15, 3.0)):
            bad = i >= 36  # newest 9 of 16 frames bad
            w.add(t, i, suitable=not bad, global_motion=bad)
        s = w.sample()
        self.assertFalse(s["suitable"])
        self.assertTrue(s["global_motion"])


class GapTests(unittest.TestCase):
    def test_detect_gap(self):
        self.assertFalse(detect_gap(None, 10.0))
        self.assertFalse(detect_gap(10.0, 10.0667))
        self.assertFalse(detect_gap(10.0, 10.29))
        self.assertTrue(detect_gap(10.0, 10.31))
        self.assertTrue(detect_gap(10.0, 15.0))

    def test_window_clears_on_gap_and_needs_fresh_second(self):
        w = ClipWindow()
        for t in stamps(15, 3.0):
            w.add(t, "old")
        self.assertIsNotNone(w.sample())
        gap_ts = 50.0 + 3.0 + 2.0  # 2 s hole
        self.assertTrue(w.add(gap_ts, "new"))
        self.assertEqual(len(w), 1)
        self.assertIsNone(w.sample())  # old+new footage is never glued
        for i in range(1, 30):
            w.add(gap_ts + i / 15, "new")
        s = w.sample()
        self.assertIsNotNone(s)
        self.assertTrue(all(x == "new" for x in s["items"]))


class CoverageAndResampler(unittest.TestCase):
    def test_coverage_full_when_stride_below_window(self):
        ts = [100 + i * 8 / 15 for i in range(40)]
        self.assertEqual(window_coverage(ts), 100.0)

    def test_coverage_drops_when_stride_exceeds_window(self):
        ts = [100 + i * 2.0 for i in range(20)]  # 1 s window every 2 s
        cov = window_coverage(ts)
        self.assertAlmostEqual(cov, 50.0, delta=6.0)

    def test_coverage_none_with_single_window(self):
        self.assertIsNone(window_coverage([1.0]))

    def test_resampler_30fps_to_15(self):
        r = FpsResampler(15)
        kept = [i for i in range(90) if r.keep(i / 30.0, 1 / 30.0)]
        self.assertAlmostEqual(len(kept), 45, delta=2)

    def test_resampler_keeps_slow_video_entirely(self):
        r = FpsResampler(15)
        kept = [i for i in range(40) if r.keep(i / 8.0, 1 / 8.0)]
        self.assertEqual(len(kept), 40)

    def test_resampler_25fps(self):
        r = FpsResampler(15)
        kept = [i for i in range(250) if r.keep(i / 25.0, 1 / 25.0)]
        self.assertAlmostEqual(len(kept), 150, delta=6)


def streamed_schedule(ts, stride_frames=8, fps=15.0):
    """What analyze_windows scores, computed by streaming into a ClipWindow."""
    w = ClipWindow()
    out, last = [], None
    for i, t in enumerate(ts):
        if w.add(t, i):
            last = None
        if w.sample() is None:
            continue
        if last is not None and (t - last) < stride_frames / fps - 1e-6:
            continue
        last = t
        out.append(i)
    return out


class ScheduleTests(unittest.TestCase):
    def test_matches_streaming_clipwindow(self):
        for fps in (8, 12, 15, 25, 30):
            ts = stamps(fps, seconds=7.0)
            self.assertEqual([w.end for w in scoring_schedule(ts)], streamed_schedule(ts), f"fps={fps}")

    def test_matches_streaming_with_gaps(self):
        ts = stamps(15, 3.0) + stamps(15, 2.5, t0=55.0) + stamps(15, 0.4, t0=60.0)
        sched = scoring_schedule(ts)
        self.assertEqual([w.end for w in sched], streamed_schedule(ts))
        self.assertEqual(sorted({w.segment for w in sched}), [0, 1])  # last segment too short
        for w in sched:
            self.assertLessEqual(ts[w.seg_start], ts[w.end])
            self.assertTrue(all(not detect_gap(a, b) for a, b in zip(ts[w.seg_start:w.end], ts[w.seg_start + 1:w.end + 1])))

    def test_stride_and_first_window(self):
        ts = stamps(15, seconds=4.0)
        ends = [w.end for w in scoring_schedule(ts)]
        self.assertEqual(ends[0], 14)  # first full clip (0.1 s start tolerance)
        self.assertTrue(all(b - a == 8 for a, b in zip(ends, ends[1:])))

    def test_stride_zero_gives_every_ready_frame(self):
        ts = stamps(15, seconds=2.0)
        self.assertEqual([w.end for w in scoring_schedule(ts, stride_frames=0)], list(range(14, 30)))

    def test_empty(self):
        self.assertEqual(scoring_schedule([]), [])

    def test_aggregate_flags(self):
        self.assertEqual(aggregate_clip_flags([1, 1, 0, 0], [0, 0, 0, 1]), (True, False))
        self.assertEqual(aggregate_clip_flags([1, 0, 0], [1, 1, 0]), (False, True))


if __name__ == "__main__":
    unittest.main()
