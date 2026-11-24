"""IncidentTracker scripted-score tests (pure Python; no torch/cv2/numpy)."""
import unittest

from tests import _helpers  # noqa: F401
from incident import IncidentTracker
from thresholds import POST_ROLL_SECONDS, MIN_INCIDENT_SECONDS, MAX_INCIDENT_SECONDS

DT = 8 / 15  # one inference stride in seconds


class Feeder:
    def __init__(self, tracker=None, t0=1000.0):
        self.tr = tracker or IncidentTracker()
        self.t = t0
        self.events = []

    def feed(self, scores, **flags):
        out = []
        for s in scores:
            ev = self.tr.update(self.t, s, **flags)
            self.events.extend(ev)
            out.extend(ev)
            self.t += DT
        return out

    def types(self, kind):
        return [e for e in self.events if e["type"] == kind]


class IncidentTests(unittest.TestCase):
    def test_spike_confirmed_opens_once(self):
        f = Feeder()
        f.feed([0.1] * 4)
        out = f.feed([0.95])
        self.assertEqual([e["type"] for e in out], ["spike_pending"])
        self.assertFalse(f.tr.active)
        out = f.feed([0.80])
        self.assertEqual([e["type"] for e in out], ["open"])
        self.assertEqual(out[0]["trigger"], "spike")
        self.assertAlmostEqual(out[0]["peak"], 0.95)
        self.assertTrue(f.tr.active)

    def test_unconfirmed_spike_does_not_open(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.97, 0.2])
        self.assertEqual(f.types("open"), [])
        self.assertEqual(len(f.types("spike_unconfirmed")), 1)
        f.feed([0.1] * 10)
        self.assertFalse(f.tr.active)

    def test_spike_without_confirmation_requirement(self):
        f = Feeder(IncidentTracker(spike_requires_confirmation=False))
        f.feed([0.1] * 4)
        out = f.feed([0.95])
        self.assertEqual([e["type"] for e in out], ["open"])

    def test_sustained_opens_after_two_consecutive_avg_windows(self):
        f = Feeder()
        out = f.feed([0.7] * 3)
        self.assertEqual(out, [])  # history not yet full
        out = f.feed([0.7])  # first full window above threshold
        self.assertEqual(out, [])
        out = f.feed([0.7])  # second consecutive
        self.assertEqual([e["type"] for e in out], ["open"])
        self.assertEqual(out[0]["trigger"], "sustained")

    def test_single_above_average_window_is_not_enough(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.85, 0.85, 0.2, 0.1, 0.1])
        self.assertEqual(f.types("open"), [])

    def test_spike_dip_spike_within_post_roll_is_one_incident(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])          # open
        f.feed([0.1, 0.1, 0.1])      # dip (~1.6 s)
        f.feed([0.95, 0.9])          # re-trigger -> extend
        self.assertEqual(len(f.types("open")), 1)
        self.assertGreaterEqual(len(f.types("extend")), 2)
        self.assertTrue(f.tr.active)
        f.feed([0.05] * 40)          # long calm
        self.assertEqual(len(f.types("open")), 1)
        closes = f.types("close")
        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0]["reason"], "calm")
        self.assertGreaterEqual(closes[0]["n_triggers"], 3)
        self.assertGreaterEqual(closes[0]["duration"], MIN_INCIDENT_SECONDS)
        self.assertGreaterEqual(closes[0]["duration"], POST_ROLL_SECONDS)

    def test_min_length_respected(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        # calm: post-roll (5 s) elapses before min length (10 s)
        f.feed([0.05] * 14)  # ~7.5 s of calm
        self.assertTrue(f.tr.active)
        f.feed([0.05] * 10)
        self.assertFalse(f.tr.active)
        close = f.types("close")[0]
        self.assertGreaterEqual(close["duration"], MIN_INCIDENT_SECONDS)

    def test_unsuitable_mid_incident_keeps_recording(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        self.assertTrue(f.tr.active)
        # 30 s of unsuitable windows with calm scores: must NOT close
        f.feed([0.0] * 56, suitable=False)
        self.assertTrue(f.tr.active)
        self.assertEqual(f.types("close"), [])
        # as soon as clean calm windows arrive it closes
        f.feed([0.0] * 3)
        self.assertFalse(f.tr.active)

    def test_degraded_and_global_motion_cannot_open(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.99, 0.99], degraded=True)
        self.assertEqual(f.types("open"), [])
        f2 = Feeder()
        f2.feed([0.9] * 10, global_motion=True)
        self.assertEqual(f2.types("open"), [])
        f3 = Feeder()
        f3.feed([0.9] * 10, suitable=False)
        self.assertEqual(f3.types("open"), [])

    def test_pending_spike_not_confirmed_by_glitchy_window(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95])
        f.feed([0.9], global_motion=True)
        self.assertEqual(f.types("open"), [])

    def test_degraded_does_not_stop_active_incident(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        f.feed([0.0] * 60, degraded=True)
        self.assertTrue(f.tr.active)

    def test_cooldown_between_incidents(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        for _ in range(60):  # feed calm windows only until the incident closes
            f.feed([0.0])
            if f.types("close"):
                break
        self.assertEqual(len(f.types("close")), 1)
        # right after closing: spike+confirm inside the 5 s cooldown -> ignored
        f.feed([0.95, 0.9])
        self.assertEqual(len(f.types("open")), 1)
        f.feed([0.0] * 4)
        # after cooldown a new spike opens a second incident
        f.feed([0.0] * 12)
        f.feed([0.95, 0.9])
        self.assertEqual(len(f.types("open")), 2)

    def test_max_length_forces_close_then_new_incident_after_cooldown(self):
        f = Feeder()
        f.feed([0.95] * 4)
        n = int((MAX_INCIDENT_SECONDS + 30) / DT)
        f.feed([0.95] * n)
        closes = f.types("close")
        self.assertGreaterEqual(len(closes), 1)
        self.assertEqual(closes[0]["reason"], "max_length")
        self.assertLessEqual(closes[0]["duration"], MAX_INCIDENT_SECONDS + DT + 1e-6)
        self.assertGreaterEqual(len(f.types("open")), 2)

    def test_reset_closes_active_incident_without_cooldown(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        ev = f.tr.reset(f.t, reason="gap")
        self.assertEqual([e["type"] for e in ev], ["close"])
        self.assertEqual(ev[0]["reason"], "gap")
        self.assertFalse(f.tr.active)
        self.assertEqual(len(f.tr.history), 0)

    def test_gap_reset_keeps_incident_open(self):
        # A brief Wi-Fi gap mid-fight must not split one incident into two.
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        self.assertTrue(f.tr.active)
        ev = f.tr.reset(f.t, reason="gap", keep_incident=True)
        self.assertEqual(ev, [])
        self.assertTrue(f.tr.active)
        self.assertEqual(len(f.tr.history), 0)
        f.feed([0.95] * 6)
        self.assertEqual(len(f.types("open")), 1)

    def test_finish_closes_for_offline_use(self):
        f = Feeder()
        f.feed([0.1] * 4)
        f.feed([0.95, 0.8])
        ev = f.tr.finish(f.t)
        self.assertEqual(ev[0]["reason"], "end_of_video")


class FactoryTests(unittest.TestCase):
    def test_make_tracker_defaults_and_overrides(self):
        from incident import default_thresholds, make_tracker
        from thresholds import ALERT_THRESHOLD, SPIKE_THRESHOLD

        t = make_tracker()
        self.assertEqual((t.alert_threshold, t.spike_threshold), (ALERT_THRESHOLD, SPIKE_THRESHOLD))
        t = make_tracker({"alert_threshold": 0.5}, spike_threshold=0.8, score_history_len=None)
        self.assertEqual((t.alert_threshold, t.spike_threshold), (0.5, 0.8))
        self.assertEqual(t.history.maxlen, default_thresholds()["score_history_len"])


if __name__ == "__main__":
    unittest.main()
