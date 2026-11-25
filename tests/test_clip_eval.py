"""Deployment-aligned evaluation: metrics, live-rule replay, selection (numpy only)."""
import math
import unittest

from tests import _helpers  # noqa: F401

try:
    import numpy as np

    import clip_eval as ce
    from incident import make_tracker

    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    HAVE_SK = True
except Exception:  # pragma: no cover
    HAVE_SK = False


def vs(raw, label=1, cls="Violence", suitable=None, glob=None, segment=None, dt=8 / 15):
    raw = np.asarray(raw, float)
    n = len(raw)
    return ce.VideoScores(
        path=f"clip_{label}_{n}",
        label=label,
        cls=cls,
        t_end=np.arange(n) * dt + 1.0,
        raw=raw,
        suitable=np.ones(n, bool) if suitable is None else np.asarray(suitable, bool),
        global_motion=np.zeros(n, bool) if glob is None else np.asarray(glob, bool),
        segment=np.zeros(n, int) if segment is None else np.asarray(segment, int),
    )


@unittest.skipUnless(HAVE, "numpy not installed")
class MetricTests(unittest.TestCase):
    def test_average_precision_hand_computed(self):
        # ranks: 0.9(+) 0.8(-) 0.7(+) 0.1(-) -> AP = 0.5*1 + 0.5*(2/3)
        self.assertAlmostEqual(ce.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), 0.5 + 1 / 3)
        self.assertEqual(ce.average_precision([1, 1], [0.2, 0.3]), 1.0)
        self.assertTrue(math.isnan(ce.average_precision([0, 0], [0.2, 0.3])))

    def test_roc_auc_hand_computed(self):
        self.assertEqual(ce.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(ce.roc_auc([0, 1], [0.5, 0.5]), 0.5)  # tie
        self.assertAlmostEqual(ce.roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), 0.75)
        self.assertTrue(math.isnan(ce.roc_auc([1, 1], [0.1, 0.2])))

    @unittest.skipUnless(HAVE_SK, "sklearn not installed")
    def test_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            n = int(rng.integers(2, 60))
            y = rng.integers(0, 2, n)
            if y.min() == y.max():
                continue
            s = np.round(rng.random(n), int(rng.integers(1, 3)))
            self.assertAlmostEqual(ce.average_precision(y, s), average_precision_score(y, s), places=12)
            self.assertAlmostEqual(ce.roc_auc(y, s), roc_auc_score(y, s), places=12)

    def test_recall_at_normal_recall_is_optimal(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(4, 50))
            y = rng.integers(0, 2, n)
            if y.min() == y.max():
                continue
            s = np.round(rng.random(n), 1)
            r = ce.recall_at_normal_recall(y, s, 0.9)
            self.assertGreaterEqual(r["normal_recall"], 0.9 - 1e-12)
            best = max(
                np.mean(s[y == 1] > t)
                for t in np.r_[-1.0, np.unique(s)]
                if np.mean(s[y == 0] <= t) >= 0.9
            )
            self.assertAlmostEqual(r["dangerous_recall"], best)

    def test_decision_metrics(self):
        m = ce.decision_metrics([0, 0, 0, 1, 1], [0, 1, 0, 1, 0])
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["normal_recall"], 2 / 3)
        self.assertAlmostEqual(m["dangerous_recall"], 0.5)
        self.assertAlmostEqual(m["false_alarm_rate"], 1 / 3)
        f1_d = 2 * 0.5 * 0.5 / 1.0
        f1_n = 2 * (2 / 3) * (2 / 3) / (4 / 3)
        self.assertAlmostEqual(m["macro_f1"], (f1_d + f1_n) / 2)


@unittest.skipUnless(HAVE, "numpy not installed")
class LiveReplayTests(unittest.TestCase):
    def manual(self, v, thresholds=None):
        tr = make_tracker(thresholds)
        opens = 0
        for t, r, s, g in zip(v.t_end, v.raw, v.suitable, v.global_motion):
            opens += sum(e["type"] == "open" for e in tr.update(t, r, suitable=s, global_motion=g))
        return opens

    def test_confirmed_spike_alerts(self):
        out = ce.simulate_live(vs([0.1, 0.95, 0.7, 0.1]))
        self.assertTrue(out["would_alert"])
        self.assertEqual(out["trigger"], "spike")

    def test_unconfirmed_spike_does_not_alert(self):
        self.assertFalse(ce.simulate_live(vs([0.1, 0.95, 0.2, 0.1]))["would_alert"])

    def test_sustained_alerts(self):
        out = ce.simulate_live(vs([0.7, 0.7, 0.7, 0.7, 0.7]))
        self.assertTrue(out["would_alert"])
        self.assertEqual(out["trigger"], "sustained")
        self.assertAlmostEqual(out["peak_smoothed"], 0.7)

    def test_unsuitable_and_global_motion_block_opening(self):
        self.assertFalse(ce.simulate_live(vs([0.1, 0.95, 0.9, 0.95, 0.9], suitable=[0] * 5))["would_alert"])
        self.assertFalse(ce.simulate_live(vs([0.1, 0.95, 0.9, 0.95, 0.9], glob=[1] * 5))["would_alert"])

    def test_gap_resets_history(self):
        # Without the reset the 4-window average would reach 0.67 twice.
        raw = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        seg = [0, 0, 0, 1, 1, 1]
        self.assertTrue(ce.simulate_live(vs(raw))["would_alert"])
        self.assertFalse(ce.simulate_live(vs(raw, segment=seg))["would_alert"])

    def test_thresholds_override(self):
        v = vs([0.6, 0.6, 0.6, 0.6, 0.6])
        self.assertFalse(ce.simulate_live(v)["would_alert"])
        self.assertTrue(ce.simulate_live(v, {"alert_threshold": 0.5})["would_alert"])

    def test_matches_manual_tracker_on_random_scores(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            v = vs(rng.random(int(rng.integers(1, 30))), suitable=None)
            self.assertEqual(ce.simulate_live(v)["n_incidents"], self.manual(v))

    def test_clip_row(self):
        row = ce.clip_row(vs([0.2, 0.95, 0.8, 0.4], label=1, cls="Weaponized"))
        self.assertEqual(row["y_true"], 1)
        self.assertEqual(row["class"], "Weaponized")
        self.assertEqual(row["n_windows"], 4)
        self.assertAlmostEqual(row["first_p"], 0.2)
        self.assertAlmostEqual(row["max_p"], 0.95)
        self.assertAlmostEqual(row["mean_p"], (0.2 + 0.95 + 0.8 + 0.4) / 4)
        self.assertTrue(row["would_alert"])


@unittest.skipUnless(HAVE, "numpy not installed")
class SummaryTests(unittest.TestCase):
    def rows(self):
        clips = [
            vs([0.1, 0.95, 0.8], 1, "Violence"),
            vs([0.2, 0.7, 0.7, 0.7, 0.7, 0.7], 1, "Weaponized"),
            vs([0.1, 0.2, 0.1], 0, "Normal"),
            vs([0.3, 0.92, 0.75], 0, "Normal"),  # a false alarm
        ]
        return [ce.clip_row(c) for c in clips]

    def test_summary_structure_and_values(self):
        s = ce.summarize(self.rows())
        self.assertEqual((s["n_clips"], s["n_dangerous"]), (4, 2))
        self.assertEqual(set(s["scores"]), set(ce.CLIP_SCORES))
        self.assertEqual((s["live"]["tp"], s["live"]["fp"], s["live"]["tn"]), (2, 1, 1))
        self.assertEqual(s["live_alert_rate_by_class"], {"Normal": 0.5, "Violence": 1.0, "Weaponized": 1.0})
        text = ce.format_summary(s)
        self.assertIn("max_p", text)
        self.assertIn("TN=1 FP=1", text)

    def test_selection(self):
        s = ce.summarize(self.rows())
        for name in ce.SELECTION_METRICS:
            self.assertIsInstance(ce.selection_value(s, name), float)
        self.assertEqual(ce.DEFAULT_SELECTION_METRIC, "pr_auc_max")
        with self.assertRaises(KeyError):
            ce.selection_value(s, "accuracy")
        one_class = ce.summarize([r for r in self.rows() if r["y_true"] == 0])
        self.assertEqual(ce.selection_value(one_class, "pr_auc_max"), -math.inf)


@unittest.skipUnless(HAVE, "numpy not installed")
class UncertaintyTests(unittest.TestCase):
    @staticmethod
    def rows(sep=1.0, n=300, seed=0, n_windows=3):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            y = int(i % 3 != 0)
            p = float(np.clip(rng.normal(0.45 + 0.3 * sep * y, 0.2), 0, 1))
            out.append({"path": f"clip{i}", "class": "Violence" if y else "Normal", "y_true": y,
                        "n_windows": n_windows, "short": False, "first_p": p, "max_p": p, "mean_p": p,
                        "peak_smoothed": p, "would_alert": p > 0.8, "n_incidents": int(p > 0.8),
                        "trigger": "spike" if p > 0.8 else "", "suitable_frac": 1.0})
        return out

    def test_intervals_contain_point_value(self):
        ci = ce.bootstrap_metrics(self.rows(), n_boot=300)
        self.assertEqual(set(ci), set(ce.CI_METRICS))
        for m in ci.values():
            self.assertLessEqual(m["low"], m["value"])
            self.assertLessEqual(m["value"], m["high"])

    def test_cluster_resampling_widens_intervals(self):
        # 300 clips that are really 30 clusters of 10 near-identical copies:
        # resampling clips pretends there are 300 independent samples.
        base = self.rows(n=30, seed=3)
        rows = [dict(r, path=f"{r['path']}_{k}") for r in base for k in range(10)]
        groups = {r["path"]: r["path"].split("_")[0] for r in rows}
        naive = ce.bootstrap_metrics(rows, metrics=["roc_auc_max"], n_boot=400)["roc_auc_max"]
        grouped = ce.bootstrap_metrics(rows, groups=groups, metrics=["roc_auc_max"], n_boot=400)["roc_auc_max"]
        self.assertGreater(grouped["high"] - grouped["low"], 1.8 * (naive["high"] - naive["low"]))

    def test_compare_identical_runs(self):
        a = self.rows()
        cmp = ce.compare_runs(a, [dict(r) for r in a], n_boot=200)
        for name in ce.CI_METRICS:
            self.assertEqual(cmp[name]["diff"], 0.0)
            self.assertEqual((cmp[name]["low"], cmp[name]["high"]), (0.0, 0.0))
            self.assertEqual(cmp[name]["p_b_not_better"], 1.0)

    def test_compare_detects_a_better_run(self):
        a = self.rows(sep=0.6, seed=1)
        b = [dict(r, max_p=min(1.0, r["max_p"] + (0.25 if r["y_true"] else 0.0))) for r in a]
        cmp = ce.compare_runs(a, b, n_boot=300)["roc_auc_max"]
        self.assertGreater(cmp["diff"], 0)
        self.assertGreater(cmp["low"], 0)
        self.assertLess(cmp["p_b_not_better"], 0.05)
        text = ce.format_comparison(ce.compare_runs(a, b, n_boot=50), "base", "new")
        self.assertIn("difference = new - base", text)

    def test_compare_uses_shared_clips_and_checks_labels(self):
        a = self.rows()
        self.assertEqual(ce.compare_runs(a, a[:100], n_boot=20)["_n_clips"]["value"], 100)
        with self.assertRaises(ValueError):
            ce.compare_runs(a, [dict(r, path="other" + r["path"]) for r in a])
        flipped = [dict(r, y_true=1 - r["y_true"]) for r in a]
        with self.assertRaises(ValueError):
            ce.compare_runs(a, flipped)

    def test_csv_roundtrip(self):
        import os
        import tempfile

        rows = self.rows(n=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scores.csv")
            ce.write_rows_csv(rows, path)
            self.assertEqual(ce.read_rows_csv(path), rows)

    def test_live_rule_reported_on_long_clips(self):
        self.assertEqual(ce.LIVE_RULE_MIN_WINDOWS, 5)
        rows = self.rows(n=30, n_windows=1) + self.rows(n=12, seed=4, n_windows=6)
        s = ce.summarize(rows)
        self.assertEqual(s["live_long_clips"]["n_clips"], 12)
        self.assertEqual(s["live_long_clips"]["min_windows"], 5)
        self.assertIn("clips with >= 5 windows (12 clips", ce.format_summary(s))


if __name__ == "__main__":
    unittest.main()
