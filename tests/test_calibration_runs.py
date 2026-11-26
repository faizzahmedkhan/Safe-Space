"""Calibration run history (calibration_runs.py) and multi-folder retention."""
import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime

from tests import _helpers  # noqa: F401

import calibration_runs as cr
from retention import prune_dir
from validators import is_clip_filename


class _Spec:
    name = "v1"


MODEL_INFO = {"name": "safespace_bullying_model_v4_baseline", "sha256_12": "d95c2623b3d4",
              "version": "safespace_bullying_model_v4_baseline@d95c2623b3d4",
              "preprocessing": _Spec(), "thresholds_from_sidecar": True}
SETTINGS = {"target_fps": 15.0, "spike_threshold": 0.97, "alert_threshold": 0.67, "history_len": 4,
            "pad_seconds": 2.0, "merge_gap_seconds": 1.5}

# Shaped exactly like the August feature/video-calibration records.
LEGACY = {
    "ok": True,
    "video_path": "uploads/calibration/inputs\\upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira.mp4",
    "model_type": "3_class",
    "native_fps": 29.97,
    "events": [{"start_sec": 0.0, "end_sec": 264.13, "peak_score": 1.0, "trigger": "spike",
                "url": "/calibration/clip/upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira_3_class_event01_spike_0.0s-264.1s_peak1.00.mp4"}],
    "per_sample_log": [{"time_sec": 1.07, "raw_score": 1.0, "smoothed_avg": 1.0, "triggered": True,
                        "trigger_reason": "spike", "suitable_for_model": True},
                       {"time_sec": 1.6, "raw_score": 0.4, "smoothed_avg": 0.7, "triggered": False,
                        "trigger_reason": "", "suitable_for_model": False}],
}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ss_calib_")
        self.store = cr.CalibrationStore(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def touch(self, *parts, data=b"x"):
        p = os.path.join(self.root, *parts)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_layout_created(self):
        self.assertTrue(os.path.isdir(os.path.join(self.root, "inputs")))
        self.assertTrue(os.path.isdir(os.path.join(self.root, "outputs")))

    def test_names_are_safe_and_servable(self):
        rid = cr.CalibrationStore.new_run_id("uploads/calibration/inputs/upload_2026 Bar Fight#3.mp4",
                                             MODEL_INFO["name"], MODEL_INFO["sha256_12"],
                                             datetime(2026, 11, 26, 16, 30, 5))
        self.assertEqual(rid, "upload_2026_Bar_Fight_3__safespace_bullying_model_v4_baseline-d95c2623__20261126_163005")
        self.assertTrue(cr.is_run_id(rid))
        clip = cr.CalibrationStore.clip_name(rid, 3, "spike", 1.84, 6.5, 0.9412)
        self.assertTrue(clip.endswith("_event03_spike_1.8s-6.5s_peak0.94.mp4"))
        self.assertTrue(is_clip_filename(clip))
        inp = os.path.basename(self.store.new_input_path("../../evil name.MOV", "20261126_163005"))
        self.assertEqual(inp, "upload_20261126_163005_evil_name.mov")
        self.assertTrue(is_clip_filename(inp))

    def test_run_id_and_name_validation(self):
        for bad in ("", "../x", "a/b", "a\\b", "x" * 201, "a..b", "run id"):
            self.assertFalse(cr.is_run_id(bad), bad)
        self.assertIsNone(self.store.load("../outputs/x"))
        self.touch("inputs", "v.mp4")
        for bad in ("../inputs/v.mp4", "..", "v.txt", "inputs/v.mp4"):
            self.assertIsNone(self.store.find_input(bad), bad)
            self.assertIsNone(self.store.find_clip(bad), bad)

    def test_find_input_and_clip_search_order(self):
        self.touch("old_flat.mp4")                       # pre-history flat layout
        self.touch("inputs", "src.mp4")
        self.touch("outputs", "run_event01_spike_0.0s-1.0s_peak0.90.mp4")
        self.assertTrue(self.store.find_input("old_flat.mp4").endswith("old_flat.mp4"))
        self.assertTrue(self.store.find_input("src.mp4").endswith(os.path.join("inputs", "src.mp4")))
        self.assertIsNone(self.store.find_input("run_event01_spike_0.0s-1.0s_peak0.90.mp4"))
        self.assertIsNotNone(self.store.find_clip("run_event01_spike_0.0s-1.0s_peak0.90.mp4"))
        self.assertIsNotNone(self.store.find_clip("src.mp4"))

    def test_new_record_roundtrip(self):
        self.touch("inputs", "upload_x.mp4")
        rid = cr.CalibrationStore.new_run_id("upload_x.mp4", MODEL_INFO["name"], MODEL_INFO["sha256_12"])
        clip = cr.CalibrationStore.clip_name(rid, 1, "spike", 1.0, 3.0, 0.98)
        self.touch("outputs", clip)
        rec = cr.build_record(run_id=rid, video_path=os.path.join(self.root, "inputs", "upload_x.mp4"),
                              source={"kind": "upload"}, model_info=MODEL_INFO, settings=SETTINGS,
                              native_fps=30.0,
                              events=[{"start_sec": 1.0, "end_sec": 3.0, "peak_score": 0.98, "trigger": "spike",
                                       "url": f"/calibration/clip/{clip}"}],
                              per_sample_log=LEGACY["per_sample_log"], now=1_700_000_000.0)
        self.store.save(rec)
        got = self.store.load(rid)
        self.assertEqual(got["schema"], 2)
        self.assertEqual(got["model"]["version"], MODEL_INFO["version"])
        self.assertEqual(got["model"]["preprocessing"], "v1")
        self.assertEqual(got["settings"]["spike_threshold"], 0.97)
        self.assertTrue(got["video_available"])
        self.assertTrue(got["events"][0]["clip_available"])
        self.assertEqual(got["summary"]["n_windows"], 2)
        self.assertAlmostEqual(got["summary"]["peak_raw"], 1.0)
        self.assertEqual(got["summary"]["framing_suitable_ratio"], 0.5)
        self.assertNotIn("\\", got["video_path"])
        self.assertFalse(any(n.endswith(".tmp-%d" % os.getpid()) for n in os.listdir(self.store.outputs)))

    def test_legacy_record_is_listed_and_opened(self):
        with open(os.path.join(self.store.outputs, "upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira_3_class_run.json"), "w") as f:
            json.dump(LEGACY, f)
        self.touch("inputs", "upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira.mp4")
        runs = self.store.list()
        self.assertEqual(len(runs), 1)
        s = runs[0]
        self.assertEqual(s["schema"], 1)
        self.assertEqual(s["model_name"], "legacy: 3_class")
        self.assertIsNone(s["spike_threshold"])
        self.assertEqual(s["video_name"], "upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira.mp4")
        self.assertTrue(s["video_available"])
        rec = self.store.load(s["run_id"])
        self.assertEqual(rec["video_url"], "/calibration/clip/upload_20260831_152704_Ilia_Topuria_vs_Charles_Oliveira.mp4")
        self.assertFalse(rec["events"][0]["clip_available"])  # clip file not present here
        self.assertIsNone(rec["settings"])

    def test_list_newest_first_and_skips_garbage(self):
        for i, ts in enumerate((1_700_000_000.0, 1_700_000_500.0, 1_699_000_000.0)):
            rid = f"vid{i}__m-abc__2026010{i}_000000"
            self.store.save(cr.build_record(run_id=rid, video_path=f"inputs/vid{i}.mp4", source={"kind": "upload"},
                                            model_info=MODEL_INFO, settings=SETTINGS, native_fps=15.0,
                                            events=[], per_sample_log=[], now=ts))
        with open(os.path.join(self.store.outputs, "broken_run.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(self.store.outputs, "notes.txt"), "w") as f:
            f.write("ignored")
        ids = [r["run_id"] for r in self.store.list()]
        self.assertEqual(ids, ["vid1__m-abc__20260101_000000", "vid0__m-abc__20260100_000000",
                               "vid2__m-abc__20260102_000000"])
        self.assertEqual(self.store.run_json_names(), {f"{i}_run.json" for i in ids} | {"broken_run.json"})


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ss_ret_")
        self.store = cr.CalibrationStore(self.root)
        now = time.time()
        self.now = now
        # (folder, name, size MB, age days)
        for folder, name, mb, days in (
            ("inputs", "old_input.mp4", 3, 90),
            ("outputs", "old_clip.mp4", 2, 60),
            ("outputs", "old_run.json", 0.01, 90),
            ("outputs", "new_clip.mp4", 2, 1),
            ("", "flat_legacy.mp4", 1, 30),
        ):
            p = os.path.join(self.root, folder, name)
            with open(p, "wb") as f:
                f.write(b"\0" * int(mb * 1024 * 1024))
            os.utime(p, (now - days * 86400, now - days * 86400))
        os.rename(os.path.join(self.root, "outputs", "old_run.json"),
                  os.path.join(self.root, "outputs", "old_run_run.json"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def dirs(self):
        return [self.root, self.store.inputs, self.store.outputs]

    def test_size_only_pool_keeps_history_records(self):
        removed = prune_dir(self.dirs(), 4.5, 0, protected_names=self.store.run_json_names(), now=self.now)
        # 8 MB of videos over a 4.5 MB budget: oldest go first, across folders
        self.assertEqual(removed, ["old_input.mp4", "old_clip.mp4"])
        self.assertTrue(os.path.exists(os.path.join(self.store.outputs, "old_run_run.json")))
        self.assertTrue(os.path.exists(os.path.join(self.store.outputs, "new_clip.mp4")))

    def test_no_age_limit_by_default(self):
        from retention import retention_budgets

        old = os.environ.pop("SAFESPACE_CALIBRATION_MAX_DAYS", None)
        try:
            self.assertEqual(retention_budgets()["calibration"][1], 0)
        finally:
            if old is not None:
                os.environ["SAFESPACE_CALIBRATION_MAX_DAYS"] = old
        self.assertEqual(prune_dir(self.dirs(), 100, 0, protected_names=self.store.run_json_names(), now=self.now), [])

    def test_single_folder_call_unchanged(self):
        removed = prune_dir(self.store.outputs, 0, 30, now=self.now)
        self.assertEqual(sorted(removed), ["old_clip.mp4", "old_run_run.json"])


if __name__ == "__main__":
    unittest.main()
