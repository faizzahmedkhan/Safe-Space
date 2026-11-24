import os
import tempfile
import time
import unittest

from tests import _helpers  # noqa: F401
from retention import prune_dir
from validators import (
    is_clip_filename,
    is_evidence_filename,
    is_private_upstream,
    is_video_filename,
    keys_match,
    parse_origins,
    validate_camera_id,
    validate_host,
    validate_path,
)


class ValidatorTests(unittest.TestCase):
    def test_hosts_ok(self):
        for h in ("192.168.1.5", "10.0.0.1", "phone.local", "my-phone", "a.b-c.example.com"):
            self.assertTrue(validate_host(h), h)

    def test_hosts_rejected(self):
        for h in (
            "", "http://192.168.1.5", "192.168.1.5:8080", "192.168.1.5/video", "999.1.1.1",
            "1.2.3", "host name", "host;rm -rf", "evil.com/../x", "-bad.com", "a" * 300,
            "192.168.1.5\n", "user@host", "[::1]",
        ):
            self.assertFalse(validate_host(h), repr(h))

    def test_paths(self):
        for p in ("/video", "/shot.jpg", "/a/b-c_d.mjpeg", "/"):
            self.assertTrue(validate_path(p), p)
        for p in ("video", "", "/a b", "/a?x=1", "/a/../b", "/a;b", "http://x/y", "/a\\b"):
            self.assertFalse(validate_path(p), repr(p))

    def test_camera_ids(self):
        self.assertTrue(validate_camera_id("Camera-1"))
        self.assertTrue(validate_camera_id("entrance_2"))
        for c in ("", "a b", "../x", "a/b", "x" * 65):
            self.assertFalse(validate_camera_id(c), c)

    def test_private_upstream(self):
        res = lambda h: ["192.168.0.20"]
        self.assertTrue(is_private_upstream("192.168.1.5"))
        self.assertTrue(is_private_upstream("10.1.2.3"))
        self.assertTrue(is_private_upstream("172.16.5.5"))
        self.assertFalse(is_private_upstream("8.8.8.8"))
        self.assertFalse(is_private_upstream("127.0.0.1"))
        self.assertFalse(is_private_upstream("169.254.1.1"))
        self.assertTrue(is_private_upstream("phone.local", resolver=res))
        self.assertFalse(is_private_upstream("evil.com", resolver=lambda h: ["93.184.216.34"]))
        self.assertFalse(is_private_upstream("mixed.com", resolver=lambda h: ["192.168.0.1", "8.8.8.8"]))
        self.assertFalse(is_private_upstream("x.com", resolver=lambda h: (_ for _ in ()).throw(OSError())))

    def test_filenames(self):
        self.assertTrue(is_video_filename("a.MP4"))
        self.assertFalse(is_video_filename("a.exe"))
        self.assertTrue(is_evidence_filename("incident_Camera-1_2026-08-06_05-24-53.mp4"))
        self.assertFalse(is_evidence_filename("../x.mp4"))
        self.assertFalse(is_evidence_filename("a/b.mp4"))
        self.assertFalse(is_evidence_filename("x.avi"))
        self.assertTrue(is_clip_filename("upload_1_clip_event01.mp4"))
        self.assertTrue(is_clip_filename("a.webm"))
        self.assertFalse(is_clip_filename("a.txt"))
        self.assertFalse(is_clip_filename("../a.mp4"))

    def test_keys(self):
        self.assertTrue(keys_match(None, ""))       # auth disabled
        self.assertTrue(keys_match("abc", "abc"))
        self.assertFalse(keys_match("abd", "abc"))
        self.assertFalse(keys_match(None, "abc"))
        self.assertFalse(keys_match("", "abc"))

    def test_origins(self):
        d = ["http://localhost:3000"]
        self.assertEqual(parse_origins(None, d), d)
        self.assertEqual(parse_origins("  ", d), d)
        self.assertEqual(parse_origins("http://a.com, http://b.com/", d), ["http://a.com", "http://b.com"])


class RetentionTests(unittest.TestCase):
    def _mk(self, d, name, size, age_days):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"x" * size)
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))

    def test_age_and_protection(self):
        with tempfile.TemporaryDirectory() as d:
            self._mk(d, "old.mp4", 10, 40)
            self._mk(d, "old_protected.mp4", 10, 40)
            self._mk(d, "new.mp4", 10, 1)
            removed = prune_dir(d, max_mb=0, max_days=30, protected_names={"old_protected.mp4"})
            self.assertEqual(removed, ["old.mp4"])
            self.assertEqual(sorted(os.listdir(d)), ["new.mp4", "old_protected.mp4"])

    def test_size_budget_oldest_first_and_min_age(self):
        with tempfile.TemporaryDirectory() as d:
            mb = 1024 * 1024
            self._mk(d, "a.mp4", mb, 3)
            self._mk(d, "b.mp4", mb, 2)
            self._mk(d, "c.mp4", mb, 1)
            self._mk(d, "fresh.mp4", mb, 0)  # modified just now: never deleted
            removed = prune_dir(d, max_mb=2, max_days=0)
            self.assertEqual(removed, ["a.mp4", "b.mp4"])
            self.assertIn("fresh.mp4", os.listdir(d))

    def test_missing_dir(self):
        self.assertEqual(prune_dir("/nonexistent/dir/xyz", 1, 1), [])


if __name__ == "__main__":
    unittest.main()
