"""Offline unit tests (no game, no GUI).   Run:  python -m unittest discover tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from app.db import match, clean
from app.learning import StateEncoder, edit_similarity, compare_examples
from app.perception import (TemplateDetector, Tracker, detect_captcha, extract_scene,
                            infer_class)
from app.profiles import validate, PROFILE_TEMPLATE


class Quiet:
    def __getattr__(self, _):
        return lambda *a, **k: None


class TestDb(unittest.TestCase):
    def test_match(self):
        d = {"a": 1, "b": "x", "t": 5}
        self.assertTrue(match(d, {"a": 1}))
        self.assertFalse(match(d, {"a": 2}))
        self.assertTrue(match(d, {"t": {"$gte": 5, "$lt": 6}}))
        self.assertTrue(match(d, {"b": {"$in": ["x", "y"]}}))

    def test_clean_numpy(self):
        out = clean({"v": np.float32(1.5), "arr": np.arange(3), "p": Path("x")})
        self.assertEqual(out["arr"], [0, 1, 2])
        self.assertIsInstance(out["v"], float)

    def test_tinydb_roundtrip(self):
        try:
            import tinydb  # noqa
        except ImportError:
            self.skipTest("tinydb not installed")
        from app.config import Settings
        from app.db import Database
        s = Settings()
        tmp = Path(tempfile.mkdtemp())
        s["tinydb_path"] = str(tmp / "t.json")
        s["logs_dir"] = str(tmp / "logs")
        db = Database(s, Quiet())
        i = db.insert("sessions", {"start_time": 1.0, "status": "x"})
        self.assertEqual(db.get("sessions", i)["status"], "x")
        db.update("sessions", i, {"status": "y"})
        self.assertEqual(db.find("sessions", {"status": "y"})[0]["_id"], i)
        self.assertEqual(db.count("sessions"), 1)
        db.delete("sessions", {"_id": i})
        self.assertEqual(db.count("sessions"), 0)
        db.close()


class TestProfiles(unittest.TestCase):
    def test_validate_fills_defaults(self):
        p = validate({"name": "x", "detector": {"labels": ["a"]}})
        self.assertIn("input_mapping", p)
        self.assertEqual(p["detector"]["labels"], ["a"])

    def test_validate_rejects(self):
        with self.assertRaises(ValueError):
            validate({"name": ""})


class TestPerception(unittest.TestCase):
    def test_template_detection_and_scene(self):
        root = Path(tempfile.mkdtemp())
        (root / "p" / "enemy").mkdir(parents=True)
        frame = np.random.default_rng(3).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "p" / "enemy" / "slime.png"), frame[50:90, 300:340])
        ents = TemplateDetector(root, "p", 0.8, 0.5, Quiet()).detect(frame)
        self.assertEqual(len(ents), 1)
        self.assertLess(abs(ents[0].bbox[0] - 300), 4)
        Tracker().update(ents, 640)
        self.assertEqual(ents[0].track_id, 1)
        sc = extract_scene(ents, frame.shape, [], ["player", "enemy"])
        self.assertEqual(sc["counts"]["enemy"], 1)

    def test_captcha_is_detected_not_solved(self):
        self.assertTrue(detect_captcha(["Please verify you are human"], [], [])["detected"])
        self.assertFalse(detect_captcha(["Level 5"], [], [])["detected"])

    def test_unknown_class_when_no_evidence(self):
        prof = {"character": {"class_signatures": {"Mage": ["staff", "orb"]}}}
        self.assertEqual(infer_class(prof, [], [])["candidate"], "Unknown")


class TestLearning(unittest.TestCase):
    def test_edit_similarity(self):
        self.assertEqual(edit_similarity(list("abc"), list("abc")), 1.0)
        self.assertAlmostEqual(edit_similarity(list("abc"), list("abd")), 2 / 3)

    def test_compare_examples(self):
        enc = StateEncoder(["player", "enemy"])
        s = {"counts": {"player": 1}, "mean_pos": {}, "player": {"cx": .5, "cy": .5}}
        ex = {"state_features": enc.encode(s), "action": {"control": "z"},
              "context_actions": ["a"], "outcome": {"enemy_delta": -1}}
        r = compare_examples(ex, ex)
        self.assertEqual(r["state_distance"], 0.0)
        self.assertTrue(r["same_action"])


if __name__ == "__main__":
    unittest.main()
