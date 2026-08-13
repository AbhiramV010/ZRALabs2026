"""Tests for DetectionResult, the shape everything else passes around."""

import json
import sys
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from result import DetectionResult


class TestColors(unittest.TestCase):

    def test_a_known_class_gets_its_own_color(self):
        res = DetectionResult(label="Train")

        self.assertEqual(res.get_color(), "#F87171")
        self.assertEqual(res.get_badge_color(), "red")

    def test_lookup_ignores_case(self):
        # predict.py title-cases labels, the maps are keyed lower
        self.assertEqual(
            DetectionResult(label="Overhead Wire").get_color(),
            DetectionResult(label="overhead wire").get_color()
        )

    def test_an_unknown_class_falls_back_to_grey(self):
        res = DetectionResult(label="Locomotive Shed")

        self.assertEqual(res.get_color(), "#94A3B8")
        self.assertEqual(res.get_badge_color(), "gray")

    def test_every_color_has_a_matching_badge(self):
        self.assertEqual(
            set(DetectionResult.COLOR_MAP),
            set(DetectionResult.BADGE_MAP)
        )


class TestSerialisation(unittest.TestCase):

    def make_one(self):
        return DetectionResult(
            label="Signal",
            description="Controls train movements",
            confidence=0.8123,
            rect_1=(12, 34),
            rect_2=(56, 78)
        )

    def test_round_trip_preserves_every_field(self):
        original = self.make_one()

        restored = DetectionResult.from_dict(original.to_dict())

        self.assertEqual(restored.label, original.label)
        self.assertEqual(restored.description, original.description)
        self.assertAlmostEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.rect_1, original.rect_1)
        self.assertEqual(restored.rect_2, original.rect_2)

    def test_round_trip_through_json(self):
        original = self.make_one()

        restored = DetectionResult.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        # JSON has no tuples, from_dict has to put them back
        self.assertEqual(restored.rect_1, (12, 34))
        self.assertEqual(restored.rect_2, (56, 78))

    def test_corners_serialise_as_lists(self):
        data = self.make_one().to_dict()

        self.assertIsInstance(data["rect_1"], list)
        self.assertIsInstance(data["rect_2"], list)

    def test_from_dict_fills_in_what_is_missing(self):
        res = DetectionResult.from_dict({"label": "Track"})

        self.assertEqual(res.label, "Track")
        self.assertEqual(res.description, "")
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(res.rect_1, (0, 0))
        self.assertEqual(res.rect_2, (0, 0))

    def test_from_dict_of_an_empty_payload(self):
        res = DetectionResult.from_dict({})

        self.assertEqual(res.label, "")
        self.assertEqual(res.get_badge_color(), "gray")

    def test_a_restored_result_still_knows_its_color(self):
        restored = DetectionResult.from_dict(self.make_one().to_dict())

        self.assertEqual(restored.get_color(), "#FACC15")


if __name__ == "__main__":
    unittest.main()
