"""Tests for the scan profiles and the window grid they generate."""

import sys
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from model.predict import (
    SCAN_PROFILES,
    ScanProfile,
    count_windows,
    get_profile,
    merge_hits,
    prettify,
    same_object,
    softmax,
    window_boxes,
)

# a middling phone photograph
FRAME = (1280, 960)


class TestProfiles(unittest.TestCase):

    def test_the_three_profiles_are_there(self):
        self.assertEqual(set(SCAN_PROFILES), {"full", "balanced", "edge"})

    def test_a_name_resolves_to_its_profile(self):
        self.assertIs(get_profile("edge"), SCAN_PROFILES["edge"])

    def test_a_profile_passes_through_unchanged(self):
        profile = SCAN_PROFILES["full"]

        self.assertIs(get_profile(profile), profile)

    def test_no_argument_gives_a_usable_default(self):
        self.assertIsInstance(get_profile(), ScanProfile)

    def test_an_unknown_name_is_refused(self):
        with self.assertRaises(ValueError):
            get_profile("tiny")


class TestWindowGrid(unittest.TestCase):

    def test_lighter_profiles_mean_fewer_forward_passes(self):
        full = count_windows(*FRAME, "full")
        balanced = count_windows(*FRAME, "balanced")
        edge = count_windows(*FRAME, "edge")

        # this ordering is the whole reason the profiles exist
        self.assertGreater(full, balanced)
        self.assertGreater(balanced, edge)

    def test_the_edge_profile_is_far_cheaper(self):
        self.assertLess(
            count_windows(*FRAME, "edge"),
            count_windows(*FRAME, "full") / 5
        )

    def test_windows_stay_inside_the_frame(self):
        width, height = FRAME

        for name in SCAN_PROFILES:
            with self.subTest(profile=name):
                for x1, y1, x2, y2 in window_boxes(width, height, name):
                    self.assertGreaterEqual(x1, 0)
                    self.assertGreaterEqual(y1, 0)
                    self.assertLessEqual(x2, width)
                    self.assertLessEqual(y2, height)
                    self.assertLess(x1, x2)
                    self.assertLess(y1, y2)

    def test_the_far_edges_are_covered(self):
        width, height = FRAME

        boxes = window_boxes(width, height, "full")

        self.assertTrue(any(box[2] == width for box in boxes))
        self.assertTrue(any(box[3] == height for box in boxes))

    def test_no_window_is_generated_twice(self):
        boxes = window_boxes(*FRAME, "full")

        self.assertEqual(len(boxes), len(set(boxes)))

    def test_a_thumbnail_sized_frame_still_produces_windows(self):
        self.assertGreater(count_windows(200, 150, "edge"), 0)

    def test_scales_below_the_pixel_floor_are_dropped(self):
        # a 100px frame at scale 0.33 is 33px, under min_pixels
        self.assertLess(
            count_windows(100, 100, "full"),
            count_windows(1000, 1000, "full")
        )


class TestMerging(unittest.TestCase):

    def test_two_windows_on_the_same_spot_are_one_object(self):
        self.assertTrue(same_object((0, 0, 100, 100), (10, 10, 110, 110)))

    def test_windows_far_apart_are_not(self):
        self.assertFalse(same_object((0, 0, 50, 50), (500, 500, 550, 550)))

    def test_a_small_window_inside_a_big_one_is_the_same_object(self):
        self.assertTrue(same_object((0, 0, 400, 400), (100, 100, 200, 200)))

    def test_merging_widens_the_survivor(self):
        merged = merge_hits([
            ("Track", 0.9, (0, 0, 100, 100)),
            ("Track", 0.8, (10, 10, 110, 110)),
        ])

        self.assertEqual(len(merged), 1)

        label, confidence, box = merged[0]

        self.assertEqual(label, "Track")

        # the stronger hit's score is kept, its box is stretched to cover both
        self.assertAlmostEqual(confidence, 0.9)
        self.assertEqual(box, (0, 0, 110, 110))

    def test_windows_that_barely_touch_stay_separate(self):
        # 0.14 IoU, under the merge threshold - two objects, not one
        merged = merge_hits([
            ("Track", 0.9, (0, 0, 100, 100)),
            ("Track", 0.8, (50, 50, 150, 150)),
        ])

        self.assertEqual(len(merged), 2)

    def test_different_classes_are_never_merged(self):
        merged = merge_hits([
            ("Track", 0.9, (0, 0, 100, 100)),
            ("Signal", 0.8, (10, 10, 110, 110)),
        ])

        self.assertEqual(len(merged), 2)


class TestHelpers(unittest.TestCase):

    def test_labels_are_made_readable(self):
        self.assertEqual(prettify("overhead_wire"), "Overhead Wire")

    def test_softmax_rows_are_distributions(self):
        rows = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))

        np.testing.assert_allclose(rows.sum(axis=1), [1.0, 1.0], rtol=1e-6)
        self.assertTrue((rows >= 0).all())

    def test_softmax_survives_large_logits(self):
        # the shift inside softmax is what stops this overflowing
        rows = softmax(np.array([[1000.0, 1001.0]]))

        self.assertTrue(np.isfinite(rows).all())
        self.assertAlmostEqual(rows.sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
