"""Tests for the on-device capture store.

    python -m unittest discover test
"""

import json
import shutil
import sys
import tempfile
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from result import DetectionResult
from store import (
    SYNC_PENDING,
    SYNC_SYNCED,
    THUMBNAIL_EDGE,
    CaptureStore,
    read_exif_location,
)


def make_image(width=1600, height=1200, color=(90, 120, 160)):
    return Image.new("RGB", (width, height), color)


def make_detections():
    return [
        DetectionResult(
            label="Track",
            description="The steel rails",
            confidence=0.91,
            rect_1=(10, 20),
            rect_2=(300, 400)
        ),
        DetectionResult(
            label="Signal",
            description="Controls movements",
            confidence=0.62,
            rect_1=(500, 60),
            rect_2=(620, 240)
        ),
    ]


class StoreTestCase(unittest.TestCase):
    """Each test gets its own empty store directory."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="zra-store-"))
        self.store = CaptureStore(root=self.root)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=True)


class TestSchema(StoreTestCase):

    def test_starts_empty(self):
        stats = self.store.stats()

        self.assertEqual(stats["captures"], 0)
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["detections"], 0)
        self.assertEqual(stats["thumb_bytes"], 0)

    def test_database_file_is_created(self):
        self.assertTrue((self.root / "captures.db").is_file())

    def test_device_id_survives_a_restart(self):
        first = self.store.device_id

        self.store.close()
        reopened = CaptureStore(root=self.root)

        self.addCleanup(reopened.close)

        self.assertEqual(reopened.device_id, first)

    def test_device_ids_differ_between_units(self):
        other_root = Path(tempfile.mkdtemp(prefix="zra-store-"))
        other = CaptureStore(root=other_root)

        self.addCleanup(shutil.rmtree, other_root, ignore_errors=True)
        self.addCleanup(other.close)

        self.assertNotEqual(other.device_id, self.store.device_id)


class TestRecording(StoreTestCase):

    def test_record_returns_a_uuid_and_counts_the_row(self):
        capture_uuid = self.store.record(make_image(), make_detections())

        self.assertTrue(capture_uuid)

        stats = self.store.stats()

        self.assertEqual(stats["captures"], 1)
        self.assertEqual(stats["detections"], 2)
        self.assertEqual(stats["pending"], 1)

    def test_original_dimensions_are_kept_not_the_thumbnails(self):
        capture_uuid = self.store.record(make_image(1600, 1200), [])

        capture = self.store.get(capture_uuid)

        self.assertEqual(capture["width"], 1600)
        self.assertEqual(capture["height"], 1200)

    def test_metadata_is_stored(self):
        capture_uuid = self.store.record(
            make_image(),
            [],
            source_name="DSC_0001.jpg",
            location=(51.5074, -0.1278),
            model_name="railway_classifier",
            model_version="resnet18-seed42"
        )

        capture = self.store.get(capture_uuid)

        self.assertEqual(capture["source_name"], "DSC_0001.jpg")
        self.assertAlmostEqual(capture["latitude"], 51.5074)
        self.assertAlmostEqual(capture["longitude"], -0.1278)
        self.assertEqual(capture["model_name"], "railway_classifier")
        self.assertEqual(capture["model_version"], "resnet18-seed42")
        self.assertEqual(capture["device_id"], self.store.device_id)

    def test_detections_come_back_strongest_first(self):
        capture_uuid = self.store.record(make_image(), make_detections())

        labels = [
            hit["label"] for hit in self.store.get(capture_uuid)["detections"]
        ]

        self.assertEqual(labels, ["Track", "Signal"])

    def test_detection_boxes_survive_the_round_trip(self):
        capture_uuid = self.store.record(make_image(), make_detections())

        top = self.store.get(capture_uuid)["detections"][0]

        self.assertEqual(tuple(top["rect_1"]), (10, 20))
        self.assertEqual(tuple(top["rect_2"]), (300, 400))
        self.assertAlmostEqual(top["confidence"], 0.91, places=5)

    def test_an_image_with_nothing_in_it_is_still_recorded(self):
        capture_uuid = self.store.record(make_image(), [])

        capture = self.store.get(capture_uuid)

        self.assertEqual(capture["detections"], [])
        self.assertEqual(self.store.stats()["captures"], 1)

    def test_unknown_uuid_is_none(self):
        self.assertIsNone(self.store.get("not-a-real-uuid"))

    def test_a_capture_is_json_serialisable(self):
        capture_uuid = self.store.record(make_image(), make_detections())

        # what a sync client will actually put on the wire
        encoded = json.dumps(self.store.get(capture_uuid))

        self.assertIn("Track", encoded)


class TestThumbnails(StoreTestCase):

    def test_thumbnail_is_written_and_readable(self):
        capture_uuid = self.store.record(make_image(), [])

        capture = self.store.get(capture_uuid)
        path = self.store.thumbnail_path(capture)

        self.assertTrue(path.is_file())

        with Image.open(path) as thumb:
            self.assertEqual(thumb.format, "JPEG")

    def test_large_images_are_downscaled(self):
        capture_uuid = self.store.record(make_image(4032, 3024), [])

        path = self.store.thumbnail_path(self.store.get(capture_uuid))

        with Image.open(path) as thumb:
            self.assertLessEqual(max(thumb.size), THUMBNAIL_EDGE)

            # 4:3 in, 4:3 out
            self.assertAlmostEqual(thumb.width / thumb.height, 4 / 3, places=2)

    def test_small_images_are_not_blown_up(self):
        capture_uuid = self.store.record(make_image(320, 240), [])

        path = self.store.thumbnail_path(self.store.get(capture_uuid))

        with Image.open(path) as thumb:
            self.assertEqual(thumb.size, (320, 240))

    def test_thumbnail_path_is_stored_relative(self):
        capture_uuid = self.store.record(make_image(), [])

        capture = self.store.get(capture_uuid)

        self.assertFalse(Path(capture["thumb_path"]).is_absolute())
        self.assertTrue(self.store.thumbnail_path(capture).is_file())

    def test_recorded_size_matches_the_file_on_disk(self):
        capture_uuid = self.store.record(make_image(), [])

        capture = self.store.get(capture_uuid)

        self.assertEqual(
            capture["thumb_bytes"],
            self.store.thumbnail_path(capture).stat().st_size
        )

    def test_a_greyscale_image_is_converted_rather_than_refused(self):
        capture_uuid = self.store.record(Image.new("L", (800, 600), 128), [])

        path = self.store.thumbnail_path(self.store.get(capture_uuid))

        with Image.open(path) as thumb:
            self.assertEqual(thumb.mode, "RGB")


class TestSyncState(StoreTestCase):

    def test_new_captures_are_pending(self):
        capture_uuid = self.store.record(make_image(), [])

        self.assertEqual(self.store.get(capture_uuid)["sync_state"], SYNC_PENDING)
        self.assertIsNone(self.store.get(capture_uuid)["synced_at"])

    def test_pending_returns_oldest_first(self):
        first = self.store.record(make_image(), [], captured_at="2026-08-01T00:00:00+00:00")
        second = self.store.record(make_image(), [], captured_at="2026-08-02T00:00:00+00:00")

        uuids = [capture["uuid"] for capture in self.store.pending()]

        self.assertEqual(uuids, [first, second])

    def test_captures_returns_newest_first(self):
        first = self.store.record(make_image(), [], captured_at="2026-08-01T00:00:00+00:00")
        second = self.store.record(make_image(), [], captured_at="2026-08-02T00:00:00+00:00")

        uuids = [capture["uuid"] for capture in self.store.captures()]

        self.assertEqual(uuids, [second, first])

    def test_mark_synced_moves_rows_out_of_pending(self):
        capture_uuid = self.store.record(make_image(), [])

        self.assertEqual(self.store.mark_synced([capture_uuid]), 1)

        capture = self.store.get(capture_uuid)

        self.assertEqual(capture["sync_state"], SYNC_SYNCED)
        self.assertIsNotNone(capture["synced_at"])
        self.assertEqual(self.store.pending(), [])
        self.assertEqual(self.store.stats()["pending"], 0)

    def test_marking_twice_changes_nothing_the_second_time(self):
        capture_uuid = self.store.record(make_image(), [])

        self.store.mark_synced([capture_uuid])

        # a server that acknowledges a batch twice must not rewrite the
        # timestamp, or a retry would look like a fresh upload
        self.assertEqual(self.store.mark_synced([capture_uuid]), 0)

    def test_marking_nothing_is_not_an_error(self):
        self.assertEqual(self.store.mark_synced([]), 0)

    def test_pending_respects_its_limit(self):
        for _ in range(5):
            self.store.record(make_image(320, 240), [])

        self.assertEqual(len(self.store.pending(limit=2)), 2)

    def test_filtering_captures_by_state(self):
        synced = self.store.record(make_image(), [])
        pending = self.store.record(make_image(), [])

        self.store.mark_synced([synced])

        rows = self.store.captures(sync_state=SYNC_PENDING)

        self.assertEqual([capture["uuid"] for capture in rows], [pending])


class TestExifLocation(unittest.TestCase):

    def test_a_photo_without_gps_reports_no_fix(self):
        self.assertEqual(read_exif_location(make_image()), (None, None))

    def test_explicit_location_beats_exif(self):
        root = Path(tempfile.mkdtemp(prefix="zra-store-"))
        store = CaptureStore(root=root)

        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.addCleanup(store.close)

        capture_uuid = store.record(make_image(), [], location=(1.5, -2.5))

        capture = store.get(capture_uuid)

        self.assertAlmostEqual(capture["latitude"], 1.5)
        self.assertAlmostEqual(capture["longitude"], -2.5)


if __name__ == "__main__":
    unittest.main()
