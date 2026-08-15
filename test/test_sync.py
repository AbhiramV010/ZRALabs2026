"""Tests for the outbox client, with nothing on the other end.

The network half is covered by test_api.py, which runs a real server and
pushes a real batch at it. This file is the parts that have to be right
before a byte leaves: what goes on the wire, and what stays home.
"""

import shutil
import sys
import tempfile
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from result import DetectionResult
from store import CaptureStore
from sync import (
    BACKOFF_CAP,
    LOCAL_ONLY,
    TIERS,
    Syncer,
    Tier,
    backoff_delay,
    get_tier,
)


def make_image(width=800, height=600, seed=0):
    """Noise, not a flat colour.

    A solid-colour JPEG compresses to almost nothing, which would make
    the thumbnail look about the same size as the metadata and quietly
    void every payload-size test below. Noise is the other extreme; a
    real photograph sits between the two.
    """
    pixels = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )

    return Image.fromarray(pixels)


def make_detections():
    return [
        DetectionResult(
            label="Track",
            description="The steel rails",
            confidence=0.88,
            rect_1=(4, 8),
            rect_2=(200, 300)
        ),
    ]


class TestTiers(unittest.TestCase):

    def test_the_three_tiers_are_there(self):
        self.assertEqual(set(TIERS), {"low", "medium", "high"})

    def test_the_narrow_tier_sends_no_pictures(self):
        self.assertFalse(TIERS["low"].thumbnails)

    def test_the_wider_tiers_do(self):
        self.assertTrue(TIERS["medium"].thumbnails)
        self.assertTrue(TIERS["high"].thumbnails)

    def test_batch_size_grows_with_the_link(self):
        self.assertLess(TIERS["low"].batch, TIERS["medium"].batch)
        self.assertLess(TIERS["medium"].batch, TIERS["high"].batch)

    def test_a_name_resolves(self):
        self.assertIs(get_tier("low"), TIERS["low"])

    def test_a_tier_passes_through(self):
        tier = Tier("custom", batch=2, thumbnails=False)

        self.assertIs(get_tier(tier), tier)

    def test_an_unknown_tier_is_refused(self):
        with self.assertRaises(ValueError):
            get_tier("dialup")


class TestBackoff(unittest.TestCase):

    def test_delays_grow(self):
        # jittered, so compare the floors rather than single draws
        early = min(backoff_delay(0) for _ in range(50))
        later = min(backoff_delay(3) for _ in range(50))

        self.assertLess(early, later)

    def test_the_delay_is_capped(self):
        # an hour offline must not become an hour between retries
        self.assertLessEqual(max(backoff_delay(20) for _ in range(50)), BACKOFF_CAP * 1.5)

    def test_the_delay_is_never_zero(self):
        self.assertGreater(min(backoff_delay(0) for _ in range(50)), 0)


class SyncerTestCase(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="zra-sync-"))
        self.store = CaptureStore(root=self.root)

        self.uuid = self.store.record(
            make_image(),
            make_detections(),
            source_name="DSC_0002.jpg",
            model_name="railway_classifier",
            model_version="resnet18"
        )

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def syncer(self, tier):
        return Syncer(url="http://example.invalid", store=self.store, tier=tier)


class TestPayload(SyncerTestCase):

    def test_local_bookkeeping_stays_home(self):
        capture = self.syncer("medium").build_capture(self.store.get(self.uuid))

        for field in LOCAL_ONLY:
            self.assertNotIn(field, capture)

    def test_what_the_server_needs_is_sent(self):
        capture = self.syncer("low").build_capture(self.store.get(self.uuid))

        for field in ("uuid", "captured_at", "device_id", "width", "height"):
            self.assertIn(field, capture)

        self.assertEqual(capture["source_name"], "DSC_0002.jpg")
        self.assertEqual(len(capture["detections"]), 1)

    def test_the_narrow_tier_leaves_the_picture_behind(self):
        capture = self.syncer("low").build_capture(self.store.get(self.uuid))

        self.assertNotIn("thumbnail", capture)

    def test_the_wider_tier_carries_it(self):
        capture = self.syncer("medium").build_capture(self.store.get(self.uuid))

        self.assertIn("thumbnail", capture)
        self.assertGreater(len(capture["thumbnail"]), 0)

    def test_dropping_the_picture_is_most_of_the_payload(self):
        record = self.store.get(self.uuid)

        thin = self.syncer("low").build_capture(record)
        fat = self.syncer("medium").build_capture(record)

        # the reason the tiers exist at all
        self.assertLess(len(str(thin)) * 10, len(str(fat)))

    def test_the_batch_names_its_sender(self):
        payload = self.syncer("medium").build_payload(
            [self.store.get(self.uuid)]
        )

        self.assertEqual(payload["device_id"], self.store.device_id)
        self.assertEqual(payload["tier"], "medium")
        self.assertEqual(len(payload["captures"]), 1)
        self.assertIn("sent_at", payload)

    def test_a_record_with_no_thumbnail_still_builds(self):
        # what a metadata-only row looks like on the receiving end
        self.store.accept({"uuid": "no-picture-here", "device_id": "other"})

        capture = self.store.get("no-picture-here")

        self.assertNotIn(
            "thumbnail",
            self.syncer("medium").build_capture(capture)
        )


class TestOutbox(SyncerTestCase):

    def test_nothing_pending_sends_nothing(self):
        self.store.mark_synced([self.uuid])

        self.assertEqual(self.syncer("medium").run_once(), (0, 0))

    def test_a_batch_is_capped_by_the_tier(self):
        for _ in range(15):
            self.store.record(make_image(320, 240), [])

        # low sends ten at a time, and there are sixteen waiting
        self.assertEqual(
            len(self.store.pending(limit=TIERS["low"].batch)),
            10
        )

    def test_an_unreachable_server_raises_rather_than_clearing(self):
        syncer = Syncer(
            url="http://127.0.0.1:1",
            store=self.store,
            tier="low",
            attempts=1,
            timeout=1
        )

        with self.assertRaises(Exception):
            syncer.run_once()

        # the whole point: a failed upload leaves the row to try again
        self.assertEqual(self.store.stats()["pending"], 1)


if __name__ == "__main__":
    unittest.main()
