"""Integration tests against a real server on a real socket.

FastAPI's own TestClient wants httpx, which is not a dependency here, so
this starts uvicorn on a spare port in a background thread and talks to
it over HTTP like anything else would. Slower than a mocked transport,
and it exercises the parts that only exist once there is a socket -
gzipped request bodies, multipart uploads, and the sync round trip.
"""

import io
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests
import uvicorn
from PIL import Image

import api
from store import CaptureStore
from sync import Syncer


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))

        return probe.getsockname()[1]


def image_bytes(width=640, height=480, colour=(70, 100, 130)):
    buffer = io.BytesIO()

    Image.new("RGB", (width, height), colour).save(buffer, "JPEG")

    return buffer.getvalue()


def noisy_image(width=800, height=600, seed=0):
    """A stand-in for a photograph, for the payload-size tests.

    A flat colour compresses away to nothing and would make a thumbnail
    look no bigger than its own metadata, which is the opposite of the
    thing the bandwidth tiers exist to exploit.
    """
    pixels = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )

    return Image.fromarray(pixels)


class ServerTestCase(unittest.TestCase):
    """One server and one temporary store, shared by every test below."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="zra-api-"))

        # the server's own store, rather than the developer's real one
        api._store = CaptureStore(root=cls.root)

        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"

        server = uvicorn.Server(
            uvicorn.Config(
                api.app,
                host="127.0.0.1",
                port=cls.port,
                log_level="error"
            )
        )

        # uvicorn installs signal handlers, which only the main thread
        # may do. This server is stopped through should_exit instead
        server.install_signal_handlers = lambda: None

        cls.server = server
        cls.thread = threading.Thread(target=server.run, daemon=True)
        cls.thread.start()

        for _ in range(200):
            try:
                requests.get(f"{cls.base}/v1/health", timeout=1)

                break

            except requests.RequestException:
                time.sleep(0.1)

        else:
            raise RuntimeError("the test server never came up")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=15)

        api._store.close()
        api._store = None

        shutil.rmtree(cls.root, ignore_errors=True)


class TestHealthAndModel(ServerTestCase):

    def test_health_answers(self):
        payload = requests.get(f"{self.base}/v1/health", timeout=10).json()

        self.assertEqual(payload["status"], "ok")
        self.assertIn("store", payload)
        self.assertGreaterEqual(payload["uptime_seconds"], 0)

    def test_the_model_endpoint_names_the_classes(self):
        payload = requests.get(f"{self.base}/v1/model", timeout=120).json()

        self.assertIn("train", payload["classes"])
        self.assertEqual(len(payload["classes"]), 6)
        self.assertEqual(payload["runtime"], "torch")

    def test_the_model_endpoint_reports_its_architecture(self):
        payload = requests.get(f"{self.base}/v1/model", timeout=120).json()

        # a device needs to know what it is talking to before it trusts it
        self.assertEqual(payload["architecture"], "resnet18")
        self.assertIn("test_accuracy", payload["metrics"])


class TestClassify(ServerTestCase):

    def post_images(self, files, **params):
        return requests.post(
            f"{self.base}/v1/classify",
            files=files,
            params=params,
            timeout=180
        )

    def test_one_image_comes_back_scanned(self):
        response = self.post_images(
            [("files", ("a.jpg", image_bytes(), "image/jpeg"))],
            profile="edge"
        )

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["filename"], "a.jpg")
        self.assertEqual(payload["results"][0]["width"], 640)
        self.assertIsInstance(payload["results"][0]["detections"], list)

    def test_several_images_in_one_request(self):
        # the thing the 08/11 meeting asked for and the interface could not do
        response = self.post_images(
            [
                ("files", ("a.jpg", image_bytes(), "image/jpeg")),
                ("files", ("b.jpg", image_bytes(320, 240), "image/jpeg")),
                ("files", ("c.jpg", image_bytes(800, 600), "image/jpeg")),
            ],
            profile="edge"
        )

        payload = response.json()

        self.assertEqual(payload["count"], 3)
        self.assertEqual(
            [entry["filename"] for entry in payload["results"]],
            ["a.jpg", "b.jpg", "c.jpg"]
        )

    def test_a_file_that_is_not_an_image_is_reported_not_fatal(self):
        response = self.post_images(
            [
                ("files", ("broken.jpg", b"not a jpeg at all", "image/jpeg")),
                ("files", ("fine.jpg", image_bytes(), "image/jpeg")),
            ],
            profile="edge"
        )

        payload = response.json()

        self.assertEqual(payload["results"][0]["error"], "not a readable image")

        # the good one in the same batch still went through
        self.assertNotIn("error", payload["results"][1])

    def test_persist_writes_a_capture(self):
        before = requests.get(f"{self.base}/v1/health", timeout=10).json()

        response = self.post_images(
            [("files", ("kept.jpg", image_bytes(), "image/jpeg"))],
            profile="edge",
            persist="true"
        )

        self.assertIn("capture_uuid", response.json()["results"][0])

        after = requests.get(f"{self.base}/v1/health", timeout=10).json()

        self.assertEqual(
            after["store"]["captures"],
            before["store"]["captures"] + 1
        )

    def test_detections_are_shaped_for_from_dict(self):
        response = self.post_images(
            [("files", ("a.jpg", image_bytes(), "image/jpeg"))],
            profile="full"
        )

        for hit in response.json()["results"][0]["detections"]:
            self.assertIn("label", hit)
            self.assertIn("confidence", hit)
            self.assertEqual(len(hit["rect_1"]), 2)
            self.assertEqual(len(hit["rect_2"]), 2)


class TestSyncEndpoint(ServerTestCase):

    def post_sync(self, payload):
        return requests.post(f"{self.base}/v1/sync", json=payload, timeout=30)

    def test_a_batch_is_stored_and_acknowledged(self):
        response = self.post_sync({
            "device_id": "test-device",
            "captures": [{
                "uuid": "capture-one",
                "captured_at": "2026-08-14T10:00:00+00:00",
                "device_id": "test-device",
                "width": 1280,
                "height": 960,
                "detections": [{
                    "label": "Signal",
                    "description": "",
                    "confidence": 0.77,
                    "rect_1": [1, 2],
                    "rect_2": [3, 4],
                }],
            }],
        })

        payload = response.json()

        self.assertEqual(payload["stored"], 1)
        self.assertEqual(payload["accepted"], ["capture-one"])

    def test_resending_the_same_batch_does_not_double_it(self):
        batch = {
            "captures": [{
                "uuid": "capture-twice",
                "device_id": "test-device",
                "width": 640,
                "height": 480,
                "detections": [],
            }]
        }

        self.post_sync(batch)

        second = self.post_sync(batch).json()

        # a device that lost the first acknowledgement will send again,
        # and still has to be told it can clear the row
        self.assertEqual(second["stored"], 0)
        self.assertEqual(second["duplicates"], 1)
        self.assertEqual(second["accepted"], ["capture-twice"])

    def test_a_body_that_is_not_a_batch_is_refused(self):
        self.assertEqual(self.post_sync({"nope": []}).status_code, 400)

    def test_rows_without_a_uuid_are_skipped(self):
        payload = self.post_sync({
            "captures": [{"device_id": "test-device"}, "not even a dict"]
        }).json()

        self.assertEqual(payload["stored"], 0)
        self.assertEqual(payload["accepted"], [])


class TestRoundTrip(ServerTestCase):
    """A device with an outbox, pointed at the server, both tiers."""

    def setUp(self):
        self.device_root = Path(tempfile.mkdtemp(prefix="zra-device-"))
        self.device = CaptureStore(root=self.device_root)

    def tearDown(self):
        self.device.close()
        shutil.rmtree(self.device_root, ignore_errors=True)

    def record(self, count=3):
        for index in range(count):
            self.device.record(
                noisy_image(seed=index),
                [],
                source_name=f"field_{index}.jpg"
            )

    def test_a_full_outbox_drains(self):
        self.record(5)

        self.assertEqual(self.device.stats()["pending"], 5)

        report = Syncer(url=self.base, store=self.device, tier="medium").run()

        self.assertEqual(report["captures"], 5)
        self.assertEqual(report["pending"], 0)
        self.assertGreater(report["bytes"], 0)

    def test_the_thumbnail_arrives_with_the_medium_tier(self):
        self.record(1)

        uuid = self.device.pending()[0]["uuid"]

        Syncer(url=self.base, store=self.device, tier="medium").run()

        landed = api._store.get(uuid)

        self.assertIsNotNone(landed)
        self.assertGreater(landed["thumb_bytes"], 0)
        self.assertTrue(api._store.thumbnail_path(landed).is_file())

    def test_the_low_tier_arrives_without_one(self):
        self.record(1)

        uuid = self.device.pending()[0]["uuid"]

        Syncer(url=self.base, store=self.device, tier="low").run()

        landed = api._store.get(uuid)

        self.assertIsNotNone(landed)
        self.assertEqual(landed["thumb_bytes"], 0)
        self.assertIsNone(api._store.thumbnail_path(landed))

    def test_metadata_survives_the_trip(self):
        self.device.record(
            Image.new("RGB", (1280, 960)),
            [],
            source_name="named.jpg",
            location=(51.5, -0.12)
        )

        uuid = self.device.pending()[-1]["uuid"]

        Syncer(url=self.base, store=self.device, tier="low").run()

        landed = api._store.get(uuid)

        self.assertEqual(landed["source_name"], "named.jpg")
        self.assertEqual(landed["width"], 1280)
        self.assertAlmostEqual(landed["latitude"], 51.5, places=4)
        self.assertEqual(landed["device_id"], self.device.device_id)

    def test_the_low_tier_puts_far_less_on_the_wire(self):
        self.record(3)

        thin = Syncer(url=self.base, store=self.device, tier="low").run()

        self.record(3)

        fat = Syncer(url=self.base, store=self.device, tier="medium").run()

        self.assertEqual(thin["captures"], fat["captures"])
        self.assertLess(thin["bytes"] * 5, fat["bytes"])

    def test_a_drained_outbox_sends_nothing_more(self):
        self.record(2)

        syncer = Syncer(url=self.base, store=self.device, tier="medium")

        syncer.run()

        self.assertEqual(syncer.run()["captures"], 0)


if __name__ == "__main__":
    unittest.main()
