"""Tests for the runtime abstraction.

Only the torch backend is always exercised - it is the one whose package
the project already depends on. The rest are skipped unless their
runtime happens to be installed, which is the same rule the module
itself follows.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from model.backends import (
    RUNTIMES,
    TorchBackend,
    available_runtimes,
    classes_path,
    load_backend,
    read_classes,
    write_classes,
)
from model.network import CHECKPOINT_PATH

CLASSES = [
    "crossing_gate", "overhead_wire", "platform", "signal", "track", "train"
]


def installed(name):
    return importlib.util.find_spec(name) is not None


class TestDispatch(unittest.TestCase):

    def test_every_known_extension_names_a_runtime(self):
        self.assertEqual(
            set(RUNTIMES.values()),
            {"torch", "onnx", "executorch", "tflite"}
        )

    def test_an_unknown_extension_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            load_backend("model.bin")

        self.assertIn(".onnx", str(caught.exception))

    def test_an_unknown_runtime_is_refused(self):
        with self.assertRaises(ValueError):
            load_backend("model.onnx", runtime="tensorrt")

    def test_available_runtimes_covers_all_four(self):
        found = available_runtimes()

        self.assertEqual(set(found), {"torch", "onnx", "executorch", "tflite"})
        self.assertTrue(found["torch"])


class TestClassesSidecar(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zra-backend-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_sidecar_sits_beside_the_model(self):
        path = classes_path(self.tmp / "railway.onnx")

        self.assertEqual(path.parent, self.tmp)
        self.assertEqual(path.name, "railway.classes.json")

    def test_labels_round_trip(self):
        path = self.tmp / "railway.tflite"

        write_classes(path, CLASSES)

        self.assertEqual(read_classes(path), CLASSES)

    def test_a_model_with_no_sidecar_reports_none(self):
        self.assertIsNone(read_classes(self.tmp / "railway.onnx"))

    def test_the_sidecar_is_readable_json(self):
        path = self.tmp / "railway.pte"

        written = write_classes(path, CLASSES)

        self.assertEqual(json.loads(written.read_text()), CLASSES)


@unittest.skipUnless(CHECKPOINT_PATH.is_file(), "no trained checkpoint")
class TestTorchBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = load_backend(CHECKPOINT_PATH, device="cpu")

    def test_it_reads_the_classes_out_of_the_checkpoint(self):
        self.assertEqual(self.backend.classes, CLASSES)

    def test_it_is_chosen_for_a_pt_file(self):
        self.assertIsInstance(self.backend, TorchBackend)
        self.assertEqual(self.backend.runtime, "torch")

    def test_a_batch_of_images_becomes_one_row_of_logits_each(self):
        logits = self.backend.run(np.zeros((3, 3, 224, 224), dtype=np.float32))

        self.assertEqual(logits.shape, (3, len(CLASSES)))
        self.assertTrue(np.isfinite(logits).all())

    def test_it_returns_numpy_not_tensors(self):
        logits = self.backend.run(np.zeros((1, 3, 224, 224), dtype=np.float32))

        self.assertIsInstance(logits, np.ndarray)

    def test_describe_reports_something_useful(self):
        described = self.backend.describe()

        self.assertEqual(described["runtime"], "torch")
        self.assertGreater(described["bytes"], 0)


@unittest.skipUnless(
    installed("tensorflow") and installed("ai_edge_litert"),
    "needs tensorflow to build a .tflite and LiteRT to run it"
)
class TestTFLiteBackend(unittest.TestCase):
    """A tiny model, converted both ways, run through the backend.

    The two things worth checking are the ones a converted model can get
    wrong silently: the layout swap, since a LiteRT model wants NHWC and
    everything upstream of it here is NCHW, and the integer scaling on a
    fully quantised model.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        import tensorflow as tf

        cls.tmp = Path(tempfile.mkdtemp(prefix="zra-tflite-"))

        model = tf.keras.Sequential([
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Conv2D(8, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES)),
        ])

        cls.float_path = cls.tmp / "float.tflite"
        cls.float_path.write_bytes(
            tf.lite.TFLiteConverter.from_keras_model(model).convert()
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            [np.random.rand(1, 32, 32, 3).astype(np.float32)] for _ in range(10)
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        cls.int8_path = cls.tmp / "int8.tflite"
        cls.int8_path.write_bytes(converter.convert())

        for path in (cls.float_path, cls.int8_path):
            write_classes(path, CLASSES)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_tflite_file_picks_the_tflite_backend(self):
        self.assertEqual(load_backend(self.float_path).runtime, "tflite")

    def test_nchw_input_is_transposed_for_an_nhwc_model(self):
        backend = load_backend(self.float_path)

        self.assertTrue(backend.channels_last)

        logits = backend.run(np.random.rand(4, 3, 32, 32).astype(np.float32))

        self.assertEqual(logits.shape, (4, len(CLASSES)))

    def test_a_quantised_model_takes_floats_and_returns_floats(self):
        backend = load_backend(self.int8_path)

        self.assertNotEqual(backend.input_detail["dtype"], np.float32)

        logits = backend.run(np.random.rand(2, 3, 32, 32).astype(np.float32))

        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(logits.shape, (2, len(CLASSES)))
        self.assertTrue(np.isfinite(logits).all())

    def test_quantising_shrinks_the_file(self):
        self.assertLess(
            self.int8_path.stat().st_size,
            self.float_path.stat().st_size
        )

    def test_labels_come_from_the_sidecar(self):
        self.assertEqual(load_backend(self.float_path).classes, CLASSES)


if __name__ == "__main__":
    unittest.main()
