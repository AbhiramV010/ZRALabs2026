"""Tests for the architecture registry and the checkpoint format.

Everything here builds with pretrained=False - these check the wiring,
not the weights, and downloading six ImageNet checkpoints to do it would
make the suite useless offline.
"""

import os
import shutil
import sys
import tempfile
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.network import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    architecture_names,
    build_model,
    count_parameters,
    load_architecture,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    unfreeze_last_block,
)

CLASSES = [
    "crossing_gate", "overhead_wire", "platform", "signal", "track", "train"
]


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestRegistry(unittest.TestCase):

    def test_the_default_is_registered(self):
        self.assertIn(DEFAULT_ARCHITECTURE, ARCHITECTURES)

    def test_names_are_sorted(self):
        self.assertEqual(architecture_names(), sorted(ARCHITECTURES))

    def test_every_spec_says_what_to_fine_tune(self):
        for name, spec in ARCHITECTURES.items():
            with self.subTest(architecture=name):
                self.assertTrue(
                    "blocks" in spec or "tail" in spec,
                    f"{name} has no fine tuning target"
                )

    def test_an_unknown_architecture_is_refused(self):
        with self.assertRaises(ValueError):
            build_model(6, architecture="resnet50")


class TestBuilding(unittest.TestCase):

    def test_every_architecture_outputs_one_score_per_class(self):
        batch = torch.randn(2, 3, 224, 224)

        for name in architecture_names():
            with self.subTest(architecture=name):
                model = build_model(6, architecture=name, pretrained=False)

                model.eval()

                with torch.no_grad():
                    self.assertEqual(model(batch).shape, (2, 6))

    def test_a_different_class_count_is_honoured(self):
        model = build_model(3, architecture="mobilenet_v3_small", pretrained=False)

        model.eval()

        with torch.no_grad():
            self.assertEqual(model(torch.randn(1, 3, 224, 224)).shape, (1, 3))

    def test_freezing_leaves_only_the_head_trainable(self):
        for name in architecture_names():
            with self.subTest(architecture=name):
                model = build_model(
                    6, architecture=name, freeze_backbone=True, pretrained=False
                )

                self.assertGreater(trainable(model), 0)
                self.assertLess(trainable(model), count_parameters(model))

    def test_unfreezing_opens_up_more_than_the_head(self):
        for name in architecture_names():
            with self.subTest(architecture=name):
                model = build_model(
                    6, architecture=name, freeze_backbone=True, pretrained=False
                )

                head_only = trainable(model)

                unfreeze_last_block(model, name)

                self.assertGreater(trainable(model), head_only)

    def test_nothing_is_frozen_when_not_asked(self):
        model = build_model(6, freeze_backbone=False, pretrained=False)

        self.assertEqual(trainable(model), count_parameters(model))


class TestCheckpoints(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zra-ckpt-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_checkpoint_reloads_as_the_architecture_it_was_saved_as(self):
        # the whole point of the registry: before this, every checkpoint
        # was rebuilt as a resnet18 whatever it had been trained as
        for name in architecture_names():
            with self.subTest(architecture=name):
                path = self.tmp / f"{name}.pt"

                model = build_model(6, architecture=name, pretrained=False)

                save_checkpoint(model, CLASSES, path, architecture=name)

                self.assertEqual(load_architecture(path), name)

                rebuilt, classes = load_checkpoint(path)

                self.assertEqual(classes, CLASSES)

                with torch.no_grad():
                    self.assertEqual(
                        rebuilt(torch.randn(1, 3, 224, 224)).shape, (1, 6)
                    )

    def test_weights_survive_the_round_trip(self):
        path = self.tmp / "round-trip.pt"

        model = build_model(6, architecture="squeezenet1_1", pretrained=False)
        model.eval()

        save_checkpoint(model, CLASSES, path, architecture="squeezenet1_1")

        rebuilt, _ = load_checkpoint(path)

        batch = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            self.assertTrue(torch.allclose(model(batch), rebuilt(batch), atol=1e-5))

    def test_a_checkpoint_with_no_architecture_is_read_as_the_default(self):
        path = self.tmp / "legacy.pt"

        model = build_model(6, pretrained=False)

        # what train.py wrote before the architecture was recorded
        torch.save(
            {"state_dict": model.state_dict(), "classes": CLASSES, "metrics": {}},
            path
        )

        self.assertEqual(load_architecture(path), DEFAULT_ARCHITECTURE)

        rebuilt, classes = load_checkpoint(path)

        self.assertEqual(classes, CLASSES)

    def test_a_missing_checkpoint_says_how_to_make_one(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_checkpoint(self.tmp / "nothing-here.pt")

        self.assertIn("train.py", str(caught.exception))


class TestDevice(unittest.TestCase):

    def test_the_environment_overrides_the_search(self):
        previous = os.environ.get("ZRA_DEVICE")

        os.environ["ZRA_DEVICE"] = "cpu"

        try:
            self.assertEqual(pick_device().type, "cpu")

        finally:
            if previous is None:
                del os.environ["ZRA_DEVICE"]
            else:
                os.environ["ZRA_DEVICE"] = previous

    def test_it_always_returns_something(self):
        self.assertIsInstance(pick_device(), torch.device)


if __name__ == "__main__":
    unittest.main()
