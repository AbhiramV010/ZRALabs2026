"""The classifier itself: a ResNet18 with a new head."""

import torch
import torch.nn as nn
from torchvision import models

from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent / "railway_classifier.pt"


def build_model(num_classes, freeze_backbone=True, pretrained=True):
    """ResNet18 with its 1000-class head swapped for ours."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # a fresh head, its parameters default to requires_grad=True
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


def unfreeze_last_block(model):
    """Open up layer4 and the head for fine tuning."""
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def save_checkpoint(model, classes, path=CHECKPOINT_PATH, metrics=None):
    """Store weights and class names together."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "architecture": "resnet18",
            "metrics": metrics or {},
        },
        path
    )

    return path


def load_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    """Rebuild a trained model from disk, ready for inference."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run 'python model/train.py' first."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    classes = checkpoint["classes"]

    # the saved weights are about to overwrite the pretrained ones
    model = build_model(
        len(classes),
        freeze_backbone=False,
        pretrained=False
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, classes


def load_metrics(path=CHECKPOINT_PATH):
    """The scores stored alongside the weights, without building a model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    return checkpoint.get("metrics", {})


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
