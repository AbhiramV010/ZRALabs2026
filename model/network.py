"""The classifier itself: a torchvision backbone with a new head.

Six backbones are on offer rather than one, because the target hardware
runs from a phone down to a single-board computer and no single network
suits both. `resnet18` stays the default so existing checkpoints and
scores mean what they did before.

    from network import build_model, ARCHITECTURES

    model = build_model(6, architecture="mobilenet_v3_small")

The architecture is written into the checkpoint and read back out of it,
so a saved file knows what to rebuild itself as.
"""

import os

import torch
import torch.nn as nn
from torchvision import models

from pathlib import Path

CHECKPOINT_PATH = Path(__file__).resolve().parent / "railway_classifier.pt"

DEFAULT_ARCHITECTURE = "resnet18"

DROPOUT = 0.3

# What each backbone calls its classifier, and which part of it is worth
# fine tuning. `blocks` names modules directly; `tail` means the last N
# children of `model.features`, which is how the mobile-style nets are
# built. Sizes are the ImageNet checkpoints, as a rough guide to cost.
ARCHITECTURES = {
    # 11.7M parameters. The baseline, and the heaviest here
    "resnet18": {
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights,
        "head": "fc",
        "blocks": ("layer4",),
    },
    # 5.5M. A middle option for a Jetson or a recent phone
    "mobilenet_v3_large": {
        "builder": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights,
        "head": "classifier",
        "tail": 2,
    },
    # 2.5M. The one to try first on a Pi
    "mobilenet_v3_small": {
        "builder": models.mobilenet_v3_small,
        "weights": models.MobileNet_V3_Small_Weights,
        "head": "classifier",
        "tail": 2,
    },
    # 5.3M, but the best accuracy per FLOP of this set. Note torchvision
    # ships b0, not the lite0 variant - lite0 drops the squeeze-excite
    # blocks and hard-swish that some NPU delegates refuse, and it lives
    # in timm rather than here
    "efficientnet_b0": {
        "builder": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights,
        "head": "classifier",
        "tail": 2,
    },
    # 1.4M. Cheap enough to be worth measuring against the small mobilenet
    "shufflenet_v2_x0_5": {
        "builder": models.shufflenet_v2_x0_5,
        "weights": models.ShuffleNet_V2_X0_5_Weights,
        "head": "fc",
        "blocks": ("stage4", "conv5"),
    },
    # 1.2M, and the head is a 1x1 convolution rather than a linear layer
    "squeezenet1_1": {
        "builder": models.squeezenet1_1,
        "weights": models.SqueezeNet1_1_Weights,
        "head": "classifier",
        "tail": 3,
    },
}


def architecture_names():
    return sorted(ARCHITECTURES)


def get_spec(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Pick one of: {', '.join(architecture_names())}"
        )

    return ARCHITECTURES[architecture]


def swap_head(head, num_classes, dropout=DROPOUT):
    """Give a torchvision classifier the right number of outputs.

    Three shapes turn up. A bare `Linear` (resnet, shufflenet) is wrapped
    in a dropout of our own. A `Sequential` already carries its own
    dropout, so only its final layer is replaced - and in squeezenet that
    final layer is a 1x1 convolution, not a linear one.
    """
    if isinstance(head, nn.Linear):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head.in_features, num_classes)
        )

    if isinstance(head, nn.Sequential):

        for index in reversed(range(len(head))):
            layer = head[index]

            if isinstance(layer, nn.Linear):
                head[index] = nn.Linear(layer.in_features, num_classes)
                return head

            if isinstance(layer, nn.Conv2d):
                head[index] = nn.Conv2d(
                    layer.in_channels,
                    num_classes,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride
                )
                return head

    raise TypeError(f"Cannot replace a head of type {type(head).__name__}")


def build_model(num_classes, architecture=DEFAULT_ARCHITECTURE,
                freeze_backbone=True, pretrained=True):
    """A pretrained backbone with its 1000-class head swapped for ours."""
    spec = get_spec(architecture)

    weights = spec["weights"].DEFAULT if pretrained else None

    model = spec["builder"](weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    setattr(
        model,
        spec["head"],
        swap_head(getattr(model, spec["head"]), num_classes)
    )

    # freezing happened before the swap, and some heads keep a pretrained
    # layer of their own, so the whole head is opened up explicitly
    for parameter in getattr(model, spec["head"]).parameters():
        parameter.requires_grad = True

    # squeezenet reads this when it reshapes its output
    if hasattr(model, "num_classes"):
        model.num_classes = num_classes

    return model


def last_blocks(model, architecture):
    """The modules worth fine tuning, for this backbone."""
    spec = get_spec(architecture)

    if "blocks" in spec:
        return [model.get_submodule(name) for name in spec["blocks"]]

    children = list(model.features.children())

    return children[-spec["tail"]:]


def unfreeze_last_block(model, architecture=DEFAULT_ARCHITECTURE):
    """Open up the last block and the head for fine tuning."""
    spec = get_spec(architecture)

    for block in last_blocks(model, architecture):
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in getattr(model, spec["head"]).parameters():
        parameter.requires_grad = True

    return model


def save_checkpoint(model, classes, path=CHECKPOINT_PATH, metrics=None,
                    architecture=DEFAULT_ARCHITECTURE):
    """Store weights, class names and which backbone they belong to."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "architecture": architecture,
            "metrics": metrics or {},
        },
        path
    )

    return path


def read_checkpoint(path=CHECKPOINT_PATH):
    """The raw saved dict, with a readable error when it is not there."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run 'python model/train.py' first."
        )

    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    """Rebuild a trained model from disk, ready for inference."""
    checkpoint = read_checkpoint(path)

    classes = checkpoint["classes"]

    # the file says what it is. Older checkpoints predate the choice and
    # are all resnet18, which is what the default resolves to anyway
    architecture = checkpoint.get("architecture") or DEFAULT_ARCHITECTURE

    # the saved weights are about to overwrite the pretrained ones
    model = build_model(
        len(classes),
        architecture=architecture,
        freeze_backbone=False,
        pretrained=False
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, classes


def load_architecture(path=CHECKPOINT_PATH):
    """Which backbone a checkpoint holds, without building it."""
    return read_checkpoint(path).get("architecture") or DEFAULT_ARCHITECTURE


def load_metrics(path=CHECKPOINT_PATH):
    """The scores stored alongside the weights, without building a model."""
    return read_checkpoint(path).get("metrics", {})


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def pick_device():
    """The best accelerator present, or the CPU.

    ZRA_DEVICE overrides the search. That matters on the edge devices
    this is headed for, where an accelerator can be present but slower
    than the CPU for a model this small, or simply missing a kernel the
    backbone needs.
    """
    override = os.environ.get("ZRA_DEVICE")

    if override:
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple silicon, which covers the tablet end of the brief
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Intel integrated and discrete graphics, present on mini PCs
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")
