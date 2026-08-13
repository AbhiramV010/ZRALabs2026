"""Classify an image with the trained model.

    python model/predict.py path/to/photo.jpg --top 3
    python model/predict.py path/to/photo.jpg --detect --profile edge

Finding objects means classifying a grid of overlapping crops, since the
network itself only ever answers what one whole picture is. How dense
that grid is decides almost everything about what this costs to run, so
it is a named profile rather than a constant - see SCAN_PROFILES.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from backends import load_backend
    from dataset import IMAGE_SIZE, MEAN, STD, get_eval_transform
    from network import CHECKPOINT_PATH
else:
    from .backends import load_backend
    from .dataset import IMAGE_SIZE, MEAN, STD, get_eval_transform
    from .network import CHECKPOINT_PATH

from torchvision import transforms


class ScanProfile:
    """How thoroughly to sweep a frame, and what to keep from the sweep.

    The window grid is the whole cost of a scan. A full sweep of a phone
    photograph is 60-70 crops, each one a forward pass, which is fine on
    a workstation and hopeless on a board mounted to a drone. The `edge`
    profile trades recall for roughly an eighth of the work.
    """

    def __init__(self, name, scales, stride, confidence=0.7, limit=8,
                 min_pixels=64, merge_iou=0.25, merge_containment=0.6,
                 batch_size=32):
        self.name = name
        self.scales = tuple(scales)
        self.stride = stride
        self.confidence = confidence
        self.limit = limit
        self.min_pixels = min_pixels
        self.merge_iou = merge_iou
        self.merge_containment = merge_containment
        self.batch_size = batch_size

    def __repr__(self):
        return f"<ScanProfile {self.name}, {len(self.scales)} scales>"


SCAN_PROFILES = {
    # everything, for a machine that can afford it
    "full": ScanProfile(
        "full",
        scales=(0.75, 0.5, 0.33),
        stride=0.5,
    ),
    # two scales and a longer step. Roughly a third of the windows
    "balanced": ScanProfile(
        "balanced",
        scales=(0.6, 0.4),
        stride=0.65,
        limit=6,
        batch_size=16,
    ),
    # one scale, coarse step, small batches. Single-board computers
    "edge": ScanProfile(
        "edge",
        scales=(0.6,),
        stride=0.75,
        limit=4,
        batch_size=8,
    ),
}

DEFAULT_PROFILE = os.environ.get("ZRA_SCAN_PROFILE", "full")


def get_profile(profile=None):
    """A ScanProfile from a name, an object, or the environment."""
    if isinstance(profile, ScanProfile):
        return profile

    name = profile or DEFAULT_PROFILE

    if name not in SCAN_PROFILES:
        raise ValueError(
            f"Unknown scan profile {name!r}. "
            f"Pick one of: {', '.join(SCAN_PROFILES)}"
        )

    return SCAN_PROFILES[name]


# the old module-level names, kept pointing at the default profile so
# anything importing them still reads the values it used to
WINDOW_SCALES = SCAN_PROFILES["full"].scales
WINDOW_STRIDE = SCAN_PROFILES["full"].stride
MIN_WINDOW_PIXELS = SCAN_PROFILES["full"].min_pixels
WINDOW_CONFIDENCE = SCAN_PROFILES["full"].confidence
MERGE_IOU = SCAN_PROFILES["full"].merge_iou
MERGE_CONTAINMENT = SCAN_PROFILES["full"].merge_containment
BATCH_SIZE = SCAN_PROFILES["full"].batch_size


def prettify(label):
    return label.replace("_", " ").title()


def softmax(scores):
    """Row-wise softmax over a (N, classes) array."""
    shifted = scores - scores.max(axis=1, keepdims=True)

    exponentiated = np.exp(shifted)

    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def window_boxes(width, height, profile=None):
    """Overlapping square crops covering the frame at a few sizes."""
    profile = get_profile(profile)

    short_edge = min(width, height)

    boxes = []

    for scale in profile.scales:
        size = round(short_edge * scale)

        if size < profile.min_pixels:
            continue

        step = max(1, round(size * profile.stride))

        # pin the last window to the far edge, so no strip is missed
        lefts = sorted({*range(0, max(width - size, 0) + 1, step), width - size})
        tops = sorted({*range(0, max(height - size, 0) + 1, step), height - size})

        for top in tops:
            for left in lefts:
                boxes.append((
                    max(left, 0),
                    max(top, 0),
                    min(left + size, width),
                    min(top + size, height)
                ))

    return list(dict.fromkeys(boxes))


def count_windows(width, height, profile=None):
    """How many forward passes a scan of this frame would take."""
    return len(window_boxes(width, height, profile))


def overlap_area(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    if right <= left or bottom <= top:
        return 0

    return (right - left) * (bottom - top)


def encloses(outer, inner):
    """Whether one box sits wholly inside another."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def drop_containers(hits):
    """Keep the boxes drawn inside, drop the same-class box around them."""
    kept = []

    for label, confidence, box in hits:
        area = (box[2] - box[0]) * (box[3] - box[1])

        # a box wrapping same-class boxes is the wider, vaguer read of them.
        # strictly bigger, so a pair of identical boxes can't cancel out
        wraps_another = any(
            other_label == label
            and encloses(box, other_box)
            and (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]) < area
            for other_label, _, other_box in hits
        )

        if not wraps_another:
            kept.append((label, confidence, box))

    return kept


def same_object(a, b, profile=None):
    """Whether two windows are looking at one thing."""
    profile = get_profile(profile)

    overlap = overlap_area(a, b)

    if not overlap:
        return False

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou = overlap / (area_a + area_b - overlap)

    # a small window inside a big one is the same object, close up
    containment = overlap / min(area_a, area_b)

    return iou > profile.merge_iou or containment > profile.merge_containment


def merge_hits(hits, profile=None):
    """Fold overlapping same-class windows into one box per object."""
    merged = []

    for label, confidence, box in sorted(hits, key=lambda hit: -hit[1]):

        for index, (kept_label, kept_confidence, kept_box) in enumerate(merged):

            # widen the survivor rather than dropping the weaker box
            if kept_label == label and same_object(box, kept_box, profile):
                merged[index] = (
                    kept_label,
                    kept_confidence,
                    (
                        min(kept_box[0], box[0]),
                        min(kept_box[1], box[1]),
                        max(kept_box[2], box[2]),
                        max(kept_box[3], box[3])
                    )
                )
                break

        else:
            merged.append((label, confidence, box))

    return merged


class RailwayClassifier:
    """Holds the model in memory so repeated calls stay cheap.

    `model` may be a training checkpoint or any exported file that
    `backends.load_backend` recognises, so the same scanning code runs on
    a workstation and on whatever the drone is carrying.
    """

    def __init__(self, model=CHECKPOINT_PATH, device=None, profile=None,
                 runtime=None):
        self.profile = get_profile(profile)

        options = {"runtime": runtime}

        # only the torch backend knows what a device is
        if runtime in (None, "torch") and Path(model).suffix in (".pt", ".pth"):
            options["device"] = device

        self.backend = load_backend(model, **options)

        self.classes = self.backend.classes
        self.transform = get_eval_transform()

        # no center crop, a window is already the crop we want
        self.window_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @property
    def device(self):
        return getattr(self.backend, "device", "cpu")

    def probabilities(self, crops, transform, batch_size=None):
        """Run a list of PIL images through the model in batches."""
        batch_size = batch_size or self.profile.batch_size

        scores = []

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]

            tensor = torch.stack([transform(crop) for crop in batch])

            logits = self.backend.run(tensor.numpy())

            scores.extend(softmax(np.asarray(logits)).tolist())

        return scores

    def classify(self, image, top_k=None):
        """Return [(label, confidence), ...] sorted best first."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        probabilities = self.probabilities([image], self.transform)[0]

        ranked = sorted(
            zip(self.classes, probabilities),
            key=lambda pair: pair[1],
            reverse=True
        )

        results = [(prettify(name), score) for name, score in ranked]

        return results[:top_k] if top_k else results

    def detect(self, image, confidence=None, limit=-1, profile=None):
        """The assets in one frame: [(label, confidence, box), ...].

        Boxes are window shaped, not object outlines. `limit` caps how
        many come back - pass None for no cap, or leave it alone to use
        the profile's own.
        """
        profile = get_profile(profile) if profile is not None else self.profile

        confidence = profile.confidence if confidence is None else confidence

        # -1 means 'not specified', since None is a meaningful value here
        limit = profile.limit if limit == -1 else limit

        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes = window_boxes(image.width, image.height, profile)

        crops = [image.crop(box) for box in boxes]

        scores = self.probabilities(crops, self.window_transform, profile.batch_size)

        hits = []

        for box, probabilities in zip(boxes, scores):

            best = max(range(len(probabilities)), key=lambda i: probabilities[i])

            if probabilities[best] >= confidence:
                hits.append((prettify(self.classes[best]), probabilities[best], box))

        merged = drop_containers(merge_hits(hits, profile))

        # the strongest objects, whatever mix of classes that turns out to be
        ranked = sorted(merged, key=lambda hit: -hit[1])

        return ranked[:limit] if limit else ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("image", type=Path, help="image file to classify")
    parser.add_argument("--model", type=Path, default=CHECKPOINT_PATH,
                        help="checkpoint or exported model to run")
    parser.add_argument("--runtime", default=None,
                        help="force a backend: torch, onnx, executorch, tflite")
    parser.add_argument("--profile", default=None,
                        choices=sorted(SCAN_PROFILES),
                        help="scan density for --detect")
    parser.add_argument("--detect", action="store_true",
                        help="find assets in the frame instead of ranking it")
    parser.add_argument("--top", type=int, default=None,
                        help="only show the top N classes")

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"No such image: {args.image}")

    classifier = RailwayClassifier(
        args.model,
        profile=args.profile,
        runtime=args.runtime
    )

    image = Image.open(args.image)

    print(f"\n{args.image.name}")

    if args.detect:
        windows = count_windows(image.width, image.height, classifier.profile)

        print(f"  {classifier.profile.name} profile, {windows} windows\n")

        for label, confidence, box in classifier.detect(image):
            print(f"  {label:<16}{confidence:>7.1%}  {box}")

        return

    for label, confidence in classifier.classify(image, args.top):
        bar = "#" * round(confidence * 30)
        print(f"  {label:<16}{confidence:>7.1%}  {bar}")


if __name__ == "__main__":
    main()
