"""Classify an image with the trained model.

    python model/predict.py path/to/photo.jpg --top 3
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import IMAGE_SIZE, MEAN, STD, get_eval_transform
    from network import CHECKPOINT_PATH, load_checkpoint, pick_device
else:
    from .dataset import IMAGE_SIZE, MEAN, STD, get_eval_transform
    from .network import CHECKPOINT_PATH, load_checkpoint, pick_device

from torchvision import transforms

# sliding window settings for detect()
WINDOW_SCALES = (0.75, 0.5, 0.33)
WINDOW_STRIDE = 0.5
MIN_WINDOW_PIXELS = 64
WINDOW_CONFIDENCE = 0.7
MERGE_IOU = 0.25
MERGE_CONTAINMENT = 0.6
MAX_DETECTIONS = 3
BATCH_SIZE = 32


def prettify(label):
    return label.replace("_", " ").title()


def window_boxes(width, height):
    """Overlapping square crops covering the frame at a few sizes."""
    short_edge = min(width, height)

    boxes = []

    for scale in WINDOW_SCALES:
        size = round(short_edge * scale)

        if size < MIN_WINDOW_PIXELS:
            continue

        step = max(1, round(size * WINDOW_STRIDE))

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


def overlap_area(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    if right <= left or bottom <= top:
        return 0

    return (right - left) * (bottom - top)


def same_object(a, b):
    """Whether two windows are looking at one thing."""
    overlap = overlap_area(a, b)

    if not overlap:
        return False

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou = overlap / (area_a + area_b - overlap)

    # a small window inside a big one is the same object, close up
    containment = overlap / min(area_a, area_b)

    return iou > MERGE_IOU or containment > MERGE_CONTAINMENT


def merge_hits(hits):
    """Fold overlapping same-class windows into one box per object."""
    merged = []

    for label, confidence, box in sorted(hits, key=lambda hit: -hit[1]):

        for index, (kept_label, kept_confidence, kept_box) in enumerate(merged):

            # widen the survivor rather than dropping the weaker box
            if kept_label == label and same_object(box, kept_box):
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
    """Holds the model in memory so repeated calls stay cheap."""

    def __init__(self, checkpoint=CHECKPOINT_PATH, device=None):
        self.device = device or pick_device()
        self.model, self.classes = load_checkpoint(checkpoint, self.device)
        self.transform = get_eval_transform()

        # no center crop, a window is already the crop we want
        self.window_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def probabilities(self, crops, transform):
        """Run a list of PIL images through the model in batches."""
        scores = []

        for start in range(0, len(crops), BATCH_SIZE):
            batch = crops[start:start + BATCH_SIZE]

            tensor = torch.stack(
                [transform(crop) for crop in batch]
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)

            scores.extend(F.softmax(logits, dim=1).cpu().tolist())

        return scores

    def classify(self, image, top_k=None):
        """Return [(label, confidence), ...] sorted best first."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probabilities = F.softmax(logits[0], dim=0)

        ranked = sorted(
            zip(self.classes, probabilities.cpu().tolist()),
            key=lambda pair: pair[1],
            reverse=True
        )

        results = [(prettify(name), score) for name, score in ranked]

        return results[:top_k] if top_k else results

    def detect(self, image, confidence=WINDOW_CONFIDENCE, limit=MAX_DETECTIONS):
        """The best few assets in one frame: [(label, confidence, box), ...].

        Classifies a grid of overlapping crops, since the network itself
        only ever answers what one whole picture is. Boxes are window
        shaped, not object outlines.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes = window_boxes(image.width, image.height)

        crops = [image.crop(box) for box in boxes]

        scores = self.probabilities(crops, self.window_transform)

        hits = []

        for box, probabilities in zip(boxes, scores):

            best = max(range(len(probabilities)), key=lambda i: probabilities[i])

            if probabilities[best] >= confidence:
                hits.append((prettify(self.classes[best]), probabilities[best], box))

        merged = merge_hits(hits)

        # the strongest few objects, whatever mix of classes that turns out to be
        return sorted(merged, key=lambda hit: -hit[1])[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("image", type=Path, help="image file to classify")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--top", type=int, default=None,
                        help="only show the top N classes")

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"No such image: {args.image}")

    classifier = RailwayClassifier(args.checkpoint)

    results = classifier.classify(Image.open(args.image), args.top)

    print(f"\n{args.image.name}")

    for label, confidence in results:
        bar = "#" * round(confidence * 30)
        print(f"  {label:<16}{confidence:>7.1%}  {bar}")


if __name__ == "__main__":
    main()
