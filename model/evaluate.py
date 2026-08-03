"""Score a trained checkpoint on the held-out test split.

    python model/evaluate.py

Prints a per-class breakdown and a confusion matrix, not just accuracy.
"""

import argparse
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import build_loaders
    from network import CHECKPOINT_PATH, load_checkpoint, pick_device
else:
    from .dataset import build_loaders
    from .network import CHECKPOINT_PATH, load_checkpoint, pick_device


def collect_predictions(model, loader, device):
    """Run the loader through the model, return (true, predicted)."""
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))

            predicted_labels += outputs.argmax(dim=1).cpu().tolist()
            true_labels += labels.tolist()

    return true_labels, predicted_labels


def confusion_matrix(true_labels, predicted_labels, num_classes):
    """counts[actual][predicted], by hand to avoid a scikit dependency."""
    counts = [[0] * num_classes for _ in range(num_classes)]

    for actual, predicted in zip(true_labels, predicted_labels):
        counts[actual][predicted] += 1

    return counts


def print_report(classes, counts):
    """Per-class precision, recall and F1 from the matrix."""
    print(f"\n{'class':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}")
    print("-" * 51)

    for index, name in enumerate(classes):
        true_positive = counts[index][index]

        actual_total = sum(counts[index])
        predicted_total = sum(row[index] for row in counts)

        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"{name:<16}{precision:>10.2f}{recall:>9.2f}"
            f"{f1:>7.2f}{actual_total:>9}"
        )


def print_matrix(classes, counts):
    """Rows are the true class, columns what the model guessed."""
    width = max(len(name) for name in classes) + 2

    header = " " * width + "".join(f"{name[:7]:>9}" for name in classes)

    print("\nconfusion matrix (rows = actual, columns = predicted)")
    print(header)

    for index, name in enumerate(classes):
        row = "".join(f"{value:>9}" for value in counts[index])
        print(f"{name:<{width}}{row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42,
                        help="must match the seed used for training")
    parser.add_argument("--split", choices=["test", "val", "train"],
                        default="test")

    args = parser.parse_args()

    device = pick_device()

    model, classes = load_checkpoint(args.checkpoint, device)

    train_loader, val_loader, test_loader, split_classes = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed
    )

    # a mismatch means the images folder changed since training
    if split_classes != classes:
        print(
            f"Warning: checkpoint classes {classes} do not match the "
            f"images folder {split_classes}. Retrain before trusting this."
        )

    loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }[args.split]

    true_labels, predicted_labels = collect_predictions(model, loader, device)

    correct = sum(a == p for a, p in zip(true_labels, predicted_labels))
    total = len(true_labels)

    if total == 0:
        print(f"\nThe {args.split} split is empty, nothing to score.")
        return

    print(f"\n{args.split} split: {correct}/{total} correct "
          f"({correct / total:.1%})")

    counts = confusion_matrix(true_labels, predicted_labels, len(classes))

    print_report(classes, counts)
    print_matrix(classes, counts)


if __name__ == "__main__":
    main()
