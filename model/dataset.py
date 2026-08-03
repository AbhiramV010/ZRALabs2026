"""Loading and splitting the images/ folder, one directory per class."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGE_ROOT = Path(__file__).resolve().parent.parent / "images"

# ImageNet statistics, the pretrained backbone expects these
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224

# the remainder goes to the test split
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# marks a file written by augment.py, as '<original-stem>__augN.jpg'
AUG_MARKER = "__aug"


def is_augmented(path):
    return AUG_MARKER in Path(path).stem


def source_stem(path):
    """The original a file came from, or its own name if it is one."""
    return Path(path).stem.split(AUG_MARKER)[0]


def get_train_transform():
    """Augmentation for the training split."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_eval_transform():
    """Deterministic pipeline for validation, test and inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def stratified_indices(targets, seed=42):
    """Split positions per class, so each split holds every class."""
    generator = torch.Generator().manual_seed(seed)

    by_class = {}

    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    train_idx, val_idx, test_idx = [], [], []

    for label in sorted(by_class):
        indices = by_class[label]

        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in order]

        n_train = int(len(shuffled) * TRAIN_FRACTION)
        n_val = int(len(shuffled) * VAL_FRACTION)

        train_idx += shuffled[:n_train]
        val_idx += shuffled[n_train:n_train + n_val]
        test_idx += shuffled[n_train + n_val:]

    return train_idx, val_idx, test_idx


def build_datasets(image_root=IMAGE_ROOT, seed=42):
    """Return (train, val, test) datasets plus the class names."""
    image_root = Path(image_root)

    if not image_root.is_dir():
        raise FileNotFoundError(f"No image folder at {image_root}")

    # the same folder twice: augmented for training, plain for the rest
    train_source = datasets.ImageFolder(image_root, get_train_transform())
    eval_source = datasets.ImageFolder(image_root, get_eval_transform())

    if len(train_source) == 0:
        raise RuntimeError(f"{image_root} has no images in it")

    samples = train_source.samples

    # only real photographs are split, so the seed gives the same three
    # groups whether or not augment.py has been run
    real = [
        position for position, (path, _) in enumerate(samples)
        if not is_augmented(path)
    ]

    if not real:
        raise RuntimeError(f"{image_root} holds nothing but augmented copies")

    train_idx, val_idx, test_idx = [
        [real[local] for local in group]
        for group in stratified_indices(
            [samples[position][1] for position in real],
            seed=seed
        )
    ]

    # a generated copy follows its original into training, and is discarded
    # when that original was held out - otherwise it would leak the answer
    trained_on = {
        (samples[position][1], source_stem(samples[position][0]))
        for position in train_idx
    }

    train_idx += [
        position for position, (path, label) in enumerate(samples)
        if is_augmented(path) and (label, source_stem(path)) in trained_on
    ]

    train_set = Subset(train_source, train_idx)
    val_set = Subset(eval_source, val_idx)
    test_set = Subset(eval_source, test_idx)

    return train_set, val_set, test_set, train_source.classes


def build_loaders(batch_size=16, num_workers=0, image_root=IMAGE_ROOT, seed=42):
    """Wrap the three splits in DataLoaders."""
    train_set, val_set, test_set, classes = build_datasets(image_root, seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, classes
