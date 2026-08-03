"""Write augmented copies of the dataset into images/, alongside the originals.

    python model/augment.py                 # 3 variants per original
    python model/augment.py --variants 5
    python model/augment.py --clean         # delete every generated file

Copies are named '<original-stem>__augN.jpg', which is how dataset.py tells
them apart from real photographs. Only originals that land in the training
split are ever used, so nothing generated here can leak into validation or
test.

Attribution for a generated file is the same as its original: find the row
for '<original-stem>.jpg' in images/ATTRIBUTION.csv.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import AUG_MARKER, IMAGE_ROOT, is_augmented
else:
    from .dataset import AUG_MARKER, IMAGE_ROOT, is_augmented


EXTENSIONS = {".jpg", ".jpeg", ".png"}

# generated files are saved no larger than this, to keep the folder small
MAX_EDGE = 640
JPEG_QUALITY = 88


def random_crop(image, rng):
    """Take between 65% and 100% of the frame, from anywhere in it."""
    scale = rng.uniform(0.65, 1.0)
    ratio = rng.uniform(0.85, 1.18)

    width = min(image.width, round(image.width * scale * ratio))
    height = min(image.height, round(image.height * scale))

    left = rng.randint(0, image.width - width)
    top = rng.randint(0, image.height - height)

    return image.crop((left, top, left + width, top + height))


def random_rotate(image, rng):
    """A few degrees either way, expanded so no corner is cut off."""
    angle = rng.uniform(-12, 12)

    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def random_colour(image, rng):
    for enhancer, spread in (
        (ImageEnhance.Brightness, 0.30),
        (ImageEnhance.Contrast, 0.30),
        (ImageEnhance.Color, 0.35),
        (ImageEnhance.Sharpness, 0.40),
    ):
        image = enhancer(image).enhance(rng.uniform(1 - spread, 1 + spread))

    return image


def augment(image, rng):
    """One randomly altered version of a photograph."""
    image = random_crop(image, rng)

    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if rng.random() < 0.6:
        image = random_rotate(image, rng)

    image = random_colour(image, rng)

    # a stand-in for a poorly focused or low light phone photo
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))

    if rng.random() < 0.10:
        image = image.convert("L").convert("RGB")

    return shrink(image)


def shrink(image):
    longest = max(image.width, image.height)

    if longest <= MAX_EDGE:
        return image

    factor = MAX_EDGE / longest

    return image.resize(
        (round(image.width * factor), round(image.height * factor)),
        Image.LANCZOS
    )


def originals(class_dir):
    """Real photographs in a class folder, generated copies excluded."""
    return sorted(
        path for path in class_dir.iterdir()
        if path.suffix.lower() in EXTENSIONS and not is_augmented(path)
    )


def generated(class_dir):
    return sorted(
        path for path in class_dir.iterdir()
        if path.suffix.lower() in EXTENSIONS and is_augmented(path)
    )


def class_dirs(image_root):
    return sorted(path for path in Path(image_root).iterdir() if path.is_dir())


def clean(image_root):
    removed = 0

    for class_dir in class_dirs(image_root):
        for path in generated(class_dir):
            path.unlink()
            removed += 1

    print(f"Removed {removed} generated files")

    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--variants", type=int, default=3,
                        help="augmented copies per original (default 3)")
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="delete generated files and stop")

    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"No image folder at {args.images}")

    if args.clean:
        clean(args.images)
        return

    # start from a clean slate, so re-running doesn't pile copies on copies
    clean(args.images)

    rng = random.Random(args.seed)

    written = 0

    for class_dir in class_dirs(args.images):
        sources = originals(class_dir)

        for source in sources:
            with Image.open(source) as handle:
                image = handle.convert("RGB")

                for index in range(args.variants):
                    target = class_dir / f"{source.stem}{AUG_MARKER}{index}.jpg"

                    augment(image, rng).save(
                        target,
                        "JPEG",
                        quality=JPEG_QUALITY
                    )

                    written += 1

        print(f"  {class_dir.name:<16}{len(sources):>4} originals "
              f"-> {len(sources) * args.variants:>4} copies")

    print(f"\nWrote {written} augmented images into {args.images}")
    print("Only copies of training-split originals are used, see dataset.py")


if __name__ == "__main__":
    main()
