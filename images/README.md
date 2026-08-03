# Sample image dataset

The image collection listed as deliverable 5 in the project brief ("a small
sample image dataset or test image collection"), and the ~30-50 images per
category the advanced version needs for training a custom classifier.

## Layout

One folder per category, matching the six labels in the brief:

```
images/
  train/            trains, locomotives, multiple units
  track/            rails, sleepers, ballast, points
  signal/           colour light and semaphore signals
  platform/         station platforms
  overhead_wire/    catenary, masts, pantographs
  crossing_gate/    level crossing barriers and gates
  ATTRIBUTION.csv   source, author and licence for every file
```

Filenames are `<category>_<source-file-name>.jpg`, so an image's row in
`ATTRIBUTION.csv` is easy to find.

Images are downloaded at up to 1024px wide - large enough to train on, small
enough to keep in the repo.

## Augmented copies

Files with `__aug0`, `__aug1`, `__aug2` in the name are **generated**, not
photographs. `model/augment.py` writes them:

```bash
python model/augment.py                 # 3 copies per original
python model/augment.py --variants 5    # more
python model/augment.py --clean         # delete every generated file
```

Each copy is a random crop, flip, small rotation, colour shift, and
occasionally a blur or a desaturation, saved at up to 640px.

A copy is only ever used when its original landed in the **training** split.
`dataset.py` splits the real photographs first and attaches the copies
afterwards, so a generated version of a held-out photo is discarded rather
than scored against - otherwise validation and test accuracy would be
measuring images the model had effectively already seen.

Attribution carries over from the original: strip the `__augN` suffix and
look that filename up in `ATTRIBUTION.csv`.

They cost about 50 MB on top of the 71 MB of originals. Delete them with
`--clean` before committing if that matters, training regenerates them.

## Where they come from

Everything is from [Wikimedia Commons](https://commons.wikimedia.org), so
every file is under a free licence (mostly CC BY-SA and public domain). The
exact licence and author of each image are in `ATTRIBUTION.csv`.

**When presenting or publishing this project, credit the images.** The CSV has
what's needed: author, licence and a link back to the source page.

## Rebuilding or extending the set

`collect_images.py` in the project root builds this folder:

```bash
python collect_images.py                        # 45 per category
python collect_images.py --count 30             # fewer
python collect_images.py --only signal platform # top up one or two classes
```

It skips files already on disk, so re-running it tops up the set rather than
starting over, and rewrites `ATTRIBUTION.csv` each time.

## Before training

These are search results, not a curated dataset - look through each folder and
delete anything that doesn't clearly show its category. A search for "railway
platform" will occasionally return a train that happens to have a platform
behind it. Class purity matters more than hitting 45 images.
