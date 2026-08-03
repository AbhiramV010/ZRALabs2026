# Railway asset classifier

A PyTorch image classifier trained on `images/`. Given a photo it says
which railway asset the photo is *of* - one label for the whole image.

This is a classifier, not a detector. It does not draw boxes. YOLO in
`app.py` does the boxes; this is the custom-trained model the brief asks
for on top of it.

## The scripts

| file | what it does |
| --- | --- |
| `dataset.py` | loads `images/`, splits it, defines the transforms |
| `augment.py` | writes augmented copies of the photos into `images/` |
| `network.py` | builds the ResNet18, saves and loads checkpoints |
| `train.py` | trains the model, writes `railway_classifier.pt` |
| `evaluate.py` | per-class scores and a confusion matrix on the test split |
| `predict.py` | classifies a single image, and the class the app calls |

## Training it

```bash
pip install torch torchvision
python model/train.py
```

About 90 seconds on a CPU. It writes two files next to the scripts:

- `railway_classifier.pt` - the weights, plus the class names and scores
- `training_history.json` - per-epoch losses, useful for a write-up graph

Options worth knowing:

```bash
python model/train.py --epochs 20 --finetune-epochs 15   # train longer
python model/train.py --no-finetune                      # head only, quicker
python model/train.py --seed 7                           # a different split
```

## How it works

There are 225 images, 45 per class. That is far too few to train a
network from scratch, so this uses **transfer learning**: a ResNet18
that already learned general visual features from ImageNet, with its
1000-class output layer replaced by a 5-class one.

Training runs in two phases:

1. **Head only.** The pretrained weights are frozen and only the new
   layer learns. A randomly initialised head produces large, wild
   gradients at first, and freezing stops those from damaging features
   that took ImageNet a long time to learn.
2. **Fine tuning.** The last convolution block (`layer4`) is unfrozen
   and both train together at a 10x lower rate. Early layers detect
   edges and textures that transfer as-is; the last block is the
   class-specific one, so it is the part worth adapting.

The split is 70/15/15, stratified, so every class appears in every
split in the same proportion. A plain random split of 225 images can
easily leave a class with one or two test images. The test split is
only ever touched at the very end.

Because the set is small the training images are augmented twice over.

**On the fly**, in `dataset.py`: random crop, flip, rotation and colour
jitter applied as each image is loaded, so the model sees a slightly
different version of every photo every epoch instead of memorising the
same exact pictures.

**On disk**, in `augment.py`: three altered copies of each photograph
written into `images/` as `<name>__augN.jpg`, which grows the training
split from 155 images to 620. Run it before training:

```bash
python model/augment.py            # 3 copies each
python model/augment.py --clean    # remove them again
```

The two stack - a generated copy still gets the on-the-fly treatment
when it is loaded.

Only copies of **training** photographs are used. `dataset.py` splits
the real images first, then attaches the copies that belong to the
training group and discards the rest. Splitting the folder as a whole
would scatter copies of one photo across all three splits, and the test
score would then be measuring images the model trained on. Validation
and test remain 30 and 40 untouched originals either way.

The weights that get saved are the ones from the **best validation
epoch**, not the last one. On a dataset this size the final epoch is
often a worse model than one from a few epochs back.

## How well it does

With the default settings and seed 42:

```
validation accuracy   83.3%
test accuracy         70.0%   (28 of 40)

class            precision   recall     f1  support
crossing_gate         0.86     0.75   0.80        8
platform              0.67     0.75   0.71        8
signal                0.67     0.75   0.71        8
track                 0.50     0.50   0.50        8
train                 0.86     0.75   0.80        8
```

Chance is 20%, so 70% is a real result, but the honest reading is that
the dataset is the limit here, not the architecture. Two things stand
out:

- **`track` is the weakest class.** That is a labelling problem more
  than a model one. Rails are visible in the background of nearly every
  platform, signal and train photo, so "is this a picture of track" is
  genuinely ambiguous. The confusion matrix shows it mostly losing to
  `platform`.
- **Train accuracy hits ~99% while validation stalls around 80%.** That
  gap is overfitting, and it is what 45 images per class buys you.
  Training for longer does not fix it - 30 fine-tuning epochs scored
  *worse* on test than 10.

The fix is more and cleaner images, not more epochs. The README in
`images/` says the same thing: these are search results, not a curated
set, and class purity matters more than the count.

Also note `overhead_wire` is one of the six categories in the brief but
has no folder in `images/`, so the model only knows five classes. Run
`python collect_images.py --only overhead_wire` and retrain to add it -
nothing here hardcodes the class list, it comes from the folder names.

## Checking a trained model

```bash
python model/evaluate.py                # test split, full breakdown
python model/evaluate.py --split val    # or the validation split
python model/predict.py some_photo.jpg  # one image
```

`evaluate.py` needs the same `--seed` that training used, otherwise it
rebuilds a different split and scores the model on images it trained on.

## Using it from the app

```python
import streamlit as st
from model.predict import RailwayClassifier

@st.cache_resource
def get_classifier():
    return RailwayClassifier()

results = get_classifier().classify(image, top_k=3)
# [("Train", 0.914), ("Crossing Gate", 0.058), ("Track", 0.013)]
```

Labels come back title-cased (`crossing_gate` -> `Crossing Gate`), which
is the form `DetectionResult` in `result.py` already colour-maps. The
`@st.cache_resource` matters - without it the checkpoint reloads on
every rerun.
