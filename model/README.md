# Railway asset classifier

A PyTorch image classifier trained on `images/`. Given a photo it says
which railway asset the photo is *of* - one label for the whole image.

This is a classifier, not a detector: the network only ever answers what
one whole picture is. The boxes in the app come from `detect()` in
`predict.py`, which runs that classifier over a grid of overlapping
crops and merges the ones that agree, so a box is window shaped rather
than an object outline. There is no YOLO in this project.

## The scripts

| file | what it does |
| --- | --- |
| `collect_images.py` | downloads `images/` from Wikimedia Commons |
| `collect_wire.py` | downloads `images/overhead_wire`, cropped to the wire |
| `dataset.py` | loads `images/`, splits it, defines the transforms |
| `augment.py` | writes augmented copies of the photos into `images/` |
| `network.py` | builds the ResNet18, saves and loads checkpoints |
| `train.py` | trains the model, writes `railway_classifier.pt` |
| `predict.py` | classifies an image, finds the assets in one, and is the class the app calls |

## Training it

```bash
pip install torch torchvision
python model/collect_images.py     # the five ordinary classes
python model/collect_wire.py       # the sixth, cropped as it downloads
python model/augment.py            # 3 generated copies per photograph
python model/train.py
```

It writes two files next to the scripts:

- `railway_classifier.pt` - the weights, plus the class names and scores
- `training_history.json` - per-epoch losses, useful for a write-up graph

`network.pick_device()` picks the GPU when there is one, so no flag turns
this on. The 25-epoch default run takes about 5 minutes on an RTX 3060 Ti
and about 19 on the CPU. A run sent elsewhere with `--output` keeps its
history beside it rather than overwriting `training_history.json`.

Options worth knowing:

```bash
python model/train.py --epochs 20 --finetune-epochs 15   # train longer
python model/train.py --no-finetune                      # head only, quicker
python model/train.py --seed 7                           # a different split
```

Nothing hardcodes the class list - it comes from the folder names under
`images/`, so adding a seventh class means adding a seventh folder.

## How it works

There are six classes and a few hundred photographs. That is far too few
to train a network from scratch, so this uses **transfer learning**: a
ResNet18 that already learned general visual features from ImageNet,
with its 1000-class output layer replaced by a 6-class one.

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
split in the same proportion. A plain random split this small can
easily leave a class with one or two test images. The test split is
only ever touched at the very end.

Because the set is small the training images are augmented twice over.

**On the fly**, in `dataset.py`: random crop, flip, rotation and colour
jitter applied as each image is loaded, so the model sees a slightly
different version of every photo every epoch instead of memorising the
same exact pictures.

**On disk**, in `augment.py`: three altered copies of each photograph
written into `images/` as `<name>__augN.jpg`. Run it before training:

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
score would then be measuring images the model trained on.

The weights that get saved are the ones from the **best validation
epoch**, not the last one. On a dataset this size the final epoch is
often a worse model than one from a few epochs back.

## The overhead wire class

`overhead_wire` is collected by its own script, and it is the only class
whose images are cropped before they are saved. Both of those are
answers to the same problem.

**Whether a photo contains wire is independent of what the photo is
of.** Half of `images/platform` is modern electrified station with
catenary across the frame; a good share of `images/train` is electric
traction working under the wires it draws current from. So a photograph
that matches "railway catenary" on Commons is, nine times in ten, a
picture of a train or a station that happens to have wire above it. Kept
whole, those images teach the model that `overhead_wire` means *track
photographed from a platform*, because that is what most of the pixels
in them are.

`collect_wire.py` therefore keeps only the band of sky the wires run
through:

```bash
python model/collect_wire.py              # download and crop
python model/collect_wire.py --recrop     # re-crop what is on disk
```

It finds that band by looking for what a wire physically is - a line a
few pixels across, darker or brighter than the sky either side of it -
and taking the rows that carry enough of them. Four things then have to
hold before the crop is saved, and roughly three candidates in four fail
at least one:

- **The crop stops above the ground.** Ballast, sleepers, undergrowth
  and brick are *busy*; sky is flat, and a wire crossing it is too thin
  to change that. Local roughness is measured on a copy of the picture
  with its thin horizontal lines median-filtered out, so a dense
  catenary cannot be mistaken for texture, and the crop is cut off above
  the first sustained run of rough rows.
- **The crop is still mostly sky** once it has been made.
- **The lines cross the frame.** A bare winter tree is thousands of thin
  dark lines against a bright sky and passes every test above, but its
  lines are bunched in the corner it grows in. Contact wire is strung
  between two masts.
- **The photograph is in colour.** Commons answers these queries with a
  lot of digitised pre-war trade press, and a scan of a wire crops as
  cleanly as a photograph of one.

### What the cropper cannot decide

That a band of wire is present is a question about pixels. That the band
is what the picture is *about* is not. A montage of four catenary photos,
a close-up with a white ellipse drawn on it, and a street of tram wire
over a shopfront all crop perfectly well.

Those are caught by eye and written down in `model/wire_excluded.txt`,
one filename per line with the reason. `collect_wire.py` reads that file
and will not download anything listed in it, so the audit survives a
re-run - the first prune was done by deleting files, and the next
collection put every one of them straight back.

### Why there is no separate wire model any more

There was one: a two-class specialist with its own dataset, trained to
answer "is there overhead line equipment in this photo?" and consulted
alongside the five-class model. It existed because the first attempt at
folding wire into the main model collapsed the class folders into
"wire" against "everything else", which labels every electrified
platform and every electric train a negative, and scored 40% recall for
it.

Cropping the class down to the wire fixes that at the source, so the
sixth folder can just be a folder. The specialist and its checkpoint are
gone; `wire_training_history.json` is what is left of that run.

## How well it does

351 photographs - 60 each of the five ordinary classes, 51 of
`overhead_wire` - split 980 training images (augmented copies included),
52 validation and 54 test. With the default settings and seed 42:

```
validation accuracy   78.8%
test accuracy         70.4%   (38 of 54)

class            precision   recall     f1  support
crossing_gate         0.54     0.78   0.64        9
overhead_wire         1.00     1.00   1.00        9
platform              0.60     0.67   0.63        9
signal                0.80     0.89   0.84        9
track                 0.80     0.44   0.57        9
train                 0.57     0.44   0.50        9
```

Chance is 17%, so 70% is a real result, and adding a sixth class cost
the model nothing - the five-class version scored 70.0%.

**`overhead_wire` is the best class in the set, at 9 of 9.** That is the
cropping, not the architecture. Every other class is a photograph of a
railway scene, and railway scenes contain each other: rails are visible
in the background of nearly every platform, signal and train photo. A
cropped wire image contains sky and wire and nothing else, so there is
nothing in it for another class to claim. It scored 9 of 9 on two
separate training runs, so it is not one lucky split.

It is worth being clear about what that number does and does not say. It
says the model can recognise a band of catenary against sky, measured on
images cropped the way it was trained. A photograph of a whole station
is a harder question, and the app asks that one.

**`track` and `train` are the weakest**, both at 0.44 recall, and both
scatter rather than failing in one direction: `train` loses three to
`platform` and two to `crossing_gate`, `track` loses two to
`crossing_gate`. That is a labelling problem more than a model one. A
train photographed at a station is a picture of a platform about as much
as it is a picture of a train, and nothing in a one-label-per-image
dataset can settle which.

The five weather-and-scenery classes move around by 10 to 20 points of
recall between runs while the totals barely shift - 54 test images is
nine per class, so one image is 11 points. Read the ranking, not the
decimals. `overhead_wire` sitting at the top of both runs is the part
that means something.

Training accuracy reaches 99% while validation stalls in the low 80s.
That gap is overfitting, and it is what a few dozen images per class
buys you. Training for longer does not fix it. The fix is more and
cleaner images, not more epochs - and the wire class is the evidence,
since it is the one class that got cleaner rather than bigger.

One thing to know about `overhead_wire` in the app: `predict.py` finds
objects by classifying a grid of **square** crops, and a square crop of
catenary is mostly sky. The whole-frame ranking is where this class is
strongest; it is the least likely of the six to get a tight box drawn
round it.

## Checking a trained model

```bash
python model/predict.py some_photo.jpg
```

The scores stored in the checkpoint come from the run that produced it -
`network.load_metrics()` reads them back without building a model.

## Using it from the app

```python
import streamlit as st
from model.predict import RailwayClassifier

@st.cache_resource
def get_classifier():
    return RailwayClassifier()

results = get_classifier().classify(image, top_k=3)
# [("Train", 0.914), ("Overhead Wire", 0.058), ("Track", 0.013)]
```

Labels come back title-cased (`overhead_wire` -> `Overhead Wire`), which
is the form `DetectionResult` in `result.py` already colour-maps. The
`@st.cache_resource` matters - without it the checkpoint reloads on
every rerun.
