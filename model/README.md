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
| `network.py` | the backbone registry, saves and loads checkpoints |
| `train.py` | trains the model, writes `railway_classifier.pt` |
| `predict.py` | classifies an image, finds the assets in one, and is the class the app calls |
| `export.py` | converts a checkpoint to ONNX, ExecuTorch, TFLite or TorchScript |
| `backends.py` | runs any of those formats behind one interface |

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
python model/train.py --arch mobilenet_v3_small          # a lighter backbone
```

Training a backbone other than the default writes to
`railway_classifier_<arch>.pt` rather than over the checkpoint the app
loads, so a comparison run cannot cost you the working model.

Nothing hardcodes the class list - it comes from the folder names under
`images/`, so adding a seventh class means adding a seventh folder.

## How it works

There are six classes and a few hundred photographs. That is far too few
to train a network from scratch, so this uses **transfer learning**: a
backbone that already learned general visual features from ImageNet,
with its 1000-class output layer replaced by a 6-class one. Which
backbone is `--arch`, and the default is a ResNet18.

Training runs in two phases:

1. **Head only.** The pretrained weights are frozen and only the new
   layer learns. A randomly initialised head produces large, wild
   gradients at first, and freezing stops those from damaging features
   that took ImageNet a long time to learn.
2. **Fine tuning.** The last convolution block is unfrozen and both
   train together at a 10x lower rate. Early layers detect edges and
   textures that transfer as-is; the last block is the class-specific
   one, so it is the part worth adapting. Which modules those are is per
   backbone - `layer4` in the ResNet, the last few entries of
   `features` in the mobile-style nets - and `ARCHITECTURES` records it.

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

## Choosing a backbone

`--arch` picks one of six, and the choice is written into the checkpoint
so `load_checkpoint` rebuilds whatever was actually trained. Parameter
counts are with the 6-class head attached:

| architecture | parameters | note |
| --- | ---: | --- |
| `resnet18` | 11,179,590 | the default, and the heaviest |
| `mobilenet_v3_large` | 4,209,718 | |
| `efficientnet_b0` | 4,015,234 | best accuracy per FLOP of this set |
| `mobilenet_v3_small` | 1,524,006 | |
| `squeezenet1_1` | 725,574 | its head is a 1x1 convolution, not a linear layer |
| `shufflenet_v2_x0_5` | 347,942 | the floor. 32x smaller than the default |

Torchvision ships `efficientnet_b0`, not the `lite0` variant. Lite0
drops the squeeze-excite blocks and hard-swish activations that some
accelerator delegates refuse to accept, so if a board rejects b0 that is
the first thing to try instead - it lives in `timm` rather than
torchvision.

**None of these fit an ESP-32.** The 08/11 meeting named an Arduino or
an ESP-32 as the drone-mounted target; an ESP32-S3 running ESP-DL is in
the neighbourhood of a 96x96 int8 person-detector, and the smallest
model here is an order of magnitude past that at 224x224. For that class
of board the honest options are capture-and-forward, or a separately
trained binary presence check. The devices the notes name for offline
work - phone, tablet, low end computer - are where these six become
real. Measuring that gap is the whitepaper's job, and `statistics/bench.py`
now does it - see below.

### What the six actually cost

Measured by `statistics/bench.py` on a 6-thread AMD CPU, torch 2.13, batch of 32.
The `x faster` column is against the resnet18 baseline:

| architecture | parameters | x smaller | ms/image | x faster |
| --- | ---: | ---: | ---: | ---: |
| `resnet18` | 11,179,590 | 1.0 | 47.3 | 1.00 |
| `mobilenet_v3_large` | 4,209,718 | 2.7 | 27.6 | 1.75 |
| `efficientnet_b0` | 4,015,234 | 2.8 | 43.9 | 1.09 |
| `mobilenet_v3_small` | 1,524,006 | 7.3 | 8.1 | 5.89 |
| `squeezenet1_1` | 725,574 | 15.4 | 26.3 | 1.80 |
| `shufflenet_v2_x0_5` | 347,942 | 32.1 | 7.0 | 6.73 |

**The two orderings are not the same ordering.** `squeezenet1_1` has
half the parameters of `mobilenet_v3_small` and takes three times as
long per image; `efficientnet_b0` is a third the size of the ResNet and
is not meaningfully faster than it. Parameters measure what a model
weighs on disk, and these are convolutional nets where the cost is the
feature maps rather than the weights - squeezenet holds full resolution
far longer than the mobilenets do. Pick a backbone on the measured
number, not the size.

Latency is the mean of three runs. Absolute figures move 5-15% between
runs on a machine doing anything else, so the ratio is the quantity
worth quoting - `shufflenet` against `resnet18` held between 6.55x and
6.89x across all three.
## Exporting for a device

```bash
python model/export.py --format onnx
python model/export.py --format all --quantize
```

Writes into `model/exported/`, one `<name>.classes.json` beside each
artifact - none of these formats carry the label list, and a model that
cannot name its own outputs is not much use at the far end.

| format | file | needs |
| --- | --- | --- |
| `torchscript` | `.pt` | nothing, torch is already here |
| `onnx` | `.onnx` | `onnx`, `onnxscript`; `onnxruntime` to run it |
| `executorch` | `.pte` | `executorch` |
| `tflite` | `.tflite` | `ai-edge-torch` to write, `ai-edge-litert` to run |

A missing package skips that format with a line saying which one, rather
than failing the run. `--quantize` adds an int8 ONNX calibrated on the
validation split - static quantisation, not dynamic, because dynamic
only touches linear layers and these backbones are nearly all
convolution.

Only TorchScript is exercised on the machine this was written on. ONNX
and ExecuTorch need their packages installed; `ai-edge-torch` needs
Linux, so a `.tflite` cannot be produced on Windows at all.

Two things about the ONNX target are worth knowing, because both were
bugs here and both are the kind that stay quiet until a device fails.

Torch 2.9+ writes the weights to an external `<name>.onnx.data` and
leaves the `.onnx` holding only the graph, a few dozen kB of it. An
export like that is two files, and copying the `.onnx` on its own ships a
model with no weights - inference is the first thing that notices.
`export_onnx()` therefore folds the sidecar back into the graph file and
deletes it, so an export is one portable file. `artifact_bytes()` counts
any sidecar that does turn up, so the reported size is the size of what
actually has to travel.

The exporter also prints progress lines ending in an emoji, which raises
`UnicodeEncodeError` on a Windows console still defaulting to cp1252 -
part way through an export that was otherwise working.
`allow_unicode_output()` switches the streams to UTF-8 before the export
runs, so nothing has to be set in the environment first.

`backends.py` then runs any of them behind one interface, so nothing
upstream has to know which it got:

```bash
python model/backends.py     # what this machine can run
```

Batches cross that interface as numpy rather than torch tensors, on
purpose: the machines most likely to be running an exported model are
the ones least able to afford a torch install.

## Scan profiles

`detect()` classifies a grid of overlapping crops, and how dense that
grid is decides nearly the whole cost of a scan. Three profiles, on a
1280x960 photograph:

| profile | windows | what it is for |
| --- | ---: | --- |
| `full` | 69 | the default. Three scales, half-window step |
| `balanced` | 29 | two scales, longer step |
| `edge` | 6 | one scale, coarse step, small batches |

```bash
python model/predict.py photo.jpg --detect --profile edge
```

`ZRA_SCAN_PROFILE` sets the default, so a device picks its own without
any code knowing. The cap on how many detections come back is part of
the profile rather than a constant - it was three, which is fewer than
some frames genuinely contain, and is now eight on `full`.

## Measuring what it costs

```bash
python statistics/bench.py                       # every suite
python statistics/bench.py --suite backbones     # just one, repeatable
python statistics/bench.py --output bench.json
```

Accuracy is written into the checkpoint by `train.py`. Cost was not
written down anywhere, and cost is what decides whether this runs on the
device the brief names. `statistics/bench.py` measures it and writes JSON -
`statistics/bench_results.json` is a committed run, so the numbers quoted in
this README and in the whitepaper can be checked against the machine
they came from.

| suite | what it answers |
| --- | --- |
| `environment` | which machine, which torch, which runtimes are installed |
| `checkpoint` | file size, load time, cold start of `RailwayClassifier` |
| `runtimes` | torch against each exported format, on speed *and* on agreement |
| `threads` | latency against core count |
| `training` | the per-epoch curves, reshaped into plottable series |
| `windows` | how many crops a scan costs, by profile and frame shape |
| `backbones` | size and speed of all six architectures |
| `profiles` | end-to-end `detect()` cost per profile |
| `store` | what a capture costs to write, and what it fills |
| `sync` | what a batch weighs on the wire, per tier |
| `api` | endpoint latency, and whether a redelivery double-stores |

Nothing here needs `images/`. Accuracy is deliberately not measured: a
fresh accuracy figure needs the exact photographs the original run was
split from, and re-collecting the folder from `ATTRIBUTION.csv` gives a
different and larger set, which would put training images into the test
split. Cost, unlike accuracy, does not depend on the pictures.

Three things worth knowing about the numbers:

- **A scan costs its window count, not its resolution.** Every window is
  resized to 224 before the model sees it, so a 12-megapixel drone still
  and a VGA frame of the same shape cost the same. Measured at 52-67 ms
  per window across every profile and resolution. The window grid keys
  off the frame's *aspect ratio* - a 4:3 photograph is 69 windows on
  `full` whether it is 640x480 or 4000x3000; 16:9 is 89.
- **ONNX Runtime is about twice PyTorch's speed on the same weights**,
  and the two agree to 1.9e-06 on the logits. `--suite runtimes` checks
  the agreement, because a fast backend that is quietly wrong is worse
  than no backend.
- **Threads stop paying.** Going 1 to 6 threads on the ResNet is 3.6x,
  not 6x - efficiency falls from 86% at two threads to 60% at six. A
  four-core board gets about 2.6x out of its four cores.

## Checking a trained model

```bash
python model/predict.py some_photo.jpg
python model/predict.py some_photo.jpg --model exported/railway.onnx
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
