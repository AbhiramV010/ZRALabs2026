# Statistics

Everything that measures this project, and everything those measurements
produced. Nothing in here is imported by the application - `app.py`,
`api.py` and `model/` do not depend on this folder, and deleting it would
cost the project only its numbers.

## Contents

| path | what it is |
| --- | --- |
| `bench.py` | the measurement harness. Runs the suites and writes JSON |
| `bench_results.json` | a committed run, and the source of every figure below |
| `report.py` | builds `benchmarks.html` from that JSON |
| `make_assets.py` | writes `graphs/*.svg` and `tables/*.csv` from that JSON |
| `common.py` | loads the results, escaping and table helpers |
| `charts.py` | the four charts, as inline SVG |
| `style.py` | the report page's tokens and stylesheet |
| `benchmarks.html` | the report, readable on its own |
| `graphs/` | one standalone SVG per figure, for dropping into a document |
| `tables/` | one CSV per table |

## Running it

```bash
python statistics/bench.py                      # every suite, printed as JSON
python statistics/bench.py --suite backbones    # one suite; the flag stacks
python statistics/bench.py --output statistics/bench_results.json

python statistics/report.py                     # rebuild benchmarks.html
python statistics/make_assets.py                # rebuild graphs/ and tables/
```

`report.py` and `make_assets.py` read `bench_results.json` and never
measure anything themselves, so a figure in the report and the same
figure in a CSV cannot disagree. Re-running `bench.py` with `--output`
pointed at that file is what updates them both.

On Windows, `--suite runtimes` wants `PYTHONIOENCODING=utf-8` if you have
just re-exported an ONNX model - torch's exporter prints an emoji that a
cp1252 console cannot encode.

## The suites

| suite | what it answers |
| --- | --- |
| `environment` | which machine, which torch, which runtimes are installed |
| `code` | how much code there is, and how much of it is tests |
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

`environment` is always included, because it is what makes the rest mean
anything.

## What is not measured here

**Accuracy.** The figures quoted in the report are read out of the
checkpoint, where `train.py` recorded them at the end of the run that
produced the weights. Nothing here re-computes them, and that is
deliberate: `images/` holds only `ATTRIBUTION.csv` on a fresh clone, and
re-collecting the folder from that manifest gives 633 photographs where
the original run used 351. Splitting the larger set with the same seed
produces a different split, which would put original training images into
the test set and report an accuracy that is too high. A fresh confusion
matrix needs the exact 351 files, and they are not recoverable from
what the repository holds.

Cost does not have that problem. The same convolutions run over the same
number of pixels whatever is in them, so every timing here is measured on
generated frames and is unaffected by the dataset's absence. Sizes in
*bytes* do depend on content, so those are measured on photo-like frames
and bracketed with a flat-colour floor and a random-noise worst case.

## Reading the numbers

Every figure is from one machine - a six-thread AMD CPU with no GPU -
and is a baseline, not a prediction for a phone or a single-board
computer.

Absolute latency moved 5-15% between three runs of the same suite on that
machine. **Ratios are the stable quantity**: `shufflenet_v2_x0_5` against
`resnet18` held between 6.55x and 6.89x across all three, while both
their absolute figures moved. Quote ratios, and treat any gap under about
15% as noise.

`bench.py` guards one failure mode directly. `perf_counter` keeps counting
through a machine suspend, and a long run left alone will eat one - an
early scan measured 6,988 seconds for work that takes 5.5. Any sample
more than 20x its own fastest is now flagged `stall_suspected` rather than
reported as a result. No figure in `bench_results.json` carries that flag.
