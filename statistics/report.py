"""Builds statistics/benchmarks.html from bench_results.json."""

from pathlib import Path

from common import RESULTS, fmt, table, datatable
from charts import chart_training, chart_backbones, chart_profiles, chart_threads
from style import CSS

R = RESULTS
ENV = R["environment"]
CK = R["checkpoint"]
TR = R["training"]["main"]
VAR = R["variance"]["backbones_batch_32"]
SPECS = {b["architecture"]: b for b in R["backbones"]["backbones"]}
STORE = R["store"]
SYNC = R["sync"]
API = R["api"]
TESTS = R["test_suite"]
THREADS = R["threads"]["measurements"]


def kpi(label, value, unit="", note=""):
    unit_html = f' <span class="kpi-unit">{unit}</span>' if unit else ""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}{unit_html}</div>{note_html}</div>')


def kpis(items):
    return '<div class="kpis">' + "".join(items) + "</div>"


def figure(title, note, chart, legend="", twin=""):
    legend_html = f'<div class="legend">{legend}</div>' if legend else ""
    return (f'<figure class="figure">'
            f'<div class="figure-head"><span class="figure-title">{title}</span>'
            f'<span class="figure-note">{note}</span></div>'
            f"{chart}{legend_html}{twin}</figure>")


def swatch(slot, label):
    return (f'<span><i class="swatch" style="background:var(--{slot})"></i>'
            f"{label}</span>")


def eyebrow(text, suite):
    return f'<div class="eyebrow">{text} <em>{suite}</em></div>'


# ------------------------------------------------------------------ sections

def sec_summary():
    return f"""
<section id="summary">
  {eyebrow("Summary", "all suites")}
  <h2>What the measurements say</h2>
  <p class="lede">Five findings carry most of the weight. Each is measured
  below, and each comes from a suite that can be re-run.</p>
  <ul>
    <li><strong>Model size does not predict model speed.</strong> Across the six
    backbones, parameter count spans 32x and measured latency spans 6.8x, and the
    two do not rank in the same order.</li>
    <li><strong>A scan costs its window count, not its resolution.</strong> Every
    crop is resized to 224 before the model sees it, so a 12-megapixel drone
    still and a VGA frame of the same shape cost the same to scan.</li>
    <li><strong>ONNX Runtime is about twice PyTorch's speed on identical
    weights</strong>, agreeing to 1.9e-06 on the logits.</li>
    <li><strong>Threads stop paying early.</strong> Six threads return 3.6x, not
    6x; parallel efficiency falls from 86% to 60%.</li>
    <li><strong>The thumbnail is the storage budget.</strong> Metadata is 57
    bytes per capture on the wire, an image is 9,310 - a factor of 163.</li>
  </ul>
</section>"""


def sec_training():
    rows = []
    series = TR["series"]
    for index in range(TR["epochs"]):
        phase = "head only" if index + 1 < TR["phase_boundary_epoch"] else "fine-tuning"
        rows.append([
            str(index + 1), phase,
            f'{series["train_loss"][index]:.4f}',
            f'{series["train_acc"][index]:.4f}',
            f'{series["val_loss"][index]:.4f}',
            f'{series["val_acc"][index]:.4f}',
            f'{series["train_acc"][index] - series["val_acc"][index]:+.4f}',
        ])

    twin = datatable(
        "Per-epoch figures (25 rows)",
        ["epoch", "phase", "train loss", "train acc", "val loss", "val acc",
         "gap"],
        rows,
        ["right", "left", "right", "right", "right", "right", "right"],
    )

    return f"""
<section id="training">
  {eyebrow("Training", "training / checkpoint")}
  <h2>Two phases, and where the overfitting enters</h2>
  <p class="lede">The run trains in two phases: fifteen epochs with the
  backbone frozen, then ten with the last block unfrozen at a tenth of the
  learning rate. The curves separate sharply at the boundary.</p>

  {figure(
      "Accuracy per epoch",
      "resnet18, seed 42, 25 epochs",
      chart_training(),
      swatch("s1", "training") + swatch("s2", "validation"),
      twin,
  )}

  <p>Phase one plateaus: validation accuracy sits around 65% from epoch 6
  onward and the train/validation gap stays under 12 points. Unfreezing changes
  both at once. Within two epochs validation jumps from 69.6% to 77.2% and
  training accuracy climbs from 76.7% to 90.9%; by the end training reaches
  98.9% while validation has stopped moving at 76.1%.</p>

  <p>So the fine-tuning phase bought roughly <strong>13 points of validation
  accuracy and all of the overfitting</strong>. The final gap is 22.8 points.
  Validation loss bottoms at epoch 23 and rises afterwards, which is why the
  saved weights are that epoch's rather than the last one's.</p>

  {kpis([
      kpi("Best validation", "78.3", "%", "epoch 23, the weights that were kept"),
      kpi("Training at that epoch", "97.8", "%", "a 19-point gap already"),
      kpi("Held-out test accuracy", "70.7", "%", "chance is 16.7% across six classes"),
      kpi("Final train/val gap", "22.8", "pts", "up from 7 points before fine-tuning"),
  ])}

  <div class="callout">
    <p>The test figure in the checkpoint is <strong>70.71%</strong>. The model
    README quotes 70.4% (38 of 54) alongside a per-class table. The two do not
    come from the same run, so a whitepaper should cite one run and say which -
    they are close enough to look like a rounding difference and are not one.</p>
  </div>
</section>"""


def sec_backbones():
    order = sorted(VAR, key=lambda a: -SPECS[a]["parameters"])
    rows = []
    for arch in order:
        spec, var = SPECS[arch], VAR[arch]
        rows.append([
            f"<code>{arch}</code>",
            fmt(spec["parameters"]),
            f'{spec["vs_resnet18_parameters"]:.1f}x',
            f'{var["per_image_ms_mean"]:.1f}',
            f'{var["speed_vs_resnet18_mean"]:.2f}x',
            f'{var["per_image_ms_cv_percent"]:.1f}%',
            f'{spec["torchscript_mb"]:.1f}',
        ])

    return f"""
<section id="backbones">
  {eyebrow("Backbones", "backbones")}
  <h2>Parameters are the wrong axis</h2>
  <p class="lede">Six architectures are selectable with <code>--arch</code>.
  Ordering them by size and by speed produces two different orders, and only one
  of them is the one that matters when picking hardware.</p>

  {figure(
      "Size against measured cost",
      "batch of 32, mean of three runs, rows ordered by parameter count",
      chart_backbones(),
      swatch("s1", "parameters") + swatch("s2", "milliseconds per image"),
      datatable(
          "Backbone figures",
          ["architecture", "parameters", "x smaller", "ms/image", "x faster",
           "spread", "TorchScript MB"],
          rows,
          ["left", "right", "right", "right", "right", "right", "right"],
      ),
  )}

  <p>The left column descends cleanly; the right column does not follow it.
  <code>squeezenet1_1</code> holds half the parameters of
  <code>mobilenet_v3_small</code> and takes <strong>three times as long per
  image</strong>. <code>efficientnet_b0</code> is a third of the ResNet's size
  and is not meaningfully faster than it - 1.09x, inside the run-to-run
  spread.</p>

  <p>The reason is that these are convolutional networks whose cost is the
  feature maps rather than the weights. SqueezeNet holds full spatial resolution
  much longer than the MobileNets do, so it moves far more activation data for
  far fewer parameters. Parameter count measures what a model weighs on disk;
  it is not a proxy for what it costs to run.</p>

  <div class="callout">
    <p>Absolute latency moved 5-15% between three runs on this machine. The
    <em>ratio</em> between two backbones is the stable quantity:
    <code>shufflenet_v2_x0_5</code> against <code>resnet18</code> held between
    6.55x and 6.89x across all three. Quote ratios, and treat any gap under
    about 15% as noise.</p>
  </div>
</section>"""


def sec_scan():
    win_rows = [
        [w["resolution"], f'{w["width"]}x{w["height"]}',
         f'{w["aspect_ratio"]:.2f}', f'{w["megapixels"]:.2f}',
         str(w["full"]), str(w["balanced"]), str(w["edge"])]
        for w in R["windows"]
    ]
    scan_rows = [
        [s["resolution"], s["profile"], str(s["windows"]),
         fmt(s["headline_ms"], 0), f'{s["ms_per_window"]:.1f}',
         f'{s["frames_per_second"]:.2f}']
        for s in R["profiles"]["scans"]
    ]

    return f"""
<section id="scan">
  {eyebrow("Scan cost", "windows / profiles")}
  <h2>The frame's shape sets the cost, not its size</h2>
  <p class="lede">The network classifies whole images, so finding objects means
  sweeping a grid of overlapping square crops and merging the ones that agree.
  The grid is the entire cost of a scan.</p>

  <p>Window size is a fraction of the frame's <em>short edge</em>, so the count
  depends on aspect ratio and not on resolution at all. A 4:3 photograph is 69
  windows on the <code>full</code> profile whether it is 640x480 or 4000x3000.
  A 16:9 frame is 89. A square one is 49.</p>

  {table(
      ["frame", "pixels", "aspect", "MP", "full", "balanced", "edge"],
      win_rows,
      ["left", "right", "right", "right", "right", "right", "right"],
      "Windows per scan. Computed from the grid, so these hold on any machine.",
  )}

  {figure(
      "Scan latency by profile",
      "resnet18 on CPU, fastest of repeated runs, window count on each bar",
      chart_profiles(),
      swatch("s1", "full") + swatch("s2", "balanced") + swatch("s3", "edge"),
      datatable(
          "Scan figures",
          ["frame", "profile", "windows", "ms", "ms/window", "scans/sec"],
          scan_rows,
          ["left", "left", "right", "right", "right", "right"],
      ),
  )}

  <p>Time per window is nearly constant across every profile and every frame
  size measured - between 52 and 67 milliseconds, with no trend against
  resolution. That gives a usable planning rule:</p>

  {kpis([
      kpi("Cost model", "~57", "ms x windows", "on this CPU, for any frame shape"),
      kpi("full profile", "3.7", "sec", "69 windows on a 4:3 frame"),
      kpi("edge profile", "0.34", "sec", "6 windows, about 3 scans a second"),
      kpi("edge vs full", "10.8", "x faster", "for 11.5x fewer windows"),
  ])}

  <p>The <code>edge</code> profile is close to linear in what it removes, which
  is the honest way to read it: it is not cheaper per window, it simply looks at
  fewer of them, and it trades recall for that.</p>
</section>"""


def sec_runtime():
    rt = {r["runtime"]: r for r in R["runtimes"]["runtimes"]}
    onnx = rt["onnx"]
    thread_rows = [
        [str(m["threads"]), fmt(m["median_ms"], 1),
         f'{m["per_image_ms"]:.1f}', f'{m["speedup_vs_one_thread"]:.2f}x',
         f'{m["parallel_efficiency"] * 100:.1f}%']
        for m in THREADS
    ]

    return f"""
<section id="runtime">
  {eyebrow("Runtimes", "runtimes / threads")}
  <h2>Where the speed actually is</h2>
  <p class="lede">The same weights run through a different runtime, and through
  a different number of cores. One of these is worth far more than the other.</p>

  {kpis([
      kpi("ONNX vs PyTorch", "2.0", "x faster", "batch of 8, identical weights"),
      kpi("Logit agreement", "1.9e-06", "max diff", "float32 rounding, not a different model"),
      kpi("Checkpoint on disk", f'{CK["megabytes"]:.1f}', "MB", "resnet18 with a 6-class head"),
      kpi("Cold start", f'{CK["cold_start"]["median_ms"] / 1000:.2f}', "sec", "build the classifier and load weights"),
  ])}

  <p>Exporting to ONNX is the cheapest real speedup available here: no retraining,
  no accuracy change, and the outputs agree to within float32 rounding. The
  <code>runtimes</code> suite checks that agreement on every run, because a
  backend that is fast and quietly wrong is worse than no backend.</p>

  <p>The export is a single 44.8 MB file - the same size as the checkpoint it
  came from, since both hold the same float32 weights. It was two files until
  the defect below was fixed.</p>

  {figure(
      "Speedup against thread count",
      "resnet18, batch of 8, against perfectly linear scaling",
      chart_threads(),
      swatch("s1", "measured"),
      datatable(
          "Thread figures",
          ["threads", "ms / 8 images", "ms/image", "speedup", "efficiency"],
          thread_rows,
          ["right", "right", "right", "right", "right"],
      ),
  )}

  <p>Cores pay less the more of them there are. Two threads return 1.72x - 86%
  efficient. Six return 3.57x, which is 60%. Extrapolating the shape rather than
  the numbers, <strong>a four-core single-board computer should expect roughly
  2.6x out of its four cores</strong>, and its cores are slower per clock than
  these to begin with.</p>

  <p>Single-threaded, the ResNet takes about 191 ms per image. That figure, not
  the six-thread one, is the right starting point for estimating a small
  board.</p>
</section>"""


def sec_device():
    sync_rows = [
        [t["tier"], str(t["batch_size"]), "yes" if t["thumbnails"] else "no",
         fmt(t["json_bytes"]), fmt(t["gzip_bytes"]),
         f'{t["compression_ratio"]:.2f}x', fmt(t["gzip_bytes_per_capture"])]
        for t in SYNC["tiers"]
    ]
    read_rows = [
        [name.replace("_", " "), f'{value["median_ms"]:.3f}']
        for name, value in STORE["reads"].items()
    ]

    return f"""
<section id="device">
  {eyebrow("On the device", "store / sync")}
  <h2>What a capture costs to keep and to send</h2>
  <p class="lede">A device records locally and uploads when a network appears.
  The full photograph is never kept - only a 640-pixel thumbnail and the boxes -
  so the thumbnail sets both the storage and the bandwidth budget.</p>

  {kpis([
      kpi("Thumbnail", fmt(STORE["thumbnail_bytes_photo_like"]), "bytes", "photo-like frame at 640px, quality 70"),
      kpi("Database row", fmt(STORE["database_bytes_per_capture"]), "bytes", "capture plus two detections"),
      kpi("Captures per GB", fmt(STORE["captures_per_gigabyte"]), "", "thumbnails and database together"),
      kpi("Write rate", f'{STORE["writes_per_second"]:.1f}', "per sec", f'{STORE["ms_per_write"]:.0f} ms each, mostly the resize'),
  ])}

  <p>JPEG size depends entirely on content, so the thumbnail figure is a range
  rather than a constant. The same 1280x960 frame stores as
  <strong>{fmt(STORE["thumbnail_bytes_flat"])} bytes</strong> flat,
  <strong>{fmt(STORE["thumbnail_bytes_photo_like"])}</strong> photo-like, and
  <strong>{fmt(STORE["thumbnail_bytes_noise"])}</strong> as pure noise - a
  spread of 46x. The middle figure is the one to plan with; the noise figure is
  a worst case that no camera produces.</p>

  <h3>Sync tiers</h3>
  <p>Three tiers differ in one thing: whether the thumbnail rides along. That
  one thing is essentially the whole payload.</p>

  {table(
      ["tier", "batch", "images", "JSON bytes", "gzipped", "ratio",
       "bytes/capture"],
      sync_rows,
      ["left", "right", "left", "right", "right", "right", "right"],
      "One batch per tier, each capture a distinct frame.",
  )}

  <p>Metadata alone gzips 7.35x and lands at <strong>57 bytes per
  capture</strong>. With a thumbnail it is <strong>9,310</strong> - a factor of
  <strong>{SYNC["thumbnail_cost_multiple"]:.0f}</strong>, which is the "two
  orders of magnitude" the module docstring claims, now measured. Base64 inflates
  the JPEG by a third and gzip recovers almost exactly that much back, so a
  transported thumbnail costs about what the file on disk costs; gzip cannot
  compress an already-compressed image.</p>

  <p>The <code>low</code> tier is therefore not a small saving. On a narrowband
  link it is the difference between a day's captures fitting and not fitting.</p>

  {datatable(
      "Store read latency",
      ["operation", "median ms"],
      read_rows,
      ["left", "right"],
  )}
</section>"""


def sec_api():
    api_rows = [
        ["GET /v1/health", f'{API["health"]["median_ms"]:.1f}',
         "store counts and uptime"],
        ["GET /v1/model", f'{API["model_info"]["median_ms"]:.1f}',
         "first call builds the model"],
        ["GET /v1/captures?limit=50", f'{API["captures_query"]["median_ms"]:.1f}',
         "50 rows with their detections"],
        ["POST /v1/sync", f'{API["sync"]["latency"]["median_ms"]:.1f}',
         "25 captures, gzipped, metadata only"],
        ["POST /v1/classify (1 image)",
         f'{API["classify_edge_batch_1"]["median_ms"]:.0f}', "edge profile"],
        ["POST /v1/classify (4 images)",
         f'{API["classify_edge_batch_4"]["median_ms"]:.0f}',
         f'{API["classify_edge_batch_4"]["per_image_ms"]:.0f} ms per image'],
    ]

    return f"""
<section id="api">
  {eyebrow("Server", "api")}
  <h2>Endpoint cost, and one correctness result</h2>
  <p class="lede">Measured in-process, so these are the server's own cost with
  no network underneath - the floor a deployment approaches, not what a device
  on a radio link sees.</p>

  {table(
      ["endpoint", "median ms", "note"],
      api_rows,
      ["left", "right", "left"],
  )}

  <p>Everything that does not touch the model answers in single-digit
  milliseconds. Classification is the model, and nothing else: four images cost
  about four times one, so batching a request saves the HTTP round trip and not
  the inference.</p>

  <div class="callout">
    <p><strong>Redelivery is idempotent, and it is now measured rather than
    assumed.</strong> A batch of 25 captures delivered twice reported
    <code>stored: 25, duplicates: 0</code> and then <code>stored: 0,
    duplicates: 25</code>, with all 25 uuids acknowledged both times. That second
    acknowledgement is the part that matters: a device can only clear its outbox
    against one, so a redelivery that went unacknowledged would leave those rows
    being sent forever.</p>
  </div>
</section>"""


def sec_tests():
    rows = []
    for name, data in TESTS["modules"].items():
        subtests = data.get("subtests")
        rows.append([
            f"<code>{name}</code>",
            str(data["passed"]),
            str(data["skipped"]) if data["skipped"] else "-",
            str(subtests) if subtests else "-",
            f'{data["seconds"]:.2f}',
        ])

    totals = R["code"]["totals"]

    return f"""
<section id="tests">
  {eyebrow("Test suite", "pytest")}
  <h2>125 passing, 5 skipped</h2>
  <p class="lede">The suite runs in about 40 seconds and needs no network and no
  dataset.</p>

  {table(
      ["module", "passed", "skipped", "subtests", "seconds"],
      rows,
      ["left", "right", "right", "right", "right"],
  )}

  {kpis([
      kpi("Passing", "125", "", "plus 33 subtests"),
      kpi("Skipped", "5", "", "all of them TFLite"),
      kpi("Application code", fmt(totals["product_code_lines"]), "lines", f'across {totals["product_files"]} modules'),
      kpi("Test code", fmt(totals["test_code_lines"]), "lines", f'{totals["test_to_product_ratio"]:.2f} lines per line of source'),
  ])}

  <p>Every skip is the same cause: producing or running a <code>.tflite</code>
  needs TensorFlow or LiteRT, and <code>ai-edge-torch</code> is Linux-only, so
  that format cannot be built on this machine at all. Nothing is skipped for
  being broken.</p>

  <div class="defect">
    <span class="tag tag-gap">coverage gap</span>
    <h3>Nothing tests that an exported model computes the right thing</h3>
    <p><code>test_backends.py</code> covers ONNX only through extension routing
    and the classes sidecar. No test loads a real <code>.onnx</code>, runs a
    batch through <code>OnnxBackend.run</code>, and compares the result to
    torch's. The export path's correctness is currently unverified by the suite -
    <code>bench.py --suite runtimes</code> is the only thing checking it, and a
    benchmark is not a test.</p>
  </div>
</section>"""


def sec_defects():
    return f"""
<section id="defects">
  {eyebrow("Found while measuring", "export")}
  <h2>Two defects in the export path, both now fixed</h2>
  <p class="lede">Both surfaced from running the tooling rather than reading
  it, and both affected the edge story this document is about. The fixes are in
  <code>model/export.py</code>; the measurements above were re-taken
  afterwards.</p>

  <div class="defect">
    <span class="tag tag-fixed">fixed</span>
    <h3>The ONNX artifact was two files, and the size report saw one</h3>
    <p><code>export.py</code> printed <strong>0.1 MB</strong> for the ONNX
    export. Torch 2.13 writes the weights to an external
    <code>.onnx.data</code> sidecar of 44.7 MB and keeps only the graph -
    97 KB - in the <code>.onnx</code> file, and the script measured just that
    file. Copying the <code>.onnx</code> to a device shipped a model with no
    weights in it, and nothing complained until inference.</p>
    <p>Rather than only correcting the arithmetic, the export now folds the
    sidecar back into the graph file and deletes it, so an export is one
    portable file again. Sizes are computed by a new
    <code>artifact_bytes()</code> that counts any sidecar it does find, so a
    future exporter that reintroduces one cannot re-open the same hole. The
    ONNX export now reports <strong>44.8 MB</strong>, one file, and still
    agrees with torch to 1.9e-06.</p>
  </div>

  <div class="defect">
    <span class="tag tag-fixed">fixed</span>
    <h3>ONNX export crashed on a default Windows console</h3>
    <p>Torch's exporter prints progress lines ending in a check-mark emoji.
    On a console still defaulting to cp1252 that raised
    <code>UnicodeEncodeError</code> part way through a working export, with a
    traceback pointing into an encoder rather than at anything to do with the
    model.</p>
    <p><code>export.py</code> now switches its own streams to UTF-8 with a
    replacing error handler before invoking the exporter, so a cosmetic
    character cannot end the run. Verified by exporting under
    <code>PYTHONIOENCODING=cp1252</code>, which previously failed and now
    completes.</p>
  </div>
</section>"""


def sec_caveats():
    return f"""
<section id="caveats">
  {eyebrow("Limits", "method")}
  <h2>What these numbers are not</h2>

  <p><strong>No accuracy was re-measured.</strong> The
  <code>images/</code> folder holds only <code>ATTRIBUTION.csv</code> on this
  machine - the photographs are not on disk. Every accuracy figure quoted here
  is read from the checkpoint, where <code>train.py</code> recorded it.</p>

  <p>Re-collecting the dataset would not fix that. The manifest lists 633 images
  across the six classes, while the run that produced this checkpoint used 351.
  Downloading the manifest and splitting it with the same seed produces a
  <em>different</em> split, which would place some of the original training
  photographs into the test set and report an accuracy that is too high. A fresh
  confusion matrix needs the exact 351 files, and they are not recoverable from
  what is in the repository.</p>

  <p><strong>Cost, unlike accuracy, does not depend on the pictures.</strong>
  The same convolutions run over the same number of pixels whatever is in them,
  which is why every timing here is measured on generated frames and is
  unaffected by the dataset's absence. Sizes in bytes <em>do</em> depend on
  content, so those were measured on photo-like frames and bracketed with a flat
  and a noise case.</p>

  <p><strong>One machine, one CPU.</strong> Every figure is from the AMD
  six-thread CPU named at the top, with no GPU. They are a baseline and a set of
  ratios, not a prediction for a phone or a single-board computer. The
  <code>threads</code> suite is the closest thing here to an estimate for
  smaller hardware, and it is a shape rather than a number.</p>

  <p><strong>One reading was discarded.</strong> An early run recorded a 1080p
  full-profile scan at 6,988 seconds - about 1.9 hours for work that takes 5.5.
  The machine had suspended mid-run and <code>perf_counter</code> counted
  through it. The harness now flags any sample more than 20x its fastest as a
  suspected stall rather than reporting it; no figure on this page carries that
  flag.</p>
</section>"""


def sec_repro():
    return f"""
<section id="repro">
  {eyebrow("Reproducing", "bench.py")}
  <h2>Re-running any of this</h2>
  <p>Every number on this page comes from <code>statistics/bench.py</code>, added for
  this work, and the committed run it produced is
  <code>statistics/bench_results.json</code>.</p>

  <div class="scroll"><table><tbody>
    <tr><th scope="row"><code>python statistics/bench.py</code></th>
        <td class="a-left">every suite, printed as JSON</td></tr>
    <tr><th scope="row"><code>python statistics/bench.py --suite backbones</code></th>
        <td class="a-left">one suite; repeatable, and the flag stacks</td></tr>
    <tr><th scope="row"><code>python statistics/bench.py --output bench_results.json</code></th>
        <td class="a-left">write the results to a file</td></tr>
    <tr><th scope="row"><code>python -m pytest test -q</code></th>
        <td class="a-left">the test suite</td></tr>
  </tbody></table></div>

  <p>Two more scripts sit beside it and measure nothing themselves, so a
  figure in this page and the same figure in a spreadsheet cannot disagree:
  <code>statistics/report.py</code> rebuilds this page, and
  <code>statistics/make_assets.py</code> writes every figure to
  <code>statistics/graphs/</code> as standalone SVG and every table to
  <code>statistics/tables/</code> as CSV, ready to drop into a document.</p>

  <p>The suites are <code>environment</code>, <code>code</code>,
  <code>checkpoint</code>,
  <code>runtimes</code>, <code>threads</code>, <code>training</code>,
  <code>windows</code>, <code>backbones</code>, <code>profiles</code>,
  <code>store</code>, <code>sync</code> and <code>api</code>.
  <code>environment</code> is always included, because it is what makes the rest
  mean anything.</p>

  <p>Two suites need optional packages. <code>runtimes</code> measures whatever
  is already in <code>model/exported/</code>, so it reports only TorchScript
  until <code>export.py</code> has been run for something else; ONNX figures here
  needed <code>pip install onnx onnxscript onnxruntime</code>. Nothing needs the
  <code>images/</code> folder.</p>
</section>"""


# ------------------------------------------------------------------- assembly

def build():
    machine = (
        f'<span><b>CPU</b> AMD, {ENV["torch_threads"]} threads</span>'
        f'<span><b>OS</b> Windows 11</span>'
        f'<span><b>Python</b> {ENV["python"]}</span>'
        f'<span><b>Torch</b> {ENV["torch"]}</span>'
        f'<span><b>GPU</b> none</span>'
        f'<span><b>Runtimes</b> torch, onnx</span>'
        f'<span><b>Run</b> {R["generated_at"][:10]}</span>'
    )

    body = "".join([
        sec_summary(),
        sec_training(),
        sec_backbones(),
        sec_scan(),
        sec_runtime(),
        sec_device(),
        sec_api(),
        sec_tests(),
        sec_defects(),
        sec_caveats(),
        sec_repro(),
    ])

    return f"""<title>Railway Classifier Benchmarks</title>
<style>{CSS}</style>
<div class="wrap">
  <header class="masthead">
    <div class="stamp">ZRA Labs 2026 &middot; measurement record</div>
    <h1>What the railway classifier costs to run</h1>
    <p class="standfirst">Timing, size and correctness measurements across the
    model, the scan, the device store and the REST layer &mdash; gathered as
    source material for the whitepaper, and reproducible from
    <code>statistics/bench.py</code>.</p>
    <div class="machine">{machine}</div>
  </header>
  {body}
  <footer>
    Generated from <code>statistics/bench_results.json</code>. Every figure is
    measured on one machine; ratios travel further than absolutes.
  </footer>
</div>
"""


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "benchmarks.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
