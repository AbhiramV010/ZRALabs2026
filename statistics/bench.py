"""Measure what this project costs to run, and write the numbers out.

    python statistics/bench.py                      # everything
    python statistics/bench.py --suite backbones    # just one
    python statistics/bench.py --output out.json

The README has always been able to say how *accurate* the model is,
because train.py writes that into the checkpoint. What it could not say
is what any of it costs - how long a scan takes, how much a smaller
backbone actually saves, how many bytes a sync puts on the wire. Those
are the numbers the whitepaper needs, and this is where they come from.

Everything here is measured on the machine it runs on, so a figure is
only ever a figure for that machine. `environment` in the output records
which one it was, and nothing should be quoted without it.

Nothing here needs the images/ dataset. Accuracy is not measured - the
checkpoint carries the scores from the run that produced it, and a fresh
accuracy number would need the exact photographs that run was split
from. Cost, unlike accuracy, does not depend on the pictures.
"""

import argparse
import gzip
import json
import platform
import statistics
import sys
import tempfile
import time

from pathlib import Path

import numpy as np
import torch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.backends import available_runtimes                    # noqa: E402
from model.dataset import IMAGE_SIZE                             # noqa: E402
from model.network import (ARCHITECTURES, CHECKPOINT_PATH,       # noqa: E402
                           build_model, count_parameters,
                           last_blocks, read_checkpoint,
                           unfreeze_last_block)
from model.predict import SCAN_PROFILES, RailwayClassifier, count_windows

# frame sizes worth quoting. A phone photograph, a 1080p video frame and
# a drone still - the three things anyone would actually feed this - plus
# a square, because the window grid turns out to key off the shape of a
# frame rather than its size and a third aspect ratio shows that up
RESOLUTIONS = [
    ("VGA", 640, 480),
    ("phone", 1280, 960),
    ("1080p", 1920, 1080),
    ("square", 1000, 1000),
    ("drone still", 4000, 3000),
]

# one forward pass alone is dominated by noise, so each configuration is
# timed repeatedly and the middle is reported. Enough repeats to fill
# roughly this long, within the bounds below
TARGET_SECONDS = 2.5
MIN_REPEATS = 3
MAX_REPEATS = 30

# slowest sample over fastest, past which a run is treated as having been
# interrupted rather than measured. Real jitter on this workload is well
# under 2x; the machine going to sleep mid-run is thousands
STALL_RATIO = 20

BATCH_SIZES = (1, 8, 32)


def timed(call, target=TARGET_SECONDS, min_repeats=MIN_REPEATS,
          max_repeats=MAX_REPEATS):
    """Run something repeatedly and describe how long it took.

    One warm-up call first, and it is not counted. The first pass through
    a torch model allocates workspaces and picks kernels, and it can be
    several times slower than every pass after it - counting it would
    make a fast model look slow in proportion to how little work it does.
    """
    call()

    start = time.perf_counter()
    call()
    single = time.perf_counter() - start

    repeats = int(target / single) if single > 0 else max_repeats
    repeats = max(min_repeats, min(max_repeats, repeats))

    samples = []

    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()

    measured = {
        "repeats": repeats,
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        # the slow tail is what a user actually notices
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 3),
    }

    # perf_counter keeps counting through a suspend, and a long run left
    # alone on a laptop will eat one. A sample orders of magnitude past
    # the fastest one is the machine stopping, not the code being slow,
    # so say so rather than letting it through as a result
    if samples[0] > 0 and samples[-1] / samples[0] > STALL_RATIO:
        measured["stall_suspected"] = True
        measured["stall_ratio"] = round(samples[-1] / samples[0], 1)

    return measured


def noise_image(width, height, seed=0):
    """A random RGB image of a given size.

    Content does not affect timing - the same convolutions run over the
    same number of pixels whatever is in them - so a photograph is not
    needed to measure cost, and the dataset is not on disk anyway.

    It does affect *size*, though. Uniform noise is incompressible, so
    anything measured in bytes off one of these is a worst case and is
    labelled as such. `photo_like_image` is the other end of that.
    """
    generator = np.random.default_rng(seed)

    return Image.fromarray(
        generator.integers(0, 256, (height, width, 3), dtype=np.uint8),
        "RGB"
    )


def photo_like_image(width, height, seed=0):
    """Something a JPEG can compress about as well as it compresses a photo.

    Large smooth regions with a few edges across them and a little grain,
    which is what a photograph looks like to a DCT. Measuring thumbnail
    and payload sizes on uniform noise instead overstates them several
    times over, and those sizes are the whole argument for the sync tiers.
    """
    generator = np.random.default_rng(seed)

    rows = np.linspace(0, 1, height)[:, None]
    columns = np.linspace(0, 1, width)[None, :]

    # a sky-to-ground gradient with some low frequency variation in it
    base = (
        0.55 * rows
        + 0.2 * np.sin(columns * 6.0)
        + 0.15 * np.cos(rows * 9.0)
    )

    field = np.stack([base * 0.9, base * 1.0, base * 1.15], axis=-1)

    # a few hard edges, the way a mast or a rail crosses a frame
    for _ in range(6):
        left = generator.integers(0, max(width - 12, 1))
        field[:, left:left + generator.integers(2, 10), :] *= 0.45

    top = int(height * 0.68)
    field[top:, :, :] *= 0.6

    field += generator.normal(0, 0.02, field.shape)

    return Image.fromarray(
        np.clip(field * 255, 0, 255).astype(np.uint8),
        "RGB"
    )


def environment():
    """What the numbers below are numbers for."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
        "runtimes": available_runtimes(),
    }


def bench_backbones(classes=6):
    """Size and speed of all six architectures, on this CPU.

    Weights are random rather than pretrained. Nothing here reads an
    output, and downloading six ImageNet checkpoints to measure how long
    a matrix multiply takes would be a waste of a network.
    """
    rows = []

    for name in sorted(ARCHITECTURES):
        model = build_model(
            classes,
            architecture=name,
            freeze_backbone=True,
            pretrained=False
        )

        # what phase 1 trains, with the backbone frozen
        head_only = sum(
            parameter.numel()
            for parameter in model.parameters() if parameter.requires_grad
        )

        total = count_parameters(model)

        unfreeze_last_block(model, name)

        # what phase 2 trains, once the last block opens up
        finetune = sum(
            parameter.numel()
            for parameter in model.parameters() if parameter.requires_grad
        )

        model.eval()

        entry = {
            "architecture": name,
            "parameters": total,
            "trainable_head_only": head_only,
            "trainable_finetune": finetune,
            "finetune_fraction": round(finetune / total, 4),
            "float32_mb": round(total * 4 / 1024 ** 2, 2),
            "tuned_blocks": len(last_blocks(model, name)),
            "latency": {},
        }

        with torch.no_grad():

            for batch in BATCH_SIZES:
                tensor = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE)

                result = timed(lambda: model(tensor))

                result["per_image_ms"] = round(result["median_ms"] / batch, 3)
                result["images_per_second"] = round(
                    batch * 1000 / result["median_ms"], 2
                )

                entry["latency"][f"batch_{batch}"] = result

        # what the file would weigh on the device, which is not the same
        # as the parameter count once torch has written its own metadata
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{name}.pt"

            traced = torch.jit.trace(model, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
            traced.save(str(path))

            entry["torchscript_mb"] = round(path.stat().st_size / 1024 ** 2, 2)

        rows.append(entry)

        print(f"  {name:<20} {total:>10,} params  "
              f"{entry['latency']['batch_1']['median_ms']:>7.1f} ms/image")

    baseline = next(row for row in rows if row["architecture"] == "resnet18")

    # the whole point of offering six is the ratio, so state it rather
    # than leaving every reader to divide the columns themselves
    for row in rows:
        row["vs_resnet18_parameters"] = round(
            baseline["parameters"] / row["parameters"], 2
        )
        row["vs_resnet18_latency"] = round(
            baseline["latency"]["batch_1"]["median_ms"]
            / row["latency"]["batch_1"]["median_ms"],
            2
        )

    return {"baseline": "resnet18", "backbones": rows}


def bench_windows():
    """How many crops a scan costs, by profile and frame size.

    Pure arithmetic on the window grid - no model is run - so this is the
    one table here that is true on every machine rather than only this
    one.
    """
    grid = []

    for label, width, height in RESOLUTIONS:
        row = {"resolution": label, "width": width, "height": height}

        for name in SCAN_PROFILES:
            row[name] = count_windows(width, height, name)

        row["megapixels"] = round(width * height / 1e6, 2)
        row["aspect_ratio"] = round(width / height, 3)

        grid.append(row)

    return grid


def bench_profiles(model_path=CHECKPOINT_PATH):
    """End-to-end detect() cost for each profile, on the real checkpoint."""
    if not Path(model_path).is_file():
        return {"error": f"no checkpoint at {model_path}"}

    classifier = RailwayClassifier(model_path)

    scans = []

    # the drone still is left out of the timed runs - it is 12 megapixels
    # and 200-odd windows on the full profile, which is minutes of CPU to
    # measure something the window table already predicts
    for label, width, height in RESOLUTIONS[:3]:
        image = noise_image(width, height)

        for name in sorted(SCAN_PROFILES):
            windows = count_windows(width, height, name)

            result = timed(
                lambda: classifier.detect(image, profile=name),
                target=4.0,
                min_repeats=2,
                max_repeats=6
            )

            # these runs are seconds long each, so only a handful fit in
            # the budget and a median of four is not robust to one bad
            # sample. The fastest run is the honest measure of the work -
            # nothing makes a scan faster than it is, only slower
            result.update({
                "resolution": label,
                "profile": name,
                "windows": windows,
                "ms_per_window": round(result["min_ms"] / windows, 2),
                "frames_per_second": round(1000 / result["min_ms"], 3),
                "headline_ms": result["min_ms"],
            })

            scans.append(result)

            print(f"  {label:<12} {name:<10} {windows:>4} windows  "
                  f"{result['min_ms']:>9.1f} ms"
                  f"{'  (stall seen)' if result.get('stall_suspected') else ''}")

    single = timed(lambda: classifier.classify(noise_image(1280, 960)))

    return {
        "checkpoint": Path(model_path).name,
        "runtime": classifier.backend.runtime,
        "classify_whole_frame": single,
        "scans": scans,
    }


def bench_store(count=200):
    """What a capture costs to write and to read back.

    The interesting number is the thumbnail: a device holding weeks of
    captures on a memory card is limited by these, not by the rows.
    """
    from result import DetectionResult
    from store import CaptureStore

    detections = [
        DetectionResult(
            label="Train",
            confidence=0.91,
            rect_1=(10, 10),
            rect_2=(300, 300)
        ),
        DetectionResult(
            label="Track",
            confidence=0.74,
            rect_1=(0, 200),
            rect_2=(640, 480)
        ),
    ]

    with tempfile.TemporaryDirectory() as folder:
        store = CaptureStore(root=folder)

        # built up front so that generating them is not counted as part
        # of what a write costs. A distinct frame each, so the thumbnail
        # figure is an average over varied content rather than one image
        # measured `count` times
        frames = [photo_like_image(1280, 960, seed=index) for index in range(count)]

        # the other two points on the compressibility scale, because a
        # thumbnail is what fills the card and JPEG size depends entirely
        # on content: noise as the worst case, flat colour as the floor
        noisy = noise_image(1280, 960, seed=1)
        smooth = Image.new("RGB", (1280, 960), (90, 120, 160))

        start = time.perf_counter()

        for index, frame in enumerate(frames):
            store.record(
                frame,
                detections,
                source_name=f"frame_{index:05d}.jpg",
                model_name="railway_classifier",
                model_version="torch"
            )

        elapsed = time.perf_counter() - start

        smooth_uuid = store.record(smooth, detections, source_name="smooth.jpg")
        noisy_uuid = store.record(noisy, detections, source_name="noise.jpg")

        stats = store.stats()

        reads = {
            "stats": timed(store.stats),
            "captures_limit_50": timed(lambda: store.captures(limit=50)),
            "pending_limit_50": timed(lambda: store.pending(limit=50)),
            "get_one_by_uuid": timed(lambda: store.get(smooth_uuid)),
        }

        database = Path(folder) / "captures.db"

        smooth_bytes = store.get(smooth_uuid)["thumb_bytes"]
        noisy_bytes = store.get(noisy_uuid)["thumb_bytes"]
        photo_bytes = round(
            (stats["thumb_bytes"] - smooth_bytes - noisy_bytes) / count
        )

        store.close()

        return {
            "captures_written": count,
            "write_seconds": round(elapsed, 3),
            "writes_per_second": round(count / elapsed, 2),
            "ms_per_write": round(elapsed * 1000 / count, 2),
            "detections_per_capture": len(detections),
            "source_frame": "1280x960",
            "thumbnail_edge": 640,
            "thumbnail_bytes_photo_like": photo_bytes,
            "thumbnail_bytes_noise": noisy_bytes,
            "thumbnail_bytes_flat": smooth_bytes,
            "database_bytes": database.stat().st_size,
            "database_bytes_per_capture": round(
                database.stat().st_size / (count + 2)
            ),
            # what a card actually holds, which is the question a device
            # owner is really asking
            "captures_per_gigabyte": round(
                1024 ** 3 / (photo_bytes + database.stat().st_size / (count + 2))
            ),
            "stats_reported": stats,
            "reads": reads,
        }


def bench_sync(count=50):
    """What a batch weighs on the wire, per tier.

    The tiers differ by one thing - whether a thumbnail rides along - and
    that one thing is most of the payload, so the ratio here is the whole
    argument for having tiers at all.
    """
    from result import DetectionResult
    from store import CaptureStore
    from sync import TIERS, Syncer

    with tempfile.TemporaryDirectory() as folder:
        store = CaptureStore(root=folder)

        detections = [
            DetectionResult(
                label="Signal",
                confidence=0.88,
                rect_1=(40, 40),
                rect_2=(200, 400)
            ),
        ]

        # a different frame per capture, and photo-like rather than noise.
        # Both matter here. Storing one image `count` times lets gzip find
        # the repeats across the batch and reports a per-capture cost that
        # falls as the batch grows, which is an artefact of the fixture
        # rather than anything a device would ever see
        for index in range(count):
            store.record(
                photo_like_image(1280, 960, seed=index),
                detections,
                source_name=f"capture_{index}.jpg"
            )

        rows = []

        for name in sorted(TIERS, key=lambda key: TIERS[key].batch):
            tier = TIERS[name]

            syncer = Syncer(store=store, tier=name)

            captures = store.pending(limit=tier.batch)

            payload = syncer.build_payload(captures)

            raw = json.dumps(payload).encode()
            compressed = gzip.compress(raw)

            rows.append({
                "tier": name,
                "batch_size": tier.batch,
                "thumbnails": tier.thumbnails,
                "captures_in_batch": len(captures),
                "json_bytes": len(raw),
                "gzip_bytes": len(compressed),
                "compression_ratio": round(len(raw) / len(compressed), 2),
                "gzip_bytes_per_capture": round(len(compressed) / len(captures)),
                "build_payload": timed(lambda: syncer.build_payload(captures)),
                "gzip": timed(lambda: gzip.compress(raw)),
            })

            print(f"  {name:<8} {tier.batch:>3} per batch  "
                  f"{len(compressed):>9,} B gzipped  "
                  f"{rows[-1]['gzip_bytes_per_capture']:>7,} B/capture")

        store.close()

        metadata_only = next(row for row in rows if not row["thumbnails"])
        with_thumbnails = next(row for row in rows if row["thumbnails"])

        return {
            "tiers": rows,
            "thumbnail_cost_multiple": round(
                with_thumbnails["gzip_bytes_per_capture"]
                / metadata_only["gzip_bytes_per_capture"],
                1
            ),
        }


def bench_api():
    """Latency of the REST endpoints, in-process.

    A TestClient skips the socket, so these are the server's own cost
    with no network under it - the floor a deployment can approach, not
    what a device on a radio link will see.
    """
    try:
        from fastapi.testclient import TestClient

    # starlette raises RuntimeError rather than ImportError when httpx is
    # the missing half, and httpx is deliberately not a dependency of this
    # project - test_api.py drives a real uvicorn server to avoid it. So
    # this suite is the one thing that wants it, and skips itself politely
    # rather than taking the whole run down with it
    except (ImportError, RuntimeError) as error:
        return {"error": f"needs fastapi and httpx: {error}"}

    import io

    with tempfile.TemporaryDirectory() as folder:
        import os

        os.environ["ZRA_STORE_ROOT"] = folder

        # api reads ZRA_STORE_ROOT at import, so it has to be set first
        for module in ("api",):
            sys.modules.pop(module, None)

        import api

        api.STORE_ROOT = folder
        api._store = None

        client = TestClient(api.app)

        results = {
            "health": timed(lambda: client.get("/v1/health")),
            "model_info": timed(lambda: client.get("/v1/model")),
        }

        buffer = io.BytesIO()
        noise_image(1280, 960, seed=3).save(buffer, "JPEG", quality=85)
        jpeg = buffer.getvalue()

        results["jpeg_bytes"] = len(jpeg)

        # the endpoint takes a list on purpose, so the question is what
        # the second and tenth image in one request cost next to the first
        for batch in (1, 4):

            def call(n=batch):
                return client.post(
                    "/v1/classify?profile=edge",
                    files=[("files", (f"f{i}.jpg", jpeg, "image/jpeg"))
                           for i in range(n)]
                )

            measured = timed(call, target=6.0, min_repeats=2, max_repeats=5)
            measured["per_image_ms"] = round(measured["median_ms"] / batch, 2)

            results[f"classify_edge_batch_{batch}"] = measured

            print(f"  classify batch {batch:<3} {measured['median_ms']:>9.1f} ms  "
                  f"({measured['per_image_ms']:.1f} ms/image)")

        # a sync body the way sync.py sends it: gzipped JSON
        captures = [
            {
                "uuid": f"bench-{index:05d}",
                "captured_at": "2026-08-17T00:00:00+00:00",
                "device_id": "bench",
                "width": 1280,
                "height": 960,
                "detections": [
                    {
                        "label": "Track",
                        "confidence": 0.8,
                        "rect_1": (0, 0),
                        "rect_2": (100, 100),
                    }
                ],
            }
            for index in range(25)
        ]

        body = gzip.compress(json.dumps({"captures": captures}).encode())

        first = client.post(
            "/v1/sync",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        # the same batch again. A device that lost the acknowledgement
        # resends, and the second delivery must not double the rows
        repeat = client.post(
            "/v1/sync",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        results["sync"] = {
            "gzip_bytes": len(body),
            "captures": len(captures),
            "first_delivery": first.json(),
            "redelivery": repeat.json(),
            "idempotent": (
                first.json()["stored"] == len(captures)
                and repeat.json()["stored"] == 0
                and repeat.json()["duplicates"] == len(captures)
            ),
            "latency": timed(
                lambda: client.post(
                    "/v1/sync",
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip",
                    },
                ),
                target=2.0,
                max_repeats=8
            ),
        }

        results["captures_query"] = timed(
            lambda: client.get("/v1/captures?limit=50")
        )

        os.environ.pop("ZRA_STORE_ROOT", None)

        # Windows will not delete a sqlite file that still has a handle
        # on it, and the module holds one in a global until something
        # says otherwise. Closing it here rather than in a finally is
        # deliberate - if the suite failed, the traceback is worth more
        # than the temporary folder is
        if api._store is not None:
            api._store.close()
            api._store = None

        return results


def bench_training():
    """The per-epoch curves, pulled out of the histories train.py wrote.

    Nothing is measured here - it is reshaping what already exists into
    series a plotting library can take directly.
    """
    out = {}

    for key, filename in (
        ("main", "training_history.json"),
        ("wire_specialist", "wire_training_history.json"),
    ):
        path = CHECKPOINT_PATH.parent / filename

        if not path.is_file():
            continue

        data = json.loads(path.read_text())

        history = data.get("history", [])

        series = {}

        for field in ("train_loss", "train_acc", "val_loss", "val_acc",
                      "accuracy", "precision", "recall", "f1"):
            values = [row[field] for row in history if field in row]

            if values:
                series[field] = values

        phases = {}

        for row in history:
            phases.setdefault(row["phase"], []).append(row["epoch"])

        # each phase numbers its own epochs from 1, so a plot against the
        # raw 'epoch' field doubles back on itself halfway along. This is
        # the x axis to actually use
        cumulative = list(range(1, len(history) + 1))

        # where phase 2 starts, for the vertical line on that plot
        boundary = None
        seen = None

        for index, row in enumerate(history):
            if seen is not None and row["phase"] != seen:
                boundary = index + 1
                break

            seen = row["phase"]

        entry = {
            "metrics": data.get("metrics", {}),
            "epochs": len(history),
            "phases": {
                name: {"first_epoch": min(epochs), "last_epoch": max(epochs),
                       "epochs": len(epochs)}
                for name, epochs in phases.items()
            },
            "cumulative_epoch": cumulative,
            "phase_boundary_epoch": boundary,
            "phase_per_epoch": [row["phase"] for row in history],
            "series": series,
        }

        # the gap between the two is the overfitting the README describes,
        # and it is the one number a reader should take from these curves
        if "train_acc" in series and "val_acc" in series:
            best = max(range(len(series["val_acc"])),
                       key=lambda i: series["val_acc"][i])

            entry["best_val_epoch"] = best + 1
            entry["best_val_acc"] = series["val_acc"][best]
            entry["train_acc_at_best_val"] = series["train_acc"][best]
            entry["final_train_val_gap"] = round(
                series["train_acc"][-1] - series["val_acc"][-1], 4
            )
            entry["gap_per_epoch"] = [
                round(train - val, 4)
                for train, val in zip(series["train_acc"], series["val_acc"])
            ]

        out[key] = entry

    return out


def bench_checkpoint(model_path=CHECKPOINT_PATH):
    """What is actually in the file the app loads."""
    path = Path(model_path)

    if not path.is_file():
        return {"error": f"no checkpoint at {path}"}

    checkpoint = read_checkpoint(path)

    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "megabytes": round(path.stat().st_size / 1024 ** 2, 2),
        "architecture": checkpoint.get("architecture"),
        "classes": checkpoint.get("classes"),
        "metrics": checkpoint.get("metrics", {}),
        "load": timed(lambda: read_checkpoint(path), target=2.0, max_repeats=5),
        "cold_start": timed(
            lambda: RailwayClassifier(path),
            target=4.0,
            min_repeats=2,
            max_repeats=4
        ),
    }


def bench_runtimes(model_path=CHECKPOINT_PATH):
    """Torch against every exported runtime present, on speed and on agreement.

    Speed is the reason to export at all. Agreement is the reason to
    check: a backend that is quick and quietly wrong is worse than no
    backend, and nothing else in the project compares their numbers.

    Only formats already sitting in model/exported/ are measured - this
    does not export anything, so what it reports depends on what
    export.py has been run for.
    """
    from model.backends import load_backend

    path = Path(model_path)

    if not path.is_file():
        return {"error": f"no checkpoint at {path}"}

    exported = path.parent / "exported"

    torch_backend = load_backend(path)

    batch = np.asarray(
        np.random.default_rng(0).standard_normal((8, 3, IMAGE_SIZE, IMAGE_SIZE)),
        dtype=np.float32
    )

    reference = torch_backend.run(batch)

    rows = [{
        "runtime": "torch",
        "file": path.name,
        "bytes": path.stat().st_size,
        "max_abs_difference": 0.0,
        "agrees_with_torch": True,
        "latency": {
            f"batch_{size}": timed(
                lambda s=size: torch_backend.run(batch[:s]),
                target=2.0,
                max_repeats=10
            )
            for size in (1, 8)
        },
    }]

    if exported.is_dir():

        for artifact in sorted(exported.iterdir()):

            if artifact.suffix.lower() not in (".onnx", ".pte", ".tflite"):
                continue

            try:
                backend = load_backend(artifact)
                logits = backend.run(batch)

            except Exception as error:
                rows.append({
                    "runtime": artifact.suffix.lstrip("."),
                    "file": artifact.name,
                    "error": f"{type(error).__name__}: {error}",
                })

                continue

            difference = float(np.abs(np.asarray(logits) - reference).max())

            # torch and onnxruntime schedule the same convolutions
            # differently, so the last bits of a float32 will not match.
            # Anything past this is a different computation, not rounding
            agrees = difference < 1e-3

            # a torch export can put its weights in a sidecar rather than
            # in the graph file, and then the graph file alone is 97 kB of
            # nothing. What ships to a device is both, so count both
            sidecars = list(exported.glob(f"{artifact.name}.data")) + \
                list(exported.glob(f"{artifact.stem}.data"))

            rows.append({
                "runtime": backend.runtime,
                "file": artifact.name,
                "bytes": artifact.stat().st_size,
                "sidecar_bytes": sum(item.stat().st_size for item in sidecars),
                "total_bytes": artifact.stat().st_size
                + sum(item.stat().st_size for item in sidecars),
                "sidecar_files": [item.name for item in sidecars],
                "max_abs_difference": difference,
                "agrees_with_torch": agrees,
                "latency": {
                    f"batch_{size}": timed(
                        lambda s=size: backend.run(batch[:s]),
                        target=2.0,
                        max_repeats=10
                    )
                    for size in (1, 8)
                },
            })

            backend.close()

            print(f"  {backend.runtime:<12} {artifact.name:<38} "
                  f"max diff {difference:.2e}  "
                  f"{rows[-1]['latency']['batch_8']['median_ms']:>8.1f} ms/8")

    baseline = rows[0]["latency"]["batch_8"]["median_ms"]

    for row in rows:
        if "latency" in row:
            row["speedup_vs_torch"] = round(
                baseline / row["latency"]["batch_8"]["median_ms"], 2
            )

    return {"reference": "torch", "runtimes": rows}


def bench_threads(architecture="resnet18", classes=6):
    """Latency against core count, which is the edge question restated.

    The machine this runs on has more cores than anything the model is
    headed for. Capping torch to two or four threads is a rough stand-in
    for a small board's CPU - rough because a Pi's core is also slower
    per clock than a desktop's, so this shows the shape of the scaling
    rather than the number a Pi would print.
    """
    original = torch.get_num_threads()

    model = build_model(
        classes,
        architecture=architecture,
        freeze_backbone=False,
        pretrained=False
    )

    model.eval()

    rows = []

    counts = sorted({1, 2, 4, original})

    try:
        with torch.no_grad():

            for threads in counts:

                if threads > original:
                    continue

                torch.set_num_threads(threads)

                tensor = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)

                result = timed(lambda: model(tensor), target=2.0, max_repeats=10)

                result.update({
                    "threads": threads,
                    "per_image_ms": round(result["median_ms"] / 8, 2),
                })

                rows.append(result)

                print(f"  {threads} thread(s)  {result['median_ms']:>9.1f} ms / 8 images")

    finally:
        torch.set_num_threads(original)

    single = next(row for row in rows if row["threads"] == 1)

    for row in rows:
        row["speedup_vs_one_thread"] = round(
            single["median_ms"] / row["median_ms"], 2
        )
        # perfect scaling would put this at 1.0, and it never is
        row["parallel_efficiency"] = round(
            single["median_ms"] / row["median_ms"] / row["threads"], 3
        )

    return {
        "architecture": architecture,
        "batch": 8,
        "host_threads": original,
        "measurements": rows,
    }


def bench_code(root=ROOT):
    """How much code there is, and how much of it is tests.

    Not a measurement of the machine, but the whitepaper wants a figure
    for the size of the thing being described, and counting it here keeps
    it honest - the alternative is a number typed into a document once
    and never true again.
    """
    import ast

    rows = []

    for path in sorted(Path(root).rglob("*.py")):
        relative = path.relative_to(root).as_posix()

        if "__pycache__" in relative or relative.startswith("assets/"):
            continue

        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()

        try:
            tree = ast.parse(source)

        except SyntaxError:
            continue

        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]

        rows.append({
            "file": relative,
            "total": len(lines),
            "code": sum(
                1 for line in lines
                if line.strip() and not line.strip().startswith("#")
            ),
            "comment": sum(1 for line in lines if line.strip().startswith("#")),
            "funcs": len(functions),
            "classes": len(classes),
            "documented": sum(
                1 for node in functions + classes if ast.get_docstring(node)
            ),
        })

    def total(group, field):
        return sum(row[field] for row in group)

    tests = [r for r in rows if r["file"].startswith("test/")]
    tooling = [r for r in rows if r["file"].startswith("statistics/")]
    product = [r for r in rows if r not in tests and r not in tooling]

    return {
        "files": rows,
        "totals": {
            "product_code_lines": total(product, "code"),
            "product_files": len(product),
            "test_code_lines": total(tests, "code"),
            "test_files": len(tests),
            "tooling_code_lines": total(tooling, "code"),
            "test_to_product_ratio": round(
                total(tests, "code") / total(product, "code"), 2
            ),
        },
    }


SUITES = {
    "environment": environment,
    "code": bench_code,
    "checkpoint": bench_checkpoint,
    "runtimes": bench_runtimes,
    "threads": bench_threads,
    "training": bench_training,
    "windows": bench_windows,
    "backbones": bench_backbones,
    "profiles": bench_profiles,
    "store": bench_store,
    "sync": bench_sync,
    "api": bench_api,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--suite", action="append", choices=sorted(SUITES),
                        help="run only this suite, repeatable")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the results here as JSON")

    args = parser.parse_args()

    chosen = args.suite or list(SUITES)

    # the environment is what makes every other number mean anything, so
    # it goes in whether it was asked for or not
    if "environment" not in chosen:
        chosen = ["environment"] + chosen

    results = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    for name in chosen:
        print(f"\n[{name}]")

        start = time.perf_counter()

        try:
            results[name] = SUITES[name]()

        except Exception as error:
            print(f"  failed: {type(error).__name__}: {error}")

            results[name] = {"error": f"{type(error).__name__}: {error}"}

        print(f"  {time.perf_counter() - start:.1f}s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))

        print(f"\nWrote {args.output}")

    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
