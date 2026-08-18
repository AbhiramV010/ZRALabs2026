"""Turn a trained checkpoint into something an edge device can run.

    python model/export.py --format onnx
    python model/export.py --format all --quantize

Four targets, because the hardware in the brief does not agree on one:

  torchscript  a self-contained .pt, still needs torch. Laptops, servers
  onnx         .onnx, the widest reach - Pi, mini PC, phone, browser
  executorch   .pte, PyTorch's own on-device runtime, XNNPACK lowered
  tflite       .tflite, via ai-edge-torch, for boards with a vendor
               delegate (Coral, some NPUs)

Only the first two are exercised by the tests here; the rest need their
own package installed and are guarded with a message saying so. Each
export writes a `<name>.classes.json` beside it, because none of these
formats carry the label list and a model that cannot name its outputs is
useless on the far end.
"""

import argparse
import json
import sys

from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from backends import write_classes
    from dataset import IMAGE_SIZE, build_loaders
    from network import CHECKPOINT_PATH, load_architecture, load_checkpoint
else:
    from .backends import write_classes
    from .dataset import IMAGE_SIZE, build_loaders
    from .network import CHECKPOINT_PATH, load_architecture, load_checkpoint

EXPORT_DIR = Path(__file__).resolve().parent / "exported"

ONNX_OPSET = 17

# how many validation images to push through a quantiser so it can see
# the range of real activations. More than a few hundred stops helping
CALIBRATION_BATCHES = 8


def example_input(batch=1, size=IMAGE_SIZE):
    """One normalised batch, the shape every exporter traces against."""
    return torch.randn(batch, 3, size, size)


def sidecars(path):
    """Weight files an exporter left beside the graph, if any.

    An ONNX graph can carry its tensors externally, in which case the
    .onnx is a few dozen kB of structure and the weights are in a
    neighbouring .data. Both have to be copied to a device, so both count
    as the artifact.
    """
    path = Path(path)

    candidates = {
        path.parent / f"{path.name}.data",
        path.parent / f"{path.stem}.data",
    }

    return sorted(item for item in candidates if item.is_file())


def artifact_bytes(path):
    """The size of an export, sidecars included."""
    return (Path(path).stat().st_size
            + sum(item.stat().st_size for item in sidecars(path)))


def megabytes(path):
    return artifact_bytes(path) / 1_000_000


def allow_unicode_output():
    """Let the exporter's progress lines print on a legacy console.

    torch's ONNX exporter prints status lines ending in a check-mark
    emoji. On a Windows console still defaulting to cp1252 that raises
    UnicodeEncodeError from inside the exporter's own printing, part way
    through a working export, and the traceback points at an encoder
    rather than at anything to do with the model. A cosmetic character
    should not end the run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if reconfigure is None:
            continue

        try:
            reconfigure(encoding="utf-8", errors="replace")

        except (ValueError, OSError):
            # a stream that will not be reconfigured is not worth failing
            # an export over - the replacing handler below is the backstop
            pass


def export_torchscript(model, path, example=None):
    """Trace the model into a standalone file that still needs torch."""
    example = example_input() if example is None else example

    traced = torch.jit.trace(model, example)

    traced = torch.jit.freeze(traced)

    traced.save(str(path))

    return path


def export_onnx(model, path, example=None):
    """The portable one. Batch stays dynamic so a device can pick its size."""
    # torch has two exporters and they want different packages - the
    # current one needs onnxscript, the legacy one needs onnx, and each
    # complains in its own way. Check here so the answer is one sentence
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401

    except ImportError as error:
        raise ImportError(
            "ONNX export needs its own packages. pip install onnx onnxscript"
        ) from error

    example = example_input() if example is None else example

    allow_unicode_output()

    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["images"],
        output_names=["logits"],
        # a sliding-window scan sends a whole grid of crops at once, and
        # the right grid size is a property of the device, not the model
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=ONNX_OPSET,
    )

    # torch 2.9+ puts the weights in a sidecar and leaves the .onnx
    # holding only the graph - a few dozen kB. Two files is a deployment
    # hazard: copying the .onnx on its own ships a model with no weights
    # in it, and nothing complains until inference. Fold them back into
    # one file, which is what every runtime here expects to be handed.
    extra = sidecars(path)

    if extra:
        onnx.save_model(
            onnx.load(str(path)),
            str(path),
            save_as_external_data=False,
        )

        for item in extra:
            item.unlink()

    return path


def calibration_arrays(batches=CALIBRATION_BATCHES, seed=42):
    """Real validation images, for a quantiser to measure ranges against.

    Quantising against random noise picks activation ranges that no
    photograph produces, and the accuracy drop that follows gets blamed
    on int8 rather than on the calibration.
    """
    _, val_loader, _, _ = build_loaders(batch_size=8, seed=seed)

    arrays = []

    for index, (images, _) in enumerate(val_loader):
        if index >= batches:
            break

        arrays.append(images.numpy())

    return arrays


def export_onnx_int8(float_path, path, arrays=None):
    """Static int8 quantisation of an ONNX model, calibrated on real images.

    Static rather than dynamic: dynamic quantisation only touches linear
    layers, and these backbones are almost entirely convolutions, so it
    would shrink the file by a few percent and speed up nothing.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

    except ImportError as error:
        raise ImportError(
            "int8 export needs ONNX Runtime. pip install onnxruntime onnx"
        ) from error

    arrays = calibration_arrays() if arrays is None else arrays

    class Reader(CalibrationDataReader):

        def __init__(self):
            self.batches = iter(arrays)

        def get_next(self):
            batch = next(self.batches, None)

            return None if batch is None else {"images": batch}

    quantize_static(
        str(float_path),
        str(path),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    return path


def export_executorch(model, path, example=None):
    """Lower to a .pte program, partitioned onto XNNPACK."""
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower

    except ImportError as error:
        raise ImportError(
            "ExecuTorch is not installed. pip install executorch"
        ) from error

    example = example_input() if example is None else example

    exported = torch.export.export(model, (example,))

    program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    Path(path).write_bytes(program.buffer)

    return path


def export_tflite(model, path, example=None):
    """Convert straight from PyTorch with ai-edge-torch.

    Note this keeps the PyTorch NCHW input layout rather than the NHWC a
    TensorFlow-trained model would have. `TFLiteBackend` detects which it
    was handed, so either converts cleanly.
    """
    try:
        import ai_edge_torch

    except ImportError as error:
        raise ImportError(
            "TFLite export needs ai-edge-torch. pip install ai-edge-torch"
        ) from error

    example = example_input() if example is None else example

    converted = ai_edge_torch.convert(model.eval(), (example,))

    converted.export(str(path))

    return path


FORMATS = ("torchscript", "onnx", "executorch", "tflite")


def export_all(checkpoint=CHECKPOINT_PATH, formats=FORMATS, output_dir=EXPORT_DIR,
               quantize=False, stem=None):
    """Write every requested format, and report what each one cost."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # export runs on the CPU whatever trained it - a traced graph should
    # not carry a cuda device around inside it
    model, classes = load_checkpoint(checkpoint, device="cpu")
    model.eval()

    architecture = load_architecture(checkpoint)

    stem = stem or f"railway_classifier_{architecture}"

    example = example_input()

    written = {}
    failed = {}

    for name in formats:
        suffix = {
            "torchscript": ".pt",
            "onnx": ".onnx",
            "executorch": ".pte",
            "tflite": ".tflite",
        }[name]

        path = output_dir / f"{stem}{suffix}"

        try:
            if name == "torchscript":
                export_torchscript(model, path, example)

            elif name == "onnx":
                export_onnx(model, path, example)

            elif name == "executorch":
                export_executorch(model, path, example)

            else:
                export_tflite(model, path, example)

            write_classes(path, classes)

            written[name] = path

        except ImportError as error:
            failed[name] = str(error)

    if quantize and "onnx" in written:
        path = output_dir / f"{stem}_int8.onnx"

        try:
            export_onnx_int8(written["onnx"], path)

            write_classes(path, classes)

            written["onnx_int8"] = path

        except ImportError as error:
            failed["onnx_int8"] = str(error)

    return written, failed, architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--format", default="onnx",
                        choices=(*FORMATS, "all"),
                        help="which format to write (default onnx)")
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--quantize", action="store_true",
                        help="also write an int8 ONNX, calibrated on val images")
    parser.add_argument("--stem", default=None,
                        help="base filename, defaults to the architecture")

    args = parser.parse_args()

    formats = FORMATS if args.format == "all" else (args.format,)

    written, failed, architecture = export_all(
        checkpoint=args.checkpoint,
        formats=formats,
        output_dir=args.output_dir,
        quantize=args.quantize,
        stem=args.stem,
    )

    print(f"\n{architecture}, from {args.checkpoint.name}")
    print(f"  {'source checkpoint':<20}{megabytes(args.checkpoint):>8.1f} MB")

    for name, path in written.items():
        print(f"  {name:<20}{megabytes(path):>8.1f} MB   {path}")

    for name, message in failed.items():
        print(f"  {name:<20}{'skipped':>8}   {message}")

    if not written:
        print("\nNothing was written.")


if __name__ == "__main__":
    main()
