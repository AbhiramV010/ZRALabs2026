"""One interface over the runtimes a trained model can be run through.

    from backends import load_backend

    backend = load_backend("railway_classifier.onnx")
    logits = backend.run(batch)          # (N, 3, 224, 224) float32 -> (N, C)

The point is that `predict.py` should not care which of these it got. A
workstation runs the PyTorch checkpoint; a Pi runs the same weights as
ONNX or TFLite; a phone runs them through ExecuTorch. Same numbers in,
same numbers out.

Batches are numpy, not torch tensors, deliberately. Torch is a large
install and the small machines are exactly the ones that should not need
it - each backend imports its own runtime lazily, so having none of the
optional ones installed costs nothing until you ask for one.
"""

import json

from pathlib import Path

import numpy as np

# what each file extension is run by
RUNTIMES = {
    ".pt": "torch",
    ".pth": "torch",
    ".onnx": "onnx",
    ".pte": "executorch",
    ".tflite": "tflite",
}

CLASSES_SUFFIX = ".classes.json"


def classes_path(path):
    """Where the label list for an exported model lives."""
    return Path(path).with_suffix("").with_suffix(CLASSES_SUFFIX)


def read_classes(path):
    """The labels written beside an exported model, if they are there."""
    sidecar = classes_path(path)

    if sidecar.is_file():
        return json.loads(sidecar.read_text())

    return None


def write_classes(path, classes):
    sidecar = classes_path(path)

    sidecar.write_text(json.dumps(classes, indent=2))

    return sidecar


class Backend:
    """A loaded model that can turn a batch of images into logits."""

    runtime = "abstract"

    def __init__(self, path, classes=None):
        self.path = Path(path)
        self.classes = classes

    def run(self, batch):
        raise NotImplementedError

    def close(self):
        pass

    def describe(self):
        return {
            "runtime": self.runtime,
            "path": str(self.path),
            "bytes": self.path.stat().st_size if self.path.is_file() else 0,
            "classes": self.classes,
        }

    def __repr__(self):
        return f"<{type(self).__name__} {self.path.name}>"


class TorchBackend(Backend):
    """The training runtime. Every other backend is measured against it."""

    runtime = "torch"

    def __init__(self, path, classes=None, device=None):
        super().__init__(path, classes)

        import torch

        if __package__ in (None, ""):
            from network import load_checkpoint, pick_device
        else:
            from .network import load_checkpoint, pick_device

        self.torch = torch
        self.device = device or pick_device()

        self.module, checkpoint_classes = load_checkpoint(path, self.device)

        self.classes = classes or checkpoint_classes

    def run(self, batch):
        tensor = self.torch.from_numpy(np.ascontiguousarray(batch))

        with self.torch.no_grad():
            logits = self.module(tensor.to(self.device))

        return logits.detach().cpu().numpy()


class OnnxBackend(Backend):
    """ONNX Runtime. The widest hardware reach of the four."""

    runtime = "onnx"

    def __init__(self, path, classes=None, providers=None):
        super().__init__(path, classes or read_classes(path))

        try:
            import onnxruntime

        except ImportError as error:
            raise ImportError(
                "ONNX Runtime is not installed. pip install onnxruntime"
            ) from error

        self.session = onnxruntime.InferenceSession(
            str(self.path),
            providers=providers or onnxruntime.get_available_providers()
        )

        self.input_name = self.session.get_inputs()[0].name

    def run(self, batch):
        outputs = self.session.run(
            None,
            {self.input_name: np.ascontiguousarray(batch, dtype=np.float32)}
        )

        return outputs[0]

    def close(self):
        self.session = None


class ExecuTorchBackend(Backend):
    """PyTorch's own on-device runtime, reading a .pte program."""

    runtime = "executorch"

    def __init__(self, path, classes=None):
        super().__init__(path, classes or read_classes(path))

        try:
            import torch
            from executorch.runtime import Runtime

        except ImportError as error:
            raise ImportError(
                "ExecuTorch is not installed. pip install executorch"
            ) from error

        self.torch = torch

        program = Runtime.get().load_program(self.path)

        self.method = program.load_method("forward")

    def run(self, batch):
        tensor = self.torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))

        outputs = self.method.execute([tensor])

        return outputs[0].numpy()


class TFLiteBackend(Backend):
    """LiteRT, for boards whose vendor ships a delegate for it."""

    runtime = "tflite"

    def __init__(self, path, classes=None, threads=None):
        super().__init__(path, classes or read_classes(path))

        Interpreter = self.find_interpreter()

        self.interpreter = Interpreter(
            model_path=str(self.path),
            num_threads=threads
        )

        self.interpreter.allocate_tensors()

        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        # a model converted from TensorFlow wants NHWC, one converted
        # straight from PyTorch keeps NCHW. The channel count gives it away
        self.channels_last = self.input_detail["shape"][-1] == 3

    @staticmethod
    def find_interpreter():
        """Whichever of the three names this machine has it under.

        TFLite was renamed LiteRT, and `tf.lite.Interpreter` moved out to
        the `ai_edge_litert` package in TensorFlow 2.16. LiteRT is tried
        first because it is standalone - reaching TensorFlow for an
        interpreter pulls in the whole framework, which is the opposite
        of the point on a small board.
        """
        try:
            from ai_edge_litert.interpreter import Interpreter

            return Interpreter

        except ImportError:
            pass

        # the classic slim runtime, still what the Pi and Coral guides use
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter

        except ImportError:
            pass

        try:
            from tensorflow.lite import Interpreter

            return Interpreter

        except ImportError as error:
            raise ImportError(
                "No LiteRT runtime. pip install ai-edge-litert "
                "(or tflite-runtime on a Pi)"
            ) from error

    def quantize(self, batch, detail):
        """Scale float input into whatever integer type the model wants."""
        scale, zero_point = detail["quantization"]

        if not scale:
            return batch.astype(detail["dtype"])

        return np.round(batch / scale + zero_point).astype(detail["dtype"])

    def dequantize(self, values, detail):
        scale, zero_point = detail["quantization"]

        if not scale:
            return values.astype(np.float32)

        return (values.astype(np.float32) - zero_point) * scale

    def run(self, batch):
        batch = np.ascontiguousarray(batch, dtype=np.float32)

        if self.channels_last:
            batch = batch.transpose(0, 2, 3, 1)

        if self.input_detail["dtype"] != np.float32:
            batch = self.quantize(batch, self.input_detail)

        # tflite interpreters are usually built for a fixed batch of one,
        # so the batch is fed through a frame at a time and stacked back up
        results = []

        for frame in batch:
            self.interpreter.set_tensor(
                self.input_detail["index"],
                frame[np.newaxis, ...]
            )

            self.interpreter.invoke()

            results.append(
                self.interpreter.get_tensor(self.output_detail["index"])[0]
            )

        return self.dequantize(np.stack(results), self.output_detail)


BACKENDS = {
    "torch": TorchBackend,
    "onnx": OnnxBackend,
    "executorch": ExecuTorchBackend,
    "tflite": TFLiteBackend,
}


def load_backend(path, classes=None, runtime=None, **options):
    """Open a model file with whichever runtime reads that extension."""
    path = Path(path)

    runtime = runtime or RUNTIMES.get(path.suffix.lower())

    if runtime is None:
        raise ValueError(
            f"Nothing here reads {path.suffix!r}. "
            f"Known extensions: {', '.join(sorted(RUNTIMES))}"
        )

    if runtime not in BACKENDS:
        raise ValueError(
            f"Unknown runtime {runtime!r}. "
            f"Pick one of: {', '.join(sorted(BACKENDS))}"
        )

    return BACKENDS[runtime](path, classes=classes, **options)


def available_runtimes():
    """Which of the four could actually be used on this machine."""
    found = {}

    for runtime, backend in BACKENDS.items():
        try:
            if runtime == "torch":
                import torch  # noqa: F401

            elif runtime == "onnx":
                import onnxruntime  # noqa: F401

            elif runtime == "executorch":
                from executorch.runtime import Runtime  # noqa: F401

            else:
                backend.find_interpreter()

            found[runtime] = True

        except ImportError:
            found[runtime] = False

    return found


def main():
    for runtime, present in available_runtimes().items():
        print(f"  {runtime:<12} {'available' if present else 'not installed'}")


if __name__ == "__main__":
    main()
