"""REST interface to the classifier and the capture store.

    uvicorn api:app --host 0.0.0.0 --port 8000

Two jobs, and they face opposite directions.

`/v1/classify` is the one the web interface calls: hand it images, get
back what the model found. It exists so the interface does not have to
hold a copy of the model, which is what lets the same front end sit in
front of a workstation or a device out in the field.

`/v1/sync` is the far end of `sync.py`: devices that have been scanning
offline push what they recorded up to whatever is running this. Nothing
in that path re-runs the model - the detections were made on the device,
and this only files them.

Environment:
    ZRA_MODEL         checkpoint or exported model to serve
    ZRA_SCAN_PROFILE  full, balanced or edge
    ZRA_STORE_ROOT    where this instance keeps its captures
"""

import base64
import gzip
import io
import json
import os
import time

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from model.network import CHECKPOINT_PATH, load_architecture, load_metrics
from model.predict import RailwayClassifier
from result import DetectionResult
from store import CaptureStore

MODEL_PATH = Path(os.environ.get("ZRA_MODEL", CHECKPOINT_PATH))

STORE_ROOT = os.environ.get("ZRA_STORE_ROOT")

# a sync batch is metadata plus base64 JPEGs, and a device that has been
# offline for a week can have a lot of both
MAX_SYNC_BYTES = 64 * 1024 * 1024

app = FastAPI(
    title="ZRA Railway Detection",
    version="1.0",
    description=__doc__,
)

STARTED = time.time()

# the model is expensive to build and most requests do not need it, so
# it is left until something actually asks for a classification
_model = None
_store = None


def get_model():
    global _model

    if _model is None:
        _model = RailwayClassifier(MODEL_PATH)

    return _model


def get_store():
    global _store

    if _store is None:
        _store = CaptureStore(**({"root": STORE_ROOT} if STORE_ROOT else {}))

    return _store


def to_results(found):
    """Model output as DetectionResult, strongest first."""
    results = [
        DetectionResult(
            label=label,
            confidence=confidence,
            rect_1=(x1, y1),
            rect_2=(x2, y2)
        )
        for label, confidence, (x1, y1, x2, y2) in found
    ]

    return sorted(results, key=lambda res: -res.confidence)


@app.get("/v1/health")
def health():
    """Whether this instance is up, and what it is holding."""
    store = get_store()

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED, 1),
        "model_loaded": _model is not None,
        "store": store.stats(),
    }


@app.get("/v1/model")
def model_info():
    """What is being served, so a device can check it matches."""
    if not MODEL_PATH.is_file():
        raise HTTPException(503, f"No model at {MODEL_PATH}")

    info = {
        "path": MODEL_PATH.name,
        "runtime": get_model().backend.runtime,
        "classes": get_model().classes,
        "profile": get_model().profile.name,
    }

    # only a torch checkpoint carries its training scores around with it
    if MODEL_PATH.suffix in (".pt", ".pth"):
        info["architecture"] = load_architecture(MODEL_PATH)
        info["metrics"] = load_metrics(MODEL_PATH)

    return info


@app.post("/v1/classify")
async def classify(
    files: list[UploadFile] = File(..., description="one or more images"),
    profile: str | None = Query(None, description="full, balanced or edge"),
    persist: bool = Query(False, description="also record these in the store"),
):
    """Run the model over a batch of images.

    Multiple files per request on purpose - the meeting that asked for
    this asked for it because the interface could only do one at a time.
    """
    model = get_model()

    store = get_store() if persist else None

    results = []

    for upload in files:
        payload = await upload.read()

        try:
            image = Image.open(io.BytesIO(payload))
            image.load()

        except (UnidentifiedImageError, OSError):
            results.append({
                "filename": upload.filename,
                "error": "not a readable image",
                "detections": [],
            })

            continue

        found = to_results(model.detect(image, profile=profile))

        entry = {
            "filename": upload.filename,
            "width": image.width,
            "height": image.height,
            "detections": [res.to_dict() for res in found],
        }

        if store is not None:
            entry["capture_uuid"] = store.record(
                image,
                found,
                source_name=upload.filename,
                model_name=MODEL_PATH.stem,
                model_version=model.backend.runtime,
            )

        results.append(entry)

    return {"count": len(results), "results": results}


async def read_json_body(request):
    """The request body, ungzipped when the sender says it is gzipped.

    Starlette hands over exactly the bytes that arrived. A device on a
    slow link is going to compress, and nothing upstream of here will
    have undone that.
    """
    body = await request.body()

    if len(body) > MAX_SYNC_BYTES:
        raise HTTPException(413, "sync batch too large")

    if request.headers.get("content-encoding", "").lower() == "gzip":
        try:
            body = gzip.decompress(body)

        except (OSError, EOFError) as error:
            raise HTTPException(400, f"could not ungzip body: {error}")

    try:
        return json.loads(body)

    except json.JSONDecodeError as error:
        raise HTTPException(400, f"body is not JSON: {error}")


@app.post("/v1/sync")
async def sync(request: Request):
    """Take a batch of captures from a device that has been offline.

    The reply names every uuid that is now safely stored, including ones
    that were already here. A device can only clear its outbox against
    an acknowledgement, so a redelivery has to be acknowledged too or
    the row would be stuck being sent forever.
    """
    payload = await read_json_body(request)

    captures = payload.get("captures")

    if not isinstance(captures, list):
        raise HTTPException(400, "expected a 'captures' list")

    store = get_store()

    accepted = []
    stored = 0
    duplicates = 0

    for capture in captures:
        if not isinstance(capture, dict) or not capture.get("uuid"):
            continue

        thumbnail = capture.get("thumbnail")

        if thumbnail:
            try:
                thumbnail = base64.b64decode(thumbnail)

            except (ValueError, TypeError):
                thumbnail = None

        if store.accept(capture, thumbnail):
            stored += 1
        else:
            duplicates += 1

        accepted.append(capture["uuid"])

    return {
        "accepted": accepted,
        "stored": stored,
        "duplicates": duplicates,
    }


@app.get("/v1/captures")
def captures(
    limit: int = Query(50, ge=1, le=500),
    sync_state: str | None = Query(None, description="pending or synced"),
):
    """What this instance has on file."""
    return {"captures": get_store().captures(limit=limit, sync_state=sync_state)}


@app.exception_handler(FileNotFoundError)
def missing_model(request, error):
    return JSONResponse({"detail": str(error)}, status_code=503)
