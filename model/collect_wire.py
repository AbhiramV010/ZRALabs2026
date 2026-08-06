# Builds images/overhead_wire, the sixth class, from Wikimedia Commons.
#
#   python model/collect_wire.py --count 60
#
# Everything downloaded here is cropped down to the wire before it is
# saved. A catenary photograph is mostly not catenary - the frame it
# arrives in holds track, ballast, a platform, usually a train - and a
# class whose folder is full of track teaches the model that track means
# wire. So the crop keeps the band of sky the contact and messenger
# wires run through, and nothing below it.
#
# Cropping is done by looking for what the wire actually is in an image:
# a thin line, a few pixels across, darker or brighter than the sky
# either side of it. Rows carrying enough of those lines are the wire
# band. The band is then clipped to sit above the horizon, so track that
# happens to be underneath the same wires cannot come with it, and an
# image whose crop still is not mostly sky is thrown away rather than
# saved dirty.
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from collect_images import (
        THROTTLE,
        download,
        makeFilename,
        readExcluded,
        search,
        slugOf,
        writeManifest,
    )
else:
    from .collect_images import (
        THROTTLE,
        download,
        makeFilename,
        readExcluded,
        search,
        slugOf,
        writeManifest,
    )


ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "images"

MANIFEST = IMAGE_DIR / "ATTRIBUTION.csv"

CATEGORY = "overhead_wire"

EXCLUDED_PATH = Path(__file__).resolve().parent / "wire_excluded.txt"

# Commons is full of digitised trade periodicals that match anything
# with 'electric railway' in it, so these lean on words that only turn
# up on modern photographs of the wires themselves. The German, French
# and Italian terms are here because they return photos rather than
# scans far more reliably than the English ones do.
QUERIES = [
    "railway catenary wires",
    "railway overhead line equipment wires",
    "oberleitung fahrdraht eisenbahn",
    "oberleitung eisenbahn strecke",
    "catenaire ferroviaire fil de contact",
    "25 kv overhead line electrification railway",
    "electrified railway contact wire above track",
    "linea aerea contatto ferrovia catenaria",
    "railway overhead line masts wires sky",
    "bahnstromleitung fahrleitung strecke",
    "railway electrification overhead wires station",
    "catenaria ferroviaria linea aerea contacto",
    "bovenleiding spoorweg",
    "kontaktledning jarnvag",
    "trakcja kolejowa sieci",
    "catenary mast insulator railway",
    "fahrleitungsmast bahn oberleitung",
    "overhead contact line railway wires",
    "ligne aerienne contact chemin de fer",
    "catenaria ferrocarril electrificacion",
    "kontaktnetz eisenbahn fahrleitung",
    "railway overhead line tensioning weights",
    "catenary registration arm railway",
    "railway neutral section overhead line",
    "bovenleiding portaal spoor",
    "linha aerea catenaria ferrovia",
    "railway overhead wire span wires",
    "fahrleitung bahnhof gleis oberleitung",
]

# A trolleybus draws off overhead wire too, and its wiring photographs
# almost identically, but the app looks at railways: a road vehicle in
# the positives teaches the model that a city street is a wire scene.
# The audit rejected these one at a time until it was quicker to say so.
ROAD_REJECT = re.compile(
    r"\b(trolleybus|trolejbus|obus|o-bus|bushaltestelle|autobus)\b",
    re.IGNORECASE
)

# --- what counts as sky -------------------------------------------------
#
# Overcast sky is white, clear sky is blue, and neither is ever much
# redder than it is blue. Ballast and rooftops can be as bright as an
# overcast sky, which is why the blue test is here as well as the
# brightness one, and why the horizon is found by walking down from the
# top rather than by testing the frame as a whole.

SKY_VALUE = 85          # a sky pixel is at least this bright, 0-255

SKY_BLUE_BIAS = 12      # ...and no more than this much redder than blue

# ...and it is smooth. Brightness and colour alone let grey ballast,
# concrete and a white station wall through, which is how track ended
# up inside a crop on the first pass. Nothing man-made near a railway
# is as featureless as the sky above it, so local roughness is the test
# that actually separates them.
SKY_TEXTURE = 2.5

SKY_TEXTURE_RADIUS = 6  # ...measured over a square this many pixels wide

# the column a median runs down to wipe the wires out before roughness
# is measured; must be wider than twice the thickest wire
WIRE_SPAN = 11

# A ground row is mostly pixels that are neither sky nor smooth:
# ballast, sleepers, vegetation, brick. The crop is cut off above the
# first run of them. Asking instead where the *sky* stops does not
# work - it needs the top row of the frame to be sky, and a close-up
# of catenary is often a gantry beam straight across the top of the
# picture, which is the class at its most obvious.
GROUND_ROW = 0.5

# a mast, a bridge or a signal gantry crosses the sky without ending it
GROUND_BREAK_ROWS = 0.03  # ...but this many ground rows in a row do

# --- what counts as wire ------------------------------------------------

# line thicknesses probed, in pixels either side of the line's centre
WIRE_WIDTHS = (1, 2, 3, 4)

WIRE_CONTRAST = 7.0     # intensity a line must stand out from the sky by

WIRE_ROW = 0.035        # a wire row carries this much line, across its width

MIN_WIRE_ROWS = 6       # fewer than this is noise, not a run of wire

# Commons answers these queries with a lot of digitised pre-war trade
# press and postcard-era photographs of tramway construction. They crop
# as cleanly as anything - a scan of a wire is still a thin dark line -
# but a modern phone photo is what the app is shown, so they are
# rejected on the one thing a scan cannot fake: mean colour spread.
MIN_CHROMA = 12.0

# --- the crop -----------------------------------------------------------

MIN_CROP_EDGE = 110     # a thinner band than this is not worth keeping

CROP_ASPECT = 1.6       # width:height the band is trimmed towards

CROP_MARGIN = 0.03      # breathing room above and below the wire, of height

# the finished crop is checked again: this much of it must still be sky,
# and this much of it must have wire running through it - a band of
# clean sky above one wire passes every test up to here and teaches the
# model that sky is the class
CROP_PURITY = 0.50

CROP_DENSITY = 0.30

# ...and the lines in it have to go somewhere. A bare winter tree is
# thousands of thin dark lines against a bright sky and reads as wire on
# every test above, but its lines are bunched into the corner it grows
# in. Contact wire is strung between two masts and crosses the frame.
WIRE_EXTENT = 0.55

MAX_EDGE = 1024

# Everything above is measured in pixels - how thick a wire is, how
# wide a smooth patch of sky has to be - so the picture is scaled to a
# known size before any of it is applied. Otherwise the same photograph
# passes or fails on the resolution Commons happened to serve it at.
ANALYSIS_EDGE = 900


def toArrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """An image as (float greyscale, uint8 RGB)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    grey = np.asarray(image.convert("L"), dtype=np.float32)

    return grey, rgb


def chroma(rgb: np.ndarray) -> float:
    """How far from grey the picture is, averaged over every pixel."""
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)

    return float(spread.mean())


def boxMean(values: np.ndarray, radius: int) -> np.ndarray:
    """Local mean over a square window, by integral image."""
    padded = np.pad(values.astype(np.float64), radius, mode="edge")

    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    height, width = values.shape

    span = 2 * radius + 1

    total = (
        integral[span:, span:]
        - integral[:height, span:]
        - integral[span:, :width]
        + integral[:height, :width]
    )

    return total / (span * span)


def deWire(grey: np.ndarray, span: int = WIRE_SPAN) -> np.ndarray:
    """The picture with its thin horizontal lines taken out.

    A median down a short column drops anything narrower than half the
    column, which is every wire in the frame and nothing else. Without
    this the roughness test below counts a dense catenary as texture
    and decides the sky is not sky - which is to say it throws out
    exactly the photographs the class is made of.
    """
    radius = span // 2

    padded = np.pad(grey, ((radius, radius), (0, 0)), mode="edge")

    stack = np.stack([padded[i:i + grey.shape[0]] for i in range(span)])

    return np.median(stack, axis=0)


def texture(grey: np.ndarray) -> np.ndarray:
    """How rough each neighbourhood is: flat on sky, busy on everything else."""
    flat = deWire(grey)

    dx = np.abs(np.diff(flat, axis=1, append=flat[:, -1:]))
    dy = np.abs(np.diff(flat, axis=0, append=flat[-1:, :]))

    return boxMean(dx + dy, SKY_TEXTURE_RADIUS)


def skyMask(rgb: np.ndarray, rough: np.ndarray) -> np.ndarray:
    """Per-pixel: does this look like open sky?"""
    red = rgb[:, :, 0].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)

    bright = rgb.max(axis=2) >= SKY_VALUE

    # white, grey and blue all pass; brick, rust and ballast do not
    cool = blue >= red - SKY_BLUE_BIAS

    # a wire crossing the sky is thin enough to leave the window it
    # sits in flat, so this does not cut the class out of its own frame
    flat = rough < SKY_TEXTURE

    return bright & cool & flat


def groundLimit(mask: np.ndarray, rough: np.ndarray) -> int:
    """The first row, from the top, that the ground has taken over.

    Steelwork is not sky and it is not ground either - it is smooth,
    and a gantry across the top of the frame leaves the rows below it
    perfectly croppable. Ballast, sleepers and undergrowth are the
    things that must not get in, and what they have in common is being
    busy rather than being dark.
    """
    ground = (~mask) & (rough >= SKY_TEXTURE)

    rows = ground.mean(axis=1)

    tolerance = max(4, round(len(rows) * GROUND_BREAK_ROWS))

    running = 0

    for y, fraction in enumerate(rows):
        if fraction < GROUND_ROW:
            running = 0
            continue

        running += 1

        if running >= tolerance:
            return y - running + 1

    return len(rows)


def lineResponse(grey: np.ndarray) -> np.ndarray:
    """How much each pixel looks like a thin horizontal line.

    A wire is a few pixels of one shade with sky either side of it, so
    it shows up as the gap between a pixel and its neighbours some rows
    above and below. Both polarities are measured: contact wire reads
    dark against a bright sky and bright against a dark one, and a
    dataset of only the first would miss every backlit photograph.
    """
    best = np.zeros_like(grey)

    for width in WIRE_WIDTHS:
        centre = grey[width:-width]
        above = grey[: -2 * width]
        below = grey[2 * width:]

        dark = np.minimum(above, below) - centre
        bright = centre - np.maximum(above, below)

        response = np.maximum(dark, bright)

        best[width:-width] = np.maximum(best[width:-width], response)

    return best


def wireRows(grey: np.ndarray, limit: int) -> np.ndarray:
    """Which rows above the horizon carry a run of wire."""
    response = lineResponse(grey)

    carrying = (response > WIRE_CONTRAST).mean(axis=1) >= WIRE_ROW

    # nothing below the horizon is wire, whatever it looks like
    carrying[limit:] = False

    return carrying


def rowExtent(hits: np.ndarray) -> np.ndarray:
    """How much of the width each row's lines are spread across."""
    width = hits.shape[1]

    first = np.argmax(hits, axis=1)

    last = width - 1 - np.argmax(hits[:, ::-1], axis=1)

    return np.where(hits.any(axis=1), (last - first + 1) / width, 0.0)


def bestWindow(weights: np.ndarray, span: int) -> int:
    """Where to start a window of `span` to cover the most weight."""
    if span >= len(weights):
        return 0

    totals = np.cumsum(np.concatenate([[0.0], weights]))

    sums = totals[span:] - totals[:-span]

    return int(np.argmax(sums))


def wireCrop(image: Image.Image) -> tuple[int, int, int, int] | None:
    """The box round the wire in a photograph, or None if there is none."""
    scale = min(1.0, ANALYSIS_EDGE / max(image.width, image.height))

    if scale < 1.0:
        image = image.resize(
            (
                max(round(image.width * scale), 1),
                max(round(image.height * scale), 1)
            ),
            Image.LANCZOS
        )

    grey, rgb = toArrays(image)

    height, width = grey.shape

    if chroma(rgb) < MIN_CHROMA:
        return None

    rough = texture(grey)

    mask = skyMask(rgb, rough)

    limit = groundLimit(mask, rough)

    if limit < MIN_CROP_EDGE:
        return None

    carrying = wireRows(grey, limit)

    if carrying.sum() < MIN_WIRE_ROWS:
        return None

    rows = np.flatnonzero(carrying)

    margin = round(height * CROP_MARGIN)

    top = max(int(rows[0]) - margin, 0)
    bottom = min(int(rows[-1]) + margin, limit)

    # a band thinner than the minimum is grown downwards first, since
    # there is only ever sky above the topmost wire and more of it adds
    # nothing the model can learn from
    if bottom - top < MIN_CROP_EDGE:
        bottom = min(top + MIN_CROP_EDGE, limit)
        top = max(bottom - MIN_CROP_EDGE, 0)

    if bottom - top < MIN_CROP_EDGE:
        return None

    # a full-width band is a letterbox, and the eval transform would
    # centre-crop most of it away, so trim to the stretch of the frame
    # holding the most wire
    span = min(width, round((bottom - top) * CROP_ASPECT))

    span = max(span, MIN_CROP_EDGE)

    response = lineResponse(grey)[top:bottom]

    columns = (response > WIRE_CONTRAST).sum(axis=0).astype(np.float64)

    left = bestWindow(columns, span)

    right = min(left + span, width)

    if mask[top:bottom, left:right].mean() < CROP_PURITY:
        return None

    hits = response[:, left:right] > WIRE_CONTRAST

    window = hits.mean(axis=1)

    carrying = window >= WIRE_ROW

    if carrying.mean() < CROP_DENSITY:
        return None

    if np.median(rowExtent(hits)[carrying]) < WIRE_EXTENT:
        return None

    if scale < 1.0:
        left, top, right, bottom = (round(edge / scale) for edge in
                                    (left, top, right, bottom))

    return left, top, right, bottom


def shrink(image: Image.Image) -> Image.Image:
    longest = max(image.width, image.height)

    if longest <= MAX_EDGE:
        return image

    factor = MAX_EDGE / longest

    return image.resize(
        (round(image.width * factor), round(image.height * factor)),
        Image.LANCZOS
    )


def cropToWire(path: Path) -> bool:
    """Replace a downloaded photo with the wire in it. False if there is none."""
    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")

            box = wireCrop(image)

            if box is None:
                return False

            cropped = shrink(image.crop(box))
    except Exception as err:
        print(f"    unreadable ({err})")
        return False

    cropped.save(path.with_suffix(".jpg"), "JPEG", quality=90)

    # a .png that has just been re-saved as .jpg would otherwise stay
    if path.suffix.lower() != ".jpg":
        path.unlink()

    return True


def collect(target: int, excluded: set[str]) -> list[dict]:
    folder = IMAGE_DIR / CATEGORY
    folder.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "__aug" not in path.stem
    )

    print(f"{CATEGORY}: {len(existing)} already on disk, want {target}")

    # ask for as much as the API will give. Three candidates in four are
    # thrown away by the cropper, so a modest search returns a thin class
    per_query = 50

    candidates = []

    for query in QUERIES:
        try:
            candidates.extend(search(query, per_query, CATEGORY))
        except Exception as err:
            print(f"  search failed for '{query}': {err}")

        time.sleep(THROTTLE)

    rows = []
    seen = set()
    skipped = 0
    uncroppable = 0

    count = len(existing)

    for item in candidates:
        if count >= target:
            break

        if item["title"] in seen:
            continue

        seen.add(item["title"])

        if ROAD_REJECT.search(item["title"]):
            skipped += 1
            continue

        dest = folder / makeFilename(CATEGORY, item["title"])

        # rejected by the visual audit, and not wanted back
        if slugOf(item["title"]) in excluded or dest.name in excluded:
            skipped += 1
            continue

        if dest.exists() or dest.with_suffix(".jpg").exists():
            rows.append({
                **item,
                "category": CATEGORY,
                "file": dest.with_suffix(".jpg").name
                if dest.with_suffix(".jpg").exists() else dest.name,
            })
            continue

        if not download(item["src"], dest):
            continue

        time.sleep(THROTTLE)

        # no wire found in it means no crop that is only wire, and an
        # uncropped frame is exactly what this class must not contain
        if not cropToWire(dest):
            dest.unlink(missing_ok=True)
            uncroppable += 1
            continue

        count += 1

        rows.append({
            **item,
            "category": CATEGORY,
            "file": dest.with_suffix(".jpg").name,
        })

        print(f"  [{count}/{target}] {dest.with_suffix('.jpg').name}")

    if skipped:
        print(f"  skipped {skipped} rejected by the audit")

    if uncroppable:
        print(f"  dropped {uncroppable} with no wire the cropper could find")

    if count < target:
        print(f"  only found {count} usable images for {CATEGORY}")

    return rows


def recrop(folder: Path) -> None:
    """Re-run the cropper over what is already on disk, and stop."""
    kept = 0
    dropped = 0

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        if "__aug" in path.stem:
            continue

        if cropToWire(path):
            kept += 1
        else:
            print(f"  no wire found: {path.name}")
            dropped += 1

    print(f"\n{kept} recropped, {dropped} left alone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--count", type=int, default=60,
                        help="wire images wanted (default 60)")

    parser.add_argument("--recrop", action="store_true",
                        help="re-crop images already on disk and stop")

    args = parser.parse_args()

    folder = IMAGE_DIR / CATEGORY

    if args.recrop:
        if not folder.is_dir():
            parser.error(f"No folder at {folder}")

        recrop(folder)
        return

    excluded = readExcluded(EXCLUDED_PATH)

    rows = collect(args.count, excluded)

    writeManifest(rows, MANIFEST, IMAGE_DIR)

    print(f"\nwrote {MANIFEST}")


if __name__ == "__main__":
    main()
