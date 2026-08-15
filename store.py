"""Local record of what the model saw, for devices that are offline.

A capture is one image the model was run on. The full-size photograph is
never kept - only a downscaled copy and the boxes - so a device with a
small card can hold weeks of them and still have something to upload
when a network turns up.

    from store import CaptureStore

    store = CaptureStore()
    uuid = store.record(image, detections, source_name="DSC_0001.jpg")

Nothing here imports torch. The store has to run on the same hardware as
the camera, which is the hardware least likely to have a deep learning
stack on it, so it stays on PIL and the standard library. The model that
produced a row is recorded as two strings the caller passes in.
"""

import json
import sqlite3
import threading
import uuid as uuid_module

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from result import DetectionResult

STORE_ROOT = Path(__file__).resolve().parent / "captures"

DATABASE_NAME = "captures.db"
THUMBNAIL_DIRNAME = "thumbs"

# the longest edge of the stored copy. 640 is small enough that a few
# thousand rows fit on a microSD card, and large enough that a person
# reviewing a flagged detection can still see what the box is round
THUMBNAIL_EDGE = 640
THUMBNAIL_QUALITY = 70

# a row is 'pending' until something has carried it to a server
SYNC_PENDING = "pending"
SYNC_SYNCED = "synced"

# GPS tags inside the EXIF GPS IFD, which PIL hands back keyed by number
GPS_IFD = 0x8825
GPS_LATITUDE_REF, GPS_LATITUDE = 1, 2
GPS_LONGITUDE_REF, GPS_LONGITUDE = 3, 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captures (
    id            INTEGER PRIMARY KEY,
    uuid          TEXT    NOT NULL UNIQUE,
    captured_at   TEXT    NOT NULL,
    device_id     TEXT    NOT NULL,
    source_name   TEXT,
    latitude      REAL,
    longitude     REAL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    thumb_path    TEXT    NOT NULL,
    thumb_bytes   INTEGER NOT NULL,
    model_name    TEXT,
    model_version TEXT,
    sync_state    TEXT    NOT NULL DEFAULT 'pending',
    synced_at     TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY,
    capture_id  INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    label       TEXT    NOT NULL,
    description TEXT,
    confidence  REAL    NOT NULL,
    x1          INTEGER NOT NULL,
    y1          INTEGER NOT NULL,
    x2          INTEGER NOT NULL,
    y2          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS captures_sync_state
    ON captures (sync_state, captured_at);

CREATE INDEX IF NOT EXISTS detections_capture
    ON detections (capture_id);
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_degrees(values):
    """One EXIF (degrees, minutes, seconds) triple as a single float."""
    degrees, minutes, seconds = (float(value) for value in values)

    return degrees + minutes / 60 + seconds / 3600


def read_exif_location(image):
    """(latitude, longitude) out of a photo's EXIF, or (None, None).

    A drone or a phone writes its position into the file it takes, so a
    caller that never thinks about GPS still gets it for free. Anything
    unexpected in there is treated as an absent fix rather than an error
    - a missing coordinate is not worth losing the detection over.
    """
    try:
        gps = image.getexif().get_ifd(GPS_IFD)

    except Exception:
        return None, None

    if not gps:
        return None, None

    try:
        latitude = to_degrees(gps[GPS_LATITUDE])
        longitude = to_degrees(gps[GPS_LONGITUDE])

    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None, None

    # the hemisphere lives in its own tag, the numbers are unsigned
    if str(gps.get(GPS_LATITUDE_REF, "N")).upper().startswith("S"):
        latitude = -latitude

    if str(gps.get(GPS_LONGITUDE_REF, "E")).upper().startswith("W"):
        longitude = -longitude

    return latitude, longitude


def make_thumbnail(image, path, edge=THUMBNAIL_EDGE, quality=THUMBNAIL_QUALITY):
    """Write a downscaled JPEG copy, and return how many bytes it took."""
    copy = ImageOps.exif_transpose(image)

    if copy.mode != "RGB":
        copy = copy.convert("RGB")

    # thumbnail() only ever shrinks, so a photo already under the limit
    # is stored at its own size rather than blown up to 640
    copy.thumbnail((edge, edge), Image.LANCZOS)

    path.parent.mkdir(parents=True, exist_ok=True)

    copy.save(path, "JPEG", quality=quality, optimize=True)

    return path.stat().st_size


class CaptureStore:
    """The on-device database of captures and their detections."""

    def __init__(self, root=STORE_ROOT, device_id=None):
        self.root = Path(root)
        self.thumb_root = self.root / THUMBNAIL_DIRNAME

        self.thumb_root.mkdir(parents=True, exist_ok=True)

        # streamlit reruns the script on its own threads, and this object
        # is meant to be cached across those, so the connection is shared
        # deliberately and every write goes through the lock below
        self.connection = sqlite3.connect(
            self.root / DATABASE_NAME,
            check_same_thread=False
        )

        self.connection.row_factory = sqlite3.Row
        self.lock = threading.Lock()

        with self.lock:
            # WAL lets a sync client read while a capture is being written
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")

            self.connection.executescript(SCHEMA)
            self.connection.commit()

        self.device_id = device_id or self.get_or_create_device_id()

    def get_or_create_device_id(self):
        """A stable name for this unit, minted once and kept in the file.

        Telemetry from a fleet is only useful if a row says which unit it
        came from, and hostnames on small boards collide constantly -
        every unflashed Pi is 'raspberrypi'.
        """
        with self.lock:
            row = self.connection.execute(
                "SELECT value FROM meta WHERE key = 'device_id'"
            ).fetchone()

            if row:
                return row["value"]

            device_id = str(uuid_module.uuid4())

            self.connection.execute(
                "INSERT INTO meta (key, value) VALUES ('device_id', ?)",
                (device_id,)
            )

            self.connection.commit()

        return device_id

    def record(self, image, detections, source_name=None, captured_at=None,
               location=None, model_name=None, model_version=None):
        """Store one scanned image and everything found in it.

        `location` is (latitude, longitude) and overrides whatever the
        photo's own EXIF says. Returns the capture's uuid.
        """
        capture_uuid = str(uuid_module.uuid4())

        thumb_path = self.thumb_root / f"{capture_uuid}.jpg"
        thumb_bytes = make_thumbnail(image, thumb_path)

        if location is None:
            latitude, longitude = read_exif_location(image)
        else:
            latitude, longitude = location

        with self.lock:
            cursor = self.connection.execute(
                """
                INSERT INTO captures (
                    uuid, captured_at, device_id, source_name,
                    latitude, longitude, width, height,
                    thumb_path, thumb_bytes,
                    model_name, model_version, sync_state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_uuid,
                    captured_at or utc_now(),
                    self.device_id,
                    source_name,
                    latitude,
                    longitude,
                    image.width,
                    image.height,
                    # relative, so moving the folder does not strand rows
                    str(thumb_path.relative_to(self.root)),
                    thumb_bytes,
                    model_name,
                    model_version,
                    SYNC_PENDING,
                )
            )

            self.connection.executemany(
                """
                INSERT INTO detections (
                    capture_id, label, description, confidence, x1, y1, x2, y2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        cursor.lastrowid,
                        res.label,
                        res.description,
                        res.confidence,
                        res.rect_1[0], res.rect_1[1],
                        res.rect_2[0], res.rect_2[1],
                    )
                    for res in detections
                ]
            )

            self.connection.commit()

        return capture_uuid

    def accept(self, capture, thumbnail=None):
        """Take in a capture that arrived from somewhere else.

        This is the far end of `sync.py`: the device keeps its own row
        and its own uuid, and this side stores that uuid rather than
        minting a new one. Idempotent, because a device that uploads a
        batch and then loses the acknowledgement will send it again, and
        that must not double the rows here.

        Returns True when the capture was new to this store.
        """
        capture_uuid = capture["uuid"]

        thumb_path = ""
        thumb_bytes = 0

        if thumbnail:
            destination = self.thumb_root / f"{capture_uuid}.jpg"

            destination.write_bytes(thumbnail)

            thumb_path = str(destination.relative_to(self.root))
            thumb_bytes = len(thumbnail)

        with self.lock:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO captures (
                    uuid, captured_at, device_id, source_name,
                    latitude, longitude, width, height,
                    thumb_path, thumb_bytes,
                    model_name, model_version, sync_state, synced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_uuid,
                    capture.get("captured_at") or utc_now(),
                    capture.get("device_id") or "unknown",
                    capture.get("source_name"),
                    capture.get("latitude"),
                    capture.get("longitude"),
                    capture.get("width") or 0,
                    capture.get("height") or 0,
                    thumb_path,
                    thumb_bytes,
                    capture.get("model_name"),
                    capture.get("model_version"),
                    # it is already where it was being sent
                    SYNC_SYNCED,
                    utc_now(),
                )
            )

            if not cursor.rowcount:
                self.connection.commit()

                return False

            self.connection.executemany(
                """
                INSERT INTO detections (
                    capture_id, label, description, confidence, x1, y1, x2, y2
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        cursor.lastrowid,
                        hit.get("label", ""),
                        hit.get("description", ""),
                        hit.get("confidence", 0.0),
                        *(hit.get("rect_1") or (0, 0)),
                        *(hit.get("rect_2") or (0, 0)),
                    )
                    for hit in capture.get("detections", [])
                ]
            )

            self.connection.commit()

        return True

    def build_capture(self, row, with_detections=True):
        """One captures row as a plain dict, ready for JSON."""
        capture = dict(row)

        capture.pop("id", None)

        if with_detections:
            capture["detections"] = [
                DetectionResult(
                    label=hit["label"],
                    description=hit["description"] or "",
                    confidence=hit["confidence"],
                    rect_1=(hit["x1"], hit["y1"]),
                    rect_2=(hit["x2"], hit["y2"])
                ).to_dict()
                for hit in self.connection.execute(
                    """
                    SELECT label, description, confidence, x1, y1, x2, y2
                    FROM detections
                    WHERE capture_id = ?
                    ORDER BY confidence DESC
                    """,
                    (row["id"],)
                )
            ]

        return capture

    def captures(self, limit=50, sync_state=None):
        """The most recent captures, newest first."""
        query = "SELECT * FROM captures"
        parameters = []

        if sync_state:
            query += " WHERE sync_state = ?"
            parameters.append(sync_state)

        query += " ORDER BY captured_at DESC, id DESC LIMIT ?"
        parameters.append(limit)

        with self.lock:
            rows = self.connection.execute(query, parameters).fetchall()

            return [self.build_capture(row) for row in rows]

    def pending(self, limit=50):
        """The oldest unsynced captures, which is the order to send them."""
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT * FROM captures
                WHERE sync_state = ?
                ORDER BY captured_at ASC, id ASC
                LIMIT ?
                """,
                (SYNC_PENDING, limit)
            ).fetchall()

            return [self.build_capture(row) for row in rows]

    def get(self, capture_uuid):
        """One capture by uuid, or None."""
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM captures WHERE uuid = ?",
                (capture_uuid,)
            ).fetchone()

            return self.build_capture(row) if row else None

    def thumbnail_path(self, capture):
        """A capture's stored image, or None if it arrived without one.

        A metadata-only sync carries no picture, so a row on the
        receiving end can legitimately have nothing to point at.
        """
        if not capture.get("thumb_path"):
            return None

        return self.root / capture["thumb_path"]

    def mark_synced(self, uuids, synced_at=None):
        """Flag rows a server has acknowledged. Returns how many changed."""
        uuids = list(uuids)

        if not uuids:
            return 0

        with self.lock:
            cursor = self.connection.executemany(
                """
                UPDATE captures
                SET sync_state = ?, synced_at = ?
                WHERE uuid = ? AND sync_state != ?
                """,
                [
                    (SYNC_SYNCED, synced_at or utc_now(), value, SYNC_SYNCED)
                    for value in uuids
                ]
            )

            self.connection.commit()

            return cursor.rowcount

    def stats(self):
        """Counts and disk use, for a status panel or a health endpoint."""
        with self.lock:
            captures, pending, thumb_bytes = self.connection.execute(
                """
                SELECT COUNT(*),
                       SUM(sync_state = ?),
                       COALESCE(SUM(thumb_bytes), 0)
                FROM captures
                """,
                (SYNC_PENDING,)
            ).fetchone()

            detections = self.connection.execute(
                "SELECT COUNT(*) FROM detections"
            ).fetchone()[0]

        return {
            "device_id": self.device_id,
            "captures": captures,
            "pending": pending or 0,
            "detections": detections,
            "thumb_bytes": thumb_bytes,
        }

    def close(self):
        with self.lock:
            self.connection.close()


def main():
    """Print what is on this device, as JSON."""
    store = CaptureStore()

    print(json.dumps(
        {"stats": store.stats(), "captures": store.captures(limit=10)},
        indent=2
    ))


if __name__ == "__main__":
    main()
