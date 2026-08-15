"""Carry what a device recorded up to a server, over a bad link.

    python sync.py --url http://192.168.1.10:8000
    python sync.py --url http://192.168.1.10:8000 --tier low --once

The device writes captures locally and forgets about the network. This
walks the outbox - the rows `store.py` still has marked pending - and
pushes them in batches, oldest first, clearing each batch only once the
server has said it has them.

Three things make it survivable on a slow link. Batches are sized to the
tier, so a narrowband radio sends ten records rather than five hundred.
The body is gzipped. And a failed batch backs off and retries rather
than dropping, because the alternative on an intermittent link is
losing the run.

Only the standard library, deliberately. This is the half of the system
that runs on the smallest machine, and `requests` is not worth a
dependency for one POST.
"""

import argparse
import base64
import gzip
import json
import random
import time
import urllib.error
import urllib.request

from store import CaptureStore, utc_now

DEFAULT_URL = "http://localhost:8000"

SYNC_PATH = "/v1/sync"

# fields that are this device's own bookkeeping, not the server's business
LOCAL_ONLY = ("sync_state", "synced_at", "thumb_path", "thumb_bytes")


class Tier:
    """How much to send at once, and whether to send pictures at all.

    A detection record is a few hundred bytes and a thumbnail is tens of
    kilobytes, so the presence of the image is the whole difference
    between the tiers - roughly two orders of magnitude.
    """

    def __init__(self, name, batch, thumbnails):
        self.name = name
        self.batch = batch
        self.thumbnails = thumbnails

    def __repr__(self):
        return f"<Tier {self.name}, {self.batch} per batch>"


TIERS = {
    # narrowband. Metadata only: what was seen, where, and when
    "low": Tier("low", batch=10, thumbnails=False),
    # enough for a thumbnail to come along
    "medium": Tier("medium", batch=25, thumbnails=True),
    "high": Tier("high", batch=50, thumbnails=True),
}

DEFAULT_TIER = "medium"

# retry schedule: 1s, 2s, 4s ... plus jitter, so a fleet coming back
# online together does not arrive as one thundering herd
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
MAX_ATTEMPTS = 5


def get_tier(tier):
    if isinstance(tier, Tier):
        return tier

    if tier not in TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}. Pick one of: {', '.join(TIERS)}"
        )

    return TIERS[tier]


def backoff_delay(attempt):
    """Seconds to wait before retry number `attempt`, counting from zero."""
    delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)

    return delay * random.uniform(0.5, 1.5)


class Syncer:
    """Pushes a store's outbox at a server."""

    def __init__(self, url=DEFAULT_URL, store=None, tier=DEFAULT_TIER,
                 timeout=30, attempts=MAX_ATTEMPTS):
        self.url = url.rstrip("/") + SYNC_PATH
        self.store = store or CaptureStore()
        self.tier = get_tier(tier)
        self.timeout = timeout
        self.attempts = attempts

    def build_capture(self, capture):
        """One row, trimmed to what the far end needs."""
        wire = {
            key: value for key, value in capture.items()
            if key not in LOCAL_ONLY
        }

        if not self.tier.thumbnails:
            return wire

        path = self.store.thumbnail_path(capture)

        if path and path.is_file():
            wire["thumbnail"] = base64.b64encode(path.read_bytes()).decode()

        return wire

    def build_payload(self, captures):
        return {
            "device_id": self.store.device_id,
            "sent_at": utc_now(),
            "tier": self.tier.name,
            "captures": [self.build_capture(capture) for capture in captures],
        }

    def post(self, payload):
        """Send one batch, retrying on anything that might be temporary.

        Returns the decoded reply. Raises after the last attempt.
        """
        body = gzip.compress(json.dumps(payload).encode())

        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        last_error = None

        for attempt in range(self.attempts):

            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                    return json.loads(reply.read()), len(body)

            except urllib.error.HTTPError as error:
                # the request itself is wrong, sending it again will not
                # fix it. 429 is the exception - that one means slow down
                if 400 <= error.code < 500 and error.code != 429:
                    raise

                last_error = error

            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error

            if attempt < self.attempts - 1:
                time.sleep(backoff_delay(attempt))

        raise last_error

    def run_once(self):
        """Send one batch. Returns (sent, bytes) - (0, 0) when idle."""
        captures = self.store.pending(limit=self.tier.batch)

        if not captures:
            return 0, 0

        reply, sent_bytes = self.post(self.build_payload(captures))

        accepted = reply.get("accepted", [])

        # only what the server named is cleared. Anything it did not
        # acknowledge stays pending and goes out with the next batch
        self.store.mark_synced(accepted)

        return len(accepted), sent_bytes

    def run(self, max_batches=None):
        """Drain the outbox, or as much of it as `max_batches` allows."""
        total = 0
        total_bytes = 0
        batches = 0

        while max_batches is None or batches < max_batches:
            sent, sent_bytes = self.run_once()

            if not sent:
                break

            total += sent
            total_bytes += sent_bytes
            batches += 1

        return {
            "captures": total,
            "batches": batches,
            "bytes": total_bytes,
            "pending": self.store.stats()["pending"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"server base URL (default {DEFAULT_URL})")
    parser.add_argument("--tier", default=DEFAULT_TIER, choices=sorted(TIERS),
                        help="how much to send per batch")
    parser.add_argument("--once", action="store_true",
                        help="send a single batch instead of draining")
    parser.add_argument("--status", action="store_true",
                        help="show what is waiting, and send nothing")
    parser.add_argument("--timeout", type=float, default=30)

    args = parser.parse_args()

    store = CaptureStore()

    if args.status:
        print(json.dumps(store.stats(), indent=2))

        return

    syncer = Syncer(
        url=args.url,
        store=store,
        tier=args.tier,
        timeout=args.timeout
    )

    try:
        report = syncer.run(max_batches=1 if args.once else None)

    except Exception as error:
        print(f"Sync failed: {type(error).__name__}: {error}")

        raise SystemExit(1)

    print(
        f"Sent {report['captures']} captures in {report['batches']} batches, "
        f"{report['bytes'] / 1000:.1f} kB on the wire, "
        f"{report['pending']} still pending"
    )


if __name__ == "__main__":
    main()
