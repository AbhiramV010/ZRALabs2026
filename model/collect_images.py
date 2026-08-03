# Downloads the dataset from Wikimedia Commons, one folder per class.
# Licenses are recorded in images/ATTRIBUTION.csv. Safe to re-run, it
# skips images already on disk.
#   python collect_images.py --count 30
import argparse
import csv
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"

# a contact is required by the Wikimedia API etiquette policy
USER_AGENT = (
    "ZRALabs2026-dataset/1.0 (student project; abhiram.vadali@hotmail.com)"
)

IMAGE_DIR = Path("images")

MANIFEST = IMAGE_DIR / "ATTRIBUTION.csv"

# seconds between requests, Commons returns 429 if we go faster
THROTTLE = 1.5

# several searches per class, for variety
QUERIES = {
    "train": [
        "passenger train railway",
        "diesel locomotive railway",
        "electric multiple unit train",
        "freight train railway",
    ],
    "track": [
        "railway track rails sleepers",
        "railway permanent way ballast",
        "railway points turnout",
        "railroad track curve",
    ],
    "signal": [
        "railway signal light",
        "semaphore railway signal",
        "colour light railway signal",
        "railway signal gantry",
    ],
    "platform": [
        "railway station platform",
        "train platform passengers waiting",
        "island platform railway station",
        "platform edge railway station",
    ],
    "overhead_wire": [
        "railway overhead line equipment",
        "railway catenary wires",
        "overhead wire electrification railway mast",
        "pantograph overhead line railway",
    ],
    "crossing_gate": [
        "level crossing barrier railway",
        "railroad crossing gate",
        "level crossing boom barrier train",
        "manned level crossing gates railway",
    ],
}

# maps, diagrams and logos slip in, drop them by title
REJECT = re.compile(
    r"\b(map|diagram|logo|coat of arms|plan|chart|graph|drawing|"
    r"timetable|poster|ticket|stamp|banner|icon)\b",
    re.IGNORECASE
)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def apiGet(params: dict) -> dict:
    """One call to the Commons API, returned as parsed JSON."""
    params = {**params, "format": "json", "formatversion": "2"}

    url = f"{API}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def stripHtml(value: str) -> str:
    """Commons returns author/license fields as small HTML snippets."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())


def search(query: str, limit: int) -> list[dict]:
    """Photos matching a query, best search-rank first."""
    data = apiGet({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",          # namespace 6 is File:
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": "1024",
    })

    pages = data.get("query", {}).get("pages", [])

    results = []

    for page in pages:
        title = page.get("title", "")

        info = (page.get("imageinfo") or [{}])[0]

        src = info.get("thumburl") or info.get("url")

        if not src or REJECT.search(title):
            continue

        meta = info.get("extmetadata", {})

        results.append({
            "title": title,
            "src": src,
            "descriptionurl": info.get("descriptionurl", ""),
            "author": stripHtml(meta.get("Artist", {}).get("value", "")),
            "license": stripHtml(
                meta.get("LicenseShortName", {}).get("value", "")
            ),
        })

    return results


def makeFilename(category: str, title: str) -> str:
    """A short, safe filename derived from the Commons file name."""
    name = title.removeprefix("File:")

    stem = Path(name).stem

    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_EXT:
        ext = ".jpg"

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()

    return f"{category}_{slug[:60]}{ext}"


def download(url: str, dest: Path, attempts: int = 4) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    data = None

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            break
        except urllib.error.HTTPError as err:
            # 429 means too fast, back off and try again
            if err.code == 429 and attempt < attempts - 1:
                time.sleep(THROTTLE * 4 * (attempt + 1))
                continue
            print(f"    skipped ({err})")
            return False
        except Exception as err:
            print(f"    skipped ({err})")
            return False

    if data is None:
        return False

    if len(data) < 8_000:          # tiny files are placeholders
        return False

    dest.write_bytes(data)
    return True


def collect(category: str, queries: list[str], target: int) -> list[dict]:
    folder = IMAGE_DIR / category
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    existing = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in ALLOWED_EXT
    )

    print(f"\n{category}: {len(existing)} already on disk, want {target}")

    # over-ask, since many results get filtered out
    per_query = max(10, (target // len(queries)) * 3)

    candidates = []

    for query in queries:
        try:
            candidates.extend(search(query, per_query))
        except Exception as err:
            print(f"  search failed for '{query}': {err}")

        time.sleep(THROTTLE)

    count = len(existing)

    for item in candidates:
        if count >= target:
            break

        if item["title"] in seen:
            continue

        seen.add(item["title"])

        dest = folder / makeFilename(category, item["title"])

        if dest.exists():
            rows.append({**item, "category": category, "file": dest.name})
            continue

        if not download(item["src"], dest):
            continue

        time.sleep(THROTTLE)

        count += 1

        rows.append({**item, "category": category, "file": dest.name})

        print(f"  [{count}/{target}] {dest.name}")

    if count < target:
        print(f"  only found {count} usable images for {category}")

    return rows


def writeManifest(rows: list[dict]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["category", "file", "source_page", "author", "license"]
        )

        for row in sorted(rows, key=lambda r: (r["category"], r["file"])):
            writer.writerow([
                row["category"],
                row["file"],
                row["descriptionurl"],
                row["author"],
                row["license"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--count",
        type=int,
        default=45,
        help="images per category"
    )

    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(QUERIES),
        help="limit to specific categories"
    )

    args = parser.parse_args()

    wanted = args.only or sorted(QUERIES)

    rows = []

    for category in wanted:
        rows.extend(collect(category, QUERIES[category], args.count))

    writeManifest(rows)

    print(f"\nwrote {MANIFEST} ({len(rows)} images)")


if __name__ == "__main__":
    main()
