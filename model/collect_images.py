# Downloads the dataset from Wikimedia Commons, one folder per class.
# Licenses are recorded in images/ATTRIBUTION.csv. Safe to re-run, it
# skips images already on disk.
#   python collect_images.py --count 30
import argparse
import csv
import hashlib
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

# how augment.py names its generated copies, see dataset.AUG_MARKER
AUG_MARKER = "__aug"

MANIFEST = IMAGE_DIR / "ATTRIBUTION.csv"

# seconds between requests, Commons returns 429 if we go faster
THROTTLE = 1.5

# Several searches per class, for variety. The second group in each list
# asks for the same asset from a different distance, height, hour or
# season. A folder built from the first group alone is largely eye-level
# daylight photographs, and a model trained on that reads an overcast
# dusk shot from a phone held low as a different thing entirely.
QUERIES = {
    "train": [
        "passenger train railway",
        "diesel locomotive railway",
        "electric multiple unit train",
        "freight train railway",
        "train railway aerial view from above",
        "train railway night long exposure",
        "train railway snow winter",
        "high speed train railway line",
        "train approaching close up front",
    ],
    "track": [
        "railway track rails sleepers",
        "railway permanent way ballast",
        "railway points turnout",
        "railroad track curve",
        "railway track aerial view from above",
        "railway track night",
        "railway track snow winter",
        "railway track close up rail head",
        "railway track tunnel bridge",
    ],
    "signal": [
        "railway signal light",
        "semaphore railway signal",
        "colour light railway signal",
        "railway signal gantry",
        "railway signal at night illuminated",
        "railway signal close up detail",
        "ground shunting signal railway",
        "railway signal snow winter",
        "railway signal rear view lineside",
    ],
    "platform": [
        "railway station platform",
        "train platform passengers waiting",
        "island platform railway station",
        "platform edge railway station",
        "railway platform at night",
        "underground metro station platform",
        "rural railway station platform",
        "railway platform canopy roof",
        "railway platform aerial view from above",
    ],
    # Commons is full of digitised trade periodicals that match anything
    # with 'electric railway' in it, so these lean on words that only
    # turn up on modern photographs of the wires themselves. The German
    # and French terms are here because they return photos rather than
    # scans far more reliably than the English ones do.
    "overhead_wire": [
        "railway catenary wires",
        "oberleitung eisenbahn strecke",
        "railway overhead line masts track",
        "catenaire ferroviaire ligne",
        "25 kv overhead line electrification railway",
        "electrified railway contact wire above track",
    ],
    "crossing_gate": [
        "level crossing barrier railway",
        "railroad crossing gate",
        "level crossing boom barrier train",
        "manned level crossing gates railway",
        "level crossing at night lights",
        "level crossing snow winter",
        "level crossing aerial view from above",
        "pedestrian level crossing gate railway",
        "level crossing barrier close up",
    ],
}

# maps, diagrams and logos slip in, drop them by title
REJECT = re.compile(
    r"\b(map|diagram|logo|coat of arms|plan|chart|graph|drawing|"
    r"timetable|poster|ticket|stamp|banner|icon|layout|alignments?)\b",
    re.IGNORECASE
)

# Commons holds a large dump of digitised trade press, and a page of
# 1906 advertising copy matches a railway search as readily as a photo
# does. Two signatures catch nearly all of it: the publication name, and
# the '<year> (<11 digit id>)' filename the Internet Archive upload used.
SCAN_REJECT = re.compile(
    r"\b(journal|review|magazine|library|gazette|treatise|proceedings|"
    r"almanac|catalogue|handbook|encyclop\w*|familjebok|advertisement|"
    r"a book for students|annual report)\b",
    re.IGNORECASE
)

BOOK_SCAN = re.compile(r"\b1[6-9]\d{2}\b.*\(\s*\d{9,}\s*\)")

# False friends, per class. The Liverpool Overhead Railway was an
# elevated line, not an electrified one, and its museum pieces - a cap
# badge, replica seats, a preserved carriage - match 'overhead railway'
# perfectly while showing no overhead wire at all.
CLASS_REJECT = {
    "overhead_wire": re.compile(r"liverpool overhead railway", re.IGNORECASE),
}


def isPhotograph(title: str, category: str = "") -> bool:
    """Whether a Commons title looks like a photo of the real thing."""
    if REJECT.search(title) or SCAN_REJECT.search(title):
        return False

    if BOOK_SCAN.search(title):
        return False

    blocked = CLASS_REJECT.get(category)

    return not (blocked and blocked.search(title))

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Slugs a caller wants left alone - images held out for a benchmark, or
# rejected by a visual audit. Set this before calling collect().
SKIP_SLUGS: set[str] = set()


def slugOf(title: str) -> str:
    """The slug part of the filename a Commons title turns into."""
    stem = Path(title.removeprefix("File:")).stem

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()[:60]

    # A title written in a non-Latin script has no Latin characters to
    # keep and slugs away to nothing, so every Japanese or Cyrillic file
    # in a run lands on '<category>_.jpg' and quietly overwrites the one
    # before it. A digest of the title is stable between runs, which a
    # counter or hash() would not be.
    return slug or f"img-{hashlib.md5(stem.encode()).hexdigest()[:10]}"


def readExcluded(path: Path) -> set[str]:
    """Filenames rejected by hand, one per line, '#' starts a comment.

    Title filters catch scans and false friends, but some rejects need
    eyes on the image - a steam special standing under Swiss catenary is
    a perfectly good photograph that simply is not the class it landed
    in. Without this the next collector run downloads it straight back.
    """
    if not path.is_file():
        return set()

    names = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#")[0].strip()

        if name:
            names.add(name)

    return names


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


def search(query: str, limit: int, category: str = "") -> list[dict]:
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

        if not src or not isPhotograph(title, category):
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
    ext = Path(title.removeprefix("File:")).suffix.lower()

    if ext not in ALLOWED_EXT:
        ext = ".jpg"

    return f"{category}_{slugOf(title)}{ext}"


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
    skipped = 0

    # augment.py writes its copies into these same folders, and counting
    # them here puts every class past any sane target - a run then quietly
    # downloads nothing. --count is a count of photographs.
    existing = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in ALLOWED_EXT and AUG_MARKER not in p.stem
    )

    print(f"\n{category}: {len(existing)} already on disk, want {target}")

    # over-ask, since many results get filtered out
    per_query = max(10, (target // len(queries)) * 3)

    candidates = []

    for query in queries:
        try:
            candidates.extend(search(query, per_query, category))
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

        # held out for a benchmark, or rejected by an audit
        if slugOf(item["title"]) in SKIP_SLUGS:
            skipped += 1
            continue

        dest = folder / makeFilename(category, item["title"])

        if dest.name in SKIP_SLUGS:
            skipped += 1
            continue

        if dest.exists():
            rows.append({**item, "category": category, "file": dest.name})
            continue

        if not download(item["src"], dest):
            continue

        time.sleep(THROTTLE)

        count += 1

        rows.append({**item, "category": category, "file": dest.name})

        print(f"  [{count}/{target}] {dest.name}")

    if skipped:
        print(f"  skipped {skipped} held out or rejected")

    if count < target:
        print(f"  only found {count} usable images for {category}")

    return rows


def readManifest(manifest: Path = MANIFEST) -> list[dict]:
    """Rows already recorded, so a --only run does not drop the rest."""
    if not manifest.is_file():
        return []

    with manifest.open(newline="", encoding="utf-8") as f:
        return [
            {
                "category": row["category"],
                "file": row["file"],
                "descriptionurl": row["source_page"],
                "author": row["author"],
                "license": row["license"],
            }
            for row in csv.DictReader(f)
        ]


def mergeRows(rows: list[dict], manifest: Path = MANIFEST,
              image_dir: Path = IMAGE_DIR) -> list[dict]:
    """This run's rows, laid over whatever was recorded before."""
    merged = {
        (row["category"], row["file"]): row
        for row in readManifest(manifest)
    }

    # a re-download of the same file wins, its metadata is fresher
    for row in rows:
        merged[(row["category"], row["file"])] = row

    # a row whose image has since been deleted is just noise
    return [
        row for row in merged.values()
        if (image_dir / row["category"] / row["file"]).is_file()
    ]


def writeManifest(rows: list[dict], manifest: Path = MANIFEST,
                  image_dir: Path = IMAGE_DIR) -> None:
    rows = mergeRows(rows, manifest, image_dir)

    with manifest.open("w", newline="", encoding="utf-8") as f:
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
