"""Regenerate the icon set from the Railway Academy logo.

    python assets/make_icons.py
"""

import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

SOURCE_URL = "https://railwayacademy.org/wp-content/uploads/2025/11/FAVICON.png"

OUT = Path(__file__).resolve().parent

PNG_SIZES = [16, 32, 48, 64, 128, 180, 192, 256, 512]

# a .ico holds several sizes, the browser picks one
ICO_SIZES = [16, 32, 48, 64, 128, 256]


def fetch():
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request) as response:
        return Image.open(BytesIO(response.read())).convert("RGBA")


def square(img):
    """Pad to a square so nothing is stretched on the way down."""
    side = max(img.size)

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))

    canvas.paste(
        img,
        ((side - img.width) // 2, (side - img.height) // 2)
    )

    return canvas


def main():
    logo = square(fetch())

    logo.save(OUT / "logo.png")

    for size in PNG_SIZES:
        # LANCZOS, cheaper filters turn the thin train outline to mush
        logo.resize((size, size), Image.LANCZOS).save(OUT / f"icon-{size}.png")

    logo.resize((180, 180), Image.LANCZOS).save(OUT / "apple-touch-icon.png")

    logo.save(
        OUT / "favicon.ico",
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES]
    )

    print(f"wrote {len(PNG_SIZES) + 3} files to {OUT}")


if __name__ == "__main__":
    main()
