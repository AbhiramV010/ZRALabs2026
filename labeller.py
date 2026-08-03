# draws the detection boxes onto the uploaded image
from result import DetectionResult
from typing import List
from PIL import Image, ImageDraw, ImageFont


def getFont(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labelImage(res: List[DetectionResult], img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # scale the markup with the image, so it reads the same on any upload
    scale = max(img.width, img.height) / 900
    stroke = max(2, round(3 * scale))
    pad = max(4, round(6 * scale))
    font = getFont(max(13, round(17 * scale)))

    for item in res:
        x1, y1 = item.rect_1
        x2, y2 = item.rect_2
        color = item.get_color()

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=stroke
        )

        label = f"{item.label}  {item.confidence:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        chip_h = text_h + pad * 2
        # sit the chip above the box, or inside it when there's no room
        chip_top = y1 - chip_h if y1 - chip_h >= 0 else y1

        draw.rounded_rectangle(
            [(x1, chip_top),
             (x1 + text_w + pad * 2, chip_top + chip_h)],
            radius=max(3, round(4 * scale)),
            fill=color
        )

        draw.text(
            (x1 + pad, chip_top + pad),
            label,
            fill="#0B1120",
            font=font,
            anchor="lt"
        )
    return img

if __name__ == "__main__":
    raise RuntimeError("This isn't meant to be executed")
