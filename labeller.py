# this file recieves input (as an image), with coordinates on where to draw the rectangle masks
# it then draws them using cv2's overlay & returns the completed image to the app
# not intended for execution, just houses some of the functions
from result import DetectionResult
from typing import List
from PIL import Image, ImageDraw, ImageFont


def labelImage(res: List[DetectionResult], img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", 18)

    for item in res:
        x1, y1 = item.rect_1
        x2, y2 = item.rect_2
        color = item.get_color()

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=3
        )

        label = f"{item.label} ({item.confidence:.0%})"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle(
            [(x1, max(0, y1 - text_h - 4)),
             (x1 + text_w + 6, y1)],
            fill=color
        )

        draw.text((x1 + 3, max(0, y1 - text_h - 2)), label, fill="black", font=font)
    return img

if __name__ == "__main__":
    raise RuntimeError("This isn't meant to be executed")