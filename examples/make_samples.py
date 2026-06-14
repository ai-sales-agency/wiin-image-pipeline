"""Generate synthetic product-like test images.

Why synthetic: real scraped product photos are third-party copyright and reveal
a private catalog, so they cannot ship in a public repo. These generated images
are MIT-clean, reproducible, and deliberately cover the cases that matter:

  clean_white   device on a clean white background  -> recipe should SKIP rembg
  dark_on_dark  near-black device on a dark backdrop -> the case neural rembg
                struggles with (it eats the body or leaves the background)
  textured_bg   device on a noisy colored backdrop  -> recipe should go NEURAL

Run:  python examples/make_samples.py [out_dir]
"""
import sys
import os
from PIL import Image, ImageDraw

S = 1000


def _device(draw, box, body, screen):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=40, fill=body)
    draw.rounded_rectangle((x0 + 30, y0 + 30, x1 - 30, y1 - 30), radius=24, fill=screen)


def clean_white(path):
    im = Image.new("RGB", (S, S), (255, 255, 255))
    _device(ImageDraw.Draw(im), (300, 180, 700, 820), (60, 60, 65), (20, 22, 28))
    im.save(path)


def dark_on_dark(path):
    im = Image.new("RGB", (S, S), (38, 40, 44))           # dark backdrop
    _device(ImageDraw.Draw(im), (300, 180, 700, 820), (18, 18, 22), (10, 10, 14))
    im.save(path)


def textured_bg(path):
    im = Image.new("RGB", (S, S), (120, 140, 170))
    d = ImageDraw.Draw(im)
    for i in range(0, S, 40):                              # simple texture
        d.line((0, i, S, i), fill=(110, 130, 160), width=12)
    _device(d, (300, 180, 700, 820), (200, 200, 205), (30, 34, 44))
    im.save(path)


def with_distractor(path):
    # device plus a second similar object (a retail box) on a non-white backdrop.
    # off-the-shelf saliency removal overcuts here, the white-threshold fallback
    # cannot clear a non-white background, so the background survives. A real
    # failure case (see docs/examples).
    im = Image.new("RGB", (S, S), (150, 150, 155))
    d = ImageDraw.Draw(im)
    _device(d, (120, 350, 430, 720), (60, 60, 65), (20, 22, 28))
    d.rectangle((560, 330, 860, 720), fill=(190, 180, 160))
    d.rectangle((560, 330, 860, 400), fill=(170, 160, 140))
    im.save(path)


def low_contrast(path):
    # device color almost equal to the backdrop: the case neural rembg fails on,
    # it either leaves the background or eats the body
    im = Image.new("RGB", (S, S), (44, 46, 50))
    _device(ImageDraw.Draw(im), (300, 180, 700, 820), (48, 50, 54), (40, 42, 46))
    im.save(path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(out, exist_ok=True)
    clean_white(os.path.join(out, "clean_white.png"))
    dark_on_dark(os.path.join(out, "dark_on_dark.png"))
    textured_bg(os.path.join(out, "textured_bg.png"))
    with_distractor(os.path.join(out, "with_distractor.png"))
    low_contrast(os.path.join(out, "low_contrast.png"))
    print("wrote samples to", out)


if __name__ == "__main__":
    main()
