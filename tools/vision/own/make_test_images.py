#!/usr/bin/env python3
"""Makes the picture test set of the own vision tower (Pillow only, deterministic).

    python tools/vision/own/make_test_images.py <out dir>

The three pictures of tools/qwen35/pictures are copied as they are; the rest is drawn: small, odd aspect ratios, sizes that are not multiples of
32, a very large one (the resize path), the same picture as JPEG / BMP / GIF, grayscale, RGBA and palette PNGs, and a photo-like one (gradients +
noise).  Nothing here is large enough to matter in the repo, and nothing is committed: the folder is made when a check runs.
"""
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PICS = os.path.join(HERE, "..", "..", "qwen35", "pictures")


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def scene(w, h, text, seed=0):
    """A canvas with a few coloured shapes and a line of text, scaled to the size (so small and large pictures show the same things)."""
    im = Image.new("RGB", (w, h), (245, 245, 235))
    d = ImageDraw.Draw(im)
    m = min(w, h)
    cols = [(220, 40, 40), (40, 90, 220), (30, 160, 70), (240, 190, 20)]
    for i, c in enumerate(cols):
        x0 = int(w * (0.08 + 0.22 * i))
        y0 = int(h * (0.15 + 0.1 * (i % 2)))
        r = max(2, int(m * 0.16))
        if i % 2 == 0:
            d.ellipse([x0, y0, x0 + r, y0 + r], fill=c)
        else:
            d.rectangle([x0, y0, x0 + r, y0 + r], fill=c)
    d.text((int(w * 0.06), int(h * 0.62)), text, fill=(20, 20, 20), font=font(max(8, int(m * 0.18))))
    return im


def photo(w, h):
    """Gradients plus noise, blurred a little: a photo-like picture with no flat areas."""
    base = Image.merge("RGB", [Image.linear_gradient("L").resize((w, h)),
                               Image.linear_gradient("L").rotate(90).resize((w, h)),
                               Image.linear_gradient("L").rotate(45, expand=False).resize((w, h))])
    noise = Image.effect_noise((w, h), 40).convert("RGB")
    out = Image.blend(base, noise, 0.25).filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(out)
    d.ellipse([w * 0.3, h * 0.3, w * 0.7, h * 0.7], outline=(255, 255, 255), width=max(2, w // 100))
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for n in ("shapes.png", "code4217.png", "big1024.png"):
        shutil.copyfile(os.path.join(PICS, n), os.path.join(out, n))
    shapes = Image.open(os.path.join(PICS, "shapes.png")).convert("RGB")
    save = lambda im, n, **kw: im.save(os.path.join(out, n), **kw)  # noqa: E731
    save(scene(64, 48, "Hi"), "tiny_64x48.png")
    save(scene(90, 1000, "TALL"), "tall_90x1000.png")
    save(scene(1000, 90, "WIDE strip"), "wide_1000x90.png")
    save(scene(33, 517, "odd"), "odd_33x517.png")
    save(scene(999, 777, "999 x 777"), "odd_999x777.png")
    save(scene(333, 333, "333"), "odd_333x333.png")
    save(photo(1600, 1200), "photo_1600x1200.jpg", quality=90)
    save(scene(4000, 3000, "HUGE 4000x3000"), "huge_4000x3000.png")
    save(scene(8000, 500, "HUGE 8000x500"), "huge_8000x500.png")
    save(shapes, "shapes.jpg", quality=90)
    save(shapes, "shapes.bmp")
    save(shapes.convert("P", palette=Image.ADAPTIVE, colors=64), "shapes_p.gif")
    save(shapes.convert("L"), "shapes_gray.png")
    rgba = shapes.convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize(shapes.size))
    save(rgba, "shapes_rgba.png")
    save(shapes.convert("P", palette=Image.ADAPTIVE, colors=32), "shapes_pal.png")
    print("wrote %d pictures to %s" % (len(os.listdir(out)), out))


if __name__ == "__main__":
    main()
