"""The two test images: a landscape 640x480 (a red circle, a blue square, "HELLO 4172") and a portrait 360x640
(a green triangle, "PELICAN"), drawn with Pillow and DejaVu Sans Bold.
    python make_images.py OUT_DIR"""
import sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
im = Image.new("RGB", (640, 480), "white"); d = ImageDraw.Draw(im)
d.ellipse((40, 60, 240, 260), fill=(220, 30, 30)); d.rectangle((380, 60, 580, 260), fill=(30, 60, 220))
d.text((60, 330), "HELLO 4172", fill="black", font=f); im.save(out + "/landscape.png")
im = Image.new("RGB", (360, 640), "white"); d = ImageDraw.Draw(im)
d.polygon([(180, 60), (320, 320), (40, 320)], fill=(30, 160, 60)); d.text((40, 420), "PELICAN", fill="black", font=f)
im.save(out + "/portrait.png")
