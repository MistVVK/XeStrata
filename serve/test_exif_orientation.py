# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-License-Identifier: MIT
"""#1229: a phone JPEG stored turned (EXIF Orientation 6 or 8) is turned upright before the vision encoder reads it."""
import io
import unittest
from unittest import mock

from serve.server import PICTURE_FORMATS, Vision

try:
    from PIL import Image
except ImportError:                 # Pillow is optional: the image then goes through as it was
    Image = None


def jpeg(size, orientation=None) -> bytes:
    im = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    out = io.BytesIO()
    im.save(out, format="JPEG", exif=exif)
    return out.getvalue()


@unittest.skipIf(Image is None, "Pillow is not installed")
class ExifOrientation(unittest.TestCase):
    def test_a_turned_jpeg_comes_out_upright_as_a_png(self):
        for tag in (6, 8):
            out = Vision.normalize(jpeg((40, 20), tag))
            self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(Image.open(io.BytesIO(out)).size, (20, 40))

    def test_a_jpeg_without_the_tag_or_with_tag_1_passes_through_unchanged(self):
        for tag in (None, 1):
            data = jpeg((40, 20), tag)
            self.assertIs(Vision.normalize(data), data)

    def test_png_passes_through(self):
        out = io.BytesIO()
        Image.new("RGB", (4, 4)).save(out, format="PNG")
        self.assertEqual(Vision.normalize(out.getvalue()), out.getvalue())

    def test_a_broken_jpeg_is_left_as_it_was_sent(self):
        data = b"\xff\xd8\xff" + b"not a jpeg"
        self.assertEqual(Vision.normalize(data), data)


if __name__ == "__main__":
    unittest.main()


class DecoderAllowlist(unittest.TestCase):
    """upstream #4220 / GHSA-78g4-xqwv-q3g8: the picture reader opens only the formats it documents, so a crafted JPEG
    2000 file never reaches OpenJPEG (Pillow picks the decoder by the file's first bytes)."""

    @staticmethod
    def sample(fmt) -> bytes:
        out = io.BytesIO()
        Image.new("RGB", (16, 8), (10, 120, 200)).save(out, format=fmt)
        return out.getvalue()

    def test_a_jpeg_2000_picture_is_refused_before_any_decoder_runs(self):
        try:
            data = self.sample("JPEG2000")
        except (KeyError, OSError, ValueError):
            self.skipTest("this Pillow cannot write JPEG 2000")
        with mock.patch.object(Image.Image, "load", side_effect=AssertionError("the JPEG 2000 decoder ran")):
            with self.assertRaises(ValueError) as caught:
                Vision.normalize(data)
        self.assertIn("could not be read", str(caught.exception))

    def test_the_documented_formats_still_become_a_png(self):
        for fmt in ("WEBP", "TIFF", "GIF", "PNG"):
            with self.subTest(fmt):
                self.assertEqual(Image.open(io.BytesIO(Vision.normalize(self.sample(fmt)))).size, (16, 8))

    def test_the_conversion_passes_the_list(self):
        seen = []
        real = Image.open

        def spy(*a, **kw):
            seen.append(kw.get("formats"))
            return real(*a, **kw)

        with mock.patch.object(Image, "open", spy):
            Vision.normalize(self.sample("WEBP"))
        self.assertTrue(seen)
        self.assertTrue(all(f == PICTURE_FORMATS for f in seen), seen)
