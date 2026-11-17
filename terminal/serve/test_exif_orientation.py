"""#1229: a phone JPEG stored turned (EXIF Orientation 6 or 8) is turned upright before the vision encoder reads it."""
import io
import unittest
from unittest import mock

from serve.server import Vision

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


@unittest.skipIf(Image is None, "Pillow is not installed")
class DecoderAllowlist(unittest.TestCase):
    """GHSA-78g4-xqwv-q3g8 (#4220): a crafted JPEG 2000 reached OpenJPEG's heap overflow through Image.open().load(), because
    Pillow picks the decoder by the file's first bytes.  Only the formats the picture reader documents are opened."""

    @staticmethod
    def sample(fmt, **kw) -> bytes:
        out = io.BytesIO()
        Image.new("RGB", (16, 8), (10, 120, 200)).save(out, format=fmt, **kw)
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

    def test_other_formats_outside_the_list_are_refused_too(self):
        for fmt in ("PPM", "ICO", "TGA"):
            with self.subTest(fmt):
                with self.assertRaises(ValueError):
                    Vision.normalize(self.sample(fmt))

    def test_the_documented_formats_still_become_a_png(self):
        for fmt, kw in (("WEBP", {}), ("TIFF", {}), ("GIF", {}), ("PNG", {})):
            with self.subTest(fmt):
                data = self.sample(fmt, **kw)
                out = Vision.normalize(data)
                self.assertEqual(Image.open(io.BytesIO(out)).size, (16, 8))

    def test_a_phone_mpo_file_is_still_read_and_turned(self):
        a, b = Image.new("RGB", (40, 20), (200, 30, 30)), Image.new("RGB", (40, 20), (30, 200, 30))
        exif = Image.Exif()
        exif[0x0112] = 6
        out = io.BytesIO()
        a.save(out, format="MPO", save_all=True, append_images=[b], exif=exif)
        self.assertEqual(Image.open(io.BytesIO(out.getvalue())).format, "MPO")
        self.assertEqual(Image.open(io.BytesIO(Vision.normalize(out.getvalue()))).size, (20, 40))

    def test_the_exif_probe_and_both_decoders_pass_the_list(self):
        seen = []
        real = Image.open

        def spy(*a, **kw):
            seen.append(kw.get("formats"))
            return real(*a, **kw)

        data = self.sample("WEBP")
        with mock.patch.object(Image, "open", spy):
            Vision.normalize(data)
            Vision._exif_rotated(jpeg((8, 8), 6))
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(f and "JPEG2000" not in f and "JPEG" in f for f in seen), seen)


if __name__ == "__main__":
    unittest.main()
