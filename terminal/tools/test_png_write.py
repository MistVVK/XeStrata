"""tools/test_png_write.py - our PNG writer (src/image/png/png_write.cpp): CRC-32 / Adler-32 against published check
values; PNG files of several shapes (grey, RGB, RGBA; one row, many stored deflate blocks) decoded back by an
independent reader (Python's zlib, every chunk CRC checked, plus Pillow when it is installed) to the same pixels; the
"parameters" tEXt chunk kept right after IHDR as stable-diffusion.cpp's stb_image_write writes it.  No GPU.

    python -m pytest tools/test_png_write.py"""
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


def read_png(data: bytes):
    """chunks (type, payload), checking each CRC; the decoded rows (filter 0 only) as bytes"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, chunks = 8, []
    while at < len(data):
        n, = struct.unpack(">I", data[at:at + 4])
        typ, body = data[at + 4:at + 8], data[at + 8:at + 8 + n]
        crc, = struct.unpack(">I", data[at + 8 + n:at + 12 + n])
        assert crc == zlib.crc32(typ + body), typ
        chunks.append((typ, body))
        at += 12 + n
    w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", chunks[0][1])
    ch = {0: 1, 4: 2, 2: 3, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(b for t, b in chunks if t == b"IDAT"))
    row = w * ch
    assert len(raw) == h * (row + 1)
    pixels = bytearray()
    for y in range(h):
        assert raw[y * (row + 1)] == 0
        pixels += raw[y * (row + 1) + 1:(y + 1) * (row + 1)]
    return chunks, (w, h, ch, depth, comp, filt, inter), bytes(pixels)


def pattern(w, h, c):
    return bytes(((7 * x + 13 * y + 29 * k) & 255) for y in range(h) for x in range(w) for k in range(c))


@unittest.skipUnless(CXX, "no C++ compiler")
class PngWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.td.name) / "strata-png-write"
        r = subprocess.run([CXX, "-std=c++17", "-O2", "-I", str(ROOT.parent / "engine" / "include"),
                            str(ROOT.parent / "engine" / "src" / "image" / "png" / "png_write.cpp"),
                            str(ROOT.parent / "engine" / "src" / "image" / "png" / "png_write_main.cpp"), "-o", str(cls.exe)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_vectors(self):
        r = subprocess.run([str(self.exe), "--vectors"], capture_output=True, text=True, timeout=60)
        self.assertIn("VECTORS PASS", r.stdout, r.stdout)
        self.assertIn("adler32 of 1 MiB of 0xff: %08x" % zlib.adler32(b"\xff" * (1 << 20)), r.stdout)

    def test_shapes_round_trip(self):
        params = "a red fox in the snow\nSteps: 8, Sampler: euler, CFG scale: 1, Seed: 42, Size: 64x48"
        for w, h, c in ((1, 1, 3), (3, 2, 3), (64, 48, 1), (64, 48, 4), (300, 200, 3), (1024, 1, 3)):
            with self.subTest(shape=(w, h, c)):
                out = Path(self.td.name) / ("p_%d_%d_%d.png" % (w, h, c))
                subprocess.run([str(self.exe), "--pattern", str(w), str(h), str(c), str(out), params], check=True)
                chunks, (W, H, C, depth, comp, filt, inter), px = read_png(out.read_bytes())
                self.assertEqual((W, H, C, depth, comp, filt, inter), (w, h, c, 8, 0, 0, 0))
                self.assertEqual(px, pattern(w, h, c))
                self.assertEqual([t for t, _ in chunks], [b"IHDR", b"tEXt", b"IDAT", b"IEND"])
                self.assertEqual(chunks[1][1], b"parameters\x00" + params.encode())
                try:
                    from PIL import Image
                except ImportError:
                    continue
                with Image.open(out) as im:
                    self.assertEqual(im.tobytes(), px)
                    self.assertEqual(im.info.get("parameters"), params)

    def test_no_parameters_and_raw(self):
        w, h, c = 17, 5, 3
        raw = Path(self.td.name) / "in.raw"
        data = bytes((i * 31) & 255 for i in range(w * h * c))
        raw.write_bytes(data)
        out = Path(self.td.name) / "raw.png"
        subprocess.run([str(self.exe), "--raw", str(raw), str(w), str(h), str(c), str(out)], check=True)
        chunks, _, px = read_png(out.read_bytes())
        self.assertEqual([t for t, _ in chunks], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(px, data)


if __name__ == "__main__":
    unittest.main()
