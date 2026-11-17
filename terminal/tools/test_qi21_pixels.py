"""tools/test_qi21_pixels.py - the decoded picture to 8-bit pixels (src/image/vae/qi21_pixels.cpp) as
stable-diffusion.cpp 2988060 makes them: tools/image/gate/data/qi21_pixels_fixture.bin holds 8,184 values of sd.cpp's
decoded float picture (512x512, prompt 0, seed 42; 729 of them within 1e-3 of a rounding boundary) with the bytes of
sd-cli's own PNG at the same places; plus edge values whose expected bytes are computed here with numpy's float32
(v x 255 rounded, + 0.5 rounded, truncated; 0 / 255 at or outside the ends).  Compiled twice (-O2; -O3
-ffp-contract=fast: the file must not fuse).  No GPU, no sd.cpp, no model file.

    python -m pytest tools/test_qi21_pixels.py"""
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
FIXTURE = ROOT / "tools" / "image" / "gate" / "data" / "qi21_pixels_fixture.bin"


def edge_fixture(path: Path) -> None:
    k = np.arange(256, dtype=np.float32)
    v = np.concatenate([
        np.array([-1.0, -0.0, 0.0, 1e-45, 1.0, 1.5, 0.99999994], dtype=np.float32),
        k / np.float32(255), (k + np.float32(0.5)) / np.float32(255),
        np.nextafter((k + np.float32(0.5)) / np.float32(255), np.float32(0)),
        np.nextafter((k + np.float32(0.5)) / np.float32(255), np.float32(1))]).astype(np.float32)
    m = (v * np.float32(255)).astype(np.float32)
    s = (m + np.float32(0.5)).astype(np.float32)
    want = np.where(v <= 0, 0, np.where(v >= 1, 255, np.clip(s, 0, 255).astype(np.int64))).astype(np.uint8)
    path.write_bytes(struct.pack("<i", len(v)) + v.astype("<f4").tobytes() + want.tobytes())


@unittest.skipUnless(CXX, "no C++ compiler")
class PixelsMatchSdcpp(unittest.TestCase):
    def build_and_check(self, flags):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-pixels-check"
            r = subprocess.run([CXX, "-std=c++17", *flags, "-I", str(ROOT.parent / "engine" / "include"),
                                str(ROOT.parent / "engine" / "src" / "image" / "vae" / "qi21_pixels.cpp"),
                                str(ROOT.parent / "engine" / "src" / "image" / "vae" / "qi21_pixels_check_main.cpp"), "-o", str(exe)],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            edges = Path(td) / "edges.bin"
            edge_fixture(edges)
            r = subprocess.run([str(exe), str(FIXTURE), str(edges)], capture_output=True, text=True, timeout=60)
            self.assertIn("PIXELS-CHECK PASS", r.stdout, r.stdout)

    def test_o2(self):
        self.build_and_check(["-O2"])

    def test_o3_contract_fast(self):
        self.build_and_check(["-O3", "-ffp-contract=fast"])


if __name__ == "__main__":
    unittest.main()
