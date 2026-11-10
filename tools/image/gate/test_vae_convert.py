"""tools/image/gate/test_vae_convert.py - TEST-ONLY: src/image/vae/convert.hpp (the VAE decoder's weight conversion at
open, split over threads) gives the same bytes with 1 thread and with several, on every bf16 bit pattern (zeros,
subnormals, infinities, NaN payloads, values that round up into the next exponent), agrees with numpy's float16
rounding on every non-NaN pattern, keeps NaN a NaN, and lays a convolution's weights out as [oc][tap][ic].
Builds a small C++ program with the host compiler; skipped when there is none.

    python -m pytest tools/image/gate/test_vae_convert.py"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CXX = shutil.which("g++") or shutil.which("clang++")

PROGRAM = r'''
#include "convert.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
// argv: in.bin (bf16 bytes) oc ic taps threads out.bin  -> out: f32 of every value, then the f16 conv layout
int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    std::vector<unsigned char> raw;
    int c;
    while ((c = std::fgetc(f)) != EOF) raw.push_back((unsigned char) c);
    std::fclose(f);
    const int oc = std::atoi(argv[2]), ic = std::atoi(argv[3]), taps = std::atoi(argv[4]), th = std::atoi(argv[5]);
    const size_t n = raw.size() / 2;
    std::vector<float> w(n);
    strata::image::bf16_to_f32(raw.data(), n, w.data(), th);
    std::vector<uint16_t> h((size_t) oc * ic * taps);
    strata::image::bf16_conv_to_f16(raw.data(), oc, ic, taps, h.data(), th);
    FILE* o = std::fopen(argv[6], "wb");
    std::fwrite(w.data(), 4, n, o);
    std::fwrite(h.data(), 2, h.size(), o);
    std::fclose(o);
    return 0;
}
'''


@unittest.skipUnless(CXX, "no host C++ compiler")
class Convert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "t.cpp").write_text(PROGRAM)
        cls.exe = d / "conv_test"
        subprocess.run([CXX, "-std=c++17", "-O2", "-pthread", "-I", str(ROOT / "src" / "image" / "vae"),
                        "-I", str(ROOT / "include"), str(d / "t.cpp"), "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_prog(self, bits: np.ndarray, oc, ic, taps, threads):
        d = Path(self.tmp.name)
        (d / "in.bin").write_bytes(bits.astype("<u2").tobytes())
        subprocess.run([str(self.exe), str(d / "in.bin"), str(oc), str(ic), str(taps), str(threads),
                        str(d / "out.bin")], check=True)
        out = (d / "out.bin").read_bytes()
        n = bits.size
        return np.frombuffer(out[:4 * n], dtype="<f4"), np.frombuffer(out[4 * n:], dtype="<u2")

    def test_every_bf16_pattern_same_bytes_for_any_thread_count(self):
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)           # oc 256 x ic 64 x taps 4
        f1, h1 = self.run_prog(bits, 256, 64, 4, 1)
        for th in (2, 7, 16):
            with self.subTest(threads=th):
                fn, hn = self.run_prog(bits, 256, 64, 4, th)
                self.assertEqual(f1.tobytes(), fn.tobytes())
                self.assertEqual(h1.tobytes(), hn.tobytes())

    def test_values_against_numpy(self):
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        f, h = self.run_prog(bits, 65536, 1, 1, 3)                             # layout = identity for ic = taps = 1
        want_f = (bits.astype(np.uint32) << 16).view(np.float32)
        self.assertEqual(f.view(np.uint32).tolist(), want_f.view(np.uint32).tolist())     # bf16 -> f32 is exact
        nan = np.isnan(want_f)
        with np.errstate(over="ignore"):
            want_h = want_f.astype(np.float16).view(np.uint16)
        self.assertEqual(h[~nan].tolist(), want_h[~nan].tolist())               # RNE, subnormals, overflow -> inf
        self.assertTrue(np.isnan(h[nan].view(np.float16)).all())                # a NaN stays a NaN
        # the interesting cases are really in the set
        hv = h.view(np.float16)
        self.assertTrue(((want_f != 0) & (np.abs(want_f) < 2 ** -14) & (hv != 0)).any())   # f16 subnormals
        self.assertTrue(np.isinf(hv[np.isfinite(want_f)]).any())                # finite bf16 that overflows f16
        self.assertTrue(np.isinf(want_f).any() and nan.any())

    def test_layout(self):
        rng = np.random.default_rng(5)
        oc, ic, taps = 3, 4, 9
        w = rng.standard_normal((oc, ic, taps)).astype(np.float32)
        bits = (w.view(np.uint32) >> 16).astype(np.uint16)                      # truncate to bf16
        _, h = self.run_prog(bits.reshape(-1), oc, ic, taps, 2)
        wf = (bits.astype(np.uint32) << 16).view(np.float32)
        want = wf.transpose(0, 2, 1).reshape(oc, taps * ic).astype(np.float16).view(np.uint16)
        self.assertEqual(h.reshape(oc, taps * ic).tolist(), want.tolist())


if __name__ == "__main__":
    unittest.main()
