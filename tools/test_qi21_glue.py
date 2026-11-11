"""tools/test_qi21_glue.py - the steps between Strata's image parts (src/image/pipeline/qi21_glue.cpp) give the bytes
stable-diffusion.cpp 2988060's code gives: the starting latent (DiscreteFlowDenoiser::noise_scaling on a zero latent),
the condition (apply_token_weights + the system rows dropped) and the decoder output to 0..1 (scale_tensor_to_0_1).
The expectations are those operations written out with numpy float32 (and a sequential float64 sum for sd.cpp's
Tensor<float>::mean).  Compiled twice (-O2; -O3 -ffp-contract=fast: the file must not fuse).  No GPU, no model file.

    python -m pytest tools/test_qi21_glue.py"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
F = np.float32


def sd_mean(v: np.ndarray) -> np.float32:
    """sd::Tensor<float>::mean: a double sum in element order, divided in double, then float."""
    s = np.cumsum(v.reshape(-1).astype(np.float64))[-1]
    return F(s / np.float64(v.size))


def want_condition(h: np.ndarray, w: np.ndarray, n_system: int) -> np.ndarray:
    h = h.astype(np.float32).copy()
    if not np.all(w == F(1)):
        m0 = sd_mean(h)
        h = (h * w[:, None]).astype(np.float32)
        m1 = sd_mean(h)
        if np.isfinite(m0) and np.isfinite(m1) and m1 != 0:
            h = (h * F(m0 / m1)).astype(np.float32)
    return h[n_system:]


def want_unit(x: np.ndarray) -> np.ndarray:
    v = ((x + F(1)).astype(np.float32) * F(0.5)).astype(np.float32)
    return np.where(v < 0, F(0), np.where(v > 1, F(1), v)).astype(np.float32)


@unittest.skipUnless(CXX, "no C++ compiler")
class GlueMatchesSdcpp(unittest.TestCase):
    def run_all(self, flags):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            exe = t / "strata-glue-check"
            r = subprocess.run([CXX, "-std=c++17", *flags, "-I", str(ROOT / "include"),
                                str(ROOT / "src" / "image" / "pipeline" / "qi21_glue.cpp"),
                                str(ROOT / "src" / "image" / "pipeline" / "qi21_glue_check_main.cpp"), "-o", str(exe)],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])

            def run(*args):
                p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, timeout=60)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

            # starting latent: sigma0 1.0 (Turbo) and another sigma; -0.0 noise becomes +0.0 as in sd.cpp
            noise = np.concatenate([rng.standard_normal(4096), [-0.0, 0.0, 1e-45, -1e-45]]).astype(np.float32)
            noise.tofile(t / "noise.bin")
            for sigma in ("1.0", "0.978453"):
                run("latent", t / "noise.bin", sigma, t / "x0.bin")
                s = F(float(sigma))
                want = ((np.zeros_like(noise) * (F(1) - s)).astype(np.float32) + (noise * s).astype(np.float32))
                got = np.fromfile(t / "x0.bin", dtype=np.float32)
                self.assertEqual(got.tobytes(), want.astype(np.float32).tobytes(), sigma)

            # condition: plain weights (rows dropped only), weighted rows, a BREAK-like -1, a zero new mean
            n, hd = 23, 96
            h = (rng.standard_normal((n, hd)) * 40).astype(np.float32)
            h.tofile(t / "h.bin")
            cases = [np.ones(n, np.float32),
                     np.array([1.0] * 10 + [1.21] * 5 + [0.9] * 8, np.float32),
                     np.array([1.0] * 20 + [-1.0, 1.1, 1.0], np.float32)]
            for i, w in enumerate(cases):
                w.tofile(t / "w.bin")
                run("cond", t / "h.bin", n, hd, t / "w.bin", 14, t / "c.bin")
                got = np.fromfile(t / "c.bin", dtype=np.float32)
                self.assertEqual(got.tobytes(), want_condition(h, w, 14).tobytes(), f"weights case {i}")
            z = np.zeros((n, hd), np.float32)
            z.tofile(t / "z.bin")
            cases[1].tofile(t / "w.bin")
            run("cond", t / "z.bin", n, hd, t / "w.bin", 14, t / "c.bin")
            self.assertEqual(np.fromfile(t / "c.bin", dtype=np.float32).tobytes(), z[14:].tobytes())

            # decoder output to 0..1, including the ends and values next to them
            x = np.concatenate([rng.uniform(-1.3, 1.3, 20000),
                                [-1.0, 1.0, -3.0, 3.0, 0.0, -0.0]]).astype(np.float32)
            x = np.concatenate([x, np.nextafter(np.float32([-1, 1]), np.float32(0)),
                                np.nextafter(np.float32([-1, 1]), np.float32([-2, 2]))]).astype(np.float32)
            x.tofile(t / "u.bin")
            run("unit", t / "u.bin", t / "u_out.bin")
            self.assertEqual(np.fromfile(t / "u_out.bin", dtype=np.float32).tobytes(), want_unit(x).tobytes())

    def test_o2(self):
        self.run_all(["-O2"])

    def test_o3_contract_fast(self):
        self.run_all(["-O3", "-ffp-contract=fast"])


if __name__ == "__main__":
    unittest.main()
