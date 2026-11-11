"""tools/test_qi21_noise.py - our initial noise (src/image/noise/qi21_noise.cpp, Strata's Philox generator) against
stable-diffusion.cpp 2988060's own PhiloxRNG as its x86 build computes it: tools/image/gate/data/qi21_noise_sha.json
holds the sha256 of sd.cpp's noise for seeds 42, 1234, 7, 2026 at 65,536 and 262,144 values (the 512 and 1024
latents), made by compiling sd.cpp's core/rng_philox.hpp with g++ -O3 on x86 (p1006).  Compiles our generator with
the system C++ compiler; no GPU, no sd.cpp, no model file.

    python -m pytest tools/test_qi21_noise.py"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


@unittest.skipUnless(CXX, "no C++ compiler")
class NoiseMatchesSdcppX86(unittest.TestCase):
    def test_seeds(self):
        cases = json.loads((ROOT / "tools" / "image" / "gate" / "data" / "qi21_noise_sha.json").read_text())
        self.assertEqual(len(cases), 8)
        with tempfile.TemporaryDirectory() as td:
            inc = Path(td) / "inc"
            subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tools" / "image" / "sdcpp_noise" / "apply.py"),
                            "--prepare", str(inc)], check=True, capture_output=True)
            exe = Path(td) / "strata-noise-check"
            r = subprocess.run([CXX, "-std=c++17", "-O2", "-I", str(ROOT / "include"),
                                "-I", str(ROOT / "tools" / "image" / "sdcpp_noise"), "-I", str(inc),
                                str(ROOT / "src" / "image" / "noise" / "qi21_noise.cpp"),
                                str(ROOT / "src" / "image" / "noise" / "qi21_noise_check_main.cpp"), "-o", str(exe)],
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            bad = []
            for c in cases:
                out = Path(td) / "n.bin"
                subprocess.run([str(exe), "--write", str(c["seed"]), str(c["count"]), str(out)], check=True)
                if hashlib.sha256(out.read_bytes()).hexdigest() != c["sha256"]:
                    bad.append((c["seed"], c["count"]))
            self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
