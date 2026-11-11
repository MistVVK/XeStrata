"""tools/test_qi21_sampler.py - our Turbo sampler step (src/image/sample/qi21_sampler.cpp) against stable-diffusion.cpp
2988060's own values: tools/image/gate/data/qi21_sampler_fixture.bin holds, for steps 0-6 of one picture (512x512,
prompt 0, seed 42), 512 values spread over the latent of sd.cpp's DiT input x_k, its DiT output v_k and its next DiT
input x_{k+1}, with the sigmas and sd.cpp's timestep bits (written by strata-sampler-check --fixture from the stage
dumps).  Our step on x_k, v_k must give x_{k+1} byte for byte, our parsed schedule and timestep the same bits.
Compiled twice: plain -O2, and -O3 with contraction allowed by default (-ffp-contract=fast; the file must still not
fuse).  No GPU, no sd.cpp, no model file.

    python -m pytest tools/test_qi21_sampler.py"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
FIXTURE = ROOT / "tools" / "image" / "gate" / "data" / "qi21_sampler_fixture.bin"


@unittest.skipUnless(CXX, "no C++ compiler")
class SamplerMatchesSdcpp(unittest.TestCase):
    def build_and_check(self, flags):
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-sampler-check"
            r = subprocess.run([CXX, "-std=c++17", *flags, "-I", str(ROOT / "include"),
                                str(ROOT / "src" / "image" / "sample" / "qi21_sampler.cpp"),
                                str(ROOT / "src" / "image" / "sample" / "qi21_sampler_check_main.cpp"), "-o", str(exe)],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            r = subprocess.run([str(exe), "--fixture-check", str(FIXTURE)], capture_output=True, text=True, timeout=60)
            self.assertIn("SAMPLER-FIXTURE PASS", r.stdout, r.stdout[-3000:])
            self.assertEqual(r.stdout.count("identical"), 7, r.stdout)

    def test_o2(self):
        self.build_and_check(["-O2"])

    def test_o3_contract_fast(self):
        self.build_and_check(["-O3", "-ffp-contract=fast"])


if __name__ == "__main__":
    unittest.main()
