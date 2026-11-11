"""tools/test_dit_rope.py - our Qwen-Image-2.1 RoPE table (src/image/dit/qi21_rope.cpp) against the tables
stable-diffusion.cpp 2988060 built for the same pictures: their sha256, recorded from sd.cpp's stock and native runners
through the stage hook (tools/image/gate/stage_hooks.py pe_stock_<k>) in tools/image/gate/data/qi21_rope_sha.json
(512x512 with four prompts, 1024x1024, 1536x1024, a 1,025-token prompt).  Compiles with the system C++ compiler; no GPU,
no sd.cpp, no model file.

    python -m pytest tools/test_dit_rope.py"""
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


@unittest.skipUnless(CXX, "no C++ compiler")
class RopeTableMatchesSdcpp(unittest.TestCase):
    def test_tables(self):
        cases = json.loads((ROOT / "tools" / "image" / "gate" / "data" / "qi21_rope_sha.json").read_text())
        self.assertGreaterEqual(len(cases), 7)
        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "strata-dit-rope"
            r = subprocess.run([CXX, "-std=c++17", "-O2", "-I", str(ROOT / "include"),
                                str(ROOT / "src" / "image" / "dit" / "qi21_rope.cpp"),
                                str(ROOT / "src" / "image" / "dit" / "qi21_rope_main.cpp"), "-o", str(exe)],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            bad = []
            for c in cases:
                out = Path(td) / "t.bin"
                subprocess.run([str(exe), str(c["text_tokens"]), str(c["H"]), str(c["W"]), str(out)], check=True)
                if hashlib.sha256(out.read_bytes()).hexdigest() != c["sha256"]:
                    bad.append(c["from"])
            self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
