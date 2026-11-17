"""tools/test_dit_rope.py - our Qwen-Image-2.1 RoPE table (src/image/dit/qi21_rope.cpp) against the tables
stable-diffusion.cpp 2988060 built for the same pictures: their sha256, recorded from sd.cpp's stock and native runners
through the stage hook (tools/image/gate/stage_hooks.py pe_stock_<k>) in tools/image/gate/data/qi21_rope_sha.json
(512x512 with four prompts, 1024x1024, 1536x1024, a 1,025-token prompt).  Compiles with the system C++ compiler; no GPU,
no sd.cpp, no model file.

The table is std::cos / std::sin of float angles, so its last bit is the C library's: the recorded sha256 are sd.cpp's on
glibc (Linux), and the byte check runs where the C library is glibc.  Elsewhere sd.cpp gets that C library's bits too, not
glibc's (macOS 27 on an M4: every float correctly rounded, 3,360 of the 269,568 floats of the first case 1 ulp from glibc's,
so no recorded hash matches).  The second test runs everywhere: every float of the table against a float64 reference of the
same float steps, within 2 ulp plus the error of a 2-ulp powf in the angle.

    python -m pytest tools/test_dit_rope.py"""
import hashlib
import json
import math
import platform
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
CASES = ROOT / "tools" / "image" / "gate" / "data" / "qi21_rope_sha.json"
GLIBC = platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"


def build(td: Path) -> Path:
    exe = td / "strata-dit-rope"
    r = subprocess.run([CXX, "-std=c++17", "-O2", "-I", str(ROOT.parent / "engine" / "include"),
                        str(ROOT.parent / "engine" / "src" / "image" / "dit" / "qi21_rope.cpp"),
                        str(ROOT.parent / "engine" / "src" / "image" / "dit" / "qi21_rope_main.cpp"), "-o", str(exe)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise AssertionError("build failed:\n" + r.stderr[-3000:])
    return exe


def table(exe: Path, td: Path, c: dict) -> bytes:
    out = td / "t.bin"
    subprocess.run([str(exe), str(c["text_tokens"]), str(c["H"]), str(c["W"]), str(out)], check=True)
    return out.read_bytes()


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def reference(text: int, H: int, W: int):
    """(value, angle) for every float of the table: sd.cpp's float steps in float64, each rounded to float32, and cos / sin of
    the float32 angle rounded once to float32."""
    pos = [(float(p),) * 3 for p in range(text)]
    pos += [(float(text), float(h - (H - H // 2)), float(w - (W - W // 2))) for h in range(H) for w in range(W)]   # every picture token at text_len
    val, ang = [0.0] * (len(pos) * 256), [0.0] * (len(pos) * 256)
    off = 0
    for a, dim in enumerate((16, 56, 56)):
        half = dim // 2
        step = f32(f32(f32(dim * 1.0 - 2) / dim) / (half - 1))
        omega = [f32(1.0 / f32(10000.0 ** f32(i * step))) for i in range(half)]
        for i, p in enumerate(pos):
            for j, om in enumerate(omega):
                t = f32(p[a] * om)
                c, s = f32(math.cos(t)), f32(math.sin(t))
                k = i * 256 + off + 4 * j
                val[k:k + 4] = (c, -s, s, c)
                ang[k:k + 4] = (t,) * 4
        off += half * 4
    return val, ang


def ulp32(x: float) -> float:
    return 2.0 ** (math.floor(math.log2(max(abs(x), 2.0 ** -126))) - 23)


@unittest.skipUnless(CXX, "no C++ compiler")
class RopeTableMatchesSdcpp(unittest.TestCase):
    @unittest.skipUnless(GLIBC, "the recorded sha256 are sd.cpp's tables on glibc, and this C library rounds cosf / sinf "
                                "differently in the last bit (test_tables_close_to_float64 checks every float here)")
    def test_tables(self):
        cases = json.loads(CASES.read_text())
        self.assertGreaterEqual(len(cases), 7)
        with tempfile.TemporaryDirectory() as td:
            exe = build(Path(td))
            bad = [c["from"] for c in cases if hashlib.sha256(table(exe, Path(td), c)).hexdigest() != c["sha256"]]
            self.assertEqual(bad, [])

    def test_tables_close_to_float64(self):
        cases = json.loads(CASES.read_text())
        self.assertGreaterEqual(len(cases), 7)
        with tempfile.TemporaryDirectory() as td:
            exe = build(Path(td))
            for c in cases:
                raw = table(exe, Path(td), c)
                got = struct.unpack("<%df" % (len(raw) // 4), raw)
                val, ang = reference(c["text_tokens"], c["H"], c["W"])
                self.assertEqual(len(got), len(val), c["from"])
                far = [k for k, (g, v, t) in enumerate(zip(got, val, ang))
                       if abs(g - v) > 2 * ulp32(v) + abs(t) * 2.0 ** -22]
                self.assertEqual(far[:5], [], f"{c['from']}: {len(far)} of {len(got)} floats off "
                                              f"(first at {far[0] if far else '-'}: ours {got[far[0]] if far else '-'}, "
                                              f"float64 {val[far[0]] if far else '-'})")


if __name__ == "__main__":
    unittest.main()
