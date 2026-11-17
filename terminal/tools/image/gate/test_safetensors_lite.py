"""tools/image/gate/test_safetensors_lite.py - TEST-ONLY: src/image/vae/safetensors_lite.hpp (the VAE decoder's file
reader) on files written here: names, dtypes, shapes, offsets, the bytes of a tensor, __metadata__ skipped, escaped
strings, and the refusals (bad length, broken JSON).  Builds a small C++ program with the host compiler; skipped when
there is none.

    python -m pytest tools/image/gate/test_safetensors_lite.py"""
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CXX = shutil.which("g++") or shutil.which("clang++")

PROGRAM = r'''
#include "safetensors_lite.hpp"
#include <cstring>
#include <cstdio>
int main(int argc, char** argv) {
    strata::image::StFile f;
    std::string err = strata::image::st_read_header(argv[1], f);
    if (!err.empty()) { std::printf("ERR %s\n", err.c_str()); return 1; }
    for (auto& kv : f.tensors) {
        std::printf("T %s %s", kv.first.c_str(), kv.second.dtype.c_str());
        for (auto s : kv.second.shape) std::printf(" %lld", (long long) s);
        std::vector<unsigned char> b;
        err = strata::image::st_read_tensor(argv[1], f, kv.second, b);
        if (!err.empty()) { std::printf(" READERR %s\n", err.c_str()); return 1; }
        std::printf(" |");
        for (auto c : b) std::printf(" %02x", c);
        std::printf("\n");
    }
    for (int th : {1, 3, 16}) {   // the whole data section in parallel parts: the same bytes as the tensor reads
        std::unique_ptr<unsigned char[]> all;
        size_t size = 0;
        err = strata::image::st_read_all_parallel(argv[1], f, all, size, th);
        if (!err.empty()) { std::printf("PARERR %s\n", err.c_str()); return 1; }
        for (auto& kv : f.tensors) {
            std::vector<unsigned char> b;
            strata::image::st_read_tensor(argv[1], f, kv.second, b);
            if (b.size() && std::memcmp(b.data(), all.get() + kv.second.begin, b.size()) != 0) { std::printf("PARDIFF\n"); return 1; }
        }
    }
    return 0;
}
'''


def write_st(path: Path, header: dict, data: bytes, raw_header: bytes = None):
    h = raw_header if raw_header is not None else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


@unittest.skipUnless(CXX, "no host C++ compiler")
class SafetensorsLite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "t.cpp").write_text(PROGRAM)
        cls.exe = d / "st_test"
        subprocess.run([CXX, "-std=c++17", "-O1", "-pthread", "-I", str(ROOT.parent / "engine" / "src" / "image" / "vae"), str(d / "t.cpp"), "-o",
                        str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_on(self, header, data, raw_header=None):
        p = Path(self.tmp.name) / "m.safetensors"
        write_st(p, header, data, raw_header)
        r = subprocess.run([str(self.exe), str(p)], capture_output=True, text=True)
        return r.returncode, r.stdout

    def test_tensors_and_metadata(self):
        hdr = {"__metadata__": {"format": "pt", "nested": {"a": [1, 2, {"b": "c}"}]}},
               "decoder.conv_in.weight": {"dtype": "BF16", "shape": [2, 1, 1, 1], "data_offsets": [0, 4]},
               "x\\\"y": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
        rc, out = self.run_on(hdr, bytes([1, 2, 3, 4, 0, 0, 128, 63]))
        self.assertEqual(rc, 0, out)
        lines = sorted(out.splitlines())
        self.assertIn("T decoder.conv_in.weight BF16 2 1 1 1 | 01 02 03 04", lines)
        self.assertIn('T x\\"y F32 1 | 00 00 80 3f', lines)
        self.assertEqual(len(lines), 2)

    def test_empty_shape_scalar(self):
        rc, out = self.run_on({"s": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, bytes(4))
        self.assertEqual((rc, out.strip()), (0, "T s F32 | 00 00 00 00"))

    def test_refusals(self):
        rc, out = self.run_on(None, b"", raw_header=b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}')
        self.assertEqual(rc, 1)
        self.assertIn("bad data_offsets", out)
        rc, out = self.run_on(None, b"", raw_header=b'{"a": {"dtype": "F32"')
        self.assertEqual(rc, 1)
        self.assertTrue(out.startswith("ERR header"), out)
        rc, out = self.run_on({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, bytes(4))
        self.assertEqual(rc, 1)
        self.assertIn("short read", out)


if __name__ == "__main__":
    unittest.main()
