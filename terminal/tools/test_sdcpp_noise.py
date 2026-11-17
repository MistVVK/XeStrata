"""Tests for Strata's noise generator for sd-cli (tools/image/sdcpp_noise): the patch for the pinned
stable-diffusion.cpp (a stand-in file with the code it edits), the table as a C array, and, where a C++ compiler is
there, the generator's stream for three seeds against the hashes of stable-diffusion.cpp's Linux x86 build.  The
exhaustive proofs (every input of the generator, 1005 seeds, three CPUs and compilers) are tools/image/gate/noise.

    python -m unittest tools.test_sdcpp_noise
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOISE = ROOT / "tools" / "image" / "sdcpp_noise"
_spec = importlib.util.spec_from_file_location("strata_sdcpp_noise_apply_test", NOISE / "apply.py")
A = importlib.util.module_from_spec(_spec)             # not "import apply": sdcpp_native has an apply.py too
_spec.loader.exec_module(A)

# the lines of src/pipeline/diffusion_engine.cpp at the pinned commit that apply.py edits, with their neighbours
ENGINE = """#include "core/rng.hpp"
#include "core/rng_mt19937.hpp"
#include "core/rng_philox.hpp"
#include "core/util.h"

bool StableDiffusionGGML::init(const sd_ctx_params_t* sd_ctx_params) {
    std::string rpc_servers_spec = SAFE_STR(sd_ctx_params->rpc_servers);
    add_rpc_devices(rpc_servers_spec);

    rng = get_rng(sd_ctx_params->rng_type);
    if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
        sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
    } else {
        sampler_rng = rng;
    }

    ggml_log_set(sd_ggml_log_callback, nullptr);
}
"""

# fnv64 of the 262144 floats randn(seed, 262144) gives, as stable-diffusion.cpp 2988060's Linux x86 build (GCC -O3, no
# -march) makes them (tools/image/gate/noise/seeds.cpp, 1005 seeds, p2006 x86 11-10; the same on aarch64 and MSVC)
GOLDEN = {0: "ed8d8d740dbd5fe6", 42: "ab9095ca6f8db096", 4611686018427387904: "ecc168e692cf1189"}

PROGRAM = """#include <cstdio>
#include <cstdlib>
#include "strata_noise.hpp"
int main(int argc, char** argv) {
    const uint64_t seed = std::strtoull(argv[1], nullptr, 10);
    const std::vector<float> o = strata_noise::randn(seed, 262144);
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = (const unsigned char*) o.data();
    for (size_t i = 0; i < o.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
    std::printf("%016llx\\n", (unsigned long long) h);
    return 0;
}
"""


class Apply(unittest.TestCase):
    def tree(self, d):
        sd = Path(d) / "sd"
        (sd / "src" / "pipeline").mkdir(parents=True)
        (sd / "src" / "core").mkdir(parents=True)
        (sd / "src" / "pipeline" / "diffusion_engine.cpp").write_text(ENGINE, encoding="utf-8")
        return sd

    def test_patch_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            sd = self.tree(d)
            A.apply(sd)
            engine = (sd / "src" / "pipeline" / "diffusion_engine.cpp").read_text(encoding="utf-8")
            self.assertIn('#include "core/strata_philox_rng.hpp"', engine)
            self.assertEqual(engine.count("strata_noise::requested(sd_ctx_params->model_args)"), 1)
            self.assertIn("rng = std::make_shared<StrataPhiloxRNG>();", engine)
            # the stock code is all still there, in the same order: the block only follows it
            self.assertTrue(engine.index("sampler_rng = rng;") < engine.index("strata_noise::requested"))
            self.assertTrue(engine.index("strata_noise::requested") < engine.index("ggml_log_set"))
            for name in A.FILES + ("strata_sin_exc_data.inc",):
                self.assertTrue((sd / "src" / "core" / name).is_file(), name)
            before = {p: p.read_bytes() for p in sd.rglob("*") if p.is_file()}
            A.apply(sd)                                   # again: nothing changes
            self.assertEqual({p: p.read_bytes() for p in sd.rglob("*") if p.is_file()}, before)

    def test_another_commit_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            sd = self.tree(d)
            p = sd / "src" / "pipeline" / "diffusion_engine.cpp"
            p.write_text(ENGINE.replace("sampler_rng = rng;", "sampler_rng = other;"), encoding="utf-8")
            with self.assertRaises(SystemExit):
                A.apply(sd)

    def test_the_table_as_a_c_array(self):
        data = (NOISE / A.BIN).read_bytes()
        src = A.data_inc(data)
        body = src.split("= {", 1)[1].rsplit("};", 1)[0]
        got = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", body))
        self.assertEqual(got, data)
        self.assertIn(f"STRATA_SIN_EXC_DATA[]", src)
        self.assertLess(len(src), 1_200_000)             # about 1.1 MB of source for the 184 KB table

    def test_prepare_writes_the_generator_files(self):
        with tempfile.TemporaryDirectory() as d:
            A.prepare(Path(d) / "inc")
            names = sorted(p.name for p in (Path(d) / "inc").iterdir())
            self.assertEqual(names, sorted(A.FILES + ("strata_sin_exc_data.inc",)))


@unittest.skipUnless(shutil.which("g++") or shutil.which("c++") or shutil.which("clang++"), "no C++ compiler")
class Stream(unittest.TestCase):
    def test_three_seeds_match_the_x86_stream(self):
        cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
        with tempfile.TemporaryDirectory() as d:
            inc = Path(d) / "inc"
            A.prepare(inc)
            (Path(d) / "t.cpp").write_text(PROGRAM, encoding="utf-8")
            exe = Path(d) / ("t.exe" if sys.platform == "win32" else "t")
            r = subprocess.run([cxx, "-std=c++17", "-O2", "-I", str(inc), str(Path(d) / "t.cpp"), "-o", str(exe)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for seed, want in GOLDEN.items():
                out = subprocess.run([str(exe), str(seed)], capture_output=True, text=True, timeout=300)
                self.assertEqual(out.stdout.strip(), want, seed)


if __name__ == "__main__":
    unittest.main()
