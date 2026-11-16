"""OWN-1B: the quant header's text lives in third_party/strata-ggml/strata-quant-common.h; third_party/ggml/ggml-common.h only forwards to it (the
dpct-migrated SYCL-mode copy, sycl/third_party/ggml, is gone since 0.2.15: the SYCL build includes the shared header). The Metal runtime reads the text under the new name, the Mac package ships both names (the old one for two releases).

    python -m unittest tools.test_quant_header_names -v
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "third_party" / "strata-ggml" / "strata-quant-common.h"
OLD = ROOT / "third_party" / "ggml" / "ggml-common.h"


class QuantHeaderNames(unittest.TestCase):
    def test_the_text_is_in_the_new_file_and_the_old_path_forwards(self):
        text = NEW.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#ifndef GGML_COMMON_DECL"), "the new file is the header's text, verbatim (the Metal runtime reads it as text)")
        self.assertIn("GGML_COMMON_IMPL_METAL", text)
        self.assertGreater(text.count("\n"), 1900)
        old = [ln for ln in OLD.read_text(encoding="utf-8").splitlines() if not ln.startswith("//") and ln.strip()]
        self.assertEqual(old, ['#include "../strata-ggml/strata-quant-common.h"'])
        # the SYCL build takes the shared header (sycl/CMakeLists.txt: ${STRATA_ROOT}/third_party/ggml), not a migrated copy of its own
        self.assertFalse((ROOT / "sycl" / "third_party" / "ggml").exists(), "sycl/third_party/ggml is a dpct output copy no build reads")
        self.assertIn("${STRATA_ROOT}/third_party/ggml", (ROOT / "sycl" / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_the_metal_runtime_reads_the_new_name_and_the_old_one_second(self):
        rp = (ROOT / "metal" / "src" / "resource_path.h").read_text(encoding="utf-8")
        self.assertIn("resource2", rp)
        rt = (ROOT / "metal" / "src" / "mtl_rt.mm").read_text(encoding="utf-8")
        self.assertIn('resource2("STRATA_GGML_COMMON", "strata-quant-common.h", "ggml-common.h", STRATA_GGML_COMMON_DEFAULT)', rt)
        cm = (ROOT / "metal" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("third_party/strata-ggml/strata-quant-common.h", cm)

    def test_the_repo_mac_qwen_package_script_stages_the_real_header_under_both_names(self):
        """packaging/mac/build_engine.sh is the repo version of the release scripts' Mac stage (rel213/scripts/mac_rel213.sh copied third_party/ggml/ggml-common.h,
        which is only a forwarder now: staged like that the Metal kernels would have no block layouts)."""
        sh = (ROOT / "packaging" / "mac" / "build_engine.sh").read_text(encoding="utf-8")
        self.assertIn('QUANT=third_party/strata-ggml/strata-quant-common.h', sh)
        self.assertIn('cp "$QUANT" "$STAGE/metal/strata-quant-common.h"', sh)
        self.assertIn('cp "$QUANT" "$STAGE/metal/ggml-common.h"', sh)
        self.assertNotIn("cp third_party/ggml/ggml-common.h", sh)
        self.assertIn('cmp "$PKG/metal/$n" "$QUANT"', sh)
        self.assertNotIn("STRATA_GGML_DIR=$LL", sh)

    @unittest.skipUnless(shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"), "no C++ compiler")
    def test_an_include_of_the_old_name_still_compiles(self):
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "t.cpp"
            src.write_text("#define GGML_COMMON_DECL_CPP\n#define GGML_COMMON_IMPL_CPP\n#include \"ggml-common.h\"\n"
                           "int main() { return sizeof(block_q4_K) == 144 ? 0 : 1; }\n", encoding="utf-8")
            exe = Path(d) / "t"
            done = subprocess.run([cxx, "-std=c++17", "-I", str(OLD.parent), str(src), "-o", str(exe)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-800:])
            self.assertEqual(subprocess.run([str(exe)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
