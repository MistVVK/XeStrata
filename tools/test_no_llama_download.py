"""OWN-1B: nothing downloads llama.cpp to build, package or install Strata.

ggml (third_party/strata-ggml), strata-vision's sources (third_party/strata-llama) and gguf-py (third_party/strata-gguf-py) are vendored at the llama.cpp
pin. This test reads the release scripts, Docker files, CI workflows and setup.py and fails when one of them fetches llama.cpp again or requires a checkout.
The overrides (STRATA_GGML_DIR, LLAMA, GG, --llama, STRATA_GGUF_PY) stay for two releases and are allowed: they only POINT at a checkout.

    python -m unittest tools.test_no_llama_download -v
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS = ["packaging/build_cuda_linux.sh", "packaging/build_cuda_linux_aarch64.sh", "packaging/build_cuda_windows.py", "packaging/build_hip_linux.sh", "packaging/build_hip_gfx1103.sh",
           "packaging/build_sycl_linux.sh", "packaging/build_split_cuda.sh", "packaging/build_split_hip.sh",
           "packaging/build_vision_windows.bat", "packaging/build_vision_mac.sh",
           "packaging/mac/build_engine.sh", "spark/build_engine.sh",
           "tools/hip/build_windows.bat", "Dockerfile", "docker/Dockerfile", "docker/entrypoint.sh", ".github/workflows/backends.yml",
           ".github/workflows/metal-build.yml", "metal/setup_mac.py", "sycl/setup_intel.py"]
# a download of llama.cpp: its git remote or archive URL, setup's fetch function, the downloaded folder as a requirement
FETCH = [r"ggml-org/llama\.cpp(\.git|/archive)", r"get_llama_cpp\s*\(", r"LLAMA_CPP_ZIP", r"FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP"]
REQUIRED = [r"third_party[/\\]llama\.cpp", r":\?[^}]*llama\.cpp", r"set STRATA_GGML_DIR to"]


class NoLlamaDownload(unittest.TestCase):
    def test_no_script_dockerfile_or_workflow_fetches_or_requires_a_llama_cpp_tree(self):
        for rel in SCRIPTS:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for pat in FETCH + REQUIRED:
                self.assertIsNone(re.search(pat, text), f"{rel} still matches /{pat}/")

    def test_setup_has_no_llama_cpp_zip_and_the_vendored_trees_are_the_defaults(self):
        import setup
        self.assertFalse(hasattr(setup, "LLAMA_CPP_ZIP"))
        self.assertIsNone(setup.get_llama_cpp() if not setup.os.environ.get("STRATA_GGML_DIR") else None)
        self.assertEqual(setup.VENDORED_LLAMA, ROOT / "third_party" / "strata-llama")
        self.assertEqual(setup.vision_llama(None), ROOT / "third_party" / "strata-llama")
        self.assertEqual(setup.ggml_defs(None), [])
        for part in ("third_party/strata-ggml",):
            self.assertIn(part, setup.ENGINE_SOURCES)
        self.assertIn("third_party/strata-llama", setup.VISION_SOURCES)
        self.assertIn("tools/vision", setup.VISION_SOURCES)

    def test_tools_vision_takes_the_vendored_tree_by_default(self):
        cm = (ROOT / "tools" / "vision" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-llama"', cm)
        tree = ROOT / "third_party" / "strata-llama"
        for rel in ("CMakeLists.txt", "src/llama.cpp", "tools/mtmd/mtmd.cpp", "tools/mtmd/models/qwen3vl.cpp", "ggml/CMakeLists.txt", "ggml/src/ggml-backend-reg.cpp",
                    "ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-sycl/ggml-sycl.cpp", "ggml/src/ggml-metal/ggml-metal.cpp", "ggml/src/ggml-hip/CMakeLists.txt",
                    "vendor/miniaudio/miniaudio.h", "vendor/stb/stb_image.h", "LICENSE"):
            self.assertTrue((tree / rel).is_file(), rel)

    def test_the_nvfp4_tools_take_the_vendored_converter(self):
        """The NVFP4-local tools import torch and transformers (not on the test boxes), so only the wiring is checked: they compile, name no downloaded tree and
        find the vendored converter, whose conversion/qwen4exp.py the policy hooks patch."""
        sys.path.insert(0, str(ROOT / "tools"))
        import _paths
        for name in ("nvfp4_convert.py", "nvfp4_verify.py", "nvfp4_verify_gguf.py"):
            src = (ROOT / "tools" / name).read_text(encoding="utf-8")
            compile(src, name, "exec")
            self.assertNotIn("llama.cpp\"", src.replace("llama.cpp's", ""), name)
            self.assertNotIn('"third_party" / "llama.cpp"', src, name)
        conv = Path(_paths.converter_dir())
        self.assertEqual(conv, ROOT / "third_party" / "strata-gguf-py" / "converter")
        src = (conv / "conversion" / "qwen4exp.py").read_text(encoding="utf-8")
        self.assertIn("class Qwen4ExpTextModel", src)
        self.assertIn("_place_ple_shard", src)

    def test_the_terminal_and_control_carry_gguf_py_so_a_model_is_prepared_without_a_download(self):
        pt = (ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8")
        self.assertIn('INSTALLER_THIRD_PARTY = ("third_party/strata-gguf-py",)', pt)


if __name__ == "__main__":
    unittest.main()
