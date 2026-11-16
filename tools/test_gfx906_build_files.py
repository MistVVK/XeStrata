"""#3172 #3271: the gfx906 (HIP wave64 compat) build stays buildable. The real check is a compile for gfx906 (done on a box with ROCm: the
`strata` target links); these cover what a text check can: the pieces of CMakeLists.txt / cmake/*.cmake a 0.1.44+ gfx906 build was missing."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CMakeLists.txt is cut into cmake/*.cmake macros (cleanup 5.4): the text checks read all of them
HIP_TEXT = (ROOT / "cmake" / "backends" / "hip.cmake").read_text(encoding="utf-8").replace("\r\n", "\n")
TEXT = "\n".join(p.read_text(encoding="utf-8").replace("\r\n", "\n")
                 for p in [ROOT / "CMakeLists.txt", *sorted((ROOT / "cmake").glob("*.cmake")), *sorted((ROOT / "cmake" / "backends").glob("*.cmake"))])


class Gfx906CMake(unittest.TestCase):
    def test_qwen35_is_not_built_for_gfx906(self):
        # q36_engine.cu calls cublasSgemm / cublasGemmStridedBatchedEx, which the wave64 compat layer does not map; main() guards every use
        # with STRATA_HAVE_QWEN35, so leaving the library out gives a gfx906 build that links
        block = re.search(r"if\(NOT STRATA_HIP_GFX906\)\n  add_library\(strata_qwen35.*?target_compile_definitions\(strata PRIVATE STRATA_HAVE_QWEN35=1\)\n  endif\(\)", TEXT, re.S)
        self.assertIsNotNone(block)
        self.assertIn("add_executable(strata-q36 ", block.group(0))
        self.assertIn("target_link_libraries(strata PRIVATE strata_qwen35)", block.group(0))

    def test_every_use_of_qwen35_in_the_engine_is_guarded(self):
        text = (ROOT / "src" / "program" / "generate.cpp").read_text(encoding="utf-8").replace("\r\n", "\n")
        for match in re.finditer(r"qwen35::|strata/qwen35/engine\.hpp", text):
            before = text[:match.start()]
            self.assertGreater(before.rfind("#if defined(STRATA_HAVE_QWEN35)"), before.rfind("#endif"), match.group(0))

    def test_the_f16_stub_is_in_the_gfx906_build(self):
        # 0.2.16: llama.cpp's MMQ (and with it the trellis overlay and the W4A4 unit) is out of the tree; strata_pf_own, Strata's own prompt
        # kernels, takes moe_f16_kq.cu and moe_p1a.cu on gfx906 (moe_fused_iq.cu calls them), and every .cu of src/ is compiled as HIP
        own = re.search(r"macro\(strata_cuda_pf_own\).*?endmacro\(\)", TEXT, re.S)
        self.assertIsNotNone(own)
        block = own.group(0)[own.group(0).index("if(STRATA_HIP_GFX906)"):]
        self.assertIn("target_sources(strata_pf_own PRIVATE src/prefill/moe_f16_kq.cu)", block)
        self.assertIn("target_sources(strata_pf_own PRIVATE src/prefill/moe_p1a.cu)", block)
        tail = HIP_TEXT[HIP_TEXT.index("# HIP gfx906: every CUDA source"):]
        self.assertIn("set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)", tail)
        self.assertNotIn("strata_mmq", TEXT.replace("strata_mmq_ids", ""))

if __name__ == "__main__":
    unittest.main()
