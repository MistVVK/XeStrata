"""tools/test_hip_compat.py - the AMD build compiles the CUDA-shaped sources through include/strata/hip_compat/: every
CUDA runtime / cuBLAS name it compiles must have a mapping there (v1.0.21 shipped three without, and no AMD build
compiled).  No ROCm or GPU needed.

    python -m unittest tools.test_hip_compat
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_hip_compat as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class HipCompat(unittest.TestCase):
    def test_every_cuda_name_the_hip_build_compiles_is_mapped(self):
        bad = C.unmapped(REPO)
        self.assertEqual(bad, [], "\n".join(f"{f}:{n}: {name} - add it to include/strata/hip_compat/"
                                            for f, n, name in bad))

    def test_the_check_sees_what_it_must(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            compat = root / "include" / "strata" / "hip_compat"
            compat.mkdir(parents=True)
            (compat / "cuda_runtime.h").write_text("#define cudaMalloc hipMalloc\n#define cudaFree hipFree\n")
            (root / "src").mkdir()
            (root / "src" / "a.cu").write_text(
                "void f() {\n"
                "  cudaMalloc(&p, 4);                       // mapped\n"
                "  cudaDeviceGetPCIBusId(b, 16, 0);         // not: the AMD build breaks\n"
                "  puts(\"cudaMemcpyPeer failed\");         // a string\n"
                "#ifndef STRATA_USE_HIP\n"
                "  cudaMemcpyPeerAsync(a, 1, b, 0, n, s);   // CUDA only\n"
                "#endif\n"
                "#if defined(STRATA_USE_HIP)\n"
                "  hipFree(p);\n"
                "#else\n"
                "  cudaFuncGetAttributes(&x, k);            // CUDA only: the #else of a HIP block\n"
                "#endif\n"
                "}\n")
            self.assertEqual([(n, name) for _, n, name in C.unmapped(root)], [(3, "cudaDeviceGetPCIBusId")])


if __name__ == "__main__":
    unittest.main()
