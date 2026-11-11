"""tools/image/gate/test_dit_step_check.py - TEST-ONLY: dit_step_check.py on a fake folder (no model, no GPU).

    python -m pytest tools/image/gate/test_dit_step_check.py"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit_step_check as K  # noqa: E402


def put(path: Path, a: np.ndarray, npy: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if npy:
        np.save(path, a)
    else:
        a.astype("<f4").tofile(path)


class StepCheck(unittest.TestCase):
    def test_table_and_rule(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as td:
            root, ours = Path(td) / "own5b", Path(td) / "ours"
            for p, scale in ((0, 1.0), (5, 3.0)):
                P = K.case_paths(root, f"512 p{p}")
                ref = rng.standard_normal((64, 4, 4)).astype(np.float32)
                noise = lambda s: ref + s * rng.standard_normal(ref.shape).astype(np.float32)  # noqa: E731
                put(Path(str(P["ref"]) + "_f32.npy"), ref, True)
                put(Path(str(P["ref"]) + "_f16op.npy"), noise(1e-3), True)
                put(P["acc32"], noise(1e-3 * scale), True)
                put(P["stock"], noise(1e-2))
                put(P["native"], noise(1e-2))
                put(P["d1d2"], noise(1e-3))
                put(ours / P["dump"].parent.name / P["dump"].name / "vel.bin", noise(1e-3 * scale))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                K.main(["--root", str(root), "--ours", str(ours)])
            text = out.getvalue()
            self.assertIn("| 512 p0 |", text)
            self.assertIn("| 512 p5 |", text)
            f32_rule = next(line for line in text.splitlines() if line.startswith("RULE vs the float32"))
            exact_rule = next(line for line in text.splitlines() if line.startswith("RULE vs the exact"))
            self.assertIn("(iii) PASS", f32_rule)                 # 1e-3 vs 1e-2: ours closer than stock and native
            self.assertIn("(ii) largest", f32_rule)
            self.assertIn("FAIL", exact_rule)                     # p5 ours ~3x the exact-sum emulation

    def test_case_paths(self):
        P = K.case_paths(Path("/r"), "1024 p0")
        self.assertEqual(P["acc32"].name, "1024_p0_acc32.npy")
        self.assertEqual(K.case_paths(Path("/r"), "512 p6")["d1d2"].parent.parent.name, "z_ours_512")
        self.assertEqual(K.case_paths(Path("/r"), "512 p2")["d1d2"].parent.parent.name, "d2_g15")


if __name__ == "__main__":
    unittest.main()
