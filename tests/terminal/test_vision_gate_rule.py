"""The picture reader's GPU gate reads our tower against the float32 reference (VitRef), not against llama.cpp's reader.

tools/vision/own/gate_gpu.py: rule 1 = cosine(own GPU, float32) >= 0.999, rule 2 = no further from float32 than mtmd's GPU reader (1e-4 slack, or >= 0.9999).
The numbers are code4217.png at 1,024 picture tokens, measured on 11-14 (RTX PRO 5000 and a Zen 3 CPU, Qwen3.8 BF16 mmproj).
"""

import importlib.util
import unittest
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "tools" / "vision" / "own" / "gate_gpu.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("gate_gpu", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VisionGateRuleTest(unittest.TestCase):
    def setUp(self):
        self.g = load_gate()

    def test_code4217_float32_is_the_reference(self):
        mtmd_gpu = 0.999029                      # mtmd's GPU reader against float32 on code4217 (the floor)
        self.assertTrue(self.g.gate_pass(1.000000, mtmd_gpu))   # our tower without input rounding equals float32
        self.assertFalse(self.g.gate_pass(0.998392, mtmd_gpu))  # our CUDA tower with BF16 inputs: under rule 1 against float32
        self.assertFalse(self.g.gate_pass(0.998312, mtmd_gpu))  # our CPU tower with the old BF16 rounding default: under rule 1

    def test_rule_two_against_the_mtmd_floor(self):
        self.assertTrue(self.g.gate_pass(0.99920, 0.99925))     # inside the 1e-4 slack
        self.assertFalse(self.g.gate_pass(0.99920, 0.99940))    # further from float32 than mtmd's GPU reader
        self.assertTrue(self.g.gate_pass(0.99990, 1.0))         # 0.9999 passes whatever the floor

    def test_largest_pictures_compare_the_gpu_readers(self):
        self.assertTrue(self.g.gate_pass(0.995, 1.0, skip_cpu=True))
        self.assertFalse(self.g.gate_pass(0.985, 1.0, skip_cpu=True))

    def test_mtmd_is_information_only(self):
        src = GATE.read_text(encoding="utf-8")
        self.assertIn('"STRATA_VIT_CPU": "ref"', src)            # the reference reader the gate starts where --ref-dir lacks a picture
        self.assertIn("own/mtmdC", src)                          # mtmd's CPU path stays a printed column


if __name__ == "__main__":
    unittest.main()
