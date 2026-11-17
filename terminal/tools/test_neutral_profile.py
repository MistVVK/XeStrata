"""A feed model of another shape than the shipped expert profile gets a neutral profile (found by the Q4-P8 rehearsal: the engine refused the shipped 48x512 one for a 4x256 model)."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strata_control import setup_config  # noqa: E402


class Fake:
    def __init__(self, layers, experts):
        self.metadata = {"qwen4exp.block_count": layers, "qwen4exp.expert_count": experts, "general.name": "x"}

    def __call__(self, path):
        return self


class NeutralProfile(unittest.TestCase):
    def test_the_shipped_shape_keeps_the_shipped_profile(self):
        shipped = ROOT / "data" / "expert-profile.bin"
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(setup_config.neutral_profile_if_other_shape(shipped, "x.gguf", d, Fake(48, 512)), shipped)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_another_shape_gets_every_pair_once_in_the_engines_format(self):
        shipped = ROOT / "data" / "expert-profile.bin"
        with tempfile.TemporaryDirectory() as d:
            said = []
            out = setup_config.neutral_profile_if_other_shape(shipped, "x.gguf", d, Fake(4, 256), said.append)
            self.assertEqual(out.name, "expert-profile-4x256.bin")
            blob = out.read_bytes()
            self.assertEqual(blob[:4], b"STRP")
            ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
            self.assertEqual((ver, nl, ne, slots, n), (1, 4, 256, 1024, 1024))
            pairs = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
            self.assertEqual(len(set(pairs)), 1024)
            self.assertEqual(pairs[:5], [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)])          # interleaved across the layers
            self.assertEqual(len(blob), 24 + 4 * n + 4 * nl * ne)
            self.assertEqual(said, ["  This model's shape differs from Flash-Next, so a neutral expert profile is used."])      # the owner's wording


if __name__ == "__main__":
    unittest.main()
