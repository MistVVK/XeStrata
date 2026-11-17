"""tools/image/gate/test_sampler_ref.py - TEST-ONLY: sampler_ref.py's euler step on numbers that can be done by hand.

    python -m pytest tools/image/gate/test_sampler_ref.py"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampler_ref as S  # noqa: E402


class Euler(unittest.TestCase):
    def test_by_hand(self):
        """x = 4, sigmas 1 -> 0.5 -> 0, velocity 4 each time: denoised = 0 both times, so x goes 4 -> 2 -> 0."""
        x = np.array([4.0, -4.0], dtype=np.float32)
        sig = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        v = np.array([4.0, -4.0], dtype=np.float32)
        out = S.euler_flow(x, [v, v], sig)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.tolist(), [0.0, -0.0])

    def test_zero_velocity_with_c_out_keeps_x_denoised(self):
        """v = 0: denoised = x, d = 0, x stays."""
        x = np.linspace(-3, 3, 11, dtype=np.float32)
        sig = S.sigmas_f32()
        out = S.euler_flow(x, [np.zeros_like(x)] * 8, sig)
        self.assertTrue(np.array_equal(out, x))

    def test_turbo_sigmas(self):
        sig = S.sigmas_f32()
        self.assertEqual(len(sig), 9)
        self.assertEqual((float(sig[0]), float(sig[-1])), (1.0, 0.0))
        self.assertTrue(all(sig[i] > sig[i + 1] for i in range(8)))
        from serve import images
        self.assertEqual(S.TURBO_SIGMAS, images.TURBO_SIGMAS)

    def test_self_test(self):
        S.self_test()


if __name__ == "__main__":
    unittest.main()
