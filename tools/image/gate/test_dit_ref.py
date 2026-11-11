"""tools/image/gate/test_dit_ref.py - TEST-ONLY: dit_ref.py's pieces against their definitions (no model file needed).

    python -m pytest tools/image/gate/test_dit_ref.py"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit_ref as R  # noqa: E402


class Layout(unittest.TestCase):
    def test_positions(self):
        p = R.layout_positions(3, 2, 3)
        self.assertEqual(p.shape, (3 + 6, 3))
        self.assertEqual(p[:3].tolist(), [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        # image tokens: (text length, h - (H - H/2), w - (W - W/2)), token h*W + w
        self.assertEqual(p[3].tolist(), [3, -1, -2])
        self.assertEqual(p[3 + 5].tolist(), [3, 0, 0])

    def test_rope_table_and_rotation(self):
        pos = R.layout_positions(4, 4, 4)
        c, s = R.rope_table(pos)
        self.assertEqual(c.shape, (4 + 16, 64))
        self.assertTrue(np.all(c[0] == 1) and np.all(s[0] == 0))          # position 0: no rotation
        x = np.random.default_rng(2).standard_normal((20, 3, 128)).astype(np.float32)
        y = R.apply_rope(x, c, s)
        self.assertTrue(np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5))
        self.assertTrue(np.array_equal(y[0], x[0]))


class Pieces(unittest.TestCase):
    def test_timestep_embedding(self):
        e = R.timestep_embedding(np.array([0.0, 500.0], dtype=np.float32))
        self.assertTrue(np.all(e[0, :128] == 1) and np.all(e[0, 128:] == 0))
        j = 7
        f = math.exp(-math.log(10000) * j / 128)
        self.assertAlmostEqual(float(e[1, j]), math.cos(500 * f), places=5)
        self.assertAlmostEqual(float(e[1, 128 + j]), math.sin(500 * f), places=5)

    def test_norms_and_activations(self):
        x = np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32) * 3 + 1
        y = R.layer_norm(x)
        self.assertTrue(np.allclose(y.mean(-1), 0, atol=1e-5) and np.allclose(y.std(-1), 1, atol=1e-3))
        self.assertAlmostEqual(float(R.gelu_tanh(np.float32([1.0]))[0]), 0.8411920, places=5)
        self.assertAlmostEqual(float(R.silu(np.float32([1.0]))[0]), 1 / (1 + math.exp(-1)), places=6)

    def test_attention_mask(self):
        rng = np.random.default_rng(4)
        q, k, v = (rng.standard_normal((7, 2, 128)).astype(np.float32) for _ in range(3))
        o = R.attention(q, k, v, 3, False)
        self.assertTrue(np.allclose(o[0], v[0], atol=1e-6))                 # first text token sees only itself
        o2 = R.attention(q[:3], k[:3], v[:3], 3, False)                     # text rows ignore the image tokens
        self.assertTrue(np.allclose(o[:3], o2, atol=1e-6))
        k2 = k.copy()
        k2[5] += 1                                                          # image rows see every token
        self.assertFalse(np.allclose(R.attention(q, k2, v, 3, False)[6], o[6]))


if __name__ == "__main__":
    unittest.main()
