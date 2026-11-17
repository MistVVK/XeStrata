"""tools/image/gate/test_te_ref.py - TEST-ONLY: te_ref.py's pieces against independent scalar forms of the same maths
(the GGUF block formats as include/strata/artifact/dequant.hpp writes them, the prompt-weight parser's documented cases),
and the encoder on a tiny random model (no model file needed).

    python -m pytest tools/image/gate/test_te_ref.py"""
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import te_ref as T  # noqa: E402

F = np.float32


def h2f(u):
    return F(np.frombuffer(struct.pack("<H", u), "<f2")[0])


def scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def scalar_q4_k(b):
    d, mn = h2f(b[0] | b[1] << 8), h2f(b[2] | b[3] << 8)
    sc, q, y, is_ = b[4:16], b[16:144], [], 0
    for j in range(0, 256, 64):
        s, m = scale_min_k4(is_, sc)
        d1, m1 = F(d * F(s)), F(mn * F(m))
        s, m = scale_min_k4(is_ + 1, sc)
        d2, m2 = F(d * F(s)), F(mn * F(m))
        qq = q[j // 2:j // 2 + 32]
        y += [F(F(d1 * F(x & 0xF)) - m1) for x in qq]
        y += [F(F(d2 * F(x >> 4)) - m2) for x in qq]
        is_ += 2
    return np.array(y, F)


def scalar_q5_k(b):
    d, mn = h2f(b[0] | b[1] << 8), h2f(b[2] | b[3] << 8)
    sc, qh, ql, y, is_, u1, u2 = b[4:16], b[16:48], b[48:176], [], 0, 1, 2
    for j in range(0, 256, 64):
        s, m = scale_min_k4(is_, sc)
        d1, m1 = F(d * F(s)), F(mn * F(m))
        s, m = scale_min_k4(is_ + 1, sc)
        d2, m2 = F(d * F(s)), F(mn * F(m))
        qq = ql[j // 2:j // 2 + 32]
        y += [F(F(d1 * F((qq[l] & 0xF) + (16 if qh[l] & u1 else 0))) - m1) for l in range(32)]
        y += [F(F(d2 * F((qq[l] >> 4) + (16 if qh[l] & u2 else 0))) - m2) for l in range(32)]
        is_ += 2
        u1, u2 = (u1 << 2) & 0xFF, (u2 << 2) & 0xFF
    return np.array(y, F)


def scalar_q6_k(b):
    ql, qh = b[0:128], b[128:192]
    sc = np.frombuffer(bytes(b[192:208]), np.int8)
    d = h2f(b[208] | b[209] << 8)
    y = [F(0)] * 256
    for n in range(2):
        L, H, S, o = ql[64 * n:], qh[32 * n:], sc[8 * n:], 128 * n
        for l in range(32):
            is_ = l // 16
            q1 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
            q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
            q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
            q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
            y[o + l] = F(F(d * F(S[is_])) * F(q1))
            y[o + l + 32] = F(F(d * F(S[is_ + 2])) * F(q2))
            y[o + l + 64] = F(F(d * F(S[is_ + 4])) * F(q3))
            y[o + l + 96] = F(F(d * F(S[is_ + 6])) * F(q4))
    return np.array(y, F)


def scalar_iq4_xs(b):
    d = h2f(b[0] | b[1] << 8)
    sh, sl, qs, y = b[2] | b[3] << 8, b[4:8], b[8:136], []
    for ib in range(8):
        ls = ((sl[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        dl = F(d * F(ls - 32))
        lo = [F(dl * T.KV_IQ4NL[qs[16 * ib + j] & 0xF]) for j in range(16)]
        hi = [F(dl * T.KV_IQ4NL[qs[16 * ib + j] >> 4]) for j in range(16)]
        y += lo + hi
    return np.array(y, F)


def random_blocks(rng, n, size, f16_at):
    b = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    for off in f16_at:                                  # finite, modest scales
        v = rng.uniform(-0.02, 0.02, n).astype(np.float16).view(np.uint8).reshape(n, 2)
        b[:, off:off + 2] = v
    return b


class Dequant(unittest.TestCase):
    def check(self, fn, scalar, size, f16_at):
        rng = np.random.default_rng(7)
        b = random_blocks(rng, 6, size, f16_at)
        got = fn(b.reshape(-1))
        want = np.concatenate([scalar(list(map(int, row))) for row in b])
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)))

    def test_q4_k(self):
        self.check(T.deq_q4_k, scalar_q4_k, 144, (0, 2))

    def test_q5_k(self):
        self.check(T.deq_q5_k, scalar_q5_k, 176, (0, 2))

    def test_q6_k(self):
        self.check(T.deq_q6_k, scalar_q6_k, 210, (208,))

    def test_iq4_xs(self):
        self.check(T.deq_iq4_xs, scalar_iq4_xs, 136, (0,))


class PromptWeights(unittest.TestCase):
    """The cases documented above sd.cpp's core/util.cpp parse_prompt_attention, as its code (not its comment) runs them."""

    def same(self, text, want):
        got = T.parse_prompt_attention(text)
        self.assertEqual([g[0] for g in got], [w[0] for w in want])
        for g, w in zip(got, want):
            self.assertAlmostEqual(float(g[1]), w[1], places=5)

    def test_cases(self):
        self.same("normal text", [["normal text", 1.0]])
        self.same("an (important) word", [["an ", 1.0], ["important", 1.1], [" word", 1.0]])
        self.same("(unbalanced", [["unbalanced", 1.1]])
        # the comment above sd.cpp's function says "(literal]"; its code unescapes only "\(" and keeps "\]" as text
        self.same("\\(literal\\]", [["(literal\\]", 1.0]])
        self.same("(unnecessary)(parens)", [["unnecessaryparens", 1.1]])
        self.same("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).",
                  [["a ", 1.0], ["house", 1.573], [" ", 1.1], ["on", 1.0], [" a ", 1.1], ["hill", 0.55],
                   [", sun, ", 1.1], ["sky", 1.4641], [".", 1.1]])

    def test_gate_prompts_are_plain(self):
        from run_set import PROMPTS
        for p in PROMPTS:
            self.assertEqual(T.parse_prompt_attention(p), [[p, np.float32(1.0)]])


class Tokens(unittest.TestCase):
    def test_bpe_and_specials(self):
        bu = T.bytes_to_unicode()
        sp = bu[ord(" ")]
        tok = T.Qwen2Tok(["h e", "l l", "he ll", f"{sp} w", "o r"])
        ids = tok.encode("<|im_start|>hello world<|im_end|>")
        base = 256 + 5
        self.assertEqual(ids[0], base + 1)
        self.assertEqual(ids[-1], base + 2)
        self.assertEqual(tok.vocab[ids[1]], "hell")
        self.assertEqual("".join(tok.vocab[i] for i in ids[1:-1]), "hello" + sp + "world")

    def test_split_specials(self):
        self.assertEqual(T.split_with_special_tokens("a<|im_end|>\nb", T.SPECIALS), ["a", "<|im_end|>", "\nb"])


class TinyModel(unittest.TestCase):
    """The layer on random weights: causal (token t sees only 0..t), f16op close to f32, NEOX rope = rotation."""

    def setUp(self):
        rng = np.random.default_rng(3)
        C, Hh, KV, D, Ff = 64, 4, 2, 16, 96

        class W:
            n_head, n_kv, hd, theta, eps = Hh, KV, D, 5e6, 1e-6

        class L:
            pass
        self.W = W
        L.attn_norm = rng.uniform(0.5, 1.5, C).astype(F)
        L.ffn_norm = rng.uniform(0.5, 1.5, C).astype(F)
        L.qn = rng.uniform(0.5, 1.5, D).astype(F)
        L.kn = rng.uniform(0.5, 1.5, D).astype(F)
        mk = lambda o, i: (rng.standard_normal((o, i)) / math.sqrt(i)).astype(F).astype(np.float64)  # noqa: E731
        L.q, L.k, L.v, L.o = mk(Hh * D, C), mk(KV * D, C), mk(KV * D, C), mk(C, Hh * D)
        L.g, L.u, L.d = mk(Ff, C), mk(Ff, C), mk(C, Ff)
        self.L = L
        self.x = rng.standard_normal((9, C)).astype(F)

    def test_causal(self):
        full = T.layer_forward(self.W, self.L, self.x, False, None, 0)
        part = T.layer_forward(self.W, self.L, self.x[:5], False, None, 0)
        self.assertTrue(np.allclose(full[:5], part, rtol=1e-6, atol=1e-6))

    def test_f16op_close(self):
        a = T.layer_forward(self.W, self.L, self.x, False, None, 0)
        b = T.layer_forward(self.W, self.L, self.x, True, {}, 0)
        self.assertLess(T.compare(b, a)["rel_l2"], 5e-3)

    def test_rope_is_rotation(self):
        x = np.random.default_rng(1).standard_normal((6, 2, 16)).astype(F)
        r = T.rope_neox(x, 5e6)
        self.assertTrue(np.allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5))
        self.assertTrue(np.array_equal(r[0], x[0]))


if __name__ == "__main__":
    unittest.main()
