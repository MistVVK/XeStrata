"""tools/image/gate/test_vae_ref.py - TEST-ONLY: vae_ref.py's pieces against independent forms of the same maths, and
the whole decoder on a tiny random model (no model file needed).

    python -m pytest tools/image/gate/test_vae_ref.py"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_ref as V  # noqa: E402


def dup_up_torch_form(x_chw, out_c, ft, fs=2):
    """DupUp3D the way the Wan2.2 Python code writes it, for one frame:
    repeat_interleave(repeats, dim=channel) -> view(out_c, ft, fs, fs, T, H, W) -> permute(out_c, T, ft, H, fs, W, fs)
    -> view(out_c, T*ft, H*fs, W*fs) -> keep frames from ft-1 (first chunk)."""
    C, H, W = x_chw.shape
    repeats = out_c * ft * fs * fs // C
    x = np.repeat(x_chw[:, None], 1, axis=1)                      # [C, T=1, H, W]
    x = np.repeat(x, repeats, axis=0)                              # repeat_interleave over channels
    x = x.reshape(out_c, ft, fs, fs, 1, H, W).transpose(0, 4, 1, 5, 2, 6, 3)
    x = x.reshape(out_c, ft, H * fs, W * fs)
    return x[:, ft - 1]                                            # [out_c, 2H, 2W]


class Pieces(unittest.TestCase):
    def test_dup_up_matches_the_torch_form(self):
        rng = np.random.default_rng(1)
        for c, out_c, ft in [(8, 8, 2), (8, 4, 2), (8, 4, 1), (12, 6, 1)]:
            x = rng.standard_normal((3, 5, c)).astype(np.float32)
            got = V.Decoder.dup_up(x, out_c, ft)
            want = dup_up_torch_form(x.transpose(2, 0, 1), out_c, ft).transpose(1, 2, 0)
            self.assertTrue(np.array_equal(got, want), (c, out_c, ft))

    def test_conv3x3_matches_direct_loops(self):
        rng = np.random.default_rng(2)
        H, W, C, O = 5, 6, 3, 4
        w = rng.standard_normal((O, C, 3, 3)).astype(np.float32)
        b = rng.standard_normal(O).astype(np.float32)
        x = rng.standard_normal((H, W, C)).astype(np.float32)
        dec = V.Decoder({"c.weight": w, "c.bias": b})
        got = dec.conv("c", x, rows_per_chunk=2)
        pad = np.zeros((H + 2, W + 2, C), np.float64)
        pad[1:-1, 1:-1] = x
        want = np.zeros((H, W, O))
        for y in range(H):
            for xx in range(W):
                for o in range(O):
                    want[y, xx, o] = np.sum(pad[y:y + 3, xx:xx + 3, :] * w[o].transpose(1, 2, 0)) + b[o]
        self.assertLess(np.abs(got - want).max(), 1e-5)

    def test_rms_norm_by_hand(self):
        dec = V.Decoder({"n.gamma": np.array([[[1.0]], [[2.0]]], np.float32)})
        out = dec.rms("n", np.array([[[3.0, 4.0]]], np.float32))      # rms = sqrt(12.5)
        r = np.sqrt(12.5)
        self.assertTrue(np.allclose(out, [[[3 / r, 8 / r]]], rtol=1e-6))

    def test_to_u8(self):
        u = V.to_u8(np.array([-2.0, -1.0, 0.0, 1.0, 3.0, 0.5], np.float32))
        self.assertEqual(u.tolist(), [0, 0, 128, 255, 255, 191])

    def test_f16op_refuses_inputs_beyond_f16(self):
        dec = V.Decoder({"c.weight": np.ones((1, 1, 1, 1), np.float32), "c.bias": np.zeros(1, np.float32)}, mode="f16op")
        with self.assertRaises(OverflowError):
            dec.conv("c", np.full((1, 1, 1), 70000.0 * 128, np.float32))

    def test_shift_table_matches_the_decoder(self):
        """vae_ref's CONV_SHIFTS is the table of src/image/vae/wan_vae.cu conv_shift(), so "f16op" models our operands."""
        import re
        cu = (Path(__file__).resolve().parents[3] / "src" / "image" / "vae" / "wan_vae.cu").read_text(encoding="utf-8")
        body = cu[cu.index("int conv_shift(const std::string& name) {"):]
        body = body[:body.index("\n}\n")]
        table = {m.group(1): int(m.group(2)) for m in re.finditer(r'if \(name == "([^"]+)"\) return (\d+);', body)}
        self.assertEqual(table, V.CONV_SHIFTS)
        self.assertIn("return CONV_SHIFT;", body)
        self.assertIn("constexpr int CONV_SHIFT = 7;", cu)

    def test_bf16_reader(self):
        vals = np.array([1.0, -2.5, 0.15625], np.float32)
        raw = (vals.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        hdr = json.dumps({"t": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(raw)]}}).encode()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.safetensors"
            p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + raw)
            self.assertEqual(V.load_safetensors(p)["t"].tolist(), vals.tolist())


def tiny_model(rng, dims=(16, 16, 16, 8, 4, 2), z=64):
    w = {}

    def conv(name, o, i, k):
        w[name + ".weight"] = (rng.standard_normal((o, i, k, k)) / np.sqrt(i * k * k)).astype(np.float32)
        w[name + ".bias"] = (0.1 * rng.standard_normal(o)).astype(np.float32)

    def norm(name, c):
        w[name + ".gamma"] = (1 + 0.1 * rng.standard_normal((c, 1, 1, 1))).astype(np.float32)

    def resnet(p, i, o):
        norm(p + ".norm1", i)
        conv(p + ".conv1", o, i, 3)
        norm(p + ".norm2", o)
        conv(p + ".conv2", o, o, 3)
        if i != o:
            conv(p + ".conv_shortcut", o, i, 1)

    conv("post_quant_conv", z, z, 1)
    conv("decoder.conv_in", dims[0], z, 3)
    resnet("decoder.mid_block.resnets.0", dims[0], dims[0])
    norm("decoder.mid_block.attentions.0.norm", dims[0])
    conv("decoder.mid_block.attentions.0.to_qkv", 3 * dims[0], dims[0], 1)
    conv("decoder.mid_block.attentions.0.proj", dims[0], dims[0], 1)
    resnet("decoder.mid_block.resnets.1", dims[0], dims[0])
    for i in range(5):
        cin, cout = dims[i], dims[i + 1]
        for j in range(3):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", cin if j == 0 else cout, cout)
        if V.UP_BLOCKS[i][3]:
            conv(f"decoder.up_blocks.{i}.upsampler.resample.1", cout, cout, 3)
    norm("decoder.norm_out", dims[-1])
    conv("decoder.conv_out", 4, dims[-1], 3)
    return w


class TinyDecoder(unittest.TestCase):
    def test_shapes_and_modes(self):
        rng = np.random.default_rng(3)
        w = tiny_model(rng)
        lat = rng.standard_normal((64, 2, 3)).astype(np.float32)
        a = V.Decoder(w, "f32").decode(lat)
        b = V.Decoder(w, "f16op").decode(lat)
        self.assertEqual(a.shape, (32, 48, 4))
        self.assertTrue(np.isfinite(a).all())
        rel = np.linalg.norm(a - b) / np.linalg.norm(a)
        self.assertLess(rel, 1e-2)          # f16 operands move the result a little, not a lot
        self.assertGreater(rel, 0.0)

    def test_stage_dump(self):
        rng = np.random.default_rng(4)
        d = {}
        V.Decoder(tiny_model(rng), "f32").decode(rng.standard_normal((64, 2, 2)).astype(np.float32), dump=d)
        self.assertEqual([d[k].shape[:2] for k in ("conv_in", "up0", "up3", "up4", "out")],
                         [(2, 2), (4, 4), (32, 32), (32, 32), (32, 32)])


if __name__ == "__main__":
    unittest.main()
