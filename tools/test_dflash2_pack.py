# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/dflash2_pack.py: the pack's header, the ggml quantization of the widened values, and the selector conversion.

    python -m unittest tools.test_dflash2_pack
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash2_pack as pk  # noqa: E402

L, H, INTER, HEADS, KV, HD, RANK, VOCAB = 2, 64, 96, 2, 1, 32, 32, 128
CFG = {"num_hidden_layers": L, "hidden_size": H, "num_attention_heads": HEADS, "num_key_value_heads": KV,
       "head_dim": HD, "intermediate_size": INTER, "sliding_window": 128,
       "rope_parameters": {"rope_theta": 1000000.0}, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
       "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 7,
                         "selector_rank": RANK, "selector_top_k": 4, "target_layer_ids": [1, 2, 3]}}


def bf16(a):
    """float32 -> BF16 bits by truncation (a valid BF16 file for the reader; the widening is exact)."""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def tensors():
    rng = np.random.default_rng(7)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1
    t = {"fc.weight": r(H, H * 3), "hidden_norm.weight": r(H), "norm.weight": r(H)}
    for ly in range(L):
        p = "layers.%d." % ly
        t[p + "self_attn.q_proj.weight"] = r(HEADS * HD, H)
        t[p + "self_attn.k_proj.weight"] = r(KV * HD, H)
        t[p + "self_attn.v_proj.weight"] = r(KV * HD, H)
        t[p + "self_attn.o_proj.weight"] = r(H, HEADS * HD)
        t[p + "mlp.gate_proj.weight"] = r(INTER, H)
        t[p + "mlp.up_proj.weight"] = r(INTER, H)
        t[p + "mlp.down_proj.weight"] = r(H, INTER)
        t[p + "attention_conv.kernel_projection.weight"] = r(32, H)
        t[p + "mlp_conv.kernel_projection.weight"] = r(32, H)
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            t[p + n + ".weight"] = r(HD if "_norm" in n else H)
        t[p + "attention_conv.base_kernel"] = r(2, 16)
        t[p + "mlp_conv.base_kernel"] = r(2, 16)
    t["candidate_selector.hidden_projection.weight"] = r(RANK, H)
    t["candidate_selector.predecessor_codebook"] = r(VOCAB, RANK)
    t["candidate_selector.successor_codebook"] = r(VOCAB, RANK)
    return t


def write_safetensors(path, arrays):
    """A BF16 safetensors file written by hand (u64 header length, JSON header, raw data)."""
    header, blobs, at = {}, [], 0
    for name, a in arrays.items():
        data = bf16(a).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    js = json.dumps(header).encode()
    js += b" " * ((8 - len(js) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def index_of(data):
    index_len = struct.unpack("<I", data[8:12])[0]
    base = 12 + index_len
    rows = {ln.split()[1]: ln.split() for ln in data[12:base].decode().splitlines() if ln.startswith("t ")}
    return base, rows


class PackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.arrays = tensors()
        write_safetensors(cls.dir / "model.safetensors", cls.arrays)
        (cls.dir / "config.json").write_text(json.dumps(CFG))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pack(self, out, **kw):
        return pk.pack(str(self.dir / "model.safetensors"), str(self.dir / "config.json"), str(out),
                       say=lambda *_: None, **kw)

    def test_reader_widens_bf16_exactly(self):
        f = pk.SafeTensors(str(self.dir / "model.safetensors"))
        got = f.get_float32("fc.weight")
        want = (bf16(self.arrays["fc.weight"]).astype(np.uint32) << 16).view(np.float32)
        self.assertTrue(np.array_equal(got, want))
        self.assertEqual(f.get_bf16_bits("candidate_selector.predecessor_codebook").dtype, np.uint16)
        self.assertEqual(f.raw("norm.weight")[0], "BF16")

    def test_the_file_has_the_header_the_engine_reads(self):
        out = self.dir / "out.pack"
        size = self.pack(out)
        data = out.read_bytes()
        self.assertEqual(data[:4], b"DF2P")
        version, index_len = struct.unpack("<II", data[4:12])
        self.assertEqual(version, 1)
        self.assertEqual((12 + index_len) % 4096, 0, "the data starts on a 4096 multiple")
        index = data[12:12 + index_len].decode().splitlines()
        self.assertTrue(any(ln.startswith("cfg hidden %d" % H) for ln in index))
        self.assertTrue(any(ln.startswith("cfg taps 1,2,3") for ln in index))
        self.assertIn("end", [ln.strip() for ln in index])
        names = [ln.split()[1] for ln in index if ln.startswith("t ")]
        self.assertIn("fc.weight", names)
        self.assertIn("candidate_selector.successor_codebook", names)
        self.assertEqual(len(data), size)

    def test_a_matrix_is_ggml_q8_0_of_the_widened_values(self):
        out = self.dir / "q8.pack"
        self.pack(out)
        data = out.read_bytes()
        base, rows = index_of(data)
        _t, _name, dtype, _rows, _cols, off, nbytes = rows["fc.weight"]
        self.assertEqual(dtype, "q8_0")
        got = data[base + int(off): base + int(off) + int(nbytes)]
        widened = (bf16(self.arrays["fc.weight"]).astype(np.uint32) << 16).view(np.float32)
        self.assertEqual(got, pk.quant_q8_0(widened))

    def test_q4_0_blocks_are_ggml_q4_0_and_only_the_matrices_change(self):
        q8, q4 = self.dir / "q8b.pack", self.dir / "q4.pack"
        self.pack(q8)
        self.pack(q4, q4=True)
        d8, d4 = q8.read_bytes(), q4.read_bytes()
        b8, t8 = index_of(d8)
        b4, t4 = index_of(d4)
        self.assertEqual(set(t8), set(t4))
        for name, r in t4.items():
            dtype4, nbytes4 = r[2], int(r[6])
            if t8[name][2] == "q8_0":
                self.assertEqual(dtype4, "q4_0", name)
                self.assertEqual(nbytes4, int(r[3]) * int(r[4]) // 32 * 18, name)
            else:   # norms, conv kernels and the codebooks are byte for byte the same
                self.assertEqual(d4[b4 + int(r[5]): b4 + int(r[5]) + nbytes4],
                                 d8[b8 + int(t8[name][5]): b8 + int(t8[name][5]) + int(t8[name][6])], name)
        # one matrix: dequantize the blocks; the error is within one step of each block scale (the +8 end clamps to 7)
        w = (bf16(self.arrays["fc.weight"]).astype(np.uint32) << 16).view(np.float32)
        r = t4["fc.weight"]
        raw = d4[b4 + int(r[5]): b4 + int(r[5]) + int(r[6])]
        blocks = np.frombuffer(raw, np.uint8).reshape(w.shape[0], w.shape[1] // 32, 18)
        d = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
        q = np.concatenate([blocks[..., 2:] & 15, blocks[..., 2:] >> 4], axis=-1).astype(np.float32) - 8
        deq = (q * d[..., None]).reshape(w.shape)
        step = np.abs(d)[..., None].repeat(32, -1).reshape(w.shape)
        self.assertTrue(np.all(np.abs(deq - w) <= 1.01 * step + 1e-6))

    def test_selector_conversion_is_byte_identical_to_packing_with_q4_selector(self):
        a, b, c = self.dir / "sel_a.pack", self.dir / "sel_b.pack", self.dir / "sel_c.pack"
        self.pack(a, q4=True)
        pk.convert_selector_q4(str(a), str(b), say=lambda *_: None)
        self.pack(c, q4=True, q4_selector=True)
        self.assertEqual(b.read_bytes(), c.read_bytes(), "the conversion writes the pack --q4_0-selector writes")
        # converting twice is refused (the codebooks are no longer bf16)
        with self.assertRaises(ValueError):
            pk.convert_selector_q4(str(b), str(self.dir / "sel_d.pack"), say=lambda *_: None)

    def test_read_index_round_trip(self):
        out = self.dir / "idx.pack"
        self.pack(out, q4=True)
        cfg_lines, tensors, base = pk.read_index(str(out))
        self.assertTrue(any(ln == "cfg hidden %d" % H for ln in cfg_lines))
        self.assertEqual(base % 4096, 0)
        fc = next(t for t in tensors if t["name"] == "fc.weight")
        self.assertEqual(fc["dtype"], "q4_0")
        self.assertEqual(fc["bytes"], fc["rows"] * fc["cols"] // 32 * 18)


if __name__ == "__main__":
    unittest.main()
