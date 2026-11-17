#!/usr/bin/env python3
"""tools/dflash2_pack.py reads a .safetensors file with numpy alone (the person's PC has neither torch nor the safetensors package) and writes
the same runtime file the torch-based version wrote.

    python tools/test_dflash2_pack.py
"""
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dflash2_pack as new  # noqa: E402

L, H, INTER, HEADS, KV, HD, RANK, VOCAB = 2, 64, 96, 2, 1, 32, 32, 128
CFG = {"num_hidden_layers": L, "hidden_size": H, "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
       "intermediate_size": INTER, "sliding_window": 128, "rope_parameters": {"rope_theta": 1000000.0}, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
       "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 7, "selector_rank": RANK, "selector_top_k": 4,
                         "target_layer_ids": [1, 2, 3]}}


def bf16(a):
    """float32 -> BF16 bytes by truncation (what torch's .to(bfloat16) does for these values up to rounding; the test only needs a valid BF16 file)."""
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def tensors():
    rng = np.random.default_rng(7)
    r = lambda *shape: rng.standard_normal(shape).astype(np.float32) * 0.1
    t = {"fc.weight": r(H, H * 3), "hidden_norm.weight": r(H), "norm.weight": r(H)}
    for l in range(L):
        p = "layers.%d." % l
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
    """A BF16 safetensors file written by hand (the format is documented: u64 header length, JSON header, raw data)."""
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


class PackWithoutTorch(unittest.TestCase):
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

    def pack(self, out):
        return new.pack(str(self.dir / "model.safetensors"), str(self.dir / "config.json"), str(out), say=lambda *_: None)

    def test_reader_widens_bf16_exactly(self):
        f = new.SafeTensors(str(self.dir / "model.safetensors"))
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
        self.assertTrue(any(l.startswith("cfg hidden %d" % H) for l in index))
        self.assertIn("end", [l.strip() for l in index])
        names = [l.split()[1] for l in index if l.startswith("t ")]
        self.assertIn("fc.weight", names)
        self.assertIn("candidate_selector.successor_codebook", names)
        self.assertEqual(len(data), size)

    def test_a_matrix_is_ggml_q8_0_of_the_widened_values(self):
        out = self.dir / "q8.pack"
        self.pack(out)
        data = out.read_bytes()
        index_len = struct.unpack("<I", data[8:12])[0]
        base = 12 + index_len
        row = next(l.split() for l in data[12:base].decode().splitlines() if l.startswith("t fc.weight "))
        _t, _name, dtype, rows, cols, off, nbytes = row
        self.assertEqual(dtype, "q8_0")
        got = data[base + int(off): base + int(off) + int(nbytes)]
        widened = (bf16(self.arrays["fc.weight"]).astype(np.uint32) << 16).view(np.float32)
        self.assertEqual(got, new.quant_q8_0(widened))

    def test_q4_0_blocks_are_ggml_q4_0_and_only_the_matrices_change(self):
        q8, q4 = self.dir / "q8b.pack", self.dir / "q4.pack"
        self.pack(q8)
        new.pack(str(self.dir / "model.safetensors"), str(self.dir / "config.json"), str(q4), say=lambda *_: None, q4=True)
        rows = {}
        for path in (q8, q4):
            data = path.read_bytes()
            index_len = struct.unpack("<I", data[8:12])[0]
            rows[path] = (data, 12 + index_len, {l.split()[1]: l.split() for l in data[12:12 + index_len].decode().splitlines() if l.startswith("t ")})
        d8, b8, t8 = rows[q8]
        d4, b4, t4 = rows[q4]
        self.assertEqual(set(t8), set(t4))
        for name, r in t4.items():
            dtype4, nbytes4 = r[2], int(r[6])
            if t8[name][2] == "q8_0":
                self.assertEqual(dtype4, "q4_0", name)
                self.assertEqual(nbytes4, int(r[3]) * int(r[4]) // 32 * 18, name)
            else:   # norms, conv kernels and the codebooks are byte for byte the same
                self.assertEqual(d4[b4 + int(r[5]): b4 + int(r[5]) + nbytes4], d8[b8 + int(t8[name][5]): b8 + int(t8[name][5]) + int(t8[name][6])], name)
        # one matrix: dequantize the blocks and check the error is within one step of each block scale (the +8 end is clamped to 7)
        w = (bf16(self.arrays["fc.weight"]).astype(np.uint32) << 16).view(np.float32)
        r = t4["fc.weight"]
        blocks = np.frombuffer(d4[b4 + int(r[5]): b4 + int(r[5]) + int(r[6])], np.uint8).reshape(w.shape[0], w.shape[1] // 32, 18)
        d = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
        q = np.concatenate([blocks[..., 2:] & 15, blocks[..., 2:] >> 4], axis=-1).astype(np.float32) - 8
        deq = (q * d[..., None]).reshape(w.shape)
        step = np.abs(d)[..., None].repeat(32, -1).reshape(w.shape)
        self.assertTrue(np.all(np.abs(deq - w) <= 1.01 * step + 1e-6))

    def test_it_matches_the_torch_version_when_torch_is_here(self):
        try:
            import torch  # noqa: F401
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("torch or safetensors is not installed here (the point of the change)")
        old_src = subprocess.run(["git", "show", "f1b6a2144:tools/dflash2_pack.py"], cwd=HERE, capture_output=True, text=True)
        if old_src.returncode != 0:
            self.skipTest("the earlier version is not in this clone")
        old_path = self.dir / "old_pack.py"
        old_path.write_text(old_src.stdout)
        spec = importlib.util.spec_from_file_location("old_pack", old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        a, b = self.dir / "a.pack", self.dir / "b.pack"
        sys.argv = ["x", str(self.dir / "model.safetensors"), str(self.dir / "config.json"), str(a)]
        old.main()
        self.pack(b)
        self.assertEqual(a.read_bytes(), b.read_bytes(), "byte for byte the same file")


if __name__ == "__main__":
    unittest.main()
