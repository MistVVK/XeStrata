#!/usr/bin/env python3
"""Pack the incoai DFlash2 drafter for Qwen3.8-Flash-Next (safetensors) into the engine's runtime file.

    python tools/dflash2_pack.py model.safetensors config.json out.pack [--q4_0] [--q4_0-selector]
    python tools/dflash2_pack.py --convert-selector in.pack out.pack      (a finished Q4_0 pack: its bf16 selector codebooks become Q4_0, nothing else changes)

Format (little endian): "DF2P", u32 version (1), u32 index_len, a text index (padded with spaces so the data starts on a 4096 multiple), then
the tensors, each 256-byte aligned at its offset (from the end of the index).  Index lines: "cfg <key> <value>", "t <name> <dtype> <rows>
<cols> <offset> <bytes>" (a matrix [out][in] is rows x cols; a vector or codebook row set is the leading dims folded into rows), "end".
dtypes: "q8_0" (ggml Q8_0 blocks, 34 bytes per 32 values, rows of the weight matrix [out][in] one after the other: what the engine's
multi-column MMVQ reads), "f32" (norms, the two conv base kernels), "q4_0" (--q4_0: ggml Q4_0 blocks, 18 bytes per 32 values, the matrices only; the engine tiles them like Q8_0, see gemm4_kernel in
src/kernels/cuda/dflash2.cu), "bf16" (the selector's two codebooks: only gathered rows are read; with --q4_0-selector they are "q4_0" rows in block order, not tiled).
The matrices' quantization matches ggml's quantize_row_q8_0 (d = amax / 127, q = round(x / d), d stored as fp16); the research harness
(tools/dflash2/q8test.py) measured no acceptance loss from it.
"""
import json
import struct
import sys

import numpy as np


class SafeTensors:
    """A .safetensors file read with numpy alone (no torch, no safetensors package: this runs on the person's PC, where neither is
    installed). The format is a little-endian u64 header length, a JSON header {name: {dtype, shape, data_offsets}}, then the raw data."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n > (64 << 20):
                raise ValueError("not a safetensors file (header too large)")
            self.header = json.loads(fh.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def raw(self, name):
        """-> (dtype string, shape, uint8 memmap of the tensor's bytes)"""
        info = self.header[name]
        lo, hi = info["data_offsets"]
        return info["dtype"], info["shape"], np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.base + lo, shape=(hi - lo,))

    def get_float32(self, name):
        """-> float32 ndarray (bf16 and f16 tensors are widened exactly)"""
        dtype, shape, data = self.raw(name)
        if dtype == "BF16":
            return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dtype == "F16":
            return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
        if dtype == "F32":
            return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
        raise ValueError("%s: unsupported dtype %s" % (name, dtype))

    def get_bf16_bits(self, name):
        dtype, shape, data = self.raw(name)
        if dtype != "BF16":
            raise ValueError("%s: expected BF16, found %s" % (name, dtype))
        return np.frombuffer(data, dtype=np.uint16).reshape(shape).copy()


def load_tensor(f, name):
    """-> float32 ndarray (bf16 tensors are widened exactly)"""
    return f.get_float32(name)


def raw_bf16(f, name):
    return np.ascontiguousarray(f.get_bf16_bits(name))


def quant_q8_0(w):
    """w: float32 [rows, cols], cols % 32 == 0 -> bytes of rows * cols/32 blocks (fp16 d, 32 int8)"""
    rows, cols = w.shape
    assert cols % 32 == 0
    b = w.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(axis=-1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    q = np.rint(b * inv[..., None]).astype(np.int8)       # round half to even; ggml uses roundf (ties are measure-zero for these weights)
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes()


def quant_q4_0(w):
    """w: float32 [rows, cols], cols % 32 == 0 -> bytes of rows * cols/32 ggml Q4_0 blocks (fp16 d, 16 bytes: value i in the low nibble of byte i, value i + 16 in the high
    nibble).  ggml's quantize_row_q4_0_ref: d = (the value of largest magnitude) / -8, q = min(15, int(x / d + 8.5)); the scale is stored as fp16, q uses the float d."""
    rows, cols = w.shape
    assert cols % 32 == 0
    b = w.reshape(rows, cols // 32, 32).astype(np.float32)
    idx = np.abs(b).argmax(axis=-1)
    mx = np.take_along_axis(b, idx[..., None], axis=-1)[..., 0]
    d = mx / np.float32(-8.0)
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0).astype(np.float32)
    q = np.minimum(15, (b * inv[..., None] + np.float32(8.5)).astype(np.int32)).astype(np.uint8)
    out = np.empty((rows, cols // 32, 18), np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q[..., :16] | (q[..., 16:] << 4)
    return out.tobytes()


def pack(st_path, cfg_path, out_path, say=print, q4=False, q4_selector=False):
    """Write the engine's runtime file for the drafter in `st_path` (+ its config.json). Returns the number of bytes written."""
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    dc = cfg["dflash_config"]
    f = SafeTensors(st_path)
    tensors = []
    blobs = []

    def add(name, dtype, shape, data):
        tensors.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": len(data)})
        blobs.append(data)

    def mat(name, new=None):
        w = load_tensor(f, name)
        assert w.ndim == 2
        if q4:
            add(new or name, "q4_0", w.shape, quant_q4_0(w))
        else:
            add(new or name, "q8_0", w.shape, quant_q8_0(w))

    def vec(name, new=None):
        w = load_tensor(f, name)
        add(new or name, "f32", w.shape, np.ascontiguousarray(w, np.float32).tobytes())

    L = cfg["num_hidden_layers"]
    mat("fc.weight")
    vec("hidden_norm.weight")
    vec("norm.weight")
    for l in range(L):
        p = "layers.%d." % l
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj", "attention_conv.kernel_projection", "mlp_conv.kernel_projection"):
            mat(p + n + ".weight")
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            vec(p + n + ".weight")
        vec(p + "attention_conv.base_kernel")
        vec(p + "mlp_conv.base_kernel")
    mat("candidate_selector.hidden_projection.weight")
    for n in ("predecessor_codebook", "successor_codebook"):
        if q4_selector:   # --q4_0-selector: ggml Q4_0 rows in block order (NOT tiled): 8 blocks of 18 bytes a row of 256, the community GGUF's selector_* bytes
            w = load_tensor(f, "candidate_selector." + n)
            add("candidate_selector." + n, "q4_0", w.shape, quant_q4_0(w))
            continue
        a = raw_bf16(f, "candidate_selector." + n)
        add("candidate_selector." + n, "bf16", a.shape, a.tobytes())

    meta = {
        "hidden": cfg["hidden_size"], "layers": L, "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"], "inter": cfg["intermediate_size"], "window": cfg["sliding_window"], "rope_theta": cfg["rope_parameters"]["rope_theta"],
        "eps": cfg["rms_norm_eps"], "block": dc["block_size"], "conv_group": dc["conv_group_size"], "conv_kernel": dc["conv_kernel_size"],
        "mask_token": dc["mask_token_id"], "rank": dc["selector_rank"], "top_k": dc["selector_top_k"], "taps": dc["target_layer_ids"],
        "vocab": cfg["vocab_size"], "source": "incoai/Qwen3.8-Flash-Next-DFlash2 a22a1744",
    }
    # offsets: relative to the end of the (padded) index; 256-byte aligned
    off = 0
    for t in tensors:
        off = (off + 255) & ~255
        t["offset"] = off
        off += t["bytes"]
    lines = []
    for k, v in meta.items():
        if k == "taps":
            v = ",".join(str(x) for x in v)
        lines.append("cfg %s %s" % (k, str(v).replace(" ", "_")))
    for t in tensors:
        sh = t["shape"] + [1] * (2 - len(t["shape"])) if len(t["shape"]) < 2 else t["shape"]
        lines.append("t %s %s %d %d %d %d" % (t["name"], t["dtype"], int(np.prod(sh[:-1])), sh[-1], t["offset"], t["bytes"]))
    lines.append("end")
    js = ("\n".join(lines) + "\n").encode()
    js += b" " * ((-(len(js) + 12)) % 4096)
    with open(out_path, "wb") as o:
        o.write(b"DF2P" + struct.pack("<II", 1, len(js)) + js)
        base = o.tell()
        for t, b in zip(tensors, blobs):
            o.seek(base + t["offset"])
            o.write(b)
    say("wrote %s: %d tensors, %.1f MB" % (out_path, len(tensors), (base + off) / 1e6))
    return base + off


def convert_selector_q4(src_path, dst_path, say=print, rows_per_step=16384):
    """Rewrite a finished DF2P pack with the selector's two codebooks as Q4_0 rows: every other tensor's bytes are carried over unchanged, the two bf16 codebooks
    (tensors `candidate_selector.predecessor_codebook` / `successor_codebook`, [vocab x 256]) are widened exactly to float32 and quantized by quant_q4_0, which is
    ggml's quantize_row_q4_0_ref.  The result is byte-identical to `pack(..., q4=True, q4_selector=True)` from the same safetensors file (and the codebooks are the
    bytes of the community GGUF nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF, selector_predecessor / selector_successor).  Memory stays under ~300 MB: the codebooks go
    through `rows_per_step` rows at a time.  Returns the size written.  Needs only numpy, no network."""
    with open(src_path, "rb") as fh:
        if fh.read(4) != b"DF2P":
            raise ValueError("not a DF2P pack")
        version, index_len = struct.unpack("<II", fh.read(8))
        if version != 1:
            raise ValueError("unknown DF2P version %d" % version)
        index = fh.read(index_len).decode()
        base = 12 + index_len
        cfg_lines, tensors = [], []
        for line in index.split("\n"):
            w = line.split()
            if not w or w[0] == "end":
                continue
            if w[0] == "cfg":
                cfg_lines.append(line)
            elif w[0] == "t":
                tensors.append({"name": w[1], "dtype": w[2], "rows": int(w[3]), "cols": int(w[4]), "offset": int(w[5]), "bytes": int(w[6])})
        books = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
        for t in tensors:
            if t["name"] in books:
                if t["dtype"] != "bf16":
                    raise ValueError("%s is already %s" % (t["name"], t["dtype"]))
                if t["cols"] % 32:
                    raise ValueError("%s: %d columns" % (t["name"], t["cols"]))
        off = 0
        for t in tensors:
            if t["name"] in books:
                t["new_bytes"] = t["rows"] * (t["cols"] // 32) * 18
                t["new_dtype"] = "q4_0"
            else:
                t["new_bytes"], t["new_dtype"] = t["bytes"], t["dtype"]
            off = (off + 255) & ~255
            t["new_offset"] = off
            off += t["new_bytes"]
        lines = list(cfg_lines)
        for t in tensors:
            lines.append("t %s %s %d %d %d %d" % (t["name"], t["new_dtype"], t["rows"], t["cols"], t["new_offset"], t["new_bytes"]))
        lines.append("end")
        js = ("\n".join(lines) + "\n").encode()
        js += b" " * ((-(len(js) + 12)) % 4096)
        with open(dst_path, "wb") as out:
            out.write(b"DF2P" + struct.pack("<II", 1, len(js)) + js)
            new_base = out.tell()
            for t in tensors:
                out.seek(new_base + t["new_offset"])
                fh.seek(base + t["offset"])
                if t["name"] not in books:
                    left = t["bytes"]
                    while left:
                        block = fh.read(min(left, 1 << 24))
                        if not block:
                            raise ValueError("%s: the pack is shorter than its index" % t["name"])
                        out.write(block)
                        left -= len(block)
                    continue
                for r0 in range(0, t["rows"], rows_per_step):
                    n = min(rows_per_step, t["rows"] - r0)
                    raw = np.frombuffer(fh.read(n * t["cols"] * 2), dtype=np.uint16)
                    if raw.size != n * t["cols"]:
                        raise ValueError("%s: the pack is shorter than its index" % t["name"])
                    w = (raw.astype(np.uint32) << 16).view(np.float32).reshape(n, t["cols"])
                    out.write(quant_q4_0(w))
            out.seek(new_base + off)        # the last tensor ends the file (the pack writer's size is base + off too)
            out.truncate(new_base + off)
    say("wrote %s: %d tensors, %.1f MB" % (dst_path, len(tensors), (new_base + off) / 1e6))
    return new_base + off


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--convert-selector":
        convert_selector_q4(sys.argv[2], sys.argv[3])
        return
    flags = ("--q4_0", "--q4_0-selector")
    args = [a for a in sys.argv[1:] if a not in flags]
    if len(args) < 3:
        sys.exit(__doc__)
    pack(*args[:3], q4="--q4_0" in sys.argv[1:] or "--q4_0-selector" in sys.argv[1:], q4_selector="--q4_0-selector" in sys.argv[1:])


if __name__ == "__main__":
    main()
