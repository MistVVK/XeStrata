# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/dflash2_pack.py - a DFlash2 drafter (Inco AI's, Apache-2.0: Qwen3.8-Flash-Next's, GLM-5.3-Flash's,
Qwen3.6-35B-A3B's) as the engine's runtime file.

    python tools/dflash2_pack.py model.safetensors config.json out.pack [--q4_0] [--q4_0-selector]
    python tools/dflash2_pack.py --convert-selector in.pack out.pack

The safetensors file is read with numpy alone (the person's PC has neither torch nor the safetensors package).
The quantization matches ggml's reference rows byte for byte, so the pack's size and SHA-256 are the same on every
machine and tools/dflash2_fetch.py pins them.

Format (little endian): "DF2P", u32 version (1), u32 index_len, a text index (padded with spaces so the data starts
on a 4096 multiple), then the tensors, each 256-byte aligned at its offset (from the end of the index).  Index lines:
"cfg <key> <value>", "t <name> <dtype> <rows> <cols> <offset> <bytes>" (a matrix [out][in] is rows x cols; a vector
or codebook row set folds its leading dims into rows), "end".  dtypes:
  q8_0   ggml Q8_0 blocks (34 bytes per 32 values), the weight matrix's rows one after another
  q4_0   ggml Q4_0 blocks (18 bytes per 32 values), the matrices only (--q4_0); the engine tiles them at load
  f32    norms, the two conv base kernels
  bf16   the selector's two codebooks (only gathered rows are read); with --q4_0-selector they are "q4_0" rows in
         block order (NOT tiled like the matrices): the bytes of nazeshinjite's community GGUF (Apache-2.0)
"""
import json
import struct
import sys

import numpy as np


class SafeTensors:
    """A .safetensors file read with numpy alone: a little-endian u64 header length, a JSON header
    {name: {dtype, shape, data_offsets}}, then the raw data."""

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
        return (info["dtype"], info["shape"],
                np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.base + lo, shape=(hi - lo,)))

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


def quant_q8_0(w):
    """w: float32 [rows, cols], cols % 32 == 0 -> bytes of rows * cols/32 blocks (fp16 d, 32 int8).
    ggml's quantize_row_q8_0: d = amax / 127, q = round(x / d), d stored as fp16."""
    rows, cols = w.shape
    assert cols % 32 == 0
    b = w.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(axis=-1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)
    # round half to even; ggml uses roundf (ties are measure-zero for these weights)
    q = np.rint(b * inv[..., None]).astype(np.int8)
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes()


def quant_q4_0(w):
    """w: float32 [rows, cols], cols % 32 == 0 -> bytes of rows * cols/32 ggml Q4_0 blocks (fp16 d, 16 bytes: value i
    in the low nibble of byte i, value i + 16 in the high nibble).  ggml's quantize_row_q4_0_ref: d = (the value of
    largest magnitude) / -8, q = min(15, int(x / d + 8.5)); the scale is stored as fp16, q uses the float d."""
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


def pack(st_path, cfg_path, out_path, say=print, q4=False, q4_selector=False,
         source="incoai/Qwen3.8-Flash-Next-DFlash2 a22a1744"):
    """Write the engine's runtime file for the drafter in `st_path` (+ its config.json). Returns the bytes written.
    `source`: the repository and revision the index names (a cfg line, so a pack's bytes are those of its source)."""
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    dc = cfg["dflash_config"]
    f = SafeTensors(st_path)
    tensors = []
    blobs = []

    def add(name, dtype, shape, data):
        tensors.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": len(data)})
        blobs.append(data)

    def mat(name):
        w = f.get_float32(name)
        assert w.ndim == 2
        add(name, "q4_0" if q4 else "q8_0", w.shape, quant_q4_0(w) if q4 else quant_q8_0(w))

    def vec(name):
        w = f.get_float32(name)
        add(name, "f32", w.shape, np.ascontiguousarray(w, np.float32).tobytes())

    L = cfg["num_hidden_layers"]
    mat("fc.weight")
    vec("hidden_norm.weight")
    vec("norm.weight")
    for ly in range(L):
        p = "layers.%d." % ly
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                  "attention_conv.kernel_projection", "mlp_conv.kernel_projection"):
            mat(p + n + ".weight")
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            vec(p + n + ".weight")
        vec(p + "attention_conv.base_kernel")
        vec(p + "mlp_conv.base_kernel")
    mat("candidate_selector.hidden_projection.weight")
    for n in ("predecessor_codebook", "successor_codebook"):
        if q4_selector:   # ggml Q4_0 rows in block order (NOT tiled): the community GGUF's selector_* bytes
            w = f.get_float32("candidate_selector." + n)
            add("candidate_selector." + n, "q4_0", w.shape, quant_q4_0(w))
            continue
        a = f.get_bf16_bits("candidate_selector." + n)
        add("candidate_selector." + n, "bf16", a.shape, a.tobytes())

    meta = {
        "hidden": cfg["hidden_size"], "layers": L, "heads": cfg["num_attention_heads"],
        "kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "inter": cfg["intermediate_size"],
        "window": cfg["sliding_window"], "rope_theta": cfg["rope_parameters"]["rope_theta"],
        "eps": cfg["rms_norm_eps"], "block": dc["block_size"], "conv_group": dc["conv_group_size"],
        "conv_kernel": dc["conv_kernel_size"], "mask_token": dc["mask_token_id"], "rank": dc["selector_rank"],
        "top_k": dc["selector_top_k"], "taps": dc["target_layer_ids"],
        "vocab": cfg["vocab_size"], "source": source,
    }
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
        lines.append("t %s %s %d %d %d %d"
                     % (t["name"], t["dtype"], int(np.prod(sh[:-1])), sh[-1], t["offset"], t["bytes"]))
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


def read_index(path):
    """A DF2P pack's (cfg lines, tensor dicts, data base offset). Raises ValueError on a file that is not one."""
    with open(path, "rb") as fh:
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
            tensors.append({"name": w[1], "dtype": w[2], "rows": int(w[3]), "cols": int(w[4]),
                            "offset": int(w[5]), "bytes": int(w[6])})
    return cfg_lines, tensors, base


def convert_selector_q4(src_path, dst_path, say=print, rows_per_step=16384):
    """Rewrite a finished DF2P pack with the selector's two codebooks as Q4_0 rows: every other tensor's bytes carry
    over unchanged.  The two bf16 codebooks ([vocab x rank]) are widened exactly to float32 and quantized by
    quant_q4_0 (ggml's quantize_row_q4_0_ref).  The result is byte-identical to pack(..., q4=True, q4_selector=True)
    from the same safetensors file.  Memory stays under ~300 MB (rows_per_step rows at a time).  No network."""
    cfg_lines, tensors, base = read_index(src_path)
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
        lines.append("t %s %s %d %d %d %d"
                     % (t["name"], t["new_dtype"], t["rows"], t["cols"], t["new_offset"], t["new_bytes"]))
    lines.append("end")
    js = ("\n".join(lines) + "\n").encode()
    js += b" " * ((-(len(js) + 12)) % 4096)
    with open(src_path, "rb") as fh, open(dst_path, "wb") as out:
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
    pack(*args[:3], q4="--q4_0" in sys.argv[1:] or "--q4_0-selector" in sys.argv[1:],
         q4_selector="--q4_0-selector" in sys.argv[1:])


if __name__ == "__main__":
    main()
