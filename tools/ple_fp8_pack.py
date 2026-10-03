# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/ple_fp8_pack.py - the PLE n-gram table exactly as Qwen shipped it (FP8 E4M3), as a GGUF the engine reads.

    python tools/ple_fp8_pack.py --model <checkpoint dir> --out <ple-fp8.gguf>
    python tools/ple_fp8_pack.py --hf-repo Qwen/Qwen3.8-Flash-Next --out <ple-fp8.gguf>

With --hf-repo the checkpoint is not downloaded: the index, each safetensors header and the n-gram shards' bytes are
read from Hugging Face with HTTP range requests (51.2 GB of the 104 GB in the 33 files that hold them).

Qwen3.8-Flash-Next's checkpoint stores the table as 128 shards `...ngram_embedding.shard_{k}.weight` of
[2500012, 160] F8_E4M3 with one BF16 `weight_scale`: 320,001,536 rows of 160 values, 51.2 GB. The engine used to take
it from ISTA-DASLab's GGUF as IQ4_NL (90 B/row), which is 8% off these values per row; this keeps them bit for bit.

The output is a one-tensor GGUF: `per_layer_token_embd.weight`, ne = [160, 320001536], type I8 holding the FP8 bytes
(GGUF has no FP8 type), with `strata.ple.format` = "f8_e4m3" and `strata.ple.scale` (F32). The bytes are copied
straight out of the safetensors files in shard order - nothing is decoded or rounded - so memory stays flat.
"""
import argparse
import json
import pathlib
import struct
import sys

from checkpoint_source import Source, safetensors_header

GGUF_TYPE_STRING, GGUF_TYPE_F32 = 8, 6
GGML_TYPE_I8 = 24
ALIGN = 32
CHUNK = 64 << 20


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def kv_f32(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_F32) + struct.pack("<f", val)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="the checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--hf-repo", help="read the checkpoint from this Hugging Face repository instead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.model is None) == (a.hf_repo is None):
        sys.exit("give --model or --hf-repo")
    src = Source(model=pathlib.Path(a.model) if a.model else None, repo=a.hf_repo)
    out = pathlib.Path(a.out)
    wmap = json.loads(src.text("model.safetensors.index.json"))["weight_map"]

    shards = {}
    scale_name = None
    for name in wmap:
        if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
            shards[int(name.rsplit(".shard_", 1)[1].split(".")[0])] = name
        elif name.endswith(".ngram_embedding.weight_scale"):
            scale_name = name
    if not shards or scale_name is None:
        sys.exit("no ngram_embedding shards / weight_scale in " + src.name)
    if sorted(shards) != list(range(len(shards))):
        sys.exit("the n-gram shards are not numbered 0..%d without gaps" % (len(shards) - 1))

    hdr, base = safetensors_header(src, wmap[scale_name])
    t = hdr[scale_name]
    if t["dtype"] != "BF16" or t["shape"] not in ([1], []):
        sys.exit("unexpected weight_scale: %s %s" % (t["dtype"], t["shape"]))
    (bits,) = struct.unpack("<H", src.read(wmap[scale_name], base + t["data_offsets"][0], 2))
    scale = struct.unpack("<f", struct.pack("<I", bits << 16))[0]

    parts, rows, dim = [], 0, None
    headers = {}
    for k in range(len(shards)):
        name = shards[k]
        if wmap[name] not in headers:
            headers[wmap[name]] = safetensors_header(src, wmap[name])
        hdr, base = headers[wmap[name]]
        t = hdr[name]
        if t["dtype"] != "F8_E4M3" or len(t["shape"]) != 2:
            sys.exit("%s is %s %s, not a 2-D F8_E4M3 tensor" % (name, t["dtype"], t["shape"]))
        if dim is None:
            dim = t["shape"][1]
        if t["shape"][1] != dim or t["data_offsets"][1] - t["data_offsets"][0] != t["shape"][0] * dim:
            sys.exit("%s: inconsistent shape %s" % (name, t["shape"]))
        parts.append((wmap[name], base + t["data_offsets"][0], t["shape"][0] * dim))
        rows += t["shape"][0]
    print("%d shards, %d rows x %d, scale %.9g, %.2f GB" % (len(parts), rows, dim, scale, rows * dim / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, FP8 E4M3 as shipped"),
           kv_string("strata.ple.format", "f8_e4m3"),
           kv_f32("strata.ple.scale", scale),
           kv_string("strata.ple.source", src.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, rows)
    head += struct.pack("<I", GGML_TYPE_I8) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    written = 0
    with open(tmp, "wb") as w:
        w.write(head)
        for i, (path, off, n) in enumerate(parts):
            done = 0
            while done < n:
                b = src.read(path, off + done, min(CHUNK, n - done))
                if not b:
                    sys.exit("short read in " + str(path))
                w.write(b)
                done += len(b)
            written += n
            if i % 16 == 15 or i == len(parts) - 1:
                print("  %3d/%d shards, %.1f GB" % (i + 1, len(parts), written / 1e9), flush=True)
    if tmp.stat().st_size != len(head) + rows * dim:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + rows * dim))
    tmp.replace(out)
    print("wrote %s (%d rows x %d FP8, header %d B)" % (out, rows, dim, len(head)))


if __name__ == "__main__":
    main()
