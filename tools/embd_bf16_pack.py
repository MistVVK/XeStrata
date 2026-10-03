# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/embd_bf16_pack.py - the token embedding exactly as the checkpoint ships it (BF16), as a GGUF the engine reads.

    python tools/embd_bf16_pack.py --model <checkpoint dir> --out <token-embd-bf16.gguf>
    python tools/embd_bf16_pack.py --hf-repo Qwen/Qwen3.8-Flash-Next --out <token-embd-bf16.gguf>

With --hf-repo only the embedding's bytes are read from Hugging Face (HTTP range requests, tools/checkpoint_source.py).

The GGUF stores token_embd quantized (IQ4_XS in the GSQ-RCO files). The engine keeps the table in mapped host memory
and reads one row per token, so BF16 costs 0.6 GB of RAM more and no VRAM; `--embd-gguf` takes it from this file.

The output is a one-tensor GGUF: `token_embd.weight`, ne = [n_embd, n_vocab], type BF16, the bytes copied straight
out of the safetensors file - nothing decoded or rounded.
"""
import argparse
import json
import pathlib
import struct
import sys

from checkpoint_source import Source, safetensors_header

GGUF_TYPE_STRING = 8
GGML_TYPE_BF16 = 30
ALIGN = 32
CHUNK = 64 << 20


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


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

    names = [n for n in wmap if n.endswith("embed_tokens.weight") and "visual" not in n and "mtp" not in n]
    if len(names) != 1:
        sys.exit("expected one text embed_tokens.weight in %s, found %s" % (src.name, names))
    name = names[0]
    hdr, base = safetensors_header(src, wmap[name])
    t = hdr[name]
    if t["dtype"] != "BF16" or len(t["shape"]) != 2:
        sys.exit("%s is %s %s, not a 2-D BF16 tensor" % (name, t["dtype"], t["shape"]))
    vocab, dim = t["shape"]
    n = vocab * dim * 2
    if t["data_offsets"][1] - t["data_offsets"][0] != n:
        sys.exit("%s: data size does not match its shape" % name)
    print("%s: %d x %d BF16, %.2f GB" % (name, vocab, dim, n / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-embd"),
           kv_string("general.name", "token embedding, BF16 as shipped"),
           kv_string("strata.embd.source", src.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, vocab)
    head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as w:
        w.write(head)
        done = 0
        while done < n:
            b = src.read(wmap[name], base + t["data_offsets"][0] + done, min(CHUNK, n - done))
            if not b:
                sys.exit("short read in " + wmap[name])
            w.write(b)
            done += len(b)
    if tmp.stat().st_size != len(head) + n:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + n))
    tmp.replace(out)
    print("wrote %s (header %d B)" % (out, len(head)))


if __name__ == "__main__":
    main()
