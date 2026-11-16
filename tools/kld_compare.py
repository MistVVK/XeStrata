#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The engine's teacher-forced logits against a reference llama.cpp saved with llama-perplexity's
--kl-divergence-base (its CPU's, for PLaMo 3 Translate's checks): for each chunk the engine reads the chunk's first
half as the prompt (BOS first, as llama-perplexity evaluates it) and follows the rest (--spec-follow), its logits
dumped (--dump-logits); the rows llama-perplexity scored (the second half) are compared.

    python tools/kld_compare.py BASE.kld --engine build/contrib-llvm/xestrata-engine --model M.gguf [--chunks N]
                                [--engine-arg ARG]...

Prints, over every compared row, the mean and largest KL divergence (reference || engine), the share of rows whose
most likely token is the same, and the RMS of the difference of the reference's top token's probability, as
llama-perplexity's own report has them (its sums: the reference's entries above a log-probability of -16, the next
token's probability for the difference).  --json writes them too.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def read_base(path: str):
    f = open(path, "rb")
    if f.read(8) != b"_logits_":
        raise SystemExit(f"{path}: not a llama-perplexity log-probability file")
    n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
    tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    rows = n_ctx - 1 - first
    return f, n_ctx, n_vocab, n_chunk, tokens, nv, first, rows


def base_log_probs(raw: np.ndarray, n_vocab: int) -> np.ndarray:
    """The rows' log-probabilities, as llama-perplexity decodes them (scale and the smallest log-probability first)."""
    head = raw[:, :4].copy().view(np.float32)   # [scale, min_log_prob]
    q = raw[:, 4:4 + n_vocab].astype(np.float32)
    return q * head[:, :1] + head[:, 1:2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", type=int, default=0, help="at most this many chunks (0: all)")
    ap.add_argument("--bos", type=int, default=1,
                    help="the chunks' first token (-1: the text's own, a model without BOS)")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--json")
    args = ap.parse_args()
    f, n_ctx, n_vocab, n_chunk, tokens, nv, first, rows = read_base(args.base)
    chunks = n_chunk if args.chunks <= 0 else min(n_chunk, args.chunks)
    kl_all, same, dp = [], 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for c in range(chunks):
            raw = np.frombuffer(f.read(2 * rows * nv), dtype=np.uint16).reshape(rows, nv)
            ref = base_log_probs(raw, n_vocab)
            toks = tokens[c].tolist()
            prompt = [args.bos if args.bos >= 0 else toks[0]] + toks[1:first + 1]
            follow = toks[first + 1:]
            pf, ff, lf = (os.path.join(tmp, n) for n in ("p.ids", "f.ids", "l.bin"))
            open(pf, "w").write(",".join(map(str, prompt)))
            open(ff, "w").write(",".join(map(str, follow)))
            cmd = [args.engine, "--native", args.model, "--tokens-file", pf, "--spec-follow", ff,
                   "--max-new", str(len(follow)), "--max-context", str(n_ctx + 64), "--dump-logits", lf]
            cmd += args.engine_arg
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stderr[-4000:])
                return 1
            got = np.fromfile(lf, dtype=np.float32)
            if got.size < rows * n_vocab:
                sys.stderr.write(f"chunk {c}: {got.size // n_vocab} rows of logits, {rows} expected\n")
                return 1
            got = got[:rows * n_vocab].reshape(rows, n_vocab).astype(np.float64)
            got -= got.max(axis=1, keepdims=True)
            lg = got - np.log(np.exp(got).sum(axis=1, keepdims=True))
            # as llama-perplexity sums it: the reference's entries above a log-probability of -16 only (those at its
            # floor carry the quantization's tail, not the model's), and the next token's probability for dp
            ref64 = ref.astype(np.float64)
            p_ref = np.where(ref64 > -16.0, np.exp(ref64), 0.0)
            kl = (p_ref * (ref64 - lg)).sum(axis=1)
            kl_all.append(kl)
            same += int((lg.argmax(axis=1) == ref.argmax(axis=1)).sum())
            nxt = np.asarray(follow[:rows], dtype=np.int64)
            dp.append(np.exp(lg[np.arange(rows), nxt]) - np.exp(ref64[np.arange(rows), nxt]))
            print(f"chunk {c}: mean KL {kl.mean():.6f}, max {kl.max():.4f}", flush=True)
    kl = np.concatenate(kl_all)
    d = np.concatenate(dp)
    out = {"rows": int(kl.size), "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
           "kl_99": float(np.quantile(kl, 0.99)), "same_top": same / kl.size, "rms_dp": float(np.sqrt((d * d).mean()))}
    print(f"rows {out['rows']}: mean KL {out['mean_kl']:.6f}, 99% {out['kl_99']:.4f}, max {out['max_kl']:.4f}, "
          f"same top {100 * out['same_top']:.3f}%, RMS dp {100 * out['rms_dp']:.3f}%")
    if args.json:
        open(args.json, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
