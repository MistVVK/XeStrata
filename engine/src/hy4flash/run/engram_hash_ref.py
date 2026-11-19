"""Independent re-implementation of tools/squant/hy_v4_flash.py EngramTableReader.hash_tokens (numpy + pure python ints),
used to produce expected row ids for engine/src/hy4flash/run/engram_test.cpp.

usage: python engram_hash_ref.py <manifest_sq64.json> <out.bin> [T=3000] [seed=7]
out.bin layout (little endian): magic 'EHR1', int32 T, int32 tok[T], then for m in 0..3: int64 ids[T*16], uint8 valid[T*16]
Tokens: zipf-like over the vocab, plus a few out-of-range ids (-1, 120832, 999999) to exercise the 'dead' rule.
"""
import json
import struct
import sys

import numpy as np


def hash_np(meta, mod, tokens):
    tm = np.array(meta["engram_token_map"], dtype=np.int64)
    pad_cid = int(tm[meta["engram_pad_id"]])
    n_head = int(meta["ple_heads_per_ngram"])
    n_cols = (int(meta["ple_ngram_size"]) - 1) * n_head
    nm = len(meta["ple_layers"])
    mults = np.array(meta["ple_layer_multipliers"], dtype=np.uint64).reshape(nm, int(meta["ple_ngram_size"]))
    primes = np.array(meta["ple_head_vocab_sizes"], dtype=np.uint64).reshape(nm, n_cols)
    offs = np.array(meta["ple_head_offsets"], dtype=np.int64).reshape(nm, n_cols)
    T = len(tokens)
    cids = np.where((tokens >= 0) & (tokens < len(tm)), tm[np.clip(tokens, 0, len(tm) - 1)], -1)
    dead = cids < 0
    c0 = np.where(dead, pad_cid, cids).astype(np.uint64)
    prev1 = np.concatenate([[-1], cids[:-1]])
    prev2 = np.concatenate([[-1, -1], cids[:-2]])
    c1 = np.where(prev1 < 0, pad_cid, prev1).astype(np.uint64)
    c2 = np.where(prev2 < 0, pad_cid, prev2).astype(np.uint64)
    m = mults[mod]
    with np.errstate(over="ignore"):
        r1 = (c0 * m[0]) ^ (c1 * m[1])
        r2 = r1 ^ (c2 * m[2])
    ids = np.empty((T, n_cols), dtype=np.int64)
    for h in range(n_head):
        ids[:, h] = (r1 % primes[mod, h]).astype(np.int64) + offs[mod, h]
        ids[:, n_head + h] = (r2 % primes[mod, n_head + h]).astype(np.int64) + offs[mod, n_head + h]
    valid = np.ones((T, n_cols), dtype=np.uint8)
    valid[dead, :] = 0
    return ids, valid


def hash_py(meta, mod, tokens, i):
    """pure python big-int check for one position i"""
    M = (1 << 64) - 1
    tm = meta["engram_token_map"]
    pad = tm[meta["engram_pad_id"]]
    n_head = int(meta["ple_heads_per_ngram"])
    mults = meta["ple_layer_multipliers"][mod * 3: mod * 3 + 3]
    primes = meta["ple_head_vocab_sizes"][mod * 16: mod * 16 + 16]
    offs = meta["ple_head_offsets"][mod * 16: mod * 16 + 16]

    def cid(j):
        if j < 0:
            return -1
        t = int(tokens[j])
        return tm[t] if 0 <= t < len(tm) else -1

    c0 = cid(i)
    dead = c0 < 0
    t0 = pad if dead else c0
    t1 = pad if cid(i - 1) < 0 else cid(i - 1)
    t2 = pad if cid(i - 2) < 0 else cid(i - 2)
    r1 = ((t0 * mults[0]) & M) ^ ((t1 * mults[1]) & M)
    r2 = r1 ^ ((t2 * mults[2]) & M)
    ids = [(r1 % primes[h]) + offs[h] for h in range(n_head)] + [(r2 % primes[n_head + h]) + offs[n_head + h] for h in range(n_head)]
    return ids, (0 if dead else 1)


def main():
    path, out = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    meta = json.load(open(path, encoding="utf-8"))["engram_meta"]
    rng = np.random.default_rng(seed)
    V = len(meta["engram_token_map"])
    z = rng.zipf(1.2, size=T)
    tok = ((z - 1) * 37 + rng.integers(0, 50, size=T)) % V
    tok = tok.astype(np.int64)
    # a few dead / out-of-range ids
    for pos, v in ((0, -1), (5, V), (6, 999999), (7, 7), (1500, -1), (2999, V + 3)):
        if pos < T:
            tok[pos] = v
    with open(out, "wb") as f:
        f.write(b"EHR1")
        f.write(struct.pack("<i", T))
        f.write(tok.astype("<i4").tobytes())
        for mod in range(len(meta["ple_layers"])):
            ids, valid = hash_np(meta, mod, tok)
            # cross-check with the independent big-int implementation on a subset
            for i in list(range(0, 40)) + [1499, 1500, 1501, 2998, T - 1]:
                if i >= T:
                    continue
                pi, pv = hash_py(meta, mod, tok, i)
                assert list(ids[i]) == pi, (mod, i, list(ids[i]), pi)
                assert int(valid[i, 0]) == pv
            f.write(ids.astype("<i8").tobytes())
            f.write(valid.tobytes())
    print("wrote", out, "T", T)


if __name__ == "__main__":
    main()
