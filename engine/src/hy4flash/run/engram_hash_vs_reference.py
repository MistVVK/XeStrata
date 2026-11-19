"""Check engram_hash_expected.bin against the repo's own tools/squant/hy_v4_flash.py EngramTableReader.hash_tokens
(instantiated without __init__, fed from the manifest engram_meta). usage: python engram_hash_vs_reference.py <repo> <manifest> <expected.bin>"""
import json
import struct
import sys

import numpy as np

repo, manifest, expected = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo + "/tools")
from squant.hy_v4_flash import EngramTableReader  # noqa: E402

meta = json.load(open(manifest, encoding="utf-8"))["engram_meta"]
r = object.__new__(EngramTableReader)
r.meta = meta
r.token_map = np.array(meta["engram_token_map"], dtype=np.int64)
r.pad_cid = int(r.token_map[meta["engram_pad_id"]])
r.n_gram = int(meta["ple_ngram_size"])
r.n_head = int(meta["ple_heads_per_ngram"])
r.n_cols = (r.n_gram - 1) * r.n_head
r.ple_layers = list(meta["ple_layers"])
r.mults = np.array(meta["ple_layer_multipliers"], dtype=np.uint64).reshape(len(r.ple_layers), r.n_gram)
r.primes = np.array(meta["ple_head_vocab_sizes"], dtype=np.uint64).reshape(len(r.ple_layers), r.n_cols)
r.offsets = np.array(meta["ple_head_offsets"], dtype=np.int64).reshape(len(r.ple_layers), r.n_cols)

b = open(expected, "rb").read()
assert b[:4] == b"EHR1"
T = struct.unpack_from("<i", b, 4)[0]
tok = np.frombuffer(b, dtype="<i4", count=T, offset=8).astype(np.int64)
off = 8 + 4 * T
bad = 0
for mod in range(4):
    eid = np.frombuffer(b, dtype="<i8", count=T * 16, offset=off).reshape(T, 16)
    ev = np.frombuffer(b, dtype=np.uint8, count=T * 16, offset=off + T * 16 * 8).reshape(T, 16)
    off += T * 16 * 9
    # the reference hashes 3000 tokens one python loop at a time; do the first 600 only
    N = 600
    ids, valid = r.hash_tokens(mod, tok[:N])
    bad += int((ids != eid[:N]).sum()) + int((valid.astype(np.uint8) != ev[:N]).sum())
print("reference hash_tokens vs expected file: mismatches", bad, "->", "PASS" if bad == 0 else "FAIL")
