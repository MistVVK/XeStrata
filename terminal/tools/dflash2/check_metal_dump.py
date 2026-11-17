"""Checks the Metal engine's DFlash2 drafter against the torch port (dflash2_ref.py) on a real round.

    python check_metal_dump.py <dump.bin> <drafter.safetensors> <drafter_config.json>

dump.bin is written by the Metal engine with STRATA_DF2_DUMP=<file> [STRATA_DF2_DUMP_ROUND=n] (metal/src/engine/dflash2.inc): every context row the
drafter was given (positions 0 .. s-1: a prompt short enough that the whole context was fed), the anchor and mask embeddings, and the engine's answer
of that round (the 8 final normalized rows, the selector's hidden projection, the top-16 per row, the picked tokens and their probabilities, the
draft head's logits).  The checks:
  1. the 8 final rows against the torch port run on the same context (weights fp32 against the engine's Q8_0: cosine similarity, not equality),
  2. the engine's top-16 per row against a sort of its own logits,
  3. the selector's walk (torch, on the engine's own rows and candidates) against the engine's picks.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import dflash2_ref as R


class Embed:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, ids):
        return torch.stack([self.rows[int(i)] for i in ids])


def main():
    dump, st, cfgp = sys.argv[1:4]
    raw = open(dump, "rb").read()
    magic, ver, s, anchor, nd, nt, H, mask = struct.unpack_from("<8i", raw, 0)
    assert magic == 0x32465344, "not a DF2 dump"
    off = 32
    def take(n, dt=np.float32):
        nonlocal off
        a = np.frombuffer(raw, dt, n, off)
        off += a.nbytes
        return a
    taps = take(s * nt * H).reshape(s, nt * H)
    emb_a, emb_m = take(H), take(H)
    fn = take(8 * H).reshape(8, H)
    hsel = take(7 * 256).reshape(7, 256)
    topval = take(7 * 16).reshape(7, 16)
    toptok = take(7 * 16, np.int32).reshape(7, 16)
    out = take(32, np.int32)
    out_prob = np.frombuffer(out.tobytes(), np.float32)[16:23]
    out_tok = out[:7]
    rest = len(raw) - off
    has_map = rest > 7 * nd * 4
    vmap = take(nd, np.int32) if has_map else None
    logits = take(7 * nd).reshape(7, nd)
    print("dump: %d context rows, anchor %d, %d-token head%s" % (s, anchor, nd, " (mapped)" if has_map else ""))

    d = R.Drafter(st, cfgp, Embed({anchor: torch.from_numpy(emb_a.copy()), mask: torch.from_numpy(emb_m.copy())}), torch.randn(64, H), threads=6)
    d.set_context(torch.from_numpy(taps.copy()))
    d.draft(s, anchor, 8)
    ref = d.last_all.numpy()
    bad = 0
    print("1. final rows against the torch port (fp32 weights):")
    worst = 1.0
    for r in range(8):
        a, b = fn[r].astype(np.float64), ref[r].astype(np.float64)
        cs = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        worst = min(worst, cs)
        print("   row %d: cos %.5f  norm ratio %.4f" % (r, cs, np.linalg.norm(a) / (np.linalg.norm(b) + 1e-30)))
    if worst < 0.98:
        print("   FAIL: a row's cosine similarity %.4f is below 0.98" % worst)
        bad += 1

    print("2. top-16 of the engine's logits:")
    for i in range(7):
        order = np.lexsort((np.arange(nd), -logits[i]))[:16]   # value descending, ties to the lower column
        want = (vmap[order] if has_map else order).astype(np.int32)
        if not (np.array_equal(want, toptok[i]) and np.allclose(logits[i][order], topval[i])):
            print("   FAIL row %d: engine %s, sort %s" % (i, toptok[i].tolist(), want.tolist()))
            bad += 1
    if not bad:
        print("   all 7 rows equal")

    print("3. the selector's walk on the engine's rows and candidates:")
    w = d.w
    hp = torch.from_numpy(fn[1:].copy()) @ w["candidate_selector.hidden_projection.weight"].T
    A, B = w["candidate_selector.predecessor_codebook"], w["candidate_selector.successor_codebook"]
    pred = anchor
    same = 0
    for i in range(7):
        cand = torch.from_numpy(toptok[i].astype(np.int64))
        sc = torch.from_numpy(topval[i].copy()) + (B[cand].float() @ (A[pred].float() * hp[i]))
        j = int(torch.argmax(sc))
        p = float(torch.softmax(sc, dim=-1)[j])
        pick = int(cand[j])
        ok = pick == int(out_tok[i]) and abs(p - float(out_prob[i])) < 0.02
        same += ok
        print("   %d: engine %d (p %.3f)  torch %d (p %.3f)  %s" % (i, out_tok[i], out_prob[i], pick, p, "ok" if ok else "DIFF"))
        pred = int(out_tok[i])
    if same < 7:
        bad += 1
    print("hsel against torch: max abs diff %.4f" % float(np.abs(hsel - hp.numpy()).max()))
    print("check_metal_dump: %s" % ("FAIL" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
