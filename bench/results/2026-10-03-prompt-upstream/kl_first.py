"""kl_first.py REF.logits OTHER.logits...: KL(REF || OTHER) of the first logits row of `strata --dump-logits` files
(int32 V, int32 n, then n rows of V floats), and whether the argmax agrees."""
import sys

import numpy as np


def first(path):
    b = open(path, "rb").read()
    v = int(np.frombuffer(b[:4], np.int32)[0])
    return np.frombuffer(b[8:8 + 4 * v], np.float32).astype(np.float64)


def kl(p, q):
    lp = p - p.max()
    lp -= np.log(np.exp(lp).sum())
    lq = q - q.max()
    lq -= np.log(np.exp(lq).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


ref = first(sys.argv[1])
for f in sys.argv[2:]:
    q = first(f)
    print(f, "KL %.3e argmax %s" % (kl(ref, q), "same" if q.argmax() == ref.argmax() else "differs"))
