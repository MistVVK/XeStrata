"""Greedy token agreement between the Xe engine and the llama.cpp reference (ref_logits.cpp).

    python compare_greedy.py XE_RUN.txt REF_RUN.txt [REF_LOGITS.txt]

XE_RUN is `strata generate` output (its `output :` line), REF_RUN the reference's (`output :` line). With the
reference's logits (one line per generated position), the first divergence is reported with the reference's top
candidates there, so a split at a near-tie can be told from a real disagreement.
"""
import re
import sys


def tokens(path, pattern):
    return [int(x) for x in re.search(pattern, open(path).read(), re.M).group(1).split()]


xe = tokens(sys.argv[1], r"^output\s*:\s*([\d ]+)")
ref = tokens(sys.argv[2], r"^output :([\d ]+)")
n = min(len(xe), len(ref))
d = next((i for i in range(n) if xe[i] != ref[i]), None)
print(f"xe {len(xe)} tokens, ref {len(ref)} tokens, first divergence at generated index {d}")
if d is not None and len(sys.argv) > 3:
    with open(sys.argv[3]) as f:
        for i, row in enumerate(f):
            if i == d:
                logits = [float(v) for v in row.split()]
                break
    order = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
    print("reference logits at that position (top 5):", [(i, round(logits[i], 4)) for i in order])
    print("xe chose", xe[d], "logit", round(logits[xe[d]], 4), "; reference chose", ref[d], "logit",
          round(logits[ref[d]], 4), "; margin", round(logits[ref[d]] - logits[xe[d]], 4))
