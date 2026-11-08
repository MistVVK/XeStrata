# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two XeStrata runs against each other on one GPU, alternating as the merge's tables do: A, the base, and B, the
candidate.  For one idea of upstream's: the same binary with its switch off (A) and on (B), measured where the idea
works (a tier, the serve path, a conversation).  For a cell that lost against the previous XeStrata: that version's
binary (A) and HEAD's (B), in the same session.  The runs are tools/bench_core.py's, as bench_generate.py,
bench_serve.py and bench_usage.py make them; A takes upstream's place (with FOLLOW it writes the tier's text and B
follows it) and B XeStrata's.

    A="CMD..." B="CMD..." PROFILE=... DATA=DIR IDS=DIR [A_ARGS="..."] [B_ARGS="..."] \\
      python .claude/skills/upstream-sync/scripts/ab_pair.py generate|serve|usage OUT GPU [--dry-run]

A and B: each run's binary or wrapper, optionally after `env K=V ...` (the switch: `env XESTRATA_X=0 build/...`).
A_ARGS and B_ARGS: options one of them takes besides the model's and setup's.  PROFILE: the expert profile of both
(A_PROFILE and B_PROFILE when they differ).  A_ROOT and B_ROOT (default this tree): the trees whose serve/server.py
everyday use starts.  A_NAME and B_NAME (default "A" and "B"): their names in the tables.  TIERS (default "32k"),
MODES (default "greedy"), ROUNDS (default 3), CONFIG, PACK, NATIVE, PLE, EXTRA, FOLLOW, CONTEXT, CONVS, PY and PORT
as for the merge's scripts.  SLOW defaults to 0: neither run is stopped as too slow.
Writes what those scripts write into OUT, and then OUT's tables (sync_collect.py's); judge.py OUT prints B against A
per cell.  --dry-run: print the commands instead of running them.
"""
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

KINDS = ("generate", "serve", "usage")
args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 3 or args[0] not in KINDS:
    sys.exit(f"usage: {sys.argv[0]} {'|'.join(KINDS)} OUT GPU [--dry-run]")
kind, out, gpu = args[0], Path(args[1]), args[2]
dry = "--dry-run" in sys.argv
tree = Path(__file__).resolve().parents[4]
for arm, side in (("UP", "A"), ("XE", "B")):     # the merge's arms: upstream's place is the base's
    os.environ[arm] = B.need(side)
    os.environ[f"{arm}_PROFILE"] = os.environ.get(f"{side}_PROFILE") or B.need("PROFILE")
    os.environ[f"{arm}_ROOT"] = os.environ.get(f"{side}_ROOT", str(tree))
os.environ.setdefault("SLOW", "0")
m = B.Merge(usage=kind == "usage", rounds=3)
extra = {"up": shlex.split(os.environ.get("A_ARGS", "")), "xe": shlex.split(os.environ.get("B_ARGS", ""))}
tiers = os.environ.get("TIERS", "32k").split()

if kind == "generate":
    B.ab_generate(out, gpu, m.engines, lambda e, t: m.args(e, t) + extra[e.arm], Path(B.need("IDS")), tiers,
                  m.rounds, os.environ.get("FOLLOW", "1") != "0", m.slow, dry=dry)
elif kind == "serve":
    B.ab_serve(out, gpu, m.engines, lambda e, t: m.args(e, t) + extra[e.arm], B.read_ids(Path(B.need("IDS"))),
               tiers, os.environ.get("MODES", "greedy").split(), m.rounds, m.slow, dry=dry)
else:
    ctx = int(os.environ.get("CONTEXT", "262144"))

    def cfg(e: B.Engine) -> dict:
        return {"exe": e.cmd[0], "cwd": str(e.root), "tokenizer": f"{m.pack}/tokenizer", "model_name": "bench",
                "args": B.engine_args(m.model, e.profile, m.mtp, B.usage_context(m.model, ctx)) + extra[e.arm]}

    convs = tuple(os.environ.get("CONVS", " ".join(B.CONVERSATIONS)).split())
    if set(convs) - set(B.CONVERSATIONS):
        sys.exit(f"CONVS: some of {' '.join(B.CONVERSATIONS)}")
    B.ab_usage(out, gpu, m.engines, cfg, ctx, B.load_corpus()["agent"], m.rounds, m.slow,
               port=int(os.environ.get("PORT", "18095")), dry=dry,
               temperature=float(os.environ.get("TEMPERATURE", str(B.TEMPERATURE))), convs=convs)
if not dry:
    B.collect(out, {"xe": os.environ.get("B_NAME", "B"), "up": os.environ.get("A_NAME", "A")})
