# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The serve table of an upstream merge on one GPU: XeStrata and upstream's release run as the server runs them
(`xestrata-engine --serve`, the options setup writes), one chat of four turns per run: the tier's prompt, then the three
follow-ups of sync_prompts.py, each with the answer before it, 256 tokens a turn.  bench_generate.py measures the
generate path; this one the serve loop users meet (the conversation checkpoints, the per-request sampling, its own
decode loop).  The engines are driven over stdin (GEN, DONE), not through serve/server.py.  The runs are
tools/bench_core.py's (ab_serve), which setup's --bench makes too.

    XE="CMD..." UP="CMD..." DATA=DIR IDS=DIR XE_PROFILE=... UP_PROFILE=... \\
      python .claude/skills/upstream-sync/scripts/bench_serve.py OUT GPU [--dry-run]

XE, UP, XE_PROFILE, UP_PROFILE, DATA, IDS, CONFIG, PACK, NATIVE, PLE, EXTRA and TIERS (default "64k 32k 1k") as for
bench_generate.py; a wrapper command must pass stdin through (podman run -i).  MODES (default "greedy t07"): greedy,
and tNN = temperature NN / 10 (t07 = 0.7, t10 = 1.0), top_p 0.95, top_k 20, a seed per round.  ROUNDS (default 3):
runs per arm, tier and mode, each a new process, the arms alternating; one discarded warm-up per arm first.  No
--spec-follow: serve has none, so each engine writes its own text and the drafts it accepts count in its output
speed, as they do for a user.
OUT/serve-runs/GPU-ARM-TIER-MODE-ROUND.txt gets each run's log, and .json the four turns' DONE fields;
sync_collect.py OUT reads them.  An upstream run whose first turn reads its prompt or writes its output at SLOW tok/s
or less (default 10) is not run again in that tier and mode (marked "stopped"); a run past 2400 s is killed.
--dry-run: print the engine commands instead of running them.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
m = B.Merge(rounds=3)          # the median of three: a round moves by about 10% with the text at t07
B.ab_serve(Path(args[0]), args[1], m.engines, m.args, B.read_ids(Path(B.need("IDS"))),
           os.environ.get("TIERS", "64k 32k 1k").split(), os.environ.get("MODES", "greedy t07").split(), m.rounds,
           m.slow, dry="--dry-run" in sys.argv)
