# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reads bench_generate.py's OUT/runs/*.txt and writes OUT/matrix.json (every run) and, per GPU, the prompt and output
tables (the median of the rounds; a cell no run finished is "-", one stopped as too slow "too slow") as Markdown to
stdout.  An upstream run stopped as too slow has "stopped" (prompt or output); one stopped in its output takes its
prompt rate from the last PP line (or, without PP lines, from its "xestrata generate: prefill" line, upstream's
"strata generate: prefill").
When bench_serve.py ran too, OUT/serve-runs/*.json go to OUT/matrix-serve.json and three more tables per GPU and mode:
the tier's prompt, the follow-ups (the median time a follow-up's prompt takes, in ms) and the output of all four
turns.
When bench_usage.py ran too, OUT/usage-runs/*.json go to OUT/matrix-usage.json and a table per GPU: per conversation,
the time to the first token (the median of its turns), the prompt and the output speed (the server's timings, summed).
When bench_image.py ran too, OUT/image-runs/*.json go to OUT/matrix-image.json and a table per GPU: the seconds an
image, its sampling and its VAE decode.
The reading is tools/bench_core.py's (collect).

    python .claude/skills/upstream-sync/scripts/sync_collect.py OUT XE_NAME UP_NAME   # e.g. OUT xe0.1.41 v0.1.41
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

if len(sys.argv) != 4:
    sys.exit(f"usage: {sys.argv[0]} OUT XE_NAME UP_NAME")
B.collect(Path(sys.argv[1]), {"xe": sys.argv[2], "up": sys.argv[3]})
