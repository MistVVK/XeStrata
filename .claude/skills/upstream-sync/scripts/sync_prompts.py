# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The code-agent prompts of the speed table: a coding agent's system prompt, XeStrata's own sources at commit
58dbbe5 as the files it has read, and a task about them; one prompt per length.  The prompts.py of
the 2026-10-04-speed-matrix record with a 192K tier added (the same tokens up to each length), from the text
tools/bench_core.py ships (data/bench-corpus.json.xz), which setup's --bench reads too.

    python .claude/skills/upstream-sync/scripts/sync_prompts.py PACK_DIR OUT_DIR

PACK_DIR is any pack (its `tokenizer/` is read).  Writes OUT_DIR/<tier>.ids, comma-separated, for `xestrata-engine
--tokens-file` (upstream's `strata --tokens-file`), and OUT_DIR/turn<N>.ids, the user's next messages of the same chat
for bench_serve.py (each opens with the <|im_end|> that closes the assistant's answer before it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} PACK_DIR OUT_DIR")
B.write_ids(Path(sys.argv[1]), Path(sys.argv[2]))
