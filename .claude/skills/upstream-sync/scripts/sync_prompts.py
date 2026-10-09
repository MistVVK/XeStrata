# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The code-agent prompts of the speed table: a coding agent's system prompt, XeStrata's own sources at commit
58dbbe5 as the files it has read, and a task about them; one prompt per length.  The prompts.py of
bench/results/2026-10-04-speed-matrix with a 192K tier added (the same tokens up to each length).

    python .claude/skills/upstream-sync/scripts/sync_prompts.py PACK_DIR OUT_DIR

Run from the repository root.  PACK_DIR is any pack (its `tokenizer/` is read).  Writes OUT_DIR/<tier>.ids,
comma-separated, for `strata generate --tokens-file`.
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, "tools")
import strata_tokenizer as ST  # noqa: E402

COMMIT = "58dbbe5"
# the token counts of upstream's code-agent prompts (bench/results/2026-09-29-speed-0126/matrix.json; 262K: the
# README's 259,943), so each tier fills the same share of its context; 192K: 262K's share of 196,608 (setup's
# context for it is 204,800)
TIERS = {"1k": 1017, "4k": 3562, "32k": 31566, "64k": 64162, "128k": 128478, "192k": 194956, "262k": 259943}

p = pathlib.Path(sys.argv[1]) / "tokenizer"
out = pathlib.Path(sys.argv[2])
vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tok = ST.Tokenizer(tokens, (p / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((p / "token_type.json").read_text()))


def enc(s):
    return tok.encode(s, parse_special=True)


def git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


system = ("<|im_start|>system\nYou are a coding agent working in the user's repository. You read files with the "
          "read_file tool, edit them with apply_patch, and run commands with the shell tool. Keep changes small, "
          "explain what you change and why, and do not touch files the task does not need.<|im_end|>\n"
          "<|im_start|>user\nThe repository is a C++/SYCL inference engine. These are the files you have read so "
          "far:\n\n")
task = ("\nTask: find where the prompt (prefill) path splits a long prompt into chunks, explain how the chunk size is "
        "chosen, and propose a patch that logs the chosen size once per prompt. Start with a short plan."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
head, tail = enc(system), enc(task)
files = [f for f in git("ls-tree", "-r", "--name-only", COMMIT, "src", "include").split("\n")
         if f.endswith((".cpp", ".hpp", ".h"))]
blocks = []                                  # each file's tokens, in the tree's order, until the largest tier is full
need = max(TIERS.values()) - len(head) - len(tail)
have = 0
for f in files:
    b = enc(f"<file path=\"{f}\">\n{git('show', f'{COMMIT}:{f}')}</file>\n")
    blocks.append(b)
    have += len(b)
    if have >= need:
        break
body = [t for b in blocks for t in b]
for tier, n in TIERS.items():
    ids = head + body[:n - len(head) - len(tail)] + tail
    (out / f"{tier}.ids").write_text(",".join(map(str, ids)))
    print(tier, len(ids))
