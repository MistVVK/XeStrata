"""Builds the scenario of tools/dflash2/serve_test.py from the prompt id files of the DFlash2 experiments.

    python mk_serve_scenario.py <prompt_dir> <tokenizer_dir> <out.json> [max_new=100]

prompt_dir holds <name>.prompt.ids (comma separated token ids of a chat-templated prompt).  Conversations: a (three turns), then b (a different conversation, so a's
live state is replaced), then a's fourth turn (a parked or checkpointed conversation resumes), c (two turns), d (one).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
import strata_tokenizer as ST  # noqa: E402


def main():
    pdir, tdir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((tdir / "token_type.json").read_text()))

    def ids(name):
        return [int(x) for x in (pdir / (name + ".prompt.ids")).read_text().strip().split(",")]

    def follow(text):
        return tok.encode("<|im_end|>\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n", parse_special=True)

    reqs = [
        {"name": "a1", "ids": ids("chat_explain")},
        {"name": "a2", "after": "a1", "follow": follow("Now say the same in two short sentences.")},
        {"name": "a3", "after": "a2", "follow": follow("Give one concrete example of it.")},
        {"name": "b1", "ids": ids("code_py")},
        {"name": "a4", "after": "a3", "follow": follow("Thanks. Last question: what is the main limit of this idea?")},
        {"name": "c1", "ids": ids("json_extract")},
        {"name": "c2", "after": "c1", "follow": follow("Now list only the keys.")},
        {"name": "d1", "ids": ids("tool_call")},
    ]
    json.dump({"max_new": max_new, "requests": reqs}, open(out, "w"))
    print("scenario with", len(reqs), "requests ->", out)


if __name__ == "__main__":
    main()
