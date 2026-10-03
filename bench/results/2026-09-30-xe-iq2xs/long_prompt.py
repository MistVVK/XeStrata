"""The long-context prompt: a needle at the start, ~26k tokens of filler, then a question about the needle.

    python long_prompt.py PACK_DIR OUT_IDS.txt

Run from the repository root.  PACK_DIR is the IQ2_XS pack (its `tokenizer/` is read).  The ids are written
comma-separated, for `strata generate --tokens-file`.
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, "tools")
import strata_tokenizer as ST

p = pathlib.Path(sys.argv[1]) / "tokenizer"
vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tok = ST.Tokenizer(tokens, (p / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((p / "token_type.json").read_text()))

random.seed(7)
subjects = ["The river", "A merchant", "The old library", "Every winter", "The committee", "A small boat", "The garden",
            "Her grandfather", "The railway", "A distant bell"]
verbs = ["carried", "remembered", "described", "ignored", "measured", "painted", "followed", "repaired"]
objects = ["the northern road", "a quiet harbour", "the market square", "three silver keys", "the evening fog",
           "a forgotten letter", "the stone bridge", "an empty lantern"]
filler = []
while len(filler) < 3600:
    filler.append(f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}.")
needle = "Important: the secret code for the vault is PELICAN-4172."
text = needle + " " + " ".join(filler) + "\n\nQuestion: what is the secret code for the vault? Answer with the code only."
ids = tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
print(len(ids), file=sys.stderr)
open(sys.argv[2], "w").write(",".join(map(str, ids)))
