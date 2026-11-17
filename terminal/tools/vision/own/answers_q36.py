#!/usr/bin/env python3
"""The answers rule for Qwen3.6: greedy picture answers of strata-q36 (the engine protocol, one engine for every arm) for embeddings made by different readers.

    python3 answers_q36.py --engine <strata-q36> --model <Qwen3.6 .gguf> --vc <dir with shapes.png.q0.llama.log / .q1.llama.log> --out <dir>
                           [--n 48] [--engine-args "--max-context 4096 --vision"] --arm mtmd-cpu=<dir of .sve> --arm own-cpu=<dir> --arm mtmd-gpu=<dir> --arm own-gpu=<dir>
                           [--pictures a.png b.png ...]

Every arm is a folder of <picture name>.sve files (strata-vision outputs of one reader for the same pictures at the same token limit).  The prompt ids are the
chat template's: the text before the picture, the picture's pad tokens (as many as the .sve has rows), then the text after it with the question.  The two
questions are the ones q36_vision_cmp.py asks (their token ids are read from llama.cpp's logs of shapes.png, which the question's tokens do not depend on).
The engine decodes greedily; the ids are turned back into text with the model's own vocabulary (byte-level BPE).  Prints the pairwise counts of identical
answers and every differing answer with all arms side by side.  Exit 0 when every answer is identical, 1 otherwise.  Standard library plus tools/qwen35/q36_harness.py.
"""
import argparse
import json
import os
import struct
import sys

PAD = 248056


def read_tokens(gguf):
    """The vocabulary (token strings) of a GGUF: a stdlib reader of the header's tokenizer.ggml.tokens array."""
    f = open(gguf, "rb")

    def u(fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, f.read(n))[0]

    def s():
        return f.read(u("Q"))

    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t, keep):
        if t == 8:
            return s().decode("utf8", "replace") if keep else s() and None
        if t == 9:
            et, n = u("I"), u("Q")
            if et == 8:
                items = [s() for _ in range(n)]
                return [x.decode("utf8", "replace") for x in items] if keep else None
            sz = struct.calcsize(sizes[et])
            f.seek(n * sz, 1)
            return None
        return u(sizes[t])

    assert f.read(4) == b"GGUF"
    u("I")
    u("Q")
    nk = u("Q")
    for _ in range(nk):
        k = s().decode()
        t = u("I")
        v = val(t, k == "tokenizer.ggml.tokens")
        if k == "tokenizer.ggml.tokens":
            return v
    raise SystemExit("no vocabulary in " + gguf)


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode(ids, vocab, bd):
    out = bytearray()
    for i in ids:
        t = vocab[i] if 0 <= i < len(vocab) else ""
        if t.startswith("<|") and t.endswith("|>"):
            continue          # special tokens (the end of the turn) are not text
        out += bytes(bd.get(ch, ord("?")) for ch in t)
    return out.decode("utf8", "replace")


def chunks(logfile):
    """(text before the picture, text after it) of a llama.cpp dump log: the 'text:' lines around the 'image:' line."""
    before, after, seen = [], [], False
    for line in open(logfile):
        if line.startswith("text:"):
            (after if seen else before).extend(int(x) for x in line.split(":", 1)[1].split())
        elif line.startswith("image:"):
            seen = True
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--engine-args", default="--max-context 4096 --vision")
    ap.add_argument("--arm", action="append", required=True, help="label=<dir of .sve files>")
    ap.add_argument("--pictures", nargs="*", default=[])
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qwen35"))
    from q36_harness import Eng  # noqa: E402

    os.makedirs(a.out, exist_ok=True)
    arms = [x.split("=", 1) for x in a.arm]
    pics = a.pictures or sorted(n[:-4] for n in os.listdir(arms[0][1]) if n.endswith(".sve"))
    questions = []
    q = 0
    while os.path.isfile(os.path.join(a.vc, "shapes.png.q%d.llama.log" % q)):   # q0, q1, ... : one log per question
        questions.append(chunks(os.path.join(a.vc, "shapes.png.q%d.llama.log" % q)))
        q += 1
    vocab, bd = read_tokens(a.model), byte_decoder()
    eng = Eng(a.engine, ["--model", a.model] + a.engine_args.split(), log=os.path.join(a.out, "engine.log"))
    print("engine ready after %.0f s" % eng.load_s, flush=True)
    res = {label: {} for label, _ in arms}
    for pic in pics:
        for label, d in arms:
            sve = os.path.join(d, pic + ".sve")
            if not os.path.isfile(sve):
                continue
            n_img = struct.unpack("<5i", open(sve, "rb").read(20))[1]
            for qi, (pre, post) in enumerate(questions):
                ids = pre + [PAD] * n_img + post
                try:
                    toks, done, _, wall = eng.solo(ids, a.n, emb=sve)
                    text = decode(toks, vocab, bd)
                    err = None
                except SystemExit as e:   # the engine said ERR for this request
                    text, err = "", str(e)
                res[label]["%s|q%d" % (pic, qi)] = {"text": text, "ids": toks if err is None else [], "err": err}
                print("%-10s %-24s q%d: %r%s" % (label, pic, qi, text[:90], "  ERR " + err if err else ""), flush=True)
        json.dump(res, open(os.path.join(a.out, "answers.json"), "w"), indent=1)

    labels = [l for l, _ in arms]
    keys = list(res[labels[0]])
    ok = lambda l, k: k in res[l] and not res[l][k]["err"]  # noqa: E731
    print("ANSWERS, pair by pair (identical / compared):")
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            same = sum(1 for k in keys if ok(la, k) and ok(lb, k) and res[la][k]["ids"] == res[lb][k]["ids"])
            print("  %-10s vs %-10s %d / %d" % (la, lb, same, len(keys)))
    differing = [k for k in keys if len({tuple(res[l][k]["ids"]) if ok(l, k) else None for l in labels}) > 1]
    print("%d of %d answers differ between some readers; all of them:" % (len(differing), len(keys)))
    for k in differing:
        print("DIFFERS %s" % k)
        for l in labels:
            print("  %-10s: %r" % (l, res[l][k]["text"] if ok(l, k) else "ERR"))
    eng.send("QUIT")
    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
