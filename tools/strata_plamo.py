# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/strata_plamo.py - the PLaMo 3 tokenizer (a GGUF's `tokenizer.ggml.model` = `plamo3`), PLaMo 3 Translate's.

Transcribed from llama.cpp's llm_tokenizer_plamo2 with its PLaMo 3 pre-segmentation (src/llama-vocab.cpp at llama.cpp
7d7f0287, the commit the 2026-11-09-translate record compares with): the text is cut before and after each
`<|plamo:...|>` spelled in it, around runs of four or more equal characters (but line breaks) and of two or more
spaces, and at U+EE00 (dropped); each segment is then searched for the best-scoring split into the vocabulary's
pieces (scores rounded to 1e-4, the longest piece first among equals), a code point no piece covers falling back to its
UTF-8 bytes.  The special tokens are split off first, as llama_vocab::impl::tokenizer_st_partition does.  The
interface is strata_tokenizer.Tokenizer's: encode, decode, token_bytes, tokens, special_ids, control_tokens.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import json
import math
import pathlib

# GGUF token types
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
UNKNOWN_SCORE = -10000000   # a code point no piece covers: its bytes
TAG_PREFIX = "<|plamo:"
TAG_BODY_MAX = 64           # the most characters between `<|plamo:` and `|>` that still make a tag
BOUNDARY = ""         # the tokenizer's private-use segment boundary: splits, and is not encoded


def _round_score(score: float) -> int:
    # std::round of score * 1e4: half away from zero (Python's round is half to even)
    v = score * 1e4
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


class PlamoTokenizer:
    def __init__(self, tokens: list[str], scores: list[float], token_types: list[int],
                 special_ids: dict[str, int] | None = None, pre_segment: bool = True):
        self.tokens = tokens
        self.scores = scores
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.pre_segment = pre_segment
        self.ids = {t: i for i, t in enumerate(tokens)}
        self._bytes = [0] * 256
        self._pieces: dict[str, tuple[int, int]] = {}   # every token but the byte ones: (id, rounded score)
        self._prefixes: set[str] = set()                # every prefix of a piece, to stop a search early
        self._max_len = 1
        for i, (t, ty) in enumerate(zip(tokens, token_types)):
            if ty == BYTE:
                if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                    self._bytes[int(t[3:5], 16)] = i
                continue
            if t in self._pieces:   # the first id of a text wins, as the C++ map's insertion
                continue
            self._pieces[t] = (i, _round_score(scores[i]))
            for k in range(1, len(t) + 1):
                self._prefixes.add(t[:k])
            self._max_len = max(self._max_len, len(t))
        if not all(self._bytes):
            raise ValueError("a byte token <0xXX> is missing from the vocabulary")
        # the special tokens split off before the search (llama.cpp's cache_special_tokens: the CONTROL, USER_DEFINED
        # and UNKNOWN ones, the longest text in bytes first); CONTROL and UNKNOWN only with parse_special
        self.special_tokens = {t: i for i, (t, ty) in enumerate(zip(tokens, token_types))
                               if ty in (CONTROL, USER_DEFINED, UNKNOWN) and t}
        self.control_tokens = [t for t, i in self.special_tokens.items() if token_types[i] == CONTROL]
        self._specials = sorted(self.special_tokens, key=lambda t: len(t.encode("utf-8")), reverse=True)

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "PlamoTokenizer":
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        model = md.get("tokenizer.ggml.model")
        if model not in ("plamo2", "plamo3"):
            raise ValueError("expected a PLaMo tokenizer (model 'plamo3'), got %r" % model)
        special = {k: int(md[k]) for k in md if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), [float(x) for x in md["tokenizer.ggml.scores"]],
                   [int(x) for x in md["tokenizer.ggml.token_type"]], special, model == "plamo3")

    def write_dir(self, out: pathlib.Path) -> dict:
        """The files from_dir reads back: plamo.json (the tokens and their scores), token_type.json, and vocab.json
        (token -> id, which the server checks for)."""
        (out / "plamo.json").write_text(json.dumps([[t, s] for t, s in zip(self.tokens, self.scores)],
                                                   ensure_ascii=False), encoding="utf-8")
        (out / "token_type.json").write_text(json.dumps(self.token_types), encoding="utf-8")
        (out / "vocab.json").write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")
        return {"model": "plamo3" if self.pre_segment else "plamo2", "vocab_size": len(self.tokens),
                "special_ids": self.special_ids}

    @classmethod
    def from_dir(cls, path, cfg: dict) -> "PlamoTokenizer":
        path = pathlib.Path(path)
        pairs = json.loads((path / "plamo.json").read_text(encoding="utf-8"))
        return cls([p[0] for p in pairs], [float(p[1]) for p in pairs],
                   json.loads((path / "token_type.json").read_text()), cfg.get("special_ids"),
                   cfg.get("model", "plamo3") == "plamo3")

    # -------------------------------------------------------------- the search
    def _encode_cpts(self, text: str) -> list[int]:
        n = len(text)
        if n == 0:
            return []
        inf = 1 << 60
        cost = [inf] * (n + 1)   # the lowest cost (the negated score sum) of text[i:]
        cost[n] = 0
        step = [0] * (n + 1)     # the length of the piece the best split of text[i:] starts with
        tid = [-1] * (n + 1)     # that piece's token (-1: the code point's bytes)
        pieces, prefixes, max_len = self._pieces, self._prefixes, self._max_len
        for i in range(n - 1, -1, -1):
            found = []
            for k in range(1, min(max_len, n - i) + 1):
                s = text[i:i + k]
                if s not in prefixes:
                    break
                p = pieces.get(s)
                if p is not None:
                    found.append((k, p))
            best = inf
            for k, (t, score) in reversed(found):   # the longest first; a later one must be strictly better
                c = cost[i + k] - score
                if c < best:
                    best, step[i], tid[i] = c, k, t
            c = cost[i + 1] - UNKNOWN_SCORE
            if c < best:
                best, step[i], tid[i] = c, 1, -1
            cost[i] = best
        out: list[int] = []
        i = 0
        while i < n:
            if tid[i] >= 0:
                out.append(tid[i])
            else:
                out.extend(self._bytes[b] for b in text[i].encode("utf-8", "surrogatepass"))
            i += step[i]
        return out

    def _segment(self, text: str) -> list[int]:
        if not self.pre_segment:
            if text.startswith("﻿"):
                text = text[1:]
            return self._encode_cpts(text)
        n = len(text)
        cut = [False] * (n + 1)
        # pass 1: <|plamo:...|>, a body of at most 64 characters, no `|` and no white space
        plen = len(TAG_PREFIX)
        i = 0
        while i + plen <= n:
            if not text.startswith(TAG_PREFIX, i):
                i += 1
                continue
            j = i + plen
            while j < n and j - (i + plen) < TAG_BODY_MAX and text[j] != "|" and not text[j].isspace():
                j += 1
            if j + 1 < n and text[j] == "|" and text[j + 1] == ">":
                cut[i] = True
                cut[j + 2] = True
                i = j + 2
            else:
                i += 1
        # pass 2: runs of four or more equal characters (but line breaks) and of two or more spaces; a run does not
        # cross a cut of pass 1
        i = 0
        while i < n:
            c = text[i]
            run = 1
            while i + run < n and text[i + run] == c and not cut[i + run]:
                run += 1
            if (c != "\n" and run >= 4) or (c == " " and run >= 2):
                cut[i] = True
                cut[i + run] = True
            i += run
        out: list[int] = []
        start = 0
        for end in range(n + 1):
            boundary = end < n and text[end] == BOUNDARY
            if end == n or cut[end] or boundary:
                if start < end:
                    out.extend(self._encode_cpts(text[start:end]))
                start = end + boundary
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`: the special tokens' texts split off first (CONTROL and UNKNOWN ones only with
        `parse_special`), the text between them searched.  `plain`: (start, end) spans of `text` that are ordinary
        text even where they spell a special token.  No BOS is added."""
        fragments: list[tuple[int, int] | int] = [(0, len(text))]   # (start, end) of raw text, or a token id
        for sp in self._specials:
            ty = self.token_types[self.special_tokens[sp]]
            if not parse_special and ty in (CONTROL, UNKNOWN):
                continue
            nxt: list[tuple[int, int] | int] = []
            for f in fragments:
                if not isinstance(f, tuple):
                    nxt.append(f)
                    continue
                a, b = f
                pos = a
                while True:
                    k = text.find(sp, pos, b)
                    while k >= 0 and plain and any(x <= k < y for x, y in plain):
                        k = text.find(sp, k + 1, b)
                    if k < 0:
                        break
                    if k > pos:
                        nxt.append((pos, k))
                    nxt.append(self.special_tokens[sp])
                    pos = k + len(sp)
                if pos < b:
                    nxt.append((pos, b))
            fragments = nxt
        out: list[int] = []
        for f in fragments:
            if isinstance(f, tuple):
                out.extend(self._segment(text[f[0]:f[1]]))
            else:
                out.append(f)
        return out

    # -------------------------------------------------------------- back to text
    def token_bytes(self, i: int) -> bytes:
        if i < 0 or i >= len(self.tokens):
            raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
        t, ty = self.tokens[i], self.token_types[i]
        if ty == BYTE and len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            return bytes([int(t[3:5], 16)])
        return t.encode("utf-8")

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)
