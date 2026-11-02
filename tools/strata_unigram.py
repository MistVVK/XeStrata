# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2023-2026 The ggml authors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/strata_unigram.py - SentencePiece's Unigram tokenizer (a GGUF's `tokenizer.ggml.model` = `t5`), llm-jp-4.1's.

Transcribed from llama.cpp's llm_tokenizer_ugm and llm_tokenizer_ugm_session (src/llama-vocab.cpp at the llama.cpp
commit the 2026-10-10-llmjp-parity record compares with, de7fa0a3): the normalization through the precompiled
charsmap (a XOR-compressed compact double array of prefix replacements), the whitespace rules, the Viterbi search over
the vocabulary's trie with the unknown token's penalty, and the special tokens split off first
(llama_vocab::impl::tokenizer_st_partition).  The interface is strata_tokenizer.Tokenizer's: encode, decode,
token_bytes, tokens, special_ids, control_tokens.

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
import pathlib
import struct

# GGUF token types
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
ESCAPED_SPACE = "▁".encode("utf-8")    # SentencePiece's space: U+2581, LOWER ONE EIGHTH BLOCK
UNKNOWN_PENALTY = 10.0


def _utf8_len(lead: int) -> int:
    # llama.cpp's unicode_len_utf8: the sequence length its first byte announces (a continuation byte counts as 1)
    return (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)[lead >> 4]


class _Trie:
    """llama.cpp's naive_trie over bytes: a node's children by byte, and the token id it ends (or None)."""
    __slots__ = ("children", "value")

    def __init__(self) -> None:
        self.children: dict[int, _Trie] = {}
        self.value: int | None = None

    def insert(self, key: bytes, value: int | None) -> None:
        node = self
        for b in key:
            node = node.children.setdefault(b, _Trie())
        if node.value is None:   # the first insertion of a key wins, as naive_trie's
            node.value = value if value is not None else -1

    def longest_prefix(self, data: bytes, start: int) -> int:
        """The length of the longest key that `data[start:]` starts with (0: none)."""
        node, best, i = self, 0, start
        while i < len(data):
            nxt = node.children.get(data[i])
            if nxt is None:
                break
            node = nxt
            i += 1
            if node.value is not None:
                best = i - start
        return best


class UnigramTokenizer:
    def __init__(self, tokens: list[str], scores: list[float], token_types: list[int], charsmap: bytes = b"",
                 add_space_prefix: bool = False, remove_extra_whitespaces: bool = False,
                 special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.scores = scores
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        self.unk_id = int(self.special_ids.get("tokenizer.ggml.unknown_token_id", 2))
        self.add_space_prefix = add_space_prefix
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.escape_whitespaces = True          # llama.cpp's default; the GGUF does not set it for t5
        self.treat_whitespace_as_suffix = False
        # the charsmap: a 4-byte XCDA size, the XCDA's 32-bit entries, then NUL-terminated replacement strings
        self._xcda: tuple[int, ...] = ()
        self._repl = b""
        if charsmap:
            (size,) = struct.unpack_from("<I", charsmap, 0)
            if size + 4 >= len(charsmap):
                raise ValueError("the precompiled charsmap's XCDA overruns it")
            self._xcda = struct.unpack_from("<%dI" % (size // 4), charsmap, 4)
            self._repl = charsmap[4 + size:]
        self._matcher = _Trie()        # normal, user-defined and unused tokens: what the Viterbi search may take
        self._user = _Trie()           # user-defined tokens: kept whole by the normalization
        min_score = float("inf")
        for i, (t, ty) in enumerate(zip(tokens, token_types)):
            raw = t.encode("utf-8")
            if ty == NORMAL:
                min_score = min(min_score, scores[i])
            if ty in (NORMAL, USER_DEFINED, UNUSED):
                self._matcher.insert(raw, i)
            if ty == USER_DEFINED:
                self._user.insert(raw, i)
        self._unk_score = min_score - UNKNOWN_PENALTY
        # the special tokens split off before the search (llama.cpp's cache_special_tokens: the CONTROL, USER_DEFINED
        # and UNKNOWN ones, the longest text in bytes first); CONTROL and UNKNOWN only with parse_special
        self.special_tokens = {t: i for i, (t, ty) in enumerate(zip(tokens, token_types))
                               if ty in (CONTROL, USER_DEFINED, UNKNOWN) and t}
        self.control_tokens = [t for t, i in self.special_tokens.items() if token_types[i] == CONTROL]
        self._specials = sorted(self.special_tokens, key=lambda t: len(t.encode("utf-8")), reverse=True)

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "UnigramTokenizer":
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        if md.get("tokenizer.ggml.model") != "t5":
            raise ValueError("expected a Unigram tokenizer (model 't5'), got %r" % md.get("tokenizer.ggml.model"))
        cm = md.get("tokenizer.ggml.precompiled_charsmap") or []
        special = {k: int(md[k]) for k in md if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), [float(x) for x in md["tokenizer.ggml.scores"]],
                   [int(x) for x in md["tokenizer.ggml.token_type"]], bytes(int(x) & 0xFF for x in cm),
                   bool(md.get("tokenizer.ggml.add_space_prefix", False)),
                   bool(md.get("tokenizer.ggml.remove_extra_whitespaces", False)), special)

    @classmethod
    def from_hf_json(cls, path) -> "UnigramTokenizer":
        """A Hugging Face tokenizer.json of a SentencePiece unigram model (T5's: Anima's LLM adapter reads its ids): the
        vocabulary and its scores, the Precompiled normalizer's charsmap, its added special tokens as control tokens
        (<unk> the unknown one), the space before the text and the whitespace runs collapsed as T5's spiece.model
        has them (Metaspace with a prefix space, nmt_nfkc's remove_extra_whitespaces).  No </s> is added."""
        import base64
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        m = d["model"]
        if m.get("type", "Unigram") != "Unigram" or "vocab" not in m:
            raise ValueError(f"{path}: not a unigram tokenizer.json")
        tokens = [p[0] for p in m["vocab"]]
        scores = [float(p[1]) for p in m["vocab"]]
        unk = int(m.get("unk_id") or 0)
        types = [NORMAL] * len(tokens)
        for a in d.get("added_tokens") or ():
            if a.get("special") and 0 <= int(a["id"]) < len(types):
                types[int(a["id"])] = CONTROL
        types[unk] = UNKNOWN
        norm = d.get("normalizer") or {}
        steps = norm.get("normalizers", [norm]) if norm.get("type") == "Sequence" else [norm]
        charsmap = b""
        for st in steps:
            if st and st.get("type") == "Precompiled" and st.get("precompiled_charsmap"):
                charsmap = base64.b64decode(st["precompiled_charsmap"])
        return cls(tokens, scores, types, charsmap, True, True, {"tokenizer.ggml.unknown_token_id": unk})

    def write_dir(self, out: pathlib.Path) -> dict:
        """The files from_dir reads back: unigram.json (the tokens and their scores), token_type.json, charsmap.bin,
        and vocab.json (token -> id, which the server checks for)."""
        (out / "unigram.json").write_text(json.dumps([[t, s] for t, s in zip(self.tokens, self.scores)],
                                                     ensure_ascii=False), encoding="utf-8")
        (out / "token_type.json").write_text(json.dumps(self.token_types), encoding="utf-8")
        charsmap = b""
        if self._xcda:
            charsmap = struct.pack("<I", 4 * len(self._xcda)) + struct.pack("<%dI" % len(self._xcda), *self._xcda) + \
                self._repl
        (out / "charsmap.bin").write_bytes(charsmap)
        (out / "vocab.json").write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")
        return {"model": "t5", "vocab_size": len(self.tokens), "special_ids": self.special_ids,
                "add_space_prefix": self.add_space_prefix, "remove_extra_whitespaces": self.remove_extra_whitespaces}

    @classmethod
    def from_dir(cls, path, cfg: dict) -> "UnigramTokenizer":
        path = pathlib.Path(path)
        pairs = json.loads((path / "unigram.json").read_text(encoding="utf-8"))
        return cls([p[0] for p in pairs], [float(p[1]) for p in pairs],
                   json.loads((path / "token_type.json").read_text()), (path / "charsmap.bin").read_bytes(),
                   bool(cfg.get("add_space_prefix", False)), bool(cfg.get("remove_extra_whitespaces", False)),
                   cfg.get("special_ids"))

    # -------------------------------------------------------------- normalization
    def _xcda_node(self, index: int) -> int:
        if index >= len(self._xcda):
            raise IndexError("index out of the precompiled charsmap's XCDA")
        return self._xcda[index]

    def _normalize_prefix(self, data: bytes, offset: int) -> tuple[bytes, int]:
        """(the replacement of the input's prefix at `offset`, the input bytes it consumed)"""
        n = self._user.longest_prefix(data, offset)
        if n > 0:
            return data[offset:offset + n], n
        longest_len, longest_off = 0, 0
        if self._xcda:
            # BASE in bits 10-30 (shifted by bit 9's flag), LCHECK in bits 0-7 and 31, LEAF in bit 8, a value in 0-30
            def base(i):
                p = self._xcda_node(i)
                return (p >> 10) << ((p & (1 << 9)) >> 6)
            node = base(0)
            for k in range(offset, len(data)):
                c = data[k]
                if c == 0:
                    break
                node ^= c
                p = self._xcda_node(node)
                if (p & ((1 << 31) | 0xFF)) != c:
                    break
                leaf = (p >> 8) & 1
                node ^= base(node)
                if leaf:
                    longest_len = k - offset + 1
                    longest_off = self._xcda_node(node) & ((1 << 31) - 1)
        if longest_len > 0:
            if longest_off >= len(self._repl):
                raise IndexError("index out of the precompiled charsmap's replacements")
            end = self._repl.find(b"\0", longest_off)
            if end < 0:
                raise ValueError("unterminated string in the precompiled charsmap")
            return self._repl[longest_off:end], longest_len
        n = _utf8_len(data[offset])
        piece = data[offset:offset + n]
        try:
            piece.decode("utf-8")
            if len(piece) == n:
                return piece, n
        except UnicodeDecodeError:
            pass
        return "�".encode("utf-8"), 1   # an invalid sequence: one byte, as U+FFFD

    def _normalize(self, data: bytes) -> bytes:
        space = ESCAPED_SPACE if self.escape_whitespaces else b" "
        prepend = not self.treat_whitespace_as_suffix and self.add_space_prefix
        append = self.treat_whitespace_as_suffix and self.add_space_prefix
        merge = self.remove_extra_whitespaces
        out = bytearray()
        prepended = False
        in_word = False
        off = 0
        while off < len(data):
            piece, used = self._normalize_prefix(data, off)
            for c in piece:
                if c != 0x20:
                    if not in_word:
                        in_word = True
                        if (prepend and not prepended) or merge:
                            out += space
                            prepended = True
                    out.append(c)
                else:
                    in_word = False
                    if not merge:
                        out += space
            off += used
        if append:
            out += space
        return bytes(out)

    # -------------------------------------------------------------- the search
    def _encode_plain(self, text: str) -> list[int]:
        data = self._normalize(text.encode("utf-8"))
        n = len(data)
        if n == 0:
            return []
        # best[i]: (token ending at i, where it starts, the score sum); every sum starts below any real one
        best: list[tuple[int, int, float]] = [(self.unk_id, 0, -float("inf"))] * (n + 1)
        best[0] = (self.unk_id, 0, 0.0)
        off = 0
        while off < n:
            units = min(_utf8_len(data[off]), n - off)
            whole = False
            score0 = best[off][2]
            node, k = self._matcher, off
            while k < n:
                nxt = node.children.get(data[k])
                if nxt is None:
                    break
                node = nxt
                k += 1
                if node.value is not None and node.value >= 0:
                    if k - off == units:
                        whole = True
                    tid = node.value
                    s = score0 + (0.0 if self.token_types[tid] == USER_DEFINED else self.scores[tid])
                    if s > best[k][2]:
                        best[k] = (tid, off, s)
            if not whole:   # no token for the whole code point: the unknown token, with its penalty
                s = score0 + self._unk_score
                if s > best[off + units][2]:
                    best[off + units] = (self.unk_id, off, s)
            off += units
        out: list[int] = []
        prev_unk = False
        i = n
        while True:   # back from the end; a run of unknown tokens becomes one
            tid, start, _ = best[i]
            unk = tid == self.unk_id
            if not (prev_unk and unk):
                out.append(tid)
            if start == 0:
                break
            prev_unk = unk
            i = start
        out.reverse()
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`: the special tokens' texts split off first (CONTROL and UNKNOWN ones only with
        `parse_special`), the text between them searched.  `plain`: (start, end) spans of `text` that are ordinary
        text even where they spell a special token."""
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
                out.extend(self._encode_plain(text[f[0]:f[1]]))
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
        if ty == NORMAL:
            return t.encode("utf-8").replace(ESCAPED_SPACE, b" ")
        return t.encode("utf-8")

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)
