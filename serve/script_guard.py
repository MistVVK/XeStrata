# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/script_guard.py - the script guard: the characters of other languages kept out of an answer.

Asked for by a request's "script_guard" (a language below, "off", or {"lang": "ja", "allow_context": false}), by the
header X-XeStrata-Script-Guard (the same values, for clients that cannot add a field to the body), by the Chat settings
shared with other apps, or for every request by the run config's "script_guard" / STRATA_SCRIPT_GUARD (the
environment wins); off when none of them says so.

"ja" bans the Han characters of GB2312 (Simplified Chinese's character set) that JIS X 0213 does not have - the forms
only Simplified Chinese writes (这, 们, 说, 时), not the ones Japanese shares with it (没, 来) - every Hangul
character, and the letters only Vietnamese writes (ả, ệ, ơ, ư, đ; not the á, à and é pinyin and French share too).
The tokens that hold one get logit_bias -100 in the answer; the thinking is left as the model writes it (the answer's
pass starts with the bias where the thinking ends).  With "allow_context" (the default) a character the prompt already
holds is not banned, so a Chinese, Korean or Vietnamese text the user pasted can be quoted back.  "en" bans the same
way every Han character, Japanese's kana, Hangul, Bopomofo, the CJK punctuation, symbols and full-width forms, and the
letters only Vietnamese writes: it is for the languages that write none of them (English and the like).  "ko" bans the
kana and the Han characters KS X 1001 (Korean's character set, its Hanja) does not have.  "zh-cn" and "zh-tw" ban the
kana, Hangul and the forms only Japanese writes (in JIS X 0208 and in neither GB2312 nor Big5: 気, 駅, 込), and
"zh-cn" the traditional forms (in Big5, not in GB2312: 這, 說, 國), "zh-tw" the simplified ones (in GB2312, not in
Big5: 这, 说, 国).  Every language but "en" also bans the letters only Vietnamese writes.

A byte-level vocabulary can still spell a banned character in byte tokens, which no single token's bias stops.  The
server sees the character as it completes, before it is sent: the tokens that spell it are dropped and the engine
writes that place again with the first of them banned for one token (at most REWIND_MAX times an answer; after that,
or when part of it has gone out already, it is let through and counted).
"""
from __future__ import annotations

import threading

from serve import logit_bias as bias_api

LANGS = ("ja", "en", "ko", "zh-cn", "zh-tw")
REWIND_MAX = 8
BAN = -100.0
USAGE = ('script_guard: "off", "ja", "en", "ko", "zh-cn", "zh-tw", or {"lang": one of them, "allow_context": true | '
         'false}')

# Hangul: the jamo, the compatibility jamo, the jamo extensions A and B, the syllables and the halfwidth forms
HANGUL = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7A3), (0xD7B0, 0xD7FF), (0xFFA0, 0xFFDC))
# Vietnamese: Latin Extended Additional's Vietnamese letters (ạ to ỹ) and ă, đ, ĩ, ũ, ơ, ư with their capitals, and ƣ
# (OI), which old Vietnamese fonts wrote for ư and which Qwen3.8 wrote in its place once ư was banned
VIETNAMESE = ((0x1EA0, 0x1EF9), (0x0102, 0x0103), (0x0110, 0x0111), (0x0128, 0x0129), (0x0168, 0x0169),
              (0x01A0, 0x01A3), (0x01AF, 0x01B0))
# CJK for "en": the radicals, the ideographic description characters, the CJK symbols and punctuation, the kana,
# Bopomofo, kanbun, the strokes, the enclosed and compatibility characters, the Han blocks (extension A, the unified
# ideographs, the compatibility ideographs, extensions B to H), the vertical and the full-width forms, and the kana
# supplements
CJK = ((0x2E80, 0x2FDF), (0x2FF0, 0x2FFF), (0x3000, 0x312F), (0x3190, 0x33FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
       (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF), (0x1B000, 0x1B16F), (0x20000, 0x2FA1F),
       (0x30000, 0x323AF))
# the kana: Hiragana, Katakana and its phonetic extensions, the half-width Katakana, the kana supplements
KANA = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F), (0x1B000, 0x1B16F))
HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))   # the Han blocks the national character sets draw from


def _encodes(ch: str, codec: str) -> bool:
    try:
        ch.encode(codec)
    except UnicodeEncodeError:
        return False
    return True


def _han(keep) -> set[str]:
    """The Han characters of HAN for which keep(ch) is false."""
    return {chr(c) for lo, hi in HAN for c in range(lo, hi + 1) if not keep(chr(c))}


def _banned(lang: str) -> frozenset:
    """The characters `lang` bans.  The codecs are Python's: gb2312 (GB2312), cp950 (Big5), euc_jis_2004 (JIS X 0213),
    shift_jis (JIS X 0208), euc_kr (KS X 1001)."""
    ranges = {"en": CJK + HANGUL, "ja": HANGUL, "ko": KANA, "zh-cn": KANA + HANGUL, "zh-tw": KANA + HANGUL}[lang]
    out = {chr(c) for lo, hi in ranges + VIETNAMESE for c in range(lo, hi + 1)}
    gb, big5 = (lambda ch: _encodes(ch, "gb2312")), (lambda ch: _encodes(ch, "cp950"))
    if lang == "ja":                                    # the forms only Simplified Chinese writes
        out |= _han(lambda ch: not gb(ch) or _encodes(ch, "euc_jis_2004"))
    elif lang == "ko":                                  # what Korean's Hanja are not
        out |= _han(lambda ch: _encodes(ch, "euc_kr"))
    elif lang in ("zh-cn", "zh-tw"):
        out |= _han(lambda ch: not _encodes(ch, "shift_jis") or gb(ch) or big5(ch))     # the forms only Japanese writes
        if lang == "zh-cn":
            out |= _han(lambda ch: not big5(ch) or gb(ch))                              # the traditional forms
        else:
            out |= _han(lambda ch: not gb(ch) or big5(ch))                              # the simplified forms
    return frozenset(out)


def parse(value, where="script_guard"):
    """A request's, the header's, a shared setting's or the config's value -> "off" or {"lang", "allow_context"};
    None for no value.  ValueError for anything else."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "off":
            return "off"
        if v in LANGS:
            return {"lang": v, "allow_context": True}
    elif isinstance(value, dict) and set(value) <= {"lang", "allow_context"}:
        lang = value.get("lang")
        ctx = value.get("allow_context", True)
        if isinstance(ctx, bool) and isinstance(lang, str) and lang.strip().lower() in ("off",) + LANGS:
            lang = lang.strip().lower()
            return "off" if lang == "off" else {"lang": lang, "allow_context": ctx}
    raise ValueError(USAGE.replace("script_guard", where, 1))


def setting(cfg: dict | None, env) -> dict | str | None:
    """The server's default: STRATA_SCRIPT_GUARD when set, else the run config's "script_guard"."""
    v = env.get("STRATA_SCRIPT_GUARD")
    if v is not None and str(v).strip():
        return parse(v, "STRATA_SCRIPT_GUARD")
    return parse((cfg or {}).get("script_guard"), "config \"script_guard\"")


class Vocab:
    """One tokenizer's banned tokens for one language: each with the banned characters it holds."""

    def __init__(self, tok, lang: str):
        self.chars = _banned(lang)
        self.token_chars: dict[int, frozenset] = {}
        token_bytes = getattr(tok, "token_bytes", None)
        if token_bytes is not None:
            for i in range(len(getattr(tok, "tokens", ()))):
                try:
                    raw = token_bytes(i)
                except (KeyError, IndexError):          # a token outside the byte alphabet: never one to ban
                    continue
                # a token's whole characters: the bytes of one split across tokens are not one of them
                hit = frozenset(c for c in raw.decode("utf-8", "ignore") if c in self.chars)
                if hit:
                    self.token_chars[i] = hit
        self.bias = {i: BAN for i in self.token_chars}


_vocabs: dict = {}                                       # (id(tok), lang) -> (tok, Vocab): built once, on first use
_lock = threading.Lock()


def vocab(tok, lang: str) -> Vocab:
    with _lock:
        hit = _vocabs.get((id(tok), lang))
        if hit is None or hit[0] is not tok:
            hit = _vocabs[(id(tok), lang)] = (tok, Vocab(tok, lang))
        return hit[1]


class Guard:
    """One request's guard.  `active`: the answer is being written (the bias applies); the server sets it."""

    def __init__(self, voc: Vocab, prompt_ids, allow_context: bool):
        self.voc = voc
        self.seen: frozenset = frozenset()
        if allow_context:
            self.seen = frozenset().union(*(voc.token_chars[i] for i in set(prompt_ids) if i in voc.token_chars))
        self.bias = voc.bias if not self.seen else \
            {i: b for i, b in voc.bias.items() if not voc.token_chars[i] <= self.seen}
        self.active = False
        self.restart = False                            # the pass ended to go on with the bias, or after a rewind
        self.ban_next: int | None = None                # the token the next pass, of one token, must not write
        self.one_token = False                          # this pass is that one-token pass
        self.pend_from, self.pend_clean = 0, True       # where the bytes the detokenizer holds began
        self.rewinds = self.passed = 0

    def banned(self, text: str) -> bool:
        return any(c in self.voc.chars and c not in self.seen for c in text)

    def sampling(self, smp: dict | None, n_call: int):
        """The pass's sampling and token count: the request's logit_bias with the bans over it, in the answer."""
        self.one_token = False
        if not self.active:
            return smp, n_call
        own = (smp or {}).get("logit_bias")
        bias = self.bias
        if own or self.ban_next is not None:
            bias = {**bias_api.normalize(own), **bias}
            if self.ban_next is not None:
                bias[self.ban_next] = BAN
                self.ban_next, self.one_token, n_call = None, True, min(n_call, 1)
        return {**(smp or {}), "logit_bias": bias}, n_call

    def follow(self, was_pending: bool, piece: str, still_pending: bool, index: int) -> None:
        """After a token that was let through: where the bytes the detokenizer now holds began, and whether that
        token sent text before them (then a character they complete cannot be taken back)."""
        if still_pending and (not was_pending or piece):
            self.pend_from, self.pend_clean = index, not piece

    def leak(self, was_pending: bool, index: int, pass_len: int) -> int | None:
        """A banned character completed at token `index`: the first token to drop, or None to let it through."""
        start, clean = (self.pend_from, self.pend_clean) if was_pending else (index, True)
        if clean and self.rewinds < REWIND_MAX and index + 1 - start <= pass_len:
            self.rewinds += 1
            return start
        self.passed += 1
        return None
