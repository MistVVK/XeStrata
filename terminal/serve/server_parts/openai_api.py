"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the OpenAI part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import collections
import codecs
import json
import math
from serve.structured import extract_json


# The engine's last PP line stops short of the prompt: up to `--short-read` tokens (generate.cpp's `short_read`, 64 by
# default) are read through the verify windows instead of the batched path, so the batched path never reports the whole
# prompt.  A PP that close to the end says the prompt is read.  It matters because a prompt shorter than one `--prefill`
# chunk sends exactly one PP, that one: a client would see a bar that appears already full.  Such a line is dropped, and
# the answer's first token ends the progress for the client anyway.
PP_DONE_TAIL = 64


class LPLines(collections.deque):
    """One request's LP lines, tagged with its cancel Event (one per request, handed unchanged from the handler
    through Service.run to Engine.generate).  The last events of a request - the parser's and the stop strings'
    tail - come after it has left the queue, when the next request may already have replaced the engine's lp_lines
    with its own: LogprobEntries reads only the lines tagged with its own request's Event."""

    def __init__(self, owner=None):
        super().__init__()
        self.owner = owner


class LogprobEntries:
    """The engine's LP lines as OpenAI's logprobs.content entries.

    Each token's "token" is the text it completes (an incremental UTF-8 decode, as the reply's
    text is made), so a character split across tokens lands in the token that finishes it and
    the tokens, joined, spell the reply.  "bytes" are the token's own bytes.  The stop token
    ends the reply and is not listed (OpenAI does not list it either).  Tokens that make no
    text of the reply are left out, because spell() gives each piece of text the tokens that
    spell it.

    `allowed_token_ids`: only those ids are kept, and the logprobs are renormalised over them,
    so a caller asking about specific candidates gets a distribution over exactly those instead
    of a top-K list it has to hope contains them.  vLLM, which named the field, masks generation
    with these ids instead (every other logit -inf before sampling) and reports the unmasked
    distribution; here they filter what is reported, so the chosen token can be outside them.

    On `--parallel`: only a `max_tokens` 1 request gets entries (an admission, decoded through the single-token path
    that writes an LP line); the slot's `BT` lines carry none, so a longer reply is refused (400) before it starts.
    """

    def __init__(self, svc, allowed=(), owner=None):
        self.tok, self.stop_ids = svc.tok, svc.stop_ids
        self.allowed = [i for i in allowed]
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = []                          # entries read but not yet given to a piece of text
        self.owner = owner                         # the request's cancel Event (LPLines)
        self.lines = None                          # this request's current LPLines, kept once seen

    def spell(self, eng, text: str) -> list:
        """The entries whose tokens spell `text`, the next piece of the reply."""
        self.pending += self.take(eng)
        at = [0]
        for e in self.pending:
            at.append(at[-1] + len(e["token"]))
        joined = "".join(e["token"] for e in self.pending)
        i = joined.find(text)
        while i >= 0 and text:
            if i in at:
                a = at.index(i)                    # the first token there
                end = i + len(text)
                b = next((j for j in range(a + 1, len(at)) if at[j] == end), None)
                if b is not None:
                    out, self.pending = self.pending[a:b], self.pending[b:]
                    return out
            i = joined.find(text, i + 1)
        out, self.pending = self.pending, []
        return out

    def raw(self, t):
        if hasattr(self.tok, "token_bytes"):
            return self.tok.token_bytes(t)
        return self.tok.decode([t]).encode("utf-8")

    def take(self, eng) -> list:
        # the engine's current lines are this request's only when they carry its Event (a later generate pass
        # of the same request makes new ones: the old ones are read to the end first); another request's never are.
        # A batch admission keeps its own lines on the request's cancel Event: the engine's is one attribute, and
        # the next request to read control lines may already have replaced it.
        own = getattr(self.owner, "_strata_lp_lines", None)
        cur, old = (own if own is not None else getattr(eng, "lp_lines", None)), None
        if cur is not self.lines and getattr(cur, "owner", None) is self.owner:
            old, self.lines = self.lines, cur
        out = []
        for lines in (old, self.lines):
            while lines:
                t, line = lines.popleft()
                if t in self.stop_ids:
                    continue
                f = line.split()
                if len(f) < 2:
                    continue
                try:                               # "nan" (no head logits for that window row), "-nan(ind)"
                    lp = float(f[1])               # (MSVC) or a cut line: no entry for that token
                    alts = [(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
                except ValueError:
                    continue
                if not math.isfinite(lp) or not all(math.isfinite(v) for _, v in alts):
                    continue
                b = self.raw(t)
                e = {"token": self.inc.decode(b), "logprob": lp, "bytes": list(b),
                     "top_logprobs": []}
                if self.allowed:
                    alts = [(i, v) for i, v in alts if i in self.allowed]
                for i, v in alts:
                    alt = self.raw(i)
                    e["top_logprobs"].append({"token": alt.decode("utf-8", "replace"),
                                              "logprob": v, "bytes": list(alt)})
                if self.allowed:
                    e = self._renormalise(e, t, lp)
                out.append(e)
        return out

    def _renormalise(self, e, t, lp):
        """Keep only the named ids and normalise over them, so they sum to 1.

        The line carries the chosen token's own logprob on its head and the named ids after it, and never
        repeats the chosen one among them.  So the chosen token is put back when the caller named it: with
        `allowed_token_ids` naming only the token the model chose, that token is then the whole distribution,
        at 1 (logprob 0), which is what vLLM's mask gives -- here it used to leave nothing to normalise over,
        which returned None, the entry was appended all the same, and the reply failed on e["token"].  A line
        that carries none of the named ids has nothing to normalise over either: the entry keeps the chosen
        token and its own logprob, with no alternatives.  It is never None and never dropped, because the
        reply's tokens have to keep spelling its text.
        """
        cand = e["top_logprobs"]
        own = None
        if t in self.allowed:                      # named, and not repeated on the line: it is a candidate
            b = self.raw(t)
            own = {"token": b.decode("utf-8", "replace"), "logprob": lp, "bytes": list(b)}
            cand = [own] + cand
        if not cand:
            return e                               # no named id observed for this token: its own logprob, no top
        mx = max(x["logprob"] for x in cand)
        se = sum(math.exp(x["logprob"] - mx) for x in cand)
        lse = mx + math.log(se)
        for x in cand:
            x["logprob"] -= lse
        cand.sort(key=lambda x: -x["logprob"])     # most likely first, as the unfiltered top-k already is
        e["top_logprobs"] = cand
        if own is not None:                        # the chosen token is in the distribution: report its own value
            e["logprob"] = own["logprob"]           # (it used to copy the first alternative's, another token's)
        return e


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    lps = {}                                       # "content" / "reasoning_content" -> logprob entries
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        for k, v in (c["choices"][0].get("logprobs") or {}).items():
            lps.setdefault(k, []).extend(v)
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning).strip() and "".join(reasoning).strip() == "".join(content).strip():
        reasoning = []              # a turn with no </think>: the text is the answer; it is not also reasoning
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if lps:
        out["choices"][0]["logprobs"] = {"content": lps.get("content", []), "refusal": None, **(
            {"reasoning_content": lps["reasoning_content"]} if lps.get("reasoning_content") else {})}
    if last.get("timings"):
        out["timings"] = last["timings"]
    if last.get("strata_rejected_tool_calls"):
        out["strata_rejected_tool_calls"] = last["strata_rejected_tool_calls"]
    if last.get("strata_unread_tool_markup"):
        out["strata_unread_tool_markup"] = last["strata_unread_tool_markup"]
    return out


def logprob_span(text: str, part: str, entries: list):
    """The entries whose tokens spell `part`, a piece of `text`, or None when the tokens do not
    spell `text` or no token boundary falls at both ends of `part`."""
    if "".join(e["token"] for e in entries) != text:
        return None
    start = text.find(part)
    if start < 0:
        return None
    at, first = 0, None
    for i, e in enumerate(entries):
        if at == start and first is None:
            first = i
        if at == start + len(part) and first is not None:
            return entries[first:i]
        at += len(e["token"])
    return entries[first:] if first is not None and at == start + len(part) else None


def _same_json(a: str, b: str) -> bool:
    """Are two strings the same JSON value (whitespace and key order aside)?  No -> False."""
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return False


def _structured_logprobs(choice: dict, content: str):
    """`content`'s logprobs, and the content the tokens actually spell.

    Returns `(content, logprobs)`: on failure `(the content as validated, None)`.

    `structured_chunks` returns the canonical JSON the validator produced, which the model never
    generated, so its tokens do not spell it.  Two honest outcomes:

      1. a run of tokens spells what is being returned -> use it as it is;
      2. the tokens spell the model's own JSON, which is the same JSON value (the fenced-with-spaces
         case, the common one) -> return the model's form, so `''.join(token) == content` again.

    Neither holds (a rearranged key order, or a reply that is not that JSON at all) and there are no
    logprobs: a caller that joins the tokens must not silently get something that does not match.
    """
    lps = choice.get("logprobs")
    if not lps or not lps.get("content"):
        return content, None
    raw = choice["message"]["content"] or ""
    entries = lps["content"]
    if raw == content:                          # not rewritten: the entries are already its
        return content, _with_reasoning(lps, entries)
    span = logprob_span(raw, content, entries)
    if span is not None:
        return content, _with_reasoning(lps, span)
    inner = extract_json(raw)
    if inner and _same_json(inner, content):
        span = logprob_span(raw, inner, entries)
        if span is not None:
            return inner, _with_reasoning(lps, span)   # the model's own form: the tokens spell it
    return content, None


def _with_reasoning(lps: dict, entries: list) -> dict:
    out = {"content": entries}
    if lps.get("reasoning_content"):
        out["reasoning_content"] = lps["reasoning_content"]
    return out
