"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the core part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import codecs
import os
import re
from serve.frontend import CALL_END, CALL_START, FUNC_END, FUNC_START, Event

from serve.server_parts.base import RATE_MIN_SPAN_S


class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


OPENAI_MAX_STOP = 4                                  # OpenAI's limit on `stop`


#: What a request may ask for that the all-cards split engine cannot do (its GEN parser has no such keys): answered with a 400 before anything is sent, not with an error in
#: the middle of a reply. logit_bias is refused the same way by logit_bias.validate_request (the split engine does not report logit_bias=1).
SPLIT_REFUSED_FEATURES = ("logprobs", "logit_bias")


def top_logprobs(req: dict):
    """The K for the engine's logprobs=K, or None when the request did not ask.

    OpenAI: "logprobs": true with "top_logprobs" 0..20; the legacy completions form,
    "logprobs": N, is taken as N."""
    lp = req.get("logprobs")
    if lp is True:
        k = req.get("top_logprobs")
        return max(0, min(20, k)) if isinstance(k, int) and not isinstance(k, bool) else 0
    if isinstance(lp, int) and not isinstance(lp, bool) and lp >= 0:
        return min(20, lp)
    return None


def allowed_ids(req: dict) -> list:
    """`allowed_token_ids`: the ids a caller named, or [] (vLLM's semantics, see LogprobEntries)."""
    ids = req.get("allowed_token_ids")
    if not isinstance(ids, (list, tuple)):
        return []
    return [x for x in ids if isinstance(x, int) and not isinstance(x, bool) and x >= 0]


def batch_slots_on(engine) -> bool:
    """Does this engine run requests in batch slots ("parallel", whose `BT` lines carry no LP lines)?

    Asked of each engine behind a ReplicaEngine: its own `batch` is how many requests it can run at once, so it
    counts a replica without slots as one (serve/replicas.py) -- a server with `--replicas 2` and no `--parallel`
    has none, and reads every reply by the GEN path, which does write them."""
    return any(int(getattr(e, "batch", 0) or 0) for e in (getattr(engine, "engines", None) or [engine]))


NO_LINE = object()                                   # StrataEngine.generate: no line held after a token


def token_limit(value, name: str):
    """#4037: a reply's token limit (max_tokens, max_completion_tokens) is a whole number: 1 or more, or -1 / 0 for no cap
    (docs/DETAILS.md).  None (absent or null) stays None.  A whole number written as a float (4096.0) or as a string of
    digits with an optional sign ("1024": clients rely on it, as int() always took it) is converted.  A boolean, a fraction
    (2.5, "2.5"), any other string or any other type is refused: int() used to turn true into 1 and 2.5 into 2.
    ValueError = a 400."""
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not value.is_integer()) \
            or value < -1:
        raise ValueError(f"{name} must be a whole number, 1 or more, or -1 or 0 for no cap")
    return int(value)


def stop_strings(req: dict) -> list[str]:
    """The request's stop strings: OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences`
    (a list).  Empty strings are left out.  A bad value is a ValueError, so a 400 before anything is generated."""
    req = req or {}
    if req.get("stop") is not None:
        stop = req["stop"]
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > OPENAI_MAX_STOP:
            raise ValueError(f"stop takes at most {OPENAI_MAX_STOP} strings")
    elif req.get("stop_sequences") is not None:
        stop = req["stop_sequences"]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop_sequences must be a list of strings")
    else:
        return []
    return [x for x in stop if x]


class StopMatcher:
    """Cuts the answer's text at the first stop string.  Text that could still be the start of a stop string is held
    back (only that tail, never more), so a stop string split across tokens or chunks is still found and never sent."""

    def __init__(self, stops: list[str]):
        self.stops, self.held, self.hit = stops, "", None

    def push(self, text: str) -> str:
        """-> the text that is safe to send now; after a match, self.hit is the stop string and nothing more comes."""
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0                                      # the longest end of buf that a stop string starts with
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held


# Windows names that open a device whatever their extension or folder (Microsoft, "Naming Files, Paths, and Namespaces")
_WIN_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in "123456789\u00b9\u00b2\u00b3"),
                *(f"lpt{i}" for i in "123456789\u00b9\u00b2\u00b3")}


def slot_filename_problem(name) -> str | None:
    """/slots/0 {"filename": NAME}: why NAME is not one plain file name inside the slot save path, on Linux and on
    Windows alike (None when it is).  No separator or drive (`D:x`), no stream (`a:b`), no device (`NUL.txt`), no
    trailing dot or space (Windows drops them), no control character, no leading dot (temporary files use it)."""
    if not isinstance(name, str) or not name:
        return "must be a non-empty string"
    if len(name.encode("utf-8", "surrogatepass")) > 200:
        return "is too long (at most 200 bytes)"
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return "must not contain control characters"
    if any(c in name for c in '/\\:*?"<>|'):
        return "must be a plain file name (no / \\ : * ? \" < > |)"
    if name.startswith("."):
        return "must not start with a dot"
    if name.endswith((".", " ")):
        return "must not end with a dot or a space"
    if name.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES:
        return "names a Windows device"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid Unicode"
    return None


def slot_save_dir(value, base: str | None = None) -> str:
    """The slot save path as one absolute, normalized directory (relative to `base`, else the server's working
    directory), created private to this user when missing.  ValueError for anything else."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slot_save_path must be a directory path (a string)")
    if any(ord(c) < 32 for c in value):
        raise ValueError("slot_save_path must not contain control characters")
    path = os.path.abspath(os.path.join(base, value) if base and not os.path.isabs(value) else value)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path):
        raise ValueError(f"slot_save_path {path} is not a directory")
    return path


def closest_declared(name: str, declared: list[str]) -> str | None:
    """#5904 / #5933: the declared tool name an undeclared one comes closest to (`read_file` -> `read`), for the hint in
    strata_rejected_tool_calls; None when none is clearly close.  The one declared name the written one contains or is
    contained in (ignoring case and `_` / `-` / `.`), else the best difflib match at a ratio of 0.75 or more."""
    import difflib
    if not name or not declared:
        return None
    norm = lambda s: re.sub(r"[_\-.\s]", "", s.lower())     # noqa: E731
    n = norm(name)
    inside = [d for d in declared if norm(d) and (norm(d) in n or n in norm(d))]
    if len(inside) == 1:
        return inside[0]
    best = difflib.get_close_matches(name.lower(), [d.lower() for d in declared], n=1, cutoff=0.75)
    if best:
        return next(d for d in declared if d.lower() == best[0])
    return None


TOOL_RETRY_MIN_ROOM = 64     # (a): the retry runs only with this many output tokens left after its note


_MARKUP_OPENER = re.compile(r"<tool_call>|<function=")


def _line_start(s: str, i: int) -> bool:
    head = s[:i].rstrip(" \t")
    return not head or head.endswith("\n")


def dead_markup_tail(text: str) -> tuple[int, str] | None:
    """The tool-call markup a reply's text ends in: (where the run of blocks at the end starts, "cut_off" | "unreadable"),
    or None when the text does not end in one.  A block opens with `<tool_call>` (or a bare `<function=`) at a line
    start, outside a ``` fence; "cut_off": the last block has no closer (`</tool_call>`, or `</function>` for a bare one),
    "unreadable": it closed (the parser read it as no call).  Text after a block's closer means the reply went on: None."""
    s = text.rstrip()
    start, reason, end = None, None, len(s)
    while True:
        part = s[:end]
        iw = max((m.start() for m in re.finditer(re.escape(CALL_START), part) if _line_start(part, m.start())), default=-1)
        ib = max((m.start() for m in re.finditer(re.escape(FUNC_START), part) if _line_start(part, m.start())), default=-1)
        bare = ib > iw and (iw < 0 or CALL_END in part[iw:ib])
        i = ib if bare else iw
        if i < 0:
            break
        body = part[i:]
        closer = FUNC_END if bare else CALL_END
        k = body.find(closer)
        if k >= 0 and body[k + len(closer):].strip():
            break                                       # the block closed and text follows it: not the reply's end
        closed = k >= 0
        if reason is None:
            reason = "unreadable" if closed or body.endswith(FUNC_END) else "cut_off"
        elif not closed:
            break                                       # an unclosed block before the last one: the run starts after it
        start, end = i, len(part[:i].rstrip())
    if start is None or s[:start].count("```") % 2:
        return None
    return start, reason


def markup_names(text: str) -> list[str]:
    """The names written in the `<function=NAME>` openers of a piece of text."""
    return [m.group(1).strip() for m in re.finditer(r"<function=([^<>\n]+?)>", text)]


def tool_retry_note(reason: str, names: list[str], declared: list[str], near: str | None) -> str:
    """(a): the user-turn note the one retry reads: why the call was not run, the declared tools, the hint."""
    what = {"rejected": "it names " + ", ".join(f"`{x}`" for x in names) + ", which is not one of the tools you can call",
            "cut_off": "it was cut off before its end",
            "unreadable": "it is not in a form that can be read"}[reason]
    listed = ", ".join(declared[:40]) + (f" and {len(declared) - 40} more" if len(declared) > 40 else "")
    hint = f" Did you mean `{near}`?" if near else ""
    return (f"[Strata server] Your last reply ended with a tool call that was not run: {what}.{hint} The tools you can "
            f"call are: {listed}. Write the call again in this form, with a tool from that list:\n"
            f"{CALL_START}\n{FUNC_START}TOOL_NAME>\n<parameter=PARAMETER_NAME>\nvalue\n</parameter>\n{FUNC_END}\n{CALL_END}\n"
            "or answer in plain text if no tool is needed.")


class ToolMarkupHold:
    """(a) The answer's text from a tool-call opener at a line start is held back until the reply goes on past the block
    (then it is sent as it is) or ends in it (then the server may write the reply's end again, once, in place of it).
    Unarmed it holds nothing; it always keeps the text it sent, for strata_unread_tool_markup."""

    def __init__(self, armed: bool):
        self.armed = armed
        self.held: list[Event] = []
        self.text = ""
        self.closed = False                             # a closer (`</tool_call>` / `</function>`) is in the held text
        self.sent: list[str] = []
        self.prev = ""                                  # the last characters sent (a line start at the reply's start)
        self.fences = 0                                 # ``` marks sent so far
        self.calls = False                              # a tool call came out: the reply does not end in dead markup
        self.started: set = set()                       # the ids of calls announced (tool_start) ...
        self.finished: set = set()                      # ... and of those whose final tool_call came

    def _send(self, ev: Event) -> Event:
        if ev.text:
            self.sent.append(ev.text)
            self.prev = (self.prev + ev.text)[-16:]
            self.fences += ev.text.count("```")
        return ev

    def _opener(self, text: str) -> int:
        for m in _MARKUP_OPENER.finditer(text):
            seen = self.prev + text[:m.start()]
            if _line_start(seen, len(seen)) and (self.fences + text[:m.start()].count("```")) % 2 == 0:
                return m.start()
        return -1

    def push(self, evs: list) -> list:
        out = []
        for ev in evs:
            if ev.kind != "content":
                if ev.kind in ("tool_start", "tool_call"):
                    self.calls = True
                    (self.started if ev.kind == "tool_start" else self.finished).add(ev.call.id)
                out += self.release()
                out.append(ev)
                continue
            text = ev.text or ""
            if not self.held:
                p = self._opener(text) if self.armed and text else -1
                if p < 0:
                    out.append(self._send(ev))
                    continue
                if p:
                    out.append(self._send(Event("content", text[:p])))
                ev, text = Event("content", text[p:]), text[p:]
            self.held.append(ev)
            self.text += text
            if not self.closed:                         # (until a closer arrives the block is open: held, no scan)
                near = self.text[-(len(text) + len(CALL_END)):]
                self.closed = CALL_END in near or FUNC_END in near
            if self.closed and text.strip() and dead_markup_tail(self.text) is None:
                out += self.release()                   # text after the block: the reply goes on
        return out

    def release(self) -> list:
        evs, self.held, self.text, self.closed = self.held, [], "", False
        return [self._send(ev) for ev in evs]

    def drop(self) -> str:
        text, self.held, self.text, self.closed = self.text, [], "", False
        return text

    def reply_text(self) -> str:
        return "".join(self.sent)


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """#471: the prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way (then the reused ones plus those read).  /metrics' history and totals count these, so a
    cancelled read is neither recorded as the whole prompt nor given a rate from tokens it never read.  An engine
    before 0.1.36 does not say (no `prompt_read`): the whole prompt, as before."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def carry_leg(carry: dict, leg: dict) -> None:
    """#1869: a request that decoded in a batch slot and went back to the solo path ends with the solo path's DONE, which
    replaces the engine's figures: keep the slot leg's decode time, tokens and prompt figures to add to it."""
    for key in ("decode_ms", "generated"):
        previous, value = carry.get(key, 0), leg.get(key)
        carry[key] = previous + value if previous is not None and value is not None else None
    carry.setdefault("prompt_ms", float(leg.get("prompt_ms") or 0.0))
    if leg.get("prompt_read") is not None:
        carry.setdefault("prompt_read", int(leg["prompt_read"]))


def add_carry(carry: dict, last: dict) -> dict:
    """The request's totals: the earlier legs' decode time and tokens plus the final one's; the prompt figures of the leg
    that read the prompt (the solo continuation reads next to nothing)."""
    out = {**last, "prompt_ms": max(float(last.get("prompt_ms") or 0.0), carry.get("prompt_ms", 0.0))}
    for key in ("decode_ms", "generated"):
        previous, value = carry.get(key, 0), last.get(key)
        out[key] = previous + value if previous is not None and value is not None else None
    if "prompt_read" in carry:
        out["prompt_read"] = carry["prompt_read"]
    return out


def answer_speed(last: dict, generated: int, wall_s: float | None = None) -> float | None:
    """The one definition of "answer speed": every token the model produced (thinking and answer, the tokens it kept - not
    the MTP drafts it threw away) over the decode time, first generated token to last.  The engine's DONE line holds both
    numbers; /metrics (`decode_tok_s`), the timings (`predicted_per_second`), the Terminal monitor and the app all use them.
    Only an engine that sent no DONE falls back to the server's own clock from the first token, and a clock younger than
    RATE_MIN_SPAN_S says nothing (it printed 1000000 tok/s for a one-token reply).  Prompt reading is never in it."""
    produced, decode_ms = last.get("generated"), last.get("decode_ms")
    if produced and produced >= 2 and decode_ms and decode_ms > 0:       # one token has no first-to-last span (a 1-token probe printed 1,250 tok/s)
        return round(produced / (decode_ms / 1000), 1)
    if generated and wall_s is not None and wall_s >= RATE_MIN_SPAN_S:
        return round(generated / wall_s, 1)
    return None


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    if last.get("prompt_read") is not None and last.get("finish") != "cancel":
        # #1870: the engine says how many prompt tokens it read: the slot's own session may have held the rest
        # (not the conversation cache), so the rate is not made of tokens it never read
        cache_n = max(cache_n, max(0, prompt_tokens - int(last["prompt_read"])))
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), last.get("decode_ms")
    decoded = last.get("generated")          # explicit zero and missing native work are not the API's output count
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1) if decode_ms is not None else None,
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded and decode_ms is not None else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def request_stats(segments: list[dict]) -> dict:
    """Keep input/cache accounting at the original API boundary and total native work across continuations."""
    if not segments:
        return {}
    result = dict(segments[-1])
    for key in ("reused", "prompt_read", "chunk"):
        if key in segments[0]:
            result[key] = segments[0][key]
        else:
            result.pop(key, None)
    for key in ("prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted",
                "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb", "experts_streamed"):
        if key in ("decode_ms", "generated") and any(segment.get(key) is None for segment in segments):
            result[key] = None
        elif any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def prefill_leg_stats(earlier: list[dict], last: dict, prompt_tokens: int) -> dict:
    """Total yielded reads once, retaining only reuse from before this API request."""
    legs = [*earlier, last]
    total = request_stats(legs)
    if all(leg.get("prompt_read") is not None for leg in legs):
        fresh = max(0, prompt_tokens - int(total.get("reused") or 0))
        total["prompt_read"] = min(fresh, sum(int(leg["prompt_read"]) for leg in legs))
    return total
