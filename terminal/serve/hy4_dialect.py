"""Hy4-Flash's text spelling, translated to the Qwen form serve's OutputParser reads.

Hy4's chat template documents (and the model writes) its reasoning and its tool calls with tagged special tokens:

    <think:opensource> ... </think:opensource>
    <tool_calls:opensource>
      <tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>...</tool_call:opensource>
    </tool_calls:opensource>

serve's OutputParser reads `<think>` / `</think>` and `<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>`.  `Hy4Dialect.feed()` turns the
first into the second as the text arrives (a tag split across deltas is held back until it is complete; a value is passed on as it comes, so a long file write still streams),
and nothing else changes: the parser, the three HTTP frontends and the tool-call recovery see the form they were written for.  The model's own template still renders the
history in Hy4's spelling (the prompt side is not touched).
"""
from __future__ import annotations

HYTK = ":opensource"
THINK, THINK_END = f"<think{HYTK}>", f"</think{HYTK}>"
CALLS, CALLS_END = f"<tool_calls{HYTK}>", f"</tool_calls{HYTK}>"
CALL, CALL_END = f"<tool_call{HYTK}>", f"</tool_call{HYTK}>"
KEY, KEY_END = f"<arg_key{HYTK}>", f"</arg_key{HYTK}>"
VAL, VAL_END = f"<arg_value{HYTK}>", f"</arg_value{HYTK}>"

#: the tags each state looks for (anything else is text; a `<` in a value is part of the value)
_TAGS = {
    "text": (THINK, THINK_END, CALLS, CALLS_END, CALL),
    "name": (KEY, CALL_END),
    "key": (KEY_END,),
    "afterkey": (VAL,),
    "value": (VAL_END,),
    "args": (KEY, CALL_END),
}
_MAX_TAG = max(len(t) for ts in _TAGS.values() for t in ts)


class Hy4Dialect:
    """Stateful: one per reply."""

    def __init__(self):
        self.state = "text"
        self.buf = ""
        self.name = ""
        self.acc = ""
        self.key = ""
        self.opened = False
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    def _open(self) -> str:
        self.opened = True
        self.calls += 1
        return f"<tool_call>\n<function={self.name.strip()}>\n"

    def _split(self, tags):
        """-> (index of the earliest complete tag or -1, that tag, index where a possible partial tag starts or -1)."""
        best, which = -1, None
        for t in tags:
            i = self.buf.find(t)
            if i >= 0 and (best < 0 or i < best):
                best, which = i, t
        if best >= 0:
            return best, which, -1
        start = max(0, len(self.buf) - _MAX_TAG + 1)
        j = self.buf.find("<", start)
        while j >= 0:
            tail = self.buf[j:]
            if any(t.startswith(tail) for t in tags):
                return -1, None, j
            j = self.buf.find("<", j + 1)
        return -1, None, -1

    # ------------------------------------------------------------------ the stream
    def feed(self, text: str = "", final: bool = False) -> str:
        self.buf += text
        out: list[str] = []
        while True:
            tags = _TAGS[self.state]
            i, tag, hold = self._split(tags)
            if tag is None:
                end = len(self.buf) if (hold < 0 or final) else hold
                chunk, self.buf = self.buf[:end], self.buf[end:]
                self._text(chunk, out)
                break
            chunk, self.buf = self.buf[:i], self.buf[i + len(tag):]
            self._text(chunk, out)
            self._tag(tag, out)
        if final:
            out.append(self._finish())
        return "".join(out)

    def _text(self, chunk: str, out: list) -> None:
        if not chunk:
            return
        if self.state in ("text", "value"):
            out.append(chunk)
        elif self.state in ("name", "key"):
            self.acc += chunk
        # "afterkey" / "args": the whitespace between the tags is dropped

    def _tag(self, tag: str, out: list) -> None:
        s = self.state
        if s == "text":
            if tag == THINK:
                out.append("<think>")
            elif tag == THINK_END:
                out.append("</think>")
            elif tag == CALL:
                self.state, self.acc, self.name, self.opened = "name", "", "", False
            # the tool_calls wrapper has no counterpart in the Qwen form
        elif s == "name":
            self.name, self.acc = self.acc, ""
            out.append(self._open())
            if tag == KEY:
                self.state = "key"
            else:
                out.append("</function>\n</tool_call>")
                self.state = "text"
        elif s == "key":
            self.key, self.acc, self.state = self.acc.strip(), "", "afterkey"
        elif s == "afterkey":
            out.append(f"<parameter={self.key}>\n")
            self.state = "value"
        elif s == "value":
            out.append("\n</parameter>\n")
            self.state = "args"
        elif s == "args":
            if tag == KEY:
                self.state = "key"
            else:
                out.append("</function>\n</tool_call>")
                self.state = "text"

    def _finish(self) -> str:
        """The reply ended inside a call: say what was open, so the parser sees a cut call the way it would see a cut Qwen call."""
        tail = ""
        if self.state == "name":
            self.name, self.acc = self.acc, ""
            tail = self._open().rstrip("\n")
        elif self.state == "value":
            tail = ""            # the parser's own rules for a cut value apply to the text that was passed on
        self.state = "text"
        return tail


def force_text(force):
    """forced_call's opening of the call the reply must make (Qwen's spelling) in Hy4's: `<tool_call>\\n<function=NAME>\\n` -> `<tool_calls:opensource><tool_call:opensource>NAME`.
    It is written into the prompt (and fed to the parser, which reads it through the translator)."""
    if not isinstance(force, str):
        return force
    import re
    m = re.fullmatch(r"<tool_call>\n(?:<function=([^>\n]+)>\n)?", force)
    if not m:
        return force
    return CALLS + CALL + (m.group(1) or "")


def translate(text: str) -> str:
    """The whole text at once (tests, the Monitor's views)."""
    return Hy4Dialect().feed(text, final=True)


# ---- reasoning effort (serve/frontend.py use_family): the template takes no_think | high | max ----
EFFORT = {
    "none": None, "off": None, "no_think": None, "disabled": None, "false": None,
    "minimal": "high", "min": "high", "low": "high", "medium": "high", "med": "high", "high": "high", "auto": "high", "default": "high",
    "xhigh": "max", "max": "max", "maximum": "max",
}


def normalize_effort(value) -> str:
    """Clamp any client reasoning_effort / thinking_mode into Hy4-Flash's three valid template modes: no_think | high | max."""
    if value is False:
        return "no_think"
    if value is True or value is None or value == "":
        return "high"
    key = str(value).strip().lower()
    if key in EFFORT:
        mapped = EFFORT[key]
        return "no_think" if mapped is None else mapped
    return "high"


def template_kwargs(kwargs: dict) -> dict:
    """The kwargs the Hy4 template reads, from the ones the frontends build (they say `enable_thinking: false` or `reasoning_effort: <level>`): thinking off is
    reasoning_effort=no_think; any external client effort (minimal, low, medium, xhigh, ...) is clamped into {no_think, high, max} before chat_template.jinja runs."""
    out = dict(kwargs)
    if "thinking_mode" in out:
        tm = out.pop("thinking_mode")
        if out.get("reasoning_effort") in (None, ""):
            out["reasoning_effort"] = tm
    if out.get("enable_thinking") is False:
        out["reasoning_effort"] = "no_think"
    elif out.get("reasoning_effort") not in (None, ""):
        out["reasoning_effort"] = normalize_effort(out["reasoning_effort"])
    else:
        out.pop("reasoning_effort", None)
    if out.get("reasoning_effort") == "no_think":
        out["enable_thinking"] = False
    return out
