"""Inkling-2-Mini's wire dialect, translated to Strata's OutputParser / JsonOutputParser form.

Inkling-2-Mini renders its assistant output with explicit channel tokens:
    <|message_model|><|content_thinking|> ... <|end_message|>
    <|message_model|><|content_text|> ... <|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_text|>...<|end_message|>
    <|content_model_end_sampling|>

`InklingDialect.feed()` streams these tokens into:
    <think>...</think>
    plain text
    <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>

Using JSON inside `<tool_call>...</tool_call>` (read by InklingOutputParser -> JsonOutputParser) preserves
arbitrary XML special characters (`<`, `>`, `&`, `</parameter>`), nested quotes, booleans/numbers/arrays,
and multiline code losslessly while reusing all downstream OpenAI / Anthropic / Responses / MCP streaming!
"""
from __future__ import annotations

import json
import re

MSG_MODEL = "<|message_model|>"
CONTENT_THINK = "<|content_thinking|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_XML = "<|content_xml|>"
END_MSG = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"

_HEADER_TAGS = (CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT, MSG_MODEL)
_IDLE_TAGS = (MSG_MODEL, CONTENT_THINK, CONTENT_TEXT, CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MSG, END_SAMPLING, ENDOFTEXT)
_BODY_TAGS = (END_MSG, END_SAMPLING, ENDOFTEXT, MSG_MODEL)
_MAX_TAG = max(len(t) for t in _HEADER_TAGS)


def _safe_json_call(name: str, arguments) -> str:
    """Serialize a tool call as `<tool_call>\\n{...}\\n</tool_call>` with `</` escaped as `<\\/` so literal `</tool_call>` in args never breaks framing."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False).replace("</", "<\\/")
    return f"<tool_call>\n{payload}\n</tool_call>"


def _parse_tool_body(header_name: str, body: str, kind: str) -> str:
    raw = body.strip()
    fallback_name = header_name.strip()
    if kind == "tool_json":
        try:
            obj = json.loads(raw)
        except ValueError:
            # Recover common trailing garbage / unclosed brace if cut
            obj = None
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except ValueError:
                    obj = None
        if isinstance(obj, dict):
            name = str(obj.get("name") or fallback_name).strip()
            args = obj.get("args") if "args" in obj else obj.get("arguments")
            if args is None:
                args = {k: v for k, v in obj.items() if k != "name"} if "name" in obj else {}
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except ValueError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}
            if name:
                return _safe_json_call(name, args)
        if fallback_name:
            return _safe_json_call(fallback_name, {"raw": raw} if raw else {})
        return raw
    # tool_text / content_xml: check if it is JSON or XML `<function=...>` / raw text
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                name = str(obj.get("name") or fallback_name).strip()
                args = obj.get("args") if "args" in obj else obj.get("arguments", {})
                if name and isinstance(args, dict):
                    return _safe_json_call(name, args)
        except ValueError:
            pass
    if "<function=" in raw:
        return f"<tool_call>\n{raw}\n</tool_call>"
    if "<tool_call" in raw or "<invoke" in raw:
        m_call = re.search(r'<(?:tool_call|invoke)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:tool_call|invoke)>', raw, re.S)
        if m_call:
            xname = m_call.group(1).strip()
            xbody = m_call.group(2)
            xargs = {
                pm.group(1).strip(): pm.group(2)
                for pm in re.finditer(r'<(?:param|parameter)\b[^>]*\bname=["\']([^"\']+)["\'][^>]*>(.*?)</(?:param|parameter)>', xbody, re.S)
            }
            if xname:
                return _safe_json_call(xname, xargs)
    if fallback_name:
        return _safe_json_call(fallback_name, {"input": raw} if raw else {})
    return raw


class InklingDialect:
    """Stateful streaming translator: one instance per assistant reply."""

    def __init__(self):
        self.state = "idle"          # idle | header | think | text | tool_json | tool_text
        self.buf = ""
        self.header_acc = ""
        self.tool_acc = ""
        self.calls = 0
        self._in_think = False

    def _tags_for(self):
        if self.state == "idle":
            return _IDLE_TAGS
        if self.state == "header":
            return _HEADER_TAGS
        return _BODY_TAGS

    def _split(self, tags):
        best, which = -1, None
        for t in tags:
            i = self.buf.find(t)
            if i >= 0 and (best < 0 or i < best):
                best, which = i, t
        if best >= 0:
            return best, which, -1
        start = max(0, len(self.buf) - _MAX_TAG + 1)
        j = self.buf.find("<|", start)
        while j >= 0:
            tail = self.buf[j:]
            if any(t.startswith(tail) for t in tags):
                return -1, None, j
            j = self.buf.find("<|", j + 1)
        if self.buf.endswith("<") and any(t.startswith("<") for t in tags):
            return -1, None, len(self.buf) - 1
        return -1, None, -1

    def feed_channels(self, text: str = "", final: bool = False) -> list[tuple[str, str]]:
        """Stream raw Inkling-2-Mini output into typed `("reasoning" | "content", text)` chunks.
        Tool invocations are emitted in the `"content"` channel as `<tool_call>\\n{...}\\n</tool_call>` JSON frames."""
        self.buf += text
        out: list[tuple[str, str]] = []
        while True:
            tags = self._tags_for()
            i, tag, hold = self._split(tags)
            if tag is None:
                end = len(self.buf) if (hold < 0 or final) else hold
                chunk, self.buf = self.buf[:end], self.buf[end:]
                self._emit_typed(chunk, out)
                break
            chunk, self.buf = self.buf[:i], self.buf[i + len(tag):]
            self._emit_typed(chunk, out)
            self._handle_tag_typed(tag, out)
        if final:
            self._finish_typed(out)
        return out

    def feed(self, text: str = "", final: bool = False) -> str:
        """Stream raw Inkling-2-Mini output into `<think>...</think>` + text + `<tool_call>...</tool_call>` string."""
        parts: list[str] = []
        for kind, chunk in self.feed_channels(text, final=final):
            if kind == "reasoning":
                if not self._in_think:
                    parts.append("<think>")
                    self._in_think = True
                parts.append(chunk)
            else:
                if self._in_think:
                    parts.append("</think>")
                    self._in_think = False
                parts.append(chunk)
        if final and self._in_think:
            parts.append("</think>")
            self._in_think = False
        return "".join(parts)

    def _emit_typed(self, chunk: str, out: list[tuple[str, str]]) -> None:
        if not chunk:
            return
        if self.state == "think":
            # Inkling-2-Mini has no thinking mode (off / "none" only): silently discard any raw <|content_thinking|> span.
            return
        elif self.state in ("idle", "text"):
            out.append(("content", chunk))
        elif self.state == "header":
            self.header_acc += chunk
        elif self.state in ("tool_json", "tool_text"):
            self.tool_acc += chunk

    def _close_channel_typed(self, out: list[tuple[str, str]]) -> None:
        if self.state in ("tool_json", "tool_text"):
            rendered = _parse_tool_body(self.header_acc, self.tool_acc, self.state)
            if rendered:
                if "<tool_call>" in rendered:
                    self.calls += 1
                out.append(("content", rendered))
            self.tool_acc = ""
            self.header_acc = ""
        elif self.state == "header":
            self.header_acc = ""

    def _handle_tag_typed(self, tag: str, out: list[tuple[str, str]]) -> None:
        if tag in (END_SAMPLING, ENDOFTEXT):
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == MSG_MODEL:
            self._close_channel_typed(out)
            self.state = "header"
            self.header_acc = ""
            return
        if tag == END_MSG:
            self._close_channel_typed(out)
            self.state = "idle"
            return
        if tag == CONTENT_THINK:
            self.header_acc = ""
            self.state = "think"
            return
        if tag == CONTENT_TEXT:
            self.header_acc = ""
            self.state = "text"
            return
        if tag == CONTENT_TOOL_JSON:
            self.state = "tool_json"
            self.tool_acc = ""
            return
        if tag in (CONTENT_TOOL_TEXT, CONTENT_XML):
            self.state = "tool_text"
            self.tool_acc = ""
            return

    def _finish_typed(self, out: list[tuple[str, str]]) -> None:
        if self.state in ("tool_json", "tool_text") and (self.tool_acc.strip() or self.header_acc.strip()):
            rendered = _parse_tool_body(self.header_acc, self.tool_acc, self.state)
            if rendered:
                if "<tool_call>" in rendered:
                    self.calls += 1
                out.append(("content", rendered))
        self.header_acc = ""
        self.tool_acc = ""
        self.state = "idle"


def force_text(force):
    """Translate Qwen-style forced tool call opener into Inkling-2-Mini's native invocation prefix."""
    if not isinstance(force, str):
        return force
    m = re.fullmatch(r"<tool_call>\n(?:<function=([^>\n]+)>\n)?", force)
    if not m:
        return force
    name = m.group(1) or ""
    if name:
        return f'{MSG_MODEL}{name}{CONTENT_TOOL_JSON}{{"name":{json.dumps(name)},"args":'
    return MSG_MODEL


def translate(text: str) -> str:
    return InklingDialect().feed(text, final=True)


# ---- thinking mode: Inkling-2-Mini has NO thinking mode (default is off / "none", and that's it) ----
DEFAULT_EFFORT = "none"
EFFORT = {
    "none": "none", "off": "none", "no_think": "none", "disabled": "none", "false": "none", "0": "none", "0.0": "none",
}


def normalize_effort(value=None) -> str:
    """Inkling-2-Mini has no thinking mode: always normalizes to `"none"` (`Thinking effort level: 0`)."""
    return DEFAULT_EFFORT


def template_kwargs(kwargs: dict) -> dict:
    """Normalize kwargs for Inkling-2-Mini's chat_template.jinja: always enforces thinking off (`reasoning_effort="none"`)."""
    out = dict(kwargs)
    out.pop("thinking_mode", None)
    out["enable_thinking"] = False
    out["reasoning_effort"] = DEFAULT_EFFORT
    return out


def prepare_messages(messages: list[dict]) -> list[dict]:
    """Canonicalize assistant tool_call arguments from JSON strings to dicts so Inkling-2-Mini's chat_template.jinja
    never raises `tool call arguments must be a parsed object, not a JSON string`."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or not m.get("tool_calls"):
            out.append(m)
            continue
        new_calls = []
        changed = False
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                new_calls.append(tc)
                continue
            fn = dict(tc["function"])
            args = fn.get("arguments")
            if isinstance(args, str):
                changed = True
                try:
                    parsed = json.loads(args) if args.strip() else {}
                    fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                except ValueError:
                    fn["arguments"] = {"raw": args}
            elif args is None:
                changed = True
                fn["arguments"] = {}
            new_calls.append({**tc, "function": fn})
        out.append({**m, "tool_calls": new_calls} if changed else m)
    return out
