"""Inkling-2-Mini typed-segment output & reasoning_effort translator for Strata's OutputParser.

Translates Inkling-2-Mini's native typed segments as tokens stream:
  - `<|content_thinking|>...<|end_message|>` -> `<think>...</think>` (collapsible reasoning accordion)
  - `<|content_text|>...<|end_message|>`     -> answer text (closes open `<think>` if active)
  - `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"...","args":{...}}<|end_message|>`
    -> `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`
  - `<|content_xml|>...<|end_message|>`      -> XML tool-call recovery into `<tool_call>` when a tool call
    is emitted in XML form, or plain text fallback on malformed XML!
  - Drops framing markers (`<|message_model|>`, `<|end_message|>`, `<|content_model_end_sampling|>`) cleanly.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

INK_THINK = "<|content_thinking|>"
INK_TEXT = "<|content_text|>"
INK_CALL = "<|content_invoke_tool_json|>"
INK_XML = "<|content_xml|>"
INK_MODEL = "<|message_model|>"
INK_END = "<|end_message|>"
INK_EOS = "<|content_model_end_sampling|>"
INK_OTHER_TEXT = ("<|content_tool_error|>", "<|content_invoke_tool_text|>")

MARKS = (INK_THINK, INK_TEXT, INK_CALL, INK_XML, INK_MODEL, INK_END, INK_EOS) + INK_OTHER_TEXT

# Reasoning effort mapping matching official Inkling-2-Mini chat_template.jinja:
# {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
EFFORT = {
    "none": None, "off": None, "no_think": None, "disabled": None, "false": None,
    "minimal": "minimal", "min": "minimal",
    "low": "low",
    "medium": "medium", "med": "medium",
    "high": "high", "auto": "high", "default": "high",
    "xhigh": "xhigh", "max": "max", "maximum": "max",
}

EFFORT_NUMERIC = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}


def normalize_effort(value) -> str:
    """Normalize client reasoning_effort / thinking_mode into Inkling-2-Mini's valid keys."""
    if value is False:
        return "none"
    if value is True or value is None or value == "":
        return "high"
    if isinstance(value, (int, float)):
        v = max(0.0, min(0.99, float(value)))
        if v <= 0.0:
            return "none"
        if v <= 0.15:
            return "minimal"
        if v <= 0.45:
            return "low"
        if v <= 0.8:
            return "medium"
        if v <= 0.94:
            return "high"
        return "max"
    key = str(value).strip().lower()
    if key in EFFORT:
        mapped = EFFORT[key]
        return "none" if mapped is None else mapped
    return "high"


def template_kwargs(kwargs: dict) -> dict:
    """Map Strata's frontend kwargs to Inkling-2-Mini's chat_template.jinja kwargs (`reasoning_effort`)."""
    out = dict(kwargs)
    if "thinking_mode" in out:
        tm = out.pop("thinking_mode")
        if out.get("reasoning_effort") in (None, ""):
            out["reasoning_effort"] = tm
    if out.get("enable_thinking") is False:
        out["reasoning_effort"] = "none"
    elif out.get("reasoning_effort") not in (None, ""):
        out["reasoning_effort"] = normalize_effort(out["reasoning_effort"])
    else:
        out["reasoning_effort"] = "high"
    if out["reasoning_effort"] == "none":
        out["enable_thinking"] = False
    return out


def _format_tool_call(name: str, args: dict) -> str:
    def _val(v) -> str:
        if isinstance(v, str):
            return v.replace("<", "\\u003c") if "</parameter>" in v else v
        return json.dumps(v, ensure_ascii=False).replace("<", "\\u003c")

    params = "".join(f"<parameter={k}>\n{_val(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _parse_json_tool_body(body: str, fallback_name: str) -> tuple[str, dict] | None:
    text = body.strip()
    if not text:
        return None
    try:
        j = json.loads(text)
    except ValueError:
        # Malformed trailing-brace / markdown-fence recovery
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            j = json.loads(m.group(0))
        except ValueError:
            return None
    if not isinstance(j, dict):
        return None
    name = (j.get("name") or fallback_name or "").strip()
    args = j.get("args")
    if args is None:
        args = j.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"raw": args}
    if not name or not isinstance(args, dict):
        return None
    return name, args


def _parse_xml_tool_body(body: str, fallback_name: str) -> list[tuple[str, dict]]:
    """Recover XML tool invocations (`<tool_call name="...">`, `<invoke name="...">`, `<call name="...">`)."""
    text = body.strip()
    if not text:
        return []
    calls: list[tuple[str, dict]] = []
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        for elem in root:
            tag = elem.tag.lower()
            if tag in ("tool_call", "invoke", "call", "tool"):
                name = (elem.attrib.get("name") or fallback_name or "").strip()
                args: dict = {}
                for child in elem:
                    k = child.attrib.get("name") or child.tag
                    v = (child.text or "").strip()
                    try:
                        args[k] = json.loads(v)
                    except ValueError:
                        args[k] = v
                if name:
                    calls.append((name, args))
    except ET.ParseError:
        # Regex fallback for malformed XML (e.g. unescaped `<` or `&` inside code values!)
        m = re.search(r"<(?:tool_call|invoke|call|tool)\s+name=[\"']([^\"']+)[\"']>(.*?)</(?:tool_call|invoke|call|tool)>", text, re.S)
        if m:
            name = m.group(1).strip()
            inner = m.group(2)
            args = {}
            for pm in re.finditer(r"<(?:param|parameter|arg)\s+name=[\"']([^\"']+)[\"']>(.*?)</(?:param|parameter|arg)>", inner, re.S):
                args[pm.group(1).strip()] = pm.group(2).strip()
            if name:
                calls.append((name, args))
    return calls


class InklingDialect:
    """Stateful streaming translator for Inkling-2-Mini typed segments (one per reply)."""

    def __init__(self, thinking: bool = True):
        self.buf = ""
        self.mode = "lead"  # lead | think | text | name | call | xml | between
        self.open = thinking
        self.name = ""
        self.body = ""

    def _close(self) -> str:
        if self.open:
            self.open = False
            return "</think>\n"
        return ""

    def _emit_call(self) -> str:
        parsed = _parse_json_tool_body(self.body, self.name)
        if parsed is None:
            return self._close() + self.body
        name, args = parsed
        return self._close() + _format_tool_call(name, args)

    def _emit_xml(self) -> str:
        calls = _parse_xml_tool_body(self.body, self.name)
        if not calls:
            return self._close() + self.body
        return self._close() + "".join(_format_tool_call(n, a) for n, a in calls)

    def _text(self, s: str) -> str:
        if not s:
            return ""
        if self.mode == "name":
            self.name += s
            return ""
        if self.mode in ("call", "xml"):
            self.body += s
            return ""
        if self.mode == "lead":
            if s.strip():
                self.mode = "text"
                return self._close() + s
            return ""
        return s if self.mode in ("think", "text") else ""

    def _mark(self, m: str) -> str:
        if m == INK_THINK:
            self.mode = "think"
            if not self.open:
                self.open = True
                return "<think>"
            return ""
        if m == INK_TEXT or m in INK_OTHER_TEXT:
            self.mode = "text"
            return self._close()
        if m == INK_MODEL:
            self.mode = "name"
            self.name = ""
            return ""
        if m == INK_CALL:
            self.mode = "call"
            self.body = ""
            return ""
        if m == INK_XML:
            self.mode = "xml"
            self.body = ""
            return ""
        out = ""
        if self.mode == "call":
            out = self._emit_call()
        elif self.mode == "xml":
            out = self._emit_xml()
        self.mode = "between"
        return out

    def feed(self, text: str = "", final: bool = False) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            hits = [(i, m) for m in MARKS for i in (self.buf.find(m),) if i >= 0]
            if not hits:
                if final:
                    out.append(self._text(self.buf))
                    self.buf = ""
                    break
                keep = max((n for m in MARKS for n in range(1, len(m)) if self.buf.endswith(m[:n])), default=0)
                out.append(self._text(self.buf[:len(self.buf) - keep]))
                self.buf = self.buf[len(self.buf) - keep:]
                break
            i, m = min(hits, key=lambda x: (x[0], -len(x[1])))
            if i:
                out.append(self._text(self.buf[:i]))
            self.buf = self.buf[i + len(m):]
            out.append(self._mark(m))
        if final:
            if self.mode == "call":
                out.append(self._emit_call())
                self.mode = "between"
            elif self.mode == "xml":
                out.append(self._emit_xml())
                self.mode = "between"
        return "".join(out)


def force_text(force):
    """Rewrite forced tool call opening from Qwen syntax into Inkling-2-Mini syntax."""
    if not isinstance(force, str):
        return force
    m = re.fullmatch(r"<tool_call>\n(?:<function=([^>\n]+)>\n)?", force)
    if not m:
        return force
    name = m.group(1) or ""
    return f"{INK_MODEL}{name}{INK_CALL}" if name else INK_MODEL


def translate(text: str, thinking: bool = True) -> str:
    return InklingDialect(thinking=thinking).feed(text, final=True)
