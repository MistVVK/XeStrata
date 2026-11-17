"""Which tool-call spelling a chat template documents, read from the template's text.

No imports from the rest of the server: tools/qwen4_intake.py and serve/frontend.py share this one function, so the report
that says "this model writes XML calls" and the parser the server picks cannot disagree.

    xml   `<tool_call>` around `<function=NAME>` and `<parameter=NAME>` blocks (Qwen3.6, Qwen3.8-Flash-Next: every template Strata ships)
    json  `<tool_call>` around a JSON object `{"name": ..., "arguments": {...}}` (Qwen3, Hermes)
    dsml  DeepSeek-V4's `<｜DSML｜...>` blocks
    none  the template has no tool section
    other the template documents tools in a form this module does not know
"""
from __future__ import annotations

import re

_DSML = "｜DSML｜"          # the full-width bars DeepSeek's template writes
_XML_OPEN = re.compile(r"<function=[^>]*>")


def tool_format(source: str) -> str:
    """The tool-call form `source` (a chat template's Jinja text) tells the model to write."""
    if not isinstance(source, str) or not source:
        return "none"
    if _DSML in source:
        return "dsml"
    has_call = "<tool_call>" in source
    if has_call and _XML_OPEN.search(source) and "<parameter=" in source:
        return "xml"
    if has_call and ("tools" in source) and ('"name"' in source or "'name'" in source or "| tojson" in source or "|tojson" in source):
        return "json"
    if "tools" not in source and not has_call:
        return "none"
    return "other"


def thinking_tags(source: str) -> dict:
    """The reasoning tags the template mentions: whether it writes `<think>` itself (into the generation prompt) and takes an
    `enable_thinking` switch."""
    src = source if isinstance(source, str) else ""
    return {"has_think_tags": "<think>" in src and "</think>" in src,
            "enable_thinking_switch": "enable_thinking" in src,
            "generation_prompt_opens_think": bool(re.search(r"add_generation_prompt.*?<think>", src, re.S))}
