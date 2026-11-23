# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import mimetypes
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from serve import envvars

LOGGER = logging.getLogger(__name__)
# The earlier turns' thinking kept where a template drops it unless asked (Qwen3.6-35B-A3B's preserve_thinking; the
# Qwen3.8-Flash-Next and GLM-5.3-Flash templates keep it already).  A client that sends the thinking back, as an agent
# harness does in its tool loop, then gets the earlier turns rendered as the model wrote them: the prompt keeps its
# start and the conversation cache serves it, instead of reading it again from the first assistant turn after each
# new user message.  A request's own chat_template_kwargs win.
TEMPLATE_DEFAULTS = {"preserve_thinking": True}
# A template that drops the earlier turns' thinking with no switch (Qwen3.5-9B's: `{%- if loop.index0 >
# ns.last_query_index %}` right before the assistant turn's `<think>` and reasoning_content) gets Qwen3.6-35B-A3B's
# switch there, so TEMPLATE_DEFAULTS reaches it too.  Only that branch, and only in a template that has no
# preserve_thinking of its own; MOSS-Image's text encoder writes its own prompt form (images.prompt_ids) and never
# renders a chat template.
_DROPS_THINKING = re.compile(r"(\{%-?\s*if\s+)(loop\.index0\s*>\s*ns\.last_query_index)"
                             r"(\s*-?%\}[^\n]*\n[^\n]*<think>[^\n]*reasoning_content)")


def keep_thinking_switch(source: str) -> str:
    """`source` with the thinking switch in its dropping branch (see _DROPS_THINKING), else as it is."""
    if "preserve_thinking" in source:
        return source
    return _DROPS_THINKING.sub(r"\1(preserve_thinking is defined and preserve_thinking is true) or (\2)\3", source,
                               count=1)


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def strftime_now(fmt):   # transformers' global for templates that write today's date (llm-jp-4.1's)
            return datetime.datetime.now().strftime(fmt)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        # Hugging Face's {% generation %} ... {% endgeneration %} only mark the assistant's tokens for training
        # (Inkling-2-Mini's template has them): Jinja has no such tag, and rendering needs nothing of them
        self.source = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "",
                             keep_thinking_switch(Path(path).read_text(encoding="utf-8")))
        self.template = env.from_string(self.source)
        # the Harmony format's templates (llm-jp-4.1, GPT-OSS): its replies are channels, read by HarmonyParser
        self.harmony = "<|channel|>" in self.source
        # the tools as OpenAI sends them ({"type": "function", "function": {...}}) where the template unwraps them
        # itself; otherwise their inner objects, as Qwen's templates read them
        self.wrapped_tools = re.search(r"tool\.function\b|tool\[['\"]function['\"]\]", self.source) is not None
        # GLM's templates (GLM-5.3-Flash): a call is `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
        # </tool_call>`, read through GlmCallRewriter; the thinking always opened, its effort low, high or max
        self.call_form = "glm" if "<arg_key>" in self.source else "qwen"
        # Inkling-2-Mini's template: its replies are channel tokens (serve/inkling_dialect.py reads them), its calls'
        # arguments parsed objects, and no thinking (upstream inkling2's no-thinking mode)
        self.inkling = "<|content_invoke_tool_json|>" in self.source
        self.caps = self._detect_caps()
        self.thinking = self._thinking_levels()

    def _thinking_levels(self) -> dict:
        """The thinking levels this template takes, by their own names, and the one a request that names none gets
        (/health's "thinking", the web page's buttons): GLM-5.3-Flash's low, high and max; Harmony's (llm-jp-4.1's)
        low, medium and high, medium its default; Qwen3.8-Flash-Next's low, medium and high (xhigh); a template that
        only turns its thinking on and off, on; none at all, no levels."""
        if self.inkling:
            return {"levels": [], "default": None}
        if self.call_form == "glm":
            return {"levels": ["none", "low", "high", "max"], "default": "max"}
        if self.harmony:
            return {"levels": ["none", "low", "medium", "high"], "default": "medium"}
        if "reasoning_effort" in self.source:
            return {"levels": ["none", "low", "medium", "high"], "default": "high"}
        if "enable_thinking" in self.source:
            return {"levels": ["none", "on"], "default": "on"}
        return {"levels": [], "default": None}

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  XESTRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        keep_empty = envvars.get("KEEP_EMPTY_TURNS") == "1"
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or keep_empty or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                                    and not _text_of(m.get("content")).strip()
                                                                    and not _has_image(m.get("content"))
                                                                    and not m.get("tool_calls"))]
        if self.harmony:   # as llama.cpp's LLM-jp-4.1 path: the reasoning in the field the template reads, `thinking`,
            # and no content beside the calls of a turn that made them
            adjusted = []
            for m in messages:
                if isinstance(m, dict) and isinstance(m.get("reasoning_content"), str):
                    m = {**m, "thinking": m["reasoning_content"]}
                    if m.get("tool_calls"):
                        m.pop("content", None)
                adjusted.append(m)
            messages = adjusted
        if self.inkling:
            from serve import inkling_dialect
            messages, kwargs = inkling_dialect.prepare_messages(messages), inkling_dialect.template_kwargs(kwargs)
        if tools and self.wrapped_tools:   # the template reads OpenAI's form, `tool.function` (llm-jp-4.1's)
            tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if self.call_form == "glm":        # its efforts: low, high and max (the default), taken by name
            effort = kwargs.get("reasoning_effort")
            kwargs = {**kwargs, "reasoning_effort": {"low": "low", "medium": "high", "high": "high"}.get(str(effort))}
        elif self.harmony and kwargs.get("reasoning_effort") == "xhigh":   # Harmony's efforts: low, medium and high
            kwargs = {**kwargs, "reasoning_effort": "high"}
        text = self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **{**TEMPLATE_DEFAULTS, **kwargs})
        if (self.call_form == "glm" and add_generation_prompt and kwargs.get("enable_thinking", True) is False
                and text.endswith("<think>")):   # GLM's template always opens the thinking: closed at once
            text += THINK_END
        if (self.harmony and add_generation_prompt and kwargs.get("enable_thinking", True) is False
                and text.endswith(HARMONY_TURN)):
            # Harmony has no switch for thinking: an empty analysis message, so the model goes on with the final
            # answer or a call (a forced final channel would leave it no call)
            text += HARMONY_NO_THINKING + HARMONY_TURN
        return text

    def _detect_caps(self) -> dict[str, bool]:
        """llama.cpp's capability names, checked at load time against this template and Strata's tool-call format.
        These are rendering hints, not a guarantee that the model will follow a request."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception as exc:                # noqa: BLE001 - a hint only: never stop the server for a probe
                # A template may reject a role or feature (a custom one may fail in any way): the feature is off,
                # discovery still answers for the others, and the start-up goes on.
                LOGGER.debug("chat template capability probe failed: %s", exc, exc_info=True)
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
                 for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "",
                               "tool_calls": [{"function": call} for call in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            messages += [{"role": "tool", "content": reply} for reply in replies] + [user]
            prompt = render(messages, tools)
            # Instructions include example XML and literal tag names, which are not model output. Parse only
            # complete calls to our probe functions, using the same body parser as OutputParser.
            if self.call_form == "glm":
                found = re.findall(r"<tool_call>(strata_caps_call_\d+.*?)</tool_call>", prompt, re.S)
                bodies = [glm_call_body(b) for b in found]
            else:
                bodies = re.findall(r"<tool_call>\s*(<function=strata_caps_call_\d+>.*?</function>)\s*</tool_call>",
                                    prompt, re.S)
            try:
                parsed = [parse_tool_call(body) for body in bodies]
            except ValueError:
                return False
            return [{"name": call.name, "arguments": call.arguments} for call in parsed] == calls \
                and all(reply in prompt for reply in replies)

        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        form_tag = "<arg_key>" if self.call_form == "glm" else "<function="
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", "<tool_call>", form_tag)),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}


# ------------------------------------------------------------------------------------------------ requests
# The content-part types the readers take as text (a part with no type that has a "text" field reads as text too)
TEXT_PARTS = ("text", "input_text", None)


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   TEXT_PARTS)


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  Qwen3.8-Flash-Next's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto none, low, medium, high and
# xhigh, high kept apart from xhigh for the templates whose high is below their top (GLM-5.3-Flash's max).
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


# A file a request sends (Chat's "file", Responses' "input_file", an Anthropic PDF document as converted by
# _anthropic_documents): a "file" item of the template's content until Service.expand_files reads it into text and
# page images
FILE_PARTS = ("file", "input_file")


def _has_image(content) -> bool:
    """Content that must stay a list of parts: a picture in it (or a file, which becomes text and pictures)."""
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS + FILE_PARTS
                                             for p in content)


def file_item(part: dict, path: str) -> dict:
    """A file part -> {"type": "file", "data": a data: URL or None, "url": an http(s) URL or None, "name", "mime"}.
    A file_id (OpenAI's and Anthropic's Files APIs) is a ValueError: this server keeps no files."""
    inner = part.get("file")
    src: dict = inner if part.get("type") == "file" and isinstance(inner, dict) else part
    if src.get("file_id"):
        raise ValueError(f"{path}: a file_id needs a Files API, which this server does not have: send the file "
                         "itself (file_data)")
    data, url, name = src.get("file_data"), src.get("file_url"), str(src.get("filename") or "")
    if not (isinstance(data, str) and data) and not (isinstance(url, str) and url):
        raise ValueError(f"{path}: a file needs file_data (a data: URL or base64) or file_url")
    if isinstance(data, str) and data and not data.startswith("data:"):
        guess = mimetypes.guess_type(name)[0] or "application/octet-stream"   # bare base64: the name's type
        data = f"data:{guess};base64,{data}"
    mime = data[5:data.find(";")] if isinstance(data, str) and data and ";" in data else None
    return {"type": "file", "data": data or None, "url": None if data else url, "name": name, "mime": mime}


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in FILE_PARTS:
            items.append(file_item(part, "content"))
        elif part.get("type") in IMAGE_PARTS:
            item = {"type": "image", "source": _image_source(part)}
            url = part.get("image_url")
            if (url.get("detail") if isinstance(url, dict) else part.get("detail")) == "low":
                item["detail"] = "low"               # OpenAI's low detail: read at a smaller size (Service.prepare)
            items.append(item)
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + FILE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking")


def _check_parts(content, path: str, kinds):
    """A request's content parts (on chat) or content blocks (on messages), looked at while the request is converted.
    A part whose type this server does not read (a file, audio, a document, Anthropic's tool_reference,
    search_result, server_tool_use / web_search_tool_result blocks, ...) is left out of the prompt exactly as it always
    was, with a debug line: real clients (Claude Code, Codex, opencode, Cline, Continue, Open WebUI) send such parts
    next to text the model can answer from, so refusing them would break requests that work.  A "refusal" part is read
    as its text, as /v1/responses does.  Content that is not a list, and a part that is not an object, passes through
    unchanged."""
    if not isinstance(content, list):
        return content
    out = content
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "refusal":
            if out is content:
                out = list(content)
            out[j] = {"type": "text", "text": part.get("refusal") or ""}
        elif kind not in kinds:
            LOGGER.debug("%s[%d].type: content part of type %r is not read by this server; left out of the prompt",
                         path, j, kind)
    return out


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# #554: the vision markers the same way.  They are special tokens only where the template writes them for an image
# item; the same strings in a message's text (an agent reading chat_template.jinja, a tool result quoting it) became
# the same special ids, so text with <|vision_start|><|image_pad|> before a picture took that picture's embeddings
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
# #2026: `</tool_call>` the same way.  A user asking for the string verbatim handed the model the real end-of-call token
# (type 4, matched everywhere); it copied that token into its answer and the reply came back empty.  An assistant
# message keeps its own: a client replaying the model's raw call text sends the tokens the model wrote.
# 0.1.44: `<tool_call>`, `<tool_response>` and `</tool_response>` are type 4 too (ids 248058, 248066, 248070), so a
# quoted one handed the model the real call / tool-result marker in the same way (found by 47Hunter47 in #2026).  The
# template writes the tool-result markers around a tool message itself; only the text inside a message is marked.
TOOL_LITERAL_TAGS = {"</tool_call>": "\U000F0E07", "<tool_call>": "\U000F0E08", "<tool_response>": "\U000F0E09",
                     "</tool_response>": "\U000F0E0A"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_LITERAL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # the control tokens' marks start here, clear of the two sets above


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text (`controls`: the tokenizer's CONTROL literals,
    <|im_start|>, <|im_end|>, <|endoftext|>, ...).  The rendered prompt is encoded with those literals parsed, so one
    written inside a message - a file an agent reads, a pasted chat template - opened or ended a turn there.  Longest
    first, as the tokenizer matches them: a literal inside a longer one is not marked before it."""
    own = [c for c in controls if c not in LITERAL_TAGS]
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(own)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


_TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE = "<tool_response>", "</tool_response>"


def _keep_tool_result_wrap(content, tags: dict[str, str]):
    """A user message whose whole text is <tool_response>...</tool_response> (a client that formats a tool result the
    way the template does for a tool message) keeps that outer pair as the markers - the template tells such a turn from
    a real question by exactly that pair - and only the text inside is marked.  None: not such a message."""
    o, c = _TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE

    def wrap(text: str, first: bool, last: bool) -> str:
        i = text.index(o) + len(o) if first else 0
        j = text.rindex(c) if last else len(text)
        head = _mark(text[:i - len(o)], tags) + o if first else ""
        tail = c + _mark(text[j + len(c):], tags) if last else ""
        return head + _mark(text[i:j], tags) + tail

    if isinstance(content, str):
        body = content.strip()
        if len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c):
            return wrap(content, True, True)
        return None
    if not (isinstance(content, list) and content and all(isinstance(x, dict) and "text" in x for x in content)):
        return None
    texts = [x["text"] for x in content]
    if not all(isinstance(t, str) for t in texts):
        return None
    body = "".join(texts).strip()
    if not (len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c)):
        return None
    if not (texts[0].lstrip().startswith(o) and texts[-1].rstrip().endswith(c)):
        return None          # the pair is split across parts: mark it all, as before
    if len(content) == 1:
        return [{**content[0], "text": wrap(texts[0], True, True)}]
    return ([{**content[0], "text": wrap(texts[0], True, False)}] +
            [_mark_deep(x, tags) for x in content[1:-1]] +
            [{**content[-1], "text": wrap(texts[-1], False, True)}])


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """#537: (messages, tools) with every literal <think> / </think> (#554: and vision marker) in their text swapped for
    LITERAL_TAGS' marks (or `tags`': literal_tags() adds the control tokens' texts), and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    out = []
    own = {t: v for t, v in tags.items() if t not in TOOL_LITERAL_TAGS}     # what an assistant message keeps as written
    for m in messages:
        m = dict(m)
        content = m.get("content")
        mine = own if m.get("role") == "assistant" else tags
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, mine)
        if m.get("role") == "user":
            kept = _keep_tool_result_wrap(content, tags)
            if kept is not None:
                m["content"] = kept
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + _mark(content[i:j], own) + "</think>" +
                            _mark(content[j + len("</think>"):], own))
        out.append(m)
    return out, _mark_deep(tools, tags), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _THINK_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


# Config "env_to_user" (opt-in): opencode writes its <env> block (working directory, today's date) near the START of
# the system message, before ~40K characters of instructions and tool definitions that are the same in every session.
# The prefix cache only reuses a prompt's start, so a new folder or a new day read all of it again (~14K tokens,
# 8-11 s).  Moved to the front of the first user message, the system turn (and the tools the template writes into it)
# is the same for every session and the disk prefix cache restores it.  The model still sees the same lines, one turn
# later.
ENV_TO_USER = False
_ENV_BLOCK = re.compile(r"(?:[^\n]*environment you are running in:\n)?<env>\n.*?\n</env>\n?", re.S)


def _env_to_user(messages: list[dict]) -> list[dict]:
    if not ENV_TO_USER or not messages or messages[0].get("role") != "system":
        return messages
    sys_text = messages[0].get("content")
    if not isinstance(sys_text, str):
        return messages
    m = _ENV_BLOCK.search(sys_text)
    if m is None:
        return messages
    for i in range(1, len(messages)):
        if messages[i].get("role") != "user":
            continue
        block = m.group(0).rstrip("\n")
        c = messages[i].get("content")
        nc: str | list
        if isinstance(c, str):
            nc = block + "\n\n" + c
        elif isinstance(c, list):
            nc = [{"type": "text", "text": block + "\n\n"}] + c
        else:
            return messages
        out = list(messages)
        out[0] = dict(messages[0], content=sys_text[:m.start()] + sys_text[m.end():])
        out[i] = dict(messages[i], content=nc)
        return out
    return messages


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def _stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list before the system prompt: one swap moves the shared prefix back thousands of
    tokens and the whole conversation is read again (opencode issue 23571).  A request's first sight of a tool set
    is untouched (so its prompt is byte for byte what the client sent); later requests with the same names in
    another order are put back in that first order.  Duplicate names, or a single tool: left as sent."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    if len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always.  A tool's schema -
    "parameters" in the OpenAI shape, "input_schema" in the Anthropic one - is an object when it is there, for the
    same reason: a string or a list passed and raised later instead, on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            # it reached parse_tool_call and the stream parser as the tool's schema, whose .get("properties") raised
            # AttributeError in the request thread - after the 200 and whatever the model had said before the call
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return _stable_tool_order(tools,
                              lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def tool_arguments(raw) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (both APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}; anything else (cut-off or malformed JSON, a list, a
    number) is kept as text under "arguments" instead of refusing the whole conversation (#510)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def require_messages(messages: list) -> None:
    """A chat request with no message at all is a 400 (OpenAI: "empty array. Expected an array with minimum length
    1"); the template would otherwise render just the assistant turn and the model would answer nothing."""
    if not messages:
        raise ValueError("messages must be a non-empty list: send at least one message")


def _order_chat_tool_results(messages: list[dict]) -> None:
    """The template pairs tool results with the calls by position, so a run of results the client returned in another
    order is put back in the order of the calls while their ids are still there (upstream 11237f831, #3395).  Only a
    complete, unambiguous run is reordered: the calls' and the results' ids non-empty unique strings, the same set.
    Anything else keeps the order received.  The ids are private to this conversion and never reach the template."""
    i = 0
    while i < len(messages):
        if messages[i]["role"] != "tool":
            i += 1
            continue
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            j += 1
        prev = messages[i - 1] if i else {}
        calls = [c.get("_call_id") for c in prev.get("tool_calls") or []] if prev.get("role") == "assistant" else []
        results = [m.get("_call_id") for m in messages[i:j]]
        if len(calls) == len(results) and all(isinstance(cid, str) and cid for cid in calls + results):
            if len(set(calls)) == len(calls) and len(set(results)) == len(results) and set(calls) == set(results):
                rank = {cid: n for n, cid in enumerate(calls)}
                messages[i:j] = sorted(messages[i:j], key=lambda m: rank[m["_call_id"]])
        i = j
    for message in messages:
        message.pop("_call_id", None)
        for call in message.get("tool_calls") or []:
            call.pop("_call_id", None)


def custom_tool_names(req: dict) -> set[str]:
    """The names of a Chat Completions request's custom tools ({"type": "custom", "custom": {"name": ...}}): their
    calls are answered as {"type": "custom", "custom": {"name", "input"}}."""
    tools = req.get("tools")
    return {t["custom"]["name"] for t in tools if isinstance(t, dict) and t.get("type") == "custom"
            and isinstance(t.get("custom"), dict) and isinstance(t["custom"].get("name"), str)} \
        if isinstance(tools, list) else set()


def _chat_tool(t):
    """A Chat Completions tool -> the template's.  A custom tool takes free text, maybe in a grammar the model reads
    (it is not enforced: the text is checked by nothing), as /v1/responses' custom tools."""
    if not isinstance(t, dict) or t.get("type") != "custom" or not isinstance(t.get("custom"), dict):
        return t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
    c = t["custom"]
    description = c.get("description") or ""
    fmt = c.get("format") if isinstance(c.get("format"), dict) else {}
    grammar = fmt.get("grammar") if isinstance(fmt.get("grammar"), dict) else fmt
    if fmt.get("type") == "grammar" and grammar.get("definition"):
        description += f"\n\nThe input must follow this {grammar.get('syntax', '')} grammar:\n{grammar['definition']}"
    return {"name": c.get("name"), "description": description,
            "parameters": {"type": "object", "required": ["input"],
                           "properties": {"input": {"type": "string", "description": "the raw input"}}}}


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role == "function":                       # the old shape of a tool's result: the call it answers by name
            role = "tool"
        raw_content = m.get("content")
        for j, part in enumerate(raw_content if isinstance(raw_content, list) else []):
            if isinstance(part, dict) and part.get("type") == "input_audio":
                raise ValueError(f"messages[{i}].content[{j}]: input_audio is not supported: this model does not "
                                 "read audio")
            if isinstance(part, dict) and part.get("type") == "file":
                file_item(part, f"messages[{i}].content[{j}]")   # a file_id or no data is a 400 here
        # an image (or a file) is read in a user, tool or assistant message; the system prompt takes text only, as
        # _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS + FILE_PARTS if images else TEXT_PARTS)
        if role == "assistant" and content in (None, "") and isinstance(m.get("refusal"), str):
            content = m["refusal"]                   # a refusal the client sends back is what the model said
        out = {"role": role, "content": _parts_of(content) if images else _text_of(content)}
        name = m.get("name")
        if role in ("user", "system") and isinstance(name, str) and name:
            # the speaker's name, which the template has no place for: written before what they said
            said = out["content"]
            out["content"] = f"{name}: {said}" if isinstance(said, str) else \
                [{"type": "text", "text": f"{name}: "}] + list(said or [])
        if role == "tool":
            out["_call_id"] = m.get("tool_call_id")
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        raw = m.get("tool_calls")
        if not raw and isinstance(m.get("function_call"), dict):   # the old shape of one call
            raw = [{"function": m["function_call"]}]
        if raw:
            calls = []
            for c in _object_list(raw, "tool_calls"):
                if c.get("type") == "custom" and isinstance(c.get("custom"), dict):
                    calls.append({"function": {"name": c["custom"].get("name"),
                                               "arguments": {"input": str(c["custom"].get("input") or "")}},
                                  "_call_id": c.get("id")})
                    continue
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                calls.append({"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))},
                              "_call_id": c.get("id")})
            out["tool_calls"] = calls
        messages.append(out)
    _order_chat_tool_results(messages)
    given = req.get("tools")
    plain = [t for t in given if not (isinstance(t, dict) and t.get("type") == "custom")] \
        if isinstance(given, list) else given
    listed = [_chat_tool(t) for t in _tool_list(plain, "function")]
    if isinstance(given, list):                      # the custom tools, after the function tools
        listed += [_chat_tool(t) for t in given if isinstance(t, dict) and t.get("type") == "custom"
                   and isinstance(t.get("custom"), dict) and isinstance(t["custom"].get("name"), str)]
    tools = listed or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req["reasoning"] if isinstance(req.get("reasoning"), dict) else {}
    given = req.get("reasoning_effort")
    # `false` is a level (thinking off, effort_kwargs): an `or` between the two fields dropped it
    kwargs.update(effort_kwargs(given if given is not None and given != "" else reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    # the earlier turns' thinking, where the template takes the switch (TEMPLATE_DEFAULTS keeps it otherwise)
    keep = (req.get("chat_template_kwargs") or {}).get("preserve_thinking")
    if isinstance(keep, bool):
        kwargs["preserve_thinking"] = keep
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


# Anthropic's thinking signature, and the data of a redacted_thinking block: the thinking's text in the form of
# responses.py's encrypted_content (base64, not encrypted: the client already holds the text), so a block this
# server wrote can be read back and checked against its text.  Another server's signature is opaque and ignored.
THINKING_PREFIX = "strata.r1:"


def thinking_signature(text: str) -> str:
    return THINKING_PREFIX + base64.b64encode(json.dumps({"text": text}, ensure_ascii=False).encode()).decode("ascii")


def _signed_thinking(signature, where: str) -> str | None:
    """The text a signature of this server's carries; None for another server's (or none)."""
    if not isinstance(signature, str) or not signature.startswith(THINKING_PREFIX):
        return None
    try:
        return str(json.loads(base64.b64decode(signature[len(THINKING_PREFIX):], validate=True).decode())["text"])
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise ValueError(f"{where}: the thinking's signature is damaged") from None


def _anthropic_documents(content, path: str):
    """Anthropic's text documents and search results -> text blocks the template reads: a document's text
    (source "text", or "content" blocks) under its title and context, a "content" document's images after it as image
    blocks of their own, a search result's text under its title and source.  A PDF document is left as it is (not
    read yet).  citations.enabled: true is a 400: no citations are made."""
    if not isinstance(content, list):
        return content
    out = []
    for j, block in enumerate(content):
        kind = block.get("type") if isinstance(block, dict) else None
        if kind in ("document", "search_result"):
            cite = block.get("citations")
            if isinstance(cite, dict) and cite.get("enabled") is True:
                raise ValueError(f"{path}[{j}].citations: citations are not supported by this server")
        text, images = None, []
        if kind == "document":
            src = block.get("source") if isinstance(block.get("source"), dict) else {}
            if src.get("type") == "text":
                text = str(src.get("data") or "")
            elif src.get("type") == "content":
                inner = src.get("content")
                text = inner if isinstance(inner, str) else "\n\n".join(
                    b.get("text") or "" for b in inner or [] if isinstance(b, dict) and b.get("type") == "text")
                images = [] if isinstance(inner, str) else [
                    b for b in inner or [] if isinstance(b, dict) and b.get("type") == "image"]
            if text is not None:
                head = [x for x in (block.get("title"), block.get("context")) if isinstance(x, str) and x]
                text = "\n".join(["Document: " + head[0]] + head[1:] + ["", text]) if head else text
            elif src.get("type") in ("base64", "url", "file"):   # a PDF: read by Service.expand_files
                name = str(block.get("title") or "")
                part: dict = {"type": "input_file", "filename": name}
                if src.get("type") == "base64":
                    part["file_data"] = f"data:{src.get('media_type') or 'application/pdf'};base64,{src.get('data')}"
                elif src.get("type") == "url":
                    part["file_url"] = str(src.get("url") or "")
                else:
                    part["file_id"] = src.get("file_id") or "?"
                file_item(part, f"{path}[{j}]")     # a file_id is a 400 here
                out.append(part)
                continue
        elif kind == "search_result":
            body = "\n\n".join(b.get("text") or "" for b in block.get("content") or []
                               if isinstance(b, dict) and b.get("type") == "text")
            text = f"Search result: {block.get('title') or ''} ({block.get('source') or ''})\n\n{body}"
        out.append({"type": "text", "text": text} if text is not None else block)
        out += images
    return out


def _anthropic_tool(t: dict) -> dict:
    """An Anthropic tool -> the template's form of it (also what the output parser reads its schema from)."""
    return {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}


def anthropic_tools(req: dict) -> tuple[list[dict] | None, bool]:
    """#4663, Anthropic's tool search: (every tool the request declared, the `defer_loading` ones included, whether any
    tool is deferred).  A deferred tool is callable although the prompt does not list it.  A client that defers tools
    (Claude Code) holds back more than it declares - it names the rest in the conversation and runs a call for any of
    them - so with one deferred tool a call for an undeclared name is a call too (OutputParser's `deferred`)."""
    tools = _tool_list(req.get("tools"), None)
    return [_anthropic_tool(t) for t in tools] or None, any(t.get("defer_loading") for t in tools)


def _tool_references(content, deferred: dict):
    """A tool_result's tool_reference blocks (a tool search's results) -> the referenced deferred tools' definitions,
    in the form the prompt lists tools: the model sees a tool where it was loaded, and the prompt's start (its tool
    list) stays the same, so the prompt cache does too.  Any other reference is left out as before (_check_parts): a
    listed tool is in the prompt already, and an old one (its MCP server gone) must not fail the rest of the session."""
    if not isinstance(content, list):
        return content
    return [{"type": "text",
             "text": "<tools>\n" + json.dumps(deferred[p.get("tool_name")], ensure_ascii=False) + "\n</tools>"}
            if isinstance(p, dict) and p.get("type") == "tool_reference" and isinstance(p.get("tool_name"), str)
            and p["tool_name"] in deferred else p
            for p in content]


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31 (the
    config's "anthropic_thinking": "model"); False renders it without thinking, Anthropic's rule (#278, the server's
    default "on_request")."""
    declared = _tool_list(req.get("tools"), None)
    # #4663: a `defer_loading` tool stays out of the prompt's tool list until a tool_reference loads it
    deferred = {t["name"]: _anthropic_tool(t) for t in declared if t.get("defer_loading")}
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS)))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(_anthropic_documents(m.get("content"), f"messages[{i}].content"),
                               f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content or []):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS + FILE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                said = block.get("thinking", "")
                signed = _signed_thinking(block.get("signature"), f"messages[{i}].content[{j}]")
                if signed is not None and signed != said:
                    raise ValueError(f"messages[{i}].content[{j}]: the thinking does not match its signature")
                reasoning.append(said)
            elif kind == "redacted_thinking":     # one of this server's is read; another server's carries nothing
                signed = _signed_thinking(block.get("data"), f"messages[{i}].content[{j}]")
                if signed is not None:
                    reasoning.append(signed)
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))},
                              "_call_id": block.get("id")})
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.  Its tool_use_id
                # puts it beside its call (_order_chat_tool_results): parallel results may come in any order.
                where = f"messages[{i}].content[{j}].content"
                content = _tool_references(_anthropic_documents(block.get("content"), where), deferred)
                result = _parts_of(_check_parts(content, where, TEXT_PARTS + IMAGE_PARTS + FILE_PARTS))
                if block.get("is_error") is True:   # the tool failed: the model is told so, not only its text
                    result = "Error: " + result if isinstance(result, str) else \
                        [{"type": "text", "text": "Error: "}] + list(result or [])
                messages.append({"role": "tool", "content": result, "_call_id": block.get("tool_use_id")})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    _order_chat_tool_results(messages)
    tools = [_anthropic_tool(t) for t in declared if t["name"] not in deferred] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    if thinking is not None and (not isinstance(thinking, dict)
                                 or thinking.get("type") not in ("enabled", "disabled", "adaptive")):
        raise ValueError('thinking must be {"type": "enabled", "budget_tokens": N}, {"type": "adaptive"} or '
                         '{"type": "disabled"}')
    # "adaptive": the model decides how much to think (the template's default, an effort if output_config has one)
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    # 0.1.45: Anthropic's budget_tokens is a real budget, not only a level: the thinking is closed (wrap-up, </think>,
    # the answer) after that many tokens.  A reasoning_budget_tokens of the request's own wins.
    bt = thinking.get("budget_tokens") if isinstance(thinking, dict) and thinking.get("type") != "disabled" else None
    if isinstance(bt, int) and not isinstance(bt, bool) and bt > 0 and req.get("reasoning_budget_tokens") is None:
        req["reasoning_budget_tokens"] = bt
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # The server's default (the config's "anthropic_thinking": "on_request"; "model" thinks as 0.1.31 did, for
        # a client that never asks).  Anthropic's thinking is opt-in. Claude Code's helper calls (a session title,
        # a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


def anthropic_format(req: dict):
    """Anthropic's structured output, `output_config.format` (or the beta's name for it, `output_format`):
    {"type": "json_schema", "schema": {...}} -> the Chat Completions response_format prepare_format reads, or None."""
    config = req.get("output_config")
    fmt = config.get("format") if isinstance(config, dict) else None
    if fmt is not None and req.get("output_format") is not None:
        raise ValueError("give the format once: output_config.format or output_format, not both")
    fmt = req.get("output_format") if fmt is None else fmt
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise ValueError('output_config.format must be {"type": "json_schema", "schema": {...}}')
    if not isinstance(fmt.get("schema"), dict):
        raise ValueError("output_config.format needs a schema object")
    # Anthropic's structured output always follows its schema: strict, as OpenAI's strict: true
    return {"type": "json_schema", "json_schema": {"name": "output", "strict": True, "schema": fmt.get("schema")}}


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_START = "<function="
FUNC_END = "</function>"
# #3441: the opener a model writes when it drifts by a token inside the envelope the template documents:
# `<=NAME>` (the word `function` dropped), `<function:NAME>`, `<function =NAME>`, or the bare `<NAME>`
_LOOSE_OPEN = re.compile(r"<\s*(?:function\s*[=:]\s*|=\s*)?([A-Za-z0-9_.\-]+)\s*>")
# #5409: `<function=tools actio>` / `<function=tools>`; the prefixes are the ones of #4751
_PREFIX_SPACED = re.compile(r"(<function=\s*(?:tools|functions|default_api)[ \t]+)([^<>\s]+)\s*>")
_PREFIX_PART = re.compile(r"<function=\s*(?:tools|functions|default_api)[ \t]+[^<>\s]{0,80}\s*$")
_PREFIX_BARE = re.compile(r"(<tool_call>\s*)<function=\s*(tools|functions|default_api)\s*>(.*?)</function>", re.S)
_PREFIX_BARE_OPEN = re.compile(r"<tool_call>\s*<function=\s*(tools|functions|default_api)\s*>(?!.*</function>)", re.S)
_LOOSE_PART = re.compile(r"<\s*(?:function[ \t]*[=:]?\s*|=\s*)?[A-Za-z0-9_.\-]*\s*$")
# #5276: `<toolcall>` / `<tool-call>` as the opener (then whitespace, `<function=NAME>`), and such an opener still
# arriving
_MISSPELT_OPEN = re.compile(r"<tool-?call>(\s*)<function=(\s*[^<>\s]+\s*)>")
_MISSPELT_PART = re.compile(r"<tool-?call>\s*<function=[^<>]{0,80}$|"
                            r"<tool-?call>\s*(?:<(?:f(?:u(?:n(?:c(?:t(?:i(?:o(?:n)?)?)?)?)?)?)?)?)?$")


# The output's end right after a call's own `</function>` (and `</tool_call>`), the parameter before it left open:
# OutputParser._closed_at_end
_OPEN_PARAM_AT_END = re.compile(r"\n?" + re.escape(FUNC_END) + r"\s*(?:" + re.escape(CALL_END) + r")?\s*\Z")
# a quoted call's opener in a value: `<function=NAME>` and its first parameter or close; a bare FUNC_START is text
_QUOTED_OPEN = re.compile(r"<function=[^<>\s]+>\s*(?:<parameter=|</function>)")


#: #4701: how much text after a closing sequence is looked at before the value is taken as ended there (a call
#: followed by a long sentence)
CLOSE_LOOKAHEAD_CHARS = 1500


def _close_verdict(after: str, final: bool) -> str:
    """`after` starts with `</function>` and/or `</tool_call>` right after a `</parameter>`.  A file that documents the
    call format has those three tags on lines of its own inside a value (#4701), so the sequence only ends the value
    when what follows is the end of the turn, another call, or prose with no closing tag after it.  "end": the value
    ends here; "continue": it goes on to the next `</parameter>`; "hold": not decidable yet (streaming): wait for more
    text.  Bounded: the end of the turn (`final`), a new `<tool_call>`, or CLOSE_LOOKAHEAD_CHARS of text decide, so a
    call is never held longer than that."""
    m = re.match(r"(?:</function>)?\s*</tool_call>", after)
    if not m:
        tail = after[len(FUNC_END):].lstrip() if after.startswith(FUNC_END) else after
        if not final and CALL_END.startswith(tail):
            return "hold"                                  # #5040: `</tool_call>` is still arriving
        return "end"                                       # the call is cut, or it has no wrapper: as before
    rest = after[m.end():].lstrip()
    if not rest:
        return "end" if final else "hold"                  # the usual call: the turn ends right after it
    if rest.startswith(CALL_START):
        return "end"
    if not final and CALL_START.startswith(rest):
        return "hold"                                      # `<tool_c` of a second call is arriving
    if CALL_START in rest:
        return "end"                                       # prose, then another call: this one is complete
    if PARAM_END in rest or CALL_END in rest or FUNC_END in rest:
        return "continue"                                  # prose, then closing tags again: the text was part of it
    if final or len(rest) > CLOSE_LOOKAHEAD_CHARS:
        return "end"
    return "hold"


def param_end(text: str, final: bool = False, name: str | None = None, hold: bool = True) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
    `<parameter=` or `</function>`, or by the wrapper's `</tool_call>` when the call's `</function>` was left out
    (#4427) - the same text inside a value (a file that documents the call format, #210) is part of the value.
    #4078: a close inside a call the value itself quotes (more _QUOTED_OPEN than `</function>` before it), or one
    followed by the same parameter `name` again, is part of the value too.  `text` starts at the value.  -1: none yet;
    -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    at = text.find(PARAM_END)
    while at >= 0:
        if len(_QUOTED_OPEN.findall(text, 0, at)) <= text.count(FUNC_END, 0, at):
            after = text[at + len(PARAM_END):].lstrip()
            if after.startswith(PARAM_START):
                gt = after.find(">")
                if gt < 0:
                    return at if final else -2
                if name is None or after[len(PARAM_START):gt].strip() != name:
                    return at
            elif after.startswith((FUNC_END, CALL_END)):          # #4427: the model left out its `</function>`
                verdict = _close_verdict(after, final) if hold else "end"   # hold=False: a bare call keeps its old rule
                if verdict == "end":
                    return at
                if verdict == "hold":
                    return -2
            elif not after or any(tag.startswith(after) for tag in (PARAM_START, FUNC_END, CALL_END)):
                return at if final else -2
        at = text.find(PARAM_END, at + 1)
    return -1


def call_end(text: str, final: bool = False, hold: bool = True) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith("<parameter="):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:], final, text[pos:gt].split("=", 1)[1].strip(), hold)
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


GLM_KEY, GLM_KEY_END, GLM_VALUE, GLM_VALUE_END = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"


def glm_call_body(body: str) -> str:
    """A GLM call's body (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`, between <tool_call> and
    </tool_call>) in the XML form parse_tool_call reads (`<function=NAME>` with `<parameter=K>` values).  The values
    stay as written: a string as it is, other types as JSON, which parse_tool_call decodes by the tool's schema."""
    k = body.find(GLM_KEY)
    out = [f"<function={(body if k < 0 else body[:k]).strip()}>\n"]
    for key, value in re.findall(re.escape(GLM_KEY) + r"(.*?)" + re.escape(GLM_KEY_END) + r"\s*" + re.escape(GLM_VALUE)
                                 + r"(.*?)" + re.escape(GLM_VALUE_END), body, re.S):
        out.append(f"<parameter={key.strip()}>\n{value}\n{PARAM_END}\n")
    return "".join(out) + FUNC_END


class GlmCallRewriter:
    """GLM's calls, as they stream, in the XML form OutputParser reads: the text outside `<tool_call>...</tool_call>`
    passes as it comes (a partial `<tool_call>` held back), a call is held until its `</tool_call>` and then written
    out rewritten (glm_call_body).  A `<tool_call>` that a `</think>` follows before its end, or that runs past
    RCALL_MAX, was not a call: it goes on as it was written."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, delta: str) -> str:
        self.buf += delta
        out = []
        while True:
            if not self.inside:
                i = self.buf.find(CALL_START)
                if i < 0:
                    keep = next((n for n in range(min(len(CALL_START) - 1, len(self.buf)), 0, -1)
                                 if CALL_START.startswith(self.buf[-n:])), 0)
                    out.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    return "".join(out)
                out.append(self.buf[:i])
                self.buf, self.inside = self.buf[i + len(CALL_START):], True
            end, think = self.buf.find(CALL_END), self.buf.find(THINK_END)
            if end >= 0 and (think < 0 or think > end):
                out.append(CALL_START + "\n" + glm_call_body(self.buf[:end]) + "\n" + CALL_END)
                self.buf, self.inside = self.buf[end + len(CALL_END):], False
            elif think >= 0 or len(self.buf) > RCALL_MAX:
                out.append(CALL_START)
                self.inside = False
            else:
                return "".join(out)

    def flush(self) -> str:
        """The end of the output: what is held, as it was written."""
        text = (CALL_START if self.inside else "") + self.buf
        self.buf, self.inside = "", False
        return text


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named", name).  OpenAI:
    "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat {"type": "function",
    "name": N}; Anthropic: {"type": "auto" | "any" | "none"} and {"type": "tool", "name": N}.  A value it does not
    know is ("unknown", None)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        return (tool_choice, None) if tool_choice in ("auto", "none", "required") else ("unknown", None)
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool", "custom"):
            fn = tool_choice.get("function") or tool_choice.get("custom")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def allowed_tools(tool_choice, tools: list[dict] | None, more: dict | None = None):
    """OpenAI's {"type": "allowed_tools"} tool_choice (Chat: {"allowed_tools": {"mode", "tools"}}; Responses: "mode"
    and "tools" beside the type) -> (the tools the model is offered: those it lists, the tool_choice to apply:
    "auto" or "required").  Any other tool_choice comes back as it is, with all the tools.  `more`: a listed tool
    type's template names (Responses' {"type": "image_generation"}: the image tools).  A list naming a tool the
    request does not have, or an unknown mode, is a ValueError (a 400)."""
    if not (isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools"):
        return tools, tool_choice
    spec = tool_choice["allowed_tools"] if isinstance(tool_choice.get("allowed_tools"), dict) else tool_choice
    mode, listed = spec.get("mode", "auto"), spec.get("tools")
    if mode not in ("auto", "required"):
        raise ValueError('tool_choice.allowed_tools: mode must be "auto" or "required"')
    if not isinstance(listed, list) or not listed:
        raise ValueError("tool_choice.allowed_tools: tools must be a non-empty list of the request's tools")
    names: set = set()
    for t in listed:
        if not isinstance(t, dict):
            raise ValueError("tool_choice.allowed_tools: each tool is an object")
        if t.get("type") in (more or {}):
            names |= set((more or {})[t["type"]])
            continue
        fn = t.get("function") or t.get("custom") or t
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.allowed_tools: each tool names a tool of the request")
        names.add(f"{t['namespace']}.{name}" if t.get("namespace") else name)
    have = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    unknown = sorted(names - have)
    if unknown:
        raise ValueError(f"tool_choice.allowed_tools names tools the request does not have: {', '.join(unknown)}")
    return [t for t in tools or [] if t.get("name") in names], mode


def legacy_functions(req: dict) -> dict | None:
    """OpenAI's deprecated `functions` and `function_call` (before tools) -> the request with `tools` and `tool_choice`
    in their place and one call at a time (the old shape answers with one), or None when it asks for neither (a null
    is no field, and function_call "none" without functions is the default).  Mixed with tools or tool_choice, or a
    function_call "auto" or name without functions, it is a 400."""
    fns, choice = req.get("functions"), req.get("function_call")
    if fns is None and choice is None:
        return None
    if req.get("tools") is not None or req.get("tool_choice") is not None:
        raise ValueError("functions and function_call cannot be used with tools or tool_choice")
    if fns is None and choice == "none":
        return None
    if not isinstance(fns, list) or not fns or not all(isinstance(f, dict) for f in fns):
        raise ValueError("functions must be a non-empty array of function objects")
    if isinstance(choice, dict) and isinstance(choice.get("name"), str):
        choice = {"type": "function", "function": {"name": choice["name"]}}
    elif choice not in (None, "none", "auto"):
        raise ValueError('function_call must be "none", "auto" or {"name": ...}')
    out = {k: v for k, v in req.items() if k not in ("functions", "function_call")}
    out.update(tools=[{"type": "function", "function": f} for f in fns], parallel_tool_calls=False)
    if choice is not None:
        out["tool_choice"] = choice
    return out


def legacy_function_call(choice: dict) -> dict:
    """A reply's choice (a message or a stream's delta) for a `functions` request: its call as the old function_call
    and finish_reason "function_call", as OpenAI answers one."""
    for key in ("message", "delta"):
        m = choice.get(key)
        if not isinstance(m, dict):
            continue
        calls = m.pop("tool_calls", None)
        if calls:
            fn = calls[0].get("function") or {}
            m["function_call"] = {k: fn[k] for k in ("name", "arguments") if k in fn}
    if choice.get("finish_reason") == "tool_calls":
        choice["finish_reason"] = "function_call"
    return choice


def forced_call(tool_choice, tools: list[dict] | None, form: str = "qwen") -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": any of the tools; a named function: that one.  "none" is handled by the caller
    (no tools).  A value it cannot honour (an unknown shape, a name that is not one of the tools, "required" with no
    tools) is a ValueError - a 400, as OpenAI and Anthropic answer it - not a request that runs as "auto"."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    if kind == "required" and names:
        # the model writes "<function=NAME>" itself: from a written "<function=", Qwen3.8 went on with names no tool
        # has, and the reply came back as text (upstream d7e555bdb)
        return CALL_START if form == "glm" else CALL_START + "\n"
    if kind == "named" and name in names:
        return CALL_START + str(name) if form == "glm" else CALL_START + f"\n<function={name}>\n"
    if kind == "named":
        raise ValueError(f"tool_choice names {name!r}, which is not one of the request's tools")
    if kind == "required":
        raise ValueError("tool_choice asks for a tool call, but the request has no tools")
    raise ValueError(f"tool_choice {json.dumps(tool_choice)[:200]} is not a tool_choice this server takes")


PARAM_START = "<parameter="


def function_end(text: str) -> int:
    """Where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`), walking its
    parameters with param_end like call_end does; -1: not complete (or not in the call format)."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith(FUNC_START):
        return -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith(PARAM_START):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:], name=text[pos:gt].split("=", 1)[1].strip(), hold=False)  # the old rule
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            return pos + len(FUNC_END)
        else:
            return -1


_JSON_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n\s*", re.I)


def json_tool_call(body: str, schemas: dict) -> ToolCall | None:
    """`{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool -> its
    call; anything else -> None.  #2037: the object may sit in a Markdown fence (```json ... ```)."""
    body = body.strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = body[m.end():].rstrip()
        body = body[:-3] if body.endswith("```") else body
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in schemas:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {}
    return ToolCall(name=obj["name"], arguments=args) if isinstance(args, dict) else None


LENIENT_MAX = 1 << 20           # the longest value lenient_json will try (it is a python loop)


class _Lenient:
    """A JSON reader for the slips a model makes inside an array or object argument (#4427): a trailing or doubled
    comma, single quotes and Python's True/False/None, a bare string between two members of an object (`"status":
    "pending", "in_progress", "pending": false`), brackets left open at the end, extra closers at the end.  It is only
    asked about text json.loads has refused.  It never guesses at a value: a string cut short, an unknown word, a
    missing colon after a key, anything else raises ValueError."""

    _WORDS = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
    _ESC = {'"': '"', "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    _TOKEN = re.compile(r"[^\s,\]}:]+")

    def __init__(self, text: str):
        self.s, self.i = text, 0

    def peek(self) -> str:
        s, n = self.s, len(self.s)
        while self.i < n and s[self.i] in " \t\r\n":
            self.i += 1
        return s[self.i:self.i + 1]

    def string(self) -> str:
        s, q = self.s, self.s[self.i]
        self.i += 1
        out: list[str] = []
        while self.i < len(s):
            c = s[self.i]
            if c == q:
                self.i += 1
                return "".join(out)
            if c == "\\":
                self.i += 1
                e = s[self.i:self.i + 1]
                if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[self.i + 1:self.i + 5]):
                    out.append(chr(int(s[self.i + 1:self.i + 5], 16)))
                    self.i += 5
                    continue
                if e not in self._ESC:
                    raise ValueError("bad escape")
                out.append(self._ESC[e])
            else:
                out.append(c)
            self.i += 1
        raise ValueError("string not closed")

    def value(self, depth: int = 0):
        if depth > 64:
            raise ValueError("too deep")
        c = self.peek()
        if c == "{":
            return self.obj(depth + 1)
        if c == "[":
            return self.arr(depth + 1)
        if c in ('"', "'"):
            return self.string()
        m = self._TOKEN.match(self.s, self.i)
        if not m:
            raise ValueError("no value")
        self.i = m.end()
        if m.group() in self._WORDS:
            return self._WORDS[m.group()]
        return json.loads(m.group())            # a number, or ValueError

    def obj(self, depth: int):
        self.i += 1
        out: dict = {}
        while True:
            c = self.peek()
            if not c or c == "}":               # (no closer: the brackets were left open at the end)
                self.i += 1
                return out
            if c == ",":
                self.i += 1
                continue
            if c not in ('"', "'"):
                raise ValueError("object member")
            key = self.string()
            if self.peek() == ":":
                self.i += 1
                out[key] = self.value(depth)
            elif self.peek() not in (",", "}", ""):     # a bare string where a member belongs is dropped
                raise ValueError("object member")

    def arr(self, depth: int):
        self.i += 1
        out: list = []
        while True:
            c = self.peek()
            if not c or c == "]":
                self.i += 1
                return out
            if c == ",":
                self.i += 1
                continue
            out.append(self.value(depth))
            if self.peek() not in (",", "]", ""):
                raise ValueError("array element")


def lenient_json(text: str):
    """The JSON array or object `text` was meant to be, read past the slips of _Lenient; ValueError for anything else
    (and for text that is not an array or object).  Valid JSON is not its business: callers try json.loads first."""
    t = text.strip()
    m = _JSON_FENCE.match(t)
    if m:
        t = t[m.end():].rstrip()
        t = t[:-3].rstrip() if t.endswith("```") else t
    if not t.startswith(("[", "{")) or len(t) > LENIENT_MAX:
        raise ValueError("not an array or object")
    r = _Lenient(t)
    v = r.value()
    if t[r.i:].strip("]}) \t\r\n"):
        raise ValueError("text after the value")
    return v


def _json_type(value) -> str:
    # JSON Schema treats 1.0 as an integer too; Python's representation must not turn a valid enum number into text.
    return ("null" if value is None else "boolean" if isinstance(value, bool)
            else "integer" if isinstance(value, int) or isinstance(value, float) and value.is_integer()
            else "number" if isinstance(value, float) else "string" if isinstance(value, str)
            else "array" if isinstance(value, list) else "object")


def _schema_hints(schema, root=None, depth: int = 0, refs: tuple = ()) -> tuple:
    """Upper bounds on JSON types and on string enum/const values, not validation. None means unconstrained or
    unknown; an empty set means impossible. Siblings (including $ref) intersect; anyOf/oneOf branches union.
    Only same-resource JSON pointers are followed, from the supplied parameters root, never from a property.
    Anchors, remote refs, nested resource IDs and cycles are unknown; strict validation remains the final check."""
    if schema is False:
        return set(), set()
    if not isinstance(schema, dict) or depth > 8:
        return None, None
    if "$id" in schema and schema is not root:
        root = None                         # a nested resource changes the base of its references

    def intersect(a, b):
        return b if a is None else a if b is None else a & b

    def merge(a, b):
        return intersect(a[0], b[0]), intersect(a[1], b[1])

    def types_of(values):
        kinds = set(values)
        if "number" in kinds:
            kinds.add("integer")            # JSON Schema's number includes integer
        return kinds

    t = schema.get("type")
    types = types_of([t]) if isinstance(t, str) else types_of(t) if isinstance(t, list) and \
        all(isinstance(x, str) for x in t) else None
    hint = types, None
    for key in ("enum", "const"):
        values = schema.get(key) if key == "enum" else [schema[key]] if key in schema else None
        if isinstance(values, list):
            hint = merge(hint, (types_of(_json_type(v) for v in values),
                                {v for v in values if isinstance(v, str)}))
    ref = schema.get("$ref")
    if isinstance(ref, str) and root is not None and ref not in refs and (ref == "#" or ref.startswith("#/")):
        target = root
        for part in unquote(ref[2:]).split("/") if ref != "#" else ():
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and (target is root or not isinstance(target.get("$id"), str)):
                target = target.get(part)
            elif isinstance(target, list) and len(part) <= len(str(len(target))) and part.isdecimal() and \
                    str(int(part)) == part and int(part) < len(target):
                target = target[int(part)]
            else:
                target = None
                break
        hint = merge(hint, _schema_hints(target, root, depth + 1, refs + (ref,)))
    for key in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list):
            continue
        combined: tuple = (None, None) if key == "allOf" else (set(), set())
        for branch in branches:
            more = _schema_hints(branch, root, depth + 1, refs)
            if key == "allOf":
                combined = merge(combined, more)
            else:
                combined = tuple(None if a is None or b is None else a | b for a, b in zip(combined, more))
        hint = merge(hint, combined)
    return hint


def _schema_types(schema, depth: int = 0, *, root=None) -> set | None:
    """A conservative type bound; unknown branches cannot erase a sibling's explicit type constraint."""
    return _schema_hints(schema, root, depth)[0]


def _text_only(schema, root=None) -> bool:
    """Raw string text can stream immediately only when no enum/const may disambiguate JSON quoting."""
    types, strings = _schema_hints(schema, root)
    return types == {"string"} and strings is None


def param_value(text: str, schema, root=None, notes: list | None = None):
    """Restore parameter types, not validity. Plain string schemas keep raw text. An enum/const permits removing
    JSON string quotes only when the raw text is excluded and the decoded string is a candidate; ambiguity keeps
    the original. Nullable strings still decode null, but keep e.g. 1 as text when numbers are not allowed.
    #4427: where no string is allowed, an array or object json.loads refuses is read with lenient_json (a note says
    so in `notes`), and one that arrived as a JSON string of itself is decoded once more."""
    types, strings = _schema_hints(schema, root)
    if types == {"string"} and (strings is None or text in strings):
        return text
    try:
        value = json.loads(text)
    except ValueError:
        if not text.lstrip().startswith(("[", "{", "`")) or (types is not None and "string" in types):
            return text
        try:
            value = lenient_json(text)
        except (ValueError, RecursionError):
            return text
        if notes is not None:
            notes.append("a damaged JSON " + ("array" if isinstance(value, list) else "object")
                         + " argument was read as written")
        return value
    kind = _json_type(value)
    if kind == "string" and types and types <= {"array", "object"} and value.lstrip().startswith(("[", "{")):
        return param_value(value, schema, root, notes)
    if kind == "string" and strings is not None and (text in strings or value not in strings):
        return text
    if types is not None and "string" in types and kind not in types:
        return text
    return value


def _properties(schema) -> dict:
    """A tool's parameters' properties, {} where the schema has none or is not an object (a boolean schema)."""
    params = schema.get("parameters") if isinstance(schema, dict) else None
    props = params.get("properties") if isinstance(params, dict) else None
    return props if isinstance(props, dict) else {}


def parse_tool_call(body: str, schema: dict | None = None, notes: list | None = None) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")].strip()      # `<function= capture>`: the model's stray space
    rest = body[body.index(">") + 1:]
    props = _properties(schema)
    root = schema.get("parameters") if isinstance(schema, dict) else None
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        pname = rest[:rest.index(">")]
        rest = rest[rest.index(">") + 1:]
        end = param_end(rest, final=True, name=pname.strip())
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len(PARAM_END):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        args[pname] = param_value(value, props.get(pname), root, notes)
    return ToolCall(name=name, arguments=args)


# ------------------------------------------------------------------------------------------------ harmony
HARMONY_ENDS = ("<|end|>", "<|return|>", "<|call|>")
HARMONY_MESSAGE = "<|message|>"
HARMONY_TURN = "<|start|>assistant"                    # the generation prompt's end
HARMONY_NO_THINKING = "<|channel|>analysis<|message|><|end|>"


class HarmonyParser:
    """The output parser for the Harmony format (llm-jp-4.1's `chat_format=llm-jp-harmony-v1`; GPT-OSS's), with
    OutputParser's interface (feed, finish, state).  After the generation prompt the model writes messages
    `[<|start|>assistant[ to=functions.NAME]]<|channel|>CHANNEL[ to=functions.NAME][<|constrain|>TYPE]<|message|>BODY`
    ended by <|end|>, <|return|> or <|call|>: the analysis channel's body is the reasoning, final's and a commentary
    without a recipient (a preamble) the answer, a body to `functions.NAME` that tool's JSON arguments.  As llama.cpp's
    LLM-jp-4.1 parser (common/parsers/llm-jp-harmony.cpp): a space after a special token is the tokenizer's, and
    parallel calls are consecutive messages, each ended by <|end|>."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, **_):
        self.thinking = thinking
        self.names = {t.get("name") for t in tools or []}
        self.head = True             # reading a message's header (up to <|message|>)
        self.buf = ""                # text held: a header, or a body's tail that may begin a terminator
        self.channel = ""
        self.to: str | None = None
        self.args = ""               # a call's body so far
        self.lead = False            # the body's first character is the tokenizer's space after <|message|>
        self.pending: list = []      # OutputParser's: calls waiting in the reasoning (none here)
        self.rescued = self.refused = 0

    @property
    def state(self) -> str:
        if self.head:
            return "content" if self.channel in ("final", "commentary") else "reasoning" if self.thinking else "content"
        if self.to is not None:
            return "call"
        return "reasoning" if self.channel == "analysis" else "content"

    def _header(self, text: str) -> None:
        to = re.search(r"to=functions\.([A-Za-z0-9_.\-]+)", text)
        self.to = to.group(1) if to else None
        ch = re.search(r"<\|channel\|>\s*([A-Za-z]+)", text)
        self.channel = ch.group(1) if ch else self.channel or "final"

    def _body_event(self, text: str) -> list[Event]:
        if not text:
            return []
        if self.to is not None:
            self.args += text
            return []
        return [Event("reasoning" if self.channel == "analysis" else "content", text)]

    def _end_message(self) -> list[Event]:
        out: list[Event] = []
        if self.to is not None:
            try:
                args = json.loads(self.args) if self.args.strip() else {}
            except ValueError:
                args = None
            if isinstance(args, dict):
                out.append(Event("tool_call", call=ToolCall(name=self.to, arguments=args)))
            else:                    # not a call's JSON: the text it is
                out.append(Event("content", self.args))
        self.head, self.to, self.args = True, None, ""
        return out

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while self.buf:
            if self.head:
                at = self.buf.find(HARMONY_MESSAGE)
                if at < 0:
                    return out       # the header is still arriving
                self._header(self.buf[:at])
                self.buf = self.buf[at + len(HARMONY_MESSAGE):]
                self.head, self.lead = False, True
                continue
            if self.lead:
                if self.buf.startswith(" "):
                    self.buf = self.buf[1:]
                self.lead = False
            ends = [(self.buf.find(e), e) for e in HARMONY_ENDS if e in self.buf]
            if ends:
                at, tag = min(ends)
                out += self._body_event(self.buf[:at])
                self.buf = self.buf[at + len(tag):]
                out += self._end_message()
                continue
            # hold back a tail that could start a terminator
            keep = 0
            for e in HARMONY_ENDS:
                for k in range(min(len(e) - 1, len(self.buf)), 0, -1):
                    if self.buf.endswith(e[:k]):
                        keep = max(keep, k)
                        break
            out += self._body_event(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            return out
        return out

    def finish(self, reason: str | None = None) -> list[Event]:
        out: list[Event] = []
        if not self.head:
            out += self._body_event(self.buf)
            self.buf = ""
            if self.to is not None and reason in (None, "stop"):
                # the stop token ended it: `<|call|>` is one (as in llama.cpp), and the engine does not pass it on
                out += self._end_message()
            elif self.to is not None:   # a call the output was cut inside: its text, never a call
                out.append(Event("content", self.args))
                self.to, self.args = None, ""
        elif self.buf.strip() and HARMONY_MESSAGE not in self.buf:
            self.buf = ""            # an unfinished header carries no text
        return out


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""
    _line_parts: list[str]              # the current line in pieces (the `line` property)
    _line_blocks: list[str] | None

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False, deferred: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # #4663: the request defers tools (Anthropic's tool search, see anthropic_tools): the client holds back more
        # tools than it declares, so a call for an undeclared name is a call too - Claude Code runs it, where text ended
        # the turn.
        self.deferred = deferred
        # #804/#3075: in the answer too, a call counts only for a tool the request declared.  With tools declared a
        # call for another name stays text (the Responses spelling `namespace__tool` of a `namespace.tool` counts as
        # that tool); `declared_only` extends the rule to a request that declared none (Chat Completions, Anthropic:
        # the client has no tool to run; Responses but for a Codex compaction request, which keeps its conversation's
        # tools in the prompt only, and whose calls come back as before).
        self.declared_only = declared_only
        # A declared namespace.member tool is also accepted as namespace__member (PR #3556), namespace=member (Qwen
        # writes the <function= tag's equals sign into the name), namespace_member (one underscore dropped) and the
        # bare member (upstream fd8d37e93, #3972), without advertising a duplicate tool.  Only an unambiguous alias
        # counts (two namespaces can end in the same member), and an exactly declared name always wins over an alias.
        aliases: dict[str, set[str]] = {}
        for name in self.schemas:
            if isinstance(name, str) and "." in name:
                namespace, member = name.rsplit(".", 1)
                for alias in (f"{namespace}__{member}", f"{namespace}={member}", f"{namespace}_{member}", member):
                    aliases.setdefault(alias, set()).add(name)
        # #3796 (upstream 805ebe17b): the same tool written with every `__` collapsed to `_`
        # (`mcp__playwright__browser_navigate` as `mcp_playwright_browser_navigate`, the way Qwen often flattens a
        # name) is that tool.  Only a spelling of a declared name: a reordered or renamed tool (`server1_label_values`
        # for `label_values_mcp_server1`) is not guessed.
        for name in self.schemas:
            if isinstance(name, str) and "__" in name:
                aliases.setdefault(name.replace("__", "_"), set()).add(name)
                # #5281: a namespaced MCP tool (`mcp__assets__get_asset_info`) written with its last segment only
                # (`get_asset_info`), as the bare member of a dotted name is accepted above; only when exactly one
                # declared tool ends that way
                last = name.rsplit("__", 1)[1]
                if last:
                    aliases.setdefault(last, set()).add(name)
        # #4751: a programmatic-tool-calling harness (the DeepSeek Harness's PTC mode) shows the model its tools as
        # `tools.write(...)` inside code, and the model then writes the native call with that spelling (`tools.write`,
        # also `functions.write`).  The prefix is not part of the declared name; the same unambiguous-alias rule
        # applies, an exactly declared name always wins.
        for name in self.schemas:
            if isinstance(name, str) and name:
                for prefix in ("tools.", "functions.", "default_api."):
                    aliases.setdefault(prefix + name, set()).add(name)
        self.aliases = {alias: next(iter(names)) for alias, names in aliases.items()
                        if len(names) == 1 and alias not in self.schemas}
        self.unread = 0              # #3441: `<tool_call>` blocks that opened like a call and were returned as text
        self.unread_names: list[str] = []   # the names (first few) of those naming a tool the request did not declare
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code, current line) to tell an act from a quote.
        self.pending: list[list] = []
        self.rescued = 0             # calls delivered from the reasoning
        self.notes: list[str] = []   # #4427: what was repaired in the calls' arguments (a damaged JSON array or object)
        self.after_think = False     # #4077: the reasoning ended at a bare `</think>` (another in the answer: a stray)
        self.stray_think = 0         # ... and the strays of that kind dropped from the answer
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        # #4703: the line before was blank / this line is inside an indented (4-space) code block
        self._prev_blank = self._indented = False
        # recover (on in the server, off for a bare OutputParser; the config's "tool_call_recovery": false turns it
        # off): a declared tool's call written next to the template's
        # form is the call - `<parameter=NAME>` as the opener, JSON inside <tool_call>, a bare <function=NAME> at the
        # start of a line outside code, or a second call in the same <tool_call> (which otherwise merges into one).
        self.recover = recover
        self.bare = False            # the call being read opened with <function= alone (no <tool_call> around it)
        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._final = False          # #4701: the turn has ended: a call held for what might follow it is decided now
        self._held = ""              # #5276: the end of a misspelt opener still arriving (see _opener)
        self._reset_scan()

    def _ok_at(self, p: int) -> bool:
        """self.buf[p] would open a call: at the start of a line, outside a code fence and inline code."""
        snap = self._snap()
        self._track(self.buf[:p])
        ok = self._opener_ok()
        self._restore(snap)
        return ok

    def _opens_at(self, p: int) -> bool:
        """self.buf[p] would open a call as `<tool_call>` does: in the reasoning at the start of a line outside code, in
        the answer anywhere outside code (a fence, inline code, an indented block)."""
        if self.state == "reasoning":
            return self._ok_at(p)
        snap = self._snap()
        self._track(self.buf[:p])
        ok = not self._in_code()
        self._restore(snap)
        return ok

    def _opener(self) -> str | None:
        """#5276: `<toolcall>` / `<tool-call>` (the opener written as text, misspelled) followed by `<function=NAME>` of
        a declared tool is `<tool_call>`, where a `<tool_call>` would open a call (start of a line, outside code).
        Prose that names the misspelled tag, or any other follower, is left as it was.  Returns the tail of an opener
        still arriving (held back)."""
        buf = self.buf
        if "<tool" not in buf:
            return None
        pos, out = 0, []
        for m in _MISSPELT_OPEN.finditer(buf):
            name = m.group(2).strip()
            name = self.aliases.get(name, name)
            if name in self.schemas and self._opens_at(m.start()):
                out.append(buf[pos:m.start()] + CALL_START + m.group(1) + FUNC_START + m.group(2) + ">")
                pos = m.end()
        buf = "".join(out) + buf[pos:]
        held = None
        pm = _MISSPELT_PART.search(buf)
        tail = buf[pm.start():] if pm else None
        if tail is None:                                    # the opener itself still arriving: a prefix of `<toolcall>`
            for k in range(min(len(buf), 11), 5, -1):
                if any(w.startswith(buf[-k:]) for w in ("<toolcall>", "<tool-call>")) and buf[-k:] != "<tool_":
                    tail = buf[-k:]
                    break
        if tail:
            held, buf = tail, buf[:len(buf) - len(tail)]
        self.buf = buf
        return held

    def _callable(self, name: str) -> bool:
        """#3075: may a call for `name` be a call?  A tool the request declared (or its Responses spelling); with
        none declared, only where `declared_only` is off."""
        if not self.schemas:
            return not self.declared_only
        return name in self.schemas or name in self.aliases or self.deferred

    def _loose_open(self, after: str):
        """#3441: `after` (text after a `<tool_call>`, whitespace dropped) opens a call in a drifted form that names a
        declared tool -> (name, length of the opener); "wait" while it may still become one; None otherwise."""
        m = _LOOSE_OPEN.match(after)
        if m:
            name = m.group(1)
            return (name, m.end()) if self._callable(name) else None
        if ">" in after or "\n" in after or not after.startswith("<"):
            return None
        if _LOOSE_PART.match(after):
            bare = after[1:].strip().lstrip("=").strip()
            if after[1:].lstrip().startswith(("=", "function")):
                return "wait"
            if any(k.startswith(bare) for k in list(self.schemas) + list(self.aliases) if isinstance(k, str)):
                return "wait"
        return None

    def _follower(self, after: str) -> str:
        """recover: what a `<tool_call>` followed (after whitespace) by `after` opens: "call" (`<function=`), "param"
        (a declared tool's name written as `<parameter=NAME>`), "json" (`{`), "wait" (not decided yet) or "text"."""
        if after.startswith(FUNC_START):
            return "call"
        if after.startswith(PARAM_START):
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "param" if after[len(PARAM_START):gt] in self.schemas else "text"
        if after.startswith("{"):
            return "json" if self.schemas else "text"
        if after.startswith("```") or (after and "```".startswith(after)):
            # #2037: the JSON call inside a Markdown fence (```json ... ```) in the explicit wrapper
            m = _JSON_FENCE.match(after)
            if m:
                rest = after[m.end():]
                if rest.startswith("{"):
                    return "json" if self.schemas else "text"
                return "wait" if not rest else "text"
            return "wait" if "\n" not in after else "text"
        if not after or FUNC_START.startswith(after) or PARAM_START.startswith(after):
            return "wait"
        return "text"

    def _bare_opener(self) -> tuple[int, bool]:
        """recover: the first `<function=` in self.buf that opens a call without the wrapper (start of a line, outside
        code, a declared tool) -> (position, decided); (-1, False) for none.  A name still arriving is undecided."""
        p = self.buf.find(FUNC_START)
        while p >= 0:
            if self._ok_at(p):
                gt = self.buf.find(">", p)
                raw_name = self.buf[p + len(FUNC_START):gt if gt >= 0 else len(self.buf)]
                name = raw_name.strip()
                name = self.aliases.get(name, name)
                if gt < 0 and re.fullmatch(r"\s*[^<>\s]*\s*", raw_name):
                    return p, False
                if gt >= 0 and name in self.schemas:
                    return p, True
            p = self.buf.find(FUNC_START, p + 1)
        return -1, False

    @property
    def line(self) -> str:
        """Materialize the current line only for a decision, newline, or state snapshot."""
        if self._line_blocks is not None:
            self._line_blocks.extend(self._line_parts)
            self._line_parts = ["".join(self._line_blocks)]
            self._line_blocks = None
        elif len(self._line_parts) > 1:
            self._line_parts = ["".join(self._line_parts)]
        return self._line_parts[0] if self._line_parts else ""

    @line.setter
    def line(self, text: str):
        self._line_parts = [text] if text else []
        self._line_blocks = None

    def _track(self, text: str) -> str:
        """Follow the reasoning text that has gone out: the open code fence, the current line, and the backticks of
        the current paragraph.  Returns the text."""
        # Most token fragments stay on the current line. Fence and paragraph
        # bookkeeping happens only when a newline completes that line.
        if "\n" not in text:
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
                    # Release small fragment objects in bounded groups. Completed
                    # blocks are never recopied until the full line is requested.
                    if self._line_blocks is None:
                        self._line_blocks = []
                    self._line_blocks.append("".join(self._line_parts))
                    self._line_parts.clear()
            return text
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line = self.line + part if self._line_parts or self._line_blocks else part
                self._line_parts.clear()
                self._line_blocks = None
                s = line.lstrip()
                was_fenced = bool(self.fence)
                if self.fence:
                    # #4702: a fence is closed only by a line of at least as many of the same character as opened it
                    # (CommonMark), so a ``` example inside a ```` file stays inside it
                    run = len(s) - len(s.lstrip(self.fence[0]))
                    if run >= len(self.fence):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0] * (len(s) - len(s.lstrip(s[0])))
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
                # #4703: an indented (4 spaces or a tab) code block starts after a blank line and goes on until a line
                # that is not indented
                if was_fenced or self.fence:
                    self._prev_blank = self._indented = False
                elif not s:
                    self._prev_blank = True
                elif line.startswith(("    ", "\t")):
                    self._indented = self._prev_blank or self._indented
                    self._prev_blank = False
                else:
                    self._prev_blank = self._indented = False
            else:
                if part:
                    self._line_parts.append(part)
        return text

    def _opener_ok(self) -> bool:
        """The reasoning text so far puts a `<tool_call>` at the start of a line, outside a code fence and outside
        inline code."""
        return not self.fence and not self.line.strip() and self.ticks % 2 == 0 and not self._in_indented_block()

    def _in_indented_block(self) -> bool:
        """#4703: the current line is part of an indented code block: four spaces or a tab, after a blank line or
        another such line."""
        return (self._prev_blank or self._indented) and self.line.startswith(("    ", "\t"))

    def _snap(self) -> tuple:
        return (self.fence, self.line, self.ticks, self._prev_blank, self._indented)

    def _restore(self, snap: tuple) -> None:
        self.fence, self.line, self.ticks, self._prev_blank, self._indented = snap

    def _reset_track(self) -> None:
        self.fence, self.line, self.ticks = "", "", 0
        self._prev_blank = self._indented = False

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence, inline code or an indented code block."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1 or self._in_indented_block()

    def _quoted_think_end(self, i: int) -> bool | None:
        """#537: the `</think>` at self.buf[i] is the model quoting the tag while it thinks, not the end of the
        thinking: written right after a quote or backtick and followed by anything but a line break, as in `</think>`
        or "</think>...".  The end of the thinking is followed by a line break.  None while the character after the
        tag has not arrived."""
        before = self.buf[i - 1] if i else self.line[-1:]
        if before not in ('"', "'", "`"):
            return False
        after = self.buf[i + len(THINK_END):]
        if not after:
            return None
        return after[0] not in "\r\n"

    def _stray_think_at(self) -> int:
        """#4077: where self.buf has a `</think>` at the start of a line, in the answer of a turn whose reasoning
        already ended at one; -1 for none.  A fence or backtick is no reason to keep it: the reasoning may have left
        one open, and the answer's own tracking of code is wrong from there on."""
        k = self.buf.find(THINK_END)
        while k >= 0:
            before = self.buf[:k]
            if not (before.rsplit("\n", 1)[1] if "\n" in before else self.line + before).strip():
                return k
            k = self.buf.find(THINK_END, k + 1)
        return -1

    def _release(self, deliver: bool) -> list[Event]:
        """Settle the calls waiting in self.pending: events for real calls, or all of it back as reasoning text."""
        out = []
        for raw, call in self.pending:
            if call is not None and deliver:
                out.append(Event("tool_call", call=call))
                self.rescued += 1
            elif call is None or not deliver:
                if call is not None:
                    self.refused += 1
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _reset_scan(self):
        self._call_end_checked = 0   # closing-tag search progress for the current wrapped call
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> function_done -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sprops = {}             # the streamed call's parameters' schemas
        self.sroot = None            # their reference resolution root (the full parameters schema)
        self.spname = ""             # the parameter being streamed
        self.svstart = 0             # where its string value starts in self.buf (#4078)

    def _wrapped_call_end(self) -> int:
        # Without a closing tag the structural walk cannot finish. Search only
        # newly appended text plus enough overlap for a tag split across feeds.
        # A literal closing tag inside a value needs structural disambiguation:
        # once any candidate appears, keep the original full walk for this call.
        # Recovery can rewrite the buffer, so it always uses the full walk.
        if not self.recover and self._call_end_checked >= 0:
            start = max(0, self._call_end_checked - len(CALL_END) + 1)
            if self.buf.find(CALL_END, start) < 0:
                self._call_end_checked = len(self.buf)
                return -1
            self._call_end_checked = -1
        return call_end(self.buf, self._final)

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                raw_name = rest[a + 10:b]
                name = raw_name.strip()
                name = self.aliases.get(name, name)
                if raw_name != raw_name.strip() and name not in self.schemas:
                    return out
                self.scall = ToolCall(name=name, arguments={})
                tool = self.schemas.get(name)
                self.sprops = _properties(tool)
                self.sroot = tool.get("parameters") if isinstance(tool, dict) else None
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = self.spname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    self.sp += b + 1
                    if _text_only(self.sprops.get(pname), self.sroot):
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    self.sp += len("</function>")
                    self.ss = "function_done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                    self.svstart = self.sp
                # the walk starts at the value (a quoted call's opener counts); only a close in the new text can end it
                end = param_end(self.buf[self.svstart:], self._final, name=self.spname.strip()) \
                    if PARAM_END in rest else -1
                end = end - (self.sp - self.svstart) if end >= 0 else end
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving; so is a
                # </function>, which a later </parameter> makes part of the value and the output's end makes the
                # call's own close (_closed_at_end)
                safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, (PARAM_END, FUNC_END))
                f = rest.find(FUNC_END)
                if f >= 0:
                    safe = min(safe, f)
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest, self._final, name=self.spname.strip())
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                args(json.dumps(param_value(value, self.sprops.get(self.spname), self.sroot), ensure_ascii=False))
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """Close streamed JSON only after its call boundary has been confirmed."""
        out: list[Event] = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _closed_at_end(self, body: str) -> str | None:
        """The call body `body` (after `<tool_call>`) as the output's end leaves it: a parameter left open whose value
        runs into the call's own `</function>` (and `</tool_call>`), nothing after them.  Within the output a value may
        hold those tags; at its end nothing can follow, so they close the call (a model's slip in an agent loop:
        `<parameter=worker>admin>` then `</function>`, which left an announced call unfinished, #211, and the turn
        stopped).  The body with the parameter closed, or None when it is not that case."""
        m = _OPEN_PARAM_AT_END.search(body)
        if m is None or function_end(body) >= 0:
            return None
        fixed = body[:m.start()] + "\n" + PARAM_END + "\n" + FUNC_END
        return fixed if function_end(fixed) >= 0 else None

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        size = len(text)
        for tag in tags:
            if len(tag) <= 1:
                continue
            # Only a suffix beginning with the tag's first character can match.
            # Ordinary token deltas contain no '<': one C-level search replaces
            # checking every possible prefix. Earliest candidate is the longest.
            start = text.find(tag[0], max(0, size - len(tag) + 1))
            while start >= 0 and size - start > best:
                if tag.startswith(text[start:]):
                    best = size - start
                    break
                start = text.find(tag[0], start + 1)
        return best

    def _prefixed(self) -> str | None:
        """#5409: a call written `<function=tools NAME>` (a space for the dot of `tools.NAME`) is the call NAME, and one
        written `<function=tools>` (the bare prefix) is the one declared tool that takes the call's parameters (with
        tool_call_recovery).  Rewrites the complete ones in self.buf and returns the tail of a call still arriving
        (held back by feed), else None.  Only a declared name is ever the result; with none, or with two that fit, the
        text is left as it was."""
        buf = self.buf
        if "<function=" not in buf:
            return None

        def spaced(m):
            name = m.group(2).strip()
            return f"<function={name}>" if name in self.schemas else m.group(0)
        buf = _PREFIX_SPACED.sub(spaced, buf)
        held = None
        m = _PREFIX_PART.search(buf)
        if m:
            held, buf = buf[m.start():], buf[:m.start()]
        if self.recover:
            def bare(m):
                tool = None if m.group(2) in self.schemas else self._tool_by_params(
                    re.findall(r"<parameter=\s*([^<>\s]+?)\s*>", m.group(3)))
                return f"{m.group(1)}<function={tool}>{m.group(3)}</function>" if tool else m.group(0)
            buf = _PREFIX_BARE.sub(bare, buf)
            m = _PREFIX_BARE_OPEN.search(buf)
            if m and m.group(1) not in self.schemas and held is None and len(buf) - m.start() <= RCALL_MAX:
                held, buf = buf[m.start():], buf[:m.start()]
        self.buf = buf
        return held

    def _tool_by_params(self, names: list[str]) -> str | None:
        """The one declared tool whose properties hold every one of `names` and whose required parameters are all in
        them."""
        found = []
        for tname, schema in self.schemas.items():
            params = (schema or {}).get("parameters") or {}
            props = params.get("properties") or {}
            if names and all(n in props for n in names) and all(r in names for r in params.get("required") or []):
                found.append(tname)
        return found[0] if len(found) == 1 else None

    def feed(self, delta: str) -> list[Event]:
        if self._held:
            delta, self._held = self._held + delta, ""
        self.buf += delta
        if self.schemas and delta:
            held = self._opener()
            if held:
                self._held = held
        held = self._prefixed() if self.schemas else None
        out = self._feed()
        if held:
            self.buf += held
        return out

    def _feed(self) -> list[Event]:
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                body = self.buf[len(CALL_START):]
                end, think = call_end(body, hold=False), body.find(THINK_END)
                drop = len(CALL_END)
                # #4427: as in the answer, a following call opener confirms an omitted closer
                f = function_end(body)
                if f >= 0 and (end < 0 or f < end) and re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>",
                                                                body[f:].lstrip()):
                    end, drop = f, 0
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = body[:end].strip()[len("<function="):].split(">", 1)[0].strip()
                        name = self.aliases.get(name, name)
                        if self._callable(name):
                            call = parse_tool_call(body[:end], self.schemas.get(name), self.notes)
                            call.name = name
                    except ValueError:
                        pass
                    raw = self.buf[:len(CALL_START) + end + drop]
                    if call is not None or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + drop:]
                    self.state = "reasoning"
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(CALL_START) + think])))
                    self.buf = body[think:]
                    self.state = "reasoning"
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state = "reasoning"
                else:
                    return out
            elif self.state == "reasoning":
                if self.pending:
                    # calls wait for what follows: more calls or whitespace keep them, `</think>` makes them acts,
                    # any other text means they were quoted
                    stripped = self.buf.lstrip()
                    if len(stripped) < len(self.buf):
                        self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                        self.buf = stripped
                    if not self.buf:
                        return out
                    if self.buf.startswith(CALL_START):
                        self.state = "rcall"
                    elif self.buf.startswith(THINK_END):
                        out += self._release(True)
                    elif CALL_START.startswith(self.buf) or THINK_END.startswith(self.buf):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    if self._opener_ok():
                        self.buf = self.buf[tool:]
                        self.state = "rcall"
                    else:                            # mid-sentence, in a fence or in inline code: a quote
                        out.append(Event("reasoning", self._track(CALL_START)))
                        self.buf = self.buf[tool + len(CALL_START):]
                    continue
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END, CALL_START) if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is here
                    if i:
                        out.append(Event("reasoning", self._track(self.buf[:i])))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # #537: a quoted `</think>` is reasoning text
                    out.append(Event("reasoning", self._track(self.buf[:i + len(THINK_END)])))
                    self.buf = self.buf[i + len(THINK_END):]
                    continue
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead, self.after_think = "content", True, True
                self._reset_track()                                # the answer's own text starts here
            elif self.state == "content":
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                stray = self._stray_think_at() if self.after_think else -1
                if stray >= 0 and (i < 0 or stray < i):
                    # #4077: the reasoning ended at a `</think>` the model wrote while thinking (quoting the format on a
                    # line of its own), and this is the one that ends it.  What came between was reasoning, already sent
                    # as the answer; the tag is dropped and the text after it starts over, so a call that follows is
                    # not taken for a quote by the code fence or backtick the reasoning left open.
                    if stray:
                        out.append(Event("content", self._track(self.buf[:stray])))
                    self.buf = self.buf[stray + len(THINK_END):]
                    self._reset_track()
                    self.lead = True
                    self.stray_think += 1
                    continue
                b, decided = self._bare_opener() if self.recover and self.schemas else (-1, False)
                if b >= 0 and (i < 0 or b < i):
                    # recover: <function=NAME> alone at the start of a line - the call without its <tool_call>.
                    # Until the name has arrived it is held.
                    j = b
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0 and self.buf[:j].strip():
                        out.append(Event("content", self._track(self.buf[:j])))
                    if not decided:
                        self.buf = self.buf[j:]
                        return out
                    self.buf = self.buf[b:]
                    self.state, self.bare = "call", True
                    continue
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START, THINK_END) if self.after_think
                                                   else (CALL_START,))
                    if self.recover and self.schemas:            # a line that may be becoming a bare <function=
                        k = self.buf.rfind("\n") + 1
                        tail = self.buf[k:]
                        if tail and FUNC_START.startswith(tail) and self._ok_at(k):
                            j = min(j, k)
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.  The
                # text before it decides, so streamed and whole outputs agree.  Mid-sentence openers still count.
                snap = self._snap()
                self._track(self.buf[:i])
                in_code = self._in_code()
                self._restore(snap)
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                # A call is `<tool_call>` and then (after whitespace) `<function=`; the tag with anything else after
                # it is prose that names the format ("I'll use a <tool_call> block") - content, not a malformed call
                # that ends the request.  Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(CALL_START):].lstrip()
                if self.schemas and not after.startswith(FUNC_START):
                    loose = self._loose_open(after)
                    if loose == "wait":                  # may still become one: held like a partial tag
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if loose:                            # #3441: `<=NAME>` etc. is `<function=NAME>`
                        at = len(self.buf) - len(after)
                        self.buf = self.buf[:at] + FUNC_START + loose[0] + ">" + after[loose[1]:]
                        continue
                kind = self._follower(after) if self.recover else None
                if (after.startswith("<") and kind is None and not FUNC_START.startswith(after)
                        and not after.startswith(FUNC_START)):
                    self.unread += 1
                if kind in ("param", "json"):
                    if i and self.buf[:i].strip():
                        out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                    rest = self.buf[i + len(CALL_START):]
                    if kind == "param":                  # <parameter=NAME> as the opener: the call's <function=NAME>
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + after[len(PARAM_START):]
                    self.buf = rest
                    self.state = "call" if kind == "param" else "json"
                    continue
                if kind == "wait":
                    after = ""
                is_call = re.match(r"<function=(\s*[^<>\s]+\s*)>", after)
                if is_call:
                    raw_name = is_call.group(1)
                    name = raw_name.strip()
                    name = self.aliases.get(name, name)
                    if raw_name != raw_name.strip() and name not in self.schemas:
                        is_call = None     # only declared names gain whitespace normalization
                is_partial = FUNC_START.startswith(after) or re.fullmatch(r"<function=\s*[^<>\s]*\s*", after)
                if not is_call and not is_partial:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if not is_call:
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                if self.schemas or self.declared_only:
                    # #3075: a call counts only for a declared tool; the name must be complete to tell
                    gt = after.find(">")
                    nl = after.find("\n")
                    if gt < 0 and nl < 0:
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if gt >= 0 and (nl < 0 or gt < nl) and not self._callable(after[len(FUNC_START):gt].strip()):
                        # logged by the server: otherwise "the model ignored my tool" is the only clue
                        self.unread += 1
                        if len(self.unread_names) < 5:
                            self.unread_names.append(after[len(FUNC_START):gt].strip()[:80])
                        out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                        self.buf = self.buf[i + len(CALL_START):]
                        continue
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            elif self.state == "json":
                # recover: `{"name": ..., "arguments": {...}}` inside the wrapper is the call once its </tool_call> has
                # arrived and it names a declared tool; anything else is the text it is
                i = self.buf.find(CALL_END)
                if i < 0:
                    return out
                call = json_tool_call(self.buf[:i].strip(), self.schemas)
                if call is None:
                    out.append(Event("content", self._track(CALL_START + self.buf[:i + len(CALL_END)])))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[i + len(CALL_END):]
                self.state, self.lead = "content", call is not None
            else:
                drop = None
                if self.bare:                       # no wrapper: the call ends at its </function>; a stray
                    i = function_end(self.buf)      # </tool_call> after it is the wrapper's closer, dropped
                    if i >= 0:
                        after = self.buf[i:].lstrip()
                        if after.startswith(CALL_END):
                            drop = len(self.buf[i:]) - len(after) + len(CALL_END)
                        elif after and not CALL_END.startswith(after):
                            drop = 0
                else:
                    i = self._wrapped_call_end()
                    if i >= 0:
                        drop = len(CALL_END)
                    # Only a following call opener confirms an omitted wrapper closer.
                    f = function_end(self.buf)
                    nxt = self.buf[f:].lstrip() if f >= 0 else ""
                    if re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>", nxt):
                        i, drop = f, 0
                    if self.recover:                # a batch in one wrapper: the next call follows </function>
                        if nxt.startswith(PARAM_START) and self._follower(nxt) == "param":
                            at = len(self.buf) - len(nxt)
                            self.buf = self.buf[:at] + FUNC_START + nxt[len(PARAM_START):]
                            nxt = self.buf[at:]
                        if nxt.startswith(FUNC_START):
                            i, drop, self.batch = f, 0, True
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        final, self._final = self._final, True       # #5040: its end is known, so no tag is arriving
                        out += self._scan()
                        out += self._close_scan()
                        self.buf, self._final = whole, final
                    else:
                        out += self._scan()
                if i < 0 or drop is None:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + drop:]
                batch, self.batch = self.batch, False
                out += self._finish_call(body)
                if batch and out[-1].kind == "tool_call":
                    self.state, self.lead = "call", False

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211).  `reason` is
        how the turn ended: calls waiting from the reasoning become real calls only on a natural stop (None = stop);
        a turn cut by max tokens (or cancelled, or failed) keeps them as reasoning text (#1058)."""
        out = []
        if self._held:
            self.buf, self._held = self.buf + self._held, ""
        if self.buf and self.state in ("call", "rcall"):
            self._final = True      # #4701: a call held after its closing tags (could a file go on?) is whole now
            out += self.feed("")
        if self.state == "rcall" and reason in (None, "stop"):
            # #4427: a call in the reasoning whose `</function>` is the end of the output (the `</tool_call>` left out)
            # is as whole as one in the answer: it waits for the turn's end like the others
            body = self.buf[len(CALL_START):]
            f = function_end(body)
            name = body.lstrip()[len(FUNC_START):].split(">", 1)[0].strip() if f >= 0 else ""
            name = self.aliases.get(name, name)
            if f >= 0 and not body[f:].strip() and self._callable(name):
                try:
                    call = parse_tool_call(body[:f], self.schemas.get(name), self.notes)
                    call.name = name
                    self.pending.append([self.buf, call])
                    self.buf, self.state = "", "reasoning"
                except ValueError:
                    pass
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call" and self.bare and function_end(self.buf) >= 0:
            i = function_end(self.buf)          # recover: a bare call is whole at its </function>
            body, rest = self.buf[:i], self.buf[i:].strip()
            self.buf = ""
            out += self._finish_call(body)
            if rest and not CALL_END.startswith(rest):
                out.append(Event("content", self._track(rest)))
            return out
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            fixed = self._closed_at_end(self.buf) if self.ss in ("str", "raw") and reason in (None, "stop") else None
            closer = _OPEN_PARAM_AT_END.search(self.buf) if fixed is not None else None
            if fixed is not None and closer is not None:
                rest = self.buf[self.sp:closer.start()]
                if self.ss == "str":
                    if not self.sval_started and rest.startswith("\n"):   # the value's leading newline
                        rest = rest[1:]
                    tail = json.dumps(rest)[1:-1] + '"'
                else:
                    value = rest[1:] if rest.startswith("\n") else rest
                    tail = json.dumps(param_value(value, self.sprops.get(self.spname), self.sroot), ensure_ascii=False)
                out.append(Event("tool_args", tail + "}", call=self.scall))
                self.ss = "done"
                call = self.scall
                try:            # the final event carries the call's arguments, as the one _finish_call sends does
                    call = parse_tool_call(fixed, self.schemas.get(call.name), self.notes)
                    call.name, call.id = self.scall.name, self.scall.id
                except ValueError:
                    pass
                out.append(Event("tool_call", call=call))
                self.buf = ""
                self._reset_scan()
                return out
            if self.ss == "function_done" and not self.buf[self.sp:].strip() and reason in (None, "stop"):
                # A natural end immediately after </function> retains native streaming recovery.
                out += self._close_scan()
                call = self.scall
                try:            # the final event carries the call's arguments, as the one _finish_call sends does
                    call = parse_tool_call(self.buf, self.schemas.get(call.name), self.notes)
                    call.name, call.id = self.scall.name, self.scall.id
                except ValueError:
                    pass
                out.append(Event("tool_call", call=call))
            elif self.ss == "function_done":
                out.append(Event("content", self._track(CALL_START + self.buf)))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "call" and not self.bare and reason in (None, "stop"):
            fixed = self._closed_at_end(self.buf)
            if fixed is not None:               # as the streamed case above: its </function> closes it
                self.buf = ""
                return out + self._finish_call(fixed)
        if self.state == "reasoning" and self.buf == THINK_END:   # the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + self.buf if self.state in ("call", "json") and not self.bare else self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out

    def _finish_call(self, body: str) -> list[Event]:
        """A whole call body in the template's form -> its tool_call event.  With recover, one that still does not
        parse is the text it is instead of an error that ends the request."""
        raw_name = body.strip()[len(FUNC_START):].split(">", 1)[0]
        name = raw_name.strip()
        name = self.aliases.get(name, name)
        bare, self.bare = self.bare, False
        try:
            if raw_name != raw_name.strip() and name not in self.schemas:
                raise ValueError("whitespace tool name is not declared")
            call = parse_tool_call(body, self.schemas.get(name), self.notes)
            call.name = name
        except ValueError:
            if not self.recover:
                raise
            self._reset_scan()
            self.state = "content"
            return [Event("content", self._track(body if bare else CALL_START + body + CALL_END))]
        if self.scall is not None:
            call.id = self.scall.id
        self._reset_scan()
        self.state, self.lead = "content", True
        return [Event("tool_call", call=call)]


class GlmOutputParser(OutputParser):
    """OutputParser for GLM's call form: the output rewritten as it streams (GlmCallRewriter) before it is parsed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewriter = GlmCallRewriter()

    def feed(self, delta: str) -> list[Event]:
        return super().feed(self.rewriter.feed(delta))

    def finish(self, reason: str | None = None) -> list[Event]:
        return super().feed(self.rewriter.flush()) + super().finish(reason)
