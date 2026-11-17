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

import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

LOGGER = logging.getLogger(__name__)


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

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        self.caps = self._detect_caps()

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1" or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                      and not _text_of(m.get("content")).strip()
                                                      and not _has_image(m.get("content")) and not m.get("tool_calls"))]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)

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
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", "<tool_call>", "<function=")),
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

# Thinking levels.  The model's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


_EFFORT_QWEN = dict(EFFORT)


def use_family(family: str | None) -> None:
    """The reasoning-effort words of the model family this server runs (module state, like ENV_TO_USER): the Qwen table."""
    EFFORT.clear()
    EFFORT.update(_EFFORT_QWEN)


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
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else EFFORT["high"]}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


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
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking",
                                               "tool_addition")


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
# 0.1.44: `<tool_call>`, `<tool_response>` and `</tool_response>` are type 4 too (ids 248058, 248066, 248070), so a quoted
# one handed the model the real call / tool-result marker in the same way (found by 47Hunter47 in #2026).  The template
# writes the tool-result markers around a tool message itself; only the text inside a message is marked.
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
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(c for c in controls
                                                                                if c not in LITERAL_TAGS)}}
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
    """A user message whose whole text is <tool_response>...</tool_response> (a client that formats a tool result the way
    the template does for a tool message) keeps that outer pair as the markers - the template tells such a turn from a
    real question by exactly that pair - and only the text inside is marked.  None: not such a message."""
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
# The prefix cache only reuses a prompt's start, so a new folder or a new day read all of it again (~14K tokens, 8-11 s).
# Moved to the front of the first user message, the system turn (and the tools the template writes into it) is the same
# for every session and the disk prefix cache restores it.  The model still sees the same lines, one turn later.
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
    return _stable_tool_order(tools, lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


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
    """A chat request with no message at all is a 400 (OpenAI: "empty array. Expected an array with minimum length 1"); the
    template would otherwise render just the assistant turn and the model would answer nothing."""
    if not messages:
        raise ValueError("messages must be a non-empty list: send at least one message")


def _order_chat_tool_results(messages: list[dict]) -> None:
    """The template pairs results by position, so restore call order while IDs are still available.

    Only a complete, unambiguous run of results is reordered. Legacy or incomplete histories keep their
    received order. IDs are private to this conversion and never reach the template.
    """
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


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        # an image is read in a user, tool or assistant message; the system prompt takes text only, as _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS if images else TEXT_PARTS)
        out = {"role": role, "content": _parts_of(content) if images else _text_of(content)}
        if role == "tool":
            out["_call_id"] = m.get("tool_call_id")
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                calls.append({"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))},
                              "_call_id": c.get("id")})
            out["tool_calls"] = calls
        messages.append(out)
    _order_chat_tool_results(messages)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    given = req.get("reasoning_effort")
    # `false` is a level (thinking off, effort_kwargs): an `or` between the two fields dropped it
    kwargs.update(effort_kwargs(given if given is not None and given != "" else reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
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


def _anthropic_tool(t: dict) -> dict:
    """An Anthropic tool -> the template's form of it (also what the output parser reads its schema from)."""
    return {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}


def anthropic_tools(req: dict) -> tuple[list[dict] | None, bool]:
    """#4663, Anthropic's tool search: (every tool the request declared, the `defer_loading` ones included, whether any
    tool is deferred).  A deferred tool is callable although the prompt does not list it.  A client that defers tools
    (Claude Code) holds back more than it declares - it names the rest in the conversation and runs a call for any of
    them - so with one deferred tool a call for an undeclared name is a call too (OutputParser's `deferred`)."""
    tools = _anthropic_callables(req)
    return [_anthropic_tool(t) for t in tools] or None, any(t.get("defer_loading") for t in tools)


def _anthropic_callables(req: dict) -> list[dict]:
    """#5952: the request's callable tools, each name once where it comes from a tool_addition: the top-level `tools`
    first (as declared, `defer_loading` ones included), then each name a system message's tool_addition adds, its
    FIRST definition.  A name the top level declares keeps the top-level definition (a later addition of it adds
    nothing).  The prompt shows an added tool with this same definition (anthropic_to_messages), so what the model
    reads and what the parser takes a call against are one schema."""
    tools = list(_tool_list(req.get("tools"), None))      # a copy: the request's own list is not changed
    names = {t["name"] for t in tools}
    for d in _tool_additions(req):
        if d["name"] not in names:
            names.add(d["name"])
            tools.append(d)
    return tools


def _tool_addition(block) -> dict | None:
    """#5952: a `tool_addition` block's tool definition ({"name", "description", "input_schema"}), or None when the
    block does not carry one in that shape.  Claude Code sends some tools (MCP servers that connect after the session
    started) this way, in a system message inside `messages`, instead of in the top-level `tools`:
    {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {"name": ..., "input_schema": ...}}}.
    The outer `tool` must say "tool_definition"; the name a non-empty string; a description, when there, a string; a
    schema, when there, an object."""
    tool = block.get("tool") if isinstance(block, dict) else None
    if not isinstance(tool, dict) or tool.get("type") != "tool_definition":
        return None
    d = tool.get("definition")
    if (isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"]
            and isinstance(d.get("input_schema", {}), dict) and isinstance(d.get("description", ""), str)):
        return d
    return None


def _tool_additions(req: dict) -> list[dict]:
    """#5952: every tool definition a SYSTEM message of the conversation adds with a `tool_addition` block, in order.
    The block is the client's (Claude Code puts it in a system message); one in a user or assistant turn - text a
    user pasted, a tool's output echoed back - adds no tool."""
    out = []
    messages = req.get("messages")
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        content = m.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_addition":
                d = _tool_addition(block)
                if d is not None:
                    out.append(d)
    return out


def _tool_references(content, deferred: dict):
    """A tool_result's tool_reference blocks (a tool search's results) -> the referenced deferred tools' definitions,
    in the form the prompt lists tools: the model sees a tool where it was loaded, and the prompt's start (its tool
    list) stays the same, so the prompt cache does too.  Any other reference is left out as before (_check_parts): a
    listed tool is in the prompt already, and an old one (its MCP server gone) must not fail the rest of the session."""
    if not isinstance(content, list):
        return content
    return [{"type": "text", "text": "<tools>\n" + json.dumps(deferred[p.get("tool_name")], ensure_ascii=False) + "\n</tools>"}
            if isinstance(p, dict) and p.get("type") == "tool_reference" and isinstance(p.get("tool_name"), str)
            and p["tool_name"] in deferred else p
            for p in content]


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request")."""
    declared = _tool_list(req.get("tools"), None)
    # #4663: a `defer_loading` tool stays out of the prompt's tool list until a tool_reference loads it
    deferred = {t["name"]: _anthropic_tool(t) for t in declared if t.get("defer_loading")}
    # #5952: the names the prompt already shows the model (its tool list); a tool_addition of one of them adds nothing.
    # An added name is shown with its one callable definition (_anthropic_callables: a deferred top-level tool's own,
    # else the first addition's), where the first addition of it stands.
    shown = {t["name"] for t in declared if t["name"] not in deferred}
    callable_defs = {}
    for t in _anthropic_callables(req):
        callable_defs.setdefault(t["name"], t)
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS)))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(m.get("content"), f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_addition":
                # #5952: the added tool's definition where the client added it, in the form a tool_reference loads
                # one (_tool_references): the prompt's start (its tool list) stays the same, and so does its cache
                d = _tool_addition(block)
                if d is None or m.get("role") != "system":
                    LOGGER.debug("messages[%d].content[%d]: tool_addition %s; left out of the prompt", i, j,
                                 "without a tool definition" if d is None else f"in a {m.get('role')} message")
                elif d["name"] not in shown:
                    shown.add(d["name"])
                    added = ("<tools>\n" + json.dumps(_anthropic_tool(callable_defs[d["name"]]), ensure_ascii=False) +
                             "\n</tools>")
                    text.append(added)
                    parts.append({"type": "text", "text": added})
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))},
                              "_call_id": block.get("id")})
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.
                messages.append({"role": "tool", "content": _parts_of(_check_parts(
                    _tool_references(block.get("content"), deferred),
                    f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS)),
                                 "_call_id": block.get("tool_use_id")})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    # A client may send results in the order its tools finished (Claude Code does for parallel calls)
    _order_chat_tool_results(messages)
    tools = [_anthropic_tool(t) for t in declared if t["name"] not in deferred] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
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
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    return _env_to_user(_late_system_to_user(messages)), tools, kwargs


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
_PREFIX_LOOKBACK = 256     # how far before a new delta a prefixed name or a misspelt opener can start (see OutputParser._prefixed)
# #5409: `<function=tools actio>` / `<function=tools>`; the prefixes are the ones of #4751
_PREFIX_SPACED = re.compile(r"(<function=\s*(?:tools|functions|default_api)[ \t]+)([^<>\s]+)\s*>")
_PREFIX_PART = re.compile(r"<function=\s*(?:tools|functions|default_api)[ \t]+[^<>\s]{0,80}\s*$")
_PREFIX_BARE = re.compile(r"(<tool_call>\s*)<function=\s*(tools|functions|default_api)\s*>(.*?)</function>", re.S)
_PREFIX_BARE_OPEN = re.compile(r"<tool_call>\s*<function=\s*(tools|functions|default_api)\s*>(?!.*</function>)", re.S)
# #5276: `<toolcall>` / `<tool-call>` as the opener (then whitespace, `<function=NAME>`), and such an opener still arriving
_MISSPELT_OPEN = re.compile(r"<tool-?call>(\s*)<function=(\s*[^<>\s]+\s*)>")
_MISSPELT_PART = re.compile(r"<tool-?call>\s*<function=[^<>]{0,80}$|<tool-?call>\s*(?:<(?:f(?:u(?:n(?:c(?:t(?:i(?:o(?:n)?)?)?)?)?)?)?)?)?$")
# #5969 (recover): Claude's `<invoke name="NAME"><parameter name="K">V</parameter></invoke>` dialect inside the wrapper
_INVOKE_OPEN = re.compile(r'<invoke\s+name="([^"<>\n]+)"\s*>')
_INVOKE_PART = re.compile(r'<(?:i(?:n(?:v(?:o(?:k(?:e(?:\s+(?:n(?:a(?:m(?:e(?:=(?:"[^"<>\n]*"?)?)?)?)?)?)?)?)?)?)?)?)?)?\s*$')
_INVOKE_PARAM = re.compile(r'((?:<invoke\s+name="[^"<>\n]+"\s*>|</parameter>)\s*)<parameter\s+name="([^"<>\n]+)"\s*>')
# the call's end: `</invoke>` right after a value's `</parameter>` or the opener (a value may hold the words)
_INVOKE_END = re.compile(r'(?:</parameter>|<invoke\s+name="[^"<>\n]+"\s*>)\s*</invoke>')


def invoke_to_template(raw: str) -> str:
    """#5969: an `<invoke ...>` call body (up to its `</invoke>`) in the template's form: `<function=T>`,
    `<parameter=K>`, `</function>`; the tags are rewritten only where the call's structure has them."""
    m = _INVOKE_OPEN.match(raw.lstrip())
    body = raw.lstrip()
    body = _INVOKE_PARAM.sub(lambda p: p.group(1) + PARAM_START + p.group(2) + ">", body)
    body = _INVOKE_PARAM.sub(lambda p: p.group(1) + PARAM_START + p.group(2) + ">", body)   # (overlapping neighbours)
    body = FUNC_START + m.group(1) + ">" + body[m.end():]
    return body[:len(body) - len("</invoke>")] + FUNC_END if body.endswith("</invoke>") else body


_LOOSE_PART = re.compile(r"<\s*(?:function[ \t]*[=:]?\s*|=\s*)?[A-Za-z0-9_.\-]*\s*$")


# a quoted call's opener in a value: `<function=NAME>` and its first parameter or close; a bare FUNC_START is text
_QUOTED_OPEN = re.compile(r"<function=[^<>\s]+>\s*(?:<parameter=|</function>)")


#: #4701: how much text after a closing sequence is looked at before the value is taken as ended there (a call followed by a long sentence)
CLOSE_LOOKAHEAD_CHARS = 1500


def _close_verdict(after: str, final: bool) -> str:
    """`after` starts with `</function>` and/or `</tool_call>` right after a `</parameter>`.  A file that documents the call format has those
    three tags on lines of its own inside a value (#4701), so the sequence only ends the value when what follows is the end of the turn, another
    call, or prose with no closing tag after it.  "end": the value ends here; "continue": it goes on to the next `</parameter>`; "hold": not
    decidable yet (streaming): wait for more text.  Bounded: the end of the turn (`final`), a new `<tool_call>`, or CLOSE_LOOKAHEAD_CHARS of
    text decide, so a call is never held longer than that."""
    m = re.match(r"(?:</function>)?\s*</tool_call>", after)
    if not m:
        tail = after[len(FUNC_END):].lstrip() if after.startswith(FUNC_END) else after
        if not final and CALL_END.startswith(tail):
            return "hold"                                  # #5040: `</tool_call>` is still arriving: a streamed view must not end the value yet
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
        return "continue"                                  # prose, then closing tags again: the text was part of the value
    if final or len(rest) > CLOSE_LOOKAHEAD_CHARS:
        return "end"
    return "hold"



def param_end(text: str, final: bool = False, name: str | None = None, hold: bool = True,
              allow_reopen: bool = False, names=None, closer: str | None = None) -> int:
    """Where a parameter value in `text` ends: the first PARAM_END followed (after whitespace) by the next
    PARAM_START or FUNC_END, or by the wrapper's CALL_END when the call's `</function>` was left out (#4427) - the
    same text inside a value (a file that documents the call format, #210) is
    part of the value.  #4078: a close inside a call the value itself quotes (more _QUOTED_OPEN than FUNC_END before
    it), or one followed by the same string parameter `name` again, is part of the value too.  `text` starts at the value.
    #6032 R4: `closer` (`</NAME>` of a value opened as the element `<NAME>`) ends it too, under the same follower rule;
    close_len() says which closer stands at the answer.
    -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    def nxt(start):
        a = text.find(PARAM_END, start)
        b = text.find(closer, start) if closer else -1
        return a if b < 0 or 0 <= a < b else b
    at = nxt(0)
    while at >= 0:
        if len(_QUOTED_OPEN.findall(text, 0, at)) <= text.count(FUNC_END, 0, at):
            after = text[at + close_len(text, at, closer):].lstrip()
            if after.startswith(PARAM_START):
                gt = after.find(">")
                if gt < 0:
                    return at if final else -2
                if not allow_reopen or name is None or after[len(PARAM_START):gt].strip() != name:
                    return at
            elif after.startswith((FUNC_END, CALL_END)):          # #4427: the model left out its `</function>`
                verdict = _close_verdict(after, final) if hold else "end"      # hold=False: a call inside the reasoning keeps its old rule
                if verdict == "end":
                    return at
                if verdict == "hold":
                    return -2
            elif after.startswith(FOLLOWERS) or _follower_form(after):    # #6032 R1: the damaged followers
                return at
            elif names and opener_at(after, names, final) not in (None, "pending"):
                return at                       # #6032 R2: `<NAME>` / `</NAME>` of a declared parameter not set yet
            elif names is not None and _stub_follows(after):
                return at                       # #6032 R5 (recover): the model's empty stub, or a repeated closer
            elif not after or PARAM_START.startswith(after) or FUNC_END.startswith(after) or CALL_END.startswith(after) \
                    or any(t.startswith(after) for t in FOLLOWERS) or _follower_form(after) is None \
                    or (names and opener_at(after, names, final) == "pending") \
                    or (names is not None and _stub_follows(after) is None):
                return at if final else -2
        at = nxt(at + 1)
    return -1


def close_len(text: str, at: int, closer: str | None = None) -> int:
    """#6032 R4: the length of the value closer param_end found at `at`: `</parameter>`, or the element's `</NAME>`."""
    return len(closer) if closer and not text.startswith(PARAM_END, at) and text.startswith(closer, at) else len(PARAM_END)


# #6032 R1: what else ends a value after its `</parameter>` (Gyro / Coder-L recordings): the Anthropic XML form's
# `</invoke>` and `<parameter name=` written in place of the call format's own tags, and a garbled `</<`
FOLLOWERS = ("</<", "</invoke>", "<parameter name=")
_DAMAGED_OPENER = re.compile(r"<=[^<>\s]+>")          # `<=NAME>`: a damaged parameter opener
_STRAY_CLOSER = re.compile(r"</(?!parameter>|function>|tool_call>)[^<>\s=/]+>\s*")


def _follower_form(after: str):
    """#6032 R1: `after` (what follows a `</parameter>`, no leading whitespace) starts with a damaged opener `<=NAME>`,
    or with a stray closer `</NAME>` standing where `</function>` belongs (then `</tool_call>`): True; None while it
    may still become one (streaming); False otherwise (a tag of the value itself: the value goes on)."""
    if _DAMAGED_OPENER.match(after):
        return True
    if re.fullmatch(r"<=[^<>\s]*", after):
        return None
    m = _STRAY_CLOSER.match(after)
    if m:
        rest = after[m.end():]
        if rest.startswith(CALL_END):
            return True
        return None if not rest or CALL_END.startswith(rest) else False
    return None if re.fullmatch(r"</(?!parameter>|function>|tool_call>)[^<>\s=/]*", after) else False


# #6032 R5: the model's empty stub after a value (Gyro / Coder-L recordings): a lone `<` or `</` line, a line of one tag
# (`</head_after>`), either closed by a `</parameter>` or followed by the next opener; and a closer written again
_STUB = re.compile(r"(?:</?[ \t]*\n|<[/=]?[^<>\s=/]+>[ \t]*\n)\s*")
_STUB_NEXT = (PARAM_END, "<parameter=", "</function>", "</tool_call>") + ("</<", "</invoke>", "<parameter name=")


def _stub_follows(after: str):
    """#6032 R5: `after` (what follows a value's `</parameter>`, no leading whitespace) is a closer written again, or
    an empty stub line followed by a closer or an opener: True; None while it may still become one; False otherwise."""
    if after.startswith(PARAM_END):
        return True
    m = _STUB.match(after)
    if m:
        rest = after[m.end():]
        if rest.startswith(_STUB_NEXT):
            return True
        return None if not rest or any(t.startswith(rest) for t in _STUB_NEXT) else False
    return None if re.fullmatch(r"</?[ \t]*|<[/=]?[^<>\s=/]*>?[ \t]*", after) else False


NOT_PARAMS = ("function", "think", "parameter")


def param_name(opener: str) -> str | None:
    """#6032 R1: the parameter a `<parameter=` opener names (`opener`: what follows `=` up to the `>` found): the name
    cut at its first whitespace, `<` or `>` (the model ran a closer or `</think>` into it, or put a space before the
    `>`); None - the value is read and dropped - when that leaves nothing, a placeholder (null / none) or one of the
    call format's own tag names."""
    name = re.split(r"[\s<>]", opener.lstrip(), 1)[0]
    if not name or name in NOT_PARAMS or name.lower() in ("null", "none"):
        return None
    return name


ARGS_START = "<arguments>"


def json_args(s: str, props=None, notes: list | None = None):
    """#6032 R3 (recover): `s` starts with `<arguments>` + a JSON object (Gyro / Coder-L wrote the call's arguments so,
    instead of `<parameter=` tags), closed by `</arguments>`, by the stray `</parameter>` the model wrote, or by
    nothing.  -> (arguments, length read) with each string value of a declared non-string parameter read by its type
    (coerce_value); None while the object is not whole (a call cut inside it is never run with partial strings)."""
    t = s[len(ARGS_START):]
    body = t.lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    tail = re.match(r"\s*(?:</arguments>|" + re.escape(PARAM_END) + r")?", body[end:])
    args = {}
    for k, v in obj.items():
        declared = declared_type((props or {}).get(k))
        if isinstance(v, str) and declared is not None and not _declares(declared, "string"):
            v = coerce_value(v, declared, notes)
            if v is _SKIP:
                continue
        args[k] = v
    return args, len(ARGS_START) + len(t) - len(body) + end + tail.end()


_COLON_OPENER = re.compile(r"<parameter=([^<>\s:=]+):[ \t]*([^\n<>]*)\n")
_BARE_OPENER = re.compile(r"<(?:/|=)?([^<>\s=/]+)>")
# #5969: the Anthropic XML form's parameter opener written inside the call format's `<function=NAME>` (a declared parameter)
_NAMED_OPENER = re.compile(r'<parameter\s+name\s*=\s*"?([^"<>\s]+)"?\s*>')


def opener_at(s: str, names=None, final: bool = False):
    """#6032 R2: the parameter opener `s` starts with (no leading whitespace), at a point where one may begin (after
    the function opener or a value's `</parameter>`).  -> (name, length, inline value): name None = read and dropped;
    inline value: the colon form's own value, the rest None.  "pending": it may still become one (streaming; `final`
    decides); None: no opener here.
      `<parameter=NAME>`: the call format's own (the name as param_name reads it);
      `<parameter=NAME: value` + line break (no `>` on the line): NAME with that value, quotes around it dropped,
          then its `</parameter>` (Gyro / Coder-L, recorded);
      `<NAME>`, `</NAME>`, `<=NAME>`: an opener of NAME when NAME is in `names` (the call's declared parameters not
          set yet); any other such tag is no opener (a tag of the text, or one R1 drops)."""
    if s.startswith(PARAM_START):
        line = s.split("\n", 1)[0]
        gt = line.find(">")
        if gt >= 0:
            return param_name(s[len(PARAM_START):gt]), gt + 1, None
        m = _COLON_OPENER.match(s)
        if m:
            value = m.group(2).strip()
            if value.startswith('"'):
                value = value[1:]
                value = value[:-1] if value.endswith('"') else value
            tail = re.match(r"\s*" + re.escape(PARAM_END), s[m.end():])
            if tail is None and not final and PARAM_END.startswith(s[m.end():].lstrip()):
                return "pending"
            return param_name(m.group(1)), m.end() + (tail.end() if tail else 0), value
        if "\n" not in s:
            # the `>` still arriving, or a long name: held, except that the call's own scan reads it as before when
            # the line ends (a `<parameter=` line with neither `>` nor a colon form is no opener)
            if s.find(">") < 0 and not final:
                return "pending"
            gt = s.find(">")
            return (param_name(s[len(PARAM_START):gt]), gt + 1, None) if gt >= 0 else None
        return None, len(line), None        # #6032 R5: an opener cut before its `>` (its line ends first): dropped
    if names:
        m = _NAMED_OPENER.match(s)
        if m:                                # #5969: `<parameter name="NAME">` inside a `<function=` call
            return (m.group(1), m.end(), None) if m.group(1) in names else None
        if not final and any(f'<parameter name="{n}">'.startswith(s) or f"<parameter name={n}>".startswith(s)
                             for n in names):
            return "pending"
        m = _BARE_OPENER.match(s)
        if m:
            return (m.group(1), m.end(), None) if m.group(1) in names else None
        if not final and re.fullmatch(r"<[/=]?[^<>\s=/]*", s) and any(
                n.startswith(s.lstrip("</=")) for n in names):
            return "pending"
    return None


def call_end(text: str, final: bool = False, hold: bool = True, names=None) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before.  #6032 R4 (recover): `names` - the declared
    parameters of the call's tool - lets the walk step over damaged openers and elements as parse_tool_call reads them."""
    seen = set()
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
        # #6032 R2-R5 (recover, the tool's names known): every opener as parse_tool_call reads it - a damaged one, an
        # element, the colon form, one cut before its `>` - so the walk never runs one call's end into the next call
        got = opener_at(s, set(names) - seen, final) if names and not s.startswith(FUNC_END) else None
        if got == "pending":
            return -1
        if got is not None and got[2] is not None:            # the colon form: its value and closer are read
            seen.add(got[0])
            pos += got[1]
        elif s.startswith("<parameter=") and got is None:
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            pname = param_name(text[pos:gt].split("=", 1)[1])
            end = param_end(text[gt + 1:], final, pname, hold, allow_reopen=True,
                            names=(set(names) - seen - {pname}) if names else None)
            if end < 0:
                return -1
            seen.add(pname)
            pos = gt + 1 + end + len(PARAM_END)
        elif got is not None:                                 # #6032 R2 / R4 / R5: any opener, an element too
            pname, length, _ = got
            closer = "</%s>" % pname if pname and s.startswith("<%s>" % pname) else None
            end = param_end(text[pos + length:], final, pname, hold, allow_reopen=True,
                            names=set(names) - seen - {pname}, closer=closer)
            if end < 0:
                return -1
            seen.add(pname)
            pos = pos + length + end + close_len(text[pos + length:], end, closer)
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
        if kind in ("function", "tool"):
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def forced_call(tool_choice, tools: list[dict] | None, callable_tools: list[dict] | None = None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": any of the tools; a named function: that one.  A value it cannot honour (an
    unknown shape, a name that is not one of the tools, "required" with no tools) is logged and acts as "auto",
    not a 400: a client's odd choice must not stop its request.  "none" is handled by the caller (no tools).
    `callable_tools` (Anthropic: the tools added in the conversation and the deferred ones as well, #5952): a name
    among them may be named too; "required" still picks from the prompt's own list."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind == "named" and name not in names and any(
            isinstance(t, dict) and t.get("name") == name for t in callable_tools or []):
        return CALL_START + f"\n<function={name}>\n"
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    if kind == "required" and names:               # the model writes "<function=NAME>": from a written "<function=", Qwen3.8 went on with names no tool has
        return CALL_START + "\n"
    if kind == "named" and name in names:
        return CALL_START + f"\n<function={name}>\n"
    print(f"[strata] tool_choice {json.dumps(tool_choice)[:200]} is not supported here (or names no tool of the "
          "request): the model decides, as with \"auto\"", flush=True)
    return None


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
            end = param_end(text[gt + 1:], name=param_name(text[pos:gt].split("=", 1)[1]), hold=False, allow_reopen=True)   # a bare or unwrapped call: the old rule
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
    """A JSON reader for the slips a model makes inside an array or object argument (#4427): a trailing or doubled comma,
    single quotes and Python's True/False/None, a bare string between two members of an object (`"status": "pending",
    "in_progress", "pending": false`), brackets left open at the end, extra closers at the end.  It is only asked about
    text json.loads has refused.  It never guesses at a value: a string cut short, an unknown word, a missing colon
    after a key, anything else raises ValueError."""

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
        out = []
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


def declared_type(prop, depth: int = 0):
    """#2371: a parameter's declared JSON type(s) for coerce_value: its "type", or the union of its anyOf / oneOf
    branches' types (["boolean", "null"] for {"anyOf": [{"type": "boolean"}, {"type": "null"}]}, a nested branch
    included).  None when a branch names no type (a $ref, an open schema): the value is then read as with no schema.
    Callers used to read the top-level "type" only, so a boolean declared through anyOf / oneOf stayed the text "True"."""
    if not isinstance(prop, dict) or depth > 8:
        return None
    t = prop.get("type")
    if t is not None:
        return t
    for key in ("anyOf", "oneOf"):
        branches = prop.get(key)
        if isinstance(branches, list) and branches:
            out = []
            for b in branches:
                bt = declared_type(b, depth + 1)
                if bt is None:
                    return None
                for x in bt if isinstance(bt, list) else [bt]:
                    if x not in out:
                        out.append(x)
            return out[0] if len(out) == 1 else out
    return None


_SKIP = object()          # #5493: coerce_value's "leave this parameter out"


def _declares(declared, kind: str) -> bool:
    return declared == kind or (isinstance(declared, list) and kind in declared)


def coerce_value(value: str, declared, notes: list | None = None):
    """A non-string parameter's text -> its value by the tool's declared type: JSON when it parses; else an array or an
    object read with lenient_json (a note says so in `notes`); else the text itself, as before.  An array or object that
    arrived as a JSON string of itself is decoded once more.  #5493: a declared non-string parameter written empty or as
    a null placeholder (null / None) is null when its type allows null, else _SKIP (left out, not sent as the text)."""
    if declared == "string":
        return value
    boolean = declared == "boolean" or (isinstance(declared, list) and "boolean" in declared and "string" not in declared)
    if boolean and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"         # #5979: the model's Python spelling (True / False) of a declared boolean is the boolean
    s = value.strip()
    if declared is not None and not _declares(declared, "string") and (not s or s.lower() in ("null", "none")):
        return None if _declares(declared, "null") else _SKIP
    try:
        v = json.loads(value)
    except ValueError:
        if not value.lstrip().startswith(("[", "{", "`")) or _declares(declared, "string"):
            return value
        try:
            v = lenient_json(value)
        except (ValueError, RecursionError):
            return value
        if notes is not None:
            notes.append("a damaged JSON " + ("array" if isinstance(v, list) else "object") + " argument was read as written")
        return v
    if isinstance(v, str) and (_declares(declared, "array") or _declares(declared, "object")) and v.lstrip().startswith(("[", "{")):
        try:
            return json.loads(v)
        except ValueError:
            return coerce_value(v, declared, notes)
    return v


def parse_tool_call(body: str, schema: dict | None = None, notes: list | None = None, recover: bool = False) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")].strip()      # `<function= capture>`: the model's stray space
    rest = body[body.index(">") + 1:]
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    if recover and rest.lstrip().startswith(ARGS_START):     # #6032 R3: the arguments as one JSON object
        got = json_args(rest.lstrip(), props, notes)
        if got is None:
            raise ValueError("malformed tool call: " + body[:80])
        args.update(got[0])
        rest = rest.lstrip()[got[1]:]
    openers = dropped = 0
    while True:
        # #6032 R2 (recover): where a parameter may begin, a damaged opener of a declared parameter not set yet
        # counts (opener_at); anything else is skipped to the next `<parameter=`, as before
        unset = {n for n in props if n not in args} if recover else None
        s = rest.lstrip()
        got = opener_at(s, unset, final=True) if (recover and not s.startswith(PARAM_START)) else None
        if got is None:
            if "<parameter=" not in rest:
                break
            rest = rest[rest.index("<parameter="):]
            if recover:
                got = opener_at(rest, unset, final=True)
            if got is None:                                  # the old reading (a `<parameter=` with no `>` raises)
                got = (param_name(rest[len(PARAM_START):rest.index(">")]), rest.index(">") + 1, None)
            s = rest
        pname, length, inline = got
        openers += 1
        dropped += pname is None
        closer = "</%s>" % pname if recover and pname and s.startswith("<%s>" % pname) else None   # #6032 R4: an element
        rest = s[length:]
        declared = declared_type(props.get(pname)) if pname else None
        if inline is not None:                               # the colon form carries its value on the opener's line
            value = inline
        else:
            # #5493: a close followed by the same name again is part of a STRING value (#4078); a non-string parameter
            # written twice is two writes of it, and the later one counts
            end = param_end(rest, final=True, name=pname, allow_reopen=(declared is None or _declares(declared, "string")),
                            names={n for n in props if n not in args and n != pname} if recover else None, closer=closer)
            value = rest[:end] if end >= 0 else rest
            rest = rest[end + close_len(rest, end, closer):] if end >= 0 else ""
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
        v = coerce_value(value, declared, notes) if pname else _SKIP
        if v is not _SKIP:
            args[pname] = v
    if recover and openers and dropped == openers and not args:
        # #6032 R5: every parameter the model began was unreadable (an opener cut before its `>`, a stub): the call is
        # not run with no arguments at all - it goes back as the text it is
        raise ValueError("malformed tool call: no readable parameter: " + body[:80])
    return ToolCall(name=name, arguments=args)


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False, deferred: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # #4663: the request defers tools (Anthropic's tool search, see anthropic_tools): the client holds back more tools
        # than it declares, so a call for an undeclared name is a call too - Claude Code runs it, where text ended the turn.
        self.deferred = deferred
        # #804/#3075: in the answer too, a call counts only for a tool the request declared.  With tools declared a
        # call for another name stays text (the Responses spelling `namespace__tool` of a `namespace.tool` counts as
        # that tool); `declared_only` extends the rule to a request that declared none (Chat Completions, Anthropic:
        # the client has no tool to run).  The Responses route leaves it off: a Codex compaction request keeps its
        # conversation's tools in the prompt only, and its calls come back as before.
        self.declared_only = declared_only
        # A declared namespace.member tool is also accepted as namespace__member (PR #3556),
        # namespace=member (Qwen writes the <function= tag's equals sign into the name),
        # namespace_member (one underscore dropped), and the bare member when exactly one
        # declared tool has it. Only an unambiguous alias counts, and an exactly declared
        # name always wins over an alias.
        aliases = {}
        for name in self.schemas:
            if isinstance(name, str) and "." in name:
                namespace, member = name.rsplit(".", 1)
                for alias in (f"{namespace}__{member}", f"{namespace}={member}",
                              f"{namespace}_{member}", member):
                    aliases.setdefault(alias, set()).add(name)
        # #3796: the same tool written with every `__` collapsed to `_` (`mcp__playwright__browser_navigate` as
        # `mcp_playwright_browser_navigate`, the way Qwen often flattens a name) is that tool.  Only a spelling of a declared
        # name: a reordered or renamed tool (`server1_label_values` for `label_values_mcp_server1`) is not guessed.
        for name in self.schemas:
            if isinstance(name, str) and "__" in name:
                aliases.setdefault(name.replace("__", "_"), set()).add(name)
                # #5281: a namespaced MCP tool (`mcp__assets__get_asset_info`) written with its last segment only (`get_asset_info`),
                # as the bare member of a dotted name is accepted above; only when exactly one declared tool ends that way
                last = name.rsplit("__", 1)[1]
                if last:
                    aliases.setdefault(last, set()).add(name)
        # #4751: a programmatic-tool-calling harness (the DeepSeek Harness's PTC mode) shows the model its tools as `tools.write(...)`
        # inside code, and the model then writes the native call with that spelling (`tools.write`, also `functions.write`).  The
        # prefix is not part of the declared name; the same unambiguous-alias rule applies, an exactly declared name always wins.
        for name in self.schemas:
            if isinstance(name, str) and name:
                for prefix in ("tools.", "functions.", "default_api."):
                    aliases.setdefault(prefix + name, set()).add(name)
        self.aliases = {alias: next(iter(names)) for alias, names in aliases.items()
                        if len(names) == 1 and alias not in self.schemas}
        self.unread = 0              # #3441: `<tool_call>` blocks that opened like a call and were returned as text
        self.unread_names: list[str] = []   # the names (first few) of those that named a tool the request did not declare
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
        self.notes: list[str] = []   # #4427: what was repaired in the arguments of the calls (a damaged JSON array or object)
        self.after_think = False     # #4077: the reasoning ended at a bare `</think>` (a second one in the answer is a stray)
        self.stray_think = 0         # ... and the strays of that kind dropped from the answer
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        self._prev_blank = self._indented = False   # #4703: the line before was blank / this is inside an indented (4-space) code block
        # recover (on in the server since 0.2.8, off for a bare OutputParser; the config's "tool_call_recovery": false turns it off): a declared tool's call written next to the template's
        # form is the call - `<parameter=NAME>` as the opener, JSON inside <tool_call>, a bare <function=NAME> at the
        # start of a line outside code, or a second call in the same <tool_call> (which otherwise merges into one).
        self.recover = recover
        self.bare = False            # the call being read opened with <function= alone (no <tool_call> around it)
        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._final = False          # #4701: the turn has ended: a call held for what might follow it is decided now
        self._held = ""              # #5276/#5409: the end of a misspelt opener or a prefixed name still arriving (see _prefixed)
        self._reset_scan()

    @property
    def clean(self) -> bool:
        """Nothing is held back: a clean point to stop the thinking at (a parser that rewrites the model's text also counts
        what it holds; #4686: so does a call body moved out of the buffer while it streams)."""
        return not self.buf and not self._held and not self._scan_parts

    def _ok_at(self, p: int) -> bool:
        """self.buf[p] would open a call: at the start of a line, outside a code fence and inline code."""
        snap = self._snap()
        self._track(self.buf[:p])
        ok = self._opener_ok()
        self._restore(snap)
        return ok

    def _opens_at(self, p: int) -> bool:
        """self.buf[p] would open a call as `<tool_call>` does: in the reasoning at the start of a line outside code, in the answer
        anywhere outside code (a fence, inline code, an indented block)."""
        if self.state == "reasoning":
            return self._ok_at(p)
        snap = self._snap()
        self._track(self.buf[:p])
        ok = not self._in_code()
        self._restore(snap)
        return ok

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
                    # #4702: a fence is closed only by a line of at least as many of the same character as opened it (CommonMark), so a
                    # ``` example inside a ```` file stays inside it
                    run = len(s) - len(s.lstrip(self.fence[0]))
                    if run >= len(self.fence):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0] * (len(s) - len(s.lstrip(s[0])))
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
                # #4703: an indented (4 spaces or a tab) code block starts after a blank line and goes on until a line that is not indented
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
        """#4703: the current line is part of an indented code block: four spaces or a tab, after a blank line or another such line."""
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
        self._function_end_checked = 0
        self._scan_parts = []        # consumed call body, kept in chunks until the final call needs the whole text
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> function_done -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}
        self.sheld: dict = {}        # #5493: the non-string parameters, sent when the call closes (a later write wins)
        self.sset: set = set()       # #6032 R2: the parameters set so far (a damaged opener counts only for the others)
        self.snames = None           # ... the declared ones not set yet, for the current value's end (recover only)
        self.scloser = None          # #6032 R4: `</NAME>` when the current value was opened as the element `<NAME>`
        self.spname = None
        self.sraw_pname = ""
        self.svstart = 0

    def _restore_scan_body(self) -> None:
        """Put compacted call text back before a structural walk or the final parse."""
        if not self._scan_parts:
            return
        prefix = "".join(self._scan_parts)
        shift = len(prefix)
        self.buf = prefix + self.buf
        self.sp += shift
        if self.ss == "str" and self.sval_started:
            self.svstart += shift
        if self._call_end_checked >= 0:
            self._call_end_checked += shift
        if self._function_end_checked >= 0:
            self._function_end_checked += shift
        self._scan_parts.clear()

    def _compact_scan_body(self) -> None:
        """Move streamed, consumed call text out of the string that grows on every token."""
        cut = self.sp
        if self.ss == "str" and self.sval_started:
            # the open string value is walked from its start (param_end over buf[svstart:]) each time a closer may have arrived: its text stays in
            # self.buf, so only what lies before it can move out (cutting at sp left svstart negative: buf[svstart:] then read the end of the
            # buffer, and the value's </parameter> was not seen once the scan position passed 4096)
            cut = min(cut, self.svstart)
        if cut < 4096:
            return
        self._scan_parts.append(self.buf[:cut])
        self.buf = self.buf[cut:]
        self.sp -= cut
        if self.ss == "str" and self.sval_started:
            self.svstart -= cut
        if self._call_end_checked >= 0:
            self._call_end_checked = max(0, self._call_end_checked - cut)
        if self._function_end_checked >= 0:
            self._function_end_checked = max(0, self._function_end_checked - cut)

    def _function_end(self) -> int:
        """Run the full function walk only when newly appended text contains a possible closer."""
        start = max(0, self._function_end_checked - len(FUNC_END) + 1)
        if self.buf.find(FUNC_END, start) < 0:
            self._function_end_checked = len(self.buf)
            return -1
        self._restore_scan_body()
        end = function_end(self.buf)
        self._function_end_checked = -1 if end >= 0 else len(self.buf)
        return end

    def _call_names(self):
        """#6032 R4 (recover): the declared parameters of the tool the call in self.buf names, for call_end's walk."""
        if not self.recover:
            return None
        m = re.match(r"\s*<function=([^<>\n]*)>", self.buf)
        if not m:
            return None
        name = m.group(1).strip()
        name = self.aliases.get(name, name)
        props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
        return set(props) or None

    def _wrapped_call_end(self) -> int:
        # Without a closing tag the structural walk cannot finish. Search only
        # newly appended text plus enough overlap for a tag split across feeds.
        # A literal closing tag inside a value needs structural disambiguation:
        # once a candidate appears, restore compacted text for the structural walk. If it was a literal tag in a
        # value, wait for another newly appended candidate instead of walking the unchanged body for every token.
        if self._call_end_checked >= 0:
            start = max(0, self._call_end_checked - len(CALL_END) + 1)
            if self.buf.find(CALL_END, start) < 0:
                self._call_end_checked = len(self.buf)
                return -1
        self._restore_scan_body()
        names = self._call_names()
        end = call_end(self.buf, self._final, names=names)
        # #4701: a complete closing sequence may be waiting for lookahead or finish(), not another closing tag.
        pending = end < 0 and not self._final and call_end(self.buf, final=True, names=names) >= 0
        self._call_end_checked = -1 if end >= 0 or pending else len(self.buf)
        return end

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
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = {k: declared_type(v) for k, v in props.items()}
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                got = None
                if self.recover and (stripped.startswith(ARGS_START) or (stripped and ARGS_START.startswith(stripped))):
                    # #6032 R3: the arguments as one JSON object: held until it is whole, then sent with the call's
                    # other non-string values when it closes (as parse_tool_call reads it); a cut one sends nothing
                    props = ((self.schemas.get(self.scall.name) or {}).get("parameters") or {}).get("properties") or {}
                    whole = json_args(stripped, props) if stripped.startswith(ARGS_START) else None
                    if whole is None:
                        return out
                    for k, v in whole[0].items():
                        self.sheld.pop(k, None)
                        self.sheld[k] = v
                        self.sset.add(k)
                    self.sp += whole[1]
                    continue
                if self.recover and not stripped.startswith(FUNC_END):
                    # #6032 R2: the opener parse_tool_call reads here (a damaged one of a declared parameter not set
                    # yet, or the colon form); other text up to the next `<parameter=` is skipped, as there
                    got = opener_at(stripped, set(self.sdeclared) - self.sset, self._final)
                    if got == "pending":
                        return out
                    if got is None and not stripped.startswith("<parameter="):
                        k = stripped.find("<parameter=")
                        if k < 0:
                            return out          # trailing text, or a tag still arriving: wait
                        self.sp += k
                        continue
                if got is not None or stripped.startswith("<parameter="):
                    if got is None:
                        b = stripped.find(">")
                        if b < 0:
                            return out
                        got = (param_name(stripped[11:b]), b + 1, None)  # #6032 R1: as parse_tool_call names it
                    pname, length, inline = got
                    self.sp += length
                    self.sraw_pname = pname
                    self.spname = pname
                    self.snames = (set(self.sdeclared) - self.sset - {pname}) if self.recover else None
                    # #6032 R4: a value opened as the element `<NAME>` may end at `</NAME>` (as parse_tool_call reads it)
                    self.scloser = "</%s>" % pname if self.recover and pname and stripped.startswith("<%s>" % pname) else None
                    if inline is not None:      # the colon form: its value is on the opener's line
                        declared = self.sdeclared.get(pname) if pname else None
                        v = coerce_value(inline, declared) if pname else _SKIP
                        if v is not _SKIP and declared == "string":
                            args(("" if self.sfirst else ",") + json.dumps(pname) + ":" + json.dumps(v))
                            self.sfirst = False
                            self.sset.add(pname)
                        elif v is not _SKIP:
                            self.sheld.pop(pname, None)
                            self.sheld[pname] = v
                            self.sset.add(pname)
                        continue
                    if pname is not None and self.sdeclared.get(pname) == "string":
                        self.sset.add(pname)
                        args(("" if self.sfirst else ",") + json.dumps(pname) + ':"')
                        self.sfirst = False
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
                closers = (PARAM_END, self.scloser) if self.scloser else (PARAM_END,)
                end = param_end(self.buf[self.svstart:], self._final, name=self.spname, allow_reopen=True,
                                names=self.snames, closer=self.scloser) if any(c in rest for c in closers) else -1
                end = end - (self.sp - self.svstart) if end >= 0 else end
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + close_len(rest, end, self.scloser)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving
                safe = min(k for k in (rest.find(c) for c in closers) if k >= 0) if end == -2 \
                    else len(rest) - self._hold(rest, closers)
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                declared = self.sdeclared.get(self.sraw_pname)
                end = param_end(rest, self._final, name=self.spname,         # the rule of parse_tool_call (#5493)
                                allow_reopen=(declared is None or _declares(declared, "string")), names=self.snames,
                                closer=self.scloser)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                v = coerce_value(value, declared) if self.sraw_pname is not None else _SKIP   # the note comes with the final call
                if v is not _SKIP:
                    # #5493: held until the call closes, so a parameter the model writes again (anywhere later in the
                    # call) streams once, with the value the final call has (the later write), not as two JSON keys
                    self.sheld.pop(self.sraw_pname, None)
                    self.sheld[self.sraw_pname] = v
                    self.sset.add(self.sraw_pname)
                self.sp += end + close_len(rest, end, self.scloser)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """Close streamed JSON only after its call boundary has been confirmed."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw" and self.sraw_pname is not None:   # a value the output ended inside: its text, as before
            self.sheld.pop(self.sraw_pname, None)
            self.sheld[self.sraw_pname] = self.buf[self.sp:].strip("\n")
        for k, v in self.sheld.items():         # #5493: the non-string parameters, each once, the later write
            tail += ("" if self.sfirst else ",") + json.dumps(k) + ":" + json.dumps(v, ensure_ascii=False)
            self.sfirst = False
        self.sheld = {}
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

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

    def _prefixed(self, fresh: int) -> str:
        """#5409: a call written `<function=tools NAME>` (a space for the dot of `tools.NAME`) is the call NAME, and one written
        `<function=tools>` (the bare prefix) is the one declared tool that takes the call's parameters.  Rewrites the complete
        ones in self.buf to `<function=NAME>` and returns the tail of a call still arriving, which feed() holds back until the next
        delta.  Only a declared name is ever the result; with none, or with two that fit, the text is left as it was.
        `fresh`: the length of the text just added; only the end of the buffer (that and a little before it) can hold a new match,
        so a long call is not scanned again with every token."""
        held = self._opener(fresh)
        start = max(0, len(self.buf) - fresh - _PREFIX_LOOKBACK)
        buf = self.buf[start:]
        if "<function=" in buf:
            def spaced(m):
                name = m.group(2).strip()
                return f"<function={name}>" if name in self.schemas else m.group(0)
            buf = _PREFIX_SPACED.sub(spaced, buf)
            m = _PREFIX_PART.search(buf)
            if m and not held:
                held, buf = buf[m.start():], buf[:m.start()]
            if self.recover:
                def bare(m):
                    body = m.group(3)
                    names = re.findall(r"<parameter=\s*([^<>\s]+?)\s*>", body)
                    tool = None if m.group(2) in self.schemas else self._tool_by_params(names)
                    return f"{m.group(1)}<function={tool}>{body}</function>" if tool else m.group(0)
                buf = _PREFIX_BARE.sub(bare, buf)
                m = _PREFIX_BARE_OPEN.search(buf)
                if m and m.group(1) not in self.schemas and not held and len(buf) - m.start() <= RCALL_MAX:
                    held, buf = buf[m.start():], buf[:m.start()]
            if buf != self.buf[start:]:
                self.buf = self.buf[:start] + buf
        return held

    def _opener(self, fresh: int) -> str:
        """#5276: `<toolcall>` / `<tool-call>` (the opener written as text, misspelled) followed by `<function=NAME>` of a declared
        tool is `<tool_call>`, where a `<tool_call>` would open a call (in the reasoning the start of a line, outside code; in the
        answer outside code).  Prose that names the misspelled tag, or any other follower, is left as it was.  Returns the tail of
        an opener still arriving (held back by feed)."""
        start = max(0, len(self.buf) - fresh - _PREFIX_LOOKBACK)
        if "<tool" not in self.buf[start:]:
            return ""
        buf = self.buf
        pos, out = start, []
        for m in _MISSPELT_OPEN.finditer(buf, start):
            name = m.group(2).strip()
            name = self.aliases.get(name, name)
            if name in self.schemas and self._opens_at(m.start()):
                out.append(buf[pos:m.start()] + CALL_START + m.group(1) + FUNC_START + m.group(2) + ">")
                pos = m.end()
        buf = buf[:start] + "".join(out) + buf[pos:] if out else buf
        held = ""
        m = _MISSPELT_PART.search(buf, start)
        tail = buf[m.start():] if m else ""
        if not tail:                                        # the opener itself still arriving: a prefix of `<toolcall>`
            for k in range(min(len(buf), 11), 5, -1):
                if any(w.startswith(buf[-k:]) for w in ("<toolcall>", "<tool-call>")) and buf[-k:] != "<tool_":
                    tail = buf[-k:]
                    break
        if tail:
            held, buf = tail, buf[:len(buf) - len(tail)]
        self.buf = buf
        return held

    def _tool_by_params(self, names: list[str]) -> str | None:
        """The one declared tool whose properties hold every one of `names` and whose required parameters are all in them."""
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
            self._held = self._prefixed(len(delta))
        return self._feed()

    def _feed(self) -> list[Event]:
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                body = self.buf[len(CALL_START):]
                end, think = call_end(body, hold=False), body.find(THINK_END)
                drop = len(CALL_END)
                f = function_end(body)               # #4427: as in the answer, a following call opener confirms an omitted closer
                if f >= 0 and (end < 0 or f < end) and re.match(r"<tool_call>\s*<function=\s*[^<>\s]+\s*>", body[f:].lstrip()):
                    end, drop = f, 0
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = body[:end].strip()[len("<function="):].split(">", 1)[0].strip()
                        name = self.aliases.get(name, name)
                        if self._callable(name):
                            call = parse_tool_call(body[:end], self.schemas.get(name), self.notes, recover=self.recover)
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
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding the thinking back
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
                self._reset_track()                               # the answer's own text starts here
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
                    j = len(self.buf) - self._hold(self.buf, (CALL_START, THINK_END) if self.after_think else (CALL_START,))
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
                if self.recover and self.schemas and after.startswith("<i"):
                    # #5969 (recover): Claude's `<invoke name="NAME">` dialect for a declared tool, its opener ending its
                    # line: held whole as the "invoke" state reads it (a cut one, an undeclared name, or prose that only
                    # names the tag - more text on the opener's line - stays the text it is)
                    m = _INVOKE_OPEN.match(after)
                    line_end = after[m.end():].lstrip(" \t")[:1] if m else ""
                    if m and not line_end:
                        return out                       # what follows the opener decides
                    if m and line_end == "\n" and self._callable(m.group(1)):
                        if i and self.buf[:i].strip():
                            out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                        self.buf = self.buf[i + len(CALL_START):]
                        self.state = "invoke"
                        continue
                    if not m and _INVOKE_PART.match(after) and "\n" not in after:
                        j = i
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
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
                cut = re.match(r"<function=([^<>\s]+)[ \t]*\n", after) if self.recover else None
                if cut and self.aliases.get(cut.group(1), cut.group(1)) in self.schemas:
                    # #6032 R6 (recover): `<function=NAME` + line break - the `>` left out - of a declared tool
                    at = len(self.buf) - len(after)
                    self.buf = self.buf[:at] + FUNC_START + cut.group(1) + ">\n" + after[cut.end():]
                    continue
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
                        self.unread += 1                 # logged by the server: otherwise "the model ignored my tool" is the only clue
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
            elif self.state == "invoke":
                # #5969 (recover): `<invoke name="T">` + `<parameter name="K">V</parameter>` ... `</invoke>`, its
                # `</tool_call>` after it or left out at the end of the output: read in the template's form; a body that
                # does not read, or one the output ends inside, is the text it is
                m = _INVOKE_END.search(self.buf)
                if m is None:
                    return out                          # not whole yet; at the end, finish() returns it as text
                rest = self.buf[m.end():]
                tail = rest.lstrip()
                if not tail and not self._final:
                    return out                          # its `</tool_call>` may still come
                if tail and CALL_END.startswith(tail) and not self._final:
                    return out
                drop = len(rest) - len(tail) + len(CALL_END) if tail.startswith(CALL_END) else 0
                raw = self.buf[:m.end()]
                call = None
                try:
                    name = _INVOKE_OPEN.match(raw.lstrip()).group(1)
                    name = self.aliases.get(name, name)
                    call = parse_tool_call(invoke_to_template(raw), self.schemas.get(name), self.notes, recover=True)
                    call.name = name
                except (ValueError, AttributeError):
                    call = None
                if call is None:
                    out.append(Event("content", self._track(CALL_START + raw + (CALL_END if drop else ""))))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[m.end() + drop:]
                self.state, self.lead = "content", call is not None
            else:
                drop = None
                if self.bare:                       # no wrapper: the call ends at its </function>; a stray
                    i = self._function_end()        # </tool_call> after it is the wrapper's closer, dropped
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
                    f = self._function_end()
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
                        final, self._final = self._final, True       # #5040: its end is known, so no closing tag is still arriving
                        out += self._scan()
                        out += self._close_scan()
                        self.buf, self._final = whole, final
                    else:
                        out += self._scan()
                if i < 0 or drop is None:
                    if self.stream_tools:
                        self._compact_scan_body()
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
        self._restore_scan_body()               # #4686: the call body moved out while it streamed comes back first
        if self._held:
            self.buf, self._held = self.buf + self._held, ""
        if self.buf and self.state in ("call", "rcall", "invoke"):
            self._final = True                  # #4701: a call held after its closing tags (could a file's text go on?) is whole now
            out += self.feed("")
        # The final feed may compact a still-incomplete streamed call again.
        self._restore_scan_body()
        if self.state == "rcall" and reason in (None, "stop"):
            # #4427: a call in the reasoning whose `</function>` is the end of the output (the `</tool_call>` left out)
            # is as whole as one in the answer: it waits for the turn's end like the others
            body = self.buf[len(CALL_START):]
            f = function_end(body)
            name = body.lstrip()[len(FUNC_START):].split(">", 1)[0].strip() if f >= 0 else ""
            name = self.aliases.get(name, name)
            if f >= 0 and not body[f:].strip() and self._callable(name):
                try:
                    call = parse_tool_call(body[:f], self.schemas.get(name), self.notes, recover=self.recover)
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
            if self.ss == "function_done" and not self.buf[self.sp:].strip() and reason in (None, "stop"):
                # A natural end immediately after </function> retains native streaming recovery.
                out += self._close_scan()
                call = self.scall
                try:            # the final event carries the call's arguments, as the one _finish_call sends does
                    call = parse_tool_call(self.buf, self.schemas.get(call.name), self.notes, recover=self.recover)
                    call.name, call.id = self.scall.name, self.scall.id
                except ValueError as e:
                    if "no readable parameter" in str(e):
                        call = None             # #6032 R5: never run with no arguments (it stays announced, unfinished)
                if call is not None:
                    out.append(Event("tool_call", call=call))
            elif self.ss == "function_done":
                out.append(Event("content", self._track(CALL_START + self.buf)))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "reasoning" and self.buf == THINK_END:   # the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + self.buf if self.state in ("call", "json", "invoke") and not self.bare else self.buf
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
            call = parse_tool_call(body, self.schemas.get(name), self.notes, recover=self.recover)
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


class JsonOutputParser(OutputParser):
    """The parser for a model whose chat template tells it to write each call as JSON inside the wrapper:
    `<tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>` (Qwen3, Hermes).  OutputParser already reads that form, but only
    as a recovery for a model that drifted off the XML form; for a JSON template it is the form itself, so it is read whether or
    not the server's tool_call_recovery is on.  Everything else (reasoning, code fences, declared-tool rules) is OutputParser's."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False, declared_only: bool = False, deferred: bool = False):
        super().__init__(thinking=thinking, tools=tools, stream_tools=stream_tools, recover=True,
                         declared_only=declared_only, deferred=deferred)


def parser_for_template(source: str | None):
    """The OutputParser class for a chat template's text, by the tool-call form the template documents (serve/toolformat.py):
    the JSON form gets JsonOutputParser; the XML form every Strata model ships with, templates without tools, and forms nothing
    here reads keep OutputParser (the same class as before this existed)."""
    from serve import toolformat
    return JsonOutputParser if toolformat.tool_format(source or "") == "json" else OutputParser
