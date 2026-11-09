# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/translate.py - translation with PLaMo 3 Translate: its prompt (PFML3) and the translation APIs' requests.

PFML3 is built here and nowhere else.  `pfml3` renders a chat the way the model's GGUF chat template does (the
template llama.cpp and vLLM render; tools/test_translate_pfml.py holds the two to the same text): leading system
messages carry a `[plamo-translate]` settings block (target_lang=, source_lang=, form=, glossary=A => B, one a line,
ended by `---`) and reference context, the keyword arguments (chat_template_kwargs) win key by key, the user and
assistant messages after them alternate source texts and their translations, and the last user message is translated.
Without a target language the model's own switch is rendered: Japanese and English into each other.  A last user
message that is already PFML3 is passed through after the BOS.
"""
from __future__ import annotations

BOS = "<|plamo:bos|>"
RAW_PREFIX = "<|plamo:tag|>dataset<|plamo:msg|>translation"
_WS = " \t\r\n"


class PfmlError(ValueError):
    """A request the template would refuse (raise_exception), with the template's message."""


def _check_text(value, name: str, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise PfmlError(name + " must be a string.")
    if not allow_empty and not value.strip(_WS):
        raise PfmlError(name + " must not be empty.")
    if "<|plamo:" in value:
        raise PfmlError(name + " must not contain reserved PFML tokens (<|plamo:...).")


def _check_lang(value, name: str) -> None:
    _check_text(value, name)
    parts = value.split("-")
    if len(parts) > 2 or len(parts[0]) not in (2, 3) or (len(parts) == 2 and len(parts[1]) != 4):
        raise PfmlError(name + " must be a lowercase language code, e.g. en, ja, zh-hans, or ja-easy.")
    if any(c not in "abcdefghijklmnopqrstuvwxyz-" for c in value):
        raise PfmlError(name + " must use lowercase language codes without region subtags.")


def _ascii_lower(value: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in value)


def _calls(m: dict) -> bool:
    return bool(m.get("tool_calls") or m.get("tool_call_id") or m.get("function_call"))


_UNSET = object()


def pfml3(messages: list, target_lang=_UNSET, source_lang=_UNSET, form=_UNSET, context=_UNSET, glossary=_UNSET,
          tools=None, documents=None, bos: str = BOS) -> str:
    """The prompt the model's chat template renders for `messages` (OpenAI chat messages whose content is a string)
    with these chat_template_kwargs (left out: not given).  Raises PfmlError where the template raises."""
    last = messages[-1] if messages else None
    raw = last.get("content") if isinstance(last, dict) and last.get("role") == "user" else None
    raw = raw if isinstance(raw, str) else ""
    if raw.startswith(RAW_PREFIX):
        if tools or documents or _calls(last):
            raise PfmlError("Tool calls and documents are not supported.")
        return bos + raw
    sys_count = 0
    s_target = s_source = s_form = None
    s_context, has_context = "", False
    s_glossary: list = []
    has_glossary = False
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise PfmlError("Each message must contain a role and text content.")
        if not (i == sys_count and message.get("role") == "system"):
            continue
        if _calls(message):
            raise PfmlError("Tool calls are not supported.")
        content = message.get("content")
        _check_text(content, "system message content", allow_empty=True)
        lines = content.split("\n")
        first = lines[0].strip(" \t\r")
        settings, inline = False, []
        if first.lower().startswith("[plamo-translate"):
            if first == "[plamo-translate]":
                settings = True
            elif first.startswith("[plamo-translate] ") or first.startswith("[plamo-translate]\t"):
                settings = True
                for token in first[17:].replace("\t", " ").split(" "):
                    if not token:
                        continue
                    if token == "---":
                        raise PfmlError("Put --- on its own line to start the context.")
                    if _ascii_lower(token.split("=")[0]) not in ("target_lang", "source_lang", "form"):
                        raise PfmlError("Only target_lang, source_lang and form may follow [plamo-translate] on the "
                                        "same line; put glossary lines below it: " + token)
                    inline.append(token)
            else:
                raise PfmlError("The settings block must start with [plamo-translate], optionally followed by "
                                "key=value settings on the same line.")
        in_settings, body, started = settings, "", False
        for k, raw_line in enumerate(lines[:1] + inline + lines[1:]):
            if in_settings and k > 0:
                line = raw_line.strip(" \t\r")
                if line == "---":
                    in_settings = False
                elif line:
                    parts = line.split("=")
                    key = _ascii_lower(parts[0].strip(" \t"))
                    if key == "glossary":
                        pair = "=".join(parts[1:]).split("=>")
                        if len(pair) != 2 or not pair[0].strip(" \t") or not pair[1].strip(" \t"):
                            raise PfmlError("Each glossary line must be glossary=source term => target term: " + line)
                        _check_text(pair[0].strip(" \t"), "glossary source")
                        _check_text(pair[1].strip(" \t"), "glossary target")
                        s_glossary.append({"source": pair[0].strip(" \t"), "target": pair[1].strip(" \t")})
                        has_glossary = True
                    else:
                        if len(parts) != 2:
                            raise PfmlError("Each line of the [plamo-translate] block must be key=value: " + line)
                        value = _ascii_lower(parts[1].strip(" \t"))
                        if key not in ("target_lang", "source_lang", "form"):
                            raise PfmlError("Unknown setting in the [plamo-translate] block: " + key)
                        if not value:
                            raise PfmlError("Empty value in the [plamo-translate] block: " + key)
                        if key == "target_lang":
                            if s_target is not None:
                                raise PfmlError("target_lang is set more than once in the [plamo-translate] block.")
                            s_target = "ja-easy" if value == "ja(easy)" else value
                        elif key == "source_lang":
                            if s_source is not None:
                                raise PfmlError("source_lang is set more than once in the [plamo-translate] block.")
                            s_source = "ja-easy" if value == "ja(easy)" else value
                        else:
                            if s_form is not None:
                                raise PfmlError("form is set more than once in the [plamo-translate] block.")
                            s_form = value
            elif not (k == 0 and settings):
                body = body + ("\n" if started else "") + raw_line.rstrip("\r")
                started = True
        if body.strip(_WS):
            s_context = s_context + ("\n\n" if has_context else "") + body
            has_context = True
        sys_count += 1
    translation = messages[sys_count:]
    if s_target is not None and s_target != "auto":
        _check_lang(s_target, "target_lang")
    if s_source is not None:
        _check_lang(s_source, "source_lang")
    if s_form is not None and s_form not in ("polite", "plain"):
        raise PfmlError("form must be polite or plain.")
    if target_lang is _UNSET:
        target_lang = s_target if s_target is not None else "auto"
    if source_lang is _UNSET:
        source_lang = s_source
    if target_lang == "ja(easy)":
        target_lang = "ja-easy"
    if source_lang == "ja(easy)":
        source_lang = "ja-easy"
    if form is _UNSET:
        form = s_form
    if context is _UNSET:
        context = s_context if has_context else None
    if glossary is _UNSET:
        glossary = s_glossary if has_glossary else None
    if len(translation) == 0 or len(translation) % 2 != 1:
        raise PfmlError("Provide source/translation pairs followed by one user message.")
    for i, message in enumerate(translation):
        if not isinstance(message, dict):
            raise PfmlError("Each message must contain a role and text content.")
        if message.get("role") != ("user" if i % 2 == 0 else "assistant"):
            raise PfmlError("Alternate user source text and assistant translation; end with user.")
        if _calls(message):
            raise PfmlError("Tool calls are not supported.")
        _check_text(message.get("content"), "message content")
    if tools or documents:
        raise PfmlError("Tool calls and documents are not supported; use context for reference text.")
    auto = target_lang == "auto"
    if auto:
        if source_lang is not None:
            raise PfmlError("source_lang requires a concrete target_lang such as ja or en (not auto); omit "
                            "source_lang to switch between Japanese and English.")
        source_lang = "ja/en"
    else:
        _check_lang(target_lang, "target_lang")
        if source_lang is not None:
            _check_lang(source_lang, "source_lang")
    if context is not None:
        _check_text(context, "context", allow_empty=True)
    if form is not None and (form not in ("polite", "plain") or target_lang != "ja"):
        raise PfmlError('form must be polite or plain and requires target_lang="ja".')
    if glossary is not None:
        if not isinstance(glossary, (list, tuple)):
            raise PfmlError("glossary must be a list of objects with source and target strings.")
        for term in glossary:
            if not isinstance(term, dict) or len(term) != 2 or "source" not in term or "target" not in term:
                raise PfmlError("Each glossary entry must contain exactly source and target.")
            _check_text(term["source"], "glossary source")
            _check_text(term["target"], "glossary target")
    out = [bos, RAW_PREFIX]
    if context:
        out.append("<|plamo:tag|>context<|plamo:msg|>" + context)
    for message in translation[:-1]:
        if message["role"] == "user":
            out.append("<|plamo:tag|>input")
            if source_lang is not None:
                out.append("<|plamo:key|>lang<|plamo:val|>" + source_lang)
        else:
            out.append("<|plamo:tag|>output")
            if not auto:
                out.append("<|plamo:key|>lang<|plamo:val|>" + target_lang)
        out.append("<|plamo:msg|>" + message["content"])
    if glossary:
        out.append("<|plamo:tag|>glossary")
        for term in glossary:
            out.append("<|plamo:index|>" + term["source"] + "<|plamo:item|>" + term["target"])
    out.append("<|plamo:tag|>input")
    if source_lang is not None:
        out.append("<|plamo:key|>lang<|plamo:val|>" + source_lang)
    out.append("<|plamo:msg|>" + translation[-1]["content"] + "<|plamo:tag|>output")
    if not auto:
        out.append("<|plamo:key|>lang<|plamo:val|>" + target_lang)
    if form is not None:
        out.append("<|plamo:key|>form<|plamo:val|>" + form)
    out.append("<|plamo:msg|>")
    return "".join(out)
