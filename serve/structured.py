# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""JSON response formats at the HTTP boundary: prompt once, validate before delivery.

An engine with grammar-constrained decoding (its INFO grammar=1) writes the answer under the schema
(answer_grammar, tool_grammar below); either way the answer is checked before delivery.  Failed generations are
errors, never silently retried or returned as successful structured output.

json_schema and a strict tool's arguments are checked against their schema with `jsonschema` (requirements.txt; the
deb and rpm packages depend on the distribution's): json_object needs only the standard library.  Without the
package a strict request (json_schema's strict: true, Anthropic's format, a strict tool) is a 400, as its answer
could not be checked; any other json_schema is checked to be one JSON object, which the server says once.
jsonschema without `referencing` (before 4.18: Ubuntu 24.04's 4.10) checks the same; remote references are refused
either way.
"""
import json
import threading


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


class _Repairing:
    """A validator that also tells validated_json to retry once, after a strict parse failed, with the backslashes that
    are not valid JSON escapes doubled (response_format "repair_latex_escapes": true; upstream 2cc6ea169, #2893)."""
    repair_escapes = True

    def __init__(self, inner):
        self.inner = inner

    def iter_errors(self, value):
        return self.inner.iter_errors(value)


_HEX = "0123456789abcdefABCDEF"


def repair_latex_escapes(text: str) -> str:
    r"""Double every backslash that does not start a valid JSON escape, so \alpha and \gamma read as text.
    The valid escapes (\b \f \n \r \t \/ \" and a well-formed \uXXXX) stay as they are: they cannot be told
    from LaTeX, so a \beta next to a \alpha comes back with a backspace.  Pairs of backslashes are skipped."""
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in ("\\", '"', "/", "b", "f", "n", "r", "t"):
            out.append(ch + nxt)
            i += 2
        elif nxt == "u" and n >= i + 6 and all(c in _HEX for c in text[i + 2:i + 6]):
            out.append(text[i:i + 6])
            i += 6
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


_jsonschema = None   # (validators, SchemaError, Registry, NoSuchResource; the last two None before 4.18), False: none
_jsonschema_lock = threading.Lock()


NO_JSONSCHEMA = ("strict: true needs the Python package jsonschema on the server to check the reply against the "
                 "schema, and it is not installed (the distribution's python3-jsonschema, or ./setup.sh again)")


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import validators
                from jsonschema.exceptions import SchemaError
            except ImportError:
                _jsonschema = False
                print("[xestrata] json_schema and strict tools: the Python package jsonschema is not installed, so "
                      "strict requests are refused and other json_schema answers are only checked to be one JSON "
                      "object (the distribution's python3-jsonschema, or ./setup.sh again)", flush=True)
            else:
                try:
                    from referencing import Registry
                    from referencing.exceptions import NoSuchResource
                except ImportError:                 # jsonschema before 4.18: its own resolver, local references only
                    Registry = NoSuchResource = None
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
        return _jsonschema or None


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def schema_validator(schema: dict, modules, what: str):
    """A validator of `schema` with jsonschema's `modules` (jsonschema_modules), or one that checks for one JSON
    object without them.  A reference that is not local (#...) - nothing is fetched - or a schema jsonschema refuses
    is a ValueError (a 400) naming `what`."""

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError(f"{what} supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    if modules is None:
        return _ObjectOnly()
    validators, SchemaError, Registry, NoSuchResource = modules

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    try:
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        return cls(schema, registry=Registry(retrieve=no_remote)) if Registry is not None else cls(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid {what} schema: {exc.message}") from exc


def prepare_format(response_format, messages, with_tools=False):
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
        modules = None                 # the standard library is enough for "one JSON object"
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise ValueError("response_format.json_schema needs a name")
        if spec.get("strict") is not None and not isinstance(spec["strict"], bool):   # null: not given (OpenAI's SDKs)
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if not _only_objects(schema, schema):
            raise ValueError("response_format schema must accept only JSON objects at its root "
                             "(type object, or anyOf/oneOf of object schemas)")
        modules = jsonschema_modules()
        if modules is None and spec.get("strict") is True:
            raise ValueError(NO_JSONSCHEMA)
    else:
        raise ValueError("response_format.type must be text, json_object or json_schema")

    validator = schema_validator(schema, modules, "response_format")
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if with_tools:
        # /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not an answer
        directive += ("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                      "what you write once you are done with the tools.")
    if response_format.get("repair_latex_escapes") is True:
        validator = _Repairing(validator)
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def extract_json(text: str) -> str:
    """The first JSON object in `text`: a ```-fenced block's inside when there is one, else the text from its first
    `{`.  Empty when there is nothing to take.  The server's json_schema logprobs read the model's own JSON with it."""
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); "
                                    "increase the output budget")
    try:
        value = json.loads(extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        value = None
        if getattr(validator, "repair_escapes", False):      # opt-in (#2893); only after the strict parse failed
            try:
                value = json.loads(repair_latex_escapes(extract_json(text)), object_pairs_hook=pairs,
                                   parse_constant=constant)
                canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                print("[xestrata] structured output: repaired backslashes that were not valid JSON escapes", flush=True)
            except (ValueError, TypeError):
                value = None
        if value is None:
            raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical


# ---- grammar-constrained decoding (the engine's grammar=, src/core/grammar.cpp; xgrammar's structural tags)
# The thinking (up to </think>) is never constrained: a request that thinks gets the grammar after it.
_THINKING = {"type": "tag", "begin": "", "content": {"type": "any_text"}, "end": "</think>"}


def _structural(fmt: dict) -> dict:
    return {"kind": "structural_tag", "text": json.dumps({"type": "structural_tag", "format": fmt}, ensure_ascii=False)}


def answer_grammar(response_format, thinking: bool):
    """A json_schema / json_object response_format (prepare_format's checks done) -> the engine's grammar for the
    answer, or None for text.  After the thinking, the template's blank line, then the JSON."""
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        schema = response_format["json_schema"]["schema"]
    else:
        return None
    if not thinking:
        return {"kind": "json_schema", "text": json.dumps(schema, ensure_ascii=False)}
    return _structural({"type": "sequence", "elements": [_THINKING, {"type": "const_string", "value": "\n\n"},
                                                         {"type": "json_schema", "json_schema": schema}]})


def tool_grammar(tools, strict: set, patterns: dict, thinking: bool, form: str):
    """The calls of a request with strict tools (`strict`: their names) or custom tools with a regex (`patterns`:
    name -> pattern), in Qwen's call form (<tool_call><function=NAME><parameter=..>): a call opened with one of the
    tools' names must follow that tool's parameters; text and the other tools' calls stay free.  None without such a
    tool, or for another call form (GLM's, Harmony's), which is checked after the reply as before."""
    if form != "qwen" or not (strict or patterns):
        return None
    tags = []
    for t in tools or []:
        name = t.get("name")
        if name in strict:
            content = {"type": "json_schema", "json_schema": t.get("parameters") or {"type": "object"},
                       "style": "qwen_xml"}
        elif name in patterns:
            content = {"type": "sequence", "elements": [{"type": "const_string", "value": "<parameter=input>\n"},
                                                        {"type": "regex", "pattern": patterns[name]},
                                                        {"type": "const_string", "value": "\n</parameter>"}]}
        else:
            content = {"type": "any_text"}
        tags.append({"type": "tag", "begin": f"<tool_call>\n<function={name}>\n", "content": content,
                     "end": "\n</function>\n</tool_call>"})
    calls = {"type": "triggered_tags", "triggers": ["<tool_call>\n<function="], "tags": tags, "excludes": [],
             "at_least_one": False, "stop_after_first": False}
    return _structural({"type": "sequence", "elements": [_THINKING, calls]} if thinking else calls)


def call_names_grammar(names, thinking: bool, form: str):
    """A forced call among several tools (the web page's image button, xestrata_images.force): the server writes the
    call's opening, and this keeps its name to one of `names` - else the model chose one of its own after
    "<function=" (a memory tool it knew), which is no declared tool and went back as text, and no image was made.
    In Qwen's call form; None in another (GLM's, Harmony's) or without names."""
    if form != "qwen" or not names:
        return None
    tags = [{"type": "tag", "begin": f"<tool_call>\n<function={name}>\n", "content": {"type": "any_text"},
             "end": "\n</function>\n</tool_call>"} for name in names]
    calls = {"type": "triggered_tags", "triggers": ["<tool_call>\n<function="], "tags": tags, "excludes": [],
             "at_least_one": False, "stop_after_first": False}
    return _structural({"type": "sequence", "elements": [_THINKING, calls]} if thinking else calls)


def constrained_tools(tools, namespace: str = "") -> tuple[set, dict]:
    """A request's tools as sent (Chat's, Responses', Anthropic's shapes; a Responses namespace's tools under
    "namespace.name", the name the template has) -> (the names of those with "strict": true, {name: pattern} of the
    custom tools whose format is a regex grammar) for tool_grammar and checked_calls.  A Lark grammar is not
    constrained (it is described to the model, as before).  A strict that is not true, false or null is a 400."""
    strict, patterns = set(), {}
    for t in tools if isinstance(tools, list) else []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "namespace" and isinstance(t.get("name"), str):
            inner_strict, inner_patterns = constrained_tools(t.get("tools"), t["name"] + ".")
            strict |= inner_strict
            patterns.update(inner_patterns)
            continue
        inner = t.get("function")
        fn: dict = inner if isinstance(inner, dict) else t
        if fn.get("strict") is not None and not isinstance(fn["strict"], bool):   # null: not given
            raise ValueError(f"tool {fn.get('name')}: strict must be true, false or null")
        if fn.get("strict") is True and isinstance(fn.get("name"), str):
            strict.add(namespace + fn["name"])
        inner = t.get("custom")
        custom: dict = inner if isinstance(inner, dict) else t if t.get("type") == "custom" else {}
        inner = custom.get("format")
        fmt: dict = inner if isinstance(inner, dict) else {}
        inner = fmt.get("grammar")
        g: dict = inner if isinstance(inner, dict) else fmt
        name = custom.get("name")
        if fmt.get("type") == "grammar" and g.get("syntax") == "regex" and isinstance(g.get("definition"), str) \
                and isinstance(name, str):
            patterns[namespace + name] = g["definition"]
    return strict, patterns


def strict_schemas(tools, strict: set) -> dict:
    """The template's tools (name, parameters) -> {name: parameters} of the strict ones (constrained_tools), which
    the server checks the calls against (checked_calls); a schema jsonschema refuses, or no jsonschema to check
    with, is a 400 now."""
    out = {t["name"]: t.get("parameters") or {"type": "object"} for t in tools or []
           if isinstance(t, dict) and t.get("name") in strict}
    modules = jsonschema_modules() if out else None
    if out and modules is None:
        raise ValueError(NO_JSONSCHEMA)
    for name, schema in out.items():
        schema_validator(schema, modules, f"tool {name}'s parameters")
    return out


def checked_calls(events, schemas: dict):
    """A run's ("event", Event) items with the calls of strict tools (`schemas`: strict_schemas) checked against
    their parameters, and for numbers JSON cannot hold, before any of them is sent: such a call is not announced while
    it is written (its tool_start and tool_args are held back) and goes out whole with its tool_call, or is a
    StructuredOutputError.  This holds
    whether the engine wrote the call under a grammar (tool_grammar) or not (another call form, an engine without
    grammars)."""
    modules = jsonschema_modules()
    checks = {name: schema_validator(schema, modules, f"tool {name}'s parameters") for name, schema in schemas.items()}
    for kind, x in events:
        call = getattr(x, "call", None) if kind == "event" else None
        if call is not None and call.name in checks:
            if x.kind in ("tool_start", "tool_args"):
                continue
            if x.kind == "tool_call":
                try:      # NaN or Infinity (1e999 reads as it): no JSON, and jsonschema's "number" takes them
                    json.dumps(call.arguments, allow_nan=False)
                except ValueError:
                    raise StructuredOutputError(f"the call of strict tool {call.name} has a number that is not "
                                                "finite (NaN or Infinity), which JSON cannot hold") from None
                try:
                    error = next(checks[call.name].iter_errors(call.arguments), None)
                except Exception as exc:
                    raise StructuredOutputError(f"could not validate the call of {call.name}: {exc}") from exc
                if error is not None:
                    path = "/" + "/".join(str(part) for part in error.absolute_path)
                    raise StructuredOutputError(f"the call of strict tool {call.name} failed its parameters' "
                                                f"JSON Schema at {path}: {error.message}")
        yield kind, x
