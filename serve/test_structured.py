# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Exercise the JSON contract through HTTP, including H3's strict Story shape."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import structured
from serve.server import ByteTokenizer, MockEngine, Service, serve

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:                  # optional: json_schema answers are then only checked to be one object
    HAVE_JSONSCHEMA = False


STORY = {"title": "Photobooth", "sheet": "A single continuous shot.", "cast": [],
         "boxes": [{"box": 1, "duration_ms": 15000, "new_shot": True, "beat": "Hailee changes poses.",
                    "on_screen": ["Hailee"], "references": [], "ends": "She smiles."}], "suggestions": []}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
STORY_SCHEMA = object_schema({
    "title": TEXT, "sheet": TEXT,
    "cast": {"type": "array", "items": object_schema({"name": TEXT,
                                                      "references": {"type": "array", "items": TEXT, "maxItems": 0}})},
    "boxes": {"type": "array", "minItems": 1, "maxItems": 1, "items": object_schema({
        "box": {"type": "integer", "minimum": 1, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 1, "maximum": 15000}, "new_shot": {"type": "boolean"},
        "beat": TEXT, "on_screen": {"type": "array", "items": TEXT},
        "references": {"type": "array", "items": TEXT, "maxItems": 0}, "ends": TEXT})},
    "suggestions": {"type": "array", "items": object_schema({
        "what": TEXT, "why": TEXT, "boxes": {"type": "array", "items": {"type": "integer"}}, "description": TEXT})}})
FORMAT = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": True, "schema": STORY_SCHEMA}}


class Structured(unittest.TestCase):
    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            stream = "event-stream" in response.headers.get("Content-Type", "")
            return response.status, data if stream else json.loads(data)

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script=None, **kwargs):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": FORMAT, "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", req)

    def test_h3_schema_success_and_stream_with_reasoning(self):
        code, reply = self.chat()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", prompt)
        self.assertIn('"duration_ms"', prompt)
        code, raw = self.chat("Consider the shot.</think>\n" + json.dumps(STORY), stream=True, reasoning_effort="low")
        self.assertEqual(code, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(json.loads(content), STORY)
        self.assertIn("reasoning_content", raw)
        self.assertIn("data: [DONE]", raw)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_generation_is_an_error_and_never_streams_content(self):
        for script in ("## Film Plan - Photobooth", '{"title":"Only a title"}',
                       json.dumps({**STORY, "extra": True}),
                       json.dumps({**STORY, "boxes": [{**STORY["boxes"][0], "duration_ms": "15000"}]})):
            with self.subTest(script=script):
                code, reply = self.chat(script)
                self.assertEqual(code, 502)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
                _, raw = self.chat(script, stream=True)
                self.assertIn('"structured_output_failed"', raw)
                self.assertNotIn('"delta"', raw)
                self.assertIn("data: [DONE]", raw)

    def test_object_mode_strict_json_and_truncation(self):
        fmt = {"type": "json_object"}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)
        for script in ('[]', '{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}', 'no json at all', '{"x":'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 502)
        # #762: the JSON object is taken out of a code fence or the prose around it (the object itself stays strict)
        for script in ('```json\n{}\n```', 'Here it is: {"answer":4} - done'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 200, script)
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt, max_tokens=5)[0], 502)
        # Plain text callers retain normal streaming, without a JSON validation gate.
        self.assertEqual(self.chat("Prose.", response_format={"type": "text"})[0], 200)

    def test_rejected_schema_does_not_load_the_engine_or_retry(self):
        with mock.patch.object(self.svc, "load") as load:
            code, _ = self.chat(response_format={"type": "json_schema"})
            self.assertEqual(code, 400)
            load.assert_not_called()
        self.engine = self.svc.engine = MockEngine(self.tok, "Markdown is not JSON", max_context=16384)
        with mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as generate:
            self.assertEqual(self.chat()[0], 502)
            self.assertEqual(generate.call_count, 1)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_schema_is_400_and_local_refs_work(self):
        for schema in ({"type": "object", "properties": {"x": {"type": "invalid"}}},
                       {"type": "object", "$ref": "https://other.example/schema"}):
            fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": schema}}
            self.assertEqual(self.chat(response_format=fmt)[0], 400)
        schema = object_schema({"answer": {"$ref": "#/$defs/n"}})
        schema["$defs"] = {"n": {"type": "integer"}}
        fmt = {"type": "json_schema", "json_schema": {"name": "local", "schema": schema}}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)

    def test_malformed_formats_are_rejected_before_loading(self):
        formats = ([], {"type": "unsupported"},
                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                   {"type": "json_schema",
                    "json_schema": {"name": "x", "strict": "true", "schema": {"type": "object"}}})
        with mock.patch.object(self.svc, "load") as load:
            for fmt in formats:
                with self.subTest(fmt=fmt):
                    self.assertEqual(self.chat(response_format=fmt)[0], 400)
            self.assertEqual(self.request("/v1/chat/completions", [])[0], 400)
            load.assert_not_called()

    def test_tools_with_a_schema(self):
        # #60: the schema is the final answer's - a turn that calls a tool goes out as it is, the answer is checked
        tool = {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
        call = "<tool_call>\n<function=search>\n<parameter=q>\nphotobooth\n</parameter>\n</function>\n</tool_call>"
        code, reply = self.chat(call, tools=[tool])
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(reply["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "search")
        _, raw = self.chat(call, tools=[tool], stream=True)
        self.assertIn('"tool_calls"', raw)
        self.assertNotIn('"structured_output_failed"', raw)
        code, reply = self.chat(json.dumps(STORY), tools=[tool])
        self.assertEqual(code, 200, reply)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        code, reply = self.chat("## Film Plan - Photobooth", tools=[tool])
        self.assertEqual(code, 502)
        self.assertEqual(reply["error"]["code"], "structured_output_failed")

    def test_without_jsonschema_an_object_is_still_required(self):
        # without jsonschema a strict request (strict: true, a strict tool) is refused, as its reply could not be
        # checked; another json_schema answer is checked to be one JSON object, and json_object never needs it
        loose = {"type": "json_schema", "json_schema": {"name": "h3_story", "schema": STORY_SCHEMA}}
        tool = {"type": "function", "function": {"name": "f", "strict": True, "parameters": {"type": "object"}}}
        with mock.patch.object(structured, "_jsonschema", False):
            code, reply = self.chat()
            self.assertEqual(code, 400)
            self.assertIn("jsonschema", reply["error"]["message"])
            code, reply = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "go"}],
                                                                "tools": [tool], "reasoning_effort": "none"})
            self.assertEqual(code, 400)
            self.assertIn("jsonschema", reply["error"]["message"])
            self.assertEqual(self.chat(response_format=loose)[0], 200)
            for script in ("## Film Plan - Photobooth", "[1, 2]"):
                code, reply = self.chat(script, response_format=loose)
                self.assertEqual(code, 502, script)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
            self.assertEqual(self.chat('{"answer":4}', response_format={"type": "json_object"})[0], 200)

    def test_root_unions_of_object_shapes_are_accepted(self):
        # A root anyOf of object shapes (sent by apps written against llama.cpp's grammar path) still means
        # "one JSON object", so it is accepted; a non-object answer is still refused.
        claim = {"type": "object", "additionalProperties": False, "required": ["claim_text"],
                 "properties": {"claim_text": {"type": "string", "minLength": 1}}}
        empty = {"type": "object", "additionalProperties": False, "required": ["outcome"],
                 "properties": {"outcome": {"type": "string", "enum": ["no_substantive_content"]}}}
        schemas = [
            {"anyOf": [claim, empty]},
            {"oneOf": [claim, empty]},
            {"allOf": [claim, {"required": ["claim_text"]}]},
            {"$ref": "#/$defs/claim", "$defs": {"claim": claim}},
            {"anyOf": [{"$ref": "#/$defs/claim"}, {"anyOf": [empty]}], "$defs": {"claim": claim}},
        ]
        for schema in schemas:
            fmt = {"type": "json_schema", "json_schema": {"name": "union", "strict": True, "schema": schema}}
            with self.subTest(schema=schema):
                self.assertEqual(self.chat('{"claim_text":"The term is 12 months."}', response_format=fmt)[0], 200)
                self.assertEqual(self.chat("[1, 2]", response_format=fmt)[0], 502)

    def test_schemas_that_allow_non_objects_are_still_rejected(self):
        obj = {"type": "object"}
        schemas = [
            {"type": "array", "items": obj},
            {"type": "string"},
            {"type": ["object", "null"]},
            {"anyOf": [obj, {"type": "string"}]},
            {"anyOf": []},
            {"oneOf": [obj, {"type": "array"}]},
            {"allOf": [{"type": "string"}]},
            {"properties": {"x": {"type": "string"}}},           # no type: any JSON value validates
            {"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}},   # reference cycle
            {"$ref": "#/$defs/missing", "$defs": {}},
            {"$ref": "#/$defs/s", "$defs": {"s": {"type": "string"}}},
        ]
        with mock.patch.object(self.svc, "load") as load:
            for schema in schemas:
                fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": schema}}
                with self.subTest(schema=schema):
                    code, reply = self.chat(response_format=fmt)
                    self.assertEqual(code, 400)
                    self.assertIn("only JSON objects", reply["error"]["message"])
            load.assert_not_called()

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_strict_tool_calls_are_checked(self):
        # a strict tool's call is checked against its parameters before it is sent, with no grammar on the engine (the
        # mock has none, as another call form or an engine without grammars); a Responses namespace's and
        # additional_tools' strict tools are found under the names the template has
        tool = {"type": "function", "function": {"name": "f", "strict": True, "parameters": object_schema(
            {"days": {"type": "integer"}})}}

        def call(days):
            return f"<tool_call>\n<function=f>\n<parameter=days>\n{days}\n</parameter>\n</function>\n</tool_call>"
        req = {"messages": [{"role": "user", "content": "plan"}], "tools": [tool], "reasoning_effort": "none"}
        self.engine = self.svc.engine = MockEngine(self.tok, call(3), max_context=16384)
        code, reply = self.request("/v1/chat/completions", req)
        self.assertEqual(code, 200, reply)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]),
                         {"days": 3})
        self.engine = self.svc.engine = MockEngine(self.tok, call("three"), max_context=16384)
        code, reply = self.request("/v1/chat/completions", req)
        self.assertEqual(code, 502)
        self.assertEqual(reply["error"]["code"], "structured_output_failed")
        _, raw = self.request("/v1/chat/completions", {**req, "stream": True})
        self.assertIn('"structured_output_failed"', raw)
        self.assertNotIn('"tool_calls"', raw)
        from serve.responses import given_tools
        member = {"name": "g", "strict": True, "parameters": {"type": "object"}}
        ns = {"model": "m", "tools": [{"type": "namespace", "name": "ns", "tools": [member]}],
              "input": [{"type": "additional_tools", "tools": [{"type": "function", **member, "name": "h"}]}]}
        self.assertEqual(structured.constrained_tools(given_tools(ns))[0], {"ns.g", "h"})

    def test_strict_tool_numbers_must_be_finite(self):
        # NaN, Infinity and a number too large for a float (it reads as Infinity) are no JSON: a strict call with one
        # is refused, though jsonschema's "number" takes them
        tool = {"type": "function", "function": {"name": "f", "strict": True, "parameters": object_schema(
            {"x": {"type": "number"}})}}
        req = {"messages": [{"role": "user", "content": "go"}], "tools": [tool], "reasoning_effort": "none"}
        for value in ("NaN", "Infinity", "1e999"):
            with self.subTest(value=value):
                self.engine = self.svc.engine = MockEngine(self.tok, "<tool_call>\n<function=f>\n<parameter=x>\n"
                                                           f"{value}\n</parameter>\n</function>\n</tool_call>",
                                                           max_context=16384)
                code, reply = self.request("/v1/chat/completions", req)
                self.assertEqual(code, 502, reply)
                self.assertIn("not finite", reply["error"]["message"])

    def test_strict_of_another_type(self):
        # a tool's strict that is not true, false or null is a 400 (count_tokens too), not a silent non-strict; null
        # is not given, for a tool and for response_format (OpenAI's SDKs send it)
        def tool(strict):
            return {"type": "function", "function": {"name": "f", "strict": strict, "parameters": {"type": "object"}}}
        user = [{"role": "user", "content": "go"}]
        for strict in ("true", 1, {}, []):
            with self.subTest(strict=strict):
                code, reply = self.request("/v1/chat/completions", {"messages": user, "tools": [tool(strict)]})
                self.assertEqual(code, 400, reply)
                self.assertIn("strict must be", reply["error"]["message"])
                code, _ = self.request("/v1/messages/count_tokens", {
                    "model": "m", "messages": user,
                    "tools": [{"name": "f", "strict": strict, "input_schema": {"type": "object"}}]})
                self.assertEqual(code, 400)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": user, "tools": [tool(None)],
                                                               "reasoning_effort": "none"})[0], 200)
        fmt = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": None, "schema": STORY_SCHEMA}}
        self.assertEqual(self.chat(response_format=fmt)[0], 200)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_responses_strict_by_default_where_possible(self):
        # Responses: a function tool without strict is strict when its parameters meet strict mode's rules (every
        # object additionalProperties false, every property required), else not - and the response's tools say which
        def call(days):
            return (f"</think>\n\n<tool_call>\n<function=f>\n<parameter=days>\n{days}\n</parameter>\n</function>\n"
                    "</tool_call>")
        exact = object_schema({"days": {"type": "integer"}})
        loose = {"type": "object", "properties": {"days": {"type": "integer"}}}
        for params, strict in ((exact, True), (loose, False)):
            with self.subTest(strict=strict):
                body = {"model": "m", "input": "go", "tools": [{"type": "function", "name": "f", "parameters": params}]}
                self.engine = self.svc.engine = MockEngine(self.tok, call(3), max_context=16384)
                code, reply = self.request("/v1/responses", body)
                self.assertEqual(code, 200, reply)
                self.assertIs(reply["tools"][0]["strict"], strict)
                self.engine = self.svc.engine = MockEngine(self.tok, call("three"), max_context=16384)
                self.assertEqual(self.request("/v1/responses", body)[0], 502 if strict else 200)

    def test_the_server_does_not_import_jsonschema_at_start(self):
        import subprocess
        import sys
        code = ("import sys; import serve.server; "
                "sys.exit(1 if any(m == 'jsonschema' or m.startswith('jsonschema.') for m in sys.modules) else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
