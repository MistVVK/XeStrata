# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Responses' automatic strict choice walks schemas, not arbitrary JSON annotation values."""
import copy
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import responses
from serve.structured import constrained_tools, jsonschema_modules


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def function(schema, **extra):
    return {"type": "function", "name": "f", "parameters": schema, **extra}


# Valid annotation data, deliberately not a valid schema (RSP-R2).
ANNOTATION = {"type": "object", "properties": {}, "additionalProperties": False, "required": 1}
EXACT = object_schema({"x": {"type": "integer"}})
LOOSE = {"type": "object", "properties": {"x": {"type": "integer"}}}


class StrictCompatibility(unittest.TestCase):
    def test_annotations_are_json_not_schemas(self):
        annotations = {"default": ANNOTATION, "enum": [ANNOTATION], "const": ANNOTATION,
                       "examples": [ANNOTATION], "x-client-data": {"items": [ANNOTATION]}}
        for key, value in annotations.items():
            for schema in ({**EXACT, key: value}, object_schema({"x": {"type": "integer", key: value}})):
                with self.subTest(key=key, schema=schema):
                    self.assertTrue(responses.strict_compatible(schema))

    def test_property_and_definition_names_are_not_keywords(self):
        names = ("properties", "type", "required", "additionalProperties", "$defs", "definitions", "items",
                 "anyOf", "$ref", "default", "enum", "examples")
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(responses.strict_compatible(object_schema({name: {"type": "integer"}})))
                for keyword in ("$defs", "definitions"):
                    self.assertTrue(responses.strict_compatible({**EXACT, keyword: {name: EXACT}}))
                    self.assertFalse(responses.strict_compatible({**EXACT, keyword: {name: LOOSE}}))

    def test_schema_maps_walk_values(self):
        for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            for child, expected in ((EXACT, True), (LOOSE, False)):
                schema = {**EXACT, keyword: {"properties": child}}
                if keyword == "properties":
                    schema["required"] = ["properties"]
                with self.subTest(keyword=keyword, expected=expected):
                    self.assertIs(responses.strict_compatible(schema), expected)

    def test_single_schema_keywords(self):
        for keyword in ("items", "additionalItems", "contains", "unevaluatedItems", "unevaluatedProperties",
                        "propertyNames", "not", "if", "then", "else", "contentSchema"):
            for child, expected in ((EXACT, True), (LOOSE, False)):
                with self.subTest(keyword=keyword, expected=expected):
                    self.assertIs(responses.strict_compatible({**EXACT, keyword: child}), expected)
        # Schema-valued additionalProperties is not the literal false strict mode needs.
        self.assertFalse(responses.strict_compatible({**EXACT, "additionalProperties": EXACT}))

    def test_schema_arrays_and_nested_arrays(self):
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems", "items"):
            for child, expected in ((EXACT, True), (LOOSE, False)):
                schema = object_schema({"x": {"type": "array", "items": {"type": "array", keyword: [child]}}})
                with self.subTest(keyword=keyword, expected=expected):
                    self.assertIs(responses.strict_compatible(schema), expected)
        nullable = {"anyOf": [{"type": "null"}, EXACT]}
        self.assertTrue(responses.strict_compatible(object_schema({"x": nullable})))
        self.assertFalse(responses.strict_compatible({"anyOf": [EXACT, LOOSE]}))

    def test_legacy_dependencies_distinguish_names_and_schemas(self):
        self.assertTrue(responses.strict_compatible({**EXACT, "dependencies": {"x": ["properties"]}}))
        self.assertTrue(responses.strict_compatible({**EXACT, "dependencies": {"properties": EXACT}}))
        self.assertFalse(responses.strict_compatible({**EXACT, "dependencies": {"properties": LOOSE}}))

    def test_optional_open_and_nullable_objects_fall_back(self):
        for schema in (LOOSE, {**EXACT, "required": []}, {**EXACT, "additionalProperties": True},
                       {**LOOSE, "type": ["object", "null"]}, {"properties": {"x": {"type": "integer"}}}):
            with self.subTest(schema=schema):
                self.assertFalse(responses.strict_compatible(schema))
        self.assertTrue(responses.strict_compatible({**EXACT, "type": ["object", "null"]}))

    def test_local_references_keep_automatic_strict(self):
        schema = object_schema({"x": {"$ref": "#/$defs/closed"}})
        self.assertTrue(responses.strict_compatible({**schema, "$defs": {"closed": EXACT}}))
        self.assertFalse(responses.strict_compatible({**schema, "$defs": {"closed": LOOSE}}))
        for ref in ("#/$defs/a~1b", "#/$defs/a%7E1b"):
            with self.subTest(ref=ref):
                schema = object_schema({"x": {"$ref": ref}})
                self.assertTrue(responses.strict_compatible({**schema, "$defs": {"a/b": EXACT, "$id": EXACT}}))
        # Map keys are names even while resolving pointers; a parameter called $id is not a resource identifier.
        schema = object_schema({"$id": {"type": "integer"}, "x": {"$ref": "#/properties/$id"}})
        self.assertTrue(responses.strict_compatible(schema))
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            for ref in ("#", "#/$defs/missing", "#anchor", "https://example.invalid/schema"):
                with self.subTest(keyword=keyword, ref=ref):
                    self.assertFalse(responses.strict_compatible({**EXACT, keyword: ref}))

    def test_shared_reference_targets_do_not_expand_per_path(self):
        class Counted(dict):
            gets = 0

            def get(self, *args):
                Counted.gets += 1
                return super().get(*args)

        def obj(props):
            return Counted(object_schema(props))

        defs = {"n0": Counted(type="integer")}
        for i in range(1, 19):
            defs[f"n{i}"] = obj({k: Counted({"$ref": f"#/$defs/n{i - 1}"}) for k in ("left", "right")})
        root = obj({"x": Counted({"$ref": "#/$defs/n18"})})
        root["$defs"] = defs
        self.assertTrue(responses.strict_compatible(root))
        # Count dictionary operations, not wall time: work is proportional to schema nodes, not reference paths.
        self.assertLess(Counted.gets, 64 * (3 * len(defs) + 1))
        defs["n0"]["$ref"] = "#/$defs/n18"   # a cycle must not be hidden by successful shared subtrees
        self.assertFalse(responses.strict_compatible(root))

    def test_boolean_subschemas_keep_the_existing_syntactic_rules(self):
        # This predicate checks the two object rules, not the entire OpenAI schema subset. Boolean children do not
        # declare object properties; accepting them here is existing behaviour, not new grammar support.
        for child in (True, False):
            self.assertTrue(responses.strict_compatible(object_schema({"x": child})))
            self.assertTrue(responses.strict_compatible({**EXACT, "items": child}))
            self.assertFalse(responses.strict_compatible(child))  # parameters must still be a schema object

    def test_malformed_schema_positions_fall_back_without_raising(self):
        for schema in (None, [], {**EXACT, "required": 1}, {**EXACT, "required": [["x"]]},
                       {**EXACT, "properties": []}, {**EXACT, "$defs": []}, {**EXACT, "items": 1},
                       {**EXACT, "anyOf": {}}, {**EXACT, "anyOf": [1]},
                       {**EXACT, "dependencies": {"x": 1}}):
            with self.subTest(schema=schema):
                self.assertFalse(responses.strict_compatible(schema))


class StrictDefaults(unittest.TestCase):
    def test_defaults_preserve_schema_and_explicit_strict(self):
        tools = [function(EXACT), function(EXACT, strict=None), function(EXACT, strict=False),
                 function(LOOSE, strict=True), function(LOOSE)]
        req = {"input": "go", "tools": tools}
        original = copy.deepcopy(req)
        with mock.patch.object(responses, "jsonschema_modules", return_value=object()):
            resolved = responses.with_strict_defaults(req)
        self.assertEqual([t["strict"] for t in resolved["tools"]], [True, True, False, True, False])
        self.assertEqual(req, original)
        for before, after in zip(tools, resolved["tools"]):
            self.assertIs(after["parameters"], before["parameters"])
            self.assertEqual(after["parameters"], before["parameters"])

    def test_namespaces_and_additional_tools(self):
        schema = object_schema({"properties": {"type": "integer", "default": ANNOTATION}})
        namespace = {"type": "namespace", "name": "ns", "tools": [{"name": "f", "parameters": schema}]}
        req = {"tools": [namespace], "input": [{"type": "message", "role": "user", "content": "go"},
               {"type": "additional_tools", "tools": [function(schema), namespace]}]}
        original = copy.deepcopy(req)
        with mock.patch.object(responses, "jsonschema_modules", return_value=object()):
            resolved = responses.with_strict_defaults(req)
        strict, _ = constrained_tools(responses.given_tools(resolved))
        self.assertEqual(strict, {"f", "ns.f"})
        self.assertTrue(resolved["input"][1]["tools"][1]["tools"][0]["strict"])
        self.assertEqual(req, original)
        self.assertEqual(responses.request_tools(resolved)[0][0]["parameters"], schema)

    def test_missing_jsonschema_does_not_promote_tools(self):
        req = {"tools": [function(EXACT), function(EXACT, strict=None), function(EXACT, strict=True),
                         function(EXACT, strict=False)]}
        with mock.patch.object(responses, "jsonschema_modules", return_value=None):
            resolved = responses.with_strict_defaults(req)
        self.assertEqual([t["strict"] for t in resolved["tools"]], [False, False, True, False])


@unittest.skipUnless(jsonschema_modules() is not None, "jsonschema is not installed")
class ResponsesSchemaHTTP(unittest.TestCase):
    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        self.tok = ByteTokenizer()
        self.engine_type = MockEngine
        self.svc = Service(MockEngine(self.tok, "", max_context=16384), self.tok,
                           ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/responses"

    def request(self, body, value="1", argument="x", name="f"):
        script = (f"</think>\n\n<tool_call>\n<function={name}>\n<parameter={argument}>\n{value}\n"
                  "</parameter>\n</function>\n</tool_call>")
        self.svc.engine = self.engine_type(self.tok, script, max_context=16384)
        req = urllib.request.Request(self.url, data=json.dumps({"model": "m", "input": "go",
                                     "max_output_tokens": 256, **body}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            reply = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            reply = error
        with reply:
            raw = reply.read().decode()
            if "event-stream" in reply.headers.get("Content-Type", ""):
                return reply.status, [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
            return reply.status, json.loads(raw)

    def test_valid_default_is_not_a_400_and_call_is_checked(self):
        schema = {**EXACT, "default": ANNOTATION}
        for strict in ({}, {"strict": None}, {"strict": False}, {"strict": True}):
            for stream in (False, True):
                with self.subTest(strict=strict, stream=stream):
                    code, reply = self.request({"tools": [function(schema, **strict)], "stream": stream})
                    self.assertEqual(code, 200, reply)
                    response = reply[-1]["response"] if stream else reply
                    self.assertEqual(response["status"], "completed")
                    self.assertEqual(response["tools"][0]["parameters"], schema)
                    self.assertIs(response["tools"][0]["strict"], strict.get("strict") is not False)
                    calls = [item for item in response["output"] if item["type"] == "function_call"]
                    self.assertEqual(json.loads(calls[0]["arguments"]), {"x": 1})

    def test_reserved_looking_argument_is_strict_and_deltas_are_held(self):
        schema = object_schema({"properties": {"type": "integer"}})
        for strict in ({}, {"strict": None}, {"strict": False}):
            for stream in (False, True):
                with self.subTest(strict=strict, stream=stream):
                    code, reply = self.request({"tools": [function(schema, **strict)], "stream": stream},
                                               value="bad", argument="properties")
                    enforced = strict.get("strict") is not False
                    self.assertEqual(code, 200 if stream or not enforced else 502, reply)
                    if stream and enforced:
                        kinds = [event["type"] for event in reply]
                        self.assertEqual(kinds[-1], "response.failed")
                        self.assertNotIn("response.function_call_arguments.delta", kinds)
                    elif not stream and enforced:
                        self.assertEqual(reply["error"]["code"], "structured_output_failed")
                    else:
                        response = reply[-1]["response"] if stream else reply
                        self.assertIs(response["tools"][0]["strict"], False)
                        self.assertEqual(response["status"], "completed")
        code, reply = self.request({"tools": [function(schema)]}, argument="properties")
        self.assertEqual(code, 200, reply)
        self.assertIs(reply["tools"][0]["strict"], True)

    def test_namespace_and_additional_tools_use_the_same_validation(self):
        schema = object_schema({"properties": {"type": "integer", "default": ANNOTATION}})
        namespace = {"type": "namespace", "name": "ns", "tools": [{"name": "f", "parameters": schema}]}
        for tool, name in ((function(schema), "f"), (namespace, "ns.f")):
            for extra in (False, True):
                body = {"input": [{"type": "message", "role": "user", "content": "go"},
                                  {"type": "additional_tools", "tools": [tool]}]} if extra else {"tools": [tool]}
                with self.subTest(name=name, extra=extra):
                    code, reply = self.request(body, argument="properties", name=name)
                    self.assertEqual(code, 200, reply)
                    code, reply = self.request(body, value="bad", argument="properties", name=name)
                    self.assertEqual(code, 502, reply)
                    self.assertEqual(reply["error"]["code"], "structured_output_failed")

    def test_local_references_keep_validation_with_automatic_and_explicit_strict(self):
        schema = {**object_schema({"x": {"$ref": "#/$defs/value"}}), "$defs": {"value": {"type": "integer"}}}
        for strict in ({}, {"strict": True}):
            with self.subTest(strict=strict):
                code, reply = self.request({"tools": [function(schema, **strict)]}, value="bad")
                self.assertEqual(code, 502, reply)
                code, reply = self.request({"tools": [function(schema, **strict)]})
                self.assertEqual(code, 200, reply)
                self.assertIs(reply["tools"][0]["strict"], True)


if __name__ == "__main__":
    unittest.main()


class CutStructuredAnswer(unittest.TestCase):
    """台帳 0805: max_output_tokens で切れた JSON 形式の答えは検証せず incomplete で返す。"""

    def test_the_text_so_far_is_an_incomplete_message(self):
        asm = responses.Assembler({}, "m", 3, {}, False, json_mode=True)
        asm.json_text = ['{"a": "bcdefgh']
        called = []

        def validate(text, finish):
            called.append(text)
            raise AssertionError("a cut answer must not be validated")

        events = asm.finish({"finish": "length", "completion_tokens": 8}, validate)
        self.assertEqual(called, [])
        self.assertEqual(asm.response["status"], "incomplete")
        self.assertEqual(asm.response["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertIsNone(asm.response["error"])
        self.assertEqual(asm.response["output"][0]["status"], "incomplete")
        self.assertEqual(asm.response["output"][0]["content"][0]["text"], '{"a": "bcdefgh')
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_a_complete_answer_is_still_validated(self):
        asm = responses.Assembler({}, "m", 3, {}, False, json_mode=True)
        asm.json_text = ['{"a": 1}']
        events = asm.finish({"finish": "stop", "completion_tokens": 3}, lambda text, finish: text)
        self.assertEqual(asm.response["status"], "completed")
        self.assertEqual(events[-1]["type"], "response.completed")
