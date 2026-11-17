"""Anthropic tool_result blocks must follow tool_use_id before the template drops it."""
import copy
import http.client
import json
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, serve


TEMPLATE = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))


def history():
    """Two calls of one tool, answered in the order they finished: what Claude Code 2.1.316 sends for parallel Greps."""
    return [
        {"role": "user", "content": "Search for both."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_slow", "name": "Grep", "input": {"pattern": "slow"}},
            {"type": "tool_use", "id": "toolu_fast", "name": "Grep", "input": {"pattern": "fast"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_fast", "content": "RESULT_FROM_FAST"},
            {"type": "tool_result", "tool_use_id": "toolu_slow", "content": "RESULT_FROM_SLOW"}]},
    ]


def in_call_order():
    raw = history()
    raw[2]["content"].reverse()
    return raw


def normalized(messages):
    return anthropic_to_messages({"messages": messages})[0]


def results(messages):
    return [m["content"] for m in normalized(messages) if m["role"] == "tool"]


class AnthropicToolResults(unittest.TestCase):
    def test_reversed_results_of_the_same_tool_follow_tool_use_id(self):
        raw = history()
        before = copy.deepcopy(raw)
        self.assertEqual(results(raw), ["RESULT_FROM_SLOW", "RESULT_FROM_FAST"])
        self.assertEqual(raw, before)
        for template in (TEMPLATE,):
            with self.subTest(template=str(template)):
                prompt = template.render(normalized(raw), enable_thinking=False)
                self.assertEqual(prompt, template.render(normalized(in_call_order()), enable_thinking=False))
                self.assertLess(prompt.index("RESULT_FROM_SLOW"), prompt.index("RESULT_FROM_FAST"))

    def test_already_ordered_and_legacy_prompts_are_identical(self):
        ordered = in_call_order()
        legacy = copy.deepcopy(ordered)
        for block in legacy[1]["content"]:
            block.pop("id")
        for block in legacy[2]["content"]:
            block.pop("tool_use_id")
        self.assertEqual(normalized(ordered), normalized(legacy))
        self.assertEqual([m["content"] for m in normalized(ordered)[2:]], ["RESULT_FROM_SLOW", "RESULT_FROM_FAST"])

    def test_ids_never_reach_the_template(self):
        for raw in (history(), in_call_order()):
            messages = normalized(raw)
            self.assertTrue(all(set(m) == {"role", "content"} for m in messages[2:]))
            self.assertTrue(all(set(c) == {"function"} for c in messages[1]["tool_calls"]))

    def test_ambiguous_or_incomplete_ids_preserve_received_order(self):
        for case in ("missing_call", "missing_result", "duplicate_call", "duplicate_result", "unknown",
                     "list_call", "list_result", "empty_call", "empty_result", "partial"):
            raw = history()
            calls, blocks = raw[1]["content"], raw[2]["content"]
            if case == "missing_call":
                calls[0].pop("id")
            elif case == "missing_result":
                blocks[0].pop("tool_use_id")
            elif case == "duplicate_call":
                calls[1]["id"] = "toolu_slow"
            elif case == "duplicate_result":
                blocks[1]["tool_use_id"] = "toolu_fast"
            elif case == "unknown":
                blocks[0]["tool_use_id"] = "unknown"
            elif case == "list_call":
                calls[0]["id"] = []
            elif case == "list_result":
                blocks[0]["tool_use_id"] = []
            elif case == "empty_call":
                calls[0]["id"] = ""
            elif case == "empty_result":
                blocks[0]["tool_use_id"] = ""
            else:
                blocks.pop()
            with self.subTest(case=case):
                self.assertEqual(results(raw), [b["content"] for b in blocks])

    def test_each_tool_run_is_scoped_to_its_assistant_turn(self):
        self.assertEqual(results(history() + history()), ["RESULT_FROM_SLOW", "RESULT_FROM_FAST"] * 2)

    def test_results_split_across_user_messages_are_one_run(self):
        raw = history()
        first, second = raw[2]["content"]
        raw[2:] = [{"role": "user", "content": [first]}, {"role": "user", "content": [second]}]
        self.assertEqual(results(raw), ["RESULT_FROM_SLOW", "RESULT_FROM_FAST"])

    def test_system_message_between_calls_and_results_keeps_received_order(self):
        raw = history()
        raw.insert(2, {"role": "system", "content": "late note"})
        self.assertEqual(results(raw), ["RESULT_FROM_FAST", "RESULT_FROM_SLOW"])

    def test_text_beside_the_results_stays_after_them(self):
        raw = history()
        raw[2]["content"].append({"type": "text", "text": "and then?"})
        self.assertEqual([(m["role"], m["content"]) for m in normalized(raw)[2:]],
                         [("tool", "RESULT_FROM_SLOW"), ("tool", "RESULT_FROM_FAST"), ("user", "and then?")])

    def test_multimodal_result_moves_with_its_call(self):
        raw = history()
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
        raw[2]["content"][0]["content"] = [{"type": "text", "text": "RESULT_FROM_FAST"}, image]
        got = results(raw)
        self.assertEqual(got[0], "RESULT_FROM_SLOW")
        self.assertEqual([p.get("type") for p in got[1]], ["text", "image"])

    def test_http_engine_receives_the_ordered_prompt(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=4096)
        service = Service(engine, tok, TEMPLATE)
        httpd = serve(service, port=0)
        try:
            for stream in (False, True):
                with self.subTest(stream=stream):
                    engine.last_prompt = None
                    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
                    try:
                        connection.request("POST", "/v1/messages", json.dumps({
                            "model": "x", "messages": history(), "max_tokens": 16, "stream": stream,
                            "thinking": {"type": "disabled"}}), {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        body = response.read()
                    finally:
                        connection.close()
                    self.assertEqual(response.status, 200, body)
                    prompt = tok.decode(engine.last_prompt)
                    self.assertLess(prompt.index("RESULT_FROM_SLOW"), prompt.index("RESULT_FROM_FAST"))
        finally:
            httpd.shutdown()
            httpd.server_close()



if __name__ == "__main__":
    unittest.main()
