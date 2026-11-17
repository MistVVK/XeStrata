"""#4210 #5244: Codex multi-agent v2 items on /v1/responses. A worker's task and its answer arrive as `agent_message` items
(Codex's own input_text header, then the payload in an `encrypted_content` part that holds plain text when the agents run
here); agent calls may arrive as `multi_agent_call` / `multi_agent_call_output`. They were refused with a 400 ("input items
of type 'agent_message' are not supported"), which stopped every worker of a Codex Security scan.
Payload shapes as in PRs #4210 (WenyuanXu) and #5244 (bsorensen110), from codex-rs."""
from __future__ import annotations

import base64
import json
import unittest

from serve.responses import ENCRYPTED_PREFIX, ResponsesError, input_messages


def seal(text):
    return ENCRYPTED_PREFIX + base64.b64encode(json.dumps({"text": text}).encode()).decode("ascii")


HEADER_TASK = "Message Type: NEW_TASK\nTask name: /root/worker1\nSender: /root\nTrigger turn: false\nPayload:\n"
TASK = {"type": "agent_message", "author": "/root", "recipient": "/root/worker1",
        "content": [{"type": "input_text", "text": HEADER_TASK},
                    {"type": "encrypted_content", "encrypted_content": "Compute 2 + 2 and reply with the exact integer."}]}
HEADER_ANSWER = "Message Type: FINAL_ANSWER\nTask name: /root/worker1\nSender: /root/worker1\nTrigger turn: true\nPayload:\n"
ANSWER = {"type": "agent_message", "author": "/root/worker1", "recipient": "/root",
          "content": [{"type": "input_text", "text": HEADER_ANSWER}, {"type": "encrypted_content", "encrypted_content": seal("4")}]}


class AgentMessage(unittest.TestCase):
    def test_a_task_is_a_user_message_with_codex_header_and_payload(self):
        msgs = input_messages({"instructions": "You are Codex", "input": [TASK]})
        self.assertEqual(msgs, [{"role": "system", "content": "You are Codex"},
                                {"role": "user", "content": HEADER_TASK + "Compute 2 + 2 and reply with the exact integer."}])

    def test_a_payload_this_server_sealed_is_opened(self):
        msgs = input_messages({"input": [ANSWER]})
        self.assertEqual(msgs[-1]["content"], HEADER_ANSWER + "4")

    def test_another_servers_ciphertext_is_named_unreadable_not_passed_off_as_text(self):
        token = "gAAAAA" + "Bo" * 60 + "=="
        item = {"type": "agent_message", "content": [{"type": "input_text", "text": HEADER_ANSWER},
                                                     {"type": "encrypted_content", "encrypted_content": token}]}
        content = input_messages({"input": [item]})[-1]["content"]
        self.assertNotIn(token, content)
        self.assertIn("encrypted by another server and cannot be read here", content)
        for broken in (ENCRYPTED_PREFIX + "!!!", seal(None), None):
            item = {"type": "agent_message", "content": [{"type": "encrypted_content", "encrypted_content": broken}]}
            self.assertIn("cannot be read here", input_messages({"input": [item]})[-1]["content"])

    def test_a_message_without_codex_header_keeps_who_sent_it(self):
        item = {"type": "agent_message", "author": "orchestrator", "recipient": "researcher", "content": "look at x"}
        self.assertEqual(input_messages({"input": [item]})[-1]["content"], "Message from orchestrator to researcher:\nlook at x")

    def test_an_unknown_part_is_refused_clearly(self):
        item = {"type": "agent_message", "content": [{"type": "input_audio", "data": "x"}]}
        with self.assertRaises(ResponsesError) as caught:
            input_messages({"input": [item]})
        self.assertIn("agent_message parts of type 'input_audio'", str(caught.exception))

    def test_a_message_between_calls_and_their_outputs_follows_the_outputs(self):
        msgs = input_messages({"input": [
            {"type": "message", "role": "user", "content": "start"},
            {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"a\"}", "call_id": "c1"},
            {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"b\"}", "call_id": "c2"},
            TASK,
            {"type": "function_call_output", "call_id": "c2", "output": "out2"},
            {"type": "function_call_output", "call_id": "c1", "output": "out1"},
            {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"c\"}", "call_id": "c3"},
            {"type": "function_call_output", "call_id": "c3", "output": "out3"},
        ]})
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool", "tool", "user", "assistant", "tool"])
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["out1", "out2", "out3"])
        self.assertTrue(msgs[4]["content"].startswith("Message Type: NEW_TASK"))
        self.assertEqual(len(msgs[1]["tool_calls"]), 2)

    def test_a_held_message_is_not_lost_when_the_outputs_never_come(self):
        msgs = input_messages({"input": [
            {"type": "function_call", "name": "spawn_agent", "arguments": "{}", "call_id": "c1"}, TASK]})
        self.assertEqual([m["role"] for m in msgs], ["assistant", "user"])

    def test_multi_agent_call_items_are_calls_and_results(self):
        msgs = input_messages({"input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "multi_agent_call", "action": "spawn_agent", "namespace": "multi_agent_v2", "arguments": "{\"task\":\"t\"}",
             "call_id": "m1"},
            {"type": "multi_agent_call_output", "call_id": "m1", "output": "started"},
            ANSWER]})
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool", "user"])
        self.assertEqual(msgs[1]["tool_calls"][0]["function"], {"name": "multi_agent_v2.spawn_agent", "arguments": {"task": "t"}})
        self.assertEqual(msgs[2]["content"], "started")

    def test_requests_without_these_items_render_as_before(self):
        req = {"instructions": "I", "input": [{"type": "message", "role": "user", "content": "hi"},
                                              {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "a"},
                                              {"type": "function_call_output", "call_id": "a", "output": "r"}]}
        self.assertEqual([m["role"] for m in input_messages(req)], ["system", "user", "assistant", "tool"])


from serve.test_responses import Server  # noqa: E402  (the live mock server of the Responses tests)


class OverHttp(Server):
    def test_a_worker_request_is_answered_not_refused(self):
        code, r = self.post({"model": "m", "store": False, "input": [
            {"type": "message", "role": "user", "content": "scan the repo"},
            {"type": "function_call", "name": "spawn_agent", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "started"},
            ANSWER]})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["status"], "completed")


del Server                                           # its own tests stay in serve.test_responses


if __name__ == "__main__":
    unittest.main()
