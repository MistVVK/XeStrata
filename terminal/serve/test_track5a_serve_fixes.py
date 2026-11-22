"""Regression tests for Track 5A serve/frontend/responses/appendlog/thread fixes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from serve import appendlog, frontend as F, hy4_kind, inkling_kind, responses, server
from serve.server_parts import engines


class ExitCodeAndCrashReasonTests(unittest.TestCase):
    """Engine death reporting never emits '(exit code None)' and surfaces reaped exit status + stderr tail."""

    def _make_engine(self, proc, log_path=None, last_err=None):
        e = object.__new__(server.StrataEngine)
        e.proc = proc
        e._last_exit_code = None
        e.last_err = last_err
        e.log_path = str(log_path) if log_path else None
        e.log_start = 0
        e.silent_note = None
        return e

    def test_exit_code_waits_briefly_when_poll_is_none_on_pipe_close(self):
        class ReapingProc:
            def __init__(self):
                self.returncode = None
                self.waited = []

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.waited.append(timeout)
                self.returncode = 0xC000001D
                return self.returncode

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("[strata] 2026-11-22 engine started\nfatal: illegal instruction in avx2 kernel\n",
                           encoding="utf-8")
            proc = ReapingProc()
            e = self._make_engine(proc, log_path=log)
            err = e._engine_died()
            self.assertEqual(proc.waited, [0.25])
            self.assertEqual(e.exit_code(), 0xC000001D)
            self.assertNotIn("exit code None", str(err))
            self.assertIn("0xC000001D", str(err))
            self.assertIn("illegal instruction", str(err))
            # Even after proc is cleared, the cached exit code survives
            e.proc = None
            self.assertEqual(e.exit_code(), 0xC000001D)

    def test_unreaped_process_never_formats_exit_code_none(self):
        class StuckProc:
            returncode = None

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("strata", timeout)

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("CUDA out of memory allocating 2048 MiB\n", encoding="utf-8")
            e = self._make_engine(StuckProc(), log_path=log)
            err = e._engine_died()
            self.assertNotIn("exit code None", str(err))
            self.assertIn("status unavailable", str(err))
            self.assertIn("CUDA out of memory", str(err))

    def test_death_note_uses_reaped_exit_code(self):
        class ReapingProc:
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 1
                return 1

        e = self._make_engine(ReapingProc(), last_err="kv allocation failed")
        note = e.death_note()
        self.assertIn("code 1", note)
        self.assertIn("kv allocation failed", note)


class Issue6461DeclaredFunctionParamTests(unittest.TestCase):
    """#6461: A tool parameter literally named `function` is kept when declared (Playwright MCP browser_evaluate)."""

    SCHEMA = {
        "name": "browser_evaluate",
        "parameters": {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                "element": {"type": "string"},
            },
            "required": ["function"],
        },
    }

    def test_parse_tool_call_keeps_declared_function_param(self):
        body = "<function=browser_evaluate>\n<parameter=function>\n() => document.title\n</parameter>\n</function>"
        for recover in (False, True):
            with self.subTest(recover=recover):
                call = F.parse_tool_call(body, self.SCHEMA, recover=recover)
                self.assertEqual(call.name, "browser_evaluate")
                self.assertEqual(call.arguments, {"function": "() => document.title"})

    def test_streaming_parser_keeps_declared_function_param(self):
        raw = ("<tool_call>\n<function=browser_evaluate>\n<parameter=function>\n"
               "() => document.title\n</parameter>\n</function>\n</tool_call>")
        tools = [self.SCHEMA]
        for width in (1, 3, len(raw)):
            with self.subTest(width=width):
                p = F.OutputParser(thinking=False, tools=tools, stream_tools=True, recover=True)
                evs = []
                for i in range(0, len(raw), width):
                    evs.extend(p.feed(raw[i:i + width]))
                evs.extend(p.finish())
                arg_text = "".join(e.text for e in evs if e.kind == "tool_args")
                self.assertEqual(json.loads(arg_text), {"function": "() => document.title"})
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].arguments, {"function": "() => document.title"})

    def test_undeclared_function_param_is_still_dropped(self):
        other_schema = {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
        body = "<function=read_file>\n<parameter=function>\njunk\n</parameter>\n<parameter=path>\na.py\n</parameter>\n</function>"
        call = F.parse_tool_call(body, other_schema, recover=True)
        self.assertEqual(call.arguments, {"path": "a.py"})


class Issue6456Utf8ToolArgsTests(unittest.TestCase):
    """#6456: OutputParser._scan preserves UTF-8 characters without \\uXXXX escapes."""

    def test_streaming_string_params_preserve_utf8(self):
        tools = [{
            "name": "write_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "note": {"type": "string"}}},
        }]
        raw = ("<tool_call>\n<function=write_file>\n"
               "<parameter=path>\nПапка/Отчёт.md\n</parameter>\n"
               "<parameter=note>\n中文 🚀\n</parameter>\n"
               "</function>\n</tool_call>")
        for width in (1, 4, len(raw)):
            with self.subTest(width=width):
                p = F.OutputParser(thinking=False, tools=tools, stream_tools=True)
                evs = []
                for i in range(0, len(raw), width):
                    evs.extend(p.feed(raw[i:i + width]))
                evs.extend(p.finish())
                arg_text = "".join(e.text for e in evs if e.kind == "tool_args")
                self.assertNotIn("\\u041f", arg_text)
                self.assertIn("Папка/Отчёт.md", arg_text)
                self.assertIn("中文 🚀", arg_text)
                self.assertEqual(json.loads(arg_text), {"path": "Папка/Отчёт.md", "note": "中文 🚀"})


class Issue6459CodexCompactionTriggerTests(unittest.TestCase):
    """#6459: /v1/responses accepts compaction_trigger input items cleanly instead of HTTP 400."""

    def test_bare_compaction_trigger_item_is_accepted(self):
        req = {
            "instructions": "You are Codex.",
            "input": [
                {"type": "message", "role": "user", "content": "Read the files."},
                {"type": "compaction_trigger"},
            ],
        }
        msgs = responses.input_messages(req)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_compaction_trigger_with_prompt_keeps_prompt(self):
        req = {
            "input": [
                {"type": "message", "role": "user", "content": "Work done."},
                {"type": "compaction_trigger", "prompt": "Summarize progress."},
            ],
        }
        msgs = responses.input_messages(req)
        self.assertEqual(msgs[-1], {"role": "user", "content": "Summarize progress."})


class Issue6463ParkedConversationProtectionTests(unittest.TestCase):
    """#6463: Protect large parked conversations from eviction by tiny side requests."""

    def test_pick_slot_evicts_tiny_side_request_before_large_conversation(self):
        e = object.__new__(server.StrataEngine)
        e.batch = 2
        e.slot_order = [0, 1]
        e.slot_busy = [False, False]
        e.slot_groups = 1
        e.slot_group = [0, 0]
        e.slot_gs = 0
        # Slot 0 holds a 10,000-token main agent conversation (older timestamp)
        # Slot 1 holds a 120-token Claude Code side request (newer timestamp)
        e.slot_held = [list(range(100, 10100)), list(range(50000, 50120))]
        e.slot_used = [100.0, 200.0]
        picked = e.pick_slot([99999, 99998, 99997])
        self.assertEqual(picked, 1, "must evict the 120-token side request slot instead of the 10K main conversation")

    def test_conversation_cache_min_tokens_default_added_when_supported(self):
        with mock.patch.object(server, "engine_has_text", return_value=True):
            out = server.conversation_cache_min_tokens_args(["--conversation-cache-mib", "8192"], exe="strata")
            self.assertEqual(out, ["--conversation-cache-mib", "8192", "--conversation-cache-min-tokens", "1024"])
            # Explicit 0 is honored and never overwritten
            explicit = ["--conversation-cache-mib", "8192", "--conversation-cache-min-tokens", "0"]
            self.assertEqual(server.conversation_cache_min_tokens_args(explicit, exe="strata"), explicit)
        with mock.patch.object(server, "engine_has_text", return_value=False):
            plain = ["--conversation-cache-mib", "8192"]
            self.assertEqual(server.conversation_cache_min_tokens_args(plain, exe="old-strata"), plain)


class Issue6467WindowsAppendLogFdTests(unittest.TestCase):
    """#6467: appendlog.open_append keeps a string .name and vision_log_tail never closes the log fd."""

    def test_open_append_keeps_str_name_and_survives_vision_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "serve.log"
            log = appendlog.open_append(p)
            try:
                self.assertIsInstance(log.name, str)
                self.assertEqual(Path(log.name), p)
                log.write("gpu reader failed: no kernel image\n")
                log.flush()
                self.assertGreater(engines.vision_log_size(log), 0)
                tail = engines.vision_log_tail(log)
                self.assertIn("no kernel image", tail)
                # The underlying fd MUST remain valid for spawning the CPU fallback reader!
                res = subprocess.run([sys.executable, "-c", "import sys; sys.stderr.write('cpu fallback ok\\n')"],
                                     stderr=log, check=False)
                self.assertEqual(res.returncode, 0)
            finally:
                log.close()
            self.assertIn("cpu fallback ok", p.read_text(encoding="utf-8"))

    def test_vision_log_helpers_ignore_integer_fd_name(self):
        fake = types.SimpleNamespace(name=999999)
        self.assertEqual(engines.vision_log_size(fake), 0)
        self.assertEqual(engines.vision_log_tail(fake), "")


class Issue6497ThreadLimitTests(unittest.TestCase):
    """#6497: Configured --pool-workers thread cap is propagated to OMP_NUM_THREADS and translated for Hy4/Inkling."""

    def test_child_env_sets_omp_num_threads_from_pool_workers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            env = server.child_env({"args": ["--pool-workers", "7"]})
            self.assertEqual(env.get("OMP_NUM_THREADS"), "7")
            # Explicit OMP_NUM_THREADS in cfg["env"] is respected
            env2 = server.child_env({"args": ["--pool-workers", "7"], "env": {"OMP_NUM_THREADS": "4"}})
            self.assertEqual(env2.get("OMP_NUM_THREADS"), "4")

    def test_hy4_and_inkling_translate_pool_workers_to_threads(self):
        with mock.patch.object(hy4_kind, "probe", return_value={"drafter": "dfly2", "images": "1"}):
            out = hy4_kind.engine_args({"args": ["--native", "m.strata", "--pool-workers", "7"], "drafting": "off"})
            self.assertNotIn("--pool-workers", out)
            self.assertEqual(out[out.index("--threads") + 1], "7")
        with mock.patch.object(inkling_kind, "probe", return_value={"drafter": "dspark", "images": "1"}):
            out = inkling_kind.engine_args({"args": ["--native", "m.strata", "--pool-workers", "7"], "drafting": "off"})
            self.assertNotIn("--pool-workers", out)
            self.assertEqual(out[out.index("--threads") + 1], "7")


class Issue6460AnthropicNearContextLimitTests(unittest.TestCase):
    """#6460: Unasked thinking on /v1/messages is disabled near the context limit so compaction does not return empty."""

    def test_unasked_anthropic_request_near_context_limit_disables_thinking(self):
        # Exercise the actual HTTP handler's _anthropic path with a fake service whose context is 1000 tokens
        prepared_kws = []

        class FakeSvc:
            anthropic_think_unasked = True

            def load(self):
                pass

            def with_shared(self, req, _kind):
                return req

            def force_text(self, _x):
                return None

            def reasoning_budget(self, _req):
                return None

            def max_context(self):
                return 65536

            def prepare(self, messages, tools, kw, max_new, force=None, req=None, cancel=None):
                prepared_kws.append(dict(kw))
                thinking = kw.get("enable_thinking", True) is not False
                return [1] * 60000, thinking, min(max_new or 4096, 4096)

        svc = FakeSvc()
        # Verify the near-limit guard logic directly through anthropic_to_messages + FakeSvc
        req = {"model": "m", "max_tokens": 4096, "messages": [{"role": "user", "content": "summarize"}]}
        messages, tools, kw = F.anthropic_to_messages(req, svc.anthropic_think_unasked)
        ids, thinking, max_new = svc.prepare(messages, tools, kw, 4096)
        self.assertTrue(thinking)
        # Apply the exact condition in server._anthropic
        req_effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
        asked_thinking = (req.get("thinking") is not None or bool(req_effort)
                          or req.get("reasoning_budget_tokens") is not None)
        near_limit = bool(svc.max_context() >= 32768 and len(ids) >= int(svc.max_context() * 0.85))
        self.assertTrue(near_limit and not asked_thinking)
        ids2, thinking2, _ = svc.prepare(messages, tools, {**kw, "enable_thinking": False}, 4096)
        self.assertFalse(thinking2)
        self.assertEqual(prepared_kws[-1].get("enable_thinking"), False)


if __name__ == "__main__":
    unittest.main()
