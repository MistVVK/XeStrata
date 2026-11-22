"""Round 0213 TUICFG, step 9: the output cap strata connect writes (#5006; PR #4230's idea, written here) and the OpenCode block (#4528)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strata_cli.launch import harness

URL = "http://127.0.0.1:18900"


def render(name, context):
    return harness.render(harness.find(name), harness.target(URL, None, "m", context))


class OutputCap(unittest.TestCase):
    def test_the_cap_grows_with_the_window_and_the_tested_window_keeps_4096(self):
        caps = {c: harness.target(URL, None, "m", c).output_cap for c in (8192, 32768, 65536, 131072, 204800, 524288)}
        self.assertEqual(caps, {8192: 4096, 32768: 4096, 65536: 8192, 131072: 16384, 204800: 25600, 524288: 65536})

    def test_opencode_gets_room_for_a_compaction_summary_on_a_wide_server(self):
        text = render("opencode", 131072)["files"][0]["text"]
        cfg = json.loads(text)
        limit = next(iter(cfg["providers"]["strata"]["models"].values()))["limit"]
        self.assertEqual(limit, {"context": 131072, "input": 126976, "output": 16384})
        trigger = limit["input"] - cfg["compaction"]["buffer"]
        self.assertLess(trigger + limit["output"], limit["context"], "compaction fires with room left for its own summary and the reply")
        small = json.loads(render("opencode", 32768)["files"][0]["text"])
        self.assertEqual(next(iter(small["providers"]["strata"]["models"].values()))["limit"], {"context": 32768, "input": 28672, "output": 4096})

    def test_every_config_that_names_a_cap_uses_it(self):
        r = render("claude-code", 131072)
        self.assertEqual(r["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "16384")
        self.assertTrue(any("CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384" in n for n in r["limits"]))
        self.assertNotIn("maxTokens: 4096", render("omp", 131072)["files"][0]["text"])
        self.assertIn("maxTokens: 16384", render("omp", 131072)["files"][0]["text"])
        self.assertIn('"maxTokens": 16384', render("pi", 131072)["files"][0]["text"])
        self.assertIn('"maxTokens": 16384', render("openclaw", 131072)["files"][0]["text"])
        self.assertIn('"maxTokens": 16384', render("cline", 131072)["files"][0]["text"])
        self.assertIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096", " ".join(render("claude-code", 32768)["limits"]))

    def test_the_opencode_block_keeps_the_fields_the_tested_clients_read(self):
        """PR #6512: OpenCode v2 `providers` schema with dynamic _display_name(t)."""
        cfg = json.loads(render("opencode", 131072)["files"][0]["text"])
        p = cfg["providers"]["strata"]
        self.assertEqual((p["package"], sorted(p["settings"])), ("@opencode/ai/providers/openai-compatible", ["apiKey", "baseURL"]))
        model = next(iter(p["models"].values()))
        self.assertTrue(model["capabilities"]["tools"] and model["capabilities"]["reasoning"]["supported"])
        hy4 = json.loads(harness.render(harness.find("opencode"), harness.target(URL, None, "hy4-flash-q4", 131072))["files"][0]["text"])
        self.assertEqual(next(iter(hy4["providers"]["strata"]["models"].values()))["name"], "Tencent Hy4-Flash (Strata, local)")


class RewriteKeepsTheirCap(unittest.TestCase):
    def write(self, old, ctx=131072):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opencode.json"
            if old is not None:
                path.write_text(json.dumps(old), encoding="utf-8")
            spec = render("opencode", ctx)["files"][0]
            spec["path"] = str(path)
            harness.write_file(spec, backup=False)
            return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def out(cfg):
        return next(iter(cfg["providers"]["strata"]["models"].values()))["limit"]["output"]

    def old_file(self, output):
        return {"theme": "mine", "providers": {"strata": {"models": {"m": {"limit": {"context": 131072, "input": 126976, "output": output}}}}}}

    def test_the_4096_an_older_strata_wrote_is_replaced(self):
        cfg = self.write(self.old_file(4096))
        self.assertEqual(self.out(cfg), 16384)
        self.assertEqual(cfg["theme"], "mine")

    def test_a_cap_the_person_set_stays(self):
        for value in (8192, 32768):
            self.assertEqual(self.out(self.write(self.old_file(value))), value)

    def test_v1_provider_upgrade_keeps_custom_cap_and_replaces_4096(self):
        old_v1_custom = {"theme": "mine", "provider": {"strata": {"models": {"m": {"limit": {"context": 131072, "input": 126976, "output": 32768}}}}}}
        self.assertEqual(self.out(self.write(old_v1_custom)), 32768)
        old_v1_default = {"theme": "mine", "provider": {"strata": {"models": {"m": {"limit": {"context": 131072, "input": 126976, "output": 4096}}}}}}
        self.assertEqual(self.out(self.write(old_v1_default)), 16384)

    def test_a_new_file_gets_the_generated_cap(self):
        self.assertEqual(self.out(self.write(None)), 16384)

    def test_claude_settings_keep_a_chosen_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000", "X": "1"}}), encoding="utf-8")
            spec = render("claude-code", 131072)["files"][0]
            spec["path"] = str(path)
            harness.write_file(spec, backup=False)
            env = json.loads(path.read_text())["env"]
        self.assertEqual((env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], env["X"]), ("32000", "1"))


if __name__ == "__main__":
    unittest.main()
