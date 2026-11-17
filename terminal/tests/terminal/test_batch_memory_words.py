"""A Strix Halo start with --batch N that ended with the engine's own line "N requests at once do not fit in this PC's memory": the Terminal's sentence
must name the way out in the words of the app and the Terminal (Concurrent requests / --parallel), not the generic "run it again" and not the shared-memory
advice about the expert cache. Nothing here starts an engine."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from strata_cli.launch import runner as R
from tests.terminal.launch_support import Home

ENGINE_LINE = "strata: 4 requests at once do not fit in this PC's memory: start with --batch 3"
STEP = ("This PC's memory cannot hold that many requests at once. Lower Concurrent requests in the app (or `--parallel` in the Terminal) "
        "by one and start again.")
FAILED = {"state": "failed", "error": "server_exited", "engine_exit_code": 1}


class BatchMemoryWords(unittest.TestCase):
    def config(self, home: Home, log_lines: list[str], args: list[str] | None = None) -> Path:
        log = home.base / "engine.log"
        log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        path = home.base / "run.json"
        path.write_text(json.dumps({"exe": "e", "args": args or [], "log": str(log)}), encoding="utf-8")
        return path

    def test_the_step_says_to_lower_the_concurrent_requests(self):
        with Home() as h:
            cfg = self.config(h, ["strata serve: prefill: copy stream", ENGINE_LINE])
            text, nxt = R.explain_engine_exit(FAILED, cfg)
        self.assertIn(ENGINE_LINE, text)
        self.assertTrue(nxt.startswith(STEP), nxt)
        self.assertNotIn("Run it again", nxt)

    def test_the_step_beats_the_shared_memory_advice_on_a_strix_halo(self):
        args = ["--expert-cache", "8000", "--resident-budget-gib", "38"]
        with Home() as h:
            other = self.config(h, ["strata serve: something odd"], args)
            _, control = R.explain_engine_exit(FAILED, other, shared_memory=True)
            self.assertIn("shares its memory with the RAM", control)        # the control: this config does get the advice for another failure
            cfg = self.config(h, [ENGINE_LINE], args)
            _, nxt = R.explain_engine_exit(FAILED, cfg, shared_memory=True)
        self.assertTrue(nxt.startswith(STEP), nxt)
        self.assertNotIn("shares its memory with the RAM", nxt)

    def test_another_failure_keeps_its_own_step(self):
        with Home() as h:
            cfg = self.config(h, ["strata serve: something odd"])
            _, nxt = R.explain_engine_exit(FAILED, cfg)
        self.assertTrue(nxt.startswith("Run it again; if it repeats, report the log."), nxt)

    def test_the_new_cause_is_the_last_entry(self):
        self.assertIn("requests at once do not fit", KNOWN_PATTERN_LAST())


def KNOWN_PATTERN_LAST() -> str:
    return R.KNOWN_CAUSES[-1][0].pattern


if __name__ == "__main__":
    unittest.main()
