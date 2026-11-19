"""#4325: `strata bench compare BASELINE CANDIDATE` stopped with "unsupported benchmark report" (exit 5) on two folders the default `strata bench run`
(standard-v1) wrote; `strata bench report` read the same folders. Two standard results are compared by the medians of their speed tables; a standard result
beside a serving-suite one is said plainly. The folders come from the same mock server the standard benchmark's own tests use."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli import bench_standard as bs  # noqa: E402
from strata_cli.cli import main  # noqa: E402
from tests.terminal.test_bench_standard import StandardBase  # noqa: E402


class CompareStandardResults(StandardBase):
    def two_runs(self):
        _, a = self.run_standard("a")
        _, b = self.run_standard("b")
        return a, b

    def compare(self, a, b, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["bench", "compare", str(a), str(b), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_two_standard_runs_compare_in_plain_text(self):
        a, b = self.two_runs()
        code, out, err = self.compare(a, b)
        self.assertEqual(code, 0, err)
        self.assertIn("Strata standard benchmark comparison", out)
        self.assertIn("Decode tok/s", out)
        self.assertIn("prompt-16k", out)

    def test_json_carries_a_row_for_every_case_and_metric(self):
        a, b = self.two_runs()
        code, out, err = self.compare(a, b, "--output", "json")
        self.assertEqual(code, 0, err)
        rows = json.loads(out)["comparison"]["rows"]
        self.assertEqual(len(rows), 3 * 4 - 1)                       # the short prompt's rate is left out
        self.assertFalse(any(r["case_id"] == "short" and r["metric"] == "prompt_tps" for r in rows))
        for r in rows:
            self.assertIn(r["better"], (None, "baseline", "candidate"))

    def test_markdown_is_a_table(self):
        a, b = self.two_runs()
        code, out, err = self.compare(a, b, "--output", "markdown")
        self.assertEqual(code, 0, err)
        self.assertIn("| Case | Metric | Baseline | Candidate | Change | Better |", out)

    def test_the_change_and_the_better_side_follow_the_numbers(self):
        a, b = self.two_runs()
        base, cand = bs.load(a), bs.load(b)
        for s in base["summaries"]:
            if s["phase"] == "performance" and s["case_id"] == "prompt-4k":
                if s["metric"] == "decode_tps":
                    s["median"] = 50.0
                if s["metric"] == "ttft_s":
                    s["median"] = 1.0
        for s in cand["summaries"]:
            if s["phase"] == "performance" and s["case_id"] == "prompt-4k":
                if s["metric"] == "decode_tps":
                    s["median"] = 60.0
                if s["metric"] == "ttft_s":
                    s["median"] = 2.0
        rows = {(r["case_id"], r["metric"]): r for r in bs.compare(base, cand)["rows"]}
        self.assertEqual(rows[("prompt-4k", "decode_tps")]["change_pct"], 20.0)
        self.assertEqual(rows[("prompt-4k", "decode_tps")]["better"], "candidate")      # a higher rate is better
        self.assertEqual(rows[("prompt-4k", "ttft_s")]["change_pct"], 100.0)
        self.assertEqual(rows[("prompt-4k", "ttft_s")]["better"], "baseline")           # a longer wait is worse

    def test_a_standard_result_beside_a_serving_suite_result_is_said_plainly(self):
        a, _ = self.two_runs()
        old = self.root / "old"
        old.mkdir()
        (old / "results.json").write_text(json.dumps({"schema": "strata.bench.v1", "results": []}), encoding="utf-8")
        code, out, err = self.compare(a, old)
        self.assertEqual(code, 5)
        self.assertIn("two results of the same kind", err + out)


if __name__ == "__main__":
    unittest.main()
