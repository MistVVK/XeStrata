"""#4941: setup.py --output jsonl: one JSON object per line on stdout (schema_version 1; events step, ok, warning, progress, error, done) for a program that
installs a model; every human line, and every program setup runs, goes to stderr; exit codes are unchanged. Without the option nothing changes."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

DRIVER = textwrap.dedent('''
    import subprocess, sys
    sys.path.insert(0, sys.argv[1])
    import setup
    if sys.argv[2] == "jsonl":
        setup.jsonl_start()
    setup.step(2, "checking this PC")
    setup.ok("NVIDIA GeForce RTX 3060, 12 GB")
    print("a plain line")
    subprocess.call([sys.executable, "-c", "print('a child program writes this')"])
    subprocess.call([sys.executable, "-c", "import os; os.write(1, b'raw child bytes, no newline')"])
    sys.stdout.write("a partial line"); sys.stdout.flush()
    setup.warn("little free disk space")
    for done in (0, 10, 20, 100):
        setup.progress_event("IQ3_S shard 1", done, 100)
    setup.write_config(__import__("pathlib").Path(sys.argv[3]), {"model_name": "x"})
    if sys.argv[4] == "fail":
        setup.fail("not enough free disk space in D:/models: need ~90 GB, 40 GB free", "free up space", kind="no_space", free_bytes=40, needed_bytes=90)
''')


class Jsonl(unittest.TestCase):
    def run_driver(self, mode, end="ok"):
        tmp = Path(tempfile.mkdtemp(prefix="strata-jsonl-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        driver = tmp / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", str(driver), str(ROOT), mode, str(tmp / "strata-x.json"), end],
                              capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)

    def test_stdout_holds_only_json_lines_and_the_text_goes_to_stderr(self):
        r = self.run_driver("jsonl")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.endswith("\n"))
        events = [json.loads(line) for line in r.stdout.splitlines()]          # every line parses: no byte of text, of a child program or of a partial line
        self.assertTrue(all(e["schema_version"] == 1 for e in events))
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[:3], ["step", "ok", "warning"])
        self.assertEqual(events[0], {"schema_version": 1, "event": "step", "n": 2, "title": "checking this PC"})
        self.assertEqual(events[2]["message"], "little free disk space")
        progress = [e for e in events if e["event"] == "progress"]
        self.assertEqual((progress[0]["done_bytes"], progress[-1]["done_bytes"], progress[-1]["total_bytes"]), (0, 100, 100))
        self.assertLess(len(progress), 4, "at most one a second, and always the last")
        for text in ("raw child bytes, no newline", "a partial line", "=== Step 2: checking this PC ===", "[ok] NVIDIA GeForce RTX 3060", "a plain line", "a child program writes this", "[!]  little free"):
            self.assertIn(text, r.stderr)
            self.assertNotIn(text, r.stdout)

    def test_a_stop_is_an_error_event_with_its_fields_and_the_same_exit_code(self):
        r = self.run_driver("jsonl", "fail")
        self.assertEqual(r.returncode, 1)
        last = json.loads(r.stdout.splitlines()[-1])
        self.assertEqual((last["event"], last["kind"], last["free_bytes"], last["needed_bytes"], last["exit_code"], last["hint"]),
                         ("error", "no_space", 40, 90, 1, "free up space"))
        self.assertIn("[X]  not enough free disk space", r.stderr)

    def test_without_the_option_nothing_changes(self):
        r = self.run_driver("human")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("=== Step 2: checking this PC ===", r.stdout)
        self.assertIn("a child program writes this", r.stdout)
        self.assertNotIn('"schema_version"', r.stdout)

    def test_the_main_entry_ends_with_done_naming_the_run_config(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        tail = src[src.index('if __name__ == "__main__":'):]
        self.assertIn('event("done", exit_code=int(code or 0), run_config=str(_RUN_CONFIGS[-1]) if _RUN_CONFIGS else None)', tail)
        self.assertIn('if a.output == "jsonl":\n        jsonl_start()', src)
        self.assertIn('kind="no_space"', src)
        setup._RUN_CONFIGS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            setup.write_config(Path(tmp) / "strata-q.json", {"a": 1})
            self.assertEqual(setup._RUN_CONFIGS, [Path(tmp) / "strata-q.json"])
        setup._RUN_CONFIGS.clear()


if __name__ == "__main__":
    unittest.main()
