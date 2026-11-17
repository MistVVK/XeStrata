"""#5080: the web page's About card said "MTP drafts up to 5 tokens" for an engine that ran the DFlash2 drafter (or none): the row named the MTP
whatever /metrics said. The row now says that only when the engine's ``drafting`` is the MTP, or not reported (an older engine, which had only the MTP)."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"


def speculation_expression() -> str:
    line = next(ln for ln in APP.read_text(encoding="utf-8").splitlines() if ln.strip().startswith('["Speculation",'))
    return re.match(r'\s*\["Speculation",\s*(.*)\],\s*$', line).group(1)


class SpeculationRow(unittest.TestCase):
    def test_the_row_reads_the_engines_drafter(self):
        self.assertRegex(speculation_expression(), r'eng\.drafting == null \|\| eng\.drafting === "mtp"')

    @unittest.skipUnless(shutil.which("node"), "node is not on PATH")
    def test_what_the_row_says_for_each_drafter(self):
        cases = {"dflash2": {"spec": 6, "mtp_max": 6, "lookup": 3, "drafting": "dflash2"},
                 "off": {"spec": 4, "lookup": 3, "drafting": "off"},
                 "mtp": {"spec": 4, "lookup": 0, "drafting": "mtp"},
                 "old": {"spec": 4, "lookup": 3},
                 "none": {"spec": 0, "drafting": "mtp"}}
        script = "const out = {};\n" + "".join(
            f"{{ const eng = {json.dumps(eng)}; out[{json.dumps(k)}] = {speculation_expression()}; }}\n" for k, eng in cases.items()) + \
            "console.log(JSON.stringify(out));\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "row.js"
            path.write_text(script, encoding="utf-8")
            got = json.loads(subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True, check=True, timeout=60).stdout)
        self.assertIsNone(got["dflash2"], "a DFlash2 engine is not said to draft with the MTP")
        self.assertIsNone(got["off"])
        self.assertEqual(got["mtp"], "MTP drafts up to 3 tokens")
        self.assertEqual(got["old"], "MTP drafts up to 3 tokens, prompt lookup on")
        self.assertIsNone(got["none"])


if __name__ == "__main__":
    unittest.main()
