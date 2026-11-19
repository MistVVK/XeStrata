"""#5484: the Monitor's recent-requests table shows the VRAM hit rate column only when a request has a hit rate. On a Mac
(unified memory) and for a model wholly in VRAM the engine reports none, and a column of dashes read as something
missing. The web page (serve/web/app.js, run in node with stand-ins) and the app's Monitor (app/host) both."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"
NODE = shutil.which("node")


def requests_block(text: str) -> str:
    start = text.index("  // recent requests")
    return text[start:text.index('  const all = $("req-all");', start)]


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class WebTable(unittest.TestCase):
    def table(self, requests):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        script = ("const els = {}; const $ = (id) => (els[id] ||= {id, innerHTML: '', hidden: false});\n"
                  "const esc = (s) => String(s); const fmt = (n, d = 0) => (n == null ? '–' : Number(n).toFixed(d)); let reqShowAll = false;\n"
                  f"function run(requests) {{\n{requests_block(app)}\n}}\nrun({json.dumps(requests)});\n"
                  "console.log(JSON.stringify({hidden: $('req-hit').hidden, body: $('req-body').innerHTML}));")
        done = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        out = json.loads(done.stdout)
        return out["hidden"], [len(re.findall(r"<td", row)) for row in out["body"].split("</tr>") if "<td" in row]

    ROW = {"time": 1, "finish": "stop", "prompt_tokens": 200, "reused": 0, "output_tokens": 50, "prompt_ms": 100,
           "decode_tok_s": 80, "duration_s": 1.2}

    def test_a_mac_run_has_no_hit_rate_column(self):
        hidden, cells = self.table([dict(self.ROW, hit_rate=None), dict(self.ROW)])
        self.assertTrue(hidden)
        self.assertEqual(cells, [8, 8])

    def test_one_request_with_a_hit_rate_shows_the_column_for_all(self):
        hidden, cells = self.table([dict(self.ROW, hit_rate=0.91, pcie_share=0.05), dict(self.ROW, hit_rate=None)])
        self.assertFalse(hidden)
        self.assertEqual(cells, [9, 9])

    def test_the_empty_table_spans_the_columns_shown(self):
        hidden, _ = self.table([])
        self.assertTrue(hidden)
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn('colspan="${anyHit ? 9 : 8}"', requests_block(app))

    def test_the_header_cell_is_the_one_the_page_hides(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        self.assertRegex(html, r'<th class="num" id="req-hit"[^>]*>VRAM hit rate</th>')


class AppTable(unittest.TestCase):
    def test_the_app_monitor_shows_the_column_only_with_a_reading(self):
        src = (ROOT.parent / "app" / "host" / "src" / "screens" / "Monitor.tsx").read_text(encoding="utf-8")
        self.assertIn("const anyHit = requests.some(r => hasReading(r.hitRate));", src)
        self.assertRegex(src, r'\{tr\(anyHit \? <th className="num"[^>]*>\{tr\("VRAM hit rate"\)\}</th> : null\)\}')
        self.assertRegex(src, r"\{tr\(anyHit \? <td className=\"num\">\{tr\(hasReading\(r\.hitRate\)")
        self.assertIn("colSpan={(multipleModels ? 9 : 8) - (anyHit ? 0 : 1)}", src)


if __name__ == "__main__":
    unittest.main()
