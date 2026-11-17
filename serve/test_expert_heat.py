"""#5723: the web page's Experts view. GET /api/expert-heat reads the expert profile the engine saves when the run config has
"expert_profile_save" (serve/expert_heat.py): ranks per (layer, expert), when it was saved and how often it saves; 404 without the
key, so the tab stays hidden. Bad files are refused, not read. The page's own helpers (serve/web/experts.js) run in node."""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import expert_heat
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"
NODE = shutil.which("node")


def profile(layers: int, experts: int, pairs: list[tuple[int, int]], magic: bytes = b"STRP") -> bytes:
    return magic + struct.pack("<5I", 1, layers, experts, 0, len(pairs)) + b"".join(struct.pack("<HH", l, e) for l, e in pairs)


class Parse(unittest.TestCase):
    def test_ranks_in_file_order_first_place_kept(self):
        got = expert_heat.parse(profile(2, 3, [(1, 2), (0, 0), (1, 2), (0, 2)]))
        self.assertEqual((got["layers"], got["experts"], got["ranked"]), (2, 3, 3))
        self.assertEqual(got["rank"], [1, -1, 2, -1, -1, 0])

    def test_not_a_profile_is_refused(self):
        self.assertIsNone(expert_heat.parse(b"nope"))
        self.assertIsNone(expert_heat.parse(profile(2, 3, [(0, 0)], magic=b"XXXX")))
        short = profile(2, 3, [(0, 0), (0, 1)])[:-4]
        self.assertIsNone(expert_heat.parse(short), "fewer pairs than the header promises")
        self.assertIsNone(expert_heat.parse(profile(0, 3, [])))
        self.assertIsNone(expert_heat.parse(profile(expert_heat.MAX_LAYERS + 1, 2, [])))
        self.assertIsNone(expert_heat.parse(profile(1, 2, [(0, 0), (0, 1), (0, 0)])), "more pairs than layers x experts")

    def test_out_of_range_pairs_are_skipped(self):
        self.assertEqual(expert_heat.parse(profile(1, 2, [(0, 5), (0, 1)]))["rank"], [-1, 0])


class View(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        expert_heat._cache.clear()

    def test_without_the_key_there_is_nothing_to_show(self):
        self.assertEqual(expert_heat.view({})[0], 404)
        self.assertEqual(expert_heat.view(None)[0], 404)
        self.assertEqual(expert_heat.view({"expert_profile_save": "  "})[0], 404)

    def test_before_the_first_save_and_after(self):
        cfg = {"expert_profile_save": "learned.bin", "cwd": self.dir.name, "expert_profile_save_every": 5}
        code, body = expert_heat.view(cfg)
        self.assertEqual((code, body), (200, {"configured": True, "every_min": 5, "saved_at": None}))
        (Path(self.dir.name) / "learned.bin").write_bytes(profile(2, 2, [(1, 1), (0, 0)]))
        code, body = expert_heat.view(cfg)
        self.assertEqual(code, 200)
        self.assertIsInstance(body["saved_at"], float)
        self.assertEqual(body["rank"], [1, -1, -1, 0])
        self.assertEqual(expert_heat.view({**cfg, "expert_profile_save_every": "x"})[1]["every_min"], expert_heat.DEFAULT_EVERY_MIN)

    def test_a_bad_file_says_so(self):
        (Path(self.dir.name) / "learned.bin").write_bytes(b"garbage" * 10)
        body = expert_heat.view({"expert_profile_save": str(Path(self.dir.name) / "learned.bin")})[1]
        self.assertEqual(body["error"], "the saved file is not an expert profile")
        self.assertNotIn("rank", body)


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        expert_heat._cache.clear()
        self.cfg = Path(self.dir.name) / "strata-q2_0.json"
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def get(self):
        try:
            with urllib.request.urlopen(f"http://{self.host}/api/expert-heat", timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_route(self):
        self.cfg.write_text(json.dumps({"exe": "x"}), encoding="utf-8")
        self.assertEqual(self.get()[0], 404)
        (Path(self.dir.name) / "learned.bin").write_bytes(profile(1, 3, [(0, 2)]))
        self.cfg.write_text(json.dumps({"exe": "x", "expert_profile_save": str(Path(self.dir.name) / "learned.bin")}), encoding="utf-8")
        code, body = self.get()
        self.assertEqual((code, body["rank"], body["every_min"]), (200, [-1, -1, 0], 10))


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class Page(unittest.TestCase):
    def test_helpers(self):
        script = ("const X = require(process.argv[1]);\n"
                  "const out = {top: X.shade(0, 10), last: X.shade(9, 10), none: X.shade(-1, 10), cell: X.cellAt(150, 9, 300, 16, 2, 3),\n"
                  "  outside: X.cellAt(300, 0, 300, 16, 2, 3), saved: X.savedLine({saved_at: 0, every_min: 10}, () => '10:00'),\n"
                  "  first: X.savedLine({saved_at: null, every_min: 10}, () => ''), quit: X.savedLine({saved_at: 0, every_min: 0}, () => '10:00'),\n"
                  "  tip: X.tipFor({experts: 3, ranked: 4, rank: [3, -1, 0, 1, 2, -1]}, {layer: 0, expert: 2}),\n"
                  "  none_tip: X.tipFor({experts: 3, ranked: 4, rank: [3, -1, 0, 1, 2, -1]}, {layer: 0, expert: 1})};\n"
                  "console.log(JSON.stringify(out));")
        done = subprocess.run([NODE, "-e", script, str(WEB / "experts.js")], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        out = json.loads(done.stdout)
        self.assertEqual((out["top"], round(out["last"], 2), out["none"]), (1, 0.08, 0))
        self.assertEqual(out["cell"], {"layer": 1, "expert": 1})
        self.assertIsNone(out["outside"])
        self.assertEqual(out["saved"], "Saved by the engine at 10:00. It saves every 10 minutes and when it stops.")
        self.assertEqual(out["first"], "It saves every 10 minutes and when it stops.")
        self.assertEqual(out["quit"], "Saved by the engine at 10:00.")
        self.assertEqual(out["tip"], "Layer 0, expert 2: rank 1 of 4")
        self.assertEqual(out["none_tip"], "", "a pair the file does not rank gets no rank")

    def test_the_tab_shows_only_when_the_model_saves_its_profile(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        self.assertRegex(html, r'id="tab-btn-experts"[^>]*data-tab="experts" hidden>')
        self.assertLess(html.index("web/experts.js"), html.index("web/app.js"))
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn('$("tab-btn-experts").hidden = !on;', app)
        self.assertIn('if (tab === "experts") StrataExperts.start($("view-experts")); else StrataExperts.stop();', app)


if __name__ == "__main__":
    unittest.main()
