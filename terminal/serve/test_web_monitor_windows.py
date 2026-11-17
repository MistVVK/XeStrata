"""#4968 (PR by noahark, revised): the Monitor tab's model card, the Speed card's min and max, and the power / temperature averages.
The min and max cover the newest 500 requests, as many as the server keeps, so an extreme the server has let go of is gone from the
page too; the averages weigh each poll by the seconds since the one before it (at most 2), leave idle time out, keep a missing sensor
out of its own average and start afresh with a new totals window or another model. Tooltips are plain English title= text (the tab is
English-only): no tooltip box of the page's own and no catalog keys. Runs app.js's own functions under node with stand-ins.

    python -m pytest serve/test_web_monitor_windows.py
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")


def block(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a)]


def run(script: str):
    # the script goes in a file: a Windows command line is too short for the 600-request fixture
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "monitor_windows.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    return json.loads(out.stdout)


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class MonitorWindows(unittest.TestCase):
    def setUp(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.code = ("const els = {}; const $ = (id) => (els[id] ||= {id, textContent: '', title: ''});\n"
                     "const fmt = (n, d = 0) => (n == null ? '–' : Number(n).toFixed(d));\n"
                     + block(app, "const SPEED_KEPT", "async function pullSpeedWindow")
                     + block(app, "function fillSpeedRange", "\n}\n") + "\n}\n"
                     + block(app, "const HW_GAP_S", "\nfunction renderMonitor("))

    def test_min_and_max_drop_an_extreme_the_server_no_longer_keeps(self):
        # 600 requests, oldest first; the slowest (1 t/s) is request 50, the fastest (999 t/s) request 550
        rows = [{"id": f"r{i}", "time": 1000 + i, "decode_tok_s": 1 if i == 50 else 999 if i == 550 else 20 + i % 7,
                 "prompt_tokens": 1000, "reused": 200, "prompt_ms": 400} for i in range(600)]
        got = run(self.code + f"""
let m = new Map(); for (const r of {json.dumps(rows)}) m.set(r.id, r);
m = keepNewest(m, SPEED_KEPT); speedWin = m; fillSpeedRange();
process.stdout.write(JSON.stringify({{size: m.size, oldest: Math.min(...[...m.values()].map(r => r.time)), range: speedRange([...m.values()]),
  text: $('sr-speed').textContent, title: $('sr-speed').title, prefill: $('sr-prefill').textContent}}));""")
        self.assertEqual(got["size"], 500)
        self.assertEqual(got["oldest"], 1100)
        self.assertEqual(got["range"]["decode"], {"min": 20, "max": 999, "n": 500})
        self.assertEqual(got["text"], "min 20.0 · max 999.0")
        self.assertEqual(got["title"], "The slowest and the fastest of the last 500 requests. The server keeps up to 500.")
        self.assertEqual(got["prefill"], "min 2000 · max 2000")            # 800 read tokens in 0.4 s

    def test_averages_weigh_time_cap_gaps_and_leave_idle_and_missing_sensors_out(self):
        # polls (seconds, working, power W, temps): one second apart, then a 60 s gap (a background tab), an idle stretch, a card that
        # stops reporting its temperature
        polls = [(0, True, 100, [40, 50]), (1, True, 100, [40, 50]), (2, True, 200, [60, None]), (62, True, 300, [70, 70]),
                 (63, False, 900, [99, 99]), (64, False, 900, [99, 99]), (65, True, None, [80, 80])]
        got = run(self.code + f"""
let now = 0; Date.now = () => now * 1000;
const out = [];
for (const [t, working, p, temps] of {json.dumps(polls)}) {{
  now = t; sampleHw({{gpu_power: p, gpus: temps.map((x, i) => ({{index: i, temp: x, name: 'GPU ' + i}}))}}, working, '1|m');
  out.push([$('sr-power').textContent, $('sr-temp').textContent]);
}}
process.stdout.write(JSON.stringify({{out, avg: hwAverages(hwWin), power: $('sr-power').title, temp: $('sr-temp').title}}));""")
        # weights: 1 (first poll), 1, 1, 2 (60 s gap capped), idle polls out, 1 (the poll after an idle one)
        avg = got["avg"]
        self.assertEqual(avg["seconds"], 6)
        self.assertAlmostEqual(avg["power"], (100 + 100 + 200 + 2 * 300) / 5)    # the last sample has no power reading
        self.assertAlmostEqual(avg["temps"][0], (40 + 40 + 60 + 2 * 70 + 80) / 6)
        self.assertAlmostEqual(avg["temps"][1], (50 + 50 + 2 * 70 + 80) / 5)       # its missing reading is left out, not counted as 0
        self.assertEqual(got["out"][4], got["out"][3])                             # idle polls change nothing
        self.assertEqual(got["power"], "Average power while the server was working, over the 6 s this page has watched. Idle time is left out.")
        self.assertTrue(got["temp"].startswith("GPU 0: 60 °C · GPU 1: 64 °C\nAverage temperature of each card while the server was working, over the 6 s"))

    def test_a_new_totals_window_or_another_model_starts_afresh(self):
        got = run(self.code + """
let now = 0; Date.now = () => now * 1000;
const hw = (p) => ({gpu_power: p, gpus: [{index: 0, temp: 50, name: 'A'}]});
const seen = [];
for (const [t, p, key] of [[0, 100, '1|a'], [1, 100, '1|a'], [2, 300, '1|b'], [3, 300, '1|b'], [4, 500, '2|b']]) {
  now = t; sampleHw(hw(p), true, key); seen.push($('sr-power').textContent);
}
process.stdout.write(JSON.stringify(seen));""")
        self.assertEqual(got, ["avg 100 W", "avg 100 W", "avg 300 W", "avg 300 W", "avg 500 W"])

    def test_no_samples_and_no_cards_show_nothing(self):
        got = run(self.code + """
sampleHw({gpus: []}, true, 'k'); sampleHw({gpu_power: null, gpus: []}, false, 'k');
speedWin = new Map(); fillSpeedRange();
process.stdout.write(JSON.stringify([$('sr-power').textContent, $('sr-temp').textContent, $('sr-speed').textContent, $('sr-speed').title]));""")
        self.assertEqual(got, ["", "", "", ""])

    def test_tooltips_are_english_title_text_and_no_catalog_keys(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        css = (WEB / "app.css").read_text(encoding="utf-8")
        self.assertNotIn("data-tip", app)
        self.assertNotIn("#st-tip", css)
        self.assertNotIn('tr("monitor.', app)
        for text in ["Which model this server runs. Apps ask for it by this exact name.",
                     "The context window this model runs with, and the engine's version.",
                     "Average answer speed since the server started: answer tokens divided by the time spent writing them. Waiting in the queue and reading the prompt are not counted.",
                     "Average prompt speed since the server started, counting only the tokens that were really read. What came from the cache is left out.",
                     "The average prompt and answer length of one request.",
                     "How much of each prompt came from the engine's cache instead of being read again. High means follow-ups are fast.",
                     "When this server began counting, and how many requests it has served. Loading or unloading the model does not reset it.",
                     "Averages start with the first request."]:
            self.assertIn(text, app)
        for f in (WEB / "locales").glob("*.json"):
            self.assertFalse([k for k in json.loads(f.read_text(encoding="utf-8")) if k.startswith("monitor.")], f.name)

    def test_the_model_card_is_first_and_has_its_elements(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        metrics = block(app, "const METRICS = [", "];")
        self.assertLess(metrics.index('key: "model"'), metrics.index('key: "speed"'))
        for el in ('id="mv-model"', 'id="ms-model"', 'id="model-facts"', 'id="sr-speed"', 'id="sr-prefill"', 'id="sr-${m.key}"'):
            self.assertIn(el, app)


if __name__ == "__main__":
    unittest.main()
