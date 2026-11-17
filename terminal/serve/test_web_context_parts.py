"""#4636 (PR by thisisspartan, revised): the Monitor's Context fill gauge in parts - Cached (the RESUME prefix), Reading, Answer, Free -
with the row of what the server does now in bold and a queued request said in words. Runs app.js's own contextParts() and
renderContext() under node with stand-ins (the PR's browser test needs puppeteer-core, which no test runner has).

    python -m pytest serve/test_web_context_parts.py
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")


def function_source(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    return text[start:text.index("\n}\n", start) + 3]


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class ContextParts(unittest.TestCase):
    def run_js(self, body: str):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        script = ("const els = {}; const $ = (id) => (els[id] ||= {id, textContent: '', hidden: false, dataset: {}, style: {}, attrs: {},\n"
                  "  setAttribute(k, v) { this.attrs[k] = v; }, classList: {set: new Set(), toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); }}});\n"
                  "const kfmt = (n) => String(n); const ctxfmt = (n) => String(n);\n"
                  + function_source(app, "contextParts") + function_source(app, "renderContext") + body)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_parts.js"
            path.write_text(script, encoding="utf-8")
            out = subprocess.run([NODE, str(path)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        return json.loads(out.stdout)

    def parts(self, live, eng, last=None, on=None):
        return self.run_js(f"process.stdout.write(JSON.stringify(contextParts({json.dumps(live)}, {json.dumps(eng)}, {json.dumps(last)}, "
                           f"{json.dumps(on if on is not None else live.get('state'))})));")

    def test_reading_counts_the_cached_prefix_apart_from_what_is_read(self):
        c = self.parts({"state": "reading", "prompt_tokens": 10000, "reused": 4000, "prompt_read": 6500}, {"max_context": 32768})
        self.assertEqual((c["cached"], c["read"], c["answer"], c["used"], c["free"]), (4000, 2500, 0, 6500, 26268))
        self.assertEqual((c["now"], c["queued"], c["tone"]), ("read", False, ""))

    def test_reading_with_no_chunk_yet_and_an_engine_without_resume(self):
        c = self.parts({"state": "reading", "prompt_tokens": 10000, "reused": 0, "prompt_read": None}, {"max_context": 32768})
        self.assertEqual((c["cached"], c["read"]), (0, 0))

    def test_generating_has_the_whole_prompt_in_and_the_answer_so_far(self):
        c = self.parts({"state": "generating", "prompt_tokens": 10000, "reused": 9000, "generated": 300}, {"max_context": 16384})
        self.assertEqual((c["cached"], c["read"], c["answer"], c["used"]), (9000, 1000, 300, 10300))
        self.assertEqual(c["now"], "answer")

    def test_queued_lights_no_row_and_says_so(self):
        c = self.parts({"state": "idle", "queued": 1}, {"max_context": 16384, "gen": 2}, {"prompt_tokens": 100, "output_tokens": 20, "gen": 2}, "queued")
        self.assertEqual((c["now"], c["queued"]), (None, True))
        self.assertEqual(c["cached"], 120)

    def test_idle_shows_the_last_request_as_cached_only_on_the_same_engine_process(self):
        last = {"prompt_tokens": 5000, "output_tokens": 500, "gen": 3}
        self.assertEqual(self.parts({"state": "idle"}, {"max_context": 16384, "gen": 3}, last)["cached"], 5500)
        self.assertEqual(self.parts({"state": "idle"}, {"max_context": 16384, "gen": 4}, last)["cached"], 0)     # restarted: cache gone
        self.assertEqual(self.parts({"state": "idle"}, {"max_context": 16384}, {"prompt_tokens": 5000, "output_tokens": 500})["cached"], 5500)

    def test_warn_from_85_percent_and_danger_only_past_the_window(self):
        self.assertEqual(self.parts({"state": "generating", "prompt_tokens": 840, "generated": 0}, {"max_context": 1000})["tone"], "")
        self.assertEqual(self.parts({"state": "generating", "prompt_tokens": 850, "generated": 0}, {"max_context": 1000})["tone"], "warn")
        over = self.parts({"state": "generating", "prompt_tokens": 900, "generated": 300}, {"max_context": 1000})
        self.assertEqual((over["tone"], over["pct"], over["free"]), ("danger", 100, 0))
        self.assertAlmostEqual(over["scale"], 1000 / 1200)

    def test_an_unknown_window_shows_a_dash(self):
        c = self.parts({"state": "reading", "prompt_tokens": 10, "prompt_read": 5}, {"max_context": 0})
        self.assertEqual((c["pct"], c["tone"]), (None, ""))

    def test_the_gauge_draws_the_parts_end_to_end_and_bolds_the_row_of_now(self):
        got = self.run_js("""renderContext({state: "generating", prompt_tokens: 600, reused: 200, generated: 200}, {max_context: 1000}, null, "generating");
const seg = (k) => [Number(els['ctx-seg-' + k].attrs['stroke-dasharray'].split(' ')[0]), Number(els['ctx-seg-' + k].attrs['stroke-dashoffset'])];
process.stdout.write(JSON.stringify({cached: seg('cached'), read: seg('read'), answer: seg('answer'), pct: els['ctx-pct'].textContent,
  sub: els['ctx-sub'].textContent, now: ['cached', 'read', 'answer', 'free'].filter(k => els['ctx-row-' + k].classList.set.has('is-now')),
  legend: ['cached', 'read', 'answer', 'free'].map(k => els['ctx-leg-' + k].textContent), note: els['ctx-note'].hidden}));""")
        self.assertEqual(got["cached"], [47.1, -0.0])
        self.assertEqual(got["read"], [94.2, -47.1])
        self.assertEqual(got["answer"], [47.1, -141.4])
        self.assertEqual((got["pct"], got["sub"], got["now"], got["legend"], got["note"]), ("80%", "800 / 1000", ["answer"], ["200", "400", "200", "200"], True))

    def test_the_legend_and_its_words(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        for k, label in (("cached", "Cached"), ("read", "Reading"), ("answer", "Answer"), ("free", "Free")):
            self.assertIn(f'id="ctx-row-{k}"', html)
            self.assertIn(f"<span>{label}</span>", html)
        for tip in ("Prompt tokens the engine already had in its cache, so they were not read again.", "Prompt tokens being read now.",
                    "Tokens of the answer written so far.", "Room left in this model&rsquo;s context window."):
            self.assertIn(f'title="{tip}"', html)
        self.assertIn(">Waiting in the queue</p>", html)


if __name__ == "__main__":
    unittest.main()
