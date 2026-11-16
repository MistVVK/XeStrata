"""#5457: the web Monitor's GPU temp, Power and PCIe cards say why they have no reading (the server's gpu_note, as the
card's line "not available" and its tooltip) also when the GPU has a name - a Mac always has one - and About shows the
reason beside the name. serve/web/app.js is run in node with stand-ins for the page's helpers (no browser)."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")
MAC_NOTE = ("no GPU temperature or power reading without root (powermetrics), and an Apple Silicon GPU sits on the "
            "die, so there is no PCIe link to show; the engine's own figures are in its log")


def run_node(source: str) -> object:
    done = subprocess.run([NODE, "-e", source], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    return json.loads(done.stdout)


def function_source(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    return text[start:text.index("\n}\n", start) + 3]


def gpu_cards_block(text: str) -> str:
    """renderMonitor's lines from the GPU load card to the PCIe card's sparkline."""
    start = text.index("  const per = (f) =>")
    return text[start:text.index('  spark("sp-pcie"', start)]


STUBS = """
const els = {};
const $ = (id) => (els[id] ||= {id, textContent: "", innerHTML: "", title: ""});
const esc = (s) => String(s);
const fmt = (n, d = 0) => (n == null ? "–" : Number(n).toFixed(d));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));
const spark = () => {};
"""


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class GpuNote(unittest.TestCase):
    def cards(self, hw, st, eng=None):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        script = (STUBS + function_source(app, "setMetric") +
                  f"function run(hw, st, eng, h) {{\n{gpu_cards_block(app)}\n}}\n"
                  f"run({json.dumps(hw)}, {json.dumps(st)}, {json.dumps(eng or {})}, {{}});\n"
                  "console.log(JSON.stringify(Object.fromEntries(['temp', 'power', 'pcie', 'gpu'].map(k => "
                  "[k, [$('mv-' + k).innerHTML, $('ms-' + k).textContent, $('ms-' + k).title]]))));")
        return run_node(script)

    MAC_HW = {"gpu_util": 95, "gpu_mem_used": 88_000_000_000, "gpu_mem_total": 131_000_000_000, "gpu_temp": None,
              "gpu_power": None, "gpus": [{"index": 0}]}
    MAC_ST = {"gpu_name": "Apple M5 Max", "gpu_note": MAC_NOTE}

    def test_a_named_mac_gpu_explains_its_three_empty_cards(self):
        cards = self.cards(self.MAC_HW, self.MAC_ST)
        for key in ("temp", "power", "pcie"):
            value, line, tip = cards[key]
            self.assertEqual(value, "–", key)
            self.assertEqual(line, "not available", key)
            self.assertEqual(tip, MAC_NOTE, key)
        self.assertEqual(cards["gpu"][1], "Apple M5 Max", "the load card keeps the GPU's name")
        self.assertEqual(cards["gpu"][2], "", "and no reason: it has a reading")

    def test_a_card_with_a_reading_or_no_note_says_nothing_more(self):
        nvidia = {"gpu_util": 50, "gpu_temp": 61, "gpu_power": 210, "gpu_power_limit": 450, "gpu_pcie_gen": 4,
                  "gpu_pcie_width": 16, "gpu_pcie_rx_mb": 12.5, "gpus": [{"index": 0}]}
        cards = self.cards(nvidia, {"gpu_name": "NVIDIA GeForce RTX 4090"})
        self.assertEqual(cards["temp"], ["61<small>°C</small>", "", ""])
        self.assertEqual(cards["power"][1:], ["of 450 W limit", ""])
        self.assertEqual(cards["pcie"][2], "")
        quiet = self.cards({"gpu_temp": None, "gpu_power": None, "gpus": [{"index": 0}]}, {"gpu_name": "Some GPU"})
        self.assertEqual(quiet["temp"], ["–", "", ""], "no note: nothing invented")

    def test_about_shows_the_reason_beside_a_named_gpu(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        script = (STUBS + "const rows = {}; const facts = (el, r) => { rows[el.id] = r; }; const projectionText = () => null;\n"
                  "globalThis.location = {origin: 'http://127.0.0.1:8080'};\n" + function_source(app, "renderAbout") +
                  f"renderAbout({{}}, {json.dumps(self.MAC_HW)}, {json.dumps(self.MAC_ST)});\n"
                  "console.log(JSON.stringify(rows['facts-hw'].find(r => r[0] === 'GPU')[1]));")
        self.assertEqual(run_node(script), f"Apple M5 Max, 122 GB · {MAC_NOTE}")


if __name__ == "__main__":
    unittest.main()
