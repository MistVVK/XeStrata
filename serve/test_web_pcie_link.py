"""The Monitor tab's PCIe card on a machine with several cards (#4623 / #5801 rule, 11-16): the server keeps the top-level gpu_pcie_* fields as
card 0's and names the card with the slowest link in gpu_pcie_bottleneck (all its fields from that one card). The card shows that link; one
card, or a server from before the field, shows the top-level fields as before. Runs app.js's own pcieLink() under node.

    python -m pytest serve/test_web_pcie_link.py
"""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")


def function_source(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class PcieLinkCard(unittest.TestCase):
    def link(self, hw):
        src = function_source((WEB / "app.js").read_text(encoding="utf-8"), "pcieLink")
        out = subprocess.run([NODE, "-e", f"{src}\nprocess.stdout.write(JSON.stringify(pcieLink({json.dumps(hw)})));"],
                             capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        return json.loads(out.stdout)

    def test_several_cards_show_the_slowest_link_from_one_card(self):
        hw = {"gpu_pcie_gen": 4, "gpu_pcie_gen_max": 4, "gpu_pcie_width": 16,                                   # card 0's, kept top level
              "gpu_pcie_bottleneck": {"card": 1, "gen": 1, "gen_max": 3, "width": 8, "width_max": 16}}         # card 1, idle at Gen1
        self.assertEqual(self.link(hw), {"gen": 3, "cur": 1, "width": 8, "card": 1})

    def test_one_card_or_an_older_server_shows_the_top_level_fields(self):
        self.assertEqual(self.link({"gpu_pcie_gen": 2, "gpu_pcie_gen_max": 4, "gpu_pcie_width": 16}), {"gen": 4, "cur": 2, "width": 16, "card": None})
        self.assertEqual(self.link({"gpu_pcie_gen": 4, "gpu_pcie_width": 16, "gpu_pcie_bottleneck": None}), {"gen": 4, "cur": 4, "width": 16, "card": None})

    def test_a_bottleneck_without_a_width_or_generation_is_not_used(self):
        hw = {"gpu_pcie_gen": 4, "gpu_pcie_width": 16, "gpu_pcie_bottleneck": {"card": 1, "gen": None, "gen_max": None, "width": 8}}
        self.assertEqual(self.link(hw)["width"], 16)

    def test_the_card_uses_pcie_link_for_its_value_and_idle_note(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("const link = pcieLink(hw), gen = link.gen;", js)
        self.assertIn('setMetric("pcie", gen ? `Gen${gen}` : null, link.width ? `x${link.width}` : "",', js)
        self.assertIn("(link.cur && gen && link.cur < gen ? ` · idle Gen${link.cur}` : \"\")", js)


if __name__ == "__main__":
    unittest.main()
