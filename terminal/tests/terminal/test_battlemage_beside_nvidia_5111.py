"""#5111: an Arc B-series (Battlemage) card beside an NVIDIA card, chosen with STRATA_ENGINE_BACKEND=sycl, starts without a refusal.

The report: an Arc Pro (Battlemage) card and an RTX 3090 in one Linux PC, `STRATA_ENGINE_BACKEND=sycl strata launch`. Engine 0.1.51 / 0.1.54
started; 0.1.56 (Strata 0.2.10) refused. 0.2.10's Arc reader (strata_cli/launch/catalog.py intel_arc_cards) read card0..card9 only, skipped
every id that was not in its table, and gave up on the whole list at the first entry without a vendor file (a firmware framebuffer, card0 on
many distributions). The Arc then was not a card at all, the NVIDIA card was the only one, and a preset naming card 0 was refused as "an
NVIDIA card ... this build of Strata drives Intel cards only".

Every case here is the launcher's own path (Catalog.computer, apply_defaults, the roles line) on a temporary /sys/class/drm tree, for the
Battlemage ids in the table and ones that are not in it, beside an RTX 3090, with and without a framebuffer entry in front. Measured on a
B-series card: never (none available); the Intel engine itself is not started here."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import catalog as C

INTEL, NVIDIA = "0x8086", "0x10de"
REAL_ARC = C.intel_arc_cards
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "usable": True, "vendor": "nvidia"}

#: Battlemage ids: in the table (B580, B570, Pro B60, Pro B50, Pro B60/B65/B70) and not in it (other BMG-G21 / BMG-G31 parts).
LISTED = ("0xe20b", "0xe20c", "0xe211", "0xe212", "0xe221", "0xe222", "0xe223")
UNLISTED = ("0xe202", "0xe209", "0xe20d", "0xe210", "0xe216", "0xe220")


def build(root: Path, cards) -> None:
    """cards: (directory name, vendor or None, device id or None). A vendor of None is an entry without a vendor file (a framebuffer)."""
    for name, vendor, did in cards:
        dev = root / name / "device"
        dev.mkdir(parents=True)
        if vendor:
            (dev / "vendor").write_text(vendor + "\n", encoding="utf-8")
        if did:
            (dev / "device").write_text(did + "\n", encoding="utf-8")


def layouts(did: str):
    return {"nvidia first": [("card0", NVIDIA, "0x2204"), ("card1", INTEL, did)],
            "arc first": [("card0", INTEL, did), ("card1", NVIDIA, "0x2204")],
            "framebuffer in front": [("card0", None, None), ("card1", NVIDIA, "0x2204"), ("card2", INTEL, did)]}


class BattlemageBesideNvidia(unittest.TestCase):
    def launch(self, cards, gpus):
        """(engine make, usable card names, ignored card names, error lines, roles line) of one non-interactive launch plan."""
        from strata_cli.launch import cmd, settings as S
        from tests.terminal.launch_support import Home
        with tempfile.TemporaryDirectory() as d, Home():
            build(Path(d), cards)
            cat = C.Catalog()
            hw = {"ram_gb": 64.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": [dict(RTX)], "disk_free_gb": 500}
            fake = lambda root="/sys/class/drm": REAL_ARC(d)       # noqa: E731
            with mock.patch.dict(os.environ, {"STRATA_ENGINE_BACKEND": "sycl"}), mock.patch.object(cat.strata, "hardware", return_value=hw), \
                    mock.patch.object(C, "intel_arc_cards", fake), mock.patch("strata_control.engine_component.status", return_value={"usable": False}):
                comp = cat.computer()
                s, notes = cmd.apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpus=gpus), cat)
                return (comp.engine_vendor, [g["name"] for g in comp.usable_gpus], [g["name"] for g in comp.ignored_gpus],
                        [n.line() for n in notes if n.level == "error"], cmd.roles_line(s, cat) or "")

    def check(self, dids):
        for did in dids:
            for how, cards in layouts(did).items():
                for gpus in (None, [0]):
                    with self.subTest(id=did, layout=how, gpus=gpus):
                        vendor, usable, ignored, errors, roles = self.launch(cards, gpus)
                        self.assertEqual(vendor, "intel")
                        self.assertEqual(len(usable), 1, usable)
                        self.assertTrue(usable[0].startswith("Intel Arc"), usable)
                        self.assertIn("NVIDIA GeForce RTX 3090", ignored)
                        self.assertEqual(errors, [], "a person who chose the Intel engine is never refused")
                        self.assertNotIn("3090", roles, "the NVIDIA card never takes a role in an Intel start")

    def test_listed_battlemage_ids(self):
        self.check(LISTED)

    def test_battlemage_ids_not_in_the_table(self):
        self.check(UNLISTED)

    def test_an_unlisted_battlemage_counts_above_the_8_gb_off_rule(self):
        with tempfile.TemporaryDirectory() as d:
            build(Path(d), [("card0", INTEL, "0xe210")])
            (card,) = REAL_ARC(d)
            self.assertGreater(card["vram_gb"], 8.0, "a card of 8 GB or less starts Off (gpu_roles.OFF_AT_OR_BELOW_GB)")


if __name__ == "__main__":
    unittest.main()
