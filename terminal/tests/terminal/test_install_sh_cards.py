"""P10 (0.2.16): install.sh names every graphics card once, with clean names.

Before: "NVIDIA NVIDIA RTX A4000" (the word twice), one card named on a two-card computer, and the R9700 as the raw lspci line
"Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)".
"""
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SH = Path(__file__).resolve().parents[3] / "tools" / "packaging" / "install.sh"
SHELL = shutil.which("sh")


def helpers() -> str:
    m = re.search(r"# <cards>\n(.*?)# </cards>", SH.read_text(encoding="utf-8"), re.S)
    assert m, "the card helpers' markers are missing from install.sh"
    return m.group(1)


def names(lines, nvidia=None):
    """Run the helpers the way install.sh does: lspci lines through pci_card, or nvidia-smi names through with_maker; return the list."""
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "t.sh"
        data = Path(d) / "in.txt"
        data.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        if nvidia is not None:
            body = 'while IFS= read -r n; do if [ -n "$n" ]; then with_maker NVIDIA "$n"; fi; done < "$1" | list_cards'
        else:
            body = 'while IFS= read -r line; do pci_card "$line"; done < "$1" | list_cards'
        script.write_text("set -eu\n" + helpers() + body + "\n", encoding="utf-8", newline="\n")
        done = subprocess.run([SHELL, str(script), str(data)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@unittest.skipUnless(SHELL, "needs sh")
class Names(unittest.TestCase):
    def test_nvidia_smi_names_carry_the_maker_once(self):
        self.assertEqual(names(["NVIDIA RTX A4000"], nvidia=True), "NVIDIA RTX A4000")
        self.assertEqual(names(["Tesla P100-PCIE-16GB"], nvidia=True), "NVIDIA Tesla P100-PCIE-16GB")
        self.assertEqual(names(["NVIDIA GeForce RTX 4090"], nvidia=True), "NVIDIA GeForce RTX 4090")
        self.assertEqual(names(["Quadro RTX 5000"], nvidia=True), "NVIDIA Quadro RTX 5000")

    def test_two_equal_cards_are_counted_and_different_cards_are_all_named(self):
        self.assertEqual(names(["NVIDIA RTX A4000", "NVIDIA RTX A4000"], nvidia=True), "2 x NVIDIA RTX A4000")
        self.assertEqual(names(["NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000"], nvidia=True),
                         "2 x NVIDIA RTX A6000, NVIDIA GeForce RTX 3090")

    def test_the_r9700_is_named_from_its_id_when_pciutils_does_not_know_it(self):
        self.assertEqual(names(["Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)"]), "AMD Radeon AI PRO R9700")
        self.assertEqual(names(["Advanced Micro Devices, Inc. [AMD/ATI] Device 7550 (rev c0)"]), "AMD Radeon RX 9070 series")

    def test_an_unknown_id_still_gives_a_plain_name(self):
        self.assertEqual(names(["Advanced Micro Devices, Inc. [AMD/ATI] Device 9999 (rev 01)"]),
                         "AMD graphics card (PCI device 9999)")

    def test_the_product_name_in_brackets_is_used(self):
        self.assertEqual(names(["Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX] (rev c8)"]),
                         "AMD Radeon RX 7900 XT/7900 XTX")
        self.assertEqual(names(["Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1 [Radeon 780M] (rev c1)"]), "AMD Radeon 780M")
        self.assertEqual(names(["Intel Corporation DG2 [Arc A750] (rev 08)"]), "Intel Arc A750")
        self.assertEqual(names(["NVIDIA Corporation GA102GL [RTX A6000] (rev a1)"]), "NVIDIA RTX A6000")
        self.assertEqual(names(["NVIDIA Corporation GP100GL [Tesla P100 PCIe 16GB] (rev a1)"]), "NVIDIA Tesla P100 PCIe 16GB")

    def test_four_cards_and_a_management_chip(self):
        lines = ["Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)"] * 4 + [
            "ASPEED Technology, Inc. ASPEED Graphics Family (rev 41)"]
        self.assertEqual(names(lines), "4 x AMD Radeon AI PRO R9700")

    def test_no_card_gives_nothing(self):
        self.assertEqual(names(["ASPEED Technology, Inc. ASPEED Graphics Family (rev 41)"]), "")


@unittest.skipUnless(SHELL and sys.platform.startswith("linux"), "needs sh on Linux (uname must say Linux)")
class Printed(unittest.TestCase):
    """The line the person reads, with a fake nvidia-smi or lspci first on PATH and a feed that does not answer (the line is printed before)."""

    def run_install(self, **tools):
        with tempfile.TemporaryDirectory() as d:
            bindir = Path(d) / "bin"
            bindir.mkdir()
            for name, text in tools.items():
                p = bindir / name
                p.write_text("#!/bin/sh\n" + text + "\n", encoding="utf-8", newline="\n")
                p.chmod(p.stat().st_mode | stat.S_IEXEC)
            for need in ("curl", "sha256sum", "tar", "sed", "awk", "grep", "tr", "cut", "uname", "mktemp", "rm", "head", "cat", "dirname"):
                found = shutil.which(need)
                if found and not (bindir / need).exists():
                    os.symlink(found, bindir / need)
            home = Path(d) / "home"
            home.mkdir()
            env = {"PATH": str(bindir), "HOME": str(home), "STRATA_FEED": "http://127.0.0.1:9", "STRATA_DATA_DIR": str(home / "data"),
                   "STRATA_BIN_DIR": str(home / "bin")}
            done = subprocess.run([SHELL, str(SH)], capture_output=True, text=True, timeout=120, env=env)
        return next((l for l in done.stdout.splitlines() if "This computer:" in l), done.stdout + done.stderr)

    def test_two_a4000(self):
        line = self.run_install(**{"nvidia-smi": "printf 'NVIDIA RTX A4000\\nNVIDIA RTX A4000\\n'"})
        self.assertIn("graphics: 2 x NVIDIA RTX A4000", line)
        self.assertNotIn("NVIDIA NVIDIA", line)

    def test_r9700_from_lspci(self):
        line = self.run_install(lspci="echo '03:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)'")
        self.assertIn("graphics: AMD Radeon AI PRO R9700", line)


if __name__ == "__main__":
    unittest.main()
