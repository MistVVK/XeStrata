"""Every distribution library an engine package bundles on purpose (collect_libs.py --bundle-system, and its built-in list) has its notice in
tools/packaging/notices/<name>.txt, and a script that bundles one also packs the notice.

    python -m unittest tools.test_bundle_system_notices -v

The 0.2.16 Intel Arc package bundles libz.so.1 (libur_loader.so.0 needs it, binary_deps.py no longer lists it as a base library); without the
notice file the copy carried no licence text, and without `--include 'licenses/*'` the notice never reached the package (package_engine.py without
--tree packs top-level files only). Nothing is built here.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # terminal/
PACK = ROOT.parent / "tools" / "packaging"
NOTICES = PACK / "notices"
sys.path.insert(0, str(PACK))
import collect_libs  # noqa: E402

BUNDLE_SYSTEM = re.compile(r"""--bundle-system\s+(?:'([^']+)'|"([^"]+)"|([^\s'"\\]+))""")


def notice_of(name: str) -> Path:
    return NOTICES / (name.split(".so", 1)[0] + ".txt")


def scripts():
    for p in sorted(PACK.iterdir()):
        if p.suffix in (".sh", ".bat", ".ps1") or (p.suffix == ".py" and p.name != "collect_libs.py"):
            yield p


class BundledSystemLibraries(unittest.TestCase):
    def test_the_built_in_list_has_its_notices(self):
        self.assertTrue(collect_libs.BUNDLE_FROM_SYSTEM)
        for name in collect_libs.BUNDLE_FROM_SYSTEM:
            with self.subTest(name=name):
                self.assertTrue(notice_of(name).is_file(), name)

    def test_every_bundle_system_name_in_a_package_script_has_a_notice(self):
        seen = 0
        for p in scripts():
            for m in BUNDLE_SYSTEM.finditer(p.read_text(encoding="utf-8", errors="replace")):
                name = next(g for g in m.groups() if g)
                seen += 1
                with self.subTest(script=p.name, library=name):
                    self.assertTrue(notice_of(name).is_file(), "%s bundles %s: tools/packaging/notices/%s is missing" % (p.name, name, notice_of(name).name))
        self.assertGreaterEqual(seen, 1)           # build_sycl_linux.sh names libz.so.1

    def test_a_script_that_bundles_one_packs_the_notices_too(self):
        for p in scripts():
            text = p.read_text(encoding="utf-8", errors="replace")
            if BUNDLE_SYSTEM.search(text) and "package_engine.py" in text and "--tree" not in text:
                with self.subTest(script=p.name):
                    self.assertIn("--include 'licenses/*'", text)

    def test_the_notices_are_text_that_names_their_library(self):
        for n in sorted(NOTICES.glob("*.txt")):
            with self.subTest(notice=n.name):
                text = n.read_text(encoding="utf-8")
                self.assertTrue(text.startswith(n.stem + ".so"), n.name)
                self.assertIn("License", text)


if __name__ == "__main__":
    unittest.main()
