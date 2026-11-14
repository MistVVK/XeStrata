"""The release manifest (docs/data/release.json) and the pages that quote it."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_manifest as rm  # noqa: E402


class ReleaseManifest(unittest.TestCase):
    def test_current_pages_agree_with_the_manifest(self):
        problems = rm.check(rm.load())
        self.assertEqual(problems, [], "\n".join(problems))

    def test_old_installer_names_are_found_and_rewritten(self):
        m = rm.load()
        old = "run `Strata-0.2.9-windows-x64-setup.exe` or Strata-0.2.9-linux-x86_64.AppImage.sig"
        self.assertEqual(len(rm.INSTALLER_RE.findall(old)), 2)
        fixed = rm.INSTALLER_RE.sub(lambda x: x.group(0).replace(x.group(1), m["product_version"], 1), old)
        self.assertNotIn("0.2.9", fixed)
        self.assertIn("Strata-%s-linux-x86_64.AppImage.sig" % m["product_version"], fixed)

    def test_version_lines_are_rewritten_and_a_second_pass_changes_nothing(self):
        m = rm.load()
        page, regex, keys = [r for r in rm.LINES if r[0] == "AGENTS.md"][0]
        text = "Version 0.2.5, engine 0.1.50. The public model families"
        once = rm.rewrite(text, regex, keys, m)
        self.assertEqual(once, "Version %s, engine %s. The public model families" % (m["product_version"], m["engine_version"]))
        self.assertEqual(rm.rewrite(once, regex, keys, m), once)

    def test_a_drifting_manifest_is_reported(self):
        m = copy.deepcopy(rm.load())
        m["product_version"] = "0.2.99"
        m["tag"] = "v0.2.99"
        m.pop("terminal_version", None)                             # the Terminal then has to say the product version
        problems = "\n".join(rm.check(m))
        self.assertIn("strata_cli/__init__.py says", problems)
        self.assertIn("asset list has no Strata-0.2.99-windows-x64-setup.exe", problems)

    def test_a_control_only_hotfix_is_allowed_and_a_drifting_control_is_not(self):
        """0.2.13.1: the Terminal and Control are one hotfix part ahead of the app and the engines; nothing else may differ.
        Written so it holds for a release with a hotfix part (0.2.13.1) and without one (0.2.14)."""
        m = rm.load()
        self.assertEqual(rm.check(m), [])
        wrong = copy.deepcopy(m)
        wrong["control_version"] = m["engine_version"] + ".9"       # the manifest names a Control hotfix the source does not have
        problems = "\n".join(rm.check(wrong))
        self.assertIn("CMakeLists.txt (Control) says", problems)
        wrong = copy.deepcopy(m)
        wrong["engine_version"] = "0.1.58"                          # engines two builds behind Control: not a hotfix
        self.assertIn("CMakeLists.txt (engine) says", "\n".join(rm.check(wrong)))
        wrong = copy.deepcopy(m)
        wrong["terminal_version"] = m["product_version"] + ".9"     # the manifest names a Terminal the source does not say
        terminal = rm.code_versions()["strata_cli/__init__.py"][0]
        self.assertIn("strata_cli/__init__.py says %s" % terminal, "\n".join(rm.check(wrong)))

    def test_installer_names_in_the_manifest_asset_list(self):
        m = rm.load()
        for name in rm.expand(m):
            self.assertIn(name, m["assets"])


if __name__ == "__main__":
    unittest.main()
