"""The capability table (docs/data/capabilities.json) against the release's packages and the pages that quote it."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capability_table as ct  # noqa: E402


class CapabilityTable(unittest.TestCase):
    def setUp(self):
        self.cap = ct.read_json(ct.DATA)
        self.release = ct.read_json(ct.RELEASE)

    def test_pages_and_table_are_what_the_generator_writes(self):
        self.assertEqual(ct.validate(self.cap, self.release), [])
        self.assertEqual(ct.read(ct.OUT).replace("\r\n", "\n"), ct.page(self.cap, self.release))
        for p in ct.PAGES:
            text = ct.read(p)
            self.assertTrue(ct.MARK_RE.search(text), p)
            self.assertEqual(ct.render(self.cap, self.release, text), text, p)

    def test_a_package_contradicts_a_no(self):
        cap = copy.deepcopy(self.cap)
        {f["id"]: f for f in cap["families"]}["qwen38"]["cells"]["mac"]["supported"] = "no"
        problems = ct.validate(cap, self.release)
        self.assertTrue(any("qwen38 on mac is marked 'no'" in p for p in problems), problems)

    def test_a_yes_without_a_package_is_reported(self):
        cap = copy.deepcopy(self.cap)
        {f["id"]: f for f in cap["families"]}["qwen36"]["cells"]["intel_linux"]["supported"] = "yes"
        problems = ct.validate(cap, self.release)
        self.assertTrue(any("no package for it" in p for p in problems), problems)

    def test_qwen36_packages_are_the_cuda_and_hip_engines_only(self):
        fam = {f["id"]: f for f in self.cap["families"]}["qwen36"]
        pk = ct.packages(self.release, fam)
        self.assertNotIn("mac", pk)
        self.assertNotIn("intel_linux", pk)
        self.assertIn("nvidia_win", pk)

    def test_no_page_keeps_the_old_mac_sentence(self):
        for p in ct.PAGES:
            self.assertNotIn("no DeepSeek yet", ct.read(p))


if __name__ == "__main__":
    unittest.main()
