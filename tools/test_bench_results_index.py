"""docs/bench-results/README.md lists every result folder (tools/bench_results_index.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_results_index as bri  # noqa: E402


class BenchResultsIndex(unittest.TestCase):
    def test_the_index_is_current_and_lists_every_folder(self):
        have = open(bri.OUT, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        self.assertEqual(have, bri.build(), "run python tools/bench_results_index.py")
        for name in os.listdir(bri.RESULTS):
            if os.path.isdir(os.path.join(bri.RESULTS, name)):
                self.assertIn("[%s](%s/)" % (name, name), have)


if __name__ == "__main__":
    unittest.main()
