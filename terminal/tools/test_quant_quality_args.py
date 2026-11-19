"""Tests for the engine command tools/quant_quality.py builds, without starting an engine.

    python -m unittest tools.test_quant_quality_args
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import quant_quality as QQ  # noqa: E402


class EngineArgs(unittest.TestCase):
    def test_drops_prefill_head_alias(self):
        # A run config with MTP carries --prefill-head-alias --short-read 0; the yardstick replaces --short-read,
        # and the engine refuses --prefill-head-alias without --short-read 0, so it stopped before READY (#5553).
        cfg = {"exe": "strata", "args": ["--pack", "P", "--prefill-head-alias", "--short-read", "0", "--kv", "int8"]}
        cmd = QQ.engine_args(cfg, None, 1000)
        self.assertNotIn("--prefill-head-alias", cmd)
        self.assertEqual(cmd.count("--short-read"), 1)
        self.assertEqual(cmd[cmd.index("--short-read") + 1], "1064")


if __name__ == "__main__":
    unittest.main()
