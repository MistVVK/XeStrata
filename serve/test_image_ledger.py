# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The image side keeps going when the shared ledger's files cannot be written.

    python -m unittest serve.test_image_ledger
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import images  # noqa: E402


class BrokenLedger:
    def find(self, *args):
        raise OSError("no space left on device")

    def hold(self, *args, **kw):
        raise OSError("no space left on device")


class LedgerFileFails(unittest.TestCase):
    def gen(self, engine):
        return SimpleNamespace(lock=threading.Lock(), ledger=lambda *a: BrokenLedger(),
                               _owner=lambda: 1234, engine=engine)

    def test_sync_keeps_going_when_the_ledger_fails(self):
        engine = SimpleNamespace(alive=lambda: False, ask=lambda *a: {"held_mib": 0}, found=False)
        images.ImageGen.ledger_sync(self.gen(engine))   # no exception: the caller can restore the cache

    def test_enter_keeps_going_when_the_ledger_fails(self):
        engine = SimpleNamespace(ask=lambda *a: {"held_mib": 100, "total_mib": 8192}, found=False)
        images.ImageGen._enter_ledger(self.gen(engine))


if __name__ == "__main__":
    unittest.main()
