"""#5747: on the Intel (SYCL) engine DFlash2 is planned only where that engine runs it: one card, one request at a time.

The SYCL engine refuses `--draft dflash2` with a layer split, `--batch` slots or `--pipeline-windows` at start
(sycl/src/program/generate.cpp: "--draft dflash2 is for one card and one request at a time"). The CUDA/HIP rule lets a layer split and,
by name, request slots through; on Intel that planned a start the engine refuses (a failed boot per restart, a restart loop with
--draft-fallback none). Runs without a GPU."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import drafting  # noqa: E402


class SyclDflash2Planning(unittest.TestCase):
    def test_a_sycl_layer_split_is_blocked_auto_and_by_name(self):
        for kw in ({"split": True}, {}):
            args = [] if kw else ["--layer-split", "14"]
            self.assertTrue(drafting.blocked_by(args, backend="sycl", **kw), (args, kw, "Auto"))
            self.assertTrue(drafting.blocked_by(args, backend="sycl", named=True, **kw), (args, kw, "by name"))
            self.assertTrue(drafting.blocked_by(args, backend="sycl", slot_drafts=True, **kw), (args, kw, "slot drafts"))
        self.assertTrue(drafting.blocked_by(["--split-device", "1"], backend="sycl", named=True))

    def test_sycl_request_slots_and_pipeline_windows_are_blocked_by_name_too(self):
        self.assertTrue(drafting.blocked_by([], slots=4, backend="sycl", named=True))
        self.assertTrue(drafting.blocked_by(["--batch", "2"], backend="sycl", named=True))
        self.assertTrue(drafting.blocked_by(["--parallel", "2"], backend="SYCL", named=True))
        self.assertTrue(drafting.blocked_by(["--pipeline-windows", "2"], backend="sycl", named=True))

    def test_one_sycl_card_one_request_stays_allowed(self):
        self.assertFalse(drafting.blocked_by([], backend="sycl"))
        self.assertFalse(drafting.blocked_by(["--spec", "4"], slots=1, backend="sycl", named=True))

    def test_other_backends_keep_their_rule(self):
        self.assertFalse(drafting.blocked_by([], split=True, backend="cuda"), "a CUDA/HIP layer split runs DFlash2")
        self.assertFalse(drafting.blocked_by(["--layer-split", "auto"], backend="hip", named=True))
        self.assertFalse(drafting.blocked_by([], slots=4, named=True, backend="cuda"))
        self.assertTrue(drafting.blocked_by([], slots=4, named=True, backend="metal"))


if __name__ == "__main__":
    unittest.main()
