"""#5240: DFlash2 on the Coder is an explicit opt-in. A DFlash2 chosen by name on the Coder starts DFlash2 with the stock pack (the Coder has no tuned pack) and no MTP
layer; Auto (and no choice) keeps the MTP on the Coder; Swift and NVFP4 stay refused (the MTP, with the NOT_SUPPORTED line); setup fetches the stock pack only for the
named choice. The app's copies (resolveDrafter) are checked against the same rule by apps/host/tests/policy-generated.test.mjs."""

from __future__ import annotations

import tempfile
import unittest

from strata_control import drafting
from tests.terminal import test_drafting as TD

CODER = "qwen3.8-flash-next-coder-iq1_m"


class CoderNamed(unittest.TestCase):
    run_config = TD.UnslothAndTheOtherFlashNextFiles.run_config

    def test_named_dflash2_runs_on_the_coder_with_the_stock_pack(self):
        for backend, archs in TD.UnslothAndTheOtherFlashNextFiles.HARDWARE:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                cfg, data, _mtp = self.run_config(CODER, tmp, "dflash2")
                d = drafting.apply(cfg, data_dir=data, backend=backend, archs=archs)
                args = cfg["args"]
                self.assertEqual((d.drafter, d.note), ("dflash2", ""))
                self.assertEqual(args[args.index("--draft-model") + 1], str(drafting.pack_path(data)))
                self.assertNotIn("--mtp", args)

    def test_auto_and_no_choice_keep_the_mtp_on_the_coder(self):
        for mode in (None, "auto"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg, data, _mtp = self.run_config(CODER, tmp, mode)
                original = list(cfg["args"])
                d = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_89"])
                self.assertEqual((d.drafter, cfg["args"]), ("mtp", original))
        self.assertEqual(drafting.choice_of("auto", family="coder", backend="cuda"), "mtp")
        self.assertEqual(drafting.choice_of("auto", model_name=CODER, backend="cuda"), "mtp")

    def test_swift_and_nvfp4_stay_refused(self):
        for name in ("qwen3.8-flash-next-nvfp4-nvfp4", "swift-1.5-iq2_xs", "qwen3.8-flash-next-swift-iq2_xs"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                cfg, data, _mtp = self.run_config(name, tmp, "dflash2")
                d = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_89"])
                self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SUPPORTED))
        for family in ("swift", "nvfp4", "qwen36"):
            self.assertEqual(drafting.choice_of("dflash2", family=family, backend="cuda"), "mtp", family)

    def test_the_named_rule(self):
        self.assertFalse(drafting.supported_model(CODER))
        self.assertTrue(drafting.supported_model(CODER, named=True))
        self.assertTrue(drafting.supported_model(None, "coder", named=True))
        self.assertFalse(drafting.supported_model(None, "swift", named=True))
        self.assertFalse(drafting.supported_model("qwen3.8-flash-next-coder", named=True), "the family name alone is not a set setup writes")
        self.assertEqual(drafting.NAMED_ONLY_FAMILIES, ("coder",))

    def test_setup_is_told_to_fetch_the_stock_pack_only_for_the_named_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(model_name=CODER, family="coder", data_dir=tmp, backend="cuda", archs=["sm_89"], args=[], vram_mib=24576)
            self.assertTrue(drafting.needs_pack("dflash2", **kw))
            self.assertFalse(drafting.needs_pack("auto", **kw))
            self.assertFalse(drafting.needs_tuned("dflash2", **kw), "no tuned pack for the Coder")
            self.assertTrue(drafting.wants_one_request("dflash2", **kw))


if __name__ == "__main__":
    unittest.main()
