"""A user-pinned drafter must reach the engine instead of the automatic pack."""
from strata_control import drafting, user_args
from tests.terminal.test_drafting_tuned import Base, V31, flag


class PinnedDrafter(Base):
    def config(self, path, mode="dflash2", **extra):
        cfg = self.cfg(V31, drafting=mode,
                       user_args=["--draft-model", str(path), "--spec", "6", "--draft-conf", "0.7"], **extra)
        user_args.apply(cfg)
        return cfg

    def pinned(self):
        path = self.root / "measured-drafter.pack"
        path.write_bytes(b"p" * (2 << 20))
        return path

    def apply_cuda(self, cfg):
        return drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"],
                              engine=self.exe, vram_mib=32768)

    def test_pin_wins_over_automatic_stock_and_tuned_selection(self):
        self.put_stock()
        self.put_tuned()
        path = self.pinned()
        cfg = self.config(path)
        decision = self.apply_cuda(cfg)
        self.assertEqual(decision.drafter, "dflash2")
        self.assertEqual(flag(cfg["args"], "--draft-model"), str(path))
        self.assertEqual(flag(cfg["args"], "--spec"), "6")
        self.assertEqual(flag(cfg["args"], "--draft-conf"), "0.7")
        self.assertEqual(flag(cfg["args"], "--draft-fallback"), "none")
        self.assertNotIn("--mtp", cfg["args"])

    def test_pin_runs_without_a_default_pack(self):
        path = self.pinned()
        cfg = self.config(path)
        decision = self.apply_cuda(cfg)
        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))
        self.assertEqual(flag(cfg["args"], "--draft-model"), str(path))

    def test_missing_pin_does_not_silently_load_a_different_pack(self):
        self.put_stock()
        cfg = self.config(self.root / "missing.pack")
        decision = self.apply_cuda(cfg)
        self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NOT_DOWNLOADED))
        self.assertNotIn("--draft-model", cfg["args"])

    def test_mtp_choice_keeps_the_mtp_even_with_an_old_pin(self):
        cfg = self.config(self.pinned(), mode="mtp")
        decision = self.apply_cuda(cfg)
        self.assertEqual(decision.drafter, "mtp")
        self.assertIn("--mtp", cfg["args"])
        self.assertNotIn("--draft-model", cfg["args"])

    def test_pin_does_not_enable_an_unsupported_backend(self):
        cfg = self.config(self.pinned())
        decision = drafting.apply(cfg, data_dir=self.data, backend="cpu", engine=self.exe)
        self.assertEqual(decision.drafter, "mtp")
        self.assertNotIn("--draft-model", cfg["args"])

    def test_auto_slots_use_the_pin_and_keep_the_other_slot_counts_unchanged(self):
        path = self.pinned()
        for slots, expected in ((2, ("dflash2", "")), (3, ("mtp", drafting.NONE_IN_SLOTS))):
            with self.subTest(slots=slots):
                cfg = self.config(path, mode="auto")
                cfg["args"] += ["--expert-cache", "auto"]
                decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"],
                                          engine=self.exe, vram_mib=73728, slots=slots)
                self.assertEqual((decision.drafter, decision.note), expected)
                if slots == 2:
                    self.assertEqual(flag(cfg["args"], "--draft-model"), str(path))
                else:
                    self.assertNotIn("--draft-model", cfg["args"])

    def test_missing_pin_keeps_the_parallel_slots_explanation_with_a_stock_pack(self):
        self.put_stock()
        cfg = self.config(self.root / "missing.pack", mode="auto")
        cfg["args"] += ["--expert-cache", "auto"]
        decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"],
                                  engine=self.exe, vram_mib=73728, slots=2)
        self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NONE_IN_SLOTS))
        self.assertNotIn("--draft-model", cfg["args"])

    def test_renamed_model_keeps_its_file_recognition_with_a_pin(self):
        from tests.terminal.test_drafting_file_fit import gguf_header

        path = self.pinned()
        model = self.root / "my-copy.gguf"
        model.write_bytes(gguf_header())
        cfg = self.config(path)
        cfg["model_name"] = "my-copy"
        cfg["args"][cfg["args"].index("--native") + 1] = str(model)
        decision = self.apply_cuda(cfg)
        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))
        self.assertEqual(flag(cfg["args"], "--draft-model"), str(path))
