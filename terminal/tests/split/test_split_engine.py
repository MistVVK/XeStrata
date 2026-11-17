"""strata_control/split_engine.py (when the all-cards split engine may run) and src/split (no peer access in any path)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from strata_control import split_engine as S

ROOT = Path(__file__).resolve().parents[2]
GIB = 1 << 30

A6000 = dict(name="NVIDIA RTX A6000", vram_mib=49140, cc=86)


def cards(n=2, **kw):
    spec = {**A6000, **kw}
    return [S.Card(index=i, **spec) for i in range(n)]


def udq4(**kw):
    """UD-Q4_K_XL as measured: 71.7 GiB of routed experts, Q8_0 embeddings, a 27.5 GiB embedding table in RAM."""
    f = dict(routed_types={"Q4_K": 97, "Q5_K": 2, "Q5_1": 43, "Q8_0": 5}, routed_bytes=int(71.7 * GIB), embd_type="Q8_0",
             shexp_types={"Q8_0": 144}, embd_ram_bytes=int(27.5 * GIB), layers=48, experts=512)
    f.update(kw)
    return S.ModelFacts(**f)


def decide(**kw):
    a = dict(enabled=True, system="linux", cards=cards(), ram_total_gib=125, ram_available_gib=80, model=udq4())
    a.update(kw)
    return S.decide(**a)


class Rule(unittest.TestCase):
    def test_runs_on_the_tested_setup(self):
        d = decide()
        self.assertTrue(d.use_split, d.reasons)
        self.assertEqual(d.cards, 2)
        self.assertEqual(d.untested, [])
        self.assertIn("2 cards", d.line)

    def test_off_unless_asked(self):
        d = decide(enabled=False)
        self.assertFalse(d.use_split)
        self.assertIn("opt-in", d.line)

    def test_linux_nvidia_only(self):
        for system in ("windows", "darwin"):
            self.assertFalse(decide(system=system).use_split)

    def test_card_count_and_sameness(self):
        self.assertFalse(decide(cards=cards(1)).use_split)
        self.assertFalse(decide(cards=cards(5)).use_split)
        mixed = cards(1) + [S.Card(index=1, name="NVIDIA RTX A5000", vram_mib=24564, cc=86)]
        d = decide(cards=mixed)
        self.assertFalse(d.use_split)
        self.assertIn("same model", d.line)

    def test_untested_counts_archs_and_contexts_stay_off_unless_a_tester_allows_them(self):
        for c in (cards(3), cards(2, cc=89), cards(2, cc=90)):
            d = decide(cards=c)
            self.assertFalse(d.use_split)
            self.assertIn("only been run", d.line)
        d = decide(cards=cards(3, vram_mib=49140), allow_untested=True)
        self.assertTrue(d.use_split, d.reasons)
        self.assertIn("3 cards", d.untested)
        self.assertIn("not tested", d.line)
        self.assertFalse(decide(context=262144).use_split)
        self.assertTrue(decide(context=262144, allow_untested=True).use_split)

    def test_all_the_experts_must_fit_in_the_cards(self):
        d = decide(cards=cards(2, vram_mib=24564, name="NVIDIA RTX A5000"))
        self.assertFalse(d.use_split)
        self.assertIn("do not fit", d.line)
        self.assertFalse(decide(cards=cards(2, vram_mib=40960)).use_split)    # 71.7 / 2 + 9 = 44.9 GiB needed
        self.assertTrue(decide(cards=cards(2, vram_mib=46000)).use_split)

    def test_ram_that_stays_must_be_free(self):
        self.assertTrue(decide(ram_available_gib=52).use_split)          # 27.5 + 3 + 4 + 8 spare = 42.5
        d = decide(ram_available_gib=30)
        self.assertFalse(d.use_split)
        self.assertIn("RAM", d.line)

    def test_only_the_quant_types_it_reads(self):
        for model in (udq4(layers=64), udq4(routed_types={}),
                      udq4(routed_types={"Q6_K": 48}),                                  # UD-Q6_K_XL-like experts: no kernel
                      udq4(embd_type="IQ4_XS"),                                         # a K-quant set is not converted at load
                      udq4(shexp_types={"BF16": 48, "IQ4_NL": 47}),
                      udq4(output_type="Q6_K"),
                      udq4(ple_type="Q6_K")):
            d = decide(model=model)
            self.assertFalse(d.use_split, model)
            self.assertTrue(d.line.startswith("Using the normal layer split"))

    def test_the_iq3_s_set_and_ud_iq4_xs_run(self):
        iq3s = S.ModelFacts(routed_types={"IQ2_S": 40, "IQ3_XXS": 34, "IQ3_S": 20, "IQ4_XS": 2, "IQ4_NL": 39, "Q2_0": 9},
                            routed_bytes=int(49 * GIB), embd_type="IQ4_XS", shexp_types={"IQ4_XS": 60, "Q4_K": 40, "Q5_K": 20, "IQ4_NL": 24},
                            embd_ram_bytes=int(27.2 * GIB), layers=48, experts=512, ple_type="IQ4_NL", output_type="Q6_K")
        d = decide(model=iq3s)
        self.assertTrue(d.use_split, d.reasons)
        ud_iq4_xs = S.ModelFacts(routed_types={"IQ3_S": 94, "IQ4_XS": 2, "IQ4_NL": 43, "Q8_0": 5}, routed_bytes=int(59.5e9),
                                 embd_type="Q8_0", shexp_types={"Q8_0": 144}, embd_ram_bytes=int(27.5 * GIB), layers=48, experts=512,
                                 ple_type="IQ4_NL", output_type="Q6_K")
        d = decide(model=ud_iq4_xs)
        self.assertTrue(d.use_split, d.reasons)
        self.assertTrue(d.need_vram_gib < 48, d.need_vram_gib)

    def test_a_type_the_loader_cannot_convert_is_a_plain_refusal(self):
        for model, word in ((udq4(routed_types={"IQ3_S": 94}, embd_type="Q3_K"), "cannot convert"),
                            (udq4(routed_types={"IQ3_S": 94}, shexp_types={"Q3_K": 5}), "cannot read"),
                            (udq4(routed_types={"IQ3_S": 94}, ple_type="Q6_K"), "per-layer"),
                            (udq4(routed_types={"IQ3_S": 94, "TQ2_0": 2}), "TQ2_0")):
            d = decide(model=model)
            self.assertFalse(d.use_split)
            self.assertIn(word, d.line)
            self.assertNotIn(chr(10), d.line)          # one plain line

    def test_scan_counts_the_q2_0_repack_twice(self):
        from unittest import mock

        class T:
            def __init__(self, name, type_name, shape, nbytes):
                self.name, self.type_name, self.shape, self._n = name, type_name, shape, nbytes
            def expected_bytes(self):
                return self._n

        class F:
            def __init__(self, path):
                self.tensors = [T("blk.0.ffn_down_exps.weight", "Q2_0", (640, 2560, 512), 1000), T("blk.1.ffn_gate_exps.weight", "IQ3_S", (2560, 640, 512), 5000),
                                T("token_embd.weight", "IQ4_XS", (2560, 248320), 10), T("per_layer_token_embd.weight", "IQ4_NL", (1, 2), 20),
                                T("output.weight", "Q6_K", (2560, 248320), 30)]
        with mock.patch("tools.gguf_reader.GGUFFile", F):
            m = S.scan_model("/x/model-00001-of-00001.gguf")
        self.assertEqual(m.routed_bytes, 2 * 1000 + 5000)
        self.assertEqual((m.embd_type, m.ple_type, m.output_type), ("IQ4_XS", "IQ4_NL", "Q6_K"))
        self.assertEqual(m.embd_ram_bytes, 30)

    def test_the_reasons_are_all_reported(self):
        d = decide(system="windows", cards=cards(1), model=udq4(embd_type="Q6_K"))
        self.assertGreaterEqual(len(d.reasons), 3)


class Config(unittest.TestCase):
    def base(self):
        return {"exe": "engine/strata", "args": ["--pack", "p", "--native", "m.gguf", "--layer-split", "auto"], "cwd": "/x",
                "tokenizer": "tok", "port": 8080, "model_name": "m", "log": "l.log", "gpu": [0, 1], "env": {"STRATA_X": "1"},
                "layer_split": "auto"}

    def test_split_config_keeps_the_front_end_and_the_fallback(self):
        d = decide()
        cfg = S.split_config(self.base(), d, exe="/e/strata-split-2", native="/m/shard1.gguf", mtp_dir="/d/mtp/rt",
                             draft_vocab="/d/draft_vocab_en.bin", context=131072, lanes=6, vision=True)
        self.assertEqual(cfg["exe"], "/e/strata-split-2")
        self.assertEqual(cfg["tokenizer"], "tok")
        self.assertEqual(cfg["port"], 8080)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertIn("--vision", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "6")
        self.assertEqual(cfg["env"]["FNX_ARENA"], "0")
        self.assertEqual(cfg["env"]["FNX_EXPERT_RAM_GIB"], "off")
        self.assertEqual(cfg["env"]["STRATA_X"], "1")
        self.assertNotIn("layer_split", cfg)
        fb = cfg["split_fallback"]
        self.assertEqual(fb["exe"], "engine/strata")
        self.assertEqual(fb["args"], self.base()["args"])
        self.assertEqual(fb["layer_split"], "auto")
        self.assertEqual(self.base()["args"], ["--pack", "p", "--native", "m.gguf", "--layer-split", "auto"])   # the base is not changed

    def test_no_config_for_a_refusal(self):
        with self.assertRaises(ValueError):
            S.split_config(self.base(), decide(enabled=False), exe="e", native="n", mtp_dir="m", draft_vocab="v", context=1)


class Apply(unittest.TestCase):
    """apply_to_config: the launcher's side (facts of this PC and model in, a run config out)."""

    def cfg(self, tmp):
        (Path(tmp) / "data").mkdir(exist_ok=True)
        (Path(tmp) / "data" / "draft_vocab_en.bin").write_bytes(b"x")
        return {"exe": "component:strata-engine", "args": ["--pack", "p", "--native", "/m/a-00001-of-00004.gguf", "--mtp", "/d/mtp/rt",
                                                           "--max-context", "131072", "--layer-split", "auto"],
                "cwd": tmp, "tokenizer": "tok", "port": 8080, "gpu": [0, 1], "env": {}, "split_engine": True}

    def run_apply(self, cfg, *, cards_=None, avail=80.0, model=None, exe="/e/strata-split-2", **kw):
        return S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None: exe, system=kw.pop("system", "linux"), probe=lambda idx: cards_ or cards(),
                                 ram=lambda: (125.0, avail), scan=lambda path: model or udq4(), **kw)

    def test_a_good_pc_gets_the_split_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            d = self.run_apply(cfg)
            self.assertTrue(d.use_split, d.reasons)
            self.assertEqual(cfg["exe"], "/e/strata-split-2")
            self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], "/m/a-00001-of-00004.gguf")
            self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072")
            self.assertEqual(cfg["split_fallback"]["exe"], "component:strata-engine")
            self.assertEqual(cfg["tokenizer"], "tok")

    def test_every_refusal_leaves_the_config_alone(self):
        import copy
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for kw in (dict(cards_=cards(1)), dict(avail=20.0), dict(exe=None), dict(system="windows"),
                       dict(model=udq4(embd_type="Q6_K"))):
                cfg = self.cfg(tmp)
                before = copy.deepcopy(cfg)
                d = self.run_apply(cfg, **kw)
                self.assertFalse(d.use_split, kw)
                self.assertEqual(cfg, before, kw)
                self.assertTrue(d.line.startswith("Using the normal layer split"), d.line)

    def test_a_model_file_it_cannot_read_is_a_reason_not_an_error(self):
        import tempfile

        def boom(path):
            raise OSError("gone")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            d = S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None: "/e/x", system="linux", probe=lambda i: cards(), ram=lambda: (125.0, 80.0), scan=boom)
            self.assertFalse(d.use_split)
            self.assertIn("could not be read", d.line)

    def test_modes_auto_is_the_default_and_off_is_the_switch(self):
        self.assertEqual(S.mode({}, {}), "auto")
        for raw, want in ((True, "on"), (False, "off"), ("on", "on"), ("off", "off"), ("auto", "auto"), ("OFF", "off"), ("nonsense", "auto"), (None, "auto")):
            self.assertEqual(S.mode({"split_engine": raw}, {}), want, raw)
        self.assertEqual(S.mode({"split_engine": "on"}, {"STRATA_SPLIT_ENGINE": "0"}), "off")     # the environment wins (testers)
        self.assertEqual(S.mode({"split_engine": "off"}, {"STRATA_SPLIT_ENGINE": "1"}), "on")
        self.assertEqual(S.mode({}, {"STRATA_SPLIT_ENGINE": "off"}), "off")
        self.assertEqual(S.mode({"split_engine": "off"}, {"STRATA_SPLIT_ENGINE": ""}), "off")      # an empty variable is no variable
        self.assertTrue(S.wanted({}, {}))
        self.assertFalse(S.wanted({"split_engine": "off"}, {}))

    def test_auto_uses_the_split_where_it_is_supported_and_says_nothing_otherwise(self):
        import copy
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            del cfg["split_engine"]                                      # nothing asked: auto
            d = self.run_apply(cfg)
            self.assertTrue(d.use_split, d.reasons)
            self.assertFalse(d.quiet)
            self.assertEqual(cfg["exe"], "/e/strata-split-2")
            for kw in (dict(cards_=cards(1)), dict(avail=20.0), dict(exe=None), dict(system="windows"), dict(model=udq4(routed_types={"Q6_K": 48})),
                       dict(cards_=cards(3)), dict(cards_=cards(2, cc=89))):
                cfg = self.cfg(tmp)
                del cfg["split_engine"]
                before = copy.deepcopy(cfg)
                d = self.run_apply(cfg, **kw)
                self.assertFalse(d.use_split, kw)
                self.assertTrue(d.quiet, kw)                             # silent: the launch goes on with the layer split
                self.assertEqual(cfg, before, kw)

    def test_on_says_why_it_cannot_run(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            cfg["split_engine"] = "on"
            d = self.run_apply(cfg, cards_=cards(1))
            self.assertFalse(d.use_split)
            self.assertFalse(d.quiet)
            self.assertIn("two or more cards", d.line)

    def test_off_never_runs_and_changes_nothing(self):
        import copy
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for off in ("off", False):
                cfg = self.cfg(tmp)
                cfg["split_engine"] = off
                before = copy.deepcopy(cfg)
                d = self.run_apply(cfg)
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
                self.assertEqual(cfg, before)

    def test_choices_the_split_engine_does_not_honour_keep_the_layer_split(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for change in (lambda c: c.update(drafting="dflash2"), lambda c: c.update(user_args=["--x"]), lambda c: c.update(replicas=2),
                           lambda c: c["args"].extend(["--kv", "q4_0"])):
                cfg = self.cfg(tmp)
                del cfg["split_engine"]
                change(cfg)
                d = self.run_apply(cfg)
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
            cfg = self.cfg(tmp)
            cfg["args"].extend(["--kv", "int8"])
            self.assertTrue(self.run_apply(cfg).use_split)

    def df2_files(self, tmp, *, taking=True, pack=True):
        """A split engine program that carries (or lacks) the text of `--draft dflash2`, and a data folder with (or without) the DFlash2 pack."""
        from strata_control import drafting

        exe = Path(tmp) / "strata-split-2"
        exe.write_bytes(b"\0--draft dflash2 needs --native, --draft-model PACK and --draft-vocab FILE\0" if taking else b"\0--native and --mtp are required\0")
        data = Path(tmp) / "models"
        if pack:
            target = drafting.target_path(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.truncate((1 << 20) + 1)
        return str(exe), data

    def test_dflash2_chosen_by_name_runs_on_the_split_engine_only_with_the_persons_own_on(self):
        """Owner 2026-11-03: the person's own choice wins. DFlash2 named + --split-engine on = the split engine with `--draft dflash2 --draft-model PACK` (no MTP layer);
        Auto, or DFlash2 named and the split engine on Auto, are as they were (Auto the MTP split; DFlash2 the layer split)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            exe, data = self.df2_files(tmp)
            cfg = self.cfg(tmp)
            cfg["drafting"] = "dflash2"
            cfg["args"] = [a for a in cfg["args"] if a != "--mtp" and a != "/d/mtp/rt"]            # no draft layer on this computer at all
            before = dict(cfg, args=list(cfg["args"]))
            d = self.run_apply(cfg, exe=exe, data_dir=str(data), mode_="on")
            self.assertTrue(d.use_split, d.reasons)
            a = cfg["args"]
            self.assertEqual((a[a.index("--draft") + 1], a[a.index("--draft-model") + 1]), ("dflash2", str(data / "dflash2" / "qwen3.8-flash-next-dflash2.q4_0.pack")))
            self.assertNotIn("--mtp", a)
            self.assertEqual(a[a.index("--draft-vocab") + 1], str(Path(tmp) / "data" / "draft_vocab_en.bin"))
            self.assertEqual(a[a.index("--batch") + 1], "6")
            self.assertEqual(cfg["drafting"], "dflash2")
            self.assertEqual(cfg["split_fallback"]["drafting"], "dflash2", "the layer split that starts if this engine refuses keeps the person's word")
            self.assertEqual(cfg["split_fallback"]["args"], before["args"])
            # the same choice with the split engine on Auto (or nothing) stays on the layer split, quietly
            for mode in ("auto", None):
                cfg = self.cfg(tmp)
                cfg["drafting"] = "dflash2"
                del cfg["split_engine"]
                if mode:
                    cfg["split_engine"] = mode
                d = self.run_apply(cfg, exe=exe, data_dir=str(data))
                self.assertFalse(d.use_split, mode)
                self.assertTrue(d.quiet, mode)
                self.assertNotIn("split_fallback", cfg)
            # Auto and the MTP on the split engine "on" are the MTP split as before
            for chosen in (None, "auto", "mtp"):
                cfg = self.cfg(tmp)
                if chosen:
                    cfg["drafting"] = chosen
                d = self.run_apply(cfg, exe=exe, data_dir=str(data))
                self.assertTrue(d.use_split, chosen)
                self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], "/d/mtp/rt", chosen)
                self.assertNotIn("--draft", cfg["args"], chosen)
                self.assertEqual(cfg["drafting"], "mtp", chosen)

    def test_dflash2_on_the_split_needs_an_engine_that_takes_it_and_its_pack(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for i, (kw, why) in enumerate((({"taking": False}, "does not run"), ({"pack": False}, S.NO_DF2_PACK))):
                (Path(tmp) / f"case{i}").mkdir()
                exe, data = self.df2_files(Path(tmp) / f"case{i}", **kw)
                cfg = self.cfg(tmp)
                cfg["drafting"] = "dflash2"
                before = dict(cfg, args=list(cfg["args"]))
                d = self.run_apply(cfg, exe=exe, data_dir=str(data), mode_="on")
                self.assertFalse(d.use_split, kw)
                self.assertIn(why, d.reasons[0], kw)
                self.assertEqual(cfg, before, "a refusal leaves the config alone: the layer split starts with DFlash2")
        self.assertFalse(S.exe_takes_dflash2(None))
        self.assertFalse(S.exe_takes_dflash2("/no/such/strata-split-2"))

    def test_expected_follows_the_same_rule(self):
        for chosen, mode, want in (("dflash2", "on", True), ("dflash2", "auto", False), ("dflash2", None, False), ("mtp", "on", True), ("auto", "on", True), ("off", "on", False)):
            cfg = {"drafting": chosen, "split_engine": mode}
            got = S.expected(cfg, indexes=[0, 1], native=None, probe=lambda idx: cards(), ram=lambda: (125.0, 80.0), system="linux")
            self.assertEqual(got, want, (chosen, mode))

    def test_drafting_auto_resolves_to_the_split_with_its_mtp_and_the_fallback_keeps_the_layer_split_choice(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for chosen, split in ((None, True), ("auto", True), ("mtp", True), ("dflash2", False), ("off", False)):
                cfg = self.cfg(tmp)
                del cfg["split_engine"]
                if chosen:
                    cfg["drafting"] = chosen
                before_fallback = {k: v for k, v in cfg.items() if k == "drafting"}
                d = self.run_apply(cfg)
                self.assertEqual(d.use_split, split, chosen)
                if split:
                    # Auto/MTP: the split engine runs its MTP; the config says so, so serve/server.py's drafter step (Auto = DFlash2) leaves its arguments alone
                    self.assertEqual(cfg["drafting"], "mtp", chosen)
                    self.assertEqual(cfg["split_fallback"].get("drafting"), before_fallback.get("drafting"), chosen)   # the layer split keeps the person's word, or Auto
                    self.assertNotIn("--draft", cfg["args"])
                else:
                    self.assertEqual(cfg.get("drafting"), chosen)                                                       # untouched: the layer split gets DFlash2 as chosen
                    self.assertNotIn("split_fallback", cfg)

    def test_auto_needs_a_card_where_the_split_decodes_faster_than_the_layer_split_with_dflash2(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            del cfg["split_engine"]
            d = self.run_apply(cfg, cards_=cards(2, cc=89), allow_untested=True)                # a tester allowed it, but nobody measured it there
            self.assertFalse(d.use_split)
            self.assertIn("DFlash2", d.line)
            cfg["drafting"] = "mtp"                                                              # the person asked for the MTP: no comparison to win
            self.assertTrue(self.run_apply(cfg, cards_=cards(2, cc=89), allow_untested=True).use_split)

    def test_the_engine_is_picked_by_card_count_and_architecture(self):
        import tempfile

        seen = []

        def exe_for(n, cc=None):
            seen.append((n, cc))
            return "/e/strata-split-2-sm120"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg(tmp)
            d = S.apply_to_config(cfg, exe_for_cards=exe_for, system="linux", probe=lambda i: cards(2, cc=120, name="NVIDIA RTX PRO 6000", vram_mib=97887),
                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
            self.assertTrue(d.use_split, d.reasons)
            self.assertEqual(seen, [(2, 120)])
            self.assertEqual(cfg["exe"], "/e/strata-split-2-sm120")


class AmdDefault(unittest.TestCase):
    """AMD gfx1201 (R9700): the split is the default on two or more cards (owner, 11-02), `off` turns it off, `on` says why when it cannot run;
    one constant (AUTO_SPLIT_AMD_ARCHS) says which AMD architectures start it by themselves, every other one keeps the layer split silently."""

    R9700 = dict(name="AMD Radeon AI PRO R9700", vram_mib=32624, cc=0, vendor="amd", arch="gfx1201")
    cfg = Apply.cfg

    def amd(self, tmp, **kw):
        cfg = self.cfg(tmp)
        cfg["gpu"] = [0, 1, 2, 3]
        for k, v in kw.items():
            cfg[k] = v
        return cfg

    def run_amd(self, cfg, seen=None, **kw):
        def exe_for(n, cc=None, arch=None):
            if seen is not None:
                seen.append((n, cc, arch))
            return f"/e/strata-split-{n}-{arch}" if arch else None

        return S.apply_to_config(cfg, exe_for_cards=exe_for, system="linux", probe=lambda i: cards(4, **self.R9700), ram=lambda: (125.0, 80.0),
                                 scan=lambda path: udq4(routed_bytes=int(71.7 * GIB)), **kw)

    def test_auto_starts_the_split_on_gfx1201(self):
        import tempfile

        self.assertEqual(S.AUTO_SPLIT_AMD_ARCHS, ("gfx1201",))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.amd(tmp)
            del cfg["split_engine"]
            seen = []
            d = self.run_amd(cfg, seen)
            self.assertTrue(d.use_split, d.reasons)
            self.assertEqual(seen, [(4, 0, "gfx1201")])
            self.assertEqual(cfg["exe"], "/e/strata-split-4-gfx1201")

    def test_amd_default_lanes_are_eight_and_a_number_the_person_chose_stays(self):
        import tempfile

        self.assertEqual((S.AMD_DEFAULT_LANES, S.DEFAULT_LANES), (8, 6))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.amd(tmp)
            del cfg["split_engine"]
            self.assertTrue(self.run_amd(cfg).use_split)
            self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "8")
            cfg = self.amd(tmp)
            del cfg["split_engine"]
            cfg["parallel"] = 3
            self.assertTrue(self.run_amd(cfg).use_split)
            self.assertEqual(cfg["args"][cfg["args"].index("--batch") + 1], "3")

    def test_auto_on_amd_starts_the_ud_q4_family_and_the_iq3s_set(self):
        import tempfile

        iq3s = dict(routed_types={"IQ4_NL": 39, "IQ3_XXS": 34, "Q2_0": 9, "IQ2_S": 40, "IQ3_S": 20, "IQ4_XS": 2}, routed_bytes=int(48.8 * GIB), embd_type="IQ4_XS",
                    shexp_types={"Q4_K": 30, "Q8_0": 1, "BF16": 48}, ple_type="IQ4_NL")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.amd(tmp)
            del cfg["split_engine"]
            before = dict(cfg)
            d = S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: "/e/strata-split-4-amd", system="linux", probe=lambda i: cards(4, **self.R9700),
                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4(**iq3s))
            self.assertTrue(d.use_split, d.reasons)
            cfg = self.amd(tmp, split_engine="on")
            d = S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: "/e/strata-split-4-amd", system="linux", probe=lambda i: cards(4, **self.R9700),
                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4(**iq3s))
            self.assertTrue(d.use_split, d.reasons)

    def test_auto_is_four_cards_only_today_and_on_runs_two_and_three(self):
        import tempfile

        self.assertEqual(S.AUTO_SPLIT_AMD_CARD_COUNTS, (4,))
        with tempfile.TemporaryDirectory() as tmp:
            for n in (2, 3):
                cfg = self.amd(tmp)
                del cfg["split_engine"]
                before = dict(cfg)
                d = S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: f"/e/strata-split-{n_}-amd", system="linux",
                                      probe=lambda i: cards(n, **self.R9700), ram=lambda: (125.0, 80.0), scan=lambda path: udq4(routed_bytes=int(30 * GIB)))
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
                self.assertEqual(cfg, before)
                cfg = self.amd(tmp, split_engine="on")
                d = S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: f"/e/strata-split-{n_}-amd", system="linux",
                                      probe=lambda i: cards(n, **self.R9700), ram=lambda: (125.0, 80.0), scan=lambda path: udq4(routed_bytes=int(30 * GIB)))
                self.assertTrue(d.use_split, (n, d.reasons))
                self.assertEqual(cfg["exe"], f"/e/strata-split-{n}-amd")

    def test_the_card_count_constant_limits_auto_but_not_on(self):
        import tempfile

        old = S.AUTO_SPLIT_AMD_CARD_COUNTS
        S.AUTO_SPLIT_AMD_CARD_COUNTS = (4,)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for n, auto_runs in ((3, False), (4, True)):
                    cfg = self.amd(tmp)
                    del cfg["split_engine"]
                    d = S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: f"/e/strata-split-{n_}-amd", system="linux",
                                          probe=lambda i: cards(n, **{**self.R9700}), ram=lambda: (125.0, 80.0),
                                          scan=lambda path: udq4(routed_bytes=int(40 * GIB)))
                    self.assertEqual(d.use_split, auto_runs, (n, d.reasons))
                    if not auto_runs:
                        self.assertTrue(d.quiet)
                cfg = self.amd(tmp, split_engine="on")
                self.assertTrue(S.apply_to_config(cfg, exe_for_cards=lambda n_, cc=None, arch=None: "/e/x", system="linux", probe=lambda i: cards(3, **self.R9700),
                                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4(routed_bytes=int(40 * GIB))).use_split)
        finally:
            S.AUTO_SPLIT_AMD_CARD_COUNTS = old

    def test_off_turns_it_off_by_every_route(self):
        import os
        import tempfile

        for how in ("key", "preset", "env"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp)
                del cfg["split_engine"]
                if how == "key":
                    cfg["split_engine"] = "off"
                elif how == "preset":
                    cfg["split_engine"] = False
                else:
                    os.environ["STRATA_SPLIT_ENGINE"] = "0"
                before = dict(cfg)
                try:
                    d = self.run_amd(cfg)
                finally:
                    os.environ.pop("STRATA_SPLIT_ENGINE", None)
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
                self.assertEqual(cfg, before)                              # the layer split's config, untouched

    def test_an_untested_amd_architecture_runs_only_on_an_explicit_on_and_says_it_is_untested(self):
        import tempfile

        for arch in ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1200"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp, split_engine="on")
                seen = []

                def exe_for(n, cc=None, arch_=None):
                    seen.append((n, arch_))
                    return f"/e/strata-split-{n}-amd"

                d = S.apply_to_config(cfg, exe_for_cards=exe_for, system="linux", probe=lambda i: cards(4, **{**self.R9700, "arch": arch}),
                                      ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
                self.assertTrue(d.use_split, (arch, d.reasons))
                self.assertEqual(d.untested, [f"card architecture {arch}"])
                self.assertIn("not tested", d.line)
                self.assertEqual(seen, [(4, arch)])
                self.assertEqual(cfg["exe"], "/e/strata-split-4-amd")
                # the same card in auto: the layer split, silently
                cfg = self.amd(tmp)
                del cfg["split_engine"]
                before = dict(cfg)
                d = S.apply_to_config(cfg, exe_for_cards=exe_for, system="linux", probe=lambda i: cards(4, **{**self.R9700, "arch": arch}),
                                      ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
                self.assertEqual(cfg, before)

    def test_architectures_without_a_code_object_are_refused_even_with_on(self):
        import tempfile

        for arch in ("gfx1030", "gfx1151", "gfx90a", "gfx942"):                     # RDNA2, Strix Halo, CDNA: not built
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp, split_engine="on")
                d = S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None, arch_=None: "/e/x", system="linux",
                                      probe=lambda i: cards(4, **{**self.R9700, "arch": arch}), ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
                self.assertFalse(d.use_split, arch)

    def test_an_amd_architecture_not_in_the_list_keeps_the_layer_split_silently(self):
        import tempfile

        for arch in ("gfx1030", "gfx1151"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp)
                del cfg["split_engine"]
                before = dict(cfg)
                d = S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None, arch=None: "/e/x", system="linux",
                                      probe=lambda i: cards(4, **{**self.R9700, "arch": arch}), ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)                                   # the can-run rule, not a refusal message
                self.assertEqual(cfg, before)

    def test_explicit_on_starts_the_split_with_the_gfx1201_engine(self):
        import tempfile

        for on in (True, "on"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp, split_engine=on)
                seen = []
                d = self.run_amd(cfg, seen)
                self.assertTrue(d.use_split, d.reasons)
                self.assertEqual(d.cards, 4)
                self.assertEqual(d.untested, [])                           # four R9700 have been run
                self.assertEqual(seen, [(4, 0, "gfx1201")])
                self.assertEqual(cfg["exe"], "/e/strata-split-4-gfx1201")
                self.assertEqual(cfg["drafting"], "mtp")
                self.assertEqual(cfg["split_fallback"]["exe"], "component:strata-engine")

    def test_on_without_the_gfx1201_binary_says_so_and_off_is_off(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.amd(tmp, split_engine="on")
            d = S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None, arch=None: None, system="linux", probe=lambda i: cards(4, **self.R9700),
                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
            self.assertFalse(d.use_split)
            self.assertFalse(d.quiet)
            self.assertIn("strata-split-4-gfx1201", d.line)
            cfg = self.amd(tmp, split_engine="off")
            before = dict(cfg)
            self.assertFalse(self.run_amd(cfg).use_split)
            self.assertEqual(cfg, before)

    def test_two_r9700_cannot_hold_the_experts_and_other_amd_cards_are_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.amd(tmp, split_engine="on")
            d = S.apply_to_config(cfg, exe_for_cards=lambda n, cc=None, arch=None: "/e/x", system="linux", probe=lambda i: cards(2, **self.R9700),
                                  ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
            self.assertFalse(d.use_split)
            d = S.apply_to_config(self.amd(tmp, split_engine="on"), exe_for_cards=lambda n, cc=None, arch=None: "/e/x", system="linux",
                                  probe=lambda i: cards(4, **{**self.R9700, "arch": "gfx1030"}), ram=lambda: (125.0, 80.0), scan=lambda path: udq4())
            self.assertFalse(d.use_split)
            self.assertIn("gfx1030", d.line)

    def test_taking_the_architecture_out_of_the_constant_makes_auto_keep_the_layer_split_but_on_still_runs(self):
        import tempfile

        old = S.AUTO_SPLIT_AMD_ARCHS
        S.AUTO_SPLIT_AMD_ARCHS = ()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = self.amd(tmp)
                del cfg["split_engine"]
                d = self.run_amd(cfg)
                self.assertFalse(d.use_split)
                self.assertTrue(d.quiet)
                self.assertTrue(self.run_amd(self.amd(tmp, split_engine="on")).use_split)
        finally:
            S.AUTO_SPLIT_AMD_ARCHS = old


class HipArchSources(unittest.TestCase):
    """The AMD architecture rules that live in the engine sources (the launcher side is AmdDefault): pinned so a refactor cannot drop the guard."""

    SRC = ROOT.parent / "engine" / "src" / "split"

    def text(self, rel):
        return (self.SRC / rel).read_text(encoding="utf-8")

    def test_the_self_test_runs_before_the_model_loads_on_a_card_nobody_has_run(self):
        model = self.text("src/model.cpp")
        i = model.index("void check_cards()")
        j = model.index("selftest_decide(", i)
        self.assertLess(j, model.index("float f16_to_f32", i))           # inside check_cards, which Model::Model calls first
        self.assertIn("FNX_SELFTEST_FAIL", self.text("src/hip/selftest_policy.hpp"))
        self.assertIn("selftest.cu", self.text("CMakeLists.txt"))
        self.assertRegex(self.text("src/hip/selftest_policy.hpp"), r'arch_tested\([^)]*\)\s*\{\s*return gcn_arch\.rfind\("gfx1201", 0\) == 0;')

    def test_the_code_objects_and_the_card_check_agree(self):
        cmake = self.text("CMakeLists.txt")
        for arch in ("gfx1201", "gfx1200", "gfx11-generic"):
            self.assertIn(arch, cmake)
        build = (ROOT.parent / "tools" / "packaging" / "build_split_hip.sh").read_text(encoding="utf-8")
        self.assertIn("gfx1201;gfx1200;gfx11-generic", build)
        self.assertIn("strata-split-$n-amd", build)
        self.assertRegex(build, r"NEEDED\.\*rccl")                          # RCCL is dlopen'ed, never linked
        model = self.text("src/model.cpp")
        for arch in ("gfx1201", "gfx1200", "gfx1100", "gfx1101", "gfx1102", "gfx1103"):
            self.assertIn('"%s"' % arch, model)

    def test_the_issue_template_exists_and_the_engine_line_points_at_it(self):
        tpl = ROOT.parent / ".github" / "ISSUE_TEMPLATE" / "amd_split_report.yml"
        self.assertTrue(tpl.is_file())
        self.assertIn("AMD split engine report", tpl.read_text(encoding="utf-8"))
        self.assertIn("template=amd_split_report.yml", self.text("src/hip/selftest_policy.hpp"))


class NoPeerAccess(unittest.TestCase):
    """Pictures of the rule 'no peer access in any path' for the NVIDIA split: nothing under src/split may call the CUDA peer APIs, and the HIP peer APIs
    appear in exactly one file, src/split/src/hip/peer.h (AMD only: the owner allowed peer access on the R9700 "if faster, with fallback", 2026-11-02; it runs only
    after a start probe of every directed pair, and FNX_PEER=0 forces the host exchange). The built NVIDIA engine is also checked with nm in
    packaging/build_split_cuda.sh."""

    FORBIDDEN = re.compile(r"(cudaDeviceEnablePeerAccess|cudaDeviceDisablePeerAccess|cuCtxEnablePeerAccess|cudaMemcpyPeer(?:Async)?|"
                           r"cudaMemcpy3DPeer(?:Async)?|cuMemcpyPeer(?:Async)?|cuMemcpy3DPeer(?:Async)?)")
    FORBIDDEN_HIP = re.compile(r"(hipDeviceEnablePeerAccess|hipDeviceDisablePeerAccess|hipMemcpyPeer(?:Async)?|hipMemcpy3DPeer(?:Async)?|hipCtxEnablePeerAccess)")
    HIP_PEER_FILE = "src/split/src/hip/peer.h"

    def _code_lines(self):
        for p in sorted((ROOT.parent / "engine" / "src" / "split").rglob("*")):
            if p.suffix not in (".cu", ".cuh", ".cpp", ".hpp", ".h"):
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                yield p, n, line, line.split("//", 1)[0]

    def test_no_cuda_peer_call_in_the_engine_sources(self):
        bad = [f"{p.relative_to(ROOT)}:{n}: {line.strip()[:100]}" for p, n, line, code in self._code_lines() if self.FORBIDDEN.search(code)]
        self.assertEqual(bad, [], "CUDA peer access in the split engine:\n" + "\n".join(bad))

    def test_hip_peer_calls_only_in_the_one_probed_file(self):
        bad = [f"{p.relative_to(ROOT.parent / "engine").as_posix()}:{n}: {line.strip()[:100]}" for p, n, line, code in self._code_lines()
               if self.FORBIDDEN_HIP.search(code) and p.relative_to(ROOT.parent / "engine").as_posix() != self.HIP_PEER_FILE]
        self.assertEqual(bad, [], "HIP peer access outside " + self.HIP_PEER_FILE + ":\n" + "\n".join(bad))

    def test_the_hip_peer_file_enables_access_but_never_uses_the_copy_engine(self):
        text = (ROOT.parent / "engine" / self.HIP_PEER_FILE).read_text(encoding="utf-8")
        code = "\n".join(l.split("//", 1)[0] for l in text.splitlines())
        self.assertIn("hipDeviceEnablePeerAccess", code)
        self.assertNotRegex(code, r"hipMemcpy(?:3D)?Peer")   # kernel stores only (the owner's condition), no peer copies

    def test_the_peer_file_is_only_compiled_into_the_hip_build_and_the_default_is_pinned(self):
        engine = (ROOT.parent / "engine" / "src" / "split" / "src" / "engine.cu").read_text(encoding="utf-8")
        i = engine.index('#include "hip/peer.h"')
        self.assertEqual(engine[:i].rstrip().splitlines()[-1].strip(), "#if FNX_HIP")
        policy = (ROOT.parent / "engine" / "src" / "split" / "src" / "hip" / "peer_policy.hpp").read_text(encoding="utf-8")
        self.assertRegex(policy, r"constexpr bool PEER_DEFAULT_ON = (false|true);")   # flipped to true only on screens that show it wins

    def test_rccl_is_loaded_at_run_time_never_linked_and_its_default_is_pinned(self):
        # the packaging rule (coordinator, 11-02): dlopen only, no link; the split must start without librccl, falling back with one line
        rd = (ROOT.parent / "engine" / "src" / "split" / "src" / "hip" / "rccl_dl.hpp").read_text(encoding="utf-8")
        self.assertIn("dlopen(", rd)
        self.assertNotIn("hip_runtime", rd)   # CPU only: tools/rccl_policy_test.cpp forces a missing library without a card
        self.assertRegex(rd, r"constexpr bool RCCL_DEFAULT_ON = (false|true);")
        cmake = (ROOT.parent / "engine" / "src" / "split" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotRegex(cmake.replace("rccl_policy_test", ""), r"(?i)rccl")   # no find_package(rccl), no -lrccl
        for p in (ROOT.parent / "engine" / "src" / "split" / "src").rglob("*"):
            if p.suffix in (".cu", ".cuh", ".cpp", ".hpp", ".h") and p.name not in ("rccl.h", "rccl_dl.hpp", "prefill.cu", "engine.cu", "engine.hpp"):
                self.assertNotIn("rccl.h>", p.read_text(encoding="utf-8", errors="ignore"), p.name)

    def test_the_peer_probe_and_the_27b_files_are_gone(self):
        names = {p.name for p in (ROOT.parent / "engine" / "src" / "split" / "src").iterdir()}
        self.assertNotIn("peer_probe.cu", names)
        self.assertFalse([n for n in names if n.startswith("s27_")])

    def test_the_licence_and_the_credit_travel_with_the_code(self):
        lic = (ROOT.parent / "engine" / "third_party" / "sfnx" / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("John Fink", lic)
        self.assertIn("MIT License", lic)
        notice = (ROOT.parent / "engine" / "third_party" / "sfnx" / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn("John Fink", notice)
        self.assertIn("github.com/johnfink8/sfnx", notice)


class LaunchManifest(unittest.TestCase):
    """P0.1 (lite): STRATA_MANIFEST=<file> records the engine choice and why in <file>.launch.json. Off unless set; the same launch gives the same text;
    the AMD auto rule is recorded condition by condition, so a two- or three-card setup is not read as the four-card one."""

    R9700 = dict(name="AMD Radeon AI PRO R9700", vram_mib=32624, cc=0, vendor="amd", arch="gfx1201")
    IQ3S = dict(routed_types={"IQ2_S": 40, "IQ3_XXS": 34, "IQ3_S": 20, "IQ4_XS": 2, "IQ4_NL": 39, "Q2_0": 9}, routed_bytes=int(49 * GIB), embd_type="IQ4_XS",
                shexp_types={"IQ4_XS": 60, "Q4_K": 40, "Q5_K": 20, "IQ4_NL": 24}, embd_ram_bytes=int(27.2 * GIB), layers=48, experts=512,
                ple_type="IQ4_NL", output_type="Q6_K")

    def decided(self, n, model=None, **card):
        spec = dict(self.R9700)
        spec.update(card)
        return S.decide(enabled=True, system="linux", cards=cards(n, **spec), ram_total_gib=125, ram_available_gib=80,
                        model=model or S.ModelFacts(**self.IQ3S), allow_untested=True)

    def test_off_unless_the_variable_is_set(self):
        import tempfile

        d = self.decided(4)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(S.record_launch_manifest(d, "auto", environ={}))
            self.assertFalse(S.record_launch_manifest(d, "auto", environ={"STRATA_MANIFEST": ""}))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_four_gfx1201_with_the_iq3_s_set_is_recognised(self):
        import json
        import tempfile

        d = self.decided(4)
        self.assertTrue(d.use_split, d.reasons)
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "m.json")
            self.assertTrue(S.record_launch_manifest(d, "auto", environ={"STRATA_MANIFEST": target}))
            doc = json.loads(Path(target + ".launch.json").read_text(encoding="utf-8"))
        launch = doc["launch"]
        self.assertEqual((launch["engine"], launch["cards"], launch["card_arch"]), ("all-cards-split", 4, "gfx1201"))
        self.assertEqual(launch["amd_auto_rule"], {"architecture": True, "card_count": True, "expert_types": True})
        self.assertIn("IQ3_S", launch["routed_types"])

    def test_two_and_three_cards_are_not_promoted_by_the_four_card_rule(self):
        for n in (2, 3):
            rule = S.auto_rule(self.decided(n))
            self.assertEqual(rule, {"architecture": True, "card_count": False, "expert_types": True}, n)

    def test_another_expert_type_or_an_nvidia_card_misses_a_condition(self):
        ud = S.ModelFacts(**{**self.IQ3S, "routed_types": {"IQ3_S": 90, "Q3_K": 4}})
        d = self.decided(4, model=ud)
        self.assertFalse(d.use_split)                         # the loader reads no Q3_K experts: a plain refusal, with the reason recorded
        self.assertTrue(any("Q3_K" in r for r in d.reasons), d.reasons)
        self.assertEqual(S.auto_rule(d), {"architecture": True, "card_count": True, "expert_types": False})
        nv = S.decide(enabled=True, system="linux", cards=cards(2), ram_total_gib=125, ram_available_gib=80, model=udq4())
        self.assertIsNone(S.auto_rule(nv))

    def test_matched_launches_give_the_same_text(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            texts = []
            for name in ("a", "b"):
                target = str(Path(tmp) / name)
                S.record_launch_manifest(self.decided(4), "auto", environ={"STRATA_MANIFEST": target})
                texts.append(Path(target + ".launch.json").read_text(encoding="utf-8"))
            self.assertEqual(texts[0], texts[1])
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_an_unwritable_target_never_stops_a_start(self):
        self.assertFalse(S.record_launch_manifest(self.decided(4), "auto", environ={"STRATA_MANIFEST": "/no/such/dir/m.json"}))


if __name__ == "__main__":
    unittest.main()
