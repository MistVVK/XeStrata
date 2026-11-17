"""Round 0214: Unsloth's v3.1 Qwen3.8-Flash-Next sets (revision d9620119, 2026-11-04) in setup.py's catalog: UD-IQ1_M, UD-Q2_0, UD-IQ2_XS,
UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL and the new UD-IQ4_XS (catalog name UD-IQ4_XS-v31), the pinned UD-Q4_K_XL / UD-IQ4_XS / UD-Q5_K_XL /
UD-Q6_K_XL / Q8_0 unchanged at 38bb39ee, UD-Q4_K_XL's second (four-shard, n-gram table last) layout taken from local files, the manual-load
rule (#444), and a header regression for every set of the repository.  The tensor directories are tests/fixtures/unsloth_v31/headers.json.gz
(the first 4 MB of each shard, read with HTTP range requests; tools/unsloth_v31_fixture.py).  Mocked: no GPU, no downloads.

    python -m pytest tools/test_setup_unsloth_v31.py
"""
from __future__ import annotations

import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import unsloth_v31_fixture as FX  # noqa: E402
from test_setup_unsloth import Base, FakeGGUF  # noqa: E402

DOC = FX.load()
REV31 = "d96201194f103c906994f283f504209dc8d2a10c"
REV_OLD = "38bb39ee97821de2c9009abb7e93950eec396e66"
NEW = ["UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31"]
HUB = {m: ("UD-IQ4_XS" if m == "UD-IQ4_XS-v31" else m) for m in NEW}     # the name on the Hub
FAM = setup.FAMILIES["unsloth"]
TABLE_BYTES = 28800138432             # per_layer_token_embd in IQ4_NL: the last shard of every set below Q5
Q5_0_DOWN = {"UD-IQ2_XS", "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-IQ4_XS-v31"}


def shards_of(set_name):
    return DOC["sets"][set_name]["shards"]


def pins_of(set_name):
    return {s["name"]: (s["bytes"], s["sha256"]) for s in shards_of(set_name)}


class Catalog(unittest.TestCase):
    def test_every_new_set_is_pinned_by_the_hub_files(self):
        for model in NEW:
            with self.subTest(model=model):
                d, shards = setup.MODELS[model], shards_of(HUB[model])
                self.assertEqual(d["shards"], len(shards))
                names = [setup.model_file(FAM, model, i) for i in range(1, d["shards"] + 1)]
                self.assertEqual(names, [s["name"] for s in shards])
                for s in shards:
                    self.assertEqual(FAM["sha256"][s["name"]], (s["bytes"], s["sha256"]), s["name"])
                    self.assertEqual(setup.UNSLOTH_V31_SHARDS[s["name"]], (s["bytes"], s["sha256"]))
                    self.assertRegex(s["sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(d["download_gb"], round(sum(s["bytes"] for s in shards) / 1e9, 1))
                self.assertEqual(d["revision"], REV31)
                self.assertEqual(d["quant"], HUB[model])
                self.assertEqual(d["families"], ("unsloth",))
                self.assertTrue(d["budget"] and d["vision"] is True)
                self.assertEqual(d["engine"], (0, 1, 57))
                self.assertEqual(d["ram_gb"], setup.UNSLOTH_MIN_RAM_GB)          # the RAM-budget floor of UD-Q4_K_XL and UD-IQ4_XS
                self.assertEqual(setup.model_quant(model), HUB[model])

    def test_arena_is_the_routed_experts_bytes_of_the_headers(self):
        for model in NEW:
            with self.subTest(model=model):
                exp = 0
                for s in shards_of(HUB[model]):
                    for name, type_id, shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                        if name.endswith("_exps.weight") and "shexp" not in name:
                            t = FX.G.TensorInfo(name, shape, type_id, FX.G.GGML_TYPES[type_id], 0)
                            exp += t.expected_bytes()
                self.assertEqual(setup.MODELS[model]["arena_gb"], round(exp / 1e9, 1))

    def test_the_pinned_sets_stay_at_their_commit(self):
        self.assertEqual(setup.HF_REVISIONS[setup.UNSLOTH_REPO], REV_OLD)
        self.assertEqual(setup.UNSLOTH_V31_REVISION, REV31)
        self.assertEqual(setup.hf(setup.UNSLOTH_REPO), f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_OLD}/")
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertNotIn("revision", setup.MODELS[model], model)
            self.assertEqual(setup.model_hf(FAM, model), f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV_OLD}/{model}/")
        self.assertEqual(setup.model_file(FAM, "UD-IQ4_XS", 3), "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf")
        for model in NEW:
            self.assertEqual(setup.model_hf(FAM, model), f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV31}/{HUB[model]}/")

    def test_the_old_pins_are_untouched(self):
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_SHARDS.values()), 111334654784)
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_IQ4_XS_SHARDS.values()), 93682584224)
        self.assertEqual(setup.UNSLOTH_IQ4_XS_SHARDS["Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"][1],
                         "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7")
        # Q5 / Q6 / Q8 were not re-uploaded: the Hub's files at d9620119 are the pinned ones
        for set_name in ("UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            for n, pin in pins_of(set_name).items():
                self.assertEqual(setup.UNSLOTH_BIG_SHARDS[n], pin, n)
        # no name is pinned twice with different bytes except UD-Q4_K_XL's, which are the alternative layout
        for n in setup.UNSLOTH_V31_SHARDS:
            self.assertNotIn(n, setup.UNSLOTH_SHARDS)
            self.assertNotIn(n, setup.UNSLOTH_IQ4_XS_SHARDS)
            self.assertNotIn(n, setup.UNSLOTH_BIG_SHARDS)

    def test_a_file_is_pinned_by_the_commit_its_url_names(self):
        n = "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf"
        self.assertEqual(setup.file_pin(setup.model_hf(FAM, "UD-IQ1_M") + n), setup.UNSLOTH_V31_SHARDS[n])
        q4 = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf"
        self.assertEqual(setup.file_pin(setup.model_hf(FAM, "UD-Q4_K_XL") + q4), setup.UNSLOTH_SHARDS[q4])
        self.assertEqual(setup.file_pin(setup.hf(setup.UNSLOTH_REPO).replace(REV_OLD, REV31) + "UD-Q4_K_XL/" + q4),
                         setup.UNSLOTH_Q4_V31_SHARDS[q4])
        self.assertNotEqual(setup.UNSLOTH_SHARDS[q4], setup.UNSLOTH_Q4_V31_SHARDS[q4])
        self.assertIsNone(setup.file_pin("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/" + REV31 + "/UD-IQ1_S/x-00001-of-00003.gguf"))

    def test_the_model_names_resolve_in_both_directions(self):
        self.assertEqual(setup.FAMILIES["unsloth"]["tag"] + "UD-IQ4_XS-v31", "unsloth-UD-IQ4_XS-v31")
        with tempfile.TemporaryDirectory() as t:
            for stem, model in (("strata-unsloth-ud-iq4_xs-v31", "UD-IQ4_XS-v31"), ("strata-unsloth-ud-iq4_xs", "UD-IQ4_XS"),
                                ("strata-unsloth-ud-iq1_m", "UD-IQ1_M"), ("strata-unsloth-ud-q3_k_xl", "UD-Q3_K_XL")):
                p = Path(t) / (stem + ".json")
                p.write_text('{"args": ["--max-context", "8192"]}')
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), ("unsloth", model), stem)

    def test_budget_model_tells_the_three_and_the_four_shard_iq4_xs_apart(self):
        for model in NEW:
            first = "/m/" + setup.model_file(FAM, model, 1)
            self.assertEqual(setup.budget_model({"args": ["--native", first]}), model, model)
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"]}), "UD-IQ4_XS")
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]}), "UD-Q4_K_XL")
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"]}), "Q8_0")

    def test_the_budget_rule_for_the_smaller_sets(self):
        # every expert in RAM where the experts and 16 GiB more fit; the budget mode below that
        self.assertFalse(setup.unsloth_budget("UD-IQ1_M", 64))
        self.assertTrue(setup.unsloth_budget("UD-IQ4_XS-v31", 64))
        self.assertFalse(setup.unsloth_budget("UD-IQ4_XS-v31", 96))
        self.assertEqual(setup.resident_budget_gib("UD-IQ3_XXS", 63.7), 40)
        self.assertEqual(setup.resident_budget_gib("UD-IQ1_M", 96), 32)            # at most all of them (34.8 GB = 32.4 GiB)
        self.assertAlmostEqual(setup.unsloth_split_need_gb("UD-Q3_K_XL"), 89.5 + setup.UNSLOTH_RAM_LEFT_GB)

    def test_vision_is_on_like_the_other_unsloth_sets(self):
        for model in NEW:
            self.assertTrue(setup.images_default(model, "unsloth", 64.0))


class LastShardHoldsTheTable(unittest.TestCase):
    """The n-gram table (per_layer_token_embd, IQ4_NL) is a shard of its own, the LAST one, in every set below Q5."""

    def test_the_table_is_the_last_shard_and_shard_1_holds_no_tensor(self):
        for set_name in ("UD-IQ1_M", "UD-IQ1_S", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(set=set_name):
                shards = shards_of(set_name)
                last = shards[-1]
                self.assertEqual(last["bytes"], TABLE_BYTES)
                self.assertEqual([t[0] for t in last["header"]["tensors"]], ["per_layer_token_embd.weight"])
                self.assertEqual(last["header"]["tensors"][0][1], 20)                 # IQ4_NL
                self.assertEqual(last["header"]["tensors"][0][2], [160, 320001536])
                for s in shards[1:-1]:
                    self.assertNotIn("per_layer_token_embd.weight", [t[0] for t in s["header"]["tensors"]])
                self.assertIsNone(shards[0]["header"])                              # metadata and tokenizer only (longer than the 4 MB read)

    def test_the_table_shard_is_the_same_bytes_in_each_layout(self):
        by_count = {}
        for set_name in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS", "UD-Q4_K_XL"):
            shards = shards_of(set_name)
            by_count.setdefault(len(shards), set()).add(shards[-1]["sha256"])
        self.assertEqual({k: len(v) for k, v in by_count.items()}, {3: 1, 4: 1})     # one file for the 3-shard sets, one for the 4-shard sets (they differ in the split header)


class FinalShardGGUF(FakeGGUF):
    """gguf_reader.GGUFFile for a set whose LAST shard holds the table."""
    def __init__(self, path):
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", str(path))
        names = ["per_layer_token_embd.weight"] if m and m.group(1) == m.group(2) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


class Install(Base):
    """setup.main() for each new set: the files of the commit it names, their pins, the pack from shard 1, the table found in the last shard."""

    def test_each_new_set_installs_from_its_commit(self):
        for model in NEW:
            with self.subTest(model=model), mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FinalShardGGUF)}):
                self.setUp()
                code, out, cfg = self._main(model)
                self.assertEqual(code, 0, out)
                hub, d = HUB[model], setup.MODELS[model]
                base = f"https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/{REV31}/{hub}/"
                names = [s["name"] for s in shards_of(hub)]
                self.assertEqual(self.downloads[:-1], [base + n for n in names])
                self.assertIn("mmproj", self.downloads[-1])
                self.assertEqual(self.verified, [(n, *pins_of(hub)[n]) for n in names])
                packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
                self.assertEqual(len(packs), 1, self.runs)
                self.assertIn("--compat-bf16", packs[0])
                self.assertNotIn("--experts-bin", packs[0])
                self.assertTrue(packs[0][packs[0].index("--gguf") + 1].endswith(names[0]))
                args = cfg["args"]
                self.assertTrue(args[args.index("--native") + 1].endswith(names[0]))
                self.assertNotIn("--ple-gguf", args)                                # more than two shards: the engine finds the table by its tensor name
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-" + model.lower())
                self.assertEqual(args[args.index("--pack") + 1], str(self.t / "data" / "packs" / ("unsloth-" + model.lower())))
                self.assertEqual(setup.budget_model(cfg), model)
                self.assertIn("--vision", args)
                self.tmp.cleanup()

    def _main(self, model, **kw):
        return self.main(["--context", "8192"], version="0.1.57", m=model, **kw)

    def test_an_engine_older_than_0157_stops_before_the_download(self):
        code, out, cfg = self.main(["--context", "8192"], version="0.1.56", m="UD-IQ1_M")
        self.assertEqual(code, 1, out)
        self.assertIn("needs engine 0.1.57 or newer", out)
        self.assertEqual(self.downloads, [])

    def test_the_pinned_ones_still_download_from_38bb39ee(self):
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            self.setUp()
            code, out, cfg = self.main(["--context", "8192"], version="0.1.57", m=model)
            self.assertEqual(code, 0, out)
            self.assertTrue(all(f"/resolve/{REV_OLD}/{model}/" in u for u in self.downloads[:-1]), self.downloads)
            self.tmp.cleanup()

    def test_check_lists_the_new_sets(self):
        code, out, _cfg = self.main(["--check"], model=False, family=False)
        self.assertIn(code, (0, None), out)
        for model in NEW:
            self.assertRegex(out, rf"(?m){re.escape(model)}\s+needs ~\d+ GB of RAM \+ VRAM: (fits|may run slower)$")


class FitLabels(Base):
    """0.2.13 (owner): the weight files without the lookup table against the RAM and the card's memory together: "fits" or "may run slower", in the size menu,
    in --check and in the install's RAM warning.  The screenshot PC: a V100 32 GB with 31 GB of RAM (63 GB)."""

    def _line(self, out, model):
        return next(x for x in out.splitlines() if re.search(rf"\d\) {re.escape(model)} ", x))

    def test_the_menu_labels_and_the_warning_follow_the_one_rule(self):
        code, out, _cfg = self.main(["--context", "8192"], ram=31.0, vram=32.0, version="0.1.57", m="UD-IQ1_M")
        self.assertEqual(code, 0, out)
        for m in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL"):      # 39 to 61 GB of files (download minus the 28.8 GB table), 63 GB here
            self.assertTrue(self._line(out, m).endswith("   <- fits"), (m, self._line(out, m)))
        for m in ("UD-IQ4_XS-v31", "UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):         # UD-IQ4_XS-v31 is 64.9 GB
            self.assertTrue(self._line(out, m).endswith("   <- may run slower"), (m, self._line(out, m)))
        self.assertNotIn("<- needs 48 GB RAM", out)                          # the RAM-floor line is gone
        self.assertNotIn("single-digit tokens/s", out)                       # UD-IQ1_M fits: the install does not warn against it
        self.tmp.cleanup()
        self.setUp()
        code, out, _cfg = self.main(["--context", "8192"], ram=31.0, vram=11.9, version="0.1.57", m="UD-IQ1_M")   # 43 GB: 39 GB of files fit, UD-IQ3_XXS (55) does not
        self.assertEqual(code, 0, out)
        self.assertTrue(self._line(out, "UD-IQ1_M").endswith("   <- fits"))
        self.assertTrue(self._line(out, "UD-IQ3_XXS").endswith("   <- may run slower"))
        self.tmp.cleanup()
        self.setUp()
        code, out, _cfg = self.main(["--context", "8192"], ram=20.0, vram=11.9, version="0.1.57", m="UD-IQ1_M")   # 32 GB: 39 GB of files do not
        self.assertTrue(self._line(out, "UD-IQ1_M").endswith("   <- may run slower"))
        self.assertIn("single-digit tokens/s", out)                          # the install's own warning stays where the set does not fit

    def test_check_says_fits_or_may_run_slower(self):
        code, out, _cfg = self.main(["--check"], ram=31.0, vram=32.0, model=False, family=False)
        self.assertIn(code, (0, None), out)
        for model in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL"):
            self.assertRegex(out, rf"(?m){re.escape(model)}\s+needs ~\d+ GB of RAM \+ VRAM: fits$")
        self.assertRegex(out, r"(?m)UD-IQ4_XS-v31\s+needs ~65 GB of RAM \+ VRAM: may run slower$")
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertRegex(out, rf"(?m){re.escape(model)}\s+needs ~\d+ GB of RAM \+ VRAM: may run slower$")
        for old in ("fits, a bit slower", "does not fit", "slow here", "low-RAM mode", "in RAM, the rest"):
            self.assertNotIn(old, out)


class OwnFiles(unittest.TestCase):
    """#444: strata launch --model-path and the app's "Use a local GGUF" go through gguf_dir_shards / gguf_choice / gguf_unsupported."""

    def test_the_new_files_are_no_longer_refused(self):
        for model in NEW:
            for s in shards_of(HUB[model]):
                self.assertIsNone(setup.gguf_unsupported(s["name"]), s["name"])
            first = shards_of(HUB[model])[0]["name"]
            self.assertEqual(setup.gguf_choice(first), ("unsloth", model), first)

    def test_the_approved_texts(self):
        want = {"UD-IQ1_M": "~1-bit (Unsloth Dynamic v3.1), the smallest download", "UD-Q2_0": "2-bit (Unsloth Dynamic v3.1)",
                "UD-IQ2_XS": "2-bit i-quant (Unsloth Dynamic v3.1)", "UD-Q2_K_XL": "2-bit (Unsloth Dynamic v3.1) with some 3-bit experts",
                "UD-IQ3_XXS": "3-bit i-quant (Unsloth Dynamic v3.1)", "UD-Q3_K_XL": "3-bit (Unsloth Dynamic v3.1) with some 4-bit experts",
                "UD-IQ4_XS-v31": "~4-bit i-quant (Unsloth Dynamic v3.1), a newer file than UD-IQ4_XS with other tensor types"}
        from strata_control import mac_fit
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_mcp
        for m, a in want.items():
            self.assertEqual(setup.MODELS[m]["about"], a)
            self.assertEqual(mac_fit.MAC_ABOUT[m], a)
            self.assertEqual(strata_mcp.FALLBACK_MODELS[m]["about"], a)
        for m in ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertIn(m, setup.SUPPORTED_GGUFS)
        self.assertIn("(Unsloth's UD-IQ1_S, other K-quant files) cannot be used", setup.SUPPORTED_GGUFS)
        self.assertEqual(setup.UNSUPPORTED_WHY["UD-IQ1_S"], "UD-IQ1_S mixes two expert formats in one layer, which Strata cannot load; UD-IQ1_M is the 1-bit choice")

    def test_q3_k_xl_really_has_some_4_bit_experts(self):
        n = sum(1 for s in shards_of("UD-Q3_K_XL") for t in (s["header"] or {"tensors": []})["tensors"]
                if t[0].endswith("ffn_gate_exps.weight") and FX.G.GGML_TYPES[t[1]] == "IQ4_XS")
        self.assertEqual(n, 6)

    def test_ud_iq1_s_stays_refused(self):
        for s in shards_of("UD-IQ1_S"):
            self.assertEqual(setup.gguf_unsupported(s["name"]), "UD-IQ1_S")
        self.assertNotIn("UD-IQ1_S", setup.MODELS)
        self.assertIsNone(setup.gguf_choice(shards_of("UD-IQ1_S")[0]["name"]))

    def test_the_bf16_and_other_unsloth_files_stay_refused(self):
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf"), "BF16")
        self.assertIsNone(setup.gguf_unsupported("mmproj-BF16.gguf"))

    def _folder(self, t, set_name, only=None):
        for s in shards_of(set_name):
            if only is None or s["name"] in only:
                (Path(t) / s["name"]).write_bytes(b"")
        return Path(t)

    def test_gguf_dir_shards_finds_all_shards_by_the_count_in_the_name(self):
        for model in NEW:
            with tempfile.TemporaryDirectory() as t:
                folder = self._folder(t, HUB[model])
                got = setup.gguf_dir_shards(folder, FAM, model)
                self.assertEqual([p.name for p in got], [s["name"] for s in shards_of(HUB[model])], model)

    def test_the_v31_iq4_xs_is_not_taken_for_the_pinned_one(self):
        with tempfile.TemporaryDirectory() as t:
            folder = self._folder(t, "UD-IQ4_XS")                                   # the four-shard files
            got = setup.gguf_dir_shards(folder, FAM, "UD-IQ4_XS")                   # asked for the three-shard set
            self.assertEqual(got[0].name, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
            self.assertFalse(got[0].exists())
            problem = setup.gguf_dir_problem(folder, got[0], FAM, "UD-IQ4_XS")
            self.assertIsNotNone(problem)
            self.assertIn("--family unsloth --model UD-IQ4_XS-v31", problem[1])
            ok = setup.gguf_dir_shards(folder, FAM, "UD-IQ4_XS-v31")
            self.assertTrue(all(p.exists() for p in ok))
            self.assertIsNone(setup.gguf_dir_problem(folder, ok[0], FAM, "UD-IQ4_XS-v31"))

    def test_ud_iq1_s_in_a_folder_is_named_as_not_usable(self):
        with tempfile.TemporaryDirectory() as t:
            folder = self._folder(t, "UD-IQ1_S")
            first = setup.gguf_dir_shards(folder, FAM, "UD-IQ1_M")[0]
            problem = setup.gguf_dir_problem(folder, first, FAM, "UD-IQ1_M")
            self.assertIsNotNone(problem)
            self.assertIn("UD-IQ1_S", problem[0] + problem[1])
            self.assertIn(setup.UNSUPPORTED_WHY["UD-IQ1_S"], problem[1])
            msg, hint = setup.gguf_dir_problem(folder, shards_of("UD-IQ1_S")[0]["name"] and (Path(t) / shards_of("UD-IQ1_S")[0]["name"]), FAM, "UD-IQ1_S")
            self.assertIn(setup.UNSUPPORTED_WHY["UD-IQ1_S"], hint)


class SecondQ4Layout(unittest.TestCase):
    """UD-Q4_K_XL at d9620119: the same tensors, the table in a shard of its own.  The pinned files stay what setup downloads; this layout is accepted from files already here."""

    def test_the_names_are_the_pinned_ones_and_the_pins_differ_in_every_shard(self):
        self.assertEqual(sorted(setup.UNSLOTH_Q4_V31_SHARDS), sorted(setup.UNSLOTH_SHARDS))
        for n, pin in setup.UNSLOTH_Q4_V31_SHARDS.items():
            self.assertEqual(pins_of("UD-Q4_K_XL")[n], pin)
            self.assertNotEqual(pin[0], setup.UNSLOTH_SHARDS[n][0], n)                # the shard boundaries moved: no size is shared
            self.assertEqual(FAM["sha256"][n], setup.UNSLOTH_SHARDS[n])               # the family's pin (what is downloaded) stays the old one
            self.assertEqual(FAM["sha256_alt"][n], (pin,))
        self.assertEqual(sum(b for b, _ in setup.UNSLOTH_Q4_V31_SHARDS.values()), sum(s["bytes"] for s in shards_of("UD-Q4_K_XL")))

    @staticmethod
    def _file(name, size, exists=True):
        return types.SimpleNamespace(name=name, exists=lambda: exists, stat=lambda: types.SimpleNamespace(st_size=size))

    def test_fam_pin_picks_the_layout_by_the_size_on_the_disk(self):
        for n, old in setup.UNSLOTH_SHARDS.items():
            new = setup.UNSLOTH_Q4_V31_SHARDS[n]
            self.assertEqual(setup.fam_pin(FAM, self._file(n, old[0])), old, n)
            self.assertEqual(setup.fam_pin(FAM, self._file(n, new[0])), new, n)
            self.assertEqual(setup.fam_pin(FAM, self._file(n, 123)), old, n)                  # neither: the pinned one (verify_sha256 then reports it)
            self.assertEqual(setup.fam_pin(FAM, self._file(n, new[0], exists=False)), old, n)   # nothing there yet: the download's pin
        other = "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf"
        self.assertEqual(setup.fam_pin(FAM, self._file(other, 1)), setup.UNSLOTH_V31_SHARDS[other])

    def test_the_catalog_reads_both_layouts_as_ud_q4_k_xl(self):
        self.assertEqual(setup.gguf_choice(shards_of("UD-Q4_K_XL")[0]["name"]), ("unsloth", "UD-Q4_K_XL"))
        self.assertIsNone(setup.gguf_unsupported(shards_of("UD-Q4_K_XL")[0]["name"]))

    def test_verify_sha256_takes_the_second_layout_from_the_users_folder(self):
        """--gguf-dir with the new files: each is checked against ITS pin; the old pin would have said 'not the published file'."""
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
            f.write_bytes(b"x" * 100)
            alt = (100, "ab" * 32)
            fam = {"sha256": {f.name: setup.UNSLOTH_SHARDS[f.name]}, "sha256_alt": {f.name: (alt,)}}
            self.assertEqual(setup.fam_pin(fam, f), alt)
            with mock.patch.object(setup, "_hash_file", return_value=alt[1]):
                setup.verify_sha256(f, *setup.fam_pin(fam, f), own=True)
            self.assertIn("sha256 " + alt[1], f.with_name(f.name + ".done").read_text())


class Pinning(Base):
    def test_the_install_checks_the_files_with_fam_pin(self):
        calls = []
        real = setup.fam_pin
        with mock.patch.object(setup, "fam_pin", lambda fam, s: calls.append(s.name) or real(fam, s)):
            code, out, cfg = self.main(["--context", "8192"], version="0.1.57", m="UD-Q4_K_XL")
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, list(setup.UNSLOTH_SHARDS))


class FileSetsAndSplit(unittest.TestCase):
    """strata_control: the known-file-set table the app's "Use a local GGUF" reads, and the split engine's clean decline."""

    def test_known_file_sets(self):
        from strata_control import models
        for model in NEW:
            self.assertIn("unsloth/" + model, models.SUPPORTED_SELECTIONS)
            got = models._candidate_selection([Path("/m") / s["name"] for s in shards_of(HUB[model])])
            self.assertEqual((got["family"], got["model"]), ("unsloth", model), model)
            self.assertEqual(got["expected_tensor_count"], 1224)
        three = [Path("/m") / n for n in setup.UNSLOTH_IQ4_XS_SHARDS]
        self.assertEqual(models._candidate_selection(three)["model"], "UD-IQ4_XS")
        q4 = [Path("/m") / n for n in setup.UNSLOTH_Q4_V31_SHARDS]
        self.assertEqual(models._candidate_selection(q4)["model"], "UD-Q4_K_XL")

    def _facts(self, set_name):
        from strata_control import split_engine
        with tempfile.TemporaryDirectory() as t:
            first = None
            for i, s in enumerate(shards_of(set_name)):
                if s["header"]:
                    p = FX.materialize(s, Path(t) / s["name"], first=(i == 0))
                else:
                    p = FX.materialize({"header": {"split": [0, len(shards_of(set_name)), 1224], "tensors": []}}, Path(t) / s["name"], first=True)
                first = first or p
            return split_engine.scan_model(first)

    def _decide(self, facts):
        from strata_control import split_engine as se
        cards = [se.Card(index=i, name="RTX A6000", vram_mib=49140, cc=86) for i in range(2)]
        return se.decide(enabled=True, system="linux", cards=cards, ram_total_gib=128, ram_available_gib=100, model=facts)

    def test_the_split_engine_declines_what_it_cannot_read(self):
        for model, bad in (("UD-IQ1_M", "IQ2_XXS"), ("UD-Q2_0", "IQ2_XXS"), ("UD-IQ2_XS", "IQ2_XXS"), ("UD-Q2_K_XL", "IQ2_XXS"),
                           ("UD-Q3_K_XL", "Q5_0"), ("UD-IQ4_XS-v31", "Q5_0")):
            with self.subTest(model=model):
                d = self._decide(self._facts(HUB[model]))
                self.assertFalse(d.use_split)
                self.assertTrue(any("routed experts use" in r and bad in r for r in d.reasons), d.reasons)
                self.assertTrue(d.line.startswith("Using the normal layer split"))

    def test_the_split_engine_reads_the_sets_whose_types_it_knows(self):
        for model in ("UD-IQ3_XXS", "UD-Q4_K_XL"):
            d = self._decide(self._facts(HUB.get(model, model)))
            self.assertFalse(any("routed experts use" in r for r in d.reasons), (model, d.reasons))


class HeaderRegression(unittest.TestCase):
    """Every set of the repository at d9620119: the pins and the tensor directories agree with each other."""

    def test_thirteen_sets(self):
        self.assertEqual(sorted(DOC["sets"]), sorted(["BF16", "Q8_0", "UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_XS", "UD-IQ3_XXS", "UD-IQ4_XS", "UD-Q2_0",
                                                      "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL"]))
        self.assertEqual(DOC["revision"], REV31)

    def test_split_keys_and_tensor_counts(self):
        for set_name, d in DOC["sets"].items():
            with self.subTest(set=set_name):
                shards = d["shards"]
                m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", shards[0]["name"])
                self.assertEqual(len(shards), int(m.group(2)))
                names = [s["name"] for s in shards]
                self.assertEqual(names, sorted(names))
                for i, s in enumerate(shards, 1):
                    self.assertIn("-%05d-of-%05d.gguf" % (i, len(shards)), s["name"])
                    h = s["header"]
                    if h is None:
                        continue
                    self.assertEqual(h["split"], [i - 1, len(shards), 1224], s["name"])
                have = [s for s in shards if s["header"]]
                if len(have) == len(shards) - 1 and shards[0]["header"] is None:       # every shard but the metadata shard was read
                    seen = [t[0] for s in have for t in s["header"]["tensors"]]
                    self.assertEqual(len(seen), len(set(seen)))
                    self.assertEqual(len(seen), 1224)

    def test_every_tensor_lies_inside_its_pinned_shard(self):
        for set_name, d in DOC["sets"].items():
            for s in d["shards"]:
                h = s["header"]
                if not h:
                    continue
                with self.subTest(shard=s["name"]):
                    end = 0
                    for name, type_id, shape, off in h["tensors"]:
                        t = FX.G.TensorInfo(name, shape, type_id, FX.G.GGML_TYPES.get(type_id, "?"), off)
                        self.assertIsNotNone(t.expected_bytes(), f"{name}: type {type_id}")
                        end = max(end, off + t.expected_bytes())
                    self.assertLessEqual(h["data_start"] + end, s["bytes"])
                    self.assertGreaterEqual(h["data_start"] + end, s["bytes"] - 64)     # (the file ends where its last tensor does)

    def test_the_catalog_sets_have_the_tensor_mix_of_the_plan(self):
        """PLAN-QUANTS section 2: routed gate/up types by layer, down types, the Q5_0 downs that decide the Mac and the split engine."""
        want = {
            "UD-IQ1_M": ({"IQ2_XXS": 23, "IQ2_S": 18, "IQ2_XS": 7}, {"Q2_0": 46, "IQ4_NL": 2}),
            "UD-Q2_0": ({"IQ2_XXS": 18, "IQ2_S": 30}, {"Q2_0": 39, "IQ4_NL": 9}),
            "UD-IQ2_XS": ({"IQ2_XXS": 8, "IQ2_S": 40}, {"IQ4_NL": 24, "Q2_0": 23, "Q5_0": 1}),
            "UD-Q2_K_XL": ({"IQ2_XXS": 5, "IQ2_S": 26, "IQ3_XXS": 17}, {"IQ4_NL": 31, "Q2_0": 16, "Q5_0": 1}),
            "UD-IQ3_XXS": ({"IQ2_S": 26, "IQ3_XXS": 21, "IQ4_XS": 1}, {"IQ4_NL": 45, "Q2_0": 3}),
            "UD-Q3_K_XL": ({"IQ2_S": 4, "IQ3_XXS": 38, "IQ4_XS": 6}, {"IQ4_NL": 26, "Q5_0": 20, "Q2_0": 2}),
            "UD-IQ4_XS": ({"IQ3_XXS": 28, "IQ4_XS": 17, "IQ2_S": 3}, {"Q5_0": 33, "IQ4_NL": 13, "Q2_0": 2}),
            "UD-Q4_K_XL": ({"Q4_K": 47, "Q5_K": 1}, {"Q5_1": 43, "Q8_0": 5}),
            "UD-Q5_K_XL": ({"Q5_K": 47, "Q6_K": 1}, {"Q8_0": 48}),
            "UD-Q6_K_XL": ({"Q6_K": 47, "Q8_0": 1}, {"Q8_0": 48}),
        }
        for set_name, (gu, down) in want.items():
            gate, d = {}, {}
            for s in shards_of(set_name):
                for name, type_id, _shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                    ty = FX.G.GGML_TYPES[type_id]
                    if re.fullmatch(r"blk\.\d+\.ffn_gate_exps\.weight", name):
                        gate[ty] = gate.get(ty, 0) + 1
                    elif re.fullmatch(r"blk\.\d+\.ffn_down_exps\.weight", name):
                        d[ty] = d.get(ty, 0) + 1
            with self.subTest(set=set_name):
                self.assertEqual(gate, gu)
                self.assertEqual(d, down)

    def test_ud_iq1_s_layer_18_has_different_gate_and_up_types(self):
        types_ = {}
        for s in shards_of("UD-IQ1_S"):
            for name, type_id, _shape, _off in (s["header"] or {"tensors": []})["tensors"]:
                if name in ("blk.18.ffn_gate_exps.weight", "blk.18.ffn_up_exps.weight"):
                    types_[name] = FX.G.GGML_TYPES[type_id]
        self.assertEqual(types_, {"blk.18.ffn_gate_exps.weight": "IQ2_S", "blk.18.ffn_up_exps.weight": "IQ2_XXS"})

    def test_the_unchanged_sets_have_the_pinned_shards_at_the_new_commit(self):
        for set_name, table in (("UD-Q5_K_XL", None), ("UD-Q6_K_XL", None), ("Q8_0", None)):
            self.assertTrue(all(setup.UNSLOTH_BIG_SHARDS[n] == p for n, p in pins_of(set_name).items()))
        # and the three-shard UD-IQ4_XS at the pinned commit is not the four-shard set at the new one
        self.assertTrue(set(pins_of("UD-IQ4_XS")).isdisjoint(setup.UNSLOTH_IQ4_XS_SHARDS))


if __name__ == "__main__":
    unittest.main()
