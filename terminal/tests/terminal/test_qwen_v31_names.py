"""Round 0214: Unsloth's v3.1 Qwen3.8-Flash-Next sets in the Terminal's model names (UD-IQ1_M: the Unsloth set, bare or with --family unsloth)."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, catalog

PC = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia"}


class Names(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(PC))
        p.start()
        self.addCleanup(p.stop)

    def test_the_new_names_resolve_to_the_unsloth_family(self):
        cat = catalog(PC)
        for m in ("UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31"):
            self.assertEqual(cat.resolve(m.lower()), ("unsloth", m))
        self.assertEqual(cat.resolve("ud-iq4_xs"), ("unsloth", "UD-IQ4_XS"))

    def test_ud_iq1_m_is_the_unsloth_set(self):
        cat = catalog(PC)
        self.assertEqual(cat.resolve("ud-iq1_m"), ("unsloth", "UD-IQ1_M"))
        self.assertEqual(cat.resolve("ud-iq1_m", "unsloth"), ("unsloth", "UD-IQ1_M"))


if __name__ == "__main__":
    unittest.main()


class AdoptUnsloth(unittest.TestCase):
    """ENG-AMD (round 0214): `strata launch` refused every Unsloth set on an Intel Arc card because adopt.FAMILIES lacked "unsloth"."""

    def _plan(self, tmp, model, hub, n, backend="sycl", family="unsloth"):
        from pathlib import Path
        from strata_cli.launch import adopt
        d = Path(tmp) / "data"
        m = d / "models" / f"unsloth-{model}"
        m.mkdir(parents=True)
        for i in range(1, n + 1):
            (m / f"Qwen3.8-Flash-Next-{hub}-{i:05d}-of-{n:05d}.gguf").write_bytes(b"GGUF")
        pack = d / "packs" / f"unsloth-{model.lower()}"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "experts.bin").write_bytes(b"x")
        src = Path(__file__).resolve().parents[2]
        cfg, problems = adopt.plan(family=family, model=model, model_dir=m, data_dir=d, src=src, context=32768, kv=None,
                                   model_name="qwen3.8-flash-next-unsloth-" + model.lower(), family_def={}, log=Path(tmp) / "x.log",
                                   backend=backend, pack_name=f"unsloth-{model.lower()}")
        return cfg, problems

    def test_every_new_set_is_written_for_an_arc_card(self):
        import tempfile
        from unittest import mock
        from strata_cli.launch import adopt
        self.assertIn("unsloth", adopt.FAMILIES)
        for model, hub, n in (("UD-IQ1_M", "UD-IQ1_M", 3), ("UD-Q2_0", "UD-Q2_0", 3), ("UD-IQ2_XS", "UD-IQ2_XS", 3), ("UD-Q2_K_XL", "UD-Q2_K_XL", 4),
                              ("UD-IQ3_XXS", "UD-IQ3_XXS", 4), ("UD-Q3_K_XL", "UD-Q3_K_XL", 4), ("UD-IQ4_XS-v31", "UD-IQ4_XS", 4), ("UD-Q4_K_XL", "UD-Q4_K_XL", 4)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as t, mock.patch.object(adopt, "intel_driver", return_value="i915"):
                cfg, problems = self._plan(t, model, hub, n)
                self.assertEqual(problems, [])
                a = cfg["args"]
                self.assertTrue(a[a.index("--native") + 1].endswith(f"{hub}-00001-of-{n:05d}.gguf"))
                self.assertNotIn("--ple-gguf", a)            # the engine finds the n-gram table by its tensor name (setup.py writes no --ple-gguf for 3+ shards)
                self.assertEqual(cfg["backend"], "sycl")

    def test_the_other_families_still_pass_the_table_shard(self):
        import tempfile
        from strata_cli.launch import adopt
        with tempfile.TemporaryDirectory() as t:
            cfg, problems = self._plan(t, "IQ3_XXS", "GSQ-RCO-IQ3_XXS", 2, family="qwen")
            self.assertEqual(problems, [])
            self.assertIn("--ple-gguf", cfg["args"])
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(adopt.plan(family="deepseek41", model="x", model_dir=t, data_dir=t, src=t, context=1, kv=None, model_name="x",
                                        family_def={}, log=t, backend="sycl")[0], None)
