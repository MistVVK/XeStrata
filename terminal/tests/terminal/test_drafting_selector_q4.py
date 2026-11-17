"""The DFlash2 pack with the Q4_0 selector (174 MiB less graphics memory): made on this computer from the checked Q4_0 pack, used only when it verifies and the engine reads it.

The idea and the bytes are nazeshinjite's (https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF, Apache-2.0): the community GGUF stores the selector's two
codebooks as Q4_0, and tools/dflash2_pack.py --q4_0-selector writes exactly those bytes (checked against the GGUF with the real checkpoint: RESULTS-DF2MEM.md section 15).
These tests use the small synthetic checkpoint of tools/test_dflash2_pack.py: the real pins cannot be exercised without the 1.2 GB download, so the pinned numbers are
checked for consistency and the mechanism with pins of the synthetic pack.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting, drafting_fetch

ROOT = Path(__file__).resolve().parents[2]
VOCAB_REAL, RANK_REAL = 248320, 256


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


class PinnedNumbers(unittest.TestCase):
    def test_the_converted_pack_is_the_q4_pack_less_the_codebooks_difference(self):
        """CONVERTED_BYTES follows from PACK_BYTES by arithmetic alone: two codebooks of 248,320 x 256 go from bf16 (2 bytes a value) to Q4_0 (18 bytes per 32 values)."""
        bf16 = 2 * VOCAB_REAL * RANK_REAL * 2
        q4 = 2 * VOCAB_REAL * (RANK_REAL // 32) * 18
        self.assertEqual(drafting.PACK_BYTES - drafting.CONVERTED_BYTES, bf16 - q4)
        self.assertEqual(bf16 - q4, 182763520)
        self.assertEqual(len(drafting.CONVERTED_SHA256), 64)
        self.assertNotEqual(drafting.CONVERTED_SHA256, drafting.PACK_SHA256)


class SmallPacks(unittest.TestCase):
    """A synthetic checkpoint packed as the Q4_0 pack, then converted."""

    @classmethod
    def setUpClass(cls):
        cls.synth = _load("t_pack_sel", "tools/test_dflash2_pack.py")
        cls.packer = _load("strata_dflash2_pack_sel", "tools/dflash2_pack.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.synth.write_safetensors(self.dir / "m.safetensors", self.synth.tensors())
        (self.dir / "c.json").write_text(json.dumps(self.synth.CFG))
        self.data = self.dir / "data"
        self.q4 = drafting.target_path(self.data)
        self.q4.parent.mkdir(parents=True)
        self.packer.pack(str(self.dir / "m.safetensors"), str(self.dir / "c.json"), str(self.q4), say=lambda *_: None, q4=True)
        self.src_pins = {"source_bytes": self.q4.stat().st_size, "source_sha256": hashlib.sha256(self.q4.read_bytes()).hexdigest()}
        direct = self.dir / "direct.pack"
        self.packer.pack(str(self.dir / "m.safetensors"), str(self.dir / "c.json"), str(direct), say=lambda *_: None, q4=True, q4_selector=True)
        self.pins = {"size": direct.stat().st_size, "sha256": hashlib.sha256(direct.read_bytes()).hexdigest(), **self.src_pins}
        self.direct = direct

    def convert(self, **extra):
        return drafting_fetch.convert_selector(self.data, **{**self.pins, **extra})

    def test_the_conversion_reproduces_the_pinned_bytes(self):
        """The pack converted from the finished Q4_0 pack is byte-identical to the one packed straight from the checkpoint with --q4_0-selector (the pinned hash)."""
        out = self.convert()
        self.assertEqual(out, drafting.converted_path(self.data))
        self.assertEqual(out.read_bytes(), self.direct.read_bytes())
        self.assertLess(out.stat().st_size, self.q4.stat().st_size)
        self.assertTrue(drafting.converted_ok(self.data, size=self.pins["size"], sha256=self.pins["sha256"]))
        self.assertTrue(self.q4.is_file(), "the Q4_0 pack stays: it is the fallback")

    def test_every_tensor_but_the_two_codebooks_is_carried_over_unchanged(self):
        out = self.convert()
        a, b = self.q4.read_bytes(), out.read_bytes()

        def table(blob):
            n = int.from_bytes(blob[8:12], "little")
            rows = {}
            for line in blob[12:12 + n].decode().split("\n"):
                w = line.split()
                if w and w[0] == "t":
                    rows[w[1]] = (w[2], int(w[5]), int(w[6]))
            return rows, 12 + n
        ta, base_a = table(a)
        tb, base_b = table(b)
        self.assertEqual(set(ta), set(tb))
        books = {"candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"}
        for name in ta:
            ba = a[base_a + ta[name][1]: base_a + ta[name][1] + ta[name][2]]
            bb = b[base_b + tb[name][1]: base_b + tb[name][1] + tb[name][2]]
            if name in books:
                self.assertEqual((ta[name][0], tb[name][0]), ("bf16", "q4_0"))
                self.assertEqual(len(bb) * 32, len(ba) * 9, name)       # 18 bytes per 32 values against 2 bytes a value
            else:
                self.assertEqual(ba, bb, name)

    def test_a_missing_corrupt_or_short_converted_pack_falls_back_to_the_q4_pack(self):
        eng = self.dir / "engine"
        eng.write_bytes(b"\0" * 64 + drafting._MARK_SELQ4 + b"\0" * 64)
        self.assertEqual(drafting.pack_path(self.data, eng), self.q4, "not made yet")
        out = self.convert()
        with mock.patch.object(drafting, "CONVERTED_BYTES", self.pins["size"]), mock.patch.object(drafting, "CONVERTED_SHA256", self.pins["sha256"]):
            self.assertEqual(drafting.pack_path(self.data, eng), out)
            blob = bytearray(out.read_bytes())
            blob[len(blob) // 2] ^= 0xFF                      # one flipped byte, the same size, and the marker still describes the old file
            before = out.stat()
            out.write_bytes(bytes(blob))
            # (a real corruption happens long after the marker was written; on Linux the kernel's file clock ticks in ~4 ms steps, so a rewrite right away can keep the
            #  same mtime_ns and the marker would still describe the file: let the clock move on, as it does in real life)
            os.utime(out, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            self.assertEqual(drafting.pack_path(self.data, eng), self.q4, "a corrupt file (the marker no longer matches it) is not used")
            out.write_bytes(bytes(blob)[:-1000])
            self.assertEqual(drafting.pack_path(self.data, eng), self.q4, "a short file is not used")
            out.unlink()
            self.assertEqual(drafting.pack_path(self.data, eng), self.q4, "a missing file is not used")

    def test_a_q4_pack_that_is_not_the_pinned_one_is_left_alone(self):
        self.assertIsNone(self.convert(source_sha256="0" * 64))
        self.assertFalse(drafting.converted_path(self.data).exists())
        self.assertTrue(self.q4.is_file())

    def test_a_result_that_does_not_match_the_pin_is_deleted(self):
        self.assertIsNone(self.convert(sha256="f" * 64))
        self.assertFalse(drafting.converted_path(self.data).exists())
        self.assertFalse(drafting.converted_path(self.data).with_suffix(".tmp").exists())

    def test_a_failing_conversion_is_never_an_error_for_the_caller(self):
        with mock.patch.object(drafting_fetch, "_packer", side_effect=RuntimeError("no numpy")):
            self.assertIsNone(self.convert())
        self.assertFalse(drafting.converted_path(self.data).exists())

    def test_the_engine_must_read_the_pack_before_it_is_used(self):
        out = self.convert()
        old = self.dir / "old-engine"
        old.write_bytes(b"\0" * 256)
        new = self.dir / "new-engine"
        new.write_bytes(b"\0" * 64 + drafting._MARK_SELQ4 + b"\0" * 64)
        with mock.patch.object(drafting, "CONVERTED_BYTES", self.pins["size"]), mock.patch.object(drafting, "CONVERTED_SHA256", self.pins["sha256"]):
            self.assertEqual(drafting.pack_path(self.data), self.q4, "no engine named: the pack every engine reads")
            self.assertEqual(drafting.pack_path(self.data, old), self.q4, "an engine without the text would read the rows as garbage")
            self.assertEqual(drafting.pack_path(self.data, self.dir / "none"), self.q4)
            self.assertEqual(drafting.pack_path(self.data, new), out)

    def test_auto_arguments_point_at_the_converted_pack(self):
        self.convert()
        new = self.dir / "new-engine"
        new.write_bytes(b"\0" * 64 + drafting._MARK_SELQ4 + b"\0" * 64 + b"--draft-fallback" + b"--mtp-draft-vocab FILE (or --mtp DIR")
        vocab = self.data / drafting.FOLDER / drafting.VOCAB_NAME
        vocab.write_bytes(b"\1\0\0\0" * 8)
        cfg = {"model_name": "qwen3.8-flash-next-iq3_s", "drafting": "dflash2", "args": ["--spec", "4", "--expert-cache", "1000"]}
        with mock.patch.object(drafting, "CONVERTED_BYTES", self.pins["size"]), mock.patch.object(drafting, "CONVERTED_SHA256", self.pins["sha256"]), \
                mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True), mock.patch.object(drafting, "pack_ready", return_value=True):
            drafting.apply(cfg, klass="blackwell", data_dir=self.data, backend="cuda", archs=["sm_120"], engine=new)
            args = cfg["args"]
            self.assertEqual(args[args.index("--draft-model") + 1], str(drafting.converted_path(self.data)))
            cfg2 = {"model_name": "qwen3.8-flash-next-iq3_s", "drafting": "dflash2", "args": ["--spec", "4", "--expert-cache", "1000"]}
            drafting.apply(cfg2, klass="blackwell", data_dir=self.data, backend="cuda", archs=["sm_120"], engine=None)
            self.assertEqual(cfg2["args"][cfg2["args"].index("--draft-model") + 1], str(self.q4), "an engine that is not known gets the pack every engine reads")

    def test_needs_conversion_only_when_dflash2_is_chosen_and_the_q4_pack_is_here(self):
        want = dict(mode="dflash2", model_name="qwen3.8-flash-next-iq3_s", data_dir=self.data, backend="cuda", archs=["sm_120"])
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True), mock.patch.object(drafting, "_here", lambda p: Path(p).is_file()):
            self.assertTrue(drafting.needs_conversion(**want))
            self.assertFalse(drafting.needs_conversion(**{**want, "mode": "mtp"}))
            self.assertFalse(drafting.needs_conversion(**{**want, "data_dir": self.dir / "empty"}))
            self.convert()
            with mock.patch.object(drafting, "CONVERTED_BYTES", self.pins["size"]), mock.patch.object(drafting, "CONVERTED_SHA256", self.pins["sha256"]):
                self.assertFalse(drafting.needs_conversion(**want), "already made")

    def test_setup_and_load_convert_a_start_does_not(self):
        """ensure() (strata drafting fetch, setup) converts a pack that is already here, with no network; the serving side only reads."""
        with mock.patch.object(drafting_fetch, "convert_selector") as convert, mock.patch.object(drafting, "pack_ready", return_value=True),                 mock.patch.object(drafting, "engine_has_selector_q4", return_value=True):
            drafting_fetch.ensure(self.data)
        convert.assert_called_once()
        with mock.patch.object(drafting_fetch, "convert_selector") as convert, mock.patch.object(drafting, "pack_ready", return_value=True),                 mock.patch.object(drafting, "engine_has_selector_q4", return_value=False):
            drafting_fetch.ensure(self.data)
        convert.assert_not_called()          # an engine that cannot read the Q4_0 selector (the SYCL one) or none installed yet: the 340 MB file would never be used
        source = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertNotIn("convert_selector", source)
        self.assertNotIn("convert_selector", (ROOT / "strata_cli" / "launch" / "plan.py").read_text(encoding="utf-8") if (ROOT / "strata_cli" / "launch" / "plan.py").exists() else "")


class TheTrueGraphicsMemory(unittest.TestCase):
    """drafting.vram_gb: 0.72 GB with the converted pack (an engine that reads it), 0.89 GB otherwise; the plan and its text say the same number."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.new = self.dir / "new-engine"
        self.new.write_bytes(b"\0" * 64 + drafting._MARK_SELQ4 + b"\0" * 64)
        self.old = self.dir / "old-engine"
        self.old.write_bytes(b"\0" * 256)
        self.data = self.dir / "data"
        (self.data / drafting.FOLDER).mkdir(parents=True)

    def pack(self, name):
        (self.data / drafting.FOLDER / name).write_bytes(b"x" * ((1 << 20) + 1))

    def test_the_table(self):
        self.assertEqual((drafting.DRAFTER_MIB["dflash2"], drafting.DRAFTER_MIB["dflash2_selq4"], drafting.DRAFTER_MIB["mtp"]), (890, 715, 1082))
        self.assertEqual((drafting.DRAFTER_GB["dflash2"], drafting.DRAFTER_GB["dflash2_selq4"], drafting.DRAFTER_GB["mtp"], drafting.DRAFTER_GB["off"]), (0.89, 0.72, 1.08, 0.0))
        self.assertEqual(drafting.VRAM_GB, 0.89, "the bf16-selector figure stays the module constant")

    def test_the_figure_follows_the_engine_and_the_pack(self):
        self.pack(drafting.PACK_NAME)
        self.assertEqual(drafting.vram_gb(self.data, self.new), 0.72)
        self.assertEqual(drafting.vram_gb(self.data, self.old), 0.89, "an engine without the text keeps the bf16 selector")
        self.assertEqual(drafting.vram_gb(self.data, None), 0.89, "an engine that is not known: the larger figure")
        self.assertEqual(drafting.vram_gb(None, self.new), 0.72, "no data folder yet: the pack will be the Q4_0 one")

    def test_a_q8_pack_of_an_earlier_strata_is_never_converted_so_the_figure_stays(self):
        self.pack(drafting.LEGACY_PACK_NAME)
        self.assertEqual(drafting.vram_gb(self.data, self.new), 0.89)
        self.pack(drafting.PACK_NAME)                                        # the Q4_0 pack beside it wins (pack_path): the converted figure
        self.assertEqual(drafting.vram_gb(self.data, self.new), 0.72)

    def test_the_plan_says_the_same_number_in_words_and_in_the_fields(self):
        plan = drafting_fetch.plan(self.data, self.new)
        self.assertEqual((plan["vram_gb"], plan["selq4"]), (0.72, True))
        self.assertIn("about 0.7 GB of graphics memory", plan["text"])
        plan = drafting_fetch.plan(self.data, self.old)
        self.assertEqual((plan["vram_gb"], plan["selq4"]), (0.89, False))
        self.assertIn("about 0.9 GB of graphics memory", plan["text"])
        plan = drafting_fetch.plan(self.data)
        self.assertEqual((plan["vram_gb"], plan["selq4"]), (0.89, False))


if __name__ == "__main__":
    unittest.main()
