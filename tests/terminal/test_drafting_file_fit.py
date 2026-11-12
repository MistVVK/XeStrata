"""A model file under a name setup does not write (renamed, a hand-written config, no name) is judged by what the file IS: its GGUF header (architecture, layers, hidden size,
vocabulary size) against what the DFlash2 drafter pack records. A renamed Qwen3.8-Flash-Next file used to start the MTP with no word said (strata_control/drafting.py, apply)."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

ROOT = Path(__file__).resolve().parents[2]
REAL_HEADER = ROOT / "tests" / "terminal" / "data" / "flash_next_iq3s_header.gguf"   # the first 2,710 bytes of the Qwen3.8-Flash-Next IQ3_S file: the keys, no weights
_HARDWARE = mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"]))


def setUpModule():
    _HARDWARE.start()


def tearDownModule():
    _HARDWARE.stop()


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_header(arch="qwen4exp", layers=48, hidden=2560, vocab=248320, tokens=True) -> bytes:
    """A GGUF header with the keys the drafter's test reads; the vocabulary array is announced (its count) and not written."""
    kvs = [(_s("general.architecture"), struct.pack("<I", 8) + _s(arch)),
           (_s("general.name"), struct.pack("<I", 8) + _s("renamed")),
           (_s(f"{arch}.block_count"), struct.pack("<I", 4) + struct.pack("<I", layers)),
           (_s(f"{arch}.context_length"), struct.pack("<I", 4) + struct.pack("<I", 262144)),
           (_s(f"{arch}.embedding_length"), struct.pack("<I", 4) + struct.pack("<I", hidden)),
           (_s(f"{arch}.rope.dimension_sections"), struct.pack("<I", 9) + struct.pack("<IQ", 5, 4) + struct.pack("<4i", 11, 11, 10, 0))]
    if tokens:
        kvs.append((_s("tokenizer.ggml.tokens"), struct.pack("<I", 9) + struct.pack("<IQ", 8, vocab)))
    body = b"".join(k + v for k, v in kvs)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body


def pack_with(**cfg) -> bytes:
    """The start of a DFlash2 pack: "DF2P", version, index length, the text index with its cfg lines."""
    lines = "".join(f"cfg {k} {v}\n" for k, v in cfg.items()) + "end\n"
    index = lines.encode()
    return b"DF2P" + struct.pack("<II", 1, len(index)) + index


class ReadingTheHeader(unittest.TestCase):
    def test_the_real_header_of_the_flash_next_file(self):
        self.assertEqual(drafting.gguf_facts(REAL_HEADER), {"arch": "qwen4exp", "layers": 48, "hidden": 2560, "vocab": 248320})
        self.assertEqual(drafting.file_fit(["--native", str(REAL_HEADER)]), "fits")

    def test_a_synthetic_header_and_what_it_cannot_be(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.gguf"
            p.write_bytes(gguf_header())
            self.assertEqual(drafting.gguf_facts(p), {"arch": "qwen4exp", "layers": 48, "hidden": 2560, "vocab": 248320})
            p.write_bytes(gguf_header(tokens=False))
            self.assertEqual(drafting.gguf_facts(p)["vocab"], None, "no vocabulary key: not a reason to refuse")
            self.assertEqual(drafting.file_fit(["--native", str(p)]), "fits")
            p.write_bytes(b"not a gguf at all, just text" * 10)
            self.assertIsNone(drafting.gguf_facts(p))
            self.assertEqual(drafting.file_fit(["--native", str(p)]), "unknown")
            p.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\xff" * 40)
            self.assertEqual(drafting.file_fit(["--native", str(p)]), "unknown", "a damaged header is no answer, never an error")
            self.assertEqual(drafting.file_fit(["--native", str(Path(tmp) / "missing.gguf")]), "unknown")
        self.assertEqual(drafting.file_fit([]), "unknown")

    def test_the_fits_test_by_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.gguf"
            for kw, want in ((dict(), "fits"), (dict(hidden=4096), "family"), (dict(layers=40), "family"), (dict(layers=46), "fits"), (dict(layers=45), "family"),
                              (dict(vocab=151936), "family"), (dict(arch="qwen35moe"), "other"), (dict(arch="llama"), "other")):
                p.write_bytes(gguf_header(**kw))
                self.assertEqual(drafting.file_fit(["--native", str(p)]), want, kw)

    def test_the_pack_says_what_it_was_made_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self.assertEqual(drafting.pack_target(data), drafting.PACK_TARGET, "no pack here: what the code knows")
            pack = drafting.target_path(data)
            pack.parent.mkdir(parents=True)
            pack.write_bytes(pack_with(hidden=3072, vocab=151936, taps="3,9,15") + b"\0" * 64)
            self.assertEqual(drafting.pack_target(data), {"hidden": 3072, "vocab": 151936, "taps": (3, 9, 15)})
            m = Path(tmp) / "m.gguf"
            m.write_bytes(gguf_header())
            self.assertEqual(drafting.file_fit(["--native", str(m)], data), "family", "the pack on this computer decides, not the table in the code")
            m.write_bytes(gguf_header(hidden=3072, vocab=151936, layers=20))
            self.assertEqual(drafting.file_fit(["--native", str(m)], data), "fits")


class ARenamedFileAtStart(unittest.TestCase):
    HARDWARE = (("cuda", ["sm_120"]), ("hip", ["gfx1151"]), ("metal", None))

    def config(self, tmp, name, header, mode=None):
        data = Path(tmp)
        mtp = data / "mtp" / "rt"
        mtp.mkdir(parents=True)
        (mtp / "experts.bin").write_bytes(b"e")
        (mtp / "draft_vocab.bin").write_bytes(b"v")
        pack = drafting.pack_path(data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        model = data / "my-copy.gguf"
        model.write_bytes(header)
        args = ["--pack", str(data / "packs" / "x"), "--native", str(model), "--mtp", str(mtp), "--spec", "4", "--spec-min-p", "0.5"]
        cfg = {"args": args, **({} if name is None else {"model_name": name}), **({} if mode is None else {"drafting": mode})}
        return cfg, data

    def test_a_renamed_flash_next_file_starts_dflash2_with_nothing_said(self):
        for backend, archs in self.HARDWARE:
            for name in ("my-copy", "Qwen3.8 Flash Next (my copy)", "", None, "qwen3.8-flash-next-unslothx-q8_0"):
                for mode in (None, "auto", "dflash2"):
                    with tempfile.TemporaryDirectory() as tmp:
                        cfg, data = self.config(tmp, name, gguf_header(), mode)
                        decision = drafting.apply(cfg, data_dir=data, backend=backend, archs=archs)
                        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""), (backend, name, mode))
                        self.assertEqual(cfg["args"][cfg["args"].index("--draft") + 1], "dflash2")
                        self.assertNotIn("--mtp", cfg["args"])

    def test_the_names_setup_writes_are_decided_as_before(self):
        """A supported name never reads the file: a name setup wrote is the answer, as in every release before."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data = self.config(tmp, "qwen3.8-flash-next-iq3_s", b"")
            with mock.patch.object(drafting, "file_fit", side_effect=AssertionError("the file is not read")):
                self.assertEqual(drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"]).drafter, "dflash2")

    def test_coder_nvfp4_and_swift_stay_out_whatever_the_file_says(self):
        for name in ("qwen3.8-flash-next-coder-iq1_m", "qwen3.8-flash-next-nvfp4-nvfp4", "swift-1.5-iq2_xs", "My-Coder-copy"):
            for mode in (None, "auto"):
                with tempfile.TemporaryDirectory() as tmp:
                    cfg, data = self.config(tmp, name, gguf_header(), mode)
                    original = list(cfg["args"])
                    decision = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"])
                    self.assertEqual((decision.drafter, decision.note), ("mtp", ""), name)
                    self.assertEqual(cfg["args"], original)

    def test_a_flash_next_family_file_the_drafter_does_not_fit_starts_the_mtp_and_says_so(self):
        for header in (gguf_header(hidden=4096), gguf_header(layers=36), gguf_header(vocab=262144)):
            for name in ("my-copy", None):
                with tempfile.TemporaryDirectory() as tmp:
                    cfg, data = self.config(tmp, name, header, "auto")
                    original = list(cfg["args"])
                    decision = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"])
                    self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NOT_SUPPORTED))
                    self.assertEqual(drafting.summary_line(decision), "Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)")
                    self.assertEqual(cfg["args"], original, "the MTP's own arguments")
                    self.assertNotIn("--draft", cfg["args"])

    def test_another_model_is_not_this_choices_business(self):
        """A Qwen3.6 or Llama file under an unknown name: the MTP as before, no line (the drafter was never an option for it, so nothing is lost)."""
        for arch in ("qwen35moe", "llama"):
            with tempfile.TemporaryDirectory() as tmp:
                cfg, data = self.config(tmp, "my-copy", gguf_header(arch=arch), "auto")
                original = list(cfg["args"])
                decision = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"])
                self.assertEqual((decision.drafter, decision.note), ("mtp", ""), arch)
                self.assertEqual(cfg["args"], original)

    def test_an_unreadable_or_missing_file_is_what_it_was(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data = self.config(tmp, "my-copy", b"junk")
            decision = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"])
            self.assertEqual((decision.drafter, decision.note), ("mtp", ""))
            cfg["args"][cfg["args"].index("--native") + 1] = str(Path(tmp) / "gone.gguf")
            self.assertEqual(drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"]).note, "")

    def test_dflash2_named_on_another_model_says_it_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data = self.config(tmp, "my-copy", gguf_header(arch="llama"), "dflash2")
            decision = drafting.apply(cfg, data_dir=data, backend="cuda", archs=["sm_120"])
            self.assertEqual((decision.drafter, decision.note), ("mtp", drafting.NOT_SUPPORTED))

    def test_the_terminal_that_knows_the_family_does_not_read_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data = self.config(tmp, "my-copy", gguf_header(arch="llama"), "auto")
            with mock.patch.object(drafting, "file_fit", side_effect=AssertionError("the family is the answer")):
                self.assertEqual(drafting.apply(cfg, data_dir=data, family="qwen", backend="cuda", archs=["sm_120"]).drafter, "dflash2")
                self.assertEqual(drafting.apply(dict(cfg), data_dir=data, family="coder", backend="cuda", archs=["sm_120"]).drafter, "mtp")


if __name__ == "__main__":
    unittest.main()
