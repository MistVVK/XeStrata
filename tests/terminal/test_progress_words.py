"""0.2.2: the words for a model download and its preparation are plain and true (strata_control/progress_words.py), and the app's
and the Terminal's progress read the same lines the same way (tools/strata_mcp.py install_progress / combined_progress).

The excerpts are from real setup logs (Gyro-S on the RTX 5070 PC, 2026-10-19) and from the steps a Q2_0 install adds."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strata_mcp  # noqa: E402
from strata_control import progress_words as W  # noqa: E402

HEAD = ["Strata - Qwen3.8-Flash-Next on a normal PC", "=== Step 1: checking your PC ===", "  [ok] GPU: NVIDIA GeForce RTX 5070",
        "=== Step 2: your choices ===", "  [ok] size: Gyro-S", "=== Step 3: Python packages ===", "  [ok] numpy ...",
        "=== Step 4: the Strata engine ==="]
STEP5 = ["=== Step 5: downloading Qwen3.8-Flash-Next (Gyro) Gyro-S ===",
         "  The model files go in C:\\Users\\x\\Documents\\Strata\\models\\gyro-Gyro-S"]
MODEL_PROGRESS = "  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  30.53 / 58.49 GB (52%)   "
CHECK = "  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf (SHA-256, 58.5 GB) ..."
STEP6 = ["=== Step 6: preparing the model for Strata ==="]


def prog(lines):
    p = strata_mcp.install_progress(lines)
    return p, p.get("label"), p.get("phase")


class StepWords(unittest.TestCase):
    def test_checking_the_pc(self):
        self.assertEqual(prog(HEAD[:3])[1:], ("Checking your computer", "check"))

    def test_the_engine_and_the_helper_download_have_their_own_words(self):
        self.assertEqual(prog(HEAD + ["  llama.cpp source:   0.01 / 0.04 GB (21%)   "])[2], "helpers")
        self.assertIn("small tools", prog(HEAD + ["  llama.cpp source:   0.01 / 0.04 GB (21%)   "])[1])
        self.assertEqual(prog(HEAD + ["  Strata engine:   0.20 / 0.90 GB (22%)   "])[1], "Downloading the engine for this PC")
        self.assertEqual(prog(HEAD)[2], "engine")

    def test_the_model_download(self):
        _p, label, phase = prog(HEAD + STEP5 + [MODEL_PROGRESS])
        self.assertEqual((label, phase), ("Downloading the model files", "download"))

    def test_a_figure_from_an_earlier_step_does_not_leak_into_the_next(self):
        """Step 4's 'llama.cpp source' line must not make step 5 'Downloading ...' before any model line is printed."""
        p, label, phase = prog(HEAD + ["  llama.cpp source:   0.04 / 0.04 GB (100%)   "] + STEP5)
        self.assertNotIn("download", p)
        self.assertEqual(phase, "download")
        self.assertEqual(label, "Getting the model files")

    def test_the_hash_check_is_checking_not_downloading(self):
        lines = HEAD + STEP5 + [MODEL_PROGRESS, "  [ok] downloaded", CHECK]
        p, label, phase = prog(lines)
        self.assertEqual(phase, "verify")
        self.assertIn("Checking that the model files are complete", label)
        self.assertNotIn("Downloading", label)
        # with its own figure
        p, label, phase = prog(lines + ["  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "])
        self.assertEqual(phase, "verify")
        self.assertEqual(p["download"]["percent"], 20)

    def test_a_damaged_file_is_repaired_in_plain_words_and_the_second_check_is_checking(self):
        damaged = "  Some parts of Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf were damaged; downloading only those parts again."
        lines = HEAD + STEP5 + [MODEL_PROGRESS, CHECK, damaged, "  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "]
        _p, label, phase = prog(lines)
        self.assertEqual(phase, "repair")
        self.assertEqual(label, "Some parts of a model file were damaged; downloading only those parts again")
        done = lines + ["  2 of 1186 parts were different and have been written again; checking the whole file again ..."]
        self.assertEqual(prog(done)[2], "download")

    def test_the_picture_reader(self):
        lines = HEAD + STEP5 + ["  [ok] model files present", "  vision encoder:   0.30 / 0.91 GB (33%)   "]
        self.assertEqual(prog(lines)[1:], ("Downloading the picture reader", "pictures"))

    def test_files_already_there(self):
        lines = HEAD + STEP5 + ["  [ok] Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf already downloaded"]
        self.assertEqual(prog(lines)[2], "verify")

    def test_step_six_packing_the_experts(self):
        lines = HEAD + STEP5 + STEP6 + ["  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...",
                                        "  > python tools/strata_pack.py build --gguf x --out y --skip-hash --force"]
        self.assertEqual(prog(lines)[1:], ("Packing the model's experts for your processor (one time, a few minutes)", "pack"))
        self.assertEqual(prog(lines + ["  > python tools/pack_index.py --pack y"])[1], "Indexing the packed experts")
        self.assertEqual(prog(lines + ["  > python tools/strata_tokenizer.py --gguf x --out y"])[1], "Preparing the tokenizer")

    def test_step_six_the_draft_layer_download_then_its_preparation(self):
        fetch = ["  [ok] model prepared: y", "  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:",
                 "  only its ~5 GB of MTP tensors are downloaded.", "  > python tools/mtp_fetch.py fetch --out z",
                 "  MTP draft layer:   1.65 / 5.21 GB (31%)"]
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch)
        self.assertEqual(phase, "draft_download")
        self.assertIn("Downloading the draft layer", label)
        # the download is over: the numbers it printed are stale, the next tool is running
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch + ["  MTP draft layer:   5.21 / 5.21 GB (100%)", "  > python tools/mtp_pack.py --src z"])
        self.assertEqual((label, phase), ("Preparing the draft layer", "draft"))
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch + ["  > python tools/mtp_rt.py --gguf g --out rt"])
        self.assertEqual(phase, "draft")

    def test_step_six_without_a_draft_layer_step_is_not_the_draft_layer(self):
        # 0.2.14 (l620 offline / Off / Auto, Boston): every launch showed "Preparing the draft layer" after "[ok] model prepared", with no draft layer touched
        for after in ([], ["  [ok] the draft layer (MTP) is off: its ~5 GB are not downloaded (`--mtp on` fetches it later)"],
                      ["  [ok] DFlash2 is the drafter: the MTP draft layer's ~5 GB are not downloaded (choosing the MTP later fetches it once)"]):
            with self.subTest(after=after):
                p, label, phase = prog(HEAD + STEP5 + STEP6 + ["  [ok] model prepared: y", *after])
                self.assertEqual((label, phase), ("The model files are on this computer; checking them", "verify"))

    def test_step_six_claims_preparation_only_while_it_runs(self):
        # 0.2.14.1 check (M5 Max): "Preparing your model for Strata (one time)" on every launch, also a 4-5 s restart with the model prepared
        self.assertEqual(prog(HEAD + STEP5 + STEP6)[1:], ("The model files are on this computer; checking them", "verify"))
        self.assertEqual(prog(HEAD + STEP5 + STEP6 + ["  > python tools/iq_pack.py --gguf x --out y"])[1:],
                         ("Preparing the model data for the engine (one time)", "pack"))
        self.assertEqual(prog(HEAD + STEP5 + STEP6 + ["  > python tools/strata_pack.py --gguf x --out y"])[2], "pack")

    def test_step_six_with_everything_already_there(self):
        lines = HEAD + STEP5 + STEP6 + ["  [ok] model prepared: y", "  [ok] MTP draft layer: z"]
        self.assertEqual(prog(lines)[2], "draft")

    def test_low_memory_mode_packing(self):
        lines = HEAD + STEP5 + STEP6 + ["  Writing the experts into one file for the low-memory mode (one time, 80 GB) ...",
                                        "  > python tools/iq_pack.py --gguf x --out y --experts-bin"]
        self.assertEqual(prog(lines)[1], "Packing the experts into one file for low-memory mode (one time)")

    def test_the_last_step(self):
        lines = HEAD + STEP5 + STEP6 + ["=== Step 7: writing the start script ===", "  [ok] KV streaming on"]
        self.assertEqual(prog(lines)[1:], ("Saving the model's settings", "settings"))

    def test_the_heading_is_found_even_after_hundreds_of_tool_lines(self):
        lines = HEAD + STEP5 + STEP6 + ["  > python tools/iq_pack.py --gguf x --out y"] + [f"  layer {i}/40" for i in range(900)]
        self.assertEqual(prog(lines)[2], "pack")

    def test_no_step_yet(self):
        self.assertEqual(prog(["Strata - Qwen3.8-Flash-Next"])[1:], ("Getting your model ready", "working"))

    def test_with_numbers(self):
        w = W.describe(5, STEP5, {"file": "x.gguf", "done_gb": 12.34, "total_gb": 58.49, "percent": 21})
        self.assertEqual(W.with_numbers(w, {"done_gb": 12.34, "total_gb": 58.49}), "Downloading the model files: 12.3 of 58.5 GB")
        self.assertEqual(W.with_numbers(w, None), "Downloading the model files")


class CombinedFigure(unittest.TestCase):
    def test_a_part_file_the_pieces_filled_ahead_is_not_a_finished_download(self):
        """The parallel download writes the last pieces early, so the .part is full length while the pieces done are 50%: the log's
        figure is the truth (the bar used to jump to 99%)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            part = folder / "m-00001-of-00002.gguf.part"
            part.write_bytes(b"\0" * 1000)
            files = [{"path": str(folder / "m-00001-of-00002.gguf"), "bytes": 1_000_000_000, "role": "model"}]
            prog_ = {"step": "5/7: downloading X", "download": {"file": "m-00001-of-00002.gguf", "done_gb": 0.5, "total_gb": 1.0, "percent": 50}}
            # the .part says 1000 bytes here; make it say 1 GB without writing it
            import os
            with open(part, "r+b") as f:
                f.truncate(1_000_000_000)
            out = strata_mcp.files_progress(prog_, {"total_gb": 1.0}, files)
            self.assertAlmostEqual(out["download"]["done_gb"], 0.5, places=2)
            self.assertEqual(out["download"]["percent"], 50)

    def test_the_hash_check_has_its_own_figure(self):
        lines = HEAD + STEP5 + [CHECK, "  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "]
        p = strata_mcp.install_progress(lines)
        out = strata_mcp.combined_progress(p, {"folder": None, "total_gb": 58.5})
        self.assertEqual(out["download"]["file"], "checking files")
        self.assertEqual(out["download"]["percent"], 20)

    @staticmethod
    def _qwen36_files(folder: Path, reader_done: bool) -> list:
        """A Qwen3.6 UD-Q4_K_XL install's files (sizes made sparse, nothing written): the model whole, the picture reader whole or as a .part
        the parallel pieces have already pushed to its full length."""
        model = folder / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        reader = folder / "mmproj-F16.gguf"
        for path, size in ((model, 22_853_663_008), (reader if reader_done else reader.with_name(reader.name + ".part"), 899_283_584)):
            with open(path, "wb") as f:
                f.truncate(size)
        return [{"path": str(model), "bytes": 22_853_663_008, "role": "model"},
                {"path": str(reader), "bytes": 899_283_584, "role": "vision"}]

    def test_the_picture_reader_counts_by_its_own_figure(self):
        """DL99: setup names the picture reader "vision encoder" in its progress line, not by its file name, so its .part (filled ahead by
        the parallel pieces) was counted whole: the bar said 99% while 16-48% of it had arrived (Boston, 2026-11-15)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = self._qwen36_files(Path(tmp), reader_done=False)
            prog_ = {"step": "5/7: downloading Qwen3.6-35B-A3B UD-Q4_K_XL",
                     "download": {"file": "vision encoder", "done_gb": 0.43, "total_gb": 0.9, "percent": 48}}
            out = strata_mcp.files_progress(prog_, {"total_gb": 22.9}, files)
            reader = next(p for p in out["parts"] if p["kind"] == "vision")
            self.assertAlmostEqual(reader["done_gb"], 0.43, places=2)
            self.assertEqual(reader["state"], "downloading")
            self.assertLess(out["download"]["percent"], 99)

    def test_after_the_download_every_file_stays_counted_and_the_figure_is_whole(self):
        """DL99: from step 6 on the parts were dropped, so the app counted the engine's part alone ("99% · 0.7 / 0.7 GB"). Every file is in:
        the parts stay, all done, and the figure is 100."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = self._qwen36_files(Path(tmp), reader_done=True)
            job = {"folder": tmp, "total_gb": 22.9, "files": files}
            for lines in (HEAD + STEP5 + STEP6, HEAD + STEP5 + STEP6 + ["=== Step 7: writing the start script ==="]):
                out = strata_mcp.combined_progress(strata_mcp.install_progress(lines), job)
                self.assertEqual([(p["kind"], p["state"]) for p in out.get("parts", [])], [("model", "done"), ("vision", "done")])
                self.assertEqual(out["download"]["percent"], 100)
                self.assertAlmostEqual(out["download"]["total_gb"], 23.753, places=2)
            # while a file is still missing the figure stays below 100
            Path(files[1]["path"]).rename(files[1]["path"] + ".part")
            out = strata_mcp.combined_progress(strata_mcp.install_progress(HEAD + STEP5 + [MODEL_PROGRESS]), job)
            self.assertLessEqual(out["download"]["percent"], 99)
            # before the download step nothing is counted
            self.assertNotIn("parts", strata_mcp.combined_progress(strata_mcp.install_progress(HEAD), job))


if __name__ == "__main__":
    unittest.main()
