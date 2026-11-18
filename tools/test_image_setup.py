# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools/image/qimage.py's tables: the image models' files, their RAM fit, and HunyuanImage 3.0's run config.

HunyuanImage 3.0 is the third family the image endpoint takes: no text encoder (the DiT reads the prompt's ids), two
quantizations (w4a8, int8), and a 37-bucket resolution table of its own.  Its buckets are computed from the model's
image processor parameters here, so this file holds them to the shape they must have - 512 x 2048 to 2048 x 512, a
1024 x 1024 with 64 x 64 tokens, the four sizes its processor adds - and the RAM fit to what setup says of a machine
that is short of it.

    python -m unittest tools.test_qimage -v

Nothing is downloaded and no GPU is used: the cases are arithmetic on the tables.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "image"))
import qimage as QI  # noqa: E402


class Tables(unittest.TestCase):
    """The pinned files: every entry's repository at a pinned revision, the HunyuanImage 3.0 files where they belong."""

    def test_every_file_is_pinned(self):
        for name, (repo, path, size, sha) in QI.FILES.items():
            self.assertIn(repo, QI.REVISIONS, name)
            self.assertGreater(size, 0, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertTrue(path, name)

    def test_every_model_names_files_that_exist(self):
        for choice, (name, dit, enc) in QI.IMAGE_MODELS.items():
            self.assertIn(dit, QI.DITS, choice)
            quant = QI.default_quant(dit)
            names = QI.file_names(quant, enc, QI.DEFAULT_TEXT_QUANT, dit)
            for entry in names:
                self.assertIn(entry, QI.FILES, f"{choice}: {entry}")
            # the files, or their Q8_0 share (Mage-Flow's bf16 blocks), and the engine's own
            files_gb = sum(QI.FILES[k][2] for k in names) / 1e9
            self.assertGreater(QI.ram_need_gb(quant, enc, QI.DEFAULT_TEXT_QUANT, dit), files_gb * 0.5, choice)
            self.assertLessEqual(QI.ram_need_gb(quant, enc, QI.DEFAULT_TEXT_QUANT, dit), files_gb + QI.HOST_GB,
                                 choice)

    def test_hunyuanimage3_files(self):
        self.assertEqual(QI.HUNYUANIMAGE3_QUANTS, ("w4a8", "int8"))
        self.assertEqual(QI.default_quant("hunyuanimage3"), "w4a8")
        self.assertEqual(QI.dit_quants("hunyuanimage3"), QI.HUNYUANIMAGE3_QUANTS)
        self.assertEqual(QI.file_names("w4a8", "", "", "hunyuanimage3"),
                         ("hunyuanimage3-w4a8", "hunyuanimage3-vae", "hunyuanimage3-tokenizer",
                          "hunyuanimage3-system-prompt", "hunyuanimage3-vision"))
        for entry in QI.file_names("int8", "", "", "hunyuanimage3"):
            # one folder for the model's four files, named after the model (not after the first file's DiT)
            self.assertEqual(QI.local_path("/models", entry).parent.name, "hunyuan-image-3", entry)
        self.assertEqual(QI.local_path("/models", "hunyuanimage3-tokenizer").name, "tokenizer.json")
        self.assertEqual(QI.system_prompt_path("/models"),
                         Path("/models/hunyuan-image-3/en_unified.txt"))

    def test_license_note_names_the_territory(self):
        note = QI.license_note("hunyuanimage3")
        self.assertIn("Tencent Hunyuan Community License", note)
        self.assertIn("European Union", note)
        self.assertIn("https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE", note)


class RamFit(unittest.TestCase):
    """setup's choice of sizes from the RAM the text model leaves (fit_ram)."""

    def test_qwen_lowers_its_text_encoder_first(self):
        self.assertEqual(QI.fit_ram(24.0)[:2], ("Q8_0", "Q8_0"))
        self.assertEqual(QI.fit_ram(18.6)[:2], ("Q8_0", "Q5_K"))
        self.assertEqual(QI.fit_ram(12.0), ("Q4_K_M", "Q4_K", False))

    def test_hunyuanimage3_lowers_its_own_size(self):
        # 49.4 GB (DiT 46.9 + VAE 2.5) and 84.3 GB, plus the vision model's 0.9 and the engine's own 2 GB
        self.assertEqual(QI.fit_ram(60.0, "", "w4a8", "", "hunyuanimage3"), ("w4a8", "", True))
        self.assertEqual(QI.fit_ram(90.0, "", "int8", "", "hunyuanimage3"), ("int8", "", True))
        self.assertEqual(QI.fit_ram(45.0, "", "int8", "", "hunyuanimage3"), ("w4a8", "", False))
        self.assertEqual(QI.fit_ram(45.0, "", "w4a8", "", "hunyuanimage3"), ("w4a8", "", False))
        # an unknown size (Qwen-Image's Q8_0 asked of this model) is its smallest
        self.assertEqual(QI.fit_ram(60.0, "", "Q8_0", "", "hunyuanimage3"), ("w4a8", "", True))


class Buckets(unittest.TestCase):
    """HunyuanImage 3.0's resolution buckets (hunyuanimage3_buckets), as its own image processor builds them."""

    def setUp(self):
        self.params = QI.hunyuanimage3_resolutions()
        self.buckets = QI.hunyuanimage3_buckets(self.params)

    def test_parameters(self):
        self.assertEqual(self.params, {"base_size": 1024, "step": 64, "align": 16,
                                       "extra_resolutions": [[1024, 768], [1280, 720], [768, 1024], [720, 1280]]})

    def test_37_buckets(self):
        self.assertEqual(len(self.buckets), 37)
        self.assertEqual([b["ratio_index"] for b in self.buckets], list(range(37)))
        tall, square, wide = self.buckets[0], self.buckets[16], self.buckets[32]
        self.assertEqual((tall["height"], tall["width"], tall["tokens_h"], tall["tokens_w"]), (512, 2048, 32, 128))
        self.assertEqual((square["height"], square["width"]), (1024, 1024))
        self.assertEqual((square["tokens_h"], square["tokens_w"]), (64, 64))
        self.assertEqual((wide["height"], wide["width"]), (2048, 512))
        # the four sizes its processor adds, in the order it adds them (a ratio already there is not added twice)
        self.assertEqual([(b["height"], b["width"]) for b in self.buckets[33:]],
                         [(1024, 768), (1280, 720), (768, 1024), (720, 1280)])
        for b in self.buckets:
            self.assertGreaterEqual(b["height"], 512)
            self.assertLessEqual(b["height"], 2048)
            self.assertGreaterEqual(b["width"], 512)
            self.assertLessEqual(b["width"], 2048)
            self.assertEqual(b["height"] % 16, 0)
            self.assertEqual(b["width"] % 16, 0)
            self.assertAlmostEqual(b["ratio"], b["height"] / b["width"])

    def test_computed_buckets_are_sorted(self):
        ratios = [b["ratio"] for b in self.buckets[:33]]
        self.assertEqual(ratios, sorted(ratios))
        # 1920x1080 (0.5625) is one of the added ratios exactly, 720 x 1280
        self.assertAlmostEqual(self.buckets[36]["ratio"], 1080 / 1920)

    def test_a_tie_keeps_the_walks_order(self):
        """The sort is stable, as the model's is: a square bucket comes once, and nothing is dropped."""
        self.assertEqual(len({(b["height"], b["width"]) for b in self.buckets}), 37)


class SystemPrompt(unittest.TestCase):
    """The system prompt out of the model's `system_prompt.py` (parsed, never run)."""

    def test_extracts_and_writes(self):
        from qimage import write_system_prompt
        with tempfile.TemporaryDirectory() as tmp:
            src = QI.local_path(tmp, "hunyuanimage3-system-prompt")
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text('"""not this string"""\n'
                           'other = "no"\n'
                           'unified_system_prompt_en = """You are an advanced\\nmultimodal model."""\n'
                           '__all__ = ["unified_system_prompt_en"]\n', encoding="utf-8")
            dst = write_system_prompt(tmp)
            self.assertEqual(dst, QI.system_prompt_path(tmp))
            self.assertEqual(dst.read_text(encoding="utf-8"), "You are an advanced\nmultimodal model.")

    def test_a_file_without_it_is_refused(self):
        from qimage import write_system_prompt
        with tempfile.TemporaryDirectory() as tmp:
            src = QI.local_path(tmp, "hunyuanimage3-system-prompt")
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text('t2i_system_prompt_en_vanilla = "x"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                write_system_prompt(tmp)
            self.assertFalse(QI.system_prompt_path(tmp).exists())


class Config(unittest.TestCase):
    """The run config's "images" section for HunyuanImage 3.0."""

    def setUp(self):
        self.cfg = QI.image_config("w4a8", "/engine/xestrata-image", "/models", model="hunyuan-image-3-distil")

    def test_the_section(self):
        self.assertEqual(self.cfg["model_name"], "hunyuan-image-3-distil")
        self.assertEqual(self.cfg["family"], "hunyuan-image-3")
        self.assertEqual(self.cfg["quant"], "w4a8")
        self.assertEqual(self.cfg["steps"], 8)
        self.assertEqual(self.cfg["guidance"], 2.5)
        self.assertEqual(self.cfg["diffusion_model"],
                         "/models/hunyuan-image-3/hunyuan_image_3_instruct_distil_w4a8.safetensors")
        self.assertEqual(self.cfg["vae"], "/models/hunyuan-image-3/hunyuan_image_3_vae_fp16.safetensors")
        self.assertEqual(self.cfg["tokenizer"], "/models/hunyuan-image-3/tokenizer.json")
        self.assertEqual(self.cfg["system_prompt"], "/models/hunyuan-image-3/en_unified.txt")
        self.assertEqual(self.cfg["vision"], "/models/hunyuan-image-3/model-0032-of-0032.safetensors")
        self.assertEqual(self.cfg["max_refs"], 3)
        self.assertEqual(self.cfg["resolutions"], QI.hunyuanimage3_resolutions())

    def test_no_text_encoder_of_its_own(self):
        for key in ("llm", "text_encoder", "text_quant", "edit_model"):
            self.assertNotIn(key, self.cfg, key)
        self.assertNotIn("mask", self.cfg, "every model takes a mask: setup writes no mask flag")
        self.assertFalse(self.cfg["transparency"])
        self.assertNotIn("lora_dir", self.cfg, "HunyuanImage 3.0 applies no adapter")

    def test_the_models_that_take_adapters_name_their_folder(self):
        """serve/loras.py reads a model's LoRA files from "lora_dir": one folder for every model that applies them."""
        for model in ("qwen-image-turbo", "mage-flow-turbo", "moss-image-sft"):
            cfg = QI.image_config("Q8_0", "/engine/xestrata-image", "/models", model=model)
            self.assertEqual(cfg["lora_dir"], "/models/loras", model)


class TrainingBases(unittest.TestCase):
    def test_opt_in_downloads_bases_without_extra_edit_models(self):
        import setup
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eng = root / "engine"
            eng.mkdir()
            (eng / setup.IMAGE_EXE).touch()
            a = argparse.Namespace(images="qwen-image-turbo,mage-flow-turbo,moss-image-sft-4-step,anima-turbo",
                                   image_lora_training=False, yes=True, image_quant=None, image_gpu_policy="auto")

            def fetch(url, path, what):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with mock.patch.object(setup, "remove_old_image_engine"), mock.patch.object(setup, "download", fetch), \
                    mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "ask_sam3_license"), \
                    mock.patch.object(setup, "free_gb", return_value=1000):
                default, more = setup.setup_images(a, {}, root / "inference", eng)
                self.assertEqual(len(more), 3)
                self.assertFalse(QI.local_path(root / "inference", "base-F16").exists())
                a.image_lora_training = True
                default, more = setup.setup_images(a, {}, root / "training", eng)
                from serve.images import ImageModels
                models = ImageModels([default, *more])
                self.assertFalse(models.missing())
                cfgs = {c["model_name"]: c for c in [default, *more]}
                self.assertTrue(cfgs["qwen-image-2.1"]["training_base_only"])
                self.assertNotIn("training_base_only", cfgs["qwen-image-2.1-turbo"])
                self.assertIn("qwen-image-2.1", cfgs)
                self.assertIn("mage-flow", cfgs)
                self.assertIn("moss-image-sft", cfgs)
                self.assertIn("anima-base", cfgs)
                for key in ("base-F16", "mf-rl-bf16", "anima-base-bf16"):
                    self.assertTrue(QI.local_path(root / "training", key).is_file(), key)
                self.assertFalse(QI.local_path(root / "training", "mf-edit-rl-bf16").exists())
                self.assertNotIn("edit_model", cfgs["mage-flow"])
                self.assertNotIn("few_step", cfgs["moss-image-sft"])
                self.assertEqual(cfgs["qwen-image-2.1"]["diffusion_model"],
                                 str(QI.local_path(root / "training", "base-F16")))


if __name__ == "__main__":
    unittest.main()
