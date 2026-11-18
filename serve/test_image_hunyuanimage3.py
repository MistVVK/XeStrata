# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/images.py's HunyuanImage 3.0 branch: the sizes that snap to the model's resolution buckets, the sequence its
DiT reads (built here: it has no text encoder), the fields a request gives it, and the line a job sends the engine.

serve/test_images.py is upstream's, for the sd-cli XeStrata no longer starts; this is not that file.

    python -m unittest serve.test_image_hunyuanimage3
    XESTRATA_HUNYUANIMAGE3_TOKENIZER=<the folder with the model's tokenizer.json> python -m unittest \
serve.test_image_hunyuanimage3
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from serve import images

RESOLUTIONS = {"base_size": 1024, "step": 64, "align": 16,
               "extra_resolutions": [[1024, 768], [1280, 720], [768, 1024], [720, 1280]]}
REAL = os.environ.get("XESTRATA_HUNYUANIMAGE3_TOKENIZER")


def cfg(**kw) -> dict:
    """A run config's "images" section as setup --images writes HunyuanImage 3.0's (tools/image/qimage.py)."""
    out = {"model_name": "hunyuan-image-3-distil", "family": "hunyuan-image-3", "exe": "xestrata-image",
           "quant": "w4a8", "diffusion_model": "dit.safetensors", "vae": "vae.safetensors",
           "tokenizer": "tokenizer.json", "system_prompt": "en_unified.txt", "vision": "shard.safetensors",
           "max_refs": 3, "steps": 8, "guidance": 2.5,
           "schedule": "hunyuanimage3", "transparency": False, "resolutions": RESOLUTIONS,
           "gpu_policy": "auto", "uma": False, "ssd_offload": False}
    out.update(kw)
    return out


class FakeTokenizer:
    """What hunyuanimage3_ids asks of a tokenizer, and nothing else: the markers' ids by name, each text section
    encoded on its own (recorded, so the test can say which sections it asked for), one id a character."""

    NAMES = ("<|startoftext|>", "<answer>", "<boi>", "<img_size_1024>", "<img_size_256>", "<img_ratio_16>",
             "<img_ratio_36>", "<timestep>", "<guidance>", "<timestep_r>", "<img>", "<eoi>", "</answer>",
             "<joint_img_sep>")

    def __init__(self):
        self.special_ids = {name: 900 + i for i, name in enumerate(self.NAMES)}
        self.seen: list[str] = []

    def encode(self, text, parse_special=False):
        self.seen.append(text)
        return [len(text)]


class FakeEngine:
    """The engine's side of a job: the lines it is sent, and zeros of the size asked for as the image."""

    def __init__(self):
        self.lines: list[str] = []

    def alive(self):
        return True

    def ask(self, line, want, timeout, cancel=None, step=None, stage=None):
        self.lines.append(line)
        width, height, out = int(line.split()[1]), int(line.split()[2]), line.split()[5]
        Path(out).write_bytes(bytes(width * height * 4))
        return {"text_s": 0.5, "sampling_s": 1.0, "decode_s": 0.25}

    def close(self, shutdown=False):
        pass


class Buckets(unittest.TestCase):
    """A request's shape goes to the model's own table: the bucket nearest its height over width."""

    def test_the_table_is_the_models(self):
        buckets = images._qimage().hunyuanimage3_buckets(RESOLUTIONS)
        self.assertEqual(len(buckets), 37)
        self.assertEqual([b["ratio_index"] for b in buckets], list(range(37)))
        self.assertTrue(all(b["height"] % 16 == 0 and b["width"] % 16 == 0 for b in buckets))
        self.assertEqual((buckets[16]["height"], buckets[16]["width"], buckets[16]["tokens_h"]), (1024, 1024, 64))

    def test_the_nearest_shape(self):
        for (width, height), want in {(1024, 1024): (1024, 1024, 16), (1024, 768): (1024, 768, 35),
                                      (1920, 1080): (1280, 720, 36), (720, 1280): (720, 1280, 34),
                                      (2048, 512): (2048, 512, 0), (512, 2048): (512, 2048, 32)}.items():
            b = images.hunyuanimage3_bucket(cfg(), width, height)
            self.assertEqual((b["width"], b["height"], b["ratio_index"]), (want[0], want[1], want[2]),
                             f"{width}x{height}")

    def test_two_equally_near_ratios_take_the_earlier(self):
        """1024 / 1088 is halfway between 960 / 1088 (bucket 15) and 1 (bucket 16)."""
        self.assertEqual(images.hunyuanimage3_bucket(cfg(), 1088, 1024)["ratio_index"], 15)

    def test_the_table_comes_from_the_config(self):
        """The parameters are the config's: a smaller base is a smaller table, and no "resolutions" is the model's."""
        small = images.hunyuanimage3_bucket(cfg(resolutions={"base_size": 512, "step": 32, "align": 16}), 512, 512)
        self.assertEqual((small["height"], small["width"]), (512, 512))
        self.assertEqual(images.hunyuanimage3_bucket({}, 1024, 1024)["ratio_index"], 16)


class Requests(unittest.TestCase):
    """parse_request's fields for the model: its steps, its one guidance number, its table's sizes."""

    def test_steps_and_guidance_are_its_own(self):
        job = images.parse_request({"prompt": "a cube"}, cfg())
        self.assertEqual((job["steps"], job["guidance"]), (8, 2.5))
        self.assertNotIn("cfg", job, "its guidance is a token, not a second pass")
        job = images.parse_request({"prompt": "a cube", "steps": 20, "guidance_scale": 4.0}, cfg())
        self.assertEqual((job["steps"], job["guidance"]), (20, 4.0))
        job = images.parse_request({"prompt": "a cube", "num_inference_steps": 3}, cfg())
        self.assertEqual(job["steps"], 3)

    def test_true_cfg_scale_is_the_same_number_by_its_other_name(self):
        job = images.parse_request({"prompt": "a cube", "true_cfg_scale": 3.5, "negative_prompt": "blurry"}, cfg())
        self.assertEqual(job["guidance"], 3.5)
        self.assertNotIn("cfg", job)
        self.assertNotIn("negative_prompt", job, "there is no second pass to steer")

    def test_the_size_is_the_bucket(self):
        job = images.parse_request({"prompt": "a cube", "size": "1920x1080"}, cfg())
        self.assertEqual((job["width"], job["height"], job["bucket"]["ratio_index"]), (1280, 720, 36))
        self.assertEqual(images.made_size(job), (1280, 720), "the bucket's sides are multiples of 16 already")
        job["ids"] = list(range(10)) + [0] * (80 * 45)
        usage = images.usage(job)
        self.assertEqual(usage["output_tokens"], 80 * 45)
        self.assertEqual(usage["input_tokens"], 10, "the block is the output's, not the prompt's")

    def test_a_transparent_image_is_refused_and_an_edit_takes_the_vision_file(self):
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cube", "background": "transparent"}, cfg())
        png = b"\x89PNG\r\n\x1a\n" + bytes(8) + (1920).to_bytes(4, "big") + (1080).to_bytes(4, "big") + bytes(40)
        job = images.parse_edit_request({"prompt": "a cube"}, [png], cfg(vision=None))
        self.assertNotIn("refs", job, "without the vision file the image is no edit's reference")
        self.assertEqual((job["init"], job["strength"]), (png, images.STRENGTH_IMG2IMG),
                         "it is the image img2img starts from, at the default strength")
        job = images.parse_edit_request({"prompt": "a cube"}, [png], cfg())
        self.assertEqual((job["width"], job["height"]), (1280, 720),
                         "size auto: the first reference's shape, at its bucket")
        with self.assertRaises(images.ImageRequestError):
            images.parse_edit_request({"prompt": "a cube"}, [png] * 4, cfg())

    def test_the_other_two_models_are_as_they_were(self):
        qwen = {"exe": "xestrata-image", "diffusion_model": "dit.gguf", "vae": "vae.safetensors", "llm": "text.gguf",
                "schedule": "turbo"}
        job = images.parse_request({"prompt": "a cube", "size": "1024x1024"}, qwen)
        self.assertEqual((job["steps"], job["bucket"]), (images.TURBO_STEPS, None))
        job = images.parse_request({"prompt": "a cube", "true_cfg_scale": 4.0, "negative_prompt": "blurry"}, qwen)
        self.assertEqual(job["cfg"], 4.0)
        mage = {**qwen, "family": "mage-flow", "schedule": "mage", "steps": 4, "cfg_scale": 1.0}
        job = images.parse_request({"prompt": "a cube", "size": "1024x1024"}, mage)
        self.assertEqual((job["steps"], job["bucket"]), (4, None))
        self.assertNotIn("guidance", job)


class Extend(unittest.TestCase):
    def test_fit_canvas_and_scale_picture_and_sides(self):
        for size, spec, ratio in (((832, 1216), {"aspect": "16:9"}, 16 / 9),
                                  ((1216, 832), {"aspect": "9:16"}, 9 / 16),
                                  ((1024, 1024), {"aspect": "21:9"}, 21 / 9)):
            with self.subTest(size=size, spec=spec):
                shrunk, pads, scale = images.parse_extend(spec, *size, 2048)
                canvas = (shrunk[0] + pads[0] + pads[2], shrunk[1] + pads[1] + pads[3])
                self.assertLess(scale, 1)
                self.assertLessEqual(max(canvas), 2048)
                self.assertAlmostEqual(canvas[0] / canvas[1], ratio, delta=0.005)
        size, pads, scale = images.parse_extend({"left": 1000, "bottom": 900}, 1600, 1200, 2048)
        self.assertLess(scale, 1)
        self.assertEqual(pads[0], int(1000 * scale))
        self.assertEqual(pads[3], int(900 * scale))
        self.assertLessEqual(max(size[0] + pads[0], size[1] + pads[3]), 2048)

    def test_edit_uses_shrunk_picture_and_reports_scale(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (832, 1216), "red").save(buf, format="PNG")
        job = images.parse_edit_request({"prompt": "a cube", "extend": {"aspect": "16:9"}},
                                        [buf.getvalue()], cfg(vision=None))
        self.assertLess(job["extend_scale"], 1)
        self.assertLessEqual(max(images.image_dims(job["init"])), 2048)
        self.assertEqual(images.image_dims(job["init"]), images.image_dims(job["mask"]))
        job = images.parse_edit_request({"prompt": "a cube", "extend": {"aspect": "16:9"}},
                                        [buf.getvalue()], cfg())
        self.assertLess(job["extend_scale"], 1)
        self.assertEqual(images.image_dims(job["init"]), images.image_dims(job["mask"]))
        rw, rh = images.image_dims(job["refs"][0])
        cw, ch = images.image_dims(job["init"])
        self.assertLess(rw, cw)
        self.assertEqual(rh, ch)
        with Image.open(io.BytesIO(job["refs"][0])) as ref:
            self.assertEqual(ref.getpixel((0, 0)), (255, 0, 0))
        job = images.parse_edit_request({"prompt": "a cube", "extend": {"aspect": "4:5"}},
                                        [buf.getvalue()], cfg())
        self.assertNotIn("extend_scale", job)
        self.assertEqual(job["refs"][0], buf.getvalue())
        self.assertEqual(images.image_dims(job["init"]), images.image_dims(job["mask"]))
        self.assertGreater(images.image_dims(job["init"])[0], 832)


class AuxiliaryMemory(unittest.TestCase):
    def test_auxiliary_budget_refusal_suggests_a_smaller_model(self):
        with self.assertRaises(images.ImageRequestError) as caught:
            images.check_room({"least_mib": 8300, "cap_mib": 8192}, None, {"aux": {"kind": "birefnet"}})
        self.assertIn("smaller auxiliary model", str(caught.exception))
        self.assertNotIn("reference images", str(caught.exception))

    def test_allocation_refusal_is_a_memory_shortage(self):
        from serve import image_regions

        class Gen:
            def aux_ask(self, line, want):
                raise images.ImageJobFailed("image engine (BiRefNet): GPU memory: alloc_device: 123 bytes refused; "
                                            "device free 54 MiB")
        with self.assertRaises(image_regions.MemoryShort) as caught:
            image_regions._ask(Gen(), "MATTE", "MATTE")
        self.assertNotIn("alloc_device", str(caught.exception))


class Sequence(unittest.TestCase):
    """The sequence the DiT reads, over a tokenizer that only answers what the builder asks."""

    BUCKET = {"ratio_index": 16, "height": 256, "width": 256, "tokens_h": 16, "tokens_w": 16, "ratio": 1.0}

    def test_without_the_system_turn(self):
        tok = FakeTokenizer()
        ids = images.hunyuanimage3_ids(tok, "ab", self.BUCKET, 256, None)
        self.assertEqual(tok.seen, ["User: ", "ab", "\n\n", "Assistant: ", "\n\n"],
                         "each section on its own: as one string the pieces merge across the boundaries")
        head = [tok.special_ids[n] for n in ("<|startoftext|>",)]
        self.assertEqual(ids[:5], head + [6, 2, 2, 11], "\"User: \", the prompt, the blank line, \"Assistant: \"")
        markers = [tok.special_ids[n] for n in ("<answer>", "<boi>", "<img_size_256>", "<img_ratio_16>",
                                                "<timestep>", "<guidance>", "<timestep_r>")]
        self.assertEqual(ids[5:12], markers)
        n = 16 * 16
        self.assertEqual(ids[12:12 + n], [tok.special_ids["<img>"]] * n)
        self.assertEqual(ids[12 + n:], [tok.special_ids["<eoi>"], tok.special_ids["</answer>"], 2])

    def test_with_the_system_turn_and_the_models_own_size(self):
        tok = FakeTokenizer()
        ids = images.hunyuanimage3_ids(tok, "ab", self.BUCKET, 1024, "sys")
        self.assertEqual(tok.seen[:2], ["sys", "\n\n"])
        self.assertEqual(ids[:2], [tok.special_ids["<|startoftext|>"], 3])
        self.assertIn(tok.special_ids["<img_size_1024>"], ids)
        self.assertNotIn(tok.special_ids["<img_size_256>"], ids)

    def test_a_reference_image_stands_before_the_prompt(self):
        """An edit's reference: its block after "User: " and before the prompt, its VAE grid then 1024 slots."""
        tok = FakeTokenizer()
        ref = {"ratio_index": 36, "height": 720, "width": 1280, "tokens_h": 45, "tokens_w": 80, "ratio": 720 / 1280}
        ids = images.hunyuanimage3_ids(tok, "ab", self.BUCKET, 1024, None, [ref])
        s = tok.special_ids
        self.assertEqual(ids[:2], [s["<|startoftext|>"], 6])
        self.assertEqual(ids[2:6], [s["<boi>"], s["<img_size_1024>"], s["<img_ratio_36>"], s["<timestep>"]])
        self.assertEqual(ids[6:6 + 45 * 80], [s["<img>"]] * (45 * 80))
        at = 6 + 45 * 80
        self.assertEqual(ids[at], s["<joint_img_sep>"])
        self.assertEqual(ids[at + 1:at + 1 + 1024], [s["<img>"]] * 1024)
        self.assertEqual(ids[at + 1025], s["<eoi>"])
        self.assertEqual(ids[at + 1026:at + 1029], [2, 2, 11], "the prompt, the blank line, \"Assistant: \"")
        self.assertEqual(ids.count(s["<boi>"]), 2)

    def test_the_engine_keeps_through_eoi(self):
        """hunyuanimage3_trim keeps the last <img> and the <eoi> after it: what follows it is dropped."""
        ids = images.hunyuanimage3_ids(FakeTokenizer(), "ab", self.BUCKET, 256, None)
        last_img = len(ids) - 1 - ids[::-1].index(FakeTokenizer().special_ids["<img>"])
        self.assertEqual(ids[last_img + 1], FakeTokenizer().special_ids["<eoi>"])

    def test_a_vocabulary_without_the_markers_is_refused(self):
        class Bare:
            special_ids: dict = {}

            def encode(self, text, parse_special=False):
                return []

        with self.assertRaises(ValueError):
            images.hunyuanimage3_ids(Bare(), "ab", self.BUCKET, 256, None)


class Job(unittest.TestCase):
    """ImageGen over the config (no engine started): its keys, its command and the line a job sends."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        for name in ("xestrata-image", "dit.safetensors", "vae.safetensors", "tokenizer.json", "en_unified.txt",
                     "shard.safetensors"):
            (self.d / name).write_text("x", encoding="utf-8")
        self.cfg = cfg(exe=str(self.d / "xestrata-image"), diffusion_model=str(self.d / "dit.safetensors"),
                       vae=str(self.d / "vae.safetensors"), tokenizer=str(self.d / "tokenizer.json"),
                       system_prompt=str(self.d / "en_unified.txt"), vision=str(self.d / "shard.safetensors"))
        self.gen = images.ImageGen(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.gen.dir, ignore_errors=True)
        self.tmp.cleanup()

    def test_the_keys_are_the_models(self):
        self.assertEqual((self.gen.family, self.gen.drop, self.gen.model),
                         ("hunyuan-image-3", 0, "hunyuan-image-3-distil"))
        self.assertEqual(self.gen.missing(), [])
        (self.d / "en_unified.txt").unlink()
        self.assertEqual(self.gen.missing(), [str(self.d / "en_unified.txt")])
        with self.assertRaises(ValueError):
            images.ImageGen(cfg(exe=str(self.d / "xestrata-image"), tokenizer=""))   # the tokenizer is required

    def test_the_engine_takes_no_text_encoder(self):
        self.assertNotIn("--text", self.gen.engine.command())
        qwen = images.ImageGen({**self.cfg, "family": None, "llm": str(self.d / "dit.safetensors")})
        try:
            self.assertIn("--text", qwen.engine.command())
        finally:
            shutil.rmtree(qwen.dir, ignore_errors=True)

    def test_what_the_job_sends_and_reports(self):
        self.gen.tok, self.gen.engine = FakeTokenizer(), FakeEngine()
        job = images.parse_request({"prompt": "ab", "size": "1024x1024", "seed": 7}, self.cfg)
        made, figures = self.gen.run(job)
        self.assertEqual(len(made), 1)
        line = self.gen.engine.lines[0]
        self.assertTrue(line.startswith("IMG 1024 1024 7 0 "), line)
        self.assertIn(" STEPS:8 GUID:2.5", line)
        self.assertNotIn("SIGMAS:", line, "the engine builds the model's own sigmas from the count")
        self.assertNotIn("CFG:", line, "one pass")
        # the system prompt (its text and a blank line), the four conversation sections and the tail's blank line,
        # each one id through the stand-in, bos, the seven markers, the block, <eoi> and </answer>
        self.assertEqual(len(job["ids"]), 1 + 6 + 1 + 7 + 64 * 64 + 2)
        self.assertEqual(len(job["ids"]) - job["ids"].count(FakeTokenizer().special_ids["<img>"]), 17)
        self.assertEqual(figures["steps"], 8)
        self.assertEqual(figures["guidance"], 2.5)
        self.assertEqual(figures["ratio"], 16)
        self.assertEqual(figures["size"], "1024x1024")
        self.assertEqual(figures["quant"], "w4a8")


@unittest.skipUnless(REAL and Path(REAL).is_dir(),
                     "set $XESTRATA_HUNYUANIMAGE3_TOKENIZER to the folder with the model's files")
class RealTokenizer(unittest.TestCase):
    """The same builder over the model's own tokenizer.json (the markers are its added tokens)."""

    def setUp(self):
        self.tok = images.load_tokenizer(str(Path(REAL) / "tokenizer.json"), "hunyuan-image-3")
        self.bucket = images.hunyuanimage3_bucket(cfg(), 1024, 1024)

    def test_the_markers_are_its_added_tokens(self):
        for name in FakeTokenizer.NAMES:
            self.assertEqual(self.tok.encode(name, parse_special=True), [self.tok.special_ids[name]], name)

    def test_a_1024_sequence(self):
        ids = images.hunyuanimage3_ids(self.tok, "A red fox in the snow", self.bucket, 1024,
                                       "You are a helpful assistant.")
        self.assertEqual(ids[0], self.tok.special_ids["<|startoftext|>"])
        self.assertEqual(ids.count(self.tok.special_ids["<img>"]), 64 * 64)
        i = ids.index(self.tok.special_ids["<answer>"])
        markers = [self.tok.special_ids[n] for n in ("<answer>", "<boi>", "<img_size_1024>", "<img_ratio_16>",
                                                     "<timestep>", "<guidance>", "<timestep_r>")]
        self.assertEqual(ids[i:i + 7], markers)
        self.assertEqual(ids[-3:], [self.tok.special_ids["<eoi>"], self.tok.special_ids["</answer>"]]
                         + self.tok.encode("\n\n", parse_special=False))
        head = self.tok.decode(ids[1:i])
        self.assertTrue(head.endswith("Assistant: "), head[-40:])
        self.assertIn("A red fox in the snow", head)

    def test_the_system_turn_is_the_models_own_default(self):
        """The model's generation_config asks for en_unified; None is the seam the CPU golden was made with."""
        with_system = images.hunyuanimage3_ids(self.tok, "p", self.bucket, 1024, "sys")
        without = images.hunyuanimage3_ids(self.tok, "p", self.bucket, 1024, None)
        self.assertEqual(len(with_system) - len(without),
                         len(self.tok.encode("sys", parse_special=False)) + 1)


if __name__ == "__main__":
    unittest.main()
