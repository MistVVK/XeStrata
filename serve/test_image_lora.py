# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/loras.py and the image endpoint's adapters: the folder's listing (a file's keys read by its header), a
request's "lora" field and prompt tags, the IMG and NEED words the engine is sent, the response, and the models that
take none.  The web page's side too (serve/web/app.js): its picker, chips, strengths and trigger words.

The files are written here, not read from a model: a safetensors' header is the 8-byte length of its JSON and the
JSON; a GGUF's is its metadata's fields and its tensors' infos (the data is never read: the listing knows a weight's
shape from it).

    python -m unittest serve.test_image_lora
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import images, loras


def safetensors(path: Path, tensors: dict[str, tuple[int, ...]], meta: dict[str, str] | None = None) -> None:
    """A safetensors file of those tensors (F16) and that metadata; the tensors' data is zeros: only the header is
    read."""
    head: dict = {} if meta is None else {"__metadata__": meta}
    at = 0
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        head[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [at, at + 2 * n]}
        at += 2 * n
    raw = json.dumps(head).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(at))


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(path: Path, tensors: dict[str, tuple[int, ...]], meta: dict[str, str] | None = None) -> None:
    """A GGUF file of version 3 with those tensors (F16) and string metadata (its count and the tensors' infos are the
    header; serve/loras.py reads no further)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta or {})))
    for k, v in (meta or {}).items():
        out += _gguf_string(k) + struct.pack("<I", 8) + _gguf_string(v)
    for name, shape in tensors.items():
        dims = list(reversed(shape))    # GGUF's shape runs from the innermost dimension
        out += _gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        out += struct.pack("<IQ", 1, 0)   # F16, its offset in the data
    path.write_bytes(bytes(out) + bytes(4096))


def moss_lora(rank: int = 4, blocks: tuple[int, ...] = (0, 1),
              prefix: str = "transformer.") -> dict[str, tuple[int, ...]]:
    """MOSS-Image's adapter's tensors, as a diffusers save names them (layers.N.proj.*), at `rank` and that prefix."""
    out: dict[str, tuple[int, ...]] = {}
    for b in blocks:
        for name, cols, rows in (("proj.q", 128, 128), ("proj.ffn_up_gate", 128, 256), ("proj.ffn_down", 128, 128)):
            out[f"{prefix}layers.{b}.{name}.lora_A.weight"] = (rank, cols)
            out[f"{prefix}layers.{b}.{name}.lora_B.weight"] = (rows, rank)
    for name, cols, rows in (("qkv", 128, 384), ("ffn_up", 128, 128), ("ffn_gate", 128, 128),
                             ("ffn_down", 128, 128)):
        out[f"{prefix}text_embedder.blocks.0.{name}.lora_A.weight"] = (rank, cols)
        out[f"{prefix}text_embedder.blocks.0.{name}.lora_B.weight"] = (rows, rank)
    return out


def qwen_lora(rank: int = 4, blocks: tuple[int, ...] = (0, 1), prefix: str = "transformer.",
              extra: dict[str, tuple[int, ...]] | None = None, meta: dict[str, str] | None = None) -> dict:
    """A Qwen-Image-2.1 adapter's tensors, as diffusers' saves name them (transformer_blocks.N.attn.to_q and the
    like), at `rank` and that prefix."""
    out: dict[str, tuple[int, ...]] = {}
    for b in blocks:
        for name, cols, rows in (("attn.to_q", 256, 256), ("attn.to_out.0", 256, 256),
                                 ("img_mlp.gate_layer", 256, 512), ("img_mlp.proj", 256, 512),
                                 ("img_mlp.out", 512, 256)):
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_A.weight"] = (rank, cols)
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_B.weight"] = (rows, rank)
    out.update(extra or {})
    return out


def mage_lora(rank: int = 4, blocks: tuple[int, ...] = (0, 1),
              prefix: str = "diffusion_model.",
              extra: dict[str, tuple[int, ...]] | None = None) -> dict[str, tuple[int, ...]]:
    """Mage-Flow's adapter's tensors, as ai-toolkit's mageflow form and the dot form name them
    (transformer_blocks.N.attn.to_q and the like), at `rank` and that prefix."""
    out: dict[str, tuple[int, ...]] = {}
    for b in blocks:
        for name, cols, rows in (("attn.to_q", 256, 256), ("attn.add_q_proj", 256, 256),
                                 ("attn.to_add_out", 256, 256), ("img_mlp.net.0.proj", 256, 512),
                                 ("img_mod.1", 256, 1536)):
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_A.weight"] = (rank, cols)
            out[f"{prefix}transformer_blocks.{b}.{name}.lora_B.weight"] = (rows, rank)
    out.update(extra or {})
    return out


class Tree(unittest.TestCase):
    """A temporary LoRA folder (the config's "lora_dir")."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "loras"
        self.d.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, family: str | None = None) -> loras.Folder:
        cfg = {"diffusion_model": str(self.d / "dit.gguf"), "lora_dir": str(self.d)}
        if family:
            cfg["family"] = family
        return loras.Folder(cfg)

    def qwen(self, name: str = "style/gouache", **kw) -> Path:
        path = self.d / f"{name}.safetensors"
        safetensors(path, qwen_lora(**{k: v for k, v in kw.items() if k != "meta"}),
                    kw.get("meta", {"modelspec.trigger_phrase": "gouache style",
                                    "recommended_strength_model": "0.8"}))
        return path


class Listing(Tree):
    """What the folder holds for a model: the files its DiT takes, each as the listing's fields."""

    def test_a_file_the_dit_takes_is_listed(self):
        self.qwen()
        entries = self.folder().listing()
        self.assertEqual([e["name"] for e in entries], ["style/gouache"])
        e = entries[0]
        self.assertEqual(e["path"], "style/gouache.safetensors", "the path under the folder, with its extension")
        self.assertEqual(e["rank"], 4)
        self.assertEqual((e["scale"], e["trigger"]), (0.8, "gouache style"))
        self.assertEqual(e["targets"], ["attn.to_out.0", "attn.to_q", "img_mlp.gate_layer", "img_mlp.out",
                                        "img_mlp.proj"])
        self.assertGreater(e["size_mib"], 0)

    def test_a_gguf_and_a_fused_gate_up(self):
        gguf(self.d / "fused.gguf", {
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight": (2, 256),
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight": (1024, 2)},
            {"ss_network_dim": "2"})
        entries = self.folder().listing()
        self.assertEqual([(e["name"], e["rank"], e["targets"]) for e in entries],
                         [("fused", 2, ["img_mlp.gate_up"])])
        self.assertEqual(entries[0]["scale"], 1.0, "no recommended strength: 1.0")
        self.assertIsNone(entries[0]["trigger"])

    def test_a_lokr_is_listed_with_its_factorizations_rank(self):
        b = "transformer.transformer_blocks.0."
        safetensors(self.d / "lokr.safetensors", {
            # to_q: w1 from its two factors, w2 whole (the file's alpha divides w1's factorization's rank)
            b + "attn.to_q.lokr_w1_a.weight": (16, 8), b + "attn.to_q.lokr_w1_b.weight": (8, 16),
            b + "attn.to_q.lokr_w2": (16, 16),
            b + "attn.to_q.alpha": (1,),
            # to_out.0: both from factors, w2's under a linear Tucker core
            b + "attn.to_out.0.lokr_w1_a.weight": (16, 8), b + "attn.to_out.0.lokr_w1_b.weight": (8, 16),
            b + "attn.to_out.0.lokr_w2_a.weight": (16, 16), b + "attn.to_out.0.lokr_w2_b.weight": (16, 16),
            b + "attn.to_out.0.lokr_t2": (16, 16, 1, 1),
        })
        f = self.folder()
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["lokr"], f.skipped)
        e = entries[0]
        self.assertEqual(e["rank"], 16, "the rank of w2's factors where w2 is decomposed (the alpha's divisor)")
        self.assertEqual(e["targets"], ["attn.to_out.0", "attn.to_q"])
        self.assertEqual(loras.field([{"name": "lokr", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_a_dora_is_listed_and_one_over_the_columns_is_not(self):
        b = "transformer.transformer_blocks.0.attn.to_q"
        safetensors(self.d / "dora.safetensors", qwen_lora(
            extra={b + ".dora_scale": (256,)}))   # the weight's rows: the decomposition's output axis
        safetensors(self.d / "columns.safetensors", qwen_lora(
            extra={"transformer.transformer_blocks.0.img_mlp.out.dora_scale": (512,)}))   # the weight's columns
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["dora"], f.skipped)
        self.assertIn("columns", f.skipped["columns"], "a DoRA over the weight's columns is not the engine's")
        self.assertIn("DoRA", f.skipped["columns"])
        self.assertEqual(loras.field([{"name": "dora", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_a_doras_magnitude_under_pefts_and_ai_toolkits_names_is_the_same_magnitude(self):
        b = "transformer.transformer_blocks.0.attn.to_q"
        for name, key in (("peft", b + ".lora_magnitude_vector.default.weight"),
                          ("peft-plain", b + ".lora_magnitude_vector.weight"),
                          ("toolkit", b + ".magnitude")):
            safetensors(self.d / f"{name}.safetensors", qwen_lora(extra={key: (256,)}))
        f = self.folder()
        self.assertEqual(sorted(e["name"] for e in f.listing()), ["peft", "peft-plain", "toolkit"], f.skipped)
        for entry in f.listing():
            self.assertNotIn(entry["name"], f.ignored, "the magnitude is a tensor of the adapter, not a left-out key")
        safetensors(self.d / "lycoris.safetensors", qwen_lora(
            extra={b + ".dora_scale": (256, 1)}))   # LyCORIS writes the rows' vector [rows][1]
        f = self.folder()
        self.assertEqual(sorted(e["name"] for e in f.listing()), ["lycoris", "peft", "peft-plain", "toolkit"],
                         f.skipped)

    def test_a_magnitude_over_the_columns_is_refused_even_where_the_rows_would_fit(self):
        b = "transformer.transformer_blocks.0.attn.to_q"   # this weight's rows are its columns: 256 and 256
        safetensors(self.d / "square-columns.safetensors", qwen_lora(extra={b + ".dora_scale": (1, 256)}))
        f = self.folder()
        self.assertEqual(f.listing(), [])
        self.assertIn("over the weight's columns", f.skipped["square-columns"],
                      "a [1][cols] magnitude says which way it runs; a square weight must not read it as the rows")
        safetensors(self.d / "shape.safetensors", qwen_lora(
            extra={"transformer.transformer_blocks.0.attn.to_q.dora_scale": (7, 9)}))
        self.assertIn("neither the weight's rows nor its columns", self.folder().skipped["shape"])

    def test_rslora_is_refused_by_either_name_and_a_plain_metadata_alpha_is_not(self):
        b = "transformer.transformer_blocks.0.attn.to_q"
        for name, field in (("kohya", "use_rslora"), ("diffusers", "transformer.use_rslora")):
            safetensors(self.d / f"{name}.safetensors", qwen_lora(extra={b + ".alpha": (1,)}),
                        {"lora_adapter_metadata": '{"%s": true, "lora_alpha": 8, "rank": 4}' % field})
        f = self.folder()
        self.assertEqual(f.listing(), [])
        for name in ("kohya", "diffusers"):
            self.assertIn("rsLoRA", f.skipped[name], f.skipped)
        safetensors(self.d / "alpha.safetensors", qwen_lora(),
                    {"lora_adapter_metadata": '{"transformer.lora_alpha": 8, "use_rslora": false}'})
        self.assertEqual([e["name"] for e in self.folder().listing()], ["alpha"])
        # one field a component in the same metadata (diffusers' packed save): the transformer's rsLoRA is read even
        # where a text encoder's false comes first in the sorted keys (the engine's json_true does the same)
        safetensors(self.d / "packed.safetensors", qwen_lora(),
                    {"lora_adapter_metadata": '{"text_encoder.rank": 4, "text_encoder.use_rslora": false, '
                                              '"transformer.rank": 4, "transformer.use_rslora": true}'})
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["alpha"], f.skipped)
        self.assertIn("rsLoRA", f.skipped["packed"], f.skipped)

    def test_metadata_that_names_two_different_alphas_is_refused(self):
        # diffusers' pack_weights writes one metadata field a component into the same "lora_adapter_metadata", so a
        # file that carries a text encoder's and a transformer's adapter holds two alphas; the engine reads one alpha
        # for the weights that keep no ".alpha" of their own, so which one those take is not said (review2's FIX-1).
        two = '{"text_encoder.lora_alpha": 8, "text_encoder.rank": 4, "transformer.lora_alpha": 64, ' \
              '"transformer.rank": 4}'
        every = {f"transformer.transformer_blocks.{b}.{n}.alpha": (1,)
                 for b in (0, 1) for n in ("attn.to_q", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj",
                                           "img_mlp.out")}
        safetensors(self.d / "two-alphas.safetensors", qwen_lora(), {"lora_adapter_metadata": two})
        safetensors(self.d / "two-alphas-tensor.safetensors", qwen_lora(extra=every),
                    {"lora_adapter_metadata": two})   # every weight says its own alpha: the file is this model's
        safetensors(self.d / "same-alphas.safetensors", qwen_lora(),
                    {"lora_adapter_metadata": '{"text_encoder.lora_alpha": 16, "transformer.lora_alpha": 16}'})
        safetensors(self.d / "one-alpha.safetensors", qwen_lora(),
                    {"lora_adapter_metadata": '{"transformer.lora_alpha": 16}'})
        f = self.folder()
        self.assertEqual(sorted(e["name"] for e in f.listing()),
                         ["one-alpha", "same-alphas", "two-alphas-tensor"], f.skipped)
        self.assertIn("different lora_alpha values", f.skipped["two-alphas"], f.skipped)

    def test_keys_that_hold_no_tensor_of_the_adapter_are_said(self):
        b = "transformer.transformer_blocks.0.attn.to_q"
        safetensors(self.d / "peft.safetensors", qwen_lora(extra={
            b + ".base_layer.bias": (256,), "transformer.format_version": (1,)}))
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["peft"], "the file is still this model's")
        self.assertEqual(set(f.ignored["peft"]), {b + ".base_layer.bias", "transformer.format_version"},
                         "said, never dropped in silence")

    def test_a_loha_is_listed_with_its_factors_rank(self):
        b = "transformer.transformer_blocks.0."
        safetensors(self.d / "loha.safetensors", {
            # attn.to_q: the plain form, the file's alpha's divisor w1_b's rows
            b + "attn.to_q.hada_w1_a.weight": (256, 8), b + "attn.to_q.hada_w1_b.weight": (8, 256),
            b + "attn.to_q.hada_w2_a.weight": (256, 4), b + "attn.to_q.hada_w2_b.weight": (4, 256),
            b + "attn.to_q.alpha": (1,),
            # img_mlp.out: both sides under linear Tucker cores, their factors written the core's way (a core of one
            # dimension, w_a [1][out], is a rank of 1; both cores come together or not at all)
            b + "img_mlp.out.hada_w1_a.weight": (1, 256), b + "img_mlp.out.hada_w1_b.weight": (8, 512),
            b + "img_mlp.out.hada_t1": (1, 8, 1, 1),
            b + "img_mlp.out.hada_w2_a.weight": (5, 256), b + "img_mlp.out.hada_w2_b.weight": (4, 512),
            b + "img_mlp.out.hada_t2": (5, 4, 1, 1),
        })
        safetensors(self.d / "sides.safetensors", {   # the two sides' products are not one weight's shape
            b + "attn.to_q.hada_w1_a.weight": (256, 8), b + "attn.to_q.hada_w1_b.weight": (8, 256),
            b + "attn.to_q.hada_w2_a.weight": (256, 4), b + "attn.to_q.hada_w2_b.weight": (4, 128)})
        safetensors(self.d / "kernel.safetensors", {   # a convolution's core
            b + "attn.to_q.hada_w1_a.weight": (256, 8), b + "attn.to_q.hada_w1_b.weight": (8, 256),
            b + "attn.to_q.hada_t1": (3, 8, 3, 3),
            b + "attn.to_q.hada_t2": (3, 8, 1, 1),
            b + "attn.to_q.hada_w2_a.weight": (256, 4), b + "attn.to_q.hada_w2_b.weight": (4, 256)})
        f = self.folder()
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["loha"], f.skipped)
        e = entries[0]
        self.assertEqual(e["rank"], 8, "the file's alpha is divided by w1_b's rows")
        self.assertEqual(e["targets"], ["attn.to_q", "img_mlp.out"])
        self.assertIn("products are 256 x 256 and 256 x 128", f.skipped["sides"])
        self.assertIn("convolution's kernel", f.skipped["kernel"])
        self.assertEqual(loras.field([{"name": "loha", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_a_glora_is_listed_with_the_rank_its_reading_gives(self):
        b = "transformer.transformer_blocks.0."
        safetensors(self.d / "glora.safetensors", {
            # to_q (256 of 256): the newer reading, a1 [cols][rank], a2 [rank][cols], b1 [rows][rank], b2
            # [rank][cols]; every tensor square-or-not, the file's alpha divided by a1's columns
            b + "attn.to_q.a1.weight": (256, 8), b + "attn.to_q.a2.weight": (8, 256),
            b + "attn.to_q.b1.weight": (256, 8), b + "attn.to_q.b2.weight": (8, 256),
            b + "attn.to_q.alpha": (1,),
            # img_mlp.out (256 of 512): LyCORIS's older reading, a1 [rank][cols], a2 [cols][rank], b1 [rank][cols],
            # b2 [rows][rank] (a weight of no square shape reads only one way)
            b + "img_mlp.out.a1.weight": (4, 512), b + "img_mlp.out.a2.weight": (512, 4),
            b + "img_mlp.out.b1.weight": (4, 512), b + "img_mlp.out.b2.weight": (256, 4)})
        safetensors(self.d / "shapes.safetensors", {   # neither of the form's two readings
            b + "attn.to_q.a1.weight": (256, 8), b + "attn.to_q.a2.weight": (8, 200),
            b + "attn.to_q.b1.weight": (256, 8), b + "attn.to_q.b2.weight": (8, 256)})
        safetensors(self.d / "missing.safetensors", {b + "attn.to_q.a1.weight": (256, 8)})
        safetensors(self.d / "dora.safetensors", {   # the engine's DoRA is the plain form's
            b + "attn.to_q.a1.weight": (256, 8), b + "attn.to_q.a2.weight": (8, 256),
            b + "attn.to_q.b1.weight": (256, 8), b + "attn.to_q.b2.weight": (8, 256),
            b + "attn.to_q.dora_scale": (256,)})
        f = self.folder()
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["glora"], f.skipped)
        e = entries[0]
        self.assertEqual(e["rank"], 8, "a square weight's four fit both readings; the smaller rank is the file's")
        self.assertEqual(e["targets"], ["attn.to_q", "img_mlp.out"])
        self.assertIn("its a2 is 8 x 200", f.skipped["shapes"])
        self.assertIn("needs a1, a2, b1 and b2", f.skipped["missing"])
        self.assertIn("DoRA", f.skipped["dora"])
        self.assertEqual(loras.field([{"name": "glora", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_an_oft_is_listed_and_the_blocks_it_cannot_apply_are_not(self):
        b = "transformer.transformer_blocks.0."
        safetensors(self.d / "oft.safetensors", {
            # to_q: an OFT's blocks over the weight's 256 rows, with the file's alpha (the constraint, not a scale)
            b + "attn.to_q.oft_blocks": (32, 8, 8), b + "attn.to_q.alpha": (1,),
            # img_mlp.out: a BOFT's four dimensions, a rescale of the weight's rows beside them
            b + "img_mlp.out.oft_blocks": (2, 16, 16, 16), b + "img_mlp.out.rescale": (256,)})
        safetensors(self.d / "stages.safetensors", {   # 2^1 of the 3 blocks: a BOFT's stage 1 cannot read them
            b + "attn.to_q.oft_blocks": (2, 3, 16, 16)})
        safetensors(self.d / "squares.safetensors", {b + "attn.to_q.oft_blocks": (32, 8, 4)})
        safetensors(self.d / "rescale.safetensors", {
            b + "attn.to_q.oft_blocks": (32, 8, 8), b + "attn.to_q.rescale": (128,)})
        safetensors(self.d / "dora.safetensors", {   # the engine's DoRA is the plain form's
            b + "attn.to_q.oft_blocks": (32, 8, 8), b + "attn.to_q.dora_scale": (256,)})
        f = self.folder()
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["oft"], f.skipped)
        e = entries[0]
        self.assertEqual(e["rank"], 0, "a turn of the rows has no rank")
        self.assertEqual(e["targets"], ["attn.to_q", "img_mlp.out"])
        self.assertIn("not a multiple of 2", f.skipped["stages"])
        self.assertIn("not squares", f.skipped["squares"])
        self.assertIn("not the weight's 256 rows", f.skipped["rescale"])
        self.assertIn("DoRA", f.skipped["dora"])
        self.assertEqual(loras.field([{"name": "oft", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_a_whole_delta_is_listed_and_this_models_bias_delta_is_not(self):
        b = "transformer.transformer_blocks.0."
        safetensors(self.d / "diff.safetensors", {
            b + "attn.to_q.diff": (256, 256), b + "img_mlp.out.diff": (256, 512)})
        safetensors(self.d / "set.safetensors", {b + "attn.to_q.set_weight": (256, 256)})
        safetensors(self.d / "norm.safetensors", {b + "attn.to_q.w_norm": (256, 256)})
        # Qwen-Image-2.1's DiT weights add no bias: its file's bias's delta has no place, and the engine refuses it
        safetensors(self.d / "bias.safetensors", {
            b + "attn.to_q.diff": (256, 256), b + "attn.to_q.diff_b": (256,)})
        safetensors(self.d / "bias-only.safetensors", {b + "attn.to_q.diff_b": (256,)})
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["diff", "norm", "set"], f.skipped)
        diff = [e for e in f.listing() if e["name"] == "diff"][0]
        self.assertEqual(diff["targets"], ["attn.to_q", "img_mlp.out"])
        self.assertEqual(diff["rank"], 0, "a whole delta has no rank")
        self.assertIn("adds no bias", f.skipped["bias"], "the model's weights have no bias to add it beside")
        self.assertIn("adds no bias", f.skipped["bias-only"])
        self.assertEqual(loras.field([{"name": "set", "scale": 0.7}], f)[0]["scale"], 0.7)

    def test_a_bias_delta_is_listed_where_the_weight_adds_a_bias(self):
        # Mage-Flow's DiT: every weight of the table has a bias, so a file may hold one beside a LoRA's or alone
        b = "diffusion_model.transformer_blocks.0."
        safetensors(self.d / "with.safetensors", mage_lora(extra={
            b + "attn.to_q.diff_b": (256,), b + "img_mod.1.diff_b": (1536,)}))
        safetensors(self.d / "alone.safetensors", {b + "attn.to_q.diff_b": (256,)})
        safetensors(self.d / "norm.safetensors", {   # b_norm beside the weight's w_norm: the same bias's patch
            b + "attn.to_q.w_norm": (256, 256), b + "attn.to_q.b_norm": (256,)})
        safetensors(self.d / "loose.safetensors", {   # b_norm without w_norm: read beside w_norm only
            b + "attn.to_q.b_norm": (256,)})
        f = self.folder("mage-flow")
        self.assertEqual([e["name"] for e in f.listing()], ["alone", "norm", "with"], f.skipped)
        listed = {e["name"]: e for e in f.listing()}
        self.assertEqual(listed["alone"]["targets"], ["attn.to_q"])
        self.assertEqual(listed["alone"]["rank"], 0, "a bias's delta alone changes no weight")
        self.assertIn("img_mod.1", listed["with"]["targets"], "the file's other tensors are its rank's")
        self.assertEqual(listed["norm"]["targets"], ["attn.to_q"])
        self.assertIn("w_norm", f.skipped["loose"])

    def test_a_moss_images_bias_delta_and_the_blocks_that_add_none(self):
        # MOSS-Image: its text embedder and the weights outside the blocks add biases; the 42 blocks' shift is their
        # modulation's, and the engine refuses a bias's delta there
        safetensors(self.d / "text.safetensors", {
            "transformer.text_embedder.blocks.0.qkv.diff_b": (384,)})
        safetensors(self.d / "block.safetensors", {"transformer.layers.0.proj.q.diff_b": (128,)})
        f = self.folder("moss-image")
        self.assertEqual([e["name"] for e in f.listing()], ["text"], f.skipped)
        self.assertEqual(f.listing()[0]["targets"], ["qkv"])
        self.assertIn("adds no bias", f.skipped["block"])

    def test_another_models_file_and_the_forms_not_applied_are_left_out(self):
        self.qwen("qwen")
        safetensors(self.d / "moss.safetensors", {
            "layers.0.proj.q.lora_A.weight": (8, 256), "layers.0.proj.q.lora_B.weight": (256, 8)})
        safetensors(self.d / "dora.safetensors", {
            "transformer.transformer_blocks.0.attn.to_q.lokr_w1": (256, 256),
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2": (256, 256),
            "transformer.transformer_blocks.0.attn.to_q.dora_scale": (256,)})
        safetensors(self.d / "lokr.safetensors", {
            "transformer.transformer_blocks.0.attn.to_q.lokr_w1": (256, 256),
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2_a": (4, 64),
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2_b": (4, 64)})
        safetensors(self.d / "text-encoder.safetensors", {
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 64),
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": (256, 4)})
        safetensors(self.d / "a-norm.safetensors", {
            "transformer.transformer_blocks.0.attn.norm_q.lora_A.weight": (4, 256),
            "transformer.transformer_blocks.0.attn.norm_q.lora_B.weight": (256, 4)})
        (self.d / "broken.safetensors").write_bytes(b"not a header at all")
        (self.d / "notes.txt").write_text("x", encoding="utf-8")
        safetensors(self.d / ".hidden.safetensors", qwen_lora())
        f = self.folder()
        self.assertEqual([e["name"] for e in f.listing()], ["qwen"])
        self.assertEqual(f.skipped["moss"], "layers.0.proj.q is not one of this image model's weights")
        self.assertIn("DoRA", f.skipped["dora"], "a DoRA on a LoKr is not the engine's")
        self.assertIn("lokr_w2_b is 4 x 64, not the rank its lokr_w2_a has (64)", f.skipped["lokr"],
                      "a LoKr's factors that do not meet are not one")
        self.assertIn("text encoder", f.skipped["text-encoder"])
        self.assertIn("not one of this image model's weights", f.skipped["a-norm"])
        self.assertIn("header", f.skipped["broken"])

    def test_a_name_with_a_space_and_the_line_word(self):
        self.qwen("my style")
        e = self.folder().listing()[0]
        self.assertEqual(e["name"], "my style")
        self.assertEqual(loras.encode_path("/x/my style.safetensors"), "/x/my%20style.safetensors")
        self.assertEqual(loras.encode_path("C:\\a b\\c:d.safetensors"), "C:\\a%20b\\c:d.safetensors")

    def test_two_files_of_one_stem_keep_their_extension(self):
        safetensors(self.d / "gouache.safetensors", qwen_lora())
        gguf(self.d / "gouache.gguf", {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (2, 256),
                                       "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (256, 2)})
        self.assertEqual([e["name"] for e in self.folder().listing()], ["gouache.gguf", "gouache.safetensors"])

    def test_the_listing_follows_the_folder(self):
        self.assertEqual(self.folder().listing(), [])
        self.qwen("late")
        self.assertEqual([e["name"] for e in self.folder().listing()], ["late"],
                         "a new file: the folder's mtime changed")

    def test_a_model_that_takes_no_adapter_lists_none(self):
        self.qwen()
        self.assertEqual(self.folder("hunyuan-image-3").listing(), [])

    def test_a_mage_flow_file_and_its_two_streams(self):
        safetensors(self.d / "rooms9.safetensors", mage_lora())
        self.qwen("qwen")
        f = self.folder("mage-flow")
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["rooms9"])
        e = entries[0]
        kinds = ["attn.add_q_proj", "attn.to_add_out", "attn.to_q", "img_mlp.net.0.proj", "img_mod.1"]
        self.assertEqual(e["targets"], kinds, "a block's number is not part of the kind")
        self.assertIn("is not one of this image model's weights", f.skipped["qwen"],
                      "Qwen-Image-2.1's weights are not Mage-Flow's")
        self.assertEqual(loras.field([{"name": "rooms9", "scale": 0.6}], f)[0]["scale"], 0.6)

    def test_moss_images_file_and_the_weights_of_its_split_qkv(self):
        safetensors(self.d / "moss.safetensors", moss_lora())
        self.qwen("qwen")
        f = self.folder("moss-image")
        entries = f.listing()
        self.assertEqual([e["name"] for e in entries], ["moss"])
        e = entries[0]
        self.assertEqual(e["rank"], 4)
        self.assertEqual(e["targets"], ["ffn_down", "ffn_gate", "ffn_up", "proj.ffn_down", "proj.ffn_up_gate",
                                        "proj.q", "qkv"], "a block's number is not part of the kind")
        self.assertIn("is not one of this image model's weights", f.skipped["qwen"],
                      "Qwen-Image-2.1's weights are not MOSS-Image's")
        use = loras.field([{"name": "moss"}], f)[0]
        self.assertEqual(use["name"], "moss")


def underscore_form(tensors: dict[str, tuple[int, ...]], prefix: str = "lora_unet_") -> dict[str, tuple[int, ...]]:
    """The same tensors as an sd-scripts file (`lora_unet_`) or a LyCORIS one (`lycoris_`) writes them: the weight's
    name with every dot an underscore, the role's own suffix and `.weight` as they stand in the dot form."""
    out: dict[str, tuple[int, ...]] = {}
    for raw, shape in tensors.items():
        key, role, _ = loras.split_key(raw)
        tail = ".weight" if raw.endswith(".weight") else ""
        suffix = next(s for s, r in loras.ROLES if r == role and raw[: len(raw) - len(tail)].endswith(s))
        out[f"{prefix}{key.replace('.', '_')}{suffix}{tail}"] = shape
    return out


class Underscores(Tree):
    """sd-scripts' `lora_unet_` keys and the LyCORIS form's `lycoris_` ones (the survey's problems 1 and 2): the same
    weights a dot-form file names, with every dot of the model's own name written as an underscore.  MageTrail shares
    Mage-Flow's table, so these three tables cover the four models."""

    FAMILIES = (("qwen-image", qwen_lora), ("mage-flow", mage_lora), ("moss-image", moss_lora))

    def test_sd_scripts_and_lycoris_names_read_as_the_dot_form(self):
        for family, make in self.FAMILIES:
            dot = make(blocks=(0,))
            safetensors(self.d / f"{family}-dot.safetensors", dot)
            safetensors(self.d / f"{family}-unet.safetensors", underscore_form(dot, "lora_unet_"))
            safetensors(self.d / f"{family}-lyco.safetensors", underscore_form(dot, "lycoris_"))
            f = self.folder(family)
            entries = {e["name"]: e for e in f.listing()}
            self.assertEqual(sorted(entries), [f"{family}-dot", f"{family}-lyco", f"{family}-unet"], f.skipped)
            want = (entries[f"{family}-dot"]["rank"], entries[f"{family}-dot"]["targets"])
            for other in (f"{family}-unet", f"{family}-lyco"):
                self.assertEqual((entries[other]["rank"], entries[other]["targets"]), want,
                                 f"{family}: {other} names the same weights, every dot an underscore")

    def test_the_other_forms_under_the_underscore_prefixes(self):
        # a LoHa and a LoKr (LyCORIS' own forms, sd-scripts' and SimpleTuner's names) and a DoRA's magnitude in
        # LyCORIS' [rows][1] shape: each lists as the dot form of the very same tensors does
        loha = {"transformer.transformer_blocks.0.attn.to_q.hada_w1_a.weight": (256, 8),
                "transformer.transformer_blocks.0.attn.to_q.hada_w1_b.weight": (8, 256),
                "transformer.transformer_blocks.0.attn.to_q.hada_w2_a.weight": (256, 4),
                "transformer.transformer_blocks.0.attn.to_q.hada_w2_b.weight": (4, 256),
                "transformer.transformer_blocks.0.attn.to_q.alpha": (1,)}
        lokr = {"transformer.layers.0.proj.ffn_up_gate.lokr_w1_a": (16, 8),   # the name's own underscores too
                "transformer.layers.0.proj.ffn_up_gate.lokr_w1_b": (8, 16),
                "transformer.layers.0.proj.ffn_up_gate.lokr_w2": (16, 16)}
        dora = mage_lora(blocks=(0,),
                         extra={"diffusion_model.transformer_blocks.0.attn.to_q.dora_scale": (256, 1)})
        for family, tensors, prefix in (("qwen-image", loha, "lora_unet_"), ("moss-image", lokr, "lycoris_"),
                                        ("mage-flow", dora, "lora_unet_")):
            fields = [(e["rank"], e["targets"]) for e in self._both_forms(family, tensors, prefix)]
            self.assertEqual(fields[0], fields[1], f"{family}: {prefix} reads as the dot form ({fields})")

    def _both_forms(self, family: str, tensors: dict, prefix: str) -> list[dict]:
        """The folder's one entry, once from the dot form and once from the underscore form of the same tensors."""
        out = []
        for tensors_, name in ((tensors, "dot"), (underscore_form(tensors, prefix), "under")):
            safetensors(self.d / f"{name}.safetensors", tensors_)
            f = self.folder(family)
            self.assertEqual([e["name"] for e in f.listing()], [name], f.skipped)
            out.append(f.listing()[0])
            (self.d / f"{name}.safetensors").unlink()
        return out

    def test_an_underscore_name_the_model_has_not_is_said_as_the_file_wrote_it(self):
        safetensors(self.d / "moss-on-qwen.safetensors", {
            "lora_unet_layers_3_proj_ffn_up_gate.lora_down.weight": (4, 128),
            "lora_unet_layers_3_proj_ffn_up_gate.lora_up.weight": (256, 4)})
        f = self.folder("qwen-image")
        self.assertEqual(f.listing(), [])
        self.assertIn("layers_3_proj_ffn_up_gate", f.skipped["moss-on-qwen"],
                      "the name no table answers, said as the file wrote it")

    def test_animas_kohya_names_read_as_the_models_own_names(self):
        # Anima's names were the ones that read, through a table that held each name dotted as well; the underscore
        # table answers them by the real names, so a listing's kinds read as the DiT writes them
        names = ("self_attn_q_proj", "cross_attn_v_proj", "mlp_layer1", "adaln_modulation_mlp_2")
        safetensors(self.d / "anima-kohya.safetensors", {
            **{f"lora_unet_blocks_0_{n}.lora_down.weight": (8, 256) for n in names},
            **{f"lora_unet_blocks_0_{n}.lora_up.weight": (256, 8) for n in names}})
        self.assertEqual([e["targets"] for e in self.folder("anima").listing()],
                         [["adaln_modulation_mlp.2", "cross_attn.v_proj", "mlp.layer1", "self_attn.q_proj"]])


class Requests(Tree):
    """A request's names: the "lora" field, local_path, and the prompt's tags."""

    def setUp(self):
        super().setUp()
        self.qwen("style/gouache")
        self.f = self.folder()

    def test_the_fields_forms(self):
        want = [{"name": "style/gouache", "scale": 0.5, "path": str(self.d / "style/gouache.safetensors")}]
        for value in ([{"name": "style/gouache", "scale": 0.5}], {"name": "style/gouache", "scale": 0.5},
                      '[{"name": "style/gouache", "scale": "0.5"}]', {"name": "style/gouache",
                                                                              "scale": "0.5"}):
            self.assertEqual(loras.field(value, self.f), want, value)
        self.assertEqual(loras.field(None, self.f), [])
        self.assertEqual(loras.field([], self.f), [])
        self.assertEqual(loras.field("style/gouache", self.f)[0]["scale"], 1.0, "a name alone: 1.0")
        self.assertEqual(loras.field([{"name": "style/gouache"}], self.f)[0]["scale"], 1.0, "no scale: 1.0")

    def test_a_name_the_model_does_not_have_is_refused_with_the_names_it_has(self):
        with self.assertRaises(loras.LoraError) as e:
            loras.field([{"name": "nope"}], self.f)
        self.assertIn("nope", str(e.exception))
        self.assertIn("style/gouache", str(e.exception))
        scale = {"transformer.transformer_blocks.0.img_mlp.out.dora_scale": (512,)}   # the columns: not the engine's
        safetensors(self.d / "dora.safetensors", qwen_lora(extra=scale))
        with self.assertRaises(loras.LoraError) as e:
            loras.field([{"name": "dora"}], self.f)
        self.assertIn("DoRA", str(e.exception), "a file of the folder, and why it is not this model's")

    def test_the_limits_and_the_repeats(self):
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache"}, {"name": "style/gouache"}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache", "scale": 2.5}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{"name": "style/gouache", "scale": "big"}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field([{}], self.f)
        with self.assertRaises(loras.LoraError):
            loras.field(7, self.f)
        self.qwen("more", blocks=(0,))
        uses = [{"name": "style/gouache"}] + [{"name": "more"}] * 1
        self.assertEqual(len(loras.field(uses, self.f)), 2)
        with self.assertRaises(loras.LoraError):
            loras.limit([{"name": "x"}] * 5)
        loras.limit([{"name": "x"}] * 4)

    def test_local_path_is_a_file_inside_the_folder_only(self):
        inside = str(self.d / "style/gouache.safetensors")
        use = loras.field([{"local_path": inside, "scale": 0.25}], self.f)[0]
        self.assertEqual((use["name"], use["scale"]), ("style/gouache", 0.25))
        self.assertEqual(loras.field([{"local_path": "style/gouache.safetensors"}], self.f)[0]["name"], "style/gouache")
        outside = Path(self.tmp.name) / "outside.safetensors"
        safetensors(outside, qwen_lora())
        for bad in (str(outside), "../outside.safetensors", str(self.d / "style"), "/etc/passwd", ""):
            with self.assertRaises(loras.LoraError, msg=bad):
                loras.field([{"local_path": bad}], self.f)
        link = self.d / "link.safetensors"
        link.symlink_to(outside)
        with self.assertRaises(loras.LoraError):
            loras.field([{"local_path": str(link)}], self.f)
        with self.assertRaises(loras.LoraError):   # a file of the folder that is not this model's
            loras.field([{"local_path": str(self.d / "style/gouache.safetensors")}],
                        loras.Folder({"lora_dir": str(self.d), "family": "moss-image"}))

    def test_a_scale_text_is_ignored_like_an_unknown_field(self):
        # a text encoder's adapter is not applied (the human's decision of 2026-10-29): the field is no field of the
        # engine's, and its value is not checked (an unknown field's is not either)
        use = loras.field([{"name": "style/gouache", "scale_text": 0.4}], self.f)[0]
        self.assertEqual(set(use), {"name", "path", "scale"})
        self.assertEqual(use["scale"], 1.0)
        use = loras.field([{"name": "style/gouache", "scale_text": -3}], self.f)[0]
        self.assertEqual(use["scale"], 1.0)

    def test_tags_come_out_and_unknown_ones_are_left_out(self):
        text, uses, unknown = loras.strip_tags("a cat <lora:style/gouache:0.5> in <lora:nope:1> snow", self.f)
        self.assertEqual(text, "a cat  in  snow")
        self.assertEqual([(u["name"], u["scale"]) for u in uses], [("style/gouache", 0.5)])
        self.assertEqual(unknown, ["nope"])
        text, uses, unknown = loras.strip_tags("<lora:style/gouache>", self.f)
        self.assertEqual((text, uses[0]["scale"], unknown), ("", 1.0, []))
        text, uses, unknown = loras.strip_tags("<lora:style/gouache:1:0.5>", self.f)
        self.assertEqual((text, uses[0]["scale"], unknown), ("", 1.0, []),
                         "A1111's text encoder's weight (the third field): not applied, and no field of the engine's")
        text, uses, unknown = loras.strip_tags("<lora:style/gouache:0.3> and <lora:style/gouache:0.9>", self.f)
        self.assertEqual([u["scale"] for u in uses], [0.3], "the first naming of a name wins")
        self.assertEqual(loras.strip_tags("no tags here", self.f), ("no tags here", [], []))
        with self.assertRaises(loras.LoraError):
            loras.strip_tags("<lora:style/gouache:heavy>", self.f)
        text, uses, unused = loras.strip_tags("<lora:style/gouache:1:heavy>", self.f)
        self.assertEqual((text, uses[0]["scale"]), ("", 1.0), "the text encoder's field is not the engine's either")

    def test_a_model_that_takes_no_adapter_refuses_both(self):
        f = self.folder("hunyuan-image-3")
        with self.assertRaises(loras.LoraError) as e:
            loras.field([{"name": "style/gouache"}], f)
        self.assertIn("takes no adapters", str(e.exception))
        with self.assertRaises(loras.LoraError):
            loras.strip_tags("a cat <lora:style/gouache:1>", f)
        self.assertEqual(loras.field(None, f), [], "a request of none is no request for one")


def cfg(**kw) -> dict:
    """A run config's "images" section of Qwen-Image-2.1 Turbo."""
    out = {"exe": "xestrata-image", "diffusion_model": "dit.gguf", "vae": "vae.safetensors", "llm": "text.gguf",
           "schedule": "turbo"}
    out.update(kw)
    return out


class Added(Tree):
    """The web page's adapters (serve/loras.py's add, remove, hf_file): kept where a model takes them, refused with
    each model's reason where none does, and taken away."""

    def staged(self, tensors) -> Path:
        tmp = loras.staging([self.folder("mage-flow")], 1000)
        safetensors(tmp, tensors)
        return tmp

    def test_add_refuse_remove(self):
        mage, qwen = self.folder("mage-flow"), self.folder()
        got = loras.add([mage, qwen], self.staged(mage_lora()), "My Style (v2).safetensors")
        self.assertEqual(got, {"path": "My Style _v2.safetensors", "name": "My Style _v2", "models": ["mage-flow"]})
        self.assertEqual([e["name"] for e in mage.listing()], ["My Style _v2"])
        self.assertEqual(qwen.listing(), [])
        with self.assertRaisesRegex(loras.LoraError, "in the LoRA folder already"):
            loras.add([mage, qwen], self.staged(mage_lora()), "My Style _v2.safetensors")
        with self.assertRaisesRegex(loras.LoraError, "no image model here takes this file"):
            junk = self.staged({"foo.lora_A.weight": (4, 8), "foo.lora_B.weight": (8, 4)})
            loras.add([mage, qwen], junk, "x.safetensors")
        with self.assertRaises(loras.LoraError):
            loras.file_name("evil.exe")
        self.assertEqual(loras.file_name("../../up/x.safetensors"), "x.safetensors")
        self.assertEqual(loras.remove([mage, qwen], "My Style _v2.safetensors"),
                         {"removed": "My Style _v2.safetensors"})
        self.assertEqual(mage.listing(), [])
        with self.assertRaises(loras.LoraError):
            loras.remove([mage], "../x.safetensors")
        self.assertEqual([p.name for p in self.d.iterdir() if p.name.startswith(".")], [])   # nothing staged is left

    def test_hugging_face_links(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": ""}):
            self.assertEqual(loras.hf_file("https://huggingface.co/a/b/blob/main/style.safetensors"),
                             ("https://huggingface.co/a/b/resolve/main/style.safetensors", "style.safetensors"))
            self.assertEqual(loras.hf_file("https://huggingface.co/a/b/resolve/v1/sub/x.gguf?download=true"),
                             ("https://huggingface.co/a/b/resolve/v1/sub/x.gguf", "x.gguf"))
        for url in ("https://huggingface.co/datasets/a/b/blob/main/x.safetensors", "https://example.com/x.safetensors",
                    "https://huggingface.co/a/b/blob/main/x.bin"):
            with self.assertRaises(loras.LoraError):
                loras.hf_file(url)

    def test_mirror_and_resume(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://mirror.example/"}):
            url, _ = loras.hf_file("https://huggingface.co/a/b/blob/main/x.gguf")
        self.assertEqual(url, "https://mirror.example/a/b/resolve/main/x.gguf")
        part, tag = self.d / "part", self.d / "etag"
        part.write_bytes(b"abc")
        tag.write_text('"version"', encoding="utf-8")
        fetch = object.__new__(loras.Fetch)
        fetch.url, fetch.folders, fetch.cancel = url, [], threading.Event()
        fetch.got, fetch.total = 0, None

        class Response(io.BytesIO):
            status = 206
            headers = {"ETag": '"version"', "Content-Range": "bytes 3-5/6"}

        def respond(req, timeout):
            self.assertEqual(req.get_header("Range"), "bytes=3-")
            self.assertEqual(req.get_header("If-range"), '"version"')
            return Response(b"def")

        with mock.patch.object(loras.urllib.request, "urlopen", respond), \
                mock.patch.object(loras, "staging", return_value=self.d / "unused"):
            fetch._download(part, tag)
        self.assertEqual(part.read_bytes(), b"abcdef")
        self.assertEqual((fetch.got, fetch.total), (6, 6))

    def test_the_server_routes(self):
        import http.client
        from serve.server import ChatTemplate, MockEngine, Service, serve
        from serve.test_server import CTX, ROOT, ByteTokenizer
        for name in ("dit.safetensors", "vae.safetensors", "text.gguf"):
            (self.d.parent / name).write_bytes(b"x")
        cfg = {"model_name": "mage-test", "family": "mage-flow", "exe": "xestrata-image", "steps": 4,
               "schedule": "mage",
               "diffusion_model": str(self.d.parent / "dit.safetensors"), "vae": str(self.d.parent / "vae.safetensors"),
               "llm": str(self.d.parent / "text.gguf"), "lora_dir": str(self.d)}
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hi", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/test_chat_template.jinja"))
        svc.images = images.ImageModels([cfg])
        httpd = serve(svc, port=0)
        try:
            def post(path, body, kind):
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
                c.request("POST", path, body, {"Content-Type": kind})
                r = c.getresponse()
                out = r.status, json.loads(r.read())
                c.close()
                return out
            good, bad = self.d.parent / "good.safetensors", self.d.parent / "bad.safetensors"
            safetensors(good, mage_lora())
            safetensors(bad, {"foo.lora_A.weight": (4, 8), "foo.lora_B.weight": (8, 4)})
            status, out = post("/v1/images/loras?name=Gouache.safetensors", good.read_bytes(),
                               "application/octet-stream")
            self.assertEqual((status, out["added"]["path"]), (200, "Gouache.safetensors"))
            self.assertEqual([e["name"] for e in svc.images.default.defaults()["loras"]], ["Gouache"])
            status, _ = post("/v1/images/loras?name=bad.safetensors", bad.read_bytes(), "application/octet-stream")
            self.assertEqual(status, 400)
            status, _ = post("/v1/images/loras?name=g.safetensors", good.read_bytes(), "text/plain")
            self.assertEqual(status, 415)                    # a form or text: not an upload
            status, out = post("/v1/images/loras/remove", json.dumps({"path": "Gouache.safetensors"}),
                               "application/json")
            self.assertEqual((status, out), (200, {"removed": "Gouache.safetensors"}))
            self.assertEqual(svc.images.default.defaults()["loras"], [])
        finally:
            httpd.shutdown()
            httpd.server_close()


class FakeTokenizer:
    """What prompt_ids asks of a tokenizer: a prompt's ids (20 of them, as the chat form's turn has)."""

    def encode(self, text, parse_special=False, plain=None):
        return [7] * 20


class FakeEngine:
    """The engine's side of a job: the lines it is sent, and zeros of the size asked for as the image."""

    def __init__(self):
        self.lines: list[str] = []

    def alive(self):
        return True

    def ask(self, line, want, timeout, cancel=None, step=None, stage=None):
        self.lines.append(line)
        if want != "IMAGE":
            return {"least_mib": 1, "full_mib": 2, "held_mib": 3, "cap_mib": 4, "host_mib": 5, "free_mib": 6,
                    "lora_n": 2 if " LORA:" in line else 0, "lora_mib": 8 if " LORA:" in line else 0}
        width, height, out = int(line.split()[1]), int(line.split()[2]), line.split()[5]
        Path(out).write_bytes(bytes(width * height * 4))
        r = {"text_s": 0.5, "sampling_s": 1.0, "decode_s": 0.25}
        if " LORA:" in line:
            r.update(lora_n=2, lora_mib=8)
        return r

    def close(self, shutdown=False):
        pass


class Job(Tree):
    """parse_request's adapters, the IMG and NEED words, and the response."""

    def setUp(self):
        super().setUp()
        self.qwen("style/gouache")
        self.qwen("Strong Style", blocks=(0,))
        self.cfg = cfg(lora_dir=str(self.d))
        self.gen = images.ImageGen(self.cfg)
        self.gen.tok, self.gen.engine = FakeTokenizer(), FakeEngine()

    def tearDown(self):
        shutil.rmtree(self.gen.dir, ignore_errors=True)
        super().tearDown()

    def test_the_field_and_the_tags_reach_the_job(self):
        job = images.parse_request({"prompt": "a cat <lora:style/gouache:0.5>", "size": "256x256"}, self.cfg)
        self.assertEqual(job["prompt"], "a cat ")
        self.assertEqual([(u["name"], u["scale"]) for u in job["loras"]], [("style/gouache", 0.5)])
        job = images.parse_request({"prompt": "a cat", "size": "256x256",
                                    "lora": [{"name": "Strong Style", "scale": -0.5}]}, self.cfg)
        self.assertEqual(job["loras"], [{"name": "Strong Style", "scale": -0.5,
                                         "path": str(self.d / "Strong Style.safetensors")}])
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "lora": {"name": "style/gouache"},
                                    "negative_prompt": "blurry <lora:Strong Style:2>"}, self.cfg)
        self.assertEqual([(u["name"], u["scale"]) for u in job["loras"]],
                         [("style/gouache", 1.0), ("Strong Style", 2.0)])
        self.assertEqual(job["negative_prompt"], "blurry ", "the negative prompt's tags come out too")
        self.assertNotIn("lora_unknown", job)

    def test_a_tags_unknown_name_is_left_out_and_said(self):
        job = images.parse_request({"prompt": "a <lora:nope:1> cat", "size": "256x256"}, self.cfg)
        self.assertEqual(job["prompt"], "a  cat")
        self.assertEqual(job["lora_unknown"], ["nope"])
        self.assertNotIn("loras", job, "nothing is applied")

    def test_an_ignored_key_of_an_applied_file_is_said(self):
        b = "transformer.transformer_blocks.0.attn.to_q"
        safetensors(self.d / "peft-side.safetensors", qwen_lora(blocks=(0,), extra={b + ".base_layer.bias": (256,)}))
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "lora": [{"name": "peft-side"}]}, self.cfg)
        self.assertNotIn("ignored", job["loras"][0], "the listing's shape is unchanged")
        self.gen.log = io.StringIO()   # the log says the names too, at one key (review2's FIX-2)
        _, figures = self.gen.run(job)
        self.assertEqual(figures["lora_ignored"], [{"name": "peft-side", "n": 1, "keys": [b + ".base_layer.bias"]}],
                         "the key is said in the response and the log, not dropped in silence")
        line = self.gen.log.getvalue()
        self.assertIn(f"lora peft-side: 1 of its keys hold no adapter tensor and were left out ({b}.base_layer.bias)",
                      line)

    def test_the_refusals(self):
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "nope"}]}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache"}] * 2}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "<lora:style/gouache:1>"}, self.cfg)
        with self.assertRaises(images.ImageRequestError):
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache", "scale": 3}]}, self.cfg)
        mage = {**self.cfg, "family": "mage-flow", "schedule": "mage", "steps": 4}
        with self.assertRaises(images.ImageRequestError) as e:
            images.parse_request({"prompt": "a cat", "lora": [{"name": "style/gouache"}]}, mage)
        self.assertIn("is not one of this image model's weights", str(e.exception),
                      "Mage-Flow takes adapters now: Qwen-Image-2.1's file is another model's")
        hunyuan = {"exe": "xestrata-image", "family": "hunyuan-image-3", "diffusion_model": "d", "vae": "v",
                   "tokenizer": "t", "system_prompt": "s", "schedule": "hunyuanimage3"}
        with self.assertRaises(images.ImageRequestError) as e:
            images.parse_request({"prompt": "a cat <lora:style/gouache:1>"}, hunyuan)
        self.assertIn("HunyuanImage 3.0", str(e.exception))

    def test_a_scale_text_is_ignored(self):
        job = images.parse_request({"prompt": "a cat", "size": "256x256",
                                    "lora": [{"name": "style/gouache", "scale_text": 0.6}]}, self.cfg)
        self.assertEqual(set(job["loras"][0]), {"name", "path", "scale"})
        self.assertEqual(job["loras"][0]["scale"], 1.0, "the strength is the one given, else the default")
        self.assertNotIn("lora_text", job)
        self.gen.need(job)
        word = f" LORA:1.0:{loras.encode_path(str(self.d / 'style/gouache.safetensors'))}"
        self.assertTrue(self.gen.engine.lines[-1].endswith(word), self.gen.engine.lines[-1])
        made, figures = self.gen.run(job)
        self.assertNotIn("lora_text", figures)

    def test_the_words_the_engine_is_sent_and_the_response(self):
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "seed": 3,
                                    "lora": [{"name": "Strong Style", "scale": 0.75}]}, self.cfg)
        word = f" LORA:0.75:{loras.encode_path(str(self.d / 'Strong Style.safetensors'))}"
        need = self.gen.need(job)
        self.assertTrue(self.gen.engine.lines[-1].endswith(word), self.gen.engine.lines[-1])
        self.assertEqual((need["lora_n"], need["lora_mib"]), (2, 8))
        made, figures = self.gen.run(job)
        self.assertEqual(len(made), 1)
        img = self.gen.engine.lines[-1]
        self.assertTrue(img.startswith("IMG 256 256 3 14 "), img)
        self.assertTrue(img.endswith(word), img)
        self.assertEqual(loras.encode_path(str(self.d / "Strong Style.safetensors")).replace("%20", " "),
                         str(self.d / "Strong Style.safetensors"), "the path itself, its separators %XX'd")
        self.assertEqual((figures["lora_n"], figures["lora_mib"]), (2, 8))
        body = self.gen.response(made, job, figures, "http://x")
        self.assertEqual(body["data"][0]["lora"], [{"name": "Strong Style", "scale": 0.75}])
        self.assertEqual(body["xestrata"]["lora_n"], 2)
        job = images.parse_request({"prompt": "a cat", "size": "256x256", "seed": 3}, self.cfg)
        made, figures = self.gen.run(job)
        self.assertNotIn("lora", self.gen.response(made, job, figures, "http://x")["data"][0])
        self.assertNotIn("lora_n", figures, "no adapter: the figures are as they were")
        self.assertNotIn(" LORA:", self.gen.engine.lines[-1])

    def test_the_listing_is_the_models_capabilities(self):
        d = self.gen.defaults()
        self.assertEqual([e["name"] for e in d["loras"]], ["Strong Style", "style/gouache"])
        self.assertEqual(images.model_defaults(self.cfg).get("loras"), None, "the pure defaults carry no folder read")


APP = Path(__file__).resolve().parent / "web" / "app.js"


def slice_between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


def entry(name: str, scale: float = 1.0, trigger: str | None = None, **kw) -> dict:
    """One adapter of a model's listing (/health's image_models[].defaults.loras, serve/loras.py)."""
    return {"name": name, "path": f"{name}.safetensors", "rank": 16, "scale": scale, "trigger": trigger,
            "targets": ["attn.to_q"], "size_mib": 12.5, **kw}


# The page's own code runs in node against a small fake page: the elements the Images tab's adapters touch, the
# browser's localStorage (the store the page keeps xestrata.* in) and the model of the moment with its listing.
HARNESS = r"""
const state = __STATE__;
const els = new Map();
class El {
  constructor(tag) {
    this.tag = tag; this.hidden = false; this.value = ""; this.disabled = false; this.textContent = "";
    this.children = []; this.dataset = {}; this.attrs = {}; this.className = ""; this.title = "";
  }
  append(...c) { this.children.push(...c); }
  replaceChildren(...c) { this.children = [...c]; }
  setAttribute(k, v) { this.attrs[k] = v; }
  focus() { this.focused = true; }
}
const $ = (id) => { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); };
class Option { constructor(text, value) { this.text = text; this.value = value; } }
const data = new Map(Object.entries(state.stored || {}));
const store = {
  get(k, d) { const v = data.get("xestrata." + k); return v === undefined ? d : JSON.parse(v); },
  set(k, v) { data.set("xestrata." + k, JSON.stringify(v)); return true; },
};
const document = {createElement: (tag) => new El(tag)};
const imgModel = () => ({id: state.model, defaults: {loras: state.loras}});
const tr = (key, english, vars) => String(english == null ? key : english)
  .replace(/\{(\w+)\}/g, (m, k) => (vars && k in vars ? String(vars[k]) : m));
const chips = () => $("img-lora-chips").children.map(c => {
  const name = c.children.find(x => x.className === "img-lora-chip__name");
  const scale = c.children.find(x => x.type === "number");
  const buttons = c.children.filter(x => x.type === "button");
  return {name: name && name.textContent, title: name && name.title, scale: scale && scale.value,
          trigger: buttons.some(b => b.textContent === "Trigger"),
          remove: buttons.some(b => b.textContent === "\u00d7")};
});
const picker = () => $("img-lora-add").children.map(o => ({text: o.text, value: o.value}));
const stored = () => Object.fromEntries([...data.entries()]);
const scaleInput = (i) => $("img-lora-chips").children[i].children.find(x => x.type === "number");
const trigger = (i) => $("img-lora-chips").children[i].children
  .find(x => x.type === "button" && x.textContent === "Trigger");
const choose = (name) => { $("img-lora-add").value = name; $("img-lora-add").onchange(); };
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebPage(unittest.TestCase):
    """The Images tab's adapters in serve/web/app.js: the block the model's listing fills, the picker, a chip per
    chosen adapter (its strength, its trigger words, its removal), the count's limit and what the page stores and
    sends."""

    code: str

    @classmethod
    def setUpClass(cls):
        cls.code = slice_between(APP.read_text(encoding="utf-8"), "// the adapters this image model applies",
                                 "// the count: up to max_n")

    def run_js(self, body: str, loras: list[dict], model: str = "qwen-image-2.1-turbo",
               stored: dict | None = None) -> dict:
        page = {"model": model, "loras": loras, "stored": stored or {}}
        js = HARNESS.replace("__STATE__", json.dumps(page)) + "\n" + self.code + "\n" + body
        r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_a_model_that_takes_no_adapter_hides_the_block(self):
        out = self.run_js('imgLoras();\nconsole.log(JSON.stringify({hidden: $("img-lora").hidden, chips: chips(),'
                          ' body: imgLorasBody()}));', loras=[], model="mage-flow-turbo")
        self.assertEqual(out, {"hidden": True, "chips": [], "body": []})

    def test_the_picker_lists_the_models_adapters_and_a_chosen_one_becomes_a_chip(self):
        out = self.run_js("""
imgLoras();
const before = {hidden: $("img-lora").hidden, picker: picker(), chips: chips()};
choose("style/gouache");
console.log(JSON.stringify({before, after: chips(), picker: picker(), hint: $("img-lora-hint").hidden,
                            stored: stored(), body: imgLorasBody(), title: chips()[0].title}));
""", loras=[entry("style/gouache", 0.8, "gouache style"), entry("Strong Style", 1.0)])
        self.assertFalse(out["before"]["hidden"])
        self.assertEqual([o["value"] for o in out["before"]["picker"]], ["", "style/gouache", "Strong Style"])
        self.assertEqual(out["before"]["chips"], [])
        # the recommended strength of the listing is the chip's, the trigger words have their button, and the chosen
        # name leaves the picker; the page keeps the choice for this model
        self.assertEqual(out["after"], [{"name": "style/gouache", "title": "rank 16, 12.5 MiB · attn.to_q",
                                         "scale": "0.8", "trigger": True, "remove": True}])
        self.assertEqual([o["value"] for o in out["picker"]], ["", "Strong Style"])
        self.assertTrue(out["hint"], "under the limit: no hint")
        self.assertEqual(out["stored"], {"xestrata.img_loras.qwen-image-2.1-turbo":
                                         '[{"name":"style/gouache","scale":0.8}]'})
        self.assertEqual(out["body"], [{"name": "style/gouache", "scale": 0.8}])

    def test_a_stored_choice_comes_back_without_the_names_the_model_lost(self):
        stored = {"xestrata.img_loras.qwen-image-2.1-turbo": json.dumps(
            [{"name": "gone", "scale": 1.0}, {"name": "Strong Style", "scale": -1.5}])}
        out = self.run_js('imgLoras();\nconsole.log(JSON.stringify({chips: chips(), picker: picker(),'
                          ' body: imgLorasBody()}));', loras=[entry("Strong Style")], stored=stored)
        self.assertEqual(out["chips"], [{"name": "Strong Style", "title": "rank 16, 12.5 MiB · attn.to_q",
                                         "scale": "-1.5", "trigger": False, "remove": True}])
        self.assertEqual([o["value"] for o in out["picker"]], [""], "a chosen adapter leaves the picker")
        self.assertEqual(out["body"], [{"name": "Strong Style", "scale": -1.5}])

    def test_a_strength_is_the_apis_range_two_decimals_and_the_listings_when_unreadable(self):
        out = self.run_js("""
imgLoras();
choose("style/gouache");
const seen = [];
for (const v of ["7", "-9", "", "abc", "0.333", "1.5"]) {
  scaleInput(0).value = v;
  scaleInput(0).onchange();
  seen.push(scaleInput(0).value);
}
console.log(JSON.stringify({seen, body: imgLorasBody()}));
""", loras=[entry("style/gouache", 0.8)])
        self.assertEqual(out["seen"], ["2", "-2", "0.8", "0.8", "0.33", "1.5"])
        self.assertEqual(out["body"], [{"name": "style/gouache", "scale": 1.5}])

    def test_four_at_most_and_the_hint(self):
        loras = [entry(f"loRA{i}", 1.0) for i in range(5)]
        out = self.run_js("""
imgLoras();
for (const e of ["loRA0", "loRA1", "loRA2", "loRA3", "loRA4"]) choose(e);
console.log(JSON.stringify({chips: chips().map(c => c.name), disabled: $("img-lora-add").disabled,
                            hint: {text: $("img-lora-hint").textContent, hidden: $("img-lora-hint").hidden},
                            count: imgLorasBody().length}));
""", loras=loras)
        self.assertEqual(out["chips"], ["loRA0", "loRA1", "loRA2", "loRA3"], "the fifth is not taken")
        self.assertTrue(out["disabled"], "the picker is off at the limit")
        self.assertEqual(out["hint"], {"text": "Up to 4 LoRAs at once (the API's limit)", "hidden": False})
        self.assertEqual(out["count"], 4)

    def test_the_trigger_words_go_into_the_prompt_once(self):
        out = self.run_js("""
imgLoras();
choose("style/gouache");
trigger(0).onclick();
const once = $("img-prompt").value;
trigger(0).onclick();
const twice = $("img-prompt").value;
$("img-prompt").value = "a fox in the snow,";
trigger(0).onclick();
console.log(JSON.stringify({once, twice, after: $("img-prompt").value, focused: $("img-prompt").focused === true}));
""", loras=[entry("style/gouache", 0.8, "gouache style")])
        self.assertEqual(out["once"], "gouache style")
        self.assertEqual(out["twice"], "gouache style", "said once")
        self.assertEqual(out["after"], "a fox in the snow, gouache style")
        self.assertTrue(out["focused"])

    def test_each_model_comes_back_with_its_own_choice(self):
        out = self.run_js("""
imgLoras();
choose("style/gouache");
state.model = "mage-flow-turbo";
state.loras = [];
imgLoras();
const other = {hidden: $("img-lora").hidden, chips: chips(), body: imgLorasBody()};
state.model = "qwen-image-2.1-turbo";
state.loras = LORAS;
imgLoras();
console.log(JSON.stringify({other, back: chips().map(c => [c.name, c.scale]), stored: stored()}));
""".replace("LORAS", json.dumps([entry("style/gouache", 0.8)])),
                             loras=[entry("style/gouache", 0.8)])
        self.assertEqual(out["other"], {"hidden": True, "chips": [], "body": []})
        self.assertEqual(out["back"], [["style/gouache", "0.8"]])
        self.assertEqual(list(out["stored"]), ["xestrata.img_loras.qwen-image-2.1-turbo"])

    def test_removing_a_chip_puts_its_name_back_in_the_picker(self):
        out = self.run_js("""
imgLoras();
choose("style/gouache");
const rm = $("img-lora-chips").children[0].children.find(x => x.textContent === "\u00d7");
rm.onclick();
console.log(JSON.stringify({chips: chips(), picker: picker(), stored: stored()}));
""", loras=[entry("style/gouache", 0.8)])
        self.assertEqual(out["chips"], [])
        self.assertEqual([o["value"] for o in out["picker"]], ["", "style/gouache"])
        self.assertEqual(out["stored"], {"xestrata.img_loras.qwen-image-2.1-turbo": "[]"})


def _table(header: str, name: str) -> list[str]:
    """A C++ table's names, in order (include/strata/image/lora_names.hpp's `inline constexpr const char* kName[]`)."""
    at = header.index(name)
    body = header[header.index("{", at): header.index("};", at)]
    return re.findall(r'"([^"]*)"', body)


class TargetNames(unittest.TestCase):
    """serve/loras.py's TARGETS are the DiTs' build_lora_targets names (include/strata/image/lora_names.hpp's tables):
    a file's key the engine takes is one this model may carry, whichever of the two sides names it."""

    def test_every_models_targets_are_the_names_the_dits_build(self):
        header = (Path(__file__).resolve().parents[1] / "include/strata/image/lora_names.hpp").read_text(
            encoding="utf-8")

        def block(path: str, names: list[str]) -> set[str]:
            return {path + n for n in names}

        def tables(*names: str) -> list[str]:
            out: list[str] = []
            for n in names:
                out += _table(header, n)
            return out

        qi_outside = set(_table(header, "kQiOutside"))
        qi_block = block("transformer_blocks.{b}.", tables("kQiBlock", "kQiGateHalf"))
        moss_outside = set(_table(header, "kMossOutside"))
        moss_text = block("text_embedder.blocks.{b}.", tables("kMossTextBlock", "kMossTextUpGate"))
        moss_block = block("layers.{b}.", set(tables("kMossBlock", "kMossQkv", "kMossUpGate")))
        mage_outside = set(_table(header, "kMageOutside"))
        mage_block = block("transformer_blocks.{b}.", tables("kMageMod", "kMageBlock"))
        anima_outside = set(_table(header, "kAnimaOutside"))
        anima_block = block("blocks.{b}.", tables("kAnimaBlock"))
        anima_mod = block("blocks.{b}.", [f"{n}.{j}" for n in _table(header, "kAnimaMod") for j in (1, 2)])
        anima_adapter = block("llm_adapter.blocks.{b}.", tables("kAnimaAdapter"))
        expected = {
            "qwen-image": qi_outside | qi_block,
            "moss-image": moss_outside | moss_text | moss_block,
            "mage-flow": mage_outside | mage_block,
            "anima": anima_outside | anima_block | anima_mod | anima_adapter,
        }
        for family, names in expected.items():
            self.assertEqual(sorted(names), sorted(set(loras.TARGETS[family])), family)


if __name__ == "__main__":
    unittest.main()
