"""Tests for nvfp4_verify_gguf.py's PLE check: it reads the table types the engine reads (the FP8 table
ple_fp8_pack.py writes and the block types such as Q8_0), not only IQ4_NL, and still refuses a different table.
A tiny checkpoint and tiny GGUFs in a temporary folder - nothing is downloaded.

    python -m unittest terminal.tools.test_nvfp4_verify_gguf_ple
"""
from __future__ import annotations

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch and safetensors are not installed here (the checkpoint is built with them)")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvfp4_verify_gguf as V  # noqa: E402
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants  # noqa: E402

ROWS, WIDTH, SCALE = 64, 160, 0.0002
NAME = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def checkpoint(folder: Path, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    fp8 = (torch.randn(ROWS, WIDTH, generator=g) * 100).to(torch.float8_e4m3fn)
    save_file({NAME: fp8, NAME.replace(".weight", ".weight_scale"): torch.tensor(SCALE, dtype=torch.bfloat16)},
              str(folder / "model.safetensors"))
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        NAME: "model.safetensors", NAME.replace(".weight", ".weight_scale"): "model.safetensors"}}))
    return fp8


def ple_gguf(path: Path, fp8: torch.Tensor, kind: str) -> None:
    w = GGUFWriter(str(path), "ple")
    if kind == "fp8":
        w.add_string("strata.ple.format", "f8_e4m3")
        w.add_float32("strata.ple.scale", float(torch.tensor(SCALE, dtype=torch.bfloat16)))
        data = fp8.view(torch.int8).numpy()
        w.add_tensor("per_layer_token_embd.weight", data, raw_dtype=Q.I8)
    else:
        values = fp8.float().numpy() * SCALE
        w.add_tensor("per_layer_token_embd.weight", quants.quantize(values, Q.Q8_0), raw_dtype=Q.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


class PleCheck(unittest.TestCase):
    def run_check(self, kind: str, other_table: bool = False) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fp8 = checkpoint(d, 1)
            if other_table:
                other = d / "other"; other.mkdir(); fp8 = checkpoint(other, 2)
            ple_gguf(d / "ple.gguf", fp8, kind)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bad = V.check_ple(d, d / "ple.gguf", 16, random.Random(0))
            return bad, out.getvalue()

    def test_fp8_table_is_read_and_matches(self):
        bad, out = self.run_check("fp8")
        self.assertEqual(bad, 0, out)
        self.assertIn("cosine mean 1.0000", out)

    def test_block_type_table_is_read(self):
        bad, out = self.run_check("q8_0")
        self.assertEqual(bad, 0, out)

    def test_different_table_is_refused(self):
        bad, out = self.run_check("fp8", other_table=True)
        self.assertEqual(bad, 1, out)


if __name__ == "__main__":
    unittest.main()
