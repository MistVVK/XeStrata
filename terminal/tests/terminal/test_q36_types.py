"""#5869: "use my own files" (and the app's local-GGUF import: `strata models import PATH --plan`, the same shared importer) accepts a
Qwen3.6-35B-A3B (qwen35moe) file the Qwen3.6 engine loads and refuses one it does not, with the existing words; qwen4exp is unchanged.

The accepted types mirror the engine loader's three checks (strata_control/q36_types.py); these tests keep the mirror equal to the CUDA
source text, check that every tensor name the loader reads is classified, and that the catalog's three published sets (their real tensor
type tables, read from the GGUF headers at the pinned revision: tests/data/q36_types/*.types.json) are all accepted."""

from __future__ import annotations

import io
import json
import re
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from strata_control import models, q36_types
from tests.terminal.test_own_files import gguf_bytes, string

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted((ROOT / "tests" / "data" / "q36_types").glob("*.types.json"))
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 2: (32, 18), 3: (32, 20),
         6: (32, 22), 7: (32, 24), 20: (32, 18), 23: (256, 136), 11: (256, 110), 10: (256, 84)}


def table(fixture: Path) -> dict[str, int]:
    return {n: v["type"] for n, v in json.loads(fixture.read_text(encoding="utf-8"))["tensors"].items()}


def numbers(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"X\((\d+)\)", text)}


def q36_gguf(types: dict[str, int]) -> bytes:
    """A tiny qwen35moe GGUF with these tensor names and types (one block each): what the importer reads."""
    pairs = [("general.architecture", 8, string("qwen35moe")), ("tokenizer.ggml.model", 8, string("gpt2")),
             ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")),
             ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b"))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(types), len(pairs)) + b"".join(string(k) + struct.pack("<I", kind) + v for k, kind, v in pairs)
    offset = 0
    for name, t in types.items():
        elements, size = BLOCK[t]
        head += string(name) + struct.pack("<IQIQ", 1, elements, t, offset)
        offset += -(-size // 32) * 32
    head += b"\x00" * (-len(head) % 32)
    return head + b"\x00" * offset


class MirrorEqualsTheEngine(unittest.TestCase):
    def test_the_type_sets_are_the_cuda_sources(self):
        iq = (ROOT.parent / "engine" / "src" / "kernels" / "cuda" / "iq_kernels.cu").read_text(encoding="utf-8", errors="replace")
        # each list is defined twice: first under #ifdef STRATA_Q6K_EXPERTS (the release build), then without the Q6_K kernels
        gu = re.findall(r"^#define STRATA_GU_FMTS\(X\)(.*)$", iq, re.M)
        dn = re.findall(r"^#define STRATA_D_FMTS\(X\)(.*)$", iq, re.M)
        self.assertEqual((len(gu), len(dn)), (2, 2))
        self.assertEqual(numbers(gu[0]), set(q36_types.EXPERT_GATE_UP))
        self.assertEqual(numbers(dn[0]), set(q36_types.EXPERT_DOWN))
        self.assertLess(iq.index("#ifdef STRATA_Q6K_EXPERTS"), iq.index("#define STRATA_GU_FMTS"))
        mm = (ROOT.parent / "engine" / "src" / "kernels" / "cuda" / "native_mmvq.cu").read_text(encoding="utf-8", errors="replace")
        body = mm[mm.index("bool native_mmvq_supported(int ggml_type) noexcept {"):]
        body = body[:body.index("}")]
        self.assertEqual({int(x) for x in re.findall(r"ggml_type == (\d+)", body)}, set(q36_types.MATRIX))
        eng = (ROOT.parent / "engine" / "src" / "qwen35" / "q36_engine.cu").read_text(encoding="utf-8", errors="replace")
        up = eng[eng.index("float* upload_f32("):]
        up = up[:up.index("std::exit(1)")]
        self.assertEqual({int(x) for x in re.findall(r"t\.type == (\d+)", up)}, set(q36_types.VECTOR))
        self.assertIn('-DSTRATA_Q6K_EXPERTS=ON', (ROOT.parent / "tools" / "packaging" / "build_cuda_linux.sh").read_text(encoding="utf-8"))

    def test_every_name_the_loader_reads_is_classified(self):
        eng = (ROOT.parent / "engine" / "src" / "qwen35" / "q36_engine.cu").read_text(encoding="utf-8", errors="replace")
        for fn, kind in (("f32", "vector"), ("mat", "matrix")):
            for name in re.findall(fn + r'\(L\(l, "([^"]+)"\)\)', eng):
                self.assertEqual(q36_types.kind_of("blk.7." + name), kind, name)
        self.assertEqual(q36_types.kind_of("output.weight"), "matrix")
        self.assertEqual(q36_types.kind_of("output_norm.weight"), "vector")
        self.assertEqual(q36_types.kind_of("token_embd.weight"), "embedding")

    def test_the_catalog_sizes_are_setups(self):
        import setup
        fam = setup.OWN_PATH_FAMILIES[q36_types.FAMILY]
        self.assertEqual(tuple(fam["models"]), q36_types.SIZES)
        self.assertEqual(fam["file"], q36_types.FILE)
        self.assertEqual(fam["arch"], "qwen35moe")


class CatalogSets(unittest.TestCase):
    def test_every_published_set_is_accepted(self):
        self.assertEqual([f.name.split(".types")[0] for f in FIXTURES], sorted(q36_types.FILE.format(q=q)[:-5] for q in q36_types.SIZES))
        for f in FIXTURES:
            types = table(f)
            self.assertEqual(q36_types.problems(types), [], f.name)
            self.assertFalse([n for n in types if q36_types.kind_of(n) == "other"], f.name)

    def test_a_type_the_engine_cannot_run_is_refused(self):
        types = table(FIXTURES[-1])
        down = next(n for n in types if n.endswith("ffn_down_exps.weight"))
        self.assertTrue(q36_types.problems({**types, down: 12}))             # Q4_K is no down type of the expert kernels
        norm = next(n for n in types if n.endswith("attn_norm.weight"))
        self.assertTrue(q36_types.problems({**types, norm: 12}))             # a norm the f32 loader cannot widen
        gate = next(n for n in types if n.endswith("ffn_gate_exps.weight"))
        self.assertTrue(q36_types.problems({**types, gate: 13}))             # gate and up must be one type


class Importer(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-q36-import-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def small_table(self, **changes) -> dict[str, int]:
        """The UD-Q4_K_XL set's names and types, a few layers of it (enough for each class), with `changes` applied."""
        full = table(next(f for f in FIXTURES if "UD-Q4_K_XL" in f.name))
        keep = {n: t for n, t in full.items() if not n.startswith("blk.") or n.split(".")[1] in ("0", "3")}
        return {**keep, **changes}

    def put(self, name: str, types: dict[str, int]) -> Path:
        (self.dir / name).write_bytes(q36_gguf(types))
        return self.dir

    def test_a_published_qwen36_file_gives_a_plan_for_family_qwen36(self):
        plan = models.plan_import(str(self.put("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", self.small_table())))
        self.assertEqual(plan["issues"], [])
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("qwen36", "UD-Q4_K_XL"))
        self.assertIn("--gguf-dir", plan["candidate_selection"]["setup_args_hint"])

    def test_a_renamed_qwen36_file_is_taken_as_the_closest_size_and_said(self):
        plan = models.plan_import(str(self.put("my-qwen36.gguf", self.small_table())))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "UD-Q4_K_XL")
        self.assertTrue(any("not exactly a build Strata has tested" in m for m in plan["messages"]))

    def test_an_unsupported_quant_is_refused_with_the_existing_words(self):
        types = self.small_table()
        down = next(n for n in types if n.endswith("ffn_down_exps.weight"))
        plan = models.plan_import(str(self.put("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", {**types, down: 12})))
        self.assertIsNone(plan["candidate_selection"])
        self.assertIn("unsupported_or_missing_architecture", plan["issues"])
        self.assertIn(models.UNSUPPORTED_MODEL_TYPE, plan["messages"])

    def test_the_cli_the_app_calls_gives_the_same_plan(self):
        from strata_cli.cli import main
        folder = self.put("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", self.small_table())
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["models", "import", str(folder), "--plan", "--output", "json"])
        answer = json.loads(out.getvalue())
        plan = answer.get("plan", answer)
        self.assertEqual(code, 0, out.getvalue()[:500])
        self.assertEqual(plan["candidate_selection"]["family"], "qwen36")

    def test_qwen4exp_is_unchanged(self):
        (self.dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf").write_bytes(gguf_bytes(["IQ3_S"]))
        (self.dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf").write_bytes(gguf_bytes(["IQ3_S"]))
        plan = models.plan_import(str(self.dir))
        self.assertNotEqual((plan["candidate_selection"] or {}).get("family"), "qwen36")


class SetupTakesTheOwnFile(unittest.TestCase):
    def test_the_one_model_file_of_the_folder_under_another_name(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "mmproj-F16.gguf").write_bytes(b"x")
            self.assertIsNone(setup.own_path_gguf_in(folder))
            (folder / "my-qwen36.gguf").write_bytes(b"x")
            self.assertEqual(setup.own_path_gguf_in(folder), folder / "my-qwen36.gguf")
            (folder / "another.gguf").write_bytes(b"x")
            self.assertIsNone(setup.own_path_gguf_in(folder))                # two model files: not guessed


if __name__ == "__main__":
    unittest.main()
