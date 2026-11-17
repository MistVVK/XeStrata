"""tools/test_strata_gguf.py - Strata's own GGUF reader/writer and quantizers (tools/strata_gguf.py, tools/strata_quants.py).

    python tools/test_strata_gguf.py [-v]            run the tests
    python tools/test_strata_gguf.py --freeze        rewrite tools/strata_gguf_digests.json from gguf-py's outputs (needs gguf-py)

What is proven:
  * tools/strata_quant_tables.py is exactly what tools/gen_strata_quant_tables.py makes from the GGUF reference tables;
  * the type table agrees with tools/gguf_reader.py and (when it is available) with gguf-py;
  * with llama.cpp's gguf-py (third_party/strata-gguf-py, or $STRATA_GGUF_PY): every dequantizer and quantizer gives bit-identical results
    on random and edge-case blocks, the writer produces byte-identical files, and the reader gives the same fields and tensors;
  * without gguf-py: the same inputs give the digests recorded from gguf-py (tools/strata_gguf_digests.json), so the exactness stays checked
    after the vendored copy is gone.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import sys
import contextlib
import gc
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gen_strata_quant_tables as GEN  # noqa: E402
import gguf_reader as GR  # noqa: E402
import strata_gguf as SG  # noqa: E402
import strata_quants as SQ  # noqa: E402

DIGESTS = HERE / "strata_gguf_digests.json"

#: byte offsets of the fp16 fields of each block type (set to finite values so the comparison is not about NaN payloads)
FP16_AT = {"Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,), "Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2),
           "Q5_K": (0, 2), "Q6_K": (208,), "TQ1_0": (52,), "TQ2_0": (64,), "IQ2_XXS": (0,), "IQ2_XS": (0,), "IQ2_S": (0,),
           "IQ3_XXS": (0,), "IQ3_S": (0,), "IQ1_S": (0,), "IQ4_NL": (0,), "IQ4_XS": (0,), "Q2_0": (0,), "MXFP4": (), "NVFP4": (), "IQ1_M": ()}
DEQ_TYPES = sorted(FP16_AT)
QUANT_TYPES = ("BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1")


def load_gguf_py():
    for c in (os.environ.get("STRATA_GGUF_PY"), ROOT.parent / "engine" / "third_party" / "strata-gguf-py"):
        if c and (Path(c) / "gguf" / "__init__.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return importlib.import_module("gguf")
    return None


GGUF = load_gguf_py()


def blocks_for(name: str, seed: int, n: int = 48) -> np.ndarray:
    """n random blocks of the type with finite fp16 scales, plus all-zero, all-0xFF and all-0x55 blocks"""
    bs, ts = SQ.BLOCK[name]
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n, ts), dtype=np.uint8)
    b[0] = 0
    b[1] = 0xFF
    b[2] = 0x55
    for off in FP16_AT[name]:
        h = rng.normal(0, 0.7, size=(n,)).astype(np.float16)
        h[3] = np.float16(0)
        h[4] = np.float16(6e-5)         # smallest normal
        h[5] = np.float16(1e-7)         # subnormal
        h[6] = np.float16(-2.5)
        b[:, off:off + 2] = h.view(np.uint8).reshape(n, 2)
    if name == "IQ1_M":                 # the fp16 scale is spread over the top nibbles of four uint16 words at bytes 48..55
        h = rng.normal(0, 0.7, size=(n,)).astype(np.float16).view(np.uint16)
        for k in range(4):
            w = b[:, 48 + 2 * k:50 + 2 * k].copy().view(np.uint16).reshape(n)
            w = (w & np.uint16(0x0FFF)) | (((h >> np.uint16(4 * k)) & np.uint16(0xF)) << np.uint16(12))
            b[:, 48 + 2 * k:50 + 2 * k] = w.astype("<u2").view(np.uint8).reshape(n, 2)
    if name == "MXFP4":
        b[3:, 0] = rng.integers(100, 140, size=(n - 3,), dtype=np.uint8)     # e8m0 scales around 1
    if name == "NVFP4":
        b[3:, :4] = rng.integers(1, 0x7E, size=(n - 3, 4), dtype=np.uint8)   # ue4m3 scales, 0 and 0x7F are the special ones (rows 0..2)
    return b


def floats_for(seed: int, cols: int = 64, rows: int = 40) -> np.ndarray:
    """float32 rows with many scales, zero rows, constant rows, exact halves, negative zero"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(rows, cols)).astype(np.float32) * (10.0 ** rng.integers(-4, 4, size=(rows, 1))).astype(np.float32)
    x[0] = 0
    x[1] = 1.5
    x[2] = -np.float32(0.0)
    x[3] = np.float32(np.arange(cols) % 5) / np.float32(2)               # exact halves: the rounding ties
    x[4] = -x[3]
    x[5, :cols // 2] = 0
    x[6] = np.float32(1e-30)
    return x


def special_floats() -> np.ndarray:
    v = np.array([0.0, -0.0, 1.0, -1.0, 1.0000001, 65504.0, 3.4e38, -3.4e38, 1e-40, -1e-40, np.inf, -np.inf, np.nan, 0.15625, 3.0517578e-05,
                  1.00390625, 1.00585938, 1.0078125, 2.5, 3.5], dtype=np.float32)
    return v.reshape(1, -1)


@contextlib.contextmanager
def tmpdir():
    """a temporary folder; the readers' memmaps are dropped first (Windows cannot delete a mapped file)"""
    d = tempfile.mkdtemp(prefix="strata_gguf_")
    try:
        yield d
    finally:
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def outputs(lib) -> dict:
    """every deterministic output the tests compare, as sha256, computed with `lib` (a gguf-py module or strata_gguf)"""
    Q = lib.GGMLQuantizationType
    out = {}
    for name in DEQ_TYPES:
        if lib is GGUF and name == "Q2_0":
            continue                      # gguf-py has no dequantizer for Q2_0
        for seed in (1, 2, 3):
            b = blocks_for(name, seed)
            out["deq/%s/%d" % (name, seed)] = sha(lib.quants.dequantize(b.reshape(4, -1), Q[name]).astype(np.float32).view(np.uint32))
    for name in QUANT_TYPES:
        bs = SQ.BLOCK[name][0]
        for seed in (1, 2):
            x = floats_for(seed, cols=64)
            out["q/%s/%d" % (name, seed)] = sha(lib.quants.quantize(x, Q[name]))
        out["q/%s/special" % name] = sha(lib.quants.quantize(np.repeat(special_floats(), 2, axis=1)[:, :32].copy().reshape(1, 32), Q[name]))
    return out


class Tables(unittest.TestCase):
    def test_generated_tables_are_current(self):
        self.assertEqual((HERE / "strata_quant_tables.py").read_text(encoding="utf-8"), GEN.render(),
                         "tools/strata_quant_tables.py is stale: python tools/gen_strata_quant_tables.py")

    def test_the_type_tables_agree(self):
        for t, (bs, ts) in SG.GGML_QUANT_SIZES.items():
            if t.name in SQ.BLOCK:
                self.assertEqual(SQ.BLOCK[t.name], (bs, ts), t.name)
            if t.name in GR.BLOCK_GEOMETRY:
                self.assertEqual(GR.BLOCK_GEOMETRY[t.name], (bs, ts), t.name)
        for tid, nm in GR.GGML_TYPES.items():
            if nm in SG.GGMLQuantizationType.__members__ and tid not in (41,):
                self.assertEqual(int(SG.GGMLQuantizationType[nm]), tid, nm)

    def test_the_type_table_matches_gguf_py(self):
        if GGUF is None:
            self.skipTest("no gguf-py")
        for t, v in GGUF.GGML_QUANT_SIZES.items():
            self.assertEqual(SG.GGML_QUANT_SIZES[SG.GGMLQuantizationType(int(t))], v, t.name)
            self.assertEqual(SG.GGMLQuantizationType[t.name].value, t.value)


class Oracle(unittest.TestCase):
    """bit-identical to gguf-py"""

    def run(self, result=None):
        with np.errstate(all="ignore"):         # the all-0xFF blocks overflow to inf in both implementations
            return super().run(result)

    def setUp(self):
        if GGUF is None:
            self.skipTest("no gguf-py (third_party/strata-gguf-py or STRATA_GGUF_PY)")

    def test_dequantize_is_bit_identical(self):
        for name in DEQ_TYPES:
            if name == "Q2_0":
                continue
            for seed in (1, 2, 3, 4):
                b = blocks_for(name, seed).reshape(4, -1)
                want = GGUF.quants.dequantize(b, GGUF.GGMLQuantizationType[name]).astype(np.float32)
                got = SG.quants.dequantize(b, SG.GGMLQuantizationType[name])
                self.assertEqual(got.shape, want.shape, name)
                self.assertEqual(got.dtype, np.float32)
                self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)), "%s seed %d" % (name, seed))

    def test_dequantize_handles_chunking_and_shapes(self):
        b = blocks_for("Q4_K", 9, n=300)
        want = GGUF.quants.dequantize(b.reshape(3, 100 * 144), GGUF.GGMLQuantizationType.Q4_K)
        old = SQ._CHUNK_ELEMS
        SQ._CHUNK_ELEMS = 256 * 7
        try:
            got = SG.quants.dequantize(b.reshape(3, 100 * 144), SG.GGMLQuantizationType.Q4_K)
        finally:
            SQ._CHUNK_ELEMS = old
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)))

    def test_float_types_pass_through(self):
        x = np.arange(12, dtype=np.float32)
        for t in ("F32", "F16"):
            raw = x.astype(np.float32 if t == "F32" else np.float16).view(np.uint8)
            self.assertTrue(np.array_equal(SG.quants.dequantize(raw, SG.GGMLQuantizationType[t]), GGUF.quants.dequantize(raw, GGUF.GGMLQuantizationType[t])))

    def test_quantize_is_bit_identical(self):
        for name in QUANT_TYPES:
            for seed in (1, 2, 3, 4):
                x = floats_for(seed, cols=96 if name != "BF16" else 50)
                if name != "BF16":
                    x = x[:, :64] if name in ("Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1") else x
                want = GGUF.quants.quantize(x, GGUF.GGMLQuantizationType[name])
                got = SG.quants.quantize(x, SG.GGMLQuantizationType[name])
                self.assertEqual(got.shape, want.shape, name)
                self.assertTrue(np.array_equal(got, want), "%s seed %d" % (name, seed))

    def test_bf16_specials_and_rounding_ties(self):
        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2 ** 32, size=(8, 4096), dtype=np.uint64).astype(np.uint32)
        bits[0, :8] = [0x3F808000, 0x3F818000, 0x3F800000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001, 0x00000001]   # ties, inf, NaNs, subnormal
        x = bits.view(np.float32)
        want = GGUF.quants.quantize(x, GGUF.GGMLQuantizationType.BF16)
        got = SG.quants.quantize(x, SG.GGMLQuantizationType.BF16)
        self.assertTrue(np.array_equal(got, want))
        self.assertTrue(np.array_equal(SG.quants.quantize(special_floats(), SG.GGMLQuantizationType.BF16), GGUF.quants.quantize(special_floats(), GGUF.GGMLQuantizationType.BF16)))

    def test_quantize_roundtrip_is_the_same(self):
        x = floats_for(8, cols=64)
        for name in QUANT_TYPES:
            a = SG.quants.dequantize(SG.quants.quantize(x, SG.GGMLQuantizationType[name]), SG.GGMLQuantizationType[name])
            b = GGUF.quants.dequantize(GGUF.quants.quantize(x, GGUF.GGMLQuantizationType[name]), GGUF.GGMLQuantizationType[name])
            self.assertTrue(np.array_equal(a.view(np.uint32), b.view(np.uint32)), name)


def write_sample(lib, path, alignment=None):
    Q = lib.GGMLQuantizationType
    V = lib.GGUFValueType
    w = lib.GGUFWriter(path, "qwen4exp")
    if alignment:
        w.add_custom_alignment(alignment)
    w.add_string("general.name", "sample")
    w.add_string("empty.skipped", "")
    w.add_uint32("a.u32", 7)
    w.add_int32("a.i32", -7)
    w.add_uint16("split.no", 1)
    w.add_uint64("a.u64", 2 ** 40)
    w.add_int8("a.i8", -3)
    w.add_float32("a.f32", 0.1)
    w.add_float64("a.f64", 0.1)
    w.add_bool("a.bool", True)
    w.add_array("a.strs", ["x", "yy", "zzzé"])
    w.add_array("a.ints", [1, 2, 3])
    w.add_array("a.floats", [0.5, 1.5])
    w.add_key_value("a.u8s", [1, 2, 255], V.ARRAY, sub_type=V.UINT8)
    w.add_key_value("a.u32s", [1, 2, 4294967295], V.ARRAY, sub_type=V.UINT32)
    w.add_key_value("a.bools", [True, False], V.ARRAY, sub_type=V.BOOL)
    w.add_key_value("a.guess", 3, V.INT32)
    rng = np.random.default_rng(3)
    w.add_tensor("t.f32", rng.normal(size=(3, 5)).astype(np.float32))
    w.add_tensor("t.f16", rng.normal(size=(2, 7)).astype(np.float16))
    w.add_tensor("t.i32", np.arange(6, dtype=np.int32).reshape(2, 3))
    w.add_tensor("t.f64", rng.normal(size=(4,)))
    x = rng.normal(size=(5, 64)).astype(np.float32)
    w.add_tensor("t.q8", lib.quants.quantize(x, Q.Q8_0), raw_dtype=Q.Q8_0)
    w.add_tensor("t.q8raw", np.zeros((3, 68), np.uint8), raw_dtype=Q.Q8_0, raw_shape=[3, 68])
    w.add_tensor("t.bf16", rng.integers(0, 65535, size=(4, 6), dtype=np.uint16), raw_dtype=Q.BF16)
    w.add_tensor("t.q4k3d", rng.integers(0, 255, size=(2, 3, 144), dtype=np.uint8), raw_dtype=Q.Q4_K)
    w.add_tensor("t.q2_0", rng.integers(0, 255, size=(2, 18), dtype=np.uint8), raw_dtype=Q.Q2_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class Container(unittest.TestCase):
    def test_a_file_written_here_reads_back(self):
        with tmpdir() as d:
            p = Path(d) / "a.gguf"
            write_sample(SG, p)
            r = SG.GGUFReader(p)
            self.assertEqual(r.alignment, 32)
            self.assertEqual(r.data_offset % 32, 0)
            self.assertEqual(r.fields["general.architecture"].contents(), "qwen4exp")
            self.assertEqual(r.fields["a.strs"].contents(), ["x", "yy", "zzzé"])
            self.assertEqual(r.fields["a.strs"].contents(1), "yy")
            self.assertEqual(r.fields["a.ints"].types, [SG.GGUFValueType.ARRAY, SG.GGUFValueType.INT32])
            self.assertEqual(r.fields["a.u64"].contents(), 2 ** 40)
            self.assertIs(r.fields["a.bool"].contents(), True)
            self.assertAlmostEqual(r.fields["a.f32"].contents(), 0.1, places=7)
            self.assertNotIn("empty.skipped", r.fields)
            t = {x.name: x for x in r.tensors}
            self.assertEqual(t["t.f32"].data.shape, (3, 5))
            self.assertEqual([int(v) for v in t["t.f32"].shape], [5, 3])
            self.assertEqual(t["t.q8"].data.shape, (5, 2 * 34))
            self.assertEqual(t["t.q8"].n_bytes, 5 * 2 * 34)
            self.assertEqual(t["t.bf16"].data.dtype, np.uint8)
            self.assertEqual(t["t.bf16"].data.shape, (4, 12))
            self.assertEqual(t["t.q4k3d"].data.shape, (2, 3, 144))
            self.assertEqual(t["t.q2_0"].tensor_type, SG.GGMLQuantizationType.Q2_0)
            self.assertEqual(t["t.i32"].data.tolist(), [[0, 1, 2], [3, 4, 5]])
            for x in r.tensors:
                self.assertEqual(x.data_offset % 32, 0)

    def test_malformed_files_are_refused(self):
        with tmpdir() as d:
            p = Path(d) / "a.gguf"
            write_sample(SG, p)
            raw = p.read_bytes()
            for name, data in (("magic", b"XXXX" + raw[4:]), ("version", raw[:4] + struct.pack("<I", 9) + raw[8:]),
                               ("short", raw[:30]), ("cut", raw[:-40]), ("empty", b"GGUF"),
                               ("count", raw[:8] + struct.pack("<Q", 2 ** 62) + raw[16:])):
                q = Path(d) / (name + ".gguf")
                q.write_bytes(data)
                with self.assertRaises((ValueError, KeyError), msg=name):
                    SG.GGUFReader(q)

    def test_the_writer_refuses_what_gguf_py_refuses(self):
        w = SG.GGUFWriter(None, "x")
        w.add_tensor("a", np.zeros(4, np.float32))
        with self.assertRaises(ValueError):
            w.add_tensor("a", np.zeros(4, np.float32))
        with self.assertRaises(ValueError):
            w.add_tensor("b", np.zeros(4, np.uint8))
        with self.assertRaises(ValueError):
            w.add_tensor("c", np.zeros((2, 33), np.uint8), raw_dtype=SG.GGMLQuantizationType.Q8_0)
        with self.assertRaises(NotImplementedError):
            SG.GGUFWriter(None, "x", split_max_tensors=2)

    def test_files_are_byte_identical_to_gguf_py(self):
        if GGUF is None:
            self.skipTest("no gguf-py")
        with tmpdir() as d:
            for align in (None, 64):
                a, b = Path(d) / "ours.gguf", Path(d) / "theirs.gguf"
                write_sample(SG, a, align)
                write_sample(GGUF, b, align)
                self.assertEqual(a.read_bytes(), b.read_bytes(), "alignment %s" % align)

    def test_the_reader_agrees_with_gguf_py(self):
        if GGUF is None:
            self.skipTest("no gguf-py")
        with tmpdir() as d:
            p = Path(d) / "a.gguf"
            write_sample(GGUF, p, 64)
            ours, theirs = SG.GGUFReader(p), GGUF.GGUFReader(p)
            self.assertEqual(ours.alignment, theirs.alignment)
            self.assertEqual(ours.data_offset, theirs.data_offset)
            self.assertEqual(list(ours.fields), list(theirs.fields))
            for k, f in theirs.fields.items():
                g = ours.fields[k]
                self.assertEqual([int(t) for t in g.types], [int(t) for t in f.types], k)
                self.assertEqual(g.contents(), f.contents(), k)
                self.assertEqual(type(g.contents()), type(f.contents()), k)
            self.assertEqual([t.name for t in ours.tensors], [t.name for t in theirs.tensors])
            for a, b in zip(ours.tensors, theirs.tensors):
                self.assertEqual(int(a.tensor_type), int(b.tensor_type))
                self.assertEqual(a.tensor_type.name, b.tensor_type.name)
                self.assertEqual((a.n_elements, a.n_bytes, a.data_offset), (b.n_elements, b.n_bytes, b.data_offset), a.name)
                self.assertEqual([int(v) for v in a.shape], [int(v) for v in b.shape])
                self.assertEqual((a.field.offset, a.field.name), (b.field.offset, b.field.name), a.name)
                self.assertEqual(a.data.shape, b.data.shape, a.name)
                self.assertEqual(a.data.dtype, b.data.dtype, a.name)
                self.assertTrue(np.array_equal(np.asarray(a.data).view(np.uint8), np.asarray(b.data).view(np.uint8)), a.name)

    def test_the_slim_copy_loop_is_the_same(self):
        """tools/nvfp4_local_pack.py cmd_slim: every KV and every tensor of a file copied through reader + writer"""
        with tmpdir() as d:
            src = Path(d) / "src.gguf"
            write_sample(SG, src)
            outs = []
            for lib in [SG] + ([GGUF] if GGUF is not None else []):
                rd = lib.GGUFReader(src)
                w = lib.GGUFWriter(Path(d) / ("o%d.gguf" % len(outs)), rd.fields["general.architecture"].contents())
                for key, f in rd.fields.items():
                    if key.startswith("GGUF.") or key == "general.architecture":
                        continue
                    vt = f.types[0]
                    w.add_key_value(key, f.contents(), vt, sub_type=f.types[-1] if vt == lib.GGUFValueType.ARRAY else None)
                for t in rd.tensors:
                    data = np.asarray(t.data)
                    if t.tensor_type in (lib.GGMLQuantizationType.F32, lib.GGMLQuantizationType.F16):
                        w.add_tensor(t.name, data)
                    else:
                        raw = np.ascontiguousarray(data).view(np.uint8)
                        w.add_tensor(t.name, raw, raw_dtype=t.tensor_type, raw_shape=[int(x) for x in raw.shape])
                w.write_header_to_file()
                w.write_kv_data_to_file()
                w.write_tensors_to_file()
                w.close()
                del rd
                outs.append(Path(d) / ("o%d.gguf" % len(outs)))
            for o in outs[1:]:
                self.assertEqual(outs[0].read_bytes(), o.read_bytes())


class Frozen(unittest.TestCase):
    """the outputs recorded from gguf-py (tools/strata_gguf_digests.json): the check that remains once gguf-py is not in the tree"""

    def run(self, result=None):
        with np.errstate(all="ignore"):
            return super().run(result)

    def test_outputs_match_the_recorded_digests(self):
        want = json.loads(DIGESTS.read_text(encoding="utf-8"))
        got = outputs(SG)
        for k, v in want.items():
            if k in got:
                self.assertEqual(got[k], v, k)
        self.assertEqual(sorted(k for k in want if k not in got), [], "recorded outputs we do not compute")
        self.assertEqual(sorted(k for k in got if k not in want and not k.startswith("deq/Q2_0/")), [], "outputs not recorded: run --freeze")


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        if GGUF is None:
            sys.exit("--freeze needs gguf-py (third_party/strata-gguf-py or STRATA_GGUF_PY)")
        DIGESTS.write_text(json.dumps(outputs(GGUF), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print("wrote", DIGESTS)
        sys.exit(0)
    unittest.main(argv=[a for a in sys.argv if a != "--freeze"])
