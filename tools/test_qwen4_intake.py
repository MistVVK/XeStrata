"""tools/qwen4_intake.py against the headers of every model we have, and against synthetic variants of them.

The headers are recorded from Hugging Face by HTTP range reads (tools/qwen4_intake_fixtures/*.json.gz: metadata without the big
arrays, the tensor directory, the special tokens).  Each test writes them back as header-only GGUF files with
tools/gguf_writer.py's key/value encoder and runs the real reader over them, so the whole path (file -> reader -> report) is the
one a real run takes.  No weights exist anywhere in this test; no network is used unless STRATA_LIVE_HF=1.

The last class reads the C++ and kernel sources and fails when a table in qwen4_intake.py no longer matches them.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import pathlib
import re
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import gguf_reader as G  # noqa: E402
import gguf_writer as W  # noqa: E402
import qwen4_intake as Q  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

FIX = HERE / "qwen4_intake_fixtures"


def load_fixture(name: str) -> dict:
    with gzip.open(FIX / (name + ".json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def _tokens(count: int, special: list) -> tuple[list[str], list[int]]:
    toks: list[str | None] = [None] * count
    types = [1] * count
    for i, text, ty in special:
        toks[i] = text
        types[i] = ty
    chars = iter(sorted(ST.BYTE_TO_UNICODE.values()))
    for i in range(count):
        if toks[i] is None:
            c = next(chars, None)
            toks[i] = c if c is not None else "tok%d" % i
    return toks, types


def _kv(w: W.GGUFWriter, key: str, v) -> bytes:
    if isinstance(v, bool):
        t = "bool"
    elif isinstance(v, int):
        t = "i64"
    elif isinstance(v, float):
        t = "f64"
    elif isinstance(v, str):
        t = "string"
    else:
        raise TypeError(key)
    return w._kv_bytes(key, t, v)


def write_header_only(path: pathlib.Path, metadata: dict, tensors: list, alignment: int = 32) -> None:
    w = W.GGUFWriter()
    kvs = bytearray()
    n = 0
    for key, v in metadata.items():
        if isinstance(v, dict) and "__array__" in v:
            kind = v["__array__"]
            if key == "tokenizer.ggml.tokens":
                toks, _ = _tokens(v["count"], v.get("special", []))
                kvs += w._kv_bytes(key, "array:string", toks)
            elif key == "tokenizer.ggml.token_type":
                kvs += w._kv_bytes(key, "array:i32", v["_values"])
            elif key == "tokenizer.ggml.merges":
                kvs += w._kv_bytes(key, "array:string", [])
            else:
                items = v["items"]
                if items and isinstance(items[0], str):
                    kvs += w._kv_bytes(key, "array:string", items)
                elif items and isinstance(items[0], float):
                    kvs += w._kv_bytes(key, "array:f64", items)
                else:
                    kvs += w._kv_bytes(key, "array:i64", items or [])
        else:
            kvs += _kv(w, key, v)
        n += 1
    body = bytearray(struct.pack("<IIQQ", G.GGUF_MAGIC, 3, len(tensors), n)) + kvs
    for name, shape, tid, off in tensors:
        b = name.encode()
        body += struct.pack("<Q", len(b)) + b + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape)
        body += struct.pack("<IQ", tid, off)
    body += b"\0" * ((-len(body)) % alignment)
    path.write_bytes(bytes(body))


def materialise(fix: dict, out: pathlib.Path, mutate=None) -> list[pathlib.Path]:
    """Write the recorded set as header-only GGUF files into `out`.  `mutate(files)` may change the copied records first."""
    files = copy.deepcopy(fix["files"])
    if mutate:
        mutate(files)
    paths = []
    for f in files:
        md = f["metadata"]
        tt = md.get("tokenizer.ggml.token_type")
        if tt is not None:
            _, types = _tokens(md["tokenizer.ggml.tokens"]["count"], md["tokenizer.ggml.tokens"].get("special", []))
            tt["_values"] = types
        p = out / os.path.basename(f["file"])
        write_header_only(p, md, f["tensors"], f.get("alignment", 32))
        paths.append(p)
    return paths


_TMP = tempfile.TemporaryDirectory(prefix="q4intake-")
_CACHE: dict[tuple, list[pathlib.Path]] = {}


def files_for(name: str, variant: str = "", mutate=None) -> list[pathlib.Path]:
    key = (name, variant)
    if key not in _CACHE:
        d = pathlib.Path(_TMP.name) / ("%s%s" % (name, ("-" + variant) if variant else ""))
        d.mkdir(parents=True, exist_ok=True)
        _CACHE[key] = materialise(load_fixture(name), d, mutate)
    return _CACHE[key]


def analyse(paths, build_tokenizer: bool = False) -> dict:
    infos = [Q.read_local(p) for p in paths]
    rep = Q.run(infos, build_tokenizer=build_tokenizer)
    sets = [s for s in rep["sets"] if "files" in s]
    assert len(sets) == 1, [s.get("files") for s in rep["sets"]]
    return sets[0]


FLASH_NEXT = ["fn_unsloth_ud_q4_k_xl", "fn_unsloth_ud_iq4_xs", "fn_unsloth_ud_q2_k_xl", "fn_unsloth_q8_0", "fn_ista_iq3_s",
              "fn_ista_q2_0", "fn_coder_iq1_m", "fn_swift_iq3_s", "fn_gyro_s"]
QWEN36 = ["q36_ud_q4_k_xl", "q36_q8_0"]


class RealSets(unittest.TestCase):
    def test_every_flash_next_set_is_case_a_on_qwen4exp(self):
        for name in FLASH_NEXT:
            with self.subTest(name):
                r = analyse(files_for(name))
                v = r["verdict"]
                self.assertEqual(r["family"], "qwen4exp")
                self.assertEqual(v["case"], "A", v["blockers"] + [str(v["shape_fields_that_differ"])])
                self.assertEqual(v["blockers"], [])
                self.assertEqual(r["layout"]["ok"], True, r["layout"])
                self.assertEqual(r["layout"]["found"], r["layout"]["wanted"])
                self.assertEqual(r["layout"]["known_unlisted"], 7)     # per_layer_token_embd + the six PLE tensors of one layer
                self.assertIn("cuda", v["backends"])
                self.assertIn("hip", v["backends"])
                self.assertIn("Flash-Next", v["path"])
                self.assertEqual(r["tokenizer"]["pre_rule"], "qwen35")
                self.assertEqual(r["chat_template"]["tool_call_format"], "xml")
                self.assertTrue(r["split"]["complete"], r["split"])

    def test_qwen36_is_case_a_on_qwen35moe_for_cuda_and_hip_only(self):
        for name in QWEN36:
            with self.subTest(name):
                r = analyse(files_for(name))
                v = r["verdict"]
                self.assertEqual(r["family"], "qwen35moe")
                self.assertEqual(v["case"], "A", v["blockers"])
                self.assertEqual(sorted(v["backends"]), ["cuda", "hip"])
                self.assertIn("qwen35", v["path"])
                self.assertEqual(r["layout"]["ok"], True, r["layout"])
                self.assertTrue(r["mtp"]["in_file"])
                self.assertEqual(r["mtp"]["in_file_layers"], 1)
                self.assertEqual(r["qwen35_limit_violations"], [])
                self.assertEqual(r["geometry"]["n_layers"], 40)
                self.assertEqual(r["differs_from_qwen36_35b"], [])

    def test_deepseek_is_case_c_with_its_blockers_named(self):
        r = analyse(files_for("ds41_ud_q8_k_xl"))
        v = r["verdict"]
        self.assertEqual(v["case"], "C")
        text = "\n".join(v["blockers"])
        self.assertIn("deepseek41", text)
        self.assertIn("strata-dsv41", text)
        self.assertIn("joyai-llm", text)           # the pre-tokenizer
        self.assertIn("MXFP4", text)               # a quant type with no kernel in any Qwen path list
        self.assertEqual(v["backends"], [])
        self.assertEqual(len(r["files"]), 9)

    def test_picture_readers(self):
        a = Q.run([Q.read_local(files_for("fn_unsloth_mmproj")[0])], build_tokenizer=False)
        self.assertEqual(a["picture_readers"][0]["projector_type"], "qwen3vl_merger")
        self.assertTrue(a["picture_readers"][0]["supported"])
        self.assertEqual(a["sets"][0]["verdict"]["case"], "-")
        # a model next to its reader: the reader is listed with the set
        infos = [Q.read_local(p) for p in files_for("q36_ud_q4_k_xl") + files_for("q36_mmproj")]
        rep = Q.run(infos, build_tokenizer=False)
        s = rep["sets"][0]
        self.assertEqual([p["projector_type"] for p in s["projectors"]], ["qwen3vl_merger"])
        self.assertTrue(s["projectors"][0]["supported"])

    def test_tokenizer_build_and_template_on_a_real_header(self):
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl"), build_tokenizer=True)
        tk = r["tokenizer"]
        self.assertTrue(tk["builds"]["ok"], tk["builds"])
        self.assertIs(tk["builds"]["roundtrip"], True)
        self.assertEqual(tk["pre"], "qwen35")
        self.assertEqual(tk["n_tokens"], 248320)
        self.assertEqual(tk["key_token_differences"], [])
        self.assertEqual(tk["end_of_turn_id"], 248046)
        self.assertTrue(r["vocab_matches_embedding"])
        ct = r["chat_template"]
        self.assertTrue(ct["known_as"].startswith("Qwen3.8-Flash-Next"))
        self.assertTrue(ct["has_think_tags"])
        self.assertTrue(ct["enable_thinking_switch"])
        r2 = analyse(files_for("fn_swift_iq3_s"))
        self.assertEqual(r2["chat_template"]["known_as"], "Swift 1.5")
        r3 = analyse(files_for("q36_ud_q4_k_xl"))
        self.assertEqual(r3["chat_template"]["known_as"], "Qwen3.6-35B-A3B (Unsloth MTP files)")

    def test_quant_report_names_the_roles_and_backends(self):
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl"))
        q = r["quant"]
        combos = {(c["gate"], c["up"], c["down"]) for c in q["expert_combinations"]}
        self.assertIn(("Q4_K", "Q4_K", "Q5_1"), combos)
        for b in ("cuda", "hip", "sycl", "metal"):
            self.assertIn(b, q["backends"])
        self.assertEqual(q["unknown_types"], [])
        # Gyro's trellis experts: CUDA and HIP run them, the Metal list has them only partly; whatever the lists say, the report agrees
        g = analyse(files_for("fn_gyro_s"))
        self.assertTrue(g["quant"]["backends"]["cuda"]["all_types_supported"])


def _bump(files, key, value):
    for f in files:
        if key in f["metadata"]:
            f["metadata"][key] = value


def reshape_flash_next(files, n_embd=3072, n_layers=64):
    """A Flash-Next of another size: the layout tables of a model with `n_embd` and `n_layers`, the quant types of the original."""
    f0 = files[0]
    md = f0["metadata"]
    md["qwen4exp.embedding_length"] = n_embd
    md["qwen4exp.block_count"] = n_layers
    md["qwen4exp.attention.compress_ratios"] = {"__array__": "list", "count": n_layers,
                                                "items": [4 if l % 4 == 3 else 0 for l in range(n_layers)]}
    g = Q.Geometry(n_embd=n_embd, n_layers=n_layers)
    types = {}
    for f in files:
        for name, shape, tid, off in f["tensors"]:
            types[name] = tid
    tensors = []
    for name, shape in Q.global_tensors(g):
        tensors.append([name, shape, types[name], 0])
    for l in range(n_layers):
        src = l if l < 48 else l - 16
        for name, shape in Q.layer_tensors(g, l):
            tensors.append(["blk.%d.%s" % (l, name), shape, types["blk.%d.%s" % (src, name)], 0])
    for name in types:
        if Q.QWEN4_UNLISTED.match(name):
            tensors.append([name, next(t[1] for f in files for t in f["tensors"] if t[0] == name), types[name], 0])
    md["split.count"] = 1
    md["split.tensors.count"] = len(tensors)
    md.pop("split.no", None)
    files[:] = [f0]
    f0["tensors"] = tensors
    f0["file"] = "reshaped.gguf"


class SyntheticSets(unittest.TestCase):
    def test_reshaped_flash_next_is_case_b_with_the_differing_fields(self):
        r = analyse(files_for("fn_ista_iq3_s", "reshaped", reshape_flash_next))
        v = r["verdict"]
        self.assertEqual(v["case"], "B", v["blockers"])
        self.assertEqual(v["blockers"], [])
        self.assertEqual(r["layout"]["ok"], True, {k: r["layout"][k][:3] for k in ("missing", "reshaped", "unknown")})
        got = {d["key"]: (d["got"], d["kernels_built_for"]) for d in v["shape_fields_that_differ"]}
        self.assertEqual(got, {"qwen4exp.block_count": (64, 48), "qwen4exp.embedding_length": (3072, 2560)})
        self.assertEqual(v["backends"], ["cuda", "hip", "sycl", "metal"][:len(v["backends"])] or v["backends"])
        self.assertTrue(any("geometry header" in n for n in v["notes"]))

    def test_reshaped_with_more_differences_lists_them_all(self):
        def f(files):
            reshape_flash_next(files, 3584, 40)
            md = files[0]["metadata"]
            md["qwen4exp.expert_feed_forward_length"] = 768
            md["qwen4exp.expert_shared_feed_forward_length"] = 768
            g = Q.Geometry(n_embd=3584, n_layers=40, n_ff=768, n_ff_shexp=768)
            t = {x[0]: x[2] for x in files[0]["tensors"]}
            files[0]["tensors"] = [[n, s, t.get(n, 8), 0] for n, s in Q.global_tensors(g)] + \
                [["blk.%d.%s" % (l, n), s, t.get("blk.%d.%s" % (l, n), 8), 0] for l in range(40) for n, s in Q.layer_tensors(g, l)]
        r = analyse(files_for("fn_ista_iq3_s", "reshaped2", f))
        keys = sorted(d["key"] for d in r["verdict"]["shape_fields_that_differ"])
        self.assertEqual(r["verdict"]["case"], "B", r["verdict"]["blockers"])
        self.assertEqual(keys, ["qwen4exp.block_count", "qwen4exp.embedding_length", "qwen4exp.expert_feed_forward_length",
                                "qwen4exp.expert_shared_feed_forward_length"])

    def test_an_unknown_tensor_is_case_c_and_named(self):
        def add(files):
            files[-1]["tensors"].append(["blk.3.mystery_gate.weight", [2560, 2560], 8, 0])
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "mystery", add))
        v = r["verdict"]
        self.assertEqual(v["case"], "C")
        self.assertIn("blk.3.mystery_gate.weight", "\n".join(v["blockers"]))
        self.assertEqual(r["layout"]["unknown"], ["blk.3.mystery_gate.weight"])

    def test_a_missing_and_a_reshaped_tensor_are_case_c(self):
        def drop(files):
            for f in files:
                f["tensors"] = [t for t in f["tensors"] if t[0] != "blk.6.attn_qkv.weight"]
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "missing", drop))
        self.assertEqual(r["verdict"]["case"], "C")
        self.assertEqual(r["layout"]["missing"], ["blk.6.attn_qkv.weight"])

        def reshape(files):
            for f in files:
                for t in f["tensors"]:
                    if t[0] == "blk.3.attn_k.weight":
                        t[1] = [2560, 1024]
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "reshaped-one", reshape))
        self.assertEqual(r["verdict"]["case"], "C")
        self.assertEqual([x["name"] for x in r["layout"]["reshaped"]], ["blk.3.attn_k.weight"])

    def test_a_quant_type_with_no_kernel_is_case_c(self):
        def retype(files):
            for f in files:
                for t in f["tensors"]:
                    if t[0] == "blk.5.attn_qkv.weight":
                        t[2] = 10                     # Q2_K as a dense projection: not in any native_mmvq_supported list
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "q2k", retype))
        v = r["verdict"]
        self.assertEqual(v["case"], "C")
        self.assertTrue(any("Q2_K" in b for b in v["blockers"]), v["blockers"])

    def test_an_unknown_pre_tokenizer_is_case_c(self):
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "pre", lambda fs: fs[0]["metadata"].__setitem__("tokenizer.ggml.pre", "qwen4")),
                    build_tokenizer=True)
        v = r["verdict"]
        self.assertEqual(v["case"], "C")
        self.assertTrue(any("'qwen4'" in b and "pre" in b for b in v["blockers"]), v["blockers"])

    def test_known_layers_under_a_new_architecture_name_are_case_b(self):
        def rename(files):
            for f in files:
                md = f["metadata"]
                for k in list(md):
                    if k.startswith("qwen4exp."):
                        md["qwen4x." + k[len("qwen4exp."):]] = md.pop(k)
                md["general.architecture"] = "qwen4x"
        r = analyse(files_for("fn_unsloth_ud_q4_k_xl", "renamed", rename))
        v = r["verdict"]
        self.assertEqual(v["case"], "B", v["blockers"])
        self.assertTrue(any("qwen4x" in n and "model_family_from_arch" in n for n in v["notes"]), v["notes"])

    def test_an_incomplete_split_is_case_c(self):
        paths = files_for("fn_unsloth_ud_q4_k_xl")
        r = analyse(paths[:3])
        self.assertEqual(r["verdict"]["case"], "C")
        self.assertFalse(r["split"]["complete"])

    def test_qwen35_limits(self):
        def wide(files):
            files[0]["metadata"]["qwen35moe.attention.key_length"] = 128
            files[0]["metadata"]["qwen35moe.attention.value_length"] = 128
            for t in files[0]["tensors"]:
                n = t[0]
                if re.search(r"attn_q_norm|attn_k_norm", n):
                    t[1] = [128]
                elif n.endswith("attn_q.weight"):
                    t[1] = [2048, 2 * 16 * 128]
                elif n.endswith(("attn_k.weight", "attn_v.weight")):
                    t[1] = [2048, 2 * 128]
                elif n.endswith("attn_output.weight"):
                    t[1] = [16 * 128, 2048]
        r = analyse(files_for("q36_ud_q4_k_xl", "head128", wide))
        v = r["verdict"]
        self.assertEqual(v["case"], "B", v["blockers"])
        self.assertEqual([d["key"] for d in v["shape_fields_that_differ"]], ["qwen35moe.attention.key_length"])


class Output(unittest.TestCase):
    def test_cli_writes_json_and_prints(self):
        d = pathlib.Path(_TMP.name) / "cli"
        d.mkdir(exist_ok=True)
        for p in files_for("fn_ista_q2_0") + files_for("q36_q8_0"):
            (d / p.name).write_bytes(p.read_bytes())
        out = d / "report.json"
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(Q.main([str(d), "--out", str(out), "--no-tokenizer-build"]), 0)
        rep = json.loads(out.read_text())
        self.assertEqual(sorted(s["verdict"]["case"] for s in rep["sets"]), ["A", "A"])
        self.assertEqual(sorted(s["architecture"] for s in rep["sets"]), ["qwen35moe", "qwen4exp"])
        self.assertIn("CASE A", buf.getvalue())
        s = rep["sets"][0]
        for k in ("geometry", "layout", "quant", "tokenizer", "chat_template", "mtp", "projectors", "general", "split", "verdict",
                  "draft_catalog_entry", "draft_setup_config", "files"):
            self.assertIn(k, s)
        self.assertIn("DRAFT", s["draft_catalog_entry"]["_status"])
        self.assertIn("DRAFT", s["draft_setup_config"]["_status"])

    def test_draft_entry_for_qwen36_has_the_catalog_numbers(self):
        r = analyse(files_for("q36_ud_q4_k_xl"))
        e = r["draft_catalog_entry"]
        self.assertEqual(e["arch"], "qwen35moe")
        self.assertEqual(e["kv_bytes_per_token"], 11 * 2 * 256 * 2 * 4)       # setup.py OWN_PATH_FAMILIES['qwen36']
        self.assertEqual(e["state_gb"], 0.07)
        self.assertEqual(e["mtp"], 2)
        self.assertEqual(e["license"], "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE")

    def test_a_single_header_is_fast(self):
        t0 = time.time()
        analyse(files_for("fn_unsloth_ud_q4_k_xl"), build_tokenizer=True)
        self.assertLess(time.time() - t0, 60)


@unittest.skipUnless(os.environ.get("STRATA_LIVE_HF") == "1", "set STRATA_LIVE_HF=1 to read real headers from Hugging Face")
class LiveHuggingFace(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {"STRATA_TEST_ALLOW_NETWORK": "1"})      # a live test: the tests' network guard (conftest.py) lets it through
        p.start()
        self.addCleanup(p.stop)

    def test_a_repo_directory_in_under_a_minute(self):
        t0 = time.time()
        out = pathlib.Path(_TMP.name) / "live.json"
        Q.main(["--hf", "unsloth/Qwen3.8-Flash-Next-GGUF", "--rev", "38bb39ee97821de2c9009abb7e93950eec396e66", "--dir", "UD-Q4_K_XL",
                "--out", str(out), "--quiet"])
        dt = time.time() - t0
        rep = json.loads(out.read_text())
        self.assertLess(dt, 60, dt)
        self.assertEqual(rep["sets"][0]["verdict"]["case"], "A")
        self.assertTrue(rep["sets"][0]["files"][0]["sha256"])               # the LFS oid, not a download


# ================================================================================================ the tables against the sources
def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _ints(text: str, pat: str) -> set[int]:
    return {int(x) for x in re.findall(pat, text)}


def _func(text: str, head: str) -> str:
    i = text.index(head)
    depth, j = 0, text.index("{", i)
    k = j
    while True:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
        k += 1


def _define(text: str, name: str) -> set[int]:
    """The first (STRATA_Q6K_EXPERTS=ON) definition of a STRATA_*_FMTS list."""
    m = re.search(r"#define %s\(X\)([^\n]*)" % name, text)
    return _ints(m.group(1), r"X\((\d+)\)")


def _is_iq_ints(text: str) -> set[int]:
    body = _func(text, "bool is_iq(int t)")
    return _ints(body.split("#else")[0], r"t == (\d+)")


class TablesMatchTheSources(unittest.TestCase):
    def test_geometry_defaults(self):
        text = _read("include/strata/core/geometry.hpp")
        cpp = {m.group(1): int(m.group(2)) for m in re.finditer(r"int64_t (\w+) = (\d+);", text)}
        py = Q.COMPILED
        for name, val in cpp.items():
            with self.subTest(name):
                self.assertEqual(getattr(py, name), val)
        self.assertEqual(py.rms_eps, 1e-6)
        self.assertEqual(py.family, "qwen4exp")

    def test_layout_tables_name_the_same_tensors(self):
        text = _read("include/strata/core/family_layout.hpp")
        cpp = set(re.findall(r'add\("([^"]+)"', text)) | set(re.findall(r'push_back\(\{"([^"]+)"', text))
        py = set()
        g4, g35 = Q.Geometry(), Q.Geometry(family="qwen35moe", hc=0, hc_lr=0, idx_q_heads=0, idx_key_dim=0, n_mtp_layers=1, n_layers=40)
        for g in (g4, g35):
            for l in range(g.n_layers + g.n_mtp_layers):
                py |= {n for n, _ in Q.layer_tensors(g, l)}
            py |= {n for n, _ in Q.global_tensors(g)}
        py |= {n for n, _ in Q.layer_tensors(Q.Geometry(n_mtp_layers=1), 48)}      # the MTP kind on the hyper-connection side
        self.assertEqual(py, cpp)

    def test_family_names_and_runs(self):
        text = _read("include/strata/artifact/model_family.hpp")
        self.assertIn('arch == "qwen4exp"', text)
        self.assertIn('arch == "qwen35moe"', text)
        self.assertIn("model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen4Exp; }", text)

    def test_qwen35_limits_and_required_keys(self):
        text = _read("src/qwen35/q36_engine.cu")
        self.assertIn("g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15", text)
        hpp = _read("include/strata/artifact/gguf_reader.hpp")
        body = _func(hpp, "inline std::string check_architecture_qwen35moe")
        self.assertEqual(set(re.findall(r'"qwen35moe\.([a-z_.]+)"', body)), set(Q.QWEN35_REQUIRED))

    def test_qwen4_geometry_keys(self):
        hpp = _read("include/strata/artifact/gguf_reader.hpp")
        body = _func(hpp, "inline std::string read_geometry")
        cpp = {m.group(1): m.group(3) == "true" for m in re.finditer(r'\{"([^"]+)", &core::ModelGeometry::(\w+), (true|false)\}', body)}
        self.assertEqual(cpp, {k: r for k, f, r in Q.QWEN4_KEYS})
        pins = _func(hpp, "inline std::string geometry_unsupported")
        self.assertEqual({m.group(1) for m in re.finditer(r'\{"([^"]+)", &core::ModelGeometry::\w+\}', pins)},
                         {k for k, _ in Q.PINNED} - {"expert_shared_feed_forward_length"})

    def test_cuda_and_hip_lists(self):
        iq = _read("src/kernels/cuda/iq_kernels.cu")
        L = Q.BACKEND_LISTS["cuda"]
        self.assertEqual(_define(iq, "STRATA_GU_FMTS"), L["gu"])
        self.assertEqual(_define(iq, "STRATA_D_FMTS"), L["down"])
        self.assertEqual(_is_iq_ints(iq), L["is_iq"])
        self.assertEqual(_ints(_func(iq, "bool embed_type_supported"), r"t == (\d+)"), L["embed_extra"])
        self.assertIs(Q.BACKEND_LISTS["hip"], L)
        # HIP compiles the same file: nothing in it narrows the lists to one vendor
        self.assertNotIn("__HIPCC__", "\n".join(re.findall(r"#define STRATA_(?:GU|D)_FMTS[^\n]*", iq)))
        self.assertIn("option(STRATA_Q6K_EXPERTS", _read("CMakeLists.txt"))
        self.assertRegex(_read("CMakeLists.txt"), r'option\(STRATA_Q6K_EXPERTS "[^"]*" ON\)')

    def test_mmvq_lists(self):
        for rel in ("src/kernels/cuda/native_mmvq.cu", "sycl/src/kernels/cuda/native_mmvq.dp.cpp", "metal/src/engine/mmvq_host.cpp"):
            with self.subTest(rel):
                self.assertEqual(_ints(_func(_read(rel), "bool native_mmvq_supported"), r"ggml_type == (\d+)"), Q.MMVQ)

    def test_sycl_lists(self):
        text = _read("sycl/src/kernels/cuda/iq_kernels.dp.cpp")
        L = Q.BACKEND_LISTS["sycl"]
        self.assertEqual(_define(text, "STRATA_GU_FMTS"), L["gu"])
        self.assertEqual(_define(text, "STRATA_D_FMTS"), L["down"])
        self.assertEqual(_is_iq_ints(text), L["is_iq"])
        self.assertEqual(_ints(_func(text, "bool embed_type_supported"), r"t == (\d+)"), L["embed_extra"])

    def test_metal_lists(self):
        text = _read("metal/src/engine/iq_host.cpp")
        L = Q.BACKEND_LISTS["metal"]
        self.assertEqual(_ints(_func(text, "int gu_qk(int t)"), r"case (\d+):"), L["gu"])
        self.assertEqual(_ints(_func(text, "int d_qk(int t)"), r"case (\d+):"), L["down"])
        self.assertEqual(_is_iq_ints(text) | _ints(_func(text, "bool is_tq(int t)"), r"t == (\d+)"), L["is_iq"])
        self.assertEqual(_ints(_func(text, "bool embed_type_supported"), r"t == (\d+)"), L["embed_extra"])

    def test_block_sizes_known_for_every_expert_type(self):
        for be, L in Q.BACKEND_LISTS.items():
            for t in L["gu"] | L["down"]:
                with self.subTest((be, t)):
                    self.assertGreater(Q.block_elems(t), 0)

    def test_expert_pair_rule_matches_the_cuda_function(self):
        iq = _read("src/kernels/cuda/iq_kernels.cu")
        body = _func(iq, "bool native_expert_supported")
        for frag in ("n_embd % qg == 0", "n_ff % qd == 0", "n_embd % 256 == 0", "(n_ff * n_embd) % 256 == 0", "is_iq(gu_type) && is_iq(d_type)"):
            self.assertIn(frag, body)
        # Flash-Next shape: IQ4_XS (qk 256) cannot be a down at n_ff 640; at Qwen3.6's n_ff 512 it can
        self.assertFalse(Q.expert_pair_ok("cuda", 21, 23, 2560, 640))
        self.assertTrue(Q.expert_pair_ok("cuda", 21, 23, 2048, 512))
        self.assertTrue(Q.expert_pair_ok("cuda", 12, 7, 2560, 640))

    def test_projector_types_come_from_the_control(self):
        from strata_control.models import OUR_PROJECTOR_TYPES
        self.assertEqual(set(OUR_PROJECTOR_TYPES), {"qwen3vl_merger"})

    def test_reference_tokenizer_is_flash_nexts(self):
        fix = load_fixture("fn_unsloth_ud_q4_k_xl")["files"][0]["metadata"]
        self.assertEqual(fix["tokenizer.ggml.tokens"]["sha256"], Q.REF_TOKENIZER["tokens_sha256"])
        self.assertEqual(fix["tokenizer.ggml.merges"]["sha256"], Q.REF_TOKENIZER["merges_sha256"])
        special = {t: i for i, t, _ in fix["tokenizer.ggml.tokens"]["special"]}
        for t, i in Q.REF_TOKENIZER["key_tokens"].items():
            self.assertEqual(special[t], i)
        tpl = serve_template_hash()
        self.assertEqual(tpl[:12], "12827f24b742")


def serve_template_hash() -> str:
    import hashlib
    return hashlib.sha256((ROOT / "serve" / "chat_template.jinja").read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
