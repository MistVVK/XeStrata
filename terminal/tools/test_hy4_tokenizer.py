"""Tests for tools/hy4_tokenizer.py: the tokenizer directory of a Hy4-Flash `.strata` file (a small synthetic file in the Hugging Face tokenizer.json form, with the
real pre-tokenizer rule) and the model's real chat template (tests/data/hy4/chat_template.jinja, Section 0 `chat_template` of the SQ-64 file) rendered by serve's ChatTemplate.

    python -m unittest tools.test_hy4_tokenizer -v

The whole-vocabulary check against Hugging Face's `tokenizers` (120,832 ids, 19-string corpus: identical ids and decode) was run by hand against the real file; it is not repeated
here because it needs the 14 MB manifest.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import hy4_tokenizer as H  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

TEMPLATE = ROOT / "tests" / "data" / "hy4" / "chat_template.jinja"
HY = "｜"
START, MIDDLE, END = f"<{HY}hy_start:opensource{HY}>", f"<{HY}hy_middle:opensource{HY}>", f"<{HY}hy_end:opensource{HY}>"
REAL_PRE = {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": False},
    {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": False},
    {"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
                                            "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": False},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}


def fake_hf_tokenizer(pre=None) -> dict:
    """A byte-level BPE tokenizer.json: the 256 byte symbols, a few merges, and the added tokens Hy4's file has the kinds of (special and not)."""
    vocab = {ch: i for i, ch in enumerate(ST.bytes_to_unicode().values())}
    merges = []

    def merge(a, b):
        merges.append([a, b])
        vocab[a + b] = len(vocab)
    h, e, l, o = "h", "e", "l", "o"
    merge(h, e)
    merge(l, l)
    merge("he", "ll")
    merge("hell", o)
    added = [{"id": len(vocab), "content": START, "special": True}, {"id": len(vocab) + 1, "content": MIDDLE, "special": True},
             {"id": len(vocab) + 2, "content": END, "special": True},
             {"id": len(vocab) + 3, "content": "<think:opensource>", "special": False}, {"id": len(vocab) + 4, "content": "</think:opensource>", "special": False}]
    return {"version": "1.0", "added_tokens": added, "normalizer": {"type": "Sequence", "normalizers": []}, "pre_tokenizer": pre or REAL_PRE,
            "post_processor": {"type": "ByteLevel"}, "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": None, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges}}


def load_dir(path: Path) -> ST.Tokenizer:
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (path / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((path / "token_type.json").read_text())
    cfg = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, cfg["pre"], cfg["special_ids"])


class Convert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "tokenizer"

    def test_directory_and_ids(self):
        hf = fake_hf_tokenizer()
        cfg = H.convert(hf, "{{ messages }}", self.out, bos=261, eos=263, pad=262)
        self.assertEqual(cfg["pre"], "deepseek3")
        self.assertEqual(cfg["special_ids"], {"bos": 261, "eos": 263, "pad": 262})
        for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"):
            self.assertTrue((self.out / name).is_file(), name)
        types = json.loads((self.out / "token_type.json").read_text())
        self.assertEqual(types[-5:], [3, 3, 3, 4, 4])                    # special -> CONTROL, the others -> USER_DEFINED
        self.assertEqual(set(types[:-5]), {1})
        tk = load_dir(self.out)
        ids = tk.encode("hello", parse_special=True)
        self.assertEqual(len(ids), 1)                                    # h e l l o -> he, ll -> hell -> hello
        self.assertEqual(tk.decode(ids), "hello")

    def test_specials_follow_the_gguf_rule(self):
        H.convert(fake_hf_tokenizer(), "", self.out, bos=261, eos=263)
        tk = load_dir(self.out)
        text = f"{START}user{MIDDLE}hi <think:opensource>x</think:opensource>{END}"
        with_special = tk.encode(text, parse_special=True)
        self.assertEqual(sum(1 for i in with_special if i in (261, 262, 263)), 3)
        plain = tk.encode(text, parse_special=False)
        self.assertNotIn(261, plain)                                     # CONTROL tokens are text when the caller does not parse specials
        self.assertIn(264, plain)                                        # USER_DEFINED `<think:opensource>` is always one token
        self.assertEqual(tk.decode(with_special), text)

    def test_digits_and_cjk_are_cut_like_the_real_rule(self):
        H.convert(fake_hf_tokenizer(), "", self.out)
        tk = load_dir(self.out)
        self.assertEqual(tk._split("a12345b"), ["a", "123", "45", "b"])
        self.assertEqual(tk._split("ab你好cd"), ["ab", "你好", "cd"])
        for s in ("a 1,000.5 你好こんにちは!\n\n  x", "é café \U0001f600"):
            self.assertEqual(tk.decode(tk.encode(s)), s)

    def test_another_pre_tokenizer_is_refused(self):
        other = dict(REAL_PRE)
        other["pretokenizers"] = [dict(p) for p in REAL_PRE["pretokenizers"]]
        other["pretokenizers"][0] = {"type": "Split", "pattern": {"Regex": "\\p{N}"}, "behavior": "Isolated", "invert": False}
        with self.assertRaises(H.NotHy4Tokenizer):
            H.convert(fake_hf_tokenizer(other), "", self.out)
        with self.assertRaises(H.NotHy4Tokenizer):
            H.convert(fake_hf_tokenizer({"type": "ByteLevel", "add_prefix_space": False, "use_regex": True}), "", self.out)

    def test_a_normalizer_or_byte_fallback_is_refused(self):
        hf = fake_hf_tokenizer()
        hf["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "NFC"}]}
        with self.assertRaises(H.NotHy4Tokenizer):
            H.convert(hf, "", self.out)
        hf = fake_hf_tokenizer()
        hf["model"]["byte_fallback"] = True
        with self.assertRaises(H.NotHy4Tokenizer):
            H.convert(hf, "", self.out)

    def test_a_hole_in_the_vocabulary_is_refused(self):
        hf = fake_hf_tokenizer()
        hf["added_tokens"][0]["id"] += 7
        with self.assertRaises(H.NotHy4Tokenizer):
            H.convert(hf, "", self.out)

    def test_the_engine_export_route(self):
        d = Path(self.tmp.name)
        (d / "tokenizer.hf.json").write_text(json.dumps(fake_hf_tokenizer()), encoding="utf-8")
        (d / "chat_template.jinja").write_text("T", encoding="utf-8")
        cfg = H.convert_files(d / "tokenizer.hf.json", d / "chat_template.jinja", self.out)
        self.assertEqual(cfg["special_ids"], {"bos": 120000, "eos": 120025, "pad": 120002})
        self.assertEqual((self.out / "chat_template.jinja").read_text(encoding="utf-8"), "T")

    def test_read_manifest_checks_real_strata_v1_magic(self):
        import struct
        d = Path(self.tmp.name)
        bad = d / "bad.strata"
        bad.write_bytes(b"STRATA1\0" + b"\x00" * 4088)
        with self.assertRaises(ValueError):
            H.read_manifest(bad)
        good = d / "good.strata"
        hdr = bytearray(4096)
        hdr[:8] = b"STRATA\x00\x01"
        struct.pack_into("<QQ", hdr, 16, 4096, 0)
        good.write_bytes(bytes(hdr))
        # Magic is accepted: either raises RuntimeError (no zstd in Python) or fails on empty payload, never ValueError("not a .strata file")
        try:
            H.read_manifest(good)
        except ValueError as exc:
            self.assertNotIn("not a .strata file", str(exc))
        except Exception:
            pass


class RealTemplate(unittest.TestCase):
    """The file's own template through serve's ChatTemplate (Jinja sandbox, trim_blocks, lstrip_blocks): the text the model was trained on."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(TEMPLATE)

    def test_plain_turn_with_the_default_effort(self):
        out = self.tpl.render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
        self.assertEqual(out, f"{START}system{MIDDLE}<{HY}reasoning_mode:opensource{HY}>reasoning_effort:high{END}"
                              f"{START}user{MIDDLE}hi{END}{START}assistant{MIDDLE}<think:opensource>")

    def test_no_think_closes_the_think_block(self):
        out = self.tpl.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}], reasoning_effort="no_think")
        self.assertTrue(out.endswith(f"{START}assistant{MIDDLE}<think:opensource></think:opensource>"))
        self.assertIn(f"{START}system{MIDDLE}Be brief.<{HY}reasoning_mode:opensource{HY}>reasoning_effort:no_think{END}", out)

    def test_tools_are_written_as_hy4_documents_them(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        out = self.tpl.render([{"role": "user", "content": "x"}], tools=tools)
        self.assertIn("<tools>\n{\"name\": \"get_weather\"", out)
        self.assertIn("<tool_calls:opensource>", out)

    def test_a_call_and_its_result_in_the_history(self):
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "reasoning_content": "need a tool",
                 "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Tokyo", "n": 3}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, {"role": "user", "content": "thanks"}]
        out = self.tpl.render(msgs, tools=[{"name": "get_weather", "parameters": {}}], reasoning_effort="max")
        self.assertIn("<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>"
                      "<arg_value:opensource>Tokyo</arg_value:opensource><arg_key:opensource>n</arg_key:opensource>"
                      "<arg_value:opensource>3</arg_value:opensource></tool_call:opensource></tool_calls:opensource>", out)
        self.assertIn(f"{START}tool{MIDDLE}<tool_response:opensource>sunny</tool_response:opensource>{END}", out)
        self.assertIn("reasoning_effort:max", out)

    def test_a_bad_effort_is_a_template_error(self):
        with self.assertRaises(Exception):
            self.tpl.render([{"role": "user", "content": "x"}], reasoning_effort="medium")


if __name__ == "__main__":
    unittest.main()
