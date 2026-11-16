# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PLaMo 3 Translate's tokenizer (strata_plamo) against llama.cpp's on the same GGUF, and the PFML3 prompt
(serve/translate.py's pfml3) against the GGUF's chat template rendered by Jinja.

Needs the model, and for the ids llama.cpp's tokenizer program, so the cases run only with them and are skipped
otherwise: XESTRATA_PLAMO_GGUF (PLaMo 3 Translate's GGUF), XESTRATA_LLAMA_TOKENIZE (llama.cpp's llama-tokenize, a
version that knows `plamo3`), and jinja2 for the template.
    .venv/bin/python -m unittest discover -s tools -p test_plamo.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "serve"))

import strata_tokenizer  # noqa: E402
import translate  # noqa: E402

GGUF = os.environ.get("XESTRATA_PLAMO_GGUF")
LLAMA_TOKENIZE = os.environ.get("XESTRATA_LLAMA_TOKENIZE")
try:
    import jinja2
    import jinja2.sandbox
except ImportError:   # the template's check needs it; the rest does not
    jinja2 = None  # type: ignore[assignment]

# prose in several scripts, the pre-segmentation's cases (runs of a character and of spaces, tags spelled in the text,
# U+EE00, a leading U+FEFF), characters outside the vocabulary, control characters
TEXTS = [
    "こんにちは。今日はいい天気ですね。",
    "The quick brown fox jumps over the lazy dog.",
    "日本語と English が混ざった文。",
    "  leading and   inner   spaces  ",
    "line one\nline two\n\n\tindented",
    "ＡＢＣ１２３　全角の英数字と空白",
    "def f(x):\n    return x ** 2  # 二乗",
    "絵文字 🍣🍜 と記号 ∀x∈ℝ",
    "aaaa bbbbbbb ----- ====\n\n\n\n!!!!",
    "﻿BOM at start",
    "xy",
    "<|plamo:tag|>literal <|plamo:foo bar|> <|plamo:|>",
    "Привет, мир! مرحبا بالعالم 你好世界 안녕하세요",
    "\x07\x0b\r\n tabs\t\t\tand more",
]

# chats and chat_template_kwargs the template renders (and those it refuses)
CHATS = [
    ([{"role": "user", "content": "こんにちは"}], {}),
    ([{"role": "user", "content": "Hello"}], {"target_lang": "ja", "form": "polite"}),
    ([{"role": "user", "content": "Hello"}], {"target_lang": "zh-hant", "source_lang": "en"}),
    ([{"role": "system", "content": "[plamo-translate]\ntarget_lang=EN\nglossary=猫 => cat\nglossary=犬=>dog\n---\n"
                                    "背景の文"},
      {"role": "user", "content": "猫と犬"}], {}),
    ([{"role": "system", "content": "[plamo-translate] target_lang=ja form=plain\n---\nctx"},
      {"role": "system", "content": "more context\r\nline"},
      {"role": "user", "content": "a"}, {"role": "assistant", "content": "あ"}, {"role": "user", "content": "b"}],
     {"glossary": [{"source": "b", "target": "ビー"}]}),
    ([{"role": "system", "content": "plain context only"}, {"role": "user", "content": "x"}],
     {"target_lang": "ja(easy)"}),
    ([{"role": "user",
       "content": translate.RAW_PREFIX + "<|plamo:tag|>input<|plamo:msg|>hi<|plamo:tag|>output<|plamo:msg|>"}], {}),
    ([{"role": "user", "content": "x"}], {"target_lang": "en-US"}),            # refused: a region
    ([{"role": "user", "content": "x"}], {"source_lang": "en"}),               # refused: source without target
    ([{"role": "user", "content": "x"}], {"target_lang": "en", "form": "polite"}),   # refused: form without ja
    ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], {}),      # refused: not alternating
    ([{"role": "system", "content": "[plamo-translate]\nfoo=bar"}, {"role": "user", "content": "x"}], {}),
    ([{"role": "user", "content": "   "}], {}),                                 # refused: empty
]


def llama_ids(text: str, special: bool) -> list[int]:
    assert GGUF and LLAMA_TOKENIZE
    cmd = [LLAMA_TOKENIZE, "-m", GGUF, "-p", text, "--ids", "--no-bos", "--log-disable"]
    if not special:
        cmd.append("--no-parse-special")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("[")][-1]
    return [int(v) for v in line.strip("[]").split(",") if v.strip()]


@unittest.skipUnless(GGUF, "set XESTRATA_PLAMO_GGUF")
class PlamoTokenizerMatchesLlamaCpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        strata_tokenizer.extract(GGUF, cls.tmp.name)
        cls.dir = Path(cls.tmp.name) / "tokenizer"
        cls.tok = strata_tokenizer.Tokenizer.from_dir(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @unittest.skipUnless(LLAMA_TOKENIZE, "set XESTRATA_LLAMA_TOKENIZE")
    def test_token_ids(self):
        for text in TEXTS:
            with self.subTest(text=text):
                self.assertEqual(self.tok.encode(text), llama_ids(text, False))

    @unittest.skipUnless(LLAMA_TOKENIZE, "set XESTRATA_LLAMA_TOKENIZE")
    def test_prompt_ids(self):
        text = translate.pfml3([{"role": "user", "content": "今日は  晴れ。"}], target_lang="en")
        self.assertEqual(self.tok.encode(text, parse_special=True), llama_ids(text, True))

    def test_round_trip(self):
        for text in TEXTS:
            if "" in text or text.startswith("﻿"):
                continue   # the boundary is dropped; a leading U+FEFF is kept, as text
            with self.subTest(text=text):
                self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    @unittest.skipUnless(jinja2, "jinja2 is not installed")
    def test_template(self):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)
        env.globals["raise_exception"] = raise_exception
        template = env.from_string((self.dir / "chat_template.jinja").read_text(encoding="utf-8"))
        for messages, kwargs in CHATS:
            with self.subTest(messages=messages, kwargs=kwargs):
                try:
                    want = template.render(messages=messages, bos_token=translate.BOS, **kwargs)
                except jinja2.exceptions.TemplateError as e:
                    with self.assertRaises(translate.PfmlError) as got:
                        translate.pfml3(messages, **kwargs)
                    self.assertEqual(str(got.exception), str(e))
                    continue
                self.assertEqual(translate.pfml3(messages, **kwargs), want)


if __name__ == "__main__":
    unittest.main()
