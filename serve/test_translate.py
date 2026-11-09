# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The translation APIs' requests and answers (serve/translate_api.py) and the models' definitions
(serve/translate.py): DeepL's and LibreTranslate's parameters, codes, errors and answers, the source language's
detection, HTML whose text alone is translated, OpenAI's completions' BOS.  No engine: a translation here is a
function of the prompt.

    .venv/bin/python -m unittest discover -s serve -p test_translate.py
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import translate as T  # noqa: E402
from serve import translate_api as A  # noqa: E402

M = T.MODELS["plamo-3-translate-2b"]


def params(obj) -> dict:
    return A.form_or_json(json.dumps(obj).encode(), "application/json")


def echo(jobs):
    """A stand-in model: the text upper-cased, its target in front."""
    return [f"{j.target}:{j.text.upper()}" for j in jobs]


class ModelsTest(unittest.TestCase):
    def test_lookup(self):
        self.assertIs(T.model_for("plamo-3-translate-2b-Q4_K_M.gguf"), M)
        self.assertIs(T.model_for("PLaMo-3-Translate-2B"), M)
        self.assertIsNone(T.model_for("qwen3.8-flash-next"))
        self.assertEqual(len(M.codes()), 53)
        self.assertEqual(M.form_values("ja"), ("plain", "polite"))

    def test_prompt(self):
        p = M.prompt("Hi.", "ja", form="polite", glossary=[{"source": "Hi", "target": "やあ"}])
        self.assertTrue(p.startswith(T.BOS + T.RAW_PREFIX))
        self.assertIn("<|plamo:index|>Hi<|plamo:item|>やあ", p)
        self.assertTrue(p.endswith("<|plamo:key|>lang<|plamo:val|>ja<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"))


class DeepLTest(unittest.TestCase):
    def test_json(self):
        req = A.deepl_request(params({"text": ["Hello", "Bonjour le monde"], "target_lang": "JA", "formality": "more",
                                      "glossary_entries": {"Hello": "やあ"}}), M)
        self.assertEqual([j.target for j in req.jobs], ["ja", "ja"])
        self.assertEqual(req.jobs[0].form, "polite")
        self.assertEqual(req.jobs[0].glossary, [{"source": "Hello", "target": "やあ"}])
        got = A.deepl_answer(req, A.translate_all(req, echo))
        self.assertEqual(got["translations"][1], {"detected_source_language": "FR", "text": "ja:BONJOUR LE MONDE"})

    def test_form(self):
        p = A.form_or_json(b"text=one&text=two&target_lang=EN-US&source_lang=FR", "application/x-www-form-urlencoded")
        req = A.deepl_request(p, M)
        self.assertEqual([(j.text, j.target, j.source) for j in req.jobs], [("one", "en", "fr"), ("two", "en", "fr")])
        self.assertEqual(A.deepl_answer(req, ["a", "b"])["translations"][0]["detected_source_language"], "FR")

    def test_codes(self):
        self.assertEqual(A.deepl_code_in("ZH", M, True), "zh-hans")
        self.assertEqual(A.deepl_code_in("ZH-HANT", M, True), "zh-hant")
        self.assertEqual(A.deepl_code_in("zh", M, False), "")       # Chinese as a source: detected
        self.assertEqual(A.deepl_code_out("zh-hant"), "ZH")
        targets = A.deepl_languages(M, "target")
        self.assertEqual(targets[0], {"language": "JA", "name": "Japanese", "supports_formality": True})
        self.assertIn({"language": "ZH", "name": "Chinese"}, A.deepl_languages(M, "source"))

    def test_errors(self):
        for bad, words in (({"target_lang": "JA"}, "'text'"), ({"text": ["x"]}, "'target_lang'"),
                           ({"text": ["x"], "target_lang": "PT-BR"}, "target_lang"),
                           ({"text": ["x"], "target_lang": "EN", "formality": "more"}, "formality"),
                           ({"text": ["x"], "target_lang": "JA", "glossary_id": "g1"}, "glossary_entries")):
            with self.assertRaises(A.ApiError) as c:
                A.deepl_request(params(bad), M)
            self.assertEqual(c.exception.status, 400)
            self.assertIn(words, str(c.exception))
        # prefer_*: dropped where the target has no register
        req = A.deepl_request(params({"text": ["x"], "target_lang": "EN", "formality": "prefer_more"}), M)
        self.assertIsNone(req.jobs[0].form)


class LibreTest(unittest.TestCase):
    def test_batch_auto(self):
        req, batch, detected = A.libre_request(params({"q": ["你好世界", "這是什麼"], "source": "auto", "target": "ja"}), M)
        self.assertTrue(batch and detected)
        got = A.libre_answer(req, A.translate_all(req, echo), batch, detected)
        self.assertEqual(got["translatedText"], ["ja:你好世界", "ja:這是什麼"])
        self.assertEqual([d["language"] for d in got["detectedLanguage"]], ["zh-Hans", "zh-Hant"])

    def test_form_single(self):
        p = A.form_or_json(b"q=Hello&q=ignored&source=en&target=zh", "application/x-www-form-urlencoded")
        req, batch, detected = A.libre_request(p, M)
        self.assertFalse(batch or detected)
        self.assertEqual(A.libre_answer(req, ["x"], batch, detected), {"translatedText": "x"})
        self.assertEqual(req.jobs[0].target, "zh-hans")

    def test_html(self):
        req, batch, detected = A.libre_request(params({"q": "<p>a <b>b</b> c</p><code>k</code>", "source": "en",
                                                      "target": "ja", "format": "html"}), M)
        self.assertEqual(A.translate_all(req, echo), ["<p>ja:A <b>B</b> C</p><code>k</code>"])

    def test_errors_and_lists(self):
        with self.assertRaises(A.ApiError) as c:
            A.libre_request(params({"q": "x", "source": "en", "target": "pt"}), M)
        self.assertEqual(c.exception.status, 400)
        langs = A.libre_languages(M)
        self.assertEqual(langs[2]["code"], "zh-Hans")
        self.assertNotIn("zh-Hans", langs[2]["targets"])
        self.assertEqual(A.libre_settings(M, True)["keyRequired"], True)
        self.assertEqual(A.libre_detect({"q": "Guten Morgen, wie geht es dir heute?"}, M)[0]["language"],
                         "de")


class HtmlTest(unittest.TestCase):
    def test_marks_round_trip(self):
        pieces, texts = A.html_jobs('<div>One <a href="u">two</a> three &amp; four<br>five</div>'
                                    '<p translate="no">x</p>')
        self.assertEqual(texts, ["One [1]two[/1] three & four", "five"])
        out = A.html_join(pieces, ["UNO [1]DOS[/1] TRES & CUATRO", "CINCO"], lambda s: "?")
        self.assertEqual(out, '<div>UNO <a href="u">DOS</a> TRES &amp; CUATRO<br>CINCO</div><p translate="no">x</p>')

    def test_marks_lost(self):
        pieces, texts = A.html_jobs("<p>a <i>b</i> c</p>")
        out = A.html_join(pieces, ["no marks"], lambda s: s.upper())
        self.assertEqual(out, "<p>A <i>B</i> C</p>")


class DetectTest(unittest.TestCase):
    def test_languages(self):
        self.assertEqual(A.detect("これは日本語の文です。", M)[0][0], "ja")
        self.assertEqual(A.detect("This is an English sentence.", M)[0][0], "en")
        self.assertEqual(A.detect("这是简体中文。", M)[0][0], "zh-hans")
        self.assertEqual(A.detect("這是繁體中文。", M)[0][0], "zh-hant")
        self.assertEqual(A.detect("  ", M), [])

    def test_short_texts_together(self):
        jobs = [A.Job("Hello", "ja"), A.Job("This is an English sentence about the weather.", "ja")]
        A.detect_jobs(jobs, M)
        self.assertEqual(jobs[0].detected[0], "en")


class CompletionsTest(unittest.TestCase):
    def test_bos_once(self):
        self.assertEqual(A.completion_prompt({"prompt": "x"}), T.BOS + "x")
        self.assertEqual(A.completion_prompt({"prompt": T.BOS + "x"}), T.BOS + "x")
        self.assertEqual(A.completion_prompt({"prompt": "x", "add_special_tokens": False}), "x")
        with self.assertRaises(A.ApiError):
            A.completion_prompt({"prompt": 3})


class ToolTest(unittest.TestCase):
    def test_call_and_skill(self):
        tools = A.TranslateTools(echo, M)
        self.assertEqual([t["name"] for t in tools.template_tools()], ["translate"])
        got = tools.call("translate", {"text": "hi", "target": "ja", "form": "polite"})
        self.assertEqual((got["ok"], got["text"]), (True, "ja:HI"))
        self.assertFalse(tools.call("translate", {"text": "hi", "target": "xx"})["ok"])
        self.assertEqual(tools.call("translate", {"text": "hi", "target": "en", "form": "polite"})["text"], "en:HI")
        from serve import skills as S
        hub = S.SkillTools(S.Skills(None), tools=(A.TOOL,))
        self.assertIn("- translate:", hub.template_tools()[0]["description"])
        self.assertIn("`translate` tool", hub.call("load_skill", {"names": ["translate"]})["text"])
        self.assertEqual(S.SkillTools(S.Skills(None)).template_tools(), [])


if __name__ == "__main__":
    unittest.main()
