# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/translate_api.py - the translation APIs' requests and answers, apart from the server: the language codes of
DeepL's and LibreTranslate's APIs, the source language's detection (py3langid, bundled in third_party/main), HTML whose
text alone is translated, and each API's request read into jobs and its answer built from their translations.

A job is one text to translate with its settings; `translate(jobs)` (the server's) gives their translations in order.
Nothing here starts an engine or reads a socket, so the tests run it as it is.
"""
from __future__ import annotations

import html
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from serve.translate import TranslationModel, PfmlError

_LANGID = Path(__file__).resolve().parents[1] / "third_party" / "main" / "py3langid"


class ApiError(ValueError):
    """A request an API refuses: the status and the message, put in that API's error shape by the server."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    """One text to translate: its source language (None: the model's own switch, or detected), target (None: the
    model's switch), register, reference context and glossary ([{"source", "target"}])."""
    text: str
    target: str | None
    source: str | None = None
    form: str | None = None
    context: str | None = None
    glossary: list | None = None
    detected: tuple | None = None        # (code, confidence 0..1) when the source was detected

    def prompt(self, model: TranslationModel) -> str:
        try:
            return model.prompt(self.text, self.target, self.source, self.form, self.context, self.glossary)
        except PfmlError as e:
            raise ApiError(400, str(e)) from None


# ------------------------------------------------------------------------------------------------- detection
_ident = None
_ident_lock = threading.Lock()


def _identifier():
    global _ident
    with _ident_lock:
        if _ident is None:
            if str(_LANGID) not in sys.path:
                sys.path.insert(0, str(_LANGID))
            from py3langid.langid import LanguageIdentifier, MODEL_FILE
            _ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
        return _ident


# Characters written one way in Simplified Chinese and another in Traditional (common ones): a Chinese text is put in
# the script whose characters it has more of, the simplified one on a tie
_HANS = set("这个们说时国会来对为学发后开经动现长门问间关见车东书语马鸟鱼体万与业两应实电话认让还进运过边该"
            "从网页么样习区热头图华产员办务县变号后师钱听写读买卖报场处")
_HANT = set("這個們說時國會來對為學發後開經動現長門問間關見車東書語馬鳥魚體萬與業兩應實電話認讓還進運過邊該"
            "從網頁麼樣習區熱頭圖華產員辦務縣變號師錢聽寫讀買賣報場處")


def chinese_script(text: str) -> str:
    hans = sum(c in _HANS for c in text)
    hant = sum(c in _HANT for c in text)
    return "zh-hant" if hant > hans else "zh-hans"


def detect(text: str, model: TranslationModel, top: int = 1) -> list:
    """The likeliest source languages of `text` among the model's, [(code, confidence 0..1)], best first: py3langid's
    probabilities over the languages it knows of the model's (a language it does not know, Cebuano, is never named;
    the nearest of the others is, with its low probability); Chinese split into the model's scripts by its
    characters.  An empty text: []."""
    if not text or not text.strip():
        return []
    ident = _identifier()
    known = sorted({c.split("-")[0] for c in model.codes()} & set(ident.nb_classes))
    with _ident_lock:                               # set_languages changes the identifier: one at a time
        ident.set_languages(known)
        ranked = ident.rank(text)
    out = []
    for code, p in ranked[:max(1, top)]:
        if code == "zh":
            code = chinese_script(text)
        if code in model.codes():
            out.append((code, float(p)))
    return out


# ------------------------------------------------------------------------------------------------------ HTML
_BLOCK = {"address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "details", "dialog", "div", "dl",
          "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head",
          "header", "hr", "html", "li", "main", "nav", "ol", "p", "section", "summary", "table", "tbody", "td", "tfoot",
          "th", "thead", "title", "tr", "ul"}
_SKIP = {"script", "style", "code", "pre", "kbd", "samp", "var", "textarea", "svg", "math"}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass
class _Run:
    """A run of text between block boundaries: its parts, each a text or an inline tag (start or end) kept as it was."""
    parts: list = field(default_factory=list)   # ("text", s) | ("tag", raw)

    def has_text(self) -> bool:
        return any(k == "text" and s.strip() for k, s in self.parts)


class _Splitter(HTMLParser):
    """The document as a list of pieces: raw markup kept as it is, and runs whose text is translated."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.pieces: list = []              # str (kept) | _Run
        self.run: _Run | None = None
        self.skip = 0                       # inside a tag whose content is kept (script, code, translate="no", ...)
        self.skip_tags: list = []

    def _raw(self, s: str):
        if self.skip or self.run is None:
            self._close()
            self.pieces.append(s)
        else:
            self.run.parts.append(("tag", s))

    def _close(self):
        if self.run is not None:
            self.pieces.append(self.run)
            self.run = None

    def _start(self, tag, attrs, empty: bool):
        raw = self.get_starttag_text() or ""
        a = dict(attrs)
        no = a.get("translate") == "no" or "notranslate" in (a.get("class") or "").split()
        if self.skip:
            if not empty and tag not in _VOID:
                self.skip_tags.append(tag)
                self.skip += 1
            self.pieces.append(raw)
            return
        if tag in _SKIP or no:
            self._close()
            self.pieces.append(raw)
            if not empty and tag not in _VOID:
                self.skip_tags.append(tag)
                self.skip = 1
            return
        if tag in _BLOCK or tag in _VOID and tag != "wbr":
            self._close()
            self.pieces.append(raw)
            return
        if self.run is None:
            self.run = _Run()
        self.run.parts.append(("tag", raw))

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        # <path/> inside <svg> needs no closing tag: counting it as one leaves skip on for the rest of the document
        self._start(tag, attrs, True)

    def handle_endtag(self, tag):
        raw = "</%s>" % tag
        if self.skip:
            self.pieces.append(raw)
            if tag in self.skip_tags:
                # HTMLParser does not infer an end tag for <pre><p>a</pre>: close the skipped inner tags too
                at = len(self.skip_tags) - 1 - self.skip_tags[::-1].index(tag)
                self.skip -= len(self.skip_tags) - at
                del self.skip_tags[at:]
            return
        if tag in _BLOCK:
            self._close()
            self.pieces.append(raw)
            return
        if self.run is None:
            self.pieces.append(raw)
        else:
            self.run.parts.append(("tag", raw))

    def handle_data(self, data):
        if self.skip:
            self.pieces.append(data)
            return
        if self.run is None:
            if not data.strip():
                self.pieces.append(data)
                return
            self.run = _Run()
        self.run.parts.append(("text", data))

    def handle_entityref(self, name):
        raw = "&%s;" % name
        if self.skip:                                   # kept markup stays escaped, or &lt;script&gt; becomes a tag
            self.pieces.append(raw)
        else:
            self.handle_data(html.unescape(raw))

    def handle_charref(self, name):
        raw = "&#%s;" % name
        if self.skip:
            self.pieces.append(raw)
        else:
            self.handle_data(html.unescape(raw))

    def handle_comment(self, data):
        self._raw("<!--%s-->" % data)

    def handle_decl(self, decl):
        self._close()
        self.pieces.append("<!%s>" % decl)

    def handle_pi(self, data):
        self._close()
        self.pieces.append("<?%s>" % data)

    def close(self):
        super().close()
        self._close()


_MARK = re.compile(r"\[(/?)(\d+)\]")


def html_jobs(doc: str) -> tuple:
    """`doc` split for translation: (pieces, texts).  Each run of text with inline tags in it becomes one text, its
    tags replaced by marks ([1] ... [/1]) so that a sentence across a tag is translated whole; html_join puts the
    translations back.  The text is unescaped (the model reads characters, not entities)."""
    sp = _Splitter()
    sp.feed(doc)
    sp.close()
    texts = []
    for p in sp.pieces:
        if isinstance(p, _Run) and p.has_text():
            texts.append(_marked(p))
    return sp.pieces, texts


def _marks(run: _Run) -> list:
    """The tags' marks as (close, number, raw tag), in the run's order.  Text is not counted: a citation like [1]
    in the source is not a tag's mark and never gets a tag put in its place."""
    out: list = []
    stack: list = []
    n = 0
    for kind, s in run.parts:
        if kind != "tag":
            continue
        if s.startswith("</"):
            out.append(("/", str(stack.pop() if stack else 0), s))
        else:
            n += 1
            stack.append(n)
            out.append(("", str(n), s))
    return out


def _marked(run: _Run) -> str:
    out: list = []
    marks = iter(_marks(run))
    for kind, s in run.parts:
        if kind == "text":
            out.append(s)
        else:
            close, num, _ = next(marks)
            out.append("[/%s]" % num if close else "[%s]" % num)
    return "".join(out)


def _unmarked(run: _Run, translated: str) -> str | None:
    """The translation of a run with its tags back in place of the marks, escaped; None when the marks did not come
    back as they went (the caller translates the run's text pieces one by one then)."""
    marks = _marks(run)
    if sorted(_MARK.findall(translated)) != sorted((close, num) for close, num, _ in marks):
        return None
    opened = {num: tag for close, num, tag in marks if not close}
    closed = {num: tag for close, num, tag in marks if close}
    out, last = [], 0
    for m in _MARK.finditer(translated):
        out.append(html.escape(translated[last:m.start()], quote=False))
        out.append(closed.get(m.group(2), "") if m.group(1) else opened.get(m.group(2), ""))
        last = m.end()
    out.append(html.escape(translated[last:], quote=False))
    return "".join(out)


def html_join(pieces: list, translations: list, translate_one) -> str:
    """The document again with each run's translation (in html_jobs' order); a run whose marks did not survive has
    its text pieces translated one by one (`translate_one(text) -> str`), its tags kept."""
    out, k = [], 0
    for p in pieces:
        if not isinstance(p, _Run):
            out.append(p)
            continue
        if not p.has_text():
            out.append("".join(html.escape(s, quote=False) if kind == "text" else s for kind, s in p.parts))
            continue
        got = _unmarked(p, translations[k])
        k += 1
        if got is None:
            got = "".join((html.escape(translate_one(s), quote=False) if s.strip() else s) if kind == "text" else s
                          for kind, s in p.parts)
        out.append(got)
    return "".join(out)


def detect_jobs(jobs: list, model: TranslationModel) -> None:
    """Each job's `detected` source language: its own text's, or, where that is unsure (below 0.5: a word or two,
    which py3langid often puts in another language), the request's texts together."""
    whole = None
    for j in jobs:
        got = detect(j.text, model)
        if not got or got[0][1] < 0.5:
            if whole is None:
                whole = detect(" ".join(x.text for x in jobs), model)
            if whole and (not got or whole[0][1] > got[0][1]):
                got = whole
        j.detected = got[0] if got else None


# ---------------------------------------------------------------------------------------------- running jobs
@dataclass
class Request:
    """An API's request read: its jobs (in its order), and whether its texts are HTML."""
    jobs: list
    html: bool = False


def translate_all(req: Request, translate) -> list:
    """The translations of `req`'s jobs in order; `translate(jobs) -> [str]` is the engine's.  An HTML job's text runs
    are translated as jobs of their own (its settings), put back in the document, and a run whose marks did not come
    back is translated text piece by text piece."""
    if not req.html:
        return list(translate(req.jobs))
    out = []
    for job in req.jobs:
        pieces, texts = html_jobs(job.text)
        subs = [Job(t, job.target, job.source, job.form, job.context, job.glossary) for t in texts]
        got = list(translate(subs)) if subs else []
        out.append(html_join(pieces, got, lambda s, j=job: translate(
            [Job(s, j.target, j.source, j.form, j.context, j.glossary)])[0]))
    return out


# ------------------------------------------------------------------------------------------------ the params
def form_or_json(body: bytes, content_type: str) -> dict:
    """A request's parameters: a JSON object as it is, else an HTML form's fields (application/x-www-form-urlencoded
    or a query string), each a list of its values.  ApiError 400 for a body that is neither."""
    from urllib.parse import parse_qs
    if (content_type or "").split(";")[0].strip().lower() == "application/json":
        import json
        try:
            got = json.loads(body or b"{}")
        except ValueError:
            raise ApiError(400, "the body is not valid JSON") from None
        if not isinstance(got, dict):
            raise ApiError(400, "send a JSON object")
        return {**got, "__json__": True}
    try:
        return parse_qs((body or b"").decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError:
        raise ApiError(400, "the form is not UTF-8") from None


def one(params: dict, key: str, default=None):
    """A single parameter: a JSON value as it is, or a form field's first value."""
    v = params.get(key, default)
    if not params.get("__json__") and isinstance(v, list):
        v = v[0] if v else default
    return v


def texts(params: dict, key: str, form_list: bool) -> list:
    """The texts of `key`: a JSON string or array of strings; a form field's values (each one a text where the API
    takes several, `form_list`, else the first)."""
    v = params.get(key)
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return v if form_list or params.get("__json__") else v[:1]
    raise ApiError(400, f"'{key}' must be a string or an array of strings")


def glossary_of(value) -> list | None:
    """Glossary entries given in the request: {"source term": "target term"}, [{"source", "target"}], or lines
    "source<TAB>target" (DeepL's glossary entry format)."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, dict):
        return [{"source": str(k), "target": str(v)} for k, v in value.items()]
    if isinstance(value, str):
        out = []
        for line in value.splitlines():
            if line.strip():
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ApiError(400, "each glossary entry line must be source<TAB>target")
                out.append({"source": parts[0], "target": parts[1]})
        return out
    if isinstance(value, list) and all(isinstance(e, dict) and set(e) == {"source", "target"} for e in value):
        return [{"source": str(e["source"]), "target": str(e["target"])} for e in value]
    raise ApiError(400, "glossary_entries must be an object of source: target terms, a list of {source, target}, "
                        "or tab-separated lines")


# ---------------------------------------------------------------------------------------------------- DeepL
DEEPL_MAX_BODY = 128 << 10                  # DeepL's request size limit (its usage limits page)
DEEPL_FORMALITY = {"default": None, "more": "polite", "less": "plain", "prefer_more": "polite",
                   "prefer_less": "plain"}


def deepl_code_in(code, model: TranslationModel, target: bool) -> str:
    """A DeepL language code (EN, EN-US, ZH, ZH-HANS, ...; any case) as the model's; ApiError 400 for one it does
    not translate.  A region (EN-US, PT-BR) is dropped: the model writes one variant of each language; ZH as a
    target is Simplified Chinese (DeepL's), as a source detected (None)."""
    if not isinstance(code, str) or not code.strip():
        raise ApiError(400, f"Value for '{'target_lang' if target else 'source_lang'}' not supported.")
    c = code.strip().lower()
    if c in ("zh-hans", "zh-hant"):
        got = c
    elif c == "zh":
        got = "zh-hans" if target else ""
    else:
        got = c.split("-")[0]
    if got == "" or got in model.codes():
        return got
    raise ApiError(400, f"Value for '{'target_lang' if target else 'source_lang'}' not supported.")


def deepl_code_out(code: str) -> str:
    return "ZH" if code.startswith("zh") else code.upper()


def deepl_languages(model: TranslationModel, kind: str) -> list:
    """GET /v2/languages: the model's languages as DeepL lists them; the targets with supports_formality (the ones
    with a register, Japanese)."""
    if kind == "target":
        out = []
        for c, n in model.languages:
            out.append({"language": c.upper(), "name": n, "supports_formality": bool(model.form_values(c))})
        return out
    seen, out = set(), []
    for c, n in model.languages:
        code = deepl_code_out(c)
        if code not in seen:
            seen.add(code)
            out.append({"language": code, "name": "Chinese" if code == "ZH" else n})
    return out


def deepl_request(params: dict, model: TranslationModel) -> Request:
    """POST /v2/translate's parameters as jobs."""
    items = texts(params, "text", form_list=True)
    if not items:
        raise ApiError(400, "Parameter 'text' not specified.")
    if one(params, "target_lang") in (None, ""):
        raise ApiError(400, "Parameter 'target_lang' not specified.")
    target = deepl_code_in(one(params, "target_lang"), model, True)
    source = one(params, "source_lang")
    source = deepl_code_in(source, model, False) or None if source not in (None, "") else None
    formality = str(one(params, "formality") or "default").lower()
    if formality not in DEEPL_FORMALITY:
        raise ApiError(400, "Value for 'formality' not supported.")
    form = DEEPL_FORMALITY[formality]
    if form is not None and form not in model.form_values(target):
        if not formality.startswith("prefer_"):
            raise ApiError(400, "'formality' is not supported for the given 'target_lang'.")
        form = None
    if one(params, "glossary_id") or params.get("glossary_ids"):
        raise ApiError(400, "Glossaries by id are not kept here; send the entries in 'glossary_entries'.")
    tags = one(params, "tag_handling")
    if tags not in (None, "", "html", "xml"):
        raise ApiError(400, "Value for 'tag_handling' not supported.")
    context = one(params, "context") or None
    glossary = glossary_of(one(params, "glossary_entries"))
    jobs = [Job(t, target, source, form, context, glossary) for t in items]
    if source is None:
        detect_jobs(jobs, model)
    return Request(jobs, html=tags in ("html", "xml"))


def deepl_answer(req: Request, translations: list) -> dict:
    return {"translations": [
        {"detected_source_language": deepl_code_out(j.source or (j.detected[0] if j.detected else "en")),
         "text": t} for j, t in zip(req.jobs, translations)]}


# ------------------------------------------------------------------------------------------- LibreTranslate
def libre_code_in(code, model: TranslationModel, allow_auto: bool) -> str | None:
    """A LibreTranslate code (en, zh, zh-Hans, zh-Hant, zt, pt-BR; any case) as the model's; None for auto;
    ApiError 400 for one it does not translate."""
    if not isinstance(code, str) or not code.strip():
        raise ApiError(400, "Invalid request: missing %s parameter" % ("source" if allow_auto else "target"))
    c = code.strip().lower()
    if c == "auto" and allow_auto:
        return None
    got = {"zh": "zh-hans", "zt": "zh-hant"}.get(c, c if c in ("zh-hans", "zh-hant") else c.split("-")[0])
    if got in model.codes():
        return got
    raise ApiError(400, f"{code} is not supported")


def libre_code_out(code: str) -> str:
    return {"zh-hans": "zh-Hans", "zh-hant": "zh-Hant"}.get(code, code)


def libre_languages(model: TranslationModel) -> list:
    codes = [libre_code_out(c) for c in model.codes()]
    return [{"code": libre_code_out(c), "name": n, "targets": [t for t in codes if t != libre_code_out(c)]}
            for c, n in model.languages]


def libre_settings(model: TranslationModel, key_required: bool) -> dict:
    first = model.languages[1] if len(model.languages) > 1 else model.languages[0]
    return {"charLimit": -1, "frontendTimeout": 500, "apiKeys": key_required, "keyRequired": key_required,
            "suggestions": False, "filesTranslation": False, "supportedFilesFormat": [],
            "language": {"source": {"code": "auto", "name": "Auto Detect"},
                         "target": {"code": libre_code_out(first[0]), "name": first[1]}}}


def libre_detect(params: dict, model: TranslationModel) -> list:
    q = one(params, "q") if not isinstance(params.get("q"), list) else (params["q"][0] if params["q"] else None)
    if not isinstance(q, str) or not q:
        raise ApiError(400, "Invalid request: missing q parameter")
    return [{"confidence": round(100.0 * p, 1), "language": libre_code_out(c)} for c, p in detect(q, model, 3)]


def libre_request(params: dict, model: TranslationModel) -> tuple:
    """POST /translate's parameters as jobs: (Request, whether q was an array, whether the source is detected)."""
    batch = isinstance(params.get("q"), list) and bool(params.get("__json__"))
    items = texts(params, "q", form_list=False)
    if not items:
        raise ApiError(400, "Invalid request: missing q parameter")
    source = libre_code_in(one(params, "source"), model, True)
    target = libre_code_in(one(params, "target"), model, False)
    fmt = str(one(params, "format") or "text").lower()
    if fmt not in ("text", "html"):
        raise ApiError(400, f"{fmt} format is not supported")
    jobs = [Job(t, target, source) for t in items]
    if source is None:
        detect_jobs(jobs, model)
    return Request(jobs, html=fmt == "html"), batch, source is None


def libre_answer(req: Request, translations: list, batch: bool, detected: bool) -> dict:
    def found(j):
        c, p = j.detected or ("en", 0.0)
        return {"confidence": round(100.0 * p, 1), "language": libre_code_out(c)}
    if batch:
        out = {"translatedText": list(translations)}
        if detected:
            out["detectedLanguage"] = [found(j) for j in req.jobs]
        return out
    out = {"translatedText": translations[0]}
    if detected:
        out["detectedLanguage"] = found(req.jobs[0])
    return out


# ------------------------------------------------------------------------- OpenAI's completions (raw PFML3)
def completion_prompt(params: dict) -> str:
    """POST /v1/completions' prompt for a translation model: the PFML3 the client wrote, the BOS put in front unless
    it is there already or add_special_tokens is false (llama-server's and vLLM's behaviour, never twice)."""
    from serve.translate import BOS
    p = params.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
        p = p[0]
    if not isinstance(p, str) or not p:
        raise ApiError(400, "'prompt' must be a string")
    if params.get("add_special_tokens") is False or p.startswith(BOS):
        return p
    return BOS + p


# ------------------------------------------------------------------------------------------- the chat's tool
TOOL = "translate"
SKILL = "translate"                         # its skill (serve/skills/translate): when and how the chat calls it
TOOL_CHARS = 32000                          # the most text one call takes (the model's context, about 8K tokens)


class TranslateTools:
    """`translate` for a chat request (a hub run_with_mcp runs: template_tools, routes, call, settings): the chat's
    model hands a text to the translation model, as the translation APIs do, rather than translating it itself.
    `run(jobs) -> [str]` is the server's translation engine; `model` its TranslationModel."""

    def __init__(self, run, model: TranslationModel):
        self.run, self.model = run, model
        self.settings = {"max_rounds": 4}

    def template_tools(self, exclude=()) -> list[dict]:
        if TOOL in exclude:
            return []
        codes = list(self.model.codes())
        forms = ", ".join(f"{t}: {' or '.join(v)}" for t, v in self.model.forms)
        return [{"name": TOOL, "description": (
            f"Translate a text with {self.model.name}, this server's translation model; read the skill "
            f"'{SKILL}' first.  Returns the translation alone."),
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "the text to translate, as it is"},
                "target": {"type": "string", "enum": codes, "description": "the language to translate into"},
                "source": {"type": "string", "enum": codes, "description": "the text's language, when known"},
                "form": {"type": "string", "description": f"the register, for these targets only ({forms})"},
                "context": {"type": "string", "description": "text that helps the translation, not translated"},
                "glossary": {"type": "array", "description": "terms translated as given", "items": {
                    "type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                    "required": ["source", "target"]}}},
                "required": ["text", "target"]}}]

    def routes(self) -> dict:
        return {TOOL: (None, TOOL)}

    def call(self, name: str, arguments, cancel: threading.Event | None = None) -> dict:
        t0 = time.monotonic()
        a = arguments if isinstance(arguments, dict) else {}
        try:
            text = a.get("text")
            if name != TOOL:
                raise ApiError(400, f"there is no tool named {name!r}")
            if not isinstance(text, str) or not text.strip():
                raise ApiError(400, "text: the text to translate")
            if len(text) > TOOL_CHARS:
                raise ApiError(400, f"text: at most {TOOL_CHARS} characters a call; translate it in parts")
            target, source = a.get("target"), a.get("source") or None
            for key, code in (("target", target), ("source", source)):
                if code is not None and code not in self.model.codes():
                    raise ApiError(400, f"{key}: one of {', '.join(self.model.codes())}")
            form = a.get("form") or None
            if form is not None and form not in self.model.form_values(str(target)):
                form = None                     # a register the target has none of: left out
            job = Job(text, target, source, form, a.get("context") or None, glossary_of(a.get("glossary")))
            out, ok = translate_all(Request([job]), self.run)[0], True
        except ApiError as e:
            out, ok = f"error: {e}", False
        except (RuntimeError, ValueError) as e:     # the engine's: it did not start, the text is too long
            out, ok = f"error: {e}", False
        return {"ok": ok, "text": out, "chars": len(out), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": TOOL}
