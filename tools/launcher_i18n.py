# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Windows launchers' translations: read the English text out of launcher, check the catalogs.

    python tools/launcher_i18n.py             # check every catalog against the text in the code (exit 1 on a problem)
    python tools/launcher_i18n.py --english   # print the English text as a catalog, {"key": "text"}, for a translator

The English text lives in the code (see launcher/Core/I18n.fs):
    tr "key" "English text"
    trf "key" "English text with {name}" [ "name", box value ]
The text must be one plain string literal, so that this reader and a translator see the same text.  The catalogs are
launcher/Core/locales/<lang>.json, built into the launchers.

setup's options (setup.py's OPTIONS) are fields the window draws from setup's --describe, in setup's English: their
texts are keyed setup.NAME.label, .hint, .takes, .about and .needed, and setup.NAME.choice.VALUE (VALUE "" is unset,
a character a key does not take is _), as launcher/Gui/Fields.fs looks them up.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"
LOCALES = LAUNCHER / "Core" / "locales"
LANGUAGES = ("de", "fr", "es", "ja", "ko", "pt", "ru", "zh")
CALL = re.compile(r'(?<![\w.])(?:I18n\.)?trf?\s+"([\w.\-]+)"\s+"((?:[^"\\]|\\.)*)"')
PLACEHOLDER = re.compile(r"\{(\w+)\}")
ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def english() -> tuple[dict, list]:
    """The English text by key from the code, and the problems found (a key with two different texts)."""
    text: dict[str, str] = {}
    problems: list[str] = []

    def unescape(m: re.Match[str]) -> str:
        return ESCAPES.get(m.group(1), m.group(0))

    for f in sorted(LAUNCHER.rglob("*.fs")):
        if "bin" in f.parts or "obj" in f.parts:
            continue
        for key, raw in CALL.findall(f.read_text(encoding="utf-8")):
            value = re.sub(r"\\(.)", unescape, raw)
            if key in text and text[key] != value:
                problems.append(f"{f.name}: {key} has two texts: {text[key]!r} and {value!r}")
            text[key] = value
    return text, problems


def setup_texts() -> dict:
    """The English texts of setup's options by their keys."""
    sys.path.insert(0, str(ROOT))
    import setup                                       # noqa: PLC0415 - setup.py is read only for this check

    def key(v: str) -> str:
        return "unset" if v == "" else re.sub(r"[^\w\-]", "_", v)

    text: dict[str, str] = {f"imgpick.origin.{key}": name for name, key in setup._qimage().IMAGE_ORIGIN_KEYS.items()}
    for o in setup.OPTIONS:
        for part in ("label", "hint", "takes", "about", "needed"):
            if o.get(part):
                text[f"setup.{o['name']}.{part}"] = o[part]
        for v, t in o.get("choices", []):
            text[f"setup.{o['name']}.choice.{key(v)}"] = t
    return text


def check() -> list:
    text, problems = english()
    text.update(setup_texts())
    for lang in LANGUAGES:
        path = LOCALES / f"{lang}.json"
        if not path.is_file():
            problems.append(f"{path.name}: missing")
            continue
        cat = json.loads(path.read_text(encoding="utf-8"))
        for key in sorted(set(text) - set(cat)):
            problems.append(f"{path.name}: no {key}")
        for key in sorted(set(cat) - set(text)):
            problems.append(f"{path.name}: {key} is not in the code")
        for key in sorted(set(cat) & set(text)):
            if set(PLACEHOLDER.findall(cat[key])) != set(PLACEHOLDER.findall(text[key])):
                problems.append(f"{path.name}: {key} has other placeholders than the English text")
    return problems


def main() -> int:
    if "--english" in sys.argv[1:]:
        print(json.dumps({**english()[0], **setup_texts()}, ensure_ascii=False, indent=1, sort_keys=True))
        return 0
    problems = check()
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
