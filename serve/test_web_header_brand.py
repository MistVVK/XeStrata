"""#5510: the name in the web page's header links to the project, opens in a new tab without handing the page over, keeps
the look of the name, and its tooltip is translated in every catalog."""
from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


class _Brand(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.found: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if "st-header__brand" in (a.get("class") or "").split():
            self.found.append((tag, a))


class HeaderBrand(unittest.TestCase):
    def setUp(self) -> None:
        p = _Brand()
        p.feed((WEB / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(len(p.found), 1, p.found)
        self.tag, self.attrs = p.found[0]

    def test_the_name_is_a_link_to_the_project(self) -> None:
        self.assertEqual(self.tag, "a")
        self.assertEqual(self.attrs.get("href"), "https://github.com/Niko1221/Strata")
        self.assertEqual(self.attrs.get("target"), "_blank")
        self.assertTrue({"noopener", "noreferrer"} <= set((self.attrs.get("rel") or "").split()))

    def test_its_tooltip_is_in_every_catalog(self) -> None:
        key = self.attrs.get("data-i18n-title")
        self.assertEqual(key, "header.github")
        en = json.loads((WEB / "locales" / "en.json").read_text(encoding="utf-8"))
        self.assertEqual(en[key], self.attrs.get("title"))
        for lang in ("de", "es", "fr", "ja", "ko", "pt", "ru", "zh"):
            text = json.loads((WEB / "locales" / f"{lang}.json").read_text(encoding="utf-8")).get(key)
            self.assertTrue(text and "GitHub" in text, f"{lang}: {text!r}")

    def test_it_keeps_the_look_of_the_name(self) -> None:
        css = (WEB / "app.css").read_text(encoding="utf-8")
        rule = re.search(r"a\.st-header__brand\s*\{([^}]*)\}", css)
        self.assertIsNotNone(rule)
        self.assertRegex(rule.group(1), r"color:\s*inherit")
        self.assertRegex(rule.group(1), r"text-decoration:\s*none")


if __name__ == "__main__":
    unittest.main()
