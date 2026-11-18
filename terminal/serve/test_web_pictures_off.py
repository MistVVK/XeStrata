"""A8 (0.2.16 retest, Arc A750): with no picture reader the chat page's paperclip still opened for a picture and the only answer was a note
saying the model was "set up for text only". The paperclip does attach text files, so it stays; the page now says up front (the button's hint)
and when a picture is dropped or chosen that pictures are not available and text files still work."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

WEB = Path(__file__).parent / "web"
LOCALES = ("en", "de", "es", "fr", "he", "ja", "ko", "pt", "ru", "zh")
OLD_HINT = "Attach a text file (or drop it here)"
OLD_NOTE = "This model was set up for text only."


class PicturesOff(unittest.TestCase):
    def catalog(self, lang):
        return json.loads((WEB / "locales" / f"{lang}.json").read_text("utf-8"))

    def test_hint_and_note_say_pictures_are_not_available_but_text_works(self):
        en = self.catalog("en")
        self.assertEqual(en["composer.attach"], OLD_HINT + ". Pictures are not available with this model.")
        self.assertEqual(en["file.picturesOffText"], "This model can't read pictures. You can still attach text files.")
        self.assertEqual(en["file.picturesOff"], "Pictures are off")

    def test_the_code_default_is_the_catalog_text(self):
        code = (WEB / "app.js").read_text("utf-8")
        en = self.catalog("en")
        self.assertIn(f'tr("composer.attach", "{en["composer.attach"]}")', code)
        self.assertIn(f'tr("file.picturesOffText", "{en["file.picturesOffText"]}")', code)
        self.assertNotIn(OLD_NOTE, code)

    def test_every_language_changed_both_lines_and_kept_its_script(self):
        en = self.catalog("en")
        for lang in LOCALES[1:]:
            cat = self.catalog(lang)
            for key in ("composer.attach", "file.picturesOffText"):
                self.assertTrue(cat[key].strip(), (lang, key))
                self.assertNotEqual(cat[key], en[key], (lang, key))
            self.assertNotIn("�", cat["composer.attach"] + cat["file.picturesOffText"], lang)

    def test_the_picture_branch_still_stops_before_reading_the_file(self):
        code = (WEB / "app.js").read_text("utf-8")
        branch = re.search(r'if \(f\.type\.startsWith\("image/"\)\) \{\n\s+if \(!health\.images\) \{(.*?)\}\n', code, re.S)
        self.assertIsNotNone(branch)
        self.assertIn("continue;", branch.group(1))
        self.assertNotIn("FileReader", branch.group(1))


if __name__ == "__main__":
    unittest.main()
