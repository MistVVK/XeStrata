"""The web page's token count under an answer is a plural (Russian 1 токен / 2 токена / 5 токенов / 21 токен).
The check itself is serve/test_web_tokens_plural.js (node); this runs it with the other serve tests."""
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TokenCountPlural(unittest.TestCase):
    def test_russian_forms(self):
        script = Path(__file__).with_name("test_web_tokens_plural.js")
        result = subprocess.run([shutil.which("node"), str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).strip())


if __name__ == "__main__":
    unittest.main()
