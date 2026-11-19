"""tools/early_close_test.py is named *_test.py, so pytest imports it while it collects (#6191). Importing it must
do nothing: no read of sys.argv, no request to a server that happens to be running.

    python -m unittest tools/test_early_close_import.py -v
"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


class ImportDoesNothing(unittest.TestCase):
    def test_importing_with_pytests_own_arguments_sends_nothing(self):
        sys.path.insert(0, str(HERE))
        sys.modules.pop("early_close_test", None)
        try:
            with mock.patch.object(sys, "argv", ["pytest", "-q"]), \
                    mock.patch("urllib.request.urlopen", side_effect=AssertionError("a request was sent on import")):
                module = importlib.import_module("early_close_test")
            self.assertEqual(module.URL, "")
            self.assertTrue(callable(module.main))
        finally:
            sys.modules.pop("early_close_test", None)
            sys.path.remove(str(HERE))


if __name__ == "__main__":
    unittest.main()
