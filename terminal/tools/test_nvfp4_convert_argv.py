"""nvfp4_convert.py spools the converted tensors to a temporary file (#5098): the converter's GGUF writer otherwise keeps every
added tensor in RAM until the file is written (about 76 GB for Qwen3.8-Flash-Next). The converter itself is replaced by a
stub here: the test only reads the arguments it is given and where Python's temporary files go.

    python -m unittest tools/test_nvfp4_convert_argv.py -v
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import nvfp4_convert as N
except Exception as exc:                                    # noqa: BLE001 - the vendored converter is not in this tree
    N = None
    WHY = repr(exc)


@unittest.skipIf(N is None, "the vendored converter is not importable here")
class SpoolsToATemporaryFile(unittest.TestCase):
    def run_main(self, env):
        seen = {}
        stub = types.ModuleType("convert_hf_to_gguf")

        def fake_main():
            seen["argv"] = list(sys.argv)
            seen["tempdir"] = tempfile.tempdir
            return 0
        stub.main = fake_main
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "model.gguf"
            old_tempdir, old_argv = tempfile.tempdir, list(sys.argv)
            try:
                with mock.patch.dict(sys.modules, {"convert_hf_to_gguf": stub}), \
                        mock.patch.dict(os.environ, env), \
                        mock.patch.object(N, "install_policy"), \
                        mock.patch.object(sys, "argv", ["nvfp4_convert.py", "--model", str(Path(tmp) / "m"),
                                                        "--outfile", str(out), "--dry-run"]):
                    if "TMPDIR" not in env:
                        os.environ.pop("TMPDIR", None)
                    self.assertEqual(N.main(), 0)
            finally:
                tempfile.tempdir = old_tempdir
                sys.argv = old_argv
            return seen, str(out.parent)

    def test_the_converter_gets_use_temp_file_and_the_output_folder_holds_it(self):
        seen, outdir = self.run_main({})
        self.assertIn("--use-temp-file", seen["argv"])
        self.assertEqual(seen["tempdir"], outdir)

    def test_tmpdir_wins_when_it_is_set(self):
        seen, outdir = self.run_main({"TMPDIR": tempfile.gettempdir()})
        self.assertIn("--use-temp-file", seen["argv"])
        self.assertNotEqual(seen["tempdir"], outdir)


if __name__ == "__main__":
    unittest.main()
