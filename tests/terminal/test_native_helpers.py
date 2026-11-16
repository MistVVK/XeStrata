"""The native-helper contract (strata_control/native_helpers.py) and its first user, the GGUF preflight helper of #5961, with stand-in
helper programs (no Rust needed): off unless opted in, never discovered, one versioned envelope, bounded, strict JSON, schema-checked,
and an exact fallback to the Python reader whenever the contract is broken. A rejection given within the contract stays a rejection."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import gguf_preflight, native_helpers
from strata_control.profiles import ProfileError
from tests.terminal.test_own_files import gguf_bytes


@unittest.skipIf(os.name == "nt", "script stand-ins for the helper; the Windows helper needs a real .exe")
class Contract(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-native-")
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.helpers = self.dir / "helpers"
        self.helpers.mkdir()
        self.ran = self.dir / "ran.txt"
        self.model = self.dir / "model.gguf"
        self.model.write_bytes(gguf_bytes(["IQ3_S"]))
        env = mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "gguf-preflight", native_helpers.DIR_ENV: str(self.helpers)})
        env.start()
        self.addCleanup(env.stop)

    def helper(self, body: str, where: Path | None = None) -> Path:
        """A stand-in strata-gguf-preflight: records that it ran, then runs `body` (Python) with `argv` in scope."""
        path = (where or self.helpers) / "strata-gguf-preflight"
        path.write_text(f"#!{sys.executable}\n"
                        "import json, sys, time\n"
                        f"open({str(self.ran)!r}, 'a').write('ran\\n')\n"
                        "argv = sys.argv[1:]\n"
                        + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def python_answer(self):
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: "off"}):
            return gguf_preflight.inspect_metadata(self.model)

    def answer(self, result: dict) -> str:
        return f"print(json.dumps({{'protocol': 1, 'ok': True, 'result': json.loads({json.dumps(result)!r})}}))\n"

    def ran_count(self) -> int:
        return len(self.ran.read_text().splitlines()) if self.ran.exists() else 0

    # ---- opt-in and discovery
    def test_off_unless_named(self):
        self.helper("print('{}')\n")
        with mock.patch.dict(os.environ, {native_helpers.ENABLE_ENV: ""}):
            self.assertEqual(gguf_preflight.inspect_metadata(self.model), self.python_answer())
        self.assertEqual(self.ran_count(), 0)

    def test_never_found_on_path_or_in_a_build_folder(self):
        elsewhere = self.dir / "bin"
        elsewhere.mkdir()
        self.helper("print('{}')\n", where=elsewhere)
        with mock.patch.dict(os.environ, {"PATH": str(elsewhere) + os.pathsep + os.environ.get("PATH", ""), native_helpers.DIR_ENV: ""}):
            self.assertIsNone(native_helpers.program("gguf-preflight"))
            self.assertEqual(gguf_preflight.inspect_metadata(self.model), self.python_answer())
        self.assertEqual(self.ran_count(), 0)

    # ---- a valid answer is used, with the protocol arguments
    def test_a_valid_answer_is_used(self):
        want = self.python_answer()
        self.helper("assert argv[:3] == ['--strata-protocol', '1', 'inspect'], argv\n" + self.answer({**want, "architecture": "from-helper"}))
        self.assertEqual(gguf_preflight.inspect_metadata(self.model)["architecture"], "from-helper")
        self.assertEqual(self.ran_count(), 1)

    # ---- every broken contract falls back to the Python reader, whose answer is the answer
    def fallback(self, body: str, timeout: float | None = None):
        self.helper(body)
        patch = mock.patch.object(native_helpers, "DEFAULT_TIMEOUT", timeout) if timeout else mock.patch.dict(os.environ, {})
        with patch:
            self.assertEqual(gguf_preflight.inspect_metadata(self.model), self.python_answer())
        self.assertEqual(self.ran_count(), 1)

    def test_timeout(self):
        self.fallback("time.sleep(30)\n", timeout=1.0)

    def test_crash(self):
        self.fallback("raise SystemExit(70)\n")

    def test_not_json(self):
        self.fallback("print('GGUF ok')\n")

    def test_trailing_data(self):
        self.fallback("print(json.dumps({'protocol': 1, 'ok': True, 'result': {}}) + ' {}')\n")

    def test_duplicate_keys(self):
        self.fallback("print('{\"protocol\": 1, \"protocol\": 1, \"ok\": true, \"result\": {}}')\n")

    def test_nan(self):
        self.fallback("print('{\"protocol\": 1, \"ok\": true, \"result\": {\"x\": NaN}}')\n")

    def test_another_protocol(self):
        self.fallback(self.answer(self.python_answer()).replace("'protocol': 1", "'protocol': 2"))

    def test_a_result_of_another_shape(self):
        self.fallback("print(json.dumps({'protocol': 1, 'ok': True, 'result': {'architecture': 'x'}}))\n")

    def test_a_bool_where_a_count_belongs(self):
        bad = {**self.python_answer(), "tensor_count_reported": True}
        self.fallback(self.answer(bad))

    def test_too_much_output(self):
        self.fallback("sys.stdout.write(' ' * (2 << 20) + '{}')\n")

    # ---- a rejection within the contract stays a rejection
    def test_a_rejection_is_a_rejection(self):
        self.helper("print(json.dumps({'protocol': 1, 'ok': False, 'error': {'code': 5, 'message': 'GGUF tensor byte ranges overlap'}}))\n")
        with self.assertRaisesRegex(ProfileError, "GGUF tensor byte ranges overlap"):
            gguf_preflight.inspect_metadata(self.model)

    def test_the_python_side_checks_links_and_regular_files_first(self):
        self.helper(self.answer(self.python_answer()))
        with self.assertRaises(ProfileError):
            gguf_preflight.inspect_metadata(self.dir / "missing.gguf")
        self.assertEqual(self.ran_count(), 0)


class StrictJson(unittest.TestCase):
    """The parser is Python's own: the two losses found in the custom parser of #5972 / #5975 cannot happen here."""

    def test_an_escaped_surrogate_pair_survives(self):
        self.assertEqual(native_helpers.parse('{"name": "GPU \\ud83d\\ude80"}')["name"], "GPU \U0001F680")

    def test_a_large_integer_is_exact(self):
        self.assertEqual(native_helpers.parse('{"n": 9007199254740993}')["n"], 9007199254740993)

    def test_only_one_object(self):
        for bad in ('[1]', '{"a": 1} {"b": 2}', '{"a": Infinity}', '{"a": 1, "a": 2}'):
            with self.assertRaises(ValueError, msg=bad):
                native_helpers.parse(bad)


if __name__ == "__main__":
    unittest.main()
