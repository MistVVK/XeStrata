"""`strata doctor --profile <run config>` against a signed engine that is installed in a component folder of a clean home.

The release check (Terminal 0.1.62 from the release tarball on a fresh HOME, a feed signed with a test key) saw `doctor --profile` report the engine as
COMPONENT_MISSING although `strata launch` had just started it: the launch ran with STRATA_UPDATE_TRUSTED_KEYS naming the test key, the doctor did not.
A package signed by a key this PC does not trust fails its integrity check (engine_component.verify_signature) and the doctor reports it as not
usable. These tests pin both sides: the signed package is ok when its key is trusted (through the shipped keys' file or STRATA_UPDATE_TRUSTED_KEYS),
and is not ok when it is not."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_control import engine_component as ec, signing
from strata_control.doctor import inspect
from tests.terminal.test_component_start import FAMILY, install_fake_engine


def engine_row(report: dict) -> dict:
    return next(row for row in report["components"] if row["name"] == "engine")


class DoctorProfileSignedEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-doctor-signed-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.components = self.root / "xdg-data" / "strata" / "components"
        self.components.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {
            "HOME": str(self.root / "home"), "USERPROFILE": str(self.root / "home"), "STRATA_COMPONENTS_DIR": str(self.components),
            "STRATA_ENGINE_BACKEND": FAMILY, "STRATA_NO_ACQUIRE": "1"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_UPDATE_TRUSTED_KEYS", None)
        os.environ.pop(ec.ENV_REQUIRE_SIGNED, None)
        # the keys shipped in the release (update_keys.json) are another key than the test key: a PC with shipped keys refuses a package signed by a key it does not know
        # (with no key at all the package counts as unverified and is accepted, as before)
        shipped_file = self.root / "shipped-keys.json"
        shipped_file.write_text(json.dumps({"keys": [signing.public_part(signing.generate_key())]}), encoding="utf-8")
        shipped = mock.patch.object(signing, "BUILTIN_KEYS", shipped_file)
        shipped.start()
        self.addCleanup(shipped.stop)
        ec.register_signature_verifier(None)
        self.addCleanup(ec.register_signature_verifier, None)
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        tokens = [BYTE_TO_UNICODE[i] for i in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (tokenizer / "vocab.json").write_text(json.dumps({token: i for i, token in enumerate(tokens)}))
        (tokenizer / "merges.txt").write_text("a b")
        (tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        self.config = self.root / "launch" / "model-128k.json"            # the run config `strata launch` writes: the engine is a component reference
        self.config.parent.mkdir()
        self.config.write_text(json.dumps({"exe": "component:strata-engine", "backend": FAMILY, "tokenizer": str(tokenizer), "args": [],
                                            "cwd": str(self.root), "model_name": "fake", "host": "127.0.0.1", "port": 0}), encoding="utf-8")
        self.key = signing.generate_key()
        folder = install_fake_engine(self.components)
        manifest = folder / ec.MANIFEST
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["signature"] = signing.sign_bytes(self.key, ec.signing_payload(data))
        manifest.write_text(json.dumps(data), encoding="utf-8")
        self.trust = self.root / "test-trust.json"
        self.trust.write_text(json.dumps({"keys": [signing.public_part(self.key)]}), encoding="utf-8")

    def doctor_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["doctor", "--profile", str(self.config), "--output", "json"])
        body = json.loads(out.getvalue())
        return code, body.get("report", body)

    def test_a_signed_engine_whose_key_is_trusted_is_ok(self):
        with mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(self.trust)}):
            row = engine_row(inspect(self.config))
            self.assertEqual(row["status"], "ok", row)
            self.assertTrue(Path(row["path"]).is_relative_to(self.components / "engine"), row)
            ec.register_signature_verifier(None)
            code, report = self.doctor_cli()
        self.assertEqual(engine_row(report)["status"], "ok", report)
        self.assertNotIn("COMPONENT_MISSING", [issue["code"] for issue in report["issues"]])
        self.assertEqual(code, 0, report)

    def test_the_same_package_without_the_key_is_not_ok(self):
        """What the release check ran into: the launch had the test key, the later doctor did not."""
        self.assertEqual(engine_row(inspect(self.config))["status"], "missing")
        ec.register_signature_verifier(None)
        code, report = self.doctor_cli()
        row = engine_row(report)
        self.assertEqual(row["status"], "missing", report)
        self.assertIn("COMPONENT_MISSING", [issue["code"] for issue in report["issues"]])
        self.assertNotEqual(code, 0)
        ec.register_signature_verifier(None)
        with self.assertRaises(ec.EngineIntegrityError) as caught:
            ec.resolve(self.components, backend=FAMILY)
        self.assertIn("does not trust", str(caught.exception))
        # the row says what `strata components engine` says (the integrity sentence with its reason), and names the installed engine's folder
        self.assertEqual(row["reason"], str(caught.exception))
        self.assertTrue(row["reason"].startswith("The engine package failed its integrity check"), row)
        self.assertEqual(Path(row["path"]), next((self.components / "engine").iterdir()), row)
        self.assertNotIn("no engine for this PC", row["path"])
        issue = next(i for i in report["issues"] if i["code"] == "COMPONENT_MISSING" and i.get("component") == "engine")
        self.assertEqual(issue["message"], str(caught.exception))

    def test_a_folder_with_no_engine_still_says_no_engine_for_this_pc(self):
        import shutil
        shutil.rmtree(self.components / "engine")
        (self.components / "engine").mkdir()
        row = engine_row(inspect(self.config))
        self.assertEqual(row["status"], "missing")
        self.assertIn("no engine for this PC", row["path"])
        self.assertNotIn("reason", row)


if __name__ == "__main__":
    unittest.main()
