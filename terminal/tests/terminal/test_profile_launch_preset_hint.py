"""#5369: `strata bench run --profile <launch preset>` said only "profile 'X' is not saved; use strata profile create". A
`strata launch` preset is saved as the profile launch-<name>, so the message now names it when it exists; otherwise it is the
same as before."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import profiles
from strata_control.profiles import ProfileError


class LaunchPresetHint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "profiles"
        self.dir.mkdir()
        env = mock.patch.dict(os.environ, {"STRATA_PROFILE_DIR": str(self.dir)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)

    def message(self, name: str) -> tuple[str, int]:
        with self.assertRaises(ProfileError) as caught:
            profiles.load(name)
        return str(caught.exception), caught.exception.exit_code

    def test_a_launch_preset_is_named(self):
        (self.dir / "launch-flashnext-iq3s.json").write_text("{}", encoding="utf-8")
        text, code = self.message("flashnext-iq3s")
        self.assertEqual(text, "profile 'flashnext-iq3s' is not saved. A `strata launch` preset is the profile launch-flashnext-iq3s; "
                               "or make one with strata profile create.")
        self.assertEqual(code, 5)

    def test_without_a_preset_the_message_is_unchanged(self):
        text, code = self.message("bench1")
        self.assertEqual(text, "profile 'bench1' is not saved; use strata profile create")
        self.assertEqual(code, 5)

    def test_a_missing_launch_profile_is_not_pointed_at_itself(self):
        text, _ = self.message("launch-gone")
        self.assertEqual(text, "profile 'launch-gone' is not saved; use strata profile create")


if __name__ == "__main__":
    unittest.main()
