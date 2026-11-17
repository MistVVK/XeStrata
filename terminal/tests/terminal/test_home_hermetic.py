r"""The Home() helper of the launch tests must keep every per-user file the code reads or writes inside its temp folder.

A test that read the person's real %APPDATA%\Strata\settings.json (its remembered Strata folders) or ~/.config/strata saw the files of whoever ran it: the
DeepSeek plan tests said "tight" / "spills" for UD-Q8_K_XL on a PC that had a Strata install, and "fits" everywhere else."""
import os
import unittest
from pathlib import Path
from unittest import mock

from tests.terminal.launch_support import Home


def inside(path: Path, base: Path) -> bool:
    try:
        Path(path).resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


class HomeIsHermetic(unittest.TestCase):
    def test_every_per_user_location_is_inside_the_temp_folder(self):
        from strata_control import discovery, models, profiles
        import setup
        from strata_cli import connections
        with Home() as h:
            places = {
                "discovery.own_settings_file (Windows rule)": discovery.own_settings_file(windows=True),
                "discovery.own_settings_file (Linux/macOS rule)": discovery.own_settings_file(windows=False),
                "models.settings_path": models.settings_path(),
                "profiles.profile_dir": profiles.profile_dir(),
                "setup.settings_path": setup.settings_path(),
                "connections.connection_dir": connections.connection_dir(create=False),
            }
            for name, path in places.items():
                self.assertTrue(inside(path, h.base), f"{name} = {path} is outside {h.base}")

    def test_a_developers_own_settings_variable_does_not_reach_a_test(self):
        with mock.patch.dict(os.environ, {"STRATA_SETTINGS_FILE": "X:/mine.json", "STRATA_CONNECTION_DIR": "X:/conn"}):
            with Home():
                self.assertNotIn("STRATA_SETTINGS_FILE", os.environ)
                self.assertNotIn("STRATA_CONNECTION_DIR", os.environ)
            self.assertEqual(os.environ["STRATA_SETTINGS_FILE"], "X:/mine.json")


if __name__ == "__main__":
    unittest.main()
