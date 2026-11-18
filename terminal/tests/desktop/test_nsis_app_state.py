r"""A10 (0.2.16 retest, WIN item 4): after a silent uninstall four small files stay in %LOCALAPPDATA%\Strata (components\.updates\state.json and
update.lock, logs\desktop-<time>.log, runtime\.activity.lock). Decision: they are the app's own state inside the data folder the uninstaller is
told to keep unless the person ticks "Also remove my Strata data", so they stay with it and go with it:

  * state.json holds the engine pin and the accepted update feed (strata_control: .updates/state.json); deleting it would make a reinstall forget
    which engine the person was on, which is exactly what "keep my data" promises not to do;
  * the two lock files are made again on the next start and weigh nothing; the desktop log is the history a support request asks for (the shell
    keeps only the newest ones, desktoplog.rs prune);
  * a person who ticks the box gets the whole folder deleted, these files included.
The installer is never run by this test: it reads the script."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT.parent / "app" / "desktop" / "src-tauri" / "nsis" / "installer.nsi"
TEXT = TEMPLATE.read_text(encoding="utf-8")
DATA_RM = r'RmDir /r "$LOCALAPPDATA\Strata"'


class AppStateStays(unittest.TestCase):
    def test_the_data_folder_is_deleted_only_inside_the_ticked_box_branch(self):
        self.assertEqual(TEXT.count(DATA_RM), 1)
        start = TEXT.index("${If} $DeleteAppDataCheckboxState = 1")
        end = TEXT.index("${EndIf}\n\n  !ifmacrodef NSIS_HOOK_POSTUNINSTALL")
        self.assertTrue(start < TEXT.index(DATA_RM) < end, "the delete is inside the branch that needs the ticked box and not an update")
        self.assertIn("${AndIf} $UpdateMode <> 1", TEXT[start:end])

    def test_nothing_deletes_state_or_lock_files_by_name_outside_that_branch(self):
        outside = TEXT[:TEXT.index("${If} $DeleteAppDataCheckboxState = 1")] + TEXT[TEXT.index("!ifmacrodef NSIS_HOOK_POSTUNINSTALL"):]
        for name in ("state.json", "update.lock", ".activity.lock", "desktop-", r"\.updates", r"\runtime", r"\logs"):
            self.assertNotRegex(outside, re.escape(name) if name.startswith("\\") else rf"Delete\s+\"[^\"]*{re.escape(name)}", name)

    def test_the_page_says_the_folder_holds_engine_and_tunings_and_that_unticked_nothing_is_deleted(self):
        self.assertIn(r"$LOCALAPPDATA\Strata   (engine, installed copy, tunings)", TEXT)
        self.assertIn("Your models, chats and settings stay on this PC unless you tick the box.", TEXT)
        self.assertIn("Also remove my Strata data (models, chats, settings)", TEXT)

    def test_a_silent_uninstall_leaves_the_box_unticked(self):
        # /S never shows the page, so the state variable keeps its default 0: the data folder is kept
        self.assertIn("Var DeleteAppDataCheckboxState", TEXT)
        self.assertNotRegex(TEXT, r"StrCpy \$DeleteAppDataCheckboxState\s+1")


if __name__ == "__main__":
    unittest.main()
