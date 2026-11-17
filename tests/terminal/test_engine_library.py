"""Settings > Engine and `strata components engines | use | remove`: the engines installed on this computer, listed, switched and deleted.

The same stub packages the updater tests use (real schema-2 packages, a real test start). Proves: an update keeps every older engine; the list shows
size, the engine in use and the newest; a switch to an older engine and back to the newest holds across a fresh read (a restart); the newest is
selected without a pin; the engine in use and an engine a program uses cannot be deleted; a running model stops a switch with one plain answer;
a package this computer cannot run is shown with its reason and cannot be switched to; a DeepSeek engine folder of Strata 0.2.13 is neither listed nor
touched, and the deepseek family is answered in plain words (strata_control/retired.py)."""
import json
import os
import shutil
import unittest
from unittest import mock

from strata_control import engine_component as ec, engine_library as lib, preset_marks, retired, updater
from tests.terminal import update_fixture as fx
from strata_control.targets import host_platform
from tests.terminal.test_updater import PLATFORM, UpdaterCase


def folder_of(version):
    fam = fx.local_family()                          # (#4138) the family this machine takes: cuda / cuda13, a Mac metal
    return f"{version}-{host_platform()}-{fam['backend']}" + (f"-{fam['variant']}" if fam["variant"] else "")


class Library(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)           # 0.1.45 stays installed, 0.1.46 is in use
        patch = mock.patch.object(lib, "_process_uses", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def listing(self, **over):
        return lib.list_engines(self.ctx(**over))

    def qwen(self, **over):
        return {e["version"]: e for e in lib.rows_of(self.listing(**over), lib.QWEN)}

    def test_the_list_shows_every_installed_engine_with_size_and_the_one_in_use(self):
        rows = self.qwen()
        self.assertEqual(sorted(rows), ["0.1.45", "0.1.46"])
        self.assertTrue(rows["0.1.46"]["active"] and rows["0.1.46"]["newest"])
        self.assertFalse(rows["0.1.45"]["active"])
        self.assertFalse(rows["0.1.45"]["newest"])
        for row in rows.values():
            self.assertGreater(row["bytes"], 0)
            self.assertTrue(row["fits"] and row["why_not"] is None)
        listing = self.listing()
        self.assertEqual(listing["total_bytes"], sum(r["bytes"] for r in rows.values()))
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen"])
        self.assertEqual(rows["0.1.46"]["delete_blocked"], "This is the engine in use. Switch to another one first.")
        self.assertTrue(rows["0.1.45"]["can_delete"] and rows["0.1.45"]["can_use"])

    def test_switch_to_an_older_engine_and_back_to_the_newest_survives_a_restart(self):
        out = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual((out["status"], out["from"], out["to"], out["automatic"]), ("switched", "0.1.46", "0.1.45", False))
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(ec.pinned_folder(self.components), folder_of("0.1.45"))
        # a fresh read of the folder (what a restart sees)
        self.assertEqual(self.qwen()["0.1.45"]["active"], True)
        self.assertEqual(self.qwen()["0.1.46"]["active"], False)
        self.assertEqual(ec.status(self.components, backend=self.family["backend"])["engine"]["version"], "0.1.45")
        # both are still installed: nothing was removed by a switch
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46")])
        back = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual((back["status"], back["to"], back["automatic"]), ("switched", "0.1.46", True))
        self.assertIsNone(ec.pinned_folder(self.components), "the newest follows updates again: no pin")
        self.assertEqual(self.resolved(), "0.1.46")
        again = lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual((again["status"], again["changed"]), ("already", False))

    def test_a_switch_marks_tuning_presets_for_the_short_recheck_and_deletes_nothing_of_the_person(self):
        self.assertEqual(len(preset_marks.pending(self.state_base)), 1, "the update to 0.1.46 left the presets owing a re-check")
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual(preset_marks.pending(self.state_base), [], "back on the engine the presets were measured on: nothing owed")
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual(len(preset_marks.pending(self.state_base)), 1)
        self.assert_user_data_untouched()

    def test_an_update_after_a_switch_makes_the_new_engine_the_one_in_use_and_clears_the_pin(self):
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.publish(("0.1.47", fx.GOOD_ENGINE))
        out = updater.run_update(self.ctx(), control=False)
        self.assertEqual(out["results"][0]["status"], "updated")
        self.assertEqual(self.resolved(), "0.1.47")
        self.assertIsNone(ec.pinned_folder(self.components))
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46"), folder_of("0.1.47")])

    def test_the_engine_in_use_cannot_be_deleted_and_an_older_one_can(self):
        with self.assertRaisesRegex(updater.UpdateError, "engine in use"):
            lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        self.assertEqual(self.engine_folders(), [folder_of("0.1.45"), folder_of("0.1.46")])
        out = lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual((out["status"], out["version"]), ("deleted", "0.1.45"))
        self.assertGreater(out["freed_bytes"], 0)
        self.assertEqual(self.engine_folders(), [folder_of("0.1.46")])
        self.assertIsNone(updater.State(self.components).section("engine").get("previous"), "the deleted engine is not the one to roll back to")
        with self.assertRaisesRegex(updater.UpdateError, "not installed"):
            lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertEqual(self.resolved(), "0.1.46")
        self.assert_user_data_untouched()

    def test_an_engine_a_running_program_uses_cannot_be_deleted(self):
        with mock.patch.object(lib, "_process_uses", side_effect=lambda folder: folder.name == folder_of("0.1.45")):
            row = self.qwen()["0.1.45"]
            self.assertFalse(row["can_delete"])
            self.assertIn("Stop the model first", row["delete_blocked"])
            with self.assertRaisesRegex(updater.UpdateError, "Stop the model first"):
                lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        self.assertTrue((self.components / "engine" / folder_of("0.1.45")).is_dir())

    def test_deleting_a_pinned_older_version_after_switching_away_clears_the_old_marks(self):
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        lib.delete_engine(self.ctx(), "qwen", folder_of("0.1.45"))
        state = updater.State(self.components).section("engine")
        self.assertNotIn(folder_of("0.1.45"), (state.get("previous"), state.get("pinned")))

    def test_a_running_model_stops_a_switch_with_one_plain_answer_and_changes_nothing(self):
        busy = lambda: [{"profile": "app-1"}]                        # noqa: E731
        with self.assertRaises(updater.Busy) as caught:
            lib.use_engine(self.ctx(serving=busy), "qwen", folder_of("0.1.45"))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertIsNone(ec.pinned_folder(self.components))
        # deleting an engine nothing uses is not a change to the running one
        out = lib.delete_engine(self.ctx(serving=busy), "qwen", folder_of("0.1.45"))
        self.assertEqual(out["status"], "deleted")

    @unittest.skipIf(fx.local_family()["backend"] != "cuda", "an NVIDIA driver floor: a CUDA package's rule (#4138: a Mac takes Metal packages)")
    def test_a_package_this_computer_cannot_run_is_listed_with_its_reason_and_cannot_be_switched_to(self):
        other = self.components / "engine" / folder_of("0.1.99")
        shutil.copytree(self.components / "engine" / folder_of("0.1.46"), other)
        manifest = json.loads((other / ec.MANIFEST).read_text())
        manifest.update(version="0.1.99", min_driver={"nvidia": "999.0"})
        (other / ec.MANIFEST).write_text(json.dumps(manifest))
        row = self.qwen()["0.1.99"]
        self.assertFalse(row["fits"] or row["can_use"])
        self.assertIn("driver 999.0", row["why_not"])
        self.assertFalse(row["newest"], "the newest is the newest one that can run here")
        self.assertTrue(self.qwen()["0.1.46"]["newest"])
        with self.assertRaises(updater.UpdateError):
            lib.use_engine(self.ctx(), "qwen", folder_of("0.1.99"))
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertTrue(row["can_delete"], "an engine that cannot run here can still be deleted to free the space")

    def test_a_damaged_package_is_listed_and_can_be_deleted(self):
        broken = self.components / "engine" / "0.1.50-broken"
        broken.mkdir()
        (broken / "junk.bin").write_bytes(b"x" * 5000)
        rows = self.qwen()
        row = rows["0.1.50"]
        self.assertFalse(row["fits"])
        self.assertEqual(row["bytes"], 5000)
        lib.delete_engine(self.ctx(), "qwen", "0.1.50-broken")
        self.assertFalse(broken.exists())

    def test_a_version_or_newest_names_an_engine_for_the_terminal(self):
        listing = self.listing()
        self.assertEqual(lib.find_entry(listing, "0.1.45")["folder"], folder_of("0.1.45"))
        self.assertEqual(lib.find_entry(listing, "newest")["folder"], folder_of("0.1.46"))
        with self.assertRaisesRegex(updater.UpdateError, "no installed engine matches"):
            lib.find_entry(listing, "0.9.9")

    def test_a_path_is_not_a_folder_name(self):
        for bad in ("../x", "a/b", ".rejected", ""):
            with self.assertRaises(updater.UpdateError):
                lib.delete_engine(self.ctx(), "qwen", bad)
            with self.assertRaises(updater.UpdateError):
                lib.use_engine(self.ctx(), "qwen", bad)

    def test_rollback_and_use_share_one_truth(self):
        updater.rollback_engine(self.ctx())
        self.assertTrue(self.qwen()["0.1.45"]["active"])
        lib.use_engine(self.ctx(), "qwen", folder_of("0.1.46"))
        out = updater.rollback_engine(self.ctx())
        self.assertEqual((out["status"], out["to"]), ("rolled-back", "0.1.45"), "use remembered where it came from")


class Cli(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.publish(("0.1.46", fx.GOOD_ENGINE))
        updater.run_update(self.ctx(), control=False)
        for target, name, value in ((lib, "_process_uses", False), (updater, "serving_now", [])):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(updater, "detect_hardware", return_value=dict(self.hardware))
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *args):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from strata_cli.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def common(self):
        return ["--components-dir", str(self.components)]

    def test_engines_use_remove_from_the_terminal(self):
        code, text, _ = self.cli("components", "engines", *self.common())
        self.assertEqual(code, 0, text)
        self.assertIn("0.1.46", text)
        self.assertIn("in use", text)
        code, out, _ = self.cli("components", "engines", "--output", "json", *self.common())
        self.assertEqual(json.loads(out)["families"][0]["engines"][0]["version"], "0.1.46")
        code, text, err = self.cli("components", "use", "0.1.45", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("Engine 0.1.45 is now in use (was 0.1.46)", text)
        self.assertEqual(self.resolved(), "0.1.45")
        code, text, err = self.cli("components", "remove", "0.1.45", *self.common())
        self.assertEqual(code, 2, "the engine in use is not deleted")
        self.assertIn("engine in use", err)
        code, text, err = self.cli("components", "use", "newest", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("follows updates again", text)
        code, text, err = self.cli("components", "remove", "0.1.45", *self.common())
        self.assertEqual(code, 0, text + err)
        self.assertIn("Engine 0.1.45 deleted", text)
        self.assertEqual(self.engine_folders(), [folder_of("0.1.46")])


class DeepSeekFolder(UpdaterCase):
    """Strata 0.2.13 and earlier put the DeepSeek engine in engine-dsv41/; this Strata runs no DeepSeek model."""

    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.dest = self.components.parent / "engine-dsv41"
        folder = self.dest / f"0.1.56-{PLATFORM}-cuda-cuda13"
        folder.mkdir(parents=True)
        self.exe = folder / "strata-dsv41"
        self.exe.write_text("#!/bin/sh\necho 'strata-dsv41 0.1.56'\n")
        (self.dest / ".chosen").write_text(folder.name + "\n")
        patch = mock.patch.object(lib, "_process_uses", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def snapshot(self):
        return sorted((str(p.relative_to(self.dest)), p.read_bytes()) for p in self.dest.rglob("*") if p.is_file())

    def test_it_is_not_listed_and_nothing_in_it_changes(self):
        before = self.snapshot()
        listing = lib.list_engines(self.ctx())
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen"])
        self.assertNotIn("engine-dsv41", json.dumps(listing))
        self.assertEqual(self.snapshot(), before)

    def test_the_deepseek_family_is_answered_in_plain_words(self):
        before = self.snapshot()
        for call in (lib.use_engine, lib.delete_engine):
            with self.assertRaises(updater.UpdateError) as caught:
                call(self.ctx(), "deepseek", f"0.1.56-{PLATFORM}-cuda-cuda13", str(self.dest))
            self.assertEqual(str(caught.exception), retired.NOT_OFFERED)
            self.assertEqual(caught.exception.exit_code, 2)
        self.assertEqual(self.snapshot(), before)

if __name__ == "__main__":
    unittest.main()
