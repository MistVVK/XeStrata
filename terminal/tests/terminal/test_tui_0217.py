"""0.2.17: the Terminal's retest findings (lane 0217-TUI, NEXT-0217-LIST items 3, 5, 7 to 15, 18). Screen items are checked at 80 x 24."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import drafting, drafting_fetch
from tests.terminal.launch_support import HW_12, Home, catalog
from tests.terminal.launch_support import text_of
from tests.terminal.test_fix2_tui_0216 import goto, small
from tests.terminal.test_launch_tui import Base, make_ctx


def sparse(path: Path, size: int = 2 << 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


# ------------------------------------------------------------------------------------------------ item 3

class DrafterRow(unittest.TestCase):
    """The drafter's Download row: nothing when a drafter is here, the real download and on-disk sizes otherwise (Runner.drafter_part)."""

    def part(self, *, backend="cuda", have=(), applies=False, offered=None, variant="selq4", reads_q4=True):
        tmp = Path(tempfile.mkdtemp())
        data = tmp / "data"
        for name in have:
            sparse(data / drafting.FOLDER / name)
        cfg = tmp / "strata-q2_0.json"
        cfg.write_text(json.dumps({"backend": backend, "model_name": "qwen3.8-flash-next-iq3_s", "args": ["--pack", str(data / "packs" / "q2_0"), "--max-context", "32768"]}), encoding="utf-8")
        with Home(), mock.patch.object(Runner, "installed_state", return_value=(cfg, {"model": "q2_0", "model_name": "qwen3.8-flash-next-iq3_s"})), \
                mock.patch.object(Runner, "other_copy", return_value=None), \
                mock.patch.object(drafting, "hardware_for", return_value=(backend, ["arc_a" if backend == "sycl" else "sm_120"])), \
                mock.patch.object(drafting, "tuned_applies", return_value=applies), \
                mock.patch.object(drafting, "tuned_variant", return_value=variant), \
                mock.patch.object(drafting, "engine_has_selector_q4", return_value=reads_q4), \
                mock.patch.object(drafting, "installed_engine_exe", return_value=Path("engine") if reads_q4 is not None else None):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, drafting="dflash2"), catalog=catalog(HW_12))
            if offered is not None:
                r.tuned_offered[(backend, variant)] = offered
            got = r.drafter_part()
        return got, r

    def test_an_installed_drafter_is_not_listed_even_where_the_summary_has_no_arguments(self):
        # the Arc finding: the plan read the installed model's summary (no --pack), so it looked for the drafter in another folder and listed the one that was there
        for backend in ("sycl", "cuda"):
            got, _r = self.part(backend=backend, have=(drafting.PACK_NAME,), reads_q4=backend == "cuda")
            self.assertIsNone(got, backend)

    def test_the_standard_drafter_lists_what_stays_on_the_disk(self):
        got, _r = self.part(backend="cuda")
        self.assertEqual((got["gb"], got["disk_gb"]), (1.2, 0.86))              # 0.52 + 0.34 GB: the two files a CUDA / AMD / Mac engine reads
        got, _r = self.part(backend="sycl", reads_q4=False)
        self.assertEqual((got["gb"], got["disk_gb"]), (1.2, 0.52))              # the Intel engine reads the one file

    def test_the_small_drafter_is_listed_only_while_the_feed_has_it(self):
        got, _r = self.part(applies=True, offered=True)
        self.assertEqual((got["gb"], got["disk_gb"]), (0.34, 0.34))
        got, _r = self.part(applies=True, offered=None)                         # not asked yet: the small file, as the launch tries it first
        self.assertEqual(got["gb"], 0.34)
        got, _r = self.part(applies=True, offered=False)                        # the feed lists none (a test feed): the launch fetches the standard one (the Strix finding)
        self.assertEqual((got["gb"], got["disk_gb"]), (1.2, 0.86))

    def test_a_standard_pack_that_is_here_needs_nothing_when_the_feed_has_no_small_drafter(self):
        got, _r = self.part(applies=True, offered=False, have=(drafting.PACK_NAME,))
        self.assertIsNone(got)
        got, _r = self.part(applies=True, offered=True, have=(drafting.PACK_NAME,))      # (the small one is still fetched, quietly)
        self.assertEqual(got["gb"], 0.34)

    def test_the_tuned_file_that_is_here_needs_nothing(self):
        # (a marker beside it says it was checked)
        with mock.patch.object(drafting, "tuned_ok", return_value=True):
            got, _r = self.part(applies=True, offered=True, have=(drafting.TUNED_NAME,))
        self.assertIsNone(got)

    def test_the_feed_is_not_asked_unless_the_real_screens_ask(self):
        with mock.patch.object(drafting_fetch, "tuned_offered", side_effect=AssertionError("no network in a test")):
            got, r = self.part(applies=True)
        self.assertEqual(got["gb"], 0.34)
        self.assertEqual(r.tuned_wanted, ("cuda", "selq4"))

    def test_the_feed_answer_decides_when_asked(self):
        with mock.patch.object(drafting_fetch, "tuned_offered", return_value=False) as ask:
            tmp = Path(tempfile.mkdtemp())
            cfg = tmp / "strata-q2_0.json"
            cfg.write_text(json.dumps({"backend": "cuda", "model_name": "qwen3.8-flash-next-iq3_s", "args": ["--pack", str(tmp / "data" / "packs" / "q")]}), encoding="utf-8")
            with Home(), mock.patch.object(Runner, "installed_state", return_value=(cfg, {})), mock.patch.object(Runner, "other_copy", return_value=None), \
                    mock.patch.object(drafting, "hardware_for", return_value=("cuda", ["sm_120"])), mock.patch.object(drafting, "tuned_applies", return_value=True), \
                    mock.patch.object(drafting, "tuned_variant", return_value="selq4"):
                r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, drafting="dflash2"), catalog=catalog(HW_12))
                r.look_up_feed = True
                got = r.drafter_part()
        ask.assert_called_once()
        self.assertEqual(got["gb"], 1.2)


class DrafterRowOnTheSaveScreen(Base):
    def lines(self, part):
        plan = {"gb": part["gb"], "parts": [part], "engine_usable": True, "engine_missing": False, "own_files": False, "text": ""}
        got = []

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            got.extend([app.screen.download_lines(), app.screen.download_lines(short=True)])
        with mock.patch("strata_cli.launch.app.Ctx.download_plan", return_value=plan):
            self.run_app(case)
        return got

    def test_the_on_disk_size_is_the_one_in_the_plan(self):
        full, short = self.lines({"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 1.2, "disk_gb": 0.52})
        self.assertEqual(full, ["the DFlash2 drafter, 1.2 GB (packed to 0.5 GB on disk)"])
        self.assertEqual(short, ["DFlash2 drafter 1.2 GB"])
        full, _short = self.lines({"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 1.2, "disk_gb": 0.86})
        self.assertEqual(full, ["the DFlash2 drafter, 1.2 GB (packed to 0.9 GB on disk)"])


# ------------------------------------------------------------------------------------------------ items 5, 8, 9, 10: the Save rows

PARTS = [{"kind": "weights", "label": "the model", "gb": 84.0}, {"kind": "mtp", "label": "the draft layer", "gb": 5.2},
         {"kind": "mmproj", "label": "the picture reader", "gb": 0.9},
         {"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 0.34, "tuned": True, "variant": "selq4"}]
OFFLINE_PLAN = {"gb": 90.0, "parts": PARTS, "engine_usable": False, "engine_missing": True, "own_files": False, "text": ""}
OWN = "/home/me/models/iq3"


def lines_of(app):
    return text_of(app).splitlines()


class SaveRows(Base):
    def at(self, size, own=False, plan=None, offline=False, term=None):
        got = {}

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            if own:
                ctx.settings.model_path = OWN
            ctx.offline = True if offline else None
            app.screen.paint()
            await pilot.pause(0.3)
            got["lines"] = lines_of(app)
            got["text"] = text_of(app)
        patches = [mock.patch("strata_cli.launch.app.Ctx.download_plan", return_value=plan)] if plan is not None else []
        if term is not None:
            patches.append(mock.patch("strata_cli.launch.app.Ctx.drafter_term", return_value=term))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        small(self, case, make_ctx(), size=size)
        return got

    def test_the_labels_keep_a_gap_to_the_path_in_a_wide_window(self):
        got = self.at((120, 40), own=True)
        for label in ("Your files", "Strata data"):
            row = next(ln for ln in got["lines"] if label in ln)
            self.assertRegex(row, rf"{label} {{2,}}\S", row)
        self.assertNotIn("Storage", got["text"])

    def test_the_default_save_names_the_folder_like_the_own_files_save(self):
        got = self.at((120, 40))
        self.assertTrue([ln for ln in got["lines"] if ln.strip().startswith("Strata data")])
        self.assertNotIn("Storage", got["text"])

    def test_80x24_shows_both_folders_and_the_three_choices(self):
        got = self.at((80, 24), own=True)
        t = got["text"]
        self.assertRegex(t, r"Your files +/home/me/models/iq3")
        self.assertRegex(t, r"Strata data +\S")
        for choice in ("Save as a preset and launch", "Launch without saving", "Save the preset and stop here"):
            self.assertIn(choice, t)
        for ln in got["lines"]:
            self.assertLessEqual(len(ln), 80, ln)

    def test_80x24_without_own_files_shows_the_data_folder(self):
        got = self.at((80, 24))
        self.assertRegex(got["text"], r"Strata data +\S")
        self.assertNotIn("Your files", got["text"])

    def test_offline_80x24_gives_each_missing_thing_its_own_way(self):
        got = self.at((80, 24), own=True, plan=OFFLINE_PLAN, offline=True)
        t = got["text"]
        for way in ("engine: install it from a file", "model 84 GB: put its files", "draft layer 5.2 GB: copy its mtp folder",
                    "picture reader 0.90 GB: copy its mmproj file", "DFlash2 drafter 0.34 GB: copy its .pack file"):
            self.assertIn(way, t)
        self.assertIn("Save the preset and stop here", t)           # the choices stay on screen
        for ln in got["lines"]:
            self.assertLessEqual(len(ln), 80, ln)

    def test_offline_names_one_file_as_the_fetch_command_does(self):
        got = self.at((120, 40), plan=OFFLINE_PLAN, offline=True)
        t = flat_text(got["text"])
        self.assertIn(f"the DFlash2 drafter, 0.34 GB: copy this file: {drafting.TUNED_NAME} (340 MB) into the dflash2 folder of your data folder", t)
        self.assertNotIn("both", t)
        parts = [dict(p) for p in PARTS]
        parts[3] = {"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 1.2, "tuned": False, "reads_q4": True}
        got = self.at((120, 40), plan={**OFFLINE_PLAN, "parts": parts}, offline=True)
        t = flat_text(got["text"])
        self.assertIn(f"copy one of these files: {drafting.PACK_NAME} (522 MB), or the smaller {drafting.CONVERTED_NAME} (340 MB) into the dflash2 folder", t)

    def test_the_note_names_the_drafter_this_launch_uses(self):
        self.assertIn("(packs and the DFlash2 drafter go here)", self.at((120, 40), own=True, term="the DFlash2 drafter")["text"])
        self.assertIn("(packs and the draft layer go here)", self.at((120, 40), own=True, term="the draft layer")["text"])
        self.assertIn("(packs go here)", self.at((120, 40), own=True, term="")["text"])


class OfflineDrafterLine(unittest.TestCase):
    """Item 10 (S7) and item 9: with DFlash2 as the drafter, the offline launch line is about the DFlash2 drafter and names the one file to copy in."""

    def run_it(self, *, layer_note=True):
        tmp = Path(tempfile.mkdtemp())
        data = tmp / "data"
        cfg = tmp / "strata-q2_0.json"
        cfg.write_text(json.dumps({"backend": "cuda", "model_name": "qwen3.8-flash-next-iq3_s",
                                   "args": ["--pack", str(data / "packs" / "q2_0"), "--mtp", str(data / "mtp" / "rt")]}), encoding="utf-8")
        events = []
        with Home(), mock.patch.object(drafting, "hardware_for", return_value=("cuda", ["sm_86"])), \
                mock.patch.object(drafting, "engine_has_selector_q4", return_value=True), \
                mock.patch.object(drafting_fetch, "ensure_for", return_value=None):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, drafting="dflash2"), catalog=catalog(HW_12), emit=events.append)
            r.ensure_drafter(cfg)
            if layer_note:                                   # the safety net then asked setup for the draft layer, which was offline too
                r._mtp_offline = True
                r.say_mtp_offline(cfg)
        return r, [e.text for e in events if e.stage == "model"]

    def test_the_line_is_about_the_dflash2_drafter_and_names_one_file(self):
        r, texts = self.run_it()
        last = texts[-1]
        self.assertIn("There is no internet connection, so the DFlash2 drafter can't be downloaded now.", last)
        self.assertIn("The model starts without it.", last)
        self.assertIn(f"copy one of these files: {drafting.PACK_NAME} (522 MB), or the smaller {drafting.CONVERTED_NAME} (340 MB)", last)
        self.assertNotIn("draft layer can't", " ".join(texts))
        self.assertEqual(sum("DFlash2 drafter can't" in n for n in r.notices), 1)       # said once

    def test_the_draft_layers_own_line_stays_for_a_launch_that_does_not_use_dflash2(self):
        from strata_cli.launch import runner
        with Home():
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, drafting="mtp"), catalog=catalog(HW_12))
            self.assertIsNone(getattr(r, "_df2_offline", None))
        self.assertIn("draft layer can't be downloaded", drafting.MTP_OFFLINE_NOTE)


# ------------------------------------------------------------------------------------------------ item 7

class LastRunNote(Base):
    """P4b: the paragraph about a last run that did not end normally is on the Launch screen and on the monitor, not only in the server log."""

    def marker(self, journal_says_oomd=False):
        from serve import lastrun
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "last-run-launch-x.json"
        path.write_text(json.dumps({"pid": 2147483000, "token": "gone", "started": 1.0, "profile": "launch-x"}), encoding="utf-8")
        return path, lastrun

    def test_the_runner_says_it_once_with_the_servers_own_words(self):
        path, lastrun = self.marker()
        events = []
        with mock.patch("strata_cli.launch.runner._on_linux", return_value=True), mock.patch.object(lastrun, "marker_path", return_value=path), \
                mock.patch.object(lastrun, "previous_end_words", side_effect=lambda m: lastrun.UNCLEAN_WORDS):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=catalog(HW_12), emit=events.append)
            r.say_last_run()
        notes = [e for e in events if e.stage == "note"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].text, lastrun.UNCLEAN_WORDS.removeprefix("[strata] "))
        self.assertTrue(notes[0].text.startswith("Last time, Strata did not end normally"))
        self.assertEqual(r.last_run, notes[0].text)
        self.assertIn(notes[0].text, r.notices)
        self.assertTrue(path.exists())                     # the server still finds the marker and replaces it itself

    def test_the_memory_guard_words_when_the_journal_says_so(self):
        path, lastrun = self.marker()
        with mock.patch("strata_cli.launch.runner._on_linux", return_value=True), mock.patch.object(lastrun, "marker_path", return_value=path), \
                mock.patch.object(lastrun, "previous_end_words", side_effect=lambda m: lastrun.OOMD_WORDS):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=catalog(HW_12))
            r.say_last_run()
        self.assertTrue(r.last_run.startswith("Last time, this computer's memory guard (systemd-oomd) ended Strata"))

    def test_nothing_is_said_without_a_leftover_marker_or_off_linux(self):
        path, lastrun = self.marker()
        with mock.patch.object(lastrun, "marker_path", return_value=path):
            with mock.patch("strata_cli.launch.runner._on_linux", return_value=False):
                r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=catalog(HW_12))
                r.say_last_run()
            self.assertEqual(r.last_run, "")
            path.unlink()
            with mock.patch("strata_cli.launch.runner._on_linux", return_value=True):
                r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=catalog(HW_12))
                r.say_last_run()
            self.assertEqual(r.last_run, "")

    def test_the_launch_screen_keeps_the_note_under_its_rows(self):
        from strata_cli.launch.runner import Event
        words = "Last time, Strata did not end normally (the window was closed hard). Your files are fine and this start goes on as usual."

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            app.run_launch()
            await pilot.pause(0.3)
            app.run_screen.handle_event(Event("note", words, None, "warn", {}))
            app.run_screen.handle_event(Event("model", "Downloading 12.0 of 75.8 GB", 15.8, "info", {}))
            await pilot.pause(0.3)
            t = flat_text(text_of(app))
            self.assertIn("Last time, Strata did not end normally", t)
            self.assertIn("Downloading 12.0 of 75.8 GB", t)
            self.assertIn("this start goes on as usual", t)          # the whole paragraph, wrapped
            for ln in text_of(app).splitlines():
                self.assertLessEqual(len(ln), 80, ln)
            app.cancel.set()
        from tests.terminal.test_launch_tui import FakeRunner
        FakeRunner.behaviour = "wait"
        small(self, case, make_ctx())

    def test_the_monitor_shows_it_in_its_first_minute_and_question_mark_puts_it_away(self):
        import asyncio
        from strata_cli.launch.monitor_ext import LaunchMonitor
        from tests.terminal.test_overview import Source, metrics
        info = {"url": "http://127.0.0.1:8080", "urls": ["http://127.0.0.1:8080"], "key": "k", "model": "m", "context": 32768, "profile": "launch-x", "name": "Demo",
                "last_run": "Last time, Strata did not end normally (the window was closed hard). Your files are fine and this start goes on as usual."}

        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=info, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.6)
                t = flat_text(text_of(app))
                self.assertIn("Last time, Strata did not end normally", t)
                self.assertIn("goes on as usual", t)
                await pilot.press("?")
                await pilot.pause(0.3)
                self.assertNotIn("Last time, Strata", text_of(app))
                await pilot.press("?")
                await pilot.pause(0.3)
                self.assertIn("Space freeze", flat_text(text_of(app)))          # the key list, as before
        asyncio.run(go())

    def test_the_result_carries_it_to_the_monitor(self):
        from strata_cli.launch import app as launch_app
        from strata_cli.launch.runner import Result
        res = Result("launch-x", ["http://127.0.0.1:8080"], "k", None, [], "/nope.json", last_run="Last time, x")
        self.assertEqual(launch_app._info_for(res, S.LaunchSettings(port=8080))["last_run"], "Last time, x")


def flat_text(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    unittest.main()
