"""0.2.17: the Terminal's retest findings (lane 0217-TUI, NEXT-0217-LIST items 3, 5, 7 to 15, 18). Screen items are checked at 80 x 24."""

from __future__ import annotations

import copy
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
from strata_cli.launch import store
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


# ------------------------------------------------------------------------------------------------ items 12, 13, 15, 18 and the small ones

class MacNote(unittest.TestCase):
    def ready_line(self, result):
        events = []
        with Home(), mock.patch("strata_control.runtime.status", return_value={"state": "stopped"}), \
                mock.patch("strata_control.runtime.start", return_value=result), \
                mock.patch.object(Runner, "stop_others"), mock.patch.object(Runner, "check_port_free"), mock.patch.object(Runner, "log_marks", return_value={}), \
                mock.patch.object(Runner, "start_wait", return_value=60), mock.patch.object(Runner, "report_drafting"), \
                mock.patch.object(Runner, "mac_first_start", return_value=True):
            r = Runner(S.LaunchSettings(name="x", family="qwen", model="Q2_0", context=32768), catalog=catalog(HW_12), emit=events.append)
            r.start()
        return [e.text for e in events if e.stage == "start"][-1]

    def test_no_ten_seconds_note_when_the_starts_own_answer_already_ran(self):
        self.assertEqual(self.ready_line({"ok": True, "state": "ready", "generation_ready": True}), "The model is loaded and answering.")

    def test_the_note_stays_for_a_start_that_made_no_answer(self):
        self.assertIn("10 seconds longer", self.ready_line({"ok": True, "state": "ready"}))


class CardsSentence(unittest.TestCase):
    def cards(self, names):
        return [{"index": i, "name": n, "vendor": "nvidia", "vram_gb": 16.0, "usable": True, "memory_mib": 16376, "total_mib": 16376} for i, n in enumerate(names)]

    def test_two_alike_cards_are_named_once(self):
        from strata_control import gpu_roles
        text = gpu_roles.sentence(self.cards(["NVIDIA RTX A4000", "NVIDIA RTX A4000"]), "nvidia")
        self.assertEqual(text.count("RTX A4000"), 1, text)
        self.assertIn("Both cards are RTX A4000, so they are alike: one is Main and the other helps.", text)
        self.assertNotIn("is Main: the cards are alike", text)

    def test_unlike_cards_keep_their_reason(self):
        from strata_control import gpu_roles
        text = gpu_roles.sentence(self.cards(["NVIDIA RTX A4000", "NVIDIA RTX 4090"]), "nvidia")
        self.assertIn("is Main:", text)


class OfflineDialog(Base):
    def ask(self, offline):
        asked = {}

        async def case(app, pilot, ctx):
            ctx.offline = offline
            from tests.terminal.test_launch_tui import FakeRunner
            FakeRunner.download = 76.0
            app.run_launch()
            await pilot.pause(0.3)
            asked["text"] = flat_text(text_of(app))
            await pilot.press("escape")
        small(self, case, make_ctx())
        return asked["text"]

    def test_offline_the_dialog_says_so_and_the_first_choice_is_not_a_promise(self):
        t = self.ask(True)
        self.assertIn("This computer is offline", t)
        self.assertIn("Try the download", t)
        self.assertNotIn("Download and start", t)

    def test_online_it_is_as_before(self):
        t = self.ask(None)
        self.assertIn("Download and start", t)
        self.assertNotIn("offline", t)


class OfflineProbe(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_a_feed_on_this_computer_does_not_make_an_offline_computer_look_online(self):
        ctx = make_ctx()
        ctx.feed = "http://127.0.0.1:19881/strata-feed.json"
        seen = []
        with mock.patch.object(drafting_fetch, "online", side_effect=lambda url, timeout=0: seen.append(url) or False):
            self.assertTrue(ctx.probe_now())
        self.assertEqual(seen, [drafting.base_url()])                 # only the model host was asked; the loopback feed was not

    def test_a_feed_elsewhere_still_counts(self):
        ctx = make_ctx()
        ctx.feed = "https://mirror.example.org/strata-feed.json"
        with mock.patch.object(drafting_fetch, "online", side_effect=lambda url, timeout=0: url.startswith("https://mirror")):
            self.assertFalse(ctx.probe_now())

    def test_is_offline_asks_for_a_start_that_skipped_the_save_screen_and_keeps_the_answer(self):
        ctx = make_ctx()
        ctx.probe_offline = True
        with mock.patch.object(drafting_fetch, "online", return_value=False) as ask:
            self.assertTrue(ctx.is_offline())
            first = ask.call_count
            self.assertTrue(ctx.is_offline())
        self.assertEqual(ask.call_count, first)                        # asked once


# ------------------------------------------------------------------------------------------------ items 11, 16, 18: the 80-column cuts

class At80Columns(unittest.TestCase):
    def draw(self, raw, width=78, height=22, **kw):
        from strata_cli import overview
        from tests.terminal.test_overview import draw
        return draw(overview.build(raw), width=width, height=height, connect=True, **kw)

    def test_the_request_table_has_whole_headers_and_cells(self):
        from tests.terminal.test_overview import metrics
        text = self.draw(metrics("generating"))
        for head in ("TIME", "STATE", "TOK/S", "TOOK"):
            self.assertIn(head, text)
        self.assertNotIn("…", text.split("Recent requests")[1])                      # no cut cell, no cut header
        self.assertIn("429.8 s", text)
        self.assertIn("17,183", text)

    def test_a_wide_window_keeps_the_full_headers(self):
        from tests.terminal.test_overview import metrics
        text = self.draw(metrics("generating"), width=110, height=30)
        for head in ("STATUS", "PROMPT", "REUSED", "OUTPUT", "DURATION"):
            self.assertIn(head, text)

    def test_the_help_line_is_there_in_the_22_rows_the_window_gives(self):
        from tests.terminal.test_overview import metrics
        text = self.draw(metrics("generating"), width=78, height=22).rstrip("\n").split("\n")
        self.assertLessEqual(len(text), 22)
        self.assertIn("? help   q quit", text[-1])
        self.assertIn("7 views", text[-1])

    def test_two_cards_keep_the_help_line_too(self):
        from tests.terminal.test_overview import metrics
        gib = 2 ** 30
        card = lambda i: {"index": i, "name": "NVIDIA RTX A4000", "util": 3, "mem_used": gib, "mem_total": 16 * gib, "temp": 40, "power": 20,  # noqa: E731
                          "power_limit": 140, "pcie_gen": 1, "pcie_gen_max": 4, "pcie_width": 16, "pcie_width_max": 16}
        m = metrics("idle")
        m["hardware"] = {**m["hardware"], "gpus": [card(0), card(1)]}
        lines = self.draw(m).rstrip("\n").split("\n")
        self.assertLessEqual(len(lines), 22)
        self.assertIn("? help   q quit", lines[-1])
        tall = self.draw(m, width=110, height=36)
        self.assertIn("Cards", tall)                                                   # the strip is there where the window has the room
        self.assertIn("idle, max Gen4", tall)

    def test_nothing_in_the_state_line_is_cut(self):
        from tests.terminal.test_overview import metrics
        line = next(ln for ln in self.draw(metrics("generating")).split("\n") if ln.startswith("Model state"))
        self.assertTrue(line.rstrip().endswith("3%"), line)

    def test_the_tiles_of_a_mac_say_whole_words(self):
        from tests.terminal.test_overview import metrics
        m = metrics("generating")
        m["hardware_static"] = {"gpu_name": "Apple M5 Max", "gpu_count": 1, "cores": 18, "threads": 18}
        m["hardware"] = {**m["hardware"], "gpu_mem_used": None, "gpu_mem_total": None, "gpu_pcie_gen": None, "gpu_pcie_width": None, "ram_total": 128 * 2 ** 30}
        text = self.draw(m)
        for whole in ("shared memory", "memory shared", "18 cores"):
            self.assertIn(whole, text)
        for cut in ("shar ", "cac ", "18 cores · 18 "):
            self.assertNotIn(cut, text)

    def test_the_experts_tile_never_ends_in_half_a_word(self):
        from tests.terminal.test_overview import metrics
        text = self.draw(metrics("generating"))
        self.assertIn("1,644 experts", text)
        self.assertNotIn("experts cac", text)

    def test_an_idle_card_says_its_link_is_resting(self):
        from tests.terminal.test_overview import metrics
        m = metrics("idle")
        m["hardware"] = {**m["hardware"], "gpu_pcie_gen": 1, "gpu_pcie_gen_max": 4}
        text = self.draw(m, width=110, height=30)
        self.assertIn("Gen1 x16", text)
        self.assertIn("idle, max Gen4", text)
        busy = metrics("generating")
        busy["hardware"] = {**busy["hardware"], "gpu_pcie_gen": 4, "gpu_pcie_gen_max": 4}
        self.assertNotIn("idle, max", self.draw(busy, width=110, height=30))

    def test_the_text_views_use_the_same_rule_as_the_other_screens(self):
        from strata_cli.render import render_view
        snap = {"source": "x", "sampled_at": __import__("time").time(), "model": {"id": "m", "loaded": True}, "work": {}, "rates": {}, "devices": [], "recent": [], "totals": {},
                "topology": {}, "quality": {}, "instance": {}}
        out = render_view(snap, "hardware", width=100, height=24, glyphs="unicode")
        self.assertIn("─" * 40, out)
        self.assertNotIn("-" * 20, out)
        plain = render_view(snap, "hardware", width=100, height=24, glyphs="ascii")
        self.assertIn("-" * 40, plain)

    def test_the_header_keeps_free_and_shortens_the_cards_name(self):
        hw = copy.deepcopy(HW_12)                                                    # (the fixture is shared: a changed name would leak into other tests)
        with Home():
            comp = catalog(hw).computer()
        line = comp.line(width=62)
        self.assertTrue(line.endswith("free"), line)
        self.assertLessEqual(len(line), 62)
        comp.raw["gpus"][0]["name"] = "AMD Radeon(TM) 8060S Graphics With A Very Long Name Indeed"
        comp.usable_gpus[0]["name"] = comp.raw["gpus"][0]["name"]
        line = comp.line(width=62)
        self.assertTrue(line.endswith("free"), line)
        self.assertLessEqual(len(line), 62, line)
        self.assertIn("…", line)
        self.assertEqual(comp.line(), comp.line(width=300))                           # room enough: as before


class StartRowAt80(Base):
    def test_the_preset_row_is_whole_and_the_header_has_free(self):
        store.save(S.LaunchSettings(name="IQ3_XXS 128K", family="qwen", model="IQ3_XXS", context=131072, parallel=1, tuned_preset="x", tuned_at="2026-10-16", last_used="2026-11-12 10:00"))
        store.save(S.LaunchSettings(name="Coder 64K", family="coder", model="IQ1_M", context=65536, parallel=2))

        async def case(app, pilot, ctx):
            t = text_of(app)
            self.assertIn("tuned 10-16", t)
            self.assertIn("not tuned", t)
            for ln in t.splitlines():
                self.assertLessEqual(len(ln), 80, ln)
                self.assertNotIn("not tu ", ln + " ")
            self.assertIn("GB free", t.splitlines()[0])
        small(self, case, make_ctx())

    def test_last_used_is_shown_where_it_fits(self):
        store.save(S.LaunchSettings(name="IQ3_XXS 128K", family="qwen", model="IQ3_XXS", context=131072, parallel=1, tuned_preset="x", tuned_at="2026-10-16", last_used="2026-11-12 10:00"))

        async def case(app, pilot, ctx):
            self.assertIn("last used 11-12", text_of(app))
        small(self, case, make_ctx(), size=(120, 36))


class AdvancedAt80(Base):
    def test_same_as_is_whole_and_the_description_has_room(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "advanced")
            await pilot.pause(0.3)
            t = flat_text(text_of(app))
            self.assertIn("Same as: strata launch --model", t)
            self.assertIn("--port 8080", t)                                           # the end of the line, not cut or scrolled out
            self.assertNotIn("Tuned", t)                                              # the column is left out of a narrow window ...
            self.assertIn("Qwen: 0.2.6 behaviour", t)                                  # ... and the description has its room
            for ln in text_of(app).splitlines():
                self.assertLessEqual(len(ln), 80, ln)
        small(self, case, make_ctx())

    def test_a_wide_window_keeps_the_tuned_column(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "advanced")
            await pilot.pause(0.3)
            self.assertIn("Tuned", text_of(app))
        small(self, case, make_ctx(), size=(120, 40))


class PhaseNames(Base):
    def test_a_finished_phase_stays_named_when_the_next_bar_starts_at_zero(self):
        from strata_cli.launch.runner import Event

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            app.run_launch()
            await pilot.pause(0.3)
            scr = app.run_screen
            scr.handle_event(Event("model", "Preparing the model data for the engine (one time)", 99.0, "info", {}))
            scr.handle_event(Event("model", "Downloading the DFlash2 drafter (1.2 GB, once)…", 0.0, "info", {}))
            scr.handle_event(Event("model", "Downloading the DFlash2 drafter… 0.26 of 1.21 GB", 21.0, "info", {}))
            await pilot.pause(0.3)
            t = text_of(app)
            self.assertIn("Preparing the model data for the engine", t)           # the finished phase, by name
            self.assertIn("Downloading the DFlash2 drafter… 0.26 of 1.21 GB", t)    # the one that runs
            self.assertEqual(t.count("Preparing the model data"), 1)
            app.cancel.set()
        from tests.terminal.test_launch_tui import FakeRunner
        FakeRunner.behaviour = "wait"
        small(self, case, make_ctx())


class ReaderBar(unittest.TestCase):
    def test_which_download_figures_belong_to_the_picture_reader(self):
        from strata_cli.launch.runner import is_reader_file
        self.assertTrue(is_reader_file("vision encoder"))
        self.assertTrue(is_reader_file("mmproj-F16.gguf"))
        self.assertFalse(is_reader_file("qwen3.8-flash-next-q2_0-00001-of-00002.gguf"))     # the model's own line: its 75.8 GB are not the reader's
        self.assertFalse(is_reader_file(None))


def flat_text(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    unittest.main()
