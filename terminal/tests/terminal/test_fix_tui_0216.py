"""0.2.16 first-impression fixes of the Terminal's `strata launch` screens (lane FIX-TUI-0216, T1 to T22 of FIXLIST-0216)."""

from __future__ import annotations

import io
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from strata_cli import overview
from strata_cli.launch import catalog as catalog_mod, cmd, modelstate, runner, screens, settings as S
from strata_cli.launch.app import Ctx, LaunchApp
from strata_cli.launch.runner import DownloadPace, LaunchError, Result
from strata_cli.launch.widgets import Glyphs, Item, Menu
from strata_control import engine_update, runtime
from tests.terminal.launch_support import HW_12, HW_2X16, Home, catalog, text_of
from tests.terminal.test_launch_tui import Base, FakeRunner, flat, make_ctx, press
from tests.terminal.test_overview import draw, metrics

MAC = {**HW_12, "os": "macOS", "mac": True, "ram_gb": 128.0,
       "gpus": [{"index": 0, "name": "Apple M5 Max", "vendor": "apple", "unified_gb": 128.0, "usable": True}]}


class T1StartLine(unittest.TestCase):
    def test_no_update_line_before_any_engine_is_installed(self):
        with mock.patch.object(engine_update.ec, "cached_offers", return_value={"engine": "0.1.62"}), \
                mock.patch.object(engine_update.ec, "newest_installed_version", return_value=None):
            self.assertIsNone(engine_update.notice())
            self.assertIsNone(engine_update.notice_line())

    def test_the_line_stays_for_an_installed_older_engine(self):
        with mock.patch.object(engine_update.ec, "cached_offers", return_value={"engine": "0.1.62"}), \
                mock.patch.object(engine_update.ec, "newest_installed_version", return_value="0.1.60"):
            self.assertIn("A new engine 0.1.62 (you have 0.1.60) is available", engine_update.notice_line())


class T3ParallelOnAMac(Base):
    def test_a_mac_is_not_told_it_has_no_graphics_card(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "enter", "enter")
            t = flat(app)
            self.assertNotIn("No graphics card was found", t)
            self.assertIn("Fit to this computer (1)", t)
            self.assertIn("use the same memory as everything else", t)
        self.run_app(case, ctx=make_ctx(MAC))

    def test_no_card_at_all_keeps_the_old_words(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "enter", "enter")
            self.assertIn("No graphics card was found", flat(app))
        self.run_app(case, ctx=make_ctx({**HW_12, "gpus": []}))


class T4Columns(unittest.TestCase):
    def test_a_name_that_fills_its_column_never_touches_the_next_one(self):
        m = Menu([("Model", 41), ("Size", 8), ("Needs", 7), ("Fits?", 0)], glyphs=Glyphs(True))
        title = "Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS-v31"                  # 42 characters: wider than the column
        m.items, m.index = [Item([title, "94 GB", "65 GB", "fits"])], 5          # (no row selected: nothing needs the app)
        line = m.build().plain.splitlines()[1]
        self.assertNotIn("v3194 GB", line)
        self.assertRegex(line, r"…\s+94 GB")

    def test_the_last_cell_of_a_row_is_left_alone(self):
        m = Menu([("", 7), ("", 0)], glyphs=Glyphs(True))
        m.items, m.index = [Item(["Custom…"])], 5
        self.assertIn("Custom…", m.build().plain)


class T4ModelList(Base):
    def test_the_model_list_widens_its_name_column_to_the_longest_name(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            widths = app.screen.widths()
            longest = max(len(r.title) for r in ctx.rows)
            self.assertGreaterEqual(widths[0][1], min(longest + 2, 52))
            self.assertNotRegex(text_of(app), r"UD-IQ4_XS-v31\d\d GB")
        self.run_app(case)


class T7StepCounter(Base):
    def test_no_total_on_the_model_step_where_the_model_decides_the_steps(self):
        hw = mock.Mock(can_run=True)
        with mock.patch("strata_control.drafting.hardware_support", return_value=hw), mock.patch("strata_control.drafting.ENGINE_SERVES_DFLASH2", True):
            self.assertTrue(screens.total_depends_on_model(make_ctx()))

            async def case(app, pilot, ctx):
                await press(pilot, "enter")
                t = text_of(app)
                self.assertIn("Step 1: Model", t)
                self.assertNotIn("Step 1 of", t)
            self.run_app(case)

    def test_the_total_stays_where_nothing_changes_it(self):
        with mock.patch("strata_control.drafting.ENGINE_SERVES_DFLASH2", False):
            self.assertFalse(screens.total_depends_on_model(make_ctx()))

            async def case(app, pilot, ctx):
                await press(pilot, "enter")
                self.assertIn("Step 1 of 9: Model", text_of(app))
            self.run_app(case)


class T8CardRoles(unittest.TestCase):
    def test_two_cards_show_their_roles_in_the_summary_and_the_copyable_line(self):
        with Home():
            ctx = make_ctx(HW_2X16)
            ctx.settings.family, ctx.settings.model = "qwen", "IQ3_S"
            line = ctx.cards_line()
            self.assertIn("Main: GPU", line)
            self.assertIn("Helper: GPU", line)
            self.assertIn("--gpu-roles ", ctx.flags_line())
            sv = screens.SaveScreen()
            with mock.patch.object(screens.SaveScreen, "ctx", ctx, create=True):
                keys = [k for k, _ in sv.summary_lines()]
            self.assertIn("Cards", keys)

    def test_one_card_adds_nothing(self):
        with Home():
            ctx = make_ctx()
            self.assertIsNone(ctx.cards_line())
            self.assertNotIn("--gpu-roles", ctx.flags_line())


class T9FreeSpace(unittest.TestCase):
    def test_the_header_reads_the_free_disk_again(self):
        with Home():
            ctx = make_ctx()
            with mock.patch.object(type(ctx.catalog), "disk_free", return_value=88.0):
                self.assertIn("88 GB free", ctx.computer_line())
            with mock.patch.object(type(ctx.catalog), "disk_free", return_value=26.0):
                line = ctx.computer_line()
            self.assertIn("26 GB free", line)
            self.assertNotIn("88 GB free", line)


class T10DiskInTheFit(unittest.TestCase):
    def rows(self, free):
        with Home():
            cat = catalog({**HW_12, "disk_free_gb": free})
            return {r.model: r for r in cat.rows(cat.computer().raw)}

    def test_a_model_that_does_not_fit_the_free_disk_says_so(self):
        rows = self.rows(88.0)
        big = rows["IQ3_S"]                                                   # 84 GB download + 8 GB room > 88 GB
        self.assertEqual(big.verdict, "disk")
        self.assertEqual(big.words, "not enough disk space")
        self.assertIn("88 GB are free", big.sentence)
        self.assertIn("about 92 GB", big.sentence)
        self.assertEqual(rows["Q2_0"].verdict, "fits")                        # 66 GB + 8 GB fits 88 GB

    def test_with_room_the_verdicts_are_as_before(self):
        self.assertTrue(all(r.verdict != "disk" for r in self.rows(5000.0).values()))

    def test_an_unknown_free_space_changes_nothing(self):
        with Home():
            cat = catalog(HW_12)
            hw = {k: v for k, v in cat.computer().raw.items() if k != "disk_free_gb"}
            rows = cat.rows(hw)
            self.assertTrue(all(r.verdict != "disk" for r in rows))

    def test_a_bigger_drive_named_on_the_folder_step_clears_the_warning(self):
        with Home():
            ctx = make_ctx({**HW_12, "disk_free_gb": 88.0})
            ctx.settings.family, ctx.settings.model = "qwen", "IQ3_S"
            self.assertTrue([w for w in ctx.warnings() if "Not enough disk space" in w])
            ctx.settings.data_dir = "D:/models"
            with mock.patch.object(type(ctx.catalog), "disk_free", return_value=900.0):
                self.assertFalse([w for w in ctx.warnings() if "Not enough disk space" in w])

class T10List(Base):
    def test_the_model_list_marks_it(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            self.assertIn("not enough disk space", text_of(app))
        self.run_app(case, ctx=make_ctx({**HW_12, "disk_free_gb": 88.0}))


class T11Port(Base):
    def test_next_free_port_skips_a_port_something_holds(self):
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        try:
            p = held.getsockname()[1]
            self.assertTrue(S.port_in_use(p))
            found = S.next_free_port(p)
            self.assertIsNotNone(found)
            self.assertGreater(found, p)
            self.assertFalse(S.port_in_use(found))
        finally:
            held.close()

    def test_the_error_screen_offers_the_next_port_and_u_starts_on_it(self):
        calls = []

        class Busy(FakeRunner):
            def run(self, start=True):
                calls.append(self.s.port)
                if len(calls) == 1:
                    raise LaunchError(f"Port {self.s.port} is already in use by another program.", "Close it, or start with a different port.", 6)
                return super().run(start)

        async def case(app, pilot, ctx):
            ctx.settings.port = 8080
            with mock.patch.object(S, "next_free_port", return_value=8081):
                await press(pilot, *("enter",) * 8, "n", "enter")
                await pilot.pause(0.6)
                t = flat(app)
                self.assertIn("Port 8080 is already in use", t)
                self.assertIn("Press u to use port 8081", t)
                await press(pilot, "u")
                await pilot.pause(0.8)
            self.assertEqual(calls, [8080, 8081])
            self.assertEqual(ctx.settings.port, 8081)
            self.assertIsNotNone(app.result)
        with mock.patch("strata_cli.launch.app.Runner", Busy):
            self.run_app(case)


class T12LastPieces(unittest.TestCase):
    def test_the_end_of_a_slow_download_is_named(self):
        now = [0.0]
        pace = DownloadPace(clock=lambda: now[0])
        for i in range(10):                                                   # 0.4 MB/s for a minute: 24 MB in 60 s
            pace.add(80.0 + i * 0.0027)
            now[0] += 6.5
        text = pace.finishing({"total_gb": 84.0, "done_gb": 83.0, "percent": 98.8})
        self.assertIn("Finishing the last pieces (", text)
        self.assertIn("MB left", text)
        self.assertRegex(text, r"[0-9.]+ MB/s")

    def test_a_stall_says_so_and_the_middle_of_a_download_says_nothing(self):
        now = [0.0]
        pace = DownloadPace(clock=lambda: now[0])
        pace.add(83.0)
        now[0] += 80.0                                                        # nothing arrived for 80 s
        self.assertIn("waiting for data", pace.finishing({"total_gb": 84.0, "done_gb": 83.0, "percent": 98.8}))
        self.assertEqual(pace.finishing({"total_gb": 84.0, "done_gb": 40.0, "percent": 47.6}), "")
        self.assertEqual(pace.finishing({"total_gb": 0.0, "done_gb": 0.0, "percent": 0}), "")


class T14PicturesOnArc(Base):
    def test_arc_gets_no_yes_and_says_why(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter", "enter", "enter", "enter")
            t = flat(app)
            self.assertIn("not available on Intel Arc cards yet", t)
            self.assertEqual(app.screen.menu().index, 0)
            await press(pilot, "down", "enter")
            self.assertEqual(ctx.settings.vision, "off")
        with mock.patch.object(catalog_mod.Catalog, "pictures_not_on_card", return_value=True):
            self.run_app(case)

    def test_the_rule_is_the_intel_engine_on_linux_only(self):
        with Home():
            cat = catalog()
            with mock.patch.object(catalog_mod.Catalog, "engine_backend", return_value="sycl"):
                with mock.patch.object(sys, "platform", "linux"):
                    self.assertTrue(cat.pictures_not_on_card())
                with mock.patch.object(sys, "platform", "win32"):
                    self.assertFalse(cat.pictures_not_on_card())
            with mock.patch.object(catalog_mod.Catalog, "engine_backend", return_value="cuda"):
                with mock.patch.object(sys, "platform", "linux"):
                    self.assertFalse(cat.pictures_not_on_card())


class T15OfflinePictures(unittest.TestCase):
    def test_a_failed_picture_reader_download_says_how_to_go_on(self):
        err = "This computer could not connect to huggingface.co (it looks offline), so the picture reader could not be downloaded."
        e = runner.setup_failure([], err, "Check that this computer can open https://huggingface.co.", None, False)
        self.assertIn("--vision off", e.next)
        self.assertIn("Pictures step", e.next)
        self.assertIn("mmproj", e.next)

    def test_other_failures_are_unchanged(self):
        e = runner.setup_failure([], "No space left on the disk", "", None, False)
        self.assertNotIn("--vision off", e.next)


class T16NotNow(Base):
    class Missing:
        only_helpers = True

        def __init__(self):
            self.parts = [modelstate.Part("mtp", "the draft layer", 5.2)]

    def test_not_now_for_the_draft_layer_starts_without_it(self):
        class Asks(FakeRunner):
            download = 5.2

            def download_gb(self):
                return 5.2

            def model_state(self):
                return T16NotNow.Missing()

            def download_question(self):
                return ("Download the draft layer (5.2 GB)?", "The model itself is already on this computer.")

        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 8, "n", "enter")
            await pilot.pause(0.3)
            t = flat(app)
            self.assertIn("Download the draft layer (5.2 GB)?", t)
            self.assertIn("Start without the draft layer", t)
            await press(pilot, "down", "enter")                               # the second choice
            await pilot.pause(0.8)
            self.assertEqual(ctx.settings.mtp, "off")
            self.assertIsNotNone(app.result)                                  # it launched: it did not go back to the Save screen
        with mock.patch("strata_cli.launch.app.Runner", Asks):
            self.run_app(case)

    def test_a_whole_model_download_keeps_not_now_and_goes_back(self):
        class Asks(FakeRunner):
            download = 80.0

            def download_gb(self):
                return 80.0

            def model_state(self):
                m = T16NotNow.Missing()
                m.only_helpers = False
                m.parts = [modelstate.Part("weights", "the model", 80.0)]
                return m

        async def case(app, pilot, ctx):
            await press(pilot, *("enter",) * 8, "n", "enter")
            await pilot.pause(0.3)
            self.assertIn("Not now", flat(app))
            await press(pilot, "down", "enter")
            await pilot.pause(0.3)
            self.assertIsNone(app.result)
            self.assertIn("Save as preset, then launch", text_of(app))
        with mock.patch("strata_cli.launch.app.Runner", Asks):
            self.run_app(case)


class T18NoScreens(unittest.TestCase):
    def test_git_bash_is_only_named_on_windows(self):
        self.assertIn("Git Bash", cmd.no_screens_words("win32"))
        for platform, app in (("darwin", "Terminal app"), ("linux", "normal terminal window")):
            text = cmd.no_screens_words(platform)
            self.assertNotIn("Git Bash", text)
            self.assertNotIn("winpty", text)
            self.assertIn(app, text)


class T19UnifiedMemory(unittest.TestCase):
    def test_a_gb10_shows_the_system_memory_and_no_pcie_link(self):
        m = metrics("idle")
        m["hardware_static"] = {"gpu_name": "NVIDIA GB10", "gpu_count": 1, "cores": 20, "threads": 20}
        m["hardware"] = {**m["hardware"], "gpu_mem_used": None, "gpu_mem_total": None, "gpu_pcie_gen": 1, "gpu_pcie_width": 1, "gpu_pcie_width_max": 16}
        text = draw(overview.build(m))
        self.assertNotIn("-- / -- GB", text)
        self.assertNotIn("Gen1 x1 of x16", text)
        self.assertIn("shared with graphics", text)
        self.assertIn("not used", text)

    def test_a_card_with_memory_of_its_own_is_unchanged(self):
        text = draw(overview.build(metrics("idle")))
        self.assertIn("VRAM", text)
        self.assertIn("Gen4 x16", text)

    def test_which_chips_share_memory(self):
        for name in ("Apple M5 Max", "NVIDIA GB10", "AMD Radeon 8060S"):
            self.assertTrue(overview.shares_memory(name), name)
        for name in ("RTX 5070", "RX 7900 XTX", None):
            self.assertFalse(overview.shares_memory(name), name)


class T21HelperGone(unittest.TestCase):
    def test_the_words_are_plain_and_have_a_next_step(self):
        self.assertNotIn("inspect the private host log", runtime.HELPER_GONE)
        self.assertNotIn("runtime control", runtime.HELPER_GONE)
        with Home():
            r = runner.Runner(S.LaunchSettings(name="x"), catalog=catalog())
            nxt = r.helper_gone_next(runtime.ProfileError(runtime.HELPER_GONE, 3))
            self.assertIn("strata launch", nxt)
            self.assertIn("strata doctor", nxt)
            self.assertIn(".host.log", nxt)
            self.assertIsNone(r.helper_gone_next(ValueError("something else")))

    def test_the_host_log_gets_the_reason_when_it_was_empty(self):
        with Home() as home:
            log = Path(home.base) / "runtime" / "demo.host.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("")
            with mock.patch.object(runtime, "_paths", return_value=(None, None, log)):
                runtime._note_host_gone("demo", runtime.ProfileError(runtime.HELPER_GONE, 3))
                runtime._note_host_gone("demo", runtime.ProfileError("something else", 3))     # not ours: no line
            text = log.read_text(encoding="utf-8")
            self.assertEqual(text.count("the launcher lost the host"), 1)
            self.assertIn("[strata runtime] profile demo", text)


class T22MacFirstStart(unittest.TestCase):
    def test_only_a_fresh_engine_on_a_mac_gets_the_note(self):
        with Home():
            r = runner.Runner(S.LaunchSettings(name="x"), catalog=catalog(MAC))
            with mock.patch.object(sys, "platform", "darwin"):
                self.assertFalse(r.mac_first_start())
                r.engine_fresh = True
                self.assertTrue(r.mac_first_start())
            with mock.patch.object(sys, "platform", "linux"):
                self.assertFalse(r.mac_first_start())
        self.assertIn("10 seconds", runner.MAC_FIRST_ANSWER)


if __name__ == "__main__":
    unittest.main()
