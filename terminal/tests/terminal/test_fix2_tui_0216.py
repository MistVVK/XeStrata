"""0.2.16 round 2: the Terminal's retest findings (lane FIX2-TUI, FIXLIST2-0216: N1, T10b, S1 to S12). Screen items are checked at 80 x 24."""

from __future__ import annotations

import asyncio
import io
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from strata_cli import overview, render
from strata_cli.launch import cmd, runner, screens, settings as S
from strata_cli.launch.app import Ctx, LaunchApp
from strata_cli.launch.runner import Runner
from strata_control import drafting, drafting_fetch
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli, text_of
from tests.terminal.test_fix_tui_0216 import MAC
from tests.terminal.test_launch_tui import Base, flat, make_ctx, press, typed
from tests.terminal.test_overview import metrics


def keyed(rows):
    return {(r.family, r.model): r for r in rows}


def small(testcase, coro, ctx, size=(80, 24)):
    """Run `coro(app, pilot, ctx)` in a window of `size` (default 80 x 24, a default terminal window)."""
    app = LaunchApp(ctx)

    async def go():
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.2)
            return await coro(app, pilot, ctx)
    return asyncio.run(go())


async def goto(pilot, app, step, limit=14):
    """Press Enter until the screen of `step` is in front (the Model step first)."""
    for _ in range(limit):
        if getattr(app.screen, "step_id", None) == step:
            return
        await press(pilot, "enter")
    raise AssertionError(f"never reached {step}: {getattr(app.screen, 'step_id', None)}")


# ------------------------------------------------------------------------------------------------ N1 and T10b

class N1OwnFilesNeedNoDownload(unittest.TestCase):
    def rows(self, free, own=None, hw=HW_12):
        with Home():
            cat = catalog({**hw, "disk_free_gb": free})
            return keyed(cat.rows(cat.computer().raw, own=own))

    def test_the_model_in_the_own_folder_is_not_marked_and_the_others_still_count_what_they_download(self):
        rows = self.rows(30.0, own=("qwen", "IQ3_S"))
        self.assertEqual(rows[("qwen", "IQ3_S")].verdict, "fits")
        self.assertEqual(rows[("qwen", "Q2_0")].verdict, "disk")            # 66 GB + 8 GB room > 30 GB: a real download
        self.assertEqual(rows[("qwen", "Q2_0")].words, "not enough disk space")

    def test_without_own_files_the_row_is_marked_as_before(self):
        self.assertEqual(self.rows(30.0)[("qwen", "IQ3_S")].verdict, "disk")

    def test_the_marks_come_off_again_when_the_choice_changes(self):
        with Home():
            cat = catalog({**HW_12, "disk_free_gb": 30.0})
            rows = cat.rows(cat.computer().raw)
            hw = cat.computer().raw
            self.assertEqual(keyed(rows)[("qwen", "IQ3_S")].verdict, "disk")
            cat.apply_disk(rows, hw, own=("qwen", "IQ3_S"))
            self.assertEqual(keyed(rows)[("qwen", "IQ3_S")].verdict, "fits")
            cat.apply_disk(rows, hw, own=None)
            self.assertEqual(keyed(rows)[("qwen", "IQ3_S")].verdict, "disk")
            self.assertEqual(keyed(rows)[("qwen", "IQ3_S")].mem_verdict, "fits")

    def test_the_save_screen_line_and_the_plan_notice_are_gone_with_own_files(self):
        with Home():
            ctx = make_ctx({**HW_12, "disk_free_gb": 30.0})
            ctx.settings.family, ctx.settings.model = "qwen", "IQ3_S"
            self.assertTrue([w for w in ctx.warnings() if "Not enough disk space" in w])
            ctx.settings.model_path = "/somewhere/my-model"                 # (the rows were marked before the folder was chosen)
            self.assertFalse([w for w in ctx.warnings() if "Not enough disk space" in w])

    def test_a_flag_launch_with_own_files_has_no_disk_notice(self):
        with Home():
            cat = catalog({**HW_12, "disk_free_gb": 30.0})
            plain = S.LaunchSettings(family="qwen", model="IQ3_S", context=32768)
            _s, notes = cmd.apply_defaults(plain, cat)
            self.assertTrue([n for n in notes if "Not enough disk space" in n.line()])
            own = S.LaunchSettings(family="qwen", model="IQ3_S", context=32768, model_path="/somewhere/my-model")
            _s, notes = cmd.apply_defaults(own, cat)
            self.assertFalse([n for n in notes if "Not enough disk space" in n.line()])


class N1ModelList(Base):
    def test_choosing_own_files_clears_the_mark_on_the_model_list(self):
        plan = {"candidate_selection": {"family": "qwen", "model": "IQ3_S"}, "issues": [], "total_bytes": 84e9, "file_count": 3}

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            self.assertIn("not enough disk space", text_of(app))
            with mock.patch("strata_control.models.own_path_problem", return_value=None), \
                    mock.patch("strata_control.retired.folder_reason", return_value=None), \
                    mock.patch("strata_control.models.plan_import", return_value=plan), \
                    mock.patch("strata_control.models.import_target", side_effect=lambda path, _plan: path):
                await press(pilot, "o")
                await typed(pilot, "/somewhere/my-model")
                await press(pilot, "enter")
            line = next(l for l in text_of(app).splitlines() if "Qwen3.8-Flash-Next IQ3_S" in l)
            self.assertNotIn("not enough disk space", line)
            self.assertFalse([w for w in ctx.warnings() if "Not enough disk space" in w])
        self.run_app(case, ctx=make_ctx({**HW_12, "disk_free_gb": 30.0}))


class T10bWhateverTheMemoryVerdict(unittest.TestCase):
    def rows(self, free):
        with Home():
            cat = catalog({**HW_12, "disk_free_gb": free})
            return keyed(cat.rows(cat.computer().raw))

    def test_a_slow_model_that_does_not_fit_the_disk_says_both(self):
        slow = self.rows(100.0)[("unsloth", "UD-Q5_K_XL")]                   # 158 GB on a 12 GB card with 64 GB RAM: may run slower
        self.assertEqual(self.rows(5000.0)[("unsloth", "UD-Q5_K_XL")].verdict, "slower")
        self.assertEqual(slow.verdict, "disk")
        self.assertEqual(slow.mem_verdict, "slower")
        self.assertIn("not enough disk space", slow.words)
        self.assertIn("Not enough disk space: about 166 GB needed, 100 GB free", slow.sentence)
        self.assertIn("may also run slower", slow.sentence)

    def test_a_model_that_is_too_big_for_the_computer_keeps_only_that(self):
        with Home():
            cat = catalog({**HW_12, "disk_free_gb": 100.0})
            rows = cat.rows(cat.computer().raw)
            row = rows[0]
            row.verdict, row.words = "no", "too big"
            row.installed = False
            cat.apply_disk(rows, cat.computer().raw)
            self.assertEqual((row.verdict, row.words, row.mem_verdict), ("no", "too big", ""))

    def test_the_without_disk_row_is_the_memory_verdict_alone(self):
        slow = self.rows(100.0)[("unsloth", "UD-Q5_K_XL")].without_disk()
        self.assertEqual((slow.verdict, slow.words, slow.mem_verdict), ("slower", "may run slower", ""))

class T10bList(Base):
    def test_the_narrow_list_says_no_disk_for_it(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            self.assertIn("no disk", next(l for l in text_of(app).splitlines() if "UD-Q5_K_XL" in l))
        small(self, case, make_ctx({**HW_12, "disk_free_gb": 100.0}), size=(80, 40))

    def test_the_wide_list_says_both(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            line = next(l for l in text_of(app).splitlines() if "UD-Q5_K_XL" in l)
            self.assertIn("not enough disk space", line)
            app.screen.menu().select(("unsloth", "UD-Q5_K_XL"))
            self.assertIn("may also run slower", app.screen.explain().plain)
        self.run_app(case, ctx=make_ctx({**HW_12, "disk_free_gb": 100.0}))


# ------------------------------------------------------------------------------------------------ S1

class S1StepCounter(Base):
    def test_step_one_shows_the_total_of_the_model_under_the_cursor(self):
        hw = mock.Mock(can_run=True)
        with mock.patch("strata_control.drafting.hardware_support", return_value=hw), mock.patch("strata_control.drafting.ENGINE_SERVES_DFLASH2", True):
            async def case(app, pilot, ctx):
                await press(pilot, "enter")
                menu = app.screen.menu()
                first = app.screen.head_text().plain
                self.assertRegex(first, r"Step 1 of 10: Model")                    # qwen has a Drafting step
                other = next((it for it in menu.items if it.key[0] not in drafting.SUPPORTED_FAMILIES), None)
                self.assertIsNotNone(other, "the list has a family without a Drafting step")
                menu.select(other.key)
                app.screen.paint()
                self.assertIn("Step 1 of 9: Model", app.screen.head_text().plain)
                menu.select(next(it for it in menu.items if it.key[0] == "qwen").key)
                app.screen.paint()
                await press(pilot, "enter")
                self.assertIn("Step 2 of 10", text_of(app))                         # the next step says the same total: steady
            self.run_app(case)

    def test_the_flow_follows_the_family_asked_for(self):
        hw = mock.Mock(can_run=True)
        with mock.patch("strata_control.drafting.hardware_support", return_value=hw), mock.patch("strata_control.drafting.ENGINE_SERVES_DFLASH2", True):
            ctx = make_ctx()
            self.assertIn("drafting", screens.flow_for(ctx, "qwen"))
            self.assertNotIn("drafting", screens.flow_for(ctx, "swift"))


# ------------------------------------------------------------------------------------------------ S2

class S2Nothing80ColumnsCut(Base):
    """80 x 24: what a screen says is wrapped or shortened, never cut at the window's edge."""

    def test_the_tuning_rows_are_whole(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "tuning")
            t = flat(app)
            self.assertIn("tries more settings, measures each one several times", t)
            self.assertIn("use the default settings (already good on most computers)", t)
        small(self, case, make_ctx())

    def test_the_access_rows_are_whole(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "access")
            t = flat(app)
            for option in app.screen.opts.values():
                self.assertIn(option.note, t, option.kind)
        small(self, case, make_ctx())

    def test_the_advanced_rows_that_do_not_fit_end_with_an_ellipsis(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "advanced")
            lines = [l for l in text_of(app).splitlines() if l.lstrip(" ›").startswith("STRATA_")]
            self.assertTrue(lines)
            for line in lines:
                self.assertLessEqual(len(line), 80)
            self.assertTrue(any(l.endswith("…") for l in lines), lines)
        small(self, case, make_ctx())

    def test_the_list_keeps_the_mark_of_the_recommendation_when_a_scrollbar_takes_two_cells(self):
        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            line = next(l for l in text_of(app).splitlines() if "›" in l and "IQ3_S" in l)
            self.assertRegex(line, r"fits \((recommended|rec\.)\)$")
        small(self, case, make_ctx(), size=(78, 24))

    def draw(self, width, **over):
        m = metrics("idle")
        m["engine"] = {**m["engine"], "model": "qwen3.8-flash-next-q2_0", "max_context": 65536}
        c = Console(width=width + 2, record=True, file=io.StringIO(), force_terminal=False, legacy_windows=False)
        c.print(overview.render(overview.build(m), width=width, height=30, connection="http://127.0.0.1:19883", **over))
        return c.export_text().splitlines()

    def test_the_monitors_top_line_keeps_the_state_and_its_marker(self):
        for width in (78, 80, 70):
            first = self.draw(width)[0]
            self.assertIn("[LIVE]", first, width)
            self.assertIn("Idle", first, width)
            self.assertLessEqual(len(first.rstrip()), width + 2, first)
            self.assertIn("19883", first)

    def test_the_wide_top_line_is_as_before(self):
        first = self.draw(110)[0]
        self.assertIn("Waiting for a request", first)
        self.assertIn("qwen3.8-flash-next-q2_0 64K", first)

    def test_the_ascii_top_line_fits_too(self):
        first = self.draw(78, glyphs="ascii")[0]
        self.assertIn("[LIVE]", first)


# ------------------------------------------------------------------------------------------------ S3

class S3SharedMemoryParallel(Base):
    def test_a_mac_gets_a_figure_from_its_shared_memory(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "parallel")
            t = flat(app)
            self.assertIn("use the same memory as everything else", t)
            self.assertRegex(t, r"\d+ GB in all; the model takes about \d+ GB")
            self.assertRegex(t, r"each further request takes about \d\.\d GB")
            self.assertRegex(t, r"\+\d\.\d GB more")
            self.assertIn("Fit to this computer (1)", t)                        # the default stays one request
        self.run_app(case, ctx=make_ctx(MAC))

    def test_a_small_shared_memory_marks_the_choices_that_do_not_fit(self):
        small_mac = {**MAC, "ram_gb": 24.0, "gpus": [{**MAC["gpus"][0], "unified_gb": 24.0}]}

        async def case(app, pilot, ctx):
            await goto(pilot, app, "parallel")
            room = app.screen.shared_room()
            self.assertIsNotNone(room)
            self.assertEqual(room["more"], 0)
            self.assertIn("may not fit: slower answers", flat(app))
            self.assertIn("Not much is left", flat(app))
        self.run_app(case, ctx=make_ctx(small_mac))

    def test_no_figure_without_the_memory_keeps_the_t3_words(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "parallel")
            with mock.patch.object(type(app.screen), "shared_room", return_value=None):
                self.assertIn("cannot count graphics memory", app.screen.explain().plain)
        self.run_app(case, ctx=make_ctx(MAC))


# ------------------------------------------------------------------------------------------------ S4 and S5

class FakeDrafterRunner:
    """What ensure_drafter reads: the drafter's inputs, and whether the files are on the computer."""

    def __init__(self, test: unittest.TestCase, *, stock_missing: bool, tuned_missing: bool, tuned: bool, ready: bool, fetched):
        self.test = test
        self.events: list[str] = []
        self.stock_missing, self.tuned_missing, self.tuned, self.ready, self.fetched = stock_missing, tuned_missing, tuned, ready, fetched

    def run(self) -> list[str]:
        want = {"data_dir": "D", "mode": "auto", "model_name": "m", "backend": "cuda", "archs": [], "family": "qwen", "klass": "other", "mtp_here": True, "args": []}
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0"), catalog=catalog(HW_12), emit=lambda e: self.events.append(e.text))
        with mock.patch.object(runner, "read_config", return_value={}), mock.patch.object(Runner, "drafting_inputs", return_value=want), \
                mock.patch.object(Runner, "drafting_signals", return_value={}), mock.patch.object(drafting, "needs_conversion", return_value=False), \
                mock.patch.object(drafting, "needs_pack", return_value=self.stock_missing), mock.patch.object(drafting, "needs_tuned", return_value=self.tuned_missing), \
                mock.patch.object(drafting, "tuned_applies", return_value=self.tuned), mock.patch.object(drafting, "drafter_ready", return_value=self.ready), \
                mock.patch.object(drafting_fetch, "ensure_for", return_value=self.fetched):
            r.ensure_drafter(Path("cfg.json"))
        return self.events


class S4DrafterWords(unittest.TestCase):
    def test_a_drafter_that_is_here_is_ready_with_no_download_line_and_no_size(self):
        with Home():
            events = FakeDrafterRunner(self, stock_missing=False, tuned_missing=True, tuned=True, ready=True, fetched=None).run()
        self.assertEqual(events, ["The DFlash2 drafter is ready.", "The DFlash2 drafter is ready."])
        self.assertFalse([e for e in events if "Downloading" in e or " GB" in e])

    def test_a_drafter_that_is_missing_names_its_size_once(self):
        with Home():
            events = FakeDrafterRunner(self, stock_missing=True, tuned_missing=False, tuned=False, ready=True, fetched=Path("x")).run()
        self.assertEqual(events[0], "Downloading the DFlash2 drafter (1.2 GB, once)…")
        self.assertEqual(events[-1], "The DFlash2 drafter is ready.")

    def test_nothing_is_called_ready_when_it_is_not_there(self):
        with Home():
            events = FakeDrafterRunner(self, stock_missing=True, tuned_missing=False, tuned=False, ready=False, fetched=None).run()
        self.assertNotIn("The DFlash2 drafter is ready.", events)

    def test_the_fetch_command_prints_no_figures_for_a_drafter_that_was_here(self):
        def fake(data, *, progress=None, **_k):
            progress("The DFlash2 drafter is ready.", 1, 1)
            return Path(data) / "x.pack"
        with Home() as home, mock.patch.object(drafting_fetch, "ensure_for", side_effect=fake):
            code, out, _err = run_cli(["drafting", "fetch", "--data-dir", str(home.base / "d")])
        self.assertEqual(code, 0)
        self.assertIn("The DFlash2 drafter is ready.", out)
        self.assertNotIn("0.00 of 0.00", out)

    def test_a_real_download_still_shows_its_figures(self):
        def fake(data, *, progress=None, **_k):
            progress("Downloading the DFlash2 drafter…", 600_000_000, 1_200_000_000)
            return Path(data) / "x.pack"
        with Home() as home, mock.patch.object(drafting_fetch, "ensure_for", side_effect=fake):
            _code, out, _err = run_cli(["drafting", "fetch", "--data-dir", str(home.base / "d")])
        self.assertIn("0.60 of 1.20 GB", out)


class S5SaveListsWhatIsFetched(Base):
    def lines(self, parts, engine_usable=True):
        plan = {"gb": sum(p["gb"] for p in parts), "parts": parts, "engine_usable": engine_usable, "engine_missing": not engine_usable, "own_files": False, "text": ""}
        got = []

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            got.extend([app.screen.download_lines(), app.screen.download_lines(short=True)])
        with mock.patch.object(Ctx, "download_plan", return_value=plan):
            self.run_app(case)
        return got

    def test_the_standard_drafter_says_what_is_downloaded_and_what_stays(self):
        full, short = self.lines([{"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 1.2}])
        self.assertEqual(full, ["the DFlash2 drafter, 1.2 GB (packed to 0.9 GB on disk)"])
        self.assertEqual(short, ["DFlash2 drafter 1.2 GB"])

    def test_the_tuned_drafter_has_one_size(self):
        full, _short = self.lines([{"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 0.34}])
        self.assertEqual(full, ["the DFlash2 drafter, 0.34 GB"])

    def test_an_intel_plan_keeps_its_drafter_and_drops_the_layer_and_reader(self):
        with Home(), mock.patch.object(Runner, "intel_chosen", return_value=True), mock.patch.object(drafting, "needs_pack", return_value=True), \
                mock.patch.object(drafting, "needs_tuned", return_value=False), mock.patch.object(drafting, "tuned_applies", return_value=True), \
                mock.patch.object(drafting, "tuned_variant", return_value="bf16sel"):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="gpu"), catalog=catalog(HW_12))
            kinds = [p["kind"] for p in r.download_plan()["parts"]]
        self.assertIn("dflash2", kinds)
        self.assertNotIn("mmproj", kinds)
        self.assertNotIn("mtp", kinds)


# ------------------------------------------------------------------------------------------------ S6

class S6Offline(Base):
    PARTS = [{"kind": "weights", "label": "the model", "gb": 84.0}, {"kind": "mmproj", "label": "the picture reader", "gb": 0.9},
             {"kind": "dflash2", "label": "the DFlash2 drafter", "gb": 0.34}]

    def test_the_probe_says_offline_only_when_both_hosts_are_silent(self):
        ctx = make_ctx()
        ctx.probe_offline = True
        with mock.patch.object(drafting_fetch, "online", return_value=False):
            ctx.check_offline()
            for _ in range(50):
                if ctx.offline is not None:
                    break
                import time
                time.sleep(0.05)
        self.assertTrue(ctx.offline)
        ctx2 = make_ctx()
        ctx2.probe_offline = True
        with mock.patch.object(drafting_fetch, "online", side_effect=[False, True]):
            ctx2.check_offline()
            for _ in range(50):
                if ctx2.offline is not None:
                    break
                import time
                time.sleep(0.05)
        self.assertFalse(ctx2.offline)

    def test_no_probe_unless_the_real_screens_ask_for_it(self):
        ctx = make_ctx()
        with mock.patch.object(drafting_fetch, "online", side_effect=AssertionError("no network in a test")):
            ctx.check_offline()
        self.assertIsNone(ctx.offline)

    def test_a_folder_of_the_persons_own_is_not_probed_as_a_web_address(self):
        import time
        asked = []
        ctx = make_ctx()
        ctx.probe_offline, ctx.feed = True, "/my/feed"
        with mock.patch.object(drafting_fetch, "online", side_effect=lambda url, timeout=0: asked.append(url) or True):
            ctx.check_offline()
            for _ in range(50):
                if ctx.offline is not None:
                    break
                time.sleep(0.05)
        self.assertFalse(ctx.offline)
        self.assertEqual(asked, [drafting.base_url()])               # only the model files' address: the feed is a folder here

    def test_the_save_screen_says_what_is_missing_and_how_to_put_it_in_place(self):
        plan = {"gb": 85.2, "parts": self.PARTS, "engine_usable": False, "engine_missing": True, "own_files": False, "text": ""}

        async def case(app, pilot, ctx):
            ctx.offline = True
            await goto(pilot, app, "save")
            t = flat(app)
            self.assertIn("offline: nothing can be downloaded now. Missing:", t)
            self.assertIn("the model, 84 GB: put its files in a folder, then choose Use model files I already have", t)
            self.assertIn("the DFlash2 drafter, 0.34 GB: copy both of its .pack files into the dflash2 folder of your data folder", t)
            self.assertIn("the engine for your graphics card: install it from a file", t)
            self.assertNotIn("Strata downloads what is listed under Download first", t)
            self.assertIn("This computer is offline", t)
        with mock.patch.object(Ctx, "download_plan", return_value=plan):
            self.run_app(case)

    def test_online_or_unknown_the_rows_are_the_download_rows(self):
        plan = {"gb": 85.2, "parts": self.PARTS, "engine_usable": True, "engine_missing": False, "own_files": False, "text": ""}

        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            self.assertIn("the model, 84 GB", flat(app))
            self.assertNotIn("offline", flat(app))
        with mock.patch.object(Ctx, "download_plan", return_value=plan):
            self.run_app(case)


# ------------------------------------------------------------------------------------------------ S7

class S7OneTerm(unittest.TestCase):
    def test_the_offline_note_does_not_mix_draft_layer_and_drafter(self):
        note = drafting.MTP_OFFLINE_NOTE
        self.assertIn("draft layer", note)
        self.assertNotIn("drafter", note)

    def test_a_failed_drafter_download_does_not_say_mtp(self):
        with Home():
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0"), catalog=catalog(HW_12), emit=lambda e: None)
            want = {"data_dir": "D", "mode": "auto", "model_name": "m", "backend": "cuda", "archs": [], "family": "qwen", "klass": "other", "mtp_here": True, "args": []}
            with mock.patch.object(runner, "read_config", return_value={}), mock.patch.object(Runner, "drafting_inputs", return_value=want), \
                    mock.patch.object(Runner, "drafting_signals", return_value={}), mock.patch.object(drafting, "needs_conversion", return_value=False), \
                    mock.patch.object(drafting, "needs_pack", return_value=True), mock.patch.object(drafting, "tuned_applies", return_value=False), \
                    mock.patch.object(drafting_fetch, "ensure_for", side_effect=RuntimeError("boom")):
                r.ensure_drafter(Path("cfg.json"))
        self.assertTrue(any("draft layer instead" in n for n in r.notices), r.notices)
        self.assertFalse(any("MTP" in n for n in r.notices), r.notices)


# ------------------------------------------------------------------------------------------------ S8

GB10 = {"index": 0, "name": "NVIDIA GB10", "mem_used": None, "mem_total": None, "util": 40, "temp": 51, "power": 30,
        "pcie_gen": 1, "pcie_width": 1, "pcie_width_max": 16}
RTX = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "mem_used": 6 * 2 ** 30, "mem_total": 12 * 2 ** 30, "util": 40, "temp": 51, "power": 100,
       "pcie_gen": 4, "pcie_width": 16, "pcie_width_max": 16}


class S8HardwareTab(unittest.TestCase):
    def test_a_chip_with_shared_memory_has_no_placeholder_pair_and_no_link(self):
        for width in (60, 80, 110):
            line = render._device(GB10, width)
            self.assertNotIn("--/--", line, width)
            self.assertNotIn("Gen1", line, width)
            self.assertIn("shared", line, width)

    def test_the_detail_pane_says_it_too(self):
        text = "\n".join(render._device_detail(GB10, {}, 100))
        self.assertIn("Memory shared with the computer", text)
        self.assertIn("PCIe not used (memory is shared)", text)
        self.assertNotIn("VRAM", text)
        self.assertNotIn("Gen1", text)

    def test_a_card_with_its_own_memory_is_as_before(self):
        line = render._device(RTX, 110)
        self.assertIn("6.0/12.0 GiB", line)
        self.assertIn("Gen4 x16", line)
        self.assertIn("VRAM 6.0 GiB / 12.0 GiB", "\n".join(render._device_detail(RTX, {}, 100)))

    def test_the_cards_strip_of_the_overview_has_no_dashes_either(self):
        card = {"index": 0, "name": "NVIDIA GB10", "util": 40, "mem_used": None, "mem_total": None, "temp": 51, "power": 30,
                "pcie_gen": 1, "pcie_width": 1, "pcie_width_max": 16, "pcie_gen_max": 1}
        text = overview._cards_strip([card], "unicode", "green", False, 110).plain
        self.assertIn("shared memory", text)
        self.assertNotIn("-- GB", text)
        self.assertNotIn("Gen1", text)


# ------------------------------------------------------------------------------------------------ S9

class S9PlainStorage(Base):
    def test_the_save_screen_says_where_own_files_stay_and_where_strata_keeps_its_data(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "model")
            ctx.settings.model_path = "/home/me/models/iq3"
            await goto(pilot, app, "save")
            lines = dict(app.screen.summary_lines())
            self.assertEqual(lines["Your files"], "/home/me/models/iq3 (used where they are)")
            self.assertIn("(packs and the draft layer go here)", lines["Strata data"])
            self.assertNotIn("Storage", lines)
        self.run_app(case)

    def test_without_own_files_it_is_storage_as_before(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "save")
            lines = dict(app.screen.summary_lines())
            self.assertIn("Storage", lines)
            self.assertNotIn("Your files", lines)
        self.run_app(case)

    def test_the_small_save_screen_still_shows_the_choices(self):
        async def case(app, pilot, ctx):
            await goto(pilot, app, "model")
            ctx.settings.model_path = "/home/me/models/" + "very-long-folder-name/" * 3 + "iq3"
            await goto(pilot, app, "save")
            t = text_of(app)
            self.assertIn("Save as a preset and launch", t)
            for line in t.splitlines():
                self.assertLessEqual(len(line), 80, line)
        small(self, case, make_ctx())


# ------------------------------------------------------------------------------------------------ S10

class S10SubcommandHelp(unittest.TestCase):
    def test_components_without_a_subcommand_lists_them(self):
        code, out, err = run_cli(["components"])
        self.assertEqual(code, 2)
        self.assertIn("strata components needs one of these:", out)
        self.assertIn("verify", out)
        self.assertNotIn("components_command", out + err)
        self.assertNotIn("usage:", err)

    def test_drafting_without_a_subcommand_lists_them(self):
        code, out, err = run_cli(["drafting"])
        self.assertEqual(code, 2)
        self.assertIn("strata drafting needs one of these:", out)
        self.assertIn("plan", out)
        self.assertIn("fetch", out)
        self.assertNotIn("drafting_command", out + err)

    def test_other_usage_errors_are_untouched(self):
        code, out, err = run_cli(["components", "verify"])           # --profile is missing
        self.assertEqual(code, 2)
        self.assertIn("--profile", err)


# ------------------------------------------------------------------------------------------------ S11

class S11FirstAnswerNote(unittest.TestCase):
    def runner(self):
        return Runner(S.LaunchSettings(name="x"), catalog=catalog(MAC))

    def test_an_engine_that_something_else_installed_is_noted_once_on_its_first_start(self):
        with Home() as home:
            exe = home.base / "engine" / "strata"
            exe.parent.mkdir(parents=True)
            exe.write_text("x")
            r = self.runner()
            with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(Runner, "engine_exe_path", return_value=str(exe)):
                self.assertTrue(r.mac_first_start())
                self.assertFalse(r.mac_first_start())                   # said once
                exe.write_text("a new engine")
                self.assertTrue(r.mac_first_start())                    # another engine file: its graphics code is built again
            with mock.patch.object(sys, "platform", "linux"), mock.patch.object(Runner, "engine_exe_path", return_value=str(exe)):
                self.assertFalse(self.runner().mac_first_start())

    def test_no_engine_found_says_nothing(self):
        with Home():
            with mock.patch.object(sys, "platform", "darwin"), mock.patch.object(Runner, "engine_exe_path", return_value=""):
                self.assertFalse(self.runner().mac_first_start())


# ------------------------------------------------------------------------------------------------ S12

class S12PortAtTheAccessStep(Base):
    def test_a_taken_port_is_said_with_the_next_free_one_and_u_takes_it(self):
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        try:
            async def case(app, pilot, ctx):
                await goto(pilot, app, "context")
                ctx.settings.port = port
                await goto(pilot, app, "access")
                t = flat(app)
                self.assertIn(f"Port {port} is already in use by another program", t)
                self.assertIn("Press u to use port", t)
                free = app.screen.free_port
                self.assertIsNotNone(free)
                self.assertNotEqual(free, port)
                await press(pilot, "u")
                self.assertEqual(ctx.settings.port, free)
                self.assertIsNone(app.screen.free_port)
                self.assertIn(f"[ {free} ]", flat(app))
            self.run_app(case)
        finally:
            held.close()

    def test_a_free_port_says_nothing(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        async def case(app, pilot, ctx):
            await goto(pilot, app, "context")
            ctx.settings.port = port
            await goto(pilot, app, "access")
            self.assertNotIn("already in use", flat(app))
            self.assertIsNone(app.screen.free_port)
        self.run_app(case)

    def test_the_screen_only_ever_looks_at_this_computer(self):
        seen = []

        def look(port, host="127.0.0.1"):
            seen.append(host)
            return False

        async def case(app, pilot, ctx):
            await goto(pilot, app, "context")
            ctx.settings.access, ctx.settings.listen = "address", "192.168.1.5"
            await goto(pilot, app, "access")
            await press(pilot, "p")
        with mock.patch.object(S, "port_in_use", look):
            self.run_app(case)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {"127.0.0.1"})

    def test_a_server_of_this_launch_is_not_another_program(self):
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        try:
            async def case(app, pilot, ctx):
                await goto(pilot, app, "context")
                ctx.settings.port = port
                with mock.patch.object(Runner, "serving_launches", return_value=[{"profile": "launch-x", "name": "x", "port": port, "choices": {}}]):
                    await goto(pilot, app, "access")
                self.assertNotIn("already in use", flat(app))
            self.run_app(case)
        finally:
            held.close()


# ------------------------------------------------------------------------------------------------ S13

class S13SteadyBar(unittest.TestCase):
    def feed(self, steps):
        bar, got = runner.SteadyBar(), []
        for phase, pct, done, total in steps:
            dl = {"percent": pct, "done_gb": done, "total_gb": total}
            got.append(bar.apply(phase, dl, False, f"{phase} text"))
        return got

    def test_the_evidence_of_vast_v2_never_goes_back(self):
        # 61 % (52.3 of 84.5 GB), the check of the finished file at 32 % and 96 % (that file's own figure), then the download again at 69 %
        got = self.feed([("download", 61, 52.3, 84.5), ("verify", 32, 20.0, 62.0), ("verify", 96, 60.0, 62.0), ("download", 69, 58.6, 84.5)])
        pcts = [p for _t, p in got]
        self.assertEqual(pcts, [61, 61, 61, 69])
        self.assertEqual(pcts, sorted(pcts))
        self.assertIn("Checking the files downloaded so far (52.3 of 84.5 GB)", got[1][0])
        self.assertNotIn("nothing is downloaded", got[1][0])

    def test_a_model_that_was_all_here_keeps_its_check_figures(self):
        bar = runner.SteadyBar()
        text, pct = bar.apply("verify", {"percent": 40, "done_gb": 1, "total_gb": 2}, True, "checking")
        self.assertEqual((text, pct), ("checking", 40))

    def test_another_kind_of_file_starts_its_own_bar(self):
        got = self.feed([("download", 90, 76.0, 84.5), ("pictures", 5, 0.05, 0.9)])
        self.assertEqual([p for _t, p in got], [90, 5])


if __name__ == "__main__":
    unittest.main()
