"""The `strata launch` application: the screens in order, the run, and the hand-over to the monitor.

`run_tui` opens the screens (or, with settings already chosen, goes straight to the run), then the Strata Monitor with the
launch-aware extras: a Connect view (key 7) and a quit question (keep serving in the background, or stop).
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual.app import App
from textual.binding import Binding

from strata_cli.launch import access as acc
from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch import store
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Cancelled, Event, LaunchError, Result, Runner, auto_signals, read_config
from strata_cli.launch.screens import (AccessScreen, AdvancedScreen, CardsScreen, ConfirmScreen, ContextScreen, DraftingScreen, FolderScreen,
                                       ModelScreen, ParallelScreen, PicturesScreen, RunScreen, SaveScreen, StartScreen,
                                       TuningScreen, flow_for)
from strata_cli.launch.widgets import CSS, Glyphs

SCREENS = {"model": ModelScreen, "cards": CardsScreen, "folder": FolderScreen, "context": ContextScreen, "pictures": PicturesScreen, "drafting": DraftingScreen,
           "parallel": ParallelScreen, "tuning": TuningScreen, "advanced": AdvancedScreen, "access": AccessScreen, "save": SaveScreen}


@dataclass
class Ctx:
    """Everything the screens share: the catalog, this computer, the choices so far."""

    catalog: Catalog
    computer: Any
    rows: list
    settings: S.LaunchSettings
    glyphs: Glyphs = field(default_factory=Glyphs)
    feed: str | None = None
    move_data: bool = False
    retune: bool = False
    restart: bool = False
    pending_save: str | None = None          # the preset name "Save as a preset and launch" saves once nothing blocks the launch
    notes: list[str] = field(default_factory=list)
    start_own_files: bool = False
    picked_in_list: bool = False
    tune_chosen: bool = False
    fresh_setup: bool = True
    engine_note: str = ""
    engine_flags: list[str] | None = None
    _adv: list | None = None
    _help: str | None = None
    _help_exe: str | None = None             # the engine whose --help _help is: read once per engine, not again on every edit (#5940)
    _cmd: tuple | None = None                # (the choices and base config it was worked out for, the engine command): see command_text
    probe_offline: bool = False              # S6: ask once, in the background, whether the download hosts answer (only the real launch screens turn this on)
    offline: bool | None = None              # None: not asked yet (or not asked at all)
    _probing: bool = False
    tuned_offered: dict = field(default_factory=dict)     # (engine family, tuned file) -> True / False / None: whether the feed lists the small tuned drafter (check_tuned)
    _tuned_asking: set = field(default_factory=set)
    _tuned_wanted: tuple | None = None

    def probe_now(self) -> bool:
        """The probe itself: True when none of the download addresses answers. A feed on this computer itself (a loopback address: a test feed, a mirror running here) does not
        count as an answer from the internet (0217 item 5: with one set, an offline computer looked online); a person's own mirror elsewhere does."""
        from urllib.parse import urlparse
        from strata_control import drafting, drafting_fetch
        urls = []
        for u in (drafting.base_url(), drafting_fetch._feed_source(self.feed)):
            if not str(u).lower().startswith(("http://", "https://")):
                continue
            if (urlparse(str(u)).hostname or "").lower() in ("127.0.0.1", "localhost", "::1"):
                continue
            urls.append(u)
        return bool(urls) and not any(drafting_fetch.online(url, timeout=4.0) for url in urls)

    def is_offline(self) -> bool:
        """The answer now: the one the Save screen's probe kept, else (a preset started with one key, which never visited Save) the probe, here and now; False when the screens do not
        probe (the unit tests) or it cannot be told."""
        if self.offline is None and self.probe_offline:
            try:
                self.offline = self.probe_now()
            except Exception:                                # noqa: BLE001
                self.offline = False
        return bool(self.offline)

    def check_offline(self, on_done=None) -> None:
        """S6: is this computer offline? One short HEAD request to each address Strata downloads from: the model files' (as the drafter fetch names
        it, which follows a mirror the person set) and the update feed's, when that is a web address and not a folder of the person's own (the download's
        own opener: same certificates and proxy settings, strata_control/drafting_fetch.online), in a thread of its own; offline only when none answers.
        The answer is kept, so the Save screen says what is missing instead of "will download". Never raises, never blocks a screen."""
        if not self.probe_offline or self.offline is not None or self._probing:
            return
        self._probing = True

        def run() -> None:
            try:
                self.offline = self.probe_now()
            except Exception:                                # noqa: BLE001 - not knowing counts as online: the download itself says what is wrong
                self.offline = False
            finally:
                self._probing = False
            if on_done is not None:
                try:
                    on_done()
                except Exception:                            # noqa: BLE001
                    pass
        import threading
        threading.Thread(target=run, daemon=True, name="offline-check").start()

    def download_plan(self) -> dict | None:
        """What this launch would download (Runner.download_plan: the model's missing files, the draft layer, the picture reader, the DFlash2 drafter,
        the engine), worked out once per set of choices. None when it cannot be worked out. The Save screen lists it (T2)."""
        import dataclasses
        import json
        try:
            key = json.dumps(dataclasses.asdict(self.settings), sort_keys=True, default=str)
        except Exception:                                    # noqa: BLE001
            key = None
        cached = getattr(self, "_plan", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
        try:
            from strata_cli.launch.runner import Runner as PlanRunner
            planner = PlanRunner(self.settings, catalog=self.catalog, feed=self.feed)
            planner.tuned_offered = self.tuned_offered          # what the feed said so far (check_tuned fills it in): the plan lists the size the launch really downloads
            plan = planner.download_plan()
            self._tuned_wanted = planner.tuned_wanted
        except Exception:                                    # noqa: BLE001 - a line of information, never a reason to stop the screens
            plan = None
        if key is not None:
            self._plan = (key, plan)
        return plan

    def drafter_term(self) -> str:
        """What the drafter this launch will use is called in a sentence: "the DFlash2 drafter", "the draft layer" (the MTP) or "" (Off). One term for one case (0217 items 9 and 10: Save and the
        Folder step said "the draft layer" next to "the DFlash2 drafter"). Worked out once per set of choices."""
        import dataclasses
        import json
        try:
            key = json.dumps(dataclasses.asdict(self.settings), sort_keys=True, default=str)
        except Exception:                                    # noqa: BLE001
            key = None
        cached = getattr(self, "_term", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
        if self.settings.family == "hy4-flash":
            return ""
        try:
            from strata_cli.launch.runner import Runner as PlanRunner
            resolved = PlanRunner(self.settings, catalog=self.catalog).resolved_drafting(self.settings.family)
        except Exception:                                    # noqa: BLE001 - not known: the layer, as before
            resolved = "mtp"
        word = {"dflash2": "the DFlash2 drafter", "mtp": "the draft layer"}.get(resolved, "")
        if key is not None:
            self._term = (key, word)
        return word

    def check_tuned(self, on_done=None) -> None:
        """0217 item 3: when the plan lists the small tuned DFlash2 drafter, ask the signed feed once, in the background, whether it really lists that file for this engine family
        (the launch fetches it only then, else the 1.2 GB standard drafter). The answer is kept; the plan is worked out again and on_done repaints. Never raises, never blocks a screen."""
        key = getattr(self, "_tuned_wanted", None)
        if not self.probe_offline or key is None or key in self.tuned_offered or key in self._tuned_asking:
            return
        self._tuned_asking.add(key)

        def run() -> None:
            try:
                from strata_control import drafting_fetch
                self.tuned_offered[key] = drafting_fetch.tuned_offered(feed=self.feed, backend=key[0], variant=key[1])
            except Exception:                                # noqa: BLE001 - not known: the plan keeps the small file's size
                self.tuned_offered[key] = None
            finally:
                self._tuned_asking.discard(key)
            self._plan = None                                # worked out again with the answer
            if on_done is not None:
                try:
                    on_done()
                except Exception:                            # noqa: BLE001
                    pass
        import threading
        threading.Thread(target=run, daemon=True, name="feed-check").start()

    def refresh_disk(self) -> None:
        """Mark the Model list's disk verdicts again for the files in use: the model in the person's own folder needs no download (N1); every other
        row counts only what it would download. Called when the own folder is chosen or the choice goes back to the list."""
        s = self.settings
        try:
            self.catalog.apply_disk(self.rows, self.computer.raw, own=(s.family, s.model) if s.model_path and s.family and s.model else None)
        except Exception:                                    # noqa: BLE001 - a mark in a list is never a reason to stop the screens
            pass

    def computer_line(self, width: int | None = None) -> str:
        """The computer's one line (the header of the start screen, the model list) with the free disk read again now: it was read once when the
        screens opened and stayed "88 GB free" after a download left 26 GB (T9). A statvfs call; nothing else is asked again."""
        try:
            free = self.catalog.disk_free(self.settings.data_dir or self.catalog.data_dir())
            if free is not None:
                self.computer.disk_free_gb = free
        except Exception:                                    # noqa: BLE001 - the old figure is better than none
            pass
        return self.computer.line(width=width)

    def card_roles(self) -> tuple[dict[int, str], list[dict]] | None:
        """The role of every card this launch will use (named here, else the installed model's own, else the recommendation) with the cards, or None
        on a computer with fewer than two usable cards. The Cards step shows the same; the Save and Advanced screens did not (T8)."""
        c = self.computer
        if len(c.usable_gpus) < 2 or self.settings.family == "hy4-flash":
            return None
        try:
            from strata_control import gpu_roles
            from strata_cli.launch.runner import installed_cards
            cards = gpu_roles.rows(c.gpus, c.engine_vendor)
            s = self.settings
            if s.gpu_roles:
                return gpu_roles.normalize({int(k): v for k, v in s.gpu_roles.items()}, cards), cards
            if s.gpus:
                return gpu_roles.from_legacy(list(s.gpus), cards, "as_given"), cards
            try:
                inst = installed_cards(self.base_config(), c.usable_gpus)
            except Exception:                                # noqa: BLE001
                inst = None
            if inst:
                return gpu_roles.from_legacy(inst, cards, "as_given"), cards
            return dict(gpu_roles.recommend(cards, c.engine_vendor)["roles"]), cards
        except Exception:                                    # noqa: BLE001 - a summary line never stops the screens
            return None

    def cards_line(self) -> str | None:
        """`Main: GPU 1 RTX 4090. Helper: GPU 2 RTX 4090.` for a computer with two or more cards, else None."""
        found = self.card_roles()
        if found is None:
            return None
        from strata_control import gpu_roles
        return gpu_roles.describe(found[0], found[1]) or None

    def flags_line(self) -> str:
        """strata launch ... with the card roles in it, also when nobody changed them (the recommendation is what runs, so the copied line says it)."""
        line = S.flags_text(self.settings)
        found = self.card_roles()
        if found is not None and not self.settings.gpu_roles and not self.settings.gpus:
            line += " --gpu-roles " + ",".join(f"{k}={v}" for k, v in sorted(found[0].items()))
        return line

    # ---- names and warnings
    def default_name(self) -> str:
        s = self.settings
        base = f"{s.model or 'my model'} {S.context_label(s.context)}"
        names = {p.name for p in store.list_presets()}
        if base not in names:
            return base
        n = 2
        while f"{base} ({n})" in names:
            n += 1
        return f"{base} ({n})"

    def warnings(self) -> list[str]:
        s = self.settings
        out = [n.line() for n in S.check_access(s)]
        if s.family != "hy4-flash":
            out += [n.line() for n in S.check_parallel(s, vram_gb=self.computer.main_vram_gb or None)]
        out += [n.line() for n in S.check_context(s)]
        r = next((x for x in self.rows if (x.family, x.model) == (s.family, s.model)), None)
        if r is not None and r.verdict == "disk":
            if s.model_path:                                             # N1: the files are the person's own, used where they are: no download to make room for
                r = r.without_disk()
            elif s.data_dir:                                             # the Folder step named another drive: its own free space decides
                other = self.catalog.disk_free(s.data_dir)
                if other is not None and other >= r.disk_need_gb:
                    r = r.without_disk()
        if r is not None and r.verdict != "fits":
            out.append(f"{r.title}: {r.sentence}.")
        if self.computer.other_vendor_note:
            out.append(self.computer.other_vendor_note)
        missing = acc.missing_helper(s.access)
        if missing:
            out.append(missing)
        return out

    # ---- the installed model's configuration (for the Advanced table and the command)
    def base_config(self) -> dict | None:
        s = self.settings
        if not (s.family and s.model):
            return None
        path = self.catalog.installed_config(self.catalog.tag_of(s.family, s.model))
        if path is None:
            return None
        try:
            return read_config(path)
        except (OSError, ValueError):
            return None

    def engine_exe(self) -> str | None:
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            return (rep.get("engine") or {}).get("exe") if rep.get("usable") else None
        except Exception:                                    # noqa: BLE001
            return None

    # ---- where the model is and what it still needs
    def model_state(self, data_dir: str | None = None):
        """The chosen model's state (installed, what helper files are missing, sizes) for these choices."""
        from strata_cli.launch import modelstate
        s = self.settings
        return modelstate.assess(self.catalog, s.family, s.model, vision=s.vision, model_path=s.model_path,
                                 data_dir=data_dir if data_dir is not None else s.data_dir, rows=self.rows, mtp=s.mtp,
                                 drafting=modelstate.resolved_drafting(s.drafting, s.family, **auto_signals(self.catalog, s, s.family)))

    def default_folder(self) -> Path:
        """The folder the Folder step starts with: where the chosen model already is (installed, or the person's own files with their
        prepared data), else the data folder every command agrees on."""
        s = self.settings
        if s.model_path:
            try:
                from strata_control import discovery
                p = Path(s.model_path).expanduser()
                first = p if p.is_file() else next(iter(sorted(p.glob("*.gguf"))), None)
                found = discovery.writable_beside(first) if first else None     # not a folder Strata cannot write to, nor a Hugging Face cache
                if found:
                    return Path(found)
            except OSError:
                pass
            return self.catalog.data_dir()
        return self.catalog.data_dir_for(s.family, s.model)

    def invalidate_rows(self) -> None:
        self._adv = None

    def advanced_rows(self) -> list[P.Row]:
        if self._adv is not None:
            return self._adv
        s = self.settings
        cfg = self.base_config()
        base_args = list((cfg or {}).get("args") or [])
        base_env = dict((cfg or {}).get("env") or {})
        tuned_args, tuned_env = None, None
        if cfg:
            try:
                from strata_control import presets
                merged, info = presets.overlay(cfg)
                if info.get("applied"):
                    tuned_args, tuned_env = list(merged.get("args") or []), dict(merged.get("env") or {})
            except Exception:                                # noqa: BLE001
                pass
        exe = self.engine_exe()
        if exe and exe == self._help_exe:
            text = self._help
        else:
            text = P.engine_help_text(exe) if exe else None
            if text:
                self._help_exe, self._help = exe, text
        if exe and text:
            self.engine_flags = sorted({p.name for p in P.parse_help(text)})
            self.engine_note = ""
        else:
            self.engine_flags = None
            self.engine_note = ("The engine is not installed yet, so this list shows only what this launch uses. The full list "
                                "appears after the first launch installs it.") if not exe else \
                               "The engine's own list could not be read here; names are not checked."
        if not cfg and not s.extra:
            self.engine_note = self.engine_note or "The model is not installed yet; the defaults and tuned values appear after it is."
        from strata_control import speed_switches
        gpus = list(getattr(self.computer, "gpus", None) or [])
        switches = P.switch_params(speed_switches.applicable(cfg, gpus, self.catalog.installed_backend()))
        self._adv = P.build_rows(text, base_args, tuned_args, base_env, tuned_env, s.extra, switches)
        return self._adv

    def command_text(self) -> str:
        """The engine command these choices run. Worked out once per set of choices and base config: it asks for the cards, the engine's
        backend and the GPU's free memory (child processes such as nvidia-smi), and the Advanced step paints it on every cursor move (#5940)."""
        s = self.settings
        cfg = self.base_config()
        if cfg is None:
            return self._runtime_line() + "   (the engine command is shown once the model is installed)"
        try:
            import dataclasses
            import json
            key = json.dumps([dataclasses.asdict(s), cfg], sort_keys=True, default=str)
        except Exception:                                    # noqa: BLE001 - not cacheable: work it out every time, as before
            key = None
        if key is not None and self._cmd is not None and self._cmd[0] == key:
            command = self._cmd[1]
        else:
            command = self._engine_command(cfg)
            if key is not None:
                self._cmd = (key, command)
        return command if command is not None else self._runtime_line()

    def _runtime_line(self) -> str:
        s = self.settings
        return f"strata runtime start --profile {'launch-' + S.slug(s.name or self.default_name())}"

    def _engine_command(self, cfg: dict) -> str | None:
        s = self.settings
        try:
            from strata_control import tune_knobs
            from strata_cli.launch.runner import apply_choices, cards_for_launch, derive_config
            from serve.server import engine_args
            import contextlib
            import io
            d = derive_config(apply_choices(cfg, s), s, bool_flags=tune_knobs.BOOL_FLAGS, gpus=cards_for_launch(s, self.catalog, base=cfg))
            if s.parallel:
                d["parallel"] = s.parallel
            with contextlib.redirect_stdout(io.StringIO()):
                argv = engine_args(d)
            argv = ["***" if (i and argv[i - 1] in ("--api-key",)) else a for i, a in enumerate(argv)]
            return f"{Path(str(d.get('exe', 'engine'))).name} {' '.join(str(a) for a in argv)}"
        except Exception:                                    # noqa: BLE001
            return None


class LaunchApp(App):
    TITLE = "strata launch"
    CSS = CSS
    # Textual's own ctrl+c only shows a "Do you want to quit?" note, and Escape pressed next also went Back (B13): ctrl+c quits like ctrl+q and q
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", show=False, priority=True)]

    def __init__(self, ctx: Ctx, *, auto_run: bool = False) -> None:
        super().__init__()
        self.ctx = ctx
        self.history: list[str] = []
        self.result: Result | None = None
        self.stopped_after_save = False
        self.cancel = threading.Event()
        self.auto_run = auto_run
        self.runner: Runner | None = None
        self.run_screen: RunScreen | None = None
        self.running = False
        self.current = "start"

    def action_quit(self) -> None:
        """ctrl+q and ctrl+c, as q on the screens: leave at once. Only a running download or tuning asks first ("Stop now?"), because work would be lost."""
        if self.running and self.run_screen is not None:
            self.stop_run(self.run_screen)
        else:
            self.exit(None)

    def on_mount(self) -> None:
        if self.auto_run:
            self.run_launch()
        else:
            self.push_screen(StartScreen())

    # ---- navigation
    def flow(self) -> list[str]:
        return flow_for(self.ctx)

    def goto(self, step_id: str) -> None:
        self.current = step_id
        self.show(SCREENS[step_id]())

    def begin_setup(self, *, preset: str | None = None, own_files: bool = False) -> None:
        self.ctx.tune_chosen = False
        self.ctx.fresh_setup = not preset
        if preset:
            s = store.load(preset)
            self.ctx.settings = s
        else:
            self.ctx.settings = S.LaunchSettings()
        self.ctx.start_own_files = own_files
        self.ctx.picked_in_list = False
        self.ctx.invalidate_rows()
        self.current = "model"
        self.show(ModelScreen())

    def advance(self) -> None:
        flow = self.flow()
        i = flow.index(self.current) if self.current in flow else -1
        if i + 1 < len(flow):
            self.goto(flow[i + 1])

    def back(self) -> None:
        flow = self.flow()
        i = flow.index(self.current) if self.current in flow else 0
        if i > 0:
            self.goto(flow[i - 1])
        else:
            self.current = "start"
            self.show(StartScreen())

    # ---- starting
    def start_preset(self, name: str) -> None:
        s = store.load(name)
        self.ctx.settings = s
        self.run_launch()

    def show(self, screen) -> None:
        """Put `screen` in front. The first screen of a run that skips the questions has nothing to replace (the app starts with only its
        empty base screen), so it is pushed; every later one replaces the one on top. (switch_screen on the base screen raised
        IndexError: pop from empty list, tester report T02.)"""
        if len(self.screen_stack) <= 1:
            self.push_screen(screen)
        else:
            self.switch_screen(screen)

    def run_launch(self, confirmed: bool = False) -> None:
        s = self.ctx.settings
        self.cancel = threading.Event()
        self.runner = Runner(s, catalog=self.ctx.catalog, emit=self._emit, cancel=self.cancel, feed=self.ctx.feed, confirm=self._ask_from_run,
                             move_data=self.ctx.move_data, retune=self.ctx.retune, restart=self.ctx.restart,
                             confirm_restart=self._ask_restart, after_check=self._save_pending)
        gb = getattr(self.runner, "download_gb", lambda: 0.0)() if not confirmed else 0.0
        if gb >= 1.0:
            free = self.ctx.catalog.disk_free(s.data_dir or self.ctx.catalog.data_dir())
            room = (f" There is {free:,.0f} GB free in the folder." if free is not None else "")
            warn = (" That is less than the download needs; it will probably stop part way." if free is not None and free < gb + 8 else "")

            kinds = self.optional_parts()

            def done(choice: str | None) -> None:
                if choice == "go":
                    self.run_launch(confirmed=True)
                elif choice == "skip":                       # T16: "Not now" for an extra file starts the model without it, it does not go back
                    if "mtp" in kinds:
                        s.mtp = "off"
                    if "mmproj" in kinds:
                        s.vision = "off"
                    self.run_launch(confirmed=True)

            asker = getattr(self.runner, "download_question", None)
            title, text = asker() if asker else (f"Download about {gb:.0f} GB?",
                                                 f"This model is not on this computer yet. Strata will download about {gb:.0f} GB now; "
                                                 "an interrupted download continues where it stopped.")
            offline = self.ctx.is_offline()                  # 0217 item 13: the dialog said nothing about the computer being offline
            go = "Try the download" if offline else "Download and start"
            if offline:
                text = "This computer is offline, so this download will probably not work now. " + text
            if kinds:
                what = ("the draft layer" if kinds == {"mtp"} else "pictures" if kinds == {"mmproj"} else "them")
                options = [("go", go), ("skip", f"Start without {what}")]
                text += " Or start now without it: answers are a little slower or text only, and you can add it later."
            else:
                options = [("go", go), ("no", "Not now")]
            self.push_screen(ConfirmScreen(title, text + room + warn, options), done)
            return
        screen = RunScreen()
        self.run_screen = screen
        self.current = "run"
        self.show(screen)
        self.running = True
        threading.Thread(target=self._work, daemon=True).start()

    def optional_parts(self) -> set[str]:
        """The kinds of file ("mtp" the draft layer, "mmproj" the picture reader) when they are ALL that is missing: the model itself is here, so
        the launch can go on without them. Empty when the model has to be downloaded or the runner cannot say."""
        try:
            st = self.runner.model_state()                   # type: ignore[union-attr]
            kinds = {p.kind for p in st.parts}
            return kinds if st.only_helpers and kinds <= {"mtp", "mmproj"} else set()
        except Exception:                                    # noqa: BLE001
            return set()

    def _save_pending(self) -> None:
        name, self.ctx.pending_save = self.ctx.pending_save, None
        if name:
            try:
                store.save(self.ctx.settings, name)
            except store.PresetError:
                pass

    def _ask_restart(self, title: str, text: str) -> bool:
        """Another model is serving other choices: ask once, in the same pop-up, whether to restart it. Cancel leaves it running."""
        return self._ask_from_run(title, text, [("go", "Restart"), ("no", "Cancel")], default=False)

    def _ask_from_run(self, title: str, text: str, options: list[tuple[str, str]] | None = None, default: bool = True) -> bool:
        """A question from the run (a worker thread): show it on the screen and wait for the answer. Enter means yes."""
        box: dict[str, bool] = {"go": default}
        answered = threading.Event()

        def done(choice: str | None) -> None:
            box["go"] = choice == "go"
            answered.set()

        def show() -> None:
            self.push_screen(ConfirmScreen(title, text, options or [("go", "Download anyway"), ("no", "Not now")]), done)

        if not self._call(show):                             # the app is closing
            return False
        while not answered.wait(0.2):
            if self.cancel.is_set():
                return False
        return box["go"]

    def _emit(self, ev: Event) -> None:
        screen = self.run_screen
        if screen is not None:
            self._call(screen.handle_event, ev)

    def _call(self, fn, *args) -> bool:
        """call_from_thread from a worker thread, False when it could not be made: after a quit the app's loop is closed and
        the call raises ("Event loop is closed"); the screen is gone then, so there is nothing to tell.  A loop already closed is not
        asked at all: Textual makes the call's coroutine before it finds the loop closed, and that left a "coroutine was
        never awaited" warning on the terminal."""
        loop = getattr(self, "_loop", None)
        if loop is not None and loop.is_closed():
            return False
        try:
            self.call_from_thread(fn, *args)
        except Exception:                                    # noqa: BLE001 - the app is closing
            return False
        return True

    def _work(self) -> None:
        screen = self.run_screen
        try:
            result = self.runner.run()               # type: ignore[union-attr]
        except Cancelled:
            self.running = False
            self._call(self._after_cancel)
            return
        except LaunchError as e:
            self.running = False
            self._call(screen.on_failure, f"{e.text} {e.next}".strip())
            return
        except Exception as e:                           # noqa: BLE001
            self.running = False
            self._call(screen.on_failure, f"Something unexpected went wrong: {type(e).__name__}: {e}")
            return
        self.running = False
        self._save_pending()                         # a runner that never asked (nothing was running) still saves what was chosen
        s = self.ctx.settings
        try:
            if s.name and store.load(s.name, missing_ok=True) is not None:
                store.save(s, s.name)
                store.touch_used(s.name)
        except Exception:                                # noqa: BLE001
            pass
        self.result = result
        self._call(screen.on_finished)
        self._call(self.exit, result)

    def _after_cancel(self) -> None:
        self.run_screen.on_failure("Stopped. Nothing was started.")     # type: ignore[union-attr]

    def stop_run(self, screen) -> None:
        if not self.running:
            if screen.failed:
                self.back_to_start()
            else:
                self.exit(None)
            return

        def done(choice: str | None) -> None:
            if choice == "stop":
                self.cancel.set()

        self.push_screen(ConfirmScreen("Stop now?", "Files already downloaded are kept; a download continues where it stopped next time. "
                                       "Nothing is started.", [("stop", "Stop"), ("go", "Keep going")]), done)

    def back_to_start(self) -> None:
        self.current = "start"
        self.show(StartScreen())


# ------------------------------------------------------------------------------------------------ the monitor

def _info_for(result: Result, s: S.LaunchSettings) -> dict:
    cfg = {}
    try:
        cfg = read_config(Path(result.config))
    except (OSError, ValueError):
        pass
    return {"url": result.urls[0] if result.urls else f"http://127.0.0.1:{s.port}", "urls": result.urls, "key": result.key,
            "model": cfg.get("model_name") or "strata", "context": s.context, "name": s.name, "profile": result.profile, "running": True,
            "tunnel": result.tunnel, "last_run": getattr(result, "last_run", "")}


def run_monitor(info: dict, *, keep_default: bool = True, color: str = "auto") -> int:
    """Open the monitor for a running launch. q asks whether to keep serving or stop."""
    from strata_cli.client import Endpoint
    from strata_cli.launch.monitor_ext import LaunchMonitor
    local = next((u for u in info.get("urls", []) if "127.0.0.1" in u), None) or \
        (info.get("urls") or [None])[0] or f"http://127.0.0.1:{info.get('port', S.DEFAULT_PORT)}"    # (One address only: its own URL, #5917)
    endpoint = Endpoint(local, info.get("key"), 3.0, "")
    app = LaunchMonitor(endpoint, None, info={**info, "daily_check": True}, color=color)     # (the quiet once-a-day check for a newer engine: this monitor only)
    app.run()
    if app.stop_requested:
        from strata_control import runtime
        try:
            runtime.stop(info["profile"], timeout=180)
        except Exception as e:                               # noqa: BLE001
            print(f"strata: could not stop the server: {e}", file=sys.stderr)
            return 6
        acc.stop_tunnel(info["profile"])
        print("Stopped. The model is unloaded and the graphics memory is free.")
    else:
        print(f"Still serving at {info['url']}. Watch it again: strata monitor --url {info['url']}   Stop: strata launch --stop")
    return 0


def run_tui(settings: S.LaunchSettings | None, *, catalog: Catalog | None = None, feed: str | None = None, auto_run: bool = False,
            save: bool = False, open_page: bool = False, move_data: bool = False, retune: bool = False,
            restart: bool = False) -> int:
    """Open the launch screens. `settings` given and `auto_run`: skip the questions and go straight to the run."""
    catalog = catalog or Catalog()
    print("Looking at this computer…", flush=True)
    computer = catalog.computer()
    rows = catalog.rows(computer.raw, own=(settings.family, settings.model) if settings is not None and settings.model_path and settings.family and settings.model else None)
    from strata_cli.launch.cmd import apply_defaults
    s = settings or S.LaunchSettings()
    notes: list[S.Notice] = []
    if settings is not None:
        s, notes = apply_defaults(s, catalog)
        if not s.name:
            s.name = f"{s.model or 'model'} {S.context_label(s.context)}"
        if save:
            store.save(s, s.name)
    ctx = Ctx(catalog=catalog, computer=computer, rows=rows, settings=s, glyphs=Glyphs(), feed=feed, move_data=move_data, retune=retune, restart=restart,
              probe_offline=True)
    app = LaunchApp(ctx, auto_run=auto_run)
    result = app.run()
    if getattr(app, "return_code", 0) not in (0, None) or getattr(app, "_exception", None) is not None:
        # The screens stopped on an error of their own (not a launch failure, which the run screen shows). Never end quietly with exit 0.
        print("strata: The launch screens stopped because of an unexpected problem, and nothing was started. "
              "Run the same command with --no-tui to see the steps as plain lines, and send the output if you ask for help.", file=sys.stderr)
        return 1
    if app.stopped_after_save:
        print(f"Saved the preset '{ctx.settings.name}'. Start it with: strata launch --preset \"{ctx.settings.name}\"")
        return 0
    if result is None:
        return 0
    if not result.serving:
        print(result.final_text)
        return 0
    info = _info_for(result, ctx.settings)
    if open_page:                                            # only with an explicit --open: nothing else opens a browser
        from strata_cli.launch.cmd import open_page as open_in_browser
        open_in_browser(info["url"])
    return run_monitor(info)
