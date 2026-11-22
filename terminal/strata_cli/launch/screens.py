"""The screens of `strata launch`: the saved-preset start screen, the nine setup steps, the confirm pop-up and the progress screen.

One decision per screen. Every screen has a short plain explanation under the content (what it means, what to do next) and a line
of key hints. Esc goes back, Enter goes on, q leaves (and says what is kept). Choices are written into `app.ctx.settings`
(strata_cli/launch/settings.py); nothing is installed or started until the last screen.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Input, Static

from strata_cli.launch import access as acc
from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch import store
from strata_cli.launch.widgets import (ACCENT, BAD, DARK, FG, LIGHT, MUTE, WARN, Cell, Item, Menu, bar)

STEPS = (("model", "Model"), ("cards", "Cards"), ("folder", "Folder"), ("context", "Context"), ("pictures", "Pictures"),
         ("parallel", "Parallel"), ("tuning", "Tuning"), ("advanced", "Advanced"), ("access", "Access"), ("save", "Save"))


STEP_LABELS = {**dict(STEPS), "drafting": "Drafting"}


def middle_shorten(text: str, width: int, dots: str = "…") -> str:
    """`text` in at most `width` terminal cells: when longer, its start and its end with `dots` between them (the drive and the folder's own
    name stay readable). Unchanged when it fits; at least the dots and one character on each side."""
    from rich.cells import cell_len
    if cell_len(text) <= width:
        return text
    room = max(2, width - cell_len(dots))
    head_n = max(1, room * 2 // 5)
    tail_n = max(1, room - head_n)
    head, tail = "", ""
    for ch in text:                                  # by cells, so wide characters count twice
        if cell_len(head + ch) > head_n:
            break
        head += ch
    for ch in reversed(text):
        if cell_len(ch + tail) > tail_n:
            break
        tail = ch + tail
    return head + dots + tail


MIN_STAGE_BAR = 8        # the shortest a stage's bar is drawn


def stage_bar_width(wide: int, screen_width: int, text_len: int) -> int:
    """The cells of a stage's bar. `wide` (40 at most) is the bar's own width. It gives way so the phase words after it are whole in a small
    window (0217 item 14), but only when giving way gets the words onto the line: a phrase that cannot fit beside even the shortest bar
    ("Checking the files downloaded so far (54.8 of 84.5 GB); the download goes on after it", 89 characters) wraps under the bar at any width,
    so the bar keeps its width there (the 0.2.17 retest saw it cut to 8 cells in a 110-column window, 0.2.16 drew 40)."""
    room = screen_width - 14 - 7 - text_len
    if room < MIN_STAGE_BAR:
        return wide
    return min(wide, room)


def tuned_sentence(row) -> str:
    """The Advanced step's note for a narrow window (the Tuned column is left out there): what the tuner chose. Said only when the tuned value is
    not the default: "The tuner chose off." for a switch that is off by default, in a launch whose tuning step said Skip, claimed work nobody did
    (GUI-TUI 0.2.17 QA)."""
    tuned = getattr(row, "tuned", None)
    if tuned in ("-", "", None) or str(tuned) == str(getattr(row, "default", "") or ""):
        return ""
    return f" The tuner chose {tuned}."


def drafting_choice_available(ctx, family: str | None = None) -> bool:
    """The Drafting step exists only where there is something to choose: Qwen3.8-Flash-Next (qwen, gyro) on a computer whose engine can run DFlash2
    (CUDA or HIP, strata_control/drafting.py). Every other model has no step: the rest have only the MTP. `family`: the model family to ask for
    (the Model step asks for the one under the cursor); default the chosen one."""
    from strata_control import drafting
    family = family if family is not None else getattr(getattr(ctx, "settings", None), "family", None)
    if family not in drafting.SUPPORTED_FAMILIES or not drafting.ENGINE_SERVES_DFLASH2:
        return False
    catalog = getattr(ctx, "catalog", None)
    backend, archs = drafting.hardware_for(catalog.installed_backend() if catalog is not None else None)   # #5109: the installed engine's family
    return drafting.hardware_support(backend, archs).can_run


def flow_for(ctx, family: str | None = None) -> list[str]:
    """The steps of this computer: the Cards step exists only when two or more graphics cards can be used; the Drafting step only for
    Qwen3.8-Flash-Next on a computer that can run DFlash2; Tencent Hy4-Flash skips Qwen-only steps (cards, drafting, parallel, tuning, advanced)."""
    fam = family if family is not None else getattr(getattr(ctx, "settings", None), "family", None)
    if fam == "hy4-flash":
        return ["model", "folder", "context", "pictures", "access", "save"]
    if fam == "inkling-2-mini":
        if len(ctx.computer.usable_gpus) >= 2:
            return ["model", "folder", "cards", "context", "pictures", "parallel", "access", "save"]
        return ["model", "folder", "context", "pictures", "parallel", "access", "save"]
    ids = [sid for sid, _ in STEPS]
    if len(ctx.computer.usable_gpus) < 2:
        ids.remove("cards")
    if drafting_choice_available(ctx, family):
        ids.insert(ids.index("pictures") + 1, "drafting")
    return ids


def _t(*parts: tuple[str, str] | str) -> Text:
    out = Text(no_wrap=False)
    for p in parts:
        if isinstance(p, tuple):
            out.append(p[0], style=p[1])
        else:
            out.append(p)
    return out


class Base(Screen):
    """Shared frame: heading line, rule, content, rule, explanation, key hints, message."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=False), Binding("q", "quit_app", "Quit", show=False),
        Binding("question_mark", "help", "Help", show=False), Binding("f1", "help", "Help", show=False),
        Binding("down", "move(1)", show=False), Binding("up", "move(-1)", show=False),
        Binding("pagedown", "page(1)", show=False), Binding("pageup", "page(-1)", show=False),
        Binding("enter", "next", "Next", show=False),
        Binding("home", "edge(0)", show=False), Binding("end", "edge(1)", show=False),
    ]

    AUTO_FOCUS = ""
    step_id = ""
    title_line = "Set up"
    hints = "↑↓ choose   Enter next   Esc back   ? help   q quit"
    help_text = "Up/Down choose, Enter next, Esc back, q quit. Nothing is installed or started until the last screen."

    def __init__(self) -> None:
        super().__init__()
        self.message = ""

    # ---- frame
    @property
    def ctx(self):
        return self.app.ctx           # type: ignore[attr-defined]

    @property
    def g(self):
        return self.ctx.glyphs

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static("", id="head", markup=False)
            yield Static("", id="rule1", markup=False)
            with Vertical(id="body"):
                yield from self.body()
            yield Static("", id="rule2", markup=False)
            yield Static("", id="explain", markup=False)
            yield Static("", id="hints", markup=False)
            yield Static("", id="message", markup=False)

    def body(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.paint()

    def on_resize(self, _e) -> None:
        self.paint()

    def width(self) -> int:
        return max(40, (self.size.width or 100) - 2)

    def rule(self) -> str:
        return self.g.rule * self.width()

    def flow(self) -> list[str]:
        """The steps this screen counts with (the Model step counts for the model under the cursor: S1)."""
        return flow_for(self.ctx)

    def head_text(self) -> Text:
        flow = self.flow()
        out = Text(no_wrap=True, overflow="crop")
        out.append("STRATA  ", style=f"bold {ACCENT}")
        out.append(f"{self.title_line}", style=f"bold {ACCENT}")
        if self.step_id in flow:
            labels = STEP_LABELS
            n = flow.index(self.step_id)
            if self.width() >= 118:
                out.append("   ")
                for i, sid in enumerate(flow):
                    mark = f"[{i + 1} {labels[sid]}]" if i == n else f"{i + 1} {labels[sid]}"
                    out.append(mark + "  ", style=f"bold {ACCENT}" if i == n else "")
            else:
                out.append(f"   Step {n + 1} of {len(flow)}: {labels[self.step_id]}", style="")
        return out

    def paint(self) -> None:
        try:
            self.query_one("#head", Static).update(self.head_text())
            self.query_one("#rule1", Static).update(Text(self.rule(), style=MUTE))
            self.query_one("#rule2", Static).update(Text(self.rule(), style=MUTE))
            explanation = self.explain()
            self.query_one("#explain", Static).update(explanation)
            self.query_one("#explain", Static).display = bool(explanation.plain.strip())     # no empty row where there is nothing to say
            self.query_one("#hints", Static).update(Text(self.hints_text(), style=MUTE))
            self.query_one("#message", Static).update(Text(self.message, style=WARN if self.message.startswith("!") else MUTE))
        except Exception:                                    # noqa: BLE001 - a screen that is closing
            return
        self.refresh_body()

    def hints_text(self) -> str:
        return self.hints.replace("↑↓", self.g.updown)

    def explain(self) -> Text:
        return Text("")

    def refresh_body(self) -> None:
        pass

    def say(self, text: str, warn: bool = False) -> None:
        self.message = ("! " if warn else "") + text
        self.paint()

    # ---- actions
    def menu(self) -> Menu | None:
        try:
            return self.query_one(Menu)
        except Exception:                                    # noqa: BLE001
            return None

    def action_move(self, step: int) -> None:
        m = self.menu()
        if m is not None and not self.input_open():
            m.move(step)
            self.on_cursor()
            self.paint()

    def action_edge(self, last: int) -> None:
        m = self.menu()
        if m is not None and not self.input_open() and m.items:
            m.index = len(m.items) - 1 if last else 0
            m._refresh()
            self.on_cursor()
            self.paint()

    def action_page(self, step: int) -> None:
        m = self.menu()
        if m is not None:
            m.page(step)
            self.on_cursor()
            self.paint()

    def on_cursor(self) -> None:
        pass

    def input_open(self) -> bool:
        return any(i.has_class("open") and not i.has_class("always") for i in self.query(".field"))

    def action_help(self) -> None:
        self.say(self.help_text)

    def action_quit_app(self) -> None:
        self.app.exit(None)

    def action_back(self) -> None:
        self.app.back()          # type: ignore[attr-defined]

    def action_next(self) -> None:
        self.commit()
        self.app.advance()       # type: ignore[attr-defined]

    def commit(self) -> None:
        pass

    # ---- text input helper
    def open_field(self, field_id: str, value: str = "", placeholder: str = "") -> None:
        f = self.query_one(f"#{field_id}", Input)
        f.disabled = False
        f.add_class("open")
        f.value = value
        f.placeholder = placeholder
        f.focus()

    def close_field(self, field_id: str) -> None:
        f = self.query_one(f"#{field_id}", Input)
        f.remove_class("open")
        f.disabled = True
        self.set_focus(None)


# ------------------------------------------------------------------------------------------------ start

class StartScreen(Base):
    step_id = "start"
    title_line = "Launch"
    hints = "↑↓ choose   Enter launch   e edit preset   d delete   ? help   q quit"
    help_text = ("Pick a saved preset to start it at once, or set up from the beginning. A preset remembers the model, "
                 "context size, parallel requests, tuned settings and who can reach it.")

    # the Delete key asks the same question as "d" (#5819: a person who pressed Del on a preset got no answer at all)
    BINDINGS = Base.BINDINGS + [Binding("e", "edit", "Edit", show=False), Binding("d", "delete", "Delete", show=False),
                                Binding("delete", "delete", show=False)]

    def body(self) -> ComposeResult:
        yield Static("", id="intro", markup=False)
        yield Menu([("", 34), ("", 14), ("", 8), ("", 14), ("", 16)], id="menu")

    def on_mount(self) -> None:
        self.reload()
        self.paint()

    def reload(self, keep: int | None = None) -> None:
        self.presets = store.list_presets()
        columns, with_used = self.preset_columns()
        items: list[Item] = []
        for p in self.presets:
            sm = store.summary(p)
            is_hy4 = p.family == "hy4-flash"
            is_native_strata = p.family in ("hy4-flash", "inkling-2-mini")
            req = "1 request" if is_hy4 else ((f"{p.parallel} request" + ("s" if p.parallel != 1 else "")) if p.parallel else "fit to card")
            tuned = "built-in" if is_native_strata else (f"tuned {p.tuned_at[5:]}" if p.is_tuned and p.tuned_at else "not tuned")
            used = f"last used {p.last_used[5:10]}" if p.last_used else "not used yet"
            cells = [p.name or "?", str(sm["model"]), S.context_label(p.context), req, Cell(tuned, MUTE if (not p.is_tuned and not is_native_strata) else ""), Cell(used, MUTE)]
            items.append(Item(cells if with_used else cells[:-1], key=("preset", p.name)))
        items.append(Item([Cell("Set up from the beginning", "")], key=("new", None)))
        items.append(Item([Cell("Use model files I already have", "")], key=("own", None)))
        m = self.query_one(Menu)
        m.columns = columns
        m.show_header = False
        m.set_items(items, keep if keep is not None else 0)

    def preset_columns(self) -> tuple[list[tuple[str, int]], bool]:
        """The preset row's columns for this window (0217 item 11: at 80 columns "tuned 10-16" and "last used 11-12" ran past the edge and ended in "not tu"): the name takes
        what is left, and "last used" is shown only where it fits whole."""
        room = self.width() - 2                          # (the two cells of the cursor)
        fixed = [14, 7, 13, 13]                          # model, context, requests, tuned
        used = 17
        if room - sum(fixed) - used >= 24:
            return [("", min(34, room - sum(fixed) - used)), *[("", w) for w in fixed], ("", 0)], True
        return [("", max(16, min(34, room - sum(fixed)))), *[("", w) for w in fixed]], False

    def on_resize(self, _e) -> None:
        if getattr(self, "presets", None) is not None and self.is_mounted:
            self.reload(keep=self.query_one(Menu).index)
        self.paint()

    def refresh_body(self) -> None:
        intro = Text()
        if self.presets:
            intro.append("Start from a saved preset\n", style=f"bold {ACCENT}")
        else:
            intro.append("No saved presets yet\n", style=f"bold {ACCENT}")
            intro.append("A preset remembers all your choices so next time you start with one key.\n", style=MUTE)
        try:                                                      # one quiet line when the last check found a newer engine (a local file; no network)
            from strata_control import engine_update
            line = engine_update.notice_line()
        except Exception:                                         # noqa: BLE001
            line = None
        if line:
            intro.append(line + "\n", style=MUTE)
        self.query_one("#intro", Static).update(intro)

    def explain(self) -> Text:
        c = self.ctx.computer
        return Text(f"This computer: {self.ctx.computer_line()}.  " + ("Preset: model, context, parallel requests, tuned settings, access. "
                    "Shared with the Strata app." if self.presets else "Choose how to begin."), style=MUTE)

    def head_text(self) -> Text:
        out = Text(no_wrap=True, overflow="crop")
        out.append("STRATA  Launch", style=f"bold {ACCENT}")
        info = self.ctx.computer_line(self.width() - len("STRATA  Launch") - 2)       # (0217 item 11: the card's name gives way, "free" stays)
        gap = max(2, self.width() - len("STRATA  Launch") - len(info))
        out.append(" " * gap + info, style=MUTE)
        return out

    def action_next(self) -> None:
        it = self.query_one(Menu).current
        if it is None:
            return
        kind, name = it.key
        if kind == "preset":
            self.app.start_preset(name)          # type: ignore[attr-defined]
        elif kind == "new":
            self.app.begin_setup()               # type: ignore[attr-defined]
        else:
            self.app.begin_setup(own_files=True)   # type: ignore[attr-defined]

    def action_edit(self) -> None:
        it = self.query_one(Menu).current
        if it and it.key[0] == "preset":
            self.app.begin_setup(preset=it.key[1])   # type: ignore[attr-defined]

    def action_delete(self) -> None:
        it = self.query_one(Menu).current
        if not it or it.key[0] != "preset":
            return
        name = it.key[1]

        def done(choice: str | None) -> None:
            if choice == "yes":
                try:
                    store.delete(name)
                except store.PresetError as e:
                    self.say(str(e), True)
                self.reload()
                self.say(f"Deleted '{name}'. Models and tuning results were not touched.")

        self.app.push_screen(ConfirmScreen(f"Delete the preset '{name}'?",
                                           "Only the saved choices go. The model files and tuning results stay.",
                                           [("yes", "Delete it"), ("no", "Keep it")]), done)

    def action_back(self) -> None:
        self.app.exit(None)


# ------------------------------------------------------------------------------------------------ confirm pop-up

class ConfirmScreen(ModalScreen):
    BINDINGS = [Binding("up", "move(-1)", show=False), Binding("down", "move(1)", show=False),
                Binding("enter", "pick", show=False), Binding("escape", "cancel", show=False)]
    DEFAULT_CSS = f"""
    ConfirmScreen {{ align: center middle; background: #0e1113 80%; }}
    #dialog {{ width: 70; max-width: 100%; height: auto; max-height: 100%; background: #161a1d; border: solid {ACCENT}; padding: 1 2; }}
    """

    def __init__(self, question: str, detail: str, options: list[tuple[str, str]]):
        super().__init__()
        self.question, self.detail, self.options = question, detail, options

    def compose(self) -> ComposeResult:
        pal = LIGHT if self.app.has_class("theme-light") else DARK          # the monitor's light theme reaches its pop-ups
        with Vertical(id="dialog"):
            yield Static(Text(self.question, style=f"bold {pal.accent}"), markup=False)
            yield Static(Text(self.detail, style=pal.mute), markup=False)
            yield Static("", markup=False)
            yield Menu(id="confirm-menu", palette=pal)
            yield Static(Text("Up/Down choose   Enter confirm   Esc cancel", style=pal.mute), markup=False)

    def on_mount(self) -> None:
        m = self.query_one(Menu)
        m.set_items([Item([label], key=key) for key, label in self.options], 0)
        if self.app.has_class("theme-light"):          # the monitor's light theme: a white dialog with dark text (the dark one is the default)
            dialog = self.query_one("#dialog")
            dialog.styles.background = LIGHT.panel
            dialog.styles.border = ("solid", LIGHT.accent)
            dialog.styles.color = LIGHT.fg
            self.styles.background = "#f9fafb 70%"
            for static in dialog.query(Static):
                static.styles.background = LIGHT.panel

    def action_move(self, step: int) -> None:
        self.query_one(Menu).move(step)

    def action_pick(self) -> None:
        self.dismiss(self.query_one(Menu).current.key)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ------------------------------------------------------------------------------------------------ 1 model

class ModelScreen(Base):
    step_id = "model"
    hints = "↑↓ choose   Enter next   o use my own files   Esc back   ? help"
    help_text = ("Pick the model. 'Needs' is the RAM + graphics memory the model's files take (the app's figure; the n-gram table stays on your SSD and is not counted). 'fits' means they fit the RAM and "
                 "the graphics memory together; 'may run slower' means they do not, so part is read from the SSD while it answers. Neither is a block: you can still try it.")

    BINDINGS = Base.BINDINGS + [Binding("o", "own", "Own files", show=False)]

    def body(self) -> ComposeResult:
        yield Static("", id="question", markup=False)
        yield Menu(id="menu", wrap_last=False)               # S2: a verdict that does not fit ends with an ellipsis (the full sentence is under the list)
        yield Input(placeholder="Folder with the model files, or one .gguf file", id="own", classes="field", disabled=True)

    def on_mount(self) -> None:
        self.rows = self.ctx.rows
        s = self.ctx.settings
        self.build_menu(initial=True)
        self.paint()
        if self.ctx.start_own_files and not s.model_path:
            self.ctx.start_own_files = False
            self.action_own()

    def widths(self) -> list[tuple[str, int]]:
        w = self.width()
        longest = max((len(r.title) for r in getattr(self, "rows", [])), default=0) + 2      # T4: the longest name and a gap, so name and size never touch
        if w >= 100:
            return [("Model", min(max(42, longest), 52)), ("Download", 10), ("Needs (RAM + VRAM)", 20), ("On this computer", 0)]
        return [("Model", min(max(41, longest), 43)), ("Size", 8), ("Needs", 7), ("Fits?", 0)]      # 80 columns: the verdict column keeps room (the long words are below the list)

    def build_menu(self, initial: bool = False) -> None:
        s = self.ctx.settings
        if not (s.family and s.model) and not s.model_path:
            rec = next((r for r in self.rows if r.recommended), None) or next((r for r in self.rows if r.verdict == "fits"), None)
            if rec is not None:
                s.family, s.model = rec.family, rec.model
        items: list[Item] = []
        sel = 0
        m = self.query_one(Menu)
        cols = self.widths()
        room = (m.size.width or self.width()) - 2 - sum(w for _t, w in cols[:-1])     # what is left of the line for the verdict (a scrollbar takes two cells: S2)
        for i, r in enumerate(self.rows):
            col = {"fits": "", "slower": WARN, "tight": WARN, "no": BAD, "disk": WARN}[r.verdict]
            tag = (" (recommended)" if r.recommended else "") + (" (installed)" if r.installed else "") + (" [experimental]" if r.experimental else "")
            need = f"{r.fit_gb:.0f} GB" if r.fit_gb else "-"
            verdict = r.words + tag
            if self.width() < 100:                           # a short verdict that fits the column; the full sentence is shown under the list
                base = {"fits": "fits", "slower": "may run slower", "tight": "tight", "no": "too big", "disk": "no disk"}[r.verdict]
                full, short = ((" (recommended)", " (rec.)") if r.recommended else (" (installed)", " (inst.)") if r.installed else
                               (" [experimental]", " [exp.]") if r.experimental else ("", ""))
                verdict = base + (full if len(base + full) <= room else short)      # "no disk (rec.)": the mark stays visible when the long one does not fit
            items.append(Item([r.title, f"{r.download_gb:.0f} GB", need, Cell(verdict, col)], key=(r.family, r.model)))
            if (s.family, s.model) == (r.family, r.model):
                sel = i
        m.columns = cols
        m.show_header = True
        m.window = max(4, (self.size.height or 30) - 12)
        m.set_items(items, sel if (initial or not m.items) else None)

    def on_resize(self, _e) -> None:
        if hasattr(self, "rows"):
            self.build_menu()
        self.paint()

    def refresh_body(self) -> None:
        q = Text()
        q.append("Which model?", style=f"bold {ACCENT}")
        s = self.ctx.settings
        if s.model_path:
            q.append(f"   your files: {s.model_path}", style=MUTE)
        else:
            info = "This computer: " + self.ctx.computer_line()
            if len("Which model?") + len(info) + 3 <= self.width():
                q.append(" " * (self.width() - len("Which model?") - len(info)) + info, style=MUTE)
        self.query_one("#question", Static).update(q)

    def explain(self) -> Text:
        m = self.menu()
        cur = m.current if m else None
        out = Text()
        if cur is not None and hasattr(self, "rows"):
            r = next((x for x in self.rows if (x.family, x.model) == cur.key), None)
            if r:
                out.append(f"{r.title}: {r.about}.\n", style=FG)
                if r.verdict != "fits":
                    out.append(f"{r.sentence}. ", style=BAD if r.verdict == "no" else WARN)
        # At 80 x 24 the box holds four lines: with a verdict sentence (three lines for a model that does not fit) this last, general sentence was cut
        # in the middle ("... stays on your SSD and is"). It stays in the help (?); a tall window shows it whole (GUI-TUI 0.2.17 QA).
        verdict_shown = cur is not None and hasattr(self, "rows") and any((x.family, x.model) == cur.key and x.verdict != "fits" for x in self.rows)
        if not verdict_shown or self.app.size.height >= 30:
            out.append("The n-gram table stays on your SSD and is not counted above.", style=MUTE)
        for title, why in self.unavailable_lines():
            out.append(f"\n{title}: {why}", style=MUTE)
        note = self.ctx.computer.other_vendor_note
        if note:
            out.append(" " + note, style=WARN)
        return out

    def unavailable_lines(self) -> list:
        """Models of the signed feed this engine does not run (none for a feed without any): listed under the models, never in the list."""
        if not hasattr(self, "_unavailable"):
            self._unavailable = self.ctx.catalog.unavailable(self.ctx.computer.raw)
        return self._unavailable

    def on_cursor(self) -> None:
        pass

    def flow(self) -> list[str]:
        """S1: "Step 1 of 10" counts the steps of the model under the cursor, so the number is the one every later step shows (the Drafting step
        exists only for some families); it changes while the cursor moves, never after the model is chosen."""
        m = self.menu()
        cur = m.current if m is not None else None
        return flow_for(self.ctx, cur.key[0] if cur is not None and isinstance(cur.key, tuple) else None)

    def commit(self) -> None:
        if self.input_open():
            return
        cur = self.menu().current
        if cur is not None:
            s = self.ctx.settings
            fam, model = cur.key
            if (s.family, s.model) != (fam, model) or s.model_path:
                s.model_path = None
            s.family, s.model = fam, model
            self.ctx.refresh_disk()
            from strata_cli.launch import modelstate
            s.drafting = modelstate.drafting_for_family(s.drafting, fam)         # the other families have no Drafting choice: back to Auto
            if fam in ("hy4-flash", "inkling-2-mini"):
                if fam == "hy4-flash":
                    s.parallel = None
                s.tune = "skip"
                s.kv = None
                s.gpu_roles = None
                s.extra = {}
            r = next(x for x in self.rows if (x.family, x.model) == (fam, model))
            if r.verdict == "no":
                self.ctx.notes.append(f"{r.title} does not fit this computer; it may be slow or run out of memory.")

    def action_next(self) -> None:
        if self.input_open():
            return
        if self.ctx.settings.model_path and not self.ctx.picked_in_list:
            self.app.advance()           # type: ignore[attr-defined]
            return
        super().action_next()

    def action_move(self, step: int) -> None:
        if self.input_open():
            return
        self.ctx.picked_in_list = True
        super().action_move(step)

    # ---- own files
    def action_own(self) -> None:
        self.open_field("own", self.ctx.settings.model_path or "", "Folder with the model files, or one .gguf file")
        self.say("Type or paste the folder (or the first .gguf file), then press Enter.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "own":
            return
        path = event.value.strip().strip('"')
        if not path:
            self.close_field("own")
            return
        self.check_own(path)

    def check_own(self, path: str) -> None:
        from strata_control import models
        problem = models.own_path_problem(path)
        if problem:
            self.say(f"{problem[0]} {problem[1]}", True)
            return
        from strata_control import retired
        if retired.folder_reason(models.absolute(path)):   # DeepSeek files: named, never used (strata_control/retired.py)
            self.say(retired.own_files(models.absolute(path)), True)
            return
        try:
            plan = models.plan_import(models.absolute(path))
        except Exception as e:                               # noqa: BLE001
            self.say(f"These files cannot be used: {e}. Pick the folder with every shard of one model, or a .gguf file of it.", True)
            return
        sel = plan.get("candidate_selection") or {}
        if sel.get("family") and sel.get("model") and not plan.get("issues"):
            s = self.ctx.settings
            s.model_path, s.family, s.model = models.import_target(path, plan), sel["family"], sel["model"]
            from strata_cli.launch import modelstate
            s.drafting = modelstate.drafting_for_family(s.drafting, s.family)
            if s.family in ("hy4-flash", "inkling-2-mini"):
                if s.family == "hy4-flash":
                    s.parallel = None
                s.tune = "skip"
                s.kv = None
                s.gpu_roles = None
                s.extra = {}
            self.ctx.picked_in_list = False
            self.ctx.refresh_disk()                          # N1: these files need no download, so the list stops marking the row (and the Save screen its "!" line)
            self.build_menu()
            self.close_field("own")
            gb = plan.get("total_bytes", 0) / 1e9
            self.say(f"These files are {sel['family']} {sel['model']} ({plan['file_count']} file(s), {gb:.0f} GB). Press Enter to use them.")
            self.menu().select((sel["family"], sel["model"]))
            self.paint()
        else:
            why = models.problem_words(plan.get("issues") or [], missing=plan.get("missing_shards")) or "they are not a model Strata knows"
            holds = next((m for m in plan.get("messages") or [] if "holds" in m), "")
            self.say(f"These files cannot be used as they are ({why}). " + (holds + " " if holds else "") +
                     "Check that every shard is in the folder, or pick a model from the list.", True)

    def action_back(self) -> None:
        if self.input_open():
            self.close_field("own")
            self.say("")
            return
        super().action_back()


# ------------------------------------------------------------------------------------------------ 2 cards (only with 2+ cards)

class CardsScreen(Base):
    """Main / Helper / Off for every card (strata_control/gpu_roles.py). The recommendation is already selected; M H O set the role of the
    card under the cursor, R goes back to the recommendation, D shows what the roles mean."""

    step_id = "cards"
    help_text = ("Main does the last steps of every answer; Helper holds part of the model; Off leaves the card alone. "
                 "Keys: M Main, H Helper, O Off, R back to recommended, D details.")
    hints = "↑↓ choose card   M Main   H Helper   O Off   R recommended   D details   Enter next   Esc back   ? help"
    BINDINGS = Base.BINDINGS + [Binding("m", "role('main')", show=False), Binding("h", "role('helper')", show=False),
                                Binding("o", "role('off')", show=False), Binding("r", "recommended", show=False),
                                Binding("d", "details", show=False), Binding("right", "cycle(1)", show=False),
                                Binding("left", "cycle(-1)", show=False)]

    def body(self) -> ComposeResult:
        yield Static(Text("Which graphics cards should it use?", style=f"bold {ACCENT}"), markup=False)
        yield Static("", id="reco", markup=False)
        yield Menu([("", 0)], id="menu")
        yield Static("", id="cardnotes", markup=False)

    def model_memory(self) -> float:
        s = self.ctx.settings
        r = next((x for x in self.ctx.rows if (x.family, x.model) == (s.family, s.model)), None)
        return float(r.memory_gb) if r is not None and r.memory_gb else 44.0

    def installed(self) -> list[int] | None:
        """The cards the installed model was set to by a person's choice (kept unless this launch names others)."""
        from strata_cli.launch.runner import installed_cards
        try:
            return installed_cards(self.ctx.base_config(), self.ctx.computer.usable_gpus)
        except Exception:                                    # noqa: BLE001
            return None

    def cards(self) -> list[dict]:
        from strata_control import gpu_roles
        c = self.ctx.computer
        return gpu_roles.rows(c.gpus, c.engine_vendor)

    def start_roles(self) -> dict[int, str]:
        """The roles the screen opens with: this launch's own, else the installed model's own choice, else the recommendation."""
        from strata_control import gpu_roles
        s, cards = self.ctx.settings, self.cards()
        if s.gpu_roles:
            return gpu_roles.normalize({int(k): v for k, v in s.gpu_roles.items()}, cards)
        if s.gpus:
            return gpu_roles.from_legacy(list(s.gpus), cards, "as_given")
        return self.baseline()

    def baseline(self) -> dict[int, str]:
        """What "no choice made here" means: the installed model's own cards, else the recommendation."""
        from strata_control import gpu_roles
        inst = self.installed()
        cards = self.cards()
        if inst:
            return gpu_roles.from_legacy(inst, cards, "as_given")
        return dict(gpu_roles.recommend(cards, self.ctx.computer.engine_vendor)["roles"])

    def on_mount(self) -> None:
        from strata_control import gpu_roles
        self.detail = False
        self.note = ""
        self.cardlist = self.cards()
        self.roles = self.start_roles()
        self.rec = gpu_roles.recommend(self.cardlist, self.ctx.computer.engine_vendor)
        wide = max((len(gpu_roles.short_name(c)) for c in self.cardlist), default=20)
        self.query_one(Menu).columns = [("", 8), ("", min(wide + 2, 34)), ("", 10), ("", 32), ("", 0)]
        self.fill(0)
        self.paint()

    def role_cell(self, role: str) -> str:
        words = [("main", "Main"), ("helper", "Helper"), ("off", "Off")]
        return "".join(f"[ {w} ] " if r == role else f"  {w}   " for r, w in words)

    def fill(self, index: int = 0) -> None:
        from strata_control import gpu_roles
        items = []
        for c in self.cardlist:
            i = int(c.get("index", 0))
            role = self.roles.get(i, "off")
            rec = self.rec["roles"].get(i)
            tag = "recommended" if rec == role else f"recommended: {rec}"
            mem = "built-in" if gpu_roles._shares(c) else f"{float(c.get('vram_gb') or 0):.0f} GB"
            items.append(Item([gpu_roles.gpu_label(c), gpu_roles.short_name(c), mem, self.role_cell(role), Cell(tag, "" if rec == role else MUTE)], key=i))
        self.query_one(Menu).set_items(items, index)

    def current_card(self) -> int | None:
        cur = self.menu().current if self.menu() else None
        return None if cur is None else int(cur.key)

    def action_role(self, role: str) -> None:
        from strata_control import gpu_roles
        i = self.current_card()
        if i is None:
            return
        self.roles, code = gpu_roles.set_role(self.roles, self.cardlist, i, role)
        self.note = gpu_roles.note_text(code, self.cardlist)
        self.fill(self.menu().index)
        self.paint()

    def action_cycle(self, step: int) -> None:
        order = ["main", "helper", "off"]
        i = self.current_card()
        if i is None:
            return
        self.action_role(order[(order.index(self.roles.get(i, "off")) + step) % 3])

    def action_recommended(self) -> None:
        self.roles = dict(self.rec["roles"])
        self.note = ""
        self.fill(self.menu().index)
        self.paint()

    def action_details(self) -> None:
        self.detail = not self.detail
        self.paint()

    def is_cpu_engine(self) -> bool:
        try:
            return (self.ctx.catalog.installed_backend() or "") in ("cpu", "metal")
        except Exception:                                    # noqa: BLE001
            return False

    def refresh_body(self) -> None:
        if not hasattr(self, "roles"):
            return
        from strata_control import gpu_fit, gpu_roles
        c = self.ctx.computer
        try:
            self.query_one("#reco", Static).update(Text(gpu_roles.sentence(self.cardlist, c.engine_vendor), style=MUTE))
        except Exception:                                    # noqa: BLE001
            return
        lines = Text()
        if self.note:
            lines.append(f"  {self.note}\n", style=MUTE)
        spill = gpu_roles.spill_for(self.roles, self.cardlist, self.model_memory(), c.ram_gb)
        for w in gpu_roles.warnings(self.roles, self.cardlist, spill_gb=spill, engine_runs_cpu_only=self.is_cpu_engine()):
            fix = ""
            if w["fix"]:
                n = gpu_roles.gpu_label(next(x for x in self.cardlist if int(x.get("index", 0)) == w["fix"]["card"]))
                fix = f"   (select {n} and press {w['fix']['role'][0].upper()})"
            lines.append(f"  ! {w['text']}{fix}\n", style=WARN)
        if self.detail:
            for word, text in (("Main", "does the last steps of every answer and keeps the conversation. Only one."),
                               ("Helper", "holds a share of the model, so more of it fits on graphics cards."),
                               ("Off", "Strata does not touch the card.")):
                lines.append(f"  {word}: {text}\n", style=MUTE)
        for g in gpu_fit.for_vendor(c.gpus, c.engine_vendor)[1]:
            lines.append(f"  Not used: {g.get('name')} (another make of card than this engine drives).\n", style=MUTE)
        try:
            self.query_one("#cardnotes", Static).update(lines)
        except Exception:                                    # noqa: BLE001
            pass

    def explain(self) -> Text:
        return Text("Strata recommends the roles shown. A Helper holds part of the model; Off leaves a card alone (for a game or the screen). "
                    "Any card is one key away; the choice is yours.", style=MUTE)

    def commit(self) -> None:
        if not hasattr(self, "roles"):
            return
        s = self.ctx.settings
        if self.roles == self.baseline():
            s.gpu_roles, s.gpus = None, None                 # the recommendation (or the installed choice) stays "not set": a card added later is re-recommended
        else:
            s.gpu_roles, s.gpus = {str(i): r for i, r in sorted(self.roles.items())}, None


# ------------------------------------------------------------------------------------------------ 3 folder

class FolderScreen(Base):
    step_id = "folder"
    hints = "Type to change the folder   Enter next   Esc back   ? help"
    help_text = "Models are big. Pick a drive with free space. The default is where Strata keeps its data."

    BINDINGS = [b for b in Base.BINDINGS if b.key not in ("q", "up", "down")] + [Binding("down", "noop", show=False)]

    def body(self) -> ComposeResult:
        yield Static(Text("Where should the model files be stored?", style=f"bold {ACCENT}"), markup=False)
        yield Static("", id="folderinfo", markup=False)
        yield Input(id="folderpath", classes="field open always")

    def on_mount(self) -> None:
        s = self.ctx.settings
        f = self.query_one("#folderpath", Input)
        f.value = s.data_dir or str(self.ctx.default_folder())
        f.focus()
        self.paint()

    def action_noop(self) -> None:
        pass

    def on_input_changed(self, event: Input.Changed) -> None:
        self.paint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_next()

    def state(self, value: str | None = None):
        return self.ctx.model_state(value or None)

    def need_gb(self, st=None) -> float:
        """The room to ask for: what is still to download plus working room (a little when the model is already there)."""
        st = st or self.state()
        return st.remaining_gb + (2.0 if (st.installed or st.own_files) else 8.0)

    def refresh_body(self) -> None:
        try:
            value = self.query_one("#folderpath", Input).value.strip()
        except Exception:                                    # noqa: BLE001
            return
        free = self.ctx.catalog.disk_free(value or self.ctx.default_folder())
        st = self.state(value)
        need = self.need_gb(st)
        s = self.ctx.settings
        out = Text()
        if s.model_path:
            out.append(f"Your model files stay where they are: {s.model_path}\n", style=FG)
            term = self.ctx.drafter_term()                   # 0217 item 10: the drafter this launch uses, by its own name
            out.append(f"Strata only writes its prepared data here (packs{', ' + term if term else ''}).\n", style=MUTE)
        out.append(st.headline(), style=FG)
        out.append("\n")
        if st.installed and st.folder is not None:
            out.append(f"It is stored in {st.folder}\n", style=MUTE)
        if st.parts and (st.installed or st.own_files):
            for p in st.parts:
                out.append(f"  - missing: {p.label}, about {p.gb:.1f} GB\n", style=WARN)
        if free is not None:
            warn = free < need
            out.append(f"Free space in that folder's drive: {free:,.0f} GB", style=WARN if warn else FG)
            if warn:
                out.append(f"   Not enough for the {need:.0f} GB needed. You can still go on, but the download is likely to stop; pick a bigger drive.", style=WARN)
        else:
            out.append("The free space of that folder could not be read.", style=MUTE)
        moving = self.models_that_would_move(value)
        if moving:
            out.append(f"\nModel files Strata already keeps in {moving} stay there and keep working. This folder is used for new downloads. "
                       "To move them here, use: strata models move.", style=MUTE)
        self.query_one("#folderinfo", Static).update(out)

    def models_that_would_move(self, value: str) -> str | None:
        """The folder holding model files today when a different one is typed (setup gathers the model files into the data folder), else None."""
        try:
            old = Path(self.ctx.default_folder())
            new = Path(value).expanduser() if value else old
            if os.path.normcase(str(new.resolve())) == os.path.normcase(str(old.resolve())):
                return None
            models = old / "models"
            if models.is_dir() and any(models.iterdir()):
                return str(old)
        except OSError:
            pass
        return None

    def explain(self) -> Text:
        if self.state().installed:
            return Text("This model is already on this computer, so the folder starts where it is. Press Enter to keep it. "
                        "Another folder only matters for files that still have to be downloaded.", style=MUTE)
        return Text("Models are large files. Strata keeps them in one folder; the default is Strata's own data folder. "
                    "Press Enter to keep it, or type another folder (it is created if it does not exist). Models you already have stay where "
                    "they are and stay usable; the new folder is for new downloads.", style=MUTE)

    def typed(self) -> str:
        return self.query_one("#folderpath", Input).value.strip().strip('"')

    def action_next(self) -> None:
        """The folder is checked before going on (B05): a path that cannot hold the models is said here, in plain words, and not saved."""
        from strata_control import folders
        found = folders.problem(self.typed())
        if found:
            self.say(folders.words(found), True)
            return
        super().action_next()

    def commit(self) -> None:
        value = self.typed()
        default = str(self.ctx.default_folder())
        self.ctx.settings.data_dir = None if (not value or value == default) else value


# ------------------------------------------------------------------------------------------------ 4 context

class ContextScreen(Base):
    step_id = "context"
    hints = "↑↓ choose   Enter next   m memory format   Esc back   ? help"
    help_text = ("Context is how much text the model can see at once (your chat, files, tool output). A longer one lets agents read "
                 "bigger projects, uses more memory, and makes the first reply of a long chat slower.")

    BINDINGS = Base.BINDINGS + [Binding("m", "memory", "Memory format", show=False), Binding("c", "custom", "Custom", show=False)]

    def body(self) -> ComposeResult:
        yield Static(Text("How much text should it remember at once?", style=f"bold {ACCENT}"), markup=False)
        yield Menu([("", 7), ("", 24), ("", 12), ("", 0)], id="menu")
        yield Static("", id="kvline", markup=False)
        yield Input(placeholder="a size like 100k, or a number of tokens", id="custom", classes="field", disabled=True)

    def hints_text(self) -> str:
        text = self.hints
        if self.ctx.settings.family in ("hy4-flash", "inkling-2-mini"):
            text = text.replace("   m memory format", "")
        return text.replace("↑↓", self.g.updown)

    def kv(self) -> str:
        s = self.ctx.settings
        return s.kv or "int8"

    def _kv_gb(self, tokens: int) -> float:
        s, cat = self.ctx.settings, self.ctx.catalog
        if s.family == "hy4-flash":
            from strata_control import hy4_fit
            return tokens * hy4_fit.KV_GB_PER_TOKEN
        if s.family == "inkling-2-mini":
            from strata_control import inkling_fit
            return tokens * inkling_fit.KV_GB_PER_TOKEN
        return cat.kv_gb(tokens, self.kv())

    def on_mount(self) -> None:
        s = self.ctx.settings
        cat = self.ctx.catalog
        fam, model = s.family, s.model
        self.rec = cat.max_context(fam, model, self.ctx.computer.raw) if fam and model else 32768
        if s.context is None:
            s.context = self.rec
        self.build()
        self.paint()

    def build(self) -> None:
        s, cat = self.ctx.settings, self.ctx.catalog
        items = []
        sel = 0
        sizes = cat.contexts_for(s.family, s.model)
        for i, c in enumerate(sizes):
            kvgb = self._kv_gb(c)
            note, style = "", ""
            if c == self.rec:
                note = "recommended"
            elif c > max(self.rec * 2, 131072) and c <= 262144:
                note, style = "tight on this computer", WARN
            elif c > 262144:
                note, style = "experimental: rope scaling", WARN
            items.append(Item([S.context_label(c), S.words_for(c), f"+{kvgb:.1f} GB", Cell(note, style)], key=c))
            if c == s.context:
                sel = i
        if s.context and s.context not in sizes:
            items.append(Item([S.context_label(s.context), S.words_for(s.context), f"+{self._kv_gb(s.context):.1f} GB",
                               Cell("your own size", "")], key=s.context))
            sel = len(items) - 1
        items.append(Item(["Custom…"], key="custom"))
        self.query_one(Menu).set_items(items, sel)

    def refresh_body(self) -> None:
        s = self.ctx.settings
        if s.family == "hy4-flash":
            self.query_one("#kvline", Static).update(Text("Conversation memory: managed automatically by Hy4-Flash", style=MUTE))
            return
        if s.family == "inkling-2-mini":
            self.query_one("#kvline", Static).update(Text("Conversation memory: managed automatically by Inkling-2-Mini (SWA-1024 + compact Global KV)", style=MUTE))
            return
        kv = self.kv()
        label = {"int8": "8-bit (recommended: what the published numbers were measured with)",
                 "q4_0": "4-bit (half the memory, a little less precise on long documents)",
                 "k8v4": "mixed 8-bit and 4-bit", "fp16": "full precision"}.get(kv, kv)
        small = (s.context or 0) <= 8192
        self.query_one("#kvline", Static).update(Text(
            f"Conversation memory: {label}" + ("" if not small else " (small contexts always use full precision)") + "   press m to change",
            style=MUTE))

    def explain(self) -> Text:
        cur = self.menu().current if self.menu() else None
        out = Text("More context lets agents read bigger projects. It uses more memory and the first reply of a long chat is slower. "
                   "The memory shown is only the extra for the conversation itself.", style=MUTE)
        if cur is not None and isinstance(cur.key, int) and cur.key > 262144:
            out.append("\nThis is past what the model was trained on; setup adds rope scaling and answers may get less reliable.", style=WARN)
        return out

    def action_memory(self) -> None:
        s = self.ctx.settings
        if s.family in ("hy4-flash", "inkling-2-mini"):
            return
        order = ["int8", "q4_0"]
        cur = s.kv or "int8"
        s.kv = order[(order.index(cur) + 1) % len(order)] if cur in order else "int8"
        self.build()
        self.paint()

    def action_custom(self) -> None:
        self.open_field("custom", "", "a size like 100k, or a number of tokens")
        self.say("Type a size and press Enter.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "custom":
            return
        try:
            n = S.parse_context(event.value)
        except ValueError as e:
            self.say(str(e), True)
            return
        self.ctx.settings.context = n
        self.close_field("custom")
        self.build()
        self.say(f"{S.context_label(n)} chosen.")
        self.paint()

    def action_back(self) -> None:
        if self.input_open():
            self.close_field("custom")
            return
        super().action_back()

    def commit(self) -> None:
        cur = self.menu().current
        if cur is None:
            return
        if cur.key == "custom":
            return
        self.ctx.settings.context = int(cur.key)

    def action_next(self) -> None:
        if self.input_open():
            return
        cur = self.menu().current
        if cur is not None and cur.key == "custom":
            self.action_custom()
            return
        super().action_next()


# ------------------------------------------------------------------------------------------------ 5 pictures

class PicturesScreen(Base):
    step_id = "pictures"
    help_text = "Pictures let the model read screenshots, photos and scanned pages. It is a small extra download and a little graphics memory."

    def body(self) -> ComposeResult:
        yield Static(Text("Should the model read pictures?", style=f"bold {ACCENT}"), markup=False)
        yield Menu(id="menu")

    def row(self):
        s = self.ctx.settings
        return next((x for x in self.ctx.rows if (x.family, x.model) == (s.family, s.model)), None)

    def is_hy4(self) -> bool:
        return self.ctx.settings.family == "hy4-flash"

    def is_inkling(self) -> bool:
        return self.ctx.settings.family == "inkling-2-mini"

    def is_native_strata(self) -> bool:
        return self.is_hy4() or self.is_inkling()

    def reader(self) -> str:
        """"present" when the installed engine carries the picture reader (strata-vision; built into strata-hy4 / strata-inkling), "missing" when an engine is installed without it,
        "unknown" before any engine is installed (the first launch fetches it)."""
        if self.is_native_strata():
            return "present"
        return self.ctx.catalog.reader_state()

    def on_mount(self) -> None:
        s = self.ctx.settings
        r = self.row()
        self.supported = True if r is None else r.images
        self.not_on_card = False if self.is_native_strata() else self.ctx.catalog.pictures_not_on_card()
        items = [Item(["No pictures"], key="off")]
        if self.supported and self.not_on_card:              # T14: an Intel Arc card starts text-only for now: do not offer what does not work
            items.append(Item([Cell("Yes, read pictures: not available on Intel Arc cards yet", MUTE)], key="on", enabled=False))
        elif self.supported:
            items.append(Item(["Yes, read pictures" + ("  (this engine has no picture reader)" if self.reader() == "missing" else "")], key="on"))
        else:
            items.append(Item([Cell("Yes, read pictures: not available for this model yet", MUTE)], key="on", enabled=False))
        sel = 1 if s.vision not in ("off", "no") and self.supported and not self.not_on_card and self.reader() != "missing" else 0       # pictures are on unless someone said off
        self.query_one(Menu).set_items(items, sel)
        self.paint()

    def explain(self) -> Text:
        if not self.supported:
            return Text("This model cannot read pictures yet, so pictures stay off. Pick another model if you need them.", style=WARN)
        if self.is_hy4():
            return Text("Pictures are on by default. They add a 0.9 GB vision shard (Hy4-Flash-Vision.strata, if it is not on this computer yet), "
                        "built directly into strata-hy4. Say No if you only chat in text.", style=MUTE)
        if self.is_inkling():
            return Text("Pictures are on by default. They add a 0.6 GB lossless BF16 vision shard (Inkling-2-Mini-Vision.strata, if it is not on this computer yet), "
                        "built directly into strata-inkling. Say No if you only chat in text.", style=MUTE)
        amd = any(str(g.get("vendor")) == "amd" for g in self.ctx.computer.usable_gpus)
        on_card = self.ctx.catalog.picture_default(True) == "gpu"
        extra = ("a 0.9 GB download (if it is not on this computer yet). On your AMD card the picture reader runs on the graphics card "
                 "(about 1 GB of its memory, under a second per picture) and starts on the processor by itself if it cannot."
                 if amd and on_card and self.reader() != "missing" else
                 "a 0.9 GB download (if it is not on this computer yet), and about 1.4 GB of graphics memory kept free for the picture reader "
                 "(text is a few % slower)." if on_card else
                 "a 0.9 GB download (if it is not on this computer yet). The picture reader runs on the CPU: a few seconds per picture, nothing "
                 "taken from the graphics card.")
        if getattr(self, "not_on_card", False):
            return Text("Intel Arc cards cannot read pictures with Strata yet, so pictures stay off and the model answers text only. "
                        "Pictures work on NVIDIA and AMD cards and on a Mac.", style=WARN)
        out = Text("Pictures are on by default. They add " + extra + " Say No if you only chat in text.", style=MUTE)
        if self.reader() == "missing":
            out.append("\nHeads up: the engine installed here does not carry the picture reader, so the model would answer text only. "
                       "Update the engine, or say No.", style=WARN)
        return out

    def commit(self) -> None:
        key = self.menu().current.key
        if key == "on":                                    # yes: the reader on the card (NVIDIA, AMD) or the CPU, as the default says
            if self.is_native_strata():
                key = "on"
            else:
                device = self.ctx.catalog.picture_default(True)
                key = device if device != "off" else "on"
        self.ctx.settings.vision = key


# ------------------------------------------------------------------------------------------------ drafting

def drafting_launch_word(ctx) -> str | None:
    """The Drafting word this launch was given last time (its own run config): a chosen drafter stays chosen at a start with no word (Runner.keep_saved_word,
    keep_saved_off), so the Drafting step shows it and an Auto picked there is written to replace it."""
    try:
        from strata_cli.launch import runner
        return runner.saved_launch_word(runner.launch_name_of(ctx.settings))
    except Exception:                                        # noqa: BLE001 - no readable file, no word
        return None


def drafting_saved_choice(ctx, choices) -> str:
    """The Drafting row a start with no new choice runs, in the order the start reads them: the model's own saved choice (the app's Models screen), the draft
    layer switched off with no choice (Off), the word this launch was given last time, else Auto."""
    from strata_cli.launch import modelstate
    base = ctx.base_config() or {}
    own = str(base.get("drafting") or "").strip().lower()
    if own in choices:
        return own
    if modelstate.layer_off_without_choice(base):
        return "off"
    word = drafting_launch_word(ctx)
    return word if word in choices and modelstate.drafting_for_family(word, ctx.settings.family) is not None else "auto"


class DraftingScreen(Base):
    step_id = "drafting"
    help_text = "Drafting is how the model guesses the next words ahead to make answers faster. One drafter runs at a time."

    CHOICES = (("auto", "Auto (recommended)"), ("mtp", "MTP"), ("dflash2", "DFlash2"), ("off", "Off"))

    def body(self) -> ComposeResult:
        yield Static(Text("How should the model guess ahead?", style=f"bold {ACCENT}"), markup=False)
        yield Menu(id="menu")

    def on_mount(self) -> None:
        s = self.ctx.settings
        from strata_cli.launch import modelstate
        now = {"dflash2": "DFlash2", "mtp": "MTP", "off": "Off"}[modelstate.resolved_drafting(None, s.family)]
        items = [Item([label, Cell(now, "") if key == "auto" else Cell("", "")], key=key) for key, label in self.CHOICES]
        shown = s.drafting if s.drafting in dict(self.CHOICES) else drafting_saved_choice(self.ctx, dict(self.CHOICES))
        sel = [key for key, _ in self.CHOICES].index(shown)
        self.query_one(Menu).columns = [("", 24), ("", 0)]
        self.query_one(Menu).set_items(items, sel)
        self.paint()

    def explain(self) -> Text:
        return Text("")

    def commit(self) -> None:
        key = self.menu().current.key
        from strata_cli.launch import modelstate
        # Auto stays "nobody chose" (a saved preset writes nothing) unless the model's file or this launch holds another choice: then Auto is written, so it wins
        base = self.ctx.base_config() or {}
        layer_off = modelstate.layer_off_without_choice(base)  # the draft layer is off and nothing else is saved: the step showed Off
        if layer_off and key == "off" and self.ctx.settings.drafting is None:
            return                                            # Off left as it is: nothing new is written, the draft layer's "mtp": false keeps it
        saved = "off" if layer_off else base.get("drafting") if base.get("drafting") is not None else drafting_launch_word(self.ctx)
        self.ctx.settings.drafting = modelstate.drafting_to_save(key, saved)


# ------------------------------------------------------------------------------------------------ 6 parallel

class ParallelScreen(Base):
    step_id = "parallel"
    help_text = ("Parallel requests: how many people or apps are answered at the same time. More at once makes each answer a bit slower "
                 "and needs graphics memory the model could use otherwise.")

    def body(self) -> ComposeResult:
        yield Static(Text("How many requests at the same time?", style=f"bold {ACCENT}"), markup=False)
        yield Menu(id="menu")

    def shared_room(self) -> dict | None:
        """S3: on a chip whose memory is the computer's own (a Mac, Strix Halo) there is no card size to count, so the figure is the shared memory the model
        and one request leave free, and what each further request takes (its conversation memory at the chosen context: the Context step's own figure).
        {"total", "model", "left", "each", "more"} in GB (more = further requests that fit, 0 or more), None when the memory or the model is unknown."""
        c, s = self.ctx.computer, self.ctx.settings
        if not c.ram_gb:
            return None
        row = next((x for x in self.ctx.rows if (x.family, x.model) == (s.family, s.model)), None)
        need = float(row.fit_gb or row.memory_gb or 0) if row is not None else 0.0
        if not need:
            return None
        try:
            cat = self.ctx.catalog
            if any(g.get("unified_gb") for g in c.usable_gpus):          # a Mac: Metal keeps this share of the memory for one program (strata_control/mac_fit.py)
                from strata_control import mac_fit
                budget = mac_fit.gpu_budget_gb(float(c.ram_gb))
            else:
                from strata_cli.launch.catalog import SYSTEM_ROOM_GB
                budget = float(c.ram_gb) - SYSTEM_ROOM_GB
            tokens = s.context or cat.max_context(s.family, s.model, c.raw) or 32768
            each = cat.kv_gb(int(tokens), s.kv or "int8")
        except Exception:                                    # noqa: BLE001 - without the figure the screen says what it always said
            return None
        left = budget - need - each
        if each <= 0:
            return None
        return {"total": float(c.ram_gb), "model": need, "left": max(left, 0.0), "each": each, "more": max(0, int(left // each)) if left > 0 else 0}

    def on_mount(self) -> None:
        vram = self.ctx.computer.main_vram_gb
        s = self.ctx.settings
        is_inkling = s.family == "inkling-2-mini"
        if is_inkling:
            self.fit = 8 if (vram or 0) >= 22 else (4 if (vram or 0) >= 15 else 1)
            choices = (1, 4, 8, 32, 64)
        else:
            self.fit = S.fit_parallel(vram) if vram else 1
            choices = (1, 2, 3, 4)
        shared = not vram and bool(self.ctx.computer.usable_gpus)
        room = self.shared_room() if shared else None
        items = [Item([f"Fit to this computer ({self.fit})" if shared else f"Fit to my graphics card ({self.fit})", Cell("recommended", "")], key=None)]
        for n in choices:
            note, style = "", ""
            if room is not None and n > 1:
                note = f"+{(n - 1) * room['each']:.1f} GB more"
                if n - 1 > room["more"]:
                    note, style = f"may not fit: slower answers ({note})", WARN
            elif not is_inkling and n > self.fit and room is None:
                note, style = "may not fit: slower answers", WARN
            elif is_inkling and n >= 32 and (vram or 0) < 22:
                note, style = "high concurrency: uses more KV memory", WARN
            items.append(Item([f"{n} request" + ("s" if n != 1 else ""), Cell(note, style)], key=n))
        if s.parallel is None:
            sel = 0
        elif s.parallel in choices:
            sel = 1 + choices.index(s.parallel)
        else:
            sel = 1
        self.query_one(Menu).columns = [("", 36), ("", 0)]
        self.query_one(Menu).set_items(items, sel)
        self.paint()

    def explain(self) -> Text:
        vram = self.ctx.computer.main_vram_gb
        if not vram and self.ctx.computer.usable_gpus:      # T3: a Mac or a Strix Halo PC has a chip whose memory is the computer's own, not a card with a size
            room = self.shared_room()
            if room is not None:                            # S3: a figure from the shared memory
                fits = "so every choice fits" if room["more"] >= 3 else f"so {room['more']} more fit" if room["more"] > 1 else "so one more fits"
                tail = (f"About {room['left']:.0f} GB is left, and each further request takes about {room['each']:.1f} GB, {fits}" if room["more"] else
                        f"Not much is left, and each further request takes about {room['each']:.1f} GB")
                return Text(f"This computer's graphics use the same memory as everything else ({room['total']:.0f} GB in all; the model takes about "
                            f"{room['model']:.0f} GB). {tail}. 1 is the safe default; a team or several coding agents: more.", style=MUTE)
            return Text("This computer's graphics use the same memory as everything else, so Strata cannot count graphics memory to say how many "
                        "requests fit. 1 is the safe choice; more at once need more of that memory. One person: pick 1. A team or several coding "
                        "agents: more.", style=MUTE)
        many = len(self.ctx.computer.usable_gpus) > 1
        if self.ctx.settings.family == "inkling-2-mini":
            return Text(f"Your {'biggest card has' if many else 'card has'} {vram:.0f} GB of graphics memory, which fits {self.fit} at once by Inkling-2-Mini's rule "
                        "(12 GB: 1, 16 GB: 4, 24 GB or more: 8; continuous batching supports up to 64). One person: pick 1. A team or several coding agents: 4 to 64."
                        if vram else "No graphics card was found, so Strata cannot say how many fit. 1 is the safe choice.", style=MUTE)
        return Text(f"Your {'biggest card has' if many else 'card has'} {vram:.0f} GB of graphics memory, which fits {self.fit} at once by Strata's rule "
                    "(12 GB: 1, 20 GB: 2, 24 GB: 3, 32 GB or more: 4). One person: pick 1. A team or several coding agents: more. "
                    "More than fits still starts, only slower." if vram else
                    "No graphics card was found, so Strata cannot say how many fit. 1 is the safe choice.", style=MUTE)

    def commit(self) -> None:
        self.ctx.settings.parallel = self.menu().current.key


# ------------------------------------------------------------------------------------------------ 7 tuning

class TuningScreen(Base):
    step_id = "tuning"
    hints = "↑↓ choose   Enter next   a answer-changing settings on/off   Esc back   ? help"
    help_text = ("Tuning measures a few settings on this computer and keeps only real gains. It runs after the model is downloaded, "
                 "before serving starts. You can stop it at any time: nothing changes until it is done.")

    BINDINGS = Base.BINDINGS + [Binding("a", "allow", "Allow", show=False)]

    def body(self) -> ComposeResult:
        yield Static(Text("Tune for the best speed on this computer?", style=f"bold {ACCENT}"), markup=False)
        yield Menu([("", 12), ("", 22), ("", 0)], id="menu")
        yield Static("", id="allowline", markup=False)

    def on_mount(self) -> None:
        s = self.ctx.settings
        items = [Item(["Quick", "about 20 minutes", "checks the most likely gains"], key="quick"),
                 Item(["Thorough", "up to 2 hours", "tries more settings, measures each one several times"], key="thorough"),
                 Item(["Skip", "", "use the default settings (already good on most computers)"], key="skip")]
        # T5: a first setup starts on Skip (the default of s.tune): Quick is one key down and a person who presses Enter all the way does not wait 20 minutes
        self.query_one(Menu).set_items(items, {"quick": 0, "thorough": 1, "skip": 2}.get(s.tune, 2))
        self.paint()

    def refresh_body(self) -> None:
        on = self.ctx.settings.tune_allow_small
        # (S2: at 80 columns the "press a to change" half wrapped in the middle of its words; the key hint line under the screen says the same)
        tail = "   press a to change" if self.width() >= 90 else ""
        self.query_one("#allowline", Static).update(Text(
            f"\n  [{'x' if on else ' '}] Allow settings that change answers very slightly (same quality){tail}", style=MUTE))

    def explain(self) -> Text:
        later = ""
        if self.ctx.fresh_setup and not self.ctx.settings.is_tuned:
            later = " Skip lets you start sooner; you can tune any time later with: strata launch --tune quick."
        return Text("Tuning is optional: it keeps only real gains, and the result is a preset the Strata app can use too. Close other heavy "
                    "programs while it measures." + later, style=MUTE)

    def action_allow(self) -> None:
        self.ctx.settings.tune_allow_small = not self.ctx.settings.tune_allow_small
        self.paint()

    def commit(self) -> None:
        self.ctx.settings.tune = self.menu().current.key
        self.ctx.tune_chosen = True


# ------------------------------------------------------------------------------------------------ 8 advanced

class AdvancedScreen(Base):
    step_id = "advanced"
    hints = "Enter next   ↑↓ choose   e edit   Esc back   Del reset   x experimental   / filter   a add by name   c copy command   Tab scroll command"
    help_text = ("Every engine setting with its default, the value the tuner chose and your own value. You never have to touch this. "
                 "A name the engine does not list gets a warning, not a refusal.")

    BINDINGS = Base.BINDINGS + [
        Binding("e", "edit", show=False), Binding("delete", "reset", show=False), Binding("backspace", "reset", show=False),
        Binding("x", "experimental", show=False), Binding("slash", "filter", show=False), Binding("a", "add", show=False),
        Binding("c", "copy", show=False), Binding("n", "go_next", show=False), Binding("right", "go_next", show=False),
        Binding("tab", "scroll_command", show=False)]

    def body(self) -> ComposeResult:
        yield Static("", id="advhead", markup=False)
        yield Menu([("Setting", 26), ("Default", 11), ("Tuned", 13), ("Yours", 10), ("What it does", 0)], id="menu", window=8, wrap_last=False)   # S2: one line per setting; the selected one's full text is under the table
        with VerticalScroll(id="cmdbox"):                    # the long command has its own scroll area (Tab, then the arrows) instead of running off the screen
            yield Static("", id="cmdline", markup=False)
        yield Input(id="edit", classes="field", disabled=True)

    def on_mount(self) -> None:
        self.only_experimental = False
        self.filter = ""
        self.mode = ""         # "", "edit", "filter", "add"
        self.rows: list[P.Row] = []
        self.load_rows()
        self.paint()

    def load_rows(self) -> None:
        self.rows = self.ctx.advanced_rows()
        self.build()

    def build(self) -> None:
        items = []
        for r in self.rows:
            if self.only_experimental and not r.experimental:
                continue
            if self.filter and self.filter.lower() not in r.name.lower() and self.filter.lower() not in r.description.lower():
                continue
            d, t, y = r.cells()[1:]
            desc = ("[experimental] " if r.experimental else "") + r.description
            cells = [r.name, d, t, Cell(y, ACCENT if y != "-" else ""), Cell(desc, WARN if r.experimental else (BAD if not r.known else ""))]
            if self.narrow():
                del cells[2]                                 # the Tuned column: its value is in the line under the table
            items.append(Item(cells, key=r.name))
        add = [Cell("+ Add a setting by name…", ""), "", "", "", Cell("for flags of newer engine versions", MUTE)]
        items.append(Item(add[:3] + add[4:] if self.narrow() else add, key="__add__"))
        m = self.query_one(Menu)
        # 0217 item 11: at 80 columns the description had 16 cells ("Qwen: 0.2.6 beh…"); without the Tuned column (said under the table instead) it has 29
        m.columns = ([("Setting", 26), ("Default", 10), ("Yours", 9), ("What it does", 0)] if self.narrow() else
                     [("Setting", 26), ("Default", 11), ("Tuned", 13), ("Yours", 10), ("What it does", 0)])
        self.fit_box()
        m.set_items(items, min(m.index, len(items) - 1))

    def fit_box(self) -> None:
        """The command box is as tall as its text (the title, the command, "Same as", a note) while the list keeps at least three rows; it was a third of the body, so "Same as"
        lost its end at 80x24 (0217 item 11)."""
        import textwrap
        m = self.query_one(Menu)
        body_h = self.query_one("#body").size.height or max(6, (self.size.height or 30) - 8)
        wrap_w = max(20, self.width() - 8)                     # (the box has a scroll bar and the words wrap a little earlier than a plain count says: one spare line below)
        texts = ["  " + self.ctx.command_text(), "  Same as: " + self.ctx.flags_line(), *(["  " + self.ctx.engine_note] if self.ctx.engine_note else [])]
        need = 2 + sum(len(textwrap.wrap(text, wrap_w)) or 1 for text in texts)
        room = body_h - 7 if body_h >= 14 else body_h // 3                  # (a small body keeps the old third: the list needs its rows)
        box_h = max(3, min(need, room))
        self.query_one("#cmdbox").styles.height = box_h
        m.window = max(3, body_h - box_h - 4)                # the heading, the column titles, the "n of m" line and the box's rule

    def narrow(self) -> bool:
        return self.width() < 100

    def on_resize(self, _e) -> None:
        if hasattr(self, "rows"):
            self.build()
        self.paint()

    def refresh_body(self) -> None:
        eng = self.ctx.engine_note
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append("All settings", style=f"bold {ACCENT}")
        if self.width() >= 90:
            t.append("   (optional · for tinkerers)", style=MUTE)
            extra = "   / filter" + (f": {self.filter}" if self.filter else "") + "   x show experimental only" + (" (on)" if self.only_experimental else "")
        elif self.width() >= 70:                             # 80 columns: the line was one word too long and wrapped ("only" alone on a second row)
            t.append("   (optional)", style=MUTE)
            extra = "   / filter" + (f": {self.filter}" if self.filter else "") + "   x experimental only" + (" (on)" if self.only_experimental else "")
        else:
            extra = "  (optional)" + (f"  filter: {self.filter}" if self.filter else "") + ("  experimental only" if self.only_experimental else "")
        t.append(extra, style=MUTE)
        self.query_one("#advhead", Static).update(t)
        s = self.ctx.settings
        cmd = self.ctx.command_text()
        out = Text()
        out.append("Command that will run (your choices above are in it)\n", style=f"bold {ACCENT}")
        out.append("  " + cmd + "\n")
        out.append("  Same as: " + self.ctx.flags_line() + "\n", style=MUTE)
        if eng:
            out.append("  " + eng, style=WARN)
        self.query_one("#cmdline", Static).update(out)
        try:                                                 # the box follows its text (a flag typed just now makes the command longer)
            self.fit_box()
            self.query_one(Menu)._refresh()
        except Exception:                                    # noqa: BLE001 - a screen that is closing
            pass

    def hints_text(self) -> str:
        if (self.size.width or 80) < 60:
            return "Enter next  e edit  Esc back  Tab command"
        return super().hints_text()

    def explain(self) -> Text:
        if (self.size.height or 30) < 20:
            return Text("")                                  # a small window: the list and the command matter more than the sentence under them
        m = self.menu()
        cur = m.current if m else None
        if cur is not None and cur.key not in (None, "__add__"):
            r = next((x for x in self.rows if x.name == cur.key), None)
            if r:
                tuned = tuned_sentence(r) if self.narrow() else ""     # (the Tuned column is left out of a narrow window)
                return Text(f"{r.name}: {r.description}" + tuned + (" This one is experimental: it may change answers or not work on every card." if r.experimental else ""), style=MUTE)
        return Text("Press e on a row to set your own value; Del puts the tuned value back. You can go on without touching anything: Enter.", style=MUTE)

    # ---- editing
    def action_edit(self) -> None:
        if self.mode:
            return
        cur = self.menu().current
        if cur is None:
            return
        if cur.key == "__add__":
            return self.action_add()
        self.mode = "edit"
        self.edit_name = cur.key
        r = next(x for x in self.rows if x.name == cur.key)
        self.open_field("edit", r.yours or (r.tuned if r.tuned != "-" else r.default) or "", f"value for {cur.key} (empty = on, 'off' removes it)")
        self.say(f"Type the value for {cur.key} and press Enter. Esc cancels.")

    def action_add(self) -> None:
        if self.mode:
            return
        self.mode = "add"
        self.open_field("edit", "", "name=value, e.g. prefill=65536 or STRATA_PIN_BY_NEED=1")
        self.say("Type the setting as name=value (the name of a newer engine's flag is fine).")

    def action_filter(self) -> None:
        if self.mode:
            return
        self.mode = "filter"
        self.open_field("edit", self.filter, "part of a name or description")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "edit":
            return
        text = event.value.strip()
        mode = self.mode
        self.mode = ""
        self.close_field("edit")
        if mode == "filter":
            self.filter = text
        elif mode == "edit":
            self.set_value(self.edit_name, text)
        elif mode == "add" and text:
            try:
                key, value = S.split_setting(text)
            except ValueError as e:
                self.say(str(e), True)
                return
            self.set_value(key, value)
        self.build()
        self.paint()

    def set_value(self, key: str, value: str) -> None:
        s = self.ctx.settings
        s.extra[key] = value
        warn = P.check_name(key, self.ctx.engine_flags)
        self.ctx.invalidate_rows()
        self.rows = self.ctx.advanced_rows()
        self.say(warn or f"{key} set to {value or 'on'}. It is used for this launch.", bool(warn))

    def action_reset(self) -> None:
        if self.mode:
            return
        cur = self.menu().current
        if cur is None or cur.key in (None, "__add__"):
            return
        if self.ctx.settings.extra.pop(cur.key, None) is not None:
            self.ctx.invalidate_rows()
            self.rows = self.ctx.advanced_rows()
            self.build()
            self.say(f"{cur.key} is back to the tuned value.")
            self.paint()

    def action_experimental(self) -> None:
        if self.mode:
            return
        self.only_experimental = not self.only_experimental
        self.build()
        self.paint()

    def action_copy(self) -> None:
        if self.mode:
            return
        text = self.ctx.command_text() + "\n" + self.ctx.flags_line()
        try:
            self.app.copy_to_clipboard(text)
            self.say("Copied the command to the clipboard (if your terminal allows it).")
        except Exception:                                    # noqa: BLE001
            self.say("Your terminal does not allow copying; select the lines above with the mouse.", True)

    def action_scroll_command(self) -> None:
        box = self.query_one("#cmdbox")
        if box.has_focus:
            self.set_focus(None)
            self.say("")
        else:
            box.focus()
            self.say("Up/Down/PageUp/PageDown scroll the command; Tab or Esc goes back to the list.")

    def action_go_next(self) -> None:
        if self.mode:
            return
        self.app.advance()           # type: ignore[attr-defined]

    def action_next(self) -> None:
        self.action_go_next()

    def action_back(self) -> None:
        if self.mode:
            self.mode = ""
            self.close_field("edit")
            self.say("")
            return
        if self.query_one("#cmdbox").has_focus:
            self.set_focus(None)
            self.say("")
            return
        super().action_back()

    def action_move(self, step: int) -> None:
        if self.mode:
            return
        super().action_move(step)


# ------------------------------------------------------------------------------------------------ 9 access

class AccessScreen(Base):
    step_id = "access"
    hints = "↑↓ choose   Enter next   p port   k new key   t key on/off   Esc back   ? help"
    help_text = ("Who can use the model. 'This computer only' is the safe default and never needs a key. Beyond it Strata makes an API key "
                 "for you (recommended); press t if you would rather have no key: it still starts, but anyone who can reach it can use it. "
                 "To see the key later: strata launch --show-key. To use a key of your own: strata launch --api-key YOUR-KEY (or --api-key-env NAME).")

    BINDINGS = Base.BINDINGS + [Binding("p", "port", show=False), Binding("k", "newkey", show=False), Binding("t", "keytoggle", show=False),
                                Binding("n", "go_next", show=False), Binding("u", "use_free_port", show=False)]

    free_port: int | None = None             # S12: the next free port, while the one in the settings is held by another program

    def body(self) -> ComposeResult:
        yield Static(Text("Who can use it?", style=f"bold {ACCENT}"), markup=False)
        yield Menu([("", 22), ("", 46), ("", 0)], id="menu")
        yield Static("", id="portkey", markup=False)
        yield Input(id="portfield", classes="field", disabled=True)

    def on_mount(self) -> None:
        # #5812: the key this launch had when the step opened, and the one moving the highlight made (None: none made)
        self._entry_key, self._cursor_key = self.ctx.settings.api_key, None
        self.build()
        self.check_port_busy()
        self.paint()

    def on_resize(self, _e) -> None:
        try:
            self.build()                                     # the columns follow the window's width
        except Exception:                                    # noqa: BLE001 - a screen that is closing
            return
        self.paint()

    def check_port_busy(self, announce: bool = True) -> None:
        """S12: the port in the settings (8080 unless the person or a preset chose another) is held by another program: say so here and offer the next free
        one (u), as the failure screen does, instead of letting the start fail at the end. A server of this very launch is not "another program" (the
        start asks what to do with that one itself)."""
        s = self.ctx.settings
        self.free_port = None
        host = "127.0.0.1"                                   # (a "one address only" launch is checked by the start itself: a screen never connects to a network address)
        try:
            port = int(s.port)
            if not S.port_in_use(port, host):
                return
            from strata_cli.launch.runner import Runner
            if any(int(r.get("port", -1)) == port for r in Runner(s, catalog=self.ctx.catalog).serving_launches()):
                return
            self.free_port = S.next_free_port(port, host)
        except Exception:                                    # noqa: BLE001 - a hint, never a reason to stop the screen
            return
        if announce and self.free_port:
            self.say(f"Port {s.port} is already in use by another program. Press u to use port {self.free_port} (the next free one), or p to choose another.", True)

    def action_use_free_port(self) -> None:
        if self.input_open() or not self.free_port:
            return
        self.ctx.settings.port = self.free_port
        self.free_port = None
        self.build()
        self.say(f"Port {self.ctx.settings.port}.")
        self.paint()

    def hints_text(self) -> str:
        text = super().hints_text()
        return text.replace("p port", f"p port   u use port {self.free_port}") if self.free_port else text

    def drop_cursor_key(self) -> bool:
        """Back on 'This computer only': a key that only moving the highlight made goes again, so a local launch keeps none (#5812).
        A key the person made (k, t), gave (--api-key) or had before this step stays."""
        s = self.ctx.settings
        made = getattr(self, "_cursor_key", None)
        if made is None or s.api_key != made:
            return False
        s.api_key, self._cursor_key = getattr(self, "_entry_key", None), None
        return True

    def build(self) -> None:
        s = self.ctx.settings
        opts = acc.options(s.port, listen=s.listen)
        items = []
        sel = 0
        for i, o in enumerate(opts):
            note = o.note
            style = WARN if o.warn else ""
            items.append(Item([o.label, o.url, Cell(note, style)], key=o.kind))
            if o.kind == s.access:
                sel = i
        self.opts = {o.kind: o for o in opts}
        m = self.query_one(Menu)
        m.columns = [("", 20), ("", 36), ("", 0)] if self.width() < 100 else [("", 22), ("", 46), ("", 0)]      # S2: at 80 columns the address column gives room to the note
        m.set_items(items, sel)

    def refresh_body(self) -> None:
        s = self.ctx.settings
        on = bool(s.api_key)
        beyond = s.access != "local"
        t = Text()
        t.append(f"\n   Port      [ {s.port} ]   press p to change\n")
        t.append(f"   API key   [ {'on' if on else 'off'} ]   {acc.mask_key(s.api_key)}  ", style=FG if on else WARN)
        # #5948: a key that is on is checked on this computer too, so a local launch with one does not say it is not needed
        t.append(("(recommended for anything beyond this computer; t = no key)" if on else "(t = make a key)") if beyond else
                 "(apps on this computer must send it too; t = no key)" if on else "(not needed on this computer)", style=MUTE)
        self.query_one("#portkey", Static).update(t)

    def explain(self) -> Text:
        cur = self.menu().current if self.menu() else None
        out = Text()
        if cur is not None:
            missing = acc.missing_helper(cur.key)
            if missing:
                out.append(missing + "\n", style=WARN)
            fw = acc.firewall_note(cur.key)
            if fw:
                out.append(fw + "\n", style=WARN)
            if cur.key == "cloudflare":
                out.append("A Cloudflare quick tunnel gives a temporary public address; anyone with it can try to use the server, so an API key keeps others out.", style=MUTE)
            elif cur.key in ("lan", "address"):
                out.append("Other computers on your network can use it at the address shown. An API key is recommended.", style=MUTE)
                if cur.key == "address" and len(acc.local_addresses()) > 1:
                    out.append("\nEnter = next address   n next", style=MUTE)        # #5917
            elif cur.key == "tailscale":
                out.append("Only devices signed in to your Tailscale network can reach it.", style=MUTE)
            else:
                out.append("Only programs on this computer can reach it, so no key is needed.", style=MUTE)
        return out

    def on_cursor(self) -> None:
        kind = self.menu().current.key
        s = self.ctx.settings
        s.access = kind
        if kind == "address" and s.listen not in acc.local_addresses():
            found = acc.local_addresses()
            s.listen = found[0] if found else None
        if kind == "local" and self.drop_cursor_key():
            self.say("")
        elif kind != "local" and not s.api_key and not s.no_key:
            s.api_key = acc.kept_key() or acc.new_key()
            self._cursor_key = s.api_key
            self.say("An API key was made for you (recommended beyond this computer). Press t if you want no key.")
        elif kind != "local" and s.no_key:
            self.say(S.NO_KEY_PUBLIC if kind == "cloudflare" else S.NO_KEY_NETWORK, True)

    def action_port(self) -> None:
        self.open_field("portfield", str(self.ctx.settings.port), "port number")
        self.say("Type a port from 1 to 65535 and press Enter.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "portfield":
            return
        n = S.check_port(event.value.strip())
        if n and n.level == "error":
            self.say(n.line(), True)
            return
        self.ctx.settings.port = int(event.value.strip())
        self.close_field("portfield")
        self.build()
        self.say(n.line() if n else f"Port {self.ctx.settings.port}.", bool(n))
        self.check_port_busy(announce=not n)               # (a port that is taken is said, with the next free one, in place of "Port N.")
        self.paint()

    def action_newkey(self) -> None:
        self.ctx.settings.api_key = acc.new_key()
        self.ctx.settings.no_key = False
        self.say("A new API key was made. Apps that used the old one need the new one (strata connect shows it).")
        self.paint()

    def action_keytoggle(self) -> None:
        s = self.ctx.settings
        if s.api_key:
            s.api_key = None
            s.no_key = s.access != "local"
            self.say("The API key is off." + ((" " + (S.NO_KEY_PUBLIC if s.access == "cloudflare" else S.NO_KEY_NETWORK)) if s.access != "local" else ""), s.access != "local")
        else:
            s.api_key, s.no_key = acc.new_key(), False
            self.say("The API key is on.")
        self.paint()

    def action_next(self) -> None:
        if self.input_open():
            return
        s = self.ctx.settings
        found = acc.local_addresses() if self.menu().current.key == "address" else []
        if len(found) > 1:                                   # #5917: Enter on "One address only" goes to the next address; n goes on
            s.listen = found[(found.index(s.listen) + 1) % len(found)] if s.listen in found else found[0]
            self.build()
            self.paint()
            return
        self.action_go_next()

    def action_go_next(self) -> None:
        if self.input_open():
            return
        self.commit()
        self.app.advance()           # type: ignore[attr-defined]

    def commit(self) -> None:
        s = self.ctx.settings
        s.access = self.menu().current.key
        if s.access == "address" and not s.listen:
            found = acc.local_addresses()
            s.listen = found[0] if found else None
        if s.access == "local":
            self.drop_cursor_key()
        elif not s.api_key and not s.no_key:
            s.api_key = acc.kept_key() or acc.new_key()

    def action_back(self) -> None:
        if self.input_open():
            self.close_field("portfield")
            return
        super().action_back()


# ------------------------------------------------------------------------------------------------ 10 save + launch

class SaveScreen(Base):
    step_id = "save"
    hints = "↑↓ choose   Enter do it   Tab rename   Esc back   ? help"
    BINDINGS = Base.BINDINGS + [Binding("tab", "rename", show=False)]
    help_text = "Save remembers all your choices under a name so next time you start with one key. Launch downloads what is missing, tunes if you asked, and starts serving."

    def body(self) -> ComposeResult:
        yield Static("", id="summary", markup=False)
        yield Input(id="name", classes="field open always")
        yield Menu(id="menu")

    def on_mount(self) -> None:
        s = self.ctx.settings
        f = self.query_one("#name", Input)
        f.value = s.name or self.ctx.default_name()
        items = [Item(["Save as a preset and launch"], key="save_launch"), Item(["Launch without saving"], key="launch"),
                 Item(["Save the preset and stop here"], key="save")]
        self.query_one(Menu).set_items(items, 0)
        plan = self.ctx.download_plan()
        if plan is not None and (plan.get("parts") or not plan.get("engine_usable", True)):     # S6: something would be downloaded: is there a way to?
            self.ctx.check_offline(lambda: self.app.call_from_thread(self.paint))
        self.ctx.check_tuned(lambda: self.app.call_from_thread(self.paint))          # 0217 item 3: is the small drafter in the feed? (else the 1.2 GB one is what comes)
        self.paint()

    def hints_text(self) -> str:
        cur = self.menu().current if self.menu() else None
        label = (cur.cells[0] if isinstance(cur.cells[0], str) else cur.cells[0].text) if cur is not None else "do it"
        if (self.size.width or 80) < 60:
            return f"Enter: {label}  Esc back"
        return f"{self.g.updown} choose  Enter: {label}  Tab rename  Esc back  ? help"

    def on_cursor(self) -> None:
        self.paint()

    def summary_lines(self) -> list[tuple[str, str]]:
        s = self.ctx.settings
        from strata_cli.launch import modelstate
        from strata_cli.launch.runner import fit_parallel_for
        is_hy4 = s.family == "hy4-flash"
        is_inkling = s.family == "inkling-2-mini"
        is_native_strata = is_hy4 or is_inkling
        r = next((x for x in self.ctx.rows if (x.family, x.model) == (s.family, s.model)), None)
        cards = None if is_native_strata else self.ctx.cards_line()
        lines = [("Model", (r.title if r else (s.model or "?")) + (" (your files)" if s.model_path else "")),
                 *([("Cards", cards)] if cards else []),                       # T8: Main / Helper / Off of each card
                 ("Context", S.context_label(s.context) + ("" if is_native_strata else f", memory {s.kv or 'int8'}")),
                 ("Pictures", "on" if s.vision not in ("off", "no") else "off"),
                 *([("Drafting", "off" if s.drafting == "off" else "DFly2 (built-in)")] if is_hy4 else
                   [("Drafting", "off" if s.drafting == "off" else "DSpark Block-8 (built-in)")] if is_inkling else
                   [("Drafting", dict(DraftingScreen.CHOICES).get(s.drafting or ("off" if modelstate.layer_off_without_choice(self.ctx.base_config()) else "auto"), "Auto (recommended)"))] if drafting_choice_available(self.ctx) else []),
                 *([("Parallel", f"{s.parallel} at a time" if s.parallel else f"fit to your card ({8 if (self.ctx.computer.main_vram_gb or 0) >= 22 else 4 if (self.ctx.computer.main_vram_gb or 0) >= 15 else 1})")] if is_inkling else
                   [] if is_hy4 else [
                     ("Parallel", f"{s.parallel} at a time" if s.parallel else f"fit to your card ({fit_parallel_for(self.ctx.catalog, s, self.ctx.computer.main_vram_gb)})"),
                     ("Tuning", {"quick": "quick (about 20 min)", "thorough": "thorough (up to 2 h)", "skip": "skipped"}[s.tune]),
                 ]),
                 ("Access", f"{S.ACCESS_WORDS[s.access]}, port {s.port}" + (", API key on" if s.api_key else ", no API key")),
                 # S9: with the person's own files, "Storage" was the folder ABOVE their model files and read like where the model is kept
                 *([("Your files", f"{s.model_path} (used where they are)")] if s.model_path else []),
                 # 0217 item 8: one name for the folder, with and without the person's own files; the note names the drafter this launch uses (item 10)
                 ("Strata data", (s.data_dir or str(self.ctx.default_folder())) + self.data_note())]
        if s.extra and not is_native_strata:
            lines.append(("Advanced", ", ".join(f"{k}={v}" for k, v in s.extra.items())))
        for n, text in enumerate(self.download_lines()):                       # T2: everything that will be downloaded, with its size
            lines.append(("Download" if n == 0 else "", text))
        return lines

    def data_note(self) -> str:
        """What goes into the Strata data folder when the model's files stay where they are: " (packs and the DFlash2 drafter go here)" / "(packs and the draft layer go here)" / "(packs go here)"."""
        if not self.ctx.settings.model_path:
            return ""
        term = self.ctx.drafter_term()
        return f" (packs and {term} go here)" if term else " (packs go here)"

    def download_lines(self, short: bool = False) -> list[str]:
        """One line per thing the launch will download, with its size ("the model, 66 GB"), or the one line "nothing: ..." when everything is here.
        The engine's size is not known before the update source is read, so it is listed without one. `short`: the words for a small window
        ("model 66 GB")."""
        plan = self.ctx.download_plan()
        if plan is None:
            return []
        if self.ctx.offline and (plan.get("parts") or not plan.get("engine_usable", True)):
            return self.offline_lines(plan, short)
        out = []
        if not plan.get("engine_usable", True):
            out.append("engine" if short else "the engine for your graphics card (its size is shown when it starts)")
        for p in plan.get("parts") or []:
            gb = float(p["gb"])
            size = f"{gb:.2f} GB" if gb < 1 else f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"
            if p.get("kind") == "dflash2" and gb >= 1 and not short:     # S5: the standard drafter is a 1.2 GB download that is packed on this computer; "disk_gb" is what stays (0.9 GB: the 0.52 + 0.34 GB files a CUDA / AMD / Mac engine reads; 0.5 GB: the one file the Intel engine reads)
                from strata_control import drafting
                size += f" (packed to {float(p.get('disk_gb') or (drafting.PACK_BYTES + drafting.CONVERTED_BYTES) / 1e9):.1f} GB on disk)"
            out.append(f"{p['label'].removeprefix('the ')} {size}" if short else f"{p['label']}, {size}")
        return out or ["nothing: everything is on this computer"]

    #: S6: what to do for each thing that cannot be downloaded while the computer is offline (plain words, one short line each)
    OFFLINE_WAYS = {"engine": "install it from a file: strata components install-file",
                    "weights": "put its files in a folder, then choose Use model files I already have",
                    "mtp": "copy its mtp folder from another Strata data folder into yours",
                    "mmproj": "put its mmproj file beside your model files, or choose No pictures",
                    "dflash2": "copy one of its .pack files into the dflash2 folder of your data folder"}

    def offline_way(self, part: dict | None) -> str:
        """How to put one missing thing in place without internet. The drafter names the file(s) exactly as `strata drafting fetch` and the launch line do
        (drafting_fetch.offline_files_phrase: one file is enough; 0217 item 9: this row said "both of its .pack files")."""
        kind = (part or {}).get("kind")
        if kind == "mmproj" and self.ctx.settings.family == "hy4-flash":
            return "put Hy4-Flash-Vision.strata in the hy4-flash models folder, or choose No pictures"
        if kind == "mmproj" and self.ctx.settings.family == "inkling-2-mini":
            return "put Inkling-2-Mini-Vision.strata in the inkling-2-mini models folder, or choose No pictures"
        if kind == "dflash2":
            from strata_control import drafting_fetch
            return (f"copy {drafting_fetch.offline_files_phrase(tuned=bool(part.get('tuned')), variant=part.get('variant'), engine_reads_q4=part.get('reads_q4', True))} "
                    "into the dflash2 folder of your data folder")
        return self.OFFLINE_WAYS.get(kind, "copy it in from another computer")

    def offline_lines(self, plan: dict, short: bool = False) -> list[str]:
        """The Download rows of a computer that cannot reach the download hosts: what is missing and how to put it in place, not "will download"."""
        parts = list(plan.get("parts") or [])
        missing = ([("engine", "the engine for your graphics card", None, None)] if not plan.get("engine_usable", True) else []) + \
                  [(p["kind"], p["label"], float(p["gb"]), p) for p in parts]
        if short:
            return ["offline, missing: " + ", ".join(label.removeprefix("the ") for _k, label, _g, _p in missing)]
        out = ["offline: nothing can be downloaded now. Missing:"]
        for kind, label, gb, part in missing:
            size = "" if gb is None else f", {gb:.2f} GB" if gb < 1 else f", {gb:.0f} GB" if gb >= 10 else f", {gb:.1f} GB"
            out.append(f"{label}{size}: {self.offline_way(part) if part else self.OFFLINE_WAYS.get(kind, 'copy it in from another computer')}")
        return out

    #: the same ways as OFFLINE_WAYS, short enough for a line of a small window (item 5: at 80x24 the whole how-to was one cut line)
    OFFLINE_SHORT = {"engine": "install it from a file: strata components install-file",
                     "weights": "put its files in a folder, then use them (Folder step)",
                     "mtp": "copy its mtp folder in from another computer",
                     "mmproj": "copy its mmproj file in, or turn pictures off",
                     "dflash2": "copy its .pack file in (strata drafting fetch)"}

    def compact_download_rows(self) -> list[str]:
        """The Download rows of the small-window summary: one line, or (offline) one line for each thing that is missing with its way in place."""
        plan = self.ctx.download_plan()
        if plan is None or not (self.ctx.offline and (plan.get("parts") or not plan.get("engine_usable", True))):
            return [self.compact_download()]
        missing = ([("engine", "engine", None)] if not plan.get("engine_usable", True) else []) + \
                  [(p["kind"], p["label"].removeprefix("the "), float(p["gb"])) for p in plan.get("parts") or []]
        out = ["Download offline: nothing can be fetched now. Missing:"]
        for kind, label, gb in missing:
            size = "" if gb is None else f" {gb:.2f} GB" if gb < 1 else f" {gb:.0f} GB" if gb >= 10 else f" {gb:.1f} GB"
            out.append(f"  {label}{size}: {self.OFFLINE_SHORT.get(kind, 'copy it in from another computer')}")
        return out

    def compact_download(self) -> str:
        """The Download row of the small-window summary: one line (cut at the end where the window is narrow, so it never wraps over the choices)."""
        text = "Download " + ", ".join(self.download_lines(short=True))
        room = max(20, self.width() - 6)
        return text if len(text) <= room else text[:room - 1] + ("…" if self.g.unicode else ".")

    def refresh_body(self) -> None:
        """The summary, the name and the three choices. When the body is too short for all of it (a small window), the summary shrinks to a
        few short lines and the name field to one row, so the three choices stay on screen and Enter never acts on something you cannot see."""
        s = self.ctx.settings
        lines = self.summary_lines()
        warnings = self.ctx.warnings()
        body_h = self.query_one("#body").size.height
        full_need = 2 + len(self.full_rows(lines, warnings)) + 3 + 3        # (the title, the name row, the rows with what wraps, the field, the three choices)
        compact = bool(body_h) and full_need > body_h
        name = self.query_one("#name", Input)
        name.set_class(compact, "slim")
        if compact:
            t = Text(no_wrap=True, overflow="ellipsis")
            tiny = body_h < 9
            ellipsis = "\u2026" if self.g.unicode else "..."
            if s.family == "hy4-flash":
                model = dict(lines)["Model"]
                rows = [f"{model} \u00b7 {S.context_label(s.context)}",
                        f"Pictures {'on' if s.vision not in ('off', 'no') else 'off'} \u00b7 Drafting {'off' if s.drafting == 'off' else 'DFly2'}",
                        *self.compact_download_rows(),
                        f"Access {s.access}, port {s.port}"]
            elif s.family == "inkling-2-mini":
                model = dict(lines)["Model"]
                rows = [f"{model} \u00b7 {S.context_label(s.context)}",
                        f"Pictures {'on' if s.vision not in ('off', 'no') else 'off'} \u00b7 Parallel {s.parallel or 'auto'} \u00b7 Drafting {'off' if s.drafting == 'off' else 'DSpark'}",
                        *self.compact_download_rows(),
                        f"Access {s.access}, port {s.port}"]
            else:
                model = dict(lines)["Model"].replace("Qwen3.8-Flash-Next ", "")
                rows = [f"{model} \u00b7 {S.context_label(s.context)} \u00b7 memory {s.kv or 'int8'}",
                        f"Pictures {'on' if s.vision != 'off' else 'off'} \u00b7 Parallel {s.parallel or 'auto'} \u00b7 Tuning {s.tune}",
                        *self.compact_download_rows(),                                             # T2: also in a small window (offline: each missing thing with its way, 0217 item 5)
                        f"Access {s.access}, port {s.port}"]
            # 0217 item 8: the two folders are rows of their own in a small window too (they were left out, so "Your files" / "Strata data" never showed at 80x24)
            room = max(16, self.width() - 2 - 13 - 2)
            if s.model_path:
                rows.append(f"{'Your files':<13}" + middle_shorten(str(s.model_path), room, ellipsis))
            rows.append(f"{'Strata data':<13}" + middle_shorten(str(s.data_dir or self.ctx.default_folder()), room, ellipsis))
            t.append("Save, then launch   name:\n" if tiny else "Save as preset, then launch   name:\n", style=f"bold {ACCENT}")
            limit = 2 if tiny else 3 if body_h < 12 else max(4, body_h - 6 - (1 if warnings else 0))
            for ln in rows[:limit]:
                t.append("  " + ln + "\n")
            if warnings and not tiny:
                t.append(f"  ! {warnings[0]}" + (f"  (+{len(warnings) - 1} more)" if len(warnings) > 1 else "") + "\n", style=WARN)
            t.rstrip()
        else:
            t = Text()
            t.append("Save as preset, then launch\n", style=f"bold {ACCENT}")
            t.append("  Name   ")
            t.append("(type below)\n", style=MUTE)
            for text, style in self.full_rows(lines, warnings):
                t.append(text + "\n", style=style)
            t.rstrip()
        self.query_one("#summary", Static).update(t)
        if getattr(self, "_compact", None) != compact:       # the body's height changes with it: look again once the layout has settled
            self._compact = compact
            self.call_after_refresh(self.paint)

    def full_rows(self, lines: list[tuple[str, str]], warnings: list[str]) -> list[tuple[str, str]]:
        """The rows of the full summary as (text, style), each one line of the window: a value that is longer than its room continues under itself, in the value's column
        (0217: the offline drafter row ended in "(340" and went on at the screen's edge)."""
        col = max(10, *(len(k) for k, _v in lines)) + 2        # 0217 item 8: the label column is as wide as the longest label ("Strata data" touched its path in a 10-wide one)
        room = max(30, self.width() - col - 4)
        out: list[tuple[str, str]] = []
        for k, v in lines:
            if k in ("Your files", "Strata data"):              # one row: a long folder wrapped and pushed the last choice off an 80x24 screen
                path, sep, note = v.partition(" (")
                v = middle_shorten(path, max(16, self.width() - col - 4 - (len(note) + 2 if sep else 0)), "…" if self.g.unicode else "...") + (sep + note if sep else "")
            parts = textwrap.wrap(v, room, break_long_words=True) or [""]
            out.append((f"  {k:<{col}}{parts[0]}", ""))
            out += [(" " * (col + 2) + p, "") for p in parts[1:]]
        for n in warnings:
            parts = textwrap.wrap(n, max(30, self.width() - 6)) or [""]
            out.append((f"  ! {parts[0]}", WARN))
            out += [(f"    {p}", WARN) for p in parts[1:]]
        return out

    def explain(self) -> Text:
        if (self.size.height or 30) < 20:
            return Text("")                                  # a small window: the choices matter more than the sentence
        plan = self.ctx.download_plan()
        if plan is not None:                                 # T2: the same list as under Download above
            if plan.get("engine_usable", True) and float(plan.get("gb") or 0) < 0.05:
                fetch = "Nothing needs to be downloaded."
            elif self.ctx.offline:                           # S6
                fetch = ("This computer is offline, so what is listed under Download cannot be fetched now. Put it in place first, as it says." if (self.size.height or 30) >= 30
                         else "This computer is offline: put what is listed under Download in place first.")       # (item 5: a small window needs the lines for the list)
            else:
                fetch = "Strata downloads what is listed under Download first."
        else:
            st = self.ctx.model_state()
            fetch = "Nothing needs to be downloaded." if st.remaining_gb < 0.05 else f"Strata downloads about {st.remaining_gb:.0f} GB first."
        tune = " It tunes first." if self.ctx.settings.tune != "skip" and self.ctx.settings.family not in ("hy4-flash", "inkling-2-mini") else ""
        return Text(f"Press Enter on the first choice to save and start. {fetch}{tune} The Strata monitor opens in this window when it is serving; "
                    "no web page opens by itself.", style=MUTE)

    def action_rename(self) -> None:
        self.query_one("#name", Input).focus()
        self.say("Type the preset's name and press Enter.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name":
            self.set_focus(None)
            self.say("")

    def action_next(self) -> None:
        name = self.query_one("#name", Input).value.strip() or self.ctx.default_name()
        choice = self.menu().current.key
        s = self.ctx.settings
        s.name = name
        if choice in ("save", "save_launch"):
            try:
                if choice == "save":
                    store.save(s, name)
                else:
                    store.check_name(name)                   # saved by the run once the restart question (if any) is answered "Restart"
            except store.PresetError as e:
                self.say(str(e), True)
                return
            if choice == "save_launch":
                self.ctx.pending_save = name
        if choice == "save":
            self.app.exit(None)
            self.app.stopped_after_save = True       # type: ignore[attr-defined]
            return
        self.app.run_launch()                        # type: ignore[attr-defined]

    def action_quit_app(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ progress

class RunScreen(Base):
    step_id = "run"
    title_line = "Launch"
    hints = "s stop   q quit"
    BINDINGS = [Binding("s", "stop_run", show=False), Binding("q", "stop_run", show=False), Binding("escape", "stop_run", show=False),
                Binding("r", "retry", show=False), Binding("u", "use_port", show=False)]

    STAGE_WORDS = (("engine", "Engine"), ("model", "Model"), ("config", "Settings"), ("tune", "Tuning"), ("start", "Starting"),
                   ("access", "Access"))

    def body(self) -> ComposeResult:
        yield Static("", id="stages", markup=False)
        yield Static("", id="tunepanel", markup=False)

    def __init__(self) -> None:
        super().__init__()
        self.state: dict[str, dict] = {k: {"text": "", "pct": None, "status": "wait"} for k, _ in self.STAGE_WORDS}
        self.failed: str | None = None
        self.free_port: int | None = None
        self.tune_info: dict = {}
        self.notes: list[str] = []              # sentences that stay under the rows for the whole launch (0217 item 7: a last run that did not end normally)

    def on_mount(self) -> None:
        self.paint()

    def head_text(self) -> Text:
        out = Text(no_wrap=True, overflow="crop")
        out.append("STRATA  Launch", style=f"bold {ACCENT}")
        s = self.ctx.settings
        out.append(f"   {s.name or ''}", style=MUTE)
        return out

    def handle_event(self, ev) -> None:
        if ev.stage == "done":
            return
        if ev.stage == "note":                  # not a stage: a note that stays on the screen
            if ev.text and ev.text not in self.notes:
                self.notes.append(ev.text)
                self.paint()
            return
        st = self.state.get(ev.stage)
        if st is None:
            return
        for k, _ in self.STAGE_WORDS:           # earlier stages are finished
            if k == ev.stage:
                break
            if self.state[k]["status"] in ("wait", "run"):
                self.state[k]["status"] = "done" if self.state[k]["text"] else "skip"
        if st["text"] and st["pct"] is not None and st["pct"] >= 95 and ev.pct is not None and ev.pct < 50 and ev.text != st["text"]:
            st.setdefault("past", []).append(st["text"])       # 0217 item 14: a new phase starts its own bar at 0 %; the finished one stays on the screen by name
        st["text"] = ev.text
        st["pct"] = ev.pct
        st["level"] = ev.level
        st["status"] = "done" if (ev.pct is not None and ev.pct >= 100) else "run"
        if ev.stage == "tune":
            self.tune_info = ev.data or {}
        self.paint()

    def on_failure(self, text: str) -> None:
        self.failed = text
        self.free_port = None
        busy = re.search(r"Port (\d+) is already in use", text)
        if busy:                                             # T11: the screen offers the next free port itself, no walk back through seven screens
            s = self.ctx.settings
            host = s.listen if s.access == "address" and s.listen else "127.0.0.1"
            self.free_port = S.next_free_port(int(busy.group(1)), host)
        self.paint()

    def action_use_port(self) -> None:
        """u: start again on the next free port (the choice is kept in this launch's settings, and in the preset when it is saved)."""
        if self.failed and self.free_port:
            self.ctx.settings.port = self.free_port
            self.free_port = None
            self.app.run_launch()          # type: ignore[attr-defined]

    def on_finished(self) -> None:
        for k, st in self.state.items():
            if st["status"] in ("wait", "run"):
                st["status"] = "done" if st["text"] else "skip"
        self.paint()

    def refresh_body(self) -> None:
        t = Text()
        g = self.g
        width = max(20, min(40, self.width() - 50))
        for k, label in self.STAGE_WORDS:
            st = self.state.get(k) if hasattr(self, "state") else None
            if st is None:
                continue
            mark = {"wait": "  ", "run": "> ", "done": g.tick + " ", "skip": "- "}[st["status"]]
            style = ACCENT if st["status"] in ("run", "done") else MUTE
            if st.get("level") == "warn" and st["status"] != "run":
                style = WARN
            for past in (st.get("past") or [])[-2:]:                       # the phases of this stage that are finished (item 14: the next one's bar starts at 0 % again)
                t.append(f"  {g.tick} {label:<10}", style=f"bold {ACCENT}")
                t.append(past + "\n", style=MUTE)
            t.append(f"  {mark}{label:<10}", style=f"bold {style}")
            if st["pct"] is not None and st["status"] == "run":
                # the bar gives way so the words after it (the phase: "Downloading the DFlash2 drafter ...") are whole in a small window (item 14: at 80 columns they wrapped under it)
                own = stage_bar_width(width, self.width(), len(st["text"] or ""))
                t.append(bar(st["pct"] / 100.0, own, g) + f" {st['pct']:.0f}%  ", style=ACCENT)
            t.append((st["text"] or "") + "\n", style=FG if st["status"] == "run" else MUTE)
        for note in getattr(self, "notes", None) or []:       # 0217 item 7: stays under the rows for the whole launch
            t.append("\n")
            for ln in textwrap.wrap(note, max(30, self.width() - 4)):
                t.append("  " + ln + "\n", style=WARN)
        if self.failed:
            t.append("\n  " + self.failed + "\n", style=BAD)
            if self.free_port:
                t.append(f"  Press u to use port {self.free_port} (the next free one) and start. ", style=ACCENT)
            t.append("  Press r to try again, Esc to go back to the screens, q to quit.\n", style=MUTE)
        try:
            self.query_one("#stages", Static).update(t)
            self.query_one("#tunepanel", Static).update(self.tune_panel())
        except Exception:                                    # noqa: BLE001
            pass

    def tune_panel(self) -> Text:
        t = Text()
        info = self.tune_info
        if not info or self.state["tune"]["status"] == "wait":
            return t
        t.append("\n  Tuning", style=f"bold {ACCENT}")
        rnd = info.get("round")
        if rnd and rnd[1]:
            t.append(f"   measurement {rnd[0]} of {rnd[1]}")
        if info.get("minutes_left"):
            t.append(f"   about {info['minutes_left']} min left")
        t.append("\n")
        for sp in (info.get("speeds") or [])[-6:]:
            label = {"base": "Default settings", "base_aa": "Default settings, measured again"}.get(sp.get("label") or sp.get("id"),
                                                                                                 sp.get("label") or sp.get("id") or "?")
            val = sp.get("decode_tok_s") or sp.get("value") or sp.get("tok_s")
            t.append(f"    {str(label)[:34]:<36}{val if val is not None else '-'}" + (" tokens/s" if val is not None else "") + "\n")
        t.append("  Close other heavy programs while it measures. You can stop it any time: nothing changes until it is done.\n", style=MUTE)
        return t

    def explain(self) -> Text:
        return Text("Strata is getting everything ready. This can take a while the first time (the model is large). "
                    "Stopping is safe: files already downloaded are kept and a download continues where it stopped.", style=MUTE)

    def action_stop_run(self) -> None:
        self.app.stop_run(self)            # type: ignore[attr-defined]

    def action_retry(self) -> None:
        if self.failed:
            self.app.run_launch()          # type: ignore[attr-defined]

    def action_move(self, step: int) -> None:
        pass

    def action_next(self) -> None:
        pass
