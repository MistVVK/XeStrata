"""The model catalog, the computer's figures and the fit words for `strata launch`.

There is ONE catalog: setup.py's MODELS / FAMILIES / CONTEXTS, which the desktop app reads through tools/strata_mcp.py
(`Strata.model_table`, `Strata.hardware`, `Strata.recommend`; the app page's own fit arithmetic is
app/host/src/lib/native-fit.ts `fitOn`). This module loads that same file from the installer folder and ports `fitOn` line by
line; it holds no table of its own. Without the installer files (a Terminal bundle made before launch existed) it says so in
words instead of guessing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata_cli.launch import settings as S


def _merge_families(cat: Any, families: dict) -> dict:
    """Flash-Next families plus own-path families (Qwen3.6, Hy4-Flash) so CLI/plan/resolve handle every family setup knows (works with StubCatalog and monkeypatched Catalog)."""
    all_fams = dict(families)
    try:
        strata = getattr(cat, "strata", None)
        if strata is not None:
            all_fams.update(strata.own_families())
            s = strata.setup_module()
            if s is not None:
                for tbl in ("OWN_PATH_FAMILIES", "HY4_FAMILIES"):
                    for f, d in (getattr(s, tbl, None) or {}).items():
                        all_fams.setdefault(f, d)
    except Exception:                                  # noqa: BLE001
        pass
    return all_fams


class CatalogUnavailable(RuntimeError):
    """The installer files (setup.py, tools/strata_mcp.py) are not there; the message is for the user."""


# --- app/host/src/lib/native-fit.ts, ported (same constants, same rules) ------------------------------------------------
DENSE_GB = 5.0
SYSTEM_ROOM_GB = 10.0
DENSE_EXTRA_GB = 1.0


def _round1(n: float) -> float:
    return round(n * 10) / 10


def memory_needed(v: dict) -> float:
    if isinstance(v.get("memory_gb"), (int, float)):
        return float(v["memory_gb"])
    if v.get("mem_total") and isinstance(v.get("ram_needed_gb"), (int, float)):
        return float(v["ram_needed_gb"])
    return _round1(float(v.get("experts_gb") or v.get("ram_needed_gb") or 0) + DENSE_EXTRA_GB)


def fit_on(v: dict, ram_gb: float | None, vram_gb: float | None) -> dict[str, Any]:
    """native-fit.ts `fitOn`: where the weights go on this computer. Returns memoryGb, vramUseGb, ramUseGb, spillGb, known."""
    memory = memory_needed(v)
    vram = float(vram_gb or 0)
    if not ram_gb or ram_gb <= 0:
        return {"memoryGb": memory, "vramUseGb": 0.0, "ramUseGb": 0.0, "spillGb": 0.0, "known": False}
    experts = max(0.0, memory - DENSE_EXTRA_GB)
    on_gpu = max(0.0, min(experts, vram - DENSE_GB))
    ram_room = max(0.0, float(ram_gb) - SYSTEM_ROOM_GB)
    spill = max(0.0, experts - on_gpu - ram_room)
    in_ram = max(0.0, experts - on_gpu - spill)
    vram_use = min(vram, DENSE_GB + on_gpu) if vram > 0 else 0.0
    return {"memoryGb": memory, "vramUseGb": _round1(vram_use),
            "ramUseGb": _round1(in_ram + (DENSE_EXTRA_GB if vram <= 0 else 0)), "spillGb": _round1(spill), "known": True}


# --- the verdict words ---------------------------------------------------------------------------------------------------
FITS, SLOWER, TIGHT, NO = "fits", "slower", "tight", "no"
DISK = "disk"                    # fits the memory, but the download does not fit the free disk (T10)
DISK_ROOM_GB = 8.0               # working room beside the download (the Folder step asks for the same)


def verdict(on_this_pc: str | None) -> tuple[str, str]:
    """(category, plain words) from the catalog's own `on_this_pc` text (strata_control/fit_label.py through strata_mcp: "fits" or "may run slower").

    No card and a Mac with little memory get "may run slower" too (owner 11-08); an older Control's "does not fit..." is read the same way.  Never a block: the person can always try, and the screen says so."""
    text = (on_this_pc or "").strip()
    low = text.lower()
    if not text:
        return TIGHT, "not checked (this computer's memory could not be read)"
    if low.startswith("does not fit"):
        return SLOWER, "may run slower"
    if low == "may run slower":
        return SLOWER, "may run slower"
    if low == "fits":
        return FITS, "fits"
    if low.startswith("tight"):
        return TIGHT, "tight: close other programs"
    return TIGHT, text


@dataclass
class ModelRow:
    family: str
    family_title: str
    model: str
    id: str                       # the run config's tag (strata-<id>.json)
    about: str
    download_gb: float
    need_vram_gb: float
    need_ram_gb: float
    spill_gb: float
    verdict: str
    words: str
    experimental: bool
    images: bool
    installed: bool
    recommended: bool = False
    mem_total: bool = False
    memory_gb: float = 0.0        # the placement estimate (experts + dense), for the cards
    fit_gb: float = 0.0           # the files without the lookup table: the "Needs" column and the fit label
    disk_need_gb: float = 0.0     # verdict "disk": the disk the download and its working room need ...
    disk_free_gb: float | None = None   # ... and what is free in the data folder's drive
    mem_verdict: str = ""         # verdict "disk": the memory verdict it replaced (fits / tight / slower), "" when the disk mark is not set
    mem_words: str = ""

    @property
    def title(self) -> str:
        return f"{self.family_title} {self.model}"

    def without_disk(self) -> "ModelRow":
        """This row as it is when the download does fit after all (another drive, or the person's own files): the memory verdict alone."""
        if not self.mem_verdict:
            return self
        import dataclasses
        return dataclasses.replace(self, verdict=self.mem_verdict, words=self.mem_words, mem_verdict="", mem_words="", disk_need_gb=0.0, disk_free_gb=None)

    @property
    def sentence(self) -> str:
        """The verdict in words for a line under the list or a warning (the disk verdict says the two figures and the way out; a model that is
        also more than the memory holds says that too, T10b)."""
        if self.verdict == DISK and self.disk_free_gb is not None:
            text = (f"Not enough disk space: about {self.disk_need_gb:.0f} GB needed, {self.disk_free_gb:.0f} GB free. "
                    "Pick a bigger drive on the Folder step, or a smaller model")
            if self.mem_verdict == SLOWER:
                text += ". It may also run slower on this computer, because it is more than the memory holds"
            return text
        return self.words


@dataclass
class Computer:
    ram_gb: float | None = None
    gpus: list[dict] = field(default_factory=list)
    disk_free_gb: float | None = None
    os: str = ""
    avx2: bool | None = None
    raw: dict = field(default_factory=dict)
    engine_vendor: str | None = None           # the make of card the engine drives (nvidia, amd, intel, apple); None: the cards name none

    @property
    def usable_gpus(self) -> list[dict]:
        """The cards that count and the engine can drive (strata_control/gpu_fit.py: an integrated chip, a tiny card or a card of another
        make is left out), the largest first. One engine drives one make, so a model is never split across makes."""
        from strata_control import gpu_fit
        return gpu_fit.usable_cards(gpu_fit.for_vendor(self.gpus, self.engine_vendor)[0])[0]

    @property
    def ignored_gpus(self) -> list[dict]:
        from strata_control import gpu_fit
        mine, others = gpu_fit.for_vendor(self.gpus, self.engine_vendor)
        return gpu_fit.usable_cards(mine)[1] + [{**c, "why_ignored": "another make of card than this engine drives"} for c in others]

    @property
    def other_vendor_note(self) -> str | None:
        """One plain line when a card of another make is the bigger one (None otherwise)."""
        from strata_control import gpu_fit
        return gpu_fit.other_vendor_note(self.gpus, self.engine_vendor)

    @property
    def new_card_note(self) -> tuple[str, str] | None:
        """(what, what to do) when a usable card is an Intel Arc B-series (Battlemage): allowed, but new to Strata and untested by the maintainers."""
        import re
        from strata_control import gpu_fit
        for c in gpu_fit.usable_cards(gpu_fit.for_vendor(self.gpus, self.engine_vendor)[0])[0]:
            if c.get("vendor") == "intel" and re.search(r"Arc (?:Pro )?B\d", str(c.get("name", ""))):
                return (f"The {str(c['name']).replace('Intel ', '')} is a B-series Arc card. It is new to Strata and we have not tested it ourselves, so pictures or long answers may fail on it.",
                        "If something fails, start again with --vision off to turn pictures off, and tell us at https://github.com/Niko1221/Strata/issues (say it is a B-series card and attach the engine log).")
        return None

    @property
    def main_vram_gb(self) -> float:
        return max((float(g.get("vram_gb") or 0) for g in self.usable_gpus), default=0.0)

    def line(self, data_dir: str | None = None, width: int | None = None) -> str:
        """The computer in one line. `width`: the room it has; the graphics card's NAME gives way (ending in an ellipsis), never the memory and disk figures at the end
        (0217 item 11: "... · 28 GB" lost its "free" at 80 columns)."""
        g = self.usable_gpus
        if g and g[0].get("unified_gb"):                       # a Mac: one pool of memory, the chip is the graphics
            disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
            rest = " · ".join(x for x in (f"{self.ram_gb:.0f} GB unified memory" if self.ram_gb else "memory unknown", disk) if x)
            return " · ".join((self._fit_name(g[0].get("name", "Apple silicon"), "", rest, width), rest))
        name = g[0].get('name', 'GPU').replace('NVIDIA GeForce ', '').replace('AMD Radeon ', '') if g else "no usable graphics card found"
        tail = (f" {g[0].get('vram_gb', 0):.0f} GB" + (f" x{len(g)}" if len(g) > 1 else "")) if g else ""
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
        rest = " · ".join(x for x in (ram, disk) if x)
        return " · ".join(x for x in (self._fit_name(name, tail, rest, width) + tail, rest) if x)

    @staticmethod
    def _fit_name(name: str, tail: str, rest: str, width: int | None) -> str:
        if width is None or len(name) + len(tail) + 3 + len(rest) <= width:
            return name
        name = name.replace("(TM)", "").replace("(R)", "")
        if len(name) + len(tail) + 3 + len(rest) <= width:
            return name
        room = width - len(tail) - 3 - len(rest)
        return name if room < 6 else name[:room - 1].rstrip() + "…"


def installer_folder(explicit: str | os.PathLike | None = None) -> Path:
    """Where setup.py and tools/ are: this Terminal's own folder (the bundle, or a source checkout)."""
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[2]


def install_places(explicit: str | os.PathLike | None = None, src: str | os.PathLike | None = None) -> list:
    """Every folder with run configs, main one first: the SAME discovery `strata models list` and the own-files import use
    (strata_control/discovery.py). An explicit folder (the argument, or STRATA_INSTALL_DIR) is the only one."""
    from strata_control import discovery
    return discovery.places(here=installer_folder(src), explicit=explicit)


def install_root(explicit: str | os.PathLike | None = None, src: str | os.PathLike | None = None) -> Path:
    """Where run configs of installed models live: the main place found by the shared discovery, else the folder a new model is
    installed into (beside the components folder, what the app uses)."""
    from strata_control import discovery
    if explicit:
        return Path(explicit)
    env = os.environ.get("STRATA_INSTALL_DIR")
    if env:
        return Path(env)
    return discovery.find_install(here=installer_folder(src))[0]


# Intel Arc cards by PCI id: (name, memory in GB).  The memory figure is the fallback: a card's own size (arc_vram_gb: sysfs, else Level
# Zero sysman) is used when it can be read, and the kernel does not always tell an ordinary user. The discrete cards are listed (the same ids as sycl/setup_intel.py; tests/terminal/test_intel_cards_029.py keeps the two in step).  An id that is not
# here is NOT counted: it is skipped, because an Intel chip built into the processor shares this vendor id and must never be named as a card.
# (A770: the 8 GB model shares the id of the 16 GB one, so the smaller figure is used.)
INTEL_ARC = {"0x56a1": ("Arc A750", 8.0), "0x56a2": ("Arc A580", 8.0), "0x56a0": ("Arc A770", 8.0), "0x56a5": ("Arc A380", 6.0),
             "0x56a6": ("Arc A310", 4.0), "0x5690": ("Arc A770M", 16.0), "0xe20b": ("Arc B580", 12.0), "0xe20c": ("Arc B570", 10.0),
             "0xe223": ("Arc Pro B70", 32.0), "0xe222": ("Arc Pro B65", 32.0), "0xe221": ("Arc Pro B60", 24.0), "0xe211": ("Arc Pro B60", 24.0),
             "0xe212": ("Arc B50", 16.0)}


#: PCI id ranges of Intel's discrete cards, for a model that is not in the table above (a new Battlemage or Alchemist part): such a card is counted as an
#: Arc of the family's smallest memory, never named by model. The processors' own graphics (0x46xx, 0x7dxx, 0xa7xx, ...) are outside these ranges.
INTEL_ARC_FAMILIES = ((0x5690, 0x56c1, 8.0), (0xe200, 0xe2ff, 10.0))


def _arc_entry(did: str) -> tuple[str, float] | None:
    if did in INTEL_ARC:
        return INTEL_ARC[did]
    try:
        value = int(did, 16)
    except ValueError:
        return None
    for low, high, gb in INTEL_ARC_FAMILIES:
        if low <= value <= high:
            return "Arc", gb
    return None


#: Files under /sys/class/drm/cardN that hold an Intel card's own memory size in bytes, where the kernel lets an ordinary user read them
#: (xe: the tile's physical VRAM; i915: the local memory total). A card's PCI BAR is not used: it is a power of two (a 24 GB card has a
#: 32 GiB BAR) or a 256 MiB window.
ARC_VRAM_FILES = ("device/tile0/physical_vram_size_bytes", "lmem_total_bytes", "device/lmem_total_bytes")
_PCI_ADDRESS = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_SYSMAN_TOTAL: dict[str, int | None] = {}


def _sysman_total(pci: str) -> int | None:
    """The card's memory size in bytes from Level Zero sysman (serve/telemetry.py _Zes, the Monitor's reader), asked once per card."""
    if pci not in _SYSMAN_TOTAL:
        try:
            from serve.telemetry import _Zes
            _SYSMAN_TOTAL[pci] = _Zes(pci).read()[1]
        except Exception:                                    # noqa: BLE001 - no runtime, no card: the table's figure is used
            _SYSMAN_TOTAL[pci] = None
    return _SYSMAN_TOTAL[pci]


def arc_vram_gb(card: Path) -> float | None:
    """An Intel card's own memory in GiB (one decimal, as the NVIDIA and AMD readers count it): a sysfs size file, else sysman; None when
    neither says, and then the id table's figure stands."""
    size = None
    for rel in ARC_VRAM_FILES:
        try:
            size = int((card / rel).read_text().strip(), 0)
        except (OSError, ValueError):
            continue
        if size > 0:
            break
        size = None
    if size is None:
        try:
            pci = os.path.basename(os.path.realpath(card / "device")).lower()
        except OSError:
            pci = ""
        size = _sysman_total(pci) if _PCI_ADDRESS.fullmatch(pci) else None
    return round(size / 2**30, 1) if size and size > 0 else None


def intel_arc_cards(root: str = "/sys/class/drm") -> list[dict]:
    """The Intel Arc cards on this Linux computer, read from the kernel's device list (empty elsewhere). One unreadable entry or a card number
    above 9 never hides the others, and the cards of other makes beside them (an AMD APU's graphics) do not matter here (#4971, #5111)."""
    cards: list[dict] = []
    try:
        entries = sorted(Path(root).glob("card[0-9]*"), key=lambda p: (len(p.name), p.name))
    except OSError:
        return cards
    for card in entries:
        if "-" in card.name:                                 # a connector (card0-DP-1), not a card
            continue
        dev = card / "device"
        try:
            if (dev / "vendor").read_text().strip().lower() != "0x8086":
                continue
            did = (dev / "device").read_text().strip().lower()
        except OSError:
            continue
        entry = _arc_entry(did)
        if entry is None:
            continue
        name, gb = entry
        gb = arc_vram_gb(card) or gb                         # the card's own size; the table (an unlisted id: the family's smallest) when unread
        cards.append({"vendor": "intel", "name": "Intel " + name, "vram_gb": gb, "usable": True, "index": len(cards)})
    return cards


class Catalog:
    """setup.py's tables through strata_mcp's `Strata` object (the app's own catalog code)."""

    def __init__(self, src: str | os.PathLike | None = None, root: str | os.PathLike | None = None,
                 hardware: dict | None = None):
        self.src = installer_folder(src)
        self.places = install_places(root, self.src)
        self.root = Path(root) if root else (self.places[0].path if self.places else install_root(None, self.src))
        mcp = self.src / "tools" / "strata_mcp.py"
        if not mcp.is_file() or not (self.src / "setup.py").is_file():
            raise CatalogUnavailable(
                "This Terminal does not carry the model list. Install it again with the one-line installer, "
                "or run `strata launch` from a full Strata folder (the folder with setup.py).")
        self.root.mkdir(parents=True, exist_ok=True)
        name = "strata_mcp_for_launch"
        mod = sys.modules.get(name)
        if mod is None:
            if str(self.src) not in sys.path:
                sys.path.insert(0, str(self.src))
            spec = importlib.util.spec_from_file_location(name, mcp)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            real = sys.stdout
            sys.stdout = sys.stderr           # the module must not print into a screen
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.stdout = real
        self.mcp = mod
        self.strata = mod.Strata(self.root, self.src)
        self._mcp_data_dir = self.strata.data_dir
        self.strata.data_dir = self.data_dir          # one answer to "where is the data folder" for every caller, the app's included
        self._hardware_override = hardware

    # ---- the computer
    def hardware(self, fresh: bool = False) -> dict:
        if self._hardware_override is not None:
            hw = dict(self._hardware_override)
            hw.setdefault("disk_free_gb", self.strata.disk_free(self.strata.data_dir()))
            return hw
        hw = dict(self.strata.hardware(fresh=fresh))
        hw["gpus"] = self._admit_old_nvidia(list(hw.get("gpus") or []))
        if not str(hw.get("os", "")).startswith("Windows"):
            arc = intel_arc_cards()
            have = list(hw.get("gpus") or [])
            if arc and not any(g.get("vendor") == "intel" for g in have):
                # the shared hardware reader adds the Arc only on a PC with no usable NVIDIA / AMD card and the Intel engine
                # (strata_control/intel_app.py cards_to_add, the same reader); the rest get it here, also beside other makes (the plan
                # then says the Intel cards exist and how to use them: --backend intel); their numbers are the Intel engine's own
                # (0, 1, ...). Added once: an A750 with the Intel engine was listed twice ("Arc A750 8 GB x2", p1006 11-16).
                hw["gpus"] = have + arc
        return hw

    @staticmethod
    def _admit_old_nvidia(gpus: list[dict]) -> list[dict]:
        """Pascal / Volta cards (P100, P40, V100: compute capability 6.0-7.0) run on the experimental CUDA 12 engine. setup.py admits them when they
        are the only kind of graphics card in the computer (`old_gpus_opt_in`: "it is the only kind of NVIDIA GPU in this PC"); the shared hardware
        reader does not, so the screens said "no usable graphics card found" and left out the Cards step on a computer with several V100s.
        A newer NVIDIA card or a usable AMD card keeps the old cards out of the way, exactly as in setup."""
        def arch(g: dict) -> int:
            try:
                return int(g.get("arch"))
            except (TypeError, ValueError):
                return 0
        old = [g for g in gpus if g.get("vendor") == "nvidia" and 60 <= arch(g) <= 70 and not g.get("usable", True)]
        if not old:
            return gpus
        if any(g.get("usable", True) for g in gpus):
            return gpus
        return [({**g, "usable": True, "problem": None,
                  "note": "experimental: the CUDA 12 engine (docs/OLDER_GPUS.md)"} if g in old else g) for g in gpus]

    def computer(self, fresh: bool = False) -> Computer:
        hw = self.hardware(fresh)
        return Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []), disk_free_gb=hw.get("disk_free_gb"),
                        os=str(hw.get("os") or ""), avx2=(hw.get("cpu") or {}).get("avx2"), raw=hw,
                        engine_vendor=self.engine_vendor(hw))

    def installed_backend(self) -> str | None:
        """The installed engine's backend (cuda, hip, sycl, cpu, metal), None when there is no usable engine yet."""
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            if rep.get("usable"):
                return (rep.get("engine") or {}).get("backend")
        except Exception:                                    # noqa: BLE001
            pass
        return None

    def engine_vendor(self, hw: dict | None = None) -> str | None:
        """The make of card the engine drives: the installed engine's; with none installed, the engine setup would fetch (NVIDIA first when the
        computer has several makes)."""
        from strata_control import engine_component as ec, gpu_fit
        hw = hw if hw is not None else self.hardware()
        chosen = None
        try:
            backend, how = ec.detect_backend()
            if how in ("environment", "saved choice"):          # the person's own choice (--backend, STRATA_ENGINE_BACKEND) beats the installed engine
                chosen = backend
        except Exception:                                      # noqa: BLE001
            pass
        return gpu_fit.pick_vendor(list(hw.get("gpus") or []), chosen or self.installed_backend())

    # ---- the tables
    def tables(self):
        models, families, contexts, source = self.strata.tables()
        return models, families, list(contexts), source

    def contexts(self) -> list[int]:
        return [int(c) for c in self.tables()[2]]

    def contexts_for(self, family: str | None, model: str | None = None) -> list[int]:
        """The context sizes to offer for this model: Hy4-Flash's own list (8K..128K), else the shared list."""
        if family == "hy4-flash":
            from strata_control import hy4_family
            return list(hy4_family.CONTEXTS)
        return self.contexts()

    def recommendation(self, hw: dict | None = None) -> dict:
        return self.strata.recommend(hw or self.hardware())

    def rows(self, hw: dict | None = None, own: tuple[str, str] | None = None) -> list[ModelRow]:
        """The model list for this computer. `own`: (family, model) of the files in the person's own folder (apply_disk: no download for it)."""
        hw = hw or self.hardware()
        comp = Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []), engine_vendor=self.engine_vendor(hw))
        vram = comp.main_vram_gb
        rec = self.recommendation(hw)
        out: list[ModelRow] = []
        # #5869: every family of the shared table, Qwen3.6-35B-A3B (an engine path of its own) included, as the app lists them: its rows
        # are a group of their own (family_title), the Flash-Next rows are as before
        for fam in self.strata.model_table(hw):
            for size in fam["sizes"]:
                f = fit_on(size, hw.get("ram_gb"), vram)
                cat, words = verdict(size.get("on_this_pc"))
                out.append(ModelRow(
                    family=fam["family"], family_title=fam["title"], model=size["model"], id=size["id"],
                    about=size["about"], download_gb=float(size["download_gb"]), need_vram_gb=f["vramUseGb"],
                    need_ram_gb=f["ramUseGb"], spill_gb=f["spillGb"], verdict=cat, words=words,
                    experimental=bool(size.get("experimental") or fam.get("experimental")), images=bool(size.get("images")),
                    installed=self.installed_with_files(str(size["id"])), mem_total=bool(size.get("mem_total")),
                    memory_gb=float(size.get("memory_gb") or 0), fit_gb=float(size.get("fit_gb") or size.get("memory_gb") or 0),
                    recommended=(rec.get("family") == fam["family"] and rec.get("model") == size["model"])))
        self.apply_disk(out, hw, own)
        return out

    def apply_disk(self, rows: list[ModelRow], hw: dict, own: tuple[str, str] | None = None) -> None:
        """T10: "fits" said nothing about the disk. A model that is not on this computer yet and whose download (what is still missing, plus room
        to work) is more than the free space of the data folder's drive gets the verdict `disk`, whatever its memory verdict (T10b: "may run slower"
        and "tight" keep their words in `mem_verdict`, the sentence says both); only a model that is too big for the computer keeps just that.
        Never a block: the person can still try, as with every verdict.

        `own` = (family, model) of the files in the person's own folder (N1: "Use model files I already have", --model-path): that model needs no
        download, so it is never marked. The other rows count only what they would really download. Run again with another `own`, it first takes
        the earlier marks off, so the Model list can be refreshed when the person points at their own files."""
        for r in rows:
            if r.mem_verdict:                              # an earlier call's mark: back to the memory verdict, then decide again
                r.verdict, r.words, r.mem_verdict, r.mem_words = r.mem_verdict, r.mem_words, "", ""
                r.disk_need_gb, r.disk_free_gb = 0.0, None
        free = hw.get("disk_free_gb")
        if free is None:
            return
        try:
            base = Path(self.data_dir())
        except Exception:                                  # noqa: BLE001
            base = None
        for r in rows:
            if r.installed or r.verdict in (NO, DISK) or (own is not None and (r.family, r.model) == tuple(own)):
                continue
            got = 0.0
            if base is not None:
                try:
                    folder = self.model_folder(base, r.family, r.model)
                    if r.family == "hy4-flash":
                        from strata_control import hy4_family
                        got = sum(f.stat().st_size for f in folder.glob("*.strata")
                                  if f.name in hy4_family.FILES) / 1e9
                    else:
                        got = sum(f.stat().st_size for f in folder.glob("*.gguf")
                                  if not f.name.lower().startswith("mmproj")) / 1e9
                except OSError:
                    got = 0.0
            need = max(r.download_gb - got, 0.0) + DISK_ROOM_GB
            if need > float(free):
                r.mem_verdict, r.mem_words = r.verdict, r.words
                r.verdict = DISK
                r.words = "not enough disk space"                  # (a model that is also slow says so in the sentence under the list)
                r.disk_need_gb, r.disk_free_gb = need, float(free)

    def unavailable(self, hw: dict | None = None) -> list[tuple[str, str]]:
        """(title, the plain reason) of each model of the signed feed that this engine does not run: nothing for a feed without such a model. The model list
        shows them under the list (same sentence as setup and the app: strata_control/feed_models.py)."""
        try:
            return [(f["title"], f["unavailable_reason"]) for f in self.strata.model_table(hw or self.hardware()) if f.get("available") is False]
        except Exception:                                  # noqa: BLE001 - a line for information
            return []

    def _all_families(self, families: dict) -> dict:
        return _merge_families(self, families)

    def resolve(self, model: str, family: str | None = None) -> tuple[str, str]:
        """`iq3_xxs` or `IQ3_XXS` (and an optional family) -> (family, model); raises ValueError with the closest match."""
        models, families, _ctx, _src = self.tables()
        all_families = _merge_families(self, families)
        names = list(dict.fromkeys([*models, *self.strata.own_size_names()]))
        match = next((n for n in names if n.lower() == model.lower()), None)
        if match is None:
            near = S.closest(model, names)
            raise ValueError(f"There is no model called '{model}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Models: " + ", ".join(names) + ".")
        if family is not None and family not in all_families:
            near = S.closest(family, list(all_families))
            raise ValueError(f"There is no model family called '{family}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Families: " + ", ".join(all_families) + ".")
        offered = [f for f in all_families if match in self.strata.sizes_of(models, f)]
        if family is not None:
            if family not in offered:
                raise ValueError(f"{all_families[family]['title']} has no size {match}. It offers: "
                                 + ", ".join(self.strata.sizes_of(models, family)) + ".")
            return family, match
        if not offered:
            raise ValueError(f"{match} is not offered by any model family here.")
        return ("qwen" if "qwen" in offered else offered[0]), match

    def model_name_of(self, family: str | None, model: str | None) -> str | None:
        """The ``model_name`` setup writes into this model's run config (``qwen3.8-flash-next-iq3_s``), known before the model is installed: what the drafter table is keyed by."""
        if not family or not model:
            return None
        fam = _merge_families(self, self.tables()[1]).get(family) or {}
        return f"{fam.get('name', family)}-{model.lower()}"

    def tag_of(self, family: str, model: str) -> str:
        _m, families, _c, _s = self.tables()
        all_families = _merge_families(self, families)
        return (all_families[family].get("tag", "") + model).lower()

    def model_folder(self, base: Path, family: str, model: str) -> Path:
        """The folder a model's files are in under `base/models`. setup.py names it with the case kept (`models/IQ3_S`), the run-config tag is
        lower case (`iq3_s`): on a case-sensitive disk the two are different folders, so the one that exists wins, and a folder that differs
        only in case is found (#3996). When none exists yet it is setup's own name, because that is where the download will go."""
        return self.model_folder_trace(base, family, model)["folder"]

    def model_folder_trace(self, base: Path, family: str, model: str) -> dict:
        """model_folder's lookup with its steps (#4887, `strata models explain`): {"folder", "rule", "tried": [{"path", "rule", "exists"}]}. rule: "exact" (setup's
        spelling), "lower" (the run-config tag), "case" (a folder that differs only in letter case) or "fallback" (none is there: setup's spelling, where a download goes)."""
        _m, families, _c, _s = self.tables()
        all_families = _merge_families(self, families)
        models = Path(base) / "models"
        if all_families.get(family, {}).get("hy4"):
            hy4_dir = models / family
            return {"folder": hy4_dir, "rule": "exact" if hy4_dir.is_dir() else "fallback", "tried": [{"path": hy4_dir, "rule": "exact", "exists": hy4_dir.is_dir()}]}
        own = models / (all_families[family].get("tag", "") + model)
        lower = models / self.tag_of(family, model)
        tried = []
        for candidate, rule in ((own, "exact"), (lower, "lower")):
            if any(t["path"] == candidate for t in tried):
                continue                                     # (a tag that is lower case already: one folder)
            try:
                here = candidate.is_dir()
            except OSError:                                  # a parent this account may not read (PermissionError): not there, as far as it can tell
                here = False
            tried.append({"path": candidate, "rule": rule, "exists": here})
            if here:
                return {"folder": candidate, "rule": rule, "tried": tried}
        try:
            for entry in models.iterdir():
                if entry.name.lower() == lower.name and entry.is_dir():
                    tried.append({"path": entry, "rule": "case", "exists": True})
                    return {"folder": entry, "rule": "case", "tried": tried}
        except OSError:
            pass
        return {"folder": own, "rule": "fallback", "tried": tried}

    def kv_gb(self, tokens: int, kv: str = "int8", family: str | None = None) -> float:
        """The conversation memory for this many tokens, in GB (setup.py's UD_FIT_KV_BYTES: 13 attention layers; Hy4-Flash: hy4_fit)."""
        if family == "hy4-flash":
            from strata_control import hy4_fit
            return tokens * hy4_fit.KV_GB_PER_TOKEN
        mod = self.strata.setup_module()
        table = getattr(mod, "UD_FIT_KV_BYTES", None) or {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}
        return tokens * 13 * table.get(kv, 1056) / 1e9

    def max_context(self, family: str, model: str, hw: dict | None = None) -> int | None:
        """The context a model starts with: 128K on every card, or the largest size the model offers under it (strata_control/default_context.py,
        the one rule of the app, the Terminal and setup). A Mac keeps its unified-memory rule."""
        from strata_control import default_context
        hw = hw or self.hardware()
        if hw.get("mac") and hw.get("ram_gb") and family != "hy4-flash":
            mac = self.strata.mac()                           # a Mac: the context by unified memory (strata_control/mac_fit.py)
            if mac is not None:
                return int(mac.recommend_context(float(hw["ram_gb"])))
        return default_context.pick(self.contexts_for(family, model))

    # ---- pictures
    def engine_backend(self) -> str | None:
        """The installed engine's backend (cuda, hip, sycl, cpu, metal), else the one this computer's graphics card needs, else None."""
        installed = self.installed_backend()
        if installed:
            return installed
        for g in self.computer().usable_gpus:
            v = str(g.get("vendor") or "").lower()
            if g.get("unified_gb") or v == "apple":
                return "metal"
            return {"nvidia": "cuda", "amd": "hip", "intel": "sycl"}.get(v)
        return None

    def reader_state(self, family: str | None = None) -> str:
        """"present" when the installed engine carries the picture reader (strata-vision; built into strata-hy4 for Hy4-Flash), "missing" when an engine is installed without it,
        "unknown" before any engine is installed (the first launch fetches one that has it)."""
        if family == "hy4-flash":
            return "present"
        try:
            from strata_control import engine_component as ec
            rep = ec.status()
            exe = (rep.get("engine") or {}).get("exe") if rep.get("usable") else None
        except Exception:                                    # noqa: BLE001
            exe = None
        if not exe:
            return "unknown"
        from strata_control import vision_reader as VR
        found = VR.find(Path(str(exe)).parent, os.name == "nt")
        return "present" if (found["cpu"] or found["gpu"]) else "missing"

    def picture_default(self, images: bool, family: str | None = None) -> str:
        """The picture setting nobody chose: off for a model that cannot read pictures or an engine without the reader; else on, with the
        reader the one rule of strata_control/vision_reader.py gives: the CPU reader under 24 GB of graphics memory, the GPU reader from 24 GB,
        on every make of card (owner decision 2026-10-18); on a Mac the Metal reader (unified memory). Without a usable graphics card there is nothing to read on: the CPU reader."""
        if family == "hy4-flash":
            return "on" if images else "off"
        from strata_control import vision_reader as VR
        if not images or self.reader_state() == "missing" or self.pictures_not_on_card():
            return "off"
        if self.is_mac():
            return VR.default_mode(None, unified=True)           # a Mac (unified memory): the Metal reader, nothing to keep free for a card
        comp = self.computer()
        vram = comp.main_vram_gb
        return VR.default_mode(vram, "intel" if comp.engine_vendor == "intel" else "") if vram > 0 else "cpu"      # no usable card: nothing to read on, the CPU reader

    def pictures_not_on_card(self, family: str | None = None) -> bool:
        """T14: an Intel Arc card on Linux whose engine does not carry the picture reader (the engine route that builds the model's pack from the files itself
        fetches none) starts the engine text-only, so offering "Yes" would say "Pictures on" and then answer "I don't see a picture" (USERTEST-LIN, A750).
        An engine that carries the reader (reader_state "present", the B-series decision of 2026-10-22; or strata-hy4's native SYCL vision) keeps pictures on."""
        if family == "hy4-flash" or not sys.platform.startswith("linux"):
            return False
        try:
            return self.engine_backend() == "sycl" and self.reader_state() != "present"
        except Exception:                                    # noqa: BLE001
            return False

    def is_mac(self) -> bool:
        """An Apple Silicon Mac: the chip's memory is the system's (the Metal engine; strata_control/vision_reader.py)."""
        return self.engine_backend() == "metal"

    # ---- installed models
    def installed_configs(self) -> list[dict]:
        out, seen = [], set()
        for path in self.strata.configs():
            seen.add(path.name)
            d = self.strata.describe_config(path)
            d["path"] = str(path)
            out.append(d)
        for place in self.places:
            if place.path == self.root:
                continue
            from strata_control import discovery
            for path in discovery.config_files(place.path):
                if path.name in seen:
                    continue
                seen.add(path.name)
                d = self.strata.describe_config(path)
                d["path"] = str(path)
                out.append(d)
        return out

    def installed_config(self, tag: str) -> Path | None:
        name = f"strata-{tag.lower()}.json"
        for folder in [self.root, *[p.path for p in self.places if p.path != self.root]]:
            p = folder / name
            if p.is_file():
                return p
        return None

    def installed_with_files(self, tag: str) -> bool:
        """Is this model on this computer, as the list's "(installed)" tag means it (#6240)?  A run config that survived its files (deleted, or the
        data folder moved) is not: its weights or its prepared pack are named and not there.  A config that names no weights (nothing to check), or
        that cannot be read, still counts, as before; a missing draft layer or picture reader does not matter (they are fetched on their own)."""
        path = self.installed_config(tag)
        if path is None:
            return False
        try:
            from strata_control import discovery
            cfg = json.loads(path.read_text(encoding="utf-8"))
            gone = {m.kind for m in discovery.footprint(cfg, path).missing()}
        except (OSError, ValueError, AttributeError, TypeError):
            return True
        return not (gone & {"weights", "pack"})

    def data_dir_choice(self) -> tuple[Path, str]:
        """(the data folder, where it came from): the person's own setting, the desktop app's folder, a Documents/Strata that holds models,
        else the default beside the install. The same rule for every command."""
        from strata_control import discovery
        return discovery.data_dir(Path(self._mcp_data_dir()))

    def data_dir(self) -> Path:
        return self.data_dir_choice()[0]

    def data_dir_for(self, family: str | None, model: str | None) -> Path:
        """Where this model's files are: the folder its installed run config names, else the data folder."""
        if family and model:
            path = self.installed_config(self.tag_of(family, model))
            if path is not None:
                try:
                    from strata_control import discovery
                    found = discovery.footprint(json.loads(path.read_text(encoding="utf-8-sig")), path).data_dir
                    if found is not None:
                        return found
                except (OSError, ValueError):
                    pass
        return self.data_dir()

    def disk_free(self, path: str | os.PathLike) -> float | None:
        return self.strata.disk_free(Path(path))
