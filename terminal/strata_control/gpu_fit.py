"""Which graphics cards a model should use: ONE place for the app, `strata launch` and setup.

The rule (owner decision):
1. First try the model on ONE card (the largest) plus system RAM, the normal hybrid mode.
2. Only when that does not fit, or would be too tight, offer to add the second card as a layer split.
3. Let the person choose: every option comes with plain numbers and a recommended one is pre-selected (recommend, never force).

The memory arithmetic is native-fit.ts `fitOn` (app/host/src/lib/native-fit.ts) with the same constants (a test keeps them equal):
the experts are held first in the graphics memory beyond about DENSE_GB (the dense weights, buffers and a 32K cache), then in RAM beyond
about SYSTEM_ROOM_GB left for the system; whatever is left would be read from the SSD (`spill_gb`). A split adds the memory of the extra
cards but every extra card holds its own dense layers and prompt buffers, so each one costs SPLIT_CARD_OVERHEAD_GB of that (an estimate:
docs/MULTI_GPU.md says each card "holds the dense weights of its layers and its own prompt buffers").

Cards that count: not integrated (an iGPU shares the system RAM and is ignored), at least MIN_CARD_GB of graphics memory, flagged usable by
the caller's own driver checks (NVIDIA RTX 20 or newer, a supported AMD card, ...). A card needs SPLIT_MIN_VRAM_GB to share a model.
Pure functions, no I/O: setup.py, strata_cli/launch and tools/strata_mcp.py import it.
"""

from __future__ import annotations

from typing import Any

DENSE_GB = 5.0                 # = native-fit.ts DENSE_GB
SYSTEM_ROOM_GB = 10.0          # = native-fit.ts SYSTEM_ROOM_GB
DENSE_EXTRA_GB = 1.0           # = native-fit.ts DENSE_EXTRA_GB
SPLIT_CARD_OVERHEAD_GB = 2.0   # each extra card of a split: its own dense layers and prompt buffers
SPLIT_MIN_VRAM_GB = 8.0        # setup.py SPLIT_MIN_VRAM_GB: a card sharing a model needs this much
SPLIT_PROMPT_VRAM_GB = 12.0    # setup.py SPLIT_PROMPT_VRAM_GB (#448): a smaller second card caps the prompt chunk
MIN_CARD_GB = 3.5              # admits 4 GB GPUs reporting 3.8-3.9 GiB (GTX 1650, RTX 3050 4 GB, T500, Arc A310); still ignores 2 GB display cards
FITS_SPARE_RAM_GB = 4.0        # RAM left beyond the experts below which "fits" becomes "tight"
SSD_BOUNCE_RING_GB = 1.5       # bounded 1-2 GiB pinned DMA bounce/prefetch ring in RAM for GPU+SSD Only (0 OS swap)
SSD_MIN_RAM_LRU_GB = 1.0       # minimum bounded RAM LRU cache for CPU+SSD Only streaming
SPLIT_PROMPT_GAIN = "about 20% faster"   # setup.py's own words for prompts read on two cards (docs/MULTI_GPU.md)


def _round1(x: float) -> float:
    return round(float(x) * 10) / 10


# ---- one engine, one vendor -----------------------------------------------------------------------------------------------
# Every Strata engine package drives the cards of ONE vendor (CUDA: NVIDIA, HIP: AMD, SYCL: Intel, Metal: Apple). A model is never split
# across vendors, and two models at once, one per vendor, is not supported yet. So the cards offered for running and splitting are the
# ones the engine can drive; a better card of the other vendor is mentioned in one plain line (recommend, never force).
BACKEND_VENDOR = {"cuda": "nvidia", "hip": "amd", "sycl": "intel", "metal": "apple"}
VENDOR_NAME = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}
SECOND_MODEL_SENTENCE = ("Running two models at the same time, one on each make of graphics card, is not supported yet. "
                         "Strata runs one model at a time on the cards of one make.")


def vendor_of_backend(backend: str | None) -> str | None:
    return BACKEND_VENDOR.get(str(backend or "").lower())


def _vendor(card: dict) -> str | None:
    v = card.get("vendor")
    return str(v).lower() if v else None


def pick_vendor(cards: list[dict], backend: str | None = None) -> str | None:
    """The make of card the engine drives: the installed (or detected) engine's, else the only make among the cards that count; with more
    than one make, NVIDIA (the engine setup picks first), else the make of the card with the most memory. None: no card names a make."""
    named = vendor_of_backend(backend)
    if named:
        return named
    keep, _ = usable_cards(cards)
    makes = {_vendor(c) for c in keep if _vendor(c)}
    if len(makes) <= 1:
        return next(iter(makes), None)
    if "nvidia" in makes:
        return "nvidia"
    return _vendor(keep[0])


def for_vendor(cards: list[dict], vendor: str | None) -> tuple[list[dict], list[dict]]:
    """(the cards the engine of that make can drive, the cards of other makes). A card that names no make goes with the engine's."""
    if not vendor:
        return list(cards or []), []
    mine = [c for c in cards or [] if _vendor(c) in (None, vendor)]
    return mine, [c for c in cards or [] if _vendor(c) not in (None, vendor)]


def _short_name(c: dict, vendor: str) -> str:
    n = str(c.get("name") or "graphics card")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel "):
        if n.startswith(junk):
            n = n[len(junk):]
            break
    return f"{VENDOR_NAME.get(vendor, vendor.title())} {n}".strip()


# ---- UD-Q4_K_XL on a Strix Halo (0.2.10) ------------------------------------------------------------------------------------------------
# REMOVE this block, its three callers (setup.py "size:", strata_cli/launch/cmd.py apply_defaults, tests/terminal/test_strix_q4_note.py) when the
# stall is fixed in the engine (STRIX-2Q). The numbers are the 0.2.9 release notes'.
STRIX_Q4_MODEL = "UD-Q4_K_XL"
STRIX_Q4_NOTE = ("UD-Q4_K_XL on a Ryzen AI Max (Strix Halo) sometimes stopped on the first request after a start in our tests of older engines. "
                 "Engine 0.1.62 did not in our tests (an 8-hour chat run and several starts), but that is too few starts to be sure. If it "
                 "happens, stop the engine and start it again. UD-IQ4_XS is the choice if you want to be sure. Nothing is refused.")


def is_strix_halo_card(card: dict) -> bool:
    """A Ryzen AI Max (Strix Halo, gfx1151, Radeon 8060S / 8050S / 8040S) from the fields the hardware readers give."""
    if not isinstance(card, dict):
        return False
    text = " ".join(str(card.get(k) or "") for k in ("name", "arch", "gfx")).lower()
    return "gfx1151" in text or "strix halo" in text or any(x in text for x in ("8060s", "8050s", "8040s"))


def strix_q4_stall_note(model, cards) -> str | None:
    """The plain warning for UD-Q4_K_XL on a Strix Halo, else None. Warn only."""
    if str(model or "").lower() != STRIX_Q4_MODEL.lower():
        return None
    return STRIX_Q4_NOTE if any(is_strix_halo_card(c) for c in cards or []) else None


def other_vendor_note(cards: list[dict], vendor: str | None) -> str | None:
    """One plain line when the other make's best card is bigger than the best card the engine can use, else None:
    "This computer also has an AMD R9700 (32 GB). The AMD build of Strata can use it - it fits bigger models." """
    if not vendor:
        return None
    mine, others = for_vendor(usable_cards(cards)[0], vendor)
    others = [c for c in others if _vendor(c) and not _shares_memory(c)]
    if not others:
        return None
    intel = [c for c in others if _vendor(c) == "intel"]
    if intel:
        # Intel Arc cards beside another make: always said (the engine for one make runs at a time; the person chooses which), never refused
        total = sum(float(c.get("vram_gb") or 0) for c in intel)
        cards = (f"{len(intel)} Intel Arc cards ({total:.0f} GB together)" if len(intel) > 1
                 else f"an {_short_name(intel[0], 'intel')} ({total:.0f} GB)")
        return (f"This computer also has {cards}. Strata uses the {VENDOR_NAME.get(vendor, vendor)} card; to use the Intel card"
                f"{'s' if len(intel) > 1 else ''} instead, add --backend intel (it is remembered; --backend auto goes back).")
    if vendor == "intel":
        names = ", ".join(sorted({_short_name(c, _vendor(c)) for c in others}))
        return (f"This computer also has {names}. Strata uses the Intel cards; --backend auto goes back to the PC's own choice "
                "(NVIDIA first).")
    best = max(others, key=lambda c: float(c.get("vram_gb") or 0))
    mine_best = max((float(c.get("vram_gb") or 0) for c in mine), default=0.0)
    if float(best.get("vram_gb") or 0) <= mine_best:
        return None
    other = _vendor(best)
    return (f"This computer also has an {_short_name(best, other)} ({float(best.get('vram_gb') or 0):.0f} GB). "
            f"The {VENDOR_NAME.get(other, other.title())} build of Strata can use it — it fits bigger models.")


def memory_needed(variant: dict) -> float:
    """RAM + VRAM the model asks for (native-fit.ts memoryNeeded)."""
    if isinstance(variant.get("memory_gb"), (int, float)):
        return float(variant["memory_gb"])
    if variant.get("mem_total") and isinstance(variant.get("ram_needed_gb"), (int, float)):
        return float(variant["ram_needed_gb"])
    return _round1(float(variant.get("experts_gb") or variant.get("ram_needed_gb") or 0) + DENSE_EXTRA_GB)


def fit(memory_gb: float, ram_gb: float | None, vram_gb: float) -> dict[str, Any]:
    """native-fit.ts `fitOn` for a given memory need: where the weights go. Keys: vram_use_gb, ram_use_gb, spill_gb, spare_ram_gb, known."""
    vram = float(vram_gb or 0)
    if not ram_gb or ram_gb <= 0:
        return {"vram_use_gb": 0.0, "ram_use_gb": 0.0, "spill_gb": 0.0, "spare_ram_gb": 0.0, "known": False}
    experts = max(0.0, memory_gb - DENSE_EXTRA_GB)
    on_gpu = max(0.0, min(experts, vram - DENSE_GB))
    ram_room = max(0.0, float(ram_gb) - SYSTEM_ROOM_GB)
    spill = max(0.0, experts - on_gpu - ram_room)
    in_ram = max(0.0, experts - on_gpu - spill)
    return {"vram_use_gb": _round1(min(vram, DENSE_GB + on_gpu) if vram > 0 else 0.0),
            "ram_use_gb": _round1(in_ram + (DENSE_EXTRA_GB if vram <= 0 else 0)), "spill_gb": _round1(spill),
            "spare_ram_gb": _round1(max(0.0, ram_room - in_ram)), "known": True}


def verdict(f: dict) -> str:
    """fits / tight / spills: all in memory with room to spare / all in memory but little RAM left / part of it is read from the SSD."""
    if not f.get("known"):
        return "unknown"
    if f["spill_gb"] >= 1.0:
        return "spills"
    return "fits" if f["spare_ram_gb"] >= FITS_SPARE_RAM_GB or f["ram_use_gb"] <= 0 else "tight"


def _shares_memory(c: dict) -> bool:
    """An integrated chip or a unified-memory chip (Apple, Strix Halo): its memory is the system RAM."""
    return bool(c.get("integrated") or c.get("uma") or c.get("unified_memory") or c.get("unified_gb") or str(c.get("vendor")) == "apple")


def usable_cards(cards: list[dict]) -> tuple[list[dict], list[dict]]:
    """(cards that count, cards ignored with a plain reason in 'why_ignored'). Largest memory first, then the card number.

    A chip that shares the system memory is ignored when the computer has a card with memory of its own (it adds nothing to it); when it is
    the only device (a Mac, a Strix Halo PC) it is the one that counts. A card under MIN_CARD_GB cannot hold the dense weights."""
    flagged = [c for c in (cards or []) if c.get("usable") is not False]
    has_real = any(not _shares_memory(c) and float(c.get("vram_gb") or 0) >= MIN_CARD_GB for c in flagged)
    keep, ignored = [], []
    for c in cards or []:
        vram = float(c.get("vram_gb") or 0)
        why = None
        if c.get("usable") is False:
            why = c.get("problem") or "Strata cannot use this card"
        elif _shares_memory(c):
            if has_real:
                why = "it shares the system memory (an integrated graphics chip), so it adds nothing"
        elif vram < MIN_CARD_GB:
            why = f"only {vram:.0f} GB of graphics memory"
        if why:
            ignored.append({**c, "why_ignored": why})
        else:
            keep.append(c)
    keep.sort(key=lambda c: (-round(float(c.get("vram_gb") or c.get("unified_gb") or 0)), c.get("index", 0)))
    return keep, ignored


def _name(c: dict) -> str:
    n = str(c.get("name") or f"card {c.get('index')}")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "):
        n = n.replace(junk, "")
    return n + (f" (card {c.get('index')})" if c.get("_dup") else "")


def own_memory_gb(c: dict) -> float:
    """The graphics memory a card adds BESIDE the RAM.  A discrete card: its VRAM.  A chip that shares the system memory (Strix Halo, an APU): only
    its BIOS carve-out (`dedicated_gb`; 0 when the report does not give it) - its "shared" pool is the RAM itself, and counting it as VRAM as well as
    as RAM made a model look like it fit twice (setup.py low_ram_vram does the same)."""
    if c.get("uma") or c.get("integrated"):
        return float(c.get("dedicated_gb") or 0)
    return float(c.get("vram_gb") or 0)


def _option(key: str, cards: list[dict], memory_gb: float, ram_gb: float | None) -> dict[str, Any]:
    n = len(cards)
    total = sum(own_memory_gb(c) for c in cards)
    eff = total - SPLIT_CARD_OVERHEAD_GB * (n - 1)
    f = fit(memory_gb, ram_gb, eff)
    v = verdict(f)
    names = " + ".join(_name(c) for c in cards)
    label = f"{names} + RAM" if n == 1 else f"{names} + RAM (layer split)"
    have = (eff + max(0.0, float(ram_gb or 0) - SYSTEM_ROOM_GB))
    if v == "unknown":
        words = "memory of this computer unknown"
    elif v == "spills":
        words = f"does not fit in memory: about {f['spill_gb']:.0f} GB is read from the SSD while answering (slower)"
    elif v == "tight":
        words = f"fits, but tight: needs {memory_gb:.0f} GB of the {have:.0f} GB available"
    else:
        words = f"fits: needs {memory_gb:.0f} GB of the {have:.0f} GB available"
    notes = []
    if n >= 2:
        notes.append(f"prompts are read {SPLIT_PROMPT_GAIN}; the cards share the layers (docs/MULTI_GPU.md)")
        small = [c for c in cards[1:] if float(c.get("vram_gb") or 0) < SPLIT_PROMPT_VRAM_GB - 0.5]
        if small and float(cards[0].get("vram_gb") or 0) >= SPLIT_PROMPT_VRAM_GB - 0.5:
            notes.append(f"{_name(small[0])} is small: it can make prompts slower than the first card alone")
    return {"key": key, "cards": [c.get("index") for c in cards], "label": label, "verdict": v, "summary": f"{label}: {words}",
            "words": words, "vram_gb": _round1(total), "spill_gb": f["spill_gb"], "ram_use_gb": f["ram_use_gb"],
            "spare_ram_gb": f["spare_ram_gb"], "notes": notes, "recommended": False}


_RANK = {"fits": 0, "tight": 1, "unknown": 2, "spills": 3}


def options(memory_gb: float, ram_gb: float | None, cards: list[dict], vendor: str | None = None) -> dict[str, Any]:
    """The choices for this model on this computer: {"options": [...], "recommended": key, "ignored": [...], "cards": n counted}.

    Options (best known first): the largest card alone; with two or more cards that can share a model, the two largest as a layer split
    (and all of them when there are more than two); then every other card alone. Exactly one option is recommended:
    the single largest card when the model fits it (with RAM); otherwise the split if it fits better (a smaller verdict or less SSD use);
    otherwise the single largest card again (nothing better to offer)."""
    vendor = vendor or pick_vendor(cards)
    mine, others = for_vendor(cards, vendor)
    keep, ignored = usable_cards(mine)
    ignored = ignored + [{**c, "why_ignored": f"it needs the {VENDOR_NAME.get(_vendor(c), _vendor(c))} build of Strata, and this engine drives "
                                              f"{VENDOR_NAME.get(vendor, vendor)} cards"} for c in others]
    out: dict[str, Any] = {"options": [], "recommended": None, "ignored": ignored, "cards": len(keep), "vendor": vendor,
                           "other_vendor_note": other_vendor_note(cards, vendor)}
    if not keep:
        return out
    names = [str(c.get("name")) for c in keep]
    keep = [{**c, "_dup": names.count(str(c.get("name"))) > 1} for c in keep]       # two cards of one model: told apart by their number
    opts = [_option("single", [keep[0]], memory_gb, ram_gb)]
    share = [c for c in keep if float(c.get("vram_gb") or 0) >= SPLIT_MIN_VRAM_GB - 0.5]
    if len(share) >= 2:
        opts.append(_option("split", share[:2], memory_gb, ram_gb))
        if len(share) > 2:
            opts.append(_option("split-all", share, memory_gb, ram_gb))
    for c in keep[1:]:
        opts.append(_option(f"single-{c.get('index')}", [c], memory_gb, ram_gb))
    single = opts[0]
    best = single
    if single["verdict"] != "fits":
        for o in opts[1:]:
            if len(o["cards"]) > 1 and (_RANK[o["verdict"]], o["spill_gb"]) < (_RANK[best["verdict"]], best["spill_gb"]):
                best = o
    best["recommended"] = True
    out["options"], out["recommended"] = opts, best["key"]
    return out


def recommended_cards(memory_gb: float, ram_gb: float | None, cards: list[dict], vendor: str | None = None) -> list[int]:
    """The card numbers of the recommended option (empty: no usable card)."""
    res = options(memory_gb, ram_gb, cards, vendor)
    return next((o["cards"] for o in res["options"] if o["recommended"]), [])


def ssd_streaming_plan(memory_gb: float, ram_gb: float | None, vram_gb: float | None = 0.0) -> dict[str, Any]:
    """First-class `GPU + SSD` Only and `CPU + SSD` Only placement & CLI flag planner.

    Never blocks or overcommits RAM when `memory_gb` (e.g., 64-250 GB MoE) exceeds available RAM:
      - `gpu_ssd_only`: discrete GPU >= 4 GiB with small host RAM (`ram_room < 4 GiB` and spill > 0) ->
        uses a bounded 1.0-2.0 GiB pinned DMA bounce/prefetch ring in RAM (0 OS swap) + O_DIRECT/NO_BUFFERING
        NVMe -> GPU streaming (`--ssd-experts`, `--ram-experts-gb 0`).
      - `cpu_ssd_only`: no usable GPU (`vram < 4 GiB`) and `ram_room < experts` -> bounded RAM LRU cache
        + O_DIRECT/NO_BUFFERING + `proj_ready[3]` (`--cpu`, `--ssd-experts`, `--ram-experts-gb <budget>`).
      - `gpu_ram_ssd`: 3-tier hybrid (GPU + bounded RAM LRU + NVMe streaming).
      - `in_memory`: fits cleanly in VRAM + RAM without SSD streaming.
    """
    vram = float(vram_gb or 0.0)
    ram = float(ram_gb or 0.0)
    f = fit(memory_gb, ram if ram > 0 else None, vram)
    if not f["known"]:
        return {
            "mode": "unknown", "runnable": True, "label": "memory unknown",
            "vram_use_gb": 0.0, "ram_use_gb": 0.0, "ssd_gb": 0.0, "bounce_ring_gb": 0.0,
            "flags": [],
        }
    if f["spill_gb"] <= 0.05:
        return {
            "mode": "in_memory", "runnable": True,
            "label": "GPU + RAM" if vram >= MIN_CARD_GB else "CPU + RAM",
            "vram_use_gb": f["vram_use_gb"], "ram_use_gb": f["ram_use_gb"],
            "ssd_gb": 0.0, "bounce_ring_gb": 0.0, "flags": [],
        }
    ram_room = max(0.0, ram - SYSTEM_ROOM_GB)
    if vram >= MIN_CARD_GB:
        if ram_room < 4.0:
            bounce = _round1(min(2.0, max(1.0, ram * 0.2)))
            return {
                "mode": "gpu_ssd_only", "runnable": True,
                "label": f"GPU + SSD (low RAM, ~{bounce:g} GB DMA ring)",
                "vram_use_gb": f["vram_use_gb"], "ram_use_gb": bounce,
                "ssd_gb": _round1(max(0.0, memory_gb - f["vram_use_gb"])),
                "bounce_ring_gb": bounce,
                "flags": ["--ssd-experts", "--ram-experts-gb", "0"],
            }
        rb = int(max(1, ram_room))
        return {
            "mode": "gpu_ram_ssd", "runnable": True,
            "label": f"GPU + {rb} GB RAM + SSD",
            "vram_use_gb": f["vram_use_gb"], "ram_use_gb": float(rb),
            "ssd_gb": f["spill_gb"], "bounce_ring_gb": SSD_BOUNCE_RING_GB,
            "flags": ["--resident-budget-gib", str(rb)],
        }
    # No usable GPU: first-class CPU + SSD Only mode!
    lru_gb = _round1(max(SSD_MIN_RAM_LRU_GB, min(ram_room, max(SSD_MIN_RAM_LRU_GB, ram * 0.5))))
    return {
        "mode": "cpu_ssd_only", "runnable": True,
        "label": f"CPU + SSD (no GPU, {lru_gb:g} GB RAM LRU)",
        "vram_use_gb": 0.0, "ram_use_gb": lru_gb,
        "ssd_gb": _round1(max(0.0, memory_gb - lru_gb)),
        "bounce_ring_gb": 0.0,
        "flags": ["--cpu", "--ssd-experts", "--ram-experts-gb", f"{lru_gb:g}"],
    }


# ---- Qwen3.8 / Unsloth all-tier planner (discrete 4..32 GB + DGX Spark + Mac Metal + Strix Halo) -------------------------
# Covers every Unsloth Dynamic quant (UD-Q4_K_XL, Q8_0, UD-IQ4_XS, UD-IQ3_XXS, UD-Q6_K_XL, UD-Q5_K_XL, UD-Q3_K_XL,
# UD-Q2_K_XL, UD-IQ2_XXS, UD-IQ2_XS, UD-Q2_0, UD-IQ1_M, UD-IQ1_S) with heterogeneous expert sizing, #6354 auto int8 KV +
# elastic --kv-resident, #4870 coverage-based adaptive defaults (2/192/0.92 at 15%-65% coverage), and clean 4 GB / 6 GB
# discrete unlock (0-slot / dense-streaming / RAM+SSD cold tiers, never refusing or OOMing).
UNSLOTH_TOTAL_EXPERT_PAIRS = 48 * 512     # 24,576 (layer, expert) pairs in Qwen3.8-Flash-Next
UNSLOTH_MAX_EXPERT_MIB = 5.05             # worst-case Q8_0 outlier expert blob (used by uniform max_blob sizing)
UNSLOTH_KV_BYTES_PER_TOK = {"fp16": 2112, "int8": 1056, "k8v4": 816, "q4_0": 576}   # 13 attention layers (12 QSA + draft)
UNSLOTH_ADAPT_MIN_COV = 0.15              # #4870 lower coverage bound for fast adaptive tier
UNSLOTH_ADAPT_MAX_COV = 0.65              # #4870 upper coverage bound for fast adaptive tier

UNSLOTH_QUANTS: dict[str, dict[str, Any]] = {
    "UD-IQ1_S":      {"arena_gb":  32.5, "download_gb":  65.8, "dense_gb": 3.2, "avg_expert_mib": 1.26, "max_expert_mib": 5.05},
    "UD-IQ1_M":      {"arena_gb":  34.8, "download_gb":  68.0, "dense_gb": 3.2, "avg_expert_mib": 1.35, "max_expert_mib": 5.05},
    "UD-Q2_0":       {"arena_gb":  37.4, "download_gb":  70.5, "dense_gb": 3.2, "avg_expert_mib": 1.45, "max_expert_mib": 5.05},
    "UD-IQ2_XXS":    {"arena_gb":  39.8, "download_gb":  73.2, "dense_gb": 3.2, "avg_expert_mib": 1.54, "max_expert_mib": 5.05},
    "UD-IQ2_XS":     {"arena_gb":  42.3, "download_gb":  75.8, "dense_gb": 3.2, "avg_expert_mib": 1.64, "max_expert_mib": 5.05},
    "UD-Q2_K_XL":    {"arena_gb":  46.0, "download_gb":  79.6, "dense_gb": 3.3, "avg_expert_mib": 1.78, "max_expert_mib": 5.05},
    "UD-IQ3_XXS":    {"arena_gb":  50.3, "download_gb":  83.6, "dense_gb": 3.3, "avg_expert_mib": 1.95, "max_expert_mib": 5.05},
    "UD-Q3_K_XL":    {"arena_gb":  56.2, "download_gb":  89.5, "dense_gb": 3.4, "avg_expert_mib": 2.18, "max_expert_mib": 5.05},
    "UD-IQ4_XS":     {"arena_gb":  59.5, "download_gb":  93.7, "dense_gb": 3.4, "avg_expert_mib": 2.31, "max_expert_mib": 5.05},
    "UD-IQ4_XS-v31": {"arena_gb":  60.4, "download_gb":  93.7, "dense_gb": 3.4, "avg_expert_mib": 2.34, "max_expert_mib": 5.05},
    "UD-Q4_K_XL":    {"arena_gb":  77.0, "download_gb": 111.3, "dense_gb": 3.5, "avg_expert_mib": 2.99, "max_expert_mib": 5.05},
    "UD-Q5_K_XL":    {"arena_gb":  98.4, "download_gb": 158.3, "dense_gb": 3.6, "avg_expert_mib": 3.82, "max_expert_mib": 5.05},
    "UD-Q6_K_XL":    {"arena_gb": 109.2, "download_gb": 169.2, "dense_gb": 3.6, "avg_expert_mib": 4.24, "max_expert_mib": 5.05},
    "Q8_0":          {"arena_gb": 128.3, "download_gb": 188.2, "dense_gb": 3.8, "avg_expert_mib": 4.98, "max_expert_mib": 5.05},
}


def is_dgx_spark_card(card: dict | None) -> bool:
    """NVIDIA DGX Spark / GB10 (cc 12.1, 128 GB unified memory)."""
    if not isinstance(card, dict):
        return False
    text = " ".join(str(card.get(k) or "") for k in ("name", "arch", "cc")).lower()
    return "gb10" in text or "dgx spark" in text or "spark" in text or "12.1" in text or "sm_121" in text


def is_mac_metal_card(card: dict | None, backend: str = "") -> bool:
    """Apple Silicon unified memory GPU (Metal backend)."""
    if str(backend or "").lower() == "metal":
        return True
    if not isinstance(card, dict):
        return False
    text = " ".join(str(card.get(k) or "") for k in ("name", "vendor")).lower()
    return "apple" in text or "metal" in text


def unsloth_kv_default(vram_gb: float, ctx: int, uma: bool = False, explicit_kv: str | None = None) -> str:
    """#6354: default KV cache precision for Unsloth quants.

    - Discrete <= 16.5 GB: always int8 (prevents 21-23% 16K/32K prompt slowdown and saves 400+ expert slots on 12 GB).
    - Discrete <= 24.5 GB at >= 64K context: int8.
    - Any tier at > 8K context or UMA (DGX Spark, Mac Metal, Strix Halo): int8.
    - Explicit user choice (--kv) is always respected.
    """
    if explicit_kv:
        return str(explicit_kv)
    vram = float(vram_gb or 0.0)
    if uma:
        return "int8"
    if 0.0 < vram <= 16.5:
        return "int8"
    if 0.0 < vram <= 24.5 and int(ctx) >= 65536:
        return "int8"
    return "int8" if int(ctx) > 8192 else "fp16"


def unsloth_kv_resident_default(vram_gb: float, ram_gb: float, ctx: int, kv: str = "int8", uma: bool = False,
                                explicit_kv_resident: int | None = None) -> int | None:
    """#6354: elastic --kv-resident token window for Unsloth quants on discrete GPUs.

    Streams older KV pages to pinned host RAM so VRAM stays free for the expert cache:
      - <= 4.5 GB: 4096 when ctx > 4096
      - <= 6.5 GB: 8192 when ctx > 8192
      - <= 12.5 GB: 16384 when ctx > 16384
      - <= 16.5 GB: 16384 at 32K, 32768 at >= 64K
      - <= 24.5 GB: 32768 at >= 64K
      - > 24.5 GB: 32768 at >= 128K
    Disabled on UMA (shared memory gains nothing from copying RAM->RAM) and on < 16 GB RAM hosts.
    """
    if explicit_kv_resident is not None:
        return int(explicit_kv_resident) if int(explicit_kv_resident) > 0 else None
    vram = float(vram_gb or 0.0)
    ram = float(ram_gb or 0.0)
    c = int(ctx)
    if uma or vram <= 0.0 or ram < 16.0:
        return None
    if vram <= 4.5:
        return 4096 if c > 4096 else None
    if vram <= 6.5:
        return 8192 if c > 8192 else None
    if vram <= 12.5:
        return 16384 if c > 16384 else None
    if vram <= 16.5:
        if c >= 65536:
            return 32768
        if c > 16384:
            return 16384
        return None
    if vram <= 24.5:
        return 32768 if c >= 65536 else None
    return 32768 if c >= 131072 else None


def adaptive_cache_defaults(live_slots: int, total_pairs: int = UNSLOTH_TOTAL_EXPERT_PAIRS,
                            adapt_every: int | None = None, adapt_swaps: int | None = None,
                            adapt_decay: float | None = None) -> dict[str, Any]:
    """Issue #4870 coverage-based adaptive expert cache defaults (mirrors engine/src/program/generate.cpp).

    When the VRAM expert cache holds between 15% and 65% of total layer-experts (every 6-32 GB discrete tier running
    UD-Q4_K_XL, Q8_0, UD-IQ4_XS, UD-IQ3_XXS, or NVFP4), default to the faster 2 / 192 / 0.92 curve (+8% cache hit
    rate, +10% decode tok/s). Explicit user overrides are never overwritten.
    """
    slots = max(0, int(live_slots))
    total = max(1, int(total_pairs))
    cov = min(1.0, float(slots) / float(total))
    if slots <= 0 or slots >= total:
        def_every, def_swaps, def_decay, mode = 0, 0, 0.7, ("all_resident" if slots >= total else "zero_slot")
    elif UNSLOTH_ADAPT_MIN_COV <= cov <= UNSLOTH_ADAPT_MAX_COV:
        def_every, def_swaps, def_decay, mode = 2, 192, 0.92, "mid_coverage_fast"
    elif cov < UNSLOTH_ADAPT_MIN_COV:
        def_every, def_swaps, def_decay, mode = 4, (16 if cov < 0.125 else 32), 0.7, "low_vram_dampened"
    else:
        def_every, def_swaps, def_decay, mode = 4, 96, 0.7, "high_coverage_standard"
    return {
        "adapt_every": def_every if adapt_every is None else int(adapt_every),
        "adapt_swaps": def_swaps if adapt_swaps is None else int(adapt_swaps),
        "adapt_decay": def_decay if adapt_decay is None else float(adapt_decay),
        "coverage": round(cov, 4),
        "mode": mode,
    }


def _discrete_tier_name(vram_gb: float) -> str:
    for bound, label in ((4.5, "4gb"), (6.5, "6gb"), (8.5, "8gb"), (10.5, "10gb"),
                         (12.5, "12gb"), (16.5, "16gb"), (24.5, "24gb"), (32.5, "32gb")):
        if vram_gb <= bound:
            return f"discrete_{label}"
    return f"discrete_{int(round(vram_gb))}gb"


def unsloth_tier_plan(model: str, vram_gb: float, ram_gb: float, ctx: int = 32768, backend: str = "cuda",
                      uma: bool = False, gpu_name: str = "", arch: str = "",
                      kv_override: str | None = None, kv_resident_override: int | None = None) -> dict[str, Any]:
    """Comprehensive Qwen3.8 / Unsloth tier planner across every GPU/RAM tier.

    Handles:
      - Discrete GPUs: 4, 6, 8, 10, 12, 16, 24, 32 GB (NVIDIA CUDA, AMD HIP RDNA2/3/4, Intel Arc SYCL)
      - Unified Memory: NVIDIA DGX Spark (128 GB), Apple Mac Metal (16/24/32/48/64/128 GB), AMD Strix Halo (64/128 GB)
    Never refuses to install or OOMs on 4 GB / 6 GB GPUs (uses 0-slot / dense-streaming + RAM/SSD cold tiers).
    """
    spec = UNSLOTH_QUANTS.get(model)
    if spec is None:
        raise ValueError(f"Unknown Unsloth quant: {model!r}")
    vram = float(vram_gb or 0.0)
    ram = float(ram_gb or 0.0)
    c = int(ctx)
    be = str(backend or "cuda").lower()
    card = {"name": gpu_name, "arch": arch, "vendor": vendor_of_backend(be), "uma": uma, "vram_gb": vram}
    arena_gb = float(spec["arena_gb"])
    avg_mib = float(spec["avg_expert_mib"])
    max_mib = float(spec["max_expert_mib"])
    dense_gb = float(spec["dense_gb"])
    notes: list[str] = []
    flags: list[str] = []

    # ---- Unified memory tiers: DGX Spark, Mac Metal, Strix Halo ----
    if is_dgx_spark_card(card):
        kv = unsloth_kv_default(vram or ram, c, uma=True, explicit_kv=kv_override)
        pool_gb = max(vram, ram, 128.0)
        fits_all = (arena_gb + dense_gb + 12.0) <= pool_gb
        slots = UNSLOTH_TOTAL_EXPERT_PAIRS if fits_all else int(max(0.0, (pool_gb - dense_gb - 16.0) * 1024.0 / avg_mib))
        slots = min(slots, UNSLOTH_TOTAL_EXPERT_PAIRS)
        adapt = adaptive_cache_defaults(slots)
        flags = ["--mmap-experts", "--prefill", "auto:32768", "--mtp-q4", "all", "--mtp-hnorm", "stream", "--kv", kv]
        if not fits_all:
            budget = max(16, int(pool_gb - 20.0))
            flags += ["--resident-budget-gib", str(budget)]
        else:
            budget = 0
        return {
            "tier": "dgx_spark", "model": model, "backend": "cuda", "uma": True, "runnable": True,
            "kv": kv, "kv_resident": None, "mtp": True,
            "expert_cache_mode": "spark_mmap_resident" if fits_all else "spark_mmap_budget",
            "estimated_slots": slots, "uniform_max_blob_slots": min(UNSLOTH_TOTAL_EXPERT_PAIRS, int(slots * avg_mib / max_mib)),
            "coverage": adapt["coverage"], "adaptive": adapt,
            "ram_tier": "uma_unified" if fits_all else "uma_budget", "resident_budget_gib": budget,
            "flags": flags, "notes": notes,
        }

    if is_mac_metal_card(card, be):
        unified = max(ram, vram)
        kv = unsloth_kv_default(unified, c, uma=True, explicit_kv=kv_override)
        leave = 8.0 if unified >= 48.0 else 4.0
        usable_gb = min(unified * 0.75, max(4.0, unified - leave))
        kv_gb = c * 13 * UNSLOTH_KV_BYTES_PER_TOK.get(kv, 1056) / 1e9
        resident = (arena_gb + dense_gb + kv_gb + 2.0) <= usable_gb
        mtp = resident or unified >= 96.0
        mode = "metal_resident" if resident else "metal_stream"
        slots = UNSLOTH_TOTAL_EXPERT_PAIRS if resident else min(1500, int(max(0.0, (usable_gb - dense_gb - kv_gb) * 1024.0 / avg_mib)))
        adapt = adaptive_cache_defaults(slots)
        flags = ["--kv", kv]
        if not resident:
            flags.append("--mmap-experts")
            notes.append("STRATA_METAL_STREAM=1 (streams cold experts from SSD through unified memory)")
        if not mtp:
            flags += ["--mtp", "off"]
        return {
            "tier": f"mac_metal_{int(round(unified))}gb", "model": model, "backend": "metal", "uma": True, "runnable": True,
            "kv": kv, "kv_resident": None, "mtp": mtp,
            "expert_cache_mode": mode, "estimated_slots": slots,
            "uniform_max_blob_slots": min(UNSLOTH_TOTAL_EXPERT_PAIRS, int(slots * avg_mib / max_mib)),
            "coverage": adapt["coverage"], "adaptive": adapt,
            "ram_tier": "uma_unified" if resident else "metal_ssd_stream", "resident_budget_gib": 0,
            "flags": flags, "notes": notes,
        }

    if uma and is_strix_halo_card(card):
        total_ram = max(ram, vram)
        kv = unsloth_kv_default(total_ram, c, uma=True, explicit_kv=kv_override)
        pool_gb = total_ram * 0.78   # GTT + carve-out unified pool on Strix Halo (64 GB -> ~50 GB, 128 GB -> ~100 GB)
        fits_pool = (6.0 + arena_gb + 4.0) <= pool_gb
        slots = UNSLOTH_TOTAL_EXPERT_PAIRS if fits_pool else min(
            UNSLOTH_TOTAL_EXPERT_PAIRS, int(max(256.0, (pool_gb * 0.45 - dense_gb) * 1024.0 / avg_mib)))
        adapt = adaptive_cache_defaults(slots)
        flags = ["--kv", kv, "--prefill", "auto:16384"]
        budget = 0
        if not fits_pool:
            # Avoid 2x pin OOM on 64 GB Strix Halo or Q8_0 on 128 GB Strix Halo
            budget = max(12, int(total_ram * 0.40))
            flags += ["--expert-cache", str(slots), "--resident-budget-gib", str(budget)]
        if model.lower() == STRIX_Q4_MODEL.lower():
            notes.append(STRIX_Q4_NOTE)
        return {
            "tier": f"strix_halo_{int(round(total_ram))}gb", "model": model, "backend": "hip", "uma": True, "runnable": True,
            "kv": kv, "kv_resident": None, "mtp": True,
            "expert_cache_mode": "full_uma" if fits_pool else "uma_bounded_cache",
            "estimated_slots": slots, "uniform_max_blob_slots": min(UNSLOTH_TOTAL_EXPERT_PAIRS, int(slots * avg_mib / max_mib)),
            "coverage": adapt["coverage"], "adaptive": adapt,
            "ram_tier": "uma_unified" if fits_pool else "uma_budget_ssd", "resident_budget_gib": budget,
            "flags": flags, "notes": notes,
        }

    # ---- Discrete GPUs: 4, 6, 8, 10, 12, 16, 24, 32+ GB (CUDA / HIP / Intel Arc SYCL) ----
    kv = unsloth_kv_default(vram, c, uma=False, explicit_kv=kv_override)
    kv_res = unsloth_kv_resident_default(vram, ram, c, kv=kv, uma=False, explicit_kv_resident=kv_resident_override)
    vram_kv_tokens = min(c, kv_res) if kv_res is not None else c
    kv_vram_mib = vram_kv_tokens * 13 * UNSLOTH_KV_BYTES_PER_TOK.get(kv, 1056) / (1024.0 * 1024.0)
    # 4 GB and 6 GB GPUs: drop GPU draft layer (saves ~800 MiB) and use tight reserve so dense trunk + KV always fit
    mtp = vram > 6.5
    mtp_mib = 802.0 if mtp else 0.0
    runtime_mib = 0.0 if be == "sycl" else (256.0 if vram <= 6.5 else 384.0)
    reserve_mib = 200.0 if vram <= 4.5 else (300.0 if vram <= 8.5 else 512.0)
    # Dense trunk in VRAM (on 4 GB cards, output head moves to pinned host memory if needed, leaving ~2.8 GiB in VRAM)
    dense_vram_mib = (dense_gb * 1024.0 - 420.0) if vram <= 4.5 else (dense_gb * 1024.0)
    slack_mib = vram * 1024.0 - dense_vram_mib - mtp_mib - kv_vram_mib - runtime_mib - reserve_mib

    # 4 GB & 6 GB unlock: 0-slot / dense-streaming when VRAM slack < 160 MiB (never refuse, never OOM!)
    if vram <= 4.5 or slack_mib < 160.0:
        est_slots = 0
        uni_slots = 0
        cache_mode = "zero_slot" if vram <= 4.5 else "dense_streaming"
        notes.append("0-slot / dense-streaming mode: dense trunk + KV on GPU, routed experts stream from RAM/SSD (never refuses)")
    else:
        est_slots = min(UNSLOTH_TOTAL_EXPERT_PAIRS, int(slack_mib / avg_mib))
        uni_slots = min(UNSLOTH_TOTAL_EXPERT_PAIRS, int(slack_mib / max_mib))
        cache_mode = "heterogeneous_cache"

    adapt = adaptive_cache_defaults(est_slots)
    flags = ["--kv", kv]
    if kv_res is not None:
        flags += ["--kv-resident", str(kv_res)]
    if not mtp:
        flags += ["--mtp", "off"]
    if est_slots == 0:
        flags += ["--expert-cache", "0", "--no-prefill-borrow"]
    elif adapt["mode"] == "mid_coverage_fast":
        flags += ["--adapt-every", str(adapt["adapt_every"]), "--adapt-swaps", str(adapt["adapt_swaps"]),
                  "--adapt-decay", f"{adapt['adapt_decay']:g}"]

    # Cold tier (RAM + SSD): never overcommit host RAM
    kv_ram_gb = (c * 13 * UNSLOTH_KV_BYTES_PER_TOK.get(kv, 1056) / 1e9) if kv_res is not None else 0.0
    vram_experts_gb = est_slots * avg_mib / 1024.0
    cold_experts_gb = max(0.0, arena_gb - vram_experts_gb)
    ram_avail_gb = max(0.0, ram - 12.0 - kv_ram_gb)
    if cold_experts_gb <= ram_avail_gb:
        ram_tier = "all_ram"
        budget_gib = 0
    elif ram_avail_gb >= 8.0:
        ram_tier = "resident_budget"
        budget_gib = int(ram_avail_gb)
        flags += ["--resident-budget-gib", str(budget_gib)]
    else:
        ram_tier = "ssd_streaming"
        budget_gib = 0
        flags += ["--ssd-experts", "--ram-experts-gb", "0"]

    return {
        "tier": _discrete_tier_name(vram), "model": model, "backend": be, "uma": False, "runnable": True,
        "kv": kv, "kv_resident": kv_res, "mtp": mtp,
        "expert_cache_mode": cache_mode, "estimated_slots": est_slots, "uniform_max_blob_slots": uni_slots,
        "coverage": adapt["coverage"], "adaptive": adapt,
        "ram_tier": ram_tier, "resident_budget_gib": budget_gib,
        "flags": flags, "notes": notes,
    }
