"""Will the default start fit the card? If it is short, move work off the GPU until it does: slower, never a smaller context, never fewer requests.

Owner rule (2026-11-02): a person's context length and their number of requests at once are never changed by a fit; people set their compaction around
the context, and a start that spills to the SSD is slow where a smaller context breaks their setup.  A start that is short of VRAM still starts, by moving
work off the card in the order of least speed cost: a smaller --kv-resident window (more of the KV streams from RAM), fewer expert-cache slots (the engine's
auto sizing gives the cache what is left, so this needs no flag), a smaller prompt chunk (--prefill), a smaller reserve (--vram-reserve-mib, within the
engine's floor), and what only the engine can do (it sizes around what it cannot spill).  Which knobs a backend takes is KNOBS below, from what was measured
on it.  One plain note says it: short of GPU memory, more is read from RAM and the SSD, expect it to be slower.  Pure functions, no I/O: sycl/setup_intel.py
(setup), strata_cli/launch (strata launch) and serve/server.py (every engine start, fit_start) import it.

The arithmetic (MiB), measured on the Arc A750 (8 GB, i915) with the SYCL engine of 0.2.2, Gyro-M and Gyro-S, 12 starts, and on the CUDA engine (RTX PRO 5000
held to 7942 / 8966 / 9990 / 12000 MiB by a memory hog, DF2-MEM RESULTS-DF2MEM.md):

    slack = card - gap - base - session(ctx) - head - reserve - prompt buffers(chunk)      # what is left for the expert cache
    fits  = slack >= what the cache needs (the engine's one slot, or the working cache the CUDA engine asks for) + MARGIN

* gap: the part of the card the driver does not give (Arc: 471 MiB, the engine sees 7.54 GiB of 8; CUDA / HIP: the runtime's context, 250).
* base: dense weights, the draft layer, the n-gram head's buffers and the engine's own allocations that do not depend on the context.
  Gyro-M 5945 (measured "holds" 6.02 GiB at 8K minus the 234 MiB session), Gyro-S 4720; any other model: 2448 (1.4 GiB dense + 0.8 GiB
  draft layer + slack), a figure that only a card under about 5 GB can get past, so a default that fits today is never changed.
* session: 121 MiB + the KV cache. int8 KV measured 234 / 347 / 573 MiB at 8192 / 16384 / 32768 tokens (13.8 KiB per token).
* head: the draft head (English subset) 110 MiB.
* prompt buffers: 170 + 0.66 * chunk MiB, measured at chunks 256 .. 2048 (338 / 508 / 850 / 1525 MiB), independent of the context.
  `--prefill auto` resolved to 2048 on that card.
* slot: 1.243 MiB (0.50 GiB / 412 slots).
* request slots ("Concurrent requests" 2 or more, the engine's --batch N): every slot holds a session of its own, the size of the main one, on every backend
  (the engine's "--batch 2: 2 slot sessions (0.23 GiB each)" is session(8192) = 234 MiB), taken out of the expert cache's room.  With KV streaming
  (--kv-resident N) a session's VRAM holds max(N, 20480) cells of KV, not the whole context (qsa_state_bytes / kv_plan), so a slot costs that much, not 128K's.
  Only these starts and the CUDA single request also use a measured Qwen3.8-Flash-Next figure (BASE_SLOTTED_MIB, per vendor, per drafter): where none is
  measured, nothing is changed and the engine decides as it always did.

Why these numbers and not the engine's own: the engine only finds out after it has loaded 30 GB of weights; setup has to choose before.
It is an estimate; the engine still says what is short if it is off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CHUNK_STEPS = (1024, 512, 256)                   # a smaller prompt chunk, tried after the default one (1024 is the value measured end to end)
RESERVE_FLOOR_MIB = 100                 # the lowest reserve the engine accepts without complaint (docs/INTEL.md); never raised
AUTO_CHUNK = 2048                       # what --prefill auto came to on the 8 GB card
MIN_SLOTS = 128                         # the engine wants 128 cache slots to lend the prompt path; below that it is slow
SLOT_MIB = 1.243
MARGIN_MIB = 64                         # the estimate's error (the measurements agree to +-25 MiB)
HEAD_MIB = 110
SESSION_FIXED_MIB = 121
KV_BYTES = {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}   # per token per attention layer (13) = setup.UD_FIT_KV_BYTES
KV_MEASURED_FACTOR = 1.054              # the engine's session is 5.4% over the bare cache (scales)
PROMPT_BASE_MIB = 170
PROMPT_PER_TOKEN_MIB = 0.66
GAP_MIB = {"intel": 471}
GAP_OTHER_MIB = 250
MIN_CARD_GB = 5.0                       # under this nothing here is claimed (the engine lowers its own reserve there, #496)

BASE_MIB = {"gyro-m": 5945, "gyro-s": 4720}
BASE_GENERIC_MIB = 2448

# Starts with request slots.  The generic base above is deliberately small (a default that fits today is never changed), so it cannot tell whether two more
# sessions leave the expert cache room; for Qwen3.8-Flash-Next the engine's own figures can, where they were measured.  Arc A750, IQ3_XXS with the MTP layer, 2 slots, 8192 tokens, the
# engine log: 7721 MiB usable (8 GB less the 471 MiB gap), 2519 MiB free after everything but the expert cache, which held the weights (1466 + 1781 MiB), the
# MTP layer (802), the Q5_K head (417), the main session (234) and the two slot sessions (468):  base = 7721 - 2519 - 234 - 468 = 4500 MiB.  The cache's
# largest expert blob is 2.33 MB (2.22 MiB).  Another quant or card holds somewhat different weights; MARGIN_MIB and the engine's own refusal cover the rest.
# CUDA, RTX PRO 5000 (sm_120) held to a 12 GB and an 8 GB card with a memory hog, IQ3_S pack, the MTP layer, --kv int8 --kv-resident 32768 at 131072 tokens (DF2-MEM,
# RESULTS-DF2MEM.md section 17), the engine log.  The plan's own terms (head, 700 MiB reserve, prompt buffers at the 2048 chunk) stay in the formula, so this base is the
# one that makes slack_mib() equal the engine's auto sizing of the cache, "free, less 700 reserved, 218 for the draft head, 256 allocated after the cache":
#   12 GB, --batch 2: 3.53 GiB free before the cache  -> 2441 MiB for it (961 slots);   12 GB, --batch 4: 2.07 GiB -> 946 MiB (371 slots)
#   base = 12000 usable - 748 main session - N * 748 slot sessions - (110 + 700 + 1522) - cache = 4984 (N 2) and 4994 (N 4), 4990 taken.
# 8 GB (7800 MiB left, the same engine): --batch 2 has 0.83 GiB free after the two slot sessions, the MTP layer takes 835 and the head upload dies "out of memory
# (7 MiB free)"; --batch 4 fits three sessions and the MTP layer's dense weights die "out of memory (99 MiB free)".  The formula says -1765 MiB for the first (the engine's own
# figures say -1757), so both are refused by the plan and given a context or fewer requests that fit.
BASE_SLOTTED_MIB = {("intel", "flash-next"): 4500, ("cuda", "flash-next"): 4990,
                    ("hip", "flash-next"): 3900}      # (vendor, model family): the A750, the CUDA engine, and a discrete AMD card (R9700, see the HIP lines below)
SLOT_SLOTTED_MIB = 2.22
# CUDA, 8 GB mimic (7942 MiB left), 2 requests at 4K: a start the plan counted 177 MiB over the one-slot limit loaded, sized a 336-slot cache with the reserve lowered
# to 424 MiB ("a working cache needs 263"), had 59 MiB free, and aborted at its first request ("ggml-cuda (strata mmq): out of memory").  The threshold for request
# slots on CUDA is the working cache the engine itself asks for: 263 slots of the largest blob (2.54 MiB with the IQ3_S pack).
WORKING_CACHE_MIB = {"cuda": 263 * 2.54, "hip": 263 * 2.54}      # (the same engine code sizes the cache on HIP)
SINGLE_MEASURED_VENDORS = ("cuda", "hip")                          # a single request is priced from the measured base on these (the draft layer, the head and the sessions are the same buffers)
HIP_BASE_NOTE = ("measured on an R9700 (gfx1201, p620) squeezed by a memory hog, DFlash2 alone, 128K, --kv-resident 32768: the engine's free VRAM before the expert cache is FREE - 5045 MiB at every "
                 "squeeze from 7000 to 5200 (the CUDA engine: FREE - 6324), so a HIP start holds about 1280 MiB less before the cache; base = 4990 - 879 - the runtime's context, taken as 200 -> 3900 (+-150). "
                 "Starts: 7942 MiB 787 slots 864 MiB free 48-52 tok/s; 7000 439 slots; 6400 273 slots (loan keeps 16); 5800 head moved to the host (31 tok/s); 5200 and 4600 refused (573 / 1171 MiB short).  "
                 "A discrete card only: drafting_vram_mib() gives None for an APU, so fit_start leaves Strix Halo alone")
# What the drafter adds to the measured base (MTP layer loaded: the base itself).  DFlash2 alone with the converted Q4_0-selector pack holds about 716 MiB where the
# MTP layer holds 835 (RESULTS-DF2MEM.md section 1); DFlash2 beside the MTP layer (--draft-fallback mtp) shares the draft head (214 MiB).  No drafter: the layer is gone.
DRAFTER_DELTA_MIB = {"mtp": 0, "dflash2": -119, "dflash2+mtp": 502, "none": -835}
# The knobs a start short of VRAM turns, mildest first, per backend.  CUDA: only the KV window; its engine sizes the expert cache and the prompt chunk itself
# (an explicit --prefill N is larger than the engine's own step-down on a card this tight, and a lower --vram-reserve-mib starves DFlash2's prompt staging: both
# measured on the 8 GB mimic).  Arc: the prompt chunk is the biggest cheap knob (DF2-SYCL, A750: 1522 MiB of buffers at chunk 2048, 508 at 512), then the reserve.
KNOBS = {"cuda": ("kv_resident",), "hip": ("kv_resident",), "intel": ("chunk", "reserve", "kv_resident"), "other": ("chunk", "reserve")}
NOTE_SLOW = "short of GPU memory: more is read from RAM and the SSD, expect it to be slower"
SLOT_RUN_MIN = 1                                       # the engine's minimum: generate.cpp refuses a start whose auto sizing leaves the cache no slot
KV_RESIDENT_MIN = 4096                                 # qsa_kv_resident_min(): a smaller --kv-resident is raised to this (was 20480; 4096 and 8192 gave the same tokens, RESULTS-DF2MEM.md section 18)
KV_RESIDENT_STEPS = (20480, 8192, 4096)                # the windows a short start walks down, mildest first (a session at 128K: 0.58 / 0.44 / 0.39 GiB)
STREAM_EXTRA_BYTES = 1400                              # per cell of context a streamed session still keeps in VRAM (rope table, indexer rows, page table); measured: the CUDA engine's slot session at 128K with 32768 resident is 0.73 GiB (RESULTS-DF2MEM.md section 17), the 32K session 0.56 GiB
SLOTS_MAX = 8                                    # the engine's batch window (serve/server.py PARALLEL_MAX)


def base_mib(model: str) -> int:
    low = str(model or "").lower().replace("_", "-")
    for key, mib in BASE_MIB.items():
        if key in low:
            return mib
    return BASE_GENERIC_MIB


def slotted_base_mib(model: str, vendor: str = "intel", drafter: str = "mtp") -> int | None:
    """The measured Flash-Next base for this vendor's cards with this drafter, None where none was measured."""
    low = str(model or "").lower().replace("_", "-")
    for (v, key), mib in BASE_SLOTTED_MIB.items():
        if v == str(vendor or "").lower() and key in low:
            return mib + DRAFTER_DELTA_MIB.get(drafter, 0)
    return None


def known_model(model: str) -> bool:
    low = str(model or "").lower().replace("_", "-")
    return any(k in low for k in BASE_MIB)


def session_mib(ctx: int, kv: str = "int8", kv_resident: int = 0) -> float:
    """A session's VRAM.  With KV streaming (kv_resident N, below the context) the KV pool holds max(N, 20480) cells; what scales with the whole
    context beyond that (the rope table, the indexer rows) stays."""
    if kv_resident and 0 < kv_resident < ctx:
        cells = max(int(kv_resident), KV_RESIDENT_MIN)
        if cells < ctx:
            return SESSION_FIXED_MIB + cells * 13 * KV_BYTES.get(kv, 1056) * KV_MEASURED_FACTOR / 2 ** 20 + ctx * STREAM_EXTRA_BYTES / 2 ** 20
    return SESSION_FIXED_MIB + ctx * 13 * KV_BYTES.get(kv, 1056) * KV_MEASURED_FACTOR / 2 ** 20


def chunk_of(prefill) -> int:
    """The prompt chunk in tokens: a number as given, `auto` (or nothing, or auto:N) is what auto came to on the 8 GB card."""
    try:
        return max(1, int(str(prefill)))
    except (TypeError, ValueError):
        return AUTO_CHUNK


def measured(model: str, slots: int, vendor: str, drafter: str = "mtp") -> bool:
    """Is this start priced from a measured Flash-Next base?  Request slots on any measured vendor; a single request only where the engine's single start was
    measured too (CUDA: DF2-MEM, the 8 GB mimic).  Elsewhere nothing is claimed and the engine decides."""
    if slotted_base_mib(model, vendor, drafter) is None:
        return False
    return slots >= 2 or str(vendor or "").lower() in SINGLE_MEASURED_VENDORS


def slack_mib(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto",
              vendor: str = "intel", slots: int = 0, kv_resident: int = 0, drafter: str = "mtp") -> float:
    """MiB the estimate leaves for the expert cache (negative: the engine would stop with "no VRAM is left for the expert cache").
    `slots` >= 2: a start with that many request slots, each one a session of the main one's size."""
    gap = GAP_MIB.get(str(vendor or "").lower(), GAP_OTHER_MIB)
    prompt = PROMPT_BASE_MIB + PROMPT_PER_TOKEN_MIB * chunk_of(prefill)
    base = base_mib(model)
    extra = 0.0
    main = session_mib(ctx, kv, 0)
    if measured(model, slots, vendor, drafter):
        base = slotted_base_mib(model, vendor, drafter)
        main = session_mib(ctx, kv, kv_resident)
        if slots >= 2:
            extra = min(int(slots), SLOTS_MAX) * main
    return vram_gb * 1024 - gap - base - main - extra - HEAD_MIB - max(int(reserve_mib), 0) - prompt


def needed_mib(model: str = "", slots: int = 0, vendor: str = "intel", drafter: str = "mtp") -> float:
    if measured(model, slots, vendor, drafter):
        if slots >= 2 and str(vendor or "").lower() in WORKING_CACHE_MIB:    # CUDA: a cache the engine calls working, not its one-slot refusal limit
            return WORKING_CACHE_MIB[str(vendor).lower()] + MARGIN_MIB
        return SLOT_RUN_MIN * SLOT_SLOTTED_MIB + MARGIN_MIB                # the engine's own minimum: the cache keeps at least a slot (plus the estimate's margin)
    return MIN_SLOTS * SLOT_MIB + MARGIN_MIB


def fits(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto", vendor: str = "intel", slots: int = 0,
         kv_resident: int = 0, drafter: str = "mtp") -> bool:
    return slack_mib(model, vram_gb, ctx, kv, reserve_mib, prefill, vendor, slots, kv_resident, drafter) >= needed_mib(model, slots, vendor, drafter)


def ctx_label(ctx: int) -> str:
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else f"{ctx} tokens"


@dataclass
class Fit:
    ctx: int                            # always the context asked for: a fit never changes it
    reserve_mib: int
    prefill: str | None = None          # a new chunk size to write, or None to leave --prefill as it was
    changed: bool = False
    fits: bool = True
    slack_mib: float = 0.0
    line: str = ""                      # one plain sentence for the user, "" when nothing changed
    changes: list = field(default_factory=list)
    kv_resident: int | None = None      # a new --kv-resident window, or None to leave it as it was
    slots: int = 0                      # the request slots of this start: always the number asked for


def plan(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, prefill="auto", vendor: str = "intel",
         title: str = "", ctx_fixed: bool = False, reserve_fixed: bool = False, prefill_fixed: bool = False, slots: int = 0, kv_resident: int = 0,
         drafter: str = "mtp", kv_fixed: bool = False) -> Fit:
    """The start that runs on this card, as close to the one asked for as possible; the SAME values back when that one already fits (byte-identical).

    The context (`ctx_fixed` is kept for the callers) and the number of request slots are never touched.  The knobs of KNOBS[vendor] are turned one after the other, each
    to the end of its steps before the next one (a smaller KV window down to the engine's 20480, a smaller prompt chunk, a smaller reserve down to RESERVE_FLOOR_MIB),
    until the estimate fits; a value the person chose (`*_fixed`) is not touched.  When nothing more can be moved the start is returned as far as it went with
    fits=False: the engine still starts as far as it can, and says what is short."""
    ctx = int(ctx)
    reserve_mib = int(reserve_mib)
    slots = int(slots) if slots and int(slots) >= 2 else 0
    keep = Fit(ctx, reserve_mib, None, False, True, 0.0, slots=slots)
    if not vram_gb or vram_gb < MIN_CARD_GB:
        return keep
    if (slots or str(vendor or "").lower() in SINGLE_MEASURED_VENDORS) and not measured(model, slots, vendor, drafter):
        return keep                      # nothing measured for this card: unchanged, the engine decides
    need = needed_mib(model, slots, vendor, drafter)
    keep.slack_mib = slack_mib(model, vram_gb, ctx, kv, reserve_mib, prefill, vendor, slots, kv_resident, drafter)
    if keep.slack_mib >= need:
        return keep
    knobs = KNOBS.get(str(vendor or "").lower(), KNOBS["other"])
    kvr, chunk, res = int(kv_resident or 0), prefill, reserve_mib

    def now():
        return slack_mib(model, vram_gb, ctx, kv, res, chunk, vendor, slots, kvr, drafter)

    for knob in knobs:
        if knob == "kv_resident":
            # the window is only ever made smaller, down the steps to the engine's minimum; a start with no window and a context beyond it gets one
            cur = kvr if kvr else ctx
            steps = [w for w in KV_RESIDENT_STEPS if w < cur and w < ctx]
            if kv_fixed or not steps or not measured(model, slots, vendor, drafter):
                continue                 # (a start priced without a measured base does not count the window, so moving it would only be a claim)
        elif knob == "chunk":
            steps = [] if prefill_fixed else [c for c in CHUNK_STEPS if c < chunk_of(chunk)]
        else:
            steps = [] if reserve_fixed or res <= RESERVE_FLOOR_MIB else [RESERVE_FLOOR_MIB]
        for step in steps:
            if knob == "kv_resident":
                kvr = step
            elif knob == "chunk":
                chunk = step
            else:
                res = step
            if now() >= need:
                break
        if now() >= need:
            break
    s = now()
    out = _result(title, vram_gb, ctx, reserve_mib, prefill, kv_resident, res, chunk, kvr, s, s >= need)
    out.slots = slots
    return out


def _result(title, vram_gb, ctx, reserve0, prefill0, kvr0, reserve, chunk, kvr, slack, ok) -> Fit:
    changes = []
    if int(kvr or 0) != int(kvr0 or 0):
        changes.append(f"KV window {kvr} tokens in VRAM, the rest read from RAM")
    if str(chunk) != str(prefill0):
        changes.append(f"prompts read in chunks of {chunk}")
    if reserve != reserve0:
        changes.append(f"memory kept free for other programs {reserve} MiB")
    name = title or "This model"
    line = ""
    if changes or not ok:
        line = f"{name} on this {vram_gb:.0f} GB card: {NOTE_SLOW}" + (" (" + ", ".join(changes) + ")" if changes else "")
    return Fit(ctx, reserve, str(chunk) if str(chunk) != str(prefill0) else None, bool(changes), ok, slack, line, changes,
               kv_resident=int(kvr) if int(kvr or 0) != int(kvr0 or 0) else None)


def drafter_of(args: list) -> str:
    """Which drafter this start loads, from its engine arguments: DFlash2 alone (`--draft dflash2 --draft-fallback none`), DFlash2 with the MTP layer behind it, the
    MTP layer, or none."""
    a = [str(x) for x in args]

    def val(f):
        return a[a.index(f) + 1] if f in a[:-1] else None

    if val("--draft") == "dflash2":
        return "dflash2" if val("--draft-fallback") == "none" else "dflash2+mtp"
    return "mtp" if "--mtp" in a[:-1] else "none"


def fit_start(cfg: dict, args: list, vram_mib: float | None, backend: str | None) -> tuple[list, int | None, str]:
    """The engine arguments of one start, moved off the GPU where the card is short: (args, parallel, note).  Never the context (--max-context) and never the number of
    requests at once: `parallel` comes back None.  The knobs are the backend's (KNOBS): on CUDA the KV window (--kv-resident); on Arc the prompt chunk and the
    reserve first.  A value the person set (``user_args``) is kept.  Nothing changes without a single card's memory (several cards, a shared-memory chip: None), on a
    layer split, or for DeepSeek, which size themselves."""
    n = cfg.get("parallel")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 2 else 0
    if not vram_mib or backend in (None, "", "metal"):
        return args, None, ""
    if cfg.get("layer_split") or "--layer-split" in args:
        return args, None, ""

    def flag(name):
        return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None

    res_s = flag("--kv-resident")
    kv_resident = int(res_s) if res_s and str(res_s).isdigit() else 0
    ctx_s = flag("--max-context")
    ctx = int(ctx_s) if ctx_s and str(ctx_s).isdigit() else 131072
    reserve_s = flag("--vram-reserve-mib")
    reserve = int(reserve_s) if reserve_s and str(reserve_s).isdigit() else 700      # the engine's default reserve
    user = [str(a) for a in (cfg.get("user_args") or [])]
    model = str(cfg.get("model_name") or flag("--native") or "")
    vendor = {"sycl": "intel", "cuda": "cuda", "hip": "hip"}.get(str(backend), "other")
    if not n and ("--batch" in args or "--slots" in args):
        b = flag("--batch") or flag("--slots") or "0"
        n = int(b) if str(b).isdigit() and int(b) >= 2 else 0
    pf = flag("--prefill") or "auto"
    fr = plan(model, float(vram_mib) / 1024.0, ctx, flag("--kv") or "int8", reserve, pf, vendor, title=str(cfg.get("model_name") or "This model"),
              ctx_fixed=True, reserve_fixed=(vendor in SINGLE_MEASURED_VENDORS) or ("--vram-reserve-mib" in user),
              prefill_fixed=not str(pf).startswith("auto"), slots=n, kv_resident=kv_resident, drafter=drafter_of(args),
              kv_fixed="--kv-resident" in user)
    if not fr.changed:
        return args, None, fr.line
    out = list(args)
    if fr.kv_resident is not None:
        if "--kv-resident" in out:
            out[out.index("--kv-resident") + 1] = str(fr.kv_resident)
        else:
            out += ["--kv-resident", str(fr.kv_resident)]
    if fr.prefill is not None and "--prefill" in out:
        out[out.index("--prefill") + 1] = fr.prefill
    if fr.reserve_mib != reserve and vendor not in SINGLE_MEASURED_VENDORS:
        if "--vram-reserve-mib" in out:
            out[out.index("--vram-reserve-mib") + 1] = str(fr.reserve_mib)
        else:
            out += ["--vram-reserve-mib", str(fr.reserve_mib)]
    return out, None, fr.line


def fit_context(model: str, vram_gb: float, ctx: int, kv: str = "int8", reserve_mib: int = 300, vendor: str = "intel") -> int:
    """The context to recommend: always `ctx`.  A fit does not change the context any more (owner, 2026-11-02); kept for the callers that ask."""
    return int(ctx)


# ----- a start that ended because a card ran out of memory --------------------------------------------------------------------------------------------------
# The engine's own lines for it (src/program/generate.cpp, serve): the cache has no room, the prompt path or a weight arena does not fit, the card has too
# little free VRAM.  They start with "strata generate:" / "strata serve:", which the launcher's generic cause search skips (those logs are chatty), and the words
# are not the generic "error|failed" ones, so the last such line used to be the picture reader's harmless "ggml_cuda_init: failed to initialize CUDA: no
# CUDA-capable device" (it runs on the processor when the cards are full).  One list, so the launcher's final line and the server's retry agree.
ENGINE_STOP_PATTERNS = (
    re.compile(r"expert cache: no room", re.I),
    re.compile(r"no VRAM is left for the expert cache", re.I),
    re.compile(r"too little free VRAM", re.I),
    re.compile(r"cannot start the prompt path", re.I),
    re.compile(r"device buffers .* do not fit", re.I),
    re.compile(r"weight arena (does not fit|failed)|cudaMalloc\(\d+\) for the weight arena failed", re.I),
    re.compile(r"(session state|head|native dense projections|weights).*(out of memory|does not fit|failed)", re.I),
)
ENGINE_LINE = re.compile(r"^(?:\[strata\]\s*)?strata (?:generate|serve|mtp|dflash2|prefill): ", re.I)
# a line that only reports what the walk did, never why a start stopped
_NOT_A_STOP = re.compile(r"short of GPU memory: |trying a smaller|trying \d+ of \d+ slots|a card this small|LOW: ", re.I)


def engine_stop_reason(lines) -> str | None:
    """The engine's own last line saying a card was out of memory, without its "strata generate: " prefix; None when the start stopped for another reason.
    `lines` are the log lines of ONE start (the launcher passes only what this attempt wrote)."""
    for raw in reversed([str(x).strip() for x in lines or []]):
        if not ENGINE_LINE.match(raw) or _NOT_A_STOP.search(raw):
            continue
        if any(p.search(raw) for p in ENGINE_STOP_PATTERNS):
            return ENGINE_LINE.sub("", raw, count=1).strip()
    return None
