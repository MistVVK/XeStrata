"""Which Hy4-Flash build fits a PC, and where its parts go: the one rule setup's menu, `strata launch`, the serve start line and the app's estimate use.

Tencent Hy4-Flash (220B-A12B, `.strata` v1.0) comes as four builds of the same model (SQ-32, SQ-64, SQ-96, SQ-128: how many GiB of memory they are made for), one
Engram shard they all share (Hy4-Flash-Engram-Q8.strata, 67.58 GB, read from the SSD, never loaded) and a Vision shard (images are on by default).  The numbers and the rule
are the model's author's (docs/10-GUI-TUI.md of the quantization pack, `terminal/tests/data/hy4_squant_fit_cases.json`, at the author's 20K context), pinned by
tests/terminal/test_hy4_squant_fit.py, plus what the 0.2.18 defaults add (owner, 2026-11-19: 128K context, the drafter and images on):

    gpu_room  = max(0, vram - dense - non_expert(ctx))              the dense trunk (and the 5-layer DFly2 drafter: 0.44 GB, inside `dense`) sit in video memory
    ram_room  = max(0, ram - OS_RESERVE_GB - HOT_ENGRAM_GB - vision) what is left of main memory after the system, the Engram cache (64 MiB seed + 836 MiB LRU) and,
                                                                    with images on, the Vision shard (0.93 GB, loaded on demand)
    in VRAM   = min(bodies, gpu_room)                               the hottest experts (the file stores them hot first)
    in RAM    = min(bodies - in VRAM, ram_room)
    on SSD    = the rest                                            read from the file while it answers

    non_expert(ctx) = VRAM_NON_EXPERT_GB + kv_extra(ctx)            the author's 2.00 GB holds the KV of a 20K context; a longer one adds what it measured to take:
    kv_extra(ctx)   = max(0, ctx - 20480) x KV_GB_PER_TOKEN         HY4-STRIX measured 66.3 GiB resident at 128K against 62.2 GiB at 4K on SQ-64 (about 4.1 GiB for 124K more
                                                                    tokens, 34.7 KB a token).  It is "KV + context workspace", not the KV alone: the MLA latent cache of the 12 sparse layers (1.9 GB at 128K on the
                                                                    M5 Max), the 30 SWA rings, and the IndexPool, workspace and Engram buffers, whose size also grows with the context. HY4-METAL measured the
                                                                    same kind of total on Metal (SQ-64, on top of the mapped 62 GB file): 1.6 GB at 4K, 4.4 GB at 32K, 6.6 GB at 128K (+5.0 GB); Strix is +4.4 GB; this is +3.84 GB
                                                                    beyond the author's 20K, the same range (to be checked against the CUDA lane's RTX 5070 numbers at 128K)

Unified memory (Strata Halo, Apple silicon, DGX Spark) has one pool: the trunk, the KV and every expert body live in it and nothing is copied or read from the SSD while the
footprint fits `pool - UMA_RESERVE_GB - kv_extra - vision`.

Nothing here changes a start: the result is a recommendation and a warning (recommend, never force).  Sizes are in GB (10^9 bytes) as the model's author states them; the
per-file byte counts and SHA-256 of the download are in hy4_family.FILES.  `ctx` defaults to DEFAULT_CONTEXT (128K); the author's cases pass AUTHOR_CONTEXT.
"""

from __future__ import annotations

GIB_TO_GB = 1.073741824
VRAM_NON_EXPERT_GB = 2.00        # 1.00 GB driver/WDDM + 0.32 GB fp16 MLA KV at 20K context + 0.68 GB buffers
OS_RESERVE_GB = 5.10
HOT_ENGRAM_GB = 0.90             # Section 2 seed (64 MiB) + clock LRU (836 MiB), in RAM
UMA_RESERVE_GB = OS_RESERVE_GB + HOT_ENGRAM_GB + VRAM_NON_EXPERT_GB - 1.0   # one pool: the system, the Engram cache and the buffers (the driver's 1 GB is the pool's own)
SSD_NEAR_RESIDENT_GB = 5.0       # up to this much read from the SSD is "near resident": a few percent of the experts, mostly cold ones
ENGRAM_GB = 67.58                # the shared Engram shard, on the SSD (needs a fast one: random 4 KiB reads)
VISION_GB = 0.934                # Hy4-Flash-Vision.strata (934,092,800 B): in the default download set; loaded on demand, so its memory is reserved in RAM only when images are on
AUTHOR_CONTEXT = 20480           # the context the author's 2.00 GB (and his 7 cases) are made for
DEFAULT_CONTEXT = 131072         # owner, 2026-11-19: 128K by default
CONTEXTS = [8192, 32768, 65536, 131072]
KV_GB_PER_TOKEN = 34.7e-6        # KV + context workspace (IndexPool, workspace and Engram buffers grow with it too), per token: HY4-STRIX, Strix Halo SQ-64: (66.3 - 62.2) GiB x 1.0737 / (131072 - 4096); HY4-METAL (M5 Max): +5.0 GB from 4K to 128K, KV alone 1.9 GB of it
DRAFTER_GB = 0.44                # the 5-layer DFly2 drafter, in Section 1 already (inside every build's dense_gb): kept here to say so, not added twice
FAMILY = "hy4-flash"
FAMILY_NAME = "hy4-flash"        # the config's model_name prefix

#: name -> (file, footprint_gb, dense_gb, bodies_gb, file_gb).  footprint = dense + bodies (what the PC holds); file_gb is the download.
BUILDS = {
    "SQ-32": {"file": "Hy4-Flash-SQ-32.strata", "footprint_gb": 37.98, "dense_gb": 5.25, "bodies_gb": 32.73, "file_gb": 38.09},
    "SQ-64": {"file": "Hy4-Flash-SQ-64.strata", "footprint_gb": 61.84, "dense_gb": 5.80, "bodies_gb": 56.04, "file_gb": 62.03},
    "SQ-96": {"file": "Hy4-Flash-SQ-96.strata", "footprint_gb": 89.43, "dense_gb": 6.50, "bodies_gb": 82.93, "file_gb": 89.58},
    "SQ-128": {"file": "Hy4-Flash-SQ-128.strata", "footprint_gb": 113.06, "dense_gb": 7.30, "bodies_gb": 105.76, "file_gb": 113.65},
}
ORDER = ("SQ-128", "SQ-96", "SQ-64", "SQ-32")      # best quality first


def build_names() -> list[str]:
    return list(BUILDS)


def kv_extra_gb(ctx: int | None = None) -> float:
    """What a context of `ctx` tokens holds beyond the author's 20K (GB): the KV and the context workspace (IndexPool, workspace and Engram buffers), see the module text."""
    n = DEFAULT_CONTEXT if ctx is None else int(ctx)
    return max(0, n - AUTHOR_CONTEXT) * KV_GB_PER_TOKEN


def non_expert_gb(ctx: int | None = None) -> float:
    """Video memory the card keeps for what is not an expert body at this context: the driver, the buffers and the KV."""
    return VRAM_NON_EXPERT_GB + kv_extra_gb(ctx)


def compute(build: str, vram_gib: float, ram_gib: float, ctx: int | None = None, vision: bool = False) -> dict:
    """Where the expert bodies of `build` go on a PC with a card of `vram_gib` and `ram_gib` of main memory (GiB): GB in video memory, in RAM, on the SSD."""
    info = BUILDS[build]
    gpu_room = max(0.0, vram_gib * GIB_TO_GB - info["dense_gb"] - non_expert_gb(ctx))
    ram_room = max(0.0, ram_gib * GIB_TO_GB - OS_RESERVE_GB - HOT_ENGRAM_GB - (VISION_GB if vision else 0.0))
    in_vram = min(info["bodies_gb"], gpu_room)
    in_ram = min(info["bodies_gb"] - in_vram, ram_room)
    on_ssd = max(0.0, info["bodies_gb"] - in_vram - in_ram)
    return {"vram_gb": round(in_vram, 2), "ram_gb": round(in_ram, 2), "ssd_gb": round(on_ssd, 2)}


def compute_unified(build: str, pool_gib: float, ctx: int | None = None, vision: bool = False) -> dict:
    """One memory pool of `pool_gib` GiB (Strix Halo: what the GPU can map; Mac: the unified memory): everything in it, the rest on the SSD."""
    info = BUILDS[build]
    room = max(0.0, pool_gib * GIB_TO_GB - UMA_RESERVE_GB - kv_extra_gb(ctx) - (VISION_GB if vision else 0.0))
    held = min(info["footprint_gb"], room)
    return {"vram_gb": round(held, 2), "ram_gb": 0.0, "ssd_gb": round(max(0.0, info["footprint_gb"] - held), 2)}


def place(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, vision: bool = False) -> dict:
    """The placement of `build`: compute() for a PC with a card, compute_unified() for one pool (`ram_gib` is then the pool), compute() with no card at all (the CPU
    engine path is not offered: vram_gib None gives the whole footprint on the SSD side of the card, i.e. the RAM rule only)."""
    if unified:
        return compute_unified(build, ram_gib, ctx, vision)
    return compute(build, vram_gib or 0.0, ram_gib, ctx, vision)


def mode(ssd_gb: float) -> str:
    """The words of the author's table: resident, near resident, or streaming from the SSD."""
    if ssd_gb <= 0.05:
        return "ram_vram_zero_ssd"
    if ssd_gb <= SSD_NEAR_RESIDENT_GB:
        return "near_resident_ssd"
    return "cross_tier_ssd"


def fits(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, vision: bool = False) -> bool:
    """Every expert body stays in memory (nothing read from the SSD while it answers)."""
    return mode(place(build, vram_gib, ram_gib, unified, ctx, vision)["ssd_gb"]) == "ram_vram_zero_ssd"


def context_fits(build: str, vram_gib: float | None, ctx: int | None = None, unified: bool = False) -> bool:
    """The card holds the trunk, the drafter, the driver, the buffers and the KV of `ctx` tokens (the experts that do not fit go to RAM, so they do not count here).  A unified
    pool always says yes (the placement then says how much is read from the SSD)."""
    if unified or vram_gib is None:
        return True
    return BUILDS[build]["dense_gb"] + non_expert_gb(ctx) <= float(vram_gib) * GIB_TO_GB


def recommend_context(build: str, vram_gib: float | None, unified: bool = False, want: int | None = None) -> int:
    """The context to start with: `want` (default 128K) when the card holds its KV, else the largest of CONTEXTS that it holds (never below the smallest)."""
    top = DEFAULT_CONTEXT if want is None else int(want)
    for c in sorted((x for x in CONTEXTS if x <= top), reverse=True) or [CONTEXTS[0]]:
        if context_fits(build, vram_gib, c, unified):
            return c
    return CONTEXTS[0]


def recommend(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, vision: bool = False) -> dict:
    """The build to offer first on this PC: the best quality whose experts all stay in memory; when none does, the one with the least read from the SSD
    (ties to the smaller build).  Returns {"build", "placement", "mode", "fits"}.  Never refuses: a PC below SQ-32 still gets SQ-32 and the words say it streams."""
    best = next((b for b in ORDER if fits(b, vram_gib, ram_gib, unified, ctx, vision)), None)
    if best is None:
        best = min(reversed(ORDER), key=lambda b: place(b, vram_gib, ram_gib, unified, ctx, vision)["ssd_gb"])
    placed = place(best, vram_gib, ram_gib, unified, ctx, vision)
    return {"build": best, "placement": placed, "mode": mode(placed["ssd_gb"]), "fits": fits(best, vram_gib, ram_gib, unified, ctx, vision)}


def rows(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None, vision: bool = False) -> list[dict]:
    """Every build for this PC, best quality first, for a menu: name, download GB (the Engram and Vision shards are the caller's sum), placement, mode."""
    out = []
    for b in ORDER:
        placed = place(b, vram_gib, ram_gib, unified, ctx, vision)
        out.append({"build": b, "file": BUILDS[b]["file"], "download_gb": BUILDS[b]["file_gb"], "footprint_gb": BUILDS[b]["footprint_gb"],
                    "placement": placed, "mode": mode(placed["ssd_gb"])})
    return out


def disk_gb(build: str, engram_present: bool = False, vision: bool = False) -> float:
    """Disk the files of one choice take: the build, the Engram shard unless it is there already, the Vision shard when wanted."""
    return round(BUILDS[build]["file_gb"] + (0.0 if engram_present else ENGRAM_GB) + (VISION_GB if vision else 0.0), 2)


def build_of_model_name(model_name: str | None) -> str | None:
    """"hy4-flash-sq-64" -> "SQ-64" (a config's model_name), None for any other model."""
    low = str(model_name or "").lower()
    if not low.startswith(FAMILY_NAME + "-"):
        return None
    size = low[len(FAMILY_NAME) + 1:].upper()
    return size if size in BUILDS else None
