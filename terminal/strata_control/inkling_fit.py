"""Which Inkling-2-Mini build fits a PC, and where its parts go: the one rule setup's menu, `strata launch`, the serve start line and the app's estimate use.

Thinking Machines Inkling-2-Mini (26.4B-A3.4B / 5.4B looped-active, 24 physical layers -> 48 virtual visits, `.strata` v1.1) comes as three builds of the same model:
  - SQ-20 (19.35 GB / 18.02 GiB): bit-exact 0.00000 KLD lossless native SF4+MX8 LLM + embedded DSpark Block-8 drafter
  - SQ-15 (14.35 GB / 13.36 GiB): flagship 16 GiB GPU / 24 GiB Mac / 12 GiB hybrid quality build + DSpark
  - SQ-10 (10.47 GB / 9.75 GiB): compact 12 GiB 100%-VRAM / 8 GiB GPU / 16 GiB Mac / 780M (gfx1103) & Strix Point (gfx1150) iGPU build + DSpark
plus an on-demand lossless BF16 Vision shard (`Inkling-2-Mini-Vision.strata`, 0.64 GB / 0.59 GiB, loaded when images are enabled/attached).

Hardware auto-selection (docs/10-GUI-TUI.md + Track 1E):
  - >= 24 GiB discrete GPU OR >= 32 GiB UMA (Mac 32..128 GB, Strix Halo gfx1151, DGX Spark): SQ-20 (100% VRAM / UMA)
  - 16 GiB discrete GPU OR 24 GiB UMA (24 GiB Mac): SQ-15 (100% VRAM / UMA, 0 spill)
  - 12 GiB discrete GPU + >= 24 GiB RAM: SQ-15 in quality mode (default, 3x loop-pinned hybrid) or SQ-10 in speed mode (100% VRAM)
  - 8 GiB discrete GPU OR 16 GiB UMA OR AMD Radeon 780M (gfx1103) / Strix Point 880M/890M (gfx1150) iGPUs: SQ-10

Multi-request parallelism (`--parallel 1|4|8|32|64`):
  - Only 8 of the 48 visits are global KV (8 * 8 kv_heads * 128 dim * 2 * 2 B = 32 KiB/token FP16, 16 KiB/token FP8).
  - The 40 SWA-1024 visits cap at min(seq_len, 1024) tokens (~0.168 GB/slot FP16, ~0.084 GB/slot FP8) plus 3.83 MB ShortConv state/slot.
"""

from __future__ import annotations

GIB_TO_GB = 1.073741824
VRAM_DRIVER_GB = 0.85            # driver / WDDM / runtime buffers
OS_RESERVE_GB = 4.50             # system RAM reserve on discrete or UMA hosts
UMA_RESERVE_GB = 5.00            # unified memory OS + driver reserve
SSD_NEAR_RESIDENT_GB = 2.0
VISION_GB = 0.637                # Inkling-2-Mini-Vision.strata (636,526,592 B)
DEFAULT_CONTEXT = 32768          # docs/10-GUI-TUI.md default: 32K (up to 1M available)
CONTEXTS = [8192, 32768, 65536, 131072, 262144]
MAX_CONTEXT = 1048576
DEFAULT_PARALLEL = 1
PARALLEL_CHOICES = (1, 4, 8, 32, 64)
MAX_PARALLEL = 64

# 8 global visits x 8 kv_heads x 128 head_dim x 2 (K+V) x 2 bytes (FP16) = 32,768 B/token
GLOBAL_KV_BYTES_PER_TOK_FP16 = 32768
GLOBAL_KV_BYTES_PER_TOK_FP8 = 16384
# 40 SWA-1024 visits x 1024 x 8 x 128 x 2 x 2 B = 167,772,160 B (~0.168 GB) + 3.83 MB ShortConv
SWA_AND_SCONV_GB_FP16 = 0.172
SWA_AND_SCONV_GB_FP8 = 0.088
KV_GB_PER_TOKEN = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9

FAMILY = "inkling-2-mini"
FAMILY_NAME = "inkling-2-mini"

#: AMD iGPU architectures that auto-select SQ-10 via coherent UMA/GTT allocation (even with small BIOS VRAM carve-out).
IGPU_SQ10_ARCHS = frozenset({"gfx1103", "gfx1150"})
#: High-bandwidth unified-memory GPUs (DGX Spark GB10 sm_121, Strix Halo gfx1151) that scale to SQ-20 on >= 32 GiB UMA.
BIG_UMA_ARCHS = frozenset({"121", "sm_121", "gfx1151"})

#: name -> exact Section 1 (dense + DSpark) and Section 4 (expert bodies) sizes from the verified .strata v1.1 binaries.
BUILDS = {
    "SQ-20": {"file": "Inkling-2-Mini-SQ-20.strata", "footprint_gb": 19.35, "dense_gb": 3.67, "bodies_gb": 15.68, "file_gb": 19.35},
    "SQ-15": {"file": "Inkling-2-Mini-SQ-15.strata", "footprint_gb": 14.35, "dense_gb": 2.07, "bodies_gb": 12.28, "file_gb": 14.35},
    "SQ-10": {"file": "Inkling-2-Mini-SQ-10.strata", "footprint_gb": 10.47, "dense_gb": 1.87, "bodies_gb": 8.60, "file_gb": 10.47},
}
ORDER = ("SQ-20", "SQ-15", "SQ-10")       # best quality first


def build_names() -> list[str]:
    return list(BUILDS)


def is_igpu_arch(arch: str | None) -> bool:
    return str(arch or "").strip().lower() in IGPU_SQ10_ARCHS


def is_big_uma_arch(arch: str | None, name: str | None = None) -> bool:
    a = str(arch or "").strip().lower()
    if a in BIG_UMA_ARCHS:
        return True
    low = str(name or "").lower()
    return "gb10" in low or "dgx spark" in low or "strix halo" in low


def kv_cache_gb(ctx: int | None = None, parallel: int = 1, fp8_kv: bool = False) -> float:
    """KV + SWA-1024 + ShortConv state in GB for `ctx` context tokens across `parallel` concurrent request slots."""
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if fp8_kv or slots >= 16:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9
        fixed = SWA_AND_SCONV_GB_FP8
        # Continuous batching allocates paged SWA blocks & global KV on demand per slot; budget realistic active occupancy
        active_ctx = min(n, 4096) if slots >= 16 else n
    else:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9
        fixed = SWA_AND_SCONV_GB_FP16
        active_ctx = n
    return round(slots * (active_ctx * per_tok + fixed), 3)


def non_expert_gb(ctx: int | None = None, parallel: int = 1, vision: bool = False) -> float:
    """Video memory kept free of expert bodies: driver/runtime buffers + KV/SWA/ShortConv state (+ optional Vision shard)."""
    return round(VRAM_DRIVER_GB + kv_cache_gb(ctx, parallel) + (VISION_GB if vision else 0.0), 3)


def compute(build: str, vram_gib: float, ram_gib: float, ctx: int | None = None, vision: bool = False, parallel: int = 1) -> dict:
    """Placement of `build` on a discrete-GPU PC (vram_gib + ram_gib): GB of experts in VRAM, in RAM, and on SSD."""
    info = BUILDS[build]
    vram_gb = float(vram_gib or 0.0) * GIB_TO_GB
    ram_gb = float(ram_gib or 0.0) * GIB_TO_GB
    gpu_room = max(0.0, vram_gb - info["dense_gb"] - non_expert_gb(ctx, parallel, vision=False))
    ram_room = max(0.0, ram_gb - OS_RESERVE_GB - (VISION_GB if vision else 0.0))
    in_vram = min(info["bodies_gb"], gpu_room)
    in_ram = min(info["bodies_gb"] - in_vram, ram_room)
    on_ssd = max(0.0, info["bodies_gb"] - in_vram - in_ram)
    vram_total = info["dense_gb"] + in_vram
    return {
        "vram_gb": round(in_vram, 2),
        "ram_gb": round(in_ram, 2),
        "ssd_gb": round(on_ssd, 2),
        "full_vram": on_ssd <= 0.05 and in_ram <= 0.05 and vram_total <= vram_gb,
    }


def compute_unified(build: str, pool_gib: float, ctx: int | None = None, vision: bool = False, parallel: int = 1) -> dict:
    """Placement on a unified-memory / coherent-UMA host (Apple Silicon, Strix Halo, Strix Point 890M, Radeon 780M, DGX Spark)."""
    info = BUILDS[build]
    room = max(0.0, float(pool_gib or 0.0) * GIB_TO_GB - UMA_RESERVE_GB - kv_cache_gb(ctx, parallel) - (VISION_GB if vision else 0.0))
    held = min(info["footprint_gb"], room)
    on_ssd = max(0.0, info["footprint_gb"] - held)
    return {"vram_gb": round(held, 2), "ram_gb": 0.0, "ssd_gb": round(on_ssd, 2), "full_vram": on_ssd <= 0.05}


def place(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
          vision: bool = False, parallel: int = 1) -> dict:
    if unified:
        return compute_unified(build, ram_gib, ctx, vision, parallel)
    return compute(build, vram_gib or 0.0, ram_gib, ctx, vision, parallel)


def mode(placed: dict | float) -> str:
    """Placement verdict: full_vram (100% GPU/UMA), hybrid_ram_vram (zero SSD spill), near_resident_ssd, or cross_tier_ssd."""
    if isinstance(placed, dict):
        ssd_gb = float(placed.get("ssd_gb", 0.0))
        if ssd_gb <= 0.05:
            return "full_vram" if placed.get("full_vram") else "ram_vram_zero_ssd"
        if ssd_gb <= SSD_NEAR_RESIDENT_GB:
            return "near_resident_ssd"
        return "cross_tier_ssd"
    ssd_gb = float(placed)
    if ssd_gb <= 0.05:
        return "ram_vram_zero_ssd"
    if ssd_gb <= SSD_NEAR_RESIDENT_GB:
        return "near_resident_ssd"
    return "cross_tier_ssd"


def fits(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
         vision: bool = False, parallel: int = 1) -> bool:
    return place(build, vram_gib, ram_gib, unified, ctx, vision, parallel)["ssd_gb"] <= 0.05


def recommend(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
              vision: bool = False, *, preference: str = "quality", igpu: bool = False, arch: str | None = None,
              parallel: int = 1) -> dict:
    """Automatic hardware auto-selection for Inkling-2-Mini (docs/10-GUI-TUI.md + Track 1E):
      - Radeon 780M (gfx1103) / Strix Point (gfx1150) iGPUs -> SQ-10 (coherent UMA/GTT, zero BIOS carve-out OOM)
      - >= 24 GiB discrete GPU OR >= 32 GiB UMA -> SQ-20
      - 16 GiB discrete GPU (>= 15 GiB) OR 24 GiB UMA (>= 22 GiB) -> SQ-15
      - 12 GiB discrete GPU (11..15 GiB): SQ-15 in quality mode (with >= 24 GiB RAM) or SQ-10 in speed mode / < 24 GiB RAM
      - 8 GiB discrete GPU (< 11 GiB) OR <= 16 GiB UMA -> SQ-10
    """
    v = float(vram_gib or 0.0)
    r = float(ram_gib or 0.0)
    is_small_igpu = bool(igpu or is_igpu_arch(arch))
    if is_small_igpu:
        best = "SQ-10"
        use_unified = True
    elif unified:
        use_unified = True
        if r >= 30.0:
            best = "SQ-20"
        elif r >= 22.0:
            best = "SQ-15"
        else:
            best = "SQ-10"
    else:
        use_unified = False
        if v >= 22.0:
            best = "SQ-20"
        elif v >= 15.0:
            best = "SQ-15"
        elif v >= 11.0:
            best = "SQ-10" if (str(preference).lower() == "speed" or r < 24.0) else "SQ-15"
        else:
            best = "SQ-10"
    placed = place(best, vram_gib, ram_gib, use_unified, ctx, vision, parallel)
    return {
        "build": best,
        "placement": placed,
        "mode": mode(placed),
        "fits": placed["ssd_gb"] <= 0.05,
        "unified": use_unified,
    }


def rows(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
         vision: bool = False, parallel: int = 1) -> list[dict]:
    out = []
    for b in ORDER:
        placed = place(b, vram_gib, ram_gib, unified, ctx, vision, parallel)
        out.append({
            "build": b,
            "file": BUILDS[b]["file"],
            "download_gb": BUILDS[b]["file_gb"],
            "footprint_gb": BUILDS[b]["footprint_gb"],
            "placement": placed,
            "mode": mode(placed),
        })
    return out


def disk_gb(build: str, vision: bool = False) -> float:
    return round(BUILDS[build]["file_gb"] + (VISION_GB if vision else 0.0), 2)


def build_of_model_name(model_name: str | None) -> str | None:
    low = str(model_name or "").lower()
    if not low.startswith(FAMILY_NAME + "-"):
        return None
    size = low[len(FAMILY_NAME) + 1:].upper()
    return size if size in BUILDS else None
