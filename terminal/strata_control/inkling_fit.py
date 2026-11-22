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
VRAM_DRIVER_GB = 1.60            # driver / WDDM / HIP / desktop compositor + runtime & O(1) prefill scratch (#6613)
VRAM_HYBRID_WORKSPACE_GB = 0.85  # hybrid ping-pong DMA ring + staging buffers + desktop margin (#6587, #6613)
KV_GROW_FACTOR = 1.60            # peak transient headroom for elastic/doubling KV pool growth (HIP grow_pool #6613)
SPLIT_GPU_OVERHEAD_GB = 1.00     # per-extra-GPU runtime/trunk/comm overhead on multi-GPU layer-split
DSPARK_BASE_GB = 0.11            # DSpark activation + compact SWA-1025 rings for layers 0 & 2
DSPARK_KV_BYTES_PER_TOK = 4096   # 1 global DSpark layer (L1) x 8 kv_heads x 128 head_dim x 2 (K+V) x 2B (FP16)
OS_RESERVE_GB = 4.50             # system RAM reserve on discrete or UMA hosts
UMA_RESERVE_GB = 5.00            # unified memory OS + driver reserve
SSD_NEAR_RESIDENT_GB = 2.0
VISION_GB = 0.637                # Inkling-2-Mini-Vision.strata (636,526,592 B, CPU / on-demand on <=12 GiB GPUs)
DSPARK_GGUF_GB = 0.402           # dspark-Inkling-2-Mini-Q8_0.gguf (402,278,560 B)
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

#: name -> exact Section 1 (dense + DSpark) and Section 4 (expert bodies) sizes from the verified .strata v1.1 binaries,
#: plus direct-download Unsloth Dynamic GGUF builds from unsloth/Inkling-2-Mini-GGUF.
BUILDS = {
    "SQ-20": {"file": "Inkling-2-Mini-SQ-20.strata", "footprint_gb": 19.35, "dense_gb": 3.67, "bodies_gb": 15.68, "file_gb": 19.35, "format": "strata"},
    "SQ-15": {"file": "Inkling-2-Mini-SQ-15.strata", "footprint_gb": 14.35, "dense_gb": 2.07, "bodies_gb": 12.28, "file_gb": 14.35, "format": "strata"},
    "SQ-10": {"file": "Inkling-2-Mini-SQ-10.strata", "footprint_gb": 10.47, "dense_gb": 1.87, "bodies_gb": 8.60, "file_gb": 10.47, "format": "strata"},
    "UD-Q8_K_XL": {"file": "Inkling-2-Mini-UD-Q8_K_XL.gguf", "footprint_gb": 22.87, "dense_gb": 2.87, "bodies_gb": 20.00, "file_gb": 22.87, "format": "gguf"},
    "UD-Q4_K_XL": {"file": "Inkling-2-Mini-UD-Q4_K_XL.gguf", "footprint_gb": 17.63, "dense_gb": 2.15, "bodies_gb": 15.48, "file_gb": 17.63, "format": "gguf"},
    "UD-Q4_K_M": {"file": "Inkling-2-Mini-UD-Q4_K_M.gguf", "footprint_gb": 16.61, "dense_gb": 2.10, "bodies_gb": 14.51, "file_gb": 16.61, "format": "gguf"},
    "UD-IQ4_XS": {"file": "Inkling-2-Mini-UD-IQ4_XS.gguf", "footprint_gb": 15.73, "dense_gb": 2.05, "bodies_gb": 13.68, "file_gb": 15.73, "format": "gguf"},
    "UD-Q3_K_XL": {"file": "Inkling-2-Mini-UD-Q3_K_XL.gguf", "footprint_gb": 13.63, "dense_gb": 2.00, "bodies_gb": 11.63, "file_gb": 13.63, "format": "gguf"},
    "UD-IQ3_XXS": {"file": "Inkling-2-Mini-UD-IQ3_XXS.gguf", "footprint_gb": 11.66, "dense_gb": 1.95, "bodies_gb": 9.71, "file_gb": 11.66, "format": "gguf"},
    "UD-Q2_K_XL": {"file": "Inkling-2-Mini-UD-Q2_K_XL.gguf", "footprint_gb": 9.14, "dense_gb": 1.88, "bodies_gb": 7.26, "file_gb": 9.14, "format": "gguf"},
    "UD-IQ2_XXS": {"file": "Inkling-2-Mini-UD-IQ2_XXS.gguf", "footprint_gb": 8.49, "dense_gb": 1.85, "bodies_gb": 6.64, "file_gb": 8.49, "format": "gguf"},
    "UD-IQ1_M": {"file": "Inkling-2-Mini-UD-IQ1_M.gguf", "footprint_gb": 8.02, "dense_gb": 1.82, "bodies_gb": 6.20, "file_gb": 8.02, "format": "gguf"},
    "UD-IQ1_S": {"file": "Inkling-2-Mini-UD-IQ1_S.gguf", "footprint_gb": 7.76, "dense_gb": 1.80, "bodies_gb": 5.96, "file_gb": 7.76, "format": "gguf"},
}
STRATA_ORDER = ("SQ-20", "SQ-15", "SQ-10")       # primary recommended native .strata builds, best quality first
GGUF_ORDER = (
    "UD-Q8_K_XL", "UD-Q4_K_XL", "UD-Q4_K_M", "UD-IQ4_XS", "UD-Q3_K_XL",
    "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S",
)
ORDER = (*STRATA_ORDER, *GGUF_ORDER)


def is_gguf_build(build: str | None) -> bool:
    return BUILDS.get(str(build or ""), {}).get("format") == "gguf"


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


def dspark_gb(ctx: int | None = None, enabled: bool = True) -> float:
    """DSpark drafter KV + scratch footprint in GB for `ctx` tokens (0 when drafting is disabled)."""
    if not enabled:
        return 0.0
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    return round(DSPARK_BASE_GB + n * DSPARK_KV_BYTES_PER_TOK / 1e9, 3)


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


def auto_fp8_kv(vram_gib: float | None, ctx: int | None = None, parallel: int = 1, kv_offload: str = "auto") -> bool:
    """Whether FP8 KV is auto-selected. With Host KV Offload ('auto'/'on'), lossless FP16 stays default (#552)
    except at high concurrency (parallel >= 16) where 40 SWA rings dominate VRAM."""
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if slots >= 16:
        return True
    if str(kv_offload or "auto").lower() == "off":
        v = float(vram_gib or 0.0)
        return 0.0 < v <= 13.0
    return False


def global_kv_gb(ctx: int | None = None, parallel: int = 1, fp8_kv: bool = False) -> float:
    """Global KV cache size in GB across the 8 global visits (spillable to mapped pinned host RAM via --kv-offload)."""
    n = DEFAULT_CONTEXT if ctx is None else max(1024, int(ctx))
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    if fp8_kv or slots >= 16:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP8 / 1e9
        active_ctx = min(n, 4096) if slots >= 16 else n
    else:
        per_tok = GLOBAL_KV_BYTES_PER_TOK_FP16 / 1e9
        active_ctx = n
    return round(slots * active_ctx * per_tok, 3)


def swa_sconv_gb(parallel: int = 1, fp8_kv: bool = False) -> float:
    """VRAM-resident 40x SWA-1024 + ShortConv state in GB."""
    slots = max(1, min(MAX_PARALLEL, int(parallel or 1)))
    fixed = SWA_AND_SCONV_GB_FP8 if (fp8_kv or slots >= 16) else SWA_AND_SCONV_GB_FP16
    return round(slots * fixed, 3)


def non_expert_gb(ctx: int | None = None, parallel: int = 1, vision: bool = False, *,
                  dspark: bool = True, fp8_kv: bool = False, hybrid: bool = False,
                  host_offload_global_kv: bool = False) -> float:
    """Video memory kept free of expert bodies: driver/desktop + hybrid workspace + DSpark KV + KV/SWA/ShortConv (+ peak KV growth margin + Vision shard)."""
    if host_offload_global_kv:
        kv_vram = swa_sconv_gb(parallel, fp8_kv=fp8_kv)
    else:
        kv_vram = round(global_kv_gb(ctx, parallel, fp8_kv=fp8_kv) * KV_GROW_FACTOR + swa_sconv_gb(parallel, fp8_kv=fp8_kv), 3)
    return round(
        VRAM_DRIVER_GB
        + (VRAM_HYBRID_WORKSPACE_GB if hybrid else 0.0)
        + dspark_gb(ctx, enabled=dspark)
        + kv_vram
        + (VISION_GB if vision else 0.0),
        3,
    )


def compute(build: str, vram_gib: float, ram_gib: float, ctx: int | None = None, vision: bool = False,
            parallel: int = 1, dspark: bool = True, kv_offload: str = "auto", *,
            extra_vram_gib: float = 0.0, backend: str | None = None) -> dict:
    """Placement of `build` on a discrete-GPU PC (vram_gib + optional extra_vram_gib + ram_gib): GB of experts in VRAM, in RAM, and on SSD.

    Implements the unified 4-tier adaptive contract (GPU -> RAM+CPU -> SSD) with Host KV Offloading (#552),
    CPU/on-demand Vision on <= 12.5 GiB GPUs, O(1) prefill & KV growth headroom (#6613), and WDDM floor so
    4/6/8/10/12/16/24/32+ GB GPUs never OOM even at 32K–128K context with DSpark and Vision enabled.
    """
    info = BUILDS[build]
    v_primary = float(vram_gib or 0.0)
    v_extra = max(0.0, float(extra_vram_gib or 0.0))
    v_gib = v_primary + v_extra
    split_overhead = SPLIT_GPU_OVERHEAD_GB if v_extra > 0.0 else 0.0
    vram_gb = max(0.0, v_gib * GIB_TO_GB - split_overhead)
    ram_gb = float(ram_gib or 0.0) * GIB_TO_GB
    offload_mode = str(kv_offload or "auto").lower()
    backend_low = str(backend or "").strip().lower()
    # Only CUDA implements mapped pinned-host global KV offload (and auto-enables it on <= 13 GiB cards);
    # HIP and SYCL keep global KV in VRAM (#6613).
    supports_host_kv = backend_low not in ("hip", "sycl")
    effective_offload = offload_mode if supports_host_kv else "off"
    fp8_kv = auto_fp8_kv(v_gib, ctx, parallel, kv_offload=effective_offload)
    kv_dtype = "fp8" if fp8_kv else "fp16"
    extra_dspark_weights = DSPARK_GGUF_GB if (dspark and info.get("format") == "gguf") else 0.0
    dense_total = info["dense_gb"] + extra_dspark_weights
    # Vision stays in host RAM / CPU on <= 12.5 GiB GPUs so it never steals ~0.64 GB of hot 3x-looped expert VRAM!
    vision_in_vram = bool(vision and v_gib > 12.5)
    glob_kv = global_kv_gb(ctx, parallel, fp8_kv=fp8_kv)
    ram_avail_for_model = max(0.0, ram_gb - OS_RESERVE_GB - (VISION_GB if vision else 0.0))

    # 1. Try 100%-VRAM expert residency with VRAM-resident global KV first
    full_need = dense_total + info["bodies_gb"] + non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=False, host_offload_global_kv=False
    )
    if vram_gb > 0.0 and effective_offload != "on" and full_need <= vram_gb:
        return {
            "vram_gb": round(info["bodies_gb"], 2),
            "ram_gb": 0.0,
            "ssd_gb": 0.0,
            "host_kv_gb": 0.0,
            "kv_dtype": kv_dtype,
            "kv_offload": "off",
            "full_vram": True,
        }

    # 2. Try 100%-VRAM expert residency with Host KV Offloading (matches CUDA engine auto-offload on <= 13 GiB,
    # e.g. 12 GiB RTX 5070 + SQ-10 + lossless FP16 32K; 16 GiB cards keep global KV in VRAM so HIP never OOMs #6613).
    can_offload_kv = supports_host_kv and (effective_offload != "off") and (ram_avail_for_model >= glob_kv)
    auto_offload_full = can_offload_kv and (effective_offload == "on" or v_gib <= 13.0)
    full_need_offload = dense_total + info["bodies_gb"] + non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=False, host_offload_global_kv=True
    )
    if vram_gb > 0.0 and auto_offload_full and full_need_offload <= vram_gb:
        return {
            "vram_gb": round(info["bodies_gb"], 2),
            "ram_gb": 0.0,
            "ssd_gb": 0.0,
            "host_kv_gb": round(glob_kv, 2),
            "kv_dtype": kv_dtype,
            "kv_offload": "on",
            "full_vram": True,
        }

    # 3. Hybrid 3/4-tier placement (VRAM hot 3x-looped core -> Host RAM -> Auto-SSD)
    use_host_kv = can_offload_kv and (effective_offload == "on" or v_gib <= 13.0 or glob_kv > 1.5)
    host_kv_used = glob_kv if use_host_kv else 0.0
    gpu_room = max(0.0, vram_gb - dense_total - non_expert_gb(
        ctx, parallel, vision=vision_in_vram, dspark=dspark, fp8_kv=fp8_kv, hybrid=True, host_offload_global_kv=use_host_kv
    ))
    ram_room = max(0.0, ram_avail_for_model - host_kv_used)
    in_vram = min(info["bodies_gb"], gpu_room)
    in_ram = min(info["bodies_gb"] - in_vram, ram_room)
    on_ssd = max(0.0, info["bodies_gb"] - in_vram - in_ram)
    return {
        "vram_gb": round(in_vram, 2),
        "ram_gb": round(in_ram, 2),
        "ssd_gb": round(on_ssd, 2),
        "host_kv_gb": round(host_kv_used, 2),
        "kv_dtype": kv_dtype,
        "kv_offload": "on" if use_host_kv else "off",
        "full_vram": False,
    }


def compute_unified(build: str, pool_gib: float, ctx: int | None = None, vision: bool = False,
                    parallel: int = 1, dspark: bool = True) -> dict:
    """Placement on a unified-memory / coherent-UMA host (Apple Silicon, Strix Halo, Strix Point 890M, Radeon 780M, DGX Spark)."""
    info = BUILDS[build]
    extra_dspark = DSPARK_GGUF_GB if (dspark and info.get("format") == "gguf") else 0.0
    need_total = info["footprint_gb"] + extra_dspark
    fp8_kv = auto_fp8_kv(None, ctx, parallel)
    room = max(
        0.0,
        float(pool_gib or 0.0) * GIB_TO_GB
        - UMA_RESERVE_GB
        - dspark_gb(ctx, enabled=dspark)
        - kv_cache_gb(ctx, parallel, fp8_kv=fp8_kv)
        - (VISION_GB if vision else 0.0),
    )
    held = min(need_total, room)
    on_ssd = max(0.0, need_total - held)
    return {
        "vram_gb": round(held, 2),
        "ram_gb": 0.0,
        "ssd_gb": round(on_ssd, 2),
        "host_kv_gb": 0.0,
        "kv_dtype": "fp8" if fp8_kv else "fp16",
        "kv_offload": "off",
        "full_vram": on_ssd <= 0.05,
    }


def place(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
          vision: bool = False, parallel: int = 1, dspark: bool = True, *,
          extra_vram_gib: float = 0.0, backend: str | None = None) -> dict:
    if unified:
        return compute_unified(build, ram_gib, ctx, vision, parallel, dspark)
    return compute(build, vram_gib or 0.0, ram_gib, ctx, vision, parallel, dspark,
                   extra_vram_gib=extra_vram_gib, backend=backend)


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
         vision: bool = False, parallel: int = 1, *, extra_vram_gib: float = 0.0, backend: str | None = None) -> bool:
    return place(build, vram_gib, ram_gib, unified, ctx, vision, parallel,
                 extra_vram_gib=extra_vram_gib, backend=backend)["ssd_gb"] <= 0.05


def recommend(vram_gib: float | None, ram_gib: float, unified: bool = False, ctx: int | None = None,
              vision: bool = False, *, preference: str = "quality", igpu: bool = False, arch: str | None = None,
              parallel: int = 1, extra_vram_gib: float = 0.0, backend: str | None = None) -> dict:
    """Automatic hardware auto-selection for Inkling-2-Mini (docs/10-GUI-TUI.md + Track 1E):
      - Radeon 780M (gfx1103) / Strix Point (gfx1150) iGPUs -> SQ-10 (coherent UMA/GTT, zero BIOS carve-out OOM)
      - >= 24 GiB discrete GPU (or combined multi-GPU VRAM) OR >= 32 GiB UMA -> SQ-20
      - 16 GiB discrete GPU (>= 15 GiB): SQ-15 in quality mode (with >= 20 GiB RAM) or SQ-10 in speed mode / < 20 GiB RAM
      - 24 GiB UMA (>= 22 GiB) -> SQ-15
      - 12 GiB discrete GPU (11..15 GiB): SQ-15 in quality mode (with >= 24 GiB RAM) or SQ-10 in speed mode / < 24 GiB RAM
      - 8 GiB discrete GPU (< 11 GiB) OR <= 16 GiB UMA -> SQ-10
    """
    v = float(vram_gib or 0.0) + max(0.0, float(extra_vram_gib or 0.0))
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
            best = "SQ-10" if (str(preference).lower() == "speed" or r < 20.0) else "SQ-15"
        elif v >= 11.0:
            best = "SQ-10" if (str(preference).lower() == "speed" or r < 24.0) else "SQ-15"
        else:
            best = "SQ-10"
    placed = place(best, vram_gib, ram_gib, use_unified, ctx, vision, parallel,
                   extra_vram_gib=extra_vram_gib, backend=backend)
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
        dl = BUILDS[b]["file_gb"] + (DSPARK_GGUF_GB if is_gguf_build(b) else 0.0)
        out.append({
            "build": b,
            "file": BUILDS[b]["file"],
            "format": BUILDS[b].get("format", "strata"),
            "download_gb": round(dl, 2),
            "footprint_gb": BUILDS[b]["footprint_gb"],
            "placement": placed,
            "mode": mode(placed),
        })
    return out


def disk_gb(build: str, vision: bool = False, dspark: bool = True) -> float:
    extra_dspark = DSPARK_GGUF_GB if (dspark and is_gguf_build(build)) else 0.0
    return round(BUILDS[build]["file_gb"] + extra_dspark + (VISION_GB if vision else 0.0), 2)


def build_of_model_name(model_name: str | None) -> str | None:
    low = str(model_name or "").lower()
    if not low.startswith(FAMILY_NAME + "-"):
        return None
    size = low[len(FAMILY_NAME) + 1:].upper()
    return size if size in BUILDS else None


def ssd_mode_flags(build: str, vram_gib: float | None, ram_gib: float, unified: bool = False,
                   ctx: int | None = None, vision: bool = False, parallel: int = 1) -> dict:
    """First-class GPU+SSD Only / CPU+SSD Only runner flags for Inkling-2-Mini (never blocks or overcommits RAM)."""
    placed = place(build, vram_gib, ram_gib, unified, ctx, vision, parallel)
    v = float(vram_gib or 0.0)
    r = float(ram_gib or 0.0)
    ssd = placed["ssd_gb"]
    if ssd <= 0.05:
        if not unified and v < 4.0:
            return {"tier": "cpu_in_memory", "runnable": True, "placement": placed, "flags": ["--cpu"]}
        return {"tier": "in_memory", "runnable": True, "placement": placed, "flags": []}
    if not unified and v >= 4.0 and (placed["ram_gb"] < 8.0 and ssd > placed["ram_gb"]):
        return {
            "tier": "gpu_ssd_only", "runnable": True, "placement": placed,
            "bounce_ring_gb": round(min(2.0, max(0.5, r * 0.15)), 1),
            "flags": ["--ssd-experts", "--ram-experts-gb", "0"],
        }
    if not unified and v < 4.0:
        lru_gb = round(max(1.0, min(placed["ram_gb"] if placed["ram_gb"] > 0 else max(1.0, r * 0.5), max(1.0, r * 0.5))), 1)
        return {
            "tier": "cpu_ssd_only", "runnable": True, "placement": placed,
            "flags": ["--cpu", "--ssd-experts", "--ram-experts-gb", f"{lru_gb:g}"],
        }
    return {
        "tier": "hybrid_ssd", "runnable": True, "placement": placed,
        "flags": ["--ssd-experts", "--ram-experts-gb", f"{max(0.0, placed['ram_gb']):g}"],
    }


def multi_gpu_flags(gpus: list[dict], *, windows: bool = False) -> list[str]:
    """Automatic multi-GPU flags for 2..4 discrete CUDA GPUs: uses --layer-split auto across 2..4 GPUs
    so both experts AND global/SWA KV span the full combined VRAM without duplicating trunk/KV (#6592, #6631)."""
    n = min(4, len(gpus or []))
    if n < 2:
        return []
    return ["--gpus", str(n), "--layer-split", "auto"]
