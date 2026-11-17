"""Which conversation context fits a graphics card for Qwen3.6-35B-A3B (the strata-q36 engine path, GitHub #5715): the one rule setup's context
menu, the start line in serve/server.py and the app's estimate (via the catalog, tools/strata_mcp.py) use.  The app's copy of the formula
(apps/host/src/lib/native-fit.ts `ownPathCacheGb` / `ownPathContextFits`) is pinned to this one by apps/host/tests/own-path-fit.test.mjs.

    need(ctx) GiB = BASE_GIB + the size's hybrid minimum + STATE_GIB + ctx x BYTES_PER_TOKEN / 2^30
    fits          = need(ctx) <= the card's memory as setup and the catalog read it (nvidia-smi memory.total, in GiB)

The parts:
  BYTES_PER_TOKEN 47,104 B: derived from the engine's allocations, not measured.  Each session keeps fp32 K and V of the 10 attention layers and the
      MTP block (11 x 2 KV heads x 256 x 2 x 4 B = 45,056 B, q36_engine.cu alloc_session; setup.py's kv_bytes_per_token, a parallel slot's figure)
      and the engine keeps one fp16 copy of K and V at the full context for the prompt attention (2 x 256 x 2 x 2 B = 2,048 B, Engine::init k16 / v16).
  STATE_GIB: the 30 GDN layers' recurrent state of one session (setup.py state_gb 0.07 GB).
  BASE_GIB 8.55: everything on the card before the context: the dense weights (4.40 GiB, the same for every size in the field logs), the engine's
      buffers at a 2048-token prompt chunk, the CUDA context and what Windows keeps of the card.  Derived from the #5715 log (RTX 5070 12 GB, Windows 11,
      engine 0.1.59, UD-Q4_K_XL at --max-context 65536: "0.45 GiB of VRAM free" after the engine's buffers and the session):
      11.94 - 0.45 - 65536 x 47,104 / 2^30 - 0.065 = 8.55.  ONE field log.  The other field logs came out at 7.2-7.65 (Linux #3303 #2747,
      Windows #3508 #3687), so this is the strict end: on those cards the rule can mark a smaller context than would start (12 GB: 32K where
      48-64K may start).  Nothing the product reads today (the OS, memory.total) tells the two apart.  NOT MEASURED HERE: a start on the
      owner's RTX 5070 (with the card's used memory logged before it) replaces this one number.
  HYBRID_MIN_GIB: what the hybrid mode allocates after the session even with no expert slot (the staging pair for the prompt layers and the CPU
      part's outputs, q36_engine.cu place_experts): UD-Q4_K_XL 1256 MiB (#5715 log), UD-Q6_K_XL 1760 MiB (#3687 log, RTX 3060 12 GB); Q8_0 taken as
      UD-Q6_K_XL (#2747 log: the same 1.59 GiB staging pair).

Nothing here changes a context: the result is a recommendation and a warning (recommend, never force).  A family or size without these figures (a
model of the update feed) gets None everywhere, and the callers keep what they did before.
"""

from __future__ import annotations

GIB = 2 ** 30
BYTES_PER_TOKEN = 11 * 2 * 256 * 2 * 4 + 2 * 256 * 2 * 2      # 45,056 + 2,048 = 47,104
STATE_GIB = 0.07e9 / GIB
BASE_GIB = 8.55                                                # the one figure a measured start replaces ("5070 calibration owed")
HYBRID_MIN_GIB = {"UD-Q4_K_XL": 1256 / 1024, "UD-Q6_K_XL": 1760 / 1024, "Q8_0": 1760 / 1024}
FAMILY_NAME = "qwen3.6-35b-a3b"                                # setup.py OWN_PATH_FAMILIES['qwen36']['name'], the config's model_name prefix
DEFAULT_CONTEXT = 131072                                       # 128K (owner, 2026-11-01): the mark never goes above it
MENU = [8192, 32768, 65536, 131072, 204800, 262144]           # setup's own-path context menu (setup.own_path_contexts(); a test keeps them equal)


def fixed_gib(model: str | None) -> float | None:
    """What the card holds before the context for this size (GiB), None for a size without figures."""
    h = HYBRID_MIN_GIB.get(str(model or ""))
    return None if h is None else BASE_GIB + h + STATE_GIB


def inputs(name: str | None, model: str | None) -> dict | None:
    """The formula's inputs for a family name ("qwen3.6-35b-a3b") and size, as the catalog row carries them to the app; None: not known."""
    f = fixed_gib(model) if str(name or "").lower() == FAMILY_NAME else None
    return None if f is None else {"bytes_per_token": BYTES_PER_TOKEN, "fixed_gb": round(f, 4)}


def from_model_name(model_name: str | None) -> dict | None:
    """The inputs from a config's model_name ("qwen3.6-35b-a3b-ud-q4_k_xl"), None for any other model."""
    low = str(model_name or "").lower()
    if not low.startswith(FAMILY_NAME + "-"):
        return None
    size = low[len(FAMILY_NAME) + 1:]
    return next((inputs(FAMILY_NAME, m) for m in HYBRID_MIN_GIB if m.lower() == size), None)


def cache_gib(ctx: int) -> float:
    """The conversation memory of one session at this context (GiB)."""
    return int(ctx) * BYTES_PER_TOKEN / GIB


def need_gib(fit: dict, ctx: int) -> float:
    return float(fit["fixed_gb"]) + int(ctx) * float(fit["bytes_per_token"]) / GIB


def fits(fit: dict, vram_gb: float, ctx: int) -> bool:
    return need_gib(fit, ctx) <= float(vram_gb)


def largest_fit(fit: dict | None, vram_gb: float | None, choices) -> int | None:
    """The largest of `choices` that fits the card; 0 when none does; None when it cannot be said (no figures, no card)."""
    if not fit or not vram_gb or float(vram_gb) <= 0:
        return None
    ok = [int(c) for c in choices if fits(fit, float(vram_gb), int(c))]
    return max(ok) if ok else 0


def recommended(fit: dict | None, vram_gb: float | None, choices, default: int = DEFAULT_CONTEXT) -> int:
    """The context to mark "(recommended)": the largest choice that fits the card, at most `default` (128K); the smallest choice when none fits;
    `default` when it cannot be said."""
    best = largest_fit(fit, vram_gb, choices)
    if best is None:
        return default
    if best == 0:
        return min(int(c) for c in choices)
    return min(best, default)


def label(ctx: int) -> str:
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else f"{ctx:,} tokens"


def warning(fit: dict | None, vram_gb: float | None, ctx: int, choices) -> str | None:
    """The sentence said before the start when `ctx` does not fit the card (owner's wording, 11-10/11-11), else None.  Nothing is said when no
    choice fits at all (the sentence needs a size that fits) or when it cannot be said."""
    best = largest_fit(fit, vram_gb, choices)
    if not best or fits(fit, float(vram_gb), int(ctx)):
        return None
    return (f"At {label(int(ctx))}, this model's conversation memory needs about {int(ctx) * float(fit['bytes_per_token']) / GIB:.1f} GB of graphics memory. With the model "
            f"itself that is more than this card's {float(vram_gb):.0f} GB, so it will probably not start. {label(best)} fits on this card: choose "
            f"{label(best)} or less.")
