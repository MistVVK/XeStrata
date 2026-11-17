"""A fixed expert cache beside a RAM budget on a PC whose graphics chip shares the RAM (Strix Halo).

The GPU's expert cache and the experts kept in RAM (--resident-budget-gib) draw on one pool there, and the RAM part is pinned, which costs
TWICE its size in that pool (measured on UD-Q4_K_XL, 128 GB Strix Halo, 2026-11-01: budget 12 / 6 / 2 GiB left 4,979 / 17,267 / 25,456 MiB
free at the first window, 2.00 x per MiB cut).  `--expert-cache auto` balances the two.  A smaller cache fixed by hand
(--expert-cache 10000 or 5000) leaves more experts for the RAM part, a 40 GiB budget then pins a bigger complement (34.5 GiB), and the next
allocation fails: the engine ended with exit code 1 after 41 s and no error text.

    pool demand = fixed + cache slots x slot size + 2 x (experts held in RAM)
    experts held in RAM = the smaller of the budget and the experts the cache does not hold

The constants below are the measurements of that box (fixed 16.5 GiB: dense weights, session, draft layer, prompt buffers at a 16K chunk).
It is an estimate for a recommendation, never a limit: the person's cache is kept, only the budget is lowered, with a line that says so.
"""
from __future__ import annotations

SLOTS_TOTAL = 24576           # the experts of the Flash-Next model (48 layers x 512): the most a cache can hold
FIXED_GIB = 16.5              # pool taken by everything but the cache and the RAM part (measured, UD-Q4_K_XL, chunk 16,384)
MARGIN_GIB = 1.0              # kept free beside the estimate
MIN_BUDGET_GIB = 4.0
MEASURED_BUDGET_GIB = 12      # starts at caches of 10,000 slots and more on that box (0 hangs, 0 faults in 28 starts)
GIB_PER_GB = 0.9313           # setup's "GB" of graphics memory seen by the GPU, in GiB


def slot_gib(expert_gb: float) -> float:
    """One expert's size in the cache (GiB), from the model's experts in GB (setup's MODELS arena_gb)."""
    return float(expert_gb) * 1e9 / SLOTS_TOTAL / 2 ** 30


def demand_gib(cache_slots: int, budget_gib: float, expert_gb: float, fixed_gib: float = FIXED_GIB) -> float:
    s = slot_gib(expert_gb)
    held = min(float(budget_gib), max(0, SLOTS_TOTAL - int(cache_slots)) * s)
    return fixed_gib + int(cache_slots) * s + 2.0 * held


def pool_gib(shared_vram_gb: float) -> float:
    """The pool the GPU can use, in GiB, from setup's figure for the chip (the carve-out plus the shared memory, in GB)."""
    return float(shared_vram_gb) * GIB_PER_GB


def safe_budget_gib(cache_slots: int, expert_gb: float, shared_vram_gb: float, fixed_gib: float = FIXED_GIB) -> float:
    """The largest RAM budget (GiB) the estimate fits beside a cache of `cache_slots`; not less than MIN_BUDGET_GIB."""
    s = slot_gib(expert_gb)
    room = pool_gib(shared_vram_gb) - MARGIN_GIB - fixed_gib - int(cache_slots) * s
    return max(MIN_BUDGET_GIB, float(int(room / 2.0)))


def too_much(cache_slots: int, budget_gib: float, expert_gb: float, shared_vram_gb: float) -> bool:
    return demand_gib(cache_slots, budget_gib, expert_gb) > pool_gib(shared_vram_gb) - MARGIN_GIB


def fixed_cache_and_budget(args) -> tuple[int, float] | None:
    """(slots, GiB) when the arguments fix a numeric --expert-cache beside a --resident-budget-gib, else None (auto is balanced by the engine)."""
    a = [str(x) for x in args or []]

    def value(flag: str):
        return a[a.index(flag) + 1] if flag in a[:-1] else None
    cache, budget = value("--expert-cache"), value("--resident-budget-gib")
    if cache is None or budget is None or not cache.isdigit():
        return None
    try:
        return int(cache), float(budget)
    except ValueError:
        return None


def stop_advice(args) -> str | None:
    """The plain next step for an engine that stopped while loading on a PC whose graphics chip shares the RAM, when the settings fix the
    expert cache beside a RAM budget (the exit had no error text of its own); None when they do not."""
    fixed = fixed_cache_and_budget(args)
    if fixed is None:
        return None
    cache, budget = fixed
    return (f"This PC's graphics chip shares its memory with the RAM. The settings fix the expert cache at {cache} slots and keep a {budget:g} GiB RAM "
            f"budget; the experts the smaller cache leaves out are pinned in that shared memory, and it can run out. Lower the budget "
            f"(--resident-budget-gib {MEASURED_BUDGET_GIB} in the model's settings) or set --expert-cache auto, then start again.")
