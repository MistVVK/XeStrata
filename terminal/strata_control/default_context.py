"""The conversation context a model starts with: 128K on every card, unless the person picks another (owner, 2026-11-01).

One rule for the app (apps/host/src/lib/default-context.ts says the same), the Terminal (`strata launch`, `setup`) and the MCP tools: 128K, or the
largest size the model offers when that is under 128K. The VRAM tiers that used to lower it on small
cards are gone; a card that 128K leaves little room on gets a warning, never a lower number (the engine says plainly if it cannot start).
A Mac keeps its unified-memory rule (strata_control/mac_fit.py), which the owner has not changed.
"""
from __future__ import annotations

from typing import Iterable

DEFAULT_CONTEXT = 131072


def pick(sizes: Iterable[int] | None = None) -> int:
    """128K, or the largest of `sizes` under it, or (when every size is longer) the smallest."""
    have = sorted(int(s) for s in (sizes or ()) if int(s) > 0)
    under = [s for s in have if s <= DEFAULT_CONTEXT]
    return under[-1] if under else (have[0] if have else DEFAULT_CONTEXT)
