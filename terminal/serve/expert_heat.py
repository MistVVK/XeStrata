"""#5723: the web page's Experts view - the rank the adaptive expert cache learned for every (layer, expert) pair, from the profile
the engine saves when the run config asks for it ("expert_profile_save": "<path>", see server.learned_profile_args).

The file is tools/make_profile.py's format: "STRP", then version, layers, experts per layer, slots and the pair count (5 little-endian
uint32), then the pairs in rank order (uint16 layer, uint16 expert). It holds ranks only - no flag says which pairs are in VRAM right
now - so the view shows ranks and when the file was saved, nothing more. It is read only when it changed (size and time), and a file
that is too large, short, or not one is refused instead of read.
"""
from __future__ import annotations

import os
import struct

MAX_LAYERS = 512
MAX_EXPERTS = 4096
MAX_PAIRS = 1 << 20
DEFAULT_EVERY_MIN = 10

_cache: dict = {}


def parse(blob: bytes) -> dict | None:
    """{layers, experts, ranked, rank}: `rank` is layers * experts entries, row by row, each the pair's place among the ranked pairs (0 = first)
    or -1 for a pair the file does not rank. A pair named twice keeps its first place, so the places run 0 .. ranked - 1. None when the bytes are not a profile."""
    if len(blob) < 24 or blob[:4] != b"STRP":
        return None
    _, nl, ne, _, n = struct.unpack_from("<5I", blob, 4)
    if not (0 < nl <= MAX_LAYERS and 0 < ne <= MAX_EXPERTS and n <= min(MAX_PAIRS, nl * ne)) or len(blob) < 24 + 4 * n:
        return None
    rank = [-1] * (nl * ne)
    ranked = 0
    for layer, expert in struct.iter_unpack("<HH", blob[24:24 + 4 * n]):
        if layer < nl and expert < ne and rank[layer * ne + expert] < 0:
            rank[layer * ne + expert] = ranked
            ranked += 1
    return {"layers": nl, "experts": ne, "ranked": ranked, "rank": rank}


def view(cfg: dict | None) -> tuple[int, dict]:
    """The answer to GET /api/expert-heat for a run config: 404 when it does not save a profile, else the saved ranks (or `saved_at`
    None before the engine saved the first one)."""
    save = (cfg or {}).get("expert_profile_save")
    if not isinstance(save, str) or not save.strip():
        return 404, {"error": {"message": "this model does not save its expert profile (expert_profile_save in its config)"}}
    every = (cfg or {}).get("expert_profile_save_every", DEFAULT_EVERY_MIN)
    if isinstance(every, bool) or not isinstance(every, (int, float)) or every < 0:
        every = DEFAULT_EVERY_MIN
    path = save if os.path.isabs(save) else os.path.join((cfg or {}).get("cwd") or ".", save)
    head = {"configured": True, "every_min": every, "saved_at": None}
    try:
        st = os.stat(path)
    except OSError:
        return 200, head
    if st.st_size > 24 + 4 * MAX_PAIRS:
        return 200, {**head, "saved_at": st.st_mtime, "error": "the saved profile is too large to show"}
    key = (path, st.st_size, st.st_mtime_ns)
    if _cache.get("key") != key:
        try:
            with open(path, "rb") as f:
                blob = f.read(24 + 4 * MAX_PAIRS)
        except OSError:
            return 200, head
        _cache.clear()
        _cache.update(key=key, parsed=parse(blob))
    parsed = _cache["parsed"]
    if parsed is None:
        return 200, {**head, "saved_at": st.st_mtime, "error": "the saved file is not an expert profile"}
    return 200, {**head, "saved_at": st.st_mtime, **parsed}
