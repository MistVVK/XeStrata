"""The one fit label of a model on this computer (owner, 2026-11-08): the model's files, WITHOUT the n-gram / PLE / Engram lookup table file (it stays on the SSD and
is not counted), against the graphics memory plus the RAM.  They fit: "fits".  They do not: "may run slower".

    size  = the download size minus the lookup table's file (`files_gb`): the owner's literal rule, the "Needs (RAM + VRAM)" column of the Terminal list
    room  = RAM + the card's memory (the largest card that counts; a chip that shares the RAM only its own carve-out)

No reserve is taken off either side.  Nothing here refuses anything; a model that "may run slower" can still be started, and a computer with no card or a Mac with
little memory gets the same two words.
"""

from __future__ import annotations

FITS = "fits"
SLOWER = "may run slower"

# The lookup table's file, by family (bytes from the pinned shard lists in setup.py; the Hub's LFS pointers):
#   qwen / swift / coder / unsloth (all but the three big files): the n-gram table, IQ4_NL, 28,800,138,432 bytes, a shard of its own
#   unsloth UD-Q5_K_XL, UD-Q6_K_XL, Q8_0: the Q8_0 table, shard 3, 54,400,261,312 bytes
#   nvfp4: the FP8 table `ple-fp8.gguf`, 51,200,246,176 bytes
#   nvfp4 NVFP4-QAD: its 102.2 GB download is NVIDIA's checkpoint, which this PC converts into the pack that runs; the files that run are the converted pack, the same
#     size as the other NVFP4 sets (128.0 - 51.2 = 76.8 GB), so that figure is used (owner's rule: the model files that run, minus the table)
#   gyro: the table is inside the one GGUF; the catalog states the rest (its `ram_gb`: 27.6 / 35 GB), the table is the difference
#   qwen36 (its own path) has none.  A family this module does not know counts its whole download.
_GB = 1e9
_TABLE_BYTES = {"qwen": 28_800_138_432, "swift": 28_800_138_432, "coder": 28_800_138_432, "unsloth": 28_800_138_432, "nvfp4": 51_200_246_176}
_BIG_UNSLOTH = ("UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0")
_BIG_TABLE_BYTES = 54_400_261_312
_PACK_GB = {("nvfp4", "NVFP4-QAD"): 76.8}        # sets whose download is not the files that run: the size of what runs


def table_gb(family: str | None, model: str | None, d: dict) -> float:
    """The size of the lookup table's file, in GB, of one catalog entry `d` (download_gb, ram_gb, mem_total)."""
    if d.get("mem_total") or family == "gyro":
        return max(0.0, float(d["download_gb"]) - float(d["ram_gb"]))
    if family == "unsloth" and model in _BIG_UNSLOTH:
        return _BIG_TABLE_BYTES / _GB
    return _TABLE_BYTES.get(str(family), 0) / _GB


def files_gb(family: str | None, model: str | None, d: dict) -> float:
    """The model's files without the lookup table, in GB (one decimal): the size the label and the "Needs" column use."""
    if (family, model) in _PACK_GB:
        return _PACK_GB[(family, model)]
    return round(float(d["download_gb"]) - table_gb(family, model, d), 1)


def card_room_gb(cards: list[dict] | None) -> float:
    """The graphics memory that counts for one model: the largest card, an integrated chip only for its own carve-out (its shared memory is the RAM)."""
    best = 0.0
    for c in cards or []:
        gb = float(c.get("dedicated_gb") or 0.0) if c.get("uma") else float(c.get("vram_gb") or 0.0)
        best = max(best, gb)
    return best


def label(size_gb: float, ram_gb: float | None, vram_gb: float | None) -> str:
    """"fits" when the files are no bigger than the RAM and the card's memory together, else "may run slower"."""
    return FITS if float(size_gb) <= float(ram_gb or 0.0) + float(vram_gb or 0.0) else SLOWER
