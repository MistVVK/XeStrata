"""The Intel Arc card in the desktop app (the model setup of Strata Control, tools/strata_mcp.py and setup.py in their CONTROL mode).

The Terminal (`strata launch`) sets an Arc up in two steps (strata_cli/launch/runner.py adopt_prepared_files): `setup.py --prepare-pack` makes the pack
from the model's GGUF files, and strata_cli/launch/adopt.py writes the run config, with the engine named as the component (`component:strata-engine`).
Intel cards do not download models: the files must be on the computer. The app's install job did neither step: it ran setup.py's normal NVIDIA / AMD
path, which stops with "no NVIDIA GPU found (nvidia-smi did not answer)" (0.2.13 and 0.2.12, found by QAGUI-LIN on an Arc A750).

This module is the part both tools/strata_mcp.py (which engine the install is for; the card list) and setup.py (`--backend sycl` in CONTROL mode)
share. NVIDIA and AMD are never touched: an Arc is added to the card list only when this PC has no usable NVIDIA or AMD card, and the engine
Strata Control itself would pick (strata_control/engine_component.py detect_backend) is the Intel one.
"""

from __future__ import annotations

import sys

#: What the install says when an Arc card is chosen and the model's files are not on the computer (the Terminal's own sentence, "Intel cards do not
#: download models yet", in the app's words: the picker's entry is "Use your own GGUF…").
NO_DOWNLOAD = ("Strata can't download models for an Intel Arc card yet. Choose \"Use your own GGUF…\" and pick the folder that holds the model's "
               ".gguf files; Strata prepares them for this card.")

#: The families the run config can be written for (strata_cli/launch/adopt.py FAMILIES): the Qwen3.8-Flash-Next sizes, Swift, Coder, Gyro and Unsloth's sets.
def families() -> tuple:
    from strata_cli.launch import adopt
    return tuple(adopt.FAMILIES)


def can_set_up(family: str, family_def: dict | None = None) -> bool:
    """An Arc run config can be written for this family: one of families(), or a model of the signed feed (Qwen4 day: strata_control/feed_models.py)."""
    return family in families() or bool((family_def or {}).get("feed"))


def model_files_in(folder, family: str, family_def: dict | None = None) -> list:
    """The model's GGUF files in `folder` when the whole set is there (every shard; Gyro: its one file; a feed model: its shards or its one file), else []."""
    from pathlib import Path
    from strata_cli.launch import adopt
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return adopt.model_files(folder, family, family_def)


def supported_here(platform: str | None = None) -> bool:
    """Intel Arc engines ship for Linux x86_64 only (no Windows or Mac package in the feed)."""
    return (platform or sys.platform).startswith("linux")


def arc_cards(root: str = "/sys/class/drm") -> list[dict]:
    """The Intel Arc cards in this Linux computer, as the Terminal lists them (strata_cli/launch/catalog.py intel_arc_cards): vendor "intel", name,
    vram_gb (the card's own, from its PCI id), usable True."""
    from strata_cli.launch.catalog import intel_arc_cards
    return intel_arc_cards(root)


def engine_is_intel() -> bool:
    """Is the engine this computer's Strata Control would use the Intel one (SYCL)? The same decision the engine download makes (strata_control/
    engine_component.py detect_backend: STRATA_ENGINE_BACKEND, the saved choice, then the cards)."""
    from strata_control import engine_component as ec
    return ec.detect_backend()[0] == "sycl"


def cards_to_add(nvidia: list[dict], amd: list[dict], *, platform: str | None = None, cards=None, intel_engine=None) -> list[dict]:
    """The Arc cards to put in the hardware report beside `nvidia` and `amd`: none when either make has a usable card (those PCs are exactly what they
    were), none off Linux, none when the engine is not the Intel one. Never raises: a probe that fails adds nothing."""
    if not supported_here(platform):
        return []
    if any(g.get("usable") for g in nvidia) or any(g.get("usable") for g in amd):
        return []
    cards, intel_engine = cards or arc_cards, intel_engine or engine_is_intel
    try:
        found = cards()
        if not found or not intel_engine():
            return []
    except Exception:                                          # noqa: BLE001 - the hardware page never breaks over this
        return []
    return [dict(c) for c in found]
