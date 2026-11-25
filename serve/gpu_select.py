# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Select one GPU by PCI without opening contexts on the other GPUs in the child."""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from serve import envvars

FILTERS = ("ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "SYCL_DEVICE_ALLOWLIST", "CUDA_VISIBLE_DEVICES",
           "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ZE_AFFINITY_MASK")
BACKENDS = {"Level Zero": "level_zero", "CUDA": "cuda", "HIP": "hip"}
# Issue #41: a Level Zero runtime that fails as the GPUs are listed stops SYCL's whole listing (the CUDA and HIP GPUs
# with it), and only a selector set before the runtime starts keeps its adapter out.  setup writes this one into the
# run config's "env" when its own listing needed it (setup.py engine_tool).
L0_OFF = "!level_zero:*"
_SELECTORS: dict[tuple[str, int, str, str, str], list[dict]] = {}
_LOCK = threading.Lock()


def _list(exe: str, env: dict, flag: str) -> list[dict]:
    tool = str(Path(exe).parent / "xestrata-device")
    try:
        run = subprocess.run([tool, flag], env=env, capture_output=True, text=True, timeout=60, check=True)
        return [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise ValueError(f"cannot list GPU selectors with {tool}: {e}") from e


def without_level_zero(env: dict) -> bool:
    """The environment leaves Level Zero out (L0_OFF)."""
    return env.get("ONEAPI_DEVICE_SELECTOR") == L0_OFF


def selected_pci(exe: str, env: dict) -> str:
    """The first GPU the unmodified environment selects, when no PCI is named."""
    rows = _list(exe, env, "--list-devices")
    pci = str(rows[0].get("pci") or "").lower() if rows else ""
    if not pci or pci == "?":
        raise ValueError("the selected GPU did not report its PCI address")
    return pci


def unfiltered(env: dict) -> dict:
    """The environment with every card-selection filter and GPU_PCI taken out."""
    raw = dict(env)
    for key in FILTERS + (envvars.NEW + "GPU_PCI", envvars.OLD + "GPU_PCI"):
        raw.pop(key, None)
    return raw


def selectors(exe: str, env: dict) -> list[dict]:
    """Every GPU and its supported backends, from the unfiltered device list (cached per device tool and environment),
    even when `env` selects one card for an engine.  An empty list when the device tool is not here, so a caller that
    only asks about other cards can leave them alone."""
    raw = unfiltered(env)
    tool = Path(exe).parent / "xestrata-device"
    try:
        cache_key = (str(tool.resolve()), tool.stat().st_mtime_ns, raw.get("LD_LIBRARY_PATH", ""),
                     raw.get("CUDA_DEVICE_ORDER", ""), raw.get("ZE_FLAT_DEVICE_HIERARCHY", ""))
    except OSError:
        return []
    with _LOCK:
        if cache_key not in _SELECTORS:
            try:
                _SELECTORS[cache_key] = _list(exe, raw, "--list-device-selectors")
            except ValueError:
                return []
        return list(_SELECTORS[cache_key])


def scope(exe: str, env: dict, pci: str) -> dict:
    """The child's environment for `pci`, checked against an unfiltered SYCL list and a filtered one."""
    pci = str(pci).lower()
    raw = unfiltered(env)
    if without_level_zero(env):
        # the unfiltered list, which the selector indices come from, fails with Level Zero in it: the card by its PCI
        # address alone (the engine takes it first), the other backends' cards still listed to the child
        out = dict(raw, ONEAPI_DEVICE_SELECTOR=L0_OFF)
        envvars.put(out, "GPU_PCI", pci)
        check = _list(exe, out, "--list-devices")
        if not check or str(check[0].get("pci") or "").lower() != pci:
            raise ValueError(f"GPU {pci}: not the first GPU listed without Level Zero")
        return out
    rows = selectors(exe, env)
    if not rows:
        tool = Path(exe).parent / "xestrata-device"
        raise ValueError(f"cannot find the GPU device tool {tool}")
    found = [d for d in rows if str(d.get("pci") or "").lower() == pci]
    if len(found) != 1 or found[0].get("backend") not in BACKENDS:
        raise ValueError(f"GPU {pci}: cannot find one supported backend in the unfiltered device list")
    index = found[0].get("selector_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"GPU {pci}: no backend selector index")
    out = dict(raw, ONEAPI_DEVICE_SELECTOR=f"{BACKENDS[found[0]['backend']]}:{index}")
    envvars.put(out, "GPU_PCI", pci)
    check = _list(exe, out, "--list-devices")
    if len(check) != 1 or str(check[0].get("pci") or "").lower() != pci:
        raise ValueError(f"GPU {pci}: selector {out['ONEAPI_DEVICE_SELECTOR']} did not select only that GPU")
    return out
