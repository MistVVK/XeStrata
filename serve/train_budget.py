# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/train_budget.py - where a LoRA training runs and how much VRAM it may take: the one place the policy
lives.  serve/lora_train.py's TrainPlace calls it every second and on a chat/image signal,
writes the control file the trainer reads ({"budget_mib", "save_state"}, the trainer's --vram-control) and keeps
the GPU ledger's "train" entry at the budget.  The trainer answers with its `vram <used> <peak_est> <min>
<running|paused>` line (train::Control).

Training is the lowest priority: it shrinks at a step's boundary when chat or an image job needs the card, pauses
(the state kept, the VRAM given back) when even the least form does not fit, and grows or resumes when they leave.
An image job is never made to wait for the training to end.  The helper card is not held closed for hours: the
training uses it while the chat is idle, or shares it with the helper cache.

Only the foundations are reused from the image/translation side: the GPU list and the text model's placement
(setup.image_gpu's candidate rule), the device reports, the VRAM ledger.  The pick itself is training's: it is
re-decided every tick, and moving cards (a state file + a restart with --resume) is weighed against shrinking and
pausing by measured values only - with no measurements it does not move.

The model table's figures are estimates until measured: open_mib is the transient room the open needs
(the weights read, the int8 quantization, the LoRA allocation) before a budget-0 start pauses; min_mib the least
the 1024-class step runs in (no resident block, no tape, the inputs on the host); peak_mib the whole form.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# model -> (open_mib, min_mib at the 1024 class, peak_mib of the whole form).  Estimates (unverified): the min
# follows train::Trainer::min_bytes (the fixed allocations + two stream slots + two row buffers + the arena's
# growth + 1.5 GiB of transients, 3 GiB more before the first step); the peaks are docs/LORA_TRAIN's measurements
# (Mage-Flow bf16 about 14.5 GiB) and the 2026-11-09 LOW-VRAM design's figures (Qwen-Image-2.1 int8 about 15 GB
# whole).  Measurements replace these estimates; the trainer's own vram line is authoritative once it runs.
MODELS = {
    ("qwen-image-2.1", "int8"): (3072, 7168, 15360),
    ("qwen-image-2.1", "bf16"): (3072, 13312, 21504),
    ("mage-flow", "int8"): (2048, 6144, 10240),
    ("mage-flow", "bf16"): (2048, 11264, 14848),
    ("magetrail", "int8"): (2048, 6144, 10240),
    ("magetrail", "bf16"): (2048, 11264, 14848),
    ("moss-image", "int8"): (2048, 6144, 12288),
    ("moss-image", "bf16"): (2048, 10240, 16384),
    ("anima", "bf16"): (1536, 5120, 9216),
}


def figures(model: str, base: str) -> tuple[int, int, int]:
    key = (model, base)
    if key not in MODELS:
        raise ValueError(f"no training figures for {model} ({base})")
    return MODELS[key]


def _setup():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import setup
    return setup


def candidates(cfg: dict, devices: list[dict] | None = None) -> list[dict]:
    """The cards a training may take, best first, by the image pick's own rule (setup.image_gpu): a discrete card
    no text engine takes (the most VRAM), else the text model's single later card (the helper), else the text
    cards (shared).  `devices`: the engine's report (xestrata-device --list-devices' objects) where setup's own
    list is not had (Windows); each {pci, total_bytes, free_bytes, integrated}.  Each candidate:
    {pci, kind: free|helper|share, total_mib}."""
    setup = _setup()
    text = {str(cfg.get("gpu_pci") or "").lower(), *(str(p).lower() for p in (cfg.get("split_pci") or []))} - {""}
    if devices is None:
        devices = [{"pci": g["pci"], "total_bytes": int(g["vram_gb"] * 1024**3), "free_bytes": None,
                    "integrated": g.get("integrated", False)} for g in setup.gpus()]
    by_pci = {str(d["pci"]).lower(): d for d in devices}
    primary = setup.image_gpu(types.SimpleNamespace(image_gpu="auto"), cfg)
    out: list[dict] = []

    def add(pci: str | None, kind: str):
        if not pci:
            return
        p = str(pci).lower()
        if any(c["pci"] == p for c in out):
            return
        d = by_pci.get(p, {})
        out.append({"pci": p, "kind": kind, "total_mib": int(d.get("total_bytes") or 0) >> 20})

    later = [str(p).lower() for p in (cfg.get("split_pci") or [])]
    helper = later[0] if len(later) == 1 and not cfg.get("replicas") else ""
    if primary and str(primary).lower() not in text:
        add(primary, "free")
    elif primary and str(primary).lower() == helper:
        add(primary, "helper")
    # the other cards no text engine takes, then the helper, then the text cards (the largest first)
    free = [d for d in devices if str(d["pci"]).lower() not in text and not d.get("integrated")]
    for d in sorted(free, key=lambda x: x.get("total_bytes") or 0, reverse=True):
        add(d["pci"], "free")
    add(helper, "helper")
    for p in sorted(text, key=lambda x: by_pci.get(x, {}).get("total_bytes") or 0, reverse=True):
        add(p, "share")
    return out


def _user() -> str:
    return str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))


def bench_path(pci: str) -> Path:
    return Path(tempfile.gettempdir()) / f"xestrata-bench-{_user()}-{re.sub(r'[^A-Za-z0-9_.-]', '_', pci.lower())}.json"


def load_bench(pci: str) -> dict | None:
    """The card's measured values (xestrata-device --bench's line), cached per device; None where not measured."""
    try:
        got = json.loads(bench_path(pci).read_text(encoding="utf-8"))
        return got if isinstance(got, dict) and "h2d_gbs" in got else None
    except (OSError, ValueError):
        return None


def run_bench(exe: str, pci: str, env: dict | None = None) -> dict | None:
    """Measure one card (XESTRATA_GPU_PCI selects it) and cache the line.  None where the probe could not run: the
    decider then assumes no fast path (it shrinks or pauses, never moves on an unmeasured card)."""
    e = dict(os.environ if env is None else env)
    e["XESTRATA_GPU_PCI"] = pci
    try:
        r = subprocess.run([exe, "--bench"], env=e, capture_output=True, text=True, timeout=120, check=False)
        got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        return None
    if isinstance(got, dict) and "h2d_gbs" in got:
        bench_path(pci).write_text(json.dumps(got), encoding="utf-8")
        return got
    return None


class Decider:
    """The training's placement and budget, re-decided every tick.  `st` (TrainPlace fills it):
      free: {pci: free_mib} the device reports now; chat_busy / chat_queued: the text engine's work;
      chat_eta_s: how long the chat's work is expected to last (None: unknown); image_job: an image job waiting
      or running on the card; image_need_mib: its estimate; helper_mib: the helper cache's size on the card;
      train: the trainer's last vram line {used_mib, peak_est_mib, min_mib, state} (None before the first);
      current_pci: the card the trainer runs on (None before the start); step_s / reload_s: the measured step
      time and the state-restore + weights-reload time (None: not measured).
    The action: {pci, budget_mib, save_state, reason, mode: start|run|shrunk|paused|move|refuse, return_helper}.
    TrainPlace writes the control file from it (and restarts with --resume on a move)."""

    def __init__(self, cfg: dict, model: str, base: str, devices: list[dict] | None = None):
        self.cfg = cfg
        self.model, self.base = model, base
        self.open_mib, self.min_mib, self.peak_mib = figures(model, base)
        self.cards = candidates(cfg, devices)

    def _margin(self, total_mib: int) -> int:
        from serve import vram_ledger
        return vram_ledger.device_margin_mib(total_mib)

    def _room(self, st: dict, card: dict) -> int:
        """What a training may take on the card now: the report's free, less the margin, less the helper cache
        where the chat needs it (the helper is not closed for the training while the chat runs)."""
        free = int((st.get("free") or {}).get(card["pci"], 0))
        room = free - self._margin(card["total_mib"])
        if card["kind"] == "helper" and (st.get("chat_busy") or st.get("chat_queued")):
            room -= int(st.get("helper_mib") or 0)
        return max(0, room)

    def tick(self, st: dict) -> dict:
        cur = st.get("current_pci")
        card = next((c for c in self.cards if c["pci"] == cur), None)
        train = st.get("train") or {}
        min_mib = int(train.get("min_mib") or self.min_mib)   # the trainer's own figure once it runs
        peak = int(train.get("peak_est_mib") or self.peak_mib)
        image = bool(st.get("image_job"))
        chat = bool(st.get("chat_busy") or st.get("chat_queued"))

        if card is None:   # the start: the first card the least form fits on (the open's room checked by TrainPlace)
            for c in self.cards:
                if self._room(st, c) >= min_mib:
                    return {"pci": c["pci"], "budget_mib": min(peak, self._room(st, c)), "save_state": False,
                            "reason": "", "mode": "start", "return_helper": False}
            return {"pci": None, "budget_mib": 0, "save_state": False, "reason": "no card has the least room",
                    "mode": "refuse", "return_helper": False}

        room = self._room(st, card)
        if image:   # an image job is never made to wait for the training: shrink to what is left, else pause
            left = room - int(st.get("image_need_mib") or 0)
            if left >= min_mib:
                return self._act(card, left, "shrunk", "image job", st)
            return self._pause_or_move(card, st, min_mib, "image job")
        if chat:   # the text model first: the training takes what is left beside it
            if room >= min_mib:
                return self._act(card, room, "shrunk" if room < peak else "run", "chat", st)
            return self._pause_or_move(card, st, min_mib, "chat")
        # idle: grow to the whole form (the helper card's cache stays closed only while the chat is idle)
        return self._act(card, min(peak, room) if room >= min_mib else 0,
                         "run" if room >= min_mib else "paused", "" if room >= min_mib else "no room", st)

    def _act(self, card: dict, budget: int, mode: str, reason: str, st: dict) -> dict:
        if budget <= 0:
            return {"pci": card["pci"], "budget_mib": 0, "save_state": False, "reason": reason or "no room",
                    "mode": "paused", "return_helper": card["kind"] == "helper"}
        return {"pci": card["pci"], "budget_mib": int(budget), "save_state": False, "reason": reason,
                "mode": mode, "return_helper": False}

    def _pause_or_move(self, card: dict, st: dict, min_mib: int, reason: str) -> dict:
        """The least form does not fit beside the work: pause (the state kept).  Move only where another card has
        the room AND the measured reload is cheaper than waiting the work out (no measurements: no move).  A
        pause that still leaves more than the work needs (the runtime's carry-over, the edge weights) escalates
        to a state file + exit: TrainPlace ends the process after `state saved` and starts it again later."""
        eta = st.get("chat_eta_s")
        reload_s = st.get("reload_s")
        if eta is not None and reload_s is not None and reload_s < eta:
            for c in self.cards:
                if c["pci"] != card["pci"] and self._room(st, c) >= min_mib:
                    return {"pci": c["pci"], "budget_mib": 0, "save_state": True, "reason": reason,
                            "mode": "move", "return_helper": card["kind"] == "helper"}
        return {"pci": card["pci"], "budget_mib": 0, "save_state": False, "reason": reason,
                "mode": "paused", "return_helper": card["kind"] == "helper"}
