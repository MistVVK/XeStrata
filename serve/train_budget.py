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

The model table's open_mib is the process room observed during open (the weights read, the int8 quantization,
the LoRA allocation) before a budget-0 start pauses; min_mib estimates the least form (no resident block, no tape,
the inputs on the host); need_mib estimates its process peak with a margin, and peak_mib estimates the whole form.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Rank 32: the table has reference image+text rows, text rows, open, least-form min and process peak, and the
# whole-form peak estimate (MiB). The measured classes use the greatest B70 / RTX 4070 / RX 9060 XT observations;
# they are samples, not hard bounds. MageTrail and the whole-form peaks are estimates.
_table = Path(__file__).with_name("train_vram_models.inc").read_text(encoding="utf-8")
_rows = re.findall(r'^TRAIN_VRAM\("([^"]+)", "([^"]+)", (\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\)$',
                   _table, re.MULTILINE)
if not _rows:
    raise ValueError("the training VRAM table has no rows")
MODELS = {(model, base): (int(op), int(minimum), int(whole))
          for model, base, size, _, _, op, minimum, _, whole in _rows if int(size) == 1024}
LEAST_FORMS = {(model, base, int(size)): (int(reference), int(text), int(minimum), int(peak))
               for model, base, size, reference, text, _, minimum, peak, _ in _rows}


def initial_vram(model: str, base: str, image_tokens: int, rank: int) -> tuple[int, int]:
    """Least-form min and process-peak budget for the largest cache shape. Samples are not a hard bound."""
    if image_tokens <= 0 or rank <= 0:
        raise ValueError("training shape and rank must be positive")
    sizes = sorted(size for m, b, size in LEAST_FORMS if (m, b) == (model, base))
    if not sizes:
        raise ValueError(f"no training figures for {model} ({base})")
    size = next((n for n in sizes if n * n // 256 + LEAST_FORMS[model, base, n][1] >= image_tokens), sizes[-1])
    reference, _, minimum, peak = LEAST_FORMS[model, base, size]
    numerator, denominator = max(reference, image_tokens) * max(32, rank), reference * 32
    minimum = (minimum * numerator + denominator - 1) // denominator
    peak = (peak * numerator + denominator - 1) // denominator
    return minimum, max(minimum, peak + max(128, (peak * 2 + 99) // 100))


def dataset_tokens(model: str, data: Path) -> int:
    """Largest image-token count plus largest text-row count from the training cache."""
    shapes: list[tuple[int, int]] = []
    text_rows = 0
    if model == "qwen-image-2.1":
        for line in data.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")   # a cache path can hold spaces; the lists from mf_pre.py are tab-separated
            if len(fields) < 2:
                continue
            with Path(fields[1]).open("rb") as f:
                rows, = struct.unpack("<f", f.read(4))
            text_rows = max(text_rows, int(rows))
            if fields[0] != "empty":
                with Path(fields[0]).open("rb") as f:
                    w, h = struct.unpack("<ff", f.read(8))
                shapes.append((int(w), int(h)))
    else:
        for line in (data / "manifest.tsv").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            shapes.append((int(fields[1]), int(fields[2])))
            text_rows = max(text_rows, int(fields[4]), int(fields[5]))
    if not shapes or any(w <= 0 or h <= 0 for w, h in shapes):
        raise ValueError("training cache has no valid shapes")
    return max(w * h for w, h in shapes) + text_rows


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
      free: {pci: free_mib} the device reports now, excluding the trainer; chat_busy / chat_queued: the text work;
      chat_eta_s: how long the chat's work is expected to last (None: unknown); image_job: an image job waiting
      or running on the card; image_need_mib: its estimate (None: unknown); device_free: the device's actual free
      MiB, without adding the trainer's usage; proc_alive: whether the training process is still alive;
      helper_mib: the helper cache's size on the card;
      train: the trainer's last vram line plus process_mib and process_source sampled beside free when possible;
        process_live says that sample was available (None before the first report);
      current_pci: the card the trainer runs on (None before the start); step_s / reload_s: the measured step
      time and the state-restore + weights-reload time (None: not measured).
    The action: {pci, budget_mib, save_state, reason, mode: start|run|shrunk|paused|move|refuse, return_helper,
    image_ready}. TrainPlace writes the control file, confirms process exit before yielding when a save was
    requested, and restarts with --resume only after the image job ends."""

    def __init__(self, cfg: dict, model: str, base: str, devices: list[dict] | None = None,
                 image_tokens: int = 4096, rank: int = 32):
        self.cfg = cfg
        self.model, self.base = model, base
        self.open_mib, _, self.peak_mib = figures(model, base)
        self.min_mib, self.need_mib = initial_vram(model, base, image_tokens, rank)
        self.cards = candidates(cfg, devices)
        self._last_action: dict = {}
        self._low_ticks = 0

    def _margin(self, total_mib: int) -> int:
        from serve import vram_ledger
        return vram_ledger.device_margin_mib(total_mib)

    def _room(self, st: dict, card: dict) -> int:
        """What a training may take on the card now: the report's free, less the margin, less the helper cache
        where the chat needs it (the helper is not closed for the training while the chat runs)."""
        free = int((st.get("free") or {}).get(card["pci"], 0))
        if card["pci"] == st.get("current_pci") and st.get("proc_alive"):
            train = st.get("train") or {}
            # fdinfo and NVML name this process; delta can include other processes and global is the whole device.
            held = train.get("process_mib") if train.get("process_source") in ("fdinfo", "nvml") else None
            free += int(held if held is not None else train.get("used_mib") or 0)
        room = free - self._margin(card["total_mib"])
        if card["kind"] == "helper" and (st.get("chat_busy") or st.get("chat_queued")):
            room -= int(st.get("helper_mib") or 0)
        return max(0, room)

    def tick(self, st: dict) -> dict:
        if st.get("image_job"):
            self._low_ticks = 0
            self._last_action = {}
            card = next((c for c in self.cards if c["pci"] == st.get("current_pci")), None)
            if card is None:
                return {"pci": None, "budget_mib": 0, "save_state": False, "reason": "unknown training card",
                        "mode": "refuse", "return_helper": False, "image_ready": False}
            alive = st.get("proc_alive", True)
            paused = (st.get("train") or {}).get("state") == "paused"
            need = st.get("image_need_mib")
            free = (st.get("device_free") or {}).get(card["pci"])
            fits = need is not None and free is not None and int(free) >= int(need) + self._margin(card["total_mib"])
            ready = (not alive and (need is None or fits)) or (alive and paused and fits)
            save = alive and paused and not fits
            return {"pci": card["pci"], "budget_mib": 0, "save_state": save, "reason": "image job",
                    "mode": "paused", "return_helper": card["kind"] == "helper", "image_ready": ready}
        action = {**self._tick(st), "image_ready": False}
        last = self._last_action
        train = st.get("train") or {}
        same = last.get("pci") == st.get("current_pci") and st.get("proc_alive")
        priority = st.get("chat_busy") or st.get("chat_queued")
        budget = int(action.get("budget_mib") or 0)
        previous = int(last.get("budget_mib") or 0)
        if same and previous > 0 and not priority and not train.get("process_live"):
            if not train.get("state"):
                # The open is still allocating and has not emitted its first vram line. Its boundary enforces
                # the initial budget; do not revoke it because those allocations reduced the device's free count.
                action = {**last, "reason": "waiting for the first VRAM report"}
            elif budget < previous:
                self._low_ticks += 1
                if self._low_ticks < 2:
                    action = {**last, "reason": "confirming available GPU memory"}
                else:
                    self._low_ticks = 0
            else:
                self._low_ticks = 0
        else:
            self._low_ticks = 0
        self._last_action = action
        return action

    def _tick(self, st: dict) -> dict:
        cur = st.get("current_pci")
        card = next((c for c in self.cards if c["pci"] == cur), None)
        train = st.get("train") or {}
        need_mib = int(train.get("need_mib") or self.need_mib)
        # The trainer's estimate follows its current plan, not the whole form; using it here creates feedback.
        peak = max(self.peak_mib, need_mib)
        chat = bool(st.get("chat_busy") or st.get("chat_queued"))

        if card is None:   # the start: the first card the least form fits on (the open's room checked by TrainPlace)
            for c in self.cards:
                if self._room(st, c) >= need_mib:
                    return {"pci": c["pci"], "budget_mib": min(peak, self._room(st, c)), "save_state": False,
                            "reason": "", "mode": "start", "return_helper": False}
            room = max((self._room(st, c) for c in self.cards), default=0)
            reason = f"not enough free GPU memory: need {need_mib} MiB, available {room} MiB after reserve"
            return {"pci": None, "budget_mib": 0, "save_state": False, "reason": reason,
                    "mode": "refuse", "return_helper": False}

        room = self._room(st, card)
        if chat:   # the text model first: the training takes what is left beside it
            if room >= need_mib:
                return self._act(card, room, "shrunk" if room < peak else "run", "chat", st)
            return self._pause_or_move(card, st, need_mib, "chat")
        # idle: grow to the whole form (the helper card's cache stays closed only while the chat is idle)
        return self._act(card, min(peak, room) if room >= need_mib else 0,
                         "run" if room >= need_mib else "paused", "" if room >= need_mib else "no room", st)

    def _act(self, card: dict, budget: int, mode: str, reason: str, st: dict) -> dict:
        if budget <= 0:
            return {"pci": card["pci"], "budget_mib": 0, "save_state": False, "reason": reason or "no room",
                    "mode": "paused", "return_helper": card["kind"] == "helper"}
        return {"pci": card["pci"], "budget_mib": int(budget), "save_state": False, "reason": reason,
                "mode": mode, "return_helper": False}

    def _pause_or_move(self, card: dict, st: dict, need_mib: int, reason: str) -> dict:
        """The least form does not fit beside the work: pause (the state kept).  Move only where another card has
        the room AND the measured reload is cheaper than waiting the work out (no measurements: no move).
        Image jobs take the separate image_ready path, which can escalate a pause to a state file and exit."""
        eta = st.get("chat_eta_s")
        reload_s = st.get("reload_s")
        if eta is not None and reload_s is not None and reload_s < eta:
            for c in self.cards:
                if c["pci"] != card["pci"] and self._room(st, c) >= need_mib:
                    return {"pci": c["pci"], "budget_mib": 0, "save_state": True, "reason": reason,
                            "mode": "move", "return_helper": card["kind"] == "helper"}
        return {"pci": card["pci"], "budget_mib": 0, "save_state": False, "reason": reason,
                "mode": "paused", "return_helper": card["kind"] == "helper"}
