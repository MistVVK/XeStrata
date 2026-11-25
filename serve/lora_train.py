# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/lora_train.py - LoRA training from the browser: the chat's train_lora tool opens a panel (serve/web/app.js),
the panel uploads images, captions them (tools/image/lora_caption.py, the chat's model), and starts a job here.

A job is a folder under the data folder's lora-jobs/ (record.json, images/, export/ or cache/, out/, samples/), one
job run at a time by a worker thread (as serve/videos.py's VideoJobs).  Its stages: preprocess (tools/image/mf_pre.py
and mf_pre_export.py, the VAE and text reader on the GPU in the image jobs' turn) -> train (the model's trainer as a
subprocess) -> samples (a picture of every saved version, the same seed and prompts, in the image jobs' turn) ->
the user picks a version and registers it (serve/loras.py's add, into the model's LoRA folder).  The state survives
a server restart (interrupted -> resume from the first stage not done); a cancel stops the subprocess and keeps the
versions saved so far.

The trainers' arguments differ by model; TRAINERS below is the one place that maps them (the UI's steps are optimizer
updates: Mage-Flow's --steps counts micro-steps, so they are multiplied by its accum).  Versions are found by scanning
the output folder (the mf trainer prints no save line) and kept only once serve/loras.py's read_header reads them (a
cancel mid-save leaves a broken file, not a version).

The GPU: training is the lowest priority.  The decisions (the card, the budget, shrink / pause / move) are in
serve/train_budget.py; TrainPlace here calls its Decider at
every step boundary's cadence (each second and on a chat/image signal), writes the control file, charges the ledger
and shows the state.  Chat and image jobs come first: training shrinks or pauses at a micro-step boundary (its LoRA
and optimizer state kept) and expands or resumes when they leave.  Image jobs are never kept waiting for the
training's end (images.gpu_turn's image_begin waits at most a micro-step's seconds plus the release).  The helper
card is never closed for hours.

The budget protocol: the trainer takes --vram-budget (0 = start paused; the open still
needs open_mib of temporary room, checked before the start), --vram-control <file> ({"budget_mib", "save_state"},
atomic write, read at a step boundary), --state <file> and --resume <file>.  It reports on stdout:
`vram <used_mib> <peak_est_mib> <min_mib> <running|paused> [key=value ...]` (the trailing fields are ignored here)
and `state saved <path>` once the state is written.  save_state takes effect at an optimizer update's boundary (mf's
accum: up to accum micro-steps late); a plain budget 0 pauses at a micro-step boundary.  When a pause's measured used
still exceeds what the other work needs (Level Zero keeps allocations while the process lives), the decider asks
save_state and the process is ended after `state saved` (its exit returns every allocation), then started again with
--resume when the room is back - on another card when the decider chose a move.
"""
from __future__ import annotations

import contextlib
import filecmp
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import unicodedata
from pathlib import Path
from typing import Any

from serve import envvars
from serve import loras as lora_api
from serve.model_terms import _qimage
from serve.train_defaults import CAPTION_DROPOUT

MAX_IMAGES = 200                  # the most training images one job takes
MAX_IMAGE_BYTES = 32 << 20        # the largest one image
MAX_DATASET_BYTES = 2 << 30       # the largest whole dataset
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")   # tools/image/mf_pre.py's IMG_EXT
MAX_CAPTION = 2000                # one line, <trigger>, <description>
STEPS_MAX = 100000
RANK_MAX = 256
JOB_DISK_MARGIN = 1 << 30         # free disk an upload leaves (videos.py's JOB_DISK_MARGIN)
KINDS = ("character", "style", "other")
FAMILIES = ("qwen-image", "mage-flow", "moss-image", "anima")   # the families with a trainer
# The image job's wait for training to yield: one micro-step plus the release. Mage-Flow's 1024-class micro-step
# was measured at 8.9 s; the wait leaves room for the slowest known micro-step. The ledger still guards VRAM.
IMAGE_YIELD_WAIT_S = 60.0
OPEN_WAIT_S = 600.0               # the longest wait for open_mib of temporary room before a trainer starts
DEVICES_CACHE_S = 5.0             # xestrata-device --list-devices is a subprocess; not one every tick

# The one place of the browser's training defaults. Caption dropout is off by default;
# Qwen-Image's print-style comparison favored 0 over 0.2 for that dataset. Other models can set it explicitly.
# peak_mib: the training's peak VRAM at the 1024 class
# (Mage-Flow measured, docs/LORA_TRAIN; Qwen-Image the user's number; MOSS-Image and Anima estimates, unverified).
# The least budget and the open's room are train_budget.py's (min_mib / open_mib).
PRESETS: dict[str, dict] = {
    "qwen-image": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": CAPTION_DROPOUT,
                   "base": "bf16", "peak_mib": 15360},
    "mage-flow": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 12345, "dropout": CAPTION_DROPOUT,
                  "accum": 1, "base": "int8", "method": "lora", "peak_mib": 14848},
    "moss-image": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": CAPTION_DROPOUT,
                   "base": "int8", "peak_mib": 16384},
    "anima": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": CAPTION_DROPOUT,
              "peak_mib": 12288},
}


def preset(family: str, kind: str) -> dict:
    result = PRESETS[family].copy()
    # One Qwen-Image-2.1 character, two seeds: the middle blocks preserve scenes better than all blocks.
    if family == "qwen-image" and kind == "character":
        result.update(steps=1500, train_blocks="8-15")
    return result


# The sample pictures of every version (the same seed and prompts across versions): a subject the data does not have
# (docs/LORA_TRAIN's check: a character in new clothes and scene, a style on new content).  The panel edits them.
SAMPLE_PROMPTS = {
    "character": ["{trigger}, a girl in a red coat on a snowy street, winter evening",
                  "{trigger}, a girl reading a book under a cherry tree, spring morning"],
    "style": ["{trigger}, a fox sitting on a stone bridge over a river",
              "{trigger}, a lighthouse on a cliff above a stormy sea"],
    "other": ["{trigger}, a quiet street with a cat on a windowsill",
              "{trigger}, a bowl of fruit on a wooden table by a window"],
}
SAMPLE_SEED = 7                   # fixed, so the versions compare
SAMPLE_SIZE = (1024, 1024)

# The trainers' entry points. elastic: the trainer takes --vram-budget/--vram-control.
# ckpt: how a version's file name carries its
# step ("_N" = <prefix>_<step>.safetensors, ".N" = <prefix>.<step>.safetensors and <prefix>.final.safetensors).
TRAINERS: dict[str, dict] = {
    "qwen-image": {"exe": "xestrata-image-qi-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "qi",
                   "elastic": True},
    "mage-flow": {"exe": "xestrata-image-mf-train", "save_flag": "--checkpoint", "ckpt": ".N", "cache": "mf",
                  "elastic": True},
    "moss-image": {"exe": "xestrata-image-moss-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "ms",
                   "elastic": True},
    "anima": {"exe": "xestrata-image-anima-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "an",
              "elastic": True},
}


# Setup choices and served names differ for Qwen-Image.  Both names refer to the same training base.
_qi = _qimage()
DISTILLED = {name: (_qi.IMAGE_MODELS[base][0], base)
             for source, base in _qi.TRAINING_BASE.items()
             for name in {source, _qi.IMAGE_MODELS.get(source, (source,))[0]}}
SETUP_ALIASES = {choice: values[0] for choice, values in _qi.IMAGE_MODELS.items() if choice != values[0]}


def training_model(model: str) -> tuple[str, str | None]:
    """The served training model and setup choice if a distilled model was requested."""
    if model in DISTILLED:
        return DISTILLED[model]
    return SETUP_ALIASES.get(model, model), None


def setup_command(svc, base_choice: str) -> str:
    """Keep the configured image choices while adding the requested training base."""
    images = getattr(svc, "images", None)
    choices = []
    for gen in list(images.gens.values()) if images is not None else []:
        if gen.cfg.get("training_base_only"):
            continue
        choice = str(gen.cfg.get("setup_choice") or next(
            (k for k, v in _qi.IMAGE_MODELS.items() if v[0] == gen.model), gen.model))
        if choice not in choices:
            choices.append(choice)
    target = training_model(base_choice)[0]
    if not any(training_model(c.split(":", 1)[0])[0] == target for c in choices):
        choices.append(base_choice)
    return f"setup --images {','.join(choices)} --image-lora-training"


class TrainError(ValueError):
    """A training request that cannot run: the message says why (the panel shows it)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class _Cancelled(Exception):
    """The job's cancel reached a stage."""


def budget_mod():
    """The training budget decisions (serve/train_budget.py: figures, Decider).
    The tests replace this with a fake of the same shape."""
    from serve import train_budget
    return train_budget


def data_root() -> Path:
    """The data folder setup uses (its settings' data_dir; the skills' default beside it)."""
    from serve import places
    try:
        data = json.loads(places.settings_path().read_text(encoding="utf-8")).get("data_dir")
    except (OSError, ValueError):
        data = None
    if data:
        return Path(data)
    from serve import skills
    return skills.default_dir().parent


def trainer_family(gen) -> str | None:
    """The trainer family of an image model (MageTrail is mage-flow's, its caches the mt tag), None when none."""
    return gen.family if gen.family in FAMILIES else None


def cache_tag(gen) -> str:
    """mf_pre.py's --cache tag: MageTrail's latents are the Flux2 VAE's (mt), Mage-Flow's the Mage-VAE's (mf)."""
    if gen.family == "mage-flow" and str(gen.cfg.get("model_name") or "").startswith("magetrail"):
        return "mt"
    return TRAINERS[gen.family]["cache"]


def trainer_exe(gen) -> Path:
    return Path(gen.cfg["exe"]).parent / TRAINERS[gen.family]["exe"]


def train_base(gen) -> tuple[Path | None, str]:
    """The DiT file the trainer reads -> (it, "") or (None, why it is not here).  Qwen-Image-2.1 trains on the F16
    base (tools/image/qimage.py's base-F16), not inference's quantized file."""
    dit = Path(gen.cfg["diffusion_model"])
    if gen.family == "qwen-image":
        f16 = dit.parent / "qwen-image-2.1-F16.gguf"
        if not f16.is_file():
            return None, (f"{f16.name} is not beside the DiT: the training reads the F16 base "
                          "(the training base is missing)")
        return f16, ""
    if gen.family == "moss-image":
        parts = [dit / "config.json", *(dit / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
                                        for i in range(1, 4))]
        ready = dit.is_dir() and all(p.is_file() for p in parts)
    else:
        ready = dit.is_file()
    if not ready:
        return None, f"the DiT is not complete: {dit}"
    return dit, ""


def model_rows(svc) -> list[dict]:
    """GET /v1/lora/models: the installed image models a trainer exists for, each with its preset and whether it can
    train here (the trainer binary and the base file present)."""
    rows = []
    images = getattr(svc, "images", None)
    for gen in list(images.gens.values()) if images is not None else []:
        fam = trainer_family(gen)
        if fam is None or gen.model in DISTILLED:
            continue
        exe = trainer_exe(gen)
        dit, why = train_base(gen)
        ok = exe.is_file() and dit is not None
        if dit is None:
            choice = str(gen.cfg.get("setup_choice") or next(
                (k for k, v in _qi.IMAGE_MODELS.items() if v[0] == gen.model), gen.model)).split(":", 1)[0]
            why += f"; run {setup_command(svc, choice)} and restart the server"
        reasons = ([f"the trainer is not built: {exe.name}"] if not exe.is_file() else []) + ([why] if why else [])
        rows.append({"model": gen.model, "family": fam, "name": f"{lora_api.FAMILY_NAMES.get(fam, fam)} — {gen.model}",
                     "presets": {kind: preset(fam, kind) for kind in KINDS},
                     "sample_prompts": SAMPLE_PROMPTS, "sample_seed": SAMPLE_SEED,
                     "can_train": ok, "why": "; ".join(reasons)})
    # A base not configured on the server cannot be used, even if its derivative is installed.
    present = {r["model"] for r in rows}
    for gen in list(images.gens.values()) if images is not None else []:
        if gen.model not in DISTILLED:
            continue
        base, choice = DISTILLED[gen.model]
        if base in present:
            continue
        fam = trainer_family(gen)
        if fam is None:
            continue
        rows.append({"model": base, "family": fam, "name": base, "presets": {kind: preset(fam, kind) for kind in KINDS},
                     "sample_prompts": SAMPLE_PROMPTS, "sample_seed": SAMPLE_SEED, "can_train": False,
                     "why": f"install the base with {setup_command(svc, choice)} and restart the server"})
        present.add(base)
    for row in rows:
        row["requested_models"] = [name for name, (base, _) in DISTILLED.items() if base == row["model"]]
    return rows


# ------------------------------------------------------------------------------------------------- a request's fields
def check_name(name: str) -> str:
    stem = Path(str(name or "").replace("\\", "/")).name.strip()
    stem = re.sub(r"[^A-Za-z0-9._ +-]+", "_", stem).strip(" ._")[:120]
    if not stem:
        raise TrainError("name: the LoRA needs a name (letters, digits, spaces, . _ + -)")
    return stem


def check_trigger(trigger: str) -> str:
    raw = str(trigger or "")
    t = raw.strip()
    if not t or len(t) > 120 or any(unicodedata.category(c) == "Cc" for c in raw) or "," in t:
        raise TrainError("trigger: one word or phrase, up to 120 characters, no comma or line break")
    return t


def check_settings(fields: dict, fam: str, kind: str = "other") -> dict:
    """The start's settings over the preset: n (steps), k (save_every), rank, lr, seed, and the advanced ones."""
    p = preset(fam, kind)

    def num(key, lo, hi, cast, default):
        v = fields.get(key)
        if v in (None, ""):
            return default
        try:
            x = cast(v)
        except (TypeError, ValueError):
            raise TrainError(f"{key}: a number") from None
        if not lo <= x <= hi:
            raise TrainError(f"{key}: {lo:g} to {hi:g}")
        return x

    n = num("steps", 1, STEPS_MAX, int, p["steps"])
    k = num("save_every", 1, n, int, min(p["save_every"], n))
    size = num("max_size", 512, 1024, int, 1024)
    if size not in (512, 768, 1024):
        raise TrainError("max_size: choose 512, 768 or 1024")
    out = {"steps": n, "save_every": k, "max_size": size,
           "rank": num("rank", 1, RANK_MAX, int, p["rank"]),
           "lr": num("lr", 1e-6, 1e-2, float, p["lr"]),
           "seed": num("seed", 0, 2**31 - 1, int, p["seed"]),
           "dropout": num("dropout", 0.0, 1.0, float, p["dropout"])}
    if fam == "mage-flow":
        out["accum"] = num("accum", 1, 64, int, p["accum"])
        method = str(fields.get("method") or p["method"])
        if method not in ("lora", "lokr", "dora"):
            raise TrainError("method: lora, lokr or dora")
        out["method"] = method
    if fam in ("qwen-image", "mage-flow", "moss-image"):
        base = str(fields.get("base") or p["base"])
        if base not in ("bf16", "int8"):
            raise TrainError("base: bf16 or int8")
        out["base"] = base
    for key in ("train_blocks", "only_if_contains", "ignore_if_contains"):
        v = str(fields.get(key, p.get(key, "")) or "").strip()
        if len(v) > 400 or "\n" in v:
            raise TrainError(f"{key}: one line, up to 400 characters")
        out[key] = v
    prompts = fields.get("sample_prompts")
    if prompts in (None, ""):
        prompts = list(SAMPLE_PROMPTS.get(kind, SAMPLE_PROMPTS["other"]))
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 4 or \
            any(not isinstance(t, str) or not t.strip() or len(t) > 1000 for t in prompts):
        raise TrainError("sample_prompts: 1 to 4 lines, up to 1000 characters each")
    out["sample_prompts"] = [t.strip() for t in prompts]
    out["sample_seed"] = num("sample_seed", 0, 2**31 - 1, int, SAMPLE_SEED)
    return out


def check_caption(text: str) -> str:
    t = str(text or "").strip()
    if "\n" in t:
        t = " ".join(t.split())
    if len(t) > MAX_CAPTION:
        raise TrainError(f"caption: one line, up to {MAX_CAPTION} characters")
    return t


# ------------------------------------------------------------------------------------------------- the GPU's reports
_DEV_CACHE: dict[str, tuple[float, list[dict]]] = {}


def devices(svc, *, fresh: bool = False) -> list[dict]:
    """The GPUs as xestrata-device --list-devices reports them (one JSON a device: ordinal, pci, total_bytes,
    free_bytes, integrated, compute_units, arch).  [] when the program is not here.  Cached DEVICES_CACHE_S."""
    images = getattr(svc, "images", None)
    if images is None:
        return []
    exe = str(Path(images.default.cfg["exe"]).parent / "xestrata-device")
    got = _DEV_CACHE.get(exe)
    if not fresh and got is not None and time.monotonic() - got[0] < DEVICES_CACHE_S:
        return got[1]
    out: list[dict] = []
    env = dict(os.environ)
    from serve import gpu_select
    if gpu_select.without_level_zero(images.default.engine.env or {}):   # issue #41: the run config's selector
        env["ONEAPI_DEVICE_SELECTOR"] = gpu_select.L0_OFF
    if Path(exe).is_file():
        try:
            r = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=60, env=env)
            for line in r.stdout.splitlines():
                with contextlib.suppress(ValueError):
                    d = json.loads(line)
                    if isinstance(d, dict) and d.get("pci"):
                        out.append(d)
        except (OSError, subprocess.TimeoutExpired):
            pass
    _DEV_CACHE[exe] = (time.monotonic(), out)
    return out


def run_config(svc) -> dict:
    if getattr(svc, "config_path", None):
        with contextlib.suppress(OSError, ValueError):
            from serve import runconfig
            return runconfig.load(svc.config_path)
    return {}


def text_placement(svc) -> tuple[set[str], str, dict[str, int]]:
    """(the cards the text model takes, its helper card, the MiB it holds on each) from the run config and the
    ledgers - the same reports the image jobs' placement reads (the foundation train_budget's candidates reuse)."""
    cfg = run_config(svc)
    env = getattr(svc.engine, "env", None) or dict(os.environ)
    pcis = {str(envvars.get("GPU_PCI", env=env) or cfg.get("gpu_pci") or "").lower()}
    pcis |= {str(p).lower() for p in cfg.get("split_pci") or []}
    for r in cfg.get("replicas") or []:
        pcis |= {str(p).lower() for p in ((r.get("gpus") if isinstance(r, dict) else r) or [])}
    ss = cfg.get("split_serve")
    if isinstance(ss, dict):
        pcis |= {str(ss.get(k) or "").lower() for k in ("prefill_gpu", "decode_gpu")}
    pcis.discard("")
    from serve.server import helper_card_for
    helper = helper_card_for(cfg, str((cfg.get("images") or {}).get("gpu_pci") or ""))
    held: dict[str, int] = {}
    from serve import images as img
    from serve.vram_ledger import Ledger
    for pci in pcis | ({helper} if helper else set()):
        with contextlib.suppress(OSError, ValueError):
            led = Ledger(img.gpu_key(dict(env, XESTRATA_GPU_PCI=pci)))
            for e in led._read().get("entries", []):
                if str(e.get("purpose", "")) in ("text", "helper"):
                    held[pci] = held.get(pci, 0) + int(e.get("mib", 0))
    return pcis, helper, held


def card_env(svc, pci: str) -> dict:
    """The trainer's environment on `pci`, with only that device exposed to SYCL."""
    from serve import gpu_select
    env = dict(os.environ)
    env.update(svc.images.default.engine.env or {})
    return gpu_select.scope(svc.images.default.cfg["exe"], env, pci)


def free_mib(svc, pci: str) -> int:
    d = next((d for d in devices(svc) if str(d["pci"]).lower() == pci.lower()), None)
    return int(d.get("free_bytes") or 0) >> 20 if d else 0


# ------------------------------------------------------------------------------------------------- the jobs
def _safe_id(jid: str) -> bool:
    return bool(re.fullmatch(r"lora_[0-9a-f]{32}", jid or ""))


class TrainJobs:
    """The jobs' records and files under root/<id>/, one job run at a time by a worker thread (VideoJobs' form).
    The state survives a restart: a job caught running becomes interrupted (its orphan trainer stopped), resume
    re-enters the first stage not done."""

    def __init__(self, root: Path, svc=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.svc = svc
        self.lock = threading.Lock()
        self.record_lock = threading.Lock()
        self.mutation_lock = threading.Lock()   # serialize register, delete and continue
        self.queue: list[str] = []
        self.cancels: dict[str, threading.Event] = {}
        self.worker: threading.Thread | None = None
        self.current: str | None = None          # the job running now
        self.place: dict | None = None           # its place now (the panel and /v1/status show it)
        self.image_waiting = 0                   # image jobs waiting for the training's yield (gpu_turn)
        self.image_needs: dict[str, int | None] = {}
        self.image_seq = 0                      # a decision is valid only for the needs it saw
        self.wake = threading.Event()            # the policy tick's immediate wake (a chat/image signal)
        self.yielded = threading.Event()         # the training made room for the image jobs
        self._proc: subprocess.Popen | None = None
        self._train_state: dict = {}             # the trainer's last vram line
        self._step_clock: tuple[int, float] | None = None
        self._state_saved = threading.Event()    # the trainer wrote its state (state saved line)
        self._decider: Any = None
        self._log: Any = None
        self._reader: threading.Thread | None = None
        for d in self.root.iterdir():
            rec = self._read(d.name)
            if rec and rec.get("status") in ("queued", "preprocessing", "training", "sampling", "captioning"):
                if self._owner_alive(rec):
                    continue
                self._stop_orphan(rec)
                if rec.get("settings"):
                    rec.update(status="interrupted", error={"code": "server_restarted",
                                                            "message": "the server stopped before the job ended"})
                else:
                    rec.update(status="draft", error=None)
                self._write(rec)

    def close(self) -> None:
        """Stop this server's queued and active work before its image engines close."""
        with self.lock:
            ids = list(self.cancels)
            worker = self.worker
        for jid in ids:
            self.cancel(jid)
        if worker is not None and worker is not threading.current_thread():
            rec = self.get(self.current) if self.current else None
            worker.join(self._stop_wait(rec or {}) * 2 + 10)
            if worker.is_alive() and rec is not None:
                self._stop_orphan(rec)
                worker.join(self._stop_wait(rec) + 5)

    @staticmethod
    def _owner_alive(rec: dict) -> bool:
        from serve.vram_ledger import process_start
        pid, start = rec.get("owner_pid"), rec.get("owner_start")
        if not pid or start is None:
            return False
        current = process_start(int(pid))
        return current is not None and abs(current - float(start)) < 1e-3

    @staticmethod
    def _claim(rec: dict) -> None:
        from serve.vram_ledger import process_start
        pid = os.getpid()
        start = process_start(pid)
        if start is None:
            raise TrainError("cannot identify this server process", 503)
        rec.update(owner_pid=pid, owner_start=start)

    # ---- the record
    def dir_of(self, jid: str) -> Path:
        if not _safe_id(jid):
            raise KeyError(jid)
        return self.root / jid

    def _read(self, jid: str) -> dict | None:
        with contextlib.suppress(OSError, ValueError, KeyError):
            return json.loads((self.dir_of(jid) / "record.json").read_text(encoding="utf-8"))
        return None

    def _write(self, rec: dict) -> None:
        d = self.dir_of(rec["id"])
        tmp = d / f"record.{uuid.uuid4().hex}.tmp"
        with self.record_lock:
            try:
                tmp.write_text(json.dumps(rec), encoding="utf-8")
                os.replace(tmp, d / "record.json")
            finally:
                tmp.unlink(missing_ok=True)

    def get(self, jid: str) -> dict | None:
        try:
            return self._read(jid)
        except KeyError:
            return None

    def list_jobs(self) -> list[dict]:
        recs = [r for d in self.root.iterdir() if (r := self._read(d.name))]
        recs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return recs

    @staticmethod
    def _stop_wait(rec: dict) -> float:
        # Three observed steps leave room for the active micro-step and cleanup. Before the first report, allow
        # 120 s for model open: a fixed 10 s can terminate an active GPU copy (especially on Windows).
        step = (rec.get("progress") or {}).get("s_per_step")
        return max(30.0, float(step) * 3) if step and float(step) > 0 else 120.0

    def _stop_orphan(self, rec: dict) -> None:
        """Stop a matching orphan gracefully before any signal; never signal a reused PID."""
        from serve.vram_ledger import process_start
        pid = int(rec.get("pid") or 0)
        start = rec.get("pid_start")
        if not pid or start is None:
            return

        def alive():
            current = process_start(pid)
            return current is not None and abs(current - float(start)) < 1e-3

        if not alive():
            return
        if rec.get("stage") == "train":
            d = self.dir_of(rec["id"])
            try:
                control = json.loads((d / "vram.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                control = {}
            self._write_control(d, int(control.get("budget_mib") or 0), bool(control.get("save_state")), stop=True)
            end = time.monotonic() + self._stop_wait(rec)
            while alive() and time.monotonic() < end:
                threading.Event().wait(0.25)
        if alive():
            with contextlib.suppress(OSError):
                os.kill(pid, 15)
            end = time.monotonic() + self._stop_wait(rec)
            while alive() and time.monotonic() < end:
                threading.Event().wait(0.25)
        if alive():
            with contextlib.suppress(OSError):
                os.kill(pid, 9)

    def _gen(self, model: str):
        images = getattr(self.svc, "images", None)
        if images is None:
            raise TrainError("this server has no image model", 503)
        gen = images.pick(model)
        if model and gen.model != model:
            raise TrainError(f"model: {model!r} is not one of this server's image models")
        return gen

    def _need(self, jid: str, statuses: tuple) -> dict:
        rec = self.get(jid)
        if rec is None:
            raise TrainError(f"{jid}: no such job", 404)
        if rec["status"] not in statuses:
            raise TrainError(f"{jid}: the job is {rec['status']}, not {' or '.join(statuses)}")
        return rec

    # ---- creating and editing a draft
    def create(self, fields: dict, images: list[tuple[str, bytes]]) -> dict:
        requested = str(fields.get("model") or "")
        if not requested:
            requested = self._gen("").model
        model, choice = training_model(requested)
        try:
            gen = self._gen(model)
        except TrainError:
            if choice:
                raise TrainError(f"{requested} needs its training base: run {setup_command(self.svc, choice)} "
                                 "and restart the server", 503) from None
            raise
        fam = trainer_family(gen)
        if fam is None:
            raise TrainError(f"model: {gen.model} has no trainer (the families: {', '.join(FAMILIES)})")
        name = check_name(str(fields.get("name") or ""))
        trigger = check_trigger(str(fields.get("trigger") or ""))
        kind = str(fields.get("kind") or "")
        if kind not in KINDS:
            raise TrainError(f"kind: one of {', '.join(KINDS)}")
        if not images:
            raise TrainError("images: at least one training image")
        jid = "lora_" + uuid.uuid4().hex
        d = self.root / jid
        (d / "images").mkdir(parents=True)
        rec = {"id": jid, "status": "draft", "stage": None, "model": gen.model, "family": fam,
               "kind": kind, "name": name, "trigger": trigger, "style_words": "", "settings": None, "images": [],
               "versions": [], "progress": None, "place": None, "error": None,
               "created_at": int(time.time()), "started_at": None, "completed_at": None,
               "pid": None, "pid_start": None, "registered": None,
               "requested_model": requested, "model_switched": bool(choice)}
        self._write(rec)
        try:
            for fname, data in images:
                self.add_image(jid, fname, data)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        got = self.get(jid)
        assert got is not None
        return got

    def add_image(self, jid: str, fname: str, data: bytes) -> dict:
        rec = self._need(jid, ("draft",))
        if len(rec["images"]) >= MAX_IMAGES:
            raise TrainError(f"images: at most {MAX_IMAGES}")
        if len(data) > MAX_IMAGE_BYTES:
            raise TrainError(f"images: one image is at most {MAX_IMAGE_BYTES >> 20} MiB")
        if sum(i["bytes"] for i in rec["images"]) + len(data) > MAX_DATASET_BYTES:
            raise TrainError(f"images: the dataset is at most {MAX_DATASET_BYTES >> 20} MiB")
        if shutil.disk_usage(self.root).free < len(data) + JOB_DISK_MARGIN:
            raise TrainError("the disk has too little room for the dataset")
        ext = Path(fname).suffix.lower()
        if ext not in IMAGE_EXTS:
            raise TrainError(f"images: a {' / '.join(IMAGE_EXTS)} picture, not {fname}")
        try:
            import io
            from PIL import Image
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise TrainError(f"images: {fname} is not a picture that opens") from None
        n = len(rec["images"]) + 1
        stem = f"{n:03d}"
        while (self.dir_of(jid) / "images" / (stem + ext)).exists():
            n += 1
            stem = f"{n:03d}"
        dest = self.dir_of(jid) / "images" / (stem + ext)
        dest.write_bytes(data)
        rec["images"].append({"file": dest.name, "stem": stem, "bytes": len(data), "caption": "", "note": ""})
        self._write(rec)
        return rec

    def remove_image(self, jid: str, fname: str) -> dict:
        rec = self._need(jid, ("draft",))
        item = next((i for i in rec["images"] if i["file"] == fname), None)
        if item is None:
            raise TrainError(f"images: {fname!r} is not one of this job's", 404)
        d = self.dir_of(jid) / "images"
        for p in (d / item["file"], d / (item["stem"] + ".txt"), d / (item["stem"] + ".json")):
            with contextlib.suppress(OSError):
                p.unlink()
        rec["images"].remove(item)
        self._write(rec)
        return rec

    def set_captions(self, jid: str, captions: dict) -> dict:
        rec = self._need(jid, ("draft",))
        by_file = {i["file"]: i for i in rec["images"]}
        checked = []
        for fname, fields in captions.items():
            item = by_file.get(fname)
            if item is None:
                raise TrainError(f"captions: {fname!r} is not one of this job's images")
            if not isinstance(fields, dict) or not isinstance(fields.get("caption"), str) or \
                    not isinstance(fields.get("note"), str):
                raise TrainError("captions: each image needs caption and note strings")
            checked.append((item, check_caption(fields["caption"]), check_caption(fields["note"])))
        d = self.dir_of(jid) / "images"
        for item, caption, note in checked:
            item.update(caption=caption, note=note)
            txt = d / (item["stem"] + ".txt")
            if caption:
                txt.write_text(caption + "\n", encoding="utf-8")
            else:
                txt.unlink(missing_ok=True)   # an empty caption must not make the captioner keep it
            (d / (item["stem"] + ".json")).write_text(json.dumps({"note": note}, ensure_ascii=False), encoding="utf-8")
        self._write(rec)
        return rec

    # ---- the auto captions (before the training; the chat's model reads the pictures)
    def caption(self, jid: str, style_words: str = "") -> dict:
        rec = self._need(jid, ("draft",))
        if not isinstance(style_words, str):
            raise TrainError("style_words: a comma-separated list of style names and features")
        style_words = check_caption(style_words)
        if style_words and rec["kind"] != "style":
            raise TrainError("style_words: only for a style LoRA")
        if not rec["images"]:
            raise TrainError("images: none to caption")
        self._claim(rec)
        rec.update(status="captioning", error=None, style_words=style_words)
        self._write(rec)
        self._enqueue(jid)
        return rec

    def _caption_stage(self, rec: dict, cancel: threading.Event) -> None:
        d = self.dir_of(rec["id"])
        port = getattr(self.svc, "port", 0) or 8095
        tool = Path(__file__).resolve().parents[1] / "tools" / "image" / "lora_caption.py"
        style = "booru" if rec["family"] == "anima" or cache_tag(self._gen(rec["model"])) == "mt" else "natural"
        cmd = [sys.executable, str(tool), "--dir", str(d / "images"), "--kind", rec["kind"],
               "--trigger", rec["trigger"], "--style", style, "--server", f"http://127.0.0.1:{port}"]
        if rec.get("style_words"):
            cmd += ["--style-words=" + rec["style_words"]]
        model_skill = "anima" if rec["family"] == "anima" else ("magetrail" if style == "booru" else "")
        if model_skill:
            cmd += ["--model-skill", model_skill]
        with tempfile.TemporaryFile(mode="w+b") as output:
            proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, text=True)
            self._proc = proc
            from serve.vram_ledger import process_start
            rec.update(pid=proc.pid, pid_start=process_start(proc.pid))
            self._write(rec)
            try:
                while proc.poll() is None:
                    if cancel.is_set():
                        self._terminate(proc, 10.0)  # captioner, not a GPU training process
                        raise _Cancelled()
                    cancel.wait(0.25)
                output.seek(0, os.SEEK_END)
                end = output.tell()
                output.seek(max(0, end - 3200))
                out = output.read().decode("utf-8", errors="replace")
            finally:
                self._proc = None
                rec.update(pid=None, pid_start=None)
                self._write(rec)
        if proc.returncode != 0:
            raise TrainError(f"the captions failed: {(out or '').strip()[-800:]}", 500)
        for item in rec["images"]:
            txt = d / "images" / (item["stem"] + ".txt")
            if txt.is_file():
                item["caption"] = check_caption(txt.read_text(encoding="utf-8"))
        rec["status"] = "draft"
        self._write(rec)

    # ---- starting, cancelling, resuming
    def _enqueue(self, jid: str) -> None:
        with self.lock:
            self.queue.append(jid)
            self.cancels[jid] = threading.Event()
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._work, name="lora-train", daemon=True)
                self.worker.start()

    def start(self, jid: str, fields: dict) -> dict:
        rec = self._need(jid, ("draft", "cancelled", "interrupted", "awaiting_choice", "failed"))
        if rec["status"] in ("cancelled", "interrupted", "awaiting_choice", "failed"):
            return self.resume(jid)
        if rec["model"] in DISTILLED:
            raise TrainError("this older draft uses a distilled model: create a new job with the base model")
        gen = self._gen(rec["model"])
        fam = rec["family"]
        if not TRAINERS[fam]["elastic"]:
            raise TrainError("the trainer does not take a VRAM budget: the training cannot share the GPU "
                             "with the chat and the images", 503)
        exe = trainer_exe(gen)
        if not exe.is_file():
            raise TrainError(f"the trainer is not built: {exe}", 503)
        base, why = train_base(gen)
        if base is None:
            choice = str(gen.cfg.get("setup_choice") or next(
                (k for k, v in _qi.IMAGE_MODELS.items() if v[0] == gen.model), gen.model)).split(":", 1)[0]
            raise TrainError(f"{why}; run {setup_command(self.svc, choice)} and restart the server", 503)
        if not rec["images"]:
            raise TrainError("images: none to train on")
        missing = [i["file"] for i in rec["images"] if not i["caption"].strip()]
        if missing:
            raise TrainError(f"captions: {len(missing)} image(s) have none ({missing[0]}...): caption them first")
        if not any(i["caption"].startswith(rec["trigger"]) for i in rec["images"]):
            raise TrainError(f"captions: none starts with the trigger word {rec['trigger']!r}")
        self._claim(rec)
        rec.update(settings=check_settings(fields, fam, rec["kind"]), status="queued", stage=None, error=None,
                   started_at=int(time.time()))
        self._write(rec)
        self._enqueue(jid)
        return rec

    def cancel(self, jid: str) -> dict:
        rec = self.get(jid)
        if rec is None:
            raise TrainError(f"{jid}: no such job", 404)
        queued = False
        with self.lock:
            ev = self.cancels.get(jid)
            if ev is not None:
                ev.set()
            if jid in self.queue:
                self.queue.remove(jid)
                queued = True
        if queued:
            rec.update(status="cancelled", completed_at=int(time.time()))
            self._write(rec)
        self.wake.set()
        return rec

    def resume(self, jid: str, steps: int | None = None) -> dict:
        with self.mutation_lock:
            return self._resume(jid, steps)

    def _resume(self, jid: str, steps: int | None = None) -> dict:
        rec = self._need(jid, ("interrupted", "cancelled", "awaiting_choice", "completed", "failed"))
        if rec["model"] in DISTILLED:
            raise TrainError("this older job uses a distilled model: create a new job with the base model")
        if not rec.get("settings"):
            raise TrainError("the job was never started: start it")
        if rec["status"] in ("awaiting_choice", "completed"):
            old_steps = rec["settings"]["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not old_steps < steps <= STEPS_MAX:
                raise TrainError(f"steps: choose a new total greater than {old_steps} and at most {STEPS_MAX}")
            path = self.dir_of(jid) / "state.bin"
            try:
                with path.open("rb") as f:
                    magic, count = f.read(8), struct.unpack("<q", f.read(8))[0]
                size = path.stat().st_size
            except (OSError, struct.error):
                raise TrainError("the completed training has no readable continuation state") from None
            if magic != b"XSTRAIN2" or count != (rec.get("progress") or {}).get("step"):
                raise TrainError("the continuation state does not match the last completed update")
            if shutil.disk_usage(self.root).free < size + JOB_DISK_MARGIN:
                raise TrainError("the disk has too little room to save the next continuation state")
            # Mage-Flow writes every run's last version to the same name. Give the old one its step name before
            # another run can overwrite it. Keep the source until the record is written, including after a crash.
            for version in rec["versions"]:
                if version["file"] == "lora.final.safetensors":
                    src = self.dir_of(jid) / "out" / version["file"]
                    dst = src.with_name(f"lora.{version['step']}.safetensors")
                    if dst.exists():
                        if not filecmp.cmp(src, dst, shallow=False):
                            raise TrainError("the saved final version conflicts with its step file")
                    else:
                        tmp = dst.with_suffix(".part")
                        try:
                            shutil.copyfile(src, tmp)
                            os.replace(tmp, dst)
                        finally:
                            with contextlib.suppress(OSError):
                                tmp.unlink()
                    version["file"] = dst.name
            rec["settings"]["steps"] = steps
            rec.update(resume_state=True, registered=None, state_bytes=size)
        elif steps is not None:
            raise TrainError("steps: a new total is only for a completed training")
        self._claim(rec)
        rec.update(status="queued", error=None, completed_at=None)
        self._write(rec)
        self._enqueue(jid)
        return rec

    def delete(self, jid: str) -> dict:
        with self.mutation_lock:
            return self._delete(jid)

    def _delete(self, jid: str) -> dict:
        rec = self.get(jid)
        if rec is None:
            raise TrainError(f"{jid}: no such job", 404)
        if rec["status"] in ("queued", "preprocessing", "training", "sampling", "captioning"):
            self.cancel(jid)
            deadline = time.monotonic() + self._stop_wait(rec) * 2 + 5
            while time.monotonic() < deadline:
                cur = self.get(jid)
                if cur is None or cur["status"] not in ("queued", "preprocessing", "training", "sampling",
                                                        "captioning"):
                    break
                threading.Event().wait(0.1)
            else:
                raise TrainError("the job is still stopping; try deleting it again", 409)
        shutil.rmtree(self.dir_of(jid), ignore_errors=True)
        return {"id": jid, "deleted": True}

    def delete_version(self, jid: str, step: int) -> dict:
        with self.mutation_lock:
            return self._delete_version(jid, step)

    def _delete_version(self, jid: str, step: int) -> dict:
        rec = self._need(jid, ("awaiting_choice", "cancelled", "interrupted", "completed", "failed"))
        v = next((v for v in rec["versions"] if v["step"] == step), None)
        if v is None:
            raise TrainError(f"versions: step {step} is not one of this job's", 404)
        if v.get("registered"):
            raise TrainError("that version is registered: remove the LoRA in the Images view first")
        d = self.dir_of(jid)
        with contextlib.suppress(OSError):
            (d / "out" / v["file"]).unlink()
        for s in v.get("samples") or []:
            with contextlib.suppress(OSError):
                (d / "samples" / s).unlink()
        rec["versions"].remove(v)
        self._write(rec)
        return rec

    def register(self, jid: str, step: int | None, name: str | None) -> dict:
        with self.mutation_lock:
            return self._register(jid, step, name)

    def _register(self, jid: str, step: int | None, name: str | None) -> dict:
        rec = self._need(jid, ("awaiting_choice", "cancelled", "interrupted", "completed", "failed"))
        if not rec["versions"]:
            raise TrainError("versions: none saved yet")
        v = next((v for v in rec["versions"] if v["step"] == step), None) if step is not None \
            else rec["versions"][-1]
        if v is None:
            raise TrainError(f"versions: step {step} is not one of this job's", 404)
        if v.get("registered"):
            raise TrainError("that version is registered already")
        src = self.dir_of(jid) / "out" / v["file"]
        if not src.is_file():
            raise TrainError("the version's file is gone", 404)
        stem = check_name(name or rec["name"])
        folders = [g.loras for g in self.svc.images.gens.values()]
        try:
            tmp = lora_api.staging(folders, src.stat().st_size)
            try:
                shutil.copyfile(src, tmp)
                added = lora_api.add(folders, tmp, stem + ".safetensors")
            finally:
                tmp.unlink(missing_ok=True)
        except (lora_api.LoraError, OSError) as e:
            raise TrainError(str(e)) from None
        v["registered"] = added["name"]
        rec["registered"] = {"step": v["step"], "name": added["name"], "path": added["path"]}
        rec.update(status="completed", completed_at=int(time.time()))
        self._write(rec)
        return rec

    # ---- the worker
    def _work(self) -> None:
        while True:
            with self.lock:
                if not self.queue:
                    self.worker = None
                    return
                jid = self.queue.pop(0)
                cancel = self.cancels.get(jid) or threading.Event()
            rec = self._read(jid)
            if rec is None:
                continue
            self.current = jid
            try:
                if rec["status"] == "captioning":
                    self._caption_stage(rec, cancel)
                elif rec["status"] == "queued":
                    self._run(rec, cancel)
            except _Cancelled:
                rec.update(status="cancelled", completed_at=int(time.time()), error=None, place=None)
                self._write(rec)          # its versions stay, to pick and register
            except Exception as e:   # noqa: BLE001 - the job's failure, said in its record
                rec.update(status="failed", completed_at=int(time.time()), place=None,
                           error={"code": "train_failed", "message": str(e)})
                self._write(rec)
            finally:
                self.current = None
                self.place = None
                self._proc = None
                self._train_state = {}
                self._decider = None
                with self.lock:
                    self.cancels.pop(jid, None)

    def _run(self, rec: dict, cancel: threading.Event) -> None:
        d = self.dir_of(rec["id"])
        (d / "out").mkdir(exist_ok=True)
        (d / "samples").mkdir(exist_ok=True)
        if not self._preprocessed(rec):
            rec.update(status="preprocessing", stage="preprocess")
            self._write(rec)
            self._preprocess(rec, cancel)
        rec.update(status="training", stage="train")
        self._write(rec)
        self._train_stage(rec, cancel)
        if cancel.is_set():
            raise _Cancelled()
        rec.update(status="sampling", stage="samples", place=None)
        self._write(rec)
        self._samples(rec, cancel)
        rec.update(status="awaiting_choice", stage=None, completed_at=int(time.time()))
        self._write(rec)

    # ---- preprocess (the VAE and text reader, in the image jobs' GPU turn)
    def _preprocessed(self, rec: dict) -> bool:
        d = self.dir_of(rec["id"])
        if rec["family"] == "qwen-image":
            return (d / "cache" / "items.txt").is_file()
        return (d / "export" / "manifest.tsv").is_file()

    def preprocess_argv(self, rec: dict) -> list[list[str]]:
        """The preprocess commands (mf_pre.py, then mf_pre_export.py but for qi).  One place; the tests replace it."""
        gen = self._gen(rec["model"])
        d = self.dir_of(rec["id"])
        tool = Path(__file__).resolve().parents[1] / "tools" / "image"
        tag = cache_tag(gen)
        pre = [sys.executable, str(tool / "mf_pre.py"), "--cache", tag, "--dir", str(d / "images"),
               "--vae", str(gen.cfg["vae"]), "--text", str(gen.cfg["llm"]),
               "--pre", str(Path(gen.cfg["exe"]).parent / "xestrata-image-mf-pre"),
               "--budget", ",".join(str(n) for n in (512, 768, 1024) if n <= rec["settings"]["max_size"])]
        if rec["family"] == "qwen-image":
            pre += ["--out", str(d / "cache"), "--trigger", rec["trigger"]]
            return [pre]
        if tag in ("ms", "an"):
            pre += ["--dit", str(gen.cfg["diffusion_model"])]
        if tag == "an":
            pre += ["--t5", str(gen.cfg["t5_tokenizer"])]
        export = [sys.executable, str(tool / "mf_pre_export.py"), "--cache", tag,
                  "--dir", str(d / "images"), "--out", str(d / "export")]
        return [pre, export]

    def _preprocess(self, rec: dict, cancel: threading.Event) -> None:
        from serve import images as img
        gen = self._gen(rec["model"])
        turn = img.GpuTurn(gen.engine.env)
        turn.acquire(cancel)
        try:
            turn.keep_only(f"train-pre:{rec['id']}")   # the GPU's idle image daemons end (none is the trainer's)
            for cmd in self.preprocess_argv(rec):
                if cancel.is_set():
                    raise _Cancelled()
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise TrainError(f"the preprocess failed: {(r.stderr or r.stdout).strip()[-800:]}", 500)
        finally:
            turn.release()
        if not self._preprocessed(rec):
            raise TrainError("the preprocess wrote no cache the trainer reads", 500)

    # ---- train (the trainer subprocess; train_budget decides the card and the budget)
    def train_argv(self, rec: dict, budget: int, resume: bool) -> list[str]:
        """The trainer's command line (TRAINERS' mapping, one place)."""
        gen = self._gen(rec["model"])
        fam, s = rec["family"], rec["settings"]
        t = TRAINERS[fam]
        d = self.dir_of(rec["id"])
        base, _ = train_base(gen)
        data = str(d / "cache" / "items.txt") if fam == "qwen-image" else str(d / "export")
        argv = [str(trainer_exe(gen)), "train", "--dit", str(base), "--data", data,
                "--out", str(d / "out" / "lora"), "--rank", str(s["rank"]), "--lr", str(s["lr"]),
                "--seed", str(s["seed"]), "--kind", rec["kind"], "--trigger", rec["trigger"],
                "--caption-dropout", str(s["dropout"])]
        if fam == "mage-flow":
            argv += ["--steps", str(s["steps"] * s["accum"]), "--accum", str(s["accum"]),
                     "--checkpoint", str(s["save_every"]), "--method", s["method"],
                     "--title", rec["name"], "--base", s["base"]]
        else:
            argv += ["--steps", str(s["steps"]), t["save_flag"], str(s["save_every"])]
            if fam != "anima":
                argv += ["--base", s["base"]]
        for key, flag in (("train_blocks", "--train-blocks"), ("only_if_contains", "--only-if-contains"),
                          ("ignore_if_contains", "--ignore-if-contains")):
            if s.get(key):
                argv += [flag, s[key]]
        argv += ["--vram-budget", str(int(budget)), "--vram-control", str(d / "vram.json"),
                 "--state", str(d / "state.bin")]
        if resume:
            argv += ["--resume", str(d / "state.bin")]
        return argv

    def _state(self, rec: dict) -> dict:
        """The current device room and trainer report in train_budget.Decider's input format."""
        from serve.vram_ledger import process_memory_mib
        current = str((self.place or {}).get("pci") or "")
        proc = self._proc
        alive = proc is not None and proc.poll() is None
        source = self._train_state.get("process_source")
        before = process_memory_mib(proc.pid, current, source) if proc is not None and alive and current else None
        # Bypass the device cache even if another thread refills it: the process and free readings are paired.
        reports = devices(self.svc, fresh=alive)
        after = process_memory_mib(proc.pid, current, source) if proc is not None and alive and current else None
        free = {str(d["pci"]).lower(): int(d.get("free_bytes") or 0) >> 20 for d in reports}
        device_free = dict(free)
        _, helper, held = text_placement(self.svc)
        status = getattr(self.svc, "status", {})
        train = dict(self._train_state)
        train["process_live"] = bool(before is not None and after is not None and before[1] == after[1])
        if train["process_live"]:
            assert before is not None and after is not None
            # The lower bracketing reading is conservative when a step allocates or frees during the device query.
            train.update(process_mib=min(before[0], after[0]), process_source=before[1])
        with self.lock:
            needs = list(self.image_needs.values())
            image_seq = self.image_seq
        known = [n for n in needs if n is not None]
        image_need = max(known) if needs and len(known) == len(needs) else None
        return {"free": free, "device_free": device_free,
                "proc_alive": proc is not None and proc.poll() is None,
                "chat_busy": self._chat_busy(), "chat_queued": bool(status.get("queued")),
                "chat_eta_s": None, "image_job": bool(needs), "image_need_mib": image_need, "image_seq": image_seq,
                "helper_mib": held.get(helper, 0), "train": train, "current_pci": current,
                "step_s": (rec.get("progress") or {}).get("s_per_step"), "reload_s": None}

    def _budget_model(self, rec: dict) -> tuple[str, str]:
        fam = rec["family"]
        model = "qwen-image-2.1" if fam == "qwen-image" else fam
        if fam == "mage-flow" and cache_tag(self._gen(rec["model"])) == "mt":
            model = "magetrail"
        return model, rec["settings"].get("base", "bf16")

    def _chat_busy(self) -> bool:
        svc = self.svc
        if svc is None:
            return False
        with contextlib.suppress(Exception):
            return bool(getattr(svc, "live_reqs", None)) or bool(svc.status.get("busy"))
        return False

    def _write_control(self, d: Path, budget: int, save_state: bool, stop: bool = False) -> None:
        tmp = d / "vram.json.tmp"
        tmp.write_text(json.dumps({"budget_mib": int(budget), "save_state": bool(save_state), "stop": bool(stop)}),
                       encoding="utf-8")
        os.replace(tmp, d / "vram.json")

    def _spawn(self, rec: dict, pci: str, budget: int, resume: bool, cancel: threading.Event) -> subprocess.Popen:
        """Start the trainer on the card, after open_mib of temporary room is free (the open reads and quantizes the
        weights even when the budget starts at 0)."""
        tb = budget_mod()
        need = int(tb.figures(*self._budget_model(rec))[0])
        waited = 0.0
        while self.image_waiting or free_mib(self.svc, pci) < need:
            if cancel.is_set():
                raise _Cancelled()
            if self.image_waiting:
                state = self._state(rec)
                self._accept_yield(self._decider.tick(state), state)
            else:
                if waited >= OPEN_WAIT_S:
                    raise TrainError(f"the card {pci} has no {need} MiB of temporary room for the trainer's open", 503)
                waited += 1.0
            self.wake.wait(1.0)
            self.wake.clear()
        if cancel.is_set():
            raise _Cancelled()
        if resume and not (self.dir_of(rec["id"]) / "state.bin").is_file():
            raise TrainError("the continuation state is missing")
        env = card_env(self.svc, pci)
        self._write_control(self.dir_of(rec["id"]), budget, False)
        proc = subprocess.Popen(self.train_argv(rec, budget, resume), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        self._proc = proc
        self._state_saved.clear()
        self._train_state = {}
        self._step_clock = None
        from serve.vram_ledger import process_start
        rec.update(pid=proc.pid, pid_start=process_start(proc.pid))
        self._write(rec)
        self._reader = threading.Thread(target=self._read_train, args=(rec, proc), daemon=True)
        self._reader.start()
        return proc

    def _train_stage(self, rec: dict, cancel: threading.Event) -> None:
        from serve import images as img
        from serve.vram_ledger import Ledger
        tb = budget_mod()
        d = self.dir_of(rec["id"])
        model, base = self._budget_model(rec)
        data = d / "cache" / "items.txt" if model == "qwen-image-2.1" else d / "export"
        tokens = tb.dataset_tokens(model, data)
        rank = rec["settings"]["rank"]
        minimum, need = tb.initial_vram(model, base, tokens, rank)
        rec["training_shape"] = {"tokens": tokens, "rank": rank, "min_mib": minimum, "need_mib": need}
        self._write(rec)
        self._decider = tb.Decider(run_config(self.svc), model, base, devices=devices(self.svc),
                                   image_tokens=tokens, rank=rank)
        act = self._decider.tick(self._state(rec))
        if not act or not act.get("pci"):
            raise TrainError(str((act or {}).get("reason") or "no GPU can take the training now"), 503)
        pci = str(act["pci"]).lower()
        self.place = {"pci": pci, "budget_mib": int(act.get("budget_mib", 0)),
                      "mode": self._mode(rec, act), "reason": str(act.get("reason", ""))}
        log = open(d / "train.log", "a", encoding="utf-8")
        self._log = log
        proc = None
        pending = None
        try:
            proc = self._spawn(rec, pci, int(act.get("budget_mib", 0)),
                               resume=bool(rec.get("resume_state")), cancel=cancel)
            led = Ledger(img.gpu_key(card_env(self.svc, pci)))
            deadline = 0.0
            while True:
                if cancel.is_set():
                    raise _Cancelled()
                if proc is not None and proc.poll() is not None:
                    if pending is None:
                        break
                    assert self._reader is not None
                    self._reader.join()
                    self._reader = None
                    if proc.returncode != 0 or not self._state_saved.is_set() or not (d / "state.bin").is_file():
                        raise TrainError("the trainer stopped without saving its state", 500)
                    self._proc = None
                    self._train_state = {}
                    _DEV_CACHE.clear()
                    self._scan_versions(rec)
                    pci = str(pending.get("pci") or pci).lower()
                    self.place["pci"] = pci
                    proc = None
                    pending = None
                    rec["resume_state"] = True
                    self._write(rec)
                if proc is None:
                    if self.image_waiting:
                        # The policy also checks the device's actual free room after the process exits.
                        state = self._state(rec)
                        self._accept_yield(self._decider.tick(state), state)
                    else:
                        act = self._decider.tick(self._state(rec)) or {}
                        card = str(act.get("pci") or pci).lower()
                        budget = int(act.get("budget_mib", 0))
                        if budget > 0 and not act.get("save_state"):
                            pci = card
                            self.place["pci"] = pci
                            led = Ledger(img.gpu_key(card_env(self.svc, pci)))
                            proc = self._spawn(rec, pci, budget, resume=True, cancel=cancel)
                else:
                    state = self._state(rec)
                    act = pending or self._decider.tick(state) or {}
                    budget = int(act.get("budget_mib", 0))
                    card = str(act.get("pci") or pci).lower()
                    self.place = {"pci": pci, "budget_mib": budget, "mode": self._mode(rec, act),
                                  "reason": str(act.get("reason", "")), "name": self._device_name(pci)}
                    self._write_control(d, budget, bool(act.get("save_state")), stop=bool(act.get("save_state")))
                    led.hold("train", max(budget, int(self._train_state.get("used_mib", 0))), proc.pid,
                             exact=False, whole=True)
                    rec["place"] = self.place
                    self._write(rec)
                    if act.get("save_state"):
                        if pending is None:
                            pending, deadline = act, time.monotonic() + self._stop_wait(rec)
                        if time.monotonic() >= deadline:
                            self._terminate(proc, self._stop_wait(rec))
                    elif self.image_waiting and act.get("image_ready") is True and \
                            self._train_state.get("state") == "paused":
                        # Only the policy can certify the paused process left enough memory.
                        self._accept_yield(act, state)
                self.wake.wait(1.0)
                self.wake.clear()
            assert self._reader is not None
            self._reader.join()
            self._scan_versions(rec)
            if cancel.is_set():
                raise _Cancelled()
            if proc.returncode != 0:
                tail = (d / "train.log").read_text(encoding="utf-8", errors="replace")[-800:]
                raise TrainError(f"the trainer failed (exit {proc.returncode}): {tail}", 500)
            if not rec["versions"]:
                raise TrainError("the trainer saved no version", 500)
            state_path = d / "state.bin"
            rec["state_bytes"] = (state_path.stat().st_size if self._state_saved.is_set() and state_path.is_file()
                                  else None)
            rec["resume_state"] = bool(rec["state_bytes"])
            self._write(rec)
        finally:
            if proc is not None and proc.poll() is None:
                self._stop_trainer(rec, proc, save_state=pending is not None)
            if self._reader is not None:
                self._reader.join()
                self._reader = None
            log.close()
            self._log = None
            self._proc = None
            _DEV_CACHE.clear()
            if self.image_waiting:
                state = self._state(rec)
                self._accept_yield(self._decider.tick(state), state)
            with self.lock:
                self.place = None

    def _accept_yield(self, act: dict, state: dict) -> None:
        if act.get("image_ready") is True:
            with self.lock:
                if self.image_needs and self.image_seq == state["image_seq"]:
                    self.yielded.set()

    def _mode(self, rec: dict, act: dict) -> str:
        budget = int(act.get("budget_mib", 0))
        if budget <= 0:
            return "paused"
        return "run" if budget >= PRESETS[rec["family"]]["peak_mib"] else "shrunk"

    def _device_name(self, pci: str) -> str:
        d = next((d for d in devices(self.svc) if str(d["pci"]).lower() == pci), None)
        return str(d.get("name") or pci) if d else pci

    def _read_train(self, rec: dict, proc: subprocess.Popen) -> None:
        """The trainer's stdout: the progress lines, the vram reports (the trailing key=value fields ignored), the
        state saved line, the versions as they appear."""
        log = getattr(self, "_log", None)
        assert proc.stdout is not None
        for line in proc.stdout:
            if log is not None:
                log.write(line)
                log.flush()
            tok = line.split()
            if tok and tok[0] == "progress":
                with contextlib.suppress(ValueError, KeyError):
                    fields = dict(t.split("=", 1) for t in tok[1:] if "=" in t)
                    step, total, loss = int(fields["step"]), int(fields["total"]), float(fields["loss"])
                    if not 0 < step <= total or not math.isfinite(loss):
                        continue
                    now = time.monotonic()
                    speed = ((now - self._step_clock[1]) / (step - self._step_clock[0])
                             if self._step_clock and step > self._step_clock[0] else None)
                    self._step_clock = step, now
                    rec["progress"] = {"step": step, "total": total, "loss": loss, "s_per_step": speed,
                                       "eta_s": int(speed * (total - step)) if speed else None}
                    self._scan_versions(rec)
                    self._write(rec)
                continue
            if tok and tok[0] == "vram" and len(tok) >= 5 and tok[4] in ("running", "paused"):
                with contextlib.suppress(ValueError):
                    _DEV_CACHE.clear()  # the pause/grow changed free VRAM; the policy needs a fresh report
                    extra = dict(t.split("=", 1) for t in tok[5:] if "=" in t)
                    need = int(extra["need_mib"]) if "need_mib" in extra else None
                    process = int(extra["process_mib"]) if "process_mib" in extra else None
                    self._train_state = {"used_mib": int(tok[1]), "peak_est_mib": int(tok[2]), "min_mib": int(tok[3]),
                                         "need_mib": need, "process_mib": process,
                                         "process_source": extra.get("process_source"), "state": tok[4]}
                    self.wake.set()
                continue
            if tok and tok[0] == "state" and len(tok) >= 3 and tok[1] == "saved":
                self._state_saved.set()
                self.wake.set()
        with contextlib.suppress(Exception):
            proc.stdout.close()
        self._scan_versions(rec)

    def _scan_versions(self, rec: dict) -> None:
        """New checkpoint files in out/, kept only once their header reads (a cancel mid-save leaves a broken one)."""
        d = self.dir_of(rec["id"]) / "out"
        if not d.is_dir():
            return
        t = TRAINERS[rec["family"]]
        known = {v["file"] for v in rec["versions"]}
        pat = re.compile(r"lora_(\d+)\.safetensors" if t["ckpt"] == "_N" else r"lora\.(\d+)\.safetensors")
        found: list[tuple[int, str]] = []
        for p in sorted(d.iterdir()):
            if p.name in known or not p.is_file():
                continue
            m = pat.fullmatch(p.name)
            if m:
                found.append((int(m.group(1)), p.name))
            elif t["ckpt"] == ".N" and p.name == "lora.final.safetensors":
                found.append((-1, p.name))
        for step, fname in found:
            try:
                _, meta = lora_api.read_header(d / fname)
            except Exception:   # noqa: BLE001 - not written whole yet (or broken): not a version
                continue
            if step < 0:
                try:
                    step = int(meta.get("ss_steps", "0"))
                except ValueError:
                    step = rec["settings"]["steps"]
            if any(v["step"] == step for v in rec["versions"]):
                continue
            rec["versions"].append({"step": step, "file": fname, "samples": [], "registered": None})
            rec["versions"].sort(key=lambda v: v["step"])

    def _stop_trainer(self, rec: dict, proc: subprocess.Popen, save_state: bool = False) -> None:
        d = self.dir_of(rec["id"])
        try:
            control = json.loads((d / "vram.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            control = {}
        self._write_control(d, int(control.get("budget_mib") or 0), save_state, stop=True)
        try:
            proc.wait(self._stop_wait(rec))
        except subprocess.TimeoutExpired:
            self._terminate(proc, self._stop_wait(rec))

    def _terminate(self, proc: subprocess.Popen, wait_s: float) -> None:
        """Last resort after the trainer's JSON stop did not finish."""
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(wait_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            proc.wait(wait_s)

    # ---- samples (after the training, the GPU free; the image jobs' own turn)
    def _samples(self, rec: dict, cancel: threading.Event) -> None:
        from serve import images as img
        gen = self._gen(rec["model"])
        d = self.dir_of(rec["id"])
        s = rec["settings"]
        prompts = [p.replace("{trigger}", rec["trigger"]) for p in s["sample_prompts"]]
        for v in rec["versions"]:
            if v.get("samples"):
                continue
            use = {"name": f"step-{v['step']}", "path": str(d / "out" / v["file"]), "scale": 1.0}
            made = []
            for i, prompt in enumerate(prompts):
                if cancel.is_set():
                    raise _Cancelled()
                req = {"prompt": prompt, "size": f"{SAMPLE_SIZE[0]}x{SAMPLE_SIZE[1]}", "n": 1,
                       "seed": s["sample_seed"], "response_format": "b64_json"}
                job = img.parse_request(req, gen.cfg)
                job["loras"] = [use]            # the version's file directly (the server's own call, not the API's)
                with img.gpu_turn(self.svc, gen, job, cancel):
                    pngs, _ = gen.run(job, cancel)
                name = f"{v['step']}-{i}.png"
                (d / "samples" / name).write_bytes(pngs[0])
                made.append(name)
                gen.end(None)
            v["samples"] = made
            self._write(rec)

    # ---- the image jobs' yield (images.gpu_turn calls these)
    def image_begin(self, need_mib: int | None = None, pci: str | None = None) -> str | None:
        """Register an image's need until image_end; unknown need requires the Decider's safe stop."""
        with self.lock:
            if self.current is None or self.place is None:
                return None
            train_pci = str(self.place.get("pci") or "").lower()
            if pci and train_pci and pci.lower() != train_pci:
                return None
            token = uuid.uuid4().hex
            self.image_needs[token] = need_mib
            self.image_seq += 1
            self.image_waiting = len(self.image_needs)
            self.yielded.clear()
        self.wake.set()
        if not self.yielded.wait(IMAGE_YIELD_WAIT_S):
            self.image_end(token)
            raise TrainError("the training did not release the GPU for the image in time", 503)
        return token

    def image_end(self, token: str) -> None:
        with self.lock:
            if token in self.image_needs:
                del self.image_needs[token]
                self.image_seq += 1
            self.image_waiting = len(self.image_needs)
        self.wake.set()


def jobs_of(svc) -> TrainJobs:
    """The server's training jobs (svc.lora_jobs) under the data folder's lora-jobs/, made at their first use."""
    with svc.status_lock:
        if getattr(svc, "lora_jobs", None) is None:
            svc.lora_jobs = TrainJobs(data_root() / "lora-jobs", svc)
        return svc.lora_jobs


def status_row(svc) -> dict | None:
    """/v1/status's training row: the job, its stage, progress and place (run/shrunk/paused and the reason)."""
    jobs = getattr(svc, "lora_jobs", None)
    if jobs is None or jobs.current is None:
        return None
    rec = jobs.get(jobs.current)
    if rec is None:
        return None
    return {"id": rec["id"], "name": rec["name"], "model": rec["model"], "status": rec["status"],
            "stage": rec["stage"], "progress": rec["progress"], "place": jobs.place}


def explain(e: Exception) -> tuple[int, dict]:
    if isinstance(e, TrainError):
        return e.status, {"error": {"type": "invalid_request_error", "code": "invalid_training", "message": str(e)}}
    if isinstance(e, lora_api.LoraError):
        return 400, {"error": {"type": "invalid_request_error", "message": str(e)}}
    return 500, {"error": {"type": "server_error", "code": "train_failed", "message": str(e)}}
