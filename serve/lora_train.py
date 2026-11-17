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

The GPU (LORAWEB DESIGN §5, the user's decision): training is the lowest priority.  The decisions (the card, the
budget, shrink / pause / move) are TRAINVRAM's serve/train_budget.py, one place; TrainPlace here calls its Decider at
every step boundary's cadence (each second and on a chat/image signal), writes the control file, charges the ledger
and shows the state.  Chat and image jobs come first: training shrinks or pauses at a micro-step boundary (its LoRA
and optimizer state kept) and expands or resumes when they leave.  Image jobs are never kept waiting for the
training's end (images.gpu_turn's image_begin waits at most a micro-step's seconds plus the release).  The helper
card is never closed for hours.

The budget protocol (TRAINVRAM DESIGN §2, agreed): the trainer takes --vram-budget (0 = start paused; the open still
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
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from serve import envvars
from serve import loras as lora_api

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
# The image job's wait for the training's yield: one micro-step plus the release (TRAINVRAM DESIGN §2).  Mage-Flow's
# 1024-class micro-step is 8.9 s (records 2026-11-03-mf-trainer §3); qi/moss/anima are measured at REC - until then
# the wait covers the slowest known micro-step with room.  Past it the ledger's room still guards the VRAM.
IMAGE_YIELD_WAIT_S = 60.0
OPEN_WAIT_S = 600.0               # the longest wait for open_mib of temporary room before a trainer starts
DEVICES_CACHE_S = 5.0             # xestrata-device --list-devices is a subprocess; not one every tick

# The one place of the training defaults (LORAWEB DESIGN §3): the trainers' current defaults (ai-toolkit's job, which
# LORA2 ran) until the comparison after REC decides the values.  peak_mib: the training's peak VRAM at the 1024 class
# (Mage-Flow measured, docs/LORA_TRAIN; Qwen-Image the user's number; MOSS-Image and Anima estimates, unverified).
# The least budget and the open's room are train_budget.py's (min_mib / open_mib, TRAINVRAM's measurements).
PRESETS: dict[str, dict] = {
    "qwen-image": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": 0.05,
                   "base": "bf16", "peak_mib": 15360},
    "mage-flow": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 12345, "dropout": 0.05,
                  "accum": 1, "base": "int8", "method": "lora", "peak_mib": 14848},
    "moss-image": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": 0.05,
                   "base": "int8", "peak_mib": 16384},
    "anima": {"steps": 3000, "save_every": 250, "rank": 32, "lr": 1e-4, "seed": 42, "dropout": 0.05,
              "peak_mib": 12288},
}
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
SAMPLE_SEED = 7                   # LORA2's sweeps' seed (fixed, so the versions compare)
SAMPLE_SIZE = (1024, 1024)

# The trainers' entry points (LORAWEB DESIGN §2).  elastic: the trainer takes --vram-budget/--vram-control (TRAINVRAM);
# False until it lands - a start is refused with the reason until then.  ckpt: how a version's file name carries its
# step ("_N" = <prefix>_<step>.safetensors, ".N" = <prefix>.<step>.safetensors and <prefix>.final.safetensors).
TRAINERS: dict[str, dict] = {
    "qwen-image": {"exe": "xestrata-image-qi-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "qi",
                   "elastic": False,
                   "progress": re.compile(r"^step (\d+) loss ([\d.eE+-]+) \(avg ([\d.eE+-]+)\).*"
                                          r"\(([\d.]+) s/step\)")},
    "mage-flow": {"exe": "xestrata-image-mf-train", "save_flag": "--checkpoint", "ckpt": ".N", "cache": "mf",
                  "elastic": False, "progress": re.compile(r"^step\s+(\d+) loss ([\d.eE+-]+) lr")},
    "moss-image": {"exe": "xestrata-image-moss-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "ms",
                   "elastic": False,
                   "progress": re.compile(r"^step (\d+) loss ([\d.eE+-]+) \(avg ([\d.eE+-]+)\).*"
                                          r"\(([\d.]+) s/step\)")},
    "anima": {"exe": "xestrata-image-anima-train", "save_flag": "--save-every", "ckpt": "_N", "cache": "an",
              "elastic": False,
              "progress": re.compile(r"^step (\d+) loss ([\d.eE+-]+) \(avg ([\d.eE+-]+)\).*"
                                     r"\(([\d.]+) s/step\)")},
}


class TrainError(ValueError):
    """A training request that cannot run: the message says why (the panel shows it)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class _Cancelled(Exception):
    """The job's cancel reached a stage."""


def budget_mod():
    """TRAINVRAM's decisions (serve/train_budget.py: figures, Decider).  One place; not
    duplicated here.  The tests replace this with a fake of the same shape."""
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
                          "(setup --images downloads it, tools/image/qimage.py's base-F16)")
        return f16, ""
    if not dit.exists():
        return None, f"the DiT is not here: {dit}"
    return dit, ""


def model_rows(svc) -> list[dict]:
    """GET /v1/lora/models: the installed image models a trainer exists for, each with its preset and whether it can
    train here (the trainer binary and the base file present)."""
    rows = []
    images = getattr(svc, "images", None)
    for gen in list(images.gens.values()) if images is not None else []:
        fam = trainer_family(gen)
        if fam is None:
            continue
        exe = trainer_exe(gen)
        base, why = train_base(gen)
        ok = exe.is_file() and base is not None
        rows.append({"model": gen.model, "family": fam, "name": lora_api.FAMILY_NAMES.get(fam, fam),
                     "preset": PRESETS[fam], "sample_prompts": SAMPLE_PROMPTS, "sample_seed": SAMPLE_SEED,
                     "can_train": ok,
                     "why": "" if ok else (f"the trainer is not built: {exe.name}" if not exe.is_file() else why)})
    return rows


# ------------------------------------------------------------------------------------------------- a request's fields
def check_name(name: str) -> str:
    stem = Path(str(name or "").replace("\\", "/")).name.strip()
    stem = re.sub(r"[^A-Za-z0-9._ +-]+", "_", stem).strip(" ._")[:120]
    if not stem:
        raise TrainError("name: the LoRA needs a name (letters, digits, spaces, . _ + -)")
    return stem


def check_trigger(trigger: str) -> str:
    t = str(trigger or "").strip()
    if not t or len(t) > 120 or "\n" in t or "," in t:
        raise TrainError("trigger: one word or phrase, up to 120 characters, no comma or line break")
    return t


def check_settings(fields: dict, fam: str, kind: str = "other") -> dict:
    """The start's settings over the preset: n (steps), k (save_every), rank, lr, seed, and the advanced ones."""
    p = PRESETS[fam]

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
    out = {"steps": n, "save_every": k,
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
        v = str(fields.get(key) or "").strip()
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


def devices(svc) -> list[dict]:
    """The GPUs as xestrata-device --list-devices reports them (one JSON a device: ordinal, pci, total_bytes,
    free_bytes, integrated, compute_units, arch).  [] when the program is not here.  Cached DEVICES_CACHE_S."""
    images = getattr(svc, "images", None)
    if images is None:
        return []
    exe = str(Path(images.default.cfg["exe"]).parent / "xestrata-device")
    got = _DEV_CACHE.get(exe)
    if got is not None and time.monotonic() - got[0] < DEVICES_CACHE_S:
        return got[1]
    out: list[dict] = []
    if Path(exe).is_file():
        try:
            r = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=60)
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
    """The trainer's environment on the card `pci` (the image engine's env over this process's, narrowed to it)."""
    env = dict(os.environ)
    env.update(svc.images.default.engine.env or {})
    env["XESTRATA_GPU_PCI"] = pci
    for k in ("CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"):
        env.pop(k, None)
    return env


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
        self._state_saved = threading.Event()    # the trainer wrote its state (state saved line)
        self._decider: Any = None
        self._log: Any = None
        self._reader: threading.Thread | None = None
        for d in self.root.iterdir():
            rec = self._read(d.name)
            if rec and rec.get("status") in ("queued", "preprocessing", "training", "sampling", "captioning"):
                self._stop_orphan(rec)
                if rec.get("settings"):
                    rec.update(status="interrupted", error={"code": "server_restarted",
                                                            "message": "the server stopped before the job ended"})
                else:
                    rec.update(status="draft", error=None)
                self._write(rec)

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
        tmp = d / "record.json.tmp"
        with self.record_lock:
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            os.replace(tmp, d / "record.json")

    def get(self, jid: str) -> dict | None:
        try:
            return self._read(jid)
        except KeyError:
            return None

    def list_jobs(self) -> list[dict]:
        recs = [r for d in self.root.iterdir() if (r := self._read(d.name))]
        recs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return recs

    def _stop_orphan(self, rec: dict) -> None:
        """A trainer (or preprocessor) the last server left running: stop it when its pid and start time match."""
        from serve.vram_ledger import process_start
        pid = rec.get("pid")
        if not pid:
            return
        start = process_start(int(pid))
        if start is None or abs(start - float(rec.get("pid_start", -1))) > 1e-3:
            return
        with contextlib.suppress(OSError):
            os.kill(int(pid), 15)
            for _ in range(40):
                time.sleep(0.25)
                if process_start(int(pid)) is None:
                    return
            os.kill(int(pid), 9)

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
        gen = self._gen(str(fields.get("model") or ""))
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
               "kind": kind, "name": name, "trigger": trigger, "settings": None, "images": [],
               "versions": [], "progress": None, "place": None, "error": None,
               "created_at": int(time.time()), "started_at": None, "completed_at": None,
               "pid": None, "pid_start": None, "registered": None}
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
        rec["images"].append({"file": dest.name, "stem": stem, "bytes": len(data), "caption": ""})
        self._write(rec)
        return rec

    def remove_image(self, jid: str, fname: str) -> dict:
        rec = self._need(jid, ("draft",))
        item = next((i for i in rec["images"] if i["file"] == fname), None)
        if item is None:
            raise TrainError(f"images: {fname!r} is not one of this job's", 404)
        d = self.dir_of(jid) / "images"
        for p in (d / item["file"], d / (item["stem"] + ".txt")):
            with contextlib.suppress(OSError):
                p.unlink()
        rec["images"].remove(item)
        self._write(rec)
        return rec

    def set_captions(self, jid: str, captions: dict) -> dict:
        rec = self._need(jid, ("draft",))
        by_file = {i["file"]: i for i in rec["images"]}
        for fname, text in captions.items():
            item = by_file.get(fname)
            if item is None:
                raise TrainError(f"captions: {fname!r} is not one of this job's images")
            item["caption"] = check_caption(text)
            (self.dir_of(jid) / "images" / (item["stem"] + ".txt")).write_text(
                item["caption"] + "\n", encoding="utf-8")
        self._write(rec)
        return rec

    # ---- the auto captions (before the training; the chat's model reads the pictures)
    def caption(self, jid: str) -> dict:
        rec = self._need(jid, ("draft",))
        if not rec["images"]:
            raise TrainError("images: none to caption")
        rec.update(status="captioning", error=None)
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
        model_skill = "anima" if rec["family"] == "anima" else ("magetrail" if style == "booru" else "")
        if model_skill:
            cmd += ["--model-skill", model_skill]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._proc = proc
        try:
            while proc.poll() is None:
                if cancel.is_set():
                    self._terminate(proc)
                    raise _Cancelled()
                time.sleep(0.25)
            out, _ = proc.communicate()
        finally:
            self._proc = None
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
        rec = self._need(jid, ("draft", "cancelled", "interrupted", "awaiting_choice"))
        if rec["status"] in ("cancelled", "interrupted", "awaiting_choice"):
            return self.resume(jid)
        gen = self._gen(rec["model"])
        fam = rec["family"]
        if not TRAINERS[fam]["elastic"]:
            raise TrainError("the trainer does not take a VRAM budget yet (the elastic training, TRAINVRAM, is not "
                             "integrated): the training cannot share the GPU with the chat and the images", 503)
        exe = trainer_exe(gen)
        if not exe.is_file():
            raise TrainError(f"the trainer is not built: {exe}", 503)
        base, why = train_base(gen)
        if base is None:
            raise TrainError(why, 503)
        if not rec["images"]:
            raise TrainError("images: none to train on")
        missing = [i["file"] for i in rec["images"] if not i["caption"].strip()]
        if missing:
            raise TrainError(f"captions: {len(missing)} image(s) have none ({missing[0]}...): caption them first")
        if not any(i["caption"].startswith(rec["trigger"]) for i in rec["images"]):
            raise TrainError(f"captions: none starts with the trigger word {rec['trigger']!r}")
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

    def resume(self, jid: str) -> dict:
        rec = self._need(jid, ("interrupted", "cancelled", "awaiting_choice"))
        if rec["status"] == "awaiting_choice":
            return rec                          # its versions are already here to pick
        if not rec.get("settings"):
            raise TrainError("the job was never started: start it")
        rec.update(status="queued", error=None)
        self._write(rec)
        self._enqueue(jid)
        return rec

    def delete(self, jid: str) -> dict:
        rec = self.get(jid)
        if rec is None:
            raise TrainError(f"{jid}: no such job", 404)
        if rec["status"] in ("queued", "preprocessing", "training", "sampling", "captioning"):
            self.cancel(jid)
            for _ in range(120):
                time.sleep(0.1)
                cur = self.get(jid)
                if cur is None or cur["status"] not in ("queued", "preprocessing", "training", "sampling",
                                                        "captioning"):
                    break
        shutil.rmtree(self.dir_of(jid), ignore_errors=True)
        return {"id": jid, "deleted": True}

    def delete_version(self, jid: str, step: int) -> dict:
        rec = self._need(jid, ("awaiting_choice", "cancelled", "interrupted", "completed"))
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
        rec = self._need(jid, ("awaiting_choice", "cancelled", "interrupted", "completed"))
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
        tmp = self.dir_of(jid) / "register.part"
        shutil.copyfile(src, tmp)
        try:
            added = lora_api.add(folders, tmp, stem + ".safetensors")
        except lora_api.LoraError as e:
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
                rec.update(status="cancelled", completed_at=int(time.time()), error=None)
                self._write(rec)          # its versions stay, to pick and register
            except Exception as e:   # noqa: BLE001 - the job's failure, said in its record
                rec.update(status="failed", completed_at=int(time.time()),
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
        rec.update(status="sampling", stage="samples")
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
               "--pre", str(Path(gen.cfg["exe"]).parent / "xestrata-image-mf-pre")]
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
                "--seed", str(s["seed"]), "--kind", rec["kind"],
                "--caption-dropout", str(s["dropout"])]
        if fam == "mage-flow":
            argv += ["--steps", str(s["steps"] * s["accum"]), "--accum", str(s["accum"]),
                     "--checkpoint", str(s["save_every"]), "--method", s["method"],
                     "--trigger", rec["trigger"], "--title", rec["name"], "--base", s["base"]]
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
        if resume and (d / "state.bin").is_file():
            argv += ["--resume", str(d / "state.bin")]
        return argv

    def _state(self, rec: dict) -> dict:
        """The current device room and trainer report in train_budget.Decider's input format."""
        reports = devices(self.svc)
        current = str((self.place or {}).get("pci") or "")
        free = {str(d["pci"]).lower(): int(d.get("free_bytes") or 0) >> 20 for d in reports}
        device_free = dict(free)
        # The device's free count already excludes this trainer. Its budget can reuse what it holds.
        if current in free:
            free[current] += int(self._train_state.get("used_mib", 0))
        _, helper, held = text_placement(self.svc)
        status = getattr(self.svc, "status", {})
        train = dict(self._train_state)
        with self.lock:
            needs = list(self.image_needs.values())
            image_seq = self.image_seq
        known = [n for n in needs if n is not None]
        image_need = max(known) if needs and len(known) == len(needs) else None
        return {"free": free, "device_free": device_free,
                "proc_alive": self._proc is not None and self._proc.poll() is None,
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

    def _write_control(self, d: Path, budget: int, save_state: bool) -> None:
        tmp = d / "vram.json.tmp"
        tmp.write_text(json.dumps({"budget_mib": int(budget), "save_state": bool(save_state)}), encoding="utf-8")
        os.replace(tmp, d / "vram.json")

    def _spawn(self, rec: dict, pci: str, budget: int, resume: bool, cancel: threading.Event) -> subprocess.Popen:
        """Start the trainer on the card, after open_mib of temporary room is free (the open reads and quantizes the
        weights even when the budget starts at 0 - TRAINVRAM's note 1)."""
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
        env = card_env(self.svc, pci)
        self._write_control(self.dir_of(rec["id"]), budget, False)
        proc = subprocess.Popen(self.train_argv(rec, budget, resume), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        self._proc = proc
        self._state_saved.clear()
        self._train_state = {}
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
        self._decider = tb.Decider(run_config(self.svc), *self._budget_model(rec), devices=devices(self.svc))
        d = self.dir_of(rec["id"])
        act = self._decider.tick(self._state(rec))
        if not act or not act.get("pci"):
            raise TrainError("no GPU can take the training now (the least VRAM is not free anywhere)", 503)
        pci = str(act["pci"]).lower()
        self.place = {"pci": pci, "budget_mib": int(act.get("budget_mib", 0)),
                      "mode": self._mode(rec, act), "reason": str(act.get("reason", ""))}
        log = open(d / "train.log", "a", encoding="utf-8")
        self._log = log
        proc = None
        try:
            proc = self._spawn(rec, pci, int(act.get("budget_mib", 0)),
                               resume=bool(rec.get("resume_state")), cancel=cancel)
            led = Ledger(img.gpu_key(card_env(self.svc, pci)))
            pending = None
            deadline = 0.0
            while True:
                if cancel.is_set():
                    raise _Cancelled()
                if proc is not None and proc.poll() is not None:
                    break
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
                    self._write_control(d, budget, bool(act.get("save_state")))
                    led.hold("train", max(budget, int(self._train_state.get("used_mib", 0))), proc.pid,
                             exact=False, whole=True)
                    rec["place"] = self.place
                    self._write(rec)
                    if act.get("save_state"):
                        if pending is None:
                            pending, deadline = act, time.monotonic() + OPEN_WAIT_S
                        if self._state_saved.is_set():
                            self._terminate(proc)
                            assert self._reader is not None
                            self._reader.join()
                            self._reader = None
                            self._proc = None
                            self._train_state = {}
                            _DEV_CACHE.clear()  # process exit can return runtime-held VRAM
                            self._scan_versions(rec)
                            pci = card
                            self.place["pci"] = pci
                            proc = None
                            pending = None
                            rec["resume_state"] = True
                            self._write(rec)
                        elif time.monotonic() >= deadline:
                            raise TrainError("the trainer did not save its state in time", 503)
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
        finally:
            if proc is not None and proc.poll() is None:
                self._terminate(proc)
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
        t = TRAINERS[rec["family"]]
        log = getattr(self, "_log", None)
        assert proc.stdout is not None
        for line in proc.stdout:
            if log is not None:
                log.write(line)
                log.flush()
            m = t["progress"].match(line)
            if m:
                g = m.groups()
                step = int(g[0])
                rec["progress"] = {"step": step, "total": rec["settings"]["steps"], "loss": float(g[1]),
                                   "s_per_step": float(g[3]) if len(g) > 3 else None, "eta_s": self._eta(rec, step)}
                self._scan_versions(rec)
                self._write(rec)
                continue
            tok = line.split()
            if tok and tok[0] == "vram" and len(tok) >= 5 and tok[4] in ("running", "paused"):
                with contextlib.suppress(ValueError):
                    _DEV_CACHE.clear()  # the pause/grow changed free VRAM; the policy needs a fresh report
                    self._train_state = {"used_mib": int(tok[1]), "peak_est_mib": int(tok[2]), "min_mib": int(tok[3]),
                                         "state": tok[4]}
                    self.wake.set()
                continue
            if tok and tok[0] == "state" and len(tok) >= 3 and tok[1] == "saved":
                self._state_saved.set()
                self.wake.set()
        with contextlib.suppress(Exception):
            proc.stdout.close()
        self._scan_versions(rec)

    def _eta(self, rec: dict, step: int) -> int | None:
        sp = (rec.get("progress") or {}).get("s_per_step")
        if not sp:
            started = rec.get("started_at") or int(time.time())
            sp = (time.time() - started) / step if step else None
        return int(sp * (rec["settings"]["steps"] - step)) if sp else None

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

    def _terminate(self, proc: subprocess.Popen) -> None:
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            proc.wait(10)

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
    def image_begin(self, need_mib: int | None = None) -> str | None:
        """Register an image's need until image_end; unknown need requires the Decider's safe stop."""
        with self.lock:
            if self.current is None or self.place is None:
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
