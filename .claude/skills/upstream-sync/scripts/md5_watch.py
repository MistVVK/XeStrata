# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The answers' md5 of one build: the watch of a change that must leave the answers as they are (tools/bench_md5.py).

    DATA=DIR XE="CMD..." [XE_ROOT=DIR] [XE_PROFILE=P] [EXTRA="..."] \\
      python .claude/skills/upstream-sync/scripts/md5_watch.py OUT GPU [--dry-run]

A cell is one model at one length, or one image model at one size.  An LLM's cell: its fixed prompt
(<data>/ids/release/<tier>.ids, sync_prompts.py's; a model of another vocabulary its own, OWN_IDS below), the
settings that decide its answer (greedy, --adapt-every 0, --pcie-frac 0, --prefill-cpu-share 0, the context setup
writes for that length) and 32 tokens through the engine; the
md5 is of the answer's line, and the logits those 32 tokens were chosen from are kept beside it (--dump-logits: the KL
of a cell that moved).
An image model's cell: one image from its fixed prompt, IMAGE_SIZE, IMAGE_STEPS and IMAGE_SEED; the md5 is of the PNG.

XE: the text engine (its binary, or `env K=V ...` around it, as in the merge's scripts).  XE_ROOT: the tree the image
models' code runs from (default this tree).  XE_IMAGE_EXE and XE_LIB_DIRS: xestrata-image and its runtime's folders
(a build folder; default <XE_ROOT>/engine/xestrata-image with its BUILD.json).  XE_IMAGE_PCI: the GPU's PCI address
(XESTRATA_GPU_PCI, as setup writes it).  PY: the Python that runs the tree's image code (default this one).  QUANT:
the image denoiser's quant (default Q8_0).  XE_PROFILE: the shipped expert profile, as the merge's runs pass it.
EXTRA: options both engines take besides.  MODELS (default "q38 ornith q36 q35 llmjp"), TIERS (default "1k 32k"),
KINDS (default "llm image"), IMAGE_MODELS (default "qwen-image-turbo mage-flow-turbo moss-image anima-turbo"),
IMAGE_SIZE (default 512x512), IMAGE_STEPS (4), IMAGE_SEED (7), ROUNDS (2: a cell that answers differently from run
to run is one the watch cannot read alone), LOGITS (1: keep the dumps, 32 MB a cell) select what runs.  EXACT=1 runs
the cells with their answers fixed (XESTRATA_NO_BLAS=1 and XESTRATA_EXACT=1 in every cell; Qwen3.8 also --spec-window
min-p --spec-min-p 0 --suffix-draft 0):
the dense products' own/library choice and the verify windows' sizes are measured per start and move the answer's last
bits, and so does the prompt's CPU share where the experts' rows are not in the fixed-point sums (docs/XE.md, "Why
outputs change from run to run").  --dry-run: print the commands and the image configs instead
of running them.

OUT/md5.json (the rows), OUT/md5.txt (a cell a line), OUT/meta.json (the commit, the engines, the runs' arguments) and
the runs' files: answers/, logs/, logits/, images/.  md5_judge.py OUT --prev DIR reads it against the previous record.
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402
import bench_md5 as M  # noqa: E402

TREE = Path(__file__).resolve().parents[4]
# each model's files in the data folder, and the settings that decide its answer: the merge's (--expert-cache auto,
# --prefill auto for the generate engine, the MTP drafter and its 4 drafts); the runs add greedy, --adapt-every 0,
# --pcie-frac 0, --prefill-cpu-share 0 (COMMON) and the tier's context (bench_core's, as setup writes it).
MODELS = {
    "q38": ["--pack", "<data>/packs/qwen-iq3_xxs",
            "--native", "<data>/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
            "--ple-gguf", "<data>/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
            "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
            "--mtp", "<data>/mtp/rt"],
    "ornith": ["--native", "<data>/models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf"],
    "q36": ["--native", "<data>/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"],
    "q35": ["--native", "<data>/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf"],
    "llmjp": ["--native", "<data>/models/llm-jp-4.1-32b-a3b-thinking/llm-jp-4.1-32b-a3b-thinking-Q4_K_M.gguf"],
    # 98 GB of experts: on a 64 GB PC it reads them through mmap, and its prompt path is slow enough that a cell of it
    # costs more than every other cell together (glm53-golden's record).  Not among the default cells.
    "glm53": ["--native", "<data>/models/GLM-5.3-Flash/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
              "--mmap-experts"],
}
# --prefill-cpu-share 0: the prompt's experts all on the GPU, as a fixed share.  A share measured per run (the engine's
# old default) moves between runs, and the CPU's rows are not the GPU's bits: the answer would differ from run to run
# and the watch could not read it (2026-11-13, the B70: 0.53 to 0.83 over five runs).
COMMON = ["--greedy", "--max-new", str(M.NEW), "--adapt-every", "0", "--pcie-frac", "0", "--prefill-cpu-share", "0"]

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
out, gpu = Path(args[0]), args[1]
dry = "--dry-run" in sys.argv
data = Path(B.need("DATA"))
models = os.environ.get("MODELS", "q38 ornith q36 q35 llmjp").split()
tiers = os.environ.get("TIERS", "1k 32k").split()
kinds = os.environ.get("KINDS", "llm image").split()
image_models = os.environ.get("IMAGE_MODELS", "qwen-image-turbo mage-flow-turbo moss-image anima-turbo").split()
rounds = int(os.environ.get("ROUNDS", "2"))
extra = shlex.split(os.environ.get("EXTRA", ""))
profile = os.environ.get("XE_PROFILE", "")
EXACT = os.environ.get("EXACT", "") not in ("", "0")
logits = os.environ.get("LOGITS", "1") != "0"
for name in models:
    if name not in MODELS:
        sys.exit(f"MODELS: {name} is not one of {', '.join(MODELS)}")
for tier in tiers:
    if tier not in B.CONTEXT:
        sys.exit(f"TIERS: {tier} is not one of {', '.join(B.CONTEXT)}")
if set(kinds) - set(M.KINDS):
    sys.exit(f"KINDS: some of {', '.join(M.KINDS)}")


def argv_of(name: str) -> list[str]:
    """The model's arguments: its files in the data folder, and the settings that decide its answer."""
    a = [w.replace("<data>", str(data)) for w in MODELS[name]]
    if EXACT and name == "q38":   # the fixed window, beside the own kernels this cell's runs take
        a += ["--spec-window", "min-p", "--spec-min-p", "0", "--suffix-draft", "0"]
    return a


# a model whose vocabulary is not Qwen's reads the same tiers in its own ids, <data>/ids/release-<name>/<tier>.ids:
# Qwen's ids are out of its vocabulary (llm-jp-4.1's 196,608 against Qwen3.8's 248,069 and more) and every cell of it
# printed no answer.
OWN_IDS = {"llmjp"}


def prompt_ids(name: str, tier: str) -> Path:
    """The tier's prompt for the model: the release ids, or the model's own for another vocabulary."""
    return data / "ids" / (f"release-{name}" if name in OWN_IDS else "release") / f"{tier}.ids"


def llm_cells(rows: list[dict], meta: dict) -> None:
    engine = B.Engine.from_spec("xe", "xe", B.need("XE"), profile)
    meta["llm"] = {"engine": os.environ["XE"], "common": COMMON, "extra": extra, "profile": profile, "exact": EXACT,
                   "context": {t: B.CONTEXT[t] for t in tiers},
                   "models": {n: [*argv_of(n), *(["--expert-profile", profile] if profile else [])] for n in models}}
    for folder in ("answers", "logs", "logits"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name in models:
        argv = [*meta["llm"]["models"][name], *COMMON]
        for tier in tiers:
            ids = prompt_ids(name, tier)
            if not ids.exists():
                sys.exit(f"{ids}: the tier's prompt (sync_prompts.py writes it"
                         + ("; this model's own vocabulary's ids" if name in OWN_IDS else "") + ")")
            for rnd in range(1, rounds + 1):
                stem = f"{name}-{tier}-r{rnd}"
                answer, log = out / "answers" / f"{stem}.out", out / "logs" / f"{stem}.err"
                dump = out / "logits" / f"{stem}.f32"
                cmd = [*engine.cmd, *engine.run_args(
                    [*argv, *B.CONTEXT[tier], *extra, "--tokens-file", str(ids)]
                    + (["--dump-logits", str(dump)] if logits else []))]
                if dry:
                    print(f"DRY {shlex.join(cmd)}")
                    continue
                t0, rc = time.monotonic(), None
                env = engine.env
                if EXACT:   # the plain engine's dense products choose own/library per start as well; the experts' rows
                    env = {**env, "XESTRATA_NO_BLAS": "1", "XESTRATA_EXACT": "1"}   # in the fixed-point sums
                with open(answer, "wb") as f_out, open(log, "wb") as f_log:
                    f_log.write(f"# {shlex.join(cmd)}\n".encode())
                    if env is not engine.env:
                        f_log.write("# XESTRATA_NO_BLAS=1 XESTRATA_EXACT=1\n".encode())
                    try:
                        rc = subprocess.run(cmd, cwd=engine.cwd, env=env, stdin=subprocess.DEVNULL,
                                            stdout=f_out, stderr=f_log, timeout=B.DEADLINE).returncode
                    except subprocess.TimeoutExpired:
                        pass
                md5, _ = M.answer_file(answer)
                failed = "" if rc == 0 else f", exit {rc if rc is not None else 'timeout'}"
                note = f"{time.monotonic() - t0:.0f} s{failed}"
                row = {"kind": "llm", "model": name, "tier": tier, "round": rnd, "md5": md5, "rows": M.NEW,
                       "file": str(answer.relative_to(out)), "note": note}
                if logits:
                    row["logits"] = str(dump.relative_to(out))
                    row["bits"] = hashlib.md5(dump.read_bytes()).hexdigest() if dump.exists() else None
                rows.append(row)
                M.write(out, rows, meta)
                print(f"{name:7s} {tier:4s} r{rnd} {md5[:8] if md5 else 'no answer':9s} {note}", flush=True)
                if md5 is None:
                    print(f"  {log}: {B.engine_error(log.read_text(errors='replace'))}", flush=True)


def image_cells(rows: list[dict], meta: dict) -> None:
    root = Path(os.environ.get("XE_ROOT") or TREE)
    exe = Path(os.environ.get("XE_IMAGE_EXE") or
               root / "engine" / ("xestrata-image.exe" if os.name == "nt" else "xestrata-image"))
    if os.environ.get("XE_IMAGE_EXE"):
        lib_dirs = [d for d in os.environ.get("XE_LIB_DIRS", "").split(os.pathsep) if d]
    else:
        try:
            lib_dirs = json.loads((exe.parent / "BUILD.json").read_text()).get("lib_dirs") or []
        except (OSError, ValueError) as err:
            sys.exit(f"xe: {exe.parent / 'BUILD.json'}: {err} (the engine its setup built; or XE_IMAGE_EXE and "
                     "XE_LIB_DIRS)")
    env = {}
    if lib_dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(lib_dirs + ([os.environ[var]] if os.environ.get(var) else []))
    if os.environ.get("XE_IMAGE_PCI"):
        env["XESTRATA_GPU_PCI"] = os.environ["XE_IMAGE_PCI"]
    size = os.environ.get("IMAGE_SIZE", "512x512")
    meta["image"] = {"root": str(root), "exe": str(exe), "quant": os.environ.get("QUANT", "Q8_0"), "size": size,
                     "steps": int(os.environ.get("IMAGE_STEPS", "4")), "seed": int(os.environ.get("IMAGE_SEED", "7")),
                     "prompt": B.IMAGE_REQUEST["prompt"], "env": env, "configs": {}}
    engine = B.Engine("xe", "xe", [], "", root=root, py=os.environ.get("PY") or sys.executable)
    for folder in ("images", "logs"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for fam in image_models:
        cfg = B.image_config(engine, "xestrata-image", meta["image"]["quant"], str(exe), str(data / "models"),
                             model=fam)
        if "error" in cfg:
            sys.exit(f"{fam}: its image config: {cfg['error']}")
        meta["image"]["configs"][fam] = cfg
        if dry:
            print(f"DRY image {fam} {size}, {meta['image']['steps']} steps, seed {meta['image']['seed']}: {cfg}")
            continue
        for rnd in range(1, rounds + 1):
            stem = f"{fam}-{size}-r{rnd}"
            png = out / "images" / f"{stem}.png"
            r = B.image_job(engine, {"do": "run", "cfg": cfg, "png": str(png), "env": env,
                                     "log": str(out / "logs" / f"{stem}.txt"),
                                     "request": {"prompt": B.IMAGE_REQUEST["prompt"], "size": size,
                                                 "steps": meta["image"]["steps"], "seed": meta["image"]["seed"]}})
            md5 = hashlib.md5(png.read_bytes()).hexdigest() if png.exists() else None
            note = (f"{r['seconds']} s (sampling {r['sampling_s']} s, VAE {r['decode_s']} s)" if r.get("rc") == 0
                    else f"failed: {r.get('error') or r.get('rc')}")
            rows.append({"kind": "image", "model": fam, "tier": size, "round": rnd, "md5": md5,
                         "file": str(png.relative_to(out)), "note": note})
            M.write(out, rows, meta)
            print(f"{fam:20s} {size:5s} r{rnd} {md5[:12] if md5 else 'no image':13s} {note}", flush=True)


rows: list[dict] = []
meta: dict = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu": gpu, "data": str(data), "tiers": tiers,
              "rounds": rounds, "new": M.NEW, "models": models, "kinds": kinds, "logits": logits}
try:
    meta["commit"] = B.git(TREE, "rev-parse", "HEAD").strip()
except (OSError, subprocess.SubprocessError):
    meta["commit"] = ""
if "llm" in kinds:
    llm_cells(rows, meta)
if "image" in kinds:
    image_cells(rows, meta)
if dry:
    sys.exit(0)
M.write(out, rows, meta)
print(f"== {out}/md5.txt")
print(M.table(rows))
