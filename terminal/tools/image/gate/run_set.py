"""tools/image/gate/run_set.py - TEST-ONLY (OWN-5 gate harness): run sd-cli for a set of prompt x seed pictures with exactly the command Strata's server runs
(serve/images.py sd_cli_args + image_env from the repo copy), one process per picture, and record time and memory.

    python3 -I -B run_set.py <root> <backend> <tag> <W>x<H> <quant> <models_dir> <prompt-indices e.g. 0-7> <seeds e.g. 42,1234>
                             [--native] [--noise] [--threads N] [--min-free-gb N] [--steps N] [--build NAME]
                             [--dump] [--cond-from TAG] [--latent-from TAG] [--noise-from DIR] [--exit-after K]
                             [--env K=V ...] [--extra-args "sd-cli args"] [--model-arg K=V ...]
--noise: the config's images "noise" is "strata" (--model-args strata_noise=1); the build must have the generator.
--build NAME uses <root>/build-NAME.json (e.g. a stage_hooks.py variant); --dump/--cond-from/--latent-from/
--noise-from/--exit-after drive that variant's hooks (OWN5_* variables, see stage_hooks.py), per picture.

Writes <root>/pics/<tag>/p<i>_s<seed>.png, .log per picture and <root>/pics/<tag>/results.jsonl.
Memory: peak RSS of sd-cli (wait4), GPU memory sampled every 0.2 s (nvidia-smi on NVIDIA with dedicated memory,
amdgpu sysfs mem_info_vram_used on AMD; on unified-memory boxes the minimum MemAvailable is recorded instead)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROMPTS = [   # the I1/I2 gate set (p2006 ~/qimage/i2/qi_gate.py PROMPTS)
    "a neon-lit ramen shop in a rainy Tokyo alley at night, reflections on wet pavement, cinematic",
    "macro photograph of a honeybee on a sunflower, visible pollen grains, extreme detail",
    "a chalkboard in a classroom with the words \"Photosynthesis: light + water + CO2\" written in white chalk",
    "a medieval knight in ornate silver armor standing in a misty forest at dawn, oil painting",
    "a cartoon sloth mascot waving, flat vector illustration, bright colours",
    "portrait of an elderly fisherman with a weathered face and a knitted cap, soft window light, 85mm photo",
    "a retro travel poster of a mountain lake with the text \"VISIT LAKE BLED\" in bold art deco letters",
    "a still life of lemons, a glass carafe of water and a blue ceramic bowl on a linen tablecloth, morning light",
]


def gpu_mem_sampler(stop: threading.Event, out: dict):
    nv = shutil.which("nvidia-smi")
    amd = [p for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")]
    peak, base, minavail, startavail = None, None, None, None
    while not stop.is_set():
        v = None
        if nv:
            try:
                r = subprocess.run([nv, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                v = sum(int(x) for x in r.stdout.split() if x.strip().isdigit()) or None
            except Exception:
                v = None
        elif amd:
            try:
                v = max(int(p.read_text()) for p in amd) >> 20
            except Exception:
                v = None
        if v is not None:
            base = v if base is None else base
            peak = v if peak is None else max(peak, v)
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith("MemAvailable:"):
                    a = int(ln.split()[1]) >> 10
                    minavail = a if minavail is None else min(minavail, a)
                    startavail = a if startavail is None else startavail
        except OSError:
            pass
        stop.wait(0.2)
    out.update(gpu_mib_start=base, gpu_mib_peak=peak, memavail_mib_start=startavail, memavail_mib_min=minavail)


def main() -> int:
    a = sys.argv[1:]
    root, backend, tag, size, quant, models, pidx, seeds = a[:8]
    root = Path(root).expanduser().resolve()
    native = "--native" in a
    threads = a[a.index("--threads") + 1] if "--threads" in a else None
    min_free = float(a[a.index("--min-free-gb") + 1]) if "--min-free-gb" in a else 10.0
    steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 8
    lo, _, hi = pidx.partition("-")
    prompts = list(range(int(lo), int(hi or lo) + 1))
    seed_list = [int(s) for s in seeds.split(",")]
    w, h = (int(x) for x in size.lower().split("x"))
    repo = root / "repo"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tools" / "image"))
    import qimage as QI
    from serve import images as IM
    bname = a[a.index("--build") + 1] if "--build" in a else backend      # another build of the same backend
    build = json.loads((root / f"build-{bname}.json").read_text())
    md = Path(models).expanduser()
    cfg = QI.image_config(backend, quant, Path(build["exe"]), md.parent if md.name == QI.MODEL_DIR else md,
                          native=native, noise="--noise" in a)
    if md.name != QI.MODEL_DIR:   # files sit directly in models_dir
        for k, name in (("diffusion_model", quant), ("vae", "vae"), ("llm", "llm")):
            cfg[k] = str(md / Path(QI.FILES[name][1]).name)
    cfg["lib_dirs"] = [d for d in build.get("env_ld", "").split(os.pathsep) if d]
    if threads:
        cfg["args"] = list(cfg["args"]) + ["-t", threads]
    outd = root / "pics" / tag
    outd.mkdir(parents=True, exist_ok=True)
    base_env = IM.image_env(cfg)
    base_env["PATH"] = build.get("env_path") or base_env["PATH"]
    for i, x in enumerate(a):
        if x == "--env":
            k, _, v = a[i + 1].partition("=")
            base_env[k] = v
    extra_args = a[a.index("--extra-args") + 1].split() if "--extra-args" in a else []
    exit_after = a[a.index("--exit-after") + 1] if "--exit-after" in a else None
    for pi in prompts:
        for seed in seed_list:
            if shutil.disk_usage(root).free / 1e9 < min_free:
                raise SystemExit("ABORT: disk under the floor")
            job = {"prompt": PROMPTS[pi], "width": w, "height": h, "n": 1, "steps": steps, "seed": seed}
            work = outd / f"_work_p{pi}_s{seed}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
            pf = IM.write_prompt(work, job["prompt"])
            cmd = IM.sd_cli_args(cfg, job, str(work / "img_%d.png"), str(pf)) + extra_args
            for i, x in enumerate(a):                       # --model-arg K=V: into the one --model-args string
                if x == "--model-arg":
                    if "--model-args" in cmd:
                        j = cmd.index("--model-args") + 1
                        cmd[j] = cmd[j] + "," + a[i + 1]
                    else:
                        cmd += ["--model-args", a[i + 1]]
            env = dict(base_env)
            key = f"p{pi}_s{seed}"
            if "--dump" in a:                                   # stage hooks (stage_hooks.py builds only)
                dd = outd / f"dump_{key}"
                dd.mkdir(exist_ok=True)
                env["OWN5_DUMP_DIR"] = str(dd)
            for opt, var, fname in (("--cond-from", "OWN5_COND_IN", "cond.bin"),
                                    ("--latent-from", "OWN5_LATENT_IN", "latent.bin")):
                if opt in a:
                    env[var] = str(root / "pics" / a[a.index(opt) + 1] / f"dump_{key}" / fname)
            if "--noise-from" in a:
                env["OWN5_NOISE_DIR"] = str(Path(a[a.index("--noise-from") + 1]).expanduser())
            if exit_after is not None:
                env["OWN5_EXIT_AFTER"] = exit_after
            stop, mem = threading.Event(), {}
            th = threading.Thread(target=gpu_mem_sampler, args=(stop, mem), daemon=True)
            th.start()
            t0 = time.monotonic()
            first_step = None
            logp = outd / f"p{pi}_s{seed}.log"
            with open(logp, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, cwd=str(work), text=True, errors="replace", bufsize=1)
                for line in proc.stdout:
                    lf.write(line)
                    if first_step is None and IM.STEP_LINE.search(line):
                        first_step = time.monotonic() - t0
                _, status, ru = os.wait4(proc.pid, 0)
            secs = time.monotonic() - t0
            stop.set()
            th.join()
            rc = os.waitstatus_to_exitcode(status)
            log = logp.read_text(errors="replace")
            rec = {"tag": tag, "backend": backend, "native": native, "quant": quant, "size": size, "steps": steps,
                   "prompt": pi, "seed": seed, "rc": rc, "seconds": round(secs, 2),
                   "first_step_s": round(first_step, 2) if first_step else None, "maxrss_mib": ru.ru_maxrss >> 10,
                   "threads": threads, **mem}
            for key, pat in (("text_encode_s", r"get_learned_condition completed, taking ([0-9.]+) ?s"),
                             ("sampling_s", r"sampling completed, taking ([0-9.]+) ?s"),
                             ("vae_decode_s", r"decode_first_stage completed, taking ([0-9.]+) ?s")):
                m = re.findall(pat, log)
                if m:
                    rec[key] = float(m[-1])
            png = work / "img_0.png"
            if rc == 0 and png.is_file():
                shutil.move(str(png), str(outd / f"p{pi}_s{seed}.png"))
            elif rc == 0 and exit_after is not None:
                rec["note"] = f"stopped after model call {exit_after} (stage run, no picture)"
            else:
                rec["error"] = "\n".join(log.strip().splitlines()[-8:])
            shutil.rmtree(work, ignore_errors=True)
            with open(outd / "results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    (outd / "cmd.txt").write_text(" ".join(cmd) + "\n")
    print("RUN-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
