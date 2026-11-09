# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Setup's --bench: XeStrata's speed on this PC, measured as the upstream merges measure it (tools/bench_core.py:
generate, serve and everyday use, two rounds each by default, and image generation where the config has it), and
what a developer needs to read the numbers: the machine, the paths the engine chose, where the time goes.  With
--original-strata TAG, upstream Strata's release TAG runs beside it on the same model files, the two taking turns.

The record goes to <data>/records/bench-<time>/ in the upstream merges' layout (runs/, serve-runs/, usage-runs/,
matrix*.json, which their sync_collect.py and judge.py read), with record.json (everything, this PC's names taken
out) and summary.md (to paste into a report).  Nothing is sent anywhere.

    python tools/bench.py ref CONFIG                   # data/bench-ref/<model>.csv: this machine's 1k answer
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_core as C  # noqa: E402
from bench_core import grab  # noqa: E402

ROOT = C.ROOT
# a model's answer to the 1k prompt on the development machine (`python tools/bench.py ref CONFIG`), which
# --bench-verbose follows to count the tokens this PC's argmax differs on
REFS = ROOT / "data" / "bench-ref"
TIERS = ("64k", "32k", "1k")          # the upstream merges' IQ3_XXS tiers (the longest first), as this config holds
# --bench-verbose's extra run per tier: the engine's per-stage timings (kept out of the measured runs)
STATS_ENV = {"XESTRATA_PREFILL_TIMING": "1", "XESTRATA_DECODE_TIMING": "1"}
GPU_LABEL = "PC"                      # the record's GPU name (the merges name their machines' GPUs)


# ------------------------------------------------------------------------------------------------ the engine's args
def arg_of(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def without(args: list[str], valued=(), bare=()) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in valued:
            i += 2
        elif args[i] in bare:
            i += 1
        else:
            out.append(args[i])
            i += 1
    return out


def context_of(args: list[str]) -> int:
    return int(arg_of(args, "--max-context") or 0)


def fits(tier: str, ctx: int) -> bool:
    return int(C.CONTEXT[tier][1]) <= ctx


def is_split(cfg: dict) -> bool:
    """A layer split across GPUs (setup --gpus), which only the serve loop runs."""
    return bool(cfg.get("split_pci")) or (isinstance(cfg.get("gpu"), list) and len(cfg["gpu"]) > 1)


# ------------------------------------------------------------------------------------------------ reading the logs
def parse_generate(text: str) -> dict:
    """One generate run's figures: bench-collect.py's (the same names) and the decode round's breakdown."""
    r = {"prompt_tokens": grab(r"^prefill\s+(\d+) tokens", text, int),
         "prefill_tok_s": grab(r"^prefill\s+.*->\s+([\d.]+) tok/s", text),
         "ttft_ms": grab(r"^prefill\s+.*time to first token ([\d.]+) ms", text),
         "decoded": grab(r"^decode\s+(\d+) tokens", text, int),
         "decode_ms": grab(r"^decode\s+\d+ tokens in ([\d.]+) ms", text),
         "decode_tok_s": grab(r"^decode\s+.*->\s+([\d.]+) tok/s", text),
         "spec_accept": grab(r"^speculation\s+.*\(([\d.]+)\)", text),
         "rounds": grab(r"^speculation\s+(\d+) rounds", text, int),
         "follow_differ": grab(r"^follow\s+(\d+) of \d+ emitted tokens differ", text, int),
         "follow_emitted": grab(r"^follow\s+\d+ of (\d+) emitted tokens differ", text, int)}
    pp = re.findall(r"^PP (\d+) (\d+) \d+ ([\d.]+)$", text, re.M)
    if pp and r["prefill_tok_s"] is None and pp[-1][0] == pp[-1][1]:   # stopped after the prompt
        r["prefill_tok_s"] = float(pp[-1][2])
    m = re.search(r"^verify window\s+wait for rings ([\d.]+)\s+pool ([\d.]+)\s+host ([\d.]+)\s+commit ([\d.]+) "
                  r"ms/round; CPU experts ([\d.]+) distinct / ([\d.]+) routed", text, re.M)
    if m:
        r["round"] = dict(zip(("gpu_wait_ms", "pool_ms", "host_ms", "commit_ms", "cpu_experts_distinct",
                               "cpu_experts_routed"), map(float, m.groups())))
        r["round"]["pool_gb_s"] = grab(r"^pool multi\s+.*?; ([\d.]+) GB/s over the rows phases", text)
        r["round"]["mtp_ms"] = grab(r"^mtp\s+([\d.]+) ms/round drafting", text)
        r["round"]["pcie_experts"] = grab(r"^pcie experts\s+([\d.]+) distinct experts per layer", text)
        r["round"]["adapt_ms"] = grab(r"^adaptive tier\s+.*, ([\d.]+) ms/round\)", text)
        r["round"]["major_faults"] = grab(r"^host memory\s+(\d+) major page faults", text, int)
    m = re.search(r"prefill \d+ tokens in (\d+) chunks, [\d.]+ ms \([\d.]+ tok/s\); experts streamed (\d+) "
                  r"\((\d+) by DMA, host ([\d.]+) ms\), resident (\d+), on the CPU (\d+) \(share ([\d.]+)\); "
                  r"PLE ([\d.]+) ms", text)
    if m:
        r["prompt"] = dict(zip(("chunks", "experts_streamed", "experts_dma", "host_ms", "experts_resident",
                                "experts_cpu", "cpu_share", "ple_ms"),
                               [int(m[1]), int(m[2]), int(m[3]), float(m[4]), int(m[5]), int(m[6]), float(m[7]),
                                float(m[8])]))
    out = re.search(r"^output\s*:\s*([\d ]+)$", text, re.M)
    if out:
        ids = out.group(1).split()
        r["output_sha256_64"] = hashlib.sha256(" ".join(ids[:64]).encode()).hexdigest()[:16]
    r["engine_lines"] = engine_lines(text)
    return r


def engine_lines(text: str) -> list[str]:
    """What the engine said about itself before its first request: the GPU, the paths it chose, the cache, the
    workers (the lines up to "session is up", or up to READY)."""
    out = []
    for line in text.splitlines():
        if line.startswith(("READY", "PP ", "T ")) or "session is up" in line:
            if "session is up" in line:
                out.append(line)
            break
        if line.startswith(("xestrata", "strata", "INFO ")) or line.startswith(" " * 8):   # upstream's engine too
            out.append(line.rstrip())
    return out


# ------------------------------------------------------------------------------------------------ running the engine
def busy_engines() -> list[str]:
    """Strata engines or servers running now (any user's), which would share the GPU with the measurements."""
    found = []
    for pid, words in command_lines():
        if pid == os.getpid() or not words:
            continue
        exe = os.path.basename(words[0])
        engine = exe.startswith(("xestrata", "strata"))   # XeStrata's or upstream's
        if (engine and "--serve" in words) or \
                any(w.replace("\\", "/").endswith("serve/server.py") for w in words):
            found.append(f"{pid} {' '.join(words[:3])}")
    return found


def command_lines() -> list[tuple[int, list[str]]]:
    """Every process's id and command line: /proc on Linux, psutil on Windows (#48)."""
    if C.WIN:
        import psutil
        return [(q.info["pid"], q.info["cmdline"] or []) for q in psutil.process_iter(["pid", "cmdline"])]
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        out.append((int(d.name), [w.decode("utf-8", "replace") for w in argv if w]))
    return out


# ------------------------------------------------------------------------------------------------ the machine
class Scrub:
    """Takes this PC's names out of what the record keeps: the data folder, the repository, the home folder, the
    host and the user."""

    def __init__(self, data: Path):
        self.pairs = [(str(data), "<data>"), (str(ROOT), "<repo>"), (str(Path.home()), "~")]
        for name, mark in ((socket.gethostname(), "<host>"), (getpass.getuser(), "<user>")):
            if name and len(name) > 2:
                self.pairs.append((name, mark))

    def __call__(self, x):
        if isinstance(x, str):
            for a, b in self.pairs:
                x = x.replace(a, b)
            return x
        if isinstance(x, list):
            return [self(v) for v in x]
        if isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        return x


def read(path, default=None):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return default


def package_version(*names: str) -> dict:
    """The installed versions of these distribution packages (dpkg or rpm), the ones found."""
    out = {}
    for n in names:
        for cmd in (["dpkg-query", "-W", "-f", "${Version}", n], ["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", n]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0 and r.stdout.strip():
                out[n] = r.stdout.strip()
                break
    return out


def drivers(vendors: set) -> dict:
    out: dict = {"kernel": platform.release()}
    if "intel" in vendors:
        out["intel"] = package_version("libze-intel-gpu1", "intel-level-zero-gpu", "intel-opencl-icd", "libze1",
                                       "intel-compute-runtime", "oneapi-level-zero")
    if "nvidia" in vendors:
        v = read("/proc/driver/nvidia/version", "")
        m = re.search(r"Kernel Module\s+(?:for \S+\s+)?([\d.]+)", v or "")
        out["nvidia"] = m.group(1) if m else None
    if "amd" in vendors:
        # ROCm 7.14 keeps it in /opt/rocm/core-<version> only
        rocm = read("/opt/rocm/.info/version") or next(
            (v for p in sorted(Path("/opt/rocm").glob("core-*/.info/version")) if (v := read(p))), None)
        out["amd"] = {"rocm": rocm, **package_version("rocm-core", "hip-runtime-amd")}
    return out


CPU_ISA = ("avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi",
           "avx512_bf16", "amx_tile")      # the CPUID features the CPU kernels choose by (cpu_avx512_ok and the like)


def cpu_facts() -> dict:
    if sys.platform == "win32":  # #48: no /proc; the name from the registry, the ISA not read (the engine's log has it)
        import psutil
        import winreg
        name: str | None
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            name = platform.processor() or None
        return {"name": name, "isa": [], "threads": os.cpu_count(), "cores": psutil.cpu_count(logical=False),
                "hybrid": None, "numa_nodes": None, "sockets": None}
    txt = read("/proc/cpuinfo", "") or ""
    flags = set((grab(r"^flags\s*:\s*(.*)$", txt, str) or "").split())
    cores = {(m[0], m[1]) for m in re.findall(r"^physical id\s*:\s*(\d+)\s*$[\s\S]*?^core id\s*:\s*(\d+)", txt, re.M)}
    hybrid = {k: read(f"/sys/devices/{k}/cpus") for k in ("cpu_core", "cpu_atom") if Path(f"/sys/devices/{k}").exists()}
    return {"name": grab(r"^model name\s*:\s*(.*)$", txt, str),
            "isa": sorted(f for f in flags if f in CPU_ISA),
            "threads": os.cpu_count(), "cores": len(cores) or None, "hybrid": hybrid or None,
            "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))) or None,
            "sockets": len({c[0] for c in cores}) or None}


def storage_of(path: Path) -> dict:
    """The kind of disk a folder is on (NVMe, SATA SSD, HDD) and its model, from sysfs (Linux only: none on Windows)."""
    if C.WIN:
        return {}
    try:
        st = os.stat(path)
        dev = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
    except OSError:
        return {}
    disk = dev if (dev / "device").exists() else dev.parent
    rot = read(disk / "queue" / "rotational")
    kind = "NVMe" if disk.name.startswith("nvme") else "HDD" if rot == "1" else "SSD" if rot == "0" else None
    return {"kind": kind, "model": read(disk / "device" / "model"), "fs": fs_type(path)}


def fs_type(path: Path) -> str | None:
    best = None
    for line in (read("/proc/self/mounts", "") or "").splitlines():
        f = line.split()
        if len(f) > 2 and str(path).startswith(f[1]) and (best is None or len(f[1]) > len(best[0])):
            best = (f[1], f[2])
    return best[1] if best else None


BUILD_KEYS = ("version", "license", "backend", "source", "package", "compiler", "cuda_archs", "hip_archs")


def environment(cfg: dict, gpus: list[dict], data: Path) -> dict:
    exe = Path(cfg["exe"])
    try:
        build = json.loads((exe.parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        build = {}
    if C.WIN:
        import psutil
        os_name = f"Windows {platform.release()} ({platform.version()})"
        ram = psutil.virtual_memory().total
    else:
        os_name = grab(r'^PRETTY_NAME="?([^"\n]*)', read("/etc/os-release", "") or "", str)
        ram = int(grab(r"^MemTotal:\s+(\d+)", read("/proc/meminfo", "") or "", int) or 0) * 1024
    env = {k: v for k, v in {**os.environ, **{str(k): str(v) for k, v in (cfg.get("env") or {}).items()}}.items()
           if k.startswith(("XESTRATA_", "STRATA_",   # and upstream's names
                            "ONEAPI_", "SYCL_", "UR_", "ZE_", "CUDA_", "HIP_", "ROCR_", "HSA_"))}
    args = cfg.get("args", [])
    return {
        "xestrata": {k: build.get(k) for k in BUILD_KEYS if k in build},
        "gpus": [{k: g.get(k) for k in ("index", "name", "vendor", "vram_gb", "bar_gb", "link", "pci", "integrated",
                                        "gfx", "uma_gb") if k in g} for g in gpus],
        "drivers": drivers({g.get("vendor") for g in gpus}),
        "cpu": cpu_facts(),
        "ram_gib": round(ram / 2**30, 1),
        "os": os_name,
        "system": platform.system(),
        "data_disk": storage_of(data),
        "model": {"name": cfg.get("model_name"), "context": context_of(args), "kv": arg_of(args, "--kv"),
                  "images": "--vision" in args, "parallel": cfg.get("parallel"),
                  "split": is_split(cfg)},
        "args": args,
        "env": env,
    }


def model_files(args: list[str]) -> list[str]:
    return [x for f in ("--pack", "--native") if (v := arg_of(args, f)) is not None for x in (f, v)]


def hw_bench(exe: str, env: dict, cwd: str | None, args: list[str], log: Path) -> dict:
    """The engine's measurement of this machine (xestrata-engine --hw-bench): the RAM's read bandwidth with the expert
    pool's threads and with one, the copies over PCIe each way, the SSD under the model.  `HW key value` lines."""
    cmd = [exe, "--hw-bench", *model_files(args)]
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    log.write_text("# " + shlex.join(cmd) + "\n" + r.stdout + r.stderr, encoding="utf-8")
    out: dict = {}
    for k, v in re.findall(r"^HW (\S+) (.+)$", r.stdout, re.M):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    if not out:
        out["error"] = C.engine_error(r.stdout + r.stderr) or f"the engine ended with {r.returncode}"
    return out


# the paths the kernels are measured on: the one the engine takes here, then the device without its matrix engines,
# and without their BF16 combinations (the switches AGENTS.md checks a small card with)
KERNEL_VARIANTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("default", {}), ("no-xmx", {"XESTRATA_NO_XMX": "1"}), ("no-bf16-mma", {"XESTRATA_NO_BF16_MMA": "1"}))


def kernel_bench(exe: str, env: dict, cwd: str | None, pack: str | None, log: Path) -> list[dict]:
    """The prompt path's products on synthetic data of the model's shapes (xestrata-engine --kernel-bench, `KB <kernel>
    <path> <shape> <us>` lines), once on each of KERNEL_VARIANTS; "default" is the path the engine takes on this
    device."""
    rows: list[dict] = []
    with open(log, "w", encoding="utf-8") as f:
        for variant, extra in KERNEL_VARIANTS:
            cmd = [exe, "--kernel-bench"] + (["--pack", pack] if pack else [])
            try:
                r = subprocess.run(cmd, cwd=cwd, env={**env, **extra}, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.TimeoutExpired) as e:
                rows.append({"variant": variant, "error": str(e)})
                continue
            f.write(f"# {variant}: {shlex.join(cmd)}\n{r.stdout}{r.stderr}")
            got = [{"variant": variant, "kernel": m[0], "path": m[1], "shape": m[2],
                    "us": float(m[3]) if m[3] != "-" else None}
                   for m in re.findall(r"^KB (\S+) (\S+) (\S+) (\S+)$", r.stdout, re.M)]
            rows += got or [{"variant": variant,
                             "error": C.engine_error(r.stdout + r.stderr) or f"the engine ended with {r.returncode}"}]
    return rows


class Sampler:
    """The GPU's load, power, temperature and memory, the CPU and the disk, once a second (serve/telemetry.py),
    tagged with what runs."""

    def __init__(self, gpu_pci: str | None):
        from serve.telemetry import Telemetry
        self.t = Telemetry(gpus=[{"pci": gpu_pci, "roles": ["text"]}])
        self.tag: str | None = None
        self.rows: dict[str, list] = {}
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            time.sleep(1.0)
            if self.tag is not None:
                self.rows.setdefault(self.tag, []).append(self.t.snapshot()["now"])

    def summary(self, tag: str) -> dict:
        out = {}
        rows = self.rows.get(tag) or []
        for k in ("gpu_util", "gpu_power", "gpu_temp", "gpu_mem_used", "gpu_pcie_rx_mb", "gpu_pcie_gen", "cpu",
                  "ram_used", "disk_read_mb"):
            v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if v:
                out[k] = {"median": round(statistics.median(v), 1), "max": round(max(v), 1)}
        return out


# ------------------------------------------------------------------------------------------------ where the time goes
def median(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.median(v) if v else None


def judge(gen: dict, hw: dict, telemetry: dict) -> list[str]:
    """One line per generate tier: what a decode round's time is spent on (the engine's own breakdown), and how near
    the CPU's experts come to the RAM's measured bandwidth; one for the prompt where the engine says how it read it.
    Rules: the share of the round each part takes; the RAM line when the round has CPU experts."""
    lines = []
    ram = hw.get("ram_read_gb_s")
    for tier, runs in gen.items():
        ok = [r for r in runs if r.get("round") and r.get("rounds") and r.get("decode_ms")]
        if not ok:
            continue
        round_ms = median([r["decode_ms"] / r["rounds"] for r in ok])
        parts = {name: median([r["round"].get(key) for r in ok]) for name, key in
                 (("GPU 待ち", "gpu_wait_ms"), ("CPU のエキスパート", "pool_ms"), ("下書き（MTP）", "mtp_ms"),
                  ("ホスト", "host_ms"), ("VRAM の入れ替え", "adapt_ms"))}
        shares = sorted(((v / round_ms, k) for k, v in parts.items() if v), reverse=True)
        text = "、".join(f"{k} {100 * s:.0f}%" for s, k in shares[:3])
        line = f"{tier} 出力: 1 ラウンド {round_ms:.1f} ms のうち {text}"
        gbs = median([r["round"].get("pool_gb_s") for r in ok])
        if gbs and ram:
            line += f"。CPU のエキスパートの読み出しは {gbs:.1f} GB/s（測った RAM 帯域 {ram:.1f} GB/s の {100 * gbs / ram:.0f}%）"
        pcie = median([r["round"].get("pcie_experts") for r in ok])
        if pcie:
            line += f"。PCIe 越しのエキスパートは層あたり {pcie:.2f} 個"
        lines.append(line)
        pr = [r["prompt"] for r in runs if r.get("prompt")]
        tok_s = median([r.get("prefill_tok_s") for r in runs])
        if pr and tok_s:
            p = pr[0]
            share = f"、CPU に {100 * p['cpu_share']:.0f}%" if p["cpu_share"] else ""
            lines.append(f"{tier} プロンプト: {tok_s:,.0f} tok/s、エキスパートの転送 {p['experts_streamed']} 個"
                         f"（ホストの時間 {p['host_ms']:.0f} ms）、PLE {p['ple_ms']:.0f} ms{share}")
    for tag, t in telemetry.items():
        temp, util = t.get("gpu_temp"), t.get("gpu_util")
        if temp and temp["max"] >= 95:
            lines.append(f"{tag}: GPU の温度が {temp['max']:.0f}°C まで上がった（熱で速さが落ちている可能性）")
        if util and util["median"] < 30:
            lines.append(f"{tag}: GPU の使用率の中央値 {util['median']:.0f}%（GPU 以外が律速）")
    return lines


# ------------------------------------------------------------------------------------------------ the tables
def mark(xe, up, kind: str) -> str:
    """The difference column of the release notes: the prompt as a ratio, the output as a percentage, the time to
    the first token as a percentage the other way (shorter is better); ✅ where XeStrata is better."""
    if not xe or not up:
        return "-"
    if kind == "ratio":
        return f"{'✅' if xe >= up else '❌'} ×{xe / up:.2f}"
    d = (xe / up - 1) * 100
    better = d <= 0 if kind in ("lower", "ms") else d >= 0
    return f"{'✅' if better else '❌'} {d:+.0f}%"


def fmt(v, kind):
    if v is None:
        return "-"
    return f"{v:,.0f}" if kind in ("ratio", "ms") else f"{v:.2f}" if kind == "lower" else f"{v:.1f}"


def table(title: str, rows: list[tuple], names: list[str]) -> list[str]:
    """rows: (label, kind, [value per engine])."""
    if len(names) == 1:
        out = [f"### {title}", "", "| | " + names[0] + " |", "| --- | ---: |"]
        out += [f"| {label} | {fmt(v[0], kind)} |" for label, kind, v in rows]
    else:
        out = [f"### {title}", "", f"| | {names[0]} | {names[1]} | 差 |", "| --- | ---: | ---: | ---: |"]
        out += [f"| {label} | {fmt(v[0], kind)} | {fmt(v[1], kind)} | {mark(v[0], v[1], kind)} |"
                for label, kind, v in rows]
    return out + [""]


def summary_md(rec: dict) -> str:
    names = [e["name"] for e in rec["engines"]]
    m = rec["machine"]
    # the link as setup reads it, before the runs: a card that narrows or slows its link when idle shows that here
    gpu = " + ".join(f"{g.get('name')} ({g.get('vram_gb', 0):.0f} GB, PCIe link before the runs {g.get('link') or '?'})"
                     for g in m["gpus"])
    comp = m["xestrata"].get("compiler") or {}
    out = ["<!-- 中身を確認してから貼ってください。パスとホスト名は伏せてあります。 -->", "",
           f"## xestrata --bench ({rec['date']})", "",
           f"- XeStrata: {m['xestrata'].get('version')} (" + ", ".join(x for x in (
               m["xestrata"].get("license"),
               f"{comp.get('id') or ''} {str(comp.get('version') or '').split(' build based')[0]}".strip()) if x)
           + ")",
           f"- GPU: {gpu}",
           f"- CPU: {m['cpu']['name']}" + (f" ({', '.join(m['cpu']['isa'])})" if m['cpu']['isa'] else "") +
           f", RAM {m['ram_gib']} GiB",
           f"- OS: {m['os']}, {m.get('system', 'Linux')} {m['drivers']['kernel']}; data folder on "
           f"{m['data_disk'].get('kind') or '?'}",
           f"- Model: {m['model']['name']}, context {m['model']['context']}, KV {m['model']['kv']}"]
    hw = rec.get("hw") or {}
    if "ram_read_gb_s" in hw:
        out.append(f"- Measured: RAM {hw['ram_read_gb_s']:.1f} GB/s ({hw.get('ram_threads', '?'):g} threads), "
                   f"{hw.get('ram_read_1t_gb_s', 0):.1f} GB/s one thread; PCIe {hw.get('h2d_gb_s', 0):.1f} GB/s to the "
                   f"GPU, {hw.get('d2h_gb_s', 0):.1f} GB/s back; SSD {hw.get('ssd_read_gb_s', 0):.2f} GB/s")
    if rec.get("upstream"):
        out.append(f"- upstream: {rec['upstream'].get('tag')} ({rec['upstream'].get('how')})")
    out.append("")
    if rec["judgment"]:
        out += ["### どこに時間がかかっているか", ""] + [f"- {x}" for x in rec["judgment"]] + [""]
    mx = rec.get("matrix") or {}

    def med(rows, key, **match):
        return median([r[key] for r in rows if all(r.get(k) == v for k, v in match.items()) and r.get(key)])

    gen = mx.get("generate") or []
    if gen:
        rows = [(f"{t.upper()} {label} (tok/s)", kind, [med(gen, key, engine=n, tier=t) for n in names])
                for key, label, kind in (("prefill_tok_s", "プロンプト", "ratio"), ("decode_tok_s", "出力", "pct"))
                for t in rec["tiers"]]
        out += table(f"generate（追従・リリースと同じ条件、256 トークン greedy、{rec.get('rounds', 2)} 回の中央値）", rows, names)
    if rec.get("skipped"):
        out += [f"- {rec['skipped']}", ""]
    srv = mx.get("serve") or []
    for mode in sorted({r["mode"] for r in srv}, key=lambda m: (m != "greedy", m)):
        rows = [(f"{t.upper()} {label}", kind, [med(srv, key, engine=n, tier=t, mode=mode) for n in names])
                for key, label, kind in (("prefill_tok_s", "最初のプロンプト (tok/s)", "ratio"),
                                         ("cont_ms", "会話の続きの待ち (ms)", "ms"), ("decode_tok_s", "出力 (tok/s)", "pct"))
                for t in rec["tiers"]]
        how = "greedy" if mode == "greedy" else f"温度 {C.mode_temperature(mode):g}"
        out += table(f"serve {how}（4 ターンのチャット、256 トークン）", rows, names)
    use = mx.get("usage") or []
    if use:
        rows = [(f"{c} {label}", kind, [med(use, key, engine=n, conv=c) for n in names])
                for c in dict.fromkeys(r["conv"] for r in use)
                for key, label, kind in (("ttft_s", "最初のトークン (s)", "lower"),
                                         ("prefill_tok_s", "プロンプト (tok/s)", "ratio"),
                                         ("decode_tok_s", "出力 (tok/s)", "pct"))]
        t = rec.get("temperature", C.TEMPERATURE)
        out += table(f"usage（server.py 経由、{'greedy' if not t else f'温度 {t:g}、top_p 0.8、top_k 20'}）", rows, names)
    img = mx.get("image") or []
    if img:
        rows = [(label, "lower", [med(img, key, engine=n) for n in names])
                for key, label in (("seconds", "1 枚 (s)"), ("sampling_s", "サンプリング (s)"), ("decode_s", "VAE (s)"))]
        out += table("画像の生成（1024×1024、8 ステップ、種 42）", rows, names)
    stopped = [f"generate {r['engine']} {r['tier']} {r['round']}: {r['stopped']}" for r in gen if r.get("stopped")]
    stopped += [f"serve {r['engine']} {r['tier']} {r['mode']} {r['round']}: {r['stopped']}" for r in srv
                if r.get("stopped")]
    stopped += [f"usage {r['engine']} {r['round']}: {r['stopped']}" for r in use if r.get("stopped")]
    if stopped:
        out += ["### 止まった回", ""] + [f"- {x}" for x in stopped] + [""]
    v = rec.get("verbose") or {}
    kern = [k for k in v.get("kernels", []) if "kernel" in k]
    if kern:
        out += ["### カーネル単体（合成のデータ、µs）", "", "| 経路の設定 | カーネル | 経路 | 形 | µs |",
                "| --- | --- | --- | --- | ---: |"]
        out += [f"| {k['variant']} | {k['kernel']} | {k['path']} | {k['shape']} | {fmt(k['us'], 'pct')} |"
                for k in kern] + [""]
    tel = v.get("telemetry") or {}
    if tel:
        out += ["### 計測中の GPU と CPU（中央値 / 最大）", "", "| | GPU 使用率 % | 電力 W | 温度 °C | CPU % |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for tag, t in tel.items():
            cell = [f"{t[k]['median']:g} / {t[k]['max']:g}" if k in t else "-"
                    for k in ("gpu_util", "gpu_power", "gpu_temp", "cpu")]
            out.append(f"| {tag} | " + " | ".join(cell) + " |")
        out.append("")
    if v.get("fingerprint"):
        out += ["### 答えの指紋（generate の出力の先頭 64 トークン）", ""]
        out += [f"- {t}: {', '.join(h or '-' for h in hs)}" for t, hs in v["fingerprint"].items()]
        ref = v.get("reference")
        if ref:
            out.append(f"- 開発機の 1k の答えをなぞって、この PC の argmax が違ったのは {ref['differ']} / {ref['tokens']} "
                       f"トークン（top-1 一致率 {100 * ref['top1_agreement']:.1f}%）")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ the measurement
class BenchError(Exception):
    pass


def xe_engine(cfg: dict) -> C.Engine:
    from serve.server import child_env
    try:
        version = json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text()).get("version")
    except (OSError, ValueError):
        version = None
    return C.Engine(f"XeStrata {version or ''}".strip(), "xe", [cfg["exe"]],
                    arg_of(cfg["args"], "--expert-profile") or str(ROOT / "data" / "expert-profile.bin"),
                    env=child_env(cfg), cwd=cfg.get("cwd"), root=ROOT, py=sys.executable)


def model_of(cfg: dict) -> tuple[list[str], str | None]:
    """The merges' model arguments for this PC's model (its files, and C.CARRY as their EXTRA) and its draft layer."""
    _, model, mtp = C.config_model(cfg["args"])
    return model, mtp


def xe_license(cfg: dict) -> str:
    try:
        return json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text()).get("license") or "contrib"
    except (OSError, ValueError):
        return "contrib"


def run(cfg_path: Path, data: Path, gpus: list[dict], verbose: bool = False, original: str | None = None,
        say: Callable = print, rounds: int = C.ROUNDS, temperature: float = C.TEMPERATURE,
        cards: list[str] | None = None) -> Path:
    """Measure; write records/bench-<time>/ and return its summary.md.  `cards`: the GPUs' PCI addresses to run on
    instead of the config's, the first one first (setup's --gpu / --gpus for this bench only; #49)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if cards:
        cfg = {k: v for k, v in cfg.items() if k not in ("gpu", "split_pci")}
        cfg["gpu_pci"] = cards[0]
        if len(cards) > 1:
            cfg["split_pci"] = cards[1:]
    busy = busy_engines()
    if busy:
        raise BenchError("an XeStrata engine or server is running and would share the GPU with the measurements; stop "
                         "it first (" + "; ".join(busy[:3]) + ")")
    out = data / "records" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    extra_dir = out / "verbose"
    (out / "ids").mkdir(parents=True, exist_ok=True)
    scrub = Scrub(data)
    corpus = C.load_corpus()
    ids = C.tier_ids(C.load_tokenizer(Path(cfg["tokenizer"])), corpus)
    for name, v in ids.items():
        (out / "ids" / f"{name}.ids").write_text(",".join(map(str, v)))
    xe = xe_engine(cfg)
    engines = [xe]
    upstream = None
    if original:
        import bench_upstream
        say(f"  upstream Strata {original}: getting it (git) and its engine ...")
        try:
            up = bench_upstream.prepare(original, data, gpus, xe_license(cfg), xe.profile, say)
        except bench_upstream.UpstreamError as err:
            raise BenchError(f"upstream Strata {original}: {err}") from err
        engines.append(up)
        upstream = {"tag": original, "how": up.how}
    model, mtp = model_of(cfg)

    def args(e: C.Engine, tier: str) -> list[str]:
        return C.engine_args(model, e.profile, mtp, C.tier_context(model, tier))

    def usage_cfg(e: C.Engine) -> dict:
        # this PC's own config, as it is used; beside upstream, the merges' config for both engines at its context
        if not original:
            return dict(cfg)
        own = {k: cfg[k] for k in ("gpu", "gpu_pci", "lib_dirs") if k in cfg} if e is xe else {}
        return {"exe": e.cmd[0], "cwd": str(e.root), "tokenizer": cfg["tokenizer"], "model_name": "bench",
                "args": C.engine_args(model, e.profile, mtp, C.usage_context(model, ctx)), **own, **e.cfg_extra}

    ctx = context_of(cfg["args"])
    tiers = [t for t in TIERS if fits(t, ctx)]
    split = is_split(cfg)
    modes = tuple(dict.fromkeys(("greedy", C.temperature_mode(temperature))))
    rec: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "tiers": [] if split else tiers, "rounds": rounds,
                 "temperature": temperature,
                 "engines": [{"name": e.name, "arm": e.arm} for e in engines],
                 "machine": environment(cfg, gpus, data), "upstream": upstream, "judgment": [],
                 "verbose": {} if verbose else None}
    sampler = Sampler(cfg.get("gpu_pci")) if verbose else None

    def tagged(tag):
        if sampler:
            sampler.tag = tag

    say("  Measuring this machine (RAM, PCIe, SSD) ...")
    rec["hw"] = hw_bench(xe.cmd[0], xe.env, xe.cwd, cfg["args"], out / "hw-bench.txt")
    if split:
        rec["skipped"] = ("generate と serve は測っていない: この設定はレイヤー分割（複数 GPU）で、その 2 つは追従と同じく "
                          "1 枚の GPU で動かすため")
    else:
        say("  generate (the merges' bench_generate.py) ...")
        C.ab_generate(out, GPU_LABEL, engines, args, out / "ids", tiers, rounds=rounds, on_run=tagged)
        say("  serve (the merges' bench_serve.py) ...")
        C.ab_serve(out, GPU_LABEL, engines, args, ids, tiers, modes, rounds=rounds, on_run=tagged)
    say("  everyday use (the merges' bench_usage.py) ...")
    C.ab_usage(out, GPU_LABEL, engines, usage_cfg, ctx, corpus["agent"], rounds=rounds, on_run=tagged,
               temperature=temperature)
    if cfg.get("images"):
        # this PC's "images" section as setup wrote it, on the text engine's GPU (its environment); beside upstream,
        # its own image generator (sd-cli) where its setup built one (engine-image/BUILD.json names the backend),
        # configured as its setup configures it
        img = [C.Engine(xe.name, "xe", [], "", root=ROOT, py=sys.executable)]
        cfgs = {"xe": cfg["images"]}
        up_root = engines[1].root if len(engines) > 1 else None
        stamp = Path(up_root) / "engine-image" / "BUILD.json" if up_root else None
        if up_root is not None and stamp is not None and stamp.exists():
            meta = json.loads(stamp.read_text())
            u = C.Engine(engines[1].name, "up", [], "", root=Path(up_root), py=sys.executable)
            c = C.image_config(u, meta.get("backend", "vulkan"), cfg["images"].get("quant", "Q8_0"),
                               str(stamp.parent / ("sd-cli.exe" if os.name == "nt" else "sd-cli")),
                               str(Path(cfg["images"]["diffusion_model"]).parent.parent), cfg["images"].get("gpu"),
                               meta.get("lib_dirs"))
            if "error" not in c:
                img.append(u)
                cfgs["up"] = c
        say("  image generation (the merges' bench_image.py) ...")
        C.ab_image(out, GPU_LABEL, img, cfgs, rounds=rounds, on_run=tagged, envs={"xe": xe.env})
    tagged(None)
    tables: list[str] = []
    C.collect(out, {e.arm: e.name for e in engines}, say=tables.append)
    (out / "tables.md").write_text("\n".join(tables) + "\n", encoding="utf-8")
    rec["matrix"] = {k: json.loads((out / f).read_text()) for k, f in (
        ("generate", "matrix.json"), ("serve", "matrix-serve.json"), ("usage", "matrix-usage.json"),
        ("image", "matrix-image.json"))
        if (out / f).exists()}
    gen = {t: [parse_generate(f.read_text(errors="replace"))
               for f in sorted((out / "runs").glob(f"{GPU_LABEL}-xe-{t}-*.txt"))
               if re.fullmatch(rf"{GPU_LABEL}-xe-{t}-\d+", f.stem)] for t in rec["tiers"]}

    if verbose:
        extra_dir.mkdir(exist_ok=True)
        for t in rec["tiers"]:
            say(f"  The engine's per-stage timings at {t} ...")
            log = extra_dir / f"stats-{t}.txt"
            C.run_generate(xe, args(xe, t) + ["--stats"], out / "ids" / f"{t}.ids", log, extra_env=STATS_ENV)
            text = log.read_text(errors="replace")
            start = set(engine_lines(text))
            rec["verbose"].setdefault("stats", {})[t] = [x for x in text.splitlines()
                                                         if not x.startswith(("PP ", "output")) and x not in start]
        say("  The kernels one by one ...")
        rec["verbose"]["kernels"] = kernel_bench(xe.cmd[0], xe.env, xe.cwd, arg_of(cfg["args"], "--pack"),
                                                 extra_dir / "kernel-bench.txt")
        rec["verbose"]["fingerprint"] = {t: [r.get("output_sha256_64") for r in runs] for t, runs in gen.items()}
        # the model's reference, also one kept under a name it had before (its aliases)
        ref = next((REFS / f"{x}.csv" for x in [cfg.get("model_name"), *(cfg.get("aliases") or [])]
                    if (REFS / f"{x}.csv").exists()), REFS / f"{cfg.get('model_name')}.csv")
        if ref.exists() and not split:
            say("  The answer against the development machine's (top-1 agreement) ...")
            log = extra_dir / "reference-1k.txt"
            C.run_generate(xe, args(xe, "1k"), out / "ids" / "1k.ids", log, ref)
            row = parse_generate(log.read_text(errors="replace"))
            if row.get("follow_emitted"):
                rec["verbose"]["reference"] = {
                    "tokens": row["follow_emitted"], "differ": row["follow_differ"],
                    "top1_agreement": round(1 - row["follow_differ"] / row["follow_emitted"], 4)}
    if sampler is not None:
        rec["verbose"]["telemetry"] = {tag: sampler.summary(tag) for tag in sampler.rows}
        sampler.stop = True
    warm = [out / "runs" / f"{GPU_LABEL}-xe-warmup.txt", out / "serve-runs" / f"{GPU_LABEL}-xe-warmup.txt"]
    rec["engine_lines"] = next((lines for f in warm if f.exists()
                                and (lines := engine_lines(f.read_text(errors="replace")))), [])
    rec["judgment"] = judge(gen, rec["hw"], (rec["verbose"] or {}).get("telemetry", {}))
    rec = scrub(rec)
    (out / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = out / "summary.md"
    md.write_text(summary_md(rec) + "\n", encoding="utf-8")
    return md


def main(argv: list[str]) -> int:
    if argv[:1] == ["ref"] and len(argv) == 2:
        # the 1k prompt's greedy answer on this machine, generate's way, as the model's reference
        cfg = json.loads(Path(argv[1]).read_text(encoding="utf-8-sig"))
        xe = xe_engine(cfg)
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"xestrata-bench-ref-{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        tiers = C.tier_ids(C.load_tokenizer(Path(cfg["tokenizer"])), C.load_corpus())
        (tmp / "1k.ids").write_text(",".join(map(str, tiers["1k"])))
        log = tmp / "ref.txt"
        model, mtp = model_of(cfg)
        C.run_generate(xe, C.engine_args(model, xe.profile, mtp, C.CONTEXT["1k"]), tmp / "1k.ids", log)
        text = C.output_ids(log.read_text(errors="replace"))
        if not text:
            print(f"no answer (the log: {log})", file=sys.stderr)
            return 1
        REFS.mkdir(exist_ok=True)
        out = REFS / f"{cfg.get('model_name')}.csv"
        out.write_text(text + "\n")
        print(f"{out}: {len(text.split(','))} tokens")
        return 0
    print("usage: python tools/bench.py ref CONFIG", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
