#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/strata_mcp.py - an MCP server that lets an AI assistant install, start, check and stop Strata.

    claude mcp add xestrata -- python3 /path/to/XeStrata/tools/strata_mcp.py

Add it to Claude Code, Claude Desktop, Cursor, VS Code, Codex ... (docs/DETAILS.md, "Manage Strata from your AI
assistant", has the snippets) and ask your AI "install Strata for this PC", "start Strata", "is Strata running?".
It speaks MCP (JSON-RPC 2.0, one message per line) over stdin/stdout, uses the Python standard library only (it runs
before setup made .venv; Python 3.10 or newer), and offers a fixed set of tools: strata_status, strata_models,
strata_install, strata_start, strata_stop, strata_logs, strata_benchmark, strata_connect_info.

Safety: no tool takes a shell command or a free path.  Every argument is checked against setup.py's own choices
(models, families, contexts ...); the only path argument (install's data_dir) must be the Strata data folder, a folder
inside the Strata folder, or a new folder named XeStrata* outside the system folders.  Setup and the model run as
background processes this server starts itself; it keeps their process id and start time in .strata-mcp/ and only
ever stops a process it started (a Strata started elsewhere, e.g. from its own window, is only asked over its API to
unload the model).

Not to be confused with serve/mcp.py: that is Strata's MCP *client* (the model calling your tools from the chat page).
This file is an MCP *server* for the AI assistant you already use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "2025-06-18"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")   # the revisions this server answers in
VERSION = "0.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8095
STATE_DIR = ".strata-mcp"                             # in the Strata folder (git-ignored)
CONFIG = "xestrata-"                                  # setup's run configs: xestrata-<model>.json

# What setup.py offers, used only when setup.py cannot be imported (the numbers are setup's MODELS / FAMILIES /
# CONTEXTS; tools/test_strata_mcp.py checks they still match).
FALLBACK_MODELS = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "vision": True},
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), EXPERIMENTAL: the best quality, but on a 64 GB PC part of its "
                            "experts come from the SSD while it answers, so it is slower than the 2-3-bit models",
                   "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",), "budget": True,
                   "experimental": True},
}
FALLBACK_FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "about": "the original model", "tag": ""},
    "swift": {"title": "Swift 1.5", "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its "
                                             "authors' numbers)", "tag": "swift-"},
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "about": "half the experts (code, tools, images kept): needs ~32 GB "
                                                            "of RAM, faster; weaker outside coding", "tag": "coder-"},
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "about": "UD-IQ4_XS: a 94 GB download; with less than ~80 GB "
                                                                  "of RAM part of its experts are read from the SSD "
                                                                  "(UD-Q4_K_XL, 111 GB: experimental)",
                "tag": "unsloth-", "vision": False},
}
FALLBACK_CONTEXTS = [8192, 32768, 65536, 131072, 262144, 393216, 524288]
BENCH_PROMPT = ("Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle. "
                "Plain prose, no title.")


def log(msg: str) -> None:
    """Diagnostics go to stderr: stdout carries only the protocol."""
    try:
        sys.stderr.write(f"[strata-mcp] {msg}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


# ------------------------------------------------------------------------------------------------ processes
def proc_identity(pid) -> str | None:
    """A token for a RUNNING process: its start time (so a reused process id is never mistaken for ours), or None
    when no such process runs."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc/self/stat").exists():
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            return None
        if fields[0] in ("Z", "X"):                     # a zombie has ended
            return None
        return f"linux:{fields[19]}"                    # field 22 of stat: the start time in clock ticks
    try:                                                # elsewhere (no /proc): alive, without a start time
        os.kill(pid, 0)
        return "pid"
    except OSError:
        return None


def proc_alive(pid, ident) -> bool:
    return bool(ident) and proc_identity(pid) == ident


def proc_terminate(pid, ident) -> bool:
    """Ask a process WE started to end (SIGTERM, the server's clean path).  Never touches a process whose start time
    differs from the one recorded."""
    if not proc_alive(pid, ident):
        return False
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def proc_kill_tree(pid, ident) -> bool:
    """End a process WE started together with the processes it started (setup's pip / cmake, the engine)."""
    if not proc_alive(pid, ident):
        return False
    try:
        os.killpg(int(pid), signal.SIGKILL)             # started as a session leader: its group is its own
        return True
    except OSError:
        try:
            os.kill(int(pid), signal.SIGKILL)
            return True
        except OSError:
            return False


def end_group_leftovers(pgid) -> None:
    """After the server WE started (a session leader) is gone, end what is left of its process group - an
    engine it started but could not close (e.g. stopped while loading, before its own SIGTERM handler existed).
    While any member lives, the group id cannot belong to anybody else."""
    if not pgid:
        return
    for sig, wait in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(int(pgid), sig)
        except OSError:
            return                                      # no such group left
        end = time.time() + wait
        while time.time() < end:
            try:
                os.killpg(int(pgid), 0)
            except OSError:
                return
            time.sleep(0.25)


def wait_gone(pid, ident, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not proc_alive(pid, ident):
            return True
        time.sleep(0.25)
    return not proc_alive(pid, ident)


def spawn_detached(cmd: list, cwd: Path, log_path: Path, env: dict | None = None) -> subprocess.Popen:
    """Start a background process whose output goes to log_path.  It keeps running when the AI client (and this
    server) ends: its own process group and session."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        return subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                env=env, close_fds=True, start_new_session=True)


def run_quiet(cmd, timeout=30) -> str:
    """A command's stdout ('' when it is missing or fails)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL).stdout or ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def read_tail(path: Path, lines: int, max_bytes: int = 512 * 1024) -> list[str]:
    """The last lines of a log; a progress line rewritten with \\r counts as its last version."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for raw in data.split("\n"):
        part = [p for p in raw.split("\r") if p.strip()]
        if part:
            out.append(part[-1].rstrip())
    return out[-lines:]


def dir_size_gb(path: Path, limit_files: int = 20000) -> float:
    total, n = 0, 0
    for base, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(base, f)).st_size
            except OSError:
                pass
            n += 1
            if n >= limit_files:
                return round(total / 1e9, 1)
    return round(total / 1e9, 1)


# ------------------------------------------------------------------------------------------------ HTTP
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never through a proxy


def http_json(url: str, method="GET", body=None, api_key=None, timeout=3.0):
    """(status, JSON or None); status 0 when nothing answers."""
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw or b"null")
    except ValueError:
        return status, None


def port_free(port: int, host="127.0.0.1") -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ validation
class ToolError(Exception):
    """A clear message for the AI (and its user): the tool result is an error, the server goes on."""


def check_schema(schema: dict, args) -> dict:
    """The tools' JSON schemas, enforced: object, known properties only, types, enums, ranges, patterns."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    for k in args:
        if k not in props:
            raise ToolError(f"unknown argument {k!r}; this tool takes: {', '.join(props) or 'no arguments'}")
    for k in schema.get("required", []):
        if k not in args:
            raise ToolError(f"missing argument {k!r}")
    for k, v in args.items():
        p = props[k]
        t = p.get("type")
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            if isinstance(v, float) and v.is_integer():
                v = args[k] = int(v)
            elif isinstance(v, str) and re.fullmatch(r"\d{1,7}", v.strip()):
                v = args[k] = int(v.strip())              # some clients send numbers as strings
            else:
                raise ToolError(f"{k} must be a whole number")
        if t == "boolean" and not isinstance(v, bool):
            if v in ("true", "false"):
                v = args[k] = v == "true"
            else:
                raise ToolError(f"{k} must be true or false")
        if t == "string":
            if not isinstance(v, str):
                raise ToolError(f"{k} must be a string")
            if len(v) > p.get("maxLength", 300) or "\x00" in v or "\n" in v or "\r" in v:
                raise ToolError(f"{k} is too long or contains control characters")
            if "pattern" in p and not re.fullmatch(p["pattern"], v):
                raise ToolError(f"{k}={v!r} is not valid ({p.get('description', p['pattern'])})")
        if "enum" in p and v not in p["enum"]:
            raise ToolError(f"{k} must be one of: {', '.join(map(str, p['enum']))} (got {v!r})")
        if t == "integer":
            if "minimum" in p and v < p["minimum"] or "maximum" in p and v > p["maximum"]:
                raise ToolError(f"{k} must be between {p.get('minimum')} and {p.get('maximum')} (got {v})")
    return args


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def norm(p) -> Path:
    """A path to compare: absolute, links resolved."""
    return Path(os.path.normcase(os.path.realpath(os.path.abspath(str(p)))))


def system_folders() -> list[Path]:
    return [norm(p) for p in ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/proc", "/run", "/sbin",
                              "/sys", "/usr", "/var", "/snap")]


# ------------------------------------------------------------------------------------------------ Strata
class Strata:
    """One Strata folder: what is installed there, the PC, and the processes this server started."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self.state_dir = self.root / STATE_DIR
        self.lock = threading.Lock()                    # install / start / stop one at a time
        self.children: list[subprocess.Popen] = []      # reaped now and then (no zombies on POSIX)
        self._setup = None
        self._setup_tried = False
        self._setup_lock = threading.Lock()
        self._hw_cache: tuple | None = None
        self.python_override: str | None = None         # tests: the interpreter that runs setup / the server
        self.health_timeout = 2.0

    # ---- setup.py's own tables and logic
    def setup_module(self):
        """setup.py imported as a module (its tables and its pure checks), or None.  Its command runner is replaced
        by a quiet one; nothing of its install path runs."""
        with self._setup_lock:                          # parallel tool calls: one import, and none sees it half-done
            return self._setup_module()

    def _setup_module(self):
        if not self._setup_tried:
            self._setup_tried = True
            try:
                spec = importlib.util.spec_from_file_location("strata_setup_for_mcp", self.root / "setup.py")
                mod = importlib.util.module_from_spec(spec)
                real = sys.stdout
                sys.stdout = sys.stderr                 # nothing it prints may reach the protocol stream
                try:
                    spec.loader.exec_module(mod)
                finally:
                    sys.stdout = real
                if all(hasattr(mod, a) for a in ("MODELS", "FAMILIES", "CONTEXTS")):
                    mod.out = lambda cmd: run_quiet(cmd, 60)
                    self._setup = mod
            except (Exception, SystemExit) as e:        # noqa: BLE001 - any failure: the built-in tables
                log(f"setup.py could not be imported ({e}); using the built-in model table")
        return self._setup

    def tables(self):
        s = self.setup_module()
        if s is not None:
            return s.MODELS, s.FAMILIES, list(s.CONTEXTS), "setup.py"
        return FALLBACK_MODELS, FALLBACK_FAMILIES, FALLBACK_CONTEXTS, "built-in table (setup.py not importable)"

    @staticmethod
    def sizes_of(models, family) -> list:
        return [m for m in models if family in models[m].get("families", ("qwen", "swift"))]

    # ---- paths
    def settings_path(self) -> Path:
        return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "xestrata" / "settings.json"

    def data_dir(self) -> Path:
        try:
            s = json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        d = s.get("data_dir") if isinstance(s, dict) else None
        return Path(d) if d else self.root.parent / "XeStrata-data"

    def allowed_roots(self) -> list[Path]:
        return [norm(self.root), norm(self.data_dir())]

    def allowed(self, p) -> bool:
        p = norm(p)
        return any(is_inside(p, r) for r in self.allowed_roots())

    def venv_python(self) -> Path:
        return self.root / ".venv" / "bin" / "python"

    def run_python(self) -> str:
        return self.python_override or str(self.venv_python())

    def state(self, name: str) -> dict:
        try:
            d = json.loads((self.state_dir / f"{name}.json").read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, name: str, d: dict | None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        f = self.state_dir / f"{name}.json"
        if d is None:
            f.unlink(missing_ok=True)
        else:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            os.replace(tmp, f)

    def reap(self):
        self.children = [p for p in self.children if p.poll() is None]

    # ---- what is installed
    def configs(self) -> list[Path]:
        return sorted((p for p in self.root.glob(CONFIG + "*.json") if not p.name.endswith(".shared-settings.json")),
                      key=lambda p: p.stat().st_mtime, reverse=True)   # #346: the Chat settings file is no config

    @staticmethod
    def read_config(path: Path) -> dict:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            return cfg if isinstance(cfg, dict) else {}
        except (OSError, ValueError):
            return {}

    def describe_config(self, path: Path) -> dict:
        cfg = self.read_config(path)
        args = cfg.get("args")
        a: list = args if isinstance(args, list) else []

        def val(k):
            return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None
        tag = path.stem[len(CONFIG):]
        missing = [p for p in [cfg.get("exe"), *[x for x in a if isinstance(x, str) and x.endswith(".gguf")]]
                   if not p or not Path(p).exists()]
        script = self.root / f"run-{tag}.sh"
        vis = cfg.get("vision")
        return {"model": tag, "config": path.name, "model_name": cfg.get("model_name"),
                "context": int(val("--max-context")) if str(val("--max-context") or "").isdigit() else None,
                "kv": val("--kv"), "images": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "off",
                "port": cfg.get("port", DEFAULT_PORT), "host": cfg.get("host", "127.0.0.1"),
                "gpu": cfg.get("gpu"), "api_key_set": bool(cfg.get("api_key")),
                "start_script": script.name if script.exists() else None,
                "ready": not missing and bool(cfg), "missing_files": missing[:3],
                "last_used": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))}

    def find_config(self, model: str | None) -> Path:
        have = self.configs()
        if not have:
            raise ToolError("no model is installed in " + str(self.root) + " yet: install one first (strata_install)")
        if not model:
            return have[0]                              # the most recently used, as setup starts it
        m = model.strip().lower()
        exact = [c for c in have if c.stem[len(CONFIG):] == m]
        if exact:
            return exact[0]
        by_name = [c for c in have if str(self.read_config(c).get("model_name", "")).lower() == m]
        if by_name:
            return by_name[0]
        ends = [c for c in have if c.stem.endswith("-" + m)]
        if len(ends) == 1:
            return ends[0]
        raise ToolError(f"no installed model {model!r}; installed: " +
                        ", ".join(c.stem[len(CONFIG):] for c in have))

    def engine_info(self) -> dict:
        for d in ("engine",):
            try:
                meta = json.loads((self.root / d / "BUILD.json").read_text())
                return {"folder": d, "version": meta.get("version"), "source": meta.get("source"),
                        "compiler": meta.get("compiler")}
            except (OSError, ValueError):
                continue
        return {"folder": None, "version": None}

    def installed(self) -> dict:
        data = self.data_dir()
        items: dict = {}
        for sub in ("models", "packs"):
            for base in dict.fromkeys([data, self.root]):
                d = base / sub
                if d.is_dir():
                    for c in sorted(d.iterdir()):
                        if c.is_dir():
                            ggufs = list(c.glob("*.gguf"))
                            entry = {"folder": str(c), "size_gb": dir_size_gb(c)}
                            if sub == "models":
                                entry["complete"] = bool(ggufs) and all(
                                    g.with_name(g.name + ".done").exists() for g in ggufs)
                                entry["downloading"] = bool(list(c.glob("*.part")))
                            items.setdefault(sub, []).append(entry)
        mtp = data / "mtp" / "rt" / "experts.bin"
        return {"strata_folder": str(self.root), "data_folder": str(data), "data_folder_exists": data.is_dir(),
                "python_env": self.venv_python().exists(), "engine": self.engine_info(),
                "models": [self.describe_config(c) for c in self.configs()],
                "downloads": items.get("models", []), "prepared": items.get("packs", []),
                "draft_layer": mtp.exists()}

    # ---- the PC
    def hardware(self, fresh=False) -> dict:
        if self._hw_cache and not fresh and time.time() - self._hw_cache[0] < 600:
            hw = dict(self._hw_cache[1])
        else:
            hw = self._hardware()
            self._hw_cache = (time.time(), hw)
            hw = dict(hw)
        hw["disk_free_gb"] = self.disk_free(self.data_dir())
        return hw

    @staticmethod
    def disk_free(p: Path) -> float | None:
        p = Path(p)
        while not p.exists() and p.parent != p:
            p = p.parent
        try:
            return round(shutil.disk_usage(p).free / 1e9, 1)
        except OSError:
            return None

    def _hardware(self) -> dict:
        S = self.setup_module()
        hw: dict = {"os": f"{platform.system()} {platform.release()}", "os_version": platform.version(),
                    "machine": platform.machine(), "python": platform.python_version()}
        if Path("/proc/version").exists():
            try:
                hw["wsl"] = "microsoft" in Path("/proc/version").read_text().lower()
            except OSError:
                pass
        try:
            hw["ram_gb"] = round(S.ram_gb(), 1) if S else round(own_ram_gb(), 1)
        except Exception:                               # noqa: BLE001
            hw["ram_gb"] = round(own_ram_gb(), 1)
        try:
            if S:
                name, avx2, avx512 = S.cpu_info()
            else:
                name, avx2, avx512 = platform.processor() or "unknown CPU", None, None
            hw["cpu"] = {"name": name, "cores": os.cpu_count(), "avx2": avx2, "avx512": avx512}
        except Exception:                               # noqa: BLE001
            hw["cpu"] = {"name": platform.processor(), "cores": os.cpu_count()}
        found: list = []
        try:
            found = S.gpus() if S else []               # every Intel GPU on the xe or i915 driver, in PCI order
        except Exception:                               # noqa: BLE001
            found = []
        for g in found:
            g["vendor"] = "intel"
            g["vram_gb"] = round(g["vram_gb"], 1)
            try:
                g["problem"] = S.gpu_problem(g)
            except Exception:                           # noqa: BLE001
                g["problem"] = None
            g["usable"] = g["problem"] is None
        hw["gpus"] = [{k: v for k, v in g.items() if k not in ("render", "device_id")} for g in found]
        return hw

    def recommend(self, hw: dict) -> dict:
        """setup.py's own rules: the size it preselects (IQ3_XXS from 60 GB of RAM, else the first, Q2_0), the
        Coder below the full model's RAM (setup's RAM table), and the context it preselects for the card's VRAM (32K
        under 14 GB, 64K under 20 GB, else 128K)."""
        models, families, _ctx, _src = self.tables()
        usable = [g for g in hw.get("gpus", []) if g.get("usable")]
        ram = hw.get("ram_gb") or 0
        if not usable:
            return {"family": None, "model": None,
                    "why": "no Intel GPU on the xe (or i915) driver found (setup --check lists what it sees)"}
        best = max(usable, key=lambda g: (round(g["vram_gb"]), -g.get("index", 0)))
        vram = best["vram_gb"]
        notes = []
        if ram >= 60:
            fam, model, why = "qwen", "IQ3_XXS", f"{ram:.0f} GB of RAM: setup's own pick from 60 GB (better quality)"
        elif ram >= models["Q2_0"]["ram_gb"] - 4:
            fam, model, why = "qwen", "Q2_0", f"{ram:.0f} GB of RAM: the full model's fastest size fits"
        elif ram >= models["IQ1_M"]["ram_gb"] - 4:
            fam, model, why = "coder", "IQ1_M", (f"{ram:.0f} GB of RAM: the full model needs ~48 GB; the Coder "
                                                 "(half the experts, best for code) needs ~32 GB")
        else:
            return {"family": None, "model": None, "why": f"{ram:.0f} GB of RAM: Strata needs 32 GB or more "
                                                          "(the smallest model, the Coder, keeps ~23 GB of experts in "
                                                          "RAM)"}
        if hw.get("cpu", {}).get("avx2") is False:
            return {"family": None, "model": None, "why": "this CPU has no AVX2; Strata needs at least AVX2"}
        ctx = 32768 if vram < 14 else 65536 if vram < 20 else 131072
        if best.get("integrated"):
            notes.append("the processor's own graphics: it runs, but slowly (its memory is the system RAM)")
        elif vram < 11:
            notes.append("less than 12 GB of VRAM: it runs, but slowly (most experts stay on the CPU)")
        cmd = f"./setup.sh --setup --yes --family {fam} --model {model} --context {ctx}"
        return {"family": fam, "model": model, "title": families[fam]["title"] + " " + model, "context": ctx,
                "gpu": {k: best.get(k) for k in ("index", "name", "vram_gb", "pci")},
                "download_gb": models[model]["download_gb"], "why": why, "notes": notes, "setup_command": cmd}

    def model_table(self, hw: dict | None) -> list:
        models, families, _ctx, _src = self.tables()
        S = self.setup_module()
        ram = (hw or {}).get("ram_gb")
        usable = [g for g in (hw or {}).get("gpus", []) if g.get("usable")]
        vram = max((g["vram_gb"] for g in usable), default=0.0)
        have = {c.stem[len(CONFIG):] for c in self.configs()}
        out = []
        for f, fd in families.items():
            sizes = []
            for m in self.sizes_of(models, f):
                d = models[m]
                tag = (fd.get("tag", "") + m).lower()
                verdict = None
                if ram is not None:
                    verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
                    try:
                        if d.get("budget"):
                            verdict = (("experimental: " if d.get("experimental") else "")
                                       + "part of its experts in RAM, the rest read from the SSD"
                                       if ram >= d["ram_gb"] else "does not fit")
                        elif S and S.low_ram_needed(m, ram) and S.low_ram_fits(m, ram, vram):
                            verdict = "fits in the low-RAM mode (slower: the GPU holds part of the experts)"
                    except Exception:                   # noqa: BLE001
                        pass
                sizes.append({"model": m, "about": d["about"], "download_gb": d["download_gb"],
                              "experimental": bool(d.get("experimental")),
                              "images": d.get("vision", fd.get("vision", True)) is not False,
                              "ram_needed_gb": d["ram_gb"], "experts_gb": d.get("arena_gb"),
                              "on_this_pc": verdict, "installed": tag in have, "id": tag})
            out.append({"family": f, "title": fd["title"], "about": fd.get("about"),
                        "experimental": bool(fd.get("experimental")),
                        "images": any(x["images"] for x in sizes) if sizes else fd.get("vision", True) is not False,
                        "sizes": sizes})
        return out

    # ---- the running server
    def ports(self, port: int | None) -> list[int]:
        if port:
            return [port]
        ps = [self.state("server").get("port")] + [self.describe_config(c)["port"] for c in self.configs()]
        return list(dict.fromkeys(int(p) for p in [*ps, DEFAULT_PORT] if isinstance(p, int)))

    def api_key_for(self, port: int) -> str | None:
        """The key a config of this folder set for that port (used for our own requests; never shown)."""
        for c in self.configs():
            cfg = self.read_config(c)
            if cfg.get("api_key") and int(cfg.get("port", DEFAULT_PORT)) == port:
                return cfg["api_key"]
        for c in self.configs():
            if self.read_config(c).get("api_key"):
                return self.read_config(c)["api_key"]
        return None

    def probe(self, port: int, deep=True) -> dict | None:
        """What answers on 127.0.0.1:port: a Strata server's facts, {"other": True} for something else, None for
        nothing."""
        base = f"http://127.0.0.1:{port}"
        st, health = http_json(base + "/health", timeout=self.health_timeout)
        if st == 0:
            return None
        if st != 200 or not isinstance(health, dict) or health.get("service") != "strata":
            return {"port": port, "other": True, "note": "something else answers on this port (not a Strata server)"}
        info = {"port": port, "url": base, "model": health.get("model"), "context": health.get("max_context"),
                "images": health.get("images"), "loaded": health.get("loaded"),
                "api_key_required": bool(health.get("api_key"))}
        if not deep:
            return info
        key = self.api_key_for(port)
        st, v1 = http_json(base + "/v1/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(v1, dict):
            info["engine_version"] = v1.get("engine")
            info["uptime_s"] = v1.get("uptime_s")
            info["requests_served"] = (v1.get("activity") or {}).get("requests")
            m = v1.get("machine") or {}
            if m.get("gpu"):
                info["gpu"] = m["gpu"]
            if m.get("ram"):
                info["ram"] = m["ram"]
            if v1.get("last_timings"):
                lt = v1["last_timings"]
                info["last_request"] = {k: lt.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n",
                                                               "predicted_per_second") if k in lt}
        elif st == 401:
            info["note"] = "the server wants an API key this folder's configs do not have"
        st, s = http_json(base + "/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(s, dict):
            info["busy"] = bool(s.get("busy"))
            info["activity"] = {k: s.get(k) for k in ("phase", "generated", "tokens_per_s", "queued", "elapsed_s")
                                if s.get(k) is not None}
        if "gpu" not in info:                           # older servers: the Monitor's numbers
            st, met = http_json(base + "/metrics", api_key=key, timeout=self.health_timeout)
            if st == 200 and isinstance(met, dict):
                hwn = met.get("hardware") or {}
                if hwn.get("gpu_mem_total"):
                    info["gpu"] = {"used_mib": round(hwn.get("gpu_mem_used", 0) / 2**20),
                                   "total_mib": round(hwn["gpu_mem_total"] / 2**20)}
                if hwn.get("ram_total"):
                    info["ram"] = {"used_gib": round(hwn.get("ram_used", 0) / 2**30, 1),
                                   "total_gib": round(hwn["ram_total"] / 2**30, 1)}
                info["engine_version"] = (met.get("engine") or {}).get("version")
        return info

    def tracked_server(self) -> dict | None:
        """The server this MCP server started, while it still runs (else its record is cleared)."""
        s = self.state("server")
        if not s:
            return None
        if proc_alive(s.get("pid"), s.get("ident")):
            return s
        return {**s, "ended": True}

    # ---- install job
    def install_job(self) -> dict | None:
        j = self.state("install")
        if not j:
            return None
        res = self.state("install-result")
        running = proc_alive(j.get("pid"), j.get("ident"))
        log_path = Path(j.get("log", ""))
        tail = read_tail(log_path, 400) if log_path.name else []
        prog = install_progress(tail)
        out = {"running": running, "started": j.get("started"), "model": j.get("model"), "family": j.get("family"),
               "log": str(log_path), **prog}
        if not running:
            if res.get("job") == j.get("job") and "exit_code" in res:
                out["exit_code"] = res["exit_code"]
                out["finished"] = res.get("ended")
                out["result"] = "installed" if res["exit_code"] == 0 else "failed"
            else:
                out["result"] = "ended unexpectedly (no exit code recorded)"
        else:
            out["result"] = "running"
        out["last_lines"] = tail[-12:]
        return out


def install_progress(lines: list[str]) -> dict:
    """setup's progress from its output: the step, a download's percentage, a failure."""
    step, title, dl, err = None, None, None, None
    for i, line in enumerate(lines):
        m = re.search(r"=== Step (\d+): (.+?) ===", line)
        if m:
            step, title, err = int(m.group(1)), m.group(2), None
        m = re.match(r"\s*(\S.*?): +([\d.]+) / ([\d.]+) GB \((\d+)%\)", line)
        if m:
            dl = {"file": m.group(1), "done_gb": float(m.group(2)), "total_gb": float(m.group(3)),
                  "percent": int(m.group(4))}
        if line.strip().startswith("[X]"):
            err = line.strip()[3:].strip()
            if i + 1 < len(lines) and lines[i + 1].startswith("       "):
                err += " - " + lines[i + 1].strip()
    out: dict = {"step": f"{step}/7: {title}" if step else None}
    if dl:
        out["download"] = dl
    if err:
        out["error"] = err
    return out


def own_ram_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / 2**30
    except OSError:
        pass
    return 0.0


# ------------------------------------------------------------------------------------------------ the tools
MODEL_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,48}"


def tool_defs(models: dict, families: dict, contexts: list) -> list:
    port = {"type": "integer", "minimum": 1024, "maximum": 65535,
            "description": "the server's port (default: the installed model's port, usually 8095)"}
    installed_model = {"type": "string", "pattern": MODEL_ID_PATTERN, "maxLength": 49,
                       "description": "an installed model's id as strata_status lists it (e.g. 'q2_0', "
                                      "'coder-iq1_m', 'swift-iq3_xxs'); default: the most recently used one"}
    ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    return [
        {"name": "strata_status", "title": "Strata status",
         "description": "Is Strata (the local AI model server) running, and what is installed? Reports the running "
                        "server (model, engine version, context, busy or idle, VRAM/RAM in use), the installed models "
                        "and data folder, an install in progress, this PC's hardware (OS, CPU, RAM, GPUs, free disk) "
                        "and the model size setup recommends for this PC. Read-only; call it first.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "hardware": {"type": "boolean", "description": "include the hardware check and the recommendation "
                                                            "(default true; takes a second or two)"}}},
         "annotations": ro},
        {"name": "strata_models", "title": "Strata models",
         "description": "The models and sizes Strata's setup offers (families: qwen = Qwen3.8-Flash-Next, swift = "
                        "Swift 1.5, coder = the Coder), with download size, RAM needed, "
                        "whether each fits this PC, which are installed, the context lengths, and the recommendation. "
                        "Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
         "annotations": ro},
        {"name": "strata_install", "title": "Install Strata",
         "description": "Install a Strata model on this PC by running Strata's setup non-interactively in the "
                        "background (Python packages, the engine, a 58-111 GB model download, preparation, start "
                        "script). Without confirm=true it only returns the plan (what would be downloaded, disk "
                        "space): show it to the user and call again with confirm=true once they agree. Returns at "
                        "once; call it again (no arguments needed) to see the progress, or with cancel=true to stop "
                        "it (it resumes later). Defaults: the recommended model and context for this PC, images off. "
                        "After it finishes, start the model with strata_start.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "family": {"type": "string", "enum": list(families),
                        "description": "qwen = Qwen3.8-Flash-Next (the original), swift = Swift 1.5 (thinks "
                                       "shorter), coder = the Coder (half the experts, ~32 GB RAM, for code)"},
             "model": {"type": "string", "enum": list(models),
                       "description": "the size (quantization): see strata_models"},
             "context": {"type": "integer", "enum": contexts,
                         "description": "context length in tokens (more needs more VRAM)"},
             "vision": {"type": "string", "enum": ["no", "yes", "gpu", "cpu"],
                        "description": "let the model read images (yes = the encoder on the GPU; ~0.9 GB more "
                                       "download); default no"},
             "kv": {"type": "string", "enum": ["int8", "q4_0", "k8v4"],
                    "description": "KV cache precision above 8K context (default int8)"},
             "gpu": {"type": "integer", "minimum": 0, "maximum": 63,
                     "description": "the Intel GPU, numbered in PCI order as setup --check lists them (default: "
                                    "the one with the most VRAM)"},
             "port": {**port, "description": "the port the model will listen on (default 8095)"},
             "data_dir": {"type": "string", "maxLength": 260,
                          "description": "where the model files go (default: the XeStrata-data folder next to the "
                                         "XeStrata folder). Only the current data folder, a folder inside the "
                                         "XeStrata folder, or a new absolute path whose last folder is named "
                                         "XeStrata... (e.g. /mnt/big/XeStrata-data) outside system folders"},
             "confirm": {"type": "boolean", "description": "true = really start the install (after the user agreed "
                                                           "to the plan and the download size)"},
             "cancel": {"type": "boolean", "description": "true = stop the install in progress"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                         "openWorldHint": True}},
        {"name": "strata_start", "title": "Start Strata",
         "description": "Start an installed Strata model in the background (as its start script does, without "
                        "opening a browser) and wait until it answers. Loading takes 1-3 minutes and the PC can be "
                        "slow meanwhile; if it is not ready within wait_seconds this returns 'loading' - check again "
                        "with strata_status. The model keeps running until strata_stop.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "model": installed_model, "port": port,
             "gpu": {"type": "integer", "minimum": 0, "maximum": 63,
                     "description": "run on this GPU for this start (PCI order, as setup --check lists them)"},
             "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 600,
                              "description": "how long to wait for it to be ready (default 50: some AI apps end "
                                             "a tool call after 60 s)"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_stop", "title": "Stop Strata",
         "description": "Stop the Strata model: the engine is unloaded through the server's own API (it frees its "
                        "VRAM and RAM itself), then the server this tool started is ended. A Strata started "
                        "elsewhere (e.g. its own window) is only unloaded - close its window to end it. Refuses while "
                        "a request is running unless force=true.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "force": {"type": "boolean", "description": "stop even while a request is running"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_logs", "title": "Strata logs",
         "description": "The last lines of a Strata log: 'server' (the model server this tool started: loading, "
                        "requests, errors), 'engine' (the engine's own log for a model), 'setup' (the install). "
                        "Default: the setup log while an install runs, else the server log. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "source": {"type": "string", "enum": ["server", "engine", "setup"]},
             "model": installed_model,
             "lines": {"type": "integer", "minimum": 1, "maximum": 500, "description": "default 80"}}},
         "annotations": ro},
        {"name": "strata_benchmark", "title": "Strata speed test",
         "description": "A short fixed request (greedy, thinking off) through the running Strata server; reports "
                        "the output speed (tokens/s) and prompt speed. Takes 10-60 s. Refuses while the model is "
                        "busy.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "max_tokens": {"type": "integer", "minimum": 16, "maximum": 512, "description": "default 200"}}},
         "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}},
        {"name": "strata_connect_info", "title": "Connect to Strata",
         "description": "How to point other apps at the local Strata server: the OpenAI and Anthropic base URLs, "
                        "the model id, whether an API key is needed, and ready-to-paste settings for Claude Code, "
                        "Codex CLI, the OpenAI SDK, curl, Cursor/Continue/Open WebUI. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"port": port}},
         "annotations": ro},
    ]


class Tools:
    def __init__(self, strata: Strata):
        self.s = strata

    def defs(self) -> list:
        return tool_defs(*self.s.tables()[:3])

    def call(self, name: str, args) -> dict:
        defs = {d["name"]: d for d in self.defs()}
        if name not in defs:
            raise KeyError(name)
        args = check_schema(defs[name]["inputSchema"], args)
        return getattr(self, name)(**args)

    # ---- strata_status
    def strata_status(self, port=None, hardware=True) -> dict:
        s = self.s
        s.reap()
        servers = []
        for p in s.ports(port):
            info = s.probe(p)
            if info is not None:
                servers.append(info)
        tracked = s.tracked_server()
        out: dict = {}
        running = [x for x in servers if not x.get("other")]
        if tracked and not tracked.get("ended"):
            mine = next((x for x in running if x["port"] == tracked["port"]), None)
            if mine:
                mine["started_by"] = f"this MCP server (pid {tracked['pid']})"
            else:
                out["loading"] = {"model": tracked.get("model"), "port": tracked["port"], "pid": tracked["pid"],
                                  "since": tracked.get("started"),
                                  "last_log_line": (read_tail(Path(tracked.get("log", "")), 1) or [""])[-1] or None}
        elif tracked and tracked.get("ended"):
            out["last_start"] = {"model": tracked.get("model"), "ended": True,
                                 "last_log_lines": read_tail(Path(tracked.get("log", "")), 6)}
        for x in running:
            x.setdefault("started_by", "outside this MCP server (its own window, a start script or a service)")
        out["running"] = running
        others = [x for x in servers if x.get("other")]
        if others:
            out["ports_used_by_other_programs"] = others
        if running:
            r = running[0]
            summary = (f"Strata is running on port {r['port']}: {r.get('model')}"
                       + (", busy" if r.get("busy") else ", idle")
                       + ("" if r.get("loaded", True) else " (model unloaded; the next request loads it)"))
        elif "loading" in out:
            summary = f"Strata is loading {out['loading']['model']} (port {out['loading']['port']})"
        else:
            summary = "Strata is not running"
        inst = s.installed()
        job = s.install_job()
        if job:
            out["install"] = job
            if job["running"]:
                summary += f"; an install is running ({job.get('step') or 'starting'})"
        if not inst["models"]:
            summary += "; no model is installed yet"
        else:
            summary += "; installed: " + ", ".join(m["model"] for m in inst["models"])
        out["installed"] = inst
        if hardware:
            hw = s.hardware()
            out["hardware"] = hw
            out["recommendation"] = s.recommend(hw)
        return {"summary": summary, **out}

    # ---- strata_models
    def strata_models(self) -> dict:
        s = self.s
        hw = s.hardware()
        _m, _f, contexts, src = s.tables()
        rec = s.recommend(hw)
        return {"summary": "Model families and sizes setup offers" + (
                    f"; recommended for this PC: {rec['title']}, {rec['context'] // 1024}K context"
                    if rec.get("model") else f"; no model fits this PC: {rec['why']}"),
                "families": s.model_table(hw), "contexts": contexts,
                "context_note": "longer contexts need more VRAM (fewer experts fit on the GPU); past 262144 setup adds "
                                "rope scaling (experimental)",
                "recommendation": rec, "source": src,
                "this_pc": {"ram_gb": hw.get("ram_gb"),
                            "gpus": [{k: g.get(k) for k in ("index", "name", "vram_gb", "usable")}
                                     for g in hw.get("gpus", [])]}}

    # ---- strata_install
    def check_data_dir(self, raw: str) -> Path:
        s = self.s
        if raw.startswith("//"):
            raise ToolError("data_dir: network paths are not allowed; use a local drive")
        p = Path(os.path.expandvars(raw)).expanduser()
        if not p.is_absolute():
            raise ToolError("data_dir must be an absolute path (e.g. /mnt/big/XeStrata-data)")
        if ".." in p.parts:
            raise ToolError("data_dir must not contain '..'")
        n = norm(p)
        current = norm(s.data_dir())
        if n == current or is_inside(n, norm(s.root)) or n == norm(s.root.parent / "XeStrata-data"):
            pass
        else:
            if not p.name.lower().startswith("xestrata"):
                raise ToolError(f"data_dir {raw!r}: a new data folder must be named XeStrata... (e.g. "
                                f"{p.parent / 'XeStrata-data'}), so that no other folder is filled with "
                                "model files")
            if not p.parent.is_dir():
                raise ToolError(f"data_dir {raw!r}: its parent folder {p.parent} does not exist")
            if any(n == f or is_inside(n, f) for f in system_folders()):
                raise ToolError(f"data_dir {raw!r} is inside a system folder")
            if p.exists() and not p.is_dir():
                raise ToolError(f"data_dir {raw!r} is a file")
        return p

    def strata_install(self, family=None, model=None, context=None, vision=None, kv=None, gpu=None, port=None,
                       data_dir=None, confirm=False, cancel=False) -> dict:
        s = self.s
        with s.lock:
            job = s.install_job()
            if cancel:
                if not job or not job["running"]:
                    return {"summary": "no install is running", "install": job}
                j = s.state("install")
                res = s.state("install-result")
                child = res.get("child_pid") if res.get("job") == j.get("job") else None
                if child:
                    proc_kill_tree(child, res.get("child_ident"))
                proc_kill_tree(j["pid"], j["ident"])
                wait_gone(j["pid"], j["ident"], 15)
                return {"summary": "the install was stopped; calling strata_install again resumes it (downloads "
                                   "continue where they stopped)", "install": s.install_job()}
            if job and job["running"]:
                return {"summary": f"an install is running: {job.get('step') or 'starting'}"
                                   + (f", downloading {job['download']['file']} {job['download']['percent']}%"
                                      if job.get("download") else ""),
                        "install": job, "follow": "call strata_install again for progress, or strata_logs "
                                                  "source=setup for the full output"}
            given = [x for x in (family, model, context, vision, kv, gpu, port, data_dir)
                     if x is not None]
            if job and not given and not confirm:
                return {"summary": f"the last install {job['result']}"
                                   + (f": {job['error']}" if job.get("error") else ""),
                        "install": job,
                        "next": "strata_start" if job.get("exit_code") == 0 else
                                "fix the problem above and call strata_install again with confirm=true (finished "
                                "steps are skipped)"}
            plan, args = self.install_plan(family, model, context, vision, kv, gpu, port, data_dir)
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended"):
                raise ToolError("Strata is running (started by strata_start): stop it first with strata_stop - setup "
                                "may update the engine files it uses")
            if not confirm:
                return {"summary": f"Plan: install {plan['title']} ({plan['download_gb']:.0f} GB download"
                                   f"{' at most; part of it is already here' if plan['partly_downloaded'] else ''}). "
                                   "Nothing was done yet: ask the user, then call strata_install again with the "
                                   "same arguments and confirm=true.", "plan": plan}
            if plan["disk_short"]:
                raise ToolError(plan["disk_short"])
            return self.launch_install(plan, args)

    def install_plan(self, family, model, context, vision, kv, gpu, port, data_dir):
        s = self.s
        models, families, contexts, _src = s.tables()
        if sys.version_info < (3, 10):
            raise ToolError("this Python is too old: Strata's setup needs Python 3.10+; run ./setup.sh once instead "
                            "(it installs Python)")
        hw = s.hardware()
        rec = s.recommend(hw)
        if model is None and family is None:
            if not rec.get("model"):
                raise ToolError("no model was given and none fits this PC: " + rec["why"])
            family, model = rec["family"], rec["model"]
            context = context or rec["context"]
        elif model is None:
            sizes = s.sizes_of(models, family)
            model = rec["model"] if rec.get("family") == family else \
                ("IQ3_XXS" if "IQ3_XXS" in sizes and hw.get("ram_gb", 0) >= 60 else sizes[0])
        elif family is None:
            fams = [f for f in families if model in s.sizes_of(models, f)]
            family = "qwen" if "qwen" in fams else fams[0]
        sizes = s.sizes_of(models, family)
        if model not in sizes:
            raise ToolError(f"{families[family]['title']} has no {model}; its sizes: {', '.join(sizes)}")
        if context is None:
            usable = [g for g in hw.get("gpus", []) if g.get("usable")]
            vram = max((g["vram_gb"] for g in usable), default=12)
            context = 32768 if vram < 14 else 65536 if vram < 20 else 131072
        if vision in ("yes", "gpu", "cpu") and models[model].get("vision", families[family].get("vision")) is False:
            raise ToolError(f"images are not available with {families[family]['title']} {model} yet: use vision=no")
        target = self.check_data_dir(data_dir) if data_dir else s.data_dir()
        tag = (families[family].get("tag", "") + model)
        have_dir = target / "models" / tag
        partly = have_dir.is_dir() and any(have_dir.glob("*.gguf*"))
        need = (8 if partly else models[model]["download_gb"] + 8) + (1 if vision in ("yes", "gpu", "cpu") else 0)
        free = s.disk_free(target)
        short = None
        if free is not None and free < need:
            short = (f"not enough free disk space for {model}: about {need:.0f} GB needed at {target}, "
                     f"{free:.0f} GB free; free some space or choose data_dir on a bigger drive "
                     "(e.g. /mnt/big/XeStrata-data)")
        args = ["--yes", "--no-start", "--family", family, "--model", model, "--context", str(context),
                "--vision", vision or "no"]
        if family in ("qwen", "coder"):
            args += ["--experimental-speed-projection", "off"]
        for flag, v in (("--kv", kv), ("--gpu", gpu), ("--port", port),
                        ("--data-dir", str(target) if data_dir else None)):
            if v is not None:
                args += [flag, str(v)]
        fit = None
        ram = hw.get("ram_gb")
        if ram is not None:
            need_ram = models[model]["ram_gb"]
            fit = "fits" if ram >= need_ram else f"needs ~{need_ram} GB of RAM, this PC has {ram:.0f} GB"
        installed = (s.root / f"{CONFIG}{tag.lower()}.json").exists()
        plan = {"title": f"{families[family]['title']} {model}", "family": family, "model": model,
                "context": context, "images": vision or "no", "download_gb": models[model]["download_gb"],
                "partly_downloaded": partly, "ram": fit, "data_folder": str(target), "disk_free_gb": free,
                "disk_needed_gb": round(need), "disk_short": short, "already_installed": installed,
                "setup_command": "./setup.sh " + " ".join(args),
                "takes": "about 20-90 minutes, mostly the download; the PC stays usable",
                "recommended_for_this_pc": rec if rec.get("model") else None}
        if installed:
            plan["note"] = ("this model is already installed: running setup again updates the engine and applies "
                            "the options (finished steps are skipped)")
        if families[family].get("license"):
            plan["license"] = families[family]["license"]
        if families[family].get("experimental"):
            plan["experimental"] = True
        return plan, args

    def launch_install(self, plan: dict, args: list) -> dict:
        s = self.s
        s.state_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"{int(time.time())}-{os.getpid()}"
        log_path = s.state_dir / "setup.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== strata-mcp: install {plan['title']} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        spec = {"job": job_id, "root": str(s.root), "python": s.run_python(), "base_python": sys.executable,
                "create_venv": s.python_override is None and not s.venv_python().exists(),
                "args": args, "result": str(s.state_dir / "install-result.json")}
        spec_path = s.state_dir / "install-job.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        s.save_state("install-result", None)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        p = spawn_detached([sys.executable, str(Path(__file__).resolve()), "--install-job", str(spec_path)],
                           s.root, log_path, env)
        s.children.append(p)
        time.sleep(0.2)
        ident = proc_identity(p.pid)
        if ident is None and p.poll() and s.state("install-result").get("job") != job_id:
            raise ToolError("setup did not start: " + " | ".join(read_tail(log_path, 5)))
        s.save_state("install", {"job": job_id, "pid": p.pid, "ident": ident, "log": str(log_path),
                                 "model": plan["model"], "family": plan["family"],
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": args})
        return {"summary": f"Install of {plan['title']} started in the background (about 20-90 minutes, mostly the "
                           f"{plan['download_gb']:.0f} GB download). Call strata_install again to see the progress; "
                           "when it is done, start it with strata_start.",
                "plan": plan, "log": str(log_path), "pid": p.pid}

    # ---- strata_start
    def strata_start(self, model=None, port=None, gpu=None, wait_seconds=50) -> dict:
        s = self.s
        with s.lock:
            job = s.install_job()
            if job and job["running"]:
                raise ToolError("an install is still running (strata_install shows its progress); start the model "
                                "when it is done")
            cfg_path = s.find_config(model)
            desc = s.describe_config(cfg_path)
            if not desc["ready"]:
                raise ToolError(f"{cfg_path.name} refers to missing files ({', '.join(desc['missing_files'])}): run "
                                "the install again (strata_install with confirm=true) to repair it")
            port = port or desc["port"] or DEFAULT_PORT
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended"):
                info = s.probe(tracked["port"])
                return {"summary": f"Strata is already {'running' if info else 'loading'} ({tracked.get('model')}, "
                                   f"port {tracked['port']}, started by this MCP server)"
                                   + ("" if tracked.get("model") == desc["model"] else
                                      ": stop it first (strata_stop) to start another model"),
                        "server": info}
            info = s.probe(port, deep=False)
            if info and not info.get("other"):
                return {"summary": f"a Strata server already runs on port {port} ({info.get('model')}), started "
                                   "outside this MCP server; stop it (close its window) or use another port",
                        "server": info}
            if info or not port_free(port):
                raise ToolError(f"port {port} is used by another program; choose another port (port=...)")
            py = s.run_python()
            if not Path(py).exists():
                raise ToolError("Strata's Python environment (.venv) is missing: run the install again")
            cmd = [py, str(s.root / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
                   "--port", str(port)]
            if gpu is not None:
                cmd += ["--gpu", str(gpu)]
            log_path = s.state_dir / "server.log"
            s.state_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== strata-mcp: start {desc['model']} on port {port} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                os.utime(cfg_path)                      # the most recently used model, as setup's start marks it
            except OSError:
                pass
            p = spawn_detached(cmd, s.root, log_path, env)
            s.children.append(p)
            time.sleep(0.1)
            ident = proc_identity(p.pid)
            s.save_state("server", {"pid": p.pid, "ident": ident, "port": port, "model": desc["model"],
                                    "config": cfg_path.name, "log": str(log_path),
                                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        return self.wait_ready(p, ident, port, desc, wait_seconds, log_path)

    def wait_ready(self, p, ident, port, desc, wait_seconds, log_path) -> dict:
        s = self.s
        end = time.time() + wait_seconds
        while True:
            if p.poll() is not None or not proc_alive(p.pid, ident):
                tail = read_tail(log_path, 15)
                end_group_leftovers(p.pid)
                s.save_state("server", None)
                raise ToolError(f"Strata stopped while starting (exit code {p.poll()}). Its last output:\n"
                                + "\n".join(tail))
            info = s.probe(port)
            if info and not info.get("other"):
                return {"summary": f"Strata is running: {info.get('model')} on http://127.0.0.1:{port}/v1 "
                                   "(OpenAI and Anthropic APIs; chat page at http://127.0.0.1:" + str(port) + "/)",
                        "server": {**info, "pid": p.pid}}
            if time.time() >= end:
                return {"summary": f"Strata is loading {desc['model']} (this takes 1-3 minutes; the PC may be slow "
                                   "meanwhile). Check with strata_status in a minute.",
                        "loading": {"pid": p.pid, "port": port,
                                    "last_log_line": (read_tail(log_path, 1) or [""])[-1] or None}}
            time.sleep(1.0)

    # ---- strata_stop
    def strata_stop(self, port=None, force=False) -> dict:
        s = self.s
        with s.lock:
            tracked = s.tracked_server()
            if tracked and tracked.get("ended"):
                s.save_state("server", None)
                tracked = None
            if tracked:
                port = tracked["port"]
                key = s.api_key_for(port)
                st, body = http_json(f"http://127.0.0.1:{port}/status", api_key=key, timeout=3)
                if st == 200 and isinstance(body, dict) and body.get("busy") and not force:
                    raise ToolError("a request is running on Strata right now; wait for it, or call strata_stop "
                                    "with force=true to stop anyway")
                st, body = http_json(f"http://127.0.0.1:{port}/unload", "POST", api_key=key, timeout=120)
                unloaded = (body or {}).get("status") if isinstance(body, dict) else None
                if st == 409 and not force:
                    raise ToolError("Strata is busy (a request is running or queued); call again with force=true "
                                    "to stop anyway")
                proc_terminate(tracked["pid"], tracked["ident"])   # POSIX: SIGTERM = the server's own Ctrl+C path
                clean = wait_gone(tracked["pid"], tracked["ident"], 30)
                if not clean:
                    proc_kill_tree(tracked["pid"], tracked["ident"])
                    wait_gone(tracked["pid"], tracked["ident"], 15)
                s.reap()
                gone = not proc_alive(tracked["pid"], tracked["ident"])
                if gone:
                    end_group_leftovers(tracked["pid"])
                    s.save_state("server", None)
                return {"summary": "Strata stopped" + ("" if gone else " - but its process still runs; see strata_logs")
                                   + (" (the engine unloaded itself first)" if unloaded == "unloaded" else ""),
                        "model": tracked.get("model"), "port": port, "engine_unloaded": unloaded,
                        "ended_cleanly": clean}
            for p in s.ports(port):
                info = s.probe(p, deep=False)
                if info and not info.get("other"):
                    key = s.api_key_for(p)
                    st, body = http_json(f"http://127.0.0.1:{p}/unload", "POST", api_key=key, timeout=120)
                    r = body.get("status") if isinstance(body, dict) else None
                    if st == 409:
                        raise ToolError("Strata is busy (a request is running); try again when it is done")
                    if st == 401:
                        raise ToolError("Strata (started outside this MCP server) wants an API key this folder's "
                                        "configs do not have; close its window to stop it")
                    return {"summary": f"Strata on port {p} was started outside this MCP server (its own window or a "
                                       f"service), so only its model was unloaded ({r}): its VRAM and RAM are free, "
                                       "but the server still listens and loads the model again on the next request. "
                                       "Close its window to end it completely.",
                            "port": p, "engine_unloaded": r}
            return {"summary": "Strata is not running"}

    # ---- strata_logs
    def strata_logs(self, source=None, model=None, lines=80) -> dict:
        s = self.s
        if source is None:
            job = s.install_job()
            source = "setup" if job and job["running"] else "server"
        if source == "setup":
            path = s.state_dir / "setup.log"
        elif source == "server":
            path = s.state_dir / "server.log"
        else:
            cfg = s.find_config(model or (s.state("server").get("model")))
            raw = s.read_config(cfg).get("log")
            if not raw:
                raise ToolError(f"{cfg.name} names no engine log")
            path = Path(raw)
            if not s.allowed(path):
                raise ToolError(f"{cfg.name}'s log {raw} is outside the Strata folder and its data folder: not shown")
        if not path.exists():
            return {"summary": f"no {source} log yet ({path})", "log": str(path), "lines": []}
        tail = read_tail(path, lines)
        return {"summary": f"last {len(tail)} lines of the {source} log", "log": str(path), "lines": tail}

    # ---- strata_benchmark
    def strata_benchmark(self, port=None, max_tokens=200) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if target is None:
            raise ToolError("Strata is not running: start it first (strata_start)")
        p = target["port"]
        key = s.api_key_for(p)
        st, stat = http_json(f"http://127.0.0.1:{p}/status", api_key=key, timeout=3)
        if st == 200 and isinstance(stat, dict) and (stat.get("busy") or stat.get("queued")):
            raise ToolError("Strata is busy with a request; run the speed test when it is idle")
        body = {"model": target.get("model") or "strata", "max_tokens": max_tokens, "temperature": 0,
                "reasoning_effort": "none", "stream": False,
                "messages": [{"role": "user", "content": BENCH_PROMPT}]}
        t0 = time.time()
        st, r = http_json(f"http://127.0.0.1:{p}/v1/chat/completions", "POST", body, api_key=key, timeout=600)
        wall = time.time() - t0
        if st != 200 or not isinstance(r, dict):
            msg = (r or {}).get("error", {}).get("message") if isinstance(r, dict) else None
            raise ToolError(f"the request failed (HTTP {st}): {msg or 'no answer'}")
        usage, tm = r.get("usage") or {}, r.get("timings") or {}
        n = usage.get("completion_tokens") or tm.get("predicted_n") or 0
        out = {"model": r.get("model"), "port": p, "prompt_tokens": usage.get("prompt_tokens"),
               "output_tokens": n, "wall_s": round(wall, 2),
               "output_tok_s": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
               "prompt_tok_s": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
               "wall_tok_s": round(n / wall, 1) if wall > 0 and n else None,
               "finish_reason": ((r.get("choices") or [{}])[0]).get("finish_reason"),
               "settings": f"greedy (temperature 0), thinking off, max_tokens {max_tokens}, a fixed story prompt"}
        speed = out["output_tok_s"] or out["wall_tok_s"]
        prompt = f" (prompt {out['prompt_tok_s']} tokens/s)" if out["prompt_tok_s"] else ""
        out["summary"] = f"{n} tokens at {speed} tokens/s{prompt}, {wall:.1f} s in total"
        out["note"] = ("output speed depends on the context length, the GPU's free VRAM and what else the PC does; "
                       "the first request after a start is slower")
        return out

    # ---- strata_connect_info
    def strata_connect_info(self, port=None) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if port is None:
            cfgs = s.configs()
            port = target["port"] if target else (s.describe_config(cfgs[0])["port"] if cfgs else DEFAULT_PORT)
        model = (target or {}).get("model") or "strata"
        key_needed = (target or {}).get("api_key_required")
        if key_needed is None:
            key_needed = any(s.describe_config(c)["api_key_set"] for c in s.configs())
        key = "<your api_key from xestrata-<model>.json>" if key_needed else "none"
        base = f"http://127.0.0.1:{port}"
        return {
            "summary": (f"Strata {'runs' if target else 'will run'} at {base}: OpenAI base URL {base}/v1, Anthropic "
                        f"base URL {base}" + (" (an API key is required)" if key_needed else " (any API key works)")),
            "running": bool(target), "openai_base_url": f"{base}/v1", "anthropic_base_url": base,
            "anthropic_messages_endpoint": f"{base}/v1/messages", "chat_page": base + "/", "model": model,
            "api_key": ("required: the \"api_key\" value in the XeStrata folder's xestrata-<model>.json"
                        if key_needed else "not required (send any value)"),
            "examples": {
                "claude_code": {"env": {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key,
                                        "ANTHROPIC_MODEL": "claude-sonnet-4-5"},
                                "note": "Claude Code needs a Claude model name it knows; Strata ignores the name"},
                "codex_cli": {"file": "~/.codex/config.toml",
                              "toml": f'model = "{model}"\nmodel_provider = "strata"\n\n[model_providers.strata]\n'
                                      f'name = "Strata"\nbase_url = "{base}/v1"\nwire_api = "chat"'},
                "openai_python": (f'from openai import OpenAI\nclient = OpenAI(base_url="{base}/v1", api_key="{key}")\n'
                                  f'r = client.chat.completions.create(model="{model}", messages=[{{"role": "user", '
                                  f'"content": "Hello!"}}])\nprint(r.choices[0].message.content)'),
                "curl": (f"curl {base}/v1/chat/completions -H \"Content-Type: application/json\" "
                         + (f"-H \"Authorization: Bearer {key}\" " if key_needed else "")
                         + f"-d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": "
                           "\"Hello!\"}]}'"),
                "openai_compatible_apps": {"base_url": f"{base}/v1", "api_key": key, "model": model,
                                           "apps": "Cursor (Settings > Models > OpenAI base URL override), "
                                                   "Continue, Open WebUI, LM Studio clients, aider "
                                                   f"(--openai-api-base {base}/v1)"}},
            "other_devices": "the server listens on this PC only; to reach it from your network, run setup with "
                             "--host 0.0.0.0 --api-key <secret> (docs/DETAILS.md, 'Using it')"}


# ------------------------------------------------------------------------------------------------ MCP over stdio
INSTRUCTIONS = ("Strata runs a large AI model (Qwen3.8-Flash-Next, a mixture of experts) locally on this PC's GPU and "
                "RAM, with an OpenAI- and Anthropic-compatible server on http://127.0.0.1:8095. Call strata_status "
                "first. To install: strata_install (it returns a plan; get the user's OK for the 58-111 GB download, "
                "then call it with confirm=true; it runs in the background - call it again for progress). Then "
                "strata_start, strata_stop, strata_logs for problems, strata_connect_info to point other apps at it.")


class McpServer:
    def __init__(self, strata: Strata, out=None):
        self.strata = strata
        self.tools = Tools(strata)
        self.out = out
        self.write_lock = threading.Lock()
        self.protocol = PROTOCOL

    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            self.out.write(data)
            self.out.flush()

    @staticmethod
    def result(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    def handle(self, msg) -> dict | None:
        """One JSON-RPC message -> its response (None for a notification or a response)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self.error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
        if "method" not in msg:
            return None                                 # a response to something we never ask: ignored
        method, rid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        if "id" not in msg:
            return None                                 # notifications (initialized, cancelled ...): nothing to say
        if not isinstance(params, dict):
            return self.error(rid, -32602, "params must be an object")
        if method == "initialize":
            asked = str(params.get("protocolVersion") or PROTOCOL)
            self.protocol = asked if asked in PROTOCOLS else PROTOCOL
            return self.result(rid, {"protocolVersion": self.protocol,
                                     "capabilities": {"tools": {"listChanged": False}},
                                     "serverInfo": {"name": "xestrata", "title": "XeStrata", "version": VERSION},
                                     "instructions": INSTRUCTIONS})
        if method == "ping":
            return self.result(rid, {})
        if method == "tools/list":
            return self.result(rid, {"tools": self.tools.defs()})
        if method == "tools/call":
            name = params.get("name")
            try:
                res = self.tools.call(str(name), params.get("arguments"))
                return self.result(rid, self.content(res, error=False))
            except KeyError:
                return self.error(rid, -32602, f"unknown tool: {name}")
            except ToolError as e:
                return self.result(rid, self.content({"error": str(e)}, error=True))
            except Exception as e:                      # noqa: BLE001 - a bug must not end the server
                log("tool failed:\n" + traceback.format_exc())
                return self.result(rid, self.content({"error": f"internal error in {name}: {e}"}, error=True))
        if method in ("resources/list", "prompts/list"):
            return self.result(rid, {method.split("/")[0]: []})
        return self.error(rid, -32601, f"method not found: {method}")

    def content(self, res: dict, error: bool) -> dict:
        text = json.dumps(res, indent=1, ensure_ascii=False)
        out = {"content": [{"type": "text", "text": text}], "isError": error}
        if self.protocol >= "2025-06-18":
            out["structuredContent"] = res
        return out

    def serve(self, inp) -> None:
        """Read JSON-RPC lines until stdin closes; tool calls run in their own threads (a start waits for the model
        while pings and status calls are still answered)."""
        threads = []
        for raw in inp:
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(self.error(None, -32700, "parse error"))
                continue
            batch = isinstance(msg, list)
            if batch or (isinstance(msg, dict) and msg.get("method") == "tools/call"):
                t = threading.Thread(target=self._run, args=(msg,), daemon=True)
                t.start()
                threads.append(t)
                threads = [x for x in threads if x.is_alive()]
            else:
                self._run(msg)
        for t in threads:
            t.join(timeout=5)

    def _run(self, msg) -> None:
        if isinstance(msg, list):
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            if out:
                data = (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")
                with self.write_lock:
                    self.out.write(data)
                    self.out.flush()
            return
        r = self.handle(msg)
        if r is not None:
            self.send(r)


# ------------------------------------------------------------------------------------------------ the install job
def install_job(spec_path: str) -> int:
    """The background half of strata_install (this file started again with --install-job): make .venv like
    setup.sh does when it is missing, run setup, record its exit code."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    root = Path(spec["root"])
    result = Path(spec["result"])

    def record(**kw):
        try:
            old = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        except (OSError, ValueError):
            old = {}
        tmp = result.with_suffix(".tmp")
        tmp.write_text(json.dumps({**old, "job": spec["job"], **kw}, indent=1), encoding="utf-8")
        os.replace(tmp, result)

    py = Path(spec["python"])
    try:
        if spec.get("create_venv"):
            venv = root / ".venv"
            if py.exists() and subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True).returncode:
                print("strata-mcp: .venv has no pip (an earlier run stopped half-way): making it again", flush=True)
                shutil.rmtree(venv, ignore_errors=True)
            if not py.exists():
                print(f"strata-mcp: creating Strata's Python environment in {venv} ...", flush=True)
                r = subprocess.run([spec["base_python"], "-m", "venv", str(venv)])
                if r.returncode or not py.exists():
                    print("\n  [X]  could not create the Python environment (.venv)", flush=True)
                    print("       Linux: install python3-venv (sudo apt install python3-venv) or run ./setup.sh "
                          "once in a terminal", flush=True)
                    record(exit_code=1, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                    return 1
        cmd = [str(py), str(root / "setup.py"), *spec["args"]]
        print("strata-mcp: running " + " ".join(cmd), flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        p = subprocess.Popen(cmd, cwd=str(root), stdin=subprocess.DEVNULL, env=env)
        record(child_pid=p.pid, child_ident=proc_identity(p.pid))
        rc = p.wait()
    except Exception as e:                              # noqa: BLE001
        print(f"\n  [X]  strata-mcp could not run setup: {e}", flush=True)
        rc = 1
    print(f"strata-mcp: setup ended with exit code {rc}", flush=True)
    record(exit_code=rc, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strata's MCP server (stdio): install, start, check and stop Strata "
                                             "from an AI assistant. See docs/DETAILS.md.")
    ap.add_argument("--root", help="the Strata folder (default: the one this file is in)")
    ap.add_argument("--install-job", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.install_job:
        return install_job(a.install_job)
    if sys.version_info < (3, 10):
        log("Python 3.10 or newer is needed")
        return 2
    root = Path(a.root).expanduser().resolve() if a.root else DEFAULT_ROOT
    if not (root / "setup.py").is_file() or not (root / "serve" / "server.py").is_file():
        log(f"{root} is not a Strata folder (no setup.py / serve/server.py)")
        return 2
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                             # stray prints must never corrupt the protocol stream
    server = McpServer(Strata(root), out)
    log(f"ready (Strata folder {root})")
    try:
        server.serve(sys.stdin.buffer)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
