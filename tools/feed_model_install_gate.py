#!/usr/bin/env python3
"""Release gate: install a model of the signed feed from REAL FILES, through setup.py, the way a person's Terminal or the app does.

Why: the feed-model install path had only mock tests until the Qwen4 release-day rehearsal (RESULTS-Q4-P8.md, 2026-11-08), which then found three bugs
(F2 `KeyError: 'mmproj'` for a model with no picture reader, F3 the Flash-Next draft layer fetched for a model whose entry says `drafter: none`, F5 no RAM
budget so a 64 GB PC began writing a second 77 GB copy of the experts) and a fourth one for a model of another shape (F4, the shipped expert profile).
Each of them only shows with a real file of the model's kind, not with the few bytes of the mock tests.

What it does, in a scratch folder (nothing of the real install, %APPDATA% or the data folder is touched, no network beyond 127.0.0.1):
  1. writes a small synthetic qwen4exp GGUF with tools/q4geo_synth.py (random weights; `--tokenizer-from` gives it a real tokenizer);
  2. builds a models-only feed signed with a THROW-AWAY key (packaging/feed_tool.py's own build_feed) whose entry says `drafter: none`, has no picture reader
     and a shape that is not the shipped Flash-Next one;
  3. serves the file from a tiny Hugging Face stand-in on 127.0.0.1 and runs `setup.py` in the app's mode (STRATA_CONTROL_INSTALL=1: no engine, no pip, no build)
     with the feed as the model list: download, SHA-256 check, preparation (the pack), run config;
  4. checks: setup exits 0 and prints neither a Traceback nor `KeyError`; the Flash-Next draft layer was not asked for (the stand-in saw only the model file);
     the owner's two lines are there; the run config reads the file in place with a RAM budget and no second copy of the experts was written; a neutral expert
     profile for the file's own layers x experts is in the pack folder and the config names it; a second run changes nothing and downloads nothing.
  5. with --engine <folder of an engine built for the file's shape>: starts that engine from the run config, sends ONE chat request and checks the answer
     is finite (the file has random weights: the text is empty or noise, the usage numbers and `finish_reason` are what counts).

    python tools/feed_model_install_gate.py
    python tools/feed_model_install_gate.py --tokenizer-from <a real Flash-Next shard 1> --engine <engine folder> --full

`tools/test_feed_model_install_realfile.py` runs step 1 to 4 as a test in `python tools/check` (py-tools). The preparation tools read GGUF files with the vendored
their own GGUF code (tools/strata_gguf.py), so no gguf-py, no llama.cpp checkout, no STRATA_TEST_LLAMA_CPP and no network are needed; the check also fails if setup says it
fetched llama.cpp. Without numpy it is SKIPPED with that reason (never a pass).
Exit codes: 0 pass, 1 a check failed, 2 not run (no numpy, no NVIDIA card on a Windows box), 3 the engine phase failed.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

REV = "0123456789abcdef0123456789abcdef01234567"
REPO = "gate-test/Qwen4-Gate-GGUF"
MODEL_ID = "qwen4-gate"
QUANT = "GATE-Q4"
FILE_NAME = "Qwen4-Gate-GATE-Q4.gguf"
SETUP_TIMEOUT = int(os.environ.get("STRATA_GATE_SETUP_TIMEOUT", "300"))   # seconds for one setup run (the 0.2 GB file takes about 6)
NEUTRAL_LINE = "This model's shape differs from Flash-Next, so a neutral expert profile is used."
NO_DRAFT_LINE = "The model list says this model has no draft layer (MTP), so none is fetched or used."

# the shape of the small file: a q4_lite model cut down to 1 layer of 8 experts, so a run takes seconds and the file is 0.2 GB. --full: the case-B stand-in of the rehearsal
# (4 layers of q4_lite, the shape a `-DSTRATA_GEOMETRY_FILE` engine of that rehearsal was built for)
TINY = dict(layers=1, vocab=2048, set="n_expert=8,n_expert_used=2,n_ff=64")
FULL = dict(layers=4, vocab=None, set="")


class NotRun(Exception):
    pass


def check_gguf_tools() -> None:
    """The preparation tools read and write GGUF files with their own code (tools/strata_gguf.py, tools/strata_quants.py): nothing to find, nothing to
    fetch, and no gguf-py needed (this gate also runs on a copy of the tree without third_party/strata-gguf-py)."""
    for f in ("strata_gguf.py", "strata_quants.py", "strata_quant_tables.py"):
        if not (ROOT / "tools" / f).is_file():
            raise NotRun("tools/%s is missing from this checkout" % f)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Mirror:
    """GET/HEAD/Range for /<repo>/resolve/<rev>/<path> out of one folder, on 127.0.0.1; keeps the list of what was asked for."""

    def __init__(self, root: Path):
        self.root, self.seen = Path(root).resolve(), []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _serve(self, head):
                outer.seen.append(self.path)
                m = re.match(r"^/([^/]+/[^/]+)/resolve/[^/]+/(.+?)(\?.*)?$", self.path)
                p = (outer.root / m.group(1) / m.group(2)).resolve() if m else None
                if p is None or outer.root not in p.parents or not p.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                size = p.stat().st_size
                start, end, code = 0, size - 1, 200
                rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
                if rng:
                    a, b = rng.groups()
                    start = max(0, size - int(b)) if (a == "" and b != "") else int(a or 0)
                    end = min(int(b), size - 1) if (b and a != "") else size - 1
                    code = 206
                n = end - start + 1
                self.send_response(code)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(n))
                self.send_header("X-Linked-Size", str(size))
                if code == 206:
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
                self.end_headers()
                if head:
                    return
                with p.open("rb") as f:
                    f.seek(start)
                    left = n
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except OSError:
                            return
                        left -= len(chunk)

            def do_GET(self):
                self._serve(False)

            def do_HEAD(self):
                self._serve(True)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_synthetic(out: Path, shape: dict, tokenizer_from: str | None) -> None:
    cmd = [sys.executable, "-I", str(ROOT / "tools" / "q4geo_synth.py"), "--shape", "q4_lite", "--layers", str(shape["layers"]), "--recipe", "ud-q4k",
           "--exact-bf16", "--small-ple", "--out", str(out)]
    if shape["vocab"] and not tokenizer_from:
        cmd += ["--vocab", str(shape["vocab"])]
    if shape["set"]:
        cmd += ["--set", shape["set"]]
    if tokenizer_from:
        cmd += ["--tokenizer-from", tokenizer_from]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if "numpy" in (r.stderr or ""):
            raise NotRun("tools/q4geo_synth.py needs numpy")
        raise RuntimeError("q4geo_synth failed: " + (r.stderr or r.stdout)[-600:])


def build_feed(feed: Path, entry: dict, work: Path) -> Path:
    """A models-only feed signed with a throw-away key; returns the file of trusted keys for STRATA_UPDATE_TRUSTED_KEYS."""
    import feed_tool
    from strata_control import signing
    key = signing.generate_key()
    models = work / "models"
    models.mkdir()
    (models / (MODEL_ID + ".entry.json")).write_text(json.dumps(entry, indent=1), encoding="utf-8")
    feed_tool.build_feed(feed, key=key, sequence=1, models_dir=models)
    trust = work / "trust.json"
    trust.write_text(json.dumps({"keys": [signing.public_part(key)]}), encoding="utf-8")
    return trust


def make_entry(model: Path) -> dict:
    return {"id": MODEL_ID, "title": "Qwen4 Gate", "by": "the release gate", "about": "a synthetic model with random weights, for the install gate",
            "licence": "Apache 2.0", "licence_url": "https://example.org/licence", "family": "qwen4exp", "repo": REPO, "revision": REV,
            "pictures": False, "drafter": "none", "context": 32768, "min_engine": "0.1.46",
            # ram_gb and arena_gb are far above any test machine's RAM on purpose: setup must pick the RAM-budget mode (the GGUF read in place) and not the low-RAM mode that
            # writes a second copy of the experts (F5); the 0.2 GB file is all there is to read
            "quants": [{"name": QUANT, "about": "random weights", "ram_gb": 2048, "arena_gb": 2000,
                        "files": [{"name": FILE_NAME, "size": model.stat().st_size, "sha256": sha256_of(model)}]}]}


STUB_SMI = """#!/bin/sh
# the gate's stand-in for nvidia-smi on a box with no NVIDIA card (setup stops at step 1 without one): one 24 GB Ada card, every other question unanswered
case "$*" in
  *memory.total*) echo "0, NVIDIA GeForce RTX 4090, 24564, 8.9, 570.86.16" ;;
esac
exit 0
"""


def nvidia_stub_if_needed(env: dict, work: Path, log) -> None:
    """Setup's step 1 needs an NVIDIA card: nvidia-smi answers on the real boxes (used as it is); on a box without one a POSIX stand-in is put first on PATH."""
    try:
        ok = subprocess.run(["nvidia-smi", "-L"], env=env, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if ok:
        return
    if os.name == "nt":
        raise NotRun("no NVIDIA card (nvidia-smi does not answer) and the stand-in is POSIX only: setup stops at its step 1")
    d = work / "bin"
    d.mkdir(exist_ok=True)
    f = d / "nvidia-smi"
    f.write_bytes(STUB_SMI.encode("utf-8"))
    f.chmod(0o755)
    env["PATH"] = str(d) + os.pathsep + env.get("PATH", "")
    log("no NVIDIA card here: a stand-in nvidia-smi (one 24 GB card) is used for setup's step 1")


def sandbox_env(work: Path, install: Path, feed: Path, trust: Path, port: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "HF_")) and k not in ("PYTHONPATH", "PYTHONHOME")}
    home = work / "home"
    home.mkdir(exist_ok=True)
    env.update({"HOME": str(home), "USERPROFILE": str(home), "APPDATA": str(home / "AppData" / "Roaming"), "LOCALAPPDATA": str(home / "AppData" / "Local"),
                "XDG_DATA_HOME": str(home / ".local" / "share"), "XDG_CONFIG_HOME": str(home / ".config"), "XDG_STATE_HOME": str(home / ".local" / "state"),
                "XDG_CACHE_HOME": str(home / ".cache"), "STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(install), "STRATA_COMPONENTS_DIR": str(work / "comp"),
                "STRATA_UPDATE_FEED": str(feed), "STRATA_UPDATE_TRUSTED_KEYS": str(trust), "STRATA_NO_PATH": "1", "HF_ENDPOINT": f"http://127.0.0.1:{port}",
                "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1", "STRATA_ALLOW_UNPINNED": "1"})
    return env


def kill_tree(proc: subprocess.Popen) -> None:
    """Only the process this script started and its children."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def run_setup(env: dict, data: Path, extra: list[str], timeout: int = SETUP_TIMEOUT) -> tuple[int, str]:
    """setup.py in the app's mode. A run that does not end in `timeout` seconds (a fetch of something the model list did not name keeps retrying) is killed and reported."""
    cmd = [sys.executable, str(ROOT / "setup.py"), "--family", MODEL_ID, "--model", QUANT, "--data-dir", str(data), "--feed", env["STRATA_UPDATE_FEED"],
           "--yes", "--no-start", "--vision", "no", "--context", "32768", "--no-browser", *extra]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            start_new_session=(os.name != "nt"))
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, _ = proc.communicate()
        return 124, (out or "") + f"\n[the gate stopped setup after {timeout} s]\n"


def newest(path: Path, pattern: str) -> list[Path]:
    return sorted(path.rglob(pattern), key=lambda p: p.stat().st_mtime) if path.is_dir() else []


def check(work: Path, tokenizer_from: str | None, shape: dict, log) -> dict:
    """Steps 1 to 4. Returns {"problems": [...], "config": path or None, "out": setup's text}."""
    problems: list[str] = []
    model_dir = work / "mirror" / REPO
    model_dir.mkdir(parents=True)
    model = model_dir / FILE_NAME
    t = time.time()
    make_synthetic(model, shape, tokenizer_from)
    log(f"synthetic file {model.stat().st_size / 1e9:.2f} GB in {time.time() - t:.1f} s")
    feed = work / "feed"
    trust = build_feed(feed, make_entry(model), work)
    install, data = work / "install", work / "data"
    install.mkdir()
    with Mirror(work / "mirror") as mirror:
        env = sandbox_env(work, install, feed, trust, mirror.port)
        nvidia_stub_if_needed(env, work, log)
        t = time.time()
        code, out = run_setup(env, data, [])
        log(f"setup run 1: exit {code} in {time.time() - t:.0f} s")
        asked = [p for p in mirror.seen if "/resolve/" in p]
        if code != 0:
            problems.append(f"setup exited {code}")
        for bad in ("Traceback", "KeyError"):
            if bad in out:
                problems.append(f"setup printed {bad!r}")
        if "llama.cpp source" in out or any("llama.cpp" in p or "ggml-org" in p for p in mirror.seen):
            problems.append("setup fetched llama.cpp (the preparation tools have their own GGUF code: tools/strata_gguf.py)")
        if (install / "third_party" / "llama.cpp").exists():
            problems.append("setup wrote third_party/llama.cpp into the install folder")
        wrong = sorted({p for p in asked if FILE_NAME not in p})
        if wrong:                                       # F3: the Flash-Next draft layer (or anything else of another repository) was asked for
            problems.append("setup asked the model host for more than the model's file: " + ", ".join(wrong[:4]))
        if NO_DRAFT_LINE not in out:
            problems.append("setup did not say that the model list has no draft layer")
        if "Downloading the draft layer" in out:
            problems.append("setup announced a draft layer download for a model with `drafter: none`")
        if shape is TINY or shape["layers"] != 48:
            if NEUTRAL_LINE not in out:
                problems.append("setup did not say it made a neutral expert profile (F4)")
        cfgs = [p for p in newest(install, "*.json") if "run" in p.name.lower() or p.parent.name in ("configs", "run")] or newest(install, "config*.json")
        configs = [p for p in newest(install, "*.json") if _is_run_config(p)]
        config = configs[-1] if configs else None
        if config is None:
            problems.append("no run config was written")
        else:
            problems += config_problems(config, data, install, out)
        extra = sorted(str(p.relative_to(work)) for p in work.rglob("experts.bin*"))
        if extra:                                         # F5: the low-RAM mode copies the experts into a second file
            problems.append("a second copy of the experts was written: " + ", ".join(extra[:3]))
        # run 2: a finished install is left alone and downloads nothing
        before = len(mirror.seen)
        mtime = config.stat().st_mtime if config else None
        t = time.time()
        code2, out2 = run_setup(env, data, [])
        log(f"setup run 2: exit {code2} in {time.time() - t:.0f} s")
        if code2 != 0:
            problems.append(f"the second setup run exited {code2}")
        if len([p for p in mirror.seen[before:] if "/resolve/" in p and "Range" not in p]) and ("Downloading" in out2):
            problems.append("the second run downloaded the model again")
        for bad in ("Traceback", "KeyError"):
            if bad in out2:
                problems.append(f"the second run printed {bad!r}")
    return {"problems": problems, "config": config, "out": out, "work": work, "data": data}


def _is_run_config(p: Path) -> bool:
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "args" in d and ("exe" in d or "model_name" in d)


def config_problems(config: Path, data: Path, install: Path, out: str) -> list[str]:
    problems = []
    d = json.loads(config.read_text(encoding="utf-8-sig"))
    args = [str(a) for a in d.get("args", [])]
    text = " ".join(args)
    if not any(a.endswith(".gguf") or ".gguf" in a for a in args) and "--model" not in args:
        problems.append("the run config names no model file")
    if "--resident-budget-gib" not in args:               # F5: a feed qwen4exp size is read in place with a RAM budget of its experts
        problems.append("the run config has no --resident-budget-gib (the RAM budget of the experts)")
    profiles = [a for a in args if "expert-profile" in a]
    if not profiles:
        problems.append("the run config names no expert profile")
    else:
        for pr in profiles:
            if pr.startswith("--"):
                continue
            if not Path(pr).is_file():
                problems.append(f"the expert profile {pr} is not a file")
    return problems


def start_engine(config: Path, engine: Path, work: Path, log) -> list[str]:
    """Step 5: the engine of the run config, one request. The engine folder replaces the config's `component:` engine."""
    problems: list[str] = []
    d = json.loads(config.read_text(encoding="utf-8-sig"))
    exe = next((p for p in (engine / "strata", engine / "strata.exe", engine / "bin" / "strata", engine / "bin" / "strata.exe") if p.is_file()), None)
    if exe is None:
        return [f"no strata engine program in {engine}"]
    port = free_port()
    cfg = dict(d, exe=str(exe), port=port, host="127.0.0.1")            # the config's `component:strata-engine` is this engine folder
    cfg_file = work / "engine-run.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1")
    env.update({k: str(v) for k, v in (d.get("env") or {}).items()})
    libs = [str(engine)] + [str(x) for x in d.get("lib_dirs", [])]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")])
    args = [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(cfg_file), "--port", str(port), "--host", "127.0.0.1"]
    logfile = work / "engine.log"
    with logfile.open("wb") as lf:
        proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        deadline, ready = time.time() + 600, False
        while time.time() < deadline and proc.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).read()
                ready = True
                break
            except OSError:
                time.sleep(2)
        if not ready:
            return ["the engine did not come up: " + logfile.read_text(errors="replace")[-500:]]
        models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).read())
        body = json.dumps({"model": models["data"][0]["id"], "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 24,
                           "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
        reply = json.loads(urllib.request.urlopen(req, timeout=600).read())
        usage = reply.get("usage") or {}
        log(f"one request: finish {reply['choices'][0].get('finish_reason')}, usage {usage}")
        if not usage.get("completion_tokens"):
            problems.append("the request returned no completion tokens")
        if reply["choices"][0].get("finish_reason") not in ("length", "stop"):
            problems.append("the request did not finish normally")
    except Exception as e:                                # noqa: BLE001
        problems.append(f"the request failed: {e}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-from", help="a real qwen4exp GGUF (shard 1): the synthetic file gets its tokenizer (needed by --engine)")
    ap.add_argument("--engine", type=Path, help="an engine folder built for the file's shape: step 5 starts it and sends one request (needs --full)")
    ap.add_argument("--full", action="store_true", help="the 4-layer q4_lite stand-in of the rehearsal instead of the small file (3.5 GB with a real tokenizer)")
    ap.add_argument("--work", type=Path, help="scratch folder (default: a new temp folder, removed at the end unless --keep)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    if a.engine and not a.full:
        print("--engine needs --full: the engine is built for one shape, the small file is another", file=sys.stderr)
        return 2
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)   # noqa: E731
    try:
        check_gguf_tools()
    except NotRun as e:
        print("NOT RUN:", e)
        return 2
    work = a.work or Path(tempfile.mkdtemp(prefix="strata-feed-install-gate-"))
    if a.work:
        if work.exists() and any(work.iterdir()):
            print("--work needs a new or empty folder", file=sys.stderr)
            return 2
        work.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        res = check(work, a.tokenizer_from, FULL if a.full else TINY, log)
        problems = res["problems"]
        if not problems and a.engine:
            ep = start_engine(res["config"], a.engine, work, log)
            if ep:
                problems, code = ep, 3
        if problems:
            print("FAIL:")
            for p in problems:
                print("  -", p)
            print("---- setup output (tail) ----")
            print("\n".join(res["out"].splitlines()[-40:]))
            code = code if code == 3 else 1
        else:
            print("PASS: feed model installed from a real file" + (" and answered one request" if a.engine else ""))
            code = 0
    except NotRun as e:
        print("NOT RUN:", e)
        code = 2
    finally:
        if a.keep:
            print("kept", work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
