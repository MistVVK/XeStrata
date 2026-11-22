# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The speed measurements of the upstream merges (.claude/skills/upstream-sync/scripts) and of setup's --bench, in one
place: what is measured and how the runs are made, alternated and read.

Three ways of running an engine:

- generate: `xestrata-engine --tokens-file`, one new process a run, 256 greedy tokens after a tier's code-agent
  prompt; with two engines, upstream's first run of a tier writes the text every later run follows (--spec-follow).
- serve: `xestrata-engine --serve` over stdin (GEN, DONE), one new process a run: a chat of four turns, the tier's
  prompt and three follow-ups, 256 tokens a turn, greedy or at a temperature (t07 = 0.7, the default).
- usage: the engine's own serve/server.py, streamed /v1/chat/completions: two short chats and a coding agent, at
  the temperature Qwen recommends (0.7) or another.
- image: the engine's image generator as its server runs it (its serve/images.py: XeStrata's resident xestrata-image,
  measured warm; upstream's sd-cli, a process a request), one 1024x1024 image of the model card's 8 steps from a
  fixed prompt and seed.

The prompts come from XeStrata's sources at commit 58dbbe5.  A package has no git history, so the text is shipped in
data/bench-corpus.json.xz (written from the repository by `make_corpus`) and tokenized with the pack's tokenizer by
the speed-matrix record's rules.  The runs' files (runs/, serve-runs/, usage-runs/) and the tables `collect` writes are
the upstream merges' record layout, which their judge.py reads.
"""
from __future__ import annotations

import json
import lzma
import os
import queue
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
WIN = os.name == "nt"

# ---- Windows: the tree a run starts ends through a job object (upstream 8431dde5d, #4260)
# Windows has no process group to signal: the run's tree used to be ended with `taskkill /T`, which Windows refused
# ("Access is denied") for a tree this very user started, and the engine then kept the GPU for the next run.  A job
# object this process owns ends every process in it in one call, which cannot be refused for one member; it is made
# kill-on-close, so a bench that dies takes its engines with it.  Elsewhere, and when the job cannot be made, nothing
# changes.
if WIN:
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                      "OtherOperationCount", "ReadTransferCount",
                                                      "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _KILL_ON_JOB_CLOSE, _EXTENDED_LIMITS = 0x2000, 9
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)

    def _make_job() -> int:
        job = _k32.CreateJobObjectW(None, None)       # no security attributes: the handle is not inherited
        if not job:
            return 0
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return 0
        return job

    _JOB = _make_job()                                # one for the run: every child of this bench is in it
else:
    _JOB = 0


def contain(proc) -> bool:
    """Put a started process in the run's job, so the run can end it and what it starts in one call that cannot be
    refused for one member.  False: no job (not Windows, or it could not be made)."""
    if not _JOB or proc is None:
        return False
    try:
        return bool(_k32.AssignProcessToJobObject(_JOB, int(proc._handle)))
    except (AttributeError, OSError, ValueError):
        return False


def end_job() -> bool:
    """End every process in the run's job now.  False: no job."""
    return bool(_JOB) and bool(_k32.TerminateJobObject(_JOB, 1))


CORPUS = ROOT / "data" / "bench-corpus.json.xz"
COMMIT = "58dbbe5"
# the token counts of upstream's code-agent prompts (its bench/results/2026-09-29-speed-0126/matrix.json; 262K: the
# README's 259,943), so each tier fills the same share of its context; 192K: 262K's share of 196,608 (setup's context
# for it is 204,800)
TIERS = {"1k": 1017, "4k": 3562, "32k": 31566, "64k": 64162, "128k": 128478, "192k": 194956, "262k": 259943}
# what setup writes for each tier's context (its smallest is 8K): int8 KV above 8K; KV streaming from 64K is
# kv_resident()'s, by the model and this PC's RAM (tier_context)
CONTEXT = {"1k": ["--max-context", "8192"], "4k": ["--max-context", "8192"],
           "32k": ["--max-context", "32768", "--kv", "int8"]}
for _t, _c in (("64k", "65536"), ("128k", "131072"), ("192k", "204800"), ("262k", "262144")):
    CONTEXT[_t] = ["--max-context", _c, "--kv", "int8"]
# the engines' settings besides the model files and the context: what setup writes
FIXED = ["--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5"]

SYSTEM = ("<|im_start|>system\nYou are a coding agent working in the user's repository. You read files with the "
          "read_file tool, edit them with apply_patch, and run commands with the shell tool. Keep changes small, "
          "explain what you change and why, and do not touch files the task does not need.<|im_end|>\n"
          "<|im_start|>user\nThe repository is a C++/SYCL inference engine. These are the files you have read so "
          "far:\n\n")
TASK = ("\nTask: find where the prompt (prefill) path splits a long prompt into chunks, explain how the chunk size is "
        "chosen, and propose a patch that logs the chosen size once per prompt. Start with a short plan."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
FOLLOWUPS = ("Show the patch as a unified diff.",
             "How would you test that the size is logged only once per prompt?",
             "Summarize the change in two sentences for the commit message.")
AGENT_STEPS = 7
# the prompt files kept: this many times the characters the Qwen tokenizer needed for the longest tier, for a
# tokenizer that packs more characters into a token
CORPUS_MARGIN = 1.5

MAX_NEW = 256
ROUNDS = 2                  # measured runs per tier (and mode), after one discarded warm-up
DEADLINE = 2400.0           # seconds a run may take
SLOW = 10.0                 # tok/s: upstream this slow is stopped and not run again in that tier
MODES = ("greedy", "t07")
USAGE_MAX_TOKENS = 1024     # a turn's tokens, the thinking included
TEMPERATURE = 0.7           # the sampled runs' temperature (serve's t07, everyday use), as Qwen recommends
CHATS = {
    "chat-ja": ("あなたは親切なアシスタントです。簡潔に日本語で答えてください。",
                ["週末に東京で雨の日でも楽しめる過ごし方を3つ教えて。",
                 "その中で一番お金がかからないのはどれ？理由も。",
                 "子ども連れの場合に気をつけることは？",
                 "当日の持ち物リストを作って。",
                 "ここまでの内容を3行でまとめて。"]),
    "chat-en": ("You are a helpful assistant. Answer concisely.",
                ["What is the difference between a process and a thread?",
                 "Give a short Python example that starts two threads.",
                 "How would that change with multiprocessing?",
                 "When is the GIL a problem, and when is it not?",
                 "Summarize our conversation in three bullet points."]),
}
CONVERSATIONS = (*CHATS, "agent")       # everyday use's conversations, in the order they run
AGENT_SYSTEM = ("You are a coding agent working in the user's repository, a C++/SYCL inference engine. Use the tools "
                "to read what you need before answering. Keep changes small and explain them.")
AGENT_TASK = ("Find where the prompt (prefill) path splits a long prompt into chunks, explain how the chunk size is "
              "chosen, and propose a patch that logs the chosen size once per prompt.")
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file of the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


# ------------------------------------------------------------------------------------------------ the prompts
def git(repo: Path, *a) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout


def load_tokenizer(path: Path):
    import strata_tokenizer as ST
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    tokens = [""] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    return ST.Tokenizer(tokens, (path / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((path / "token_type.json").read_text(encoding="utf-8")))


def file_block(path: str, text: str) -> str:
    return f"<file path=\"{path}\">\n{text}</file>\n"


def make_corpus(repo: Path, tokenizer: Path) -> dict:
    """The texts at COMMIT: the prompt's files in the tree's order (as many as the longest tier needs, with
    CORPUS_MARGIN) and the agent's tool results (the first .cpp files of src/ over 12,000 characters, cut to 14,000)."""
    tok = load_tokenizer(tokenizer)
    names = [f for f in git(repo, "ls-tree", "-r", "--name-only", COMMIT, "src", "include").split("\n")
             if f.endswith((".cpp", ".hpp", ".h"))]
    need = max(TIERS.values()) - len(tok.encode(SYSTEM, parse_special=True)) - len(tok.encode(TASK, parse_special=True))
    files, have, chars, used = [], 0, 0, None
    for f in names:
        text = git(repo, "show", f"{COMMIT}:{f}")
        files.append([f, text])
        chars += len(text)
        if used is None:
            have += len(tok.encode(file_block(f, text), parse_special=True))
            if have >= need:
                used = chars
        if used is not None and chars >= used * CORPUS_MARGIN:
            break
    agent = []
    for f in git(repo, "ls-tree", "-r", "--name-only", COMMIT, "src").split("\n"):
        if f.endswith(".cpp"):
            text = git(repo, "show", f"{COMMIT}:{f}")
            if len(text) > 12000:
                agent.append(f"// {f}\n{text[:14000]}")
        if len(agent) == AGENT_STEPS - 1:
            break
    return {"commit": COMMIT, "files": files, "agent": agent}


def load_corpus() -> dict:
    return json.loads(lzma.decompress(CORPUS.read_bytes()).decode("utf-8"))


def tier_ids(tok, corpus: dict) -> dict[str, list[int]]:
    """Each tier's prompt and the follow-ups (`turn2`..: the user's next messages of the same chat, each opening with
    the <|im_end|> that closes the answer before it).  A tier the corpus cannot fill with this tokenizer is left out."""
    def enc(s):
        return tok.encode(s, parse_special=True)
    head, tail = enc(SYSTEM), enc(TASK)
    need = max(TIERS.values()) - len(head) - len(tail)
    body: list[int] = []
    for path, text in corpus["files"]:
        body += enc(file_block(path, text))
        if len(body) >= need:
            break
    out = {}
    for tier, n in TIERS.items():
        if n - len(head) - len(tail) <= len(body):
            out[tier] = head + body[:n - len(head) - len(tail)] + tail
    for i, text in enumerate(FOLLOWUPS, 2):
        out[f"turn{i}"] = enc(f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                              "</think>\n\n")
    return out


def write_ids(pack: Path, out: Path, say: Callable = print) -> dict[str, list[int]]:
    """OUT/<tier>.ids and OUT/turn<N>.ids, comma-separated, with the pack's tokenizer."""
    out.mkdir(parents=True, exist_ok=True)
    ids = tier_ids(load_tokenizer(pack / "tokenizer"), load_corpus())
    for name, v in ids.items():
        (out / f"{name}.ids").write_text(",".join(map(str, v)))
        say(f"{name} {len(v)}")
    return ids


def read_ids(folder: Path) -> dict[str, list[int]]:
    return {f.stem: [int(x) for x in f.read_text().split(",")] for f in folder.glob("*.ids")}


# ------------------------------------------------------------------------------------------------ the engines
class Engine:
    """One engine to measure.  `arm`: "xe" or "up" (the record's names).  `cmd`: what starts it (its strata binary,
    or a wrapper and its arguments); `env`: its environment; `cwd`; `root`: its tree, whose serve/server.py (or
    `server`) the usage runs start; `py`: the Python that runs that server; `profile`: its expert profile.  `map`:
    (host prefix, its prefix) when it sees the data folder elsewhere (a container).  `cfg_extra`: what its server's
    config needs besides (its CUDA libraries' folders).  `adapt`: what its setup changes in the arguments it is given
    (upstream's sycl/setup_intel.py)."""

    def __init__(self, name: str, arm: str, cmd: list[str], profile: str, env: dict | None = None,
                 cwd: str | None = None, root: Path | None = None, py: str | None = None, server: str | None = None,
                 map: tuple[str, str] | None = None, cfg_extra: dict | None = None,
                 how: str | None = None, adapt: Callable[[list[str]], list[str]] | None = None):
        self.name, self.arm, self.cmd, self.profile = name, arm, list(cmd), profile
        self.env = dict(os.environ) if env is None else env
        self.cwd, self.root, self.py = cwd, root, py or sys.executable
        self.server = server or "serve/server.py"
        self.map, self.cfg_extra, self.how, self.adapt = map, cfg_extra or {}, how, adapt

    @classmethod
    def from_spec(cls, name: str, arm: str, spec: str, profile: str, root: Path | None = None,
                  py: str | None = None) -> "Engine":
        """XE / UP of the merge scripts: `CMD ARGS...`, optionally after `env K=V ...` (those go to its
        environment)."""
        w = shlex.split(spec)
        extra = {}
        if w and w[0] == "env":
            w = w[1:]
            while w and "=" in w[0]:
                k, v = w.pop(0).split("=", 1)
                extra[k] = v
        return cls(name, arm, w, profile, env={**os.environ, **extra}, cwd=None, root=root, py=py)

    def inside(self, words: list[str]) -> list[str]:
        """The paths as the engine sees them."""
        if not self.map:
            return list(words)
        host, inside = self.map
        return [inside + w[len(host):] if w.startswith(host + "/") else w for w in words]

    def run_args(self, words: list[str]) -> list[str]:
        """An engine's whole argument list: what its setup changes in it (`adapt`), with the paths as it sees them."""
        return self.inside(self.adapt(words) if self.adapt else words)


def model_args(pack: str, native: str | None, ple: str | None, extra: list[str]) -> list[str]:
    """The model's files (the pack, the GGUF's first shard, the shard with per_layer_token_embd) and EXTRA."""
    a = ["--pack", pack] if pack else []   # PACK= (empty): a plain model's engine takes no pack at all
    if native:
        a += ["--native", native]
    if ple:
        a += ["--ple-gguf", ple]
    return a + list(extra)


# what both engines take from a model config besides its files: what setup or the calibration measured on the
# machine (its RAM budget, its share of PCIe), as the merges' EXTRA
CARRY = ("--vram-reserve-mib", "--resident-budget-gib", "--pcie-frac", "--pool-workers", "--adapt-every",
         "--adapt-swaps", "--adapt-decay")


def config_model(args: list[str]) -> tuple[str, list[str], str | None]:
    """A model config's (xestrata-<model>.json) pack, model arguments (its files and CARRY) and draft layer."""
    def arg(flag: str) -> str | None:
        return args[args.index(flag) + 1] if flag in args[:-1] else None
    pack = arg("--pack") or ""
    extra = [x for f in CARRY if (v := arg(f)) is not None for x in (f, v)]
    return pack, model_args(pack, arg("--native"), arg("--ple-gguf"), extra), arg("--mtp")


def model_name(model: list[str]) -> str | None:
    """The name in setup's MODELS of the model whose arguments these are, from its first GGUF shard's name
    (...-IQ3_XXS-00001-of-00002.gguf); None for a model setup installs otherwise (Ornith, llm-jp)."""
    import setup
    native = Path(model[model.index("--native") + 1]).name if "--native" in model[:-1] else ""
    found = [k for k in setup.MODELS if re.search(rf"(^|[-_.]){re.escape(k)}([-.]|$)", native)]
    return max(found, key=len) if found else None


def kv_resident(model: list[str], ctx: int, kv: str = "int8") -> list[str]:
    """--kv-resident 32768 where setup writes it for this model on this PC (setup.kv_streaming_wanted: from 64K, when
    the RAM holds the KV cache beside the model; never for the models outside MODELS).  upstream's setup decides the
    same, with the same RAM per model (v0.1.43), so both engines get it; its sycl/ port decides again
    (sycl/setup_intel.py, tools/bench_upstream.py)."""
    import setup
    name = model_name(model)
    return ["--kv-resident", "32768"] if name and setup.kv_streaming_wanted(name, ctx, kv, setup.ram_gb()) else []


def tier_context(model: list[str], tier: str) -> list[str]:
    """What setup writes for the tier's context, for this model on this PC."""
    return CONTEXT[tier] + kv_resident(model, int(CONTEXT[tier][1]))


def engine_args(model: list[str], profile: str, mtp: str | None, context: list[str]) -> list[str]:
    """The merge scripts' arguments: the model, the engine's profile, setup's settings, the draft layer (none: a
    model installed without it), the context."""
    return [*model, "--expert-profile", profile, *FIXED, *(["--mtp", mtp] if mtp else []), *context]


class Merge:
    """The upstream merge scripts' inputs, from the environment: DATA (the data folder), the model (CONFIG, a model
    config of setup's, xestrata-<model>.json: its files and its RAM and calibrated settings; or its files, PACK,
    NATIVE and PLE, IQ3_XXS by default, and EXTRA, options both engines take), and the two engines (XE, UP with
    XE_PROFILE, UP_PROFILE; XE_ROOT, UP_ROOT and PY for the usage runs)."""

    def __init__(self, usage: bool = False, rounds: int = ROUNDS):
        self.data = Path(need("DATA"))
        g = self.data / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
        self.pack = os.environ.get("PACK", str(self.data / "packs" / "qwen-iq3_xxs"))
        self.model = model_args(self.pack, os.environ.get("NATIVE", f"{g}-00001-of-00002.gguf"),
                                os.environ.get("PLE", f"{g}-00002-of-00002.gguf"),
                                shlex.split(os.environ.get("EXTRA", "")))
        self.mtp: str | None = str(self.data / "mtp" / "rt")
        if os.environ.get("CONFIG"):            # a smaller model on a PC with less RAM, as setup installed it
            cfg = json.loads(Path(os.environ["CONFIG"]).read_text(encoding="utf-8-sig"))
            self.pack, self.model, self.mtp = config_model(cfg["args"])
            self.model += shlex.split(os.environ.get("EXTRA", ""))
        self.engines = []
        for arm in ("xe", "up"):
            a = arm.upper()
            if arm == "up" and os.environ.get("UP") == "none":   # XeStrata alone, judged against its previous record
                continue
            root = Path(need(f"{a}_ROOT")) if usage else None
            e = Engine.from_spec(arm, arm, need(a), need(f"{a}_PROFILE"), root, os.environ.get("PY"))
            if usage and len(e.cmd) != 1:
                sys.exit(f"{a}: one binary or wrapper (with arguments, write a wrapper script)")
            self.engines.append(e)
        self.rounds = int(os.environ.get("ROUNDS", str(rounds)))
        self.slow = float(os.environ.get("SLOW", str(SLOW)))

    def args(self, e: Engine, tier: str) -> list[str]:
        return engine_args(self.model, e.profile, self.mtp, tier_context(self.model, tier))


def need(name: str) -> str:
    v = os.environ.get(name)
    if v is None:
        sys.exit(f"{Path(sys.argv[0]).name}: {name} is not set")
    return v


# ------------------------------------------------------------------------------------------------ reading the logs
def grab(pat: str, text: str, cast: Callable = float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


def output_ids(text: str) -> str | None:
    """A generate run's emitted tokens (its `output :` line), comma-separated: the text later runs follow."""
    m = re.search(r"^output\s*:\s*([\d ]+)$", text, re.M)
    return ",".join(m.group(1).split()) if m else None


def engine_error(text: str) -> str | None:
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.startswith(("#", "PP ", "T "))]
    heads = ("xestrata", "strata", "ERR")   # upstream's engine says strata
    return next((x for x in reversed(lines) if x.startswith(heads)), lines[-1] if lines else None)


def generate_row(text: str) -> dict:
    """A generate run's figures (sync_collect.py's names): the prompt and the output, the drafts accepted, and
    whether the run was stopped as too slow; one stopped in its output takes its prompt rate from the last PP line
    (without PP lines, upstream's sycl/ port, from its prefill line)."""
    stopped = grab(r"^bench: stopped, \S+ (\w+)", text, str)
    pp = re.findall(r"^PP (\d+) (\d+) \d+ ([\d.]+)$", text, re.M)
    row = {"prompt_tokens": grab(r"^prefill\s+(\d+) tokens", text, int),
           "prefill_tok_s": grab(r"^prefill\s+.*->\s+([\d.]+) tok/s", text),
           "decoded": grab(r"^decode\s+(\d+) tokens", text, int),
           "decode_tok_s": grab(r"^decode\s+.*->\s+([\d.]+) tok/s", text),
           "spec_accept": grab(r"^speculation\s+.*\(([\d.]+)\)", text),
           "stopped": stopped}
    if stopped == "output" and pp and pp[-1][0] == pp[-1][1]:
        row["prefill_tok_s"] = float(pp[-1][2])
    elif stopped == "output" and not pp:
        row["prefill_tok_s"] = grab(r"^(?:xe)?strata generate: prefill \d+ tokens in .*?\(([\d.]+) tok/s\)", text)
    return row


# ------------------------------------------------------------------------------------------------ one run
def stop(p: subprocess.Popen) -> None:
    """Ends `p` as the server ends an engine: SIGTERM, on which the engine drains its GPU queues (at most 5 s), then
    SIGKILL after 10 s.  A SIGKILL alone leaves the GPU work it queued behind it, and the next engine's queue can hang
    on the xe driver (issue #62)."""
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def watch_generate(p: subprocess.Popen, log: Path, slow: float, owner: str) -> str | None:
    """Stops `p` once it is plainly at `slow` tok/s or less (0: never): its prompt when the PP progress line says so
    after 30 s, its output when 256 tokens at `slow` tok/s and a minute have passed since the prompt was read (an
    engine without PP lines, upstream's sycl/ port: once its prefill line is out).  Returns the line the log ends
    with, "bench: stopped, <owner> prompt|output ...", or None."""
    t_done = None
    while p.poll() is None:
        time.sleep(5)
        if slow <= 0:
            continue
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        pp = re.findall(r"^PP (\d+) (\d+) (\d+) ([\d.]+)$", text, re.M)
        if not pp:
            if not re.search(r"^(?:xe)?strata generate: prefill \d+ tokens in", text, re.M):
                continue
            pp = [("1", "1", "0", "0")]
        done, total, ms, rate = int(pp[-1][0]), int(pp[-1][1]), int(pp[-1][2]), pp[-1][3]
        if done < total:
            if ms >= 30000 and float(rate) <= slow:
                stop(p)
                return f"bench: stopped, {owner} prompt at {rate} tok/s after {ms // 1000} s"
        elif t_done is None:
            t_done = time.monotonic()
        elif time.monotonic() - t_done > MAX_NEW // slow + 60:
            stop(p)
            return (f"bench: stopped, {owner} output under {slow:g} tok/s ({time.monotonic() - t_done:.0f} s after "
                    "the prompt)")
    return None


def run_generate(e: Engine, args: list[str], ids: Path, log: Path, follow: Path | None = None, slow: float = 0.0,
                 extra_env: dict | None = None, dry: bool = False) -> int | None:
    """One generate run into `log` (the engine's output, then the stop line if it was stopped); the exit code."""
    cmd = [*e.cmd, *e.run_args([*args, "--tokens-file", str(ids), "--max-new", str(MAX_NEW)])]
    if follow is not None:
        cmd += e.inside(["--spec-follow", str(follow)])
    if dry:
        print("DRY generate " + shlex.join(cmd), flush=True)
        return 0
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=e.cwd, env={**e.env, **(extra_env or {})}, stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        timer = threading.Timer(DEADLINE, stop, (p,))   # bench.sh's `timeout 2400`
        timer.start()
        why = watch_generate(p, log, slow, "upstream's" if e.arm == "up" else "XeStrata's")
        rc = p.wait()
        timer.cancel()
    if why:
        with open(log, "a", encoding="utf-8") as f:
            f.write(why + "\n")
    return rc


def temperature_mode(t: float) -> str:
    """A serve mode's name: greedy, or t and ten times the temperature in two digits (t07 = 0.7, t10 = 1.0)."""
    return "greedy" if t == 0 else f"t{round(t * 10):02d}"


def mode_temperature(mode: str) -> float:
    if mode == "greedy":
        return 0.0
    if re.fullmatch(r"t\d\d", mode) and mode != "t00":
        return int(mode[1:]) / 10
    raise ValueError(f"unknown mode {mode!r} (greedy, or t07 for temperature 0.7, t10 for 1.0, ...)")


def sampling(mode: str, rnd: int) -> str:
    """The serve run's sampling keys: greedy, or tNN = temperature NN / 10, top_p 0.95, top_k 20, a seed per round."""
    t = mode_temperature(mode)
    return f"temperature={t:g} top_p=0.95 top_k=20 seed={1000 + rnd} " if t else ""


def usage_sampling(t: float) -> dict:
    """Everyday use's sampling: temperature `t` with what Qwen recommends beside it (top_p 0.8, top_k 20); 0 is
    greedy."""
    return {"temperature": t, "top_p": 0.8, "top_k": 20} if t else {"temperature": 0}


def run_serve(e: Engine, args: list[str], ids: dict, tier: str, mode: str, rnd: int, log: Path,
              slow: float = 0.0, dry: bool = False) -> dict:
    """One process, one chat of 1 + len(FOLLOWUPS) turns: the turns' DONE fields and why it ended early, if it did.
    The first turn at `slow` tok/s or less (its prompt or its output) ends it: "stopped: prompt|output"."""
    cmd = [*e.cmd, "--serve", *e.run_args(args)]
    if dry:
        print("DRY serve " + shlex.join(cmd) + f" | {tier} {mode} {rnd}", flush=True)
        return {"turns": [], "why": None}
    f = open(log, "w", encoding="utf-8")
    f.write("# " + shlex.join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=e.cwd, env=e.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    stdin, stdout = p.stdin, p.stdout
    assert stdin is not None and stdout is not None
    lines: queue.Queue = queue.Queue()

    def pump():
        for x in stdout:
            lines.put(x)
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    end = time.monotonic() + DEADLINE
    im_end = ids["turn2"][0]

    def line():
        try:
            x = lines.get(timeout=max(end - time.monotonic(), 0.1))
        except queue.Empty:
            raise TimeoutError
        if x is not None and not x.startswith("T "):
            f.write(x)
        return x

    turns, why = [], None
    try:
        while (x := line()) is not None and not x.startswith("READY"):
            pass
        if x is None:
            raise EOFError
        prompt = list(ids[tier])
        for turn in range(len(FOLLOWUPS) + 1):
            stdin.write(f"GEN {MAX_NEW} {sampling(mode, rnd)}{','.join(map(str, prompt))}\n")
            stdin.flush()
            gen = []
            while (x := line()) is not None:
                if x.startswith("T "):
                    gen.append(int(x.split()[1]))
                elif x.startswith(("DONE ", "ERR")):
                    break
            if x is None or x.startswith("ERR"):
                why = (x or "the engine ended").strip()
                break
            d = x.split()
            # DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <accepted> <offered> <reused> ... [read]
            done = {"generated": int(d[1]), "prompt": int(d[2]), "prompt_ms": float(d[3]), "decode_ms": float(d[4]),
                    "finish": d[5], "accepted": int(d[6]), "offered": int(d[7]), "reused": int(d[8]),
                    "read": int(d[14]) if len(d) > 14 else int(d[2]) - int(d[8])}
            turns.append(done)
            if turn == 0 and slow > 0:
                if done["read"] / max(done["prompt_ms"], 1e-3) * 1e3 <= slow:
                    why = "stopped: prompt"
                elif done["generated"] / max(done["decode_ms"], 1e-3) * 1e3 <= slow:
                    why = "stopped: output"
                if why:
                    break
            if turn < len(FOLLOWUPS):
                while gen and gen[-1] == im_end:          # the follow-up closes the answer itself
                    gen.pop()
                prompt = prompt + gen + ids[f"turn{turn + 2}"]
        if not why:
            stdin.write("QUIT\n")
            stdin.flush()
    except TimeoutError:
        why = f"killed after {DEADLINE:.0f} s"
    except EOFError:
        why = "the engine ended before READY"
    finally:
        try:
            stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            stop(p)
            p.wait()
        if why:
            f.write(f"bench: {why}\n")
        f.close()
    return {"turns": turns, "why": why}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def usage_post(port: int, body: dict, timeout: float = 1200, temperature: float = TEMPERATURE) -> tuple:
    """One streamed chat completion: (the assistant message, time to the first token, the server's timings)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                 # never the script guard (serve/script_guard.py): the config setup wrote may ask
                                 # for it, and it changes the answers; upstream's server ignores the field
                                 data=json.dumps({"model": "bench", "stream": True, "max_tokens": USAGE_MAX_TOKENS,
                                                  "script_guard": "off", **usage_sampling(temperature),
                                                  **body}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, timings = time.monotonic(), None, "", None
    calls: dict[int, dict] = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            x = json.loads(line[6:])
            timings = x.get("timings") or timings
            for c in x.get("choices") or []:
                d = c.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    first = time.monotonic() - t0
                text += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    k = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "type": "function",
                                                              "function": {"name": "", "arguments": ""}})
                    k["id"] = tc.get("id") or k["id"]
                    k["function"]["name"] += (tc.get("function") or {}).get("name") or ""
                    k["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    msg: dict = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, first, timings


def usage_turns(port: int, seed: int, agent_files: list[str], temperature: float = TEMPERATURE,
                convs: tuple[str, ...] | None = None):
    """Every turn of the conversations (`convs`, default all three): (conversation, turn, ttft, timings)."""
    for name, (system, questions) in CHATS.items():
        if convs and name not in convs:
            continue
        msgs = [{"role": "system", "content": system}]
        for i, q in enumerate(questions):
            msgs.append({"role": "user", "content": q})
            m, ttft, tm = usage_post(port, {"messages": msgs, "seed": seed}, temperature=temperature)
            msgs.append(m)
            yield name, i, ttft, tm
    if convs and "agent" not in convs:
        return
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": AGENT_TASK}]
    for step in range(AGENT_STEPS):
        last = step == AGENT_STEPS - 1
        m, ttft, tm = usage_post(port, {"messages": msgs, "seed": seed, "tools": TOOLS,
                                        "tool_choice": "auto" if last else "required"}, temperature=temperature)
        yield "agent", step, ttft, tm
        if last or not m.get("tool_calls"):
            break
        msgs.append(m)
        for c in m["tool_calls"]:              # the step's file answers the first call; any others get a note
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": agent_files[step] if c is m["tool_calls"][0] else "(not available)"})


def usage_context(model: list[str], ctx: int) -> list[str]:
    """What setup writes for a context, for this model on this PC."""
    return ["--max-context", str(ctx)] + (["--kv", "int8"] if ctx > 8192 else []) + kv_resident(model, ctx)


def run_usage(e: Engine, cfg: dict, seed: int, agent_files: list[str], stem: Path, warmup: bool = False,
              slow: float = 0.0, label: str = "", port: int | None = None, dry: bool = False,
              temperature: float = TEMPERATURE, convs: tuple[str, ...] | None = None) -> dict:
    """The engine's own server with `cfg` (written to STEM.config.json, its log to STEM.txt) on local `port` (None: a
    free one), and the conversations: the turns and why it ended early, if it did.  The first answer at `slow` tok/s
    or less ends it ("stopped: output")."""
    port = port or free_port()
    cfg = {**{k: v for k, v in cfg.items() if k not in ("api_key", "host", "port", "args")}, "port": port,
           "args": e.run_args(cfg["args"])}
    cpath = stem.with_name(stem.name + ".config.json")
    # upstream's server takes strata only, XeStrata's both
    cmd = [e.py, str(Path(e.root or ROOT) / e.server), "--engine", "strata", "--config", str(cpath),  # upstream
           "--port", str(port), "--host", "127.0.0.1"]
    if dry:
        print("DRY usage " + shlex.join(cmd) + " | " + json.dumps(cfg, sort_keys=True), flush=True)
        return {"turns": [], "why": None}
    cpath.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    log = open(stem.with_name(stem.name + ".txt"), "w", encoding="utf-8")
    group: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {  # type: ignore[attr-defined]
        "start_new_session": True}
    srv = subprocess.Popen(cmd, cwd=e.root, env=e.env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           **group)
    contain(srv)                               # the run's job owns the tree it starts (Windows; upstream 8431dde5d)
    turns, why = [], None
    try:
        end = time.monotonic() + 900
        while True:                            # the server answers /health at once; the engine loads after
            if srv.poll() is not None:
                raise RuntimeError("the server ended while loading")
            if time.monotonic() > end:
                raise RuntimeError("not loaded after 900 s")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("loaded"):
                        break
            except OSError:
                pass
            time.sleep(2)
        for name, i, ttft, tm in usage_turns(port, seed, agent_files, temperature, convs):
            turns.append({"conv": name, "turn": i, "ttft_s": ttft, **{k: (tm or {}).get(k) for k in (
                "prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")}})
            t = turns[-1]
            if warmup:                         # the JIT and the first loads: one answer is enough
                break
            if slow > 0 and len(turns) == 1 and t["predicted_ms"] and \
                    t["predicted_n"] / t["predicted_ms"] * 1e3 <= slow:
                why = "stopped: output"
                break
            print(f"{label} {name} {i}: ttft {ttft or 0:.2f} s, read {t['prompt_n']} in {t['prompt_ms']} ms, out "
                  f"{t['predicted_n']} in {t['predicted_ms']} ms", flush=True)
    except (OSError, RuntimeError, ValueError) as err:
        why = str(err)
    finally:
        try:
            if WIN:                            # the whole tree at once, as setup's end_process_group (#48)
                end_job()                      # upstream 8431dde5d: the job ends the tree, and cannot be refused
                try:
                    srv.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if srv.poll() is None:         # no job, or it did not take the tree: taskkill of the tree, then it
                    subprocess.run(["taskkill", "/PID", str(srv.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=60)
                    srv.wait()
            else:
                try:
                    os.killpg(srv.pid, signal.SIGTERM)
                    srv.wait(timeout=120)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(srv.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    srv.wait()
            end_group(srv.pid, f"{label or e.arm}: its server's")
        finally:
            log.close()
    return {"turns": turns, "why": why}


class EngineLeft(RuntimeError):
    """A run's engine still runs after it was stopped and killed (stuck in its GPU driver): the next run would start
    beside it, in what it leaves of the GPU's memory, so the measurements stop here."""


GROUP_END = 120   # s: for the processes left in a run's group to end, and again after they are killed


def group_left(pgid: int) -> list[str]:
    """The processes of process group `pgid` that still run, as "PID name" (zombies left out: they hold no memory,
    and one a container's init does not reap would stay for good).  Windows has no process groups: the processes
    descended from `pgid` by their parent ids (which outlive a parent that ended), the root among them (#48)."""
    left = []
    if WIN:
        import psutil
        procs = list(psutil.process_iter(["pid", "ppid", "name"]))
        tree, grew = {pgid}, True
        while grew:
            more = {p.info["pid"] for p in procs if p.info["ppid"] in tree} - tree
            grew = bool(more)
            tree |= more
        return [f"{p.info['pid']} {p.info['name']}" for p in procs if p.info["pid"] in tree and p.is_running()]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
        except OSError:
            continue
        name_end = stat.rfind(")")
        f = stat[name_end + 2:].split()    # state, ppid, pgrp, ...
        if len(f) > 2 and int(f[2]) == pgid and f[0] != "Z":
            left.append(f"{p.name} {stat[stat.find('(') + 1:name_end]}")
    return left


def end_group(pgid: int, what: str) -> None:
    """#34: wait until every process of a run's group has ended, not only the server: the server ends once it has
    given up on its engine (EngineStuck), and an engine stuck after a GPU fault kept its VRAM while the next run
    started (6674 and 838 MiB free).  The group is killed after GROUP_END s; EngineLeft when a process is still there
    GROUP_END s later."""
    killed = False
    deadline = time.monotonic() + GROUP_END
    while left := group_left(pgid):
        if time.monotonic() > deadline:
            if killed:
                raise EngineLeft(f"{what} engine has not ended ({'; '.join(left[:3])}), {2 * GROUP_END} s after it "
                                 "was stopped, and keeps its GPU memory: the next run would start beside it, so the "
                                 "measurements stop here. The GPU's driver log (sudo dmesg) may say why; a reboot "
                                 "frees the GPU")
            if WIN:
                import psutil
                for pid in (int(x.split()[0]) for x in left):
                    try:
                        psutil.Process(pid).kill()
                    except psutil.Error:
                        pass
            else:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            killed = True
            deadline = time.monotonic() + GROUP_END
        time.sleep(1)


# ------------------------------------------------------------------------------------------------ the alternation
def ordered(engines: list[Engine], first_up: bool) -> list[Engine]:
    """The engines of a round, upstream first when `first_up` (one engine: just it)."""
    if len(engines) < 2:
        return list(engines)
    xe = [e for e in engines if e.arm == "xe"]
    up = [e for e in engines if e.arm == "up"]
    return up + xe if first_up else xe + up


def ab_generate(out: Path, gpu: str, engines: list[Engine], args: Callable[[Engine, str], list[str]], ids: Path,
                tiers: list[str], rounds: int = ROUNDS, follow: bool = True, slow: float = SLOW,
                on_run: Callable | None = None, dry: bool = False) -> None:
    """bench.sh: OUT/runs/GPU-ARM-TIER-ROUND.txt, one discarded warm-up per engine first (GPU-ARM-warmup.txt), the
    engines alternating.  With `follow` and two engines, upstream runs first in a tier and writes its answer
    (OUT/runs/GPU-TIER-follow.csv), and every later run of the tier, both engines', follows it: the rounds and the
    drafts accepted are then the same, and the output speeds differ by the engines alone.  Upstream at `slow` tok/s
    or less is stopped mid-run and not run again in that tier."""
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    follow_on = follow and len(engines) > 1

    def say(e, tier, rc, s):
        # bench.sh's line: the run's prefill and decode lines (spaces squeezed) and the stop line
        print(f"{gpu} {e.arm} {tier} rc={rc} " + "".join(
            re.sub(" +", " ", x) + " " for x in s.splitlines()
            if re.match(r"^(prefill|decode)  |^bench: stopped", x)), flush=True)

    for e in sorted(engines, key=lambda x: x.arm != "xe"):
        log = runs / f"{gpu}-{e.arm}-warmup.txt"
        rc = run_generate(e, args(e, "1k"), ids / "1k.ids", log, dry=dry)
        if not dry:
            say(e, "1k", rc, log.read_text(errors="replace"))
    for t in tiers:
        text, up_slow = None, False
        ftext = runs / f"{gpu}-{t}-follow.csv"
        if not dry:
            ftext.unlink(missing_ok=True)
        for r in range(1, rounds + 1):
            # with follow, upstream first in the first round: its answer is the tier's text
            for e in ordered(engines, (r + follow_on) % 2 == 0):
                if e.arm == "up" and up_slow:
                    continue
                log = runs / f"{gpu}-{e.arm}-{t}-{r}.txt"
                if on_run:
                    on_run(f"generate {t}")
                rc = run_generate(e, args(e, t), ids / f"{t}.ids", log, text, slow if e.arm == "up" else 0.0, dry=dry)
                if dry:
                    if follow_on and e.arm == "up" and text is None:
                        text = ftext
                    continue
                s = log.read_text(errors="replace")
                say(e, t, rc, s)
                if e.arm == "up" and re.search(r"^bench: stopped", s, re.M):
                    up_slow = True
                if follow_on and e.arm == "up" and text is None:
                    ids_text = output_ids(s)
                    if ids_text:
                        ftext.write_text(ids_text + "\n")
                        text = ftext
                        print(f"{gpu} {t}: the text of upstream's first run, {len(ids_text.split(','))} tokens",
                              flush=True)


def ab_serve(out: Path, gpu: str, engines: list[Engine], args: Callable[[Engine, str], list[str]], ids: dict,
             tiers: list[str], modes=MODES, rounds: int = ROUNDS, slow: float = SLOW, on_run: Callable | None = None,
             dry: bool = False) -> None:
    """bench_serve.py: OUT/serve-runs/GPU-ARM-TIER-MODE-ROUND.txt (the log) and .json (the turns), one discarded
    warm-up per engine first (round 0), each run a new process, the engines alternating.  Upstream whose first turn
    reads its prompt or writes its output at `slow` tok/s or less is not run again in that tier and mode."""
    runs = out / "serve-runs"
    runs.mkdir(parents=True, exist_ok=True)

    def one(e, tier, mode, rnd, stem):
        if on_run and rnd:
            on_run(f"serve {tier}")
        rec = run_serve(e, args(e, tier), ids, tier, mode, rnd, runs / f"{stem}.txt",
                        slow if e.arm == "up" else 0.0, dry=dry)
        rec = {"gpu": gpu, "arm": e.arm, "tier": tier, "mode": mode, "round": rnd, **rec}
        if dry:
            return rec
        (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
        rate = lambda n, ms: f"{n / ms * 1e3:.1f}" if ms > 0 else "-"   # noqa: E731
        print(f"{gpu} {e.arm} {tier} {mode} {rnd}: " + " | ".join(
            f"read {t['read']} at {rate(t['read'], t['prompt_ms'])}, out {t['generated']} at "
            f"{rate(t['generated'], t['decode_ms'])}" for t in rec["turns"])
            + (f" ({rec['why']})" if rec["why"] else ""), flush=True)
        return rec

    for e in sorted(engines, key=lambda x: x.arm != "xe"):
        one(e, "1k", "greedy", 0, f"{gpu}-{e.arm}-warmup")
    for tier in tiers:
        for mode in modes:
            up_stopped = False
            for r in range(1, rounds + 1):
                for e in ordered(engines, r % 2 == 0):
                    if e.arm == "up" and up_stopped:
                        continue
                    rec = one(e, tier, mode, r, f"{gpu}-{e.arm}-{tier}-{mode}-{r}")
                    up_stopped |= e.arm == "up" and (rec["why"] or "").startswith("stopped")


def ab_usage(out: Path, gpu: str, engines: list[Engine], cfg: Callable[[Engine], dict], context: int,
             agent_files: list[str], rounds: int = ROUNDS, slow: float = SLOW, on_run: Callable | None = None,
             port: int | None = None, dry: bool = False, temperature: float = TEMPERATURE,
             convs: tuple[str, ...] | None = None) -> None:
    """bench_usage.py: OUT/usage-runs/GPU-ARM-ROUND.json (the turns), .txt (its server's log) and .config.json,
    each engine's server started anew for each run, one discarded warm-up per engine first (round 0: one answer),
    the engines alternating; seed 1000 + the round.  Upstream whose first answer comes at `slow` tok/s or less is
    stopped there and not run again."""
    runs = out / "usage-runs"
    runs.mkdir(parents=True, exist_ok=True)

    def one(e, rnd, stem, warmup=False):
        if on_run:
            on_run("usage" if rnd else None)
        rec = run_usage(e, cfg(e), 1000 + rnd, agent_files, runs / stem, warmup, slow if e.arm == "up" else 0.0,
                        f"{gpu} {e.arm} {rnd}", port, dry=dry, temperature=temperature, convs=convs)
        if dry:
            return rec["why"]
        rec = {"gpu": gpu, "arm": e.arm, "round": rnd, "context": context, "temperature": temperature, **rec}
        (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
        if rec["why"]:
            print(f"{gpu} {e.arm} {rnd}: {rec['why']}", flush=True)
        return rec["why"]

    for e in sorted(engines, key=lambda x: x.arm != "xe"):
        one(e, 0, f"{gpu}-{e.arm}-warmup", warmup=True)
    up_stopped = False
    for r in range(1, rounds + 1):
        for e in ordered(engines, r % 2 == 0):
            if e.arm == "up" and up_stopped:
                continue
            why = one(e, r, f"{gpu}-{e.arm}-{r}")
            up_stopped |= e.arm == "up" and (why or "").startswith("stopped")


# ------------------------------------------------------------------------------------------------ image generation
IMAGE_REQUEST = {"prompt": "A red fox sitting in a snowy forest at dawn, photograph", "size": "1024x1024", "seed": 42}
# Run by the engine tree's own Python with its tools/image/qimage.py and serve/images.py (both trees have modules of
# those names, so each in a process of its own).  "make": the config section its setup writes for this GPU (with
# setup's timed choices where the tree makes them, qimage.tune); "run": one image as its server makes it.  A tree whose
# server keeps a resident image engine (XeStrata's xestrata-image) starts it and makes a warm-up image of another prompt
# first, so the image measured is one the server makes while it runs ("cold_s": the warm-up's, with the start); one
# whose server runs sd-cli for each request (upstream's) runs that command once.
IMAGE_RUNNER = r"""
import json, os, re, subprocess, sys, time
from pathlib import Path
job = json.load(sys.stdin)
tree = Path(job["tree"])
sys.path[:0] = [str(tree), str(tree / "serve"), str(tree / "tools" / "image")]
if job["do"] == "make":
    import qimage as QI
    m = job["make"]
    if hasattr(QI, "IMAGE_EXE"):
        kw = {"model": m["model"]} if m.get("model") else {}
        cfg = QI.image_config(m["quant"], Path(m["exe"]), Path(m["models"]), **kw)
        print(json.dumps(cfg))
        sys.exit(0)
    kw = {"lib_dirs": m.get("lib_dirs") or ()}
    if m.get("gpu") is not None:
        kw["gpu"] = m["gpu"]
    cfg = QI.image_config(m["backend"], m["quant"], Path(m["exe"]), Path(m["models"]), **kw)
    if m["backend"] == "vulkan" and hasattr(QI, "tune"):
        QI.tune(cfg, lambda s: print(s, file=sys.stderr))
    print(json.dumps(cfg))
    sys.exit(0)
import images as IM
cfg = job["cfg"]
if hasattr(IM, "ImageEngine"):
    env = dict(os.environ)
    env.update(job.get("env") or {})
    g = IM.ImageGen({**cfg, "out_dir": str(Path(job["png"]).parent / "image-engine")}, env=env)
    try:
        t0 = time.monotonic()
        g.run(IM.parse_request({"prompt": "A lighthouse on a cliff at night, oil painting", "seed": 1,
                                "size": job["request"].get("size")}, cfg))
        cold = time.monotonic() - t0
        t0 = time.monotonic()
        pngs, fig = g.run(IM.parse_request(dict(job["request"]), cfg))
        wall = time.monotonic() - t0
    except IM.ImageJobFailed as e:
        print(json.dumps({"rc": 1, "error": str(e)}))
        sys.exit(0)
    finally:
        try:   # the daemon too: it would hold its VRAM through the next measurement (the run is warm already)
            g.close(shutdown=True)
        except TypeError:          # an older tree's ImageGen.close() takes no argument
            g.close()
    Path(job["png"]).write_bytes(pngs[0])
    Path(job["log"]).write_text(json.dumps(fig, indent=1) + "\n")
    print(json.dumps({"rc": 0, "seconds": round(wall, 2), "cold_s": round(cold, 2),
                      "text_s": round(fig["text_encode_ms"] / 1000, 3), "sampling_s": fig["sampling_s"],
                      "decode_s": fig["vae_decode_s"], "backend": "xestrata-image"}))
    sys.exit(0)
cmd = IM.sd_cli_args(cfg, IM.parse_request(dict(job["request"]), cfg), job["png"])
t0 = time.monotonic()
p = subprocess.run(cmd, env=IM.image_env(cfg), capture_output=True, text=True, errors="replace")
wall = time.monotonic() - t0
log = p.stdout + p.stderr
Path(job["log"]).write_text(log)
def last(pat):
    m = re.findall(pat, log)
    return float(m[-1]) if m else None
print(json.dumps({"rc": p.returncode, "seconds": round(wall, 2),
                  "text_s": last(r"get_learned_condition completed, taking ([0-9.]+) ?s"),
                  "sampling_s": last(r"sampling completed, taking ([0-9.]+) ?s"),
                  "decode_s": last(r"decode_first_stage completed, taking ([0-9.]+) ?s")}))
"""


def image_job(e: Engine, job: dict, timeout: float = DEADLINE) -> dict:
    """IMAGE_RUNNER in `e`'s tree with its Python -> its last line's JSON, or {"error": ...}."""
    try:
        p = subprocess.run([e.py, "-c", IMAGE_RUNNER], input=json.dumps({"tree": str(e.root), **job}),
                           capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(e.root))
    except (OSError, subprocess.TimeoutExpired) as err:
        return {"error": str(err)}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (p.stderr or p.stdout or f"exit {p.returncode}").strip()[-400:]}


def image_config(e: Engine, backend: str, quant: str, exe: str, models: str, gpu: int | None = None,
                 lib_dirs: list[str] | None = None, model: str | None = None) -> dict:
    """The "images" section `e`'s setup writes for this GPU (its timed choices included) -> the config, or
    {"error": ...}.  `model`: setup's --images name, for XeStrata's tree (upstream's has Qwen-Image-2.1-Turbo
    alone)."""
    return image_job(e, {"do": "make", "make": {"backend": backend, "quant": quant, "exe": exe, "models": models,
                                                "gpu": gpu, "lib_dirs": lib_dirs or [], "model": model}})


def ab_image(out: Path, gpu: str, engines: list[Engine], cfgs: dict[str, dict], rounds: int = ROUNDS,
             on_run: Callable | None = None, dry: bool = False, envs: dict[str, dict] | None = None,
             model: str = "qwen-image-turbo") -> None:
    """OUT/image-runs/GPU-ARM-ROUND.json (the times), .txt (sd-cli's log, or the resident engine's figures) and .png:
    each engine's "images" config (`cfgs`, by arm; `envs`: what a resident engine's environment adds, by arm), one
    discarded warm-up per engine first (the files into the page cache), then the engines alternating, one process an
    image (IMAGE_RUNNER).  `model`: setup's --images name of the configs, for the tables."""
    runs = out / "image-runs"
    runs.mkdir(parents=True, exist_ok=True)

    def one(e, rnd, stem):
        if on_run:
            on_run("image" if rnd else None)
        cfg = cfgs[e.arm]
        if dry:
            print(f"{gpu} {e.arm} image {rnd}: {json.dumps(cfg)}")
            return
        r = image_job(e, {"do": "run", "cfg": cfg, "request": IMAGE_REQUEST, "png": str(runs / f"{stem}.png"),
                          "log": str(runs / f"{stem}.txt"), "env": (envs or {}).get(e.arm)})
        rec = {"gpu": gpu, "arm": e.arm, "round": rnd, "model": model, "steps": cfg.get("steps") or 8,
               "backend": cfg.get("backend"), "quant": cfg.get("quant"),
               "args": cfg.get("args"), "env": cfg.get("env"), **r}
        (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
        took = (f"{r['seconds']} s (sampling {r['sampling_s']} s, VAE {r['decode_s']} s)" if r.get("rc") == 0 else
                f"failed: {r.get('error') or r.get('rc')}")
        print(f"{gpu} {e.arm} image {rnd}: {took}", flush=True)

    for e in sorted(engines, key=lambda x: x.arm != "xe"):
        one(e, 0, f"{gpu}-{e.arm}-warmup")
    for r in range(1, rounds + 1):
        for e in ordered(engines, r % 2 == 0):
            one(e, r, f"{gpu}-{e.arm}-{r}")


# ------------------------------------------------------------------------------------------------ the tables
COLLECT_TIERS = ["1k", "4k", "32k", "64k", "128k", "192k", "262k"]


def collect(out: Path, names: dict[str, str], say: Callable = print) -> None:
    """OUT/runs/*.txt -> OUT/matrix.json (every run) and, per GPU, the prompt and output tables (the median of the
    rounds; a cell no run finished is "-", one stopped as too slow "too slow") as Markdown.  OUT/serve-runs/*.json ->
    OUT/matrix-serve.json and, per GPU and mode, the tier's prompt, the follow-ups (the tokens they read over their
    prompt time) and the output of all four turns.  OUT/usage-runs/*.json -> OUT/matrix-usage.json and a table per
    GPU: per conversation, the time to the first token (the median of its turns), the prompt and the output speed
    (the server's timings, summed).  OUT/image-runs/*.json -> OUT/matrix-image.json and a table per GPU: the seconds
    an image, its sampling and its VAE decode, and the backend.  `names`: the arms' names in the tables ({"xe": ...,
    "up": ...})."""
    rows = []
    for f in sorted((out / "runs").glob("*.txt")):
        m = re.fullmatch(r"(.+)-(xe|up)-(\w+)-(\d+)", f.stem)
        if not m:                                 # the warm-ups
            continue
        r = generate_row(f.read_text(errors="replace"))
        rows.append({"gpu": m.group(1), "arm": m.group(2), "engine": names.get(m.group(2), m.group(2)),
                     "tier": m.group(3), "round": int(m.group(4)), **{k: r[k] for k in (
                         "prompt_tokens", "prefill_tok_s", "decoded", "decode_tok_s", "spec_accept", "stopped")}})
    if rows or (out / "runs").is_dir():
        (out / "matrix.json").write_text(json.dumps(rows, indent=1) + "\n")
    for gpu in sorted({r["gpu"] for r in rows}):
        for key, title, fmt in (("prefill_tok_s", "Prefill (tokens/s)", "{:,.0f}"),
                                ("decode_tok_s", "Decode (tokens/s)", "{:.1f}")):
            say(f"\n### {gpu}: {title}\n\n| Engine | " + " | ".join(t.upper() for t in COLLECT_TIERS) + " |")
            say("| --- |" + " ---: |" * len(COLLECT_TIERS))
            for name in names.values():
                cells = []
                for t in COLLECT_TIERS:
                    v = [r[key] for r in rows if (r["gpu"], r["engine"], r["tier"]) == (gpu, name, t) and r[key]]
                    slow = [r for r in rows if (r["gpu"], r["engine"], r["tier"]) == (gpu, name, t) and r["stopped"]]
                    cells.append(fmt.format(statistics.median(v)) if v else "too slow" if slow else "-")
                say(f"| {name} | " + " | ".join(cells) + " |")

    serve = []
    for f in sorted((out / "serve-runs").glob("*.json")):
        r = json.loads(f.read_text())
        if r["round"] == 0:                       # the warm-ups
            continue
        t = r["turns"]
        whole = len(t) > 1 and not r["why"]
        rate = lambda n, ms: n / ms * 1e3 if ms > 0 else None   # noqa: E731
        serve.append({
            "gpu": r["gpu"], "arm": r["arm"], "engine": names.get(r["arm"], r["arm"]), "tier": r["tier"],
            "mode": r["mode"], "round": r["round"],
            "prefill_tok_s": rate(t[0]["read"], t[0]["prompt_ms"]) if t else None,
            # the wait before a follow-up's answer: a rate would reward reading more (a missed checkpoint reads in
            # a batch)
            "cont_ms": statistics.median(x["prompt_ms"] for x in t[1:]) if whole else None,
            "decode_tok_s": rate(sum(x["generated"] for x in t), sum(x["decode_ms"] for x in t)) if whole else None,
            "spec_accept": (sum(x["accepted"] for x in t) / max(sum(x["offered"] for x in t), 1)) if whole else None,
            "stopped": (r["why"] or "").removeprefix("stopped: ") if (r["why"] or "").startswith("stopped") else None,
        })
    if serve:
        (out / "matrix-serve.json").write_text(json.dumps(serve, indent=1) + "\n")
        tiers = [t for t in COLLECT_TIERS if any(r["tier"] == t for r in serve)]
        for gpu in sorted({r["gpu"] for r in serve}):
            for mode in sorted({r["mode"] for r in serve}):
                for key, title, fmt in (("prefill_tok_s", "Prefill (tokens/s)", "{:,.0f}"),
                                        ("cont_ms", "Follow-ups (ms)", "{:,.0f}"),
                                        ("decode_tok_s", "Decode (tokens/s)", "{:.1f}")):
                    say(f"\n### {gpu} serve {mode}: {title}\n\n| Engine | " + " | ".join(t.upper() for t in tiers)
                        + " |")
                    say("| --- |" + " ---: |" * len(tiers))
                    for name in names.values():
                        cells = []
                        for t in tiers:
                            c = [r for r in serve
                                 if (r["gpu"], r["engine"], r["tier"], r["mode"]) == (gpu, name, t, mode)]
                            v = [r[key] for r in c if r[key]]
                            cells.append(fmt.format(statistics.median(v)) if v else
                                         "too slow" if any(r["stopped"] for r in c) else "-")
                        say(f"| {name} | " + " | ".join(cells) + " |")

    image = []
    for f in sorted((out / "image-runs").glob("*.json")):
        r = json.loads(f.read_text())
        if r["round"] == 0:                       # the warm-ups
            continue
        ok = r.get("rc") == 0
        image.append({"gpu": r["gpu"], "arm": r["arm"], "engine": names.get(r["arm"], r["arm"]), "round": r["round"],
                      "model": r.get("model") or "qwen-image-turbo", "steps": r.get("steps") or 8,
                      "backend": r.get("backend"), "quant": r.get("quant"),
                      **{k: r.get(k) if ok else None for k in ("seconds", "text_s", "sampling_s", "decode_s")}})
    if image:
        (out / "matrix-image.json").write_text(json.dumps(image, indent=1) + "\n")
        for gpu, model in sorted({(r["gpu"], r["model"]) for r in image}):
            steps = next(r["steps"] for r in image if (r["gpu"], r["model"]) == (gpu, model))
            say(f"\n### {gpu} image generation, {model} (1024x1024, {steps} steps)\n\n| Engine | backend | image (s) | "
                "sampling (s) | VAE (s) |\n| --- | --- | ---: | ---: | ---: |")
            # the named arms first, then any other (bench_comfy.py's arms)
            for name in dict.fromkeys([*names.values(), *sorted(r["engine"] for r in image)]):
                c = [r for r in image if (r["gpu"], r["model"], r["engine"]) == (gpu, model, name)]
                if not c:
                    continue
                cells = []
                for key in ("seconds", "sampling_s", "decode_s"):
                    v = [r[key] for r in c if r[key]]
                    cells.append(f"{statistics.median(v):.1f}" if v else "-")
                say(f"| {name} | {c[0]['backend']} {c[0]['quant']} | " + " | ".join(cells) + " |")

    usage = []
    runs_u = []
    for f in sorted((out / "usage-runs").glob("*.json")):
        if f.name.endswith(".config.json"):
            continue
        r = json.loads(f.read_text())
        if r["round"] == 0 or (r["why"] and not r["why"].startswith("stopped")):   # warm-ups, runs that failed
            continue
        runs_u.append(r)
    # a conversation's turns that every engine reached on that GPU in that round: the agent goes on only while the model
    # calls a tool, and a turn that reads a tool's file is no measure of the turns another engine stopped before
    # (2026-10-24: upstream ended the agent at its first answer, XeStrata went on to read a 3,800-token file)
    reached: dict = {}
    for r in runs_u:
        for conv in dict.fromkeys(t["conv"] for t in r["turns"]):
            rk = (r["gpu"], r["round"], conv)
            have = {x["turn"] for x in r["turns"] if x["conv"] == conv}
            reached[rk] = have if rk not in reached else reached[rk] & have
    for r in runs_u:
        for conv in dict.fromkeys(t["conv"] for t in r["turns"]):
            common = reached[(r["gpu"], r["round"], conv)]
            ct = [x for x in r["turns"] if x["conv"] == conv and x["turn"] in common]
            if not ct:
                continue
            tot = lambda k: sum(x[k] or 0 for x in ct)   # noqa: E731
            usage.append({
                "gpu": r["gpu"], "arm": r["arm"], "engine": names.get(r["arm"], r["arm"]), "conv": conv,
                "round": r["round"], "turns": len(ct), "context": r["context"],
                "ttft_s": statistics.median([x["ttft_s"] for x in ct if x["ttft_s"] is not None] or [0]) or None,
                "prefill_tok_s": tot("prompt_n") / tot("prompt_ms") * 1e3 if tot("prompt_ms") else None,
                "decode_tok_s": tot("predicted_n") / tot("predicted_ms") * 1e3 if tot("predicted_ms") else None,
                "read": tot("prompt_n"), "reused": tot("cache_n"), "generated": tot("predicted_n"),
                "stopped": (r["why"] or "").removeprefix("stopped: ") or None,
            })
            if r["why"]:                          # stopped as too slow: its one answer is no conversation's speed
                usage[-1].update(ttft_s=None, prefill_tok_s=None, decode_tok_s=None)
    if usage:
        (out / "matrix-usage.json").write_text(json.dumps(usage, indent=1) + "\n")
        convs = list(dict.fromkeys(r["conv"] for r in usage))
        for gpu in sorted({r["gpu"] for r in usage}):
            say(f"\n### {gpu} everyday use\n\n| Engine | " + " | ".join(
                f"{c} {m}" for c in convs for m in ("first token (s)", "prompt (tok/s)", "output (tok/s)")) + " |")
            say("| --- |" + " ---: |" * (3 * len(convs)))
            for name in names.values():
                cells = []
                for c in convs:
                    for key, fmt in (("ttft_s", "{:.2f}"), ("prefill_tok_s", "{:,.0f}"), ("decode_tok_s", "{:.1f}")):
                        v = [r[key] for r in usage if (r["gpu"], r["engine"], r["conv"]) == (gpu, name, c) and r[key]]
                        cells.append(fmt.format(statistics.median(v)) if v else "-")
                say(f"| {name} | " + " | ".join(cells) + " |")


def main(argv: list[str]) -> int:
    if argv[:1] == ["corpus"] and len(argv) == 2:
        corpus = make_corpus(ROOT, Path(argv[1]) / "tokenizer")
        CORPUS.write_bytes(lzma.compress(json.dumps(corpus, ensure_ascii=False).encode("utf-8"), preset=9))
        print(f"{CORPUS}: {len(corpus['files'])} files, {len(corpus['agent'])} tool results, "
              f"{CORPUS.stat().st_size} bytes")
        return 0
    if argv[:1] == ["kv-resident"] and len(argv) == 3:
        print(shlex.join(kv_resident(["--native", argv[1]], int(argv[2]))))
        return 0
    print("usage: python tools/bench_core.py corpus PACK_DIR   # (re)write data/bench-corpus.json.xz from git\n"
          "       python tools/bench_core.py kv-resident GGUF CONTEXT   # what kv_resident() gives on this PC",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
