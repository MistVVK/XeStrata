# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8095   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time
behind a FIFO (plan: one resident sequence).
Image generation (optional, when the config has an "images" section): POST /v1/images/generations and
/v1/images/edits (OpenAI's shapes, plus seed) run the image model setup --images installed (Qwen-Image-2.1-Turbo,
Qwen-Image-2.1 or Mage-Flow) on XeStrata's own image engine (strata-image, resident after the first image), one job at
a time on the GPU the text engine uses (serve/images.py; docs/IMAGE.md).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
A request whose prompt alone does not fit the engine's context is REJECTED with 400, never truncated.  When the prompt
fits but prompt plus max tokens does not (Claude Code asks for 32000, the Harness for 16384), max tokens is shortened to
the room left (less CTX_SLACK) and the thinking budget follows it; `--no-fit-max-tokens` / `"fit_max_tokens": false`
brings back the strict 400 of llama.cpp.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import collections
import base64
import hashlib
import hmac
import io
import codecs
import json
import math
import os
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Protocol, cast
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.sh) as well as a module
SETUP_CMD = "xestrata" if (ROOT / "PACKAGED").is_file() else "./setup.sh"   # a deb or rpm package's command (setup.py)
from serve.frontend import (THINK_END, ChatTemplate, Event, GlmOutputParser, HarmonyParser,  # noqa: E402
                            OutputParser, ToolCall, allowed_tools, anthropic_format, anthropic_to_messages,
                            custom_tool_names, forced_call, images_of, legacy_function_call, legacy_functions,
                            literal_tags, thinking_signature, mark_think_literals,
                            openai_to_messages, require_messages, tool_choice_of, unmark_think_literals)
from serve.mcp import McpCancelled, McpHub, hub_from_config  # noqa: E402
from serve import runconfig  # noqa: E402
from serve import places  # noqa: E402
from serve import search as search_api  # noqa: E402
from serve.webfiles import subtree_file as web_subtree_file  # noqa: E402
from serve import logit_bias as bias_api  # noqa: E402
from serve import release_check  # noqa: E402
from serve import loop_guard  # noqa: E402
from serve import debuglog  # noqa: E402
from serve import script_guard  # noqa: E402
from serve.structured import (StructuredOutputError, answer_grammar, checked_calls, constrained_tools,  # noqa: E402
                              extract_json, prepare_format, strict_schemas, tool_grammar, validated_json)
from serve import responses as responses_api  # noqa: E402
from serve.responses import ResponsesError, error_body as responses_error_body  # noqa: E402
from serve import images as images_api  # noqa: E402
from serve import loras as lora_api  # noqa: E402

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


def popen(what: str, args: list, **kw):
    """#735: subprocess.Popen, but Windows' refusal to start a program (WinError 4551, or 1260 when a policy blocks it:
    Smart App Control on a clean Windows 11 blocks an exe that is not signed) is said in words instead of a bare
    OSError."""
    try:
        return subprocess.Popen(args, **kw)
    except OSError as e:
        if getattr(e, "winerror", None) in (4551, 1260):
            raise OSError(f"Windows blocked {what} ({args[0]}): Smart App Control, or an application-control policy, "
                          "does not let a program that is not signed run. Turn Smart App Control off (Windows "
                          "Security > App & browser control > Smart App Control settings), or - for the image encoder "
                          "only - run setup with --vision no (see docs/TROUBLESHOOTING.md)") from e
        raise


def listen_problem(host: str, port: int, e: OSError) -> str:
    """#769: why the server cannot listen.  Only an address that is taken says "already in use"; anything else (a port
    Windows keeps reserved, an address this PC does not have, no permission) says what the OS said."""
    import errno
    code = getattr(e, "winerror", None) or e.errno
    if code in (errno.EADDRINUSE, 10048):
        return (f"port {port} is already in use - is Strata (or another server) already running? "
                f"Close it, or start this one with a different --port")
    what = e.strerror or str(e)
    if code in (10013, errno.EACCES):
        hint = ("Windows may keep this port for itself (see `netsh int ipv4 show excludedportrange protocol=tcp`) "
                "or a firewall rule forbids it: start this one with a different --port")
    elif code in (10049, errno.EADDRNOTAVAIL):
        hint = f"{host} is not an address of this PC: check \"host\" in the model's strata-<model>.json or --host"
    else:
        hint = "check --host and --port"
    return f"cannot listen on {host}:{port}: {what} [{code}]. {hint}"


def combined_embeddings_path(vision_dir: Path, nbytes: int) -> Path:
    """#874: where one request's combined image-embeddings file goes.  Every request with images rewrites it (it is
    deleted after the request), which on a disk is steady heavy writing: on Linux it goes to /dev/shm when that has
    room (the engine reads it back at once), on Windows the file is opened short-lived (the cache is not forced out
    to the drive).  Anywhere else, or when /dev/shm is too small (Docker's default is 64 MB), it stays in the encoder's
    own directory, as before."""
    name = f"req-{uuid.uuid4().hex[:12]}.sve"
    shm = Path("/dev/shm")
    if os.name != "nt" and shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free > 2 * nbytes + (16 << 20):
                return shm / f"strata-vision-{os.getpid()}-{name}"
        except OSError:
            pass
    return Path(vision_dir) / name


def write_temporary(path: Path, parts: list) -> None:
    """#874: write `parts` (files) one after the other into `path`; on Windows with FILE_ATTRIBUTE_TEMPORARY
    (os.O_SHORT_LIVED), so the file normally never reaches the drive before it is deleted."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0) | getattr(os, "O_SHORT_LIVED", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as f:
        for part in parts:
            f.write(Path(part).read_bytes())


# #458: the trailing effort turn ("effort_position": "end"); low is the template's own sentence, medium (which the
# template says nothing for: no xhigh sentence is medium there) says it, since the xhigh one stays at the top
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration.",
               # the template's own xhigh sentence (chat_template.jinja), which #728's recovery swaps for the low one
               "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                        "clarity in the final answer."}
VISION_START = "<|vision_start|>"
# #606: a reply that repeats one token this many times in a row is ended there ("length"): a model in a loop, or a
# broken state that answers one token forever (an issue saw 36,689 tokens of "!"). The config's "repeat_stop_tokens"
# sets it; 0 turns it off.
REPEAT_STOP_TOKENS = 256
# 0.1.45: from this max_tokens on, thinking without a budget is closed early enough to leave room to answer
ANSWER_ROOM_MIN_CAP = 1024
# #984: the tokens kept for the answer when a thinking budget is cut down to fit max_tokens
ANSWER_RESERVE_MIN = 512
# #123: what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
# #1053 (opt-in "reasoning_close_retry": true): a reply that ends on its stop token still inside <think>, with no answer
# and no call, is continued once with the thinking closed (the same way the budget's wrap-up closes it)
REASONING_CLOSE = "\n</think>\n\n"
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# #728: opt-in handling of reasoning that repeats whole passages (which the single-token guard above cannot see).
# "reasoning_loop_recovery": "stop" ends the reply there; "recover" (or true) goes on from the same output with the
# low-effort instruction in place of the xhigh one.  Both off by default.
HIGH_EFFORT = EFFORT_TEXT["xhigh"]
LOW_EFFORT = EFFORT_TEXT["low"]
LOOP_CHECK_EVERY = 512           # output tokens between two looks at the reasoning (at a clean parser boundary)
LOOP_COVERAGE = 0.25             # the share of the last 2,000 words inside 12-word passages seen three times
LOOP_HISTORY_WORDS = 30000       # how far back the passages are counted (bounds the cost of a look)
LOOP_TAIL_CHARS = 2048           # chars at the end of the reasoning also checked for a short period (#1753)
LOOP_TAIL_PERIOD = 64            # a period at most this long in that tail reads as a fully repeated window
# the loop guard (serve/loop_guard.py, from Project Maya): the answer it gets when it ends one before any answer text
# (never a silent end after the thinking)
LOOP_STOP_NOTE = "(The answer was stopped: the model kept repeating itself.)"


def reasoning_repeat_coverage(text):
    """Coverage of recent words by 12-word passages seen at least three times.

    Only reasoning is supplied. The history (the last LOOP_HISTORY_WORDS words) detects repeated verification passes
    separated by long code drafts; the recent window excludes old repetitions.  #1753: a loop can also be one long
    word (a 24k-digit string cycling an 11-digit pattern): the word split makes it a single word and no word count
    grows, and #606's token run never grows either (the tokenizer has no multi-digit tokens), so before the word
    count the last LOOP_TAIL_CHARS chars are checked for a period of at most LOOP_TAIL_PERIOD, and a periodic tail
    reads as full coverage: it is all of the recent output, repeated.
    """
    tail = text[-LOOP_TAIL_CHARS:].lower()
    if len(tail) == LOOP_TAIL_CHARS and any(tail[p:] == tail[:-p] for p in range(1, LOOP_TAIL_PERIOD + 1)):
        return 1.0
    words = re.findall(r"\w+|[^\w\s]", text.lower())[-LOOP_HISTORY_WORDS:]
    width, window = 12, 2000
    if len(words) < window:
        return 0.0
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    counts = collections.Counter(grams)
    start = max(0, len(words) - window)
    covered = set()
    for i in range(start, len(grams)):
        if counts[grams[i]] >= 3:
            covered.update(range(i, i + width))
    return len(covered) / (len(words) - start)


def focused_recovery_prompt(tok, ids, generated):
    """The prompt's ids with the xhigh effort sentence of the first system message replaced by the low one, then
    every token generated so far.  A splice of token ids: the prompt is never decoded and encoded again, so a literal
    `</think>` or vision marker in it (#537, #554) stays what it was.  None when the sentence is not there as whole
    tokens in the first system message (a user's text never counts).  No answer or end-of-thinking is inserted."""
    end = tok.encode(IM_END, parse_special=True)
    if len(end) != 1 or end[0] not in ids:
        return None
    head = ids[:ids.index(end[0])]
    if not tok.decode(head).startswith("<|im_start|>system\n"):
        return None
    old, new = tok.encode(HIGH_EFFORT, parse_special=False), tok.encode(LOW_EFFORT, parse_special=False)
    for i in range(len(head) - len(old) + 1):
        if head[i:i + len(old)] == old:
            return ids[:i] + new + ids[i + len(old):] + list(generated)
    return None


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often (#1619)
KEEPALIVE_S = 10.0          # a held-back tool argument leaves the stream quiet: ping at least this often
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# #481: a running request whose engine prints nothing (no T, PP or any other line) for this long has lost step with the
# server (the engine's main thread waits, untimed, for its next command): the engine is ended and the request fails;
# the next request starts it again.  The config's "engine_silence_s" sets it (0: wait forever, as before).
ENGINE_SILENCE_S = 300.0
# Disk sessions: the longest allowance an engine may announce (`SWAIT <phase> <s>`) for one step that blocks in a
# single call (a file flush, the host->device transfer); the engine's own bound is the same (session_phase_limit_s).
SESSION_WAIT_MAX_S = 3600
# ... except while a prompt is read: a PP line comes once per chunk (up to 32768 tokens with --prefill auto, issue
# #282), and the slowest PCs read ~100 tok/s, so a first chunk can take minutes before the first line.  Until the first
# PP the wait adds the chunk's tokens at PP_FLOOR_TOK_S; after one, a chunk may take PP_SLACK x the last one's time.
PP_CHUNK_MAX = 32768
PP_FLOOR_TOK_S = 50.0
PP_SLACK = 3.0
# Reads that had no timeout (#1317).  A blocking read of the image encoder's pipe or of the engine's READY line that
# never returned held the request turn for good: the engine then sat idle in its command loop, so no watchdog could
# fire and every later request queued behind it (the "lost step" class of #481).  Generous on purpose: a cold model
# load, a slow disk or a slow CPU encode is not a hang.  Only a process that says nothing at all for this long is
# ended; the next request starts it again.
# STRATA_ENGINE_READY_S / STRATA_VISION_READY_S / STRATA_VISION_ENCODE_S override them (0 = wait for ever, as before).


def _timeout_env(name: str, default: float) -> float | None:
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return None if v <= 0 else v


# #1317 part 2: the quick watchdog for a FROZEN engine.  engine_silence_s (above) is generous on purpose: a slow PC
# reads a prompt chunk for minutes without a line.  But an engine that prints nothing for ENGINE_STALL_S and also uses
# no CPU time and moves no bytes of I/O in that time is not slow, it is idle or frozen (a deadlock in a CUDA call, a
# driver stall, a lost step), and waiting longer cannot help: it is ended, and the next request starts it again.  An
# engine that is silent but still working is never ended by this, only reported once (the recommendation:
# engine_silence_s, or this variable). STRATA_ENGINE_STALL_S overrides it (0 = off).  Needs psutil (setup installs it);
# without it nothing is ended.
ENGINE_STALL_S = _timeout_env("STRATA_ENGINE_STALL_S", 90.0)
# --batch: new requests take the control lines (one prompt read at a time) short prompts first: a prompt of at most
# STRATA_ADMIT_SHORT_FIRST tokens goes before longer ones (opt-in; unset or 0: whoever gets the lock first), and a
# longer one that waited STRATA_ADMIT_MAX_WAIT_S (default 30) counts as short.  A chat question then waits for the one
# read in progress, not for every agent read that came before it.
try:
    ADMIT_SHORT_FIRST = max(0, int(os.environ.get("STRATA_ADMIT_SHORT_FIRST") or 0))
except ValueError:
    ADMIT_SHORT_FIRST = 0
ADMIT_MAX_WAIT_S = _timeout_env("STRATA_ADMIT_MAX_WAIT_S", 30.0) or 0.0
STALL_CPU_EPS_S = 0.5       # CPU seconds the engine may use in the whole stall window and still count as frozen
STALL_IO_EPS_B = 1 << 20    # bytes read or written in that window


def engine_frozen(base: tuple[float, int], now: tuple[float, int]) -> bool:
    """#1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them."""
    return (now[0] - base[0]) <= STALL_CPU_EPS_S and (now[1] - base[1]) <= STALL_IO_EPS_B


VISION_READY_S = _timeout_env("STRATA_VISION_READY_S", 600.0)   # a first GPU start compiles its kernels (JIT)
VISION_ENCODE_S = _timeout_env("STRATA_VISION_ENCODE_S", 300.0)
ENGINE_READY_S = _timeout_env("STRATA_ENGINE_READY_S", 900.0)


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event,
                 embeddings: list | None = None) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        scripts = [script] if isinstance(script, str) else script
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in scripts]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet (#344): no context size to plan a request with - a 503,
    not a 400 about the prompt."""


class ClientGone(Exception):
    """#1999: the client hung up while its request was being prepared (an image URL downloading, a picture encoding or
    waiting for its turn): nothing is left to answer."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


REPLY_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")


class Reply:
    """A streamed chat reply kept apart from its connection (strata_reply): its SSE lines as they were made, so that a
    client whose connection broke (a browser that put the tab to sleep, a network that dropped for a moment) reads the
    rest again, and the answer, an image job's included, is not cancelled with the connection.  It is cancelled only
    by its own cancel (POST /v1/strata/replies/<id>/cancel: the web app's Stop)."""

    def __init__(self, cancel: threading.Event):
        self.cancel = cancel
        self.lines: list[bytes] = []
        self.done = False
        self.ended_at: float | None = None
        self.cond = threading.Condition()

    def put(self, line: bytes) -> None:
        with self.cond:
            self.lines.append(line)
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done, self.ended_at = True, time.time()
            self.cond.notify_all()

    def read(self, start: int, wait_s: float) -> tuple[list[bytes], bool]:
        """The lines from `start`, waiting up to wait_s for one, and whether the reply has ended."""
        with self.cond:
            if len(self.lines) <= start and not self.done:
                self.cond.wait(wait_s)
            return self.lines[start:], self.done


class Replies:
    """The replies kept for their clients: a finished one for KEEP_S, at most MAX of them (the oldest finished go
    first; one still running is never dropped)."""
    KEEP_S = 600.0
    MAX = 32

    def __init__(self):
        self.lock = threading.Lock()
        self.items: dict[str, Reply] = {}

    def open(self, rid: str, cancel: threading.Event) -> Reply:
        with self.lock:
            now = time.time()
            for k, r in list(self.items.items()):
                if r.done and r.ended_at is not None and now - r.ended_at > self.KEEP_S:
                    del self.items[k]
            ended = sorted((r.ended_at or 0, k) for k, r in self.items.items() if r.done)
            while len(self.items) >= self.MAX and ended:
                del self.items[ended.pop(0)[1]]
            old = self.items.get(rid)
            if old is not None and not old.done:
                raise ValueError("strata_reply: a reply with this id is still running")
            r = self.items[rid] = Reply(cancel)
            return r

    def get(self, rid: str) -> Reply | None:
        with self.lock:
            return self.items.get(rid)


REPLIES = Replies()


def reply_id(req: dict) -> str | None:
    """A streamed request's strata_reply id (the web app's: {"id": ...}), or None."""
    r = req.get("strata_reply")
    if r is None:
        return None
    rid = r.get("id") if isinstance(r, dict) else None
    if not isinstance(rid, str) or not REPLY_ID.fullmatch(rid):
        raise ValueError("strata_reply: {\"id\": 8 to 64 letters, digits, - or _}")
    if not req.get("stream"):
        raise ValueError("strata_reply needs stream: true")
    return rid


class EngineSilent(EngineDied):
    """#481: the engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


class EngineNonFinite(ValueError):
    """#879: the engine's DONE (or a slot's BDONE) said `nonfinite`: a window's logits held a NaN or an infinity.  The
    engine wrote none of that window's tokens and drops every cached state that may carry the poison, so its next
    request reads the prompt from token 0.  Service.run sends a request that had produced nothing yet once more, and
    ends one that had started with this error (as vLLM and llama.cpp end a reply their engine broke off).  A
    ValueError, so a stream that has started reports it as an error event."""


class SessionRefused(ValueError):
    """Disk sessions: the engine refused or failed a SAVE / RESTORE and is still in step (`SERR <kind> <published>
    <reason>`).  `kind` is the engine's category - invalid (400), storage (507), memory (503), io (500); `published`:
    a save whose new file already replaced the old one before a later step (the folder flush) failed."""

    STATUS = {"invalid": 400, "storage": 507, "memory": 503, "io": 500}

    def __init__(self, kind: str, message: str, published: bool = False):
        super().__init__(message)
        self.kind = kind if kind in self.STATUS else "io"
        self.published = published

    @property
    def status(self) -> int:
        return self.STATUS[self.kind]


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill: the server keeps it (and says so) rather than
    reporting its GPU and RAM as given back."""


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left.  `retry_after`: seconds
    for the Retry-After header of the 503."""

    def __init__(self, message="", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# The page's only CSP directive: its preview frames (srcdoc) may not load any address. Other directives are left out
# on purpose: a frame made from srcdoc inherits the page's policy, and the opt-in "run scripts" preview needs its own
# inline script.
PAGE_CSP = "frame-src 'none'"


def unload_wait(value, default: float) -> float:
    """#1997: the `wait` of POST /unload?wait=...: absent / 0 / false: 0 (answer "busy" at once, as before); 1 / true /
    yes: `default` seconds; a number of seconds above 1: that long (at most an hour)."""
    if value is None:
        return 0.0
    v = str(value).strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return 0.0
    if v in ("1", "true", "yes", "on"):
        return default
    try:
        return min(3600.0, max(0.0, float(v)))
    except ValueError:
        return default


def rttime_limit_note() -> str:
    """upstream #3238: Linux only. A hard RLIMIT_RTTIME that is not unlimited (PipeWire's client module sets it to 0
    when rtkit is missing; every program started from that desktop inherits it) makes the kernel SIGKILL a process as
    soon as it checks a thread with an armed CPU timer, with no OOM kill and no message. Returns one sentence naming the
    cause and the fix, or "" when the limit is unlimited, the platform has no such limit or it cannot be read."""
    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_RTTIME)
    except (ImportError, AttributeError, OSError, ValueError):
        return ""
    if hard == resource.RLIM_INFINITY:
        return ""
    return (f"This session has a real-time CPU limit (RLIMIT_RTTIME hard limit {hard}; the desktop's audio system sets "
            "it to 0 when rtkit is not installed). Linux can then end a program with signal 9 for no visible reason. "
            "Install rtkit, or start XeStrata outside the desktop session's limits, for example: "
            "systemd-run --user --collect -p LimitRTTIME=infinity -- xestrata")


ENGINE_REQUEST = re.compile(
    # "+ 12288 of 98179 read": a request cancelled while its prompt was read (#471)
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")


DRAFT_HEAD_FAIL = "the draft head does not fit"
DRAFT_HEAD_HINT = ("a smaller draft vocabulary needs less VRAM: --draft-vocab cyrillic (English, code and the Cyrillic "
                   "script) or --draft-vocab en (English and code, ~215 MiB less than the default). Run "
                   f"{SETUP_CMD} --draft-vocab en once and the model keeps it; or a smaller --context in setup.")


def exit_text(rc: int | None) -> str:
    """The engine's exit status in words (issue #12): "exit code N", or the signal that ended it ("killed by SIGSEGV")
    - a crash and the OS ending it for its RAM (SIGKILL) read alike as a negative code."""
    if rc is None:
        return "no exit status yet"
    if rc < 0:
        try:
            return f"killed by {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}"


def start_failure_hint(log: str | None, offset: int) -> str:
    """#474 (upstream f2aaba1): what to change when the engine stopped at the start because the MTP draft head did not
    fit the VRAM left: the engine's own `strata mtp:` lines after that failure (what it needs, what is free, the
    smaller subsets), else the same advice in words.  "" for any other failure: the log says why."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    if DRAFT_HEAD_FAIL not in text:
        return ""
    said = [x.strip()[len("strata mtp: "):] for x in text.splitlines()
            if x.strip().startswith("strata mtp: ") and ("draft head over" in x or "hint:" in x)]
    return ". mtp: " + DRAFT_HEAD_FAIL + ". " + (" ".join(said) if said else "Hint: " + DRAFT_HEAD_HINT)


def start_log_tail(log: str | None, offset: int, n: int = 20) -> str:
    """#496: the last n lines this start wrote to the engine log, for the error when the engine ended before READY -
    whatever the failure, the engine's own reason is in them (people posted the traceback without the log).  "" when
    there is no log or nothing in it from this start."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # enough for 20 lines, never an earlier start's
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    lines = [x.rstrip() for x in lines if x.strip()][-n:]
    if not lines:
        return ""
    return "\nthe engine log's last lines:\n" + "\n".join("  " + x for x in lines)


_echoing: set[str] = set()      # the logs echo_requests already follows (restart() runs StrataEngine.__init__ again)
# Windows NTSTATUS exit codes of a process the system ended (a crash, not an error the engine reported)
_WIN_CRASH = {0xC000001D: "illegal instruction (the CPU lacks an instruction the program uses)",
              0xC0000005: "access violation", 0xC0000409: "stack buffer overrun / fast-fail",
              0xC0000135: "a DLL the program needs was not found", 0xC0000139: "entry point not found in a DLL",
              0xC0000142: "a DLL failed to initialize", 0xC000007B: "bad image (32-bit / 64-bit mix-up)",
              0xC00000FD: "stack overflow", 0xC0000094: "integer division by zero",
              0xC0000096: "privileged instruction"}


def engine_exit_words(rc: int | None) -> str:
    """What a process's exit status says, in words: "killed by signal 4 (SIGILL, Illegal instruction)" (a negative
    status on Linux / macOS), a Windows NTSTATUS crash code, or "exit code N".  "" when there is no status."""
    if rc is None:
        return ""
    if rc < 0 and os.name != "nt":
        try:
            sig = signal.Signals(-rc)
            return f"killed by signal {-rc} ({sig.name}, {signal.strsignal(sig) or sig.name})"
        except (ValueError, AttributeError):
            return f"killed by signal {-rc}"
    code = rc & 0xFFFFFFFF
    if code in _WIN_CRASH:
        return f"exit code 0x{code:08X}: {_WIN_CRASH[code]}"
    if 0xC0000000 <= code < 0xD0000000:
        return f"exit code 0x{code:08X} (a system error status)"
    return f"exit code {rc}"


def engine_crashed(rc: int | None) -> bool:
    """True when the engine process was ended by the system (an illegal instruction, a bad memory access, the
    out-of-memory killer, a missing DLL ...) rather than leaving by itself: nothing a second try fixes."""
    if rc is None:
        return False
    if rc < 0 and os.name != "nt":
        return -rc in {int(getattr(signal, n)) for n in ("SIGILL", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGKILL")
                       if hasattr(signal, n)}
    return 0xC0000000 <= (rc & 0xFFFFFFFF) < 0xD0000000      # an NTSTATUS error code


def cpu_isa_words() -> tuple:
    """(text, missing): the CPU's model and which of AVX / AVX2 / FMA / F16C it reports (Linux: /proc/cpuinfo), and the
    ones of those it lacks.  ("", []) where it cannot be read."""
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return "", []
    model = next((x.split(":", 1)[1].strip() for x in text.splitlines() if x.lower().startswith("model name")), "")
    flags = next((x.split(":", 1)[1].split() for x in text.splitlines() if x.lower().startswith("flags")), [])
    have = ", ".join(f"{f} {'yes' if f in flags else 'NO'}" for f in ("avx", "avx2", "fma", "f16c", "avx512f"))
    missing = [f for f in ("avx", "avx2", "fma", "f16c") if f not in flags]
    return f"{model or 'this CPU'} ({have})", missing


def start_exit_note(rc: int | None) -> str:
    """The sentence the start error carries about how the engine process ended (the engine log is silent when the
    system ends it): "" without a status, else a line, with the CPU's AVX flags after an illegal instruction."""
    words = engine_exit_words(rc)
    if not words:
        return ""
    note = f"\nthe engine process ended: {words}"
    if rc is not None and ((rc < 0 and os.name != "nt" and -rc == int(getattr(signal, "SIGILL", 4))) or
                           (rc & 0xFFFFFFFF) == 0xC000001D):
        isa, missing = cpu_isa_words()
        note += "\n  An illegal instruction: the engine ran an instruction this CPU cannot execute"
        if isa and missing:
            note += f" ({', '.join(m.upper() for m in missing)} missing; {isa})"
        elif isa:
            note += f"; the CPU reports AVX, AVX2, FMA and F16C ({isa}), so this is a bug in the engine"
        note += ".  Please report it with the CPU model (github.com/MistVVK/XeStrata/issues)."
    return note


def start_log_marks(log: str | None, offset: int) -> list:
    """The GPU runtime failure texts (TRANSIENT_START_MARKS) this start wrote to the log: what lets a retry notice say
    the GPU runtime failed.  [] when none, or no log."""
    if not log:
        return []
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read(64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return []
    return [m for m in TRANSIENT_START_MARKS if m in text]


# A start that ends at once with the GPU runtime's own failure and none of the engine's lines: on ROCm, a start right
# after the previous engine exited can fail inside the runtime's device/queue set-up while the old process's queues are
# still being torn down ("terminate called after throwing an instance of 'rocr::AMD::hsa_exception'", or an exit with
# nothing in the log).  Such a start is tried again after a short wait (STRATA_START_RETRIES, default 3; 0: never).
TRANSIENT_START_MARKS = ("hsa_exception", "HSA_STATUS_ERROR", "hsaKmt")
START_RETRIES = max(0, int(os.environ.get("STRATA_START_RETRIES", "3") or 0))


def transient_start_failure(log: str | None, offset: int, elapsed_s: float, rc: int | None = None) -> bool:
    """True when the start that wrote the log from `offset` failed the transient way above: within 30 s, and its lines
    are only the launcher's own "engine started" line plus (optionally) a GPU runtime failure.  An exit with no line
    at all counts only when it was not the engine's own error exit (1: it said why on stdout)."""
    if not log or elapsed_s > 30.0:
        return False
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read(64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return False
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    engine_lines = [x for x in lines if not x.startswith("[strata]")]
    if any(m in text for m in TRANSIENT_START_MARKS):
        return all(x.startswith("terminate called") or x.startswith("what()") or
                   any(m in x for m in TRANSIENT_START_MARKS) for x in engine_lines)
    # it exited without writing a single line of its own - but not when the system ended it (an illegal instruction, a
    # bad memory access, the OOM killer, a missing DLL): that is a crash a second try repeats, and the retry notice
    # must not call it a GPU runtime start (#2324: an old CPU's illegal instruction was retried 3 times as "GPU
    # runtime" and the real reason never shown)
    return not engine_lines and rc != 1 and not engine_crashed(rc)


def echo_requests(log_path: str, offset: int) -> None:
    """STRATA_REQUEST_LINES=1: one stdout line per finished request, from the engine's own summary in its log.

    The engine's stderr goes to the log file (the start narrator reads it), so a supervisor that only sees this
    process's output - a tray, llama-swap - has no per-request numbers. This re-states the engine's line with the
    total the two times make: `request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y
    tok/s` (prompt_read: the time the engine spent reading the prompt's new tokens, not a time to first token).
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = ENGINE_REQUEST.search(line)
            if m:
                read_ms, gen_ms = float(m["read"]), float(m["gen_ms"])
                print("[strata] request prompt %s cached %s output %s prompt_read %.0f ms total %.0f ms prefill %s "
                      "tok/s decode %s tok/s" % (m["prompt"], m["reused"], m["gen"], read_ms, read_ms + gen_ms, m["pp"],
                                                 m["tg"]), flush=True)


def experts_loading_words(args: list, size: str) -> str:
    """#505: what the start does with the experts, by the engine's flags (generate.cpp's option parsing): a RAM budget
    copies the hottest N GiB into RAM (--resident-budget-gib), the resident low-RAM mode the ones the GPU does not hold
    (--resident-experts), plain --mmap-experts reads them from the model files through the OS file cache (nothing is
    loaded into RAM up front); otherwise all of them go into RAM."""
    if "--resident-budget-gib" in args:
        try:
            n = f"up to {float(args[args.index('--resident-budget-gib') + 1]):g} GiB"
        except (IndexError, ValueError):
            n = "a RAM budget"
        return (f"loading the most-used experts into RAM ({n}; the rest are read from the model files as needed) "
                "and locking part of them for the GPU.")
    if "--resident-experts" in args:
        return (f"loading the experts the GPU does not hold into RAM (of {size}) and locking part of them for the "
                "GPU.")
    if "--mmap-experts" in args:
        return (f"mapping the experts from the model files ({size}, --mmap-experts): they are not loaded into RAM - "
                "the OS file cache reads them as the GPU's expert cache fills and as requests need them.")
    return f"loading the experts into RAM ({size}) and locking part of them for the GPU."


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    loading = experts_loading_words(args, size)
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[strata] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line or "experts via mmap" in line:   # #505: mapped
                    say("arena", f"[strata] {loading}\n"
                        "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                        "         Please wait and don't close this window; the browser opens when it is ready.")
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                elif "IGC's GEP simplification is off" in line:   # Intel's compiler 2.27 to 2.30 (AGENTS.md)
                    v = line.split("; IGC ", 1)[1].split(")", 1)[0] if "; IGC " in line else "?"
                    say("igc", f"[strata] Intel's compiler IGC {v} has a bug XeStrata works around: the decode of "
                        "short prompts is up to 10% slower.\n"
                        "         For full speed, use Fedora 44 (its IGC is 2.32 or later), or install with Podman "
                        "(Fedora 44 inside).")
                elif "session is up" in line:
                    say("up", "[strata] almost ready ...")
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


_BTRACE = bool(os.environ.get("STRATA_BATCH_TRACE"))


def btrace(*a):
    if _BTRACE:
        print("[batch-trace]", threading.get_ident() % 10000, *a, file=sys.stderr, flush=True)


EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|>: the engine's default --eos-ids (Qwen3.8-Flash-Next's)


def stop_token_ids(tok) -> set[int]:
    """The tokens that end a reply: the model's end-of-sequence, end-of-turn and end-of-message tokens as its GGUF
    names them (the engine stops at the same ones), and `<|im_end|>` / `<|endoftext|>` where the vocabulary has them
    as single tokens; the Harmony format's `<|return|>` and `<|call|>` too (llm-jp-4.1: llama.cpp ends the generation
    at them, not at `<|end|>`, which closes the reasoning).  Qwen3.8-Flash-Next's EOS_IDS when the tokenizer names
    none."""
    special = getattr(tok, "special_ids", None) or {}
    ids = {int(v) for k, v in special.items()
           if k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id")}
    for text in (IM_END, "<|endoftext|>", "<|return|>", "<|call|>"):
        try:
            one = tok.encode(text, parse_special=True)
        except (KeyError, ValueError):
            continue
        if len(one) == 1:
            ids.add(one[0])
    return ids or set(EOS_IDS)


# The engine's own words for a verifier that cannot go on (its GPU waits were released, or an earlier window never
# finished, #267): the process may stay alive and even print DONE, but it must not take another request.
FATAL_PREFIXES = ("ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                  "ERR verify: an earlier window never finished", "ERR verify batch: an earlier window never finished")


class ConvCacheLog:
    """upstream #596: the engine's conversation cache as its log tells it (the engine writes "strata serve: conversation
    cache: parked N tokens ...; parked=P bytes=B evictions=E" and "restored N tokens ...; parked=P bytes=B" to stderr,
    which is the log): read on from where it was last read, from the start of the engine's current run."""
    EVENT = re.compile(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)"
                       r"(?: evictions=(\d+))?")
    DROPPED = re.compile(r"conversation cache: dropped \d+ superseded .*?parked=(\d+)")
    READ_MAX = 1 << 20                                  # at most the last MiB of new lines per read

    def __init__(self):
        self.key, self.pos = None, 0
        self.reset()

    def reset(self):
        self.state = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None,
                      "last_tokens": None, "last_at": None}

    def poll(self, path, start) -> dict:
        """The state after the log's new lines; `start` is where the engine's current run began in it."""
        if not path or start is None:
            return dict(self.state)
        if self.key != (path, start):                   # another start of the engine: its cache starts empty
            self.key, self.pos = (path, start), start
            self.reset()
        try:
            size = os.path.getsize(path)
            if size < self.pos:                         # the log was emptied or replaced
                self.pos = 0
            if size - self.pos > self.READ_MAX:
                self.pos = size - self.READ_MAX
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return dict(self.state)
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        now = time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if "conversation cache:" not in line:
                continue
            m = self.EVENT.search(line)
            if m:
                st = self.state
                st["parked"], st["bytes"] = int(m.group(3)), int(m.group(4))
                if m.group(5) is not None:
                    st["evictions"] = int(m.group(5))
                if m.group(1) != "skipped":
                    st["parks" if m.group(1) == "parked" else "restores"] += 1
                    st["last_event"], st["last_tokens"], st["last_at"] = m.group(1), int(m.group(2)), now
                continue
            m = self.DROPPED.search(line)
            if m:
                self.state["parked"] = int(m.group(1))
        return dict(self.state)


def conversation_slots(engine) -> int | None:
    """How many conversations the engine's cache holds: its --conversation-cache-slots argument, which is known
    before the engine has started (a lazy start has reported no INFO line yet), else the INFO line it reported."""
    args = engine.spawn[1] if hasattr(engine, "spawn") else []
    if "--conversation-cache-slots" in args[:-1]:
        try:
            return int(args[args.index("--conversation-cache-slots") + 1])
        except ValueError:
            pass
    slots = (getattr(engine, "info", None) or {}).get("conversation_cache_slots")
    return slots if isinstance(slots, int) else None


def conversation_cache_view(info: dict, hist: list, totals: dict, parked: dict) -> dict:
    """#596: the Monitor's Conversation cache card: the parked conversations (the engine's opt-in
    --conversation-cache-mib: budget, slots, what its log says) and how much of the prompts the cache gave back."""
    mib = info.get("conversation_cache_mib")
    last = hist[-1] if hist else None
    return {"enabled": isinstance(mib, int) and mib > 0, "budget_mib": mib if isinstance(mib, int) else None,
            "slots": info.get("conversation_cache_slots"), **parked,
            "requests": len(hist), "requests_reused": sum(1 for r in hist if (r.get("reused") or 0) > 0),
            "reused_tokens": totals.get("reused", 0), "prompt_tokens": totals.get("prompt_tokens", 0),
            "last_reused": last.get("reused") if last else None,
            "last_prompt": last.get("prompt_tokens") if last else None}


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """
    silence_s = ENGINE_SILENCE_S         # #481: main() sets the config's engine_silence_s (survives restart())
    silent_note: str | None = None       # #481: why the server ended a silent engine (death_note says it)
    # #1317: () -> bool, "the GPU is working" (the telemetry's reading); a GPU at work is not frozen
    gpu_busy = None
    last_err: str | None = None          # #997: an ERR that was its last stdout line (death_note says it)
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)
    kept_arena: "KeptArena | None" = None   # main() sets it: the experts' arena kept across an image job's unload
    eos_ids = EOS_IDS                    # the tokens that end a reply (Service sets the model's: stop_token_ids)

    # `last`: the figures of the engine's last DONE line.  With batch slots several requests run at once, each on its
    # own thread, so each one reads its own request's figures (Service.run records them); one at a time, the one dict.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False, cpus: list[int] | None = None, _attempt: int = 0,
                 *, fds: tuple[int, ...] = ()):
        self.spawn = (exe, list(args), cwd, log, env, False, cpus)   # to start it again after it died (issue #27)
        self.spawn_kw = {"fds": tuple(fds)} if fds else {}           # descriptors the engine inherits (split's arena)
        self.session_save_reclaim = '--session-save-reclaim' in args
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        self.ended, self.unloaded = True, True
        self.max_context = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 4096
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        self.progress_ms = 0             # PP's own milliseconds since the prompt started (its third field)
        self.reused = 0                  # RESUME: prompt tokens not read again (a client takes them out of the work)
        self.silent_note = None
        self.last_err = None
        try:                            # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        if lazy:
            return
        self.unloaded = False            # `ended` stays True until READY (below): not alive while starting (#344)
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0  # where this start's lines begin (start_failure_hint)
        self.log_start = log_start                      # #596: the Monitor's conversation cache reads from here
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
            # once per log: the follower keeps reading the same (appended) log across restarts and reloads
            if os.environ.get("STRATA_REQUEST_LINES") and os.path.abspath(log) not in _echoing:
                _echoing.add(os.path.abspath(log))
                threading.Thread(target=echo_requests, args=(log, os.path.getsize(log)), daemon=True).start()
        if log:                                          # #779: when an engine started (a crash report has the time)
            try:
                assert not isinstance(self.log, int)
                self.log.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {exe} --serve "
                               f"{' '.join(args)}\n")
                self.log.flush()
            except OSError:
                pass
        self.max_context = 0                             # before the new process is visible: never its predecessor's
        pin: dict[str, Any] = {}
        if cpus and hasattr(os, "sched_setaffinity"):    # replicas: this engine's own physical cores (replicas.py)
            pin["preexec_fn"] = lambda: os.sched_setaffinity(0, cpus)
        if fds:
            pin["pass_fds"] = tuple(fds)
        spawned_at = time.monotonic()
        try:
            self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env,
                              **pin)
        except OSError:                                  # nothing started: the narrator stops, the log handle goes back
            loading.set()
            if log and not isinstance(self.log, int):
                self.log.close()
            raise
        # the READY read has a timeout (#1317): the lines come from a thread, up to READY, so the rest of the stream
        # stays for _pump (one reader at a time)
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q), daemon=True).start()
        ready_deadline = None if ENGINE_READY_S is None else time.monotonic() + ENGINE_READY_S
        timed_out = False
        # past ENGINE_READY_S an engine still at work (CPU or disk in the last window, as _stall_check reads it) is
        # waited for, a window at a time; a frozen one, or one that cannot be read, is ended as before.  A big RAM copy
        # from a cold disk took longer than the limit and was killed with SIGKILL while it loaded, with nothing in its
        # log (R9700 x2, IQ3_S, 30.8 GiB page-locked, the GPU caches filled at 88 MB/s).
        work_base = None
        work_window = min(x for x in (ENGINE_STALL_S, ENGINE_READY_S) if x is not None) if ENGINE_STALL_S else None
        start_err = None                                # an ERR line before READY: the engine said it cannot start
        while True:
            left = None if ready_deadline is None else ready_deadline - time.monotonic()
            if left is not None and left <= 0:
                now = self._activity() if work_window else None
                if work_window and now is not None and (work_base is None or not engine_frozen(work_base, now)):
                    if work_base is None:
                        print(f"[strata] the engine has not reported READY within {ENGINE_READY_S:.0f} s but is still "
                              "loading (CPU or disk at work): waiting while it works (STRATA_ENGINE_READY_S, "
                              "STRATA_ENGINE_STALL_S)", flush=True)
                    work_base, ready_deadline = now, time.monotonic() + work_window
                    continue
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                break
            if line.startswith("ERR"):
                # not waited for to exit: a process that failed to load and then hung in its clean-up (two engines on
                # one pair of GPUs, 2026-11-01) kept a CPU busy, which the READY wait above reads as still loading
                start_err = line[3:].strip()
                break
        loading.set()
        if self.max_context <= 0:
            if timed_out or start_err is not None:      # still loading, hung or failed: end it, so the turn is not held
                try:
                    self.proc.kill()
                except OSError:
                    pass
            try:                                        # its pipes and our handle on its log (the log stays)
                self.proc.wait(timeout=5)
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
                if self.proc.stdout is not None:
                    self.proc.stdout.close()
                if log and not isinstance(self.log, int):
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            if not timed_out and start_err is None and _attempt < START_RETRIES and \
                    transient_start_failure(log, log_start, time.monotonic() - spawned_at, self.proc.returncode):
                wait_s = 2.0 * (2 ** _attempt)              # 2, 4, 8 s: the previous process's GPU queues go away
                exit_words = engine_exit_words(self.proc.returncode)
                if start_log_marks(log, log_start):         # the runtime said so itself (hsa_exception ...)
                    print(f"[strata] the engine stopped while the GPU runtime was starting (most often right after "
                          f"another engine exited, whose GPU queues were still being released"
                          f"{'; ' + exit_words if exit_words else ''}): starting it again in {wait_s:.0f} s "
                          f"(try {_attempt + 2} of {START_RETRIES + 1})", flush=True)
                else:                                       # an exit with nothing in the log: say only what is known
                    print(f"[strata] the engine exited before it wrote anything to its log"
                          f"{' (' + exit_words + ')' if exit_words else ''}; it may be a GPU runtime that was not "
                          f"ready yet: starting it again in {wait_s:.0f} s (try {_attempt + 2} of {START_RETRIES + 1})",
                          flush=True)
                time.sleep(wait_s)
                StrataEngine.__init__(self, exe, args, cwd, log, env, lazy, cpus, _attempt + 1, fds=fds)
                return
            why = (f"the engine did not report READY within {ENGINE_READY_S:.0f} s" if timed_out else
                   f"the engine could not start: {start_err}" if start_err is not None else
                   f"the engine exited before it was ready ({exit_text(self.proc.poll())})")
            # EngineDied, not bare RuntimeError: a request waiting on this load must end with a 503
            # through the request path's handler - a RuntimeError there left the turn with no answer (#1527)
            note = "" if timed_out or start_err is not None else start_exit_note(self.proc.returncode)
            tail = start_log_tail(log, log_start)
            lines = [x for x in tail.split("\n")[2:] if x.strip()]
            if tail and lines and all(x.strip().startswith("[strata]") for x in lines):
                tail += "\n  (those are the launcher's own lines: the engine itself wrote nothing)"
            raise EngineDied(why + (f" (see {log})" if log else "") + note +
                             start_failure_hint(log, log_start) + tail)
        self.known_ctx = self.max_context   # survives a failed restart: requests keep their limit and restart it
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        self.ended = False                              # READY: alive from here (restart() set it True, #344)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        # --batch: concurrent requests in the engine's batch slots (see generate_batched).  The engine says how many
        # it runs (INFO batch_slots=N: it may have fewer than asked, or none, when they do not fit)
        asked = next((int(args[args.index(k) + 1]) for k in ("--batch", "--slots") if k in args[:-1]), 0)
        self.batch = int(self.info.get("batch_slots") or 0)
        if asked and self.batch != asked:
            print(f"[strata] parallel requests: {asked} asked, the engine runs {self.batch or 'one at a time'} "
                  "(its log says why)", flush=True)
        # the engine resolves the groups (a number, auto, or auto by default on a layer split) and says what it runs:
        # INFO batch_groups=G (upstream 1bf6344c, ca001df5); an engine without it runs what --batch-groups asked
        bg = args[args.index("--batch-groups") + 1] if "--batch-groups" in args[:-1] else "1"
        groups = int(self.info["batch_groups"]) if "batch_groups" in self.info else int(bg) if bg.isdigit() else 1
        groups = groups if self.batch and groups > 0 and self.batch % groups == 0 else 1
        gs = self.batch // groups if self.batch else 0
        # slots in the order that spreads requests over the pipeline's groups first: 0, gs, 2gs, .., 1, gs+1, ..
        self.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
        self.slot_group = [b // gs if gs else 0 for b in range(self.batch)]   # each slot's pipeline group
        self.slot_groups = groups
        self.slot_gs = gs if groups > 1 else 0   # pipelined groups: their size (pick_slot fills a group's holes)
        self.slot_q: list[queue.Queue] = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # #1603: who owns a busy flag.  slot_claims: slots a request of this server picked or reserved and has not
        # given back; slot_releasing: slots handed to the background BDONE wait.  A busy slot in neither has no owner
        # (its flag leaked) and would keep every later request waiting for a slot that nothing will free.
        self.slot_claims: set[int] = set()
        self.slot_releasing: set[int] = set()
        self._orphan_seen: dict[int, float] = {}
        # what each slot's session holds (prompt + every token a window fed), so a conversation's next turn goes to
        # the slot that has its start (the engine checks it again); when the slot was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        # the prompt_cache_key of the request each slot last took (OpenAI's field: an agent harness such as Mainplane
        # sends one per session), so a session whose prompt no slot starts overwrites its own slot first (pick_slot)
        self.slot_key: list[str | None] = [None] * self.batch
        self.slot_live: list[dict | None] = [None] * self.batch   # /metrics: the request in each slot
        # #1012: the admission state belongs to the server, not to one engine process.  restart() runs this again
        # while requests still wait on it (they hold these very objects), so it is made once and kept: new ones
        # would leave the waiters on a lock and a condition nobody notifies, with their counts missing from the
        # new lists.
        if "slot_cv" not in self.__dict__:
            self.slot_cv = threading.Condition()
            self.waiting = 0                            # requests waiting for the control lines (ctl)
            # #1603: ... and those that gave the lines back to wait for a slot
            self.admitting = 0
            self.wait_lens: list[list] = []        # ... their prompt lengths (a long read gives way to short ones)
            self.ctl_epoch = 0                          # how often the control lines were taken
            self.ctl = threading.Lock()                 # one admission or solo request on the control lines at a time
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is (a request notes its own)
        self._ctl_erred = False                         # #1059: the control lines ended on an ERR
        self._yielded = None                            # (slot, tokens read): the last request on them gave way
        self._ctl_mode, self._ctl_result, self._last_done = "solo", None, None
        self.wlock = threading.Lock()                   # stdin writes from several request threads
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        proc, lines = self.proc, self.lines             # this process's: a restart replaces both (#344)
        self.__dict__.setdefault("lp_lines", LPLines())   # (token id, LP line): logprobs
        last_t = None
        # a request asking for logprobs: its T line is held until the LP line after it is read, so that the token is
        # never spelled before its logprob is there (it went to the next token's text, and the last token had none)
        held = None
        slot_q: list[queue.Queue] = getattr(self, "slot_q", [])
        line = None
        for line in proc.stdout:
            if held is not None and not line.startswith("LP "):   # no LP line for it after all
                lines.put(held)
                held = None
            # checked before batch routing: a fatal line is never a slot's own
            if line.startswith(FATAL_PREFIXES):
                # release_gpu_waits invalidates the verifier, even if the native
                # process stays alive and prints DONE afterwards.
                if self.proc is proc:
                    self.silent_note = "Unrecoverable native verification failure: " + line[4:].strip()
                    self.ended = True
                try:
                    proc.terminate()
                except OSError:
                    pass
                break                           # publish EOF, never the trailing DONE
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # --batch: a batch slot's own lines
                try:
                    slot_q[int(line.split()[1])].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            if line.startswith("LP "):                  # beside the token before it, not in the queue: in the lines of
                self.lp_lines.append((last_t, line))    # the request that has the control lines now (LPLines)
                if held is not None:
                    lines.put(held)
                    held = None
                continue
            if line.startswith("T "):
                try:
                    last_t = int(line[2:])
                except ValueError:
                    last_t = None
                if getattr(self.lp_lines, "owner", None) is not None:
                    held = line
                    continue
            lines.put(line)
        if held is not None:
            lines.put(held)
        if self.proc is proc:                           # a killed engine's pump must not mark its successor dead
            self.ended = True                           # its output closed: it is gone, even before the OS says so
            if line and line.startswith("ERR"):         # #997 #890: why it exited, though no request may read it
                self.last_err = line[4:].strip()        # (a failed batch window: every request reads its slot)
        lines.put(None)
        for q in slot_q:
            q.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        if self.silent_note:                            # #481: the server ended it, not the OS or the engine itself
            return self.silent_note
        tail = wide = ""
        try:
            if self.log_path is None:
                raise OSError("no log")
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                wide = f.read().decode("utf-8", "replace")
            tail = wide[-4096:]
        except (OSError, TypeError):
            pass
        # #1705: STRATA_VERIFY_TRACE dumps are ~30 KB and push the watchdog's line out of a 4 KB tail
        for line in reversed(wide.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught (issue "
                        "#29 there is upstream Strata's). Its log line: " + line.strip() + " - please report it at "
                        "github.com/MistVVK/XeStrata/issues.")
        proc = self.proc                                 # None while a restart has not started the next one yet
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc >= 0 and self.last_err:  # its own last words on stdout: they say why
            return (f"The engine exited (code {rc}) after it reported: {self.last_err} - please report it at "
                    "github.com/MistVVK/XeStrata/issues with the log.")
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("strata", "ERR"))), "")
        if rc is not None and rc < 0 and -rc != signal.SIGKILL:   # issue #12: a crash, not the OS ending it for RAM
            return (f"The engine crashed ({exit_text(rc)}). " + (f"It reported: {self.last_err}" if self.last_err
                    else f"Its last log line: {last}" if last else "It wrote nothing about it") +
                    " - please report it with the log.")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please "
                    "report it at github.com/MistVVK/XeStrata/issues with the log.")
        rt_note = rttime_limit_note() if rc == -9 else ""
        if rt_note:                                       # upstream #3238: the kernel killed it, but not for RAM
            return ("The engine was ended by signal 9. " + rt_note + " If dmesg (sudo dmesg | grep -i -E "
                    "'killed process|out of memory') shows an out-of-memory kill instead, close other programs or use "
                    "a smaller model (Q2_0 / IQ2_XS).")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'). Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart() / close() may set it to None meanwhile
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def unload(self, keep_arena: bool = False):
        """Release GPU/RAM between requests, retaining the spawn config for automatic reloading.  `keep_arena`: an image
        job's unload, after which the experts stay in the server's arena (kept_arena) but for those arena_drop gave
        up; any other unload gives the whole arena back."""
        self.close()
        self.unloaded = True
        kept = getattr(self, "kept_arena", None)
        if kept is not None:
            kept.after_unload(keep_arena)

    def arena_drop(self, mib: int, path: str, timeout: float = 60.0) -> dict:
        """`ARENA DROP <mib> <file>` before an image job's unload (the caller holds the service's FIFO): the engine
        marks at least `mib` of its coldest experts dropped from the shared arena and writes their ranges in the file
        to `path` -> {"dropped_mib", "ranges"}; ValueError with the engine's reason, EngineDied when it ended."""
        proc = self.proc
        if not self.alive() or proc is None or proc.stdin is None:
            raise EngineDied("the engine is not running")
        if any(c in path for c in "\r\n\0"):
            raise ValueError("invalid arena drop path")
        proc.stdin.write(f"ARENA DROP {int(mib)} {path}\n")
        proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the ARENA command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the ARENA command")
            if line.startswith("ARENA "):
                out = {k: int(v) for k, _, v in (kv.partition("=") for kv in line.split()[1:]) if v.isdigit()}
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the ARENA command")

    def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
        """#533: `VRAM <reserve_mib>` between requests (the caller holds the service's FIFO): the engine shrinks its
        expert cache until that much VRAM is free, or grows it back when more is free; None: back to the reserve it
        started with.  -> the engine's figures (expert_slots, expert_cache_mib, vram_free_mib, ...); raises ValueError
        with the engine's reason (e.g. it was not started with --vram-elastic), EngineDied when it ended."""
        proc = self.proc
        if not self.alive() or proc is None or proc.stdin is None:
            raise EngineDied("the engine is not running")
        proc.stdin.write("VRAM" + ("" if reserve_mib is None else f" {int(reserve_mib)}") + "\n")
        proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the VRAM command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the VRAM command")
            if line.startswith("VRAM "):
                out: dict = {}
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                self.info.update({k: out[k] for k in ("expert_slots", "vram_free_mib") if k in out})
                self.info["vram"] = out
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the VRAM command")

    def helper(self, on: bool, timeout: float = 120.0) -> dict:
        """`HELPER off|on` between requests (the caller holds the service's FIFO): the helper GPU's expert cache closed,
        its VRAM given back for an image job there, or filled again from the RAM copy (a second or two).  -> the
        engine's figures (experts, vram_free_mib, ms); ValueError with the engine's reason, EngineDied when it ended."""
        proc = self.proc
        if not self.alive() or proc is None or proc.stdin is None:
            raise EngineDied("the engine is not running")
        proc.stdin.write(f"HELPER {'on' if on else 'off'}\n")
        proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the HELPER command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the HELPER command")
            if line.startswith("HELPER "):
                out: dict = {}
                for kv in line.split()[2:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the HELPER command")

    def encode(self, ids_path: str, out_path: str, timeout: float = 600.0) -> dict:
        """`ENCODE <ids file> <states file>` between requests (the caller holds the service's FIFO; images.text_states):
        the prompt's states after the final norm into the file, the engine's conversation set aside and put back.
        -> {"tokens", "ms"}; ValueError with the engine's reason (a model it cannot encode with, a conversation it could
        not set aside), EngineDied when it ended or did not answer."""
        if " " in ids_path or any(c in ids_path + out_path for c in "\r\n\0"):   # the engine splits at the first space
            raise ValueError("invalid encode paths")
        proc = self.proc
        if not self.alive() or proc is None or proc.stdin is None:
            raise EngineDied("the engine is not running")
        proc.stdin.write(f"ENCODE {ids_path} {out_path}\n")
        proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the ENCODE command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the ENCODE command")
            if line.startswith("ENCODED "):
                f = line.split()
                return {"tokens": int(f[1]), "ms": float(f[2])}
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the ENCODE command")

    RESTART_RETRY_S = 15.0   # between the tries of restart(): a dying engine's VRAM may take a while to come back

    @staticmethod
    def _ready_pump(proc, out: queue.Queue) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def restart(self, tries: int = 3):
        """Start the engine again (the same command) after it died; the new process has its own line queue.
        close() ends and waits for the old process first (EngineStuck when it cannot be ended).  A start that still
        exits before READY - a dead engine's VRAM can take a while to come back, notably on ROCm - is retried
        (PR #637)."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 "starting" meanwhile (#344)
        try:
            self.close()
            for i in range(tries):
                # #344: not alive until READY - __init__ sets max_context to 0 and blocks until the engine says
                # READY, and a request that saw alive() in that window skipped load() and failed with "context
                # (0)".  __init__ clears `ended` itself once READY (before its pump thread can set it again).
                self.ended = True
                try:
                    # (a test's subclass starts again as itself; spawn_kw is not there for one built without __init__)
                    self.__init__(*self.spawn, **getattr(self, "spawn_kw", {}))   # type: ignore[misc]
                    break
                except RuntimeError:
                    try:
                        if self.proc is not None:
                            self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[strata] the engine did not start (try {i + 1} of {tries}); again in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
        finally:
            self.starting = False
            # a lazy first start never reached the post-READY __init__ where slot_cv is made (#1527)
            if "slot_cv" in self.__dict__:
                with self.slot_cv:               # #1012: wake the requests that waited through it: they go on with the
                    self.slot_cv.notify_all()    # new engine, or (it did not start) end with a clean EngineDied
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
        if len(f) >= 14:                                  # the expert tiers (engine 0.1.31+): RAM / file blobs, file MB
            self.last.update(ram_blobs=int(f[11]), file_blobs=int(f[12]), file_mb=float(f[13]))
        if len(f) >= 15:                                  # #471 (engine 0.1.36+): the prompt tokens actually read
            self.last.update(prompt_read=int(f[14]))
        if len(f) >= 16:                                  # #588 (engine 0.1.39+): routed experts read over PCIe
            self.last.update(offloaded=int(f[15]))

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp, pf, pp = (float(v) if isinstance(v, (int, float)) else None for v in
                      (sampling.get("repetition_penalty"), sampling.get("frequency_penalty"),
                       sampling.get("presence_penalty")))
        rp_on, pf_on, pp_on = rp is not None and rp != 1.0, pf is not None and pf != 0.0, pp is not None and pp != 0.0
        if rp_on:
            keys += f" penalty_repeat={rp!r}"
        if pf_on:
            keys += f" penalty_freq={pf!r}"
        if pp_on:
            keys += f" penalty_present={pp!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        if sampling.get("strata_route_prior") is False:   # the loop guard saw a loop: route exactly from here on
            keys += " route_prior=0"
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune = sampling.get("strata_tune")
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    keys += f" {k}={float(v)!r}"
            # --aux-cpus for this request: 1 puts the engine's threads that are neither pool workers nor the host on
            # the spare CPUs, 0 leaves them where they were created (true/false or 1/0; the engine started without a
            # spare CPU ignores a 1).  Anything else is not sent.
            ac = tune.get("aux_cpus")
            if isinstance(ac, bool) or (isinstance(ac, (int, float)) and ac in (0, 1)):
                keys += f" aux_cpus={int(ac)}"
        k = top_logprobs(sampling)
        if k is not None:
            keys += f" logprobs={k}"
            ids = allowed_ids(sampling)
            if ids:     # only these ids are printed (upstream 9ea044962, #3200); renormalised in LogprobEntries
                keys += " allowed=" + ",".join(str(i) for i in ids)
        # "strata_checkpoint": false - a one-shot call (a classification, a probe) whose turn no later request
        # extends: no conversation checkpoint for it (#830).  It still reuses a cached prefix.  Absent = as before.
        if sampling.get("strata_checkpoint") is False:
            keys += " ckpt=0"
        # #2228: the pictures' keys (hex, in prompt order): the engine matches a picture by them, not by its embeddings
        ik = sampling.get("strata_img_keys")
        if isinstance(ik, list) and ik and all(isinstance(k, str) and k and all(c in "0123456789abcdef" for c in k)
                                               for k in ik):
            keys += " img_keys=" + ",".join(ik)
        # "strata_prefix" (resolved by Service.resolve_prefix to {"tokens": N}): the first N prompt tokens are a shared
        # prefix - the engine pins the checkpoint there.  Absent or unresolved = as before.
        prefix = sampling.get("strata_prefix")
        if isinstance(prefix, dict):
            pn = prefix.get("tokens")
            if isinstance(pn, int) and not isinstance(pn, bool) and pn > 0:
                keys += f" pin={pn}"
        # "strata_grammar" (Service.attach_grammar): the reply follows this grammar (src/core/grammar.cpp)
        grammar = sampling.get("strata_grammar")
        if isinstance(grammar, dict):
            keys += " grammar=" + base64.b64encode(json.dumps(grammar, ensure_ascii=False).encode()).decode("ascii")
        # the script guard's table (serve/script_guard.py; src/core/script_guard.hpp), its path in hex (spaces)
        sgt = sampling.get("strata_script")
        if isinstance(sgt, dict):
            keys += " script_table=" + str(sgt["table"]).encode().hex()
            keys += f" script_allow={sgt['allow']}" if sgt.get("allow") else ""
            keys += f" script_pending={sgt['pending']}" if sgt.get("pending") else ""
        return keys + StrataEngine.projection_key(sampling) + bias_api.engine_key(sampling.get("logit_bias"))

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def _send(self, text: str):
        btrace("send>", text[:60])
        try:
            with self.wlock:
                proc = self.proc
                if proc is None or proc.stdin is None:
                    raise OSError("no engine")
                proc.stdin.write(text + "\n")
                proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})") from None

    def _control(self, cancel, on_token, stop_when=None, on_quiet=None):
        """Reads the control lines of the request on them (GEN / BGEN), yielding None heartbeats.  Calls on_token(id)
        for each `T`; returns ("done", None) at DONE, or ("badm", continues) at BADM (after DONE).  `stop_when()`
        true sends STOP once (the request is then read to its DONE).  `on_quiet()` runs whenever the engine has been
        quiet for half a second: between two PP lines of a read whose chunks take longer than that."""
        stopped = False
        beat = time.monotonic()
        while True:
            try:
                # A client can leave before the first PP/token.  Poll its Event as often as _take_control does;
                # the ten-second HTTP heartbeat is independent of how promptly STOP reaches the engine.
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if cancel.is_set() and not stopped:
                    self._send("STOP")
                    stopped = True
                if on_quiet is not None:
                    on_quiet()
                if time.monotonic() - beat >= 10.0:
                    beat = time.monotonic()
                    yield None
                continue
            beat = time.monotonic()
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})")
            if not line.startswith(("PP ", "INFO")):
                btrace("ctl<", self._ctl_mode, line.strip()[:60])
            if line.startswith("T "):
                on_token(int(line[2:]))
                if not stopped and (cancel.is_set() or (stop_when is not None and stop_when())):
                    self._send("STOP")
                    stopped = True
                yield False                              # a token is pending (not a heartbeat)
            elif line.startswith("PP "):
                f = line.split()
                if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                    self.progress = (int(f[1]), int(f[2]))
                    self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                    self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                if not stopped and cancel.is_set():
                    # #879: a Stop during the prompt read went out only after the first token (or after 10 s of
                    # silence, which a stream of PP lines never allows): the engine read the whole prompt first
                    self._send("STOP")
                    stopped = True
                yield None
            elif line.startswith("DONE"):
                self._parse_done(line)
                self._last_done = line
            elif line.startswith("BADM "):
                f = line.split()
                self._ctl_result = ("badm", len(f) >= 3 and f[2] == "1")
                return
            elif line.startswith("YIELDED "):            # the read gave way (BYIELD): <slot> <tokens read>
                f = line.split()
                if len(f) >= 3 and f[1].lstrip("-").isdigit() and f[2].isdigit():
                    self._yielded = (int(f[1]), int(f[2]))
            elif line.startswith("ERR"):
                self._ctl_erred = True                   # #1059: the engine refused it and is idle: no DONE follows
                raise ValueError(line[4:].strip())
            if line.startswith("DONE") and self._ctl_mode == "solo":
                self._ctl_result = ("done", None)
                return

    def _done_finish(self) -> str | None:
        """The finish field of the DONE line the control lines read last (None: none since it was cleared)."""
        f = (getattr(self, "_last_done", None) or "").split()
        return f[5] if len(f) > 5 else None

    def _drain_control(self, until: str, timeout: float = 300.0, born: int | None = None):
        """After a consumer left early: read the control lines up to the next `until` line (DONE or BADM) so the next
        request does not read this one's leftovers.  Returns that line (None: the engine ended or never answered).
        `born`: the engine process (self.gen) the request started on; once that one is gone there is nothing to
        drain - its end line was read already (by the request's own EngineDied), and waiting for another one held
        a request that had failed for the whole 300 s (#1012), or read the lines of the engine that replaced it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if born is not None and (self.gen != born or not self.alive()):
                return None
            try:
                line = self.lines.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith("DONE"):
                self._parse_done(line)
            if line.startswith(until) or line.startswith("ERR"):
                return line
        return None

    def _apply_bdone(self, f: list[str]) -> None:
        """`BDONE <slot> <produced> <finish> <ms>` into this request's figures."""
        if len(f) >= 5 and isinstance(self.last, dict):
            self.last = {**self.last, "finish": f[3], "decode_ms": float(f[4]),
                         "generated": int(f[2]) if f[2].isdigit() else self.last.get("generated")}

    def _bdone_after_eos(self, slot: int, wait_s: float = 2.0):
        """#1869: a request in a slot whose consumer left at the stop token has its BDONE right behind it (the engine
        prints it in the same window).  Read it here, so the request's figures are the slot leg's (the background wait
        would eat it, and the request kept the admission's single token).  None when it does not come in `wait_s`: then
        the background wait takes over."""
        if slot >= len(self.slot_q):
            return None
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            try:
                line = self.slot_q[slot].get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                return None
            if line is None:
                return None
            if line.startswith("BDONE "):
                return line
        return None

    def _release_slot_when_done(self, slot: int, stream: list[int] | None = None):
        """A slot whose consumer left (a stop token, a stop string, a disconnect): BSTOP it and free it once the engine
        says BDONE (in the background).  `stream`: the prompt and every token of it so far - with the tokens still to
        come before BDONE, all but the last are what the slot holds then (the next turn of its conversation)."""
        if slot >= len(self.slot_q):                    # the engine restarted with fewer slots: this one is gone
            return
        try:
            self._send(f"BSTOP {slot}")
        except EngineDied:
            pass
        q, busy, held = self.slot_q[slot], self.slot_busy, self.slot_held   # this process's: restart() replaces them
        releasing = self.slot_releasing
        releasing.add(slot)

        def wait():
            end = time.monotonic() + 600.0
            tail = list(stream or [])
            while time.monotonic() < end:
                try:
                    line = q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if line is None:
                    tail = []
                    break
                if line.startswith("BT "):
                    try:
                        tail.append(int(line.split()[2]))
                    except (IndexError, ValueError):
                        tail = []
                if line.startswith("BDONE "):
                    if line.split()[3:4] == ["nonfinite"]:   # #879: the engine dropped the slot's state
                        tail = []
                    break
            held[slot] = tail[:-1] if stream and tail else []
            with self.slot_cv:
                busy[slot] = False
                releasing.discard(slot)
                self.slot_cv.notify_all()
        threading.Thread(target=wait, daemon=True).start()

    BGEN_REFUSALS_MAX = 6    # #1994: how often one request tries another slot after the engine refused a BGEN
    SLOT_PARK_S = 2.0        # ... and how long a refused slot stays out of the picks

    def _park_slot(self, slot: int) -> None:
        """#1994: the engine refused a BGEN for `slot`.  Keep it busy here for SLOT_PARK_S (no request owns it), then
        give it back: the engine's view has caught up, or the next refusal parks it again."""
        with self.slot_cv:
            self.slot_claims.discard(slot)
            self.slot_busy[slot] = True
        busy = self.slot_busy

        def free():
            with self.slot_cv:
                if slot not in self.slot_claims and slot not in self.slot_releasing:
                    busy[slot] = False
                self.slot_cv.notify_all()
        t = threading.Timer(self.SLOT_PARK_S, free)
        t.daemon = True
        t.start()

    ORPHAN_SLOT_S = 3.0   # #1603: a busy flag nobody owns this long is a leak and is given back

    def _reap_orphan_slots(self) -> list[int]:
        """#1603: free the slots whose busy flag has no owner (no request holds or reserves them and no BDONE wait is
        running for them).  The caller holds slot_cv.  A flag that leaked kept every later request waiting for a slot
        that nothing would free - with nothing running, nothing queued and nothing for the frozen-engine check to see.
        A slot is only given back after it has looked ownerless for ORPHAN_SLOT_S, so a flag set a moment before its
        owner registers is never taken."""
        now = time.monotonic()
        freed = []
        for b, busy in enumerate(self.slot_busy):
            if busy and b not in self.slot_claims and b not in self.slot_releasing:
                first = self._orphan_seen.setdefault(b, now)
                if now - first >= self.ORPHAN_SLOT_S:
                    self.slot_busy[b] = False
                    self._orphan_seen.pop(b, None)
                    freed.append(b)
            else:
                self._orphan_seen.pop(b, None)
        if freed:
            print(f"[strata] parallel: slot(s) {', '.join(map(str, freed))} were marked busy with no request "
                  "using them "
                  "(issue #1603); freed, the waiting request goes on", flush=True)
            self.slot_cv.notify_all()
        return freed

    YIELDS_MAX = 2   # #656: how often one request's prompt read gives way to a shorter waiting one

    def _take_control(self, cancel, plen: int, after_epoch: int | None = None, new: bool = False):
        """Waits for the control lines (one prompt read at a time), yielding None heartbeats; False when cancelled.
        `after_epoch`: a request whose read gave way lets the requests waiting then go first.
        `new`: the request's first admission (STRATA_ADMIT_SHORT_FIRST orders only these)."""
        entry = [plen, time.monotonic(), new]           # (length, arrival, ordered: STRATA_ADMIT_SHORT_FIRST)
        with self.slot_cv:
            self.waiting += 1
            self.wait_lens.append(entry)
        beat = time.monotonic()
        try:
            while True:
                if not self.alive() and not getattr(self, "starting", False):
                    # #1012: the engine is gone and nobody is starting it again: this request was not sent, so it
                    # ends now (a clean 503) instead of waiting for control lines nothing will answer on.  While a
                    # restart is under way it keeps waiting and goes on with the new engine.
                    raise EngineDied("the engine stopped while this request waited; it was not sent")
                with self.slot_cv:
                    turn = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                    if turn and new and ADMIT_SHORT_FIRST > 0:
                        turn = self._admit_next() is entry
                if turn and self.ctl.acquire(timeout=0.5):
                    if new and ADMIT_SHORT_FIRST > 0:
                        with self.slot_cv:              # a shorter one came while this one waited in acquire()
                            first = self._admit_next() is entry
                        if not first:
                            self.ctl.release()
                            with self.slot_cv:
                                self.slot_cv.notify_all()
                            continue
                    if self.alive():
                        break
                    self.ctl.release()                  # it died (or is being restarted) while this one was getting
                    turn = False                        # the lines: not the engine this request was meant for
                if not turn:
                    with self.slot_cv:
                        self.slot_cv.wait(timeout=0.5)
                if cancel.is_set():
                    return False
                if time.monotonic() - beat >= 10.0:     # a heartbeat every 10 s, as the engine's own wait
                    beat = time.monotonic()
                    yield None
        finally:
            with self.slot_cv:
                self.waiting = max(0, self.waiting - 1)
                if entry in self.wait_lens:
                    self.wait_lens.remove(entry)
        with self.slot_cv:
            self.ctl_epoch += 1
            self.slot_cv.notify_all()
        return True

    def _admit_next(self):
        """STRATA_ADMIT_SHORT_FIRST=<tokens> (the caller holds slot_cv): the new request that takes the control lines
        next - one whose prompt is at most that long (it reads little) before a longer one, in arrival order within
        each; a longer one that has waited STRATA_ADMIT_MAX_WAIT_S counts as short (it is never held back for good).
        Off (unset or 0): whoever gets the lock first, as before.  Only the order changes, never what is read."""
        now = time.monotonic()
        ordered = [e for e in self.wait_lens if len(e) > 2 and e[2]]
        if not ordered:
            return None
        return min(ordered, key=lambda e: (0 if e[0] <= ADMIT_SHORT_FIRST or now - e[1] >= ADMIT_MAX_WAIT_S else 1,
                                           e[1]))

    SOLO_AGAIN_MAX = 2      # how often a request left alone in a slot goes back to the solo path
    SOLO_AGAIN_MIN_LEFT = 32  # ... only with at least this many tokens still allowed (max_tokens)

    def _may_go_solo(self, left: int, times: int, embeddings) -> bool:
        """A request decoding in a slot that is alone now (no other slot busy, nobody waiting) goes back to the solo
        path with its MTP drafts: the engine continues it from the slot's sessions (its slot cache; INFO
        slot_cache=1).  STRATA_PARALLEL_SOLO=0 keeps it in the slot."""
        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or
                not (self.info or {}).get("slot_cache") or os.environ.get("STRATA_PARALLEL_SOLO") == "0"):
            return False
        with self.slot_cv:
            return sum(1 for b in self.slot_busy if b) == 1 and self.waiting == 0

    def _shorter_waiting(self, plen: int) -> bool:
        """#656: someone waits for the control lines with a prompt under half this one's: worth giving way to."""
        with self.slot_cv:
            return any(e[0] * 2 <= plen for e in self.wait_lens)

    def own_lp_lines(self, sampling, cancel, make_current=True):
        """A request asking for logprobs gets LP lines of its own, tagged with its cancel Event and kept on it (one per
        request, handed unchanged from the handler through Service.run): its last events can come after it left the
        queue, when the next request has the engine's lines already (upstream 9ea044962, #3200).  None when it did
        not ask; the engine's lines are then a fresh, untagged deque, as before."""
        if top_logprobs(sampling or {}) is None or cancel is None:
            if make_current:
                self.lp_lines = LPLines()
            return None
        lines = LPLines(cancel)
        cancel._strata_lp_lines = lines
        if make_current:
            self.lp_lines = lines
        return lines

    def generate_batched(self, ids, max_new, sampling, cancel, embeddings=None):
        """--batch.  Alone (no slot busy, nobody waiting): the solo path (GEN, with drafts: the fastest stream)
        - and when another request arrives meanwhile, this one is STOPped and continues in a batch slot (BGEN with
        its prompt + what it generated: the engine reuses that prefix).  Otherwise: BGEN into a free slot, then the
        slot's own BT lines until BDONE.  Several requests run at once; the control lines (prompt reading, admission)
        are taken one request at a time - and a long prompt read gives way at a chunk boundary to a waiting request
        with a much shorter prompt (#656: `BYIELD <slot>`; the part read waits in a slot and the read goes on after).
        A consumer that stops early leaves the engine in step: the solo request is STOPped and read to its DONE, an
        admission to its BADM, a slot is BSTOPped and freed at its BDONE."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.progress, self.progress_ms, self.reused = None, 0, 0
        born = self.gen                                 # the engine process this request is sent to
        keys = self.sampling_keys(sampling or {})
        cache_key = (sampling or {}).get("prompt_cache_key")
        cache_key = cache_key if isinstance(cache_key, str) and cache_key else None
        # logprobs: only a request the server let through (max_tokens 1, see the handler) asks for them here; its LP
        # lines go to its own LPLines, made the engine's while it has the control lines (below)
        own_lp = self.own_lp_lines(sampling, cancel, make_current=False)
        out: list[int] = []
        pending: list[int] = []
        prompt, left = list(ids), int(max_new)
        reused0 = None      # the first read's reused tokens: a continued leg's count includes this request's own output
        # #1869: the figures of the legs that ended in a hand-back to the solo path (the final DONE adds to them)
        carry = {}
        ok = yield from self._take_control(cancel, len(prompt), new=True)
        if not ok:
            return
        holding, born = True, self.gen                  # (the engine it now has the control lines of)
        btrace("ctl acquired")
        slot, gen0, reserved = None, None, None
        phase = "none"            # solo -> (admit -> slot) ; "done" once the engine has finished with this request
        stop_sent = False
        yields, solo_again, refused = 0, 0, 0
        admit_wait, admit_since, admit_said = False, 0.0, False   # #1603: waiting for a slot, since when, said
        try:
            while True:   # a request in a slot that is left alone goes back to the solo path
                if not holding:
                    ok = yield from self._take_control(cancel, len(prompt))
                    if not ok:
                        return
                    holding, born = True, self.gen                  # (the engine it now has the control lines of)
                with self.slot_cv:
                    alone = not any(self.slot_busy) and self.waiting == 0
                if alone and left > 1:
                    head = f"GENI {left}{keys} {embeddings}" if embeddings else f"GEN {left}{keys}"
                    self._last_done = None                  # (#879: this request's DONE decides below)
                    if own_lp is not None:
                        self.lp_lines = own_lp
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    phase = "solo"
                    self._ctl_mode, self._ctl_result, self._yielded = "solo", None, None
                    self._ctl_erred = False

                    def others():
                        with self.slot_cv:
                            return self.waiting > 0

                    # at a PP line, and between two far apart (the engine takes it at its next boundary)
                    def give_way():
                        nonlocal reserved
                        if (reserved is None and not out and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            with self.slot_cv:          # the slot the part read will wait in
                                reserved = self.pick_slot(prompt, cache_key)
                                if reserved is not None:
                                    self.slot_busy[reserved] = True
                                    self.slot_claims.add(reserved)
                                    self.slot_key[reserved] = cache_key
                            if reserved is not None:
                                self._send(f"BYIELD {reserved}")
                    for x in self._control(cancel, pending.append, stop_when=others, on_quiet=give_way):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:                       # a heartbeat (False: a token, flushed above)
                            give_way()
                            yield None
                    phase = "none"                          # its DONE is read
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if self._done_finish() == "nonfinite":    # #879 (never promoted to a slot)
                        raise EngineNonFinite("the engine's logits became non-finite (upstream Strata #879)")
                    if self._yielded is not None and reserved is not None and self._yielded[0] == reserved:
                        slot, reserved = reserved, None     # gave way: the read goes on in that slot (below)
                    elif reserved is not None:              # it did not give way: the slot is free again
                        with self.slot_cv:
                            self.slot_busy[reserved] = False
                            self.slot_claims.discard(reserved)
                            self.slot_cv.notify_all()
                        reserved = None
                    if slot is None:
                        finish = (self.last or {}).get("finish") if isinstance(self.last, dict) else None
                        left = int(max_new) - len(out)
                        stopped = bool(out) and out[-1] in self.eos_ids
                        if cancel.is_set() or left <= 0 or finish in ("stop", "length") or stopped:
                            return
                        if reused0 is None:
                            reused0 = (self.last or {}).get("reused")
                        # #1869: the solo leg's figures; the slot's admission (a copy of the held prefix:
                        # prompt_read 1) and BDONE replace them
                        carry_leg(carry, self.last)
                        prompt = list(ids) + out            # promoted: it continues in a batch slot from here
                while True:
                    if slot is None:
                        # a free slot (they free themselves at BDONE, which needs no control lines): the one that holds
                        # the start of this prompt (its conversation's last turn), else the one used longest ago.
                        # With none free the control lines are given back while waiting: a slot can be held by a read
                        # that gave way (BYIELD) and needs them to go on, so waiting with them is a deadlock - long,
                        # long, short at 0.25 s steps under parallel: 2 (ENGINE_REVIEW finding 1, reproduced in 0.1.41).
                        while True:
                            with self.slot_cv:
                                slot = self.pick_slot(prompt, cache_key)
                                if slot is not None:
                                    self.slot_busy[slot] = True
                                    self.slot_claims.add(slot)
                                    self.slot_key[slot] = cache_key
                                    if admit_wait:
                                        admit_wait = False
                                        self.admitting = max(0, self.admitting - 1)
                                    break
                                if not admit_wait:          # #1603: counted while it waits for a slot (it holds no
                                    admit_wait, admit_since = True, time.monotonic()   # control lines then)
                                    self.admitting += 1
                            if holding:
                                self.ctl.release()
                                holding = False
                            with self.slot_cv:
                                if self.pick_slot(prompt, cache_key) is None:
                                    if (ENGINE_STALL_S or 0) > 0 and time.monotonic() - admit_since >= ENGINE_STALL_S:
                                        # the stall check of #1317 only watches a request on the control lines; one
                                        # waiting here is covered by looking for a slot nobody owns, and by saying so
                                        if not admit_said:
                                            admit_said = True
                                            busy = [b for b, v in enumerate(self.slot_busy) if v]
                                            waited = time.monotonic() - admit_since
                                            print(f"[strata] parallel: a request has waited {waited:.0f} s "
                                                  f"for a free slot (busy: {busy}, in use: {sorted(self.slot_claims)}, "
                                                  f"ending: {sorted(self.slot_releasing)}) (issue #1603)", flush=True)
                                    self._reap_orphan_slots()
                                if self.pick_slot(prompt, cache_key) is None:
                                    self.slot_cv.wait(timeout=1.0)
                            if cancel.is_set():
                                return
                            if not holding:
                                ok = yield from self._take_control(cancel, len(prompt))
                                if not ok:
                                    return
                                holding, born = True, self.gen
                    if self._yielded is not None:           # it gave way: the others waiting then go first
                        self.slot_held[slot] = list(prompt[:self._yielded[1]])
                        self._yielded = None
                        yields += 1
                        self.ctl.release()
                        holding = False
                        with self.slot_cv:
                            epoch = self.ctl_epoch
                        ok = yield from self._take_control(cancel, len(prompt), after_epoch=epoch)
                        if not ok:
                            return
                        holding, born = True, self.gen                  # (the engine it now has the control lines of)
                    while not self.slot_q[slot].empty():
                        self.slot_q[slot].get_nowait()
                    head = f"BGENI {slot} {left}{keys} {embeddings}" if embeddings else f"BGEN {slot} {left}{keys}"
                    live = {"slot": slot, "state": "reading", "prompt_tokens": len(prompt), "generated": 0,
                            "started": time.time(), "first_token": None}
                    self.slot_live[slot] = live
                    self._last_done = None                  # (#879: this admission's DONE decides below)
                    if own_lp is not None:
                        self.lp_lines = own_lp
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    self.slot_held[slot] = []               # the admission overwrites what the slot held
                    phase = "admit"
                    self._ctl_mode, self._ctl_result, self._yielded = "batch", None, None
                    self._ctl_erred = False
                    asked = False

                    def ask():        # as give_way above, for the slot this admission reads into
                        nonlocal asked
                        if (not asked and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            self._send(f"BYIELD {slot}")
                            asked = True
                    try:
                        for x in self._control(cancel, pending.append, on_quiet=ask):
                            while pending:
                                t = pending.pop(0)
                                out.append(t)
                                yield t
                            if x is None:
                                ask()
                                yield None
                    except ValueError as e:
                        # #1994: the engine says this slot is not free (its view and ours differ for a moment, or a
                        # slot it still holds).  That is not the request's fault and nothing was read: park the slot
                        # for a moment (busy here too), and wait for another one - never a 400 for the conversation.
                        if not (self._ctl_erred and str(e).startswith("BGEN:") and left >= 1 and not cancel.is_set()):
                            raise
                        refused += 1
                        print(f"[strata] parallel: the engine refused slot {slot} ({e}); trying another slot "
                              f"({refused}/{self.BGEN_REFUSALS_MAX}) (issue #1994)", flush=True)
                        if refused > self.BGEN_REFUSALS_MAX:
                            raise GpuBusy("the engine's batch slots are not free right now; try again in a moment",
                                          retry_after=5) from e
                        self.slot_live[slot] = None
                        self.slot_held[slot] = []
                        phase = "none"
                        self._ctl_erred = False
                        self._park_slot(slot)
                        slot = None
                        continue
                    cont = bool(self._ctl_result and self._ctl_result[1])
                    phase = "slot" if cont else "none"
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if not cont and self._done_finish() == "nonfinite":   # #879
                        raise EngineNonFinite("the engine's logits became non-finite (upstream Strata #879)")
                    if not cont and self._yielded is not None and self._yielded[0] == slot and not cancel.is_set():
                        continue                            # gave way: again once the shorter request is in
                    break
                self.ctl.release()
                holding = False
                if not cont:
                    return
                live.update(state="decoding", first_token=time.time(), generated=len(out))
                gen0 = len(out) - 1                         # the admission's own token: the slot feeds it first
                going_solo = False
                while True:
                    try:
                        line = self.slot_q[slot].get(timeout=10.0)
                    except queue.Empty:
                        if cancel.is_set() and not stop_sent:
                            self._send(f"BSTOP {slot}")
                            stop_sent = True
                        yield None
                        continue
                    if line is None:
                        phase = "none"
                        raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})")
                    if line.startswith("BT "):
                        t = int(line.split()[2])
                        out.append(t)
                        live["generated"] = len(out)
                        if cancel.is_set():
                            if not stop_sent:
                                self._send(f"BSTOP {slot}")
                                stop_sent = True
                            continue
                        yield t
                        if not going_solo and not stop_sent and self._may_go_solo(int(max_new) - len(out), solo_again,
                                                                                  embeddings):
                            self._send(f"BSTOP {slot}")         # alone now: on with the drafts (below)
                            going_solo = True
                    elif line.startswith("BDONE "):
                        phase = "none"
                        f = line.split()
                        self._apply_bdone(f)
                        # what the slot's sessions hold now: the prompt and every token fed (all but the last one)
                        self.slot_held[slot] = list(prompt) + out[gen0:-1] if len(out) > gen0 else []
                        if f[3:4] == ["nonfinite"]:              # #879: the engine dropped the slot's state
                            self.slot_held[slot] = []
                            raise EngineNonFinite("the engine's logits became non-finite (upstream Strata #879)")
                        if (going_solo and f[3:4] == ["cancel"] and not cancel.is_set() and len(out) < int(max_new)
                                and not (out and out[-1] in self.eos_ids)):
                            # the solo path continues it: the engine copies the slot's sessions back (all but the
                            # last token are in them) and decodes with MTP drafts again
                            carry_leg(carry, self.last)         # #1869: the solo path's DONE replaces these figures
                            self.slot_live[slot] = None
                            self.slot_used[slot] = time.time()
                            with self.slot_cv:
                                self.slot_busy[slot] = False
                                self.slot_claims.discard(slot)
                                self.slot_cv.notify_all()
                            slot, gen0 = None, None
                            if reused0 is None:
                                reused0 = (self.last or {}).get("reused")
                            prompt, left = list(ids) + out, int(max_new) - len(out)
                            solo_again += 1
                            btrace("back to the solo path")
                            break
                        return
        finally:
            # a consumer that left early (or an error): keep the engine and this server in step
            btrace("finally phase", phase, "slot", slot, "holding", holding)
            try:
                if phase in ("solo", "admit") and (self.gen != born or not self.alive()):
                    phase = "none"                      # #1012: its engine is gone: nothing to stop or drain
                if phase in ("solo", "admit") and self._ctl_erred:
                    phase = "none"                      # #1059: it ended on the engine's ERR: no DONE to wait 300 s for
                if phase == "solo":
                    self._send("STOP")
                    self._drain_control("DONE", born=born)
                elif phase == "admit":
                    line = self._drain_control("BADM", born=born)
                    if line and line.startswith("BADM ") and line.split()[2:3] == ["1"]:
                        phase = "slot"
            except EngineDied:
                pass
            bdone_in = False
            if phase == "slot" and slot is not None and out and out[-1] in EOS_IDS and self.gen == born:
                bd = self._bdone_after_eos(slot)            # (#1869) the slot's figures, not the admission's
                if bd is not None:
                    self._apply_bdone(bd.split())
                    bdone_in = True
            if reused0 is not None and isinstance(self.last, dict):
                self.last = {**self.last, "reused": reused0}
            if carry and isinstance(self.last, dict):
                self.last = add_carry(carry, self.last)
            if holding:
                self.ctl.release()
            if admit_wait:
                with self.slot_cv:
                    self.admitting = max(0, self.admitting - 1)
            if reserved is not None:
                with self.slot_cv:
                    self.slot_busy[reserved] = False
                    self.slot_claims.discard(reserved)
                    self.slot_cv.notify_all()
            if slot is not None:
                self.slot_live[slot] = None
                self.slot_used[slot] = time.time()
                if phase == "slot" and bdone_in:
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self.slot_held[slot] = stream[:-1] if stream else []     # as the BDONE wait leaves it
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()
                elif phase == "slot":
                    self.slot_held[slot] = []
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self._release_slot_when_done(slot, stream)    # freed at its BDONE
                    self.slot_claims.discard(slot)                # (the BDONE wait owns it now)
                else:
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()

    slot_groups = 1         # --batch-groups (set when the engine starts)
    SLOT_PREFIX_MIN = 512   # a held prefix shorter than this is not worth unbalancing the pipeline's groups
    TURN_ID = 248045        # <|im_start|>: the engine's default --turn-token (its turn checkpoint sits at the last one)

    def pick_slot(self, prompt: list[int], key: str | None = None) -> int | None:
        """A free slot for `prompt` (the caller holds slot_cv): the one whose held tokens are the longest start of the
        prompt (the engine then reads only the rest), else one the same session (`key`, the request's
        prompt_cache_key) took last - what it holds is that session's own, which this prompt supersedes - else an
        empty one, else the one used longest ago - so the conversations other slots hold stay for their next turns.
        None: every slot is busy."""
        free = [b for b in self.slot_order if not self.slot_busy[b]]
        if not free:
            return None

        def held_prefix(b):
            # what of `prompt` the slot's cache serves: the slot's whole held stream (the prompt and what it decoded)
            # when this prompt continues it, else its turn checkpoint - the engine takes one at the last <|im_start|> of
            # a prompt (the next request of the chat renders the history again, but not the reply it decoded or the
            # assistant header), so a prompt that agrees with the held tokens up to there is that chat's next turn
            # too (#2518, upstream dd08a7d0: another slot was picked, and the request there overwrote the chat whose
            # cache the next one needed)
            h = self.slot_held[b]
            if not h:
                return 0
            if len(h) < len(prompt) and prompt[:len(h)] == h:
                return len(h)
            k = len(h) - 1 - h[::-1].index(self.TURN_ID) if self.TURN_ID in h else 0
            return k if 0 < k < len(prompt) and prompt[:k] == h[:k] else 0
        best = max(free, key=held_prefix)
        # without --batch-groups a held prefix of any length is taken, as before
        if held_prefix(best) >= (self.SLOT_PREFIX_MIN if self.slot_groups > 1 else 1):
            return best
        own = [b for b in free if key is not None and self.slot_key[b] == key]
        if own:
            return min(own, key=lambda b: self.slot_used[b])
        # pipelined groups: a group's window spans its slots up to its last busy one, so a free slot below a busy one
        # of its group costs no extra row: such a hole is filled first (upstream 2c7c02ab, #793: picks that left
        # leading holes, [_ B], cost a whole row each).  After a held prefix, as a pipelined slot is a cache again
        # (upstream fc844436)
        gs = getattr(self, "slot_gs", 0)
        if gs:
            holes = [b for b in free if any(self.slot_busy[x] for x in range(b + 1, b - b % gs + gs))]
            if holes:
                return min(holes, key=self.slot_order.index)
        # with --batch-groups: the group with the fewest busy slots first, so the requests spread over the
        # pipeline's stages (two requests in one group leave the other stage idle); then empty, then least recent
        busy_in: dict[int, int] = {}
        for b in range(self.batch):
            if self.slot_busy[b]:
                busy_in[self.slot_group[b]] = busy_in.get(self.slot_group[b], 0) + 1
        return min(free, key=lambda b: (busy_in.get(self.slot_group[b], 0) if self.slot_groups > 1 else 0,
                                        bool(self.slot_held[b]), self.slot_used[b]))

    def slots_view(self) -> list[dict]:
        """/metrics: each batch slot - idle (with the tokens it holds for a next turn), reading or decoding."""
        now, view = time.time(), []
        for b in range(self.batch):
            r = self.slot_live[b]
            if r is None:
                view.append({"slot": b, "state": "idle", "held_tokens": len(self.slot_held[b])})
                continue
            ft = r.get("first_token")
            view.append({"slot": b, "state": r["state"], "prompt_tokens": r["prompt_tokens"],
                         "generated": r["generated"], "elapsed_s": round(now - r["started"], 1),
                         "tok_s": round(r["generated"] / max(1e-6, now - ft), 1) if ft else None})
        return view

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        bias_api.validate_request(sampling or {}, self, None)
        if getattr(self, "batch", 0):
            yield from self.generate_batched(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress, self.progress_ms, self.reused = None, 0, 0
        self.prefill_tok_s_mean = None
        self.own_lp_lines(sampling, cancel)             # this request's LP lines only
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        if prefill_only:                                 # split serving (serve/split.py): read the prompt, no token
            if embeddings or int(max_new) != 0:
                raise ValueError("prefill_only is a text prompt and max_new 0")
            head = "GEN 0 prefill=1"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})") from None
        done = False
        # #481: how long the engine may stay silent from here.  Until the first line: the request's first prompt
        # chunk at the slowest prompt reading on top of silence_s; a PP line resets it to its own chunk's time.
        silence = float(self.silence_s or 0)
        allow = silence + min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
        heard, read_to = time.monotonic(), 0
        beat = heard
        stall_s = ENGINE_STALL_S or 0.0
        stall_state: dict = {}
        try:
            while True:
                wait = 10.0
                if allow > 0:
                    wait = min(wait, allow - (time.monotonic() - heard))
                    if wait <= 0:
                        done = True                       # no STOP and no drain: nothing is listening
                        raise self._silent(f"the engine said nothing for {time.monotonic() - heard:.0f} s during "
                                           "the request")
                try:
                    line = self.lines.get(timeout=min(wait, 0.5))
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if stall_s > 0:
                        try:
                            self._stall_check(heard, stall_state, stall_s)
                        except EngineSilent:
                            done = True                   # as above: nothing is listening
                            raise
                    if allow > 0 and time.monotonic() - heard >= allow:
                        continue                          # enforce the expired deadline before another heartbeat
                    if time.monotonic() - beat >= 10.0:
                        beat = time.monotonic()
                        yield None                        # polling cancellation does not add HTTP heartbeats
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})")
                heard = time.monotonic()                  # any line is output: T, PP, RESUME, INFO ...
                beat = heard
                if stall_state:
                    stall_state.clear()
                if line.startswith("T "):
                    allow = silence
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    # prompt progress, one per chunk, and a heartbeat too: the lines reset the 10 s wait, so without
                    # the yield below a long prompt would send no keep-alive at all
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))
                        self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        rate, chunk = self.prefill_tok_s_mean or 0.0, int(f[1]) - read_to
                        read_to = int(f[1])
                        if silence > 0 and rate > 0 and chunk > 0:   # #481: the next chunk, as long as this one
                            allow = max(silence, PP_SLACK * chunk / rate)
                    if cancel.is_set():
                        return
                    yield None
                elif line.startswith("RESUME "):          # the reused tokens: the first chunk starts after them
                    try:
                        read_to = self.reused = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    if self.last.get("finish") == "nonfinite":   # #879
                        raise EngineNonFinite("the engine's logits became non-finite (upstream Strata #879)")
                    return
                elif line.startswith("ERR"):
                    done = True
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # #481: never an untimed wait here - it holds the request FIFO, and an engine that lost step never
                # answers.  An engine that honours STOP gets the current allowance in all (a STOP during a prompt
                # chunk is seen after it); an older one runs on to max_new, so each line only has to come in time.
                heard = time.monotonic()
                while True:
                    left = allow - (time.monotonic() - heard) if allow > 0 else None
                    try:
                        if left is not None and left <= 0:
                            raise queue.Empty
                        line = self.lines.get(timeout=left)
                    except queue.Empty:
                        raise self._silent("the engine did not finish the request after it was stopped (STOP) "
                                           f"within {allow:.0f} s") from None
                    if line is None or line.startswith("ERR"):
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break
                    if not self.can_stop:
                        heard = time.monotonic()

    def _activity(self) -> tuple[float, int] | None:
        """#1317: (CPU seconds, I/O bytes) the engine process has used so far, or None when they cannot be read."""
        try:
            import psutil
            if self.proc is None:
                return None
            pr = psutil.Process(self.proc.pid)
            t = pr.cpu_times()
            io = pr.io_counters()
            return (t.user + t.system, io.read_bytes + io.write_bytes)
        except Exception:  # noqa: BLE001 - psutil missing, the process gone, no permission: no reading
            return None

    def _stall_check(self, heard: float, state: dict, stall_s: float) -> None:
        """#1317: called while the engine is silent.  Takes a baseline soon after the last line, and at stall_s of
        silence compares: no CPU and no I/O in all that time -> the engine is ended (EngineSilent); work going on -> one
        warning."""
        age = time.monotonic() - heard
        if state.get("base") is None:
            state["base"] = self._activity()
            return
        if age < state.get("next", stall_s) or state["base"] is None:
            return
        now = self._activity()
        try:
            gpu_working = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu_working = False
        if now is not None and not gpu_working and engine_frozen(state["base"], now):
            raise self._silent(f"the engine said nothing for {age:.0f} s and used no CPU or disk in that time (frozen; "
                               "STRATA_ENGINE_STALL_S sets this, 0 = off)")
        if now is not None:   # working: look again after another window from here, and say so once
            state["base"], state["next"] = now, age + stall_s
            if not state.get("warned"):
                state["warned"] = True
                print(f"[strata] the engine has said nothing for {age:.0f} s but is still working; it is not ended "
                      "(the limit is engine_silence_s). If it is stuck, \"engine_silence_s\" can be set lower.",
                      flush=True)
        else:
            state["next"] = float("inf")   # no reading (psutil missing): nothing to decide on

    def _silent(self, what: str) -> EngineSilent:
        """#481: end an engine that lost step with the server (its main thread waits for a command the server never
        sends), so the next request starts it again: killed now, its GPU and RAM go back with the process."""
        self.silent_note = ("The engine and the server lost step (issue #481; a very slow PC can raise "
                            "\"engine_silence_s\" in the config, 0 = wait forever). If it happens again, please add "
                            "the end of the engine log to an issue at github.com/MistVVK/XeStrata/issues.")
        self.ended = True                               # not alive from now: the next request restarts it
        proc = self.proc
        try:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=20)                       # restart() -> close() handles one that is still exiting
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineSilent(f"{what}; the server ended the engine")

    def session_file(self, action: str, path: str) -> dict:
        """Disk sessions: `SAVE <path>` / `RESTORE <path>` between requests (the caller holds the service FIFO).
        -> {"tokens", "bytes", "ms"}.  SessionRefused (kind, published) when the engine refused or failed the file and
        is still in step (`SERR <kind> <0|1> <reason>`).  Any line this exchange does not allow - an unknown or
        malformed one, a bare ERR, an answer with negative counts or a non-finite time - means the two sides lost
        step: the engine is ended (EngineDied) and the next request starts it again.  EngineDied also when it ended
        by itself (a restore transfer that failed after the device state changed ends it on purpose: FATAL), and
        EngineSilent (#481) when it printed nothing valid for engine_silence_s - a SESSION progress line (one per
        block of the file moved, at most 16 MiB) counts only when its numbers are.  `SWAIT <phase> <seconds>` announces
        a step that blocks in one call (a flush, the device transfer): until the next line the wait is that step's
        explicit allowance (at most SESSION_WAIT_MAX_S) when it is longer than engine_silence_s - no more, so a step
        that never ends is still ended."""
        if action not in ("save", "restore") or any(c in path for c in "\r\n\0"):
            raise ValueError("invalid session command")
        proc = self.proc
        try:
            if proc is None or proc.stdin is None:
                raise OSError("no engine")
            proc.stdin.write(f"{'SAVE' if action == 'save' else 'RESTORE'} {path}\n")
            proc.stdin.flush()
        except OSError:
            raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})") from None
        silence = float(self.silence_s or 0)
        heard = time.monotonic()
        want = "SAVED" if action == "save" else "RESTORED"

        def out_of_step(line: str) -> EngineDied:
            self.ended = True                               # not alive from now: the next request restarts it
            try:
                proc.kill()
                proc.wait(timeout=20)
            except (OSError, AttributeError, subprocess.TimeoutExpired):
                pass
            return EngineDied(f"the engine answered a session {action} with {line.strip()[:200]!r}; "
                              "the server ended the engine")

        def count(s: str) -> int:
            if not s.isdigit():                             # no sign, no blank, no fraction
                raise ValueError
            return int(s)

        last_done = -1
        allow = 0.0                                         # the announced blocking step's allowance, seconds
        while True:
            limit = max(silence, allow)
            left = limit - (time.monotonic() - heard) if silence > 0 else None
            try:
                if left is not None and left <= 0:
                    raise queue.Empty
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise self._silent(f"the engine said nothing for {limit:.0f} s during a session {action}"
                                   + (f" (in a blocking step allowed {allow:.0f} s)" if allow > silence else "")
                                   ) from None
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly ({exit_text(self.exit_code())})")
            f = line.split()
            head = f[0] if f else ""
            if head == "SESSION":                          # a large file moving: progress, and a heartbeat
                try:
                    if len(f) != 3:
                        raise ValueError
                    done, total = count(f[1]), count(f[2])
                    if done > total or done < last_done:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                last_done = done
                heard, allow = time.monotonic(), 0.0
                continue
            if head == "SWAIT":                            # a blocking step starts: its bounded allowance
                try:
                    if len(f) != 3 or not f[1].isalpha() or not f[1].isascii():
                        raise ValueError
                    seconds = count(f[2])
                    if not 0 < seconds <= SESSION_WAIT_MAX_S:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                heard, allow = time.monotonic(), float(seconds)
                continue
            if head == "FATAL":
                raise EngineDied(line[6:].strip() or "the engine ended during a session restore")
            if head == "SERR":
                parts = line.rstrip("\r\n").split(" ", 3)
                if len(parts) != 4 or parts[1] not in SessionRefused.STATUS or parts[2] not in ("0", "1"):
                    raise out_of_step(line)
                raise SessionRefused(parts[1], parts[3].strip(), parts[2] == "1")
            if head == want:
                try:
                    if len(f) != 4:
                        raise ValueError
                    tokens, size, ms = count(f[1]), count(f[2]), float(f[3])
                    if not math.isfinite(ms) or ms < 0:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                return {"tokens": tokens, "bytes": size, "ms": ms}
            if head == "INFO" or head == "WARN":          # engine log lines may interleave; they are not answers
                heard = time.monotonic()
                continue
            raise out_of_step(line)

    def close(self):
        """End the engine process: QUIT first (the engine frees its memory itself - unpinning tens of GB can take
        a while), then terminate, then kill, each given 20 s.  Raises EngineStuck when it still runs after all three."""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()  # the engine's stdin reader sees EOF
                    self.proc.wait(timeout=20)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.proc.terminate()
                    self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise EngineStuck("Strata is still releasing GPU/RAM; retry unloading after it exits") from None
        except OSError:
            pass
        finally:
            if self.proc.poll() is not None:
                if self.pump is not None and self.pump.ident is not None:   # a restart can stop before starting it
                    self.pump.join(timeout=2)
                try:
                    self.proc.stdin.close()             # buffered data may flush again after the engine has exited
                except OSError:
                    pass
                self.proc.stdout.close()
                if self.log not in (None, subprocess.DEVNULL):
                    self.log.close()
                self.proc = None
                self.ended = True
                self.progress, self.last = None, {}
                self.progress_ms, self.reused = 0, 0


IMAGE_URL_MAX = 32 << 20        # an image URL is read up to this (the web app attaches pictures of up to 20 MB)


def private_url_host(url: str) -> str | None:
    """#28: the address an image URL's host resolves to when it is not a public one (loopback, link-local - the cloud
    metadata address -, private, reserved, multicast), else None.  A request's image URL must not make the server a
    proxy into its own network; STRATA_IMAGE_URL_PRIVATE=1 allows it (pictures served from the LAN on purpose).  An
    address that changes between this check and the connection (DNS rebinding) is not caught."""
    import ipaddress
    if (os.environ.get("STRATA_IMAGE_URL_PRIVATE") or "").strip() == "1":
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None                                     # not resolvable: the fetch itself says so
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if not addr.is_global:
            return str(addr)
    return None


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    """#28: a redirect to a private address is refused as the first URL is (a public URL could bounce inward)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (addr := private_url_host(newurl)) is not None:
            raise ValueError(f"the image URL redirects to a private address ({addr}); STRATA_IMAGE_URL_PRIVATE=1 "
                             "allows it")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def network_path(path: str) -> bool:
    r"""A path to another computer: \\host\share or //host/share (either slash), \\?\UNC\host\..., \\.\UNC\host\...
    or the NT form \??\UNC\host\...  Windows connects to that host at the first look at the path (os.path.isfile is
    enough) and signs in with the user's NTLM credentials.  \\?\ and \??\ name local files too; no client needs them."""
    p = path.replace("/", "\\")
    return p.startswith("\\\\") or p.startswith("\\??\\")


def vision_label(cfg: dict) -> str:
    return "the GPU encoder" if cfg.get("gpu") else "the CPU encoder"


def vision_attempts(cfg: dict) -> list[dict]:
    """The encoders to start, in order: the configured one, then the config's `fallback` (setup writes the CPU
    encoder there)."""
    first = dict(cfg)
    first.pop("fallback", None)
    out = [first]
    if isinstance(cfg.get("fallback"), dict):
        out.append(dict({k: cfg[k] for k in ("mmproj", "model") if k in cfg}, **cfg["fallback"]))
    return out


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once.

    Every wait on it is bounded.  An encoder that stops or stops answering is marked dead: later image requests
    fail with the reason and text requests are unaffected."""

    @staticmethod
    def work_dir() -> Path:
        """#480: the directory for the encoder's files.  The engine reads the embeddings file named in a `GENI` line
        and the encoder the output named in an `ENC <image> <out>` line, both split at spaces, so a space in the path
        (a %TEMP% under a user name with a space) broke every image.  The system temp directory is used while its path
        has no space, then the working directory, then the system drive's root; the plain temp directory is the last
        resort (the encoder's own files are then named relative to it, see `_start`)."""
        for parent in (None, Path.cwd(), Path(os.environ.get("SystemDrive", "C:") + os.sep)):
            try:
                d = Path(tempfile.mkdtemp(prefix="strata-vision-", dir=parent))
            except OSError:
                continue                                       # not writable, or no such drive
            if " " not in str(d):
                return d
            try:
                d.rmdir()                                      # just created, so still empty
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix="strata-vision-"))

    def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = False):
        args = self.reader_args(cfg)
        self.label = vision_label(cfg)
        self.dir = self.work_dir()
        self.spawn = (args, log, env)                   # to start it again after an unload (#208)
        self.proc: subprocess.Popen | None = None
        self.dead: str | None = None
        self.stopped = True                             # until _start: a lazy encoder starts with the model
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}
        # the vision section's "elastic" (vision_elastic_on: by default with a GPU encoder and an elastic expert cache):
        # the encoder starts with the first picture and stops after elastic_idle_s without one; while it runs, the
        # engine's expert cache keeps elastic_mib of VRAM free for it (Service._vision_up / _vision_idle_down)
        self.max_tokens = self.token_limit(cfg)
        self.elastic = cfg.get("elastic") is True
        self.elastic_mib = int(cfg.get("elastic_mib") or 2048)
        self.elastic_idle_s = float(cfg.get("elastic_idle_s") or 120)
        if not lazy and not self.elastic:
            try:
                self._start()
            except (OSError, RuntimeError):
                shutil.rmtree(self.dir, ignore_errors=True)
                raise

    # What a picture may become when the config does not say (setup writes the same numbers, setup.VISION): an old,
    # hand-edited or migrated config without vision.max_tokens left the reader unclamped, and #3690 saw one picture
    # become 93,700 image tokens (I7).  A key that is there is the person's: 0 means no limit, as before.
    DEFAULT_MAX_TOKENS = {"gpu": 1024, "cpu": 300}

    @staticmethod
    def token_limit(cfg: dict) -> int:
        """The --max-tokens of a reader config: its max_tokens, else the default of its kind (GPU 1024, CPU 300)."""
        if cfg.get("max_tokens") is not None:
            try:
                return max(0, int(cfg["max_tokens"]))
            except (TypeError, ValueError):
                pass
        return Vision.DEFAULT_MAX_TOKENS["gpu" if cfg.get("gpu") else "cpu"]

    @staticmethod
    def reader_args(cfg: dict) -> list:
        """The picture reader's command line from its config (upstream 828a2559a's, with XeStrata's --gpu-pci)."""
        def absolute(p):                                # the encoder runs in its own dir (#480)
            return os.path.abspath(p) if os.path.dirname(str(p)) else p
        args = [absolute(cfg["exe"]), "--mmproj", absolute(cfg["mmproj"]), "--model", absolute(cfg["model"])]
        if cfg.get("gpu"):
            args.append("--gpu")
            if cfg.get("gpu_pci"):
                args += ["--gpu-pci", str(cfg["gpu_pci"])]
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        limit = Vision.token_limit(cfg)                 # I7 (#3690): a default when the config names none
        if limit:
            args += ["--max-tokens", str(limit)]
        if cfg.get("min_tokens"):                       # #767: mtmd's image_min_tokens (a hand-edited key)
            args += ["--min-tokens", str(cfg["min_tokens"])]
        return args

    def _ensure_dir(self):
        """#3370: a temp cleaner (Windows Storage Sense) removes the empty work folder while the encoder is unloaded;
        the next start then failed with WinError 267 for as long as the server lived.  Make it again; pictures cached in
        it are gone."""
        if not self.dir.is_dir():
            self.dir.mkdir(parents=True, exist_ok=True)
            self.cache.clear()

    def _start(self):
        """Start the encoder process and wait (bounded) for its READY; RuntimeError when it does not come."""
        args, log, env = self.spawn
        self.dead = None
        self._ensure_dir()
        # cwd is the encoder's directory: the ENC lines name the files relative to it (#480: no space in the line)
        self.proc = popen("the image encoder", args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=log or subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, env=env,
                          cwd=self.dir)
        self.lines: queue.Queue = queue.Queue()    # its stdout, read on a thread so that a wait can time out
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()
        line = self._line(VISION_READY_S)
        if line is None or not line.startswith("READY"):
            why = ("no answer within %d s" % VISION_READY_S if line == "" else
                   "it exited" if line is None else line.strip())
            self._stop(why)
            raise RuntimeError("the vision encoder did not start: " + why)
        self.stopped = False

    def alive(self) -> bool:
        """False after `unload` (and before a lazy encoder's first start): the next request starts it again.  An
        encoder that died or stopped answering stays marked dead (images off, with the reason) instead of being
        restarted at every request."""
        return not self.stopped

    def unload(self):
        """Stop the encoder process (its VRAM or RAM goes back); the encoded images stay cached on disk."""
        self.close()

    def restart(self):
        """Start the encoder again after an unload; the cache of encoded images is kept.  RuntimeError when it does
        not start (it is then marked dead, so images are off until the server restarts)."""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            self._close_stdin()
        try:
            self._start()
        finally:
            self.stopped = False

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)                             # end of output: the encoder exited

    def _line(self, timeout: float | None) -> str | None:
        """Its next line; "" when none came within `timeout` (None: wait for ever), None when it exited."""
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return ""

    def _stop(self, why: str) -> None:
        self.dead = why
        proc = self.proc
        try:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            pass
        self._close_stdin()

    def _close_stdin(self) -> None:
        proc = self.proc
        try:                                        # a line it never read stays buffered; closing drops it here
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            try:
                return base64.b64decode(source.split(",", 1)[1])
            except (IndexError, ValueError) as e:                # no comma, or not base64
                raise ValueError(f"the image's data: URL could not be read ({e})") from None
        if source.startswith(("http://", "https://")):
            return Vision.download(source)
        path = source[7:] if source.startswith("file://") else source
        # refused before anything looks at the path: file://host/share/x.png is \\host\share\x.png ("file://C:/x.png",
        # a drive, is a client's spelling of file:///C:/x.png)
        host = re.split(r"[/\\]", path, maxsplit=1)[0] if source.startswith("file://") else ""
        if network_path(path) or host and not re.fullmatch(r"[A-Za-z]:|localhost", host, re.IGNORECASE):
            raise ValueError("an image file must be on this computer: network paths (\\\\host\\share, //host/share, "
                             "file://host/...) are not read")
        if path and os.path.isfile(path):
            try:
                if os.path.getsize(path) > IMAGE_URL_MAX:   # #27: not pulled into memory whole (a model file, a log)
                    raise ValueError(f"the image file is over {IMAGE_URL_MAX >> 20} MiB")
                return Path(path).read_bytes()
            except OSError as e:
                raise ValueError(f"the image could not be read from {path} ({e})") from None
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def download(url: str) -> bytes:
        """An image URL's bytes, at most IMAGE_URL_MAX of them: the whole response was read, so a huge or endless one
        filled the memory.  One that cannot be read is a ValueError (a 400 that says so), not a dropped connection.
        Vision.download_public: neither the URL nor a redirect may reach a private address (#28)."""
        return Vision._fetch(url, False)

    @staticmethod
    def download_public(url: str) -> bytes:
        """#28: a request's image URL, fetched by a running server (Service.public_urls): only from a public address,
        redirects included, so the server is no proxy into its own network (STRATA_IMAGE_URL_PRIVATE=1 allows it)."""
        return Vision._fetch(url, True)

    @staticmethod
    def _fetch(url: str, public: bool) -> bytes:
        too_big = f"the image URL's file is over {IMAGE_URL_MAX >> 20} MiB"
        opener = urllib.request.build_opener(_PublicRedirects) if public else urllib.request.build_opener()
        if public and (addr := private_url_host(url)) is not None:
            raise ValueError(f"the image URL points to a private address ({addr}): the server does not fetch from "
                             "its own network for a request; STRATA_IMAGE_URL_PRIVATE=1 allows it")
        try:
            with opener.open(urllib.request.Request(url, headers={"User-Agent": "strata"}), timeout=60) as r:
                size = r.headers.get("Content-Length") or ""
                if size.isdigit() and int(size) > IMAGE_URL_MAX:
                    raise ValueError(too_big)
                data = r.read(IMAGE_URL_MAX + 1)
        except (OSError, HTTPException) as e:
            raise ValueError(f"the image URL could not be read: {e}") from None
        if len(data) > IMAGE_URL_MAX:
            raise ValueError(too_big)
        return data

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        (stb_image) ignores the tag.  False without Pillow, or when the tag is missing, 1 or unreadable (#1229)."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes, keep_alpha: bool = False) -> bytes:
        """The formats strata-vision's decoder (stb_image) reads pass through; anything else is converted to PNG.
        A JPEG with an EXIF Orientation other than 1 is turned upright first (needs Pillow).  `keep_alpha` (an image
        edit's reference: the image model reads the alpha): a transparent image stays RGBA, else it goes on white."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied (#1229)
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if keep_alpha and (im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info):
            im = im.convert("RGBA")
        elif im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode_all(self, sources: list, cancel=None) -> list[tuple[Path, int]]:
        """encode() for every image of one request: the files of this request's earlier images are never evicted to
        make room for a later one (#1072: with more than 64 unique images the first was deleted before the request
        had read it).  `cancel` (#1999): a request whose client left does not encode its next pictures."""
        out: list[tuple[Path, int]] = []
        for src in sources:
            if cancel is not None and cancel.is_set():
                raise ClientGone()
            out.append(self.encode(src, keep=[p for p, _ in out]))
        return out

    def encode(self, source: str | bytes, keep=()) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens).  `source`: what load() reads, or the image's bytes.
        `keep`: files the caller still needs (never evicted by this call)."""
        data = self.normalize(source if isinstance(source, bytes) else self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self._ensure_dir()
            if key in self.cache:
                self.cache[key] = self.cache.pop(key)                  # most recently used last
                return self.cache[key]
            if self.dead:
                raise ValueError("images are unavailable: " + self.dead)
            if self.proc is None or self.proc.stdin is None:   # an elastic encoder not started yet, or one unloaded
                raise ValueError("the picture reader is not running")
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            try:
                img.write_bytes(data)
            except OSError:                         # a file error of the server's (a full disk): no half-written file
                with contextlib.suppress(OSError):  # left; the request's handler answers it (a 500)
                    img.unlink(missing_ok=True)
                raise
            try:
                if self.proc is None or self.proc.stdin is None:
                    raise OSError("the vision encoder is not running")
                self.proc.stdin.write(f"ENC {img.name} {out.name}\n")      # relative to the encoder's cwd (#480)
                self.proc.stdin.flush()
                line = self._line(VISION_ENCODE_S)
            except OSError:                         # its stdin is closed: it exited
                line = None
            finally:                                # #352: also when the encoder's pipe is gone
                img.unlink(missing_ok=True)
            if line is None or line == "":
                self._stop("the vision encoder stopped" if line is None else
                           "the vision encoder did not answer within %d s" % (VISION_ENCODE_S or 0))
                raise ValueError(f"the image could not be read: {self.dead}")
            line = line.strip()
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else line))
            self.cache[key] = (out, int(line.split()[1]))
            self.warn_if_over_limit(self.cache[key][1])
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):                               # oldest first, but never one still in use
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def warn_if_over_limit(self, tokens: int) -> None:
        """I7: one picture that became more than 4 times the limit is logged with the numbers, so the next report has
        them (the reader's own stderr has the warm-up line).  A reader built before the limit was enforced inside it
        (strata-vision refuses these now) still lets them by."""
        limit = getattr(self, "max_tokens", 0)
        if limit and tokens > 4 * limit:
            print(f"[strata] images: one picture became {tokens} image tokens, more than 4 times the limit of {limit} "
                  "(vision.max_tokens); the picture reader is not applying its limit. Update the engine, and send the "
                  "line 'strata-vision: warmed up at N image tokens' from the reader's log with the report", flush=True)

    def _end(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self._close_stdin()

    def close(self):
        if self.proc is not None:
            self._end()
        self.stopped = True

    def shutdown(self):
        """#914: close() for good: the server ends, so the encoder's directory (the pictures, their embeddings and the
        requests' combined files) goes with it.  unload() keeps it for the restart."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def effort_end_args(cfg: dict, exe: str, tok) -> list[str] | None:
    """#458 (opt-in): the engine arguments for "effort_position": "end" - the id of "system" as --tail-role-token, so
    the engine checkpoints in front of the trailing effort turn - or None when the config leaves it at the top (the
    default) or the engine is too old for it (said once; the prompt stays the default one).  ValueError for a value
    other than "start" / "end"."""
    pos = cfg.get("effort_position", "start")
    if pos not in ("start", "end"):
        raise ValueError(f'"effort_position" must be "start" (the default) or "end", not {pos!r}')
    if pos == "start":
        return None
    try:
        with open(exe, "rb") as f:
            known = b"--tail-role-token" in f.read()
    except OSError:
        known = False
    role = tok.encode("system", parse_special=True)
    if not known or len(role) != 1:
        print("[strata] effort_position \"end\" needs an engine that knows --tail-role-token (build it again): the "
              "reasoning effort stays at the top of the prompt", flush=True)
        return None
    print("[strata] effort_position end: a request's non-default reasoning effort goes right before the answer, so "
          "changing it keeps the cached conversation", flush=True)
    return ["--tail-role-token", str(role[0])]


def engine_silence_s(cfg: dict) -> float:
    """#481: the config's "engine_silence_s" - seconds an engine may print nothing during a request before the server
    ends it (default ENGINE_SILENCE_S; 0 = wait forever).  ValueError for anything but a number >= 0."""
    v = cfg.get("engine_silence_s")
    if v is None:
        return ENGINE_SILENCE_S
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(f'"engine_silence_s" must be a number of seconds >= 0 (0 = no limit), not {v!r}')
    return float(v)


def layer_split_value(cfg: dict) -> str:
    """upstream #644: the config's "layer_split" as the engine's --layer-split value: "auto" (the default), or the
    FIRST LAYER of each later GPU's share - one rising number per GPU after the first, e.g. "24,36,42" for 4 GPUs
    (layers 0-23 on the first card, 24-35, 36-41, 42 to the end) - as a string or a JSON list ([24, 36, 42]).
    ValueError with the format and an example for anything else, such as per-card layer counts ("24,16,12,12").
    The GPUs: "gpu", or the first card and the Xe engine's later ones ("split_pci")."""
    v = cfg.get("layer_split")
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto")):
        return "auto"
    n = len(gpu_list(cfg))
    if n <= 1:
        n = 1 + len(cfg.get("split_pci") or [])
    items = v if isinstance(v, (list, tuple)) else str(v).split(",") if isinstance(v, (str, int)) else None
    vals: list[int] = []
    for x in items or []:
        try:
            if isinstance(x, bool) or isinstance(x, float):
                raise ValueError
            vals.append(int(str(x).strip()))
        except ValueError:
            vals = []
            break
    want = max(n - 1, 1)
    example = ",".join(str(round(48 * (i + 1) / (want + 1))) for i in range(want))
    hint = (f'"layer_split" is the first layer of each later GPU, one rising number per GPU after the first '
            f'({want} for {n} GPUs), not a count of layers per card - e.g. "{example}" '
            f'(or [{example.replace(",", ", ")}]) for an even share of 48 layers, or "auto" (the default) to place '
            'them by each card\'s free VRAM')
    if not vals:
        raise ValueError(f"{hint}; got {v!r}")
    if n > 1 and len(vals) != n - 1:
        raise ValueError(f"{hint}; got {len(vals)} number(s) ({v!r}) for {n} GPUs")
    if vals[0] < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"{hint}; got {v!r}, which does not rise from 2 or more")
    return ",".join(str(x) for x in vals)


def vram_elastic_on(cfg: dict) -> bool:
    """#533: the expert cache in segments it gives back and maps again (--vram-elastic): the config's "vram_elastic",
    else on when something borrows the VRAM between requests - image generation, or a GPU vision encoder.  The engine
    keeps one allocation where it cannot (a runtime without virtual memory, a layer split, the resident low-RAM
    mode)."""
    v = cfg.get("vram_elastic")
    if isinstance(v, bool):
        return v
    vision = cfg.get("vision")
    return isinstance(cfg.get("images"), dict) or (isinstance(vision, dict) and vision.get("gpu") is True)


def vision_elastic_on(cfg: dict) -> bool:
    """Upstream ca41a6b9: a GPU vision encoder started with the first picture (and stopped when none came for a while)
    instead of with the model - the section's "elastic", else on with a GPU encoder and an elastic expert cache."""
    vision = cfg.get("vision")
    if not isinstance(vision, dict):
        return False
    v = vision.get("elastic")
    return v if isinstance(v, bool) else vision.get("gpu") is True and vram_elastic_on(cfg)


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32"; see
    layer_split_value)."""
    args = list(cfg["args"])
    # #1322: a config with a "vision" section but without --vision in its args (written by an older setup run, or edited
    # by hand) advertised images and then refused every picture ("this engine was started without --vision").  The
    # section says images are wanted: start the engine with them, and keep the encoder's VRAM free as setup does for a
    # GPU encoder (only when the config has no reserve of its own).
    if isinstance(cfg.get("vision"), dict) and "--vision" not in args:
        args.append("--vision")
        note = "added --vision"
        if cfg["vision"].get("gpu") and "--vram-reserve-mib" not in args:
            args += ["--vram-reserve-mib", "700"]
            note += " and --vram-reserve-mib 700"
        print(f'[strata] the config has a "vision" section but no --vision in its args: {note} (run setup again with '
              "--vision to write them)", flush=True)
    if len(gpu_list(cfg)) > 1 and "--layer-split" not in args:
        args += ["--layer-split", layer_split_value(cfg)]
    # the Xe engine's layer split (setup --gpus): the later GPUs by PCI address; STRATA_LAYER_SPLIT=off (setup --gpu N
    # for one start) runs on the first card alone
    if cfg.get("split_pci") and "--layer-split" not in args and os.environ.get("STRATA_LAYER_SPLIT") != "off":
        args += ["--layer-split", layer_split_value(cfg), "--split-device", ",".join(cfg["split_pci"])]
        # "gpu_order": "as_given": the auto split keeps the cards in the config's order (upstream be8db787)
        if cfg.get("gpu_order") == "as_given" and "--split-order" not in args:
            args += ["--split-order", "as-given"]
        # "split_skip_if_fits": true (opt-in, upstream 18ad40c0d): the first card alone when it holds every profiled
        # expert with the KV
        if cfg.get("split_skip_if_fits") and "--split-skip-if-fits" not in args:
            args.append("--split-skip-if-fits")
        # the image jobs on a later card of the text model's (image_helper_card): that card is a helper expert cache
        # or none, never a stage - the server closes a helper for each image job and opens it again after
        if image_helper_card(cfg) and "--split-roles" not in args:
            args += ["--split-roles", "helper"]
    if "coupled_draft" in cfg and "--coupled-draft" not in args and "--no-coupled-draft" not in args:
        args += ["--coupled-draft" if cfg["coupled_draft"] else "--no-coupled-draft"]
    # #533: the expert cache in segments (vram_elastic_on), so an image job, a vision encoder that starts with its
    # picture and POST /v1/vram can take VRAM and give it back; "vram_segment_mib" sets the segment size (the engine's
    # default: 512)
    if vram_elastic_on(cfg) and "--vram-elastic" not in args:
        args.append("--vram-elastic")
        seg = cfg.get("vram_segment_mib")
        if isinstance(seg, int) and not isinstance(seg, bool) and seg > 0 and "--vram-segment-mib" not in args:
            args += ["--vram-segment-mib", str(seg)]
    # upstream 73ed39ae: a conversation save that its RAM check refuses (an image job's unload, when the RAM is short)
    # first lets the caches it can build again go; "session_save_reclaim": false keeps the refusal
    if cfg.get("session_save_reclaim") is not False and "--session-save-reclaim" not in args:
        args.append("--session-save-reclaim")
    args += parallel_args(cfg, args)
    args += pool_cores_args(cfg, args)
    return conversation_save_args(cfg, learned_profile_args(cfg, args))


PARALLEL_MAX = 8      # the engine's batch window holds at most 8 rows (kVerifyMaxT); --batch-mtp waves more through it


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    """#465 (opt-in): "parallel": N in the config - up to N requests decode together in the engine's batch slots
    (--batch N); more wait their turn.  1 or absent: one request at a time, as before.  A value the engine cannot use
    is said and passed on: the engine warns and adjusts (recommend, never force)."""
    if "--batch" in args or "--slots" in args:
        return []
    n = cfg.get("parallel")
    if n is None or n is False:
        return []
    if isinstance(n, bool) or not isinstance(n, int):
        print(f'[strata] "parallel" must be a whole number of requests (2..{PARALLEL_MAX}), not {n!r}: ignored',
              flush=True)
        return []
    if n <= 1:
        return []
    # --batch-mtp is on by default with --mtp (the engine's --no-batch-mtp / STRATA_BATCH_MTP=0 turn it off)
    env = os.environ.get("STRATA_BATCH_MTP", "")
    batch_mtp = "--batch-mtp" in args or env not in ("", "0") or (
        "--mtp" in args and "--no-batch-mtp" not in args and env != "0")
    if n > PARALLEL_MAX and not batch_mtp:
        print(f'[strata] "parallel": {n} - the engine runs at most {PARALLEL_MAX} requests together; it will use '
              f"{PARALLEL_MAX}", flush=True)
    return ["--batch", str(n)]


def conversation_save_args(cfg: dict, args: list[str]) -> list[str]:
    """"conversation_save": "<folder>" in the config (setup writes one for each model, setup.py's
    conversation_save_default) has the engine keep the conversations its RAM cache lets go there, and across
    restarts, with "conversation_save_checkpoints", "conversation_save_mib" and "conversation_save_hours" (whole
    numbers >= 0) for the engine's defaults, and "conversation_save_compress" (true: an engine built with libblosc2
    compresses the files).  A relative path is the engine's (the config's "cwd").  Without the key, or with false,
    the arguments are the config's, unchanged."""
    folder = cfg.get("conversation_save")
    if not isinstance(folder, str) or not folder.strip() or "--conversation-save" in args:
        return args
    args = args + ["--conversation-save", folder]
    compress = cfg.get("conversation_save_compress")
    if compress is not None and not isinstance(compress, bool):
        raise ValueError(f'"conversation_save_compress" must be true or false, not {compress!r}')
    if compress and "--conversation-save-compress" not in args:
        args += ["--conversation-save-compress"]
    for key in ("conversation_save_checkpoints", "conversation_save_mib", "conversation_save_hours"):
        v = cfg.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f'"{key}" must be a whole number >= 0, not {v!r}')
        flag = "--" + key.replace("_", "-")
        if flag not in args:
            args += [flag, str(v)]
    return args


_POOL_CORES_OK = re.compile(r"^[0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+(:[0-9]+)?)?)*$")


def pool_cores_args(cfg: dict, args: list[str]) -> list[str]:
    """Experimental, off by default (upstream 2aa1ec2d5): "pool_cores": "0,2,4,6" (or "2-14:2", or a list of numbers)
    in the config pins the CPU expert workers to exactly those logical processors (--pool-cores; docs/DETAILS.md, "Which
    logical processors the CPU workers use").  The engine warns about two entries on one core or the host's own and
    leaves out processors the process may not use; a text this cannot read is said here and ignored (recommend, never
    force).  An engine option already in the args wins."""
    value = cfg.get("pool_cores")
    if value is None or value is False or value == "" or value == []:
        return []
    if "--pool-cores" in args:
        return []
    if isinstance(value, list) and value and \
            all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value):
        text = ",".join(str(v) for v in value)
    elif isinstance(value, str):
        text = value.replace(" ", "")
    else:
        text = ""
    if not _POOL_CORES_OK.match(text):
        print(f'[strata] "pool_cores" must be logical processor numbers like "0,2,4,6" or "2-14:2", not {value!r}: '
              "ignored (the CPU workers keep their default placement)", flush=True)
        return []
    return ["--pool-cores", text]


def profile_shape(path: str) -> tuple[int, int] | None:
    """An expert profile's (layers, experts per layer), from its header (tools/make_profile.py's format), or None
    when the file is missing, is not one or is shorter than the pairs its header promises."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"STRP":
        return None
    _, nl, ne, _, n = struct.unpack("<5I", head[4:])
    return (nl, ne) if size >= 24 + 4 * n else None


def learned_profile_args(cfg: dict, args: list[str]) -> list[str]:
    """#477: "expert_profile_save": "<path>" in the config has the engine save what its adaptive tier learned there -
    on QUIT and every "expert_profile_save_every" minutes (10 by default, 0 = at QUIT only) - and the next start
    begins from it instead of the config's --expert-profile, when it is a profile of the same model (its header's
    layers and experts match); otherwise from the config's own, as before.  A relative path is the engine's (the
    config's "cwd").  Without the key, expert-profile-<model>.bin there, for a config with an --expert-profile;
    "expert_profile_save": false keeps the config's arguments as they are."""
    save = cfg.get("expert_profile_save")
    if save is None and "--expert-profile" in args[:-1]:
        save = "expert-profile-" + re.sub(r"[^\w.-]", "_", str(cfg.get("model_name") or "model")) + ".bin"
    if not isinstance(save, str) or not save.strip() or "--expert-profile-save" in args:
        return args
    args = args + ["--expert-profile-save", save]
    every = cfg.get("expert_profile_save_every")
    if isinstance(every, (int, float)) and not isinstance(every, bool) and every >= 0:
        args += ["--expert-profile-save-every", str(every)]
    if "--expert-profile" in args[:-1]:
        i = args.index("--expert-profile") + 1
        here = cfg.get("cwd") or "."
        learned = profile_shape(save if os.path.isabs(save) else os.path.join(here, save))
        base = profile_shape(args[i] if os.path.isabs(args[i]) else os.path.join(here, args[i]))
        if learned is not None and learned == base:
            args[i] = save
    return args


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's own default order (fastest first) can number
    # the cards otherwise
    if gpu_list(cfg):
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries
        env[str(k)] = str(v)
    args = cfg.get("args")
    if isinstance(args, list) and "--ram-reserve-mib" in args[:-1]:   # setup's --ram-reserve-gib: the image engine too
        env["STRATA_RAM_RESERVE_MIB"] = str(args[args.index("--ram-reserve-mib") + 1])
    if cfg.get("gpu_pci"):                           # the Xe engine: the GPU setup chose, by PCI address
        env.setdefault("STRATA_GPU_PCI", str(cfg["gpu_pci"]))   # a --gpu for this start (setup sets it) wins
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"   # Windows finds a program's DLLs on PATH
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


SELECT_ENV = ("ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK", "HIP_VISIBLE_DEVICES")   # each narrows the GPUs listed


def image_helper_card(cfg: dict) -> str:
    """The "images" section's "gpu_pci" when it is one of the text engine's later cards ("split_pci", two cards in
    all) and no other text engine of the server takes it: the engine then gives that card a helper expert cache's role
    or none (--split-roles helper), and each image job runs there beside the chat while the helper is closed (images'
    separate_turn).  On a B70 + RTX 4070 a split's later stage shrunk for an image read prompts at half the speed of
    the first card alone (801 against 1,592 tok/s at 4K), and a helper closes in 29 ms and fills again in 1.9 s.
    "" otherwise."""
    images = cfg.get("images")
    want = str((images or {}).get("gpu_pci") or "").lower() if isinstance(images, dict) else ""
    later = [str(p).lower() for p in cfg.get("split_pci") or []]
    if not want or len(later) != 1 or want != later[0] or cfg.get("replicas") or cfg.get("split_serve"):
        return ""
    if os.environ.get("STRATA_LAYER_SPLIT") == "off" or cfg.get("layer_split") not in (None, "", "auto"):
        return ""
    return want


def gpu_files_missing(pci: str) -> str:
    """The card at this PCI address when its device files are not here (setup's gpu_files_missing): a container is
    given the host's config, whose image GPU can be a card the container was not passed.  "" when they are, or the
    card is not one sysfs knows (Windows)."""
    dev = Path("/sys/bus/pci/devices") / pci
    try:
        vendor = (dev / "vendor").read_text().strip()
    except OSError:
        return ""
    if vendor == "0x10de":
        need = ["/dev/nvidiactl"]
    else:
        need = [f"/dev/dri/{r.name}" for r in sorted((dev / "drm").glob("renderD*"))][:1] or ["a render node"]
        need += ["/dev/kfd"] if vendor == "0x1002" else []
    gone = [p for p in need if not Path(p).exists()]
    return "its device files are not here: " + ", ".join(gone) if gone else ""


def image_env(cfg: dict, env: dict) -> tuple[dict, bool, str]:
    """The image engine's environment, from the text engine's (`env`, child_env) -> (it, whether the image jobs have a
    card of their own (images_api.ImageModels' separate), the line the log says of it).  The "images" section's
    "gpu_pci" (setup --image-gpu) is that card unless a text engine of the server takes it: the server's card, a layer
    split's, the replicas' and a split serve's.  A helper GPU (--expert-cache-device1..3, by device number), an
    environment that narrows the GPUs the runtime lists, or a card whose device files are not here (gpu_files_missing)
    leaves the jobs on the text engine's card."""
    want = str((cfg.get("images") or {}).get("gpu_pci") or "").lower()
    if not want:
        return env, False, ""
    text = {str(env.get("STRATA_GPU_PCI") or cfg.get("gpu_pci") or "").lower()}
    text |= {str(p).lower() for p in cfg.get("split_pci") or []}
    reps = cfg.get("replicas")
    for r in reps if isinstance(reps, list) else []:
        text |= {str(p).lower() for p in ((r.get("gpus") if isinstance(r, dict) else r) or [])}
    ss = cfg.get("split_serve")
    if isinstance(ss, dict):
        text |= {str(ss.get(k) or "").lower() for k in ("prefill_gpu", "decode_gpu")}
    text.discard("")
    if want == image_helper_card(cfg):            # the text engine's helper card, handed over for each image job
        text.discard(want)
    args = [str(a) for a in cfg.get("args") or []]
    why = ("a helper GPU (--expert-cache-device1..3) is named by its device number" if any(
        a.startswith("--expert-cache-device") for a in args) else
           f"{next(k for k in SELECT_ENV if env.get(k))} narrows the GPUs" if any(env.get(k) for k in SELECT_ENV)
           else "the text model takes it" if want in text else
           "no text card is named (the engine takes the first)" if not text else gpu_files_missing(want))
    if why:
        return env, False, f"image GPU: {want} is not used ({why}); the image jobs share the text model's card"
    out = dict(env, STRATA_GPU_PCI=want)
    if cfg.get("gpu") not in (None, ""):          # CUDA_VISIBLE_DEVICES of the text card alone would hide this one
        out.pop("CUDA_VISIBLE_DEVICES", None)
    if want == image_helper_card(cfg):
        return out, True, (f"image GPU: {want}, the text model's second card: a helper expert cache there is closed "
                           "for each image job and opened again after it")
    return out, True, f"image GPU: {want} (separate from the text model's)"


def gpu_roles(cfg: dict, gpu_pci: str | None, groups: list | None = None, split_serve: dict | None = None,
              image_pci: str | None = None, helper_card: str = "") -> list[dict]:
    """The cards the server's engines run on and what each does, the Monitor's tabs (serve/telemetry.py's `gpus`):
    [{"pci", "roles", "stage", "replica"}], the text engine's cards first.  "roles": "text" (the text engine; with
    "prefill" or "decode" under split serving), "helper" (a layer split's later card holding a helper expert cache,
    image_helper_card), "image" (the image engine).  "stage": a layer split's card number, "replica": the replica's.
    `gpu_pci`: the text engine's first card (None: the one the engine takes); `groups`: the replicas' cards;
    `split_serve`: split serving's two; `image_pci`: the image engine's card of its own (image_env's), None when the
    jobs share the text model's; `helper_card`: image_helper_card's."""
    cards: list[dict] = []

    def add(pci, role, **more):
        key = str(pci).lower() if pci else None
        card = next((c for c in cards if (c["pci"] or "").lower() == (key or "")), None)
        if card is None:
            card = {"pci": pci or None, "roles": []}
            cards.append(card)
        if role not in card["roles"]:
            card["roles"].append(role)
        card.update({k: v for k, v in more.items() if v is not None})

    if groups:
        for r, g in enumerate(groups):
            for st, p in enumerate(g):
                add(p, "text", replica=r, stage=st if len(g) > 1 else None)
    elif split_serve:
        add(split_serve.get("prefill_gpu"), "text")
        add(split_serve.get("prefill_gpu"), "prefill")
        add(split_serve.get("decode_gpu"), "text")
        add(split_serve.get("decode_gpu"), "decode")
    else:
        later = [] if os.environ.get("STRATA_LAYER_SPLIT") == "off" else list(cfg.get("split_pci") or [])
        stages = [p for p in later if not (helper_card and str(p).lower() == helper_card)]
        add(gpu_pci, "text", stage=0 if stages else None)
        for p in later:
            if p in stages:
                add(p, "text", stage=stages.index(p) + 1)
            else:
                add(p, "helper")
    if cfg.get("images"):
        add(helper_card or image_pci or cards[0]["pci"], "image")
    return cards


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    ALWAYS: tuple[str, ...] = ()                    # specials matched without parse_special (type 4, as <think>)

    @property
    def max_special_len(self):
        return max(len(s) for s in self.SPECIALS)

    @property
    def control_tokens(self):
        return [s for s in self.SPECIALS if s not in self.ALWAYS]

    def encode(self, text, parse_special=False, plain=()):
        return self.encode_marked(text, parse_special, plain)[0]

    def encode_marked(self, text, parse_special=False, plain=()):
        """encode() and its resume points (after each special), as strata_tokenizer's for PromptEncoder."""
        out, marks, i = [], [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(
                        a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    marks.append((i, len(out)))
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out, marks

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


def model_switcher(svc, mode: str = "") -> dict:
    """The host's model swapper (config `model_switcher`, an RPC taking {"mode": m}): the models it serves on THIS
    port, which one holds the card, whether one is loading. With `mode`, asks for that one first; this server is
    then usually the one stopped, so the web app waits for /health to name the new model."""
    url = getattr(svc, "switcher", None)
    if not url:
        return {"enabled": False}
    try:
        body = json.dumps({"mode": mode} if mode else {}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
        st = res.get("result", res)
    except (OSError, ValueError) as e:
        return {"enabled": True, "error": f"the model switcher did not answer ({e})"}
    port = f":{getattr(svc, 'port', '')}"
    urls = st.get("urls") or {}
    mine = {k: v for k, v in (st.get("choices") or {}).items() if not urls or str(urls.get(k, "")).endswith(port)}
    return {"enabled": True, "mode": st.get("mode"), "up": st.get("up"), "starting": st.get("starting"),
            "choices": mine, "model": svc.model}


OPENAI_MAX_STOP = 4                                  # OpenAI's limit on `stop`


def stop_strings(req: dict) -> list[str]:
    """The request's stop strings: OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences`
    (a list).  Empty strings are left out.  A bad value is a ValueError, so a 400 before anything is generated."""
    req = req or {}
    if req.get("stop") is not None:
        stop = req["stop"]
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > OPENAI_MAX_STOP:
            raise ValueError(f"stop takes at most {OPENAI_MAX_STOP} strings")
    elif req.get("stop_sequences") is not None:
        stop = req["stop_sequences"]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop_sequences must be a list of strings")
    else:
        return []
    return [x for x in stop if x]


class StopMatcher:
    """Cuts the answer's text at the first stop string.  Text that could still be the start of a stop string is held
    back (only that tail, never more), so a stop string split across tokens or chunks is still found and never sent."""

    def __init__(self, stops: list[str]):
        self.stops, self.held = stops, ""
        self.hit: str | None = None

    def push(self, text: str) -> str:
        """-> the text that is safe to send now; after a match, self.hit is the stop string and nothing more comes."""
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0                                      # the longest end of buf that a stop string starts with
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held


# Windows names that open a device whatever their extension or folder (Microsoft, "Naming Files, Paths, and Namespaces")
_WIN_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in "123456789\u00b9\u00b2\u00b3"),
                *(f"lpt{i}" for i in "123456789\u00b9\u00b2\u00b3")}


def slot_filename_problem(name) -> str | None:
    """/slots/0 {"filename": NAME}: why NAME is not one plain file name inside the slot save path, on Linux and on
    Windows alike (None when it is).  No separator or drive (`D:x`), no stream (`a:b`), no device (`NUL.txt`), no
    trailing dot or space (Windows drops them), no control character, no leading dot (temporary files use it)."""
    if not isinstance(name, str) or not name:
        return "must be a non-empty string"
    if len(name.encode("utf-8", "surrogatepass")) > 200:
        return "is too long (at most 200 bytes)"
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return "must not contain control characters"
    if any(c in name for c in '/\\:*?"<>|'):
        return "must be a plain file name (no / \\ : * ? \" < > |)"
    if name.startswith("."):
        return "must not start with a dot"
    if name.endswith((".", " ")):
        return "must not end with a dot or a space"
    if name.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES:
        return "names a Windows device"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid Unicode"
    return None


def slot_save_dir(value, base: str | None = None) -> str:
    """The slot save path as one absolute, normalized directory (relative to `base`, else the server's working
    directory), created private to this user when missing.  ValueError for anything else."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slot_save_path must be a directory path (a string)")
    if any(ord(c) < 32 for c in value):
        raise ValueError("slot_save_path must not contain control characters")
    path = os.path.abspath(os.path.join(base, value) if base and not os.path.isabs(value) else value)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path):
        raise ValueError(f"slot_save_path {path} is not a directory")
    return path


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = True):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.vision_error: str | None = None           # why the configured encoder did not start (images off)
        self.vision_fallback: str | None = None        # why a later encoder serves instead of the configured one
        self.literals = literal_tags(getattr(tokenizer, "control_tokens", ()))   # texts that stay text inside a message
        # default: clamp the output cap to the room left; False (--no-fit-max-tokens) = 400
        self.fit_max_tokens = fit_max_tokens
        self.aliases: list[str] = []                  # #297: other names of the model (the config's `aliases`)
        self.answer_room = True                       # 0.1.45: see ANSWER_ROOM_MIN_CAP ("answer_room": false: off)
        self.reasoning_close_retry = False            # #1053 (opt-in): close the thinking once when a reply ends in it
        self.codex_compaction_cache = False           # #924 (opt-in): a Codex compaction rendered with its own tools
        self.codex_thread_titles = False              # #923 (opt-in): Codex's thread-title turns without the engine
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.shared: dict = {}                        # the web app's Chat settings for every client (POST /settings)
        self.shared_path: str | None = None           # where they are kept between starts (next to the config)
        # #1641: the config's "reasoning_effort": the level for requests with none
        self.config_effort = None
        # the model card whose sampling a request with none gets ("qwen4exp", "qwen35moe"): None = greedy as asked
        self.model_arch: str | None = None
        self.fifo = threading.Lock()
        self.vram_reserve: int | None = None           # #533: the last POST /v1/vram reserve (None: the start's)
        self.slot_save_path: str | None = None        # --slot-save-path: /slots/0?action=save|restore (off when None)
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        # #458 (opt-in, the config's "effort_position": "end"): a non-default reasoning effort goes in a short system
        # turn before the answer instead of the top of the prompt, so switching it keeps the cached conversation
        self.effort_end = False
        self.config_path = None                         # #564: the run config the web page's Settings view edits
        self.config_lock = threading.Lock()
        self.settings_path = places.settings_path()     # the user's settings: GLOBAL keys for every model (runconfig)
        self.search_url: str | None = None              # web search: a SearXNG-compatible server's address
        # the web search's engine (search.py)
        self.search_backend: search_api.Searxng | search_api.DeepSeek | search_api.TinyFish | None = None
        self.switcher: str | None = None                # the host's model swapper RPC (config model_switcher)
        self.port = 0                                   # the port this server listens on (the switcher's models)
        # #321: browser pages of these origins may call /v1/* (CORS; "*" = any page - only with an api_key that
        # matters); empty = no CORS headers at all, as before
        self.cors_origins: list[str] = []
        self.conv_log = ConvCacheLog()                  # #596: the parked conversations, from the engine's log
        # #321: origins that count as Strata's own page for /settings and MCP tools, e.g. the web app reached through a
        # reverse proxy or tunnel whose Host differs ("https://strata.example.com"); never a wildcard
        self.trusted_origins: list[str] = []
        # DNS rebinding: extra Host names this server answers to (the config's allowed_hosts, $STRATA_ALLOWED_HOSTS;
        # "*" = any), and every name it answers to, which serve() works out from the address it listens on
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status: dict = {"busy": False, "queued": 0}   # GET /status: what the model is doing right now
        # (time, generated) samples for the live tok/s window
        self.rate: collections.deque = collections.deque(maxlen=32)
        # "parallel" (the engine's batch slots, #465): every running request's own status and rate window; self.status
        # then says busy while any runs and shows the newest one
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        # #332: the API request monitor (/api-monitor) keeps the last 100 requests' prompts and answers in memory,
        # so it is off unless the config's "api_monitor" (or --api-monitor) turns it on
        self.api_monitor = False
        # bounded I/O in memory; no headers or API keys
        self.api_requests: collections.deque[dict] = collections.deque(maxlen=100)
        self.request_trace = threading.local()
        self.debug_log: debuglog.DebugLog | None = None   # the debug chat log (serve/debuglog.py), when on
        # the last finished requests, newest last (GET /metrics)
        self.history: collections.deque[dict] = collections.deque(maxlen=500)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0,
                       "drafts_offered": 0, "drafts_accepted": 0}   # #457: the MTP drafts, summed where reported
        self.last_timings: dict | None = None           # the last finished request's, llama.cpp's names (/v1/status)
        # /metrics/prometheus' two throughput gauges need the totals as they stood at the last scrape, because
        # llama.cpp averages them over the window between two scrapes (#2341).
        self.scrape_bucket: tuple[float, float, float, float] = (0, 0.0, 0, 0.0)
        from serve.prometheus import Latencies
        self.latencies = Latencies()                     # GET /metrics in Prometheus text: the latency histograms
        self.last_request_at: float | None = None        # when a request last started or finished
        self.started_at = time.time()
        self.engine_failure: dict | None = None   # issue #10: the last engine that died in a request, for the web app
        self.gpu_pci: str | None = None      # the engine's card (the Monitor reads it)
        self.split_pci: list | None = None   # a layer split's other cards (the Monitor reads them too)
        self.gpu_roles: list | None = None   # every engine's cards and their roles (gpu_roles; the Monitor's tabs)
        self.updates = release_check.Checker()           # GET/POST /update: whether a newer release is out
        self.status_lock = threading.Lock()
        self.mcp: McpHub | None = None                   # serve/mcp.py's McpHub when MCP servers are configured
        # sharing the GPU with other programs (all off by default): unload the engine after this many idle seconds,
        # only start it again when this much VRAM is free, and run this command first (e.g. to unload another
        # server's model); the next request after an unload starts the engine again
        self.idle_unload_s: float = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.before_load_timeout_s = 120.0               # #1997: how long the before_load command may run
        self.before_load_ignore_failure = False          # #1997: load even when it fails, without a VRAM check
        self.vram_free_wait_s = 15.0                     # how long a full card is waited on (memory given back)
        self.before_load_strict = False                  # #1997: refuse (503) whenever it fails
        self.vram_reserve_asked = False                  # a POST /v1/vram was made (vram_reserve None is then asked)
        self.vision_held = False                         # an elastic encoder holds VRAM the expert cache gave it
        self.last_image_at = 0.0                         # when an image turn last ended (the elastic encoder's idle)
        self.reasoning_loop_recovery = False
        # the loop guard (serve/loop_guard.py; STRATA_LOOP_GUARD=0 / "loop_guard": false: off) and what it did since
        # the start (the JSON /metrics and its Prometheus form)
        self.loop_guard = True
        self.loop_guard_counts = {"detected": 0, "penalised": 0, "thinking_closed": 0, "stopped": 0}
        # the script guard (serve/script_guard.py): the server's default ("script_guard" / STRATA_SCRIPT_GUARD; None:
        # off) and what it did since the start
        self.script_guard: dict | str | None = None
        self.script_guard_counts = {"requests": 0, "rewinds": 0, "passed": 0}
        self.script_guard_unsupported = False            # said once: this engine takes no logit_bias
        self.reasoning_budget_tokens = 0                 # #123: the config's default thinking budget (0: none)
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply
        self.anthropic_think_unasked = False              # #278: "anthropic_thinking": "model" -> True
        # on unless the config's "tool_call_recovery" is false: tool calls in forms next to the template's
        self.tool_call_recovery = True
        self.images: Any = None                           # serve/images.py's ImageModels when the config has "images"
        self.restore_after_load = None                    # a session file an image job saved before it unloaded
        self.stop_ids = stop_token_ids(tokenizer)
        if hasattr(engine, "eos_ids"):
            engine.eos_ids = self.stop_ids
        # #567: a prompt re-encodes only what follows the last special token it shares with a recent prompt (the
        # same ids as a full encode: tools/strata_tokenizer.py PromptEncoder).  Tokenizers without resume points
        # encode in full.
        self.prompts = None
        if hasattr(tokenizer, "encode_marked") and hasattr(tokenizer, "max_special_len"):
            from strata_tokenizer import PromptEncoder
            # one kept prompt per conversation the engine's cache can hold (--conversation-cache-slots); the
            # encoder's own default without that setting
            slots = conversation_slots(engine)
            self.prompts = PromptEncoder(tokenizer, keep=slots) if slots is not None and slots > 0 \
                else PromptEncoder(tokenizer)

    # #28: the running server (main) sets it: a request's image URLs are fetched only from public addresses
    public_urls = False

    def download(self, url: str) -> bytes:
        """A request's image URL (the chat's pictures, a tool's, an edit's references): Vision.download_public on a
        running server, else Vision.download."""
        return Vision.download_public(url) if self.public_urls else Vision.download(url)

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def slot_action(self, slot: str, action: str, filename) -> tuple[int, dict]:
        """llama-server's POST /slots/{id}?action=save|restore {"filename": ...}: the conversation the engine holds,
        to or from a file in slot_save_path.  -> (HTTP status, body)."""
        def error(code, message):
            return code, {"error": {"code": code, "message": message,
                                    "type": "invalid_request_error" if code < 500 else "server_error"}}
        if not self.slot_save_path or not hasattr(self.engine, "session_file"):
            return error(501, "slot save/restore is disabled (start the server with --slot-save-path DIR)")
        if getattr(self.engine, "batch", 0):
            # #465 parallel requests do not hold the FIFO and share the engine's control lines: a session file
            # could interleave with a request being admitted, so the two are not combined (the engine refuses too)
            return error(501, "slot save/restore is not available with parallel requests (\"parallel\" / --batch)")
        if slot != "0":
            return error(400, "this server has one slot: 0")
        if action not in ("save", "restore"):
            return error(400, "action must be save or restore")
        why = slot_filename_problem(filename)
        if why:
            return error(400, f"filename {why}")
        path = os.path.join(self.slot_save_path, filename)
        if action == "restore":
            try:
                os.lstat(path)                              # the engine opens it without following a link
            except OSError:
                return error(404, f"no saved session named {filename}")
        self.last_request_at = time.time()                  # the idle unload counts this as activity
        with self.status_lock:
            self.status["queued"] += 1
        waiting = True
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                waiting = False
                if not self.loaded():
                    return error(503, "the model is not loaded")
                with self.status_lock:
                    self.status.update(busy=True, phase="saving the session" if action == "save"
                                       else "restoring a session", started=time.time(), first_token=None,
                                       prompt_tokens=None, generated=None, max_tokens=None)
                try:
                    if action == 'save' and getattr(self.engine, 'session_save_reclaim', False):
                        from serve.session_save_retry import session_file
                        r = session_file(self, action, path, SessionRefused)
                    else:
                        r = self.engine.session_file(action, path)
                except SessionRefused as e:
                    body = error(e.status, str(e))
                    body[1]["error"]["kind"] = e.kind
                    if e.published:
                        body[1]["error"]["published"] = True
                    return body
                except ValueError as e:                     # the request itself (an invalid command)
                    return error(400, str(e))
                finally:
                    with self.status_lock:
                        self.status["busy"] = False
                    self.last_request_at = time.time()
        finally:
            if waiting:                                     # never reached the FIFO (an exception while waiting)
                with self.status_lock:
                    self.status["queued"] -= 1
        if action == "save":
            return 200, {"id_slot": 0, "filename": filename, "n_saved": r["tokens"], "n_written": r["bytes"],
                         "timings": {"save_ms": r["ms"]}}
        return 200, {"id_slot": 0, "filename": filename, "n_restored": r["tokens"], "n_read": r["bytes"],
                     "timings": {"restore_ms": r["ms"]}}

    def set_aliases(self, aliases) -> None:
        """#297: the config's `aliases` - a list of names (or one comma-separated string), like llama-server's --alias.
        ValueError for anything else."""
        if aliases is None:
            aliases = []
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"aliases={aliases!r}: expected a list of model names")
        names = []
        for x in (x.strip() for x in aliases):
            if x and x != self.model and x not in names:
                names.append(x)
        self.aliases = names

    def model_names(self) -> list[str]:
        return [self.model, *self.aliases]

    def model_for(self, req) -> str:
        """The name to answer with: the request's own when it is the model's name or an alias (#297), else the model's.
        Other names are still served, as before."""
        asked = req.get("model") if isinstance(req, dict) else None
        return asked if isinstance(asked, str) and asked in self.aliases else self.model

    def reported_ctx(self) -> int:
        """The context the endpoints report (/v1/status, /v1/models, /props, /health, /slots, /metrics): the engine's,
        or the last one it reported while it restarts (max_context is 0 until READY, and a client that sizes its
        prompt from it would send nothing, #351).  Reporting only: requests are checked in prepare()."""
        return int(getattr(self.engine, "max_context", 0) or getattr(self.engine, "known_ctx", 0) or 0)

    tokenizer_dir: str | None = None     # the model's tokenizer folder (main): the engine's grammars read it

    def attach_grammar(self, req: dict, grammar) -> bool:
        """`grammar` (structured.answer_grammar / tool_grammar) into the request, for an engine that constrains its
        decoding (INFO grammar=1: not with continuous batching).  STRATA_GRAMMAR=0 keeps
        every reply unconstrained (checked after it, as before).  -> whether it was attached."""
        info = getattr(self.engine, "info", None) or {}
        if grammar is None or str(info.get("grammar")) != "1" or not self.tokenizer_dir or \
                os.environ.get("STRATA_GRAMMAR") == "0":
            return False
        req["strata_grammar"] = {**grammar, "vocab": self.tokenizer_dir, "prefix": ""}
        return True

    def check_strict(self, req: dict, tools, strict: set) -> None:
        """The request's strict tools (`strict`: constrained_tools' names) have their calls checked against their
        parameters before they are sent (structured.checked_calls, in run), whether or not a grammar wrote them."""
        schemas = strict_schemas(tools, strict)
        if schemas:
            req["strata_strict_tools"] = schemas

    def reasoning_budget(self, req) -> int | None:
        """#123: the most tokens this request may think, or None: the request's `reasoning_budget_tokens` (or, as
        llama.cpp reads it, `thinking_budget_tokens`), else the config's.  0 (or less) means no budget, so a request can
        turn a configured one off.  ValueError (a 400) for anything that is not a whole number."""
        req = req if isinstance(req, dict) else {}
        value = req.get("reasoning_budget_tokens")
        if value is None:                                # a null does not hide the other name
            value = req.get("thinking_budget_tokens")
        if value is None:
            value = self.reasoning_budget_tokens
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens={value!r}: expected a whole number of tokens (0: no budget)")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        """The encoder should run and does not (the engine's load starts it again).  An elastic encoder is down
        between pictures by design: it starts in the image turn (_vision_up), not with the engine."""
        return (self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()
                and not getattr(self.vision, "elastic", False))

    def _engine_reserve(self) -> int | None:
        """The VRAM the engine keeps free for other programs outside an elastic encoder's turn: the last POST /v1/vram
        reserve, or 0 when none was asked (the cache grows back to its full size; None would leave the reserve the
        engine started with, a segment or more below the size it had)."""
        return self.vram_reserve if self.vram_reserve_asked else 0

    def _vision_up(self):
        """The image turn's start for an elastic encoder that is not running (the caller holds self.fifo): the expert
        cache first keeps the encoder's elastic_mib of VRAM free on top of the reserve, then the encoder starts (load
        and warm-up, a few seconds).  If the cache cannot shrink (an engine without --vram-elastic, the resident low-RAM
        mode) the encoder starts anyway, as a non-elastic one would."""
        v = self.vision
        if not getattr(v, "elastic", False) or v.alive():
            return
        if str((getattr(self.engine, "info", None) or {}).get("vram_elastic")) in ("0", "False") and self.loaded():
            # the engine cannot give VRAM back (a runtime that keeps an unmapped segment's VRAM: ROCm 7.x's HIP): the
            # encoder starts before the model from now on, as a non-elastic one does, the model loaded again once
            print("[strata] images: the expert cache cannot give VRAM back on this GPU's runtime, so the image "
                  "encoder starts before the model from now on (the model loads again once)", flush=True)
            v.elastic = False
            if getattr(self.engine, "kept_arena", None) is not None:
                self.engine.unload(keep_arena=True)
            else:
                self.engine.unload()
            self.ensure_loaded()
            return
        if hasattr(self.engine, "vram") and self.loaded():
            try:
                out = self.engine.vram((self._engine_reserve() or 0) + v.elastic_mib)
                self.vision_held = True
                print(f"[strata] images: the expert cache keeps {v.elastic_mib} MiB free for the image encoder "
                      f"({out.get('expert_cache_mib')} of {out.get('expert_cache_full_mib')} MiB left)", flush=True)
            except ValueError as e:
                print(f"[strata] images: the expert cache did not make room for the image encoder ({e}); starting "
                      "it anyway", flush=True)
        if self.vision_held:
            time.sleep(0.2)          # the driver's free-VRAM counter shows the cache's release a moment later
        free0, t0 = self.free_vram_mib(), time.time()
        try:
            v.restart()
        except Exception as e:      # a 503 as for a non-elastic encoder at a load, not a dropped connection (#2435)
            raise EngineDied(f"the image encoder failed to start: {e}") from e
        free1 = self.free_vram_mib()
        took = f", {free0 - free1} MiB of VRAM" if free0 is not None and free1 is not None else ""
        print(f"[strata] images: the image encoder started in {time.time() - t0:.1f} s{took}", flush=True)

    def _vision_idle_down(self, force: bool = False) -> str:
        """An elastic encoder idle for elastic_idle_s (or force) stops, and the expert cache takes its VRAM back:
        "down", "busy" (a request holds the turn, or a picture came more recently) or "nothing" (not elastic or not
        holding anything)."""
        v = self.vision
        if v is None or not getattr(v, "elastic", False) or not (self.vision_held or v.alive()):
            return "nothing"
        if not force and time.time() - self.last_image_at < v.elastic_idle_s:
            return "busy"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            v.unload()
            back = ""
            if self.vision_held and hasattr(self.engine, "vram") and self.loaded():
                out = self.engine.vram(self._engine_reserve())
                back = (f"; the expert cache took its VRAM back ({out.get('expert_cache_mib')} of "
                        f"{out.get('expert_cache_full_mib')} MiB)")
            # also when the cache never made room (a layer split, an engine without --vram-elastic): the stop happened
            print(f"[strata] images: the image encoder stopped after {v.elastic_idle_s:g} s without a picture{back}",
                  flush=True)
            self.vision_held = False
            return "down"
        finally:
            self.fifo.release()

    def start_vision_idle(self):
        v = self.vision
        if not getattr(v, "elastic", False):
            return
        print(f"[strata] images: the image encoder starts with the first picture and stops after "
              f"{v.elastic_idle_s:g} s without one; while it runs the expert cache keeps {v.elastic_mib} MiB free "
              "for it", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, v.elastic_idle_s / 4)))
                try:
                    self._vision_idle_down()
                except Exception as e:                 # tried again at the next turn; the thread keeps running
                    print(f"[strata] images: stopping the idle image encoder: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def free_vram_mib(self) -> int | None:
        """Free VRAM on the engine's GPU (an Intel card by PCI address, read through Level Zero's sysman);
        None when it can't be read (then nothing is refused)."""
        try:
            from serve.telemetry import free_vram_mib
            return free_vram_mib(getattr(self, "gpu_pci", None))
        except Exception:
            return None

    def ensure_loaded(self):
        """Start the engine if it is not running (unloaded, or it died - issue #27), after the before_load hook and
        the free-VRAM check.  The caller holds self.fifo."""
        if self.loaded() and not self._vision_down():
            return
        if self.images is not None and hasattr(self.images, "give_back"):
            self.images.give_back()                  # an image engine kept warm holds the VRAM the model needs
        hook_failed = None
        if self.before_load:
            cmd = self.before_load
            print(f"[strata] before loading: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
            why = None
            try:
                r = subprocess.run(cmd, shell=isinstance(cmd, str), timeout=self.before_load_timeout_s,
                                   stdin=subprocess.DEVNULL)
                if r.returncode != 0:
                    why = f"it exited with code {r.returncode}"
            except subprocess.TimeoutExpired:
                why = f"it did not finish in {self.before_load_timeout_s:g} s (before_load_timeout_s)"
            except (OSError, subprocess.SubprocessError) as e:
                why = f"it could not run ({e})"
            if why is not None:
                # #1997: a failing hook (e.g. the other server's /unload that is still busy) may mean the GPU was not
                # given back.  A peer that is simply not running also makes `curl` fail (exit 7), so the exit code alone
                # does not refuse: the card's free VRAM decides (min_free_vram_mib, checked just below); "strict" always
                # refuses, "ignore" always loads.
                if self.before_load_strict and not self.before_load_ignore_failure:
                    print(f"[strata] the before_load command failed ({why}); the model stays unloaded and this "
                          "request is answered 503 - try again (before_load_strict)", flush=True)
                    raise GpuBusy(f"the before_load command failed ({why}): the GPU was not given back, so the "
                                  "model was not loaded; try again in a moment", retry_after=15)
                hook_failed = why
                if self.before_load_ignore_failure or not self.min_free_vram_mib:
                    print(f"[strata] the before_load command failed ({why}); loading anyway"
                          + ("" if self.before_load_ignore_failure else
                             " (set min_free_vram_mib to have the free VRAM checked, or before_load_strict to refuse)"),
                          flush=True)
                else:
                    print(f"[strata] the before_load command failed ({why}); checking the free VRAM", flush=True)
        if self.min_free_vram_mib and not (hook_failed and self.before_load_ignore_failure):
            free = self.free_vram_mib()
            deadline = time.time() + self.vram_free_wait_s   # memory another process just gave back can take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                if hook_failed and not self.before_load_ignore_failure:
                    print(f"[strata] the before_load command failed ({hook_failed}) and only {free} MiB of VRAM is "
                          f"free (the model needs {self.min_free_vram_mib}): not loading; 503", flush=True)
                raise GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model needs "
                              f"{self.min_free_vram_mib} (min_free_vram_mib) - it stays unloaded until that is free"
                              + (f"; the before_load command failed ({hook_failed})" if hook_failed else ""),
                              retry_after=15)
            if hook_failed and not self.before_load_ignore_failure:
                print(f"[strata] the before_load command failed ({hook_failed}), but {free} MiB of VRAM is free: "
                      "loading", flush=True)
        vision_started = False
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[strata] starting the vision encoder ...", flush=True)   # the engine sizes its cache
            try:
                self.vision.restart()
            except Exception as e:
                raise EngineDied(f"the vision encoder failed to start: {e}") from e
            vision_started = True
        if self.loaded():
            return
        if getattr(self.engine, "unloaded", False):
            print("[strata] loading the model again (it was unloaded) ...", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[strata] the engine had stopped ({exit_text(code)}); starting it again "
                  "(a minute or two) ...", flush=True)
        # what the web app's chat says while a request waits for it (/metrics live.loading): after an image job
        # (restore_after_load) the model comes back with the conversation it had
        with self.status_lock:
            self.status["loading"] = "after_image" if self.restore_after_load else "again"
        try:
            try:
                self.engine.restart()
            except Exception as e:
                if vision_started:
                    self.vision.unload()
                # the program could not be started at all (missing, not executable, or blocked by Windows - #735):
                # the request path answers EngineDied with a 503, but takes a bare OSError for a client that went
                # away (#2496)
                if isinstance(e, OSError):
                    raise EngineDied(f"the engine could not be started: {e}") from e
                raise
            print("[strata] the engine is running again", flush=True)
            if self.restore_after_load:                  # an image job unloaded it: its conversation comes back
                images_api.restore_session(self)
        finally:
            with self.status_lock:
                self.status.pop("loading", None)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # #533: the reserve asked for last
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, EngineDied) as e:
                print(f"[strata] the VRAM reserve ({self.vram_reserve} MiB) was not applied: {e}", flush=True)

    vram_wait_s = 300.0                                  # #533: how long POST /v1/vram waits for a running request

    def vram(self, reserve_mib: int | None) -> dict:
        """#533, POST /v1/vram: keep `reserve_mib` of VRAM free for other programs (None: the reserve the engine
        started with), applied between requests - a request that is running finishes first (up to vram_wait_s).
        Only an engine started with --vram-elastic (vram_elastic_on) can do it; it never resizes
        on its own.  An unloaded engine applies it when it loads.  -> {"status": ..., and the engine's figures}."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        if not self.fifo.acquire(timeout=self.vram_wait_s):
            raise ModelBusy("a request is still running; try again when it has finished")
        # parallel requests do not hold the fifo: the engine's control lines (a prompt being admitted) are taken too,
        # and the engine refuses a resize while a slot decodes
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(timeout=self.vram_wait_s):
            self.fifo.release()
            raise ModelBusy("a request is still running; try again when it has finished")
        try:
            if not self.loaded():
                self.vram_reserve, self.vram_reserve_asked = reserve_mib, True
                return {"status": "not loaded", "reserve_mib": reserve_mib,
                        "note": "applied when the model loads"}
            # an elastic encoder that runs keeps its VRAM on top of the asked reserve
            held = self.vision.elastic_mib if self.vision_held and self.vision is not None else None
            out = self.engine.vram((reserve_mib or 0) + held if held is not None else reserve_mib)
            self.vram_reserve, self.vram_reserve_asked = reserve_mib, True
            print(f"[strata] VRAM: {out.get('vram_free_mib')} MiB free, expert cache {out.get('expert_cache_mib')} of "
                  f"{out.get('expert_cache_full_mib')} MiB ({out.get('expert_slots')} experts)", flush=True)
            return {"status": "ok", **out}
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _say_died(self, e: Exception) -> None:
        """The server window's line for an engine that died (or was ended, #481) in the middle of a request; the web
        app shows it too (issue #10: engine_failure in /metrics and /v1/status)."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        self.engine_failure = {"at": int(time.time()), "message": f"{e}. {note}".strip(), "log": log}
        print(f"[strata] {e}. {note} The next request starts the engine again."
              f"{' Its log: ' + log if log else ''}", flush=True)

    def load(self):
        """POST /load and every generation request: start the engine now if it is unloaded (raises GpuBusy)."""
        # a request is on its way: the idle thread must not unload between this and the request's own start
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        # #1603: waiting for the fifo to start the engine is queued, not nothing
        with self.status_lock:
            self.status["queued"] += 1
        counted = True
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                counted = False
                loading = time.perf_counter()
                if trace is not None:
                    with self.status_lock:
                        trace["queue_s"] += round(loading - waiting, 3)
                        trace["state"] = "loading"
                try:
                    self.ensure_loaded()
                finally:
                    if trace is not None:
                        with self.status_lock:
                            trace["load_s"] += round(time.perf_counter() - loading, 3)
                            trace["state"] = "queued"
        finally:
            if counted:
                with self.status_lock:
                    self.status["queued"] -= 1

    UNLOAD_WAIT_S = 300.0       # #1997: POST /unload?wait=1 waits this long for the running turn (wait=<s> to choose)

    def unload_waiting(self, wait_s: float) -> str:
        """#1997: unload(), but a running or queued request is waited for (up to wait_s) instead of answering "busy":
        a peer server that needs this GPU can then free the card on the server's terms.  "busy" after wait_s."""
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            r = self.unload()
            if r != "busy" or time.monotonic() >= deadline:
                return r
            time.sleep(0.25)

    def unload(self, idle_for: float | None = None) -> str:
        """Stop the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, or
        with idle_for: one ran more recently than that) or "unsupported"."""
        if not hasattr(self.engine, "unload"):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            if not self.loaded():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            self.vision_held = False
            print(f"[strata] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}; "
                  "the next request loads it again", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[strata] the model unloads after {self.idle_unload_s} s without requests", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except EngineStuck as e:                # tried again at the next turn; the thread keeps running
                    print(f"[strata] idle unload: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[strata] could not save the shared settings: {e}", flush=True)
        return self.shared

    def effort_default(self) -> str | None:
        """The thinking level of a request that names none: the Chat settings shared with apps first, then the config's
        "reasoning_effort" (#1641).  Every API reads it here (the Responses API did not: 0.1.44.1)."""
        return self.shared.get("reasoning_effort") or self.config_effort

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        effort = self.effort_default()
        if not s and not effort:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs")
                ctk = ctk if isinstance(ctk, dict) else {}
                if req.get("reasoning_effort") in (None, "") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from serve.telemetry import Telemetry
            cards = getattr(self, "gpu_roles", None) or gpu_roles({"split_pci": getattr(self, "split_pci", None)},
                                                                  getattr(self, "gpu_pci", None))
            # an image job's seconds on the text model's card are no tok/s: the speed history has a gap there, not
            # zeros; a job on a card of its own runs beside the chat, whose speed goes on
            self.telemetry = Telemetry(extra=lambda: {"tok_s": None if self._image_shares_text() else self._tok_s(),
                                                      "tok_s_mean": self._tok_s_mean(),
                                                      "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpus=cards)
            if getattr(self, "engine", None) is not None and hasattr(self.engine, "gpu_busy"):
                # #1317: the quick frozen-engine check does not end an engine whose GPU is busy (a long prompt chunk on
                # a slow card keeps the GPU at work while the host thread sleeps)
                self.engine.gpu_busy = lambda: (self.telemetry.snapshot()["now"].get("gpu_util") or 0) >= 10

    def _image_running(self) -> bool:
        return self.images is not None and self.images.job is not None

    def image_gpu_mode(self) -> str:
        """Where the image jobs run beside the chat: "separate" (a card of their own: they run at the same time),
        "helper" (the text model's second card, its helper expert cache closed for each job: the chat goes on, on the
        first) or "shared" (the text model's card: they take turns); "" without images."""
        if self.images is None:
            return ""
        if getattr(self.images, "helper_card", ""):
            return "helper"
        return "separate" if getattr(self.images, "separate", False) else "shared"

    def _image_shares_text(self) -> bool:
        return self._image_running() and self.image_gpu_mode() == "shared"

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating.  With parallel requests:
        the sum of theirs."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def begin_request(self, path, req):
        """#332: a monitor record for this request, or None when the monitor is off (nothing is kept then)."""
        if not self.api_monitor:
            return None
        raw = json.dumps(req, ensure_ascii=False, indent=2)
        record = {"id": uuid.uuid4().hex[:12], "path": path, "model": req.get("model") or self.model,
                  "started_at": time.time(), "state": "queued", "stream": bool(req.get("stream")),
                  "response_format": (req.get("response_format") or {}).get("type")
                  if isinstance(req.get("response_format"), dict) else None,
                  "input": raw[:262144], "input_truncated": len(raw) > 262144,
                  "output": "", "reasoning": "", "output_truncated": False, "reasoning_truncated": False,
                  "queue_s": 0.0, "load_s": 0.0, "first_token_s": None,
                  "_clock": time.perf_counter()}
        with self.status_lock:
            self.api_requests.append(record)
        self.request_trace.record = record
        return record

    def debug_begin(self, path, req):
        """This request's debug chat log record (serve/debuglog.py), or None when the log is off.  It goes with the
        request on this thread (request_trace.debug): prepare() adds each prompt the engine reads to it, the tool
        loop each round, the handler the answer."""
        if self.debug_log is None:
            return None
        rec = {"id": uuid.uuid4().hex[:12], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "path": path,
               "request": req, "prompts": [], "tools": [], "reasoning": "", "output": "", "tool_calls": {}}
        self.request_trace.debug = rec
        return rec

    def debug_end(self, rec) -> None:
        """The record written once (the reply's thread, or the handler when no answer started)."""
        if rec is not None and self.debug_log is not None and not rec.get("_written"):
            rec["_written"] = True
            rec["tool_calls"] = list(rec["tool_calls"].values())
            self.debug_log.request({k: v for k, v in rec.items() if not k.startswith("_")})

    def debug_prompt(self, ids, keys) -> None:
        """The prompt the engine reads, for the debug chat log: each picture's image tokens as one marker."""
        rec = getattr(self.request_trace, "debug", None)
        if rec is None:
            return
        try:
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            parts: list[str] = []
            run: list[int] = []
            k = 0
            i = 0
            while i < len(ids):
                if ids[i] == pad:
                    j = i
                    while j < len(ids) and ids[j] == pad:
                        j += 1
                    parts.append(self.tok.decode(run))
                    key = keys[k] if keys and k < len(keys) else "?"
                    parts.append(f"<|image_pad|>x{j - i} [picture {key}]")
                    run, k, i = [], k + 1, j
                else:
                    run.append(ids[i])
                    i += 1
            parts.append(self.tok.decode(run))
            rec["prompts"].append({"tokens": len(ids), "text": "".join(parts)})
        except Exception as e:  # noqa: BLE001 - the log must not fail a request
            rec["prompts"].append({"tokens": len(ids), "error": f"{type(e).__name__}: {e}"})

    def request_records(self, request_id=None):
        with self.status_lock:
            records = list(self.api_requests)
            if request_id:
                record = next((r for r in records if r["id"] == request_id), None)
                if record is None:
                    return None
                return {k: v for k, v in record.items() if not k.startswith("_")}
            return [{k: v for k, v in r.items()
                     if k not in ("input", "output", "reasoning", "response") and not k.startswith("_")}
                    | {"wallclock_s": r.get("wallclock_s", round(time.perf_counter() - r["_clock"], 3))}
                    for r in reversed(records)]

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel requests: the newest one's status
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
            totals = dict(self.totals)
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and s.get("first_token") is None:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0), "phase": s.get("phase") if s.get("busy") else None,
                "loading": s.get("loading"),
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if s.get("busy") else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        par = int(getattr(self.engine, "batch", 0) or 0)
        if par:                                         # #465: the requests running together, slot by slot
            with self.status_lock:
                running = len(self.live_reqs)
            slots = self.engine.slots_view() if hasattr(self.engine, "slots_view") else []
            # the requests in flight that are not in a slot: the one alone on the solo path, or waiting their turn
            in_slots = sum(1 for x in slots if x["state"] != "idle")
            live.update(parallel=par, running=running, slots=slots, outside_slots=max(0, running - in_slots),
                        waiting=int(getattr(self.engine, "waiting", 0) or 0),
                        admitting=int(getattr(self.engine, "admitting", 0) or 0))
            if running and state == "idle":
                live["state"] = "generating"
        if hasattr(self.engine, "replica_view"):        # opt-in replicas: each one's state and requests routed to it
            live["replicas"] = self.engine.replica_view()
        if hasattr(self.engine, "split_view"):          # opt-in split serving: the hand-overs so far (serve/split.py)
            live["split_serve"] = self.engine.split_view()
        image = self.images.live() if self.images is not None else None
        if image is not None:                           # an image job in the GPU's turn: its stage and steps
            live["image"] = image
        # the image jobs that ended beside the chat's requests, newest first (the Monitor's Requests table); the
        # conversation cache counts the chat's only
        rows = sorted(hist + (self.images.recent() if self.images is not None else []), key=lambda r: r["time"])
        vision = self.vision
        engine = {"model": self.model, "max_context": self.reported_ctx(), "images": vision is not None,
                  # the About's other models: the picture reader that serves (the configured one or its fallback) and
                  # the image models
                  "vision": None if vision is None else {
                      "encoder": vision.label, "max_tokens": vision.max_tokens,
                      "error": getattr(vision, "dead", None) or self.vision_error, "fallback": self.vision_fallback},
                  "image_models": self.images.about() if self.images is not None else [],
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = (self.telemetry.snapshot() if getattr(self, "telemetry", None)
               else {"now": {}, "history": {}, "static": {}})
        engine["image_gpu"] = self.image_gpu_mode()
        if tel["now"].get("gpus"):                      # what each card does right now (the Monitor's tabs light it)
            doing = {"text": live["state"] in ("reading", "generating"), "image": image is not None}
            tel["now"] = dict(tel["now"], gpus=[dict(g, active=[r for r in g.get("roles") or () if doing.get(r)])
                                                for g in tel["now"]["gpus"]])
        parked = self.conv_log.poll(getattr(self.engine, "log_path", None), getattr(self.engine, "log_start", None))
        return {"engine": engine, "live": live, "requests": rows[::-1][:None if all_requests else 12],
                "conversation_cache": conversation_cache_view(engine, hist, totals, parked),
                "requests_kept": len(rows), "totals": totals, "hardware": tel["now"],
                "loop_guard": dict(self.loop_guard_counts, on=bool(self.loop_guard)),
                "script_guard": dict(self.script_guard_counts, default=script_guard_name(self.script_guard)),
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now,
                "engine_failure": self.engine_failure}

    def prometheus(self) -> str:
        """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
        metric names (llamacpp:*) so its dashboards work unchanged; the drafts and requests as strata:*."""
        with self.status_lock:
            s, t = dict(self.status), dict(self.totals)
            running = len(self.live_reqs) or int(bool(s.get("busy")))
            prompt_n = t["prompt_tokens"] - t["reused"]               # what was read, as llama.cpp counts it
            prompt_s, decode_s = t["prompt_ms"] / 1000, t["decode_ms"] / 1000
            # llama.cpp reads these two gauges from a bucket it resets on every scrape (server_metrics::bucket,
            # "the gauges are averaged over the window between two scrapes"), so an idle server reports 0 rather
            # than its lifetime average. Keep the same window here, or the names lie to llama.cpp dashboards.
            was_prompt_n, was_prompt_s, was_predicted_n, was_decode_s = self.scrape_bucket
            self.scrape_bucket = (prompt_n, prompt_s, t["output_tokens"], decode_s)
        d_prompt_n, d_prompt_s = prompt_n - was_prompt_n, prompt_s - was_prompt_s
        d_predicted_n, d_decode_s = t["output_tokens"] - was_predicted_n, decode_s - was_decode_s
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held",
             t["reused"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", prompt_s),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t["output_tokens"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", decode_s),
            ("llamacpp:prompt_tokens_seconds", "gauge",
             "Average prompt read throughput in tokens/s, since the last scrape",
             d_prompt_n / d_prompt_s if d_prompt_s > 0 else 0),
            ("llamacpp:predicted_tokens_seconds", "gauge",
             "Average generation throughput in tokens/s, since the last scrape",
             d_predicted_n / d_decode_s if d_decode_s > 0 else 0),
            ("llamacpp:requests_processing", "gauge", "Requests running", running),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued") or 0),
            ("strata:requests_total", "counter", "Requests finished", t["requests"]),
            ("strata:drafts_offered_total", "counter", "Speculative draft tokens offered", t["drafts_offered"]),
            ("strata:drafts_accepted_total", "counter", "Speculative draft tokens accepted", t["drafts_accepted"]),
            ("strata:tool_calls_unread_total", "counter",
             "<tool_call> blocks returned as text (an undeclared name or a form the template does not use)",
             t.get("tool_calls_unread", 0)),
            ("strata:tool_args_repaired_total", "counter",
             "Tool call arguments (a JSON array or object) that were damaged and read as written",
             t.get("tool_args_repaired", 0)),
            ("strata:model_loaded", "gauge", "1 while the model is loaded", int(self.loaded())),
        ]
        return "".join(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name} {value}\n"
                       for name, kind, help_, value in rows)

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.reported_ctx()
        images = self.vision is not None and not getattr(self.vision, "dead", None)
        vision_error = self.vision_error or (getattr(self.vision, "dead", None) if self.vision is not None else None)
        return {
            "service": "strata", "model": self.model,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            # one request at a time (more wait their turn), or "parallel": N batch slots (#465)
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": max(1, int(getattr(self.engine, "batch", 0) or 0))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images or vision_error is not None, "available": images, "error": vision_error,
                       "encoder": getattr(self.vision, "label", None) if self.vision is not None else None,
                       "fallback": self.vision_fallback},
            "image_generation": self.images.status() if self.images is not None else {"enabled": False},
            "activity": {"requests": totals["requests"] + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "engine_failure": self.engine_failure,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic"))),
            "machine": {
                "at": int(time.time()),
                "gpu": ({"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                         "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                         "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")}
                        if static.get("gpu_name") else None),
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def render_prompt(self, messages, tools, kwargs) -> str:
        """The template rendered.  #458: with `effort_end`, a request with a non-default effort (low, medium or no
        thinking) is rendered as a default one up to the answer - the same prompt start, so the conversation cache
        keeps it - and its effort follows in a short system turn right before the answer (thinking off: the template's
        empty thinking block).  The engine (--tail-role-token) checkpoints in front of that turn."""
        # thinking unless asked off, as the server counts it (encode_prompt): said to the template, since Unsloth's
        # Qwen3.5-9B template thinks only when enable_thinking is given true (the others think unless it is false)
        kwargs = {"enable_thinking": True, **kwargs}
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template this cannot take apart: rendered as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs, store: bool = True) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens.  `store=False` (a token count) reuses the kept
        prompts without adding or replacing one."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        plain: list[tuple[int, int]] | tuple[()] = ()
        if changed:
            prompt, plain = unmark_think_literals(prompt, self.literals)
        if self.prompts is not None:            # #567: only what follows the last special shared with a recent prompt
            return self.prompts.encode(prompt, plain, store)
        return self.tok.encode(prompt, parse_special=True, plain=plain) if plain else \
            self.tok.encode(prompt, parse_special=True)

    @contextlib.contextmanager
    def _turn(self, cancel=None):
        """The engine's single turn (self.fifo) for an image encode.  #1999: a client that leaves while this request
        waits for it gives its place up (ClientGone), as a queued generation does (#1619)."""
        if cancel is None:
            self.fifo.acquire()
        else:
            while not self.fifo.acquire(timeout=0.25):
                if cancel.is_set():
                    raise ClientGone()
        try:
            yield
        finally:
            self.fifo.release()

    def _note_unreadable_tool_images(self, messages, cancel=None):
        """A picture a tool returned (Claude Code's Read of an image file) that this server cannot read - it has no
        image encoder, or the encoder refuses that picture - becomes a short note in its place instead of a 400.
        Clients resend the whole history every turn, so one unreadable tool picture would fail every later request of
        the conversation; a picture the user sends keeps the 400.  A picture that encodes stays in the cache, so the
        prompt's own encode below finds it."""
        fetched = {}                     # a URL's bytes, so prepare() does not download the same picture again
        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool" or not isinstance(content, list):
                continue
            for n, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                why = "this server has no image encoder" if self.vision is None else None
                if why is None:
                    src = item["source"]
                    try:
                        if cancel is not None and cancel.is_set():
                            raise ClientGone()
                        if src.startswith(("http://", "https://")):
                            src = fetched[item["source"]] = self.download(src)   # outside the FIFO, as in prepare()
                        with self._turn(cancel):  # the encoder takes its turn with the requests (see below)
                            self._vision_up()         # an elastic encoder is not running until a picture needs it
                            self.vision.encode(src)
                    except (ValueError, OSError, RuntimeError) as e:    # RuntimeError: the encoder did not start
                        # a file error's sentence without its path: the note goes into the model's prompt
                        why = (e.strerror or type(e).__name__) if isinstance(e, OSError) else str(e)
                if why is not None:
                    print(f"[strata] a picture a tool returned was left out of the prompt: {why}", flush=True)
                    content[n] = {"type": "text", "text": f"[image omitted: {why}]"}
        return fetched

    def forget_prompt(self, ids):
        if self.prompts is not None:
            self.prompts.forget(ids)

    def expand_files(self, messages, cancel=None):
        """The files of a request (frontend's "file" items) read into the content: a text file's text, a PDF's text
        page by page and, with the vision encoder, each page as a picture (read_file).  A file read once is kept for
        the next request (the client sends the conversation again).  The messages without files are returned as
        they are."""
        if not any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "file"
                                                              for p in m["content"]) for m in messages):
            return messages
        cache = self.__dict__.setdefault("file_cache", collections.OrderedDict())
        out = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                out.append(m)
                continue
            items: list = []
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "file"):
                    items.append(part)
                    continue
                if cancel is not None and cancel.is_set():
                    raise ClientGone()
                data = Vision.load(part["data"]) if part.get("data") else self.download(part["url"])
                if len(data) > FILE_MAX:
                    raise ValueError(f"the file {part.get('name') or ''} is over {FILE_MAX >> 20} MiB")
                key = (hashlib.sha256(data).hexdigest(), self.vision is not None, part.get("name"))
                if key not in cache:
                    cache[key] = read_file(data, part.get("mime"), part.get("name") or "", self.vision is not None)
                    while len(cache) > 8:
                        cache.popitem(last=False)
                cache.move_to_end(key)
                items += cache[key]
            out.append({**m, "content": items})
        return out

    def forced_ids(self, force, kwargs) -> list[int]:
        """The tokens of a forced call's opening (forced_call) that end the prompt: without thinking the reply starts
        with it (see run); with thinking it is written after the thinking, as part of the reply."""
        return self.tok.encode(force, parse_special=True) if force and kwargs.get("enable_thinking", True) is False \
            else []

    def prompt_ids(self, messages, tools, kwargs, force=None, cancel=None, store=True, wake=False):
        """The prompt the engine reads for these messages, as prepare() and Anthropic's count_tokens make it -> (the
        messages as read: files expanded, tool pictures resolved; the prompt's ids, each picture's marker expanded to
        its image tokens and the forced call's opening (forced_ids) at the end; the rendered text's ids; each
        picture's (embeddings file, tokens)).  The pictures are encoded here (in the FIFO's turn; one encoded before
        comes from the cache).  `store`: keep the rendering for the next request (encode_prompt).  `wake`: an encoder
        that is down (an idle unload's) is brought back by load(), as a request's own load does before prepare()."""
        # `cancel` (#1999): set when the client hangs up; the downloads, the wait for the turn and the encodes stop
        # between pictures and raise ClientGone
        messages = self.expand_files(messages, cancel)
        # Cached tool images are references in browser history, never arbitrary file paths or huge base64 strings.
        for message in messages:
            for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
                source = part.get("source", "") if isinstance(part, dict) else ""
                if (isinstance(part, dict) and part.get("type") == "image" and isinstance(source, str)
                        and source.startswith("strata-tool-media:")):
                    name = source[len("strata-tool-media:"):]
                    if not re.fullmatch(r"[0-9a-f]{64}\.(png|jpg|webp)", name):
                        raise ValueError("invalid tool image reference")
                    # Clients resend old tool results. A removed cache or disabled feature must not poison
                    # every later turn; keep the other text and replace only the unavailable picture.
                    media_dir = self.mcp.media_dir if self.mcp is not None else None
                    image = Path(media_dir) / name if media_dir is not None else None
                    if image is None or not image.is_file():
                        why = ("tool image cache is disabled" if image is None
                               else "cached tool image is no longer available")
                        part.clear()
                        part.update(type="text", text=f"[image omitted: {why}]")
                    else:
                        part["source"] = str(image)
        # A stopped reader is not an unreadable tool picture.  Wake it before the tool-image probe can replace
        # that picture with an omission note; load failures keep their admission error, as on the generation path.
        if wake and images_of(messages) and self._vision_down():
            self.load()                 # its admission (before_load, the free VRAM) and the encoder's restart
        fetched = self._note_unreadable_tool_images(messages, cancel)
        ids = encoded_ids = self.encode_prompt(messages, tools, kwargs, store=store)
        ids = ids + self.forced_ids(force, kwargs)
        encoded: list = []
        images = images_of(messages)
        if images:
            if self.vision is None:
                if self.vision_error:
                    raise ValueError("images are unavailable: " + self.vision_error)
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]

            # An image URL is downloaded first, outside the FIFO: under it, a slow server held every other request
            # for as long as urlopen waited.
            def fetch(src):
                if cancel is not None and cancel.is_set():
                    raise ClientGone()
                return fetched[src] if src in fetched else self.download(src) \
                    if src.startswith(("http://", "https://")) else src
            low = [item.get("detail") == "low" for m in messages if isinstance(m.get("content"), list)
                   for item in m["content"] if item.get("type") == "image"]
            images = [low_detail(fetch(src)) if is_low else fetch(src) for src, is_low in zip(images, low)]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self._turn(cancel):
                self._vision_up()
                try:
                    encoded = (self.vision.encode_all(images, cancel) if hasattr(self.vision, "encode_all")
                               else [self.vision.encode(src) for src in images])
                finally:
                    self.last_image_at = time.time()
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more pairs than images: text parts that cut both markers apart (the literal marks keep whole
                    # ones text)
                    if k == len(encoded):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
        return messages, ids, encoded_ids, encoded

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        self.embeddings.path = None
        self.embeddings.keys = None
        messages, ids, encoded_ids, encoded = self.prompt_ids(messages, tools, kwargs, force, cancel)
        images = encoded
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # #344: (re)starting, not a prompt that is too long
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: check against the last known context, so the request reaches
            # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                # A prompt refused for the context is not kept (#567): the client cannot resend it unchanged.  The
                # other refusals (engine starting, images) keep it: the same prompt comes back and its encode is reused.
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's xestrata-<model>.json to shorten it to the room left (#545)")
            print(f"[strata] max_tokens {max_new} shortened to {max(1, room)}: the prompt ({len(ids)} tokens) "
                  f"leaves that much of the context ({ctx}); \"fit_max_tokens\": false answers 400 instead", flush=True)
            max_new = max(1, room)          # a shorter completion beats a 400 (the default; upstream f31b6efd6)
        if images:
            # The request's images in one file for GENI (~10 MB a picture), written once nothing above refuses the
            # request: one refused after it (the engine starting, no room) left it in the vision directory for good,
            # one more for every retry of a 503.  run() deletes it; drop_embeddings() if run() never starts.
            combined = combined_embeddings_path(self.vision.dir, sum(Path(p).stat().st_size for p, _ in encoded))
            self.embeddings.path = combined             # first, so a half-written one is found as well
            # #2228: each picture's key is its normalized bytes' hash (the encoder's file name), so the engine's
            # conversation cache matches a picture by what it is, not by its embedding floats: a picture encoded again
            # (an encoder restart, the cache's eviction) with other bits still matches its checkpoints
            self.embeddings.keys = [Path(p).stem[:16] for p, _ in encoded]
            write_temporary(combined, [p for p, _ in encoded])
        self.debug_prompt(ids, getattr(self.embeddings, "keys", None))
        if req is not None and req.get("strata_prefix") is not None:
            req["strata_prefix"] = self.resolve_prefix(req["strata_prefix"], messages, tools, kwargs, ids, bool(images))
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    PREFIX_END = "\u0001strata-prefix-end\u0001"   # where a message's shared part ends in the rendered prompt

    def resolve_prefix(self, spec, messages, tools, kwargs, ids, has_images=False) -> dict:
        """R1: `"strata_prefix"` marks the start of the prompt as a shared read-only prefix: a long document that many
        short questions follow.  Three ways to say where it ends: `{"messages": K}` (the first K messages), `{"message":
        I, "chars": N}` (the first N characters of message I's text: a document and its question in one message) or
        `{"tokens": N}`.  The engine reads the prompt in two parts there, keeps the checkpoint at the boundary pinned
        (never evicted) and does not park the branch it leaves, so N questions cost one prefix.  Returns
        {"tokens": N} (what the engine's `pin=N` takes: always a prefix of the prompt's ids, the tokenizer's merge
        across the boundary included), or {} when the prefix cannot be marked (said in the log: the request is never
        refused for it).  A malformed field is a 400."""
        keys = ("messages", "message", "chars", "tokens")
        if (not isinstance(spec, dict) or not spec or any(k not in keys for k in spec) or
                set(spec) not in ({"messages"}, {"tokens"}, {"message", "chars"})):
            raise ValueError('strata_prefix: {"messages": K} (the first K messages), {"message": I, "chars": N} '
                             '(the first N characters of message I) or {"tokens": N}')
        for k, v in spec.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if k == "message" else 1):
                raise ValueError(f"strata_prefix.{k}: a whole number ({'0' if k == 'message' else '1'} or more)")
        if has_images:
            print("[strata] strata_prefix ignored: the request has images", flush=True)
            return {}
        what = ",".join(f"{k}={v}" for k, v in spec.items())

        def lcp(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        def head_ids(msgs):
            """The ids the rendered `msgs` start with (a message's trailing marker cut at PREFIX_END when there)."""
            marked, marked_tools, changed = mark_think_literals(msgs, tools, self.literals)
            effort = kwargs.get("reasoning_effort")
            off = kwargs.get("enable_thinking") is False        # render_prompt's rule for #458's effort_end
            kw = ({k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
                  if self.effort_end and (off or effort not in (None, "", "xhigh", "high")) else dict(kwargs))
            kw["add_generation_prompt"] = False
            text = self.template.render(marked, tools=marked_tools, **kw)
            cut = text.find(self.PREFIX_END)
            if cut >= 0:
                text = text[:cut]
            if changed:
                text, plain = unmark_think_literals(text, self.literals)
                return self.tok.encode(text, parse_special=True, plain=plain)
            return self.tok.encode(text, parse_special=True)

        if "tokens" in spec:
            n = spec["tokens"]
        elif "messages" in spec:
            k = spec["messages"]
            if k >= len(messages):
                print(f"[strata] strata_prefix.messages={k} ignored: the request has {len(messages)} message(s) and "
                      "the prefix must leave at least the last one", flush=True)
                return {}
            head = head_ids(messages[:k])
            n = lcp(head, ids)
        else:
            i, chars = spec["message"], spec["chars"]
            text = messages[i].get("content") if 0 <= i < len(messages) else None
            if not isinstance(text, str) or chars > len(text):
                print(f"[strata] strata_prefix ignored: message {i} has no text of {chars} characters", flush=True)
                return {}
            cut = messages[:i] + [{**messages[i], "content": text[:chars] + self.PREFIX_END}]
            head = head_ids(cut)
            n = lcp(head, ids)
        if n < 1 or n >= len(ids) - 1:
            print(f"[strata] strata_prefix {what} ignored: it leaves no usable prefix of the {len(ids)}-token prompt "
                  f"(common start {n})", flush=True)
            return {}
        return {"tokens": n}

    def drop_embeddings(self) -> None:
        """Delete the combined image file prepare() wrote when no run() took it over (a run deletes its own as it
        ends): a request can end before its answer starts - its client gone, anything raised after prepare() - and
        then nothing else would.  The handler calls this when a request is done."""
        path, self.embeddings.path = getattr(self.embeddings, "path", None), None
        if path is not None:
            Path(path).unlink(missing_ok=True)

    def _next_pass(self, ids: list, nxt: list, raw_ids: list) -> list:
        """The next pass's prompt after the server wrote into the reply (a thinking budget's wrap-up, the thinking
        closed, a forced call's opening, the loop guard's note): the reply's text encoded whole, as the template
        renders it when the client sends the reply back, not the pieces as they were joined.  Joined, the model's
        last token and the inserted text can split differently ("  \n" + "\n\n" against "  \n\n\n"): the
        engine's state then is not the prompt the next turn sends, and a recurrent model reads that turn again
        from its checkpoint at the prompt (Mainplane, the chat's segments: 11,269 tokens after one wrap-up).
        `nxt` as it is where the reply is not exactly `raw_ids` after the prompt (a pass that cut it)."""
        if nxt[len(ids):] != raw_ids:
            return nxt
        return ids + list(self.tok.encode(self.tok.decode(raw_ids), parse_special=True))

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])
            # #984: while a thinking budget applies, the thinking line shows it, not the request's max_tokens
            limit = (f"budget {s['thinking_budget']} (max {s.get('max_tokens')})"
                     if s["phase"] == "thinking" and s.get("thinking_budget") else f"max {s.get('max_tokens')}")
            print(f"[strata] {s['phase']}: {s['generated']} of {limit} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def _loop_guard_on(self, sampling) -> bool:
        """The loop guard for this request: its own "loop_guard" when it sends one, else the server's."""
        own = (sampling or {}).get("loop_guard")
        return own if isinstance(own, bool) else bool(self.loop_guard)

    def _script_guard(self, sampling, ids):
        """This request's script guard (serve/script_guard.py), None when off: its own "script_guard" (the body's, the
        header's or the shared settings') when it has one, else the server's."""
        want = (sampling or {}).get("script_guard")
        want = self.script_guard if want is None else want
        if not isinstance(want, dict):
            return None
        info = getattr(self.engine, "info", {}) or {}
        if info.get("logit_bias") != 1 or getattr(self.engine, "batch", 0):
            if not self.script_guard_unsupported:
                self.script_guard_unsupported = True
                print("[strata] script guard: not applied - this engine takes no logit_bias (an older engine, or "
                      "\"parallel\" batch slots)", flush=True)
            return None
        with self.status_lock:
            self.script_guard_counts["requests"] += 1
        voc = script_guard.vocab(self.tok, want["lang"])
        table = None
        if info.get("script_table") == 1 and voc.n_vocab and voc.token_chars:   # the engine bans and masks itself
            try:
                table = voc.table(places.user_dir("state") / "script-tables")
            except OSError as e:
                print(f"[strata] script guard: cannot write its table ({e}); the server applies it", flush=True)
        return script_guard.Guard(voc, ids, want["allow_context"], table)

    def _guard_pieces(self):
        """A fresh token-text reader for the loop guard's markers (a character split across tokens whole)."""
        return Detokenizer(self.tok).push

    def _loop_noted(self, act, written: int):
        """A loop the guard broke: a line in the server window, and the Monitor's count."""
        print(f"[strata] loop guard: {act.describe()}"
              f"{' (now ' + format(written, ',') + ' tokens)' if act.kind != loop_guard.STOP else ''}", flush=True)
        with self.status_lock:
            c = self.loop_guard_counts
            c["detected"] += 1
            c[{loop_guard.PENALTY: "penalised", loop_guard.CLOSE_THINKING: "thinking_closed",
               loop_guard.STOP: "stopped"}[act.kind]] += 1

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
            declared_only=False) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..}).
        `force` (forced_call): the opening of the call the reply must make - see prepare()."""
        strict = (sampling or {}).get("strata_strict_tools")
        if strict:                     # the strict tools' calls, checked before they are sent (check_strict)
            rest = {k: v for k, v in sampling.items() if k != "strata_strict_tools"}
            yield from checked_calls(self.run(ids, thinking, tools, max_new, rest, cancel, force, declared_only),
                                     strict)
            return
        # the model card's sampling for this thinking state, then the config's block, then the Chat settings shared with
        # apps; the request's own fields win over all of them
        defaults = {**model_card_sampling(self.model_arch, thinking), **self.sampling_defaults, **self.shared}
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        # parallel_tool_calls: false (OpenAI) / tool_choice.disable_parallel_tool_use (Anthropic): the reply ends with
        # its first call, a whole one (finish "stop", so the call is the reply's tool_calls / tool_use)
        choice = (sampling or {}).get("tool_choice")
        one_call = (sampling or {}).get("parallel_tool_calls") is False or \
            isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True
        # a tool_choice that names a tool (its opening forced): the reply's one call is that tool's - a call of another
        # (one written in the thinking, or after the forced one) stays text, as an undeclared tool's does
        kind, only = tool_choice_of(choice)
        if force and kind == "named":
            if isinstance(choice, dict) and isinstance(choice.get("namespace"), str):   # Responses' namespace.name
                only = f"{choice['namespace']}.{only}"
            tools = [t for t in tools or [] if isinstance(t, dict) and t.get("name") == only]
            declared_only = one_call = True
        # #123: read after the merge, so a budget shared through POST /settings is seen like the other keys
        # the Harmony format writes its reasoning in a channel: the budget's and the loop recovery's closing text is
        # Qwen's <think> form, so neither applies to it
        harmony = getattr(self.template, "harmony", False)
        budget = self.reasoning_budget(sampling) if thinking and not harmony else None
        if thinking and not harmony and not budget and self.answer_room and max_new and max_new >= ANSWER_ROOM_MIN_CAP:
            # 0.1.45 (upstream d0e990ba): max_tokens counts the thinking too.  A client that caps the reply at 8192
            # (the desktop agent's harness sends max_completion_tokens: 8192 on every request) and a task the model
            # thinks long about got the cut-off in the middle of the thinking and NO answer.  So thinking is closed
            # early enough to leave the answer its quarter of the cap (the budget's wrap-up: a short closing sentence,
            # </think>, the answer).  A request with a budget of its own, or max_tokens below ANSWER_ROOM_MIN_CAP, is
            # left as it was.
            budget = max_new - min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if max_new <= 32768:
                print(f"[strata] thinking limited to {budget} of max_tokens {max_new} so the answer has room "
                      "(\"answer_room\": false in the config turns this off)", flush=True)
        if budget and max_new and max_new > 0:
            # #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking, so
            # it has no answer.  Close the thinking early enough to leave room for one.
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if budget > max_new - reserve:
                print(f"[strata] thinking capped at {max_new - reserve} of max_tokens {max_new} to leave room for "
                      f"the answer (budget {budget})", flush=True)
                budget = max(1, max_new - reserve)
        reader = GlmOutputParser if getattr(self.template, "call_form", "qwen") == "glm" else OutputParser
        parser = (HarmonyParser(thinking=thinking, tools=tools) if harmony else
                  reader(thinking=thinking, tools=tools, stream_tools=True, recover=self.tool_call_recovery,
                         declared_only=declared_only))
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # #606: the current run of one repeated token
        # the loop guard (serve/loop_guard.py): every id goes through it before the parser; it holds back a short
        # tail that could still be a loop's repeats.  Harmony's thinking is not closed by the server (its closing
        # text is Qwen's), so there a loop in the thinking only gets the penalty.
        guard = loop_guard.ToolSafeGuard(parser.state == "reasoning", pieces=self._guard_pieces,
                                         can_close=not harmony) if self._loop_guard_on(sampling) else None
        guard_stopped = False                           # the guard ended the answer (a loop that came back)
        # the script guard (serve/script_guard.py): its bias applies from where the thinking ends
        sg = self._script_guard(sampling, ids)
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        opening, written = [], ""                       # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            written = force                             # (a grammar's prefix: the reply starts with it)
            opening, force = parser.feed(force), None
        tail = ""                                       # the last characters written (the newlines before a call)
        one_done = False                                # one_call: the reply's one call has been written
        answered, close_retried = False, False          # #1053: content or a call came out; the thinking was closed
        nf_retried = False                              # #879: the request was sent again after non-finite logits
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        recovery_count, reasoning_text, repeat_coverage = 0, "", 0.0
        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
        raw_ids: list = []                              # every generated id (STRATA_DEBUG: dump raw model text)
        emb, self.embeddings.path = getattr(self.embeddings, "path", None), None   # this run's to delete now
        img_keys, self.embeddings.keys = getattr(self.embeddings, "keys", None), None
        if emb and img_keys:
            sampling = {**(sampling or {}), "strata_img_keys": img_keys}   # #2228: sent beside the embeddings file
        # Only a DONE line replaces engine.last, so a request that died, errored or was disconnected must not have
        # the PREVIOUS request's decode figures recorded as its own: each pass keeps its own DONE (a segment) if
        # one arrived.  A reasoning-budget continuation is another native generation, not another API request,
        # so the generated reasoning must not become this request's cached input (#615).
        segments: list = []
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        # #465: with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st: dict = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        with self.status_lock:
            self.status["queued"] += 1
        try:
            # --batch: the engine runs several requests at once (StrataEngine.generate_batched orders them)
            turn = contextlib.ExitStack()
            if not par:
                # #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S, so a client with
                # a stream idle timeout does not abort a healthy queued request (and its retry read the prompt again);
                # a client that left while queued gives its place up instead of taking the turn only to cancel there
                got = False
                try:
                    while not got:
                        got = self.fifo.acquire(timeout=QUEUE_BEAT_S)
                        if got:
                            turn.callback(self.fifo.release)
                        elif cancel.is_set():
                            break
                        else:
                            yield "ping", None
                except BaseException:                    # the consumer closed us at a ping: no longer queued
                    with self.status_lock:
                        self.status["queued"] -= 1
                    raise
                if not got:                              # cancelled while queued
                    with self.status_lock:
                        self.status["queued"] -= 1
                    yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                                   "reasoning_tokens": 0, "reasoning_recoveries": 0}
                    return
            with turn:
                try:
                    with self.status_lock:
                        if trace is not None:
                            trace["queue_s"] += round(time.perf_counter() - waiting, 3)
                            trace["state"] = "generating"
                        if not par:
                            self.status["queued"] -= 1
                    # issue #27: it died in an earlier request (or was unloaded) - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that).
                    # #1603: a parallel request stays counted as queued until it is registered as running below -
                    # waiting here for the fifo (an image encode, a /v1/vram resize) left it neither - and one on a
                    # loaded engine has nothing to start, so it does not wait for the fifo at all
                    try:
                        if not par:
                            self.ensure_loaded()
                        elif not (self.loaded() and not self._vision_down()):
                            with self.fifo:
                                self.ensure_loaded()
                    except BaseException:
                        if par:
                            with self.status_lock:
                                self.status["queued"] -= 1
                        raise
                    with self.status_lock:
                        if par:
                            self.status["queued"] -= 1
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, reasoning_recoveries=0, thinking_budget=budget)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # thought: the reasoning tokens so far (the budget's count)
                    for ev in opening:
                        yield "event", ev
                    while True:
                        segment_before = getattr(self.engine, "last", None)
                        segments_before = len(segments)     # (#879: a pass sent again leaves no segment)
                        if isinstance((sampling or {}).get("strata_grammar"), dict) and (raw_ids or written):
                            # a pass that continues the reply (a budget's wrap-up, a forced call's opening, the
                            # thinking closed): the grammar goes on from what the reply holds so far
                            sampling = {**sampling, "strata_grammar": {
                                **sampling["strata_grammar"], "prefix": written + self.tok.decode(raw_ids)}}
                        # the loop guard's short repetition penalty after a loop: an engine call of its own, which
                        # ends where it ends (the request's own sampling after it); else the request's sampling
                        smp_call, n_call = guard.sampling(sampling, n, max_new - n) if guard is not None \
                            else (sampling, max_new - n)
                        if guard is not None and guard.events:
                            # after a loop the rest of the reply routes the experts exactly: the engine's
                            # STRATA_ROUTE_PRIOR bias towards the experts in VRAM can hold a loop
                            smp_call = {**(smp_call or {}), "strata_route_prior": False}
                        if sg is not None:
                            # (the thinking may have been closed by the server: a budget's wrap-up, the loop guard)
                            sg.active = sg.active or parser.state not in ("reasoning", "rcall")
                            sg.restart = False
                            smp_call, n_call = sg.sampling(smp_call, n_call, raw_ids)
                        n_pass = n
                        gen = self.engine.generate(prompt, n_call, smp_call, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, n_call, smp_call, cancel)
                        stream = loop_guard.HeldStream(guard, gen, self.stop_ids, n) if guard is not None else None
                        recover_prompt = None
                        seg, wrap, leaving = [], False, False   # this pass's tokens; the budget is reached; closed
                        nonfinite = False               # #879: the engine ended this pass on non-finite logits
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # A value the frontend holds back (a tool call's array or object argument is sent
                            # whole, once complete) can keep the stream silent for minutes while tokens are
                            # generated; a client's idle timeout then ends the request.  Send the same keep-alive
                            # the engine's heartbeat sends when nothing has gone out for KEEPALIVE_S.
                            last_out = time.perf_counter()
                            for t in (stream if stream is not None else gen):
                                if t is None:               # heartbeat while the engine is quiet
                                    last_print = self._progress(last_print, st=st)
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                    continue
                                n += 1
                                if trace is not None and trace["first_token_s"] is None:
                                    with self.status_lock:
                                        trace["first_token_s"] = round(time.perf_counter() - trace["_clock"], 3)
                                if t in self.stop_ids:
                                    if force and parser.state == "reasoning" and parser.buf == THINK_END:
                                        opens = True    # #537: the held </think> was the end: the call opens there
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += parser.state in ("reasoning", "rcall")
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # #606: a degenerate output, not an answer: end it here
                                    break
                                was_pending = detok.pending()
                                piece = detok.push(t)
                                if sg is not None and sg.active:
                                    if piece and sg.banned(piece):
                                        start = sg.leak(was_pending, len(raw_ids) - 1, len(seg))
                                        with self.status_lock:
                                            self.script_guard_counts["passed" if start is None else "rewinds"] += 1
                                        if start is not None:
                                            # spelt in byte tokens, none of them sent yet: they are dropped and the
                                            # engine writes this place again with the first of them banned
                                            drop = len(raw_ids) - start
                                            sg.ban_next, sg.restart = raw_ids[start], True
                                            del raw_ids[start:], seg[-drop:]
                                            n -= drop
                                            detok = Detokenizer(self.tok)
                                            break
                                        print(f"[strata] script guard: let {piece!r} through at {n} tokens (after "
                                              f"{sg.rewinds} rewinds, or part of it was sent)", flush=True)
                                    sg.follow(was_pending, piece, detok.pending(), len(raw_ids) - 1)
                                tail = (tail + piece)[-2:]
                                evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                        reasoning_text += ev.text or ""
                                    if ev.kind in ("content", "tool_start", "tool_call"):
                                        answered = True
                                    yield "event", ev
                                if evs:
                                    last_out = time.perf_counter()
                                elif time.perf_counter() - last_out >= KEEPALIVE_S:
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                one_done = one_done or one_call and any(ev.kind == "tool_call" for ev in evs)
                                if stops is not None and stops.hit is not None or one_done:
                                    finish = "stop"         # gen.close() below STOPs the engine, as for a stop token
                                    break
                                if (self.reasoning_loop_recovery and not harmony and not recovery_count
                                        and parser.state == "reasoning"
                                        and n >= next_loop_check and not parser.buf and not parser.pending
                                        and not detok.pending()):
                                    next_loop_check = n + LOOP_CHECK_EVERY
                                    repeat_coverage = reasoning_repeat_coverage(reasoning_text)
                                    if repeat_coverage >= LOOP_COVERAGE:
                                        if self.reasoning_loop_recovery == "stop":
                                            looped = True        # #728: end the reply here, as for a repeated token
                                            break
                                        if not emb:              # "recover": not for a request with pictures
                                            recover_prompt = focused_recovery_prompt(self.tok, ids, raw_ids)
                                            if recover_prompt is not None:
                                                break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # at a clean point: no tag held back, no character split across tokens
                                    if thought >= budget and not parser.buf and not detok.pending():
                                        wrap = True
                                        break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and not parser.buf and not detok.pending():
                                        opens = True
                                        break
                                if (sg is not None and not sg.active and parser.state not in ("reasoning", "rcall")
                                        and not parser.buf and not detok.pending()):
                                    sg.active = sg.restart = True   # the thinking is over: the answer with the bias
                                    break
                        except EngineNonFinite:
                            nonfinite = True
                        except EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except ValueError as e:             # the engine's ERR line (it may have ended after it)
                            finish = "error"
                            print(f"[strata] the engine reported an error: {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away: an engine that never acknowledges
                            leaving = True                  # the STOP below is ended, but no error replaces this
                            raise
                        finally:
                            try:
                                # STOP+drain to THIS request's DONE while still holding the fifo, so a stop-token
                                # break can't leave the shared engine queue mid-drain for the next request to read as
                                # its own DONE (getattr: the engine's generate is typed as a plain Iterator)
                                getattr(gen, "close")()
                            except EngineSilent as e:   # #481: the STOP was never acknowledged: the engine is ended
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                            segment_done = getattr(self.engine, "last", None)
                            if segment_done is not None and segment_done is not segment_before:
                                segments.append(dict(segment_done))
                        act = stream.act if stream is not None and stream.finished else None
                        if stream is not None and not stream.finished:
                            # the pass ended for a reason of its own (a stop string, the budget, an error): the ids
                            # it held were never read, so the guard starts again from what the reply holds
                            stream.guard.rewind(raw_ids)
                        if nonfinite and not cancel.is_set():
                            # #879: the engine emitted none of the poisoned window's tokens and reads its next prompt
                            # from token 0.  Nothing sent yet: the same request once more, unseen by the client
                            if n == 0 and not nf_retried:
                                nf_retried = True
                                # the retry reads from token 0: its DONE, not this one's, says what was reused
                                del segments[segments_before:]
                                print("[strata] the engine's logits became non-finite before the first token "
                                      "(upstream Strata #879); it dropped its cached state: sending the request again "
                                      "once", flush=True)
                                continue
                            finish = "error"
                            if n == 0:
                                print("[strata] the engine's logits became non-finite again on the retry (upstream "
                                      "Strata #879): the request ends with an error", flush=True)
                                raise EngineNonFinite("the model's output became non-finite (NaN) twice in a row "
                                                      "(upstream Strata #879); the engine has dropped its cached "
                                                      "state - send the request again")
                            # an error, as vLLM and llama.cpp end a reply their engine broke off: a stream gets its
                            # error event after the text it already has, a whole answer is not sent; a "length"
                            # finish would read as a reply cut at max_tokens
                            print(f"[strata] the engine's logits became non-finite after {n} tokens (upstream Strata "
                                  "#879): the request ends with an error; the engine dropped its cached state, so the "
                                  "next request reads its prompt from the start", flush=True)
                            raise EngineNonFinite(f"the model's output became non-finite (NaN) after {n} tokens "
                                                  "(upstream Strata #879); the engine has dropped its cached state - "
                                                  "send the request again")
                        if (sg is not None and (sg.restart or sg.one_token) and act is None and finish == "length"
                                and n < max_new and not cancel.is_set()
                                and not (wrap or opens or repeated or looped or one_done)
                                and recover_prompt is None and (stops is None or stops.hit is None)):
                            prompt = prompt + seg       # the script guard's pass: the answer's, a rewind, one token
                            continue
                        if stream is not None and act is not None and not cancel.is_set():
                            g = stream.guard
                            # a loop: its repeats the client did not get are gone (never in seg); the engine goes on
                            # from what is kept - after the thinking closed, or with the short penalty - or it ends
                            extra = []
                            # (closed only where the parser is in the thinking too; else as the penalty)
                            closing = act.kind == loop_guard.CLOSE_THINKING and parser.state == "reasoning"
                            if act.kind == loop_guard.CLOSE_THINKING and not closing:
                                act = act._replace(kind=loop_guard.PENALTY)
                            if closing:
                                extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                                if max_new - n - len(extra) < 1:
                                    extra = []
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    g.follow(t)
                                    thinking_n += parser.state in ("reasoning", "rcall")
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                            g.resume(raw_ids, act)
                            self._loop_noted(act, n)
                            if trace is not None:
                                trace["loop_guard"] = g.summary()
                            if act.kind == loop_guard.STOP:
                                finish, guard_stopped = "stop", True
                                break
                            if (closing and not extra) or n >= max_new:
                                break                   # no room left to answer: "length", as without the guard
                            if stops is not None and stops.hit is not None:
                                finish = "stop"
                                break
                            prompt = self._next_pass(ids, prompt + seg + extra, raw_ids)
                            continue
                        if (stream is not None and stream.finished and n_call < max_new - n_pass and seg
                                and finish == "length" and n < max_new and not cancel.is_set()
                                and not (wrap or opens or repeated or looped or one_done)
                                and recover_prompt is None and (stops is None or stops.hit is None)):
                            prompt = prompt + seg       # the guard's penalty call ended: on with the request's own
                            continue
                        if recover_prompt is not None and not cancel.is_set() and n < max_new:
                            recovery_count += 1
                            # Only the two settings that keep the same words coming are raised (temperature to at
                            # least 1.0, presence penalty to at least 1.5): a client's top_p and top_k are never
                            # touched, and its own request dict is copied, not changed.
                            want = (max(1.0, float((sampling or {}).get("temperature") or 0)),
                                    max(1.5, float((sampling or {}).get("presence_penalty") or 0)))
                            if want != tuple(float((sampling or {}).get(k) or 0)
                                             for k in ("temperature", "presence_penalty")):
                                sampling = {**(sampling or {}), "temperature": want[0], "presence_penalty": want[1]}
                            prompt = recover_prompt
                            with self.status_lock:
                                st["reasoning_recoveries"] = recovery_count
                            if trace is not None:
                                trace["reasoning_recoveries"] = recovery_count
                                trace["reasoning_repeat_coverage"] = round(repeat_coverage, 3)
                            print(f"[strata] repeated reasoning detected at {n} tokens "
                                  f"(coverage={repeat_coverage:.3f}); resuming the same output with the low-effort "
                                  "instruction (reasoning_loop_recovery)", flush=True)
                            continue
                        if (self.reasoning_close_retry and thinking and not harmony and finish == "stop"
                                and not close_retried
                                and not answered and not wrap and not opens and parser.state == "reasoning"
                                and parser.buf != THINK_END         # #537: not after a held </think>, the end
                                and (stops is None or stops.hit is None) and not cancel.is_set()):
                            # #1053: the model wrote its reasoning and stopped before </think>: the client would get
                            # an empty answer.  Close the thinking once and let it answer.
                            close_retried = True
                            extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                            if max_new - n - len(extra) >= 1:
                                print("[strata] the reply ended inside its thinking with no answer: closing the "
                                      "thinking once and continuing (reasoning_close_retry)", flush=True)
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    thinking_n += parser.state in ("reasoning", "rcall")
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                                prompt = self._next_pass(ids, prompt + seg + extra, raw_ids)
                                finish = "length"
                                continue
                        if not (wrap or opens) or cancel.is_set():
                            break
                        # #123: the thinking reached reasoning_budget_tokens.  Close it the way the model would (a
                        # short wrap-up and </think>) and let it answer: the next pass's prompt is this one plus what
                        # was generated plus the wrap-up, so the engine continues from the prefix it already holds.
                        # A forced call (tool_choice) is opened the same way: after the wrap-up, or where the
                        # thinking ended, after the blank line the template puts before a call.
                        if wrap:
                            budget = st["thinking_budget"] = None
                            text = REASONING_WRAP_UP + (force or "")
                        else:
                            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
                        force = None
                        extra = self.tok.encode(text, parse_special=True)
                        if max_new - n - len(extra) < 1:
                            break                       # no room left to answer: "length", as without a budget
                        if wrap:
                            print(f"[strata] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state in ("reasoning", "rcall")
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = self._next_pass(ids, prompt + seg + extra, raw_ids)
                    if cancel.is_set():
                        finish = "cancel"
                    elif guard_stopped and not answered:
                        # the guard ended an answer that has no text yet: it says so instead of ending empty
                        for ev in cut([Event("content", LOOP_STOP_NOTE)]):
                            yield "event", ev
                    elif looped:
                        print(f"[strata] the thinking repeated the same passages (coverage={repeat_coverage:.3f}) at "
                              f"{n} tokens: ended as \"length\" (reasoning_loop_recovery: \"stop\" in "
                              "strata-<model>.json; remove it to turn this off)", flush=True)
                    elif repeated:
                        print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times in a row: ended as \"length\" (repeat_stop_tokens in "
                              "xestrata-<model>.json; 0 turns this off). If a new request with a short prompt does the "
                              "same, restart the server and report it (#606)", flush=True)
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # #266: settle this request's status, history and totals while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    with self.status_lock:
                        if st.get("busy"):
                            last = request_stats(segments)          # only this request's own DONEs count
                            started = st.get("started", time.time())
                            cvec = (getattr(self.engine, "info", {}) or {}).get("cvec", 0)
                            loaded = str(cvec) not in ("0", "", "None")
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # #588: the share the GPU read over PCIe (--pcie-frac) or another GPU computed - not in
                            # the hit rate's lookups, which is the VRAM share of the rest (None: an older engine)
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # #471: < len(ids) when cancelled mid-read
                            self.history.append({
                                "projection": (sampling or {}).get("experimental_speed_projection") is not False
                                if loaded else None,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                # the request's whole prompt, and the tokens read of it (None: an older engine)
                                "prompt_total": len(ids), "prompt_read": last.get("prompt_read"),
                                "reasoning_recoveries": recovery_count,
                                "reasoning_repeat_coverage": round(repeat_coverage, 3),
                                "loop_guard": guard.summary() if guard is not None and guard.events else None,
                                "engine_generated": last.get("generated"),
                                "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and (last.get("generated") or 0) >= 2 and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share, "ram_blobs": last.get("ram_blobs"),
                                "file_blobs": last.get("file_blobs"), "file_mb": last.get("file_mb"),
                                # #457: the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            tot = self.totals
                            tot["requests"] += 1
                            tot["prompt_tokens"] += seen
                            tot["reused"] += last.get("reused") or 0
                            tot["output_tokens"] += n
                            tot["prompt_ms"] += last.get("prompt_ms") or 0.0
                            tot["decode_ms"] += last.get("decode_ms") or 0.0
                            tot["drafts_offered"] += last.get("drafts_offered") or 0
                            tot["drafts_accepted"] += last.get("drafts_accepted") or 0
                            ft0 = st.get("first_token")             # this request's own (#465: one per request)
                            self.latencies.observe(ft0 - started if ft0 else None, n,
                                                   (last.get("decode_ms") or 0.0) / 1000, time.time() - started)
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            tok_s = answer_speed(last, n, now - ft if ft else None)
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            rate_txt = f"{tok_s:.1f} tok/s" if tok_s is not None else "no speed"
                            print(f"[strata] done: {n} tokens in {el:.0f} s ({rate_txt}) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish == "length" and parser.state in ("reasoning", "rcall"):   # #530
                                print("[strata] the reply reached max tokens while still thinking, so it has no "
                                      "answer: a thinking budget (reasoning_budget_tokens, in the request or in "
                                      "xestrata-<model>.json for every request) leaves room to answer", flush=True)
                            if os.environ.get("STRATA_DEBUG") and raw_ids:
                                print(f"[strata] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # #212: the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish(finish)):
            if one_call and ev.kind in ("tool_start", "tool_args", "tool_call"):
                if one_done:
                    continue                            # calls read from the thinking at the end: the first only
                one_done = ev.kind == "tool_call"
            yield "event", ev
        if parser.rescued or parser.refused:
            self.totals["tool_calls_from_reasoning"] = self.totals.get("tool_calls_from_reasoning", 0) + parser.rescued
            print(f"[strata] tool calls inside the thinking: {parser.rescued} read as calls, {parser.refused} kept as "
                  f"reasoning (quoted, or the turn did not end on a stop)", flush=True)
        stray = getattr(parser, "stray_think", 0)
        if stray:
            # #4077: the model wrote a closing think tag while thinking; the reasoning after it was sent as the answer
            with self.status_lock:
                self.totals["stray_think_closers"] = self.totals.get("stray_think_closers", 0) + stray
            print(f"[strata] {stray} extra closing think tag(s) in the answer were dropped (the model "
                  "wrote one while thinking, so the rest of its reasoning went out as the answer; a call after it "
                  "still counts)", flush=True)
        notes = list(getattr(parser, "notes", None) or [])
        if notes:
            with self.status_lock:
                self.totals["tool_args_repaired"] = self.totals.get("tool_args_repaired", 0) + len(notes)
            print(f"[strata] {len(notes)} tool call argument(s) repaired: {'; '.join(sorted(set(notes)))}", flush=True)
        unread = getattr(parser, "unread", 0)          # the Harmony reader keeps no such count
        rejected = []
        if unread:
            named = getattr(parser, "unread_names", None)
            print(f"[strata] {unread} <tool_call> block(s) in the answer did not read as a call of a declared "
                  "tool and went back as text (a name the request did not declare, or a form the template does "
                  "not use)" + (f"; the names the model wrote: {', '.join(named)}" if named else "")
                  + "; the response carries them as strata_rejected_tool_calls, /metrics counts them", flush=True)
            with self.status_lock:
                self.totals["tool_calls_unread"] = self.totals.get("tool_calls_unread", 0) + unread
            # #3796 (upstream 805ebe17b): the same names for the client, a machine-readable marker next to the text,
            # which stays as the model wrote it
            rejected = [{"name": nm, "reason": "undeclared"} for nm in named or []]
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string after all
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n,
                "reasoning_recoveries": recovery_count}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        if rejected:
            done["rejected_tool_calls"] = rejected
        yield "done", done


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """#471: the prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way (then the reused ones plus those read).  /metrics' history and totals count these, so a
    cancelled read is neither recorded as the whole prompt nor given a rate from tokens it never read.  An engine
    before 0.1.36 does not say (no `prompt_read`): the whole prompt, as before."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def carry_leg(carry: dict, leg: dict) -> None:
    """#1869: a request that decoded in a batch slot and went back to the solo path ends with the solo path's DONE,
    which replaces the engine's figures: keep the slot leg's decode time, tokens and prompt figures to add to it."""
    carry["decode_ms"] = carry.get("decode_ms", 0.0) + float(leg.get("decode_ms") or 0.0)
    carry["generated"] = carry.get("generated", 0) + int(leg.get("generated") or 0)
    carry.setdefault("prompt_ms", float(leg.get("prompt_ms") or 0.0))
    if leg.get("prompt_read") is not None:
        carry.setdefault("prompt_read", int(leg["prompt_read"]))


def add_carry(carry: dict, last: dict) -> dict:
    """The request's totals: the earlier legs' decode time and tokens plus the final one's; the prompt figures of the
    leg that read the prompt (the solo continuation reads next to nothing)."""
    out = {**last, "decode_ms": float(last.get("decode_ms") or 0.0) + carry.get("decode_ms", 0.0),
           "generated": int(last.get("generated") or 0) + carry.get("generated", 0),
           "prompt_ms": max(float(last.get("prompt_ms") or 0.0), carry.get("prompt_ms", 0.0))}
    if "prompt_read" in carry:
        out["prompt_read"] = carry["prompt_read"]
    return out


def answer_speed(last: dict, generated: int, wall_s: float | None = None) -> float | None:
    """The one definition of "answer speed": every token the model produced (thinking and answer, the tokens it kept -
    not the MTP drafts it threw away) over the decode time, first generated token to last.  The engine's DONE line holds
    both numbers; /metrics (`decode_tok_s`) and the timings (`predicted_per_second`) use them.
    Only an engine that sent no DONE falls back to the server's own clock from the first token, and a clock younger than
    RATE_MIN_SPAN_S says nothing (it printed 1000000 tok/s for a one-token reply).  Prompt reading is never in it."""
    produced, decode_ms = last.get("generated"), last.get("decode_ms")
    # one token has no first-to-last span (a 1-token probe printed 1,250 tok/s)
    if produced and produced >= 2 and decode_ms and decode_ms > 0:
        return round(produced / (decode_ms / 1000), 1)
    if generated and wall_s is not None and wall_s >= RATE_MIN_SPAN_S:
        return round(generated / wall_s, 1)
    return None


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    if last.get("prompt_read") is not None and last.get("finish") != "cancel":
        # #1870: the engine says how many prompt tokens it read: the slot's own session may have held the rest
        # (not the conversation cache), so the rate is not made of tokens it never read
        cache_n = max(cache_n, max(0, prompt_tokens - int(last["prompt_read"])))
    prompt_n, prompt_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"])
    decode_ms = float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported
            # them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def request_stats(segments: list[dict]) -> dict:
    """Keep input/cache accounting at the original API boundary and total native work across continuations."""
    if not segments:
        return {}
    result = dict(segments[-1])
    for key in ("reused", "prompt_read"):
        if key in segments[0]:
            result[key] = segments[0][key]
        else:
            result.pop(key, None)
    for key in ("prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted",
                "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb"):
        if any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[strata] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
class ToolHubs:
    """Hubs that run_with_mcp runs as one: the MCP servers' tools and the image tools (serve/images.py ImageTools)."""

    def __init__(self, hubs: list):
        self.hubs = hubs
        self.settings = {"max_rounds": max(int(h.settings["max_rounds"]) for h in hubs)}

    def routes(self) -> dict:
        out: dict = {}
        for h in self.hubs:
            out.update(h.routes())
        return out

    def call(self, name: str, arguments, cancel=None) -> dict:
        for h in self.hubs:
            if name in h.routes():
                return h.call(name, arguments, cancel)
        text = f"error: there is no tool named {name!r}"
        return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                "tool": name}


def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names, force=None):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round. After the last tool round, one final
    reply uses the results already obtained, with no tools offered or executed.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's).  `force` (forced_call): the first round's reply opens that call;
    the rounds after a tool ran are the model's own."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds = 0, 0
    done: dict = {}
    finalizing = False   # PR #3741 (upstream e0469a585): the round after the limit answers with no tools
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel, force=force, declared_only=True):
            if kind == "done":
                done = cast(dict, x)
                continue
            if kind == "event":
                ev = cast(Event, x)
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and not finalizing and done["finish"] == "stop" and not cancel.is_set()
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        force = None
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alive lines, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box: dict = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            partials = getattr(hub, "partials", None)   # an image tool's partial images (ImageTools.stream_partials)
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    while partials is not None and not partials.empty():
                        index, data = partials.get()
                        yield "mcp", {"event": "partial", "id": c.id, "index": index,
                                      "b64": base64.b64encode(data).decode("ascii")}
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            dbg = getattr(svc.request_trace, "debug", None)
            if dbg is not None:
                made = []                                # the images the tool made, saved beside the log
                log = cast(debuglog.DebugLog, svc.debug_log)
                for u in r.get("images") or []:
                    f = svc.images.file(u.rsplit("/", 1)[-1]) if isinstance(u, str) and svc.images is not None else None
                    try:
                        made.append(log.picture(f.read_bytes(), f.suffix[1:]) if f is not None else u)
                    except OSError:
                        made.append(u)
                dbg["tools"].append({"round": rounds, "name": c.name, "arguments": c.arguments, "ok": r["ok"],
                                     "ms": r["ms"], "result": r.get("model_content", r["text"]),
                                     **({"images": made} if made else {}),
                                     **({"image": r["image"]} if r.get("image") else {})})
            # an MCP tool's pictures (mcp.image_results, upstream 7c9b6f3a2) reach the model only with a picture reader
            results.append(r.get("model_content", r["text"]) if svc.vision is not None else r["text"])
            yield "mcp", {"event": "result", "id": c.id,
                          **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")},
                          **({"images": r["images"], "vision": svc.vision is not None} if r.get("images") else {}),
                          **({"image": r["image"]} if r.get("image") else {}),
                          **({"sources": r["sources"]} if "sources" in r else {})}   # a web search's (search.py)
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        if rounds >= max_rounds:
            finalizing = True
            tools = None
            # a user turn after the results, not a system message: the chat templates take a system message only at
            # the start ("System message must be at the beginning"), as _late_system_to_user turns a client's late one
            messages.append({"role": "user", "content": (
                f"The tool-use limit of {max_rounds} rounds has been reached. No tools are available for this final "
                "reply. Answer the user's original request in their language using the results already obtained. "
                "Explain any gaps in the available evidence. Do not request or emit further tool calls.")})
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
        try:
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, cancel=cancel)
        except ClientGone:                                      # #1999: the client left while a picture was prepared
            done = {**done, "finish": "cancel"}
            break
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
FILE_MAX = 32 << 20               # a file a request sends: 32 MiB, as OpenAI's and Anthropic's PDF inputs
PDF_PAGES_MAX = 100               # a PDF's pages, as Anthropic's limit
PDF_PAGE_SIDE = 1024              # a PDF page's picture: its longer side, in pixels (the reader shrinks it further
#                                   to its own image-token limit)
PDF_TOOL_S = 120                  # how long pdftotext or pdftoppm may take on one PDF
TEXT_TYPES = ("application/json", "application/xml", "application/x-yaml", "application/yaml",
              "application/javascript", "application/x-sh", "application/sql", "application/toml")


def _pdf_poppler(data: bytes, label: str, vision: bool) -> tuple[list, list]:
    """A PDF read with poppler's programs (Linux's poppler-utils, run as programs, not linked): pdftotext's text,
    one page per form feed, and pdftoppm's pages, their longer side PDF_PAGE_SIDE -> (texts, PNG data: URLs or
    None)."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.pdf"
        src.write_bytes(data)
        # one page past the limit, to know a PDF that has more
        r = subprocess.run(["pdftotext", "-enc", "UTF-8", "-l", str(PDF_PAGES_MAX + 1), str(src), "-"],
                           capture_output=True, timeout=PDF_TOOL_S)
        if r.returncode != 0:
            why = r.stderr.decode(errors="replace").strip().splitlines()
            raise ValueError(f"{label}: the PDF could not be read ({why[-1] if why else 'pdftotext failed'})")
        pages = r.stdout.decode("utf-8", errors="replace").split("\f")[:-1]
        if len(pages) > PDF_PAGES_MAX:
            raise ValueError(f"{label}: the PDF has more than {PDF_PAGES_MAX} pages; at most {PDF_PAGES_MAX} are read")
        pictures: list = [None] * len(pages)
        if vision and pages:
            r = subprocess.run(["pdftoppm", "-png", "-scale-to", str(PDF_PAGE_SIDE), "-l", str(len(pages)), str(src),
                                str(Path(d) / "p")], capture_output=True, timeout=PDF_TOOL_S)
            drawn = sorted(Path(d).glob("p-*.png"), key=lambda f: int(f.stem.rsplit("-", 1)[1]))
            if r.returncode != 0 or len(drawn) != len(pages):
                raise ValueError(f"{label}: the PDF's pages could not be drawn (pdftoppm)")
            pictures = ["data:image/png;base64," + base64.b64encode(f.read_bytes()).decode("ascii") for f in drawn]
    return pages, pictures


def _pdf_python(data: bytes, label: str, vision: bool) -> tuple[list, list]:
    """A PDF read with the Python packages (Windows' 7z carries them): pypdf's text page by page and pypdfium2's
    pages, their longer side PDF_PAGE_SIDE -> (texts, PNG data: URLs or None)."""
    try:
        import pypdf
    except ImportError:
        raise ValueError("reading a PDF needs poppler's pdftotext and pdftoppm (Linux: poppler-utils) or the "
                         "Python packages pypdf and pypdfium2") from None
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"{label}: the PDF is encrypted")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - pypdf's errors are many: a PDF it cannot read is the request's fault
        raise ValueError(f"{label}: the PDF could not be read ({type(e).__name__}: {e})") from None
    if len(pages) > PDF_PAGES_MAX:
        raise ValueError(f"{label}: the PDF has {len(pages)} pages; at most {PDF_PAGES_MAX} are read")
    pictures: list = [None] * len(pages)
    try:
        import pypdfium2
    except ImportError:                  # pypdf alone: the text alone
        if vision and not getattr(read_file, "said", False):
            read_file.said = True        # type: ignore[attr-defined]
            print("[strata] a PDF's pages are read as text only: drawing them as pictures needs poppler's pdftoppm "
                  "(Linux: poppler-utils) or the Python package pypdfium2", flush=True)
        vision = False
    if vision:
        try:
            pdf = pypdfium2.PdfDocument(data)
            for k in range(len(pdf)):
                page = pdf[k]
                image = page.render(scale=PDF_PAGE_SIDE / max(page.get_size())).to_pil()
                png = io.BytesIO()
                image.save(png, "PNG")
                pictures[k] = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")
            pdf.close()
        except Exception as e:  # noqa: BLE001 - as above
            raise ValueError(f"{label}: the PDF's pages could not be drawn ({type(e).__name__}: {e})") from None
    return pages, pictures


def read_file(data: bytes, mime: str | None, name: str, vision: bool) -> list[dict]:
    """A file -> the template's content items.  A PDF: its text page by page and, when the model reads pictures,
    each page as one, in page order - as OpenAI and Anthropic give a PDF to their models; read with poppler's
    programs when they are there (Linux), else with the Python packages (Windows).  A text file (text/*, JSON, XML,
    YAML, ...; or no type and UTF-8): its text.  Anything else, an encrypted or broken PDF, or more than
    PDF_PAGES_MAX pages is a ValueError (a 400)."""
    label = f"File: {name}" if name else "File"
    if data[:5] == b"%PDF-" or mime == "application/pdf":
        poppler = shutil.which("pdftotext") is not None and shutil.which("pdftoppm") is not None
        try:
            pages, pictures = (_pdf_poppler if poppler else _pdf_python)(data, label, vision)
        except subprocess.TimeoutExpired:
            raise ValueError(f"{label}: reading the PDF took more than {PDF_TOOL_S} s") from None
        items: list = [{"type": "text", "text": f"{label} ({len(pages)} pages)\n"}]
        for k, (text, picture) in enumerate(zip(pages, pictures)):
            items.append({"type": "text", "text": f"\n--- page {k + 1} ---\n{text.strip()}\n"})
            if picture is not None:
                items.append({"type": "image", "source": picture})
        return items
    if mime and not (mime.startswith("text/") or mime in TEXT_TYPES):
        raise ValueError(f"{label}: files of type {mime} are not read (text files and PDFs are)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{label}: the file is not UTF-8 text (text files and PDFs are read)") from None
    return [{"type": "text", "text": f"{label}\n\n{text}\n"}]


LOW_DETAIL_SIDE = 512             # OpenAI's image_url detail "low": the picture is read at 512 x 512 at most


def low_detail(source) -> str | bytes:
    """An image to read at OpenAI's low detail: its longer side brought down to LOW_DETAIL_SIDE (fewer image tokens),
    as PNG.  Without Pillow, or for an image already that small, the image as it is."""
    try:
        import io
        from PIL import Image
    except ImportError:
        return source
    data = source if isinstance(source, bytes) else Vision.load(source)
    try:
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) <= LOW_DETAIL_SIDE:
                return data
            im.thumbnail((LOW_DETAIL_SIDE, LOW_DETAIL_SIDE))
            out = io.BytesIO()
            im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(out, "PNG")
            return out.getvalue()
    except (OSError, ValueError):            # not a picture Pillow reads: the reader says what is wrong with it
        return data


def check_logprobs(req: dict) -> None:
    """OpenAI's limits on logprobs: top_logprobs a whole number 0..20, and only with logprobs: true."""
    k = req.get("top_logprobs")
    if k is None:
        return
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 20:
        raise ValueError("top_logprobs must be a whole number from 0 to 20")
    if req.get("logprobs") is not True:
        raise ValueError("top_logprobs needs logprobs: true")


def system_fingerprint(svc) -> str:
    """OpenAI's system_fingerprint: names the engine's build and the model, so it changes when either does."""
    version = (getattr(svc.engine, "info", None) or {}).get("version")
    return "fp_" + hashlib.sha256(f"{version}|{svc.model}".encode()).hexdigest()[:12]


def top_logprobs(req: dict):
    """The K of the engine's logprobs=K, or None when the request did not ask (OpenAI: logprobs=true, top_logprobs
    0..20; the legacy completions form, logprobs=N, is taken as N)."""
    lp = req.get("logprobs")
    if lp is True:
        k = req.get("top_logprobs")
        return max(0, min(20, k)) if isinstance(k, int) and not isinstance(k, bool) else 0
    if isinstance(lp, int) and not isinstance(lp, bool) and lp >= 0:
        return min(20, lp)
    return None


def allowed_ids(req: dict) -> list:
    """`allowed_token_ids`: the ids a caller named, or [] (vLLM's field; see LogprobEntries)."""
    ids = req.get("allowed_token_ids")
    if not isinstance(ids, (list, tuple)):
        return []
    return [x for x in ids if isinstance(x, int) and not isinstance(x, bool) and x >= 0]


def batch_slots_on(engine) -> bool:
    """Does this engine run requests in batch slots ("parallel", whose `BT` lines carry no LP lines)?  Asked of each
    engine behind a ReplicaEngine: a server with `--replicas 2` and no `--parallel` has none."""
    return any(int(getattr(e, "batch", 0) or 0) for e in (getattr(engine, "engines", None) or [engine]))


class LPLines(collections.deque):
    """One request's (token id, LP line) pairs, tagged with its cancel Event (StrataEngine.own_lp_lines)."""

    def __init__(self, owner=None):
        super().__init__()
        self.owner = owner


class LogprobEntries:
    """The engine's LP lines as OpenAI's logprobs.content entries (upstream 9ea044962, #3200).

    Each token's "token" is the text it completes (an incremental UTF-8 decode, as the reply's text is made), so a
    character split across tokens lands in the token that finishes it and the tokens, joined, spell the reply.
    "bytes" are the token's own bytes.  The stop token ends the reply and is not listed (OpenAI does not list it
    either), and a token that makes no text of the reply is left out: spell() gives each piece of text the tokens that
    spell it.  A line whose values are not finite ("nan", MSVC's "-nan(ind)", a cut line) gives no entry.

    `allowed_token_ids`: only those ids are kept, renormalised over them, so a caller asking about specific candidates
    gets a distribution over exactly those.  vLLM, which named the field, masks generation with them; here they filter
    what is reported, so the chosen token can be outside them.
    """

    def __init__(self, svc, allowed=(), owner=None):
        self.tok, self.stop_ids = svc.tok, svc.stop_ids
        self.allowed = list(allowed)
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending: list = []                    # entries read but not yet given to a piece of text
        self.owner = owner                         # the request's cancel Event (LPLines)
        self.lines = None                          # this request's current LPLines, kept once seen

    def spell(self, eng, text: str) -> list:
        """The entries whose tokens spell `text`, the next piece of the reply."""
        self.pending += self.take(eng)
        at = [0]
        for e in self.pending:
            at.append(at[-1] + len(e["token"]))
        joined = "".join(e["token"] for e in self.pending)
        i = joined.find(text)
        while i >= 0 and text:
            if i in at:
                a = at.index(i)                    # the first token there
                end = i + len(text)
                b = next((j for j in range(a + 1, len(at)) if at[j] == end), None)
                if b is not None:
                    out, self.pending = self.pending[a:b], self.pending[b:]
                    return out
            i = joined.find(text, i + 1)
        out, self.pending = self.pending, []
        return out

    def raw(self, t):
        if hasattr(self.tok, "token_bytes"):
            return self.tok.token_bytes(t)
        return self.tok.decode([t]).encode("utf-8")

    def take(self, eng) -> list:
        # the request's own lines (on its cancel Event), else the engine's when they carry its Event; a later generate
        # pass of the same request (an MCP round) makes new ones, and the old ones are read to the end first
        own = getattr(self.owner, "_strata_lp_lines", None)
        cur, old = (own if own is not None else getattr(eng, "lp_lines", None)), None
        if cur is not self.lines and getattr(cur, "owner", None) is self.owner:
            old, self.lines = self.lines, cur
        out = []
        for lines in (old, self.lines):
            while lines:
                t, line = lines.popleft()
                if t is None or t in self.stop_ids:
                    continue
                f = line.split()
                if len(f) < 2:
                    continue
                try:
                    lp = float(f[1])
                    alts = [(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
                except ValueError:
                    continue
                if not math.isfinite(lp) or not all(math.isfinite(v) for _, v in alts):
                    continue
                b = self.raw(t)
                e = {"token": self.inc.decode(b), "logprob": lp, "bytes": list(b), "top_logprobs": []}
                if self.allowed:
                    alts = self._renormalised(t, lp, alts)
                    e["logprob"] = next((v for i, v in alts if i == t), lp)
                for i, v in alts:
                    alt = self.raw(i)
                    e["top_logprobs"].append({"token": alt.decode("utf-8", "replace"), "logprob": v,
                                              "bytes": list(alt)})
                out.append(e)
        return out

    def _renormalised(self, t, lp, alts):
        """Only the named ids, normalised over them so they sum to 1, most likely first.  The line carries the chosen
        token's own logprob on its head and the named ids after it; the chosen token is put back among them when the
        caller named it and the line did not list it (named alone, it is the whole distribution: logprob 0, as vLLM's
        mask gives).  A line with none of the named ids gives no alternatives, and the entry keeps the chosen token's
        own logprob: it is never dropped, since the reply's tokens have to keep spelling its text."""
        alts = [(i, v) for i, v in alts if i in self.allowed]
        if t in self.allowed and all(i != t for i, _ in alts):
            alts = [(t, lp)] + alts
        if not alts:
            return alts
        mx = max(v for _, v in alts)
        lse = mx + math.log(sum(math.exp(v - mx) for _, v in alts))
        return sorted(((i, v - lse) for i, v in alts), key=lambda x: -x[1])


# The engine's last PP line stops short of the prompt: a short last part (`--short-read` tokens, or as many as the
# windows read faster on this machine: from 64, at most 256) is read through the verify windows instead of the batched
# path, so the batched path never reports the whole prompt.  A PP that close to the end says the prompt is read.  It
# matters because a prompt shorter than one `--prefill` chunk sends exactly one PP, that one: a client would see a bar
# that appears already full.  Such a line is dropped, and the answer's first token ends the progress for the client
# anyway.
PP_DONE_TAIL = 64


def prompt_progress(svc: Service):
    """The last PP line as llama.cpp's `prompt_progress`, so a client written for llama.cpp reads both servers: the
    same four fields, `time_ms` elapsed since prompt reading started, and `processed` counting from token 0 so that a
    client takes the reused prefix out with `cache` (llama.cpp's rule: the timed progress is
    `(processed-cache)/(total-cache)`).  None when there is nothing to say: no PP yet, or the line that says the prompt
    is read.  An engine too old to print RESUME gives `cache` 0, which only makes a reused prefix count as work."""
    engine = svc.engine
    if getattr(engine, "batch", 0):      # --batch: the engine's progress is one global line, not this request's
        return None
    progress = getattr(engine, "progress", None)
    if not progress:
        return None
    read, total = progress
    if total <= 0 or total - read <= PP_DONE_TAIL:
        return None
    return {"total": int(total), "cache": int(getattr(engine, "reused", 0) or 0), "processed": int(read),
            "time_ms": int(getattr(engine, "progress_ms", 0) or 0)}


def legacy_chunks(chunks):
    """A stream's chunks for a `functions` request (frontend.legacy_functions): each choice's call as the old
    function_call (legacy_function_call).  Closing it closes the stream it reads."""
    try:
        for c in chunks:
            for choice in (c or {}).get("choices") or ():
                legacy_function_call(choice)
            yield c
    finally:
        chunks.close()


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None, force=None):
    """With `logprobs` asked (upstream's Intel port 4ba35fd), each chunk carries the log-probabilities of the tokens
    the engine wrote since the last one (the model's, before sampling, temperature and penalties; the stop token is
    not listed), under `content` or, while it thinks, `reasoning_content`."""
    chunks = _openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run, force)
    if top_logprobs(req) is None:
        yield from chunks
        return
    lp = LogprobEntries(svc, allowed_ids(req), cancel)
    try:
        for c in chunks:
            if c and c.get("choices"):
                d = c["choices"][0].get("delta") or {}
                # the entries whose tokens spell this chunk's text (upstream 9ea044962): the thinking's under
                # reasoning_content, the answer's under content
                for key in ("reasoning_content", "content"):
                    if d.get(key):
                        got = lp.spell(svc.engine, d[key])
                        if got:
                            c["choices"][0].setdefault("logprobs", {})[key] = got
            yield c
    finally:
        chunks.close()                              # a client that went: the run stops now, as without logprobs


def _openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None, force=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads.  `force`: see Service.run."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model_for(req)
    # llama.cpp's flag, and its default: a client that wants prompt progress asks for it, and nothing changes for one
    # that does not.  Strata already sends a keep-alive comment on every PP line, so this only fills that line in.
    want_progress = bool(req.get("return_progress"))
    fingerprint = system_fingerprint(svc)
    customs = custom_tool_names(req)              # their calls: {"type": "custom", "custom": {"name", "input"}}

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "system_fingerprint": fingerprint,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came (#211)
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel, force=force,
                                                       declared_only=True):
        if kind == "ping":
            # a PP line: the prompt is being read and there is nothing to say yet.  It used to be only the SSE comment;
            # now it carries the progress when the client asked, and stays the comment when there is nothing to report.
            pp = prompt_progress(svc) if want_progress else None
            yield None if pp is None else {**chunk({}), "prompt_progress": pp}
        elif kind == "mcp":
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev = cast(Event, x)
            call = cast(ToolCall, ev.call)          # (every tool_* event carries its call)
            if ev.kind == "reasoning" and ev.text:
                # llama.cpp's reasoning_content and vLLM's reasoning: the same text, for a client that reads either
                yield chunk({"reasoning_content": ev.text, "reasoning": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind in ("tool_start", "tool_args") and call.name in customs:
                pass                                # a custom tool's input is sent whole, with its tool_call
            elif ev.kind == "tool_call" and call.name in customs:
                given = call.arguments
                text = given["input"] if isinstance(given, dict) and set(given) == {"input"} \
                    and isinstance(given["input"], str) else json.dumps(given, ensure_ascii=False)
                yield chunk({"tool_calls": [{"index": calls, "id": call.id, "type": "custom",
                                             "custom": {"name": call.name, "input": text}}]})
                calls += 1
            elif ev.kind == "tool_start":
                streamed[call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[call.id], "id": call.id, "type": "function",
                                             "function": {"name": call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and call.id in streamed:
                finished.add(getattr(ev.call, "id", None))
            elif ev.kind == "tool_call":
                args = json.dumps(call.arguments, ensure_ascii=False)
                yield chunk({"tool_calls": [{"index": calls, "id": call.id, "type": "function",
                                             "function": {"name": call.name, "arguments": args}}]})
                calls += 1
        else:
            # a streamed call without its final tool_call is one the output ended inside: not "tool_calls" (#211)
            whole = calls and streamed.keys() <= finished
            fin = cast(dict, x)                     # the run's last item: the DONE figures
            ended = fin["finish"]
            finish = "tool_calls" if whole and ended == "stop" else {"cancel": "stop"}.get(ended, ended)
            last = chunk({}, finish)
            pt = fin.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": fin["completion_tokens"],
                             "total_tokens": pt + fin["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": fin.get("reused") or 0},
                             # the thinking's share of completion_tokens
                             "completion_tokens_details": {"reasoning_tokens": fin.get("reasoning_tokens") or 0}}
            if fin.get("timings"):
                last["timings"] = fin["timings"]          # llama.cpp's field: the speed its clients show
            if fin.get("rejected_tool_calls"):          # #3796: only when a call was refused as undeclared
                last["strata_rejected_tool_calls"] = fin["rejected_tool_calls"]
            yield last


def openai_choices(n: int, choice):
    """The chunks of n choices (`choice(k)`: choice k's chunks), one choice after another, each chunk under its
    choice's index.  The usage is the request's: the prompt once, every choice's completion tokens, on the last
    choice's finishing chunk.  Every chunk carries the first choice's id and created, as OpenAI's chunks of one
    completion do.  A choice that fails ends the whole request."""
    total: dict | None = None
    first: tuple | None = None
    for k in range(n):
        chunks = choice(k)
        try:
            for c in chunks:
                if c is not None and "id" in c:
                    if first is None:
                        first = c["id"], c.get("created")
                    c["id"], c["created"] = first
                if c is None or not c.get("choices"):
                    yield c
                    continue
                c["choices"][0]["index"] = k
                if "usage" in c:
                    u = c.pop("usage")
                    if total is None:
                        total = u
                    else:
                        done = total["completion_tokens"] + u["completion_tokens"]
                        thought = sum((x.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                                      for x in (total, u))
                        total = {**total, "completion_tokens": done, "total_tokens": total["prompt_tokens"] + done,
                                 "completion_tokens_details": {"reasoning_tokens": thought}}
                    if k == n - 1:
                        c["usage"] = total
                yield c
        finally:
            chunks.close()


def openai_collect_choices(chunks) -> dict:
    """openai_collect for n choices: each choice's chunks collected on their own, then one response."""
    groups: dict = {}
    for c in chunks:
        if c is not None:
            groups.setdefault(c["choices"][0]["index"], []).append(c)
    outs = [openai_collect(groups[k]) for k in sorted(groups)]
    out = outs[-1]
    out["choices"] = [{**o["choices"][0], "index": k} for k, o in zip(sorted(groups), outs)]
    return out


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def cite_chunks(chunks, search_tools):
    """The chunks, with one more before the finishing one: the url_citation annotations of the addresses the answer
    writes from what the web search found (OpenAI's message.annotations)."""
    text: list[str] = []
    for c in chunks:
        choice = c["choices"][0] if c is not None and c.get("choices") else None
        if choice is not None:
            text.append(choice["delta"].get("content") or "")
            if choice.get("finish_reason") is not None:
                notes = [{"type": "url_citation", "url_citation": {"start_index": x["start"], "end_index": x["end"],
                                                                   "url": x["url"], "title": x["title"]}}
                         for x in search_api.cited("".join(text), search_tools.sources)]
                if notes:
                    yield {**{k: c[k] for k in ("id", "object", "created", "model", "system_fingerprint") if k in c},
                           "choices": [{"index": 0, "delta": {"annotations": notes}, "finish_reason": None}]}
        yield c


def openai_collect(chunks) -> dict:
    content, reasoning, last, mcp = [], [], None, []
    notes: list[dict] = []                         # a web search's url_citation annotations (cite_chunks)
    by_index: dict = {}
    lps: dict[str, list] = {"content": [], "reasoning_content": []}
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        for k, v in (c["choices"][0].get("logprobs") or {}).items():
            lps[k].extend(v)
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        notes += d.get("annotations") or []
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            if tc.get("type") == "custom":         # a custom tool's call comes whole
                by_index[tc.get("index", len(by_index))] = {k: v for k, v in tc.items() if k != "index"}
                continue
            cur = by_index.setdefault(tc.get("index", len(by_index)),
                                      {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    assert last is not None                         # openai_chunks always ends with the finishing chunk
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if c.get("type") == "custom"
                 or _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    msg: dict = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = msg["reasoning"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if notes:
        msg["annotations"] = notes
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           **({"system_fingerprint": last["system_fingerprint"]} if "system_fingerprint" in last else {}),
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last.get("usage")}
    if last.get("timings"):
        out["timings"] = last["timings"]
    if last.get("strata_rejected_tool_calls"):
        out["strata_rejected_tool_calls"] = last["strata_rejected_tool_calls"]
    if lps["content"] or lps["reasoning_content"]:
        out["choices"][0]["logprobs"] = {"content": lps["content"],
                                         **({"reasoning_content": lps["reasoning_content"]}
                                            if lps["reasoning_content"] else {})}
    return out


def logprob_span(text: str, part: str, entries: list):
    """The entries whose tokens spell `part`, a piece of `text`, or None when the tokens do not spell `text` or no
    token boundary falls at both ends of `part`."""
    if "".join(e["token"] for e in entries) != text:
        return None
    start = text.find(part)
    if start < 0:
        return None
    at, first = 0, None
    for i, e in enumerate(entries):
        if at == start and first is None:
            first = i
        if at == start + len(part) and first is not None:
            return entries[first:i]
        at += len(e["token"])
    return entries[first:] if first is not None and at == start + len(part) else None


def _same_json(a: str, b: str) -> bool:
    """Are two strings the same JSON value (whitespace and key order aside)?"""
    try:
        return json.loads(a) == json.loads(b)
    except ValueError:
        return False


def _structured_logprobs(choice: dict, content: str):
    """(content, logprobs) of a validated json_schema / json_object reply (upstream 9ea044962, #3200).

    structured_chunks returns the canonical JSON the validator made, which the model never wrote, so its tokens do not
    spell it.  A run of tokens that spells what is returned is used as it is; tokens that spell the model's own JSON,
    the same JSON value (the fenced-with-spaces case, the common one), return the model's form, so the joined tokens
    are the content again.  Otherwise no logprobs: a caller joining the tokens must not get something that differs."""
    lps = choice.get("logprobs")
    if not lps or not lps.get("content"):
        return content, None
    raw = choice["message"]["content"] or ""
    entries = lps["content"]
    if raw == content:
        return content, _with_reasoning(lps, entries)
    span = logprob_span(raw, content, entries)
    if span is not None:
        return content, _with_reasoning(lps, span)
    inner = extract_json(raw)
    if inner and _same_json(inner, content):
        span = logprob_span(raw, inner, entries)
        if span is not None:
            return inner, _with_reasoning(lps, span)
    return content, None


def _with_reasoning(lps: dict, entries: list) -> dict:
    out = {"content": entries}
    if lps.get("reasoning_content"):
        out["reasoning_content"] = lps["reasoning_content"]
    return out


def structured_chunks(chunks, validator):
    """Buffer structured streams so a client never receives unvalidated content."""
    buffered = []
    heartbeat = time.monotonic()
    try:
        for chunk in chunks:
            progress = chunk is not None and "prompt_progress" in chunk
            if chunk is not None and not progress:
                buffered.append(chunk)      # a progress chunk carries no content: there is nothing in it to validate
            if chunk is None or progress or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield chunk if progress else None
        result = openai_collect(buffered)
        choice = result["choices"][0]
        content = validated_json(choice["message"]["content"], validator, choice["finish_reason"])
        # logprobs: the entries spell what the model wrote, which is replaced here with the canonical JSON;
        # _structured_logprobs gives back the form the tokens spell, or None
        content, logprobs = _structured_logprobs(choice, content)
        yield buffered[0]
        delta = {"content": content}
        if choice["message"].get("reasoning_content"):
            delta["reasoning_content"] = delta["reasoning"] = choice["message"]["reasoning_content"]
        piece = {"index": 0, "delta": delta, "finish_reason": None}
        if logprobs:
            piece["logprobs"] = logprobs
        yield {**buffered[0], "choices": [piece]}
        end = buffered[-1]
        if end["choices"][0].get("logprobs"):       # those went out in the piece above (or were dropped with it)
            end = {**end, "choices": [{k: v for k, v in end["choices"][0].items() if k != "logprobs"}]}
        yield end
    finally:
        chunks.close()


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_usage(ids, reused, output_tokens: int) -> dict:
    """Anthropic's usage of a prompt the conversation cache held `reused` tokens of: those are cache reads, the rest
    a cache write (it stays for the next request); input_tokens keeps the last token, as 0 makes Claude Code count
    the prompt twice (#1013)."""
    reused = max(0, min(reused or 0, len(ids) - 1))
    return {"input_tokens": 1, "cache_creation_input_tokens": len(ids) - reused - 1,
            "cache_read_input_tokens": reused, "output_tokens": output_tokens}


def anthropic_search_blocks(x: dict, calls: dict) -> list[dict]:
    """A web search tool's activity (run_with_mcp's "mcp" events) as Anthropic's server tool blocks: its call a
    server_tool_use block (web_search or web_fetch), its result a web_search_tool_result (or web_fetch_tool_result)
    block.  `calls`: call id -> the tool's name, filled here.  The results' encrypted_content is the snippet, in
    base64: this server keeps nothing to decrypt it with, and a client only sends it back."""
    sid = "srvtoolu_" + str(x.get("id", "")).removeprefix("call_")
    if x.get("event") == "call":
        raw = x.get("arguments")
        args: dict = raw if isinstance(raw, dict) else {}
        fetch = x.get("name") == "fetch_page"
        calls[x.get("id")] = "web_fetch" if fetch else "web_search"
        return [{"type": "server_tool_use", "id": sid, "name": calls[x.get("id")],
                 "input": {"url": args.get("url")} if fetch else {"query": args.get("query")}}]
    if x.get("event") != "result" or x.get("id") not in calls:
        return []
    if calls[x.get("id")] == "web_fetch":
        src = (x.get("sources") or [{}])[0]
        content: dict | list = (
            {"type": "web_fetch_result", "url": src.get("url"), "content": {
                "type": "document", "title": src.get("title"),
                "source": {"type": "text", "media_type": "text/plain", "data": x.get("text", "")}}}
            if x.get("ok") else {"type": "web_fetch_tool_error", "error_code": "unavailable"})
        return [{"type": "web_fetch_tool_result", "tool_use_id": sid, "content": content}]
    content = ([{"type": "web_search_result", "url": r["url"], "title": r["title"], "page_age": None,
                 "encrypted_content": base64.b64encode(r["url"].encode()).decode("ascii")}
                for r in x.get("sources") or []] if x.get("ok") else
               {"type": "web_search_tool_result_error", "error_code": "unavailable"})
    return [{"type": "web_search_tool_result", "tool_use_id": sid, "content": content}]


def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None, run=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp with the web search tools): their activity
    becomes Anthropic's server tool blocks (anthropic_search_blocks) and usage.server_tool_use."""
    mid = "msg_" + uuid.uuid4().hex[:24]

    def start(reused):
        # sent with the reply's first piece, once the engine has said how much of the prompt its cache held (its
        # RESUME line), so message_start's usage names the cache reads and writes as message_delta's does; until
        # then the heartbeats go out as pings
        return "message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": svc.model_for(req), "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": anthropic_usage(ids, reused, 0)}}
    started = False
    index, open_kind, used_tool = -1, None, False
    thought: list[str] = []                        # the open thinking block's text: its signature at its end

    def closing():
        if open_kind == "thinking":
            yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                "type": "signature_delta", "signature": thinking_signature("".join(thought))}}
            thought.clear()
        yield "content_block_stop", {"type": "content_block_stop", "index": index}

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came (#211)
    server_calls: dict = {}                        # the web search calls run here: id -> web_search / web_fetch
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel, force=force,
                                                       declared_only=True):
        if kind == "ping":
            yield None
            continue
        if not started:
            started = True
            yield start(cast(dict, x).get("reused") if kind == "done" else
                        0 if getattr(svc.engine, "batch", 0) else getattr(svc.engine, "reused", 0))
        if kind == "mcp":
            for server_block in anthropic_search_blocks(cast(dict, x), server_calls):
                if open_kind is not None:
                    yield from closing()
                index += 1
                open_kind = server_block["type"]
                if open_kind == "server_tool_use":       # its input as one piece, as a tool_use's
                    yield "content_block_start", {"type": "content_block_start", "index": index,
                                                  "content_block": {**server_block, "input": {}}}
                    yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(server_block["input"], ensure_ascii=False)}}
                else:
                    yield "content_block_start", {"type": "content_block_start", "index": index,
                                                  "content_block": server_block}
            continue
        if kind == "event":
            ev = cast(Event, x)
            call = cast(ToolCall, ev.call)          # (None but for the tool_* events)
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and call.id in streamed:
                finished.add(getattr(ev.call, "id", None))
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use",
                    "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield from closing()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         # Anthropic's ids start "toolu_"; the client sends it back as tool_result.tool_use_id
                         "tool_use": {"type": "tool_use", "id": "toolu_" + ev.call.id.removeprefix("call_")
                                      if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                thought.append(ev.text)
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield from closing()
            assert isinstance(x, dict)                  # the run's last item: the DONE figures
            asked = req.get("max_tokens")
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                "tool_use" if used_tool and streamed <= finished and x["finish"] == "stop" else \
                "model_context_window_exceeded" if x["finish"] == "length" and isinstance(asked, int) \
                and x["completion_tokens"] < asked else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts: the engine's own figure of the prompt its cache held
            usage: dict = anthropic_usage(ids, x.get("reused"), x["completion_tokens"])
            if server_calls:
                usage["server_tool_use"] = {
                    "web_search_requests": sum(v == "web_search" for v in server_calls.values()),
                    "web_fetch_requests": sum(v == "web_fetch" for v in server_calls.values())}
            delta_event = {"type": "message_delta",
                           "delta": {"stop_reason": stop, "stop_sequence": x.get("stop_sequence")}, "usage": usage}
            if x.get("rejected_tool_calls"):
                delta_event["strata_rejected_tool_calls"] = x["rejected_tool_calls"]     # #3796
            yield "message_delta", delta_event
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d["signature"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            raw = b.pop("_json") or "{}"
            try:
                b["input"] = json.loads(raw)
            except ValueError:                     # a call the output ended inside (#211): what it wrote, as text
                blocks[-1] = {"type": "text", "text": raw}
        elif name == "message_delta":
            assert msg is not None
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
            if e.get("strata_rejected_tool_calls"):
                msg["strata_rejected_tool_calls"] = e["strata_rejected_tool_calls"]
    assert msg is not None                          # anthropic_events always starts with message_start
    msg["content"] = blocks
    return msg


def anthropic_prepare_format(req: dict, messages, tools):
    """prepare_format for Anthropic's output_config.format -> (messages, validator or None); its errors name the
    request's own field."""
    try:
        return prepare_format(anthropic_format(req), messages, with_tools=bool(tools))
    except ValueError as e:
        raise ValueError(str(e).replace("response_format", "output_config.format")) from None


def anthropic_structured(events, validator):
    """anthropic_events for a structured answer (output_config.format), buffered so a client never receives
    unvalidated text: the text blocks are checked against the schema and sent as one, after the thinking.  An answer
    that calls a tool goes out as written: the schema is for the final answer (as on /v1/responses, #782)."""
    buffered = []
    heartbeat = time.monotonic()
    try:
        for item in events:
            if item is not None:
                buffered.append(item)
            if item is None or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield None
        msg = anthropic_collect(buffered)
        if msg["stop_reason"] == "tool_use":
            yield from buffered
            return
        text = "".join(b["text"] for b in msg["content"] if b["type"] == "text")
        finish = "length" if msg["stop_reason"] == "max_tokens" else "stop"
        content = validated_json(text, validator, finish)
        index, in_text = 0, False
        for name, e in buffered:
            if name == "content_block_start":
                in_text = e["content_block"]["type"] == "text"
            if name.startswith("content_block_"):
                if not in_text:
                    yield name, {**e, "index": index}
                    index += name == "content_block_stop"
                elif name == "content_block_stop":
                    in_text = False
                continue
            if name == "message_delta":
                yield "content_block_start", {"type": "content_block_start", "index": index,
                                              "content_block": {"type": "text", "text": ""}}
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": content}}
                yield "content_block_stop", {"type": "content_block_stop", "index": index}
            yield name, e
    finally:
        events.close()


# ------------------------------------------------------------------------------------------------ HTTP
# Anthropic's error types by HTTP status (its "errors" page); a 5xx without its own is an api_error
ANTHROPIC_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                         404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
                         529: "overloaded_error"}


def api_error_body(path: str, status: int, obj):
    """An error body in the shape of the API its path belongs to: Anthropic's routes (/v1/messages...) answer
    {"type": "error", "error": {"type", "message"}}, the other /v1/ routes OpenAI's {"error": {"message", "type",
    "param", "code"}} (param and code null when the error has none).  Anything else is left as it is."""
    err = obj.get("error") if isinstance(obj, dict) else None
    if status < 400 or not isinstance(err, dict) or not path.startswith("/v1/"):
        return obj
    message = str(err.get("message") or "")
    if path.startswith("/v1/messages"):
        if obj.get("type") == "error":
            return obj
        kind = ANTHROPIC_ERROR_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error")
        return {"type": "error", "error": {"type": kind, "message": message}}
    kind = err.get("type") or ("server_error" if status >= 500 else "invalid_request_error")
    return {**obj, "error": {**err, "message": message, "type": kind, "param": err.get("param"),
                             "code": err.get("code")}}


CHUNKED_BODY_MAX = 256 << 20                # #893: the most a Transfer-Encoding: chunked body may hold (in memory)


def body_limit() -> int:
    """The most a request body may hold, in bytes (read into memory): 256 MiB, which is a million-token conversation
    with room to spare; STRATA_MAX_BODY_MIB changes it (0 or an unreadable value: the default).  A larger one is
    answered 413 before it is read (a Content-Length of 10 TB used to be read until the client gave up)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return mib << 20 if mib > 0 else CHUNKED_BODY_MAX


def parse_multipart(ctype: str, body: bytes) -> tuple[dict, list[bytes]]:
    """A multipart/form-data body (OpenAI's /v1/images/edits) -> (its text fields, the bytes of its image or image[]
    files in order); the numbers the JSON request takes as numbers (n, seed, steps, num_inference_steps,
    output_compression, partial_images whole; vLLM-Omni's guidance_scale and true_cfg_scale) as numbers, stream's true
    or false (as OpenAI's SDK writes a boolean in a form) as a bool, the mask file's bytes as "mask"."""
    from email.parser import BytesParser
    from email.policy import HTTP
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("the multipart/form-data body could not be read")
    fields: dict = {}
    files: list[bytes] = []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        raw = part.get_payload(decode=True)
        data = raw if isinstance(raw, bytes) else b""
        if name in ("image", "image[]"):
            files.append(data)
        elif name == "mask":
            fields["mask"] = data
        elif isinstance(name, str) and name:
            value = data.decode("utf-8", "replace")
            whole = name in ("n", "seed", "steps", "num_inference_steps", "output_compression", "partial_images") \
                and value.strip().isdigit()
            if name == "stream" and value.strip().lower() in ("true", "false"):
                fields[name] = value.strip().lower() == "true"
            elif name in ("guidance_scale", "true_cfg_scale"):
                try:
                    fields[name] = float(value)          # out of range or not finite: the request's own 400
                except ValueError:
                    fields[name] = value
            else:
                fields[name] = int(value) if whole else value
    return fields, files


class BadBody(Exception):
    """A request body that cannot be read (a malformed or oversized chunked body): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        debug = None                                        # this request's debug chat log record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()

        def send_response(self, code, message=None):
            self.answer_started = True                 # a malformed-request answer can only replace one not yet begun
            super().send_response(code, message)

        def _chunked(self) -> bool:
            """#893: a body sent as Transfer-Encoding: chunked (a relay or proxy that does not buffer it).  By RFC 9112
            it wins over a Content-Length."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read_chunked(self, limit: int, keep: bool = True, deadline: float | None = None) -> bytes:
            """Decode a chunked body: at most `limit` bytes (BadBody 413 beyond it - the size is checked before a chunk
            is read, so a huge announced chunk is never allocated), BadBody 400 for a malformed one.  keep=False reads
            and drops it (the drain); `deadline` (monotonic) ends the read early."""
            parts: list[bytes] = []
            total = 0
            while True:
                if deadline is not None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        return b"".join(parts)
                    self.connection.settimeout(wait)
                line = self.rfile.readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, including the final blank line
                        trailer = self.rfile.readline(8193)
                        if trailer == b"\r\n":
                            return b"".join(parts)
                        if not trailer.endswith(b"\r\n"):
                            raise BadBody(400, "malformed or incomplete chunked request trailers")
                    raise BadBody(400, "too many chunked request trailers")
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {limit >> 20} MiB")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            self.body_read = True
            limit = body_limit()
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True               # the body is not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _drain_body(self):
            """An answer sent before the body was read (a 401, a 403, /load, a method with no handler) must not close
            the connection on unread bytes: the close then sends a reset, and a client that sends its body after the
            headers (http.client, urllib, requests) gets a connection error instead of the answer.  So the body is
            read and dropped here, once, after an answer, in pieces so that its size is never held in memory.  What
            has not arrived DRAIN_SECONDS later is left unread: the limit is time, so that a conversation of many
            megabytes, at any speed the client has, still gets its answer."""
            headers = getattr(self, "headers", None)
            if self.body_read or headers is None:
                return
            if self._chunked():
                try:
                    self._read_chunked(body_limit(), keep=False, deadline=time.monotonic() + self.DRAIN_SECONDS)
                except (BadBody, OSError):
                    self.close_connection = True
                return
            try:
                left = int(headers.get("Content-Length", 0))
            except ValueError:
                return
            if left <= 0:
                return
            deadline = time.monotonic() + self.DRAIN_SECONDS
            try:
                while left > 0 and (wait := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(wait)         # a client that never sends what it announced
                    piece = self.rfile.read1(min(left, 1 << 20))   # one socket read: read() would wait for it all
                    if not piece:
                        break
                    left -= len(piece)
            except OSError:
                pass

        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[strata] refused a request for Host {host!r} from {self.client_address[0]}: not a name this server "
                  f"answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching Strata under this "
                             f"name on purpose? Add it to \"allowed_hosts\" in the config (strata-<model>.json) or to "
                             f"the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which "
                             f"turns this check off"}})
            return False

        def _foreign_page(self) -> bool:
            """Without an API key, a /v1 POST from a browser page of another site (any site can POST text/plain
            there without a CORS preflight) would burn GPU time: an Origin header must name an allowed page, and
            then the body must be JSON.  No Origin (curl, the SDKs, other servers): any content type, as before."""
            origin = self.headers.get("Origin")
            if svc.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[strata] refused an API request from the web page {origin!r} (no API key; add its host to "
                      f"\"allowed_hosts\" or its origin to \"cors_origins\" in the config)", flush=True)
                msg = (f"web pages of {origin} may not use this server without an API key; set \"api_key\", "
                       f"or add the page's host to \"allowed_hosts\" (or its origin to \"cors_origins\") in the config")
                self._json(403, {"error": {"type": "forbidden", "message": msg}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def end_headers(self):
            # HTTP/1.0 without keep-alive: the server closes the connection after every response.  Say so, or a
            # client that pools connections (.NET's SocketsHttpHandler) can put its next request on this socket
            # before the close reaches it, and that request fails with "response ended prematurely" without the
            # server ever reading it (a .NET agent app, 2026-10-03: 25 of ~1000 requests, 1-24 ms after the previous
            # answer).
            self.send_header("Connection", "close")
            super().end_headers()

        def _watch_client(self, cancel: threading.Event) -> None:
            """#430 #431: cancel the request as soon as its client hangs up.  A non-streamed request writes nothing
            until it ends, and a streamed one only a keep-alive per prompt chunk (and the first write after a hang-up
            usually still succeeds), so a dropped request kept the engine busy until its answer or the whole prompt
            was done.  Every 0.5 s: the socket readable with nothing to read (EOF) means the client closed it.  A
            request is HTTP/1.0 and fully read here, so no later bytes are expected - data is not a hang-up."""
            done = self.watch_done = threading.Event()
            sock = self.connection

            def watch():
                while not done.wait(0.5) and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # the socket was closed here: the request has ended
                        return
                    if gone:
                        self._note(outcome="disconnected")
                        cancel.set()
                        return

            threading.Thread(target=watch, daemon=True, name="strata-client-watch").start()

        def _note(self, **values):
            """#332: what the monitor shows about this request (nothing when the monitor is off)."""
            if self.record is not None:
                with svc.status_lock:
                    self.record.update(values)

        def _cors(self):
            """#321: CORS headers for an API path (/v1/*) and an origin the config lists in cors_origins - nothing
            otherwise, so a browser keeps every other page away from the API, /settings, /unload and the MCP tools."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if "*" in svc.cors_origins:
                allow = "*"
            elif origin and origin in svc.cors_origins:
                allow = origin
                self.send_header("Vary", "Origin")
            else:
                return
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # the headers the preflight asks for (SDKs add their own; "*" does not cover Authorization)
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers",
                             asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
            self.send_header("Access-Control-Max-Age", "600")

        def do_DELETE(self):
            """DELETE /v1/responses/{id}: nothing is stored, so there is nothing to delete - OpenAI's 404, in JSON."""
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/v1/responses/"):
                self._json(404, responses_error_body("this server keeps no responses (stateless): there is nothing "
                                                     "to delete", code="not_found"))
            else:
                self._json(404, {"error": {"message": f"Invalid URL (DELETE {path})", "type": "invalid_request_error"}})

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; the headers only for cors_origins
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, obj, headers=None):
            obj = api_error_body(self.path.split("?")[0].rstrip("/"), code, obj)
            if self.record is not None:
                with svc.status_lock:
                    self.record["http_status"] = code
                    raw = json.dumps(obj, ensure_ascii=False, indent=2)
                    self.record.update(response=raw[:262144], response_truncated=len(raw) > 262144)
                    for key in ("error", "usage", "timings"):
                        if key in obj:
                            self.record[key] = obj[key]
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/v1/images/loras/fetch/"):   # a Hugging Face fetch of the Images view, as it goes
                if not self._authorized():
                    return
                f = lora_api.FETCHES.get(path[len("/v1/images/loras/fetch/"):])
                if f is None:
                    self._json(404, {"error": {"message": "no such fetch"}})
                    return
                self._json(200, f.view())
                return
            if path.startswith("/v1/strata/replies/"):      # a kept reply again, from its line `from`
                if not self._authorized():
                    return
                reply = REPLIES.get(path[len("/v1/strata/replies/"):])
                if reply is None:
                    self._json(404, {"error": {"message": "no such reply (it ended long ago, or never began)"}})
                    return
                try:
                    start = max(0, int(parse_qs(urlsplit(self.path).query).get("from", ["0"])[0]))
                except ValueError:
                    self._json(400, {"error": {"message": "from: a line number"}})
                    return
                self._sse()
                self._stream_reply(reply, start)
                return
            if path.startswith("/tool-media/"):
                if not self._authorized():
                    return
                name = path[len("/tool-media/"):]
                hub = svc.mcp
                if not re.fullmatch(r"[0-9a-f]{64}\.(png|jpg|webp)", name) or not hub or not hub.media_dir:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                file = Path(hub.media_dir) / name
                if not file.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = file.read_bytes()
                self.send_response(200)
                mime = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}[file.suffix]
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL 1.1, third_party/main/outfit); without it the page uses the
                # system font
                name = path[len("/fonts/"):]
                f = ROOT / "third_party" / "main" / "outfit" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith(("/web/locales/", "/web/vendor/")):
                # the page's translations (serve/web/locales/<lang>.json) and the vendored KaTeX
                # (third_party/main/katex): only whitelisted names and types, never a path out of those folders
                sub = web_subtree_file(path)
                if sub is None:
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = sub[0].read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", sub[1])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/metrics":
                from serve import prometheus
                if prometheus.wants_prometheus(self.headers.get("Accept", ""), urlsplit(self.path).query):
                    if self._authorized():   # a scraper: the text format, vLLM's names (serve/prometheus.py)
                        body = prometheus.render(svc.metrics(), svc.latencies.snapshot()).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", prometheus.CONTENT_TYPE)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/api/requests" and svc.api_monitor:
                if self._authorized():
                    request_id = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    records = svc.request_records(request_id)
                    self._json(404 if records is None else 200,
                               {"error": {"message": "request no longer retained"}} if records is None else
                               records if request_id else {"requests": records, "retention": 100, "persistent": False,
                                                           "loaded": svc.loaded(),
                                                           "auto_load": hasattr(svc.engine, "restart")})
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/config":                            # #564: the Settings view's keys of the run config
                if self._authorized():
                    self._config_get()
                return
            if path == "/switcher":
                if self._authorized():
                    self._json(200, model_switcher(svc))
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "" or (path == "/api-monitor" and svc.api_monitor):
                body = (ROOT / "serve" / "web" / ("monitor.html" if path else "index.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                if not path:
                    # the chat's svg/html preview is a sandboxed frame; this keeps it from navigating away to a web
                    # page (a link or <meta refresh> inside a model's html), which no sandbox flag does
                    # (serve/web/preview.js)
                    self.send_header("Content-Security-Policy", PAGE_CSP)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": svc.reported_ctx(), "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key),
                                 "image_generation": svc.images is not None,
                                 "web_search": svc.search_backend is not None,
                                 "image_note": svc.images.note() if svc.images is not None else None,
                                 "image_defaults": svc.images.default.defaults() if svc.images is not None else None,
                                 "image_models": svc.images.models() if svc.images is not None else None,
                                 "thinking": getattr(svc.template, "thinking", None),
                                 # a streamed chat reply can be kept past its connection (strata_reply)
                                 "kept_replies": True,
                                 "debug_chat_log": svc.debug_log is not None,
                                 "loaded": svc.loaded(), "service": "strata"})
            elif path == "/update":                         # whether a newer release is out (About > Updates)
                if not self._authorized():
                    return
                self._json(200, svc.updates.check() if release_check.enabled() else svc.updates.info())
            elif path.startswith("/v1/images/files/"):
                # response_format url: the name (32 random hex digits) is the access, as an OpenAI image URL is
                f = svc.images.file(path[len("/v1/images/files/"):]) if svc.images is not None else None
                if f is None:
                    self._json(404, {"error": {"message": "not found (the server keeps the newest images only)"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", images_api.FORMATS[f.suffix[1:]])
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                if not self._authorized():                  # #212: it shows the end of the last answer
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    self._json(200, {"object": "list", "data": self._models()})
            elif path.startswith("/v1/models/"):         # OpenAI's retrieve: one model by its id
                if self._authorized():
                    want = unquote(path[len("/v1/models/"):])
                    found = next((m for m in self._models() if m["id"] == want), None)
                    if found is None:
                        self._json(404, {"error": {"message": f"The model '{want}' does not exist",
                                                   "type": "invalid_request_error", "param": "model",
                                                   "code": "model_not_found"}})
                    else:
                        self._json(200, found)
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                        # llama.cpp's name for what the slot's context holds: the running request's prompt size,
                        # kept after it ends (the conversation cache carries it on).  A front-end's context meter
                        # divides it by n_ctx; without the key it reads 0 % against Strata forever (#562).
                        in_use = int(svc.status.get("prompt_tokens") or 0)
                    if getattr(svc.engine, "batch", 0) and hasattr(svc.engine, "slots_view"):
                        # --batch: one entry per slot (the same view /metrics has), not one for the whole engine
                        n_ctx = svc.reported_ctx()
                        slots = [{"id": s["slot"], "n_ctx": n_ctx, "is_processing": s["state"] != "idle",
                                  "n_prompt_tokens": int(s.get("prompt_tokens") or s.get("held_tokens") or 0)}
                                 for s in svc.engine.slots_view()]
                    else:
                        slots = [{"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy,
                                  "n_prompt_tokens": in_use}]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": f"Invalid URL (GET {path})", "type": "invalid_request_error"}})

        def _models(self) -> list[dict]:
            """GET /v1/models' list: the text model (its status: loaded, loading, or unloaded and loaded on use, as
            llama-server's router lists them), each alias under its own id (#297), and the image models.  OpenAI's
            created and owned_by: the server's start and "xestrata"."""
            status = "loaded" if svc.loaded() else "unloaded" \
                if svc.idle_unload_s or getattr(svc.engine, "unloaded", False) else "loading"
            common = {"object": "model", "created": int(svc.started_at), "owned_by": "xestrata"}
            model = {"id": svc.model, **common, "status": {"value": status},
                     "meta": {"n_ctx": svc.reported_ctx()},
                     "architecture": {"input_modalities": (["text", "image"] if svc.vision is not None else ["text"]),
                                      "output_modalities": ["text"]}}
            if svc.aliases:
                model["aliases"] = list(svc.aliases)
            data = [model, *({**model, "id": x, "alias_of": svc.model} for x in svc.aliases)]
            if svc.images is not None:                    # the image models: POST /v1/images/generations
                data += [{"id": m["id"], **common, "status": {"value": "loaded"},
                          "defaults": m["defaults"],
                          "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]}}
                         for m in svc.images.models()]
            return data

        def do_POST(self):
            try:
                self._post()
            except BadBody as e:                              # #893: a malformed or oversized chunked body
                self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _post(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            if path == "/v1/images/loras" or path.startswith("/v1/images/loras/"):   # the Images view's adapters:
                self._loras_post(path)                   # its own page checks (an upload is not JSON)
                return
            if path.startswith("/v1/") and self._foreign_page():
                return
            if path == "/settings":
                self._settings()
                return
            if path == "/switcher":                     # the web app's model menu: ask the host to swap models
                if not self._own_page("the model can be switched"):   # as /settings: a page elsewhere must not
                    return
                body = self._body()                     # #26: after the page check, and within body_limit()
                try:
                    req = json.loads(body or b"{}")
                except ValueError:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": "send a JSON object"}})
                    return
                self._json(200, model_switcher(svc, str(req.get("mode") or "") if isinstance(req, dict) else ""))
                return
            if path == "/config":
                self._config_post()
                return
            if path in ("/unload", "/load") and not self._control_body():
                return
            # JSON from Strata's own page only, as /settings: else a plain form POST from any site unloads the model
            if path in ("/unload", "/load") and not self._own_page("the model can be loaded or unloaded"):
                return
            # the same for the slot files: else any site could overwrite a saved conversation or replace the live one
            if path.startswith("/slots/") and not self._own_page("conversations can be saved or restored"):
                return
            if path == "/v1/strata/debug/chat":              # the web page's chat, for the debug chat log
                if svc.debug_log is None:
                    self._json(404, {"error": {"message": "the debug chat log is off"}})
                    return
                try:
                    chat = json.loads(self._body() or b"{}")
                except ValueError:
                    chat = None
                if not isinstance(chat, dict) or not svc.debug_log.chat(str(chat.get("id") or ""), chat):
                    self._json(400, {"error": {"type": "invalid_request_error", "message": "send a chat with its id"}})
                    return
                self._json(200, {"saved": True})
                return
            if path.startswith("/v1/strata/replies/") and path.endswith("/cancel"):   # the web app's Stop
                if not self._control_body():
                    return
                reply = REPLIES.get(path[len("/v1/strata/replies/"):-len("/cancel")])
                if reply is None:
                    self._json(404, {"error": {"message": "no such reply"}})
                    return
                reply.cancel.set()
                self._json(200, {"cancelled": not reply.done})
                return
            if path == "/update":                            # About's "Check now": ask GitHub again
                if self._control_body() and self._own_page("a check for a new release can be asked"):
                    self._json(200, svc.updates.check(force=True))
                return
            if path == "/unload":                            # give the GPU back now (between requests)
                try:
                    wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                         svc.UNLOAD_WAIT_S)
                    r = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                except EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load now, e.g. ahead of a request
                try:
                    svc.load()
                    self._json(200, {"status": "loaded"})
                except GpuBusy as e:
                    ra = getattr(e, "retry_after", None)
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                               headers={"Retry-After": int(ra)} if ra else None)
                return
            try:
                if path == "/v1/images/edits":                # a form (OpenAI's multipart) or JSON: read there
                    self._images_edit()
                    return
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # retrieve/delete/cancel/compact: nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path == "/v1/vram":                       # #533: give VRAM back to other programs, or take it
                    if not self._own_page("the VRAM reserve can be changed"):
                        return
                    r = req.get("reserve_mib")
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    try:
                        self._json(200, svc.vram(r))
                    except EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                if path in ("/v1/load", "/v1/unload"):
                    if not self._own_page("the model can be loaded or unloaded"):
                        return
                    if req.get("model") not in (None, svc.model):
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    if path == "/v1/unload":
                        wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                             svc.UNLOAD_WAIT_S)
                        result = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    else:
                        if not svc.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with svc.status_lock:
                                if svc.status.get("busy") or svc.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            svc.ensure_loaded()
                            svc.last_request_at = time.time()
                        finally:
                            svc.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **svc.v1_status()})
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    # the script guard for a client that cannot add a field to the body; the body's own field wins
                    sg_header = self.headers.get("X-XeStrata-Script-Guard")
                    if sg_header and isinstance(req, dict) and req.get("script_guard") is None:
                        req["script_guard"] = script_guard.parse(sg_header, "X-XeStrata-Script-Guard")
                    self.record = svc.begin_request(path, req)
                    self.debug = svc.debug_begin(path, req)
                if path == "/v1/responses":
                    self._responses(req)
                elif path.startswith("/slots/"):
                    query = parse_qs(self.path.partition("?")[2])
                    try:
                        code, body = svc.slot_action(path[len("/slots/"):], (query.get("action") or [""])[0],
                                                     req.get("filename"))
                    except EngineDied as e:                  # EngineSilent included: the engine was ended
                        code, body = 500, {"error": {"code": 500, "message": str(e) + "; the next request starts "
                                                     "the engine again", "type": "server_error"}}
                    self._json(code, body)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)
                elif path == "/v1/images/generations":
                    self._images(req)
                else:
                    self._json(404, {"error": {"message": f"Invalid URL (POST {path})",
                                               "type": "invalid_request_error"}})
            except EngineNonFinite as e:                    # #879, twice: not the request's fault
                print(f"[strata] 503: {e}", flush=True)
                self._json(503, responses_error_body(str(e), "server_error") if path == "/v1/responses"
                           else {"error": {"type": "server_error", "message": str(e)}})
            except ValueError as e:
                print(f"[strata] 400 invalid request: {e}", flush=True)
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ModelBusy as e:                           # the control routes (/v1/load, /v1/vram, ...): 409
                generation = path in ("/v1/chat/completions", "/v1/messages", "/v1/images/generations")
                self._json(529 if path == "/v1/messages" else 429 if generation else 409,
                           {"error": {"type": "rate_limit_exceeded" if generation else "model_busy",
                                      "code": "rate_limit_exceeded" if generation else "model_busy",
                                      "message": str(e)}}, headers={"Retry-After": 1} if generation else None)
            except StructuredOutputError as e:
                self._json(502, {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                           "message": str(e)}})
            except (GpuBusy, EngineStarting) as e:
                ra = getattr(e, "retry_after", None)       # #1997 / #1994: a refusal that is worth retrying
                # Anthropic's clients (Claude Code) retry its 529 overloaded_error
                self._json(529 if path.startswith("/v1/messages") else 503,
                           {"error": {"type": "server_error", "message": str(e)}},
                           headers={"Retry-After": int(ra)} if ra else None)
            except (TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
                # a body that is JSON but the wrong shape ("messages": 5, a content part that is a number): a 400 that
                # says so, with the traceback in the log, instead of a dropped connection - unless the answer began
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[strata] 400 malformed request ({type(e).__name__}: {e}):\n" + traceback.format_exc(),
                      flush=True)
                body = {"error": {"type": "invalid_request_error",
                                  "message": f"the request is malformed ({type(e).__name__}: {e})"}}
                self._json(400, responses_error_body(body["error"]["message"]) if path == "/v1/responses" else body)
            except EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
            except EngineStuck as e:                         # an unload or restart that could not end the engine
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            except ClientGone:                               # #1999: the client left while the request was prepared
                self._note(outcome="disconnected")
            except OSError as e:
                # the client's socket gone (or anything once the answer began): a disconnect, as before #332.  Before
                # it began, another one is a file of the server's own (a picture's work file, a full disk): a 500 that
                # says so, with the traceback in the log - not a dropped connection, and without the file's path
                if isinstance(e, (ConnectionError, TimeoutError)) or getattr(self, "answer_started", False):
                    self._note(outcome="disconnected")
                    raise                                    # the server's own handling
                import traceback
                print(f"[strata] 500: a file error ({type(e).__name__}: {e}):\n" + traceback.format_exc(), flush=True)
                self._json(500, {"error": {"type": "server_error", "message": "the server could not read or write a "
                                           f"file of its own: {e.strerror or type(e).__name__}"}})
            finally:
                svc.drop_embeddings()                        # the images' file of a request that never got to run()
                if self.watch_done is not None:
                    self.watch_done.set()
                record = self.record
                if record is not None:
                    with svc.status_lock:
                        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                        record["finished_at"] = time.time()
                        record["state"] = "error" if record.get("error") else record.get("outcome", "completed")
                    svc.request_trace.record = None
                if self.debug is not None and not self.debug.get("_streaming"):
                    svc.debug_end(self.debug)            # no answer started: what there was
                svc.request_trace.debug = None

        def _images(self, req):
            """POST /v1/images/generations (serve/images.py): one image job in the GPU's turn."""
            if svc.images is None:
                self._json(404, {"error": {"type": "invalid_request_error", "code": "images_not_set_up",
                                           "message": images_api.not_set_up_message()}})
                return
            gen = svc.images.pick(req.get("model"))             # the image model it names, else the default
            job = images_api.parse_request(req, gen.cfg)        # ImageRequestError (a ValueError): 400
            self._image_job(gen, job, req.get("stream") is True)

        def _images_edit(self):
            """POST /v1/images/edits (serve/images.py): reference images and a prompt in OpenAI's multipart form (image
            or image[]), or JSON (images: [{"image_url": ...}], or image: a URL or a list), one image job in the GPU's
            turn.  The images are read as a chat's are (data: URLs, http(s) URLs, local paths; formats stb_image does
            not read converted to PNG)."""
            if svc.images is None:
                self._json(404, {"error": {"type": "invalid_request_error", "code": "images_not_set_up",
                                           "message": images_api.not_set_up_message()}})
                return
            body = self._body()
            ctype = self.headers.get("Content-Type", "") or ""
            if ctype.lower().startswith("multipart/form-data"):
                req, files = parse_multipart(ctype, body)
                mask = req.pop("mask", None) or None
            else:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object or a multipart form")
                srcs = req.pop("images", None) or req.pop("image", None) or []
                if isinstance(srcs, (str, dict)):
                    srcs = [srcs]
                files = []
                for src in srcs:
                    url = src.get("image_url") if isinstance(src, dict) else src
                    if isinstance(url, dict):
                        url = url.get("url")
                    if not isinstance(url, str) or not url:
                        raise ValueError("images: [{\"image_url\": a data: URL, an http(s) URL or a path}]")
                    self._no_local_sources([url])     # #27: as the chat routes, before anything is read
                    files.append(svc.download(url) if url.startswith(("http://", "https://")) else Vision.load(url))
                # the mask as an image is given ({"image_url": ...} or the URL), its alpha kept (not normalized)
                mask = req.pop("mask", None)
                if isinstance(mask, dict):
                    mask = mask.get("image_url") or mask.get("url")
                if isinstance(mask, dict):
                    mask = mask.get("url")
                if mask is not None:
                    if not isinstance(mask, str) or not mask:
                        raise ValueError("mask: {\"image_url\": a data: URL, an http(s) URL or a path}")
                    self._no_local_sources([mask])
                    mask = svc.download(mask) if mask.startswith(("http://", "https://")) else Vision.load(mask)
            # the image model it names; else the default, or where that one has no edits (MageTrail), the first that
            # edits - unless the request is img2img's (a strength or a mask), which the default takes
            gen = svc.images.pick_edit(req.get("model"), req.get("strength") is not None or mask is not None)
            job = images_api.parse_edit_request(req, [Vision.normalize(d, keep_alpha=True) for d in files], gen.cfg,
                                                mask)
            self._image_job(gen, job, req.get("stream") is True, edit=True)

        def _image_job(self, gen, job, stream=False, edit=False):
            """One image job (a generation or an edit) in the GPU's turn, answered as OpenAI's images API does: one
            JSON body, or with `stream` its events (_image_stream)."""
            svc.last_request_at = time.time()
            cancel = threading.Event()
            self._watch_client(cancel)
            print(f"[strata] image: {job['width']}x{job['height']}, n={job['n']}, {job['steps']} steps, "
                  f"seed {job['seed']}" + (f", {len(job['refs'])} reference image(s)" if job.get("refs") else "")
                  + ("".join(f", adapter {u['name']} at {u['scale']:g}" for u in job.get("loras") or ())),
                  flush=True)
            if job.get("lora_unknown"):
                print(f"[strata] image: no such LoRA {', '.join(job['lora_unknown'])} in this model's folder: "
                      "left out of the prompt", flush=True)
            if stream:
                return self._image_stream(gen, job, cancel, edit)
            host = self.headers.get("Host") or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            scheme = "https" if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https" else "http"
            try:
                images, figures, policy = images_api.make_images(svc, gen, job, cancel)
                figures["gpu_policy"] = policy
                print(f"[strata] image done in {figures['seconds']:.1f} s ({policy})", flush=True)
                body = gen.response(images, job, figures, f"{scheme}://{host}")   # cjxl may fail (jxl)
            except images_api.LoraRejected as e:   # an adapter the request names, which the engine does not take
                self._json(400, {"error": {"type": "invalid_request_error", "code": "lora_rejected", "param": "lora",
                                           "message": str(e)}})
                return
            except images_api.ImageJobFailed as e:
                self._json(500, {"error": {"type": "server_error", "code": "image_generation_failed",
                                           "message": str(e)}})
                return
            self._json(200, body)

        def _image_stream(self, gen, job, cancel, edit):
            """OpenAI's streamed images: image_generation.partial_image events (image_edit.* for an edit) as the
            engine makes the job's partial_images, then one .completed event an image (b64_json, with the usage on
            each; and as the JSON body has, its seed and the "strata" figures), as SSE.  A failure after the stream
            started is an error event."""
            kind = "image_edit" if edit else "image_generation"
            fmt, opaque = job["output_format"], job["background"] == "opaque"

            def common():
                # read at each event: a size "auto" is the model's choice once it has made it (the job's, by then)
                return {"size": f"{job['width']}x{job['height']}", "quality": job["quality"],
                        "background": job["background"], "output_format": fmt}
            events: queue.Queue = queue.Queue()
            result: dict = {}

            def preview(image, index, png):
                events.put((image, index, images_api.encode_image(png, fmt, job["output_compression"], opaque)))

            def work():
                try:
                    images, figures, policy = images_api.make_images(svc, gen, job, cancel,
                                                                     preview if job.get("partial_images") else None)
                    figures["gpu_policy"] = policy
                    print(f"[strata] image done in {figures['seconds']:.1f} s ({policy}, stream)", flush=True)
                    result["images"], result["figures"] = images, figures
                except Exception as e:  # noqa: BLE001 - said in the stream: its status is sent already
                    result["error"] = e
                finally:
                    events.put(None)
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            self._sse()

            def send(name, data):
                self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                 json.dumps({"type": name, **data}).encode() + b"\n\n")
                self.wfile.flush()
            try:
                while True:
                    try:
                        item = events.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    if item is None:
                        break
                    _, index, data = item
                    send(f"{kind}.partial_image", {"b64_json": base64.b64encode(data).decode("ascii"),
                                                   "created_at": int(time.time()), **common(),
                                                   "partial_image_index": index})
                if "error" in result:
                    if isinstance(result["error"], images_api.LoraRejected):   # the request's adapter: its own fault
                        err = {"type": "invalid_request_error", "code": "lora_rejected", "param": "lora",
                               "message": str(result["error"])}
                    else:
                        err = {"type": "server_error", "code": "image_generation_failed",
                               "message": str(result["error"])}
                    self._note(error=err)
                    send("error", {"error": err})
                    return
                for i, png in enumerate(result["images"]):
                    data = images_api.encode_image(png, fmt, job["output_compression"], opaque)
                    applied = [{"name": u["name"], "scale": u["scale"]} for u in job.get("loras") or ()]
                    send(f"{kind}.completed", {"b64_json": base64.b64encode(data).decode("ascii"),
                                               "created_at": int(time.time()), **common(),
                                               **({"lora": applied} if applied else {}),
                                               "usage": images_api.usage(job), "model": gen.model,
                                               "seed": job["seed"] + i, "strata": result["figures"]})
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model not in svc.model_names():
                self._json(404, {"error": {"message": "model not found"}})
                return
            if not svc.loaded() and not getattr(svc.engine, "unloaded", False):
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**model_card_sampling(svc.model_arch, True), **svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "batch", 0) or 0)),   # #1004: the batch slots
                     "model_alias": svc.model, "chat_template": svc.template.source,
                     "chat_template_caps": svc.template.caps,
                     "modalities": {"vision": svc.vision is not None},
                     "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded(),
                     # XeStrata's: the script guard of a request that names none (serve/script_guard.py)
                     "script_guard": script_guard_name(svc.shared.get("script_guard") or svc.script_guard)}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _control_body(self) -> bool:
            """Consume the unused control body before replying/closing (Windows otherwise sends a TCP reset)."""
            if self._chunked():
                self.body_read = True
                try:
                    self.connection.settimeout(2.0)
                    self._read_chunked(65536, keep=False)
                except BadBody as e:
                    self._json(e.status, {"error": {"message": f"control request: {e}"}})
                    return False
                except OSError:
                    self._json(400, {"error": {"message": "incomplete control request body"}})
                    return False
                return True
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": {"message": "invalid Content-Length"}})
                return False
            if not 0 <= length <= 65536:
                self._json(413, {"error": {"message": "control request body is limited to 64 KiB"}})
                return False
            timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(2.0)
                complete = len(self.rfile.read(length)) == length
            except OSError:
                complete = False
            finally:
                self.connection.settimeout(timeout)
                self.body_read = True                    # nothing left for the drain to wait for (#594)
            if not complete:
                self._json(400, {"error": {"message": "incomplete control request body"}})
            return complete

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            if self._foreign_origin():
                self._json(403, {"error": {"message": f"{what} only from Strata's own page (or an origin in the "
                                                      f"config's trusted_origins)"}})
                return False
            return True

        def _loras_post(self, path):
            """The Images view's adapters (serve/loras.py): an upload (POST /v1/images/loras?name=<file>, the file
            itself the body, application/octet-stream: a page elsewhere cannot send that without a CORS preflight,
            which this server never grants), a remove (/v1/images/loras/remove {"path"}) and a Hugging Face fetch
            (/v1/images/loras/fetch {"url"}, read with GET /v1/images/loras/fetch/<id>).  From this server's own page
            (or a trusted origin), as /settings."""
            folders = svc.images.folders() if svc.images is not None else []
            if path == "/v1/images/loras":
                if not self.headers.get("Content-Type", "").startswith("application/octet-stream"):
                    self._json(415, {"error": {"message": "send the file as application/octet-stream"}})
                    return
                if self._foreign_origin():
                    self._json(403, {"error": {"message": "LoRA files are added only from Strata's own page"}})
                    return
                name = parse_qs(urlsplit(self.path).query).get("name", [""])[0]
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._json(411, {"error": {"message": "the upload needs a Content-Length"}})
                    return
                try:
                    lora_api.file_name(name)
                    tmp = lora_api.staging(folders, length)
                except (lora_api.LoraError, OSError) as e:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                    return
                left = length
                try:
                    with open(tmp, "wb") as out:
                        while left:
                            chunk = self.rfile.read(min(1 << 20, left))
                            if not chunk:
                                raise OSError("the upload ended before its Content-Length")
                            out.write(chunk)
                            left -= len(chunk)
                except OSError as e:
                    tmp.unlink(missing_ok=True)
                    self.close_connection = True
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                    return
                finally:
                    self.body_read = True
                try:
                    added = lora_api.add(folders, tmp, name)
                except (lora_api.LoraError, OSError) as e:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                    return
                print(f"[strata] LoRA added: {added['path']} ({', '.join(added['models'])})", flush=True)
                self._json(200, {"added": added})
                return
            if not self._own_page("LoRA files can be added or removed"):
                return
            try:
                req = json.loads(self._body() or b"{}")
            except ValueError:
                self._json(400, {"error": {"type": "invalid_request_error", "message": "send a JSON object"}})
                return
            req = req if isinstance(req, dict) else {}
            try:
                if path == "/v1/images/loras/remove":
                    gone = lora_api.remove(folders, req.get("path"))
                    print(f"[strata] LoRA removed: {gone['removed']}", flush=True)
                    self._json(200, gone)
                elif path == "/v1/images/loras/fetch":
                    lora_api.staging(folders)                    # a model that takes one, before the download
                    f = lora_api.FETCHES.start(folders, req.get("url"))
                    print(f"[strata] LoRA fetch: {f.url}", flush=True)
                    self._json(202, f.view())
                else:
                    self._json(404, {"error": {"message": f"Invalid URL (POST {path})"}})
            except (lora_api.LoraError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _config_get(self):
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                cfg = runconfig.load(svc.config_path)
            except (OSError, ValueError) as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be read: {e}"}})
                return
            try:
                settings = runconfig.load_settings(svc.settings_path)
            except (OSError, ValueError):
                settings = {}                           # shown as unset; a change to it is refused (POST)
            self._json(200, runconfig.view(cfg, svc.config_path, settings, svc.settings_path))

        def _config_post(self):
            """#564: change a few documented keys of the run config - JSON from XeStrata's own page only, as
            /settings (the key is checked before); every other key of the file stays as it is."""
            body = self._body()
            if not self._own_page("the run config can be changed"):
                return
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            scope, settings = "model", {}
            try:
                req = json.loads(body or b"{}")
                if isinstance(req, dict) and req.get("scope") is not None:
                    scope = req["scope"]
                with svc.config_lock:
                    cfg = runconfig.load(svc.config_path)
                    try:
                        settings = runconfig.load_settings(svc.settings_path)
                    except (OSError, ValueError) as e:
                        if scope == "global":           # setup's keys are in there: never written over
                            raise ValueError(f"{svc.settings_path} cannot be read ({e}): fix or remove it first")
                    sets = req.get("set") if isinstance(req, dict) else None
                    if scope == "global":
                        settings, changed = runconfig.apply(settings, sets, "global")
                        if changed:
                            runconfig.save_settings(svc.settings_path, settings)
                        new = cfg
                    else:
                        new, changed = runconfig.apply(cfg, sets, scope)
                        if changed:
                            bak = runconfig.save(svc.config_path, new)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            except OSError as e:
                self._json(500, {"error": {"type": "server_error",
                                           "message": f"the run config cannot be written: {e}"}})
                return
            if changed:
                where = (f"{svc.settings_path} (every model)" if scope == "global" else
                         f"{Path(svc.config_path).name} (the earlier file: {bak.name})")
                print(f"[strata] the Settings view changed {', '.join(changed)} in {where}; used from the next start",
                      flush=True)
            self._json(200, {**runconfig.view(new, svc.config_path, settings, svc.settings_path), "changed": changed})

        def _foreign_origin(self) -> bool:
            """A web page of another origin sent this.  The Origin must be this server's own address (host and port),
            or an origin the config trusts (trusted_origins: the web app behind a reverse proxy or tunnel); no Origin
            is a client that is no browser.  Headers a proxy adds (X-Forwarded-*, CF-Ray, CF-Connecting-IP) prove
            nothing about the page that sent the request, so they open nothing."""
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return bool(origin) and origin.split("://", 1)[-1] != self.headers.get("Host", "") and \
                origin not in svc.trusted_origins

        def _no_local_images(self, messages):
            """A web page of another origin must not have a file on this computer read as an image.  Without an
            api_key, _foreign_page refuses such a page unless cors_origins lets it in - and a page cors_origins lets
            in (["*"]: every page) could name any file and, CORS allowing, read what the model says about it.  data:
            and http(s) images are the page's own to send; files come from Strata's own page, a trusted origin or a
            client that is no browser.  ValueError (a 400)."""
            self._no_local_sources(images_of(messages))

        def _no_local_sources(self, sources):
            """_no_local_images for a list of image sources (the chat routes', and #27 /v1/images/edits')."""
            if self._foreign_origin() and any(not src.startswith(("data:", "http://", "https://")) for src in sources):
                raise ValueError("a web page of another origin cannot have a file on this computer read as an "
                                 "image: send it as a data: URL (or add the page's origin to the config's "
                                 "trusted_origins)")

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "defaults" not in req:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                               "message": 'the body must be {"defaults": {...}} to set them, '
                                                          'or {"defaults": null} to clear them'}})
                    return
                shared = svc.set_shared(req["defaults"])
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[strata] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self._note(http_status=200)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # nginx and similar proxies pass each event at once
            self._cors()
            self.end_headers()

        def _capture(self, items, api):
            """Retain bounded input/output for the monitor without changing the API response (the items as they
            are when the monitor is off)."""
            if self.record is None and self.debug is None:
                return items
            if self.debug is not None:
                self.debug["_streaming"] = True          # the reply's end writes it (_captured)
            return self._captured(items, api)

        def _captured(self, items, api):
            try:
                for item in items:
                    if item is not None:
                        if api == "openai":
                            delta = item["choices"][0]["delta"]
                            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
                            usage, timings = item.get("usage"), item.get("timings")
                        elif api == "responses":
                            kind = item["type"]
                            content = item["delta"] if kind == "response.output_text.delta" else ""
                            reasoning = item["delta"] if kind == "response.reasoning_text.delta" else ""
                            usage, timings = (item.get("response") or {}).get("usage"), None
                        else:
                            _, event = item
                            delta = event.get("delta", {})
                            content, reasoning = delta.get("text", ""), delta.get("thinking", "")
                            usage, timings = event.get("usage"), None
                        if self.record is not None:
                            with svc.status_lock:
                                for name, value in (("output", content), ("reasoning", reasoning)):
                                    if value:
                                        combined = self.record[name] + value
                                        self.record[name] = combined[:262144]
                                        self.record[name + "_truncated"] |= len(combined) > 262144
                                if usage:
                                    self.record["usage"] = usage
                                if timings:
                                    self.record["timings"] = timings
                        if self.debug is not None:
                            self._debug_item(item, api, content, reasoning, usage, timings)
                    yield item
            except Exception as e:
                if self.debug is not None:
                    self.debug["error"] = f"{type(e).__name__}: {e}"
                raise
            finally:
                items.close()
                svc.debug_end(self.debug)

        def _debug_item(self, item, api, content, reasoning, usage, timings):
            d = self.debug
            d["output"] += content or ""
            d["reasoning"] += reasoning or ""
            if usage:
                d["usage"] = usage
            if timings:
                d["timings"] = timings
            if api == "openai":
                choice = item["choices"][0]
                for tc in choice.get("delta", {}).get("tool_calls") or []:
                    one = d["tool_calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    one["name"] += fn.get("name") or ""
                    one["arguments"] += fn.get("arguments") or ""
                if choice.get("finish_reason"):
                    d["finish_reason"] = choice["finish_reason"]

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            legacy = legacy_functions(req)                    # the old functions / function_call, as tools
            req = legacy or req
            modalities = req.get("modalities")
            if req.get("audio") is not None or modalities is not None and (
                    not isinstance(modalities, list) or any(m != "text" for m in modalities)):
                raise ValueError("modalities: this server answers in text only (no audio output)")
            for key in ("max_completion_tokens", "max_tokens"):   # OpenAI's integers: 2.9 is a 400, not 2
                v = req.get(key)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                      or isinstance(v, float) and not v.is_integer()):
                    raise ValueError(f"{key} must be a whole number")
            bias_api.normalize(req.get("logit_bias"), len(getattr(svc.tok, "tokens", ())) or None)
            messages, tools, kw = openai_to_messages(req)
            require_messages(messages)   # an empty conversation is a 400, as OpenAI answers it, not a reply to nothing
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the Responses route: no tools are offered
                tools = None
            tools, choice = allowed_tools(req.get("tool_choice"), tools)   # the tools it lists, and its mode
            force = forced_call(choice, tools,                      # a bad value is a 400 before anything is sent
                                getattr(svc.template, "call_form", "qwen"))
            messages, validator = prepare_format(req.get("response_format"), messages)
            if validator is not None and (tools or req.get("strata_mcp")):
                raise ValueError("structured response_format with tools/MCP is not supported")
            svc.load()
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            if top_logprobs(req) is not None and batch_slots_on(svc.engine) and max_req != 1:
                # --parallel: a slot's "BT <slot> <id>" lines carry no LP line; an admission - a max_tokens 1
                # request - does.  Past that token a reply would hold entries for a prefix only, which a client cannot
                # tell from the model's own answer, so the longer one is refused (upstream 9ea044962, #3200)
                raise ValueError('logprobs is available with "parallel" (batch slots) only for max_tokens 1; '
                                 'run the server without "--parallel" for longer replies')
            n_vocab = len(getattr(svc.tok, "tokens", ())) or None
            if top_logprobs(req) is not None and n_vocab:
                # the engine indexes the logits row with each named id: one past the vocabulary is refused here
                bad = [i for i in allowed_ids(req) if i >= n_vocab]
                if bad:
                    raise ValueError(f"allowed_token_ids: {bad[0]} is outside the vocabulary (0..{n_vocab - 1})")
            bias_api.validate_request(req, svc.engine, len(getattr(svc.tok, "tokens", ())) or None)
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (serve/mcp.py)
            own = {t.get("name") for t in tools or [] if isinstance(t, dict)}   # #592: a second line of defence
            extra: list = []
            hubs: list = []
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                if extra:
                    hubs.append(svc.mcp)
            # opt-in: the image tools (true, or {"model": name}); {"force": true} (the web page's image button): the
            # reply opens a call of one of them, as the Responses route's tool_choice image_generation does, and they
            # are the request's only tools
            want = req.get("strata_images")
            image_force = isinstance(want, dict) and want.get("force") is True
            if image_force and (svc.images is None or force or tool_choice_of(req.get("tool_choice"))[0] == "none"):
                raise ValueError("strata_images.force: " + ("this server makes no images" if svc.images is None else
                                                            "it is a tool_choice of its own; send none beside it"))
            if (want is True or isinstance(want, dict)) and svc.images is not None:
                image_tools = images_api.ImageTools(svc, svc.images, messages, Vision.load,
                                                    lambda d: Vision.normalize(d, keep_alpha=True),
                                                    model=want.get("model") if isinstance(want, dict) else None)
                more = image_tools.template_tools(exclude=own | {t["name"] for t in extra})
                if image_force:
                    if not more:
                        raise ValueError("strata_images.force: the image tools are not available for this request")
                    tools, extra, hubs = None, list(more), []
                    force = forced_call("required", more, getattr(svc.template, "call_form", "qwen"))
                if more:
                    extra += [] if image_force else more
                    hubs.append(image_tools)
            search_tools = None                               # web_search_options: the search tools (search.py)
            ws = req.get("web_search_options")
            if ws is not None and not isinstance(ws, dict):
                raise ValueError("web_search_options must be an object")
            if ws is not None and svc.search_backend is not None and not image_force:   # without one: unsearched
                size = ws.get("search_context_size")
                if size is not None and size not in search_api.CONTEXT:
                    raise ValueError('web_search_options.search_context_size is "low", "medium" or "high"')
                search_tools = search_api.SearchTools(svc.search_backend, {
                    "context": size,
                    "locale": search_api.locale(ws.get("user_location"), self.headers.get("Accept-Language"))})
                more = search_tools.template_tools(exclude=own | {t["name"] for t in extra})
                if more:
                    extra += more
                    hubs.append(search_tools)
            use_mcp = bool(hubs)
            if use_mcp:
                tools = (tools or []) + extra or None
            if force and use_mcp and not image_force:
                raise ValueError("a forced tool_choice with MCP tools is not supported")
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            check_logprobs(req)                               # ... and top_logprobs out of 0..20
            n = 1 if req.get("n") is None else req["n"]
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 128:
                raise ValueError("n must be a whole number from 1 to 128")
            if n > 1 and use_mcp:
                raise ValueError("n above 1 with MCP or image tools is not supported")
            cancel = threading.Event()
            rid = reply_id(req)
            if rid is None:                                  # a kept reply is cancelled by its own cancel alone
                self._watch_client(cancel)                   # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            form = getattr(svc.template, "call_form", "qwen")
            # the engine writes the answer, or a strict tool's call, under its schema when it can (structured.py)
            strict, patterns = constrained_tools(req.get("tools"))
            svc.attach_grammar(req, answer_grammar(req.get("response_format"), thinking) if validator is not None
                               else tool_grammar(tools, strict, patterns, thinking, form))
            svc.check_strict(req, tools, strict)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            run = run_with_mcp(svc, hubs[0] if len(hubs) == 1 else ToolHubs(hubs), messages, tools, kw, ids,
                               thinking, max_new, max_req, req, cancel, {t["name"] for t in extra}) if use_mcp else None
            if n > 1:
                first = (ids, thinking, max_new)

                def choice(k):
                    # each choice its own run of the same prompt (the conversation cache holds it after the first);
                    # a request's seed gives each choice its own, seed + k
                    sub = {**req, "seed": req["seed"] + k} if k and type(req.get("seed")) is int else req
                    got = first if k == 0 else svc.prepare(messages, tools, kw, max_req, force=force, req=sub,
                                                           cancel=cancel)
                    c = openai_chunks(svc, sub, *got[:2], tools, got[2], cancel, force=force)
                    return structured_chunks(c, validator) if validator is not None else c
                chunks = openai_choices(n, choice)
            else:
                chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run, force=force)
                if validator is not None:
                    chunks = structured_chunks(chunks, validator)
                if search_tools is not None:
                    chunks = cite_chunks(chunks, search_tools)
            chunks = self._capture(chunks, "openai")
            if not req.get("stream"):
                body = openai_collect_choices(chunks) if n > 1 else openai_collect(chunks)
                for c in body["choices"] if legacy else ():
                    legacy_function_call(c)
                return self._json(200, body)
            if legacy:
                chunks = legacy_chunks(chunks)
            lines = self._openai_sse(chunks, req.get("stream_options"))
            if rid is not None:                              # kept: made on its own thread, read from the reply
                reply = REPLIES.open(rid, cancel)
                # the request's image embeddings, which prepare() keeps for its own thread: run() reads them on this
                # one, and the handler's drop_embeddings() at its end leaves them to it
                emb = (getattr(svc.embeddings, "path", None), getattr(svc.embeddings, "keys", None))
                svc.embeddings.path = svc.embeddings.keys = None
                dbg = getattr(svc.request_trace, "debug", None)   # ... and its debug chat log record

                def make():                              # its events, one a line: what a read `from` counts
                    svc.embeddings.path, svc.embeddings.keys = emb
                    svc.request_trace.debug = dbg
                    try:
                        for line in lines:
                            for event in line.split(b"\n\n"):
                                if event and not event.startswith(b":"):   # a keep-alive is _stream_reply's own
                                    reply.put(event + b"\n\n")
                    finally:
                        svc.drop_embeddings()            # when the reply ended before run() took them
                        reply.finish()
                threading.Thread(target=make, daemon=True, name="strata-reply").start()
                self._sse()
                self._stream_reply(reply, 0)
                return
            self._sse()
            try:
                for line in lines:
                    self.wfile.write(line)
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                                 # client went away: stop the engine
                lines.close()
                chunks.close()

        def _openai_sse(self, chunks, options):
            """A streamed chat completion's SSE lines: the chunks (None a keep-alive), OpenAI's usage chunk when
            asked, [DONE]; an error after the stream started as an error event, then [DONE]."""
            last = None
            try:
                for c in chunks:
                    if c is None:
                        yield b": keep-alive\n\n"                # an SSE comment: clients ignore it
                    else:
                        yield b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n"
                        last = c
                if isinstance(options, dict) and options.get("include_usage") and last and last.get("usage"):
                    # OpenAI's usage chunk, after the finishing one (which also keeps its usage, as llama.cpp's)
                    usage = {k: last[k] for k in ("id", "object", "created", "model", "system_fingerprint")
                             if k in last}
                    yield b"data: " + json.dumps({**usage, "choices": [], "usage": last["usage"]},
                                                 ensure_ascii=False).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"
                return
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
            except StructuredOutputError as e:
                err = {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                 "message": str(e)}}
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
            except Exception as e:  # noqa: BLE001 - a bug after the stream started: said, and the stream ends
                err = {"error": {"type": "server_error", "message": self._stream_bug(e)}}
            self._note(error=err["error"])
            yield b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n"

        def _stream_reply(self, reply: Reply, start: int) -> None:
            """Write a kept reply's lines from `start` as they come, a keep-alive while none does, until it ends.  A
            connection that breaks only ends this read: the reply goes on (GET /v1/strata/replies/<id> reads it
            again)."""
            at = start
            try:
                while True:
                    got, done = reply.read(at, 1.0)
                    for line in got:
                        self.wfile.write(line)
                    at += len(got)
                    if done and not got:
                        break
                    if not got:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected (the reply goes on)")

        def _responses_ready(self, req, events):
            """Send a finished Responses result that did not run on the engine."""
            if not req.get("stream"):
                return self._json(200, responses_api.collect(events))
            self._sse()
            try:
                for e in events:
                    self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                     json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")

        def _responses(self, req):
            """#451: POST /v1/responses - OpenAI's Responses API, stateless (serve/responses.py), on the chat path.
            Errors use the Responses format; once the stream has started they arrive as a response.failed event."""
            try:
                title = (responses_api.thread_title_events(req, svc.model_for(req))
                         if svc.codex_thread_titles else None)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            if title is not None:
                return self._responses_ready(req, title)
            req = responses_api.with_strict_defaults(req)    # a function tool without strict: strict where possible
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            try:
                ids, thinking, tools, max_new, asm, validator, req, force = self._responses_prepare(req, cancel)
            except ClientGone:
                self._note(outcome="disconnected")
                return
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except ModelBusy as e:
                return self._json(429, responses_error_body(str(e), "rate_limit_exceeded", code="rate_limit_exceeded"),
                                  headers={"Retry-After": 1})
            except (GpuBusy, EngineStarting, EngineStuck, EngineDied) as e:
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))

            def check(text, finish):
                return validated_json(text, validator, finish)

            image_tools, search_tools = self.image_tools, self.search_tools
            hubs = [h for h in (image_tools, search_tools) if h is not None]
            hub = None if not hubs else hubs[0] if len(hubs) == 1 else ToolHubs(hubs)
            search_names = set(search_tools.routes()) if search_tools is not None else set()

            def image_b64(url: str) -> str | None:
                f = svc.images.file(url.rsplit("/", 1)[-1]) if svc.images is not None else None
                return base64.b64encode(f.read_bytes()).decode("ascii") if f is not None else None

            def events():
                if top_logprobs(req) is not None:          # include: message.output_text.logprobs
                    lp = LogprobEntries(svc, allowed_ids(req), cancel)
                    asm.spell = lambda text: lp.spell(svc.engine, text)
                yield from asm.start()
                done = None
                if hub is not None:   # the image and search tools run here: image_generation_call, web_search_call
                    messages, kw = self.prepared
                    source = run_with_mcp(svc, hub, messages, tools, kw, ids, thinking, max_new,
                                          int(req.get("max_output_tokens") or 0), req, cancel,
                                          set(hub.routes()), force=force)
                else:
                    # Compaction may replay undeclared calls, but an explicit none still forbids calls, just as
                    # it does on the other routes.  Do not promote a quoted call to an executable output item.
                    source = svc.run(ids, thinking, tools, max_new, req, cancel, force=force,
                                     declared_only=not responses_api.codex_compaction(req)
                                     or tool_choice_of(req.get("tool_choice"))[0] == "none")
                for kind, x in source:
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "mcp":
                        x = cast(dict, x)
                        if x.get("name") in search_names if x.get("event") == "call" else \
                                asm.item is not None and asm.item["type"] == "web_search_call":
                            yield from asm.search_call(x)
                        else:
                            yield from asm.image_call(x, image_b64)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done, check)
            items = self._capture(events(), "responses")
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except StructuredOutputError as e:
                    return self._json(502, responses_error_body(str(e), "server_error",
                                                                code="structured_output_failed"))
                except EngineDied as e:
                    return self._json(503, responses_error_body(f"{e}; the next request restarts it", "server_error",
                                                                code="server_error"))
                except EngineNonFinite as e:                 # #879, twice: not the request's fault
                    return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                for item in items:
                    if item is None:
                        # Codex's idle timeout (5 minutes by default) counts events, not SSE comments: while a long
                        # prompt is read, a response.in_progress every 15 s tells it the server is still working
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(item)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                                 # client went away: stop the engine
                items.close()
            except EngineDied as e:                          # mid-stream: response.failed, then the stream ends
                self._responses_failed(asm, f"{e}; the next request restarts it", "server_error", send)
            except StructuredOutputError as e:
                self._responses_failed(asm, str(e), "structured_output_failed", send)
            except ValueError as e:                          # the engine's ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)
            except Exception as e:  # noqa: BLE001 - a bug after the stream started: response.failed, as above
                self._responses_failed(asm, self._stream_bug(e), "server_error", send)

        def _responses_failed(self, asm, message, code, send):
            # OpenAI's order: the error event, then response.failed - made in that order, as the sequence numbers
            # count as the events are made
            err = asm.error(message, code)
            e = asm.failed(message, code)
            self._note(error=e["response"]["error"])
            try:
                send(err)
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req, cancel=None):
            responses_api.check_request(req)
            messages = responses_api.input_messages(        # an earlier answer's image is kept for edit_image
                req, (lambda d: f"/v1/images/files/{svc.images.keep(d)}") if svc.images is not None else None)
            tools, names, skipped = responses_api.request_tools(req)
            # OpenAI's hosted image_generation tool: this server's image tools, run here (serve/images.py)
            self.image_tools = None
            if "image_generation" in skipped and svc.images is not None and req.get("tool_choice") != "none":
                spec = next((t for t in req.get("tools") or [] if isinstance(t, dict)
                             and t.get("type") == "image_generation"), {})
                # its model: one of this server's image models by name (OpenAI's own, gpt-image-1: the default or the
                # one the words choose); its input_image_mask: the mask of the edits it makes (a data: or http(s)
                # URL; no file_id, which OpenAI's Files API gives)
                fixed = spec.get("model") if spec.get("model") in svc.images.gens else None
                mask = spec.get("input_image_mask")
                if mask is not None:
                    url = mask.get("image_url") if isinstance(mask, dict) else None
                    if not isinstance(url, str) or not url:
                        raise ValueError('input_image_mask: {"image_url": a data: URL or an http(s) URL} (no file_id)')
                    self._no_local_sources([url])
                    mask = svc.download(url) if url.startswith(("http://", "https://")) else Vision.load(url)
                self.image_tools = images_api.ImageTools(svc, svc.images, messages, Vision.load,
                                                         lambda d: Vision.normalize(d, keep_alpha=True),
                                                         spec, model=fixed, mask=mask)
                if req.get("stream") is True:            # the tool's partial_images, sent as the image is made
                    self.image_tools.stream_partials(spec.get("partial_images"))
                offered = self.image_tools.template_tools(exclude={t["name"] for t in tools or []})
                # the tool's action: "generate" offers generate_image alone, "edit" edit_image alone
                action = spec.get("action", "auto")
                if action not in ("auto", "generate", "edit"):
                    raise ResponsesError('the image_generation tool\'s action is "auto", "generate" or "edit"',
                                         "tools")
                if action != "auto":
                    keep = "generate_image" if action == "generate" else "edit_image"
                    offered = [t for t in offered if t["name"] == keep]
                    if not offered or action == "edit" and not self.image_tools.has_image():
                        raise ResponsesError(f'the image_generation tool\'s action "{action}" cannot be done here'
                                             + (": there is no image in the conversation to edit"
                                                if action == "edit" and offered else ""), "tools")
                image_names = [t["name"] for t in offered]
                tools = (tools or []) + offered
                skipped = [k for k in skipped if k != "image_generation"]
            else:
                image_names = []
            # OpenAI's hosted web_search tool (and web_search_preview): the search tools, run here (serve/search.py)
            self.search_tools, search_names = None, []
            hosted = [k for k in skipped if k.startswith("web_search")]
            if hosted and svc.search_backend is not None and req.get("tool_choice") != "none":
                spec = next((t for t in req.get("tools") or [] if isinstance(t, dict) and t.get("type") in hosted), {})
                domains = (spec.get("filters") or {}).get("allowed_domains") if isinstance(spec.get("filters"),
                                                                                           dict) else None
                self.search_tools = search_api.SearchTools(svc.search_backend, {
                    "context": spec.get("search_context_size"),
                    "include": domains if isinstance(domains, list) else [],
                    "locale": search_api.locale(spec.get("user_location"), self.headers.get("Accept-Language"))})
                offered = self.search_tools.template_tools(exclude={t["name"] for t in tools or []})
                search_names = [t["name"] for t in offered]
                tools = (tools or []) + offered
                skipped = [k for k in skipped if not k.startswith("web_search")]
            kw = responses_api.template_kwargs(req, {"reasoning_effort": svc.effort_default()})
            messages = responses_api.verbosity_messages(req, messages)
            if req.get("truncation") not in (None, "auto", "disabled"):
                raise ResponsesError('truncation is "auto" or "disabled"', "truncation")
            calls = req.get("max_tool_calls")
            if calls is not None and (isinstance(calls, bool) or not isinstance(calls, int) or calls < 1):
                raise ResponsesError("max_tool_calls must be a positive whole number", "max_tool_calls")
            if self.image_tools is not None:              # the hosted tools this server runs: the image tools
                self.image_tools.max_calls = calls
            if self.search_tools is not None and calls is not None:   # ... and the search tools
                self.search_tools.max_uses = calls
            include = req.get("include") if isinstance(req.get("include"), list) else []
            if "message.output_text.logprobs" in include:
                req = {**req, "logprobs": True}           # the engine's LP lines, read into output_text's logprobs
            try:
                check_logprobs(req) if req.get("logprobs") else None
            except ValueError as e:
                raise ResponsesError(str(e), "top_logprobs") from None
            # "required" or a named tool: the reply opens that call (forced_call), as on the chat route
            choice = responses_api.forced_choice(req)
            try:
                if isinstance(choice, dict) and choice.get("type") == "image_generation":
                    if not image_names:
                        raise ValueError("tool_choice image_generation: the request has no image_generation tool "
                                         "this server can run")
                    tools, choice = [t for t in tools or [] if t["name"] in image_names], "required"
                if isinstance(choice, dict) and str(choice.get("type")).startswith("web_search"):
                    if not search_names:
                        raise ValueError(f"tool_choice {choice.get('type')}: the request has no web search tool this "
                                         "server can run (search_engine, search_url)")
                    tools, choice = [t for t in tools or [] if t["name"] == "web_search"], "required"
                tools, choice = allowed_tools(choice, tools, {"image_generation": image_names,
                                                              **({k: search_names for k in hosted}
                                                                 if search_names else {})})
                force = forced_call(choice, tools, getattr(svc.template, "call_form", "qwen"))
            except ValueError as e:
                raise ResponsesError(str(e), "tool_choice") from None
            try:
                self._no_local_images(messages)              # as on the chat route: no file read for a foreign page
            except ValueError as e:
                raise ResponsesError(str(e), "input") from None
            try:
                messages, validator = prepare_format(responses_api.text_format(req), messages, with_tools=bool(tools))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # said once per tool, not per request
            if set(skipped) - noted:
                print(f"[strata] /v1/responses: left out the hosted tools {', '.join(sorted(set(skipped) - noted))} "
                      f"(this server cannot run them)", flush=True)
                noted.update(skipped)
            if req.get("max_output_tokens") is None and "max_tokens" in svc.shared:   # the shared Chat settings
                req = {**req, "max_output_tokens": svc.shared["max_tokens"]}
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before anything is sent
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            try:
                check_request_sampling(req)                   # the request is also the sampling dict (see below)
            except ValueError as e:                           # its message starts with the field's name: the param
                raise ResponsesError(str(e), str(e).partition(":")[0]) from None
            svc.load()
            # #924 (opt-in): a Codex compaction request is rendered with its conversation's tools, so its prompt
            # starts as the cached one did; the parser and the response keep the request's own tools
            shown = responses_api.prompt_tools(req, tools) if svc.codex_compaction_cache else tools
            while True:
                try:
                    try:
                        ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0,
                                                             force=force, req=req, cancel=cancel)
                    except ValueError:
                        if shown is tools:
                            raise
                        shown = tools                        # too long with the kept tools: the request as sent
                        ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0,
                                                             force=force, req=req, cancel=cancel)
                    break
                except ResponsesError:
                    raise
                except ValueError as e:                      # too long for the context, an image without vision
                    # truncation "auto": the oldest items go (a call with its results), the instructions stay
                    fewer = responses_api.drop_oldest(messages) \
                        if req.get("truncation") == "auto" and "context" in str(e) else None
                    if fewer is None:
                        raise ResponsesError(str(e), "input",
                                             "context_length_exceeded" if "context" in str(e) else None) from None
                    messages = fewer
            if svc.codex_compaction_cache:
                responses_api.prompt_made(req, shown)
            req = dict(req)
            strict, patterns = constrained_tools(responses_api.given_tools(req))   # namespaces' and additional_tools'
            svc.attach_grammar(req, answer_grammar(responses_api.text_format(req), thinking)
                               if validator is not None and not tools
                               else tool_grammar(tools, strict, patterns, thinking,
                                                 getattr(svc.template, "call_form", "qwen")))
            svc.check_strict(req, tools, strict)
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            asm = responses_api.Assembler(req, svc.model_for(req), len(ids), names,
                                          "reasoning.encrypted_content" in include, validator is not None)
            if self.search_tools is not None:
                asm.sources = self.search_tools.sources      # the same list: what the searches find, as they run
            self.prepared = (messages, kw)
            # the request is also the sampling dict, as on the chat path (temperature, top_p, top_k, seed, ...)
            return ids, thinking, tools, max_new, asm, validator, req, force

        def _count_tokens(self, req):
            """Anthropic's token count, which Claude Code asks for its context figures: the prompt this server would
            read for the same request, rendered and tokenized - the model does not run."""
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the request itself: no tools offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools, getattr(svc.template, "call_form", "qwen"))
            constrained_tools(req.get("tools"))               # a tool's strict of another type: the same 400
            messages, _ = anthropic_prepare_format(req, messages, tools)   # its instruction is part of the prompt
            self._no_local_images(messages)                   # the same 400s as the request itself would get
            stop_strings(req)
            check_request_sampling(req)                       # ... likewise a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)
            # the prompt /v1/messages would read: a document's text and pages, each picture's image tokens (encoded
            # here, in the FIFO's turn, unless the cache has it; an encoder an idle unload stopped is loaded back, as
            # the request would) and a forced call's opening
            _, ids, _, _ = svc.prompt_ids(messages, tools, kw, force, cancel, store=False, wake=True)
            self._json(200, {"input_tokens": len(ids)})

        def _anthropic(self, req):
            asked = req.get("max_tokens")                     # Anthropic's required field (the shared settings' own
            if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:   # default does not stand in)
                raise ValueError("max_tokens: a whole number of 1 or more is required")
            svc.load()
            req = svc.with_shared(req, "anthropic")
            # Anthropic's server tools web_search and web_fetch (web_search_20250305, web_fetch_20250910): the search
            # tools, run here (serve/search.py); without a search server they stay the client tools they were
            given = req.get("tools") if isinstance(req.get("tools"), list) else []
            hosted = [t for t in given if isinstance(t, dict) and str(t.get("type", "")).startswith(("web_search_",
                                                                                                    "web_fetch_"))]
            search_tools = None
            if hosted and svc.search_backend is not None:
                req = {**req, "tools": [t for t in given if t not in hosted]}
                ws = next((t for t in hosted if t["type"].startswith("web_search_")), hosted[0])
                uses = [t.get("max_uses") for t in hosted if isinstance(t.get("max_uses"), int)]
                search_tools = search_api.SearchTools(svc.search_backend, {
                    "include": ws.get("allowed_domains"), "exclude": ws.get("blocked_domains"),
                    "max_uses": sum(uses) if uses else None,
                    "locale": search_api.locale(ws.get("user_location"), self.headers.get("Accept-Language"))})
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            self._no_local_images(messages)
            if search_tools is not None:
                want = {"web_search": any(t["type"].startswith("web_search_") for t in hosted),
                        "fetch_page": any(t["type"].startswith("web_fetch_") for t in hosted)}
                offered = [t for t in search_tools.template_tools(exclude={t["name"] for t in tools or []})
                           if want.get(t["name"])]
                tools = (tools or []) + offered or None
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # Anthropic's {"type": "none"}: no tools offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools,        # "any" / {"type": "tool", "name": N}
                                getattr(svc.template, "call_form", "qwen"))
            messages, validator = anthropic_prepare_format(req, messages, tools)
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            strict, patterns = constrained_tools(req.get("tools"))
            svc.attach_grammar(req, answer_grammar(anthropic_format(req), thinking) if validator is not None
                               else tool_grammar(tools, strict, patterns, thinking,
                                                 getattr(svc.template, "call_form", "qwen")))
            svc.check_strict(req, tools, strict)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            run = None
            if search_tools is not None and tools:
                if validator is not None:
                    raise ValueError("output_config.format with the web search tools is not supported")
                run = run_with_mcp(svc, search_tools, messages, tools, kw, ids, thinking, max_new,
                                   int(req.get("max_tokens") or 0), req, cancel,
                                   {t["name"] for t in tools} & set(search_tools.routes()), force=force)
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force, run=run)
            if validator is not None:
                events = anthropic_structured(events, validator)
            events = self._capture(events, "anthropic")
            if not req.get("stream"):
                try:
                    return self._json(200, anthropic_collect(events))
                except StructuredOutputError as exc:
                    return self._json(502, {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
            self._sse()
            pinged = 0.0
            try:
                for item in events:
                    if item is None:
                        # Anthropic's keep-alive is a ping event; one a second at most (a prompt being read sends a
                        # heartbeat per chunk), SSE comments between them
                        if time.monotonic() - pinged >= 1:
                            pinged = time.monotonic()
                            self.wfile.write(b'event: ping\ndata: {"type": "ping"}\n\n')
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except StructuredOutputError as e:
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except Exception as e:  # noqa: BLE001 - a bug after the stream started: the error event, as above
                err = {"type": "error", "error": {"type": "api_error", "message": self._stream_bug(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

        def _stream_bug(self, e: Exception) -> str:
            """An error no other handler expects, raised after a stream's headers went out (a bug): its traceback to
            the log, and the sentence its error event carries - a stream that only stopped read as an empty answer."""
            import traceback
            print(f"[strata] error in a stream ({type(e).__name__}: {e}):\n" + traceback.format_exc(), flush=True)
            return f"internal error ({type(e).__name__}: {e}); the server's log has the details"

    return Handler


class Server(ThreadingHTTPServer):
    svc: Service                       # set by serve()
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the
    # kernel (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        if ":" in server_address[0]:                    # an IPv6 address (`::`): an IPv6 socket (upstream eba3106c0)
            self.address_family = socket.AF_INET6
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)

    def server_close(self):
        # the hardware sampler serve() started for this server stops with it (test isolation: every Service a test
        # started used to leave its sampler running, 21 of them after test_responses, which slowed test_parallel's
        # timing)
        tel = getattr(getattr(self, "svc", None), "telemetry", None)
        if tel is not None and hasattr(tel, "close"):
            tel.close()
        super().server_close()


# upstream #560 #516: below this, an AMD card that also drives a Linux desktop can run out
DESKTOP_FREE_MIB = 2048
DESKTOP_RESERVE_MIB = 3072       # what kept KDE/Wayland alive beside a full expert cache in both reports


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X): its compositor, browser and apps take VRAM after the model has."""
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def amd_card(pci) -> bool:
    """The card at PCI address `pci` is AMD's (sysfs vendor 0x1002)."""
    try:
        return bool(pci) and Path(f"/sys/bus/pci/devices/{pci}/vendor").read_text().strip() == "0x1002"
    except OSError:
        return False


def desktop_vram_note(amd: bool, vram_free_mib, args: list, desktop: bool) -> str:
    """upstream #560 #516: on Linux, when the desktop needs VRAM the AMD card does not have, amdgpu moves GPU memory
    (the expert cache) to system RAM, which the experts already fill - the OOM killer then ends the compositor.  The
    default reserve (700 MiB) is sized for a card without a desktop.  A recommendation, nothing changes: "" when it
    does not apply."""
    if not amd or not desktop or not isinstance(vram_free_mib, int) or vram_free_mib >= DESKTOP_FREE_MIB:
        return ""
    try:
        reserve = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args else 700
    except (ValueError, IndexError):
        reserve = 700
    if reserve >= DESKTOP_RESERVE_MIB:
        return ""
    return (f"[strata] note: {vram_free_mib} MiB of VRAM free with the model loaded. If this AMD card also drives your "
            "desktop and the desktop or apps crash after the start (the driver moves the expert cache to RAM and "
            "the OOM killer ends the session), keep more VRAM free: setup with --vram-reserve-mib "
            f"{DESKTOP_RESERVE_MIB} (remembered; the expert cache gets "
            f"{(DESKTOP_RESERVE_MIB - reserve) / 1024:.1f} GB less, a few % of speed)")


def warn_tight_ram(arena_mib, experts_ram=None) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux may end the engine mid-answer when memory runs out (issue #27);
    say so at start instead of after a lost answer.  #11: an engine that says `experts_ram` gives in arena_mib what it
    placed in RAM - the whole arena, or the resident copy of the experts the GPU does not hold (`resident`) - and an
    older one the whole arena."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        what = ("the RAM copy of the experts the GPU does not hold takes" if experts_ram == "resident"
                else "the model's experts take")
        print(f"[strata] WARNING: RAM is tight - {what} {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + "Linux may stop the engine in the middle of an answer. "
              + f"Close other programs, or run {SETUP_CMD} --setup and pick a smaller size (Q2_0 / IQ2_XS).",
              flush=True)


def warn_budget_over_ram(args) -> str | None:
    """#1080: an explicit `--resident-budget-gib N` bigger than the RAM this PC has free (or than its total beside
    ~6 GB for everything else) pushes experts into swap or the file tier: decode that is "dead slow" until the number is
    removed.  A warning at start, never a refusal (recommend, never force).  -> the sentence printed, or None."""
    try:
        budget = float(args[list(args).index("--resident-budget-gib") + 1])
    except (ValueError, IndexError):
        return None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, free = vm.total / 2**30, vm.available / 2**30
    except Exception:  # noqa: BLE001 - psutil is optional here
        return None
    if budget <= free and budget <= total - 6:
        return None
    msg = (f"WARNING: the RAM budget for experts is {budget:g} GiB (--resident-budget-gib), but this PC has "
           f"{free:.0f} GiB free of {total:.0f} GiB: what does not fit goes to swap or is read from disk, which can "
           "make decoding very slow. Lower it (setup sizes it for this PC) or remove the argument to let the engine "
           "choose.")
    print("[strata] " + msg, flush=True)
    return msg


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first: str | None = None
    ips: set[str] = set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(str(info[4][0]))
    except OSError:
        pass

    def ok(ip: str) -> bool:
        return bool(ip) and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if first is not None and ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def host_name(value) -> str:
    """The name in a Host header (or an origin's host[:port]): "Example.com:8080" -> "example.com",
    "[::1]:8095" -> "::1"; "" when it is malformed."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, sep, rest = v[1:].partition("]")
        return name if sep and (not rest or (rest[:1] == ":" and rest[1:].isdigit())) else ""
    if v.count(":") == 1:
        v, port = v.split(":")
        if not port.isdigit():
            return ""
    elif ":" in v:                                       # a bare IPv6 address (a config entry)
        return v
    v = v.rstrip(".")
    return v if v and all(c.isalnum() or c in "-._" for c in v) else ""


def _is_ip(name: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name: str, names) -> bool:
    """`name` is one of `names`, or below an entry that starts with a dot (".example.com")."""
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def allowed_hosts_of(value, env: str = "") -> list[str]:
    """The config's allowed_hosts (a name or a list) plus $STRATA_ALLOWED_HOSTS (comma-separated): host names, "*" or
    ".example.com" (it and every name below it).  A scheme, port or path is dropped ("https://a.example.com:8443/"
    -> "a.example.com"); a wrong entry stops the start (ValueError)."""
    items = [] if value in (None, "") else [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("allowed_hosts: expected a host name or a list of them")
    out = []
    for raw in items + [x for x in env.split(",") if x.strip()]:
        x = raw.strip().lower()
        if x == "*":
            out.append(x)
            continue
        x = x.split("://", 1)[-1].split("/", 1)[0]
        dot = x.startswith(".")
        name = host_name(x[1:] if dot else x)
        if not name:
            raise ValueError(f"allowed_hosts: {raw!r} is not a host name like strata.example.com")
        out.append("." + name if dot else name)
    return list(dict.fromkeys(out))


def host_names_for(bind_host: str, allowed_hosts=(), trusted_origins=()) -> set[str]:
    """Every name this server answers to besides an IP address: localhost, the address it listens on, the config's
    allowed_hosts and trusted_origins' hosts, and - listening beyond this PC (0.0.0.0 or a LAN address) - this PC's
    name and LAN addresses (the LAN addresses matter for the Origin check, which takes no IP on trust)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        # the names a container reaches this PC by: Docker's and Podman's
        names.update(("host.docker.internal", "host.containers.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    names.update(filter(None, (host_name(o.split("://", 1)[-1]) for o in trusted_origins)))
    return names


def host_allowed(host, names, any_host=False) -> bool:
    """DNS rebinding: a web page of another site whose name its DNS points at 127.0.0.1 reaches this server as the
    same origin, so the browser lets it read every answer.  Its requests carry that site's name in Host, so only
    the names this server answers to pass.  An IP address passes (a page served from an IP is that IP's own page;
    rebinding needs a name), as does "*.localhost" (browsers never ask DNS for it) and a request without a Host
    header (HTTP/1.0 clients; browsers always send one)."""
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    return bool(name) and (_is_ip(name) or name == "localhost" or name.endswith(".localhost")
                           or _name_in(name, names))


def key_list(value) -> list[str]:
    """#1344: the keys a server accepts.  --api-key / STRATA_API_KEY / "api_key" take llama.cpp's form, several keys
    separated by commas ("k1,k2"; spaces around each are dropped), or a list in the config ("api_key": ["k1", "k2"],
    the way to give a key that contains a comma).  A list is taken as it is."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def explicit_no_key(value) -> bool:
    """The owner chose to serve without a key: "api_key": "none" in the config, STRATA_API_KEY=none or --api-key none
    (or --no-api-key, handled by the caller).  Never a key called "none"."""
    return isinstance(value, str) and value.strip().lower() == "none"


def open_server_warning(port, tools=False) -> str:
    """The one plain warning for a server on the network with no API key (recommend, never force)."""
    msg = f"no API key: anyone who can reach port {port} can use the model" + (" and its tools" if tools else "") + "."
    if tools:
        msg += " The tools can read and change files on this computer."
    return msg + " Add \"api_key\": \"...\" to the config to keep others out (clients send it as their API key)."


def api_key_of(value):
    """The key as a client can send it (#725).  An HTTP header loses the spaces and line ends around its value, so a
    key kept with them (a config file or an environment file with CRLF line ends, a quoted " key ") matched no
    request: every client got 401 with the right key.  A value that is only such characters raises ValueError, it
    never means "no key" (#213).  #1344: a config list of keys returns the list (each key stripped, none empty); a
    string stays a string (its commas separate keys, see key_list)."""
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value]
        if not keys or any(not k for k in keys):
            raise ValueError("an API key was given but it is empty")
        return keys
    key = "" if value is None else str(value)
    if key and not key_list(key):
        raise ValueError("an API key was given but it is empty")
    return key.strip()


def key_matches(given: str, key) -> bool:
    """given: a header's value as http.server read it, each byte one character.  A key with characters outside ASCII
    arrives as UTF-8 from most clients and as Latin-1 from some; the right key passes in both forms (#725: the
    bytes were re-encoded before, so a UTF-8 key never matched).  Constant-time (#213).  #1344: `key` may name several
    keys (see key_list); the given one must match any, and every key is compared whatever the outcome."""
    raw = given.encode("latin-1", "replace")
    ok = False
    for k in key_list(key):
        ok |= hmac.compare_digest(raw, k.encode())
        try:
            ok |= hmac.compare_digest(raw, k.encode("latin-1"))
        except UnicodeEncodeError:
            pass
    return ok


def origin_allowed(origin: str, host, names, origins=()) -> bool:
    """A browser page's Origin that may use the model without an API key: this server's own page (the Origin is the
    request's own Host), a page on one of the names this server answers to (any port), or an origin the config lists
    (trusted_origins, cors_origins).  Unlike the Host check no IP passes on trust: a page served from any other IP is
    another site.  "null" (a sandboxed frame, a file:// page) does not pass: any site can send it.  An origin of another
    scheme (chrome-extension://, moz-extension://, an Electron app's app://) does: no web site can send one."""
    origin = (origin or "").strip().rstrip("/")
    if origin in origins or "*" in origins:             # cors_origins ["*"]: the config lets every page in
        return True
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False
    if scheme not in ("http", "https"):
        return True
    if host and rest == host.strip().lower():
        return True
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def origins_of(value, key: str, wildcard: bool) -> list[str]:
    """#321: a config's origin list ("https://chat.example.com" or a list of them) - scheme://host[:port], no path;
    "*" only where `wildcard` allows it.  A wrong entry stops the start rather than opening less or more than meant."""
    if value is None or value == "" or value == []:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise SystemExit(f"[strata] {key}: expected an origin or a list of origins")
    out = []
    for x in items:
        x = x.strip().rstrip("/")
        if x == "*" and wildcard:
            out.append(x)
            continue
        scheme, sep, rest = x.partition("://")
        if scheme not in ("http", "https") or not sep or not rest or "/" in rest or "*" in rest:
            raise SystemExit(f"[strata] {key}: {x!r} is not an origin like https://chat.example.com"
                             + ("" if wildcard else " (no wildcards here)"))
        out.append(x)
    return out


SHARED_KEYS = ("reasoning_effort", "reasoning_budget_tokens", "temperature", "top_p", "top_k", "seed", "max_tokens",
               "experimental_speed_projection", "script_guard")


def script_guard_name(value) -> str:
    """A script guard setting as /props and /metrics show it: "off", "ja", or "ja" with ", allow_context: false"."""
    if not isinstance(value, dict):
        return "off"
    return value["lang"] + ("" if value["allow_context"] else ", allow_context: false")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high", "xhigh", "max"):
                raise ValueError("reasoning_effort: none, low, medium, high or max")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "reasoning_budget_tokens":                 # #123: 0 means no budget, so 0 is allowed here
            if not number or value != int(value) or value < 0:
                raise ValueError("reasoning_budget_tokens: a whole number of tokens (0: no budget)")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        elif key == "script_guard":
            value = script_guard.parse(value)
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


def split_joined_args(cfg: dict) -> None:
    """#2267: an engine option written as ONE element of the config's "args" - "--ple-io mmap" or "--ple-io=mmap" -
    reaches the engine as a single argument ("unknown argument: --ple-io mmap"; the usual form is two elements,
    "--ple-io", "mmap").  Recommend, never force: such an element is split into the flag and its value, and said at
    startup.  Only an element that starts with a flag name (--letters-digits-dashes) followed by a space or "=" is
    touched, and never one that sits in the value place of the flag before it."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out: list = []
    changed: list[str] = []
    for a in args:
        m = re.match(r"^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$", a) if isinstance(a, str) else None
        prev = out[-1] if out else None
        in_value_place = isinstance(prev, str) and re.match(r"^--[A-Za-z0-9][A-Za-z0-9-]*$", prev) is not None
        if m is None or in_value_place:
            out.append(a)
            continue
        out += [m.group(1), m.group(2).strip()]
        changed.append(a)
    if changed:
        cfg["args"] = out
        print("[strata] the config's args hold " + ", ".join(repr(c) for c in changed) + " as one element each: split "
              'into the flag and its value (write them as two elements, e.g. "--ple-io", "mmap")', flush=True)


def read_config(path) -> dict:
    """The run config's JSON.  A mistake in it (a missing comma between two args, a trailing comma) is said with its
    place in the file instead of a traceback (#2267: editing "args" by hand is where it happens)."""
    text = Path(path).read_text(encoding="utf-8-sig")   # Notepad adds a BOM
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[strata] {path} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}. Check the "
                         'commas between the entries of "args" (each option and each value is its own quoted entry, '
                         'e.g. "--ple-io", "mmap",) and that no entry is followed by a comma at the end of a '
                         'list.') from None


def lift_reasoning_effort_arg(cfg: dict) -> None:
    """#1641: "--reasoning-effort high" in a config's "args" is meant for the thinking level, which is not an engine
    option (the engine stopped at the unknown flag).  It is the config's "reasoning_effort" (none, low, medium or high:
    the level for requests that name none), so the flag is taken out of the engine's arguments and becomes that key
    unless the config already has one; said at startup."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, value, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a == "--reasoning-effort" and i + 1 < len(args):
            value, i = args[i + 1], i + 2
        elif isinstance(a, str) and a.startswith("--reasoning-effort="):
            value, i = a.partition("=")[2], i + 1
        else:
            out.append(a)
            i += 1
    if value is None:
        return
    cfg["args"] = out
    if cfg.get("reasoning_effort") is None:
        cfg["reasoning_effort"] = value
    print("[strata] '--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking "
          "level is the config's 'reasoning_effort' key (outside 'args'); it is used as " +
          f"{cfg['reasoning_effort']!r}", flush=True)


# #1819: llama.cpp clients call the repetition penalty and its window repeat_penalty / repeat_last_n; they are the
# same settings as repetition_penalty / penalty_last_n (the /props answer already reports them under those names)
SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty (also written repeat_penalty), frequency_penalty, penalty_last_n (also
    repeat_last_n), seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    block = dict(cfg.get("sampling") or {})
    for alias, canon in SAMPLING_ALIASES.items():      # #1819: llama.cpp's names for the same two fields
        if block.get(alias) is not None:
            if block.get(canon) is not None:
                raise SystemExit(f"[strata] config sampling: both {alias!r} and {canon!r} are set; they are the same "
                                 f"setting, keep one")
            block[canon] = block.pop(alias)
    for key, value in block.items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[strata] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[strata] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[strata] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[strata] config sampling.repetition_penalty={value!r}: "
                                 "expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[strata] config sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[strata] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[strata] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def loop_guard_setting(cfg: dict | None, env=None) -> bool:
    """The loop guard (serve/loop_guard.py), on unless STRATA_LOOP_GUARD says 0 / off / false / no or the run config
    has "loop_guard": false; the environment, when set, wins over the config."""
    off = ("0", "off", "false", "no", "none")
    env = os.environ if env is None else env
    v = env.get("STRATA_LOOP_GUARD")
    if v is not None and str(v).strip():
        return str(v).strip().lower() not in off
    v = (cfg or {}).get("loop_guard")
    return str(v).strip().lower() not in off if isinstance(v, str) else v is not False and v != 0


def check_request_sampling(req) -> None:
    """A request's sampling field of the wrong JSON type ("temperature": "0.7", "top_k": 40.0, "seed": true) is a 400
    that names the field, as the same value is through POST /settings or the run config.  Unchecked, the value
    replaced the configured default in run()'s merge and sampling_keys dropped it, so the reply was greedy.  Only
    types are checked: a number (int or float, never bool) for temperature, top_p, min_p and the three penalties; an
    integer (never bool or float) for top_k, penalty_last_n and seed.  Ranges behave as before (a temperature <= 0 is
    greedy, top_k 0 or above 64 is clamped, a seed <= 0 is unset), and an absent or null field is not checked."""
    if isinstance(req, dict):      # #1819: llama.cpp's names; the OpenAI-style field wins when both are sent
        if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
            req["repetition_penalty"] = req["repeat_penalty"]
        rl = req.get("repeat_last_n")
        if req.get("penalty_last_n") is None and isinstance(rl, int) and not isinstance(rl, bool) and rl > 0:
            req["penalty_last_n"] = rl      # 0 and -1 (off / the whole context) keep the engine's default window
    for key, integer in (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                         ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                         ("penalty_last_n", True), ("seed", True)):
        value = (req or {}).get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():      # some clients send numbers as strings: read and used
            try:
                f = float(value)
                num: float | None = int(f) if integer and f == int(f) else (f if not integer else None)
                if integer and num is None:
                    raise ValueError
                if not integer and re.fullmatch(r"\s*[+-]?\d+\s*", value):
                    num = int(value)
            except (ValueError, OverflowError):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}") from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
    guard = (req or {}).get("loop_guard")          # the loop guard for this request only (serve/loop_guard.py)
    if guard is not None and not isinstance(guard, bool):
        if isinstance(guard, str) and guard.strip().lower() in ("true", "false", "on", "off", "1", "0"):
            req["loop_guard"] = guard.strip().lower() in ("true", "on", "1")
        elif isinstance(guard, int) and guard in (0, 1):
            req["loop_guard"] = bool(guard)
        else:
            raise ValueError("loop_guard: true or false")
    if (req or {}).get("script_guard") is not None:  # the script guard for this request (serve/script_guard.py)
        req["script_guard"] = script_guard.parse(req["script_guard"])


def start_replicas(cfg: dict, groups: list[list[str]], exe: str, effort_end: list[str], lazy: bool):
    """Opt-in data-parallel replicas (serve/replicas.py, upstream 8d276c19): one StrataEngine per card group, each with
    its own cards, log and CPU cores; the ReplicaEngine in front spreads the requests.  On Linux the replicas share one
    expert arena in RAM (upstream 3ccfd5aed, 378200894; "replica_shared_arena": false gives each its own copy), as
    split serving's engines do: the first replica fills it, the others start after it and map what it filled."""
    from serve import replicas as rp
    from serve import split as sp
    n = len(groups)
    cpus = rp.cpu_sets(n, cfg.get("replica_cpus") is not False)
    arena, arena_fd, arena_note = sp.arena_path(cfg, {"shared_arena": cfg.get("replica_shared_arena", True)})
    print(f"[strata] {n} replicas on the cards {' | '.join(','.join(g) for g in groups)}"
          f"{' (own CPU cores each)' if cpus else ''}: requests go to the least busy one, "
          f"a conversation back to its own; {arena_note}",
          flush=True)
    engines: list = [None] * n
    errors: list = []

    def one(i: int):
        rc = rp.replica_cfg(cfg, i, groups[i])
        if arena:
            rc["args"] = list(rc.get("args") or []) + ["--shared-expert-arena", arena]
        env = child_env(rc)
        env["STRATA_GPU_PCI"] = rc["gpu_pci"]           # its own card, whatever the server's environment names
        try:
            engines[i] = StrataEngine(exe, engine_args(rc) + effort_end, cwd=rc.get("cwd"), log=rc.get("log"),
                                      env=env, lazy=lazy, cpus=cpus[i] if cpus else None,
                                      fds=(arena_fd,) if arena_fd is not None else ())
        except BaseException as e:        # noqa: BLE001
            errors.append((i, e))
    rc0 = rp.replica_cfg(cfg, 0, groups[0])
    warn_budget_over_ram(engine_args(rc0) if "args" in rc0 else [])
    if os.environ.get("STRATA_REPLICA_START", "") == "serial":
        for i in range(n):
            one(i)
    else:
        rest = range(n)
        if arena:                         # the arena filled once: the first replica, then the others beside each other
            one(0)
            rest = range(1, n) if not errors else range(0)
        ts = [threading.Thread(target=one, args=(i,)) for i in rest]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errors:
        for e in engines:
            if e is not None:
                try:
                    e.close()
                except Exception:        # noqa: BLE001
                    pass
        if arena_fd is not None:
            os.close(arena_fd)
        i, e = errors[0]
        raise SystemExit(f"[strata] replica {i} did not start: {e}")
    load = int(cfg.get("replica_load_tokens") or rp.LOAD_TOKENS)
    return rp.ReplicaEngine(engines, load_tokens=load)


class KeptArena:
    """The one engine's expert arena in memory the server holds, when image jobs may unload the engine: started again
    after the image, it maps the arena instead of reading the experts from the model files (the SSD, when the page
    cache has let them go).  On Linux a memory file (--shared-expert-arena /proc/self/fd/N, as split.arena_path makes
    it), on Windows a page-file-backed section (section:<name>, SEC_RESERVE and as large as the RAM: the engine commits
    its arena in it).  Before such an unload, when the image needs the RAM, the engine marks its coldest experts
    dropped (arena_drop); once it has ended their pages are given back here - punched out of the file, or reset in the
    section (MEM_RESET: the system takes them without writing them to the page file) - and the next start reads those
    alone.  An unload for any other reason gives the whole arena back (a new, empty one in its place)."""

    PUNCH = 0x02 | 0x01                              # FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE
    PAGE = 4096

    def __init__(self, fd: int | None = None):
        self.fd = fd
        self.handle: int | None = None               # Windows: the section
        self.name = ""
        if fd is None:
            self.name = f"Local\\xestrata-arena-{os.getpid()}-{os.urandom(4).hex()}"
            self.handle = self._section()
            self.path = f"section:{self.name}"
        else:
            self.path = f"/proc/self/fd/{fd}"
        self.pending: str | None = None              # arena_drop's ranges file, given back after the unload

    @staticmethod
    def _k32():
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)        # type: ignore[attr-defined]
        k.CreateFileMappingW.restype = wintypes.HANDLE
        k.CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.DWORD, wintypes.LPCWSTR]
        k.MapViewOfFile.restype = ctypes.c_void_p
        k.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
        k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        k.VirtualAlloc.restype = ctypes.c_void_p
        k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        return k

    def _section(self) -> int:
        """A new page-file-backed section of the RAM's size, reserved (committed by the engine's view as it fills)."""
        import ctypes
        import psutil
        size = psutil.virtual_memory().total
        h = self._k32().CreateFileMappingW(ctypes.c_void_p(-1).value, None, 0x04 | 0x4000000,   # PAGE_READWRITE |
                                           size >> 32, size & 0xFFFFFFFF, self.name)          # SEC_RESERVE
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateFileMappingW")    # type: ignore[attr-defined]
        return h

    def after_unload(self, keep: bool) -> None:
        pending, self.pending = self.pending, None
        if not keep:
            self.reset()
        elif pending:
            try:
                n = self.punch(pending)
                print(f"[strata] expert arena: {n / 2 ** 30:.2f} GiB of the coldest experts given back to the RAM "
                      "(the next start reads them again)", flush=True)
            except OSError as e:
                print(f"[strata] expert arena: the dropped experts could not be given back ({e}); the whole arena "
                      "goes", flush=True)
                self.reset()
        with contextlib.suppress(OSError):
            if pending:
                os.unlink(pending)

    def punch(self, ranges_file: str) -> int:
        """The ranges ("offset bytes" lines) given back -> their bytes.  In a section only the whole pages inside each
        range can be reset (the page at each end stays, a few KiB an expert)."""
        import ctypes
        with open(ranges_file, encoding="ascii") as f:
            ranges = [tuple(int(x) for x in line.split()) for line in f if line.strip()]
        if self.fd is not None:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
            for off, n in ranges:
                if libc.fallocate(self.fd, self.PUNCH, off, n) != 0:
                    raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
            return sum(n for _, n in ranges)
        k = self._k32()
        view = k.MapViewOfFile(self.handle, 0xF001F, 0, 0, 0)       # FILE_MAP_ALL_ACCESS, the whole section
        if not view:
            raise OSError(ctypes.get_last_error(), "MapViewOfFile")         # type: ignore[attr-defined]
        total = 0
        try:
            for off, n in ranges:
                a, e = -(-off // self.PAGE) * self.PAGE, (off + n) // self.PAGE * self.PAGE
                if e > a:
                    if not k.VirtualAlloc(view + a, e - a, 0x80000, 0x04):   # MEM_RESET, PAGE_READWRITE
                        raise OSError(ctypes.get_last_error(), "VirtualAlloc MEM_RESET")   # type: ignore[attr-defined]
                    total += e - a
        finally:
            k.UnmapViewOfFile(view)
        return total

    def reset(self) -> None:
        """A new, empty arena under the same name (the engine's argument stays valid); the old one's RAM goes with it:
        a memory file at the same descriptor, or the section closed and made again."""
        if self.fd is None:
            self._k32().CloseHandle(self.handle)
            self.handle = self._section()
            return
        new = os.memfd_create("strata-arena")
        os.dup2(new, self.fd, inheritable=False)
        os.close(new)


def kept_arena(cfg: dict, args: list[str]) -> tuple[KeptArena | None, str]:
    """The one engine's KeptArena, or None and why not: only with image generation on the text model's card (an image
    job may unload it), not with a layer split or an engine that reads its experts from the files or keeps part of
    them in RAM, nor when the "images" section says "keep_arena": false."""
    from serve import split as sp
    images = cfg.get("images")
    if not isinstance(images, dict):
        return None, ""
    if images.get("keep_arena") is False:
        return None, "the experts are read again after an image job (\"keep_arena\": false)"
    if cfg.get("split_pci"):                         # a shared mapping is not moved into host USM for the PCIe share
        return None, "the experts are read again after an image job (a layer split)"
    tier = next((f for f in ("--mmap-experts", "--resident-experts", "--resident-budget-gib", "--shared-expert-arena")
                 if f in args), None)
    if tier:
        return None, f"the experts are read again after an image job ({tier})"
    said = "the experts stay in RAM across an image job that unloads the model (a shared arena)"
    if os.name == "nt":
        try:
            return KeptArena(), said
        except (OSError, ImportError) as e:
            return None, f"the experts are read again after an image job: no section for them ({e})"
    path, fd, note = sp.arena_path({"args": args}, {"shared_arena": True})
    if fd is None:
        return None, f"the experts are read again after an image job: {note}"
    return KeptArena(fd), said


def start_split(cfg: dict, spec: dict, exe: str, effort_end: list[str], lazy: bool):
    """Opt-in split serving (serve/split.py, upstream 690172c46): a prefill engine and a decode engine, one card each,
    the same model and the same session settings.  Started one after the other, so a shared expert arena is filled
    once; the SplitEngine in front hands a long prompt from the first to the second."""
    from serve import split as sp
    sp.check_cfg(cfg, spec)
    arena, arena_fd, arena_note = sp.arena_path(cfg, spec)
    work = sp.session_dir(spec)
    print(f"[strata] split serving: the prompt is read on card {spec['prefill_gpu']}, the answer is made on card "
          f"{spec['decode_gpu']} (prompts with {spec['min_prompt_tokens']}+ new tokens are handed over); {arena_note}",
          flush=True)
    engines = {}
    try:
        for role in ("prefill", "decode"):
            rc = sp.role_cfg(cfg, spec, role)
            if arena:
                rc["args"] = list(rc["args"]) + ["--shared-expert-arena", arena]
            if role == "prefill":
                warn_budget_over_ram(engine_args(rc) if "args" in rc else [])
            env = child_env(rc)
            env["STRATA_GPU_PCI"] = rc["gpu_pci"]       # its own card, whatever the server's environment names
            engines[role] = StrataEngine(exe, engine_args(rc) + effort_end, cwd=rc.get("cwd"), log=rc.get("log"),
                                         env=env, lazy=lazy, fds=(arena_fd,) if arena_fd is not None else ())
    except BaseException:
        for e in engines.values():
            try:
                e.close()
            except Exception:        # noqa: BLE001
                pass
        shutil.rmtree(work, ignore_errors=True)
        if arena_fd is not None:
            os.close(arena_fd)
        raise
    return sp.SplitEngine(engines["prefill"], engines["decode"], min_tokens=spec["min_prompt_tokens"], work_dir=work,
                          arena_fd=arena_fd)


# The sampling each model card recommends ("Best Practices"), used when neither the request, the Chat settings nor the
# run config's `sampling` block says otherwise.  Greedy is only for a request or config that asks for temperature 0.
# min_p 0, repetition_penalty 1 and presence_penalty 0 are the engine's own "off", so they are not listed.
MODEL_CARD_SAMPLING = {
    # Qwen3.8-Flash-Next (and Gyro, the same base model)
    "qwen4exp": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                 "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # Qwen3.6-35B-A3B and Qwen3.5-9B: the general-task sets (the cards' precise-coding set is temperature 0.6,
    # presence_penalty 0)
    "qwen35moe": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                  "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # HauhauCS's Qwen3.5-9B Uncensored Aggressive: its card's two sets
    "hauhaucs": {"thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                 "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
    # Ornith-1.5: the card's set for general tasks (its benchmarks ran at temperature 1.0)
    "ornith": {"thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
               "instruct": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
    # dealignai's Ornith-1.5 (CRACK): its card's one set, with or without thinking
    "dealignai": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                  "instruct": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    # GLM-5.3-Flash: its generation_config.json
    "glm5": {"thinking": {"temperature": 1.0, "top_p": 0.95}, "instruct": {"temperature": 1.0, "top_p": 0.95}},
}


def card_family(cfg: dict) -> str | None:
    """Which MODEL_CARD_SAMPLING set a config's model takes: Qwen3.8-Flash-Next and its variants (Swift, Gyro: upstream
    gives every qwen4exp file the set, "qwen4exp"), Qwen3.6-35B-A3B and Qwen3.5-9B ("qwen35moe"), HauhauCS's Qwen3.5-9B
    ("hauhaucs"), Ornith-1.5 ("ornith"), dealignai's Ornith-1.5 ("dealignai"), GLM-5.3-Flash ("glm5"); None for
    llm-jp (its card names no sampling), which stays greedy unless the request or the config's `sampling` block says
    otherwise.  By the model's name, as the architecture alone does not tell Qwen3.6 from Ornith."""
    name = str(cfg.get("model_name") or "qwen3.8-flash-next").lower()
    if name.startswith(("qwen3.8", "swift")) or "gyro" in name:
        return "qwen4exp"
    if name.startswith("qwen3.5") and "uncensored" in name:
        return "hauhaucs"
    if name.startswith(("qwen3.6", "qwen3.5")):
        return "qwen35moe"
    if name.startswith("ornith") and "uncensored" in name:
        return "dealignai"
    if name.startswith("ornith"):
        return "ornith"
    if name.startswith("glm"):
        return "glm5"
    return None


def model_card_sampling(arch, thinking) -> dict:
    """The model card's sampling for the family `arch` and the thinking state ({} for an unknown or unset family)."""
    sets = MODEL_CARD_SAMPLING.get(arch) if arch else None
    return dict(sets["thinking" if thinking else "instruct"]) if sets else {}


def tool_call_recovery_setting(cfg: dict) -> bool:
    """The config's "tool_call_recovery" (see OutputParser's recover): on when the config does not name it,
    "tool_call_recovery": false turns it off.  Anything but true or false stops the start."""
    recovery = cfg.get("tool_call_recovery", True)
    if not isinstance(recovery, bool):
        raise SystemExit(f"[strata] config \"tool_call_recovery\" must be true or false, not {recovery!r}")
    return recovery


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), "
                         "0.0.0.0 = also other devices on your network, :: = all IPv6 interfaces (set an API key); "
                         "also \"host\" in the config")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(the default; kept so older scripts keep working)")
    ap.add_argument("--no-fit-max-tokens", action="store_true",
                    help="answer 400 when prompt + max_tokens does not fit the context, like llama.cpp "
                         "(also \"fit_max_tokens\": false in the config). A prompt that alone does not fit is "
                         "always 400")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY. "
                         "Several keys: separate them with commas (key1,key2), as llama.cpp does")
    ap.add_argument("--no-api-key", action="store_true",
                    help="serve on the network without an API key on purpose (anyone who can reach the port can use "
                         "the model; also \"api_key\": \"none\" in the config): no warning about an empty key")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--lazy", action="store_true",
                    help="start the API unloaded (the engine and the vision encoder start with the first request)")
    ap.add_argument("--api-monitor", action="store_true",
                    help="the API request monitor at /api-monitor: keeps the last 100 requests' prompts and answers in "
                         "memory (also \"api_monitor\": true in the config; off by default)")
    ap.add_argument("--debug-chat-log", action="store_true",
                    help="write every chat request's prompt, tool rounds, answer and pictures, and the web page's "
                         "chats, to the state folder's debug-chat-log/ (also \"debug_chat_log\": true in the config "
                         "or the settings; off by default)")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests, so other programs (games, other "
                         "model servers) can use the VRAM; the next request loads it again (also \"idle_unload_s\" "
                         "in the config; default: never)")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503 (also "
                         "\"min_free_vram_mib\" in the config; default: always load)")
    ap.add_argument("--before-load", help="a command run before the model is loaded again (e.g. to unload another "
                                          "server's model; also \"before_load\" in the config, a string or a list)")
    ap.add_argument("--slot-save-path", default=None, metavar="DIR",
                    help="enable POST /slots/0?action=save|restore {\"filename\": NAME} (llama-server's API): the "
                         "conversation the engine holds, to or from DIR/NAME (also \"slot_save_path\" in the config)")
    ap.add_argument("--replicas", type=int, default=None, metavar="N",
                    help="data-parallel replicas (opt-in): cut the cards (\"gpu_pci\" and \"split_pci\") into N equal "
                         "groups and run one engine on each, requests spread over them (also \"replicas\" in the "
                         "config: a number, or [{\"gpus\": [\"0000:03:00.0\"]}, {\"gpus\": [\"0000:09:00.0\"]}])")
    ap.add_argument("--split-serve", default=None, metavar="PREFILL,DECODE",
                    help="split serving (opt-in): one engine on the card PREFILL reads the prompt, a second on the "
                         "card DECODE makes the answer, the session handed over in RAM; the cards by PCI address "
                         "(0000:03:00.0,0000:09:00.0; also \"split_serve\" in the config: {\"prefill_gpu\": "
                         "\"0000:03:00.0\", \"decode_gpu\": \"0000:09:00.0\"}; docs/MULTIGPU.md)")
    a = ap.parse_args()
    if a.engine == "strata" and (_rt := rttime_limit_note()):   # upstream #3238: say it before the first unexplained -9
        print("[strata] WARNING: " + _rt, flush=True)
    cfg = read_config(a.config) if a.config else {}
    split_joined_args(cfg)
    if a.replicas is not None:
        cfg["replicas"] = a.replicas
    if a.split_serve is not None:
        pair = [x.strip() for x in a.split_serve.split(",")]
        if len(pair) != 2:
            ap.error("--split-serve takes two cards by PCI address: PREFILL,DECODE (for example "
                     "0000:03:00.0,0000:09:00.0)")
        given = cfg.get("split_serve")
        cfg["split_serve"] = {**(given if isinstance(given, dict) else {}), "prefill_gpu": pair[0],
                              "decode_gpu": pair[1]}
    lift_reasoning_effort_arg(cfg)
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError as e:
        ap.error(listen_problem(a.host, a.port, e))
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(tpath)
    hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[strata] sampling defaults from the config: {pretty}", flush=True)
        vision_error = vision_fallback = None
        lazy = a.lazy or cfg.get("lazy_load") is True
        if cfg.get("vision"):
            elastic = vision_elastic_on(cfg) and vram_elastic_on(cfg)
            print("the vision encoder starts with the first picture (elastic) ..." if elastic else
                  "loading the vision encoder ..." if not lazy else
                  "vision encoder unloaded; it starts with the model ...", flush=True)
            # relative paths are the config's cwd's, as for the engine below
            vcfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                        if k in ("exe", "mmproj", "model") and isinstance(v, str) and not os.path.isabs(v) else v)
                    for k, v in cfg["vision"].items()}
            vcfg["elastic"] = vision_elastic_on(cfg)
            if vcfg["elastic"] and not vram_elastic_on(cfg):
                print('[strata] the vision section\'s "elastic" needs "vram_elastic": true (the expert cache gives the '
                      "encoder its VRAM): the encoder starts with the model instead", flush=True)
                vcfg["elastic"] = False
            elif vcfg.get("elastic") is True and (len(gpu_list(cfg)) > 1 or (
                    cfg.get("split_pci") and os.environ.get("STRATA_LAYER_SPLIT") != "off")):
                # the engine turns --vram-elastic off on a layer split (a cache per GPU): the encoder still starts
                # with the first picture and stops when idle, but its VRAM has to be free already (upstream 092c7d1ef)
                print('[strata] images: on a layer split the expert cache cannot make room for the image encoder (the '
                      'engine turns --vram-elastic off); keep its VRAM free with "--vram-reserve-mib" on the '
                      "encoder's GPU, or the first picture may not get it", flush=True)
            failed: list = []
            for vc in vision_attempts(vcfg):
                if failed:
                    print(f"[strata] images: trying {vision_label(vc)} instead", flush=True)
                try:
                    vision = Vision(vc, log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                                    env=env, lazy=lazy)
                    break
                except (OSError, RuntimeError) as e:    # the next one; with none left the model still serves text
                    failed.append(f"{vision_label(vc)}: {e}")
                    print(f"[strata] images: {failed[-1]}", flush=True)
            if vision is None:
                vision_error = "; ".join(failed)
                print("[strata] images are off: no image encoder started", flush=True)
            elif failed:
                vision_fallback = "; ".join(failed)
                print(f"[strata] images: {vision.label} serves them ({vision_fallback})", flush=True)
        print("model unloaded; the first request loads it ..." if lazy else
              "loading the model (the first start takes a minute or two) ...", flush=True)
        try:
            split = layer_split_value(cfg)              # upstream #644: before the (minutes-long) start
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        warn_budget_over_ram(engine_args(cfg) if "args" in cfg else [])      # #1080
        if len(gpu_list(cfg)) > 1:
            print(f"[strata] layer split across GPUs {gpu_list(cfg)} ({split})", flush=True)
        # a relative "exe" is the config's cwd's, made absolute here
        exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else \
            os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
        try:
            silence = engine_silence_s(cfg)             # #481: checked before the (minutes-long) start
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        try:
            effort_end = effort_end_args(cfg, exe, tok)  # #458
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        from serve import replicas as rp
        try:
            groups = rp.gpu_groups(cfg)
        except rp.ReplicaConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        from serve import split as split_mod
        try:
            split_serve = split_mod.split_spec(cfg)
            if split_serve:
                split_mod.check_cfg(cfg, split_serve)
        except split_mod.SplitConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        if groups:
            engine = start_replicas(cfg, groups, exe, effort_end or [], lazy)
        elif split_serve:
            engine = start_split(cfg, split_serve, exe, effort_end or [], lazy)
        else:
            eargs = engine_args(cfg) + (effort_end or [])
            kept, said = (None, "") if image_env(cfg, env)[1] else kept_arena(cfg, eargs)
            if said:
                print(f"[strata] {said}", flush=True)
            if kept is not None:
                eargs += ["--shared-expert-arena", kept.path]
            engine = StrataEngine(exe, eargs, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env, lazy=lazy,
                                  fds=(kept.fd,) if kept is not None and kept.fd is not None else ())
            engine.kept_arena = kept                    # an attribute of its own: restart() keeps it
        engine.silence_s = silence                      # an attribute of its own: restart() keeps it
        warn_tight_ram(engine.info.get("arena_mib"), engine.info.get("experts_ram"))
        note = None if groups or split_serve else desktop_vram_note(amd_card(cfg.get("gpu_pci")),
                                                                    engine.info.get("vram_free_mib"),
                                                                    engine_args(cfg), linux_desktop())
        if note:                                        # upstream #560 #516: before --open starts a browser there
            print(note, flush=True)
    else:
        effort_end = None
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
        vision_error = vision_fallback = None
    # the model's own chat template (exported with its tokenizer), else the original model's, which is under the Qwen
    # Community License and so in third_party/nonfree (XeStrata runs without that folder)
    tpl = tpath / "chat_template.jinja"
    if not tpl.exists():
        tpl = ROOT / "third_party" / "nonfree" / "qwen3.8-flash-next" / "chat_template.jinja"
    if not tpl.exists():
        ap.error(f"no chat template: {tpath / 'chat_template.jinja'} is missing (prepare the model again with setup)")
    svc = Service(engine, tok, ChatTemplate(tpl),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or (not a.no_fit_max_tokens
                                                      and cfg.get("fit_max_tokens") is not False))
    svc.vision_error = vision_error
    svc.vision_fallback = vision_fallback
    if (tpath / "vocab.json").exists():
        svc.tokenizer_dir = str(tpath.resolve())
    svc.public_urls = True                             # #28: a request's image URL only from a public address
    if a.engine == "strata":                            # the model card's sampling when nothing else is said
        svc.model_arch = card_family(cfg)
    svc.answer_room = cfg.get("answer_room") is not False
    svc.reasoning_close_retry = cfg.get("reasoning_close_retry") is True    # #1053: opt-in, off by default
    if cfg.get("images"):                               # image generation (serve/images.py), set up by setup --images
        # the default image model ("images") and the others a request can name ("image_models")
        extra = cfg.get("image_models") or []
        if not isinstance(extra, list) or not all(isinstance(m, dict) for m in extra):
            raise SystemExit('[strata] config: "image_models" must be a list of "images" sections')
        icfgs = [{k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                      if k in ("exe", "diffusion_model", "vae", "llm", "tokenizer", "system_prompt", "out_dir",
                               "lora_dir")
                      and isinstance(v, str)
                      and not os.path.isabs(v) else v)
                  for k, v in section.items()} for section in [cfg["images"], *extra]]
        try:
            ilog = open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None
            # the text engine's environment: its libraries, and its GPU unless the images have a card of their own
            ienv, separate, said = image_env(cfg, child_env(cfg))
            if said:
                print(f"[strata] {said}", flush=True)
            svc.images = images_api.ImageModels(icfgs, log=ilog, env=ienv, separate=separate)
            svc.images.helper_card = image_helper_card(cfg)
            svc.images.gpu_pci = ienv.get("STRATA_GPU_PCI") if separate else None   # the Monitor's image card
        except ValueError as e:
            raise SystemExit(f"[strata] config: {e}")
        gone = svc.images.missing()
        if gone:
            print(f"[strata] image generation: missing {', '.join(gone)} - requests fail until setup --images has run "
                  "again", flush=True)
        for g in svc.images.gens.values():
            print(f"[strata] image generation: {g.model} {g.cfg.get('quant') or ''} on strata-image "
                  f"(POST /v1/images/generations{'' if g is svc.images.default else ', model ' + g.model}; "
                  f"GPU policy {g.policy})", flush=True)
        if svc.images.note():
            print(f"[strata] image generation: {svc.images.note()}", flush=True)
        import atexit
        atexit.register(svc.images.close)               # the image engine ends with the server
    if cfg.get("reasoning_effort") is not None:         # #1641: the thinking level of requests that name none
        try:
            svc.config_effort = clean_shared_defaults({"reasoning_effort": cfg["reasoning_effort"]}).get(
                "reasoning_effort")
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if svc.config_effort:
            print(f"[strata] thinking level: {svc.config_effort} (reasoning_effort in the config; a request, or the "
                  "Chat settings shared with other apps, can set its own)", flush=True)
    svc.codex_thread_titles = cfg.get("codex_thread_titles") is True    # #923: opt-in, off by default
    svc.codex_compaction_cache = cfg.get("codex_compaction_cache") is True   # #924: opt-in, off by default
    try:
        svc.set_aliases(cfg.get("aliases"))             # #297: other names the model answers to
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.aliases:
        print(f"[strata] model aliases: {', '.join(svc.aliases)}", flush=True)
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or \
            any(x == "--api-key" and i + 1 < len(sys.argv) and not sys.argv[i + 1].strip() or x.strip() == "--api-key="
                for i, x in enumerate(sys.argv)):
        # #213: an empty key would switch authentication off without a word
        print("[strata] an API key was given but it is empty: set a key, or leave --api-key / STRATA_API_KEY out",
              file=sys.stderr)
        return 2
    raw_key = a.api_key or cfg.get("api_key", "")
    no_key_chosen = a.no_api_key or explicit_no_key(raw_key)    # upstream 2f361a375: no key on purpose
    if explicit_no_key(raw_key):
        raw_key = ""
    try:
        svc.api_key = api_key_of(raw_key)
    except ValueError as e:
        print(f'[strata] {e}: set a key, or leave --api-key / STRATA_API_KEY / "api_key" out', file=sys.stderr)
        return 2
    # #569: written, but empty: the server has none (a warning, not a stop)
    if not svc.api_key and "api_key" in cfg and not no_key_chosen:
        print("[strata] the api_key in the config is empty: this server has no API key (anyone who can reach it can "
              "use it); set one, or leave api_key out", file=sys.stderr, flush=True)
    svc.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    svc.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    try:
        svc.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.allowed_hosts:
        print("[strata] Host check off: any name reaches this server (allowed_hosts \"*\")"
              if "*" in svc.allowed_hosts
              else f"[strata] also answers to the host names {', '.join(svc.allowed_hosts)} (allowed_hosts)",
              flush=True)
    svc.api_monitor = a.api_monitor or cfg.get("api_monitor") is True
    if svc.api_monitor:
        print("[strata] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in "
              "memory" + ("" if svc.api_key else "; anyone who can reach this server can read them (no API key)"),
              flush=True)
    if svc.cors_origins:
        print(f"[strata] CORS on /v1/* for {', '.join(svc.cors_origins)}"
              + ("" if svc.api_key or "*" not in svc.cors_origins else
                 " - WARNING: any web page may use the model (no API key)"), flush=True)
    svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)
    svc.min_free_vram_mib = a.min_free_vram_mib if a.min_free_vram_mib is not None else \
        int(cfg.get("min_free_vram_mib") or 0)
    svc.before_load = a.before_load or cfg.get("before_load") or None
    try:
        svc.before_load_timeout_s = float(cfg.get("before_load_timeout_s") or svc.before_load_timeout_s)
    except (TypeError, ValueError):
        print("[strata] \"before_load_timeout_s\" in the config is not a number; using "
              f"{svc.before_load_timeout_s:g} s", flush=True)
    svc.before_load_ignore_failure = cfg.get("before_load_ignore_failure") is True
    svc.before_load_strict = cfg.get("before_load_strict") is True
    # --slot-save-path is relative to the working directory the server was started in; the config's slot_save_path
    # to the config's "cwd" (the engine's folder) when it has one.  Either way the engine gets an absolute path.
    if a.slot_save_path or cfg.get("slot_save_path"):
        try:
            svc.slot_save_path = slot_save_dir(a.slot_save_path) if a.slot_save_path else \
                slot_save_dir(cfg["slot_save_path"], cfg.get("cwd"))
        except (ValueError, OSError) as e:
            raise SystemExit(f"[strata] {e}")
        print(f"[strata] slot save/restore on: {svc.slot_save_path} (session files are kept until deleted: "
              "about 1.2 GB per 63K-token conversation; restore only files this server wrote)", flush=True)
    mode = str(cfg.get("anthropic_thinking") or "on_request")   # #278: "model" thinks unasked too
    if mode not in ("model", "on_request"):
        raise SystemExit(f"[strata] config anthropic_thinking must be \"model\" or \"on_request\", not {mode!r}")
    svc.anthropic_think_unasked = mode == "model"
    svc.tool_call_recovery = tool_call_recovery_setting(cfg)
    etu = cfg.get("env_to_user", False)                # opt-in: see frontend._env_to_user
    if not isinstance(etu, bool):
        raise SystemExit(f"[strata] config \"env_to_user\" must be true or false, not {etu!r}")
    import serve.frontend as _frontend
    _frontend.ENV_TO_USER = etu
    rs = cfg.get("repeat_stop_tokens", REPEAT_STOP_TOKENS)   # #606: opt-out with 0
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise SystemExit(f"[strata] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {rs!r}")
    svc.repeat_stop_tokens = rs
    svc.loop_guard = loop_guard_setting(cfg)            # (STRATA_LOOP_GUARD=0 or "loop_guard": false: off)
    if not svc.loop_guard:
        print("[strata] loop guard: off (a request's \"loop_guard\": true turns it on for that request)", flush=True)
    svc.effort_end = bool(effort_end)                   # #458: "effort_position": "end" with an engine that has it
    if cfg.get("reasoning_budget_tokens") is not None:  # #123: a default thinking budget for every request
        try:
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"]
            budget = svc.reasoning_budget({})
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if budget:
            print(f"[strata] thinking budget: {budget} tokens (reasoning_budget_tokens; a request can set its own)",
                  flush=True)
    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover"):
        raise SystemExit("[strata] config \"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", "
                         f"not {recovery!r}")
    svc.reasoning_loop_recovery = recovery
    svc.gpu_pci = os.environ.get("STRATA_GPU_PCI") or cfg.get("gpu_pci")   # the Monitor reads the engine's card
    svc.split_pci = cfg.get("split_pci") or None       # and a layer split's other cards
    if a.engine == "strata" and split_serve:            # split serving: the decode card, and the prefill card beside it
        svc.gpu_pci, svc.split_pci = split_serve["decode_gpu"], [split_serve["prefill_gpu"]]
    strata = a.engine == "strata"
    svc.gpu_roles = gpu_roles(cfg, svc.gpu_pci, groups if strata else None, split_serve if strata else None,
                              getattr(svc.images, "gpu_pci", None), getattr(svc.images, "helper_card", ""))
    if a.config:
        svc.config_path = a.config                      # #564: the web page's Settings view
    try:                                                # the user's settings: the keys every model shares (runconfig)
        user_settings = runconfig.load_settings(svc.settings_path)
    except (OSError, ValueError) as e:
        print(f"[strata] WARNING: {svc.settings_path} cannot be read ({e}); started without it", flush=True)
        user_settings = {}
    try:                                                # the script guard: the model's config over the settings'
        svc.script_guard = script_guard.setting(
            {"script_guard": runconfig.effective(cfg, user_settings, "script_guard")}, os.environ)
    except ValueError as e:
        raise SystemExit(f"[strata] {e}") from None
    if isinstance(svc.script_guard, dict):
        print(f"[strata] script guard: {script_guard_name(svc.script_guard)} for every request that names none "
              "(a request's \"script_guard\": \"off\" turns it off for that request)", flush=True)
    try:                                                # the debug chat log (debuglog.py): model over settings
        on = a.debug_chat_log or runconfig.check(
            "debug_chat_log", runconfig.effective(cfg, user_settings, "debug_chat_log"), cfg)
        mib = runconfig.check("debug_chat_log_mib", runconfig.effective(cfg, user_settings, "debug_chat_log_mib"), cfg)
    except ValueError as e:
        raise SystemExit(f"[strata] {e}") from None
    if on:
        svc.debug_log = debuglog.DebugLog(places.user_dir("state") / "debug-chat-log", mib or 1024)
        print(f"[strata] debug chat log on: every chat request's prompt, tool rounds, answer and pictures, and the web "
              f"page's chats, are written to {svc.debug_log.folder} (at most {mib or 1024:g} MiB)" +
              ("" if svc.api_key else "; set an API key if others can reach this server"), flush=True)
    searcher = None
    try:                                                # web search: the engine and its address (search.py)
        searcher = runconfig.check("search_engine", runconfig.effective(cfg, user_settings, "search_engine"),
                                   cfg) or "searxng"
        svc.search_url = runconfig.check("search_url", runconfig.effective(cfg, user_settings, "search_url"),
                                         cfg) or None
    except ValueError as e:
        print(f"[strata] WARNING: web search is off: {e}", flush=True)
    deepseek_key = user_settings.get("deepseek_api_key")   # the settings' only (runconfig.GLOBAL_ONLY): never logged
    tinyfish_key = user_settings.get("tinyfish_api_key")
    if searcher == "deepseek" and isinstance(deepseek_key, str) and deepseek_key.strip():
        svc.search_backend = search_api.DeepSeek(deepseek_key.strip())
        print("[strata] web search: DeepSeek's search API (for requests that ask for it; billed per search)",
              flush=True)
    elif searcher == "deepseek":
        print(f"[strata] WARNING: web search is off: search_engine is deepseek, but {svc.settings_path} has no "
              "deepseek_api_key", flush=True)
    elif searcher == "tinyfish" and isinstance(tinyfish_key, str) and tinyfish_key.strip():
        svc.search_backend = search_api.TinyFish(tinyfish_key.strip())
        print("[strata] web search: TinyFish's search API (for requests that ask for it; free up to a daily "
              "allowance)", flush=True)
    elif searcher == "tinyfish":
        print(f"[strata] WARNING: web search is off: search_engine is tinyfish, but {svc.settings_path} has no "
              "tinyfish_api_key", flush=True)
    elif searcher == "searxng" and svc.search_url:
        svc.search_backend = search_api.Searxng(svc.search_url)
        print(f"[strata] web search: {svc.search_url} (for requests that ask for it)", flush=True)
    # A host that runs one model at a time and swaps them (a GPU shared with other work) can name the RPC that does
    # the swap: the web app then offers the models served on this port. {"mode": m} switches, {} only reports.
    svc.switcher = cfg.get("model_switcher") or None
    svc.port = a.port
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[strata] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
    if hub is not None:
        import atexit
        svc.mcp = hub
        if hub.settings["image_results"]:
            hub.media_dir = Path(a.config).with_suffix(".tool-media") if a.config else Path.cwd() / ".tool-media"
        print(f"[strata] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    httpd = serve(svc, host=a.host, port=a.port)
    svc.start_idle_unload()
    svc.start_vision_idle()
    # an IPv6 address goes in brackets in a URL (upstream eba3106c0, #3212)
    here = ("[::1]" if a.host == "::" else "127.0.0.1" if a.host in ("0.0.0.0", "")
            else f"[{a.host}]" if ":" in a.host else a.host)
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = [f"[{a.host}]"] if ":" in a.host and a.host != "::" else [] if a.host == "::" else lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{a.port}/   (API: http://{ip}:{a.port}/v1)", flush=True)
        if not ips:
            address = "[<this PC's IPv6 address>]" if a.host == "::" else "<this PC's IP address>"
            print(f"       from other devices: http://{address}:{a.port}/", flush=True)
        if not svc.api_key:
            print("       WARNING: " + open_server_warning(a.port, tools=svc.mcp is not None), flush=True)
    if release_check.enabled():                         # at the start and every six hours, said here once
        svc.updates.watch(lambda line: print("[strata] " + line, flush=True))
    if a.open and cfg.get("open_browser") is not False:   # #609: the config's "open_browser": false wins (an older
        import webbrowser                                  # run-<model>.sh still passes --open)
        webbrowser.open(f"http://{here}:{a.port}/")
    # #96: docker stop sends SIGTERM, which Python ignores by default, so the container's PID 1 would be killed after
    # the grace period with the engine still running. SIGTERM takes Ctrl+C's path below (QUIT to the engine).
    # SIGHUP (the terminal closed) ended Python without that path, so it is taken as well (not on Windows).
    # SIGINT keeps Python's own handler, so Ctrl+C and a second Ctrl+C work as before.

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        try:
            if sig is not None:
                signal.signal(sig, on_sigterm)
        except (ValueError, OSError, AttributeError):     # not the main thread
            pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[strata] stopping (Ctrl+C again to end the engine at once) ...", flush=True)
        closers = [httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None,
                   hub.close if hub is not None else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # a second Ctrl+C: don't wait for the engine to free its memory
                proc = getattr(engine, "proc", None)
                if proc:
                    proc.kill()
        try:
            from serve import cache_drop
            cache_drop.drop_after_exit(engine)           # opt-in (STRATA_DROP_CACHE_ON_EXIT=1), default unchanged
        except Exception as e:  # noqa: BLE001 - never block the exit
            print(f"[strata] could not release the model files' cache: {e}", flush=True)
        print("[strata] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
