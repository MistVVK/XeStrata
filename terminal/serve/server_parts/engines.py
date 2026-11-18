"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the engines part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import json
import os
import platform
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Iterator, Protocol
from strata_control import engine_component
from strata_control import reader_room
from strata_control import gpu_roles

from serve.server_parts.base import ENGINE_SILENCE_S


class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""
    restarts = True        # the engine is gone, so the next request starts it again; False for an error raised while the engine is left running (#3961)


class EngineClosing(EngineDied):
    """#6055: the server stopped (or unloaded) the engine under a request it had already accepted, and the engine ended
    that request on the close's STOP before it finished.  The request failed; it did not end by itself: never a
    successful reply with no tokens and finish "length".  Nothing restarts the engine for it."""
    restarts = False


def died_note(e: Exception) -> str:
    """What an API error says about an EngineDied: the cause, and that the next request restarts the engine - only where it really is ended."""
    return f"{e}; the next request restarts it" if getattr(e, "restarts", True) else str(e)


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet (#344): no context size to plan a request with - a 503,
    not a 400 about the prompt."""


class ClientGone(Exception):
    """#1999: the client hung up while its request was being prepared (an image URL downloading, a picture encoding or
    waiting for its turn): nothing is left to answer."""


class ServerStopping(RuntimeError):
    """The drain deadline (--shutdown-timeout) cancelled a request whose client may still be waiting for its answer."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


class QueueWaitExpired(RuntimeError):
    """An inference call waited too long for its first engine turn."""


class QueueCancelled(RuntimeError):
    """The client left before its first serial engine turn."""


class EngineSilent(EngineDied):
    """#481: the engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


class EngineNonFinite(ValueError):
    """#879: the engine's DONE (or a slot's BDONE) said `nonfinite`: a window's logits held a NaN or an infinity.  The
    engine wrote none of that window's tokens and drops every cached state that may carry the poison, so its next
    request reads the prompt from token 0.  Service.run sends a request that had produced nothing yet once more, and
    ends one that had started (as "length").  A ValueError, so a stream that has started reports it as an error event."""


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


HOOK_REASON_MAX = 300   # #5987: the most of a before_load command's own refusal line that goes into a 503


def hook_text(output) -> str:
    """A before_load command's output as text: UTF-8, else the system's own encoding (a Windows console tool writes its
    code page), never an error."""
    if not isinstance(output, bytes):
        return output or ""
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError:
        import locale
        return output.decode(locale.getpreferredencoding(False) or "utf-8", "replace")


def hook_reason(output, limit: int = HOOK_REASON_MAX, stderr=None) -> str:
    """#5987: the last non-empty line a before_load command wrote, made safe for an API message: control and format
    characters become spaces, runs of whitespace one space, at most `limit` characters ("..." marks a cut).  The two
    streams are read apart: the reason is stderr's last line, else stdout's - merged, their order depends on the
    command's buffering (a Python launcher's stdout to a pipe is flushed at its exit, after its stderr line).  "" when it
    wrote nothing."""
    for text in (stderr, output):
        for line in reversed(hook_text(text).splitlines()):
            s = " ".join("".join(" " if unicodedata.category(c)[0] == "C" else c for c in line).split())
            if s:
                return s if len(s) <= limit else s[:limit - 3].rstrip() + "..."
    return ""


def hook_echo(output, stream=None) -> None:
    """The before_load command's output, passed on to this server's own stdout (or `stream`: its stderr for the command's
    stderr), where it went before #5987 read it."""
    if not output:
        return
    lines = hook_text(output).splitlines()   # (a Windows command's \r\n would come out doubled through a text stream)
    if lines:
        print("\n".join(lines), file=stream or sys.stdout, flush=True)


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left.  `retry_after`: seconds
    for the Retry-After header of the 503."""

    def __init__(self, message="", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# The page's only CSP directive: its preview frames (srcdoc) may not load any address. Other directives are left out
# on purpose: a frame made from srcdoc inherits the page's policy, and the opt-in "run scripts" preview needs its own inline script.
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


ENGINE_REQUEST = re.compile(
    # "+ 12288 of 98179 read": a request cancelled while its prompt was read (#471)
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")


_echoing: set[str] = set()      # the logs echo_requests already follows (restart() runs StrataEngine.__init__ again)


DRAFT_HEAD_FAIL = "the draft head does not fit"


DRAFT_HEAD_HINT = ("a smaller draft vocabulary needs less VRAM: --draft-vocab cyrillic (English, code and the Cyrillic "
                   "script) or --draft-vocab en (English and code, ~215 MiB less than the default). Start once with "
                   "it - START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                   "keeps it; or a smaller --context in setup.")


def start_failure_hint(log: str | None, offset: int) -> str:
    """#474: what to change when the engine stopped at the start because the MTP draft head did not fit the VRAM
    left: the engine's own `strata mtp:` lines after that failure (0.1.36+: what it needs, what is free, the smaller
    subsets), else the same advice in words for an older engine.  "" for any other failure: the log says why."""
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


def start_log_text(log: str | None, offset: int, limit: int = 1 << 20) -> str:
    """Everything this engine start has written to the log so far (at most `limit` bytes), "" when there is no log."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, offset))
            return f.read(limit).decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""


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


# Windows NTSTATUS exit codes of a process the system ended (a crash, not an error the engine reported)
_WIN_CRASH = {0xC000001D: "illegal instruction (the CPU lacks an instruction the program uses)",
              0xC0000005: "access violation", 0xC0000409: "stack buffer overrun / fast-fail",
              0xC0000135: "a DLL the program needs was not found", 0xC0000139: "entry point not found in a DLL",
              0xC0000142: "a DLL failed to initialize", 0xC000007B: "bad image (32-bit / 64-bit mix-up)",
              0xC00000FD: "stack overflow", 0xC0000094: "integer division by zero", 0xC0000096: "privileged instruction"}


def engine_exit_words(rc: int | None) -> str:
    """What a process's exit status says, in words: "killed by signal 4 (SIGILL, Illegal instruction)" (a negative status
    on Linux / macOS), a Windows NTSTATUS crash code, or "exit code N".  "" when there is no status."""
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


def start_exit_note(rc: int | None, exe=None) -> str:
    """The sentence the start error carries about how the engine process ended (the engine log is silent when the
    system ends it): "" without a status, else a line, with the CPU's AVX flags after an illegal instruction."""
    words = engine_exit_words(rc)
    if not words:
        return ""
    note = f"\nthe engine process ended: {words}"
    if sys.platform.startswith("linux"):
        try:                                        # systemd-oomd ended it (serve/oomd_pacer.py): say so in plain words
            from serve.oomd_pacer import killed_words
            note += killed_words(rc)
        except Exception:                           # noqa: BLE001 - a nicety
            pass
    try:                                            # 0xC0000135: say which DLL (the loader prints nothing)
        from serve.loader_diag import missing_note
        note += missing_note(rc, exe)
    except Exception:
        pass
    if rc is not None and ((rc < 0 and os.name != "nt" and -rc == int(getattr(signal, "SIGILL", 4))) or
                           (rc & 0xFFFFFFFF) == 0xC000001D):
        isa, missing = cpu_isa_words()
        note += "\n  An illegal instruction: the engine ran an instruction this CPU cannot execute"
        if isa and missing:
            note += f" ({', '.join(m.upper() for m in missing)} missing; {isa})"
        elif isa:
            note += f"; the CPU reports AVX, AVX2, FMA and F16C ({isa}), so this is a bug in the engine"
        note += ".  Please report it with the CPU model (github.com/Niko1221/Strata/issues)."
    return note


def start_log_marks(log: str | None, offset: int) -> list:
    """The GPU runtime failure texts (TRANSIENT_START_MARKS) this start wrote to the log: what lets a retry notice say the
    GPU runtime failed.  [] when none, or no log."""
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


_BTRACE = bool(os.environ.get("STRATA_BATCH_TRACE"))


def btrace(*a):
    if _BTRACE:
        print("[batch-trace]", threading.get_ident() % 10000, *a, file=sys.stderr, flush=True)


# The engine's own words for a verifier that cannot go on (its GPU waits were released, or an earlier window never
# finished, #267; a GPU wait on the host ran out and the window was discarded, #5160 - the Intel engine ends after it):
# the process may stay alive and even print DONE, but it must not take another request.
FATAL_PREFIXES = ("ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                  "ERR verify: an earlier window never finished", "ERR verify batch: an earlier window never finished",
                  "ERR verify: the GPU stopped waiting")


def parse_engine_stage_line(line: str) -> list[dict] | None:
    """Accept only the engine's final, contiguous stage plan (inclusive layer ends)."""
    prefix = "strata serve: layer split: layers "
    suffix = ", one hand-off per window"
    if not line.startswith(prefix) or not line.rstrip("\r\n").endswith(suffix):
        return None
    body = line.rstrip("\r\n")[len(prefix):-len(suffix)]
    parts = body.split(", ")
    if not 2 <= len(parts) <= 16:
        return None
    stages, next_layer = [], 0
    for i, part in enumerate(parts):
        match = re.fullmatch(r"([0-9]{1,5})-([0-9]{1,5}) \(CUDA([0-9]{1,2})\)", part)
        if match is None:
            return None
        first, last, ordinal = map(int, match.groups())
        if first != next_layer or last < first or last > 65535 or ordinal > 15:
            return None
        stages.append({"stage": i, "layer_start": first, "layer_end": last,
                       "engine_ordinal": ordinal})
        next_layer = last + 1
    return stages


def read_engine_stage_log(path: str | None, start: int, limit: int = 2 << 20) -> list[dict]:
    """Read this engine start only; oversized/unavailable logs leave placement unknown."""
    if not path or start < 0:
        return []
    try:
        with open(path, "rb") as source:
            source.seek(start)
            data = source.read(limit + 1)
    except OSError:
        return []
    if len(data) > limit:
        return []
    for line in reversed(data.decode("utf-8", errors="replace").splitlines()):
        stages = parse_engine_stage_line(line)
        if stages is not None:
            return stages
    return []


def visible_device_order(env: dict | None, backend: str | None) -> list[int]:
    """Map engine ordinals to OS GPU indices only when the visibility list is unambiguous."""
    if backend not in (None, "cuda", "hip"):
        return []
    if backend != "hip" and (env or {}).get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        return []
    name = "HIP_VISIBLE_DEVICES" if backend == "hip" else "CUDA_VISIBLE_DEVICES"
    value = (env or {}).get(name)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,5}(?:,[0-9]{1,5}){0,15}", value):
        return []
    devices = [int(part) for part in value.split(",")]
    return devices if all(0 <= device <= 65535 for device in devices) and len(set(devices)) == len(devices) else []


def monitored_gpu(cards: list[int], env: dict | None, backend: str | None) -> int:
    """The card the Monitor reads, numbered as the GPU readers number them (serve/telemetry.py: AMD in KFD order, NVIDIA by PCI bus).
    The config's own cards when it names any. Otherwise the card the engine's environment shows it first: an "Automatic" AMD start
    (hip_auto_devices) or the person's own HIP_VISIBLE_DEVICES runs the engine on card N while the Monitor used to read card 0, an idle
    card next to it (#4640). 0 when the environment does not say, or says it ambiguously (visible_device_order)."""
    if cards:
        return cards[0]
    order = visible_device_order(env, backend)
    return order[0] if order else 0


def parse_batch_window_sample(line: str) -> dict | None:
    """Read an opt-in engine BSTAT sample; rows are verifier work, not emitted tokens."""
    if len(line) > 80:
        return None
    match = re.fullmatch(r"BSTAT ([1-9][0-9]{0,8}) ([1-9][0-9]{0,9}) "
                         r"([1-9][0-9]{0,9}) ([1-9][0-9]{0,9})\r?\n?", line)
    if match is None:
        return None
    windows, rows, active, elapsed_ms = map(int, match.groups())
    if rows > windows * 8 or active > rows or elapsed_ms > 86_400_000:
        return None
    return {"windows": windows, "processed_rows": rows, "active_rows": active,
            "elapsed_ms": elapsed_ms}


def parse_stage_in_flight_sample(line: str) -> dict | None:
    """Read bounded PSTAT host scheduling spans; these are not GPU utilization."""
    if len(line) > 256 or re.fullmatch(
            r"PSTAT (?:[2-9]|1[0-6]) [1-9][0-9]{0,10}(?: (?:0|[1-9][0-9]{0,10})){2,16}\r?\n?", line) is None:
        return None
    parts = line.strip().split(" ")
    stages, elapsed_us = int(parts[1]), int(parts[2])
    in_flight_us = [int(value) for value in parts[3:]]
    if len(in_flight_us) != stages or elapsed_us > 86_400_000_000 or \
            any(value > elapsed_us for value in in_flight_us):
        return None
    return {"elapsed_us": elapsed_us, "in_flight_us": in_flight_us}


class _RequestOwned:
    """#4451: a per-request control field of a --batch engine.  With batch slots several requests run on their own
    threads, and the control lines serve one at a time; a field every request writes (`_ctl_mode`, `_ctl_result`,
    `_yielded`, `_last_done`, `_ctl_erred`) must not be seen by another request that gets the lines next.  So each
    thread has its own copy; one request at a time, it is the one attribute as before."""

    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def _store(self, obj):
        d = obj.__dict__
        if d.get("batch"):
            return d.setdefault("_own_tl", threading.local()).__dict__
        return d.setdefault("_own_plain", {})

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return self._store(obj).get(self.name, self.default)

    def __set__(self, obj, value):
        self._store(obj)[self.name] = value


def network_path(path: str) -> bool:
    r"""A path to another computer: \\host\share or //host/share (either slash), \\?\UNC\host\..., \\.\UNC\host\...
    or the NT form \??\UNC\host\...  Windows connects to that host at the first look at the path (os.path.isfile is
    enough) and signs in with the user's NTLM credentials.  \\?\ and \??\ name local files too; no client needs them."""
    p = path.replace("/", "\\")
    return p.startswith("\\\\") or p.startswith("\\??\\")


def vision_log_size(log=None) -> int:
    """How long the encoder's stderr file is now (0 when there is none): a start reads only what it wrote after this."""
    try:
        name = getattr(log, "name", None)
        return os.path.getsize(name) if name else 0
    except (OSError, TypeError):
        return 0


def vision_log_tail(log=None, limit: int = 16384, since: int = 0) -> str:
    """The last `limit` bytes of the encoder's stderr file (`log` is the file object it was started with; "" when there
    is none or it cannot be read), but nothing from before `since` (the file's size when this start began: a log that several starts share)."""
    try:
        name = getattr(log, "name", None)
        if name:
            with open(name, "rb") as f:
                f.seek(0, 2)
                f.seek(max(since, f.tell() - limit))
                return f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        pass
    return ""


# What the CUDA / HIP runtimes print when the reader has no code for the card it runs on (a binary built without that
# architecture: #3669, the Windows cuda12 package's reader on a Pascal card).
_VISION_ARCH_SIGNS = ("device kernel image is invalid", "no kernel image is available", "im2col failed",
                      "unsupported gpu architecture", "invalid device function", "unknown gpu architecture")


def vision_start_error(line: str, log=None, code: int | None = None, waited: float | None = None, strays=(), since: int = 0) -> str:
    """The message when the image encoder answered something other than READY.  #1445: a card in the Exclusive_Process
    compute mode (or one another process holds) refuses the encoder's CUDA context with "busy or unavailable"; that
    cause is named instead of leaving the user the first line of a traceback.  `log` is the encoder's stderr file, whose
    tail is searched too (the CUDA error is printed there).  An encoder that dies before it prints anything (#3645, #3444,
    #3669, #3697) used to give a blank sentence: now the last lines of its stderr, its exit status (`code`) and, for a
    kernel-image error, the card/architecture hint come with it.  `waited`: it said nothing for that many seconds.  `strays`: lines it printed on
    its output before it ended that were not an answer (a debugger's, a library's); they are searched for the causes below and named in the message."""
    text = line.strip()
    tail = vision_log_tail(log, since=since)
    both = (text + chr(10) + chr(10).join(strays) + chr(10) + tail).lower()
    if "busy or unavailable" in both:
        return ("the vision encoder did not start: the GPU refused a second CUDA context (\"CUDA-capable device(s) "
                "is/are busy or unavailable\"). This is usually the compute mode Exclusive_Process: only one process "
                "may use the card, and the engine already holds it. Check with: nvidia-smi --query-gpu=compute_mode "
                "--format=csv; set it back with: sudo nvidia-smi -c DEFAULT (or run the encoder on another card). "
                "First line from the encoder: " + (text or "(none)"))
    lines = [x.strip() for x in tail.splitlines() if x.strip()]
    room = reader_room.find((text + chr(10) + chr(10).join(strays) + chr(10) + tail).splitlines())
    if room:                                           # a card another program filled: ggml's cudaMalloc line, then an abort the reader cannot answer for
        asked = (f" (it asked for {room.need_mib:.0f} MiB" + (f" on its card {room.device}" if room.device is not None else "") + ")") if room.need_mib else ""
        said = [text, *strays, *lines]
        evidence = (next((x for x in reversed(said) if "malloc failed" in x.lower()), None)
                    or next((x for x in reversed(said) if "failed to allocate" in x.lower()), text))
        return "the vision encoder did not start: the graphics card has no room for it" + asked + ". From the encoder: " + evidence.strip()[:300]
    if any(sign in both for sign in _VISION_ARCH_SIGNS):
        evidence = next((x for x in reversed(lines) if any(sign in x.lower() for sign in _VISION_ARCH_SIGNS)), text)
        return ("the vision encoder did not start: it has no code for this graphics card (a kernel-image error: the "
                "picture reader in this engine package was built for other card generations than yours). Update the "
                "engine, or read pictures on the CPU (Models, Read pictures, Image encoder device: CPU; or run setup "
                "again with --vision cpu). From the encoder: " + evidence[:300])
    if text:
        msg = "the vision encoder did not start: " + text
    elif waited is not None:
        msg = f"the vision encoder did not start: it said nothing for {waited:g} s (it never reported READY)"
    else:
        msg = "the vision encoder did not start: it ended without saying anything"
    extra = []
    if code is not None:
        words = engine_exit_words(code)
        if words:
            extra.append(words)
    if strays:
        extra.append("it also printed on its output: " + " | ".join(x[:120] for x in list(strays)[-3:]))
    shown = [x for x in lines[-3:] if x != text]
    if shown:
        extra.append("last lines of its log: " + " | ".join(x[:200] for x in shown))
    if extra:
        msg += " (" + "; ".join(extra) + ")"
    return msg


class VisionNoRoom(RuntimeError):
    """The picture reader could not start because its card has no room for it (`room`: strata_control.reader_room.NoRoom)."""

    def __init__(self, message: str, room):
        super().__init__(message)
        self.room = room


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    roles, _ = gpu_roles.saved_roles(cfg)       # "gpu_roles" (Main / Helper / Off per card) wins unless "gpu" was edited by hand
    if roles is not None:
        legacy = _gpu_numbers(cfg.get("gpu"))
        return legacy if legacy else gpu_roles.engine_order(roles)     # Main LAST; Off cards left out; [] when every card is Off
    return _gpu_numbers(cfg.get("gpu"))


def _gpu_numbers(g) -> list[int]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def gpu_roles_active(cfg: dict) -> bool:
    """True when the run settings carry usable "gpu_roles": the card order is the person's (Main last), never re-sorted by speed."""
    return gpu_roles.saved_roles(cfg)[0] is not None


def gpu_all_off(cfg: dict) -> bool:
    """Every card is set to Off ("gpu_roles" with no card on, or "gpu_none": true)."""
    roles, _ = gpu_roles.saved_roles(cfg)
    return roles is not None and not any(r != "off" for r in roles.values())


def card_roles_lines(cfg: dict) -> list[str]:
    """The plain lines printed at start: the roles (cards numbered as nvidia-smi numbers them), a hand-edited "gpu" that beat them, and the
    all-Off case.  SystemExit when every card is Off and the engine cannot start without one (the CUDA, HIP and SYCL engines)."""
    roles, problem = gpu_roles.saved_roles(cfg)
    out = [f"[strata] {problem}"] if problem else []
    if roles is None:
        return out
    if gpu_all_off(cfg):
        if cfg.get("backend") in ("cpu", "metal") or sys.platform == "darwin":
            out.append("[strata] every graphics card is Off: " + gpu_roles.CPU_ONLY_LINE)
            return out
        raise SystemExit("[strata] every graphics card is set to Off, and this engine cannot start without one. Turn a card on "
                         "(Main or Helper) in the app or with strata launch --gpu-roles.")
    word = {"main": "Main", "helper": "Helper", "off": "Off"}
    out.append("[strata] card roles: " + ", ".join(f"{word[r]} GPU {i}" for r in ("main", "helper", "off")
                                                    for i in sorted(roles) if roles[i] == r) + " (numbers as nvidia-smi gives them)")
    return out


def engine_binary(exe: str, cfg: dict | None = None) -> str:
    """The file that holds the engine's own option strings, for a feature check that reads it.  The Intel launcher
    (sycl/serve/strata-sycl.sh) is a shell script that runs its repo's STRATA_SYCL_BIN (default build-sycl-aot/strata)
    in a container, so the check reads that binary, not the script.  Any other `exe` is the engine itself."""
    p = Path(exe)
    if p.name == "strata-sycl.sh":
        rel = ((cfg or {}).get("env") or {}).get("STRATA_SYCL_BIN") or os.environ.get("STRATA_SYCL_BIN") \
            or "build-sycl-aot/strata"
        b = p.resolve().parents[2] / rel
        if b.is_file():
            return str(b)
    return exe


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
    """#644: the config's "layer_split" as the engine's --layer-split value: "auto" (the default), or the FIRST LAYER of
    each later GPU's share - one rising number per GPU after the first, e.g. "24,36,42" for 4 GPUs (layers 0-23 on the
    first card, 24-35, 36-41, 42 to the end) - as a string or a JSON list ([24, 36, 42]).  ValueError with the format
    and an example for anything else, such as per-card layer counts ("24,16,12,12")."""
    v = cfg.get("layer_split")
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto")):
        return "auto"
    n = len(gpu_list(cfg))
    items = v if isinstance(v, (list, tuple)) else str(v).split(",") if isinstance(v, (str, int)) else None
    vals = []
    for x in items or []:
        try:
            if isinstance(x, bool) or isinstance(x, float):
                raise ValueError
            vals.append(int(str(x).strip()))
        except ValueError:
            vals = None
            break
    want = max(n - 1, 1)
    example = ",".join(str(round(48 * (i + 1) / (want + 1))) for i in range(want))
    hint = (f'"layer_split" is the first layer of each later GPU, one rising number per GPU after the first '
            f'({want} for {n} GPUs), not a count of layers per card - e.g. "{example}" (or [{example.replace(",", ", ")}])'
            f' for an even share of 48 layers, or "auto" (the default) to place them by each card\'s free VRAM')
    if not vals:
        raise ValueError(f"{hint}; got {v!r}")
    if n > 1 and len(vals) != n - 1:
        raise ValueError(f"{hint}; got {len(vals)} number(s) ({v!r}) for {n} GPUs")
    if vals[0] < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"{hint}; got {v!r}, which does not rise from 2 or more")
    return ",".join(str(x) for x in vals)


def layer_split_of(cfg: dict) -> bool:
    """Several GPUs in the config are a layer split, unless the args put the later card(s) to another use: with
    --peer-device the second card is an expert-cache tier, and the engine refuses that beside --layer-split (#665)."""
    return len(gpu_list(cfg)) > 1 and "--peer-device" not in cfg["args"]


def own_path_context_warning(cfg: dict, args: list, vram_mib: float | None) -> str | None:
    """The sentence for a Qwen3.6 start whose --max-context does not fit its one card (strata_control/own_path_fit.py, the rule setup's menu
    uses); None for any other model, a context that fits, or a card that is not known.  Nothing about the start changes."""
    try:
        from strata_control import own_path_fit
        fit = own_path_fit.from_model_name(cfg.get("model_name"))
        if not fit or not vram_mib or "--max-context" not in args:
            return None
        ctx = int(args[args.index("--max-context") + 1])
        return own_path_fit.warning(fit, float(vram_mib) / 1024, ctx, own_path_fit.MENU)
    except Exception:                                        # noqa: BLE001 - an estimate that fails says nothing; the engine still says what does not fit
        return None


PARALLEL_MAX = 8      # without a layer split the engine's batch window holds at most 8 rows (kVerifyNarrowT); --batch-mtp waves more through it


SPLIT_MAX_LANES = 8   # the all-cards split engine runs 1..8 lanes (strata_control/split_engine.py MAX_LANES)


def parallel_flag_conflicts(cfg: dict) -> bool:
    """`--parallel N` next to the saved engine args' own --batch/--slots is refused, except for the all-cards split engine's run config (split_config's
    "split_engine_note"): its --batch is its lane count, and "parallel" sets it (parallel_args). The app's Concurrent requests reaches the server this way."""
    args = cfg.get("args", [])
    return ("--batch" in args or "--slots" in args) and not cfg.get("split_engine_note")


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    """#465 (opt-in): "parallel": N in the config - up to N requests decode together in the engine's batch slots
    (--batch N); more wait their turn.  1 or absent: one request at a time, as before.  A value the engine cannot use
    is said and passed on: the engine warns and adjusts (recommend, never force)."""
    if cfg.get("split_engine_note") and "--batch" in args:
        # the all-cards split engine: its run config carries --batch N (its lanes); "parallel" (the Terminal's --parallel, the app's Concurrent requests) sets N
        n = cfg.get("parallel")
        if isinstance(n, int) and not isinstance(n, bool) and n >= 1:
            args[args.index("--batch") + 1] = str(min(n, SPLIT_MAX_LANES))
        return []
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
    # (on a layer split the pipelined groups each hold up to 8 slots - 16 in a HIP engine with wide windows - so more than 8 is
    # not a limit there: the engine says how many slots it runs, INFO batch_slots)
    if (n > PARALLEL_MAX and "--batch-mtp" not in args and os.environ.get("STRATA_BATCH_MTP", "0") in ("", "0")
            and not layer_split_of(cfg)):
        print(f'[strata] "parallel": {n} - the engine runs at most {PARALLEL_MAX} requests together; it will use '
              f"{PARALLEL_MAX}", flush=True)
    return ["--batch", str(n)]


_POOL_CORES_OK = re.compile(r"^[0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+(:[0-9]+)?)?)*$")


def pool_cores_args(cfg: dict, args: list[str]) -> list[str]:
    """Experimental, off by default: "pool_cores": "0,2,4,6" (or "2-14:2", or a list of numbers) in the config pins the CPU
    expert workers to exactly those logical processors (--pool-cores; docs/DETAILS.md, "Which logical processors the CPU
    workers use").  The engine warns about two entries on one core or the host's own and leaves out processors the process may
    not use; a text this cannot read is said here and ignored (recommend, never force).  An engine option already in the args wins."""
    value = cfg.get("pool_cores")
    if value is None or value is False or value == "" or value == []:
        return []
    if "--pool-cores" in args:
        return []
    if isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value):
        text = ",".join(str(v) for v in value)
    elif isinstance(value, str):
        text = value.replace(" ", "")
    else:
        text = ""
    if not _POOL_CORES_OK.match(text):
        print(f'[strata] "pool_cores" must be logical processor numbers like "0,2,4,6" or "2-14:2", not {value!r}: ignored '
              "(the CPU workers keep their default placement)", flush=True)
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
    """#477 (opt-in): "expert_profile_save": "<path>" in the config has the engine (0.1.36+) save what its adaptive
    tier learned there - on QUIT and every "expert_profile_save_every" minutes (10 by default, 0 = at QUIT only) -
    and the next start begins from it instead of the config's --expert-profile, when it is a profile of the same
    model (its header's layers and experts match); otherwise from the config's own, as before.  A relative path is
    the engine's (the config's "cwd").  Without the key, the arguments are the config's, unchanged."""
    save = cfg.get("expert_profile_save")
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


DRAFT_COSTS_DEFAULT_PLATFORMS = ("darwin",)   # Metal: the taken-over table measured above live costs there (M5 Max: Q2_0 code +2.6 %, chat +2.4 %, JSON +0.2 %; IQ3_S +0.2 / +0.6 / +0.7 %)


def draft_costs_default(platform: str | None = None) -> bool:
    """Repeatable output (owner decision) is the default only where the table was measured at or above live costs: Metal.  On CUDA (RTX PRO 5000, IQ3_S: code
    -1.5 %, chat -1.5 %, JSON +1.9 % against live costs) and HIP the engine keeps live costs, as an engine started without --draft-costs does.  `"draft_costs": true`
    in a config turns the table on anywhere (tests and A/Bs), `"draft_costs": false` turns it off."""
    return (platform or sys.platform) in DRAFT_COSTS_DEFAULT_PLATFORMS


_engine_texts: dict = {}


def engine_has_text(exe: str, needle: bytes) -> bool | None:
    """Does the engine program contain `needle` (an option it knows)?  None when it cannot be read.  Read once per path, size and time."""
    import mmap
    try:
        st = os.stat(exe)
    except (OSError, ValueError):
        return None
    key = (str(exe), st.st_size, st.st_mtime_ns, needle)
    if key not in _engine_texts:
        try:
            with open(exe, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _engine_texts[key] = data.find(needle) >= 0
        except (OSError, ValueError):
            _engine_texts[key] = None
    return _engine_texts[key]


def draft_costs_dir() -> Path:
    """Where the learned draft cost tables live: beside the saved tunings (strata_control/presets.py), in the user's Strata settings folder
    (%APPDATA%\\Strata\\draft-costs, ~/.config/strata/draft-costs), which the Windows uninstaller's "remove my data" deletes with them.
    STRATA_DRAFT_COSTS_DIR: another folder (tests and test jobs, which never write into a real settings folder)."""
    explicit = os.environ.get("STRATA_DRAFT_COSTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from strata_control import profiles
    return profiles.profile_dir().parent / "draft-costs"


def gpu_speed_scores(indices: list[int]) -> dict[int, float] | None:
    """Multiprocessors x max clock for each NVIDIA card, numbered as nvidia-smi numbers them (the CUDA driver API,
    which every driver ships: no toolkit needed); None when it cannot say for every one of them."""
    code = (
        "import ctypes,sys,os\n"
        "os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'\n"
        "lib=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')\n"
        "assert lib.cuInit(0)==0\n"
        "n=ctypes.c_int()\n"
        "assert lib.cuDeviceGetCount(ctypes.byref(n))==0\n"
        "for i in range(n.value):\n"
        "    d=ctypes.c_int()\n"
        "    assert lib.cuDeviceGet(ctypes.byref(d),i)==0\n"
        "    sm=ctypes.c_int(); mhz=ctypes.c_int()\n"
        "    lib.cuDeviceGetAttribute(ctypes.byref(sm),16,d)\n"
        "    lib.cuDeviceGetAttribute(ctypes.byref(mhz),13,d)\n"
        "    print(i,sm.value,mhz.value)\n")
    try:
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
        got = {}
        for line in r.stdout.splitlines():
            i, sm, khz = (int(x) for x in line.split())
            got[i] = float(sm) * float(khz)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return got if all(i in got and got[i] > 0 for i in indices) else None


def hip_speed_scores(indices: list[int], root: str = "/sys/class/kfd/kfd/topology/nodes") -> dict[int, float] | None:
    """AMD on Linux: SIMDs x max engine clock of each card, from the KFD topology (the order setup and the engine use
    for "gpu" there: the GPU nodes in node order).  None when the files are not there (Windows, no ROCm driver)."""
    try:
        nodes = sorted((int(n) for n in os.listdir(root) if n.isdigit()))
    except OSError:
        return None
    cards = []
    for n in nodes:
        try:
            with open(os.path.join(root, str(n), "properties")) as fh:
                props = dict(line.split() for line in fh if len(line.split()) == 2)
            simd, clk = int(props.get("simd_count", 0)), int(props.get("max_engine_clk_fcompute", 0))
        except (OSError, ValueError):
            continue
        if simd > 0:
            cards.append(float(simd) * float(clk or 1))
    got = dict(enumerate(cards))
    return got if all(i in got for i in indices) else None


def igpu_cards(indices: list[int], archs: dict[int, str] | None) -> set[int]:
    """The cards of `indices` that are integrated (their memory is the system RAM, and they usually draw the desktop)."""
    return {i for i in indices if archs and archs.get(i) in HIP_IGPU_ARCHS}


#: AMD chips that share the system memory (an iGPU): never preferred over a card that has memory of its own.
HIP_IGPU_ARCHS = ("gfx1035", "gfx1036", "gfx1037", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx90c")


def parse_hip_devices(text: str) -> list[dict]:
    """`strata-device --list-devices`: [{"index", "name", "arch", "vram_gb", "integrated", "cannot_run"}], numbered as the HIP runtime numbers them."""
    found: list[dict] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"device\s+(\d+):\s*(.*)$", s)
        if m:
            found.append({"index": int(m.group(1)), "name": m.group(2).strip(), "arch": "", "vram_gb": 0.0, "integrated": False, "cannot_run": None})
            continue
        if not found:
            continue
        a = re.match(r"arch\s+(gfx[0-9a-f]+)\s*,\s*([\d.]+)\s*GiB(.*)$", s)
        if a and not found[-1]["arch"]:
            found[-1]["arch"], found[-1]["vram_gb"] = a.group(1), float(a.group(2))
            found[-1]["integrated"] = "integrated" in a.group(3)      # a newer engine says so itself
        elif s.startswith("cannot run:"):
            found[-1]["cannot_run"] = s[len("cannot run:"):].strip()
    for d in found:
        d["integrated"] = d["integrated"] or d["arch"] in HIP_IGPU_ARCHS
    return found


def hip_no_card_message(devices: list[dict]) -> str:
    """The plain line when the engine has code for none of the computer's cards."""
    cards = ", ".join(f"{d['name'] or 'graphics card'} ({d['arch'] or 'unknown'}{', integrated' if d['integrated'] else ''})" for d in devices)
    built = next((m.group(1) for d in devices for m in [re.search(r"compiled for \(([^)]*)\)", d["cannot_run"] or "")] if m), "")
    text = f"[strata] none of this computer's graphics cards has code in this Strata engine{f' (it is built for {built})' if built else ''}. Cards found: {cards}."
    note = engine_component.no_package_note([d["arch"] for d in devices])
    return text + (f" {note}" if note else "")


def pick_hip_devices(devices: list[dict]) -> tuple[list[int] | None, str | None]:
    """Which card an "Automatic" start runs on: (HIP_VISIBLE_DEVICES to set, or None to leave the runtime's own order; a plain problem or
    None).  Cards this engine has no code for are skipped, and so are integrated chips while a card of its own can run (#3203: a mini PC's
    Radeon 760M is HIP's device 0 and the engine died on it; the RX 6800 XT behind it was never tried).  When device 0 is already the right
    kind nothing changes."""
    if not devices:
        return None, None
    runnable = [d for d in devices if not d["cannot_run"]]
    if not runnable:
        return None, hip_no_card_message(devices)
    pool = [d for d in runnable if not d["integrated"]] or runnable
    if devices[0] in pool:
        # device 0 runs, as before; a card the engine has no code for is not shown to it at all (#6015: such a card visible to the engine
        # can stop it at start), and the order of the others is kept
        return ([d["index"] for d in runnable] if len(runnable) < len(devices) else None), None
    best = max(pool, key=lambda d: (round(d["vram_gb"]), -d["index"]))
    return [best["index"]], None


def parallel_slots(cfg: dict) -> int:
    """How many batch slots the config asks for (0 = none): the args' own --batch/--slots, else "parallel"."""
    args = list(cfg.get("args") or [])
    for k in ("--batch", "--slots"):
        if k in args:
            try:
                return int(args[args.index(k) + 1])
            except (IndexError, ValueError):
                return 0
    n = cfg.get("parallel")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 1 else 0


def _dir_writable(p: str) -> bool:
    try:
        with tempfile.TemporaryFile(dir=p):
            return True
    except OSError:
        return False


def sh_stream_warning(env: dict, cfg: dict) -> str | None:
    """#3895: the shared-expert fork (STRATA_SH_STREAM=1) together with KV streaming (--kv-resident) made one Windows RX 6900 XT fail on the first
    request ("verify: unspecified launch failure"); each alone, or Linux, did not. A warning, not a refusal: the person set the variable, the
    engine runs as asked, and the line says what to take out."""
    if os.name != "nt" or str(env.get("STRATA_SH_STREAM", "")).strip() in ("", "0"):
        return None
    args = [str(a) for a in cfg.get("args") or []]
    if "--kv-resident" not in args:
        return None
    return ("[strata] WARNING: STRATA_SH_STREAM=1 with --kv-resident (KV streaming) failed on the first request on one Windows RX 6000 PC "
            "(\"verify: unspecified launch failure\", #3895). If the engine stops that way, take STRATA_SH_STREAM out of the config's env "
            "(AMD runs without it by default) or run without --kv-resident, and tell us the card and the driver.")


def card_note(cfg: dict, env: dict) -> str | None:
    """One plain line that ties the engine's own numbering to the person's: the engine counts the cards it is given from 0 ("GPU 0",
    "CUDA0" in its log), so a person who chose GPU 1 would read "GPU 0" and think the choice was lost.  Said only when the two differ
    (NVIDIA, numbered as nvidia-smi numbers them); None otherwise."""
    if cfg.get("backend") == "hip" or not gpu_list(cfg):
        return None
    mine = [x.strip() for x in str((env or {}).get("CUDA_VISIBLE_DEVICES", "")).split(",") if x.strip() != ""]
    if not mine or mine == [str(i) for i in range(len(mine))]:
        return None
    names = [f"CUDA{i}" for i in range(len(mine))]
    if len(mine) == 1:
        return f"[strata] running on GPU {mine[0]} (as nvidia-smi numbers it); the engine's log calls it GPU 0 / CUDA0"
    return (f"[strata] cards in stage order: GPU {', then GPU '.join(mine)} (as nvidia-smi numbers them); the engine's log calls them "
            f"{', '.join(names)} - the first runs the first layers and the picture reader, the last runs the head and the draft layer")


def build_arch_numbers(value) -> list[int]:
    """The CUDA architectures of a BUILD.json `archs` value as sorted numbers.  The Linux packages write ONE string ("60;61;70;75;80;86;120;89");
    other builds write a list ([75, 86], ["sm_75", "compute_89"]); "60,61 70" is read too.  A string used to be walked character by
    character, which said "built for sm_0, sm_1, sm_2 ..." on every Pascal card (0.2.16 retest, p1004)."""
    if value is None or value == "":
        return []
    parts = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[;,\s]+", str(value))
    found = set()
    for part in parts:
        m = re.search(r"\d+", str(part))
        if m:
            found.add(int(m.group()))
    return sorted(found)


def engine_arch_note(cfg: dict, cards: dict[int, int] | None = None, built: dict | None = None) -> str | None:
    """#4662: a compiled engine holds code for the architectures it was built for (BUILD.json "archs"); on a card of another one a long prompt
    ends in "unspecified launch failure" (the kernels it needs are a trap there).  One plain line when a card this run uses is not covered
    (NVIDIA; a build with PTX also covers newer cards).  Said, never refused: the engine is started as asked.  `cards`: {nvidia-smi number:
    compute capability x 10}; `built`: the BUILD.json dict - both read here when not given.  None when everything is covered or unknown."""
    try:
        if cfg.get("backend") == "hip":
            return None
        if built is None:
            exe = cfg.get("exe")
            if not isinstance(exe, str) or engine_component.is_reference(exe):
                return None
            built = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8-sig"))
        archs = build_arch_numbers(built.get("archs"))
        if not archs:
            return None
        if cards is None:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                               errors="replace", timeout=10)
            cards = {int(i): int(round(float(c) * 10)) for i, c in (ln.split(",") for ln in (r.stdout or "").strip().splitlines())}
        used = gpu_list(cfg) or sorted(cards)
        miss = sorted({cards[i] for i in used if i in cards and cards[i] not in archs and not (built.get("ptx") and cards[i] > max(archs))})
        if not miss:
            return None
        names = ", ".join(f"sm_{m}" for m in miss)
        return (f"[strata] this engine was built for {', '.join(f'sm_{a}' for a in archs)}, but a graphics card in use is {names}: long prompts "
                "can stop with \"unspecified launch failure\". Update the engine (./setup.sh --update, or `strata update`) so it is built for every card in this PC.")
    except Exception:
        return None


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    ALWAYS = ()                                     # specials matched without parse_special (type 4, as <think>)

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
