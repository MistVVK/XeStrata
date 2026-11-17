"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the head part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path


PICTURE_FORMATS = ("JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF", "AVIF")   # the picture formats a request may send (Pillow decodes them)


IM_END = "<|im_end|>"


IMAGE_PAD = "<|image_pad|>"


def rttime_limit_note() -> str:
    """#3238: Linux only. A hard RLIMIT_RTTIME that is not unlimited (PipeWire's client module sets it to 0 when rtkit is
    missing; every program started from that desktop inherits it) makes the kernel SIGKILL a process as soon as it
    checks a thread with an armed CPU timer, with no OOM kill and no message. Returns one sentence naming the cause and
    the fix, or "" when the limit is unlimited, the platform has no such limit or it cannot be read."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTTIME)
    except (ImportError, AttributeError, OSError, ValueError):
        return ""
    if hard == resource.RLIM_INFINITY:
        return ""
    return (f"This session has a real-time CPU limit (RLIMIT_RTTIME hard limit {hard}; the desktop's audio system sets "
            "it to 0 when rtkit is not installed). Linux can then end a program with signal 9 for no visible reason. "
            "Install rtkit, or start Strata outside the desktop session's limits, for example: "
            "systemd-run --user --collect -p LimitRTTIME=infinity -- strata serve ...")


LEFTOVER_SETTLE_S = 120.0       # #4051: how long an engine ended at a start is waited for to give its memory back


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
        if sys.platform == "win32":
            hint = ("Windows may keep this port for itself (see `netsh int ipv4 show excludedportrange protocol=tcp`) "
                    "or a firewall rule forbids it: start this one with a different --port")
        else:
            hint = ("This system does not allow it (on Linux and macOS, ports below 1024 need administrator rights): "
                    "start this one with a different --port")
    elif code in (10049, errno.EADDRNOTAVAIL):
        hint = f"{host} is not an address of this PC: check \"host\" in the model's strata-<model>.json or --host"
    else:
        hint = "check --host and --port"
    return f"cannot listen on {host}:{port}: {what} [{code}]. {hint}"


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


ANSWER_ROOM_MIN_CAP = 1024    # 0.1.45: from this max_tokens on, thinking without a budget is closed early enough to leave room to answer


ANSWER_RESERVE_MIN = 512     # #984: the tokens kept for the answer when a thinking budget is cut down to fit max_tokens


FIT_MIN_ROOM = 1024          # #5924: fit_max_tokens never shortens max_tokens below this (a 400 instead; see fit_floor)


def fit_floor(ctx: int) -> int:
    """#5924: the least room fit_max_tokens shortens a max_tokens to: FIT_MIN_ROOM (thinking plus an answer or a tool
    call), at most a quarter of a small context (test and tiny contexts keep their shortening)."""
    return max(1, min(FIT_MIN_ROOM, (ctx - CTX_SLACK) // 4))


# #123: what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
# #1053 (opt-in "reasoning_close_retry": true): a reply that ends on its stop token still inside <think>, with no answer
# and no call, is continued once with the thinking closed (the same way the budget's wrap-up closes it)
REASONING_CLOSE = "\n</think>\n\n"


REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"


# The budget's pass to an engine that says "hold" in READY carries GEN hold=N:ID (N: the reasoning tokens left, ID: the one
# token of </think>): the engine stops there with its session at exactly the tokens it sent, so the next pass (+ the wrap-up)
# continues from the live session.  Without it the engine ran a window or two past the cut, and the next pass read the
# whole reasoning again from the checkpoint before it.  The server's own key in `sampling`, never a client's.
HOLD_KEY = "_strata_hold"


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


KEEPALIVE_S = 10.0          # a held-back tool argument leaves the stream quiet: ping at least this often


# Anthropic clients add /v1/messages to their base URL themselves. A base URL that already ends in /v1 (copied from an OpenAI-style
# setup) makes them post /v1/v1/messages; these two paths mean the same as the real ones. Nothing else is aliased.
V1_V1_ALIASES = {"/v1/v1/messages": "/v1/messages", "/v1/v1/messages/count_tokens": "/v1/messages/count_tokens"}


CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here


class ContextOverflow(ValueError):
    """A request that does not fit the context (a 400 on every API). It keeps the numbers, so /v1/messages can also say it the way Anthropic's API
    does: a client that decides from the wording (Claude Code compacts on "prompt is too long", and shortens max_tokens on "input length and
    `max_tokens` exceed context limit") then does what it does for Anthropic's own error (#4838)."""

    def __init__(self, text, *, prompt, context, max_new=None, min_room=None):
        super().__init__(text)
        self.prompt, self.context, self.max_new, self.min_room = prompt, context, max_new, min_room

    def anthropic_text(self) -> str:
        if self.min_room:              # #5924: room left, but less than an answer needs: the client should compact
            return f"prompt is too long: {self.prompt} tokens > {self.context - CTX_SLACK - self.min_room} maximum; {self}"
        if self.max_new is None or self.prompt > self.context - CTX_SLACK - 1:     # the prompt alone leaves no room: the largest one that still leaves a token
            return f"prompt is too long: {self.prompt} tokens > {self.context - CTX_SLACK - 1} maximum; {self}"
        return f"input length and `max_tokens` exceed context limit: {self.prompt} + {self.max_new} > {self.context - CTX_SLACK}; {self}"


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


def _timeout_env(name: str, default: float) -> float | None:
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return None if v <= 0 else v


PAR_COUNTERS = os.environ.get("STRATA_PAR_COUNTERS") == "1"   # research counters (queue wait for a slot), shown in /v1/status


# STRATA_COALESCE_PREFILL=N (research, PAR-H; needs the engine's BCO / BCOK lines): a request that takes the control lines
# for its admission also sends the prompts of up to N-1 requests still waiting for them (BCO <slot> <ids>, into free slots
# it reserves for them); the engine reads all the prompts in ONE layer-major pass, so the experts a layer streams are
# read once for all of them, and each waiting request then finds its prompt in its slot.  0 / unset: off.
COALESCE_PREFILL = max(0, int(os.environ.get("STRATA_COALESCE_PREFILL") or 0))


COALESCE_TOKENS = max(256, int(os.environ.get("STRATA_COALESCE_TOKENS") or 16384))   # prompt tokens read together at most


COALESCE_WAIT_MS = max(0, int(os.environ.get("STRATA_COALESCE_WAIT_MS") or 0))       # a burst's later requests register within this long


# Adaptive: a read is shared (and the leader waits for its followers) only when this engine has seen at least
# COALESCE_MIN_REQS requests arrive within COALESCE_BURST_MS (the leader's own included).  Fewer than that - one or two
# requests per engine, as 8 requests on 4 replicas - read plainly, with no wait: nothing to share, and the wait and the
# longer pass only cost (measured 4 x 1 replicas at 8 requests: 168 against 196 tok/s with an unconditional wait).
COALESCE_MIN_REQS = max(2, int(os.environ.get("STRATA_COALESCE_MIN_REQS") or 3))


COALESCE_BURST_MS = max(10, int(os.environ.get("STRATA_COALESCE_BURST_MS") or 250))


STALL_CPU_EPS_S = 0.5       # CPU seconds the engine may use in the whole stall window and still count as frozen


STALL_IO_EPS_B = 1 << 20    # bytes read or written in that window


def engine_frozen(base: tuple[float, int], now: tuple[float, int]) -> bool:
    """#1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them."""
    return (now[0] - base[0]) <= STALL_CPU_EPS_S and (now[1] - base[1]) <= STALL_IO_EPS_B
