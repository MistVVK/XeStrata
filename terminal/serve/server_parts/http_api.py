"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the HTTP part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import hmac
import json
import os
import re
import socket
import socketserver
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from strata_control import engine_component
from strata_control import reader_room
from strata_control import hipblaslt_tuning

from serve.server_parts.base import LOOPBACK_NAMES


class BadBody(Exception):
    """A request body that cannot be read (a malformed or oversized chunked body): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def anthropic_error_type(code: int, current) -> str:
    """error.type on /v1/messages, by the status rows of platform.claude.com/docs/en/api/errors.  502 and 503 have no
    row and keep their type: a 503 must not read as overloaded_error, which Claude Code treats as an overload."""
    known = {401: "authentication_error", 403: "permission_error", 404: "not_found_error", 409: "conflict_error",
             413: "request_too_large", 429: "rate_limit_error", 500: "api_error"}.get(code)
    return known or ("invalid_request_error" if code < 500 else current or "api_error")


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        if ":" in server_address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

    def server_bind(self):
        # http.server's own server_bind also asks socket.getfqdn(host) for a server_name nothing here reads; on a Mac whose reverse lookup
        # stalls (GitHub's macOS runner: 35 s on the first lookup of every new process) each start waited that long before it listened
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)

    def server_close(self):
        # the hardware sampler serve() started for this server stops with it (test isolation: every Service a test
        # started used to leave its sampler running, 21 of them after test_responses, which slowed test_parallel's timing)
        tel = getattr(getattr(self, "svc", None), "telemetry", None)
        if tel is not None and hasattr(tel, "close"):
            tel.close()
        super().server_close()


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[strata] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


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
           f"{free:.0f} GiB free of {total:.0f} GiB: what does not fit goes to swap or is read from disk, which can make "
           "decoding very slow. Lower it (setup sizes it for this PC) or remove the argument to let the engine choose.")
    print("[strata] " + msg, flush=True)
    return msg


DESKTOP_FREE_MIB = 2048          # #560 #516: below this, an AMD card that also drives a Linux desktop can run out


DESKTOP_RESERVE_MIB = 3072       # what kept KDE/Wayland alive beside a full expert cache in both reports


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X): its compositor, browser and apps take VRAM after the model has."""
    env = os.environ if env is None else env
    return os.name != "nt" and sys.platform != "darwin" and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_vram_note(backend, vram_free_mib, args: list, desktop: bool) -> str:
    """#560 #516: on Linux, when the desktop needs VRAM the AMD card does not have, amdgpu moves GPU memory (the expert
    cache, ~24 GB) to system RAM, which the experts already fill - the OOM killer then ends the compositor.  The
    default reserve (700 MiB) is sized for a card without a desktop.  A recommendation, nothing changes: "" when it
    does not apply."""
    if backend != "hip" or not desktop or not isinstance(vram_free_mib, int) or vram_free_mib >= DESKTOP_FREE_MIB:
        return ""
    try:
        reserve = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args else 700
    except (ValueError, IndexError):
        reserve = 700
    if reserve >= DESKTOP_RESERVE_MIB:
        return ""
    return (f"[strata] note: {vram_free_mib} MiB of VRAM free with the model loaded. If this AMD card also drives your "
            "desktop and the desktop or apps crash after the start (the driver moves the expert cache to RAM and "
            "the OOM killer ends the session), keep more VRAM free: ./setup.sh --vram-reserve-mib "
            f"{DESKTOP_RESERVE_MIB} (remembered; the expert cache gets "
            f"{(DESKTOP_RESERVE_MIB - reserve) / 1024:.1f} GB less, a few % of speed)")


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


def open_server_warning(port, public=False, tools=False) -> str:
    """The one plain warning for a server on the network with no API key (recommend, never force)."""
    msg = f"no API key: anyone who can reach port {port} can use the model" + (" and its tools" if tools else "") + "."
    if tools:
        msg += " The tools can read and change files on this computer."
    if public:
        msg += " This address is on the public internet, so anyone who finds it can use it."
    return msg + " Add an API key to keep others out."


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
               "experimental_speed_projection")


def load_shared_settings(path, warn=print) -> dict:
    """The Chat settings kept next to a config between starts, read so that one bad key cannot erase the others.

    The strict reader below belongs to POST /settings, where a client is there to be told which key is wrong and can
    fix it.  At start-up nobody is there: the old code called the same strict reader and caught its ValueError with
    `svc.shared = {}`, so a single unknown key - a typo, a key from a newer version, or a setting the server does not
    know - discarded every OTHER setting in the file, `reasoning_effort` included, without a word in the log (#4933).

    So the two failures are told apart here: a file that cannot be read, or is not a JSON object, is still ignored as
    a whole (nothing in it can be trusted); a file that IS an object keeps every key the server understands, and each
    key it does not is named on one line and skipped."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}                   # no file yet: the normal case on a first start, nothing to report
    except (OSError, ValueError) as e:
        warn(f"[strata] the Chat settings in {path} are ignored ({e}); other apps use their own settings")
        return {}
    if not isinstance(doc, dict):
        warn(f"[strata] the Chat settings in {path} are ignored (a JSON object was expected, found "
             f"{type(doc).__name__}); other apps use their own settings")
        return {}
    out = {}
    for key, value in doc.items():
        try:
            out.update(clean_shared_defaults({key: value}))
        except ValueError as e:
            warn(f"[strata] one Chat setting in {path} is skipped ({e}); the others still apply")
    return out


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
            if value not in ("none", "low", "medium", "high"):
                raise ValueError("reasoning_effort: none, low, medium or high")
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
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


def split_joined_args(cfg: dict) -> None:
    """#2267: an engine option written as ONE element of the config's "args" - "--ple-io mmap" or "--ple-io=mmap" - reaches
    the engine as a single argument ("unknown argument: --ple-io mmap"; the usual form is two elements, "--ple-io",
    "mmap").  Recommend, never force: such an element is split into the flag and its value, and said at startup.  Only an
    element that starts with a flag name (--letters-digits-dashes) followed by a space or "=" is touched, and never one
    that sits in the value place of the flag before it."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, changed = [], []
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
        print("[strata] the config's args hold " + ", ".join(repr(c) for c in changed) + " as one element each: split into the "
              'flag and its value (write them as two elements, e.g. "--ple-io", "mmap")', flush=True)


def read_config(path) -> dict:
    """The run config's JSON.  A mistake in it (a missing comma between two args, a trailing comma) is said with its
    place in the file instead of a traceback (#2267: editing "args" by hand is where it happens)."""
    return parse_config_text(Path(path).read_text(encoding="utf-8-sig"), path)   # Notepad adds a BOM


def parse_config_text(text: str, path) -> dict:
    """read_config on text already read (the owner-launched server hashes the bytes it parses, launch attestation)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[strata] {path} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}. Check the commas "
                         'between the entries of "args" (each option and each value is its own quoted entry, e.g. '
                         '"--ple-io", "mmap",) and that no entry is followed by a comma at the end of a list.') from None


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


def apply_saved_tuning(cfg: dict, *, say=None) -> dict:
    """The run config with the active saved tuning for its choices put on top (strata_control/presets.py), with no measurement.
    Every start path runs this (the Terminal, `runtime start`, the app's load, a restart of the app, an unload and a load), so the
    tuning is applied every time and never re-measured. A tune's own worker sets STRATA_NO_PRESET so that it measures the
    configuration it was given. Never fails a start: a store that cannot be read means the configuration as it is."""
    if os.environ.get("STRATA_NO_PRESET"):
        return cfg
    say = say or (lambda line: print(line, flush=True))
    try:
        from strata_control import presets as _presets
        out, applied = _presets.overlay(cfg)
        if applied.get("applied"):
            changes = {**(applied.get("args") or {}), **(applied.get("env") or {})}
            note = applied.get("notice") or ""
            say(f"[strata] saved tuning '{applied['preset']['name']}' applied: "
                f"{' '.join(f'{k} {v}' for k, v in changes.items()) or 'the standard settings (nothing to change)'}"
                f"{'; ' + note if note and 'identical' not in note else ''}")
            for warning in applied.get("warnings") or []:   # #3844: outside the range the tuner tries, kept
                say(f"[strata] warning: {warning}")
            if applied.get("state") == "soft":
                say("[strata] this PC or engine changed since it was saved (" + "; ".join(applied["reasons"]) +
                    "): a short re-check is available (strata presets recheck)")
        elif applied.get("why_not"):
            say(f"[strata] saved tuning '{applied['preset']['name']}' {applied['why_not']}; re-tune with strata tune auto")
        return out
    except Exception as e:  # noqa: BLE001
        say(f"[strata] saved tuning could not be read ({type(e).__name__}); starting with the configuration as it is")
        return cfg


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
                raise SystemExit(f"[strata] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[strata] config sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[strata] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "strata_tune":           # a saved tuning's per-request settings (strata_control/presets.py)
            if not isinstance(value, dict):
                raise SystemExit(f"[strata] config sampling.strata_tune={value!r}: expected an object")
            out[key] = {k: v for k, v in value.items() if k in ("pcie_frac", "spec_min_p", "aux_cpus")}
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[strata] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


SPLIT_KEYS = ("split_fallback", "split_engine_note")


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
                num = float(value)
                num = int(num) if integer and num == int(num) else (num if not integer else None)
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


# the API's model name per model family (the GGUF architecture, "arch" in the config) when the config names none
FAMILY_MODEL_NAMES = {"qwen4exp": "qwen3.8-flash-next", "qwen35moe": "qwen3.6-35b-a3b"}


# The sampling each model card recommends ("Best Practices"), used when neither the request, the Chat settings nor the
# run config's `sampling` block says otherwise.  Greedy is only for a request or config that asks for temperature 0.
# min_p 0, repetition_penalty 1 and presence_penalty 0 are the engine's own "off", so they are not listed.
MODEL_CARD_SAMPLING = {
    # Qwen3.8-Flash-Next (and Gyro, the same base model)
    "qwen4exp": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                 "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # Qwen3.6-35B-A3B: the general-task sets (the card's precise-coding set is temperature 0.6, presence_penalty 0)
    "qwen35moe": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                  "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
}


# The card's penalties: a greedy request must not get them (see model_card_sampling)
CARD_PENALTIES = ("presence_penalty", "frequency_penalty", "repetition_penalty")


def model_card_sampling(arch, thinking, greedy=False) -> dict:
    """The model card's sampling for the family `arch` and the thinking state ({} for an unknown or unset family).
    `greedy` (temperature 0 asked for, see asks_greedy): without the card's penalties, so a greedy request stays greedy.
    A penalty makes the engine sample from its logits: the answer is no longer the greedy one, and the MTP drafts
    (greedy requests only) never run - with Qwen3.6's presence_penalty 1.5, 0 drafts on every temperature-0 request."""
    sets = MODEL_CARD_SAMPLING.get(arch) if arch else None
    card = dict(sets["thinking" if thinking else "instruct"]) if sets else {}
    if greedy:
        for k in CARD_PENALTIES:
            card.pop(k, None)
    return card


def asks_greedy(*layers) -> bool:
    """True when greedy is asked for: the first of `layers` (highest precedence first - the request, the shared Chat
    settings, the config's `sampling` block) that sets a temperature sets 0."""
    for layer in layers:
        t = (layer or {}).get("temperature")
        if t is not None:
            return isinstance(t, (int, float)) and not isinstance(t, bool) and float(t) <= 0.0
    return False


def default_model_name(cfg: dict) -> str:
    """The config's "model_name", else its family's name ("arch": qwen4exp, qwen35moe), else Flash-Next's."""
    return cfg.get("model_name") or FAMILY_MODEL_NAMES.get(cfg.get("arch") or "qwen4exp", "qwen3.8-flash-next")


def with_current_reader(cfg: dict, package: Path) -> dict:
    """The picture reader (vision "exe", "cpu_exe") is written into the run config as an absolute path inside the engine package's folder,
    and that folder is named after the engine version: after an engine update, or in a container made for another card, the path is gone
    and the server would stop with "server_exited". Use the same file of the package that is installed now (the reader ships in the package)."""
    vision = cfg.get("vision")
    if not isinstance(vision, dict):
        return cfg
    fixed = dict(vision)
    for key in ("exe", "cpu_exe"):
        old = fixed.get(key)
        if isinstance(old, str) and old and not engine_component.is_reference(old) and os.path.isabs(old):
            new = package / old.replace("\\", "/").rsplit("/", 1)[-1]
            gone = not os.path.exists(old)
            # #4658: the old engine folder is kept after `strata update`, so the old reader still exists; one in a SIBLING engine folder
            # (components/engine/<version>-...) is the engine's reader from before the update, and the engine in use has its own
            older = (not gone and Path(old).parent != package and Path(old).parent.parent == package.parent
                     and package.parent.name.lower() == "engine")
            if (gone or older) and new.is_file():
                print(f"[strata] the picture reader {old} " + ("is not there any more" if gone else "belongs to an older engine")
                      + f"; using {new} from the engine package in use", flush=True)
                fixed[key] = str(new)
    if fixed == vision:
        return cfg
    return {**cfg, "vision": fixed}


READER_NEEDS_MIB = reader_room.READER_NEEDS_MIB     # what the GPU picture reader takes from the card (1,342 MiB measured on an RTX A6000 and an R9700)


#: Owner rule 2026-11-01: the picture reader lives in RAM, not VRAM.  On an NVIDIA card with one GPU the GPU reader is started by the first
#: picture and stopped after vision.elastic_idle_s, inside a reserve of VISION_RESERVE_MIB; a start that finds no room is read by the CPU reader.
#: Measured on an RTX A6000 as a 24 GB card (Qwen3.8-Flash-Next IQ3_S, 5 interleaved pairs): the same 20,623 expert slots and the same text speed as
#: the reader on the card, 1.3 GB less VRAM in use between pictures.  The expert cache that gives its room (vram_elastic) held 8.5% more slots
#: and was 2.2% slower in text (4 pairs of 5) in that first measurement; 11-11 it is the default (VISION_NO_RESERVE_BY_DEFAULT): on an RTX
#: PRO 5000 held to a 32 GB card (IQ3_S streamed, 5 rotated rounds) its text was level with the processor reader (2K / 8K / 20K 0.998 /
#: 0.997 / 0.998) and 2.9% faster than the room held back at 2K (12,719 against 12,022 slots), level when the model fits.
VISION_ON_DEMAND_BY_DEFAULT = True


VISION_RESERVE_MIB = 2048


_ON_DEMAND_NOT_WITH = ("--kv-grow", "--batch", "--slots", "--peer-device", "--expert-cache-device1", "--expert-cache-device2",
                       "--expert-cache-device3")


def with_component_engine(cfg: dict) -> dict:
    """A run config that names "component:strata-engine" gets the engine package for this PC: the path in `exe`, and
    the package's own folder first on the library path (the GPU runtime libraries ship beside the binary)."""
    if not engine_component.is_reference(cfg.get("exe")):
        return cfg
    cfg = dict(cfg)
    cfg["exe"] = _component_exe(cfg["exe"], cfg)
    cfg["lib_dirs"] = [*engine_component.package_lib_dirs(Path(cfg["exe"]).parent), *(cfg.get("lib_dirs") or [])]
    cfg = with_current_reader(cfg, Path(cfg["exe"]).parent)
    if cfg.get("backend") == "hip":     # AMD: the package's own hipBLASLt tuning table for this card, when its hipBLASLt version matches
        cfg = hipblaslt_tuning.apply(cfg, Path(cfg["exe"]).parent, say=lambda line: print(line, flush=True))
    return cfg


def _history_call_names(req) -> list:
    """The tool names a request's own conversation called, in order, on any of the three routes: Chat Completions (an assistant
    message's tool_calls), Anthropic (an assistant message's tool_use blocks) and Responses (function_call / custom_tool_call
    input items, a namespace in front as `namespace.name`)."""
    names = []
    for message in req.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            names.append((call.get("function") or {}).get("name") if isinstance(call, dict) else None)
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(block.get("name"))
    items = req.get("input")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("type") in ("function_call", "custom_tool_call"):
            name = item.get("name")
            if isinstance(name, str) and name and isinstance(item.get("namespace"), str) and item["namespace"]:
                name = f"{item['namespace']}.{name}"
            names.append(name)
    return names


def with_history_tools(tools, req) -> list:
    """#4663: an agent's tool set changes between turns (deferred tools, MCP servers, a compacted history), and the model calls a tool it used
    earlier in this conversation that this turn's `tools` no longer lists.  That call came back as raw text with finish_reason "stop", and the
    client took it for prose and stopped.  A name the conversation's own assistant turns called is a tool the client has, so it counts as
    declared (the call comes back as a tool call; the client says what it does with it) - on Chat Completions, Anthropic and Responses.
    Nothing is added when the request declared no tools at all, or for a name that is already declared."""
    if not tools or not isinstance(req, dict):
        return tools
    try:
        have = {t.get("name") for t in tools if isinstance(t, dict)}
        extra = []
        for name in _history_call_names(req):
            if isinstance(name, str) and name and name not in have:
                have.add(name)
                extra.append({"name": name, "parameters": {"type": "object", "properties": {}}})
        return [*tools, *extra] if extra else tools
    except Exception:
        return tools


def _component_exe(reference: str, cfg: dict) -> str:
    """The engine a run config names as "component:strata-engine". No engine for this PC is a clear notice and exit 5;
    the server never goes looking on the internet for one (strata_control/engine_component.py)."""
    try:
        return engine_component.expand(reference, cfg)
    except engine_component.EngineMissing as e:
        print(f"[strata] {e}", file=sys.stderr, flush=True)
        raise SystemExit(e.exit_code) from None


def undeclared_tool_calls_setting(cfg: dict) -> str:
    """#4622: the config's "undeclared_tool_calls": "text" (the default, when the config does not name it) keeps a well-formed call of a tool the request
    did not declare as the text the model wrote, with strata_rejected_tool_calls beside it; "call" returns it as a tool call, for a client that has
    its own policy for unknown names (it answers the model with the tool names it has).  Anything else stops the start."""
    mode = cfg.get("undeclared_tool_calls")
    if mode is None:                                  # the environment names it too (STRATA_UNDECLARED_TOOL_CALLS); the config wins
        mode = os.environ.get("STRATA_UNDECLARED_TOOL_CALLS") or "text"
    if mode not in ("text", "call"):
        raise SystemExit(f"[strata] config \"undeclared_tool_calls\" must be \"text\" or \"call\", not {mode!r}")
    return mode


def tool_call_recovery_setting(cfg: dict) -> bool:
    """The config's "tool_call_recovery" (see OutputParser's recover): ON when the config does not name it (since 0.2.8; 0.2.7 and before: off),
    "tool_call_recovery": false turns it off. Anything but true or false stops the start."""
    recovery = cfg.get("tool_call_recovery", True)
    if not isinstance(recovery, bool):
        raise SystemExit(f"[strata] config \"tool_call_recovery\" must be true or false, not {recovery!r}")
    return recovery


#: #5787: the switch that turns the server's own crash reporter off.
FAULTHANDLER_OFF = "STRATA_NO_FAULTHANDLER"


def arm_faulthandler(env: dict | None = None) -> bool:
    """#5787: arm CPython's faulthandler for this process, on its own output, and say whether it armed.

    The reports of the serve process dying natively (`0xC0000005` / `0xC0000409` on Windows, SIGSEGV / SIGBUS on Linux
    and Mac) all have the same shape: the host log carries an exit code, the crash guard restarts the server in about
    two minutes, and nothing names the Python line.  #5787's ten restarts in three days needed the reporter to set
    `PYTHONFAULTHANDLER=1` in his own launcher, plus the private `python312.pdb` from the package, to read anything.
    Armed here, in the server, every launcher (the app, the Terminal, a source start) leaves the stack of the dying
    thread, and of the request workers beside it, in the log a person can already send.

    It writes nothing while the process runs.  `STRATA_NO_FAULTHANDLER=1` turns it off.  A faulthandler that cannot arm
    (no module, a closed output) is answered with False and never stops a start.  The process still dies by the same
    signal with the same exit code, so the crash guard, the app's restart and the OS's own dump are unchanged.
    """
    target = os.environ if env is None else env
    if target.get(FAULTHANDLER_OFF) == "1":
        return False
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)         # the request workers' stacks are what names a shared cause
        return True
    except Exception:                                  # noqa: BLE001 - arming a crash reporter never breaks a start
        return False
