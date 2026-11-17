"""Native helper programs (the Rust crates under crates/): one versioned contract for every Python <-> helper boundary (#5961 and the
native-helper series #5970 #5972 #5973 #5974 #5975 #5978).

The rules, so that a helper's presence never changes behaviour by itself:

* **Explicit opt-in, off by default.** A helper runs only when ``STRATA_NATIVE_HELPERS`` names it (a comma list, or ``all``); ``off``,
  empty or unset means never. Nothing is discovered: no PATH lookup, no ``target/release`` or ``target/debug`` build of a checkout.
* **One place.** The program is ``<STRATA_NATIVE_HELPER_DIR>/strata-<name>`` (``.exe`` on Windows), a regular file, else the helper is
  unavailable.
* **Versioned protocol.** It is called as ``strata-<name> --strata-protocol 1 <command> <args...>`` and answers with exactly one JSON
  object on stdout: ``{"protocol": 1, "ok": true, "result": {...}}`` or ``{"protocol": 1, "ok": false, "error": {"code": N, "message":
  "..."}}``. Any other protocol number is a mismatch.
* **Bounded.** A wall-clock timeout (the child is killed), at most ``MAX_OUTPUT`` bytes of output, a strict JSON parser (Python's own:
  duplicate keys, NaN / Infinity and trailing data are refused), then the caller's schema check of ``result``.
* **Exact fallback.** Anything that breaks the contract (missing, not runnable, timeout, crash, too much output, bad JSON, wrong protocol,
  a result that fails the schema) raises ``Unavailable``: the caller then runs its Python path, whose answer is the answer. A well-formed
  ``ok: false`` is the helper's own verdict and raises ``Rejected``: the caller treats it as its Python path would treat the same
  rejection (it is never retried with a second implementation).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Callable

PROTOCOL = 1
ENABLE_ENV = "STRATA_NATIVE_HELPERS"
DIR_ENV = "STRATA_NATIVE_HELPER_DIR"
MAX_OUTPUT = 1 << 20
DEFAULT_TIMEOUT = 10.0


class Unavailable(Exception):
    """The helper could not give a trusted answer; the caller uses its Python path. `reason` is short and machine-readable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Rejected(Exception):
    """The helper answered within the contract that the input is not acceptable (its `ok: false`)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def enabled(name: str, environ=None) -> bool:
    raw = str((os.environ if environ is None else environ).get(ENABLE_ENV, "")).strip().lower()
    if raw in ("", "0", "off", "none", "no", "false"):
        return False
    names = {x.strip() for x in raw.split(",") if x.strip()}
    return "all" in names or name in names


def program(name: str, environ=None) -> Path | None:
    """The helper's program when it is opted in and present at the one allowed place, else None."""
    env = os.environ if environ is None else environ
    if not enabled(name, env):
        return None
    folder = str(env.get(DIR_ENV, "")).strip()
    if not folder:
        return None
    path = Path(folder) / (f"strata-{name}.exe" if os.name == "nt" else f"strata-{name}")
    try:
        details = path.stat()
    except OSError:
        return None
    return path if stat.S_ISREG(details.st_mode) else None


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constants(name):
    raise ValueError(f"{name} is not JSON")


def parse(text: str) -> dict:
    """One JSON object, strictly: no duplicate keys, no NaN / Infinity, nothing after it."""
    value = json.loads(text, object_pairs_hook=_strict_object, parse_constant=_no_constants)
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return value


def call(name: str, command: str, args: list[str], *, validate: Callable[[object], bool], timeout: float | None = None,
         environ=None) -> dict:
    """Run one helper command under the contract and return its validated `result` (see the module text for Unavailable / Rejected)."""
    exe = program(name, environ)
    if exe is None:
        raise Unavailable("not_enabled_or_missing")
    argv = [str(exe), "--strata-protocol", str(PROTOCOL), command, *map(str, args)]
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    try:
        done = subprocess.run(argv, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise Unavailable("timeout") from None
    except OSError:
        raise Unavailable("not_runnable") from None
    if len(done.stdout) > MAX_OUTPUT:
        raise Unavailable("output_too_large")
    try:
        answer = parse(done.stdout.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise Unavailable("bad_output") from None
    if answer.get("protocol") != PROTOCOL or type(answer.get("ok")) is not bool:
        raise Unavailable("protocol_mismatch")
    if answer["ok"]:
        if done.returncode != 0 or set(answer) != {"protocol", "ok", "result"} or not validate(answer["result"]):
            raise Unavailable("bad_result")
        return answer["result"]
    error = answer.get("error")
    if set(answer) != {"protocol", "ok", "error"} or not isinstance(error, dict) or type(error.get("code")) is not int or \
            not isinstance(error.get("message"), str) or not 1 <= error["code"] <= 255 or len(error["message"]) > 1000:
        raise Unavailable("bad_error")
    raise Rejected(error["code"], error["message"])
