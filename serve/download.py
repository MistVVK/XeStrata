# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared streaming and SHA-256 primitives for downloads, and the one network rule every transfer follows.

The rule (upstream fd40eb0fd and 4c0626849, lane 15 of upstream 0.2.13): the connection phase of a request (the name's
lookup, the connection, TLS, the proxy's CONNECT, the answer's headers) gets `FIRST_CONNECT_S`, then one retry with
`RETRY_CONNECT_S` `RETRY_PAUSE_S` later - about 44 s for a host that never answers, instead of the 320 s five 60 s
tries were (setup.py's size request used to do that).  After `WAIT_SAY_S` a line says what is being waited for, and
every `WAIT_EVERY_S` after it; a host whose whole window failed is not asked again for `DEAD_HOST_S` in this process
(memory only, nothing on disk); and a host that answered with an error, or whose certificate the system does not
trust, fails at once."""
import errno
import hashlib
import http.client
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF_DEFAULT = "https://huggingface.co"

FIRST_CONNECT_S = 12.0    # the first try's connection phase
RETRY_CONNECT_S = 30.0    # the one retry's, RETRY_PAUSE_S later (44 s in all with the pause)
RETRY_PAUSE_S = 2.0
WAIT_SAY_S = 5.0          # "Waiting for <host> to answer (<n> s so far) ..." after this long
WAIT_EVERY_S = 10.0       # and then every this long, until the answer or the end of the window
DEAD_HOST_S = 600.0       # a host that failed its whole window is not asked again for this long (this run)


class NetFailure(OSError):
    """A host that did not answer within the rule's window.  `reason` is the raw text for the log (a timeout, a
    refused port, a name that does not resolve); `headline` is the sentence a user reads."""

    def __init__(self, host: str, reason: str, waited: float = 0.0):
        super().__init__(f"{host}: {reason}")
        self.host, self.reason, self.waited = host, reason, waited

    def headline(self, what: str | None = None) -> str:
        tail = f" for {what}" if what else ""
        return f"cannot reach {self.host}{tail}: it did not answer ({self.waited:.0f} s)"

    def next_step(self) -> str:
        return "check your internet connection and run it again - what is already downloaded is kept"


_DEAD: dict[str, float] = {}
_DEAD_LOCK = threading.Lock()


def forget_dead_hosts() -> None:
    """Ask every host again (a new download in the same run, a test)."""
    with _DEAD_LOCK:
        _DEAD.clear()


def host_of(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).netloc or url
    except ValueError:
        return url


def _say_while_waiting(host: str, say):
    """The thread that says what the request waits for: the host's answer is all that is missing, and a transfer that
    says nothing looks like a hung program (upstream ed0c8939a's stall line, said before the first byte)."""
    stop = threading.Event()

    def run() -> None:
        start = time.monotonic()
        if stop.wait(WAIT_SAY_S):
            return
        while not stop.is_set():
            say(f"Waiting for {host} to answer ({time.monotonic() - start:.0f} s so far) ...")
            stop.wait(WAIT_EVERY_S)

    threading.Thread(target=run, name="xs-net-wait", daemon=True).start()
    return stop


def _final(exc: BaseException | None) -> bool:
    """A failure another try cannot fix: the host answered with an error (an HTTPError, raised apart), or its
    certificate is not trusted, or the request itself is wrong."""
    if exc is None:
        return False
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLError) or isinstance(exc, ssl.SSLError)


def open_url(request, *, say=None, timeout: float | None = None, tries: int = 2, check_memory: bool = True):
    """One request under the network rule (the module's docstring).  `timeout` replaces both limits (a short probe, a
    test); `tries=1` does not retry.  Raises NetFailure, or the HTTPError / other OSError the last try raised."""
    url = getattr(request, "full_url", None) or str(request)
    host = host_of(url)
    if check_memory:
        with _DEAD_LOCK:
            until = _DEAD.get(host, 0.0)
        if time.monotonic() < until:
            raise NetFailure(host, "it did not answer a moment ago")
    limits = ([timeout] if timeout is not None else [FIRST_CONNECT_S, RETRY_CONNECT_S])[:max(1, tries)]
    started = time.monotonic()
    last: BaseException | None = None
    for i, limit in enumerate(limits):
        if i:
            time.sleep(RETRY_PAUSE_S)
        stop = _say_while_waiting(host, say) if say is not None else None
        try:
            return urllib.request.urlopen(request, timeout=limit)
        except urllib.error.HTTPError:
            raise                                  # the host answered: the caller decides (a pinned file that is gone)
        except (OSError, http.client.HTTPException) as e:
            last = e
            if _final(e):
                break
        finally:
            if stop is not None:
                stop.set()
    if not _final(last):
        with _DEAD_LOCK:
            _DEAD[host] = time.monotonic() + DEAD_HOST_S
    raise NetFailure(host, str(getattr(last, "reason", last)), time.monotonic() - started)


# ---- what a disk that refuses a write means, in a sentence a user can act on (upstream 0771e3555)
_DISK_WORDS = {
    "ENOSPC": ("the drive is full", "free some space (or pick another --models-dir, --data-dir)"),
    "EDQUOT": ("the drive's quota for this user is used up", "free some space or raise the quota"),
    "EROFS": ("the folder is read-only", "pick a folder you can write to (--models-dir, --data-dir)"),
    "EACCES": ("this user may not write there", "pick a folder you can write to (--models-dir, --data-dir)"),
    "EPERM": ("this user may not write there", "pick a folder you can write to (--models-dir, --data-dir)"),
    "ENAMETOOLONG": ("the file's name is too long for the folder", "pick a shorter --models-dir path"),
}


def disk_words(exc: BaseException) -> tuple[str, str] | None:
    """(what happened, what to do) for a write that failed on the disk, or None when the error is not the disk's."""
    code = getattr(exc, "errno", None)
    if code is None:
        return None
    for name, words in _DISK_WORDS.items():
        if code == getattr(errno, name, None):
            return words
    return None


def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def range_matches(value: str | None, start: int, end: int | None = None, total: int | None = None) -> bool:
    """Check a Content-Range; an unspecified total permits the MTP reader's unknown total."""
    raw = (value or "").strip()
    if total is None and end is not None:
        return raw.startswith(f"bytes {start}-{end}/")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", raw)
    return bool(match and int(match[1]) == start and (end is None or int(match[2]) == end)
                and (total is None or int(match[3]) == total))


def resumed_at(response, have: int, total: int = 0, *, strict: bool = True) -> int:
    """The response's write offset: a full response restarts; strict callers reject an incorrect range."""
    if have:
        matches = not strict or range_matches(response.headers.get("Content-Range"), have, total=total or None)
        if response.status == 206 and matches:
            return have
        if response.status != 200 and strict:
            raise OSError(f"the server answered another range than bytes {have}- (HTTP {response.status})")
        return 0
    if strict and response.status not in (200, 206):
        raise OSError(f"HTTP {response.status}")
    return 0


def write_done(path, text: str) -> None:
    path = Path(path)
    path.with_name(path.name + ".done").write_text(text, encoding="utf-8")


def sha256_file(path, chunk: int = 16 << 20, progress=None) -> str:
    digest = hashlib.sha256()
    seen = 0
    with open(path, "rb") as f:
        while data := f.read(chunk):
            digest.update(data)
            seen += len(data)
            if progress is not None:
                progress(seen)
    return digest.hexdigest()


def copy_stream(source, out, *, have: int = 0, chunk: int = 1 << 20, limit: int = 0, progress=None, stop=None) -> int:
    while stop is None or not stop():
        data = source.read(chunk)
        if not data:
            break
        if limit and have + len(data) > limit:
            raise OSError("the server sent more bytes than the file has")
        out.write(data)
        have += len(data)
        if progress is not None:
            progress(have)
    return have
