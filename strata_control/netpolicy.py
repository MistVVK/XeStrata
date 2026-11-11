"""One rule for every network call on the install -> update -> launch path (lane NET, owner rules: offline users, no silent waits).

Many people run Strata fully local or behind a network that drops packets without refusing them (a firewall, a black-holed route, a
broken proxy). There a plain ``urlopen(timeout=60)`` waits a minute per try and setup tried five times per file: 320 s with nothing on
the screen. The rule here:

1. The *connection phase* - name lookup, TCP connect, TLS handshake, a proxy's CONNECT and the wait for the server's first answer (its
   headers) - gets at most ``FIRST_CONNECT_S`` on the first try and ``RETRY_CONNECT_S`` on the one retry (``CONNECT_LIMITS``). The
   name lookup runs in a helper thread, because ``getaddrinfo`` itself has no timeout.
2. No retry and no wait when the answer is already clear: no such host (no DNS answer), connection refused, network or host
   unreachable, a refused certificate, an HTTP error other than 408/429/5xx.
3. One retry, ``RETRY_GAP_S`` later, for a timeout, a reset before the first byte, or HTTP 408/429/5xx. A host that never answers is
   given up after about 12 + 2 + 30 = 44 s.
4. A host given up under rule 3 after the whole window (12 s, then 30 s) is not tried again for ``HOST_MEMORY_S`` *in the same process*
   (``HOSTS``, memory only, never on disk; a call with shorter limits reads it but never writes it;
   a new run always tries afresh). The next file from it fails at once with the same message.
5. Once the answer has come, a read that gets no byte for ``STALL_S`` ends that connection (the caller resumes from the byte it
   reached). A download stops only after ``GIVE_UP_S`` in a row without a new byte (``StallClock``); there is no total time cap, so a
   slow link that works keeps working.
6. When the connection phase has not ended after ``WAIT_LINE_AFTER_S``, a line names the host (``WaitLine``), again every
   ``WAIT_LINE_EVERY_S``.

The words people read are in ``WORDS`` (one place, so a wording change is one edit). Proxy settings (HTTPS_PROXY, the system's) work
as with ``urllib.request.urlopen``: only the connection class differs.
"""

from __future__ import annotations

import errno
import http.client
import re
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Iterable

FIRST_CONNECT_S = 12.0      # first try: measured, one github.com request in 25 hung while the next answered in 0.5 s
RETRY_CONNECT_S = 30.0      # the one retry
CONNECT_LIMITS = (FIRST_CONNECT_S, RETRY_CONNECT_S)
RETRY_GAP_S = 2.0
STALL_S = 60.0              # no byte for this long ends one connection (the download resumes)
GIVE_UP_S = 300.0           # no new byte for this long, across every try, ends the download
WAIT_LINE_AFTER_S = 5.0
WAIT_LINE_EVERY_S = 10.0
HOST_MEMORY_S = 600.0
RETRY_HTTP = (408, 429, 500, 502, 503, 504)

# The words. {host} is a host name, {seconds} a whole number, {what} what was being fetched ("the picture reader").
WORDS = {
    "waiting": "Waiting for {host} to answer ({seconds} s so far)",
    "no_answer": "{host} did not answer within {seconds} seconds, so {what} could not be downloaded.",
    "offline": "This computer could not connect to {host} (it looks offline), so {what} could not be downloaded.",
    "again": "{host} did not answer a moment ago either, so {what} could not be downloaded.",
    "next": ("Check that this computer can open https://{host}, then run the same command again: what is already downloaded is kept. "
             "Models that are already on this computer still start without the internet."),
    "stalled": "No data arrived for {minutes} minutes, so the download of {what} stopped at {percent}%.",
    "stalled_next": "Run the same command again: it continues where it stopped.",
    "update_no_answer": ("{host} did not answer within {seconds} seconds, so the update check could not run. Check that this computer can open "
                         "https://{host}, then try again. Strata keeps working with what is installed."),
    # (W7's two companions, owner 11-10)
    "update_offline": "this computer could not connect to {host} (it looks offline), so the update check could not run. Strata keeps working with what is installed.",
    "update_again": "{host} did not answer a moment ago, so the update check was not tried again. Strata keeps working with what is installed.",
}

#: the waiting line as setup prints it ("  Waiting for github.com to answer (15 s so far) ..."), for the progress readers
WAITING_RE = re.compile(r"^\s*Waiting for (?P<host>\S+) to answer \((?P<seconds>\d+) s so far\)")


def what_words(label: str | None, name: str | None = None) -> str:
    """What a download is, in the words of a failure message: setup's internal labels become plain words."""
    low = (label or "").lower()
    if not label:
        return "the model files"
    if "vision" in low or "picture" in low or "mmproj" in low:
        return "the picture reader"
    if "llama.cpp" in low:
        return "the tools that prepare models"
    if low.startswith(("the ", "a ")):
        return label
    if "engine" in low:
        return "the " + label
    return label


def host_of(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower() or url
    except ValueError:
        return url


# ------------------------------------------------------------------------------------------------ what went wrong

def kind(exc: BaseException | None) -> str:
    """'http' (an HTTP answer), 'http_retry' (408/429/5xx), 'cert', 'offline' (no DNS answer, refused, unreachable: clear at once),
    'no_answer' (a timeout), 'retry' (a reset or a dropped connection before the answer), or 'other'."""
    if exc is None:
        return "other"
    if isinstance(exc, urllib.error.HTTPError):
        return "http_retry" if exc.code in RETRY_HTTP else "http"
    from strata_control import tls
    if tls.is_certificate_error(exc):
        return "cert"
    seen = 0
    while isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), BaseException) and seen < 4:
        exc = exc.reason
        seen += 1
    if isinstance(exc, urllib.error.URLError):          # a reason that is only text: "timed out" is a timeout, other text is unclear
        return "no_answer" if "timed out" in str(exc.reason).lower() else "other"
    if isinstance(exc, socket.gaierror) or isinstance(exc, ConnectionRefusedError):
        return "offline"
    if isinstance(exc, OSError) and exc.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN):
        return "offline"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "no_answer"
    if isinstance(exc, ssl.SSLError) and "timed out" in str(exc).lower():
        return "no_answer"
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, http.client.HTTPException)):
        return "retry"
    return "other"


class NetFailure(OSError):
    """A host could not be reached under the rule: `kind` is 'no_answer', 'offline' or 'again' (given up earlier in this run).
    `seconds` is how long this call waited. str() is the plain headline with `what` = "the download"."""

    def __init__(self, host: str, kind_: str, cause: BaseException | None = None, seconds: float = 0.0):
        self.host, self.kind, self.cause, self.seconds = host, kind_, cause, seconds
        super().__init__(self.headline("the download"))

    def headline(self, what: str) -> str:
        key = {"offline": "offline", "again": "again"}.get(self.kind, "no_answer")
        return WORDS[key].format(host=self.host, what=what, seconds=max(1, int(round(self.seconds))))

    def next_step(self) -> str:
        return WORDS["next"].format(host=self.host)

    def detail(self) -> str:
        """The raw reason, for a log or a Details line only."""
        c = self.cause
        return f"{type(c).__name__}: {getattr(c, 'reason', c)}" if c is not None else ""


# ------------------------------------------------------------------------------------------------ host memory (rule 4)

class HostMemory:
    """Hosts that did not answer (rule 3), kept in this process's memory only for `HOST_MEMORY_S`. Nothing is written to disk."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, keep: float | None = None):
        self._clock, self._keep = clock, keep
        self._failed: dict[str, float] = {}
        self._lock = threading.Lock()

    def remember(self, host: str) -> None:
        with self._lock:
            self._failed[host] = self._clock()

    def forget(self, host: str | None = None) -> None:
        with self._lock:
            if host is None:
                self._failed.clear()
            else:
                self._failed.pop(host, None)

    def failed_recently(self, host: str) -> bool:
        keep = HOST_MEMORY_S if self._keep is None else self._keep
        with self._lock:
            at = self._failed.get(host)
            if at is None:
                return False
            if self._clock() - at >= keep:
                self._failed.pop(host, None)
                return False
            return True


HOSTS = HostMemory()


# ------------------------------------------------------------------------------------------------ the waiting line (rule 6)

class WaitLine:
    """While the block runs: after `after` seconds call say(line) naming the host, then every `every` seconds with the time so far."""

    def __init__(self, host: str, say: Callable[[str], None] | None, after: float | None = None, every: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.host, self.say, self.clock = host, say, clock
        self.after = WAIT_LINE_AFTER_S if after is None else after
        self.every = WAIT_LINE_EVERY_S if every is None else every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.said = 0

    def _run(self) -> None:
        start = self.clock()
        if self._stop.wait(self.after):
            return
        while not self._stop.is_set():
            try:
                self.say(WORDS["waiting"].format(host=self.host, seconds=int(self.clock() - start)))   # type: ignore[misc]
                self.said += 1
            except Exception:                                    # noqa: BLE001 - a line that cannot be shown never stops a download
                pass
            if self._stop.wait(self.every):
                return

    def __enter__(self) -> "WaitLine":
        if self.say is not None:
            self._thread = threading.Thread(target=self._run, daemon=True, name=f"wait-{self.host}")
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        return False


# ------------------------------------------------------------------------------------------------ the connection phase (rule 1)

def _bounded_getaddrinfo(host: str, port: int, limit: float) -> list:
    """getaddrinfo has no timeout of its own: a resolver that never answers would hold the call for as long as the system says.
    It runs in a helper thread here, given up after `limit` seconds (the thread ends by itself later)."""
    box: dict = {}

    def look() -> None:
        try:
            box["infos"] = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        except BaseException as exc:                             # noqa: BLE001 - handed to the caller
            box["error"] = exc
    t = threading.Thread(target=look, daemon=True, name=f"dns-{host}")
    t.start()
    t.join(limit)
    if t.is_alive():
        raise TimeoutError(f"no answer to the name lookup of {host} within {limit:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["infos"]


def _numeric(timeout) -> float | None:
    return float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0 else None


def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):  # type: ignore[attr-defined]
    """socket.create_connection with the whole of DNS + connect bounded by `timeout` (a deadline, not a limit per step). The socket is
    returned with the time that is left as its timeout, so the TLS handshake and the wait for the answer fit in the same deadline."""
    limit = _numeric(timeout)
    if limit is None:
        return socket.create_connection(address, timeout, source_address)
    deadline = time.monotonic() + limit
    host, port = address[0], address[1]
    infos = _bounded_getaddrinfo(host, port, limit)
    last: BaseException | None = None
    for af, socktype, proto, _canon, sa in infos:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        try:                                                     # through socket.create_connection (a numeric address: no second lookup)
            sock = socket.create_connection(sa[:2], left, source_address)
            sock.settimeout(max(0.05, deadline - time.monotonic()))
            return sock
        except OSError as exc:
            last = exc
    if last is not None and not isinstance(last, (TimeoutError, socket.timeout)):
        raise last
    raise TimeoutError(f"{host} did not accept a connection within {limit:.0f} s")


class _Bounded:
    """Mixed into http.client's connection classes: the connection phase ends at a deadline (`timeout` seconds after connect() began,
    covering the answer's headers too); after the answer the socket gets `stall` (rule 5)."""
    stall: float = STALL_S

    def _bounded_init(self) -> None:
        self._create_connection = create_connection             # type: ignore[attr-defined]
        self._deadline = None

    def connect(self):                                           # type: ignore[override]
        limit = _numeric(self.timeout)                           # type: ignore[attr-defined]
        self._deadline = time.monotonic() + limit if limit else None
        super().connect()                                        # type: ignore[misc]
        if self._deadline is not None and self.sock is not None:  # type: ignore[attr-defined]
            self.sock.settimeout(max(0.05, self._deadline - time.monotonic()))     # type: ignore[attr-defined]

    def getresponse(self):                                       # type: ignore[override]
        sock = self.sock                                         # type: ignore[attr-defined]
        response = super().getresponse()                         # type: ignore[misc]
        if self._deadline is not None and sock is not None:
            try:
                sock.settimeout(self.stall)
            except OSError:
                pass
        return response


class BoundedHTTPConnection(_Bounded, http.client.HTTPConnection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._bounded_init()


class BoundedHTTPSConnection(_Bounded, http.client.HTTPSConnection):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._bounded_init()


def _handlers(stall: float):
    http_cls = type("StallHTTPConnection", (BoundedHTTPConnection,), {"stall": stall})
    https_cls = type("StallHTTPSConnection", (BoundedHTTPSConnection,), {"stall": stall})

    class Http(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(http_cls, req)

    class Https(urllib.request.HTTPSHandler):
        def https_open(self, req):
            kw = {"context": self._context}
            if sys.version_info < (3, 12) and hasattr(self, "_check_hostname"):
                kw["check_hostname"] = self._check_hostname
            return self.do_open(https_cls, req, **kw)
    return Http, Https


def build_opener(*handlers, stall: float | None = None) -> urllib.request.OpenerDirector:
    """urllib.request.build_opener with the bounded connection classes. `timeout` given to .open() is the connection-phase limit."""
    http_h, https_h = _handlers(STALL_S if stall is None else stall)
    return urllib.request.build_opener(http_h, https_h, *handlers)


_OPENER: urllib.request.OpenerDirector | None = None
_OPENER_LOCK = threading.Lock()


def urlopen(request, timeout=None):
    """urllib.request.urlopen with the rule's connection classes: `timeout` bounds the connection phase, reads after the answer stall
    out after STALL_S. Without a timeout it behaves as urllib does. (Tests that fake the network replace this function.)"""
    global _OPENER
    with _OPENER_LOCK:
        if _OPENER is None:
            _OPENER = build_opener()
        opener = _OPENER
    if timeout is None:
        return opener.open(request)
    return opener.open(request, timeout=timeout)


# ------------------------------------------------------------------------------------------------ one request under the rule

def open_url(open_fn: Callable, request, *, host: str | None = None, say: Callable[[str], None] | None = None,
             memory: HostMemory | None = None, limits: Iterable[float] | None = None, gap: float | None = None,
             sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic, check_memory: bool = True,
             wait_after: float | None = None, wait_every: float | None = None):
    """open_fn(request, timeout=limit) under rules 1-4 and 6. Returns the answer. Raises NetFailure when the host did not answer or
    cannot be reached, HTTPError for an HTTP answer (after the one retry of 408/429/5xx), and anything else (a certificate problem, a
    refusal of the acquisition rule) unchanged."""
    url = getattr(request, "full_url", request)
    host = host or host_of(str(url))
    try:                                                         # remembered per host and port (a test server on 127.0.0.1 is not github.com)
        key = (urllib.parse.urlsplit(str(url)).netloc or host).lower()
    except ValueError:
        key = host
    memory = HOSTS if memory is None else memory
    if check_memory and memory.failed_recently(key):
        raise NetFailure(host, "again")
    limits = tuple(CONNECT_LIMITS if limits is None else limits)
    # Only a failure that had the whole window of the rule (12 s, then 30 s) is remembered: a call with shorter limits (an optional read
    # of 5 s) may READ the memory and skip a host that already failed, but never WRITES it, so a slow but working link that a short read
    # gave up on still gets its full attempt from a download that needs it (coordinator, 11-10).
    full = len(limits) >= len(CONNECT_LIMITS) and all(a >= b for a, b in zip(limits, CONNECT_LIMITS))
    gap = RETRY_GAP_S if gap is None else gap
    start = clock()
    with WaitLine(host, say, wait_after, wait_every, clock):
        for i, limit in enumerate(limits):
            last = i + 1 >= len(limits)
            try:
                return open_fn(request, timeout=limit)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRY_HTTP and not last:
                    try:
                        exc.close()
                    except Exception:                            # noqa: BLE001
                        pass
                    sleep(gap)
                    continue
                raise
            except (OSError, http.client.HTTPException) as exc:
                k = kind(exc)
                if k in ("no_answer", "retry") and not last:
                    sleep(gap)
                    continue
                if k in ("no_answer", "retry"):
                    if full:
                        memory.remember(key)
                    raise NetFailure(host, "no_answer", exc, clock() - start) from None
                if k == "offline":
                    raise NetFailure(host, "offline", exc, clock() - start) from None
                raise
    raise NetFailure(host, "no_answer", None, clock() - start)          # (no limits given)


# ------------------------------------------------------------------------------------------------ the download's clock (rule 5)

class StallClock:
    """`progress(have)` with the bytes so far after every try; `expired()` is True once no new byte came for GIVE_UP_S."""

    def __init__(self, give_up: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.give_up = GIVE_UP_S if give_up is None else give_up
        self.clock = clock
        self.have = -1
        self.since = clock()

    def progress(self, have: int) -> None:
        if have > self.have:
            self.have, self.since = have, self.clock()

    def idle(self) -> float:
        return self.clock() - self.since

    def expired(self) -> bool:
        return self.idle() >= self.give_up

    def words(self, what: str, have: int, total: int) -> tuple[str, str]:
        pct = int(100 * have // total) if total else 0
        return (WORDS["stalled"].format(minutes=max(1, int(self.give_up // 60)), what=what, percent=pct), WORDS["stalled_next"])
