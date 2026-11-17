"""Three ways a feed host can be unreachable, for the callers that must say NOTHING at a launch (lane NET; the owner's rule: every online check skips
silently when this computer is offline or the server is not there).

    for kind in net_dead.KINDS:
        with net_dead.dead_feed(kind) as source:
            ...call the code with `source` as the update feed...

* "offline": no name resolves (a computer with no network at all): the lookup fails at once.
* "refused": a local port nobody listens on (a firewall that refuses, a proxy that is down).
* "never answers": a server that accepts the connection and never sends a byte; the rule's windows (12 s, then 30 s) are scaled down so the test
  takes a second or two, and every call that gives its own window gets the same scaled one.
"""
from __future__ import annotations

import contextlib
import socket
from unittest import mock

from strata_control import netpolicy
from tests.terminal.test_netpolicy import BlackHole, refused_port

KINDS = ("offline", "refused", "never answers")
WINDOW = (0.3, 0.6)


@contextlib.contextmanager
def no_name_resolution():
    """A computer with no name service: every lookup of a name other than this computer fails at once, as with no network (loopback addresses and
    "localhost" still resolve). For the tests that stand for an offline PC: they never ask a real resolver (the test guard in conftest.py)."""
    import ipaddress
    real = socket.getaddrinfo

    def lookup(host, *args, **kwargs):
        text = str(host).strip("[]").lower() if host is not None else ""
        try:
            local = text in ("", "localhost") or ipaddress.ip_address(text).is_loopback
        except ValueError:
            local = False
        if local:
            return real(host, *args, **kwargs)
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    with mock.patch.object(socket, "getaddrinfo", lookup):
        yield


@contextlib.contextmanager
def dead_feed(kind: str):
    """Yield a feed address that is unreachable in the way `kind` says."""
    assert kind in KINDS, kind
    real_open = netpolicy.open_url

    def scaled(open_fn, request, **kw):
        kw["limits"] = WINDOW
        kw["gap"] = 0.05
        return real_open(open_fn, request, **kw)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(netpolicy, "open_url", scaled))
        stack.enter_context(mock.patch.object(netpolicy, "CONNECT_LIMITS", WINDOW))
        netpolicy.HOSTS.forget()
        if kind == "offline":
            stack.enter_context(mock.patch.object(socket, "getaddrinfo", side_effect=socket.gaierror(socket.EAI_NONAME, "Name or service not known")))
            yield "https://strata-feed.example.invalid/strata-feed.json"
        elif kind == "refused":
            yield f"http://127.0.0.1:{refused_port()}/strata-feed.json"
        else:
            hole = BlackHole()
            try:
                yield f"http://127.0.0.1:{hole.port}/strata-feed.json"
            finally:
                hole.close()
        netpolicy.HOSTS.forget()
