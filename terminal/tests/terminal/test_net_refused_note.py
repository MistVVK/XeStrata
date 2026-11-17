"""netpolicy rule 4b (owner 11-11, "Remembering refused update feeds on Windows": Ok): a feed host that REFUSED the connection is remembered
on disk for 6 hours, in Strata's own state folder, host:port and the time only.  Only automatic reads (`automatic=True`) read the note; a read
the person asked for by name ignores it; the first answer from the host removes it; a damaged or unreadable note is no note, without a message.

Windows needs about 2.0 s to report a refused connection (measured 2.03-2.04 s on the owner PC, owner-20261110-2003-a8da); a launch of a model
with its own engine shape read the feed on every start, so a PC behind a dead proxy paid that on every start.  Every test here works in a test
HOME (conftest.py and launch_support.Home redirect the state folder) or on a note file of its own."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from strata_control import netpolicy, paths, transfer
from tests.terminal.launch_support import Home
from tests.terminal.test_netpolicy import refused_port


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def note_in(folder: Path, clock=None) -> netpolicy.RefusalNote:
    return netpolicy.RefusalNote(path=folder / "net" / netpolicy.REFUSED_NOTE_NAME, clock=clock or Clock())


class Base(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix="strata-refused-")
        self.addCleanup(t.cleanup)
        self.dir = Path(t.name)
        self.clock = Clock()
        self.note = note_in(self.dir, self.clock)
        netpolicy.HOSTS.forget()
        self.addCleanup(netpolicy.HOSTS.forget)


def refuse(request, timeout):
    raise ConnectionRefusedError(10061 if os.name == "nt" else 111, "refused")


class TheNote(Base):
    def test_the_values_of_the_rule(self):
        self.assertEqual(netpolicy.REFUSED_NOTE_S, 6 * 3600)
        with Home():                                         # the default note lives in Strata's own state folder (read when used)
            self.assertEqual(netpolicy.REFUSALS.path(), paths.state_base() / "net" / "refused-hosts.json")
            base = Path(os.environ["LOCALAPPDATA" if os.name == "nt" else "XDG_STATE_HOME"])
            self.assertEqual(netpolicy.REFUSALS.path().parent.parent.parent, base, "inside the test HOME's state folder")

    def test_host_port_and_time_only_and_six_hours(self):
        self.note.remember("github.com:443")
        self.assertEqual(json.loads(self.note.path().read_text(encoding="utf-8")), {"github.com:443": self.clock.t})
        self.assertTrue(self.note.refused_recently("github.com:443"))
        self.assertFalse(self.note.refused_recently("github.com:8443"), "per host AND port")
        self.clock.t += 6 * 3600 - 1
        self.assertTrue(self.note.refused_recently("github.com:443"))
        self.clock.t += 1
        self.assertFalse(self.note.refused_recently("github.com:443"), "after 6 hours the host is asked again")
        self.note.remember("example.org:443")                # writing drops the expired entries
        self.assertEqual(sorted(json.loads(self.note.path().read_text(encoding="utf-8"))), ["example.org:443"])

    def test_a_clock_that_went_back_does_not_keep_a_host_forever(self):
        self.note.remember("github.com:443")
        self.clock.t -= 3600
        self.assertFalse(self.note.refused_recently("github.com:443"))

    def test_forget_removes_the_host_and_the_empty_note(self):
        self.note.remember("a:1")
        self.note.remember("b:2")
        self.note.forget("a:1")
        self.assertEqual(sorted(json.loads(self.note.path().read_text(encoding="utf-8"))), ["b:2"])
        self.note.forget("b:2")
        self.assertFalse(self.note.path().exists())

    def test_forget_without_a_note_writes_nothing(self):
        self.note.forget("a:1")
        self.assertFalse(self.note.path().parent.exists())

    def test_a_damaged_or_unreadable_note_is_no_note_and_says_nothing(self):
        p = self.note.path()
        p.parent.mkdir(parents=True)
        for text in ("{not json", "[1, 2]", '{"a:1": "yesterday"}', '{"a:1": true}', "\xff\xfe".encode("latin-1").decode("latin-1")):
            p.write_text(text, encoding="latin-1")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertFalse(self.note.refused_recently("a:1"), text)
                self.note.remember("c:3")                    # a damaged note is replaced by a good one
                self.assertTrue(self.note.refused_recently("c:3"))
            self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
        p.unlink()
        p.mkdir()                                            # a folder where the note should be: cannot be read or written
        self.assertFalse(self.note.refused_recently("a:1"))
        self.note.remember("a:1")                            # silently not kept
        self.note.forget("a:1")
        self.assertFalse(self.note.refused_recently("a:1"))


class OpenUrl(Base):
    def test_a_refusal_is_noted_and_the_next_automatic_read_does_not_connect(self):
        with self.assertRaises(netpolicy.NetFailure) as cm:
            netpolicy.open_url(refuse, "https://feed.example:443/strata-feed.json", refusals=self.note, automatic=True)
        self.assertEqual(cm.exception.kind, "offline")
        self.assertTrue(self.note.refused_recently("feed.example:443"))
        calls = []

        def counting(request, timeout):
            calls.append(1)
            return b"answer"
        with self.assertRaises(netpolicy.NetFailure) as cm:
            netpolicy.open_url(counting, "https://feed.example:443/strata-feed.json", refusals=self.note, automatic=True)
        self.assertEqual((cm.exception.kind, calls), ("offline", []), "the automatic read failed at once, without a connection")
        self.assertIn("could not connect", str(cm.exception), "the same offline words as a refusal")

    def test_a_read_asked_for_by_name_ignores_the_note_and_its_answer_removes_the_host(self):
        self.note.remember("feed.example:443")
        calls = []

        def answer(request, timeout):
            calls.append(1)
            return b"answer"
        self.assertEqual(netpolicy.open_url(answer, "https://feed.example:443/strata-feed.json", refusals=self.note), b"answer")
        self.assertEqual(calls, [1])
        self.assertFalse(self.note.refused_recently("feed.example:443"), "the first success removes the host")
        self.assertEqual(netpolicy.open_url(answer, "https://feed.example:443/strata-feed.json", refusals=self.note, automatic=True), b"answer")

    def test_an_http_answer_also_removes_the_host(self):
        import urllib.error
        self.note.remember("feed.example:443")

        def missing(request, timeout):
            raise urllib.error.HTTPError(str(request), 404, "Not Found", {}, None)
        with self.assertRaises(urllib.error.HTTPError):
            netpolicy.open_url(missing, "https://feed.example:443/x", refusals=self.note)
        self.assertFalse(self.note.refused_recently("feed.example:443"))

    def test_only_a_refusal_is_noted(self):
        import socket

        def no_name(request, timeout):
            raise socket.gaierror(-2, "Name or service not known")

        def timeout(request, timeout):
            raise TimeoutError("timed out")
        for fn in (no_name, timeout):
            with self.assertRaises(netpolicy.NetFailure):
                netpolicy.open_url(fn, "https://feed.example:443/x", refusals=self.note, limits=(0.01,), gap=0, sleep=lambda s: None)
        self.assertFalse(self.note.path().exists(), "no DNS answer and a timeout are not refusals (the first fails at once anyway)")

    def test_a_refusal_through_urllib_is_seen(self):
        import urllib.error
        self.assertTrue(netpolicy.refused(urllib.error.URLError(ConnectionRefusedError(111, "refused"))))
        self.assertFalse(netpolicy.refused(urllib.error.URLError("timed out")))


class RealSocket(Base):
    def test_a_refused_loopback_port_costs_its_wait_once_then_nothing(self):
        url = f"http://127.0.0.1:{refused_port()}/strata-feed.json"
        with mock.patch.object(netpolicy, "REFUSALS", self.note):
            t0 = time.monotonic()
            with self.assertRaises(transfer.Unreachable):
                transfer.read_small(url, timeout=5, automatic=True)
            first = time.monotonic() - t0
            t0 = time.monotonic()
            with self.assertRaises(transfer.Unreachable):
                transfer.read_small(url, timeout=5, automatic=True)
            second = time.monotonic() - t0
            t0 = time.monotonic()
            with self.assertRaises(transfer.Unreachable):
                transfer.read_small(url, timeout=5)              # asked for by name: it tries again
            explicit = time.monotonic() - t0
        print(f"refused loopback: first {first:.3f} s, second (noted) {second:.3f} s, explicit {explicit:.3f} s")
        self.assertLess(second, 0.2)
        if os.name == "nt":                                      # Windows reports a refusal after about 2 s: the note saves exactly that
            self.assertGreater(first, 1.0)
            self.assertGreater(explicit, 1.0)


if __name__ == "__main__":
    unittest.main()
