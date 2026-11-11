"""The network guard of conftest.py (lane NET): it refuses what is not this computer, passes three named things (a UDP connect, this computer's own host
name, the RFC 5737 documentation addresses), counts every use of them per test, and nothing else gets through. These tests call the guard's own
functions with the real connect replaced, so no packet is ever sent; each one that makes the guard refuse clears the guard's list before it ends
(the autouse fixture would otherwise fail the test for the reach it was checking)."""

from __future__ import annotations

import os
import socket
import unittest
from unittest import mock


class _Live:
    """The root conftest.py's own namespace, the live one (several conftest.py files share the module name "conftest", so it is found through the
    function the guard put on socket.socket.connect, not by an import)."""

    def __init__(self, names: dict):
        object.__setattr__(self, "_names", names)

    def __getattr__(self, name):
        try:
            return self._names[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._names[name] = value

    def __delattr__(self, name):
        del self._names[name]


assert socket.socket.connect.__name__ == "_guarded_connect", "the network guard of conftest.py is not installed"
guard = _Live(socket.socket.connect.__globals__)


def tcp() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def udp() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class Guarded(unittest.TestCase):
    """The guard is on for these tests whatever the run says: a run with STRATA_TEST_ALLOW_NETWORK=1 (a live run, or a check that sets it) switches it
    off for the others, and these tests are about the guard itself."""

    def setUp(self):
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("STRATA_TEST_ALLOW_NETWORK", None)


class Refused(Guarded):
    def tearDown(self):
        del guard._reached[:]                                  # (the refusals checked here are not this test's failure)

    def test_a_tcp_connect_to_an_address_outside_the_allowances_is_refused(self):
        for host in ("203.0.114.1", "192.0.3.1", "198.51.101.1", "10.255.255.254", "8.8.8.8", "93.184.216.34"):
            with tcp() as s, mock.patch.object(guard, "_real_connect", side_effect=AssertionError("sent")):
                with self.assertRaises(ConnectionRefusedError, msg=host):
                    s.connect((host, 80))
        self.assertEqual(len(guard._reached), 6)

    def test_a_real_host_name_is_refused(self):
        for host in ("github.com", "huggingface.co", "example.com", "strata-feed.invalid"):
            with self.assertRaises(socket.gaierror, msg=host):
                socket.getaddrinfo(host, 443)
        self.assertEqual(len(guard._reached), 4)

    def test_a_name_that_only_looks_like_this_computers_is_refused(self):
        with self.assertRaises(socket.gaierror):
            socket.getaddrinfo(guard._OWN_NAME + ".example.com", 80)

    def test_the_failure_says_where_it_came_from(self):
        with self.assertRaises(socket.gaierror):
            socket.getaddrinfo("github.com", 443)
        self.assertRegex(guard._reached[-1], r"^github\.com \[test_net_guard\.py:\d+\]$")

    def test_a_tcp_connect_with_the_ipv6_documentation_range_is_not_covered(self):
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s, mock.patch.object(guard, "_real_connect", side_effect=AssertionError("sent")):
            with self.assertRaises(ConnectionRefusedError):
                s.connect(("2001:db8::1", 80, 0, 0))             # (only the three IPv4 ranges are allowed)


class Allowed(Guarded):
    def setUp(self):
        super().setUp()
        self.before = len(guard._allowed)

    def used(self):
        return guard._allowed[self.before:]

    def test_loopback_is_not_counted_at_all(self):
        with tcp() as s, mock.patch.object(guard, "_real_connect", return_value=None):
            s.connect(("127.0.0.1", 9))
        socket.getaddrinfo("localhost", 80)
        self.assertEqual(self.used(), [])

    def test_a_udp_connect_passes_and_is_counted(self):
        with udp() as s, mock.patch.object(guard, "_real_connect", return_value=None):
            s.connect(("10.255.255.255", 1))
        self.assertEqual([(w, a) for _t, w, a in self.used()], [("(a) UDP connect", "10.255.255.255:1")])
        self.assertTrue(self.used()[0][0].endswith("::Allowed::test_a_udp_connect_passes_and_is_counted"), self.used())

    def test_this_computers_own_name_passes_and_is_counted(self):
        with mock.patch.object(guard, "_real_getaddrinfo", return_value=[]):
            socket.getaddrinfo(socket.gethostname(), 0)
        self.assertEqual([(w, a.lower()) for _t, w, a in self.used()], [("(b) own host name", guard._OWN_NAME)])

    def test_the_three_documentation_ranges_pass_and_are_counted(self):
        for host in ("192.0.2.1", "198.51.100.7", "203.0.113.1"):
            with tcp() as s, mock.patch.object(guard, "_real_connect", return_value=None):
                s.connect((host, 8080))
        self.assertEqual([(w, a) for _t, w, a in self.used()],
                         [("(c) documentation address", "192.0.2.1:8080"), ("(c) documentation address", "198.51.100.7:8080"),
                          ("(c) documentation address", "203.0.113.1:8080")])


class OptOut(Guarded):
    def test_a_live_test_that_sets_the_variable_is_let_through_and_nothing_is_counted(self):
        os.environ["STRATA_TEST_ALLOW_NETWORK"] = "1"
        before = (len(guard._reached), len(guard._allowed))
        with tcp() as s, mock.patch.object(guard, "_real_connect", return_value=None):
            s.connect(("93.184.216.34", 80))
        with mock.patch.object(guard, "_real_getaddrinfo", return_value=[]):
            socket.getaddrinfo("github.com", 443)
        self.assertEqual((len(guard._reached), len(guard._allowed)), before)


if __name__ == "__main__":
    unittest.main()
