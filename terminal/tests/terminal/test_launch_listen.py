"""#5917: "One address only" - the server listens on one address of this computer (a second network card, a VPN), chosen in the Access step
or with `strata launch --listen ADDRESS`. Its URL is that address (127.0.0.1 would not answer), a key is on as for the local network, and an
address that is no longer on this computer stops the launch with a plain line instead of falling back to every address."""

from __future__ import annotations

import argparse
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

from strata_cli.launch import access as acc
from strata_cli.launch import settings as S
from tests.terminal.test_launch_tui import Base, press, text_of

HERE = ["192.168.1.100", "192.168.2.100", "10.8.0.2"]


def nic(address, family=socket.AF_INET):
    return SimpleNamespace(family=family, address=address)


class Addresses(unittest.TestCase):
    def test_every_usable_address_and_no_loopback_or_link_local(self):
        fake = {"eth0": [nic("192.168.1.100"), nic("fe80::1%eth0", socket.AF_INET6), nic("2001:db8::5", socket.AF_INET6)],
                "lo": [nic("127.0.0.1"), nic("::1", socket.AF_INET6)], "wg0": [nic("10.8.0.2")], "x": [nic("169.254.3.4")]}
        with mock.patch.object(acc, "lan_ips", return_value=["192.168.1.100", "10.8.0.2"]), \
                mock.patch("psutil.net_if_addrs", return_value=fake):
            self.assertEqual(acc.local_addresses(), ["192.168.1.100", "10.8.0.2", "2001:db8::5"])

    def test_the_host_and_the_url_are_that_address(self):
        self.assertEqual(acc.host_for("address", "192.168.2.100"), "192.168.2.100")
        self.assertEqual((acc.host_for("lan"), acc.host_for("local"), acc.host_for("cloudflare")), ("0.0.0.0", "127.0.0.1", "127.0.0.1"))
        s = S.LaunchSettings(access="address", listen="192.168.2.100", port=8080)
        self.assertEqual(acc.urls_for(s), ["http://192.168.2.100:8080"])
        self.assertEqual(acc.urls_for(s.copy(listen="2001:db8::5")), ["http://[2001:db8::5]:8080"])
        self.assertIn("--listen 192.168.2.100", S.flags_text(s))

    def test_an_address_that_left_is_refused_in_plain_words(self):
        with mock.patch.object(acc, "local_addresses", return_value=HERE):
            self.assertIsNone(acc.listen_problem("10.8.0.2"))
            self.assertEqual(acc.listen_problem("192.168.9.9"),
                             "192.168.9.9 is no longer an address of this computer. Its addresses now: 192.168.1.100, 192.168.2.100, "
                             "10.8.0.2. Choose one in the Access step.")

    def test_the_launch_stops_before_anything_is_done(self):
        from strata_cli.launch.runner import LaunchError, Runner
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home(), mock.patch.object(acc, "local_addresses", return_value=HERE):
            r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, access="address", listen="192.168.9.9"),
                       catalog=catalog(HW_12), emit=lambda e: None)
            with mock.patch.object(Runner, "ensure_engine", side_effect=AssertionError("must not get here")), \
                    self.assertRaises(LaunchError) as caught:
                r.run(start=True)
        self.assertIn("is no longer an address of this computer", caught.exception.text)
        self.assertEqual(caught.exception.code, 2)


class MonitorReachesTheAddress(unittest.TestCase):
    """The monitor's client allowed plain HTTP to loopback only; a server on one address of this computer must be reachable there too
    (never through a proxy), and another computer's address still needs HTTPS."""

    def test_this_computers_address_over_http(self):
        from strata_cli import client
        with mock.patch.object(acc, "local_addresses", return_value=HERE + ["2001:db8::5"]):
            self.assertEqual(client.server_url("http://192.168.2.100:8080"), "http://192.168.2.100:8080")
            self.assertEqual(client.server_url("http://[2001:db8::5]:8080"), "http://[2001:db8::5]:8080")
            self.assertTrue(client.this_computer("10.8.0.2"))
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                client.server_url("http://192.168.7.7:8080")
            self.assertEqual(client.server_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080")


class LocalReaders(unittest.TestCase):
    def test_programs_on_this_computer_read_the_one_address(self):
        s = S.LaunchSettings(access="address", listen="192.168.2.100", port=8081)
        self.assertEqual(acc.local_url(s), "http://192.168.2.100:8081")
        self.assertEqual(acc.local_url(s.copy(access="lan")), "http://127.0.0.1:8081")
        self.assertEqual(acc.host_url("192.168.2.100", 8081), "http://192.168.2.100:8081")
        self.assertEqual(acc.host_url("0.0.0.0", 8081), "http://127.0.0.1:8081")
        self.assertEqual(acc.host_url("2001:db8::5", 8081), "http://[2001:db8::5]:8081")


class ListenFlag(unittest.TestCase):
    def parse(self, *argv):
        from strata_cli.launch import cmd
        ap = argparse.ArgumentParser()
        cmd.add_parsers(ap.add_subparsers(dest="command"))
        return ap.parse_args(["launch", *argv])

    def test_listen_is_the_one_address_way(self):
        from strata_cli.launch import cmd
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home():
            s, _ = cmd.settings_from_args(self.parse("--model", "Q2_0", "--listen", "10.8.0.2"), catalog=catalog(HW_12))
        self.assertEqual((s.access, s.listen), ("address", "10.8.0.2"))

    def test_access_address_alone_takes_this_computers_first_address(self):
        from strata_cli.launch import cmd
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home(), mock.patch.object(acc, "local_addresses", return_value=HERE):
            s, _ = cmd.settings_from_args(self.parse("--model", "Q2_0", "--access", "address"), catalog=catalog(HW_12))
        self.assertEqual((s.access, s.listen), ("address", "192.168.1.100"))


class AccessStepRow(Base):
    def test_enter_goes_through_the_addresses_and_n_goes_on(self):
        async def case(app, pilot, ctx):
            with mock.patch.object(acc, "local_addresses", return_value=HERE):
                await press(pilot, *(("enter",) * 7 + ("n",)))
                self.assertEqual(app.current, "access")
                app.screen.menu().select("address")
                app.screen.on_cursor()
                app.screen.build()
                app.screen.paint()
                await press(pilot)
                self.assertEqual((ctx.settings.access, ctx.settings.listen), ("address", "192.168.1.100"))
                self.assertIn("One address only", text_of(app))
                await press(pilot, "enter")
                self.assertEqual(ctx.settings.listen, "192.168.2.100")
                self.assertIn("http://192.168.2.100:", text_of(app))
                self.assertEqual(app.current, "access")
                await press(pilot, "n")
                self.assertEqual(app.current, "save")
                self.assertTrue(ctx.settings.api_key)                         # beyond this computer: a key, as for the local network
        self.run_app(case)


if __name__ == "__main__":
    unittest.main()
