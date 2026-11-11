"""Lane NET item 3: packaging/install.sh's fetch() under the network rule, against local servers: a TLS server that never answers its handshake
(given up after the 15 s connect limit and one retry, about 32 s, not 10 minutes), a refused port (at once), a working server, and a slow link that
keeps sending (never cut; real-time, STRATA_SLOW_NET_TESTS=1). curl and POSIX sh are needed; the Windows script is item 6."""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.terminal.test_netpolicy import BlackHole, Server, ok_body, refused_port, trickle

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "packaging" / "install.sh"


def curl_fetch_line() -> str:
    m = re.search(r"^\s*fetch\(\) \{ curl [^\n]*\}", SH.read_text(encoding="utf-8"), re.M)
    assert m, "install.sh has no curl fetch()"
    return m.group(0).strip()


@unittest.skipIf(os.name == "nt" or not (shutil.which("sh") and shutil.which("curl")), "needs POSIX sh and curl")
class InstallFetch(unittest.TestCase):
    def fetch(self, url, timeout=120):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "got"
            t0 = time.monotonic()
            env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
            r = subprocess.run(["sh", "-c", curl_fetch_line() + '\nfetch "$1" "$2"', "sh", url, str(out)], capture_output=True, text=True,
                               timeout=timeout, env=env)
            return r.returncode, (out.read_bytes() if out.exists() else b""), time.monotonic() - t0, r.stderr

    def test_the_limits_are_in_the_script(self):
        line = curl_fetch_line()
        for flag in ("--connect-timeout 15", "--speed-limit 1", "--speed-time 60", "--retry 1"):
            self.assertIn(flag, line)
        self.assertNotIn("--max-time", line)                 # no total cap: a slow download that works is not cut
        self.assertNotIn("--retry-connrefused", line)        # a refused connection fails at once

    def test_a_handshake_that_never_answers_is_given_up_in_about_half_a_minute(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        code, _got, took, err = self.fetch(f"https://127.0.0.1:{hole.port}/install-manifest.txt")
        self.assertNotEqual(code, 0)
        self.assertGreater(took, 25)
        self.assertLess(took, 45, err)                       # 15 + 2 + 15 s, not 10 minutes
        self.assertEqual(len(hole.held), 2)

    def test_a_refused_port_fails_at_once(self):
        code, _got, took, _err = self.fetch(f"http://127.0.0.1:{refused_port()}/install-manifest.txt")
        self.assertNotEqual(code, 0)
        self.assertLess(took, 3)

    def test_a_working_server(self):
        srv = Server(ok_body(b"linux-x86_64 0.1 a b 1\n"))
        self.addCleanup(srv.close)
        code, got, _took, err = self.fetch(srv.url("/install-manifest.txt"))
        self.assertEqual((code, got), (0, b"linux-x86_64 0.1 a b 1\n"), err)

    @unittest.skipUnless(os.environ.get("STRATA_SLOW_NET_TESTS") == "1", "real-time: STRATA_SLOW_NET_TESTS=1")
    def test_a_slow_link_that_keeps_sending_is_not_cut(self):
        srv = Server(trickle(150, 0.5))                      # 2 bytes a second for 75 s: past the 60 s window, above 1 byte/s
        self.addCleanup(srv.close)
        code, got, took, err = self.fetch(srv.url("/terminal.tar.gz"), timeout=200)
        self.assertEqual((code, len(got)), (0, 150), err)
        self.assertGreater(took, 70)


class DocumentedLines(unittest.TestCase):
    """W12 and the Windows line (owner 11-10): the documented install lines carry their limits; no copy of the old lines is left (old release notes
    are history and keep theirs)."""
    OLD = ("curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh",
           "curl -fsSLO https://github.com/Niko1221/Strata/releases/latest/download/install.sh",
           "irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex")

    def files(self):
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix.lower() not in (".md", ".ps1", ".sh", ".txt", ".json", ".ts", ".tsx", ".mjs", ".py", ".html") or not p.is_file():
                continue
            if rel.startswith(("node_modules/", ".git/", "history/")) or "/node_modules/" in rel or "release-notes-" in p.name:
                continue
            if p.resolve() == Path(__file__).resolve():
                continue
            yield rel, p

    def test_no_old_line_is_left(self):
        left = []
        for rel, p in self.files():
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            left += [rel for old in self.OLD if old in text]
        self.assertEqual(left, [])

    def test_the_readme_lines(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh "
                      "-o install.sh && sh install.sh", text)
        self.assertIn("irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex", text)


class InstallPs1Text(unittest.TestCase):
    """Item 6: packaging/install.ps1 under the same rule (its real run is on Windows, through the coordinator)."""

    def setUp(self):
        self.ps1 = (ROOT / "packaging" / "install.ps1").read_text(encoding="utf-8")

    def test_every_download_goes_through_fetch_with_the_limits(self):
        self.assertNotIn("Invoke-WebRequest", self.ps1)
        self.assertNotIn("Invoke-RestMethod", self.ps1)
        self.assertEqual(len(re.findall(r"\bFetch \"\$feed/", self.ps1)), 3)
        for text in ("$req.Timeout = 15000", "$req.ReadWriteTimeout = 60000", "Start-Sleep -Seconds 2", "$try -lt 2 -and $timedOut",
                     "[Net.WebExceptionStatus]::Timeout", "$ProgressPreference = 'SilentlyContinue'", "DefaultNetworkCredentials"):
            self.assertIn(text, self.ps1)
        self.assertNotIn("TimeoutSec", self.ps1.split("& {", 1)[1])     # no total cap inside the script (PowerShell 7 would cut a slow download)

    def test_the_messages(self):
        self.assertIn("did not answer (or this computer is offline), so the installer could not start. Check that this computer can open "
                      "https://$feedHost, then run the same line again. Nothing was installed.", self.ps1)
        self.assertIn("the download stopped: $feedHost did not answer, or no data came for 60 seconds. Run the same line again; nothing was installed.",
                      self.ps1)


if __name__ == "__main__":
    unittest.main()
