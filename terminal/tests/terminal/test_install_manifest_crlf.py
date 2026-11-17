"""The release's list files read the same with CRLF line ends. 0.2.15's publish scripts wrote UPLOAD-ORDER.txt with CRLF on Windows and every
upload of that run failed on the names' trailing CR (fixed in the release scripts, which now write LF). The one list file the released tools read is
install-manifest.txt: packaging/install.sh takes its platform line with awk and uses the last field (the size) in shell arithmetic, so a manifest
saved with CRLF (an editor, a copy through Windows) stopped the install there. install.ps1 reads it with Get-Content, which takes either ending.
packaging/feed_tool.py writes it with LF."""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT.parent / "tools" / "packaging" / "install.sh"
MANIFEST = ("linux-x86_64 0.2.15 strata-control-0.1.61-linux-x86_64.tar.gz " + "a" * 64 + " 97550582\n"
            "macos-arm64 0.2.15 strata-control-0.1.61-macos-arm64.tar.gz " + "b" * 64 + " 48715939\n")


def manifest_line() -> str:
    m = re.search(r"^\s*line=\$\(awk [^\n]*manifest\.txt\"\)", SH.read_text(encoding="utf-8"), re.M)
    assert m, "install.sh reads no platform line from install-manifest.txt"
    return m.group(0).strip()


@unittest.skipIf(os.name == "nt" or not shutil.which("sh") or not shutil.which("awk"), "needs POSIX sh and awk")
class ManifestLineEnds(unittest.TestCase):
    def read(self, text: str, platform: str = "linux-x86_64") -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.txt").write_bytes(text.encode())
            script = (f"platform={platform}; work=\"$1\"\n{manifest_line()}\n"
                      'set -- $line\nprintf "%s|%s|%s|%s|%s\\n" "$1" "$2" "$3" "$4" "$(( ${5:-0} / 1048576 ))"\n')
            return subprocess.run(["sh", "-c", script, "sh", d], capture_output=True, text=True, timeout=30)

    def test_lf_and_crlf_give_the_same_fields(self):
        want = "linux-x86_64|0.2.15|strata-control-0.1.61-linux-x86_64.tar.gz|" + "a" * 64 + "|93\n"
        for text in (MANIFEST, MANIFEST.replace("\n", "\r\n")):
            r = self.read(text)
            self.assertEqual((r.returncode, r.stdout), (0, want), (repr(text[:20]), r.stderr))
        r = self.read(MANIFEST.replace("\n", "\r\n"), "macos-arm64")
        self.assertEqual(r.stdout.split("|")[3], "b" * 64)                  # the checksum compared with the download has no CR either


class Writers(unittest.TestCase):
    def test_feed_tool_writes_the_manifest_with_lf(self):
        src = (ROOT.parent / "tools" / "packaging" / "feed_tool.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"install-manifest\.txt"\)\.write_text\(install_manifest, encoding="utf-8", newline="\\n"\)')


if __name__ == "__main__":
    unittest.main()
