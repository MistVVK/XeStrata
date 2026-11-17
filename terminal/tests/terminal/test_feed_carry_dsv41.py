"""build-feed --carry-dsv41 (owner 11-11, "Keep frozen"): the last feed's DeepSeek engine packages go into the new feed as they were, with each
package URL made absolute to the release that holds the archive, so Strata 0.2.13 and earlier keep their DeepSeek engine; this Strata never
reads the member.  No network: the reader path an older Control takes is the entry schema (feed._engine_package, unchanged since 0.2.7) and
the reference resolver (transfer.resolve_ref), both run on the carried entry here."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))

import feed_tool  # noqa: E402
from strata_control import feed as feedmod, transfer  # noqa: E402

BASE = "https://github.com/Niko1221/Strata/releases/download/v0.2.13.1/"
FEED_URL = "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json"


def entry(url):
    return {"version": "0.1.59", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13", "archs": ["sm_86", "sm_89", "sm_120", "compute_120"],
            "min_driver": {"nvidia": "580.65"}, "min_runtime": None, "build_id": "b", "format": "tar.gz", "url": url, "size": 1000,
            "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 2000, "notes": ""}


def previous(folder: Path, packages) -> Path:
    p = folder / "strata-feed.json"
    p.write_text(json.dumps({"schema_version": 1, "feed": "strata-updates", "sequence": 122, "engine": {"packages": []},
                             "dsv41": {"packages": packages}}), encoding="utf-8")
    return p


class Carry(unittest.TestCase):
    def test_relative_urls_become_absolute_and_the_rest_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            flat = entry("strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz")
            local = entry("dsv41/strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz")
            absolute = entry("https://example.org/x/strata-dsv41-0.1.59-macos-arm64-metal-folder.tar.gz")
            member = feed_tool.carried_dsv41(previous(Path(d), [flat, local, absolute]), BASE)
        urls = [p["url"] for p in member["packages"]]
        self.assertEqual(urls, [BASE + "strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz"] * 2 + [absolute["url"]])
        for got, was in zip(member["packages"], (flat, local, absolute)):
            self.assertEqual({k: v for k, v in got.items() if k != "url"}, {k: v for k, v in was.items() if k != "url"})

    def test_the_placeholder_or_a_bad_base_stops_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            prev = previous(Path(d), [entry("a.tar.gz")])
            for base in (feed_tool.DSV41_BASE_EXAMPLE, "http://example.org/r/", "https://example.org/r", "", None):
                with self.assertRaises(SystemExit, msg=repr(base)):
                    feed_tool.carried_dsv41(prev, base)
            empty = Path(d) / "empty.json"
            empty.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                feed_tool.carried_dsv41(empty, BASE)

    def test_build_feed_needs_a_choice(self):
        with tempfile.TemporaryDirectory() as d:
            feed_tool.keygen(Path(d) / "keys")
            args = ["build-feed", "--out", str(Path(d) / "rel"), "--key", str(Path(d) / "keys" / "feed-key.json"), "--sequence", "123"]
            for extra in ([], ["--no-dsv41-carry", "--carry-dsv41", str(previous(Path(d), [entry("a.tar.gz")]))]):
                with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    feed_tool.main(args + extra)
            with redirect_stdout(io.StringIO()):
                feed_tool.main(args + ["--carry-dsv41", str(previous(Path(d), [entry("a.tar.gz")])), "--carry-dsv41-base", BASE])
            doc = json.loads((Path(d) / "rel" / feedmod.FEED_NAME).read_text(encoding="utf-8"))
            self.assertEqual(doc["dsv41"]["packages"][0]["url"], BASE + "a.tar.gz")
            with redirect_stdout(io.StringIO()):
                feed_tool.main(args[:2] + [str(Path(d) / "rel2")] + args[3:] + ["--no-dsv41-carry"])
            self.assertNotIn("dsv41", json.loads((Path(d) / "rel2" / feedmod.FEED_NAME).read_text(encoding="utf-8")))


class OlderControlReadsIt(unittest.TestCase):
    """What a 0.2.13 Control does with the member: each package through the engine entry schema, then its URL through resolve_ref
    against the feed's own address (an absolute https URL is taken as it is: no fetch here)."""

    def test_the_carried_entry_is_a_valid_package_with_an_absolute_https_url(self):
        with tempfile.TemporaryDirectory() as d:
            member = feed_tool.carried_dsv41(previous(Path(d), [entry("strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz")]), BASE)
        item = member["packages"][0]
        skipped: list = []
        pkg = feedmod._engine_package(item, skipped)
        self.assertIsNotNone(pkg, skipped)
        self.assertEqual(skipped, [])
        base = transfer.base_of(FEED_URL)
        self.assertEqual(transfer.resolve_ref(base, pkg.url), BASE + "strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz")
        # the relative form would have pointed at the NEW release, which does not hold the archive
        self.assertTrue(transfer.resolve_ref(base, "strata-dsv41-0.1.59-linux-x86_64-cuda-cuda13.tar.gz").startswith(
            "https://github.com/Niko1221/Strata/releases/latest/download/"))

    def test_this_control_ignores_the_member(self):
        self.assertFalse(hasattr(feedmod.Feed, "dsv41"))


if __name__ == "__main__":
    unittest.main()
