"""A DFlash2 drafter pack that the signed feed points at an EARLIER release by an absolute url (0.2.17: the two packs are unchanged since 0.2.16, so the feed names
``releases/download/v0.2.16/<name>`` and the new release does not carry them again, 862 MB less).  No network: the feed is signed here with a throwaway key and
parsed as a client reads it from an https address; the download is replaced by a function that records the address it was asked for.

What is pinned:
  * the drafter entry keeps its absolute url through feed_tool (``pack-drafter --url-prefix``), the signature and the parser, and ``Feed.url`` resolves it to itself
    (a relative name still resolves against the feed's own release);
  * ``tuned_from_feed`` and ``ensure_tuned`` follow that address, check the pinned size and SHA-256, and place the file;
  * an address that is not https (a plain http host, a file) is refused by the url policy, so a feed cannot send the drafter anywhere that is not https.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))

import feed_tool as ft  # noqa: E402
from strata_control import drafting, drafting_fetch, signing, transfer  # noqa: E402
from strata_control import feed as feedmod  # noqa: E402

FEED_URL = "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json"
EARLIER = "https://github.com/Niko1221/Strata/releases/download/v0.2.16/"
PACK = b"drafter-pack-" * 50_000                  # stands in for the 340 MB file
NAME = drafting.TUNED_NAME


class AbsoluteDrafterUrl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "dflash2").mkdir(parents=True)
        pins = mock.patch.multiple(drafting, TUNED_BYTES=len(PACK), TUNED_SHA256=hashlib.sha256(PACK).hexdigest())
        pins.start()
        self.addCleanup(pins.stop)
        env = mock.patch.dict("os.environ", {"STRATA_COMPONENTS_DIR": str(self.root / "components")})
        env.start()
        self.addCleanup(env.stop)
        self.keys = self.root / "keys"
        ft.keygen(self.keys, allow_git=True)
        (self.root / "trust.json").write_text((self.keys / "feed-key.pub.json").read_text(encoding="utf-8"), encoding="utf-8")

    def feed(self, url_prefix: str):
        """The release folder a build would write, parsed the way a client parses the file it fetched from FEED_URL."""
        host = self.root / "host"
        host.mkdir(exist_ok=True)
        (host / NAME).write_bytes(PACK)
        entries = self.root / "entries"
        ft.pack_drafter(host / NAME, entries, version="1", backends=["metal"], url_prefix=url_prefix)
        out = self.root / "rel"
        ft.build_feed(out, key=ft._load_key(self.keys / "feed-key.json"), sequence=129, drafters_dir=entries)
        payload = (out / feedmod.FEED_NAME).read_bytes()
        signature = (out / (feedmod.FEED_NAME + ".sig")).read_bytes()
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_TRUSTED_KEYS": str(self.root / "trust.json")}):
            trusted = signing.load_trusted_keys()
            return feedmod.verify_and_parse(payload, signature, source=FEED_URL, trusted=trusted)

    def test_the_entry_keeps_its_absolute_url_and_the_client_resolves_it_to_itself(self):
        loaded = self.feed(EARLIER)
        (entry,) = loaded.drafters
        self.assertEqual(entry.url, EARLIER + NAME)
        self.assertEqual(loaded.url(entry.url), EARLIER + NAME)
        self.assertEqual((entry.size, entry.sha256), (len(PACK), hashlib.sha256(PACK).hexdigest()))

    def test_a_flat_name_still_means_the_release_that_serves_the_feed(self):
        loaded = self.feed("")
        (entry,) = loaded.drafters
        self.assertEqual(entry.url, NAME)
        self.assertEqual(loaded.url(entry.url), "https://github.com/Niko1221/Strata/releases/latest/download/" + NAME)

    def test_the_lookup_and_the_fetch_follow_the_absolute_url(self):
        loaded = self.feed(EARLIER)
        asked: list[str] = []

        def fake_download(url, partial, **kw):
            asked.append(url)
            Path(partial).write_bytes(PACK)

        env = {"STRATA_UPDATE_TRUSTED_KEYS": str(self.root / "trust.json"), "STRATA_UPDATE_FEED": FEED_URL}
        with mock.patch.dict("os.environ", env), mock.patch.object(feedmod, "fetch", return_value=loaded), \
                mock.patch.object(transfer, "download", side_effect=fake_download):
            found = drafting_fetch.tuned_from_feed(backend="metal")
            self.assertEqual(found, (EARLIER + NAME, len(PACK), hashlib.sha256(PACK).hexdigest()))
            out = drafting_fetch.ensure_tuned(self.data, backend="metal")
        self.assertEqual(asked, [EARLIER + NAME])
        self.assertEqual(out, drafting.tuned_path(self.data))
        self.assertEqual(out.read_bytes(), PACK)
        self.assertTrue(drafting.tuned_ok(self.data))

    def test_an_address_that_is_not_https_is_refused(self):
        for bad in ("http://example.org/v0.2.16/" + NAME, "file:///c:/x/" + NAME, "ftp://example.org/" + NAME):
            with self.assertRaises(transfer.TransferError, msg=bad):
                transfer.resolve_ref(transfer.base_of(FEED_URL), bad)


if __name__ == "__main__":
    unittest.main()
