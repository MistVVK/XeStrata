"""The model screen of `strata launch` with models from the signed feed: a model this Strata does not know is named under the list with the plain reason, and a feed
without models changes nothing on the screen. No engine is needed: the list comes from the cached feed in the test's own components folder (a throwaway key)."""

from __future__ import annotations

import base64
import json
import os
import unittest
from unittest import mock

from strata_control import feed as feedmod
from strata_control import feed_models, signing
from tests.terminal.launch_support import text_of
from tests.terminal.test_launch_tui import Base, make_ctx, press

REV = "0123456789abcdef0123456789abcdef01234567"


def entry(**kw):
    m = {"id": "later-one", "title": "Later One", "by": "Someone", "licence": "Apache 2.0", "licence_url": "https://example.org/l", "family": "qwen5new",
         "repo": "test/Later-GGUF", "revision": REV, "quants": [{"name": "Q4", "ram_gb": 24, "files": [{"name": "later-Q4.gguf", "size": 10, "sha256": "a" * 64}]}]}
    m.update(kw)
    return m


class FeedModelsOnTheModelScreen(Base):
    def setUp(self):
        super().setUp()
        self.key = signing.generate_key()
        keys = os.path.join(self.home.tmp.name, "keys.json")
        with open(keys, "w", encoding="utf-8") as f:
            json.dump({"keys": [signing.public_part(self.key)]}, f)
        p = mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": keys})
        p.start()
        self.addCleanup(p.stop)

    def keep_feed(self, models):
        doc = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 3, "generated_at": "2026-11-06T00:00:00Z", "expires_at": None}
        if models is not None:
            doc["models"] = {"packages": models}
        raw = (json.dumps(doc, sort_keys=True) + "\n").encode()
        sig = json.dumps(signing.sign_bytes(self.key, raw)).encode()
        feed_models.remember(_with_signature(raw, sig, self.key))

    def test_a_model_this_strata_does_not_know_is_named_with_the_reason(self):
        self.keep_feed([entry()])

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            t = " ".join(text_of(app).split())
            self.assertIn("Which model?", t)
            self.assertIn("Later One: " + feed_models.UNAVAILABLE_UNKNOWN, t)
            self.assertNotIn("qwen5new", t)                                  # the family name is for the log, not for people
        self.run_app(case, ctx=make_ctx())

    def test_a_feed_without_models_changes_nothing(self):
        self.keep_feed(None)

        async def case(app, pilot, ctx):
            await press(pilot, "enter")
            t = text_of(app)
            self.assertNotIn("update Strata to use it", t)
            self.assertIn("Which model?", t)
        self.run_app(case, ctx=make_ctx())


def _with_signature(raw: bytes, sig: bytes, key: dict):
    import dataclasses
    feed = feedmod.verify_and_parse(raw, sig, source="x", trusted={key["key_id"]: base64.b64decode(key["public"])})
    return dataclasses.replace(feed, signature=sig)


if __name__ == "__main__":
    unittest.main()
