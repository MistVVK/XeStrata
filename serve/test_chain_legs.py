# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_chain_legs.py - the chains' legs at the REST entries, with the drawing itself stubbed.

An outside caller chains image tools by handing the link the previous result gave straight into the next request
(`chain-notes.md`'s F1, F2 of the records' X1-X4).  These run the real entries over HTTP - the reading of the input,
`parse_edit_request`'s job, the answer's link - with `make_images` stubbed, so no image model runs: the plumbing, not
the picture.  A server that is running reads a request's image URL only from a public address (#28), which is why the
second leg used to fail on the first leg's own link.

    python -m unittest serve.test_chain_legs -v
"""
from __future__ import annotations

import io
import json
import secrets
import sys
import tempfile
import unittest
from typing import Callable
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import image_regions as regions_api    # noqa: E402
from serve import images as images_api            # noqa: E402
from serve.frontend import ChatTemplate           # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "serve/test_chat_template.jinja"
CT = 4096
MADE = b"made"                                # what the stubbed drawing hands back
RED = (200, 30, 30, 255)


def png(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format="PNG")
    return buf.getvalue()


def pixels(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGBA")


class ChainLegs(unittest.TestCase):
    """The entries take a kept picture's link (relative or absolute) and a grey mask, and build the job from them."""

    store: Path
    name: str
    picture: bytes
    gen: SimpleNamespace
    svc: Service
    port: int
    jobs: list[dict]
    shutdown_server: Callable[[], None]
    close_server: Callable[[], None]

    @classmethod
    def setUpClass(cls):
        cls.store = Path(tempfile.mkdtemp())
        cls.name = secrets.token_hex(16) + ".png"
        cls.picture = png((256, 256), RED)
        (cls.store / cls.name).write_bytes(cls.picture)
        cls.gen = SimpleNamespace(cfg={"max_side": 2048, "model_name": "chain-test"}, model="chain-test",
                                  engine=SimpleNamespace(alive=lambda: True))

        def keep(data: bytes, fmt: str = "png") -> str:
            made = secrets.token_hex(16) + "." + fmt
            (cls.store / made).write_bytes(data)
            return made
        cls.gen.store = keep
        cls.gen.response = lambda images, job, figures, base: {
            "created": 0, "data": [{"url": f"{base}/v1/images/files/{keep(images[0])}"}]}
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "xy" * 1000, max_context=CT), tok, ChatTemplate(TEMPLATE))
        cls.svc.public_urls = True                       # as a running server's main() sets it (#28)
        cls.svc.images = SimpleNamespace(gens={cls.gen.model: cls.gen}, default=cls.gen,
                                         pick=lambda m: cls.gen, pick_edit=lambda m, e: cls.gen,
                                         file=lambda n: (cls.store / n) if (cls.store / n).is_file() else None)
        httpd = serve(cls.svc, port=0)
        cls.port = httpd.server_address[1]
        cls.shutdown_server = httpd.shutdown
        cls.close_server = httpd.server_close
        cls.jobs = []

    @classmethod
    def tearDownClass(cls):
        cls.shutdown_server()
        cls.close_server()

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def leg(self, path: str, body: dict) -> tuple[int, dict, dict]:
        """One request with the drawing stubbed -> (the status, the body, the job that reached the drawing)."""
        self.jobs.clear()

        def made(svc, gen, job, cancel=None, preview=None):
            self.jobs.append(job)
            return [MADE], {"seconds": 0.1}, "keep"
        with (mock.patch.object(images_api, "make_images", made),
              mock.patch.object(images_api, "place", lambda s, g, gpu: (g, None))):
            status, answer = self.post(path, body)
        return status, answer, (self.jobs[-1] if self.jobs else {})

    def link(self, relative: bool = False, query: str = "") -> str:
        path = f"/v1/images/files/{self.name}"
        return path if relative else f"http://127.0.0.1:{self.port}{path}{query}"

    def test_absolute_link_is_the_picture_of_the_next_leg(self):
        """C4's second leg: remove_background's `url` (an absolute link of this server) into an extend edit."""
        status, answer, job = self.leg("/v1/images/edits", {"prompt": "wider", "image": self.link(),
                                                            "extend": {"aspect": "16:9"}})
        self.assertEqual(status, 200, answer)
        canvas = pixels(job["init"])
        self.assertEqual(canvas.height, 256)                 # the outpaint's canvas: 16:9 at the picture's height
        self.assertGreater(canvas.width, 256)
        self.assertEqual(canvas.getpixel((canvas.width // 2, 128)), RED)     # the kept picture, read from the store
        self.assertEqual(pixels(job["mask"]).getchannel("A").getpixel((2, 128)), 0)     # the new area changes
        self.assertIn("/v1/images/files/", answer["data"][0]["url"])

    def test_relative_link_is_the_picture_too(self):
        """C1's leg: the relative form a chat tool's result gives is an input at the REST entries as well."""
        status, answer, job = self.leg("/v1/images/edits", {"prompt": "", "image": self.link(relative=True),
                                                            "fix": {"parts": ["face"]}})
        self.assertEqual(status, 200, answer)
        self.assertEqual(job["fix"]["parts"], ["face"])
        self.assertEqual(job["picture"], self.picture)

    def test_upscale_takes_the_link(self):
        status, answer, job = self.leg("/v1/images/upscale", {"image": self.link(relative=True), "scale": 2})
        self.assertEqual(status, 200, answer)
        self.assertEqual((job["width"], job["height"], job["ups_only"]), (512, 512, True))

    def test_the_segmentation_mask_is_the_edit_mask(self):
        """C2's second leg: segmentations' `data[].mask` (a grey PNG, white inside) sent as the edit's mask."""
        import numpy as np
        inside = np.zeros((256, 256), dtype=bool)
        inside[10:60, 10:60] = True
        grey = "data:image/png;base64," + regions_api.mask_png(inside)
        status, answer, job = self.leg("/v1/images/edits", {"prompt": "p", "image": self.link(), "mask": grey,
                                                            "size": "256x256"})
        self.assertEqual(status, 200, answer)
        alpha = pixels(job["mask"]).getchannel("A")
        self.assertEqual(alpha.getpixel((30, 30)), 0)        # white: where it changes
        self.assertEqual(alpha.getpixel((200, 200)), 255)    # black: kept

    def test_a_url_of_another_port_or_with_a_query_is_still_the_public_reader(self):
        """#28 stays: only this request's own Host names a kept picture, and a link with a query is not one."""
        other = f"http://127.0.0.1:{self.port + 1}/v1/images/files/{self.name}"
        for source in (self.link(query="?x=1"), other):
            status, answer, _job = self.leg("/v1/images/edits", {"prompt": "p", "image": source})
            with self.subTest(source=source):
                self.assertEqual(status, 400, answer)
                self.assertIn("private address", answer["error"]["message"])

    def test_a_link_of_a_picture_no_longer_kept_is_refused(self):
        gone = f"http://127.0.0.1:{self.port}/v1/images/files/{secrets.token_hex(16)}.png"
        status, answer, _job = self.leg("/v1/images/edits", {"prompt": "p", "image": gone})
        self.assertEqual(status, 400, answer)
        self.assertIn("no picture kept here", answer["error"]["message"])


if __name__ == "__main__":
    unittest.main()
