# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_kept_links.py - a chain's next request names the picture its previous result kept.

The records' obstacles X1 and X2: a link this server handed out (`/v1/images/files/<name>`, relative in a chat tool's
result, absolute in an endpoint's `url`) was not an input another endpoint took - the relative one was read as a path of
this computer, the absolute one the #28 guard refused as a private address.  Both are now read from the store that kept
them; every other source is read as it was.

    python -m unittest serve.test_kept_links -v
"""
from __future__ import annotations

import base64
import io
import secrets
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import images as images_api        # noqa: E402
from serve.server import make_handler         # noqa: E402

OUTSIDE = b"outside"


def png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 255)).save(buf, format="PNG")
    return buf.getvalue()


PNG = png_bytes()
NAME = "0" * 32 + ".png"
LINK = f"/v1/images/files/{NAME}"


def reader(svc, host: str = "127.0.0.1:8123"):
    """A request reader of a server built on `svc`, with only its Host header set: reading an image source asks for
    nothing else of the connection."""
    handler = make_handler(svc)
    h = handler.__new__(handler)
    h.headers = {"Host": host} if host else {}
    return h


def store_svc(files: dict[str, bytes], download=None):
    """A service whose image store holds `files`, and what its outside reader was asked for."""
    root = Path(tempfile.mkdtemp())
    for name, data in files.items():
        (root / name).write_bytes(data)
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        return OUTSIDE

    images = SimpleNamespace(dir=root, file=lambda name: (root / name) if name in files else None,
                             gens={}, default=None)
    svc = SimpleNamespace(images=images, public_urls=True, download=download or fetch)
    return svc, seen


class KeptLinkNames(unittest.TestCase):
    """images.kept_link: which spellings name one of this server's own kept pictures."""

    def names(self, source, host: str = "127.0.0.1:8123"):
        return images_api.kept_link(source, host)

    def test_the_two_spellings_of_a_result_link(self):
        self.assertEqual(self.names(LINK), NAME)
        for scheme in ("http", "https"):
            self.assertEqual(self.names(f"{scheme}://127.0.0.1:8123{LINK}"), NAME)
        # behind a proxy the answer's link carries the public name, which is the request's own Host
        self.assertEqual(self.names(f"https://images.example{LINK}", "images.example"), NAME)
        self.assertEqual(self.names(f"HTTP://127.0.0.1:8123{LINK}".lower()), NAME)

    def test_another_computer_is_not_the_store(self):
        for host in ("127.0.0.1:9999", "example:8123", "10.0.0.5:8123", ""):
            self.assertIsNone(self.names(f"http://127.0.0.1:8123{LINK}", host), host)

    def test_only_a_kept_image_path(self):
        for source in (NAME, f"{NAME}.png", f"/v1/images/files/{secrets.token_hex(16)}.tiff",
                       f"{LINK}?x=1", f"{LINK}#frag", "/v1/images/files/" + "z" * 32 + ".png",
                       f"/v1/images/files/sub/{NAME}", f"data:image/png;base64,{NAME}",
                       f"http://127.0.0.1:8123/v1/images/{NAME}", "", "/v1/images/files/"):
            self.assertIsNone(self.names(source), source)
        for not_a_string in (None, 7, {"image_url": LINK}):
            self.assertIsNone(self.names(not_a_string), not_a_string)


class LoadKeptLink(unittest.TestCase):
    """_load_image_source: a link to a kept picture comes from the store; every other source is read as it was."""

    def reads(self, source, host: str = "127.0.0.1:8123", download=None):
        svc, seen = store_svc({NAME: PNG}, download)
        return reader(svc, host)._load_image_source(source), seen

    def test_relative_link_of_a_kept_picture(self):
        got, seen = self.reads(LINK)
        self.assertEqual((got, seen), (PNG, []))

    def test_absolute_link_of_the_request_host(self):
        for source in (f"http://127.0.0.1:8123{LINK}", f"https://127.0.0.1:8123{LINK}"):
            with self.subTest(source=source):
                got, seen = self.reads(source)
                self.assertEqual((got, seen), (PNG, []))

    def test_a_link_names_nothing_kept(self):
        with self.assertRaisesRegex(ValueError, "no picture kept here"):
            self.reads("/v1/images/files/" + "1" * 32 + ".png")

    def test_another_host_still_goes_to_the_outside_reader(self):
        got, seen = self.reads(f"http://127.0.0.1:9999{LINK}")
        self.assertEqual((got, seen), (OUTSIDE, [f"http://127.0.0.1:9999{LINK}"]))

    def test_a_data_url_and_a_path_are_as_they_were(self):
        inline = "data:image/png;base64," + base64.b64encode(PNG).decode()
        self.assertEqual(self.reads(inline), (PNG, []))
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(PNG)
            f.flush()
            self.assertEqual(self.reads(f.name)[0], PNG)

    def test_the_private_address_guard_is_untouched(self):
        """#28: a running server reads an image URL only from a public address.  A link of its own Host now comes from
        the store, so there is nothing to refuse there; any other URL of a private address is still refused."""
        def refuse(url: str) -> bytes:
            raise ValueError("the image URL points to a private address (127.0.0.1): the server does not fetch from "
                             "its own network for a request; XESTRATA_IMAGE_URL_PRIVATE=1 allows it")
        self.assertEqual(self.reads(LINK, download=refuse), (PNG, []))
        for source in (f"http://127.0.0.1:8123{LINK}?x=1", f"http://127.0.0.1:9999{LINK}"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "private address"):
                self.reads(source, download=refuse)


class KeptMaskForms(unittest.TestCase):
    """check_mask: the mask a chain's step before gives - this server's own grey PNG, white where it changes - is read
    as OpenAI's transparent one (the records' X3, X4); OpenAI's form is read and checked as it was."""

    def grey(self, size: tuple[int, int] = (16, 32), box: tuple[int, int, int, int] = (4, 2, 9, 12)) -> bytes:
        """segmentations' mask_format png (image_regions.mask_png): a single-channel PNG, 255 inside."""
        import base64
        import numpy as np
        from serve import image_regions as regions_api
        inside = np.zeros((size[1], size[0]), dtype=bool)
        inside[box[1]:box[3], box[0]:box[2]] = True
        return base64.b64decode(regions_api.mask_png(inside))

    def alpha(self, size: tuple[int, int] = (16, 32), box: tuple[int, int, int, int] = (4, 2, 9, 12)) -> bytes:
        """OpenAI's mask: an RGBA PNG, transparent where it changes."""
        from PIL import Image
        im = Image.new("RGBA", size, (0, 0, 0, 255))
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                im.putpixel((x, y), (0, 0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def alpha_of(self, data: bytes):
        from PIL import Image
        return Image.open(io.BytesIO(data)).getchannel("A")

    def test_the_grey_mask_is_read_as_transparency(self):
        got = images_api.check_mask(self.grey(), (16, 32))
        a = self.alpha_of(got)
        self.assertEqual(a.getpixel((5, 5)), 0)          # white inside: where it changes
        self.assertEqual(a.getpixel((14, 30)), 255)      # black elsewhere: kept
        self.assertEqual(a.getpixel((5, 5)), self.alpha_of(self.alpha()).getpixel((5, 5)))

    def test_the_chat_tools_mask_link_is_the_same_form(self):
        grey = images_api.mask_to_bw(self.alpha())
        self.assertEqual(self.alpha_of(images_api.check_mask(grey, (16, 32))).getpixel((5, 5)), 0)

    def test_a_grey_mask_at_another_size_is_fitted(self):
        from PIL import Image
        got = Image.open(io.BytesIO(images_api.check_mask(self.grey(), (32, 64))))
        self.assertEqual(got.size, (32, 64))
        self.assertEqual(got.getchannel("A").getpixel((10, 10)), 0)
        self.assertEqual(got.getchannel("A").getpixel((30, 62)), 255)

    def test_openai_s_mask_is_as_it_was(self):
        same = self.alpha()
        kept = images_api.check_mask(same, (16, 32))
        self.assertEqual(self.alpha_of(kept).getpixel((5, 5)), 0)             # transparent: where it changes
        self.assertEqual(self.alpha_of(kept).getpixel((14, 30)), 255)         # opaque: kept
        with self.assertRaisesRegex(images_api.ImageRequestError, r"^mask: 16x32, not the image's 8x8$"):
            images_api.check_mask(same, (8, 8))                               # OpenAI's own size rule
        with self.assertRaisesRegex(images_api.ImageRequestError, r"^mask: a PNG with an alpha channel$"):
            images_api.check_mask(b"not a png", (16, 32))


if __name__ == "__main__":
    unittest.main()
