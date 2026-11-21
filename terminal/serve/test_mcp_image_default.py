"""#2343: MCP tool images reach the model by default when the loaded model has a picture reader (the owner's rule: every
picture-capable model reads pictures by default). Without a reader they stay off; "image_results": false in the run config
keeps them off; true keeps them on. The limits stay as they were (4 per result, 24 MB of base64, PNG / JPEG / WebP)."""
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.mcp import hub_from_config, resolve_image_results

ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB0kAAAAASUVORK5CYII=")


class Default(unittest.TestCase):
    def test_not_set_follows_the_picture_reader(self):
        self.assertTrue(resolve_image_results({"image_results": None}, True))
        self.assertFalse(resolve_image_results({"image_results": None}, False))
        self.assertTrue(resolve_image_results({}, True))

    def test_an_explicit_value_is_respected(self):
        self.assertFalse(resolve_image_results({"image_results": False}, True))
        self.assertTrue(resolve_image_results({"image_results": True}, False))

    def tool_images(self, config: dict, has_reader: bool):
        hub = hub_from_config({"mcp_servers": {"fake": {"command": "unused"}}, "mcp": config})
        hub.settings["image_results"] = resolve_image_results(hub.settings, has_reader)
        server = hub.servers["fake"]
        server.status = "ready"
        server.tools = [{"name": "shot", "inputSchema": {}}]
        block = {"type": "image", "mimeType": "image/png", "data": base64.b64encode(PNG).decode()}
        with tempfile.TemporaryDirectory() as tmp:
            hub.media_dir = Path(tmp)
            with patch.object(server, "call", return_value={"content": [{"type": "text", "text": "a shot"}, block] * 3}):
                return hub.call("fake__shot", {})

    def test_a_model_with_a_reader_gets_the_pictures_without_any_setting(self):
        result = self.tool_images({}, has_reader=True)
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual([p["type"] for p in result["model_content"]], ["text", "image", "image", "image"])

    def test_without_a_reader_or_with_false_the_model_reads_text_only(self):
        for config, reader in (({}, False), ({"image_results": False}, True)):
            with self.subTest(config=config, reader=reader):
                result = self.tool_images(config, has_reader=reader)
                self.assertNotIn("images", result)
                self.assertNotIn("model_content", result)
                self.assertIn("not shown to the model", result["text"])

    def test_the_server_decides_it_after_the_picture_reader_is_known(self):
        src = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        at = src.index('hub.settings["image_results"] = resolve_image_results(hub.settings, svc.has_vision())')
        self.assertLess(at, src.index('if hub.settings["image_results"]:\n            hub.media_dir'))
        self.assertLess(src.index("svc = Service(engine, tok"), at)


if __name__ == "__main__":
    unittest.main()
