# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""setup --image-tools fetches only the control networks the configured image families can take: Qwen-Image's
ControlNet refuses normal and segmentation maps, a family that takes them (MOSS-Image's control LoRAs, the reference
families) still has them fetched.  The decision is checked against image_controls.mode_of, which the requests use.

    python -m unittest serve.test_image_aux
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from serve import image_aux, image_controls

BASE = ("seedvr2-7b-int8", "seedvr2-vae", "realesrgan", "yolo-face", "yolo-hand", "yolo-person",
        "florence-2-large", "florence-2-base", "florence-2-tokenizer", "birefnet-general", "birefnet-lite")


class SetupNames(unittest.TestCase):
    def names(self, *families: str, accepted: bool = False) -> list[str]:
        with tempfile.TemporaryDirectory() as root:
            if accepted:
                image_aux.accept("sam3", Path(root), "test")
            return image_aux.setup_names(8192, 32768, Path(root), families)

    def networks(self, names: list[str]) -> list[str]:
        q = image_aux._qimage()
        all_nets = {w for ws in q.CNET_KINDS.values() for w in ws}
        return [n for n in names if n in all_nets]

    def all_networks(self) -> list[str]:
        q = image_aux._qimage()
        return list(dict.fromkeys(w for ws in q.CNET_KINDS.values() for w in ws))

    def test_qwen_image_only_skips_the_networks_its_controlnet_refuses(self):
        got = self.names("qwen-image")
        q = image_aux._qimage()
        self.assertEqual(got[:len(BASE)], list(BASE))   # the other tools' weights are unchanged
        expected = list(dict.fromkeys(w for k, ws in q.CNET_KINDS.items()
                                      if k not in image_aux.CONTROLNET_REFUSED_KINDS for w in ws))
        self.assertEqual(self.networks(got), expected)
        self.assertIn("cnet-union", got)
        self.assertNotIn("cnet-normal", got)
        self.assertNotIn("cnet-segmentation", got)
        self.assertFalse(any(n.startswith("moss-control") for n in got))

    def test_consented_weights_are_still_fetched(self):
        got = self.names("qwen-image", accepted=True)
        self.assertIn("sam3", got)
        self.assertIn("clip-tokenizer", got)

    def test_moss_image_keeps_all_networks_and_its_own_weights(self):
        got = self.names("moss-image")
        q = image_aux._qimage()
        self.assertEqual(self.networks(got), self.all_networks())
        caps = q.family_capabilities("moss-image")
        self.assertIn("moss-control", got)
        for k in sorted(set(caps["control_loras"].values())):
            self.assertIn(f"moss-control-{k}", got)
        self.assertNotIn("cnet-union", got)

    def test_reference_families_keep_all_networks(self):
        for family in ("mage-flow", "hunyuan-image-3"):
            with self.subTest(family=family):
                got = self.names(family)
                self.assertEqual(self.networks(got), self.all_networks())
                self.assertNotIn("cnet-union", got)
                self.assertFalse(any(n.startswith("moss-control") for n in got))

    def test_mixed_families_union_the_networks(self):
        got = self.names("qwen-image", "moss-image")
        self.assertEqual(self.networks(got), self.all_networks())
        self.assertIn("cnet-union", got)
        self.assertIn("moss-control", got)

    def test_networks_match_the_kinds_the_requests_take(self):
        q = image_aux._qimage()
        # Editable configs: a config without vision or an edit adapter takes no reference or LoRA control.
        cfgs = {"qwen-image": {"family": "qwen-image", "vision": "v"},
                "moss-image": {"family": "moss-image", "edit_lora": "l"},
                "mage-flow": {"family": "mage-flow", "vision": "v", "edit_model": "m"},
                "hunyuan-image-3": {"family": "hunyuan-image-3", "vision": "v"}}
        for family, cfg in cfgs.items():
            got = self.names(family)
            for kind, nets in q.CNET_KINDS.items():
                if not nets:   # this kind needs no network, fetched or not
                    continue
                takes = image_controls.mode_of(cfg, kind) is not None
                self.assertEqual(takes, any(n in got for n in nets), (family, kind))

    def test_family_without_control_fetches_no_control_networks(self):
        self.assertEqual(self.networks(self.names("anima")), [])


if __name__ == "__main__":
    unittest.main()
