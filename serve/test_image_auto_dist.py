# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""When gpu_policy is auto and nothing names a GPU, an image job goes to an unused card that has room for its NEED.

    python -m unittest serve.test_image_auto_dist
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import gpu_select, images, telemetry

CFG = {"exe": "/tools/xestrata-image", "diffusion_model": "/models/dit.gguf",
       "vae": "/models/vae.gguf", "llm": "/models/text.gguf"}


class Gen:
    def __init__(self, need=None, model="default", owner=None, reserve=512, policy="auto"):
        self.cfg = {"exe": "/tools/xestrata-image"}
        self.engine = SimpleNamespace(env={"GPU_PCI": "0000:03:00.0"})
        self.model, self.owner, self.reserve, self.policy = model, owner, reserve, policy
        self.need_answer, self.needs, self.released, self.said = need, [], [], []
        self.totals = {"images": 0, "jobs": 0}
        self.state = None

    def _say(self, text):
        self.said.append(text)

    def need(self, job):
        self.needs.append(job)
        return self.need_answer

    def release(self, everything):
        self.released.append(everything)


def models_with(auto_cfgs=None, gpu_pci=None, separate=False):
    """An ImageModels instance without starting any image engine."""
    models = images.ImageModels.__new__(images.ImageModels)
    models.separate, models.gpu_pci, models.auto_cfgs = separate, gpu_pci, auto_cfgs
    models.lend_cfgs, models.lend_pci, models._lent = None, None, None
    models._auto_cards, models._auto_lock = {}, threading.Lock()
    models.gens = {"default": Gen(owner=models)}
    return models


class AutoCards(unittest.TestCase):
    def test_cards_no_text_engine_uses_are_listed_roomiest_first(self):
        svc = SimpleNamespace(
            gpu_roles=[{"pci": "0000:03:00.0", "roles": ["text"]},
                       {"pci": "0000:04:00.0", "roles": ["helper"]},
                       {"pci": "0000:09:00.0", "roles": ["image"]}],
            gpu_pci="0000:03:00.0", split_pci=["0000:04:00.0"], images=models_with(gpu_pci="0000:09:00.0"))
        rows = [{"pci": p, "backend": "Level Zero", "selector_index": i, "integrated": p == "0000:00:02.0"}
                for i, p in enumerate(["0000:03:00.0", "0000:04:00.0", "0000:09:00.0", "0000:0a:00.0",
                                       "0000:0a:00.0", "0000:0b:00.0", "0000:00:02.0"])]
        free = {"0000:03:00.0": 20000, "0000:04:00.0": 10000, "0000:09:00.0": 1024,
                "0000:0a:00.0": 8192, "0000:0b:00.0": None, "0000:00:02.0": 60000}
        with mock.patch.object(gpu_select, "selectors", return_value=rows), \
             mock.patch.object(telemetry, "free_vram_mib", side_effect=lambda pci: free[pci]):
            got = images.auto_cards(svc, Gen())
        self.assertEqual([c["pci"] for c in got], ["0000:0a:00.0"])
        self.assertEqual(got[0]["free_mib"], 8192)

    def test_no_device_list_means_no_card_and_no_error(self):
        with mock.patch.object(gpu_select, "selectors", return_value=[]):
            self.assertEqual(images.auto_cards(SimpleNamespace(), Gen()), [])

    def test_fits_uses_the_room_the_separate_turn_would_use(self):
        need = {"free_mib": 2500, "held_mib": 0, "least_mib": 2000}
        self.assertFalse(images.fits_auto_card(need, 512))
        self.assertTrue(images.fits_auto_card({"free_mib": 2600, "least_mib": 2000}, 512))
        self.assertTrue(images.fits_auto_card({"free_mib": 1000, "held_mib": 1600, "least_mib": 2000}, 512))


class AutoPick(unittest.TestCase):
    def test_asks_the_free_card_for_need_and_uses_it_when_the_job_fits(self):
        gen = Gen()
        other = Gen(need={"free_mib": 9000, "least_mib": 4000, "full_mib": 6000})
        card = {"pci": "0000:0a:00.0", "free_mib": 9000, "name": "RX 9060 XT"}
        models = models_with(auto_cfgs=([CFG], None, {}))
        models.auto_card = lambda pci: SimpleNamespace(gens={"default": other})
        with mock.patch.object(images, "auto_cards", return_value=[card]):
            picked = images.auto_pick(SimpleNamespace(images=models), gen, {"width": 1024})
        self.assertEqual(picked[0], card)
        self.assertIs(picked[1], other)
        self.assertEqual(other.needs, [{"width": 1024}])
        self.assertTrue(any("auto:" in line and "RX 9060 XT" in line for line in gen.said))
        self.assertEqual(other.released, [])

    def test_no_free_card_keeps_the_text_engines_decision(self):
        gen = Gen()
        with mock.patch.object(images, "auto_cards", return_value=[]):
            self.assertIsNone(images.auto_pick(SimpleNamespace(), gen, {}))
        self.assertTrue(any("no unused card" in line for line in gen.said))

    def test_a_card_the_job_does_not_fit_is_not_used(self):
        gen = Gen()
        other = Gen(need={"free_mib": 900, "least_mib": 4000})
        models = models_with(auto_cfgs=([CFG], None, {}))
        models.auto_card = lambda pci: SimpleNamespace(gens={"default": other})
        card = {"pci": "0000:0a:00.0", "free_mib": 900, "name": "RTX 4070"}
        with mock.patch.object(images, "auto_cards", return_value=[card]):
            self.assertIsNone(images.auto_pick(SimpleNamespace(images=models), gen, {}))
        self.assertEqual(other.released, [True])
        self.assertTrue(any("staying with the text engine" in line for line in gen.said))


class NamedCard(unittest.TestCase):
    def setUp(self):
        self.token = images.AUTO_DIST.set(False)

    def tearDown(self):
        images.AUTO_DIST.reset(self.token)

    def test_only_an_unnamed_auto_job_on_the_shared_card_is_eligible(self):
        models = models_with(auto_cfgs=([CFG], None, {}))
        gen = Gen(owner=models)
        svc = SimpleNamespace(images=models, gpu_roles=[{"pci": "0000:09:00.0", "roles": ["image"]}])
        images.place(svc, gen, "auto")
        self.assertTrue(images.AUTO_DIST.get())
        images.place(svc, gen, "image")
        self.assertFalse(images.AUTO_DIST.get())
        images.place(svc, gen, "auto")
        models.gpu_pci = "0000:09:00.0"   # a card named in the config
        images.place(svc, gen, "auto")
        self.assertFalse(images.AUTO_DIST.get())
        models.gpu_pci = None
        gen.policy = "shrink"
        images.place(svc, gen, "auto")
        self.assertFalse(images.AUTO_DIST.get())

    def test_auto_card_has_no_way_to_lend_the_job_back_to_the_text_engine(self):
        models = models_with(auto_cfgs=([CFG], None, {}))
        scoped = {"GPU_PCI": "0000:0a:00.0"}
        with mock.patch.object(gpu_select, "scope", return_value=scoped) as scope:
            got = models.auto_card("0000:0a:00.0")
        scope.assert_called_once_with("/tools/xestrata-image", {}, "0000:0a:00.0")
        self.assertTrue(got.separate)
        self.assertEqual(got.gpu_pci, "0000:0a:00.0")
        self.assertFalse(images.can_lend(got))
        self.assertIs(models.auto_card("0000:0a:00.0"), got)


class Entry(unittest.TestCase):
    def setUp(self):
        self.dist = images.AUTO_DIST.set(True)
        self.card = images.AUTO_CARD.set(None)

    def tearDown(self):
        images.AUTO_DIST.reset(self.dist)
        images.AUTO_CARD.reset(self.card)

    def test_the_job_runs_on_the_picked_card_and_says_so(self):
        models = models_with()
        gen, other = Gen(owner=models), Gen(owner=models)
        other.totals["images"] = 1
        svc = SimpleNamespace(images=models)
        seen = []

        @images.on_free_gpu
        def job(svc, gen, work):
            seen.append(gen)
            return [], {"seconds": 1.0}, "separate"

        card = {"pci": "0000:0a:00.0", "name": "RX 9060 XT"}
        need = {"free_mib": 9000, "least_mib": 4000}
        with mock.patch.object(images, "auto_pick", return_value=(card, other, need)):
            got = job(svc, gen, {"width": 1024})
        self.assertEqual(seen, [other])
        self.assertEqual(got[1]["gpu"]["card"], "auto")
        self.assertEqual(got[1]["gpu"]["pci"], card["pci"])
        self.assertIn("free card", images.place_words(got[1]["gpu"]))
        self.assertEqual(gen.last["gpu"]["pci"], card["pci"])
        self.assertEqual(gen.totals["images"], 1)
        self.assertIsNone(images.AUTO_CARD.get())

    def test_when_the_real_need_grows_the_job_falls_back_to_the_text_card(self):
        models = models_with()
        gen, other = Gen(owner=models), Gen(owner=models)
        svc = SimpleNamespace(images=models)
        seen = []

        @images.on_free_gpu
        def job(svc, g, work):
            seen.append(g)
            if g is other:
                raise images.AutoCardNoFit("no room")
            return [], {"seconds": 2.0}, "keep"

        card = {"pci": "0000:0a:00.0", "name": "RX 9060 XT"}
        need = {"free_mib": 900, "least_mib": 4000}
        with mock.patch.object(images, "auto_pick", return_value=(card, other, need)):
            got = job(svc, gen, {})
        self.assertEqual(seen, [other, gen])
        self.assertEqual(other.released, [True])
        self.assertNotIn("gpu", got[1])
        self.assertTrue(any("falling back" in line for line in gen.said))

    def test_an_unnamed_job_with_no_pick_runs_where_it_would_have(self):
        models = models_with()
        gen = Gen(owner=models)
        seen = []

        @images.on_free_gpu
        def job(svc, g, work):
            seen.append(g)
            return "done"

        with mock.patch.object(images, "auto_pick", return_value=None):
            self.assertEqual(job(SimpleNamespace(images=models), gen, {}), "done")
        self.assertEqual(seen, [gen])


class SeparateTurn(unittest.TestCase):
    def setUp(self):
        self.auto = images.AUTO_CARD.set({"card": "auto", "name": "RX 9060 XT", "pci": "0000:0a:00.0"})
        self.lend = images.AUTO_LEND.set(None)
        self.retry = images.LEND_RETRY.set(False)

    def tearDown(self):
        images.AUTO_CARD.reset(self.auto)
        images.AUTO_LEND.reset(self.lend)
        images.LEND_RETRY.reset(self.retry)

    def run_turn(self, need):
        gen = Gen(need=need)
        gen.totals = {"failed": 0}
        gen.family = "qwen-image"
        gen.begin = lambda job: None
        gen.end = lambda error, cancelled=False: None
        gen._live = lambda **kw: None
        gen.ledger_sync = lambda: None
        models = SimpleNamespace(gpu=mock.MagicMock(), helper_card="", activate=lambda g: None)
        gen.owner = models
        svc = SimpleNamespace(images=models, last_request_at=0)
        with mock.patch.object(images, "refuse_during_long_job"), \
             mock.patch.object(images, "helper_borrow", return_value=False), \
             mock.patch.object(images, "check_room"), \
             mock.patch.object(images, "ram_available_mib", return_value=100000), \
             mock.patch.object(images, "_qimage", return_value=SimpleNamespace(
                 family_capabilities=lambda family: {})), \
             mock.patch.object(images, "ledger_fit", return_value=1000), \
             mock.patch.object(images, "_give_back"), \
             mock.patch.object(images.ImageGen, "ledger_sync", lambda self: None):
            with images.separate_turn(svc, gen, {}) as policy:
                return [policy]

    def test_the_real_need_is_checked_again_on_the_picked_card(self):
        with self.assertRaises(images.AutoCardNoFit):
            self.run_turn({"free_mib": 900, "least_mib": 4000, "full_mib": 6000})
        self.assertEqual(self.run_turn({"free_mib": 9000, "least_mib": 4000, "full_mib": 6000}), ["separate"])


if __name__ == "__main__":
    unittest.main()
