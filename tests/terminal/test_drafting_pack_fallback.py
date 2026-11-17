"""#5810: a start whose stock DFlash2 pack is not here, while the set's tuned pack is (setup fetched it for a start that held more experts), starts DFlash2 with the
tuned pack at the stock pack's conf, not the MTP or no drafter (owner: the drafter stays on). Then the MTP, then off. And the launch's fetch reads the card
signals of the config as the server starts it (serve.server.apply_vision_defaults), so fetch and start cannot sit on either side of the tuned pack's slot line.

The matrix: 11,700 / 12,090 expert slots (--expert-cache 9000 / 9300, 1.3 slots each; the line is 12,000) x packs here (stock only, tuned only, both, neither)
x the MTP layer here or not. Only "tuned only, under the line" changes against the code before (it was the MTP, or off without the MTP layer)."""

from __future__ import annotations

import json
import sys
import unittest
from unittest import mock

from strata_control import drafting
from tests.terminal import test_drafting_tuned as T

IQ3S, L620, flag = T.IQ3S, T.L620, T.flag

UNDER, OVER = "9000", "9300"


class PackFallback(T.Base):
    def setUp(self):
        super().setUp()
        (self.data / "dflash2" / drafting.PACK_NAME).unlink(missing_ok=True)

    def start(self, cache, packs, mtp=True):
        for p in (drafting.PACK_NAME, drafting.TUNED_NAME, drafting.TUNED_NAME + ".ok"):
            (self.data / "dflash2" / p).unlink(missing_ok=True)
        if packs in ("stock", "both"):
            self.put_stock()
        if packs in ("tuned", "both"):
            self.put_tuned()
        args = list(self.args) + ["--expert-cache", cache]
        layer = self.data / "mtp" / "rt" / "experts.bin"
        if mtp:
            layer.write_bytes(b"e" * 100)
        else:
            layer.unlink(missing_ok=True)                      # a DFlash2-only install: --mtp names the folder, the layer was never fetched
        cfg = {"model_name": IQ3S, "args": args}
        _label, backend, archs, vram = L620
        decision = drafting.apply(cfg, data_dir=self.data, backend=backend, archs=archs, engine=self.exe, vram_mib=vram)
        return decision, cfg["args"]

    def test_the_matrix(self):
        tuned, stock = str(drafting.tuned_path(self.data)), str(self.data / "dflash2" / drafting.PACK_NAME)
        want = {   # (slots side, packs, MTP layer here) -> (drafter, --draft-model, --draft-conf)
            (OVER, "both", True): ("dflash2", tuned, "0.55"), (OVER, "tuned", True): ("dflash2", tuned, "0.55"),
            (OVER, "stock", True): ("dflash2", stock, "0.35"), (OVER, "neither", True): ("mtp", None, None),
            (UNDER, "both", True): ("dflash2", stock, "0.5"), (UNDER, "stock", True): ("dflash2", stock, "0.5"),
            (UNDER, "tuned", True): ("dflash2", tuned, "0.5"),            # #5810: was the MTP
            (UNDER, "neither", True): ("mtp", None, None),
            (UNDER, "tuned", False): ("dflash2", tuned, "0.5"),           # #5810: was no drafter at all ("no drafter is downloaded yet")
            (UNDER, "neither", False): ("off", None, None), (OVER, "neither", False): ("off", None, None),
        }
        for (cache, packs, mtp), (drafter, model, conf) in want.items():
            with self.subTest(slots=cache, packs=packs, mtp=mtp):
                d, args = self.start(cache, packs, mtp)
                self.assertEqual(d.drafter, drafter)
                self.assertEqual(flag(args, "--draft-model"), model)
                self.assertEqual(flag(args, "--draft-conf"), conf)
                if drafter == "dflash2":
                    self.assertNotIn("--mtp", args)
                    self.assertEqual(flag(args, "--draft-fallback"), "none")

    def test_a_pinned_pack_is_left_alone(self):
        self.put_tuned()
        cfg = {"model_name": IQ3S, "args": list(self.args) + ["--expert-cache", UNDER], "user_args": ["--draft-model", "/my/own.pack"]}
        d = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], engine=self.exe, vram_mib=73728)
        self.assertNotEqual(flag(cfg["args"], "--draft-model"), str(drafting.tuned_path(self.data)), d)


class FetchReadsTheStartedConfig(unittest.TestCase):
    """runner.card_signals (the launch's fetch and plan) reads the config after serve.server.apply_vision_defaults."""

    def signals(self, cfg):
        from strata_cli.launch import runner
        with mock.patch.object(drafting, "card_vram_mib", return_value=32607):
            return runner.card_signals(cfg, None, {"backend": cfg.get("backend"), "archs": ["sm_120"]})

    @unittest.skipIf(sys.platform == "darwin", "a Mac never raises the reserve for the picture reader (serve.server.vision_on_demand_by_default)")
    def test_a_reserve_the_start_raises_is_the_one_the_fetch_reads(self):
        from serve import server
        vision = {"gpu": True, "exe": "component:strata-vision", "cpu_exe": "component:strata-vision"}
        held_room = {"backend": "cuda", "model_name": IQ3S, "args": ["--max-context", "131072", "--vram-reserve-mib", "700"], "vision": dict(vision),
                     "vram_elastic": False}                          # a config that keeps the reader's room: the start raises the reserve to 2048
        args = self.signals(json.loads(json.dumps(held_room)))["args"]
        self.assertEqual(flag(args, "--vram-reserve-mib"), str(server.VISION_RESERVE_MIB))
        self.assertEqual(held_room["args"][-1], "700", "the caller's config is left as it was")
        elastic = {**held_room, "args": list(held_room["args"])}
        elastic.pop("vram_elastic")                                  # the default on NVIDIA: nothing held back, the reserve stays
        self.assertEqual(flag(self.signals(elastic)["args"], "--vram-reserve-mib"), "700")
        self.assertEqual(flag(self.signals({**elastic, "vision": None})["args"], "--vram-reserve-mib"), "700")


if __name__ == "__main__":
    unittest.main()
