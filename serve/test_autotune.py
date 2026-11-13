# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/autotune.py: the choices the machine makes ("auto") - from the cards' VRAM, the RAM, their number and the
model's size - and the measurement of the batch slots from the first requests.  No GPU and no engine: the facts are
handed to the rules.

    python -m unittest serve.test_autotune
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import autotune as A


def machine(vram=(24.0,), ram=128.0, model=71.0, experts=50.0, ctx=32768, kv="int8", streaming=False, **kw):
    cards = tuple(f"0000:0{i}:00.0" for i in range(len(vram)))
    return A.Machine(vram=tuple(vram), ram=ram, model=model, experts=experts, ctx=ctx, kv=kv, streaming=streaming,
                     cards=cards, **kw)


class Slots(unittest.TestCase):
    def test_only_where_the_experts_fit_beside_them(self):
        # the experts mostly run on the CPU: every slot's VRAM is expert cache lost
        self.assertEqual(A.parallel(machine(vram=(12.0,))), 0)
        self.assertEqual(A.parallel(machine(vram=(16.0,))), 0)
        self.assertEqual(A.parallel(machine(vram=(24.0,))), 0)
        # a card whose cache holds them beside the sessions
        self.assertEqual(A.parallel(machine(vram=(64.0,))), 4)

    def test_a_long_context_holds_fewer_or_none(self):
        self.assertEqual(A.parallel(machine(vram=(64.0,), ctx=262144, kv="fp16")), 0)
        self.assertEqual(A.parallel(machine(vram=(64.0,), ctx=262144, kv="fp16", streaming=True)), 4)

    def test_a_dense_model_needs_only_the_sessions_room(self):
        dense = machine(vram=(24.0,), model=5.0, experts=0.0, kv="fp16")
        self.assertEqual(A.parallel(dense), 3)                  # sessions of ~1.0 GB against 20% of 19 GB
        self.assertEqual(A.parallel(machine(vram=(8.0,), model=5.0, experts=0.0, kv="fp16")), 0)   # no room for it

    def test_unknown_facts_and_models_without_slots(self):
        self.assertEqual(A.parallel(machine(model=0.0)), 0)                       # a config written by hand
        self.assertEqual(A.parallel(machine(slots_ok=False)), 0)                 # GLM-5.3-Flash, or --vision

    def test_groups_follow_a_layer_split(self):
        self.assertEqual(A.groups(machine(vram=(32.0, 32.0)), 4), A.AUTO)
        self.assertIsNone(A.groups(machine(), 4))                                # one card
        self.assertIsNone(A.groups(machine(vram=(32.0, 32.0)), 1))               # one at a time: nothing to group


class Elastic(unittest.TestCase):
    def test_only_where_something_borrows_the_vram(self):
        self.assertTrue(A.elastic(machine(), borrower=True))
        self.assertFalse(A.elastic(machine(), borrower=False))
        self.assertFalse(A.elastic(machine(vram=(32.0, 32.0)), borrower=True))   # the engine's cache is per card
        # a GPU encoder still starts with its picture there (the engine turns the segmenting off and says so)
        self.assertTrue(A.elastic(machine(vram=(32.0, 32.0)), borrower=True, encoder=True))


class Replicas(unittest.TestCase):
    def test_one_engine_per_card_when_the_model_fits_each(self):
        # two 32 GB cards, 50 GB of experts in RAM twice: two engines, each with the whole model
        self.assertEqual(A.replicas(machine(vram=(32.0, 32.0)), {"args": []}), 2)
        # the second copy of the experts does not fit the RAM: a layer split's one pipeline
        self.assertEqual(A.replicas(machine(vram=(32.0, 32.0), ram=64.0), {"args": []}), 0)
        # --mmap-experts reads them from the files: no copy in RAM
        self.assertEqual(A.replicas(machine(vram=(32.0, 32.0), ram=64.0), {"args": ["--mmap-experts"]}), 2)

    def test_not_without_the_cards_or_the_room(self):
        self.assertEqual(A.replicas(machine(), {"args": []}), 0)                         # one card
        self.assertEqual(A.replicas(machine(vram=(16.0, 16.0)), {"args": []}), 0)        # the model needs ~25 GB
        self.assertEqual(A.replicas(machine(vram=(32.0, 32.0)), {"args": [], "layer_split": "24"}), 0)


class SplitServe(unittest.TestCase):
    def test_the_second_card_when_replicas_cannot_run(self):
        # two cards, the model fits each, the RAM holds one copy of the experts: the second card reads the prompts
        got = A.split_serve(machine(vram=(32.0, 32.0), ram=64.0), {"args": ["--mtp"]})
        self.assertEqual(got, {"prefill_gpu": "0000:00:00.0", "decode_gpu": "0000:01:00.0"})
        # the RAM holds two copies: replicas serve two conversations, preferred
        self.assertIsNone(A.split_serve(machine(vram=(32.0, 32.0)), {"args": ["--mtp"]}))

    def test_what_the_split_cannot_run_with(self):
        m = machine(vram=(32.0, 32.0), ram=64.0)
        self.assertIsNone(A.split_serve(m, {"args": []}))                        # no --mtp: no session file
        self.assertIsNone(A.split_serve(m, {"args": ["--mtp"]}, busy=True))      # a --batch of the config's own
        self.assertIsNone(A.split_serve(machine(), {"args": ["--mtp"]}))         # one card
        self.assertIsNone(A.split_serve(machine(vram=(16.0, 16.0), ram=64.0), {"args": ["--mtp"]}))   # does not fit


class Conversation(unittest.TestCase):
    def test_auto_leaves_it_to_the_engine(self):
        self.assertIsNone(A.conversation_mib({}))
        self.assertIsNone(A.conversation_mib({"conversation_cache_mib": "auto"}))
        self.assertIsNone(A.conversation_mib({"conversation_cache_mib": True}))
        self.assertEqual(A.conversation_mib({"conversation_cache_mib": False}), 0)
        self.assertEqual(A.conversation_mib({"conversation_cache_mib": 2048}), 2048)
        with self.assertRaises(ValueError):
            A.conversation_mib({"conversation_cache_mib": "a lot"})


class Resolve(unittest.TestCase):
    """The plan: every "auto" chosen, the user's own values kept, the config left as it was.  "auto" never takes the
    slots or re-divides the cards: both cost the one user a server mostly has."""
    CFG = {"args": ["--native", "/nonexistent/model.gguf", "--max-context", "32768", "--mtp", "/m/mtp.gguf"],
           "gpu_pci": "0000:03:00.0", "model_gb": 71.0, "experts_gb": 50.0}

    def test_auto_keeps_one_request_at_a_time(self):
        # a 64 GB card holds 4 slots beside the experts, and "auto" still starts none of them: a request alone would
        # take the smaller expert cache
        plan = A.resolve(self.CFG, machine(vram=(64.0,)))
        self.assertEqual(A.parallel(machine(vram=(64.0,))), 4)          # what the machine could hold
        self.assertEqual(plan.cfg["parallel"], 1)
        self.assertTrue(any('"parallel": 4' in x for x in plan.notes))
        self.assertNotIn("batch_groups", plan.cfg)                     # one card: one group
        self.assertIs(plan.cfg["vram_elastic"], False)                 # nothing borrows the VRAM
        self.assertNotIn("conversation_cache_mib", plan.cfg)           # the engine sizes it
        self.assertIs(plan.cfg["split_skip_if_fits"], False)           # one card: no layer split
        self.assertEqual(len(plan.notes), 7)

    def test_auto_does_not_re_divid_the_cards(self):
        # two 32 GB cards that each hold the model, and the RAM for two copies: replicas would pay for conversations,
        # but a request alone is slower there - "auto" keeps the layer split
        m = machine(vram=(32.0, 32.0), ram=128.0)
        self.assertEqual(A.replicas(m, self.CFG), 2)                    # what the machine could do
        plan = A.resolve(self.CFG, m)
        self.assertNotIn("replicas", plan.cfg)
        self.assertNotIn("split_serve", plan.cfg)
        self.assertEqual(plan.cfg["parallel"], 1)
        self.assertTrue(any('"replicas": 2' in x for x in plan.notes))

    def test_auto_offers_split_serving_but_does_not_take_it(self):
        # two cards that each hold the model, and the RAM for one copy of the experts: split serving could read the
        # prompts on the second card, but a request alone is made on one there
        m = machine(vram=(32.0, 32.0), ram=64.0)
        self.assertIsNotNone(A.split_serve(m, self.CFG))
        plan = A.resolve(self.CFG, m)
        self.assertNotIn("split_serve", plan.cfg)
        self.assertTrue(any('"split_serve"' in x for x in plan.notes))

    def test_the_users_values_are_kept(self):
        cfg = {**self.CFG, "parallel": 2, "vram_elastic": True, "conversation_cache_mib": 1024,
               "replicas": [{"gpus": ["0000:03:00.0"]}, {"gpus": ["0000:09:00.0"]}], "batch_groups": 2,
               "split_skip_if_fits": False}
        plan = A.resolve(cfg, machine(vram=(64.0,)))
        self.assertEqual(plan.cfg["parallel"], 2)
        self.assertIs(plan.cfg["vram_elastic"], True)
        self.assertEqual(plan.cfg["conversation_cache_mib"], 1024)
        self.assertEqual(plan.cfg["replicas"], cfg["replicas"])
        self.assertEqual(plan.cfg["batch_groups"], 2)
        self.assertIs(plan.cfg["split_skip_if_fits"], False)
        self.assertTrue(any("as the config says" in x for x in plan.notes))

    def test_off_is_the_users_too(self):
        plan = A.resolve({**self.CFG, "parallel": 1, "vram_elastic": False}, machine(vram=(64.0,)))
        self.assertEqual(plan.cfg["parallel"], 1)
        self.assertIs(plan.cfg["vram_elastic"], False)

    def test_the_config_passed_in_is_left_alone(self):
        before = dict(self.CFG)
        A.resolve(self.CFG, machine(vram=(64.0,)))
        self.assertEqual(self.CFG, before)

    def test_a_replica_counts_the_slots_of_its_own_card(self):
        plan = A.resolve({**self.CFG, "replicas": 2}, machine(vram=(32.0, 32.0), ram=128.0))
        self.assertEqual(plan.cfg["replicas"], 2)                      # the user's
        self.assertEqual(plan.cfg["parallel"], 1)                      # "auto" still starts none

    def test_a_layer_split_point_of_the_users_is_kept(self):
        plan = A.resolve({**self.CFG, "split_pci": ["0000:09:00.0"], "layer_split": "24"},
                         machine(vram=(32.0, 32.0), ram=64.0))
        self.assertIs(plan.cfg["split_skip_if_fits"], False)          # the split point is the user's: not skipped
        self.assertNotIn("replicas", plan.cfg)
        self.assertNotIn("split_serve", plan.cfg)
        self.assertEqual(plan.cfg["parallel"], 1)

    def test_split_serving_steps_aside_for_the_configs_parallel(self):
        plan = A.resolve({**self.CFG, "split_pci": ["0000:09:00.0"], "parallel": 2},
                         machine(vram=(32.0, 32.0), ram=64.0))
        self.assertNotIn("split_serve", plan.cfg)                      # a split cannot run several at once
        self.assertEqual(plan.cfg["parallel"], 2)

    def test_each_engine_holds_its_cache_in_pieces_only_on_one_card(self):
        # a split serving the user set: each engine is on one card, so an image job may borrow its VRAM
        plan = A.resolve({**self.CFG, "split_serve": {"prefill_gpu": "0000:03:00.0", "decode_gpu": "0000:09:00.0"},
                          "images": {"exe": "x"}}, machine(vram=(32.0, 32.0), ram=64.0))
        self.assertEqual(plan.cfg["split_serve"]["prefill_gpu"], "0000:03:00.0")
        self.assertIs(plan.cfg["vram_elastic"], True)
        # a layer split the user set: each card's cache cannot give its VRAM back
        plan = A.resolve({**self.CFG, "split_pci": ["0000:09:00.0"], "layer_split": "24", "images": {"exe": "x"}},
                         machine(vram=(32.0, 32.0), ram=64.0))
        self.assertIs(plan.cfg["vram_elastic"], False)


class ModelFiles(unittest.TestCase):
    def test_the_shards_are_summed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in (1, 2):
                (root / f"model-{i:05d}-of-00002.gguf").write_bytes(b"x" * (i * 1000))
            (root / "model-00001-of-00002.gguf").write_bytes(b"x" * 1000)   # 1000 + 2000 bytes
            self.assertAlmostEqual(A.model_gb(["--native", str(root / "model-00001-of-00002.gguf")]), 3e-6)
            self.assertAlmostEqual(A.model_gb(["--native", str(root / "single.gguf")]), 0.0)   # not there
            self.assertAlmostEqual(A.model_gb([]), 0.0)


class Read(unittest.TestCase):
    def test_the_config_and_the_machine(self):
        cfg = {"args": ["--native", "/m/a.gguf", "--pack", "/m/pack", "--max-context", "65536", "--kv", "q4_0",
                        "--kv-resident", "32768"],
               "gpu_pci": "0000:03:00.0", "model_gb": 71.0}
        with mock.patch("serve.telemetry.total_vram_mib", lambda pci=None: 32768), \
                mock.patch("serve.telemetry.free_vram_mib", lambda pci=None: 30000), \
                mock.patch("serve.telemetry.ram_gb", lambda: 128.0):
            m = A.read(cfg)
        self.assertEqual(m.vram, (32.0,))
        self.assertEqual(m.free, (30000 / 1024,))
        self.assertEqual(m.ram, 128.0)
        self.assertEqual(m.ctx, 65536)
        self.assertEqual(m.kv, "q4_0")
        self.assertTrue(m.streaming)
        self.assertEqual(m.model, 71.0)
        # a Qwen3.8 config from an older setup keeps its experts in the model's files
        self.assertEqual(m.experts, 71.0)

    def test_a_dense_model_has_no_experts(self):
        with mock.patch("serve.telemetry.total_vram_mib", lambda pci=None: 24576), \
                mock.patch("serve.telemetry.free_vram_mib", lambda pci=None: 20000), \
                mock.patch("serve.telemetry.ram_gb", lambda: 64.0):
            m = A.read({"args": ["--native", "/m/plain.gguf", "--max-context", "32768"], "model_gb": 21.7})
        self.assertEqual(m.experts, 0.0)
        self.assertEqual(m.dense, 21.7)


if __name__ == "__main__":
    unittest.main()
