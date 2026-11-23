# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""#68: a replica must get a selector for its own card, not one inherited from the server.

The engines and the device tool are mocked: this checks the wiring in start_replicas(), with no GPU.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from serve import server

EXE = "/fake/xestrata-engine"
A, B = "0000:05:00.0", "0000:19:00.0"
SELECTOR = {A: "level_zero:0", B: "level_zero:1"}


class ReplicaScopeTest(unittest.TestCase):
    def launch(self, groups, envs=None, inherited="level_zero:1", scope_error=None):
        """Start with fake engines; return (result, SystemExit, scoped envs, engine envs, scope calls)."""
        envs = envs or [None] * len(groups)
        cfg = {"args": [], "replicas": [{"gpus": g, **({"env": e} if e else {})}
                                        for g, e in zip(groups, envs)]}

        def scope(exe, env, pci):
            if scope_error:
                raise ValueError(scope_error)
            return dict(env, ONEAPI_DEVICE_SELECTOR=SELECTOR.get(pci, "level_zero:9"), XESTRATA_GPU_PCI=pci)

        error = None
        with (mock.patch.dict(os.environ, {"ONEAPI_DEVICE_SELECTOR": inherited}, clear=True),
              mock.patch("serve.gpu_select.scope", side_effect=scope) as scope_mock,
              mock.patch("serve.server.StrataEngine") as engine,
              mock.patch("serve.server.engine_args", return_value=[]),
              mock.patch("serve.server.warn_budget_over_ram"),
              mock.patch("serve.split.arena_path", return_value=(None, None, "")),
              mock.patch("serve.replicas.cpu_sets", return_value=None),
              mock.patch("serve.replicas.ReplicaEngine", return_value="engine")):
            try:
                result = server.start_replicas(cfg, groups, EXE, [], False)
            except SystemExit as e:
                result = None
                error = e
        scoped = {c.args[2]: c.args[1] for c in scope_mock.call_args_list}
        started = {c.kwargs["env"]["XESTRATA_GPU_PCI"]: c.kwargs["env"] for c in engine.call_args_list}
        return result, error, scoped, started, scope_mock

    def test_each_replica_is_scoped_to_its_own_card(self):
        result, error, scoped, started, calls = self.launch([[A], [B]])
        self.assertIsNone(error)
        self.assertEqual(result, "engine")
        self.assertEqual(set(scoped), {A, B})                      # both replicas scoped
        for call in calls.call_args_list:
            self.assertEqual(call.args[0], EXE)
            self.assertEqual(call.args[1]["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")   # what the server inherited
        for pci in (A, B):
            self.assertEqual(started[pci]["XESTRATA_GPU_PCI"], pci)
            self.assertEqual(started[pci]["ONEAPI_DEVICE_SELECTOR"], SELECTOR[pci])
            self.assertNotIn("STRATA_GPU_PCI", started[pci])

    def test_an_explicit_replica_selector_is_kept(self):
        _, error, scoped, started, calls = self.launch(
            [[A], [B]], envs=[{"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}, {"ZE_AFFINITY_MASK": "1"}])
        self.assertIsNone(error)
        self.assertEqual(calls.call_count, 0)                      # the config's own selection is not replaced
        self.assertEqual(started[A]["ONEAPI_DEVICE_SELECTOR"], "level_zero:0")
        self.assertEqual(started[B]["ZE_AFFINITY_MASK"], "1")
        self.assertEqual(started[A]["XESTRATA_GPU_PCI"], A)
        self.assertEqual(started[B]["XESTRATA_GPU_PCI"], B)

    def test_a_multi_card_replica_keeps_its_layer_split_visible(self):
        groups = [["0000:03:00.0", "0000:09:00.0"], [A, B]]
        _, error, scoped, started, calls = self.launch(groups)
        self.assertIsNone(error)
        self.assertEqual(calls.call_count, 0)                      # one selector would hide the split's other card
        for first in (groups[0][0], A):
            self.assertEqual(started[first]["ONEAPI_DEVICE_SELECTOR"], "level_zero:1")
            self.assertEqual(started[first]["XESTRATA_GPU_PCI"], first)

    def test_a_selector_error_stops_the_start_with_a_message(self):
        _, error, _, started, _ = self.launch([[A]], scope_error=f"GPU {A}: cannot find one supported backend")
        self.assertIsInstance(error, SystemExit)
        self.assertIn("replica 0 did not start", str(error))
        self.assertIn("cannot find one supported backend", str(error))
        self.assertEqual(started, {})                              # no engine left starting without an error


if __name__ == "__main__":
    unittest.main()
