"""#5625: `strata launch ... --restart` with a changed flag re-ran the model's preparation test while the old engine still held the graphics memory: the test
failed with "CUDA0 has 0.04 GB free of 50.86 GB" (a false not-enough-VRAM) and the restart was dropped. The server the restart replaces is stopped after the
engine is here and before the model is prepared (Runner.stop_before_setup). Without a restart nothing is stopped there."""

import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog


class RestartStopsTheOldServerFirst(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def runner(self, **kw):
        r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, name="gyro"), catalog=catalog(), **kw)
        r.said = []
        r.emit_cb = lambda e: r.said.append(e.text)
        return r

    def run_to_the_model_step(self, r, rows):
        order = []

        class Reached(Exception):
            pass

        def model_step(self):
            order.append("prepare the model")
            raise Reached()
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
                mock.patch.object(Runner, "ensure_engine", lambda self: order.append("engine")), \
                mock.patch.object(Runner, "ensure_model", model_step), \
                mock.patch("strata_control.runtime.stop", side_effect=lambda n, timeout=0: order.append("stop " + n)), \
                mock.patch("strata_cli.launch.access.stop_tunnel"):
            with self.assertRaises(Reached):
                r.run()
        return order

    def test_the_old_server_is_stopped_before_the_model_is_prepared(self):
        r = self.runner(restart=True)
        rows = [{"profile": r.profile, "name": r.name, "port": 8080, "choices": {**r.choices(), "context": 65536}}]
        order = self.run_to_the_model_step(r, rows)
        self.assertEqual(order, ["engine", "stop " + r.profile, "prepare the model"])
        self.assertTrue(any("holds the graphics memory" in x for x in r.said))

    def test_another_launch_on_this_port_is_stopped_first_too(self):
        r = self.runner(restart=True)
        rows = [{"profile": "launch-gyro-s-32k", "name": "gyro-s-32k", "port": 8080, "choices": {"context": 32768}}]
        order = self.run_to_the_model_step(r, rows)
        self.assertEqual(order, ["engine", "stop launch-gyro-s-32k", "prepare the model"])

    def test_nothing_is_stopped_when_nothing_is_running(self):
        r = self.runner(restart=True)
        order = self.run_to_the_model_step(r, [])
        self.assertEqual(order, ["engine", "prepare the model"])

    def test_without_a_restart_the_launch_stops_and_nothing_is_touched(self):
        r = self.runner()
        rows = [{"profile": r.profile, "name": r.name, "port": 8080, "choices": {**r.choices(), "context": 65536}}]
        with mock.patch.object(Runner, "serving_launches", lambda self: rows), \
                mock.patch.object(Runner, "ensure_engine", side_effect=AssertionError("must not run")), \
                mock.patch("strata_control.runtime.stop", side_effect=AssertionError("must not stop")):
            with self.assertRaises(LaunchError):
                r.run()


if __name__ == "__main__":
    unittest.main()
