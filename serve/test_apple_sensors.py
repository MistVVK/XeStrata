"""AppleSensors (serve/telemetry.py): the Apple Silicon GPU's temperature and power from a fake _AppleNative, so it
runs on any system.  The native calls themselves are checked on a Mac: see the pull request's validation."""
import os
import unittest
from unittest import mock

from serve import telemetry


class FakeNative:
    def __init__(self, temps=(), energy=(), rep=True, smc=()):
        self._temps, self._energy, self.rep, self._smc = list(temps), list(energy), rep, list(smc)

    def gpu_smc(self):
        return self._smc

    def temps(self):
        return self._temps

    def energy(self):
        return self._energy.pop(0) if self._energy else []


class Clock:
    def __init__(self, *ts):
        self.ts = list(ts)

    def __call__(self):
        return self.ts.pop(0)


class AppleSensorsTest(unittest.TestCase):
    def test_gpu_temp_is_the_mean_of_the_gpu_sensors(self):
        n = FakeNative(temps=[("GPU MTR Temp Sensor1", 50.0), ("GPU MTR Temp Sensor4", 54.0),
                              ("pACC MTR Temp Sensor2", 80.0), ("GPU MTR Temp Sensor7", -1.0)])
        self.assertEqual(telemetry.AppleSensors(n, clock=Clock(0)).gpu_temp(), 52.0)

    def test_no_gpu_sensor_no_temp(self):
        n = FakeNative(temps=[("PMU tdie1", 45.0), ("GPU MTR Temp Sensor1", 130.0)])
        self.assertIsNone(telemetry.AppleSensors(n, clock=Clock(0)).gpu_temp())

    def test_smc_keys_win_over_hid_and_drop_bad_values(self):
        n = FakeNative(smc=[60.0, 64.0, 0.0, 200.0], temps=[("GPU MTR Temp Sensor1", 50.0)])
        self.assertEqual(telemetry.AppleSensors(n, clock=Clock(0)).gpu_temp(), 62.0)

    def test_hid_when_the_smc_has_nothing(self):
        n = FakeNative(smc=[], temps=[("GPU MTR Temp Sensor1", 50.0)])
        self.assertEqual(telemetry.AppleSensors(n, clock=Clock(0)).gpu_temp(), 50.0)

    def test_a_native_without_smc_support_still_works(self):
        class NoSmc:
            rep = None

            def temps(self):
                return [("GPU MTR Temp Sensor1", 50.0)]

            def energy(self):
                return []
        self.assertEqual(telemetry.AppleSensors(NoSmc(), clock=Clock(0)).gpu_temp(), 50.0)

    def test_gpu_keys_follow_the_chip_and_its_tier(self):
        keys = telemetry.smc_gpu_keys
        m5 = ("Tg0U", "Tg0X", "Tg0d", "Tg0g", "Tg0j", "Tg1Y", "Tg1c", "Tg1g")     # the keys run on an M5 Pro
        for brand in ("Apple M5", "Apple M5 Pro", "Apple M5 Max", "Apple M5 Ultra"):
            self.assertEqual(keys(brand), m5, brand)
        self.assertEqual(keys("Apple M4"), ("Tg0G", "Tg0H", "Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k"))
        self.assertEqual(keys("Apple M4 Pro")[:2], ("Tg1U", "Tg1k"))
        self.assertEqual(keys("Apple M4 Max"), keys("Apple M4 Pro"))
        self.assertEqual(len(keys("Apple M4 Pro")), 8)
        self.assertIsNone(keys("Apple M4 Ultra"))                                   # no such chip: no guess
        self.assertEqual(keys("Apple M3 Max")[0], "Tf14")                           # an M3's are Tf.., not Tg..
        self.assertEqual([len(keys(b)) for b in ("Apple M1", "Apple M1 Ultra", "Apple M2 Pro", "Apple M3", "Apple M6 Max",
                                                  "Apple A18 Pro")], [4, 4, 2, 8, 12, 4])

    def test_a_chip_not_in_the_table_reads_none(self):
        for brand in ("Apple M9", "Apple M10", "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz", "", None, "Apple"):
            self.assertIsNone(telemetry.smc_gpu_keys(brand), brand)

    def test_the_table_is_well_formed(self):
        every = [telemetry.smc_gpu_keys(b) for b in ("Apple M1", "Apple M2", "Apple M3", "Apple M4", "Apple M4 Pro",
                                                     "Apple M5", "Apple M6", "Apple A18 Pro")]
        for keys in every:
            self.assertTrue(all(isinstance(k, str) and len(k) == 4 for k in keys), keys)   # an SMC key is four characters
            self.assertEqual(len(set(keys)), len(keys), keys)

    def test_gpu_power_from_the_energy_delta(self):
        # the baseline at construction, then 2 s later 30,000 mJ on the GPU channel: 15 W; the CPU channel is ignored
        n = FakeNative(energy=[[], [("GPU Energy", "mJ", 30000), ("CPU Energy", "mJ", 99999)]])
        s = telemetry.AppleSensors(n, clock=Clock(10.0, 12.0))
        self.assertEqual(s.gpu_power(), 15.0)

    def test_units_and_unknown_units(self):
        n = FakeNative(energy=[[], [("GPU Energy", "nJ", 4_000_000_000)], [("GPU Energy", "kWh", 1)]])
        s = telemetry.AppleSensors(n, clock=Clock(0.0, 1.0, 2.0))
        self.assertEqual(s.gpu_power(), 4.0)
        self.assertIsNone(s.gpu_power())

    def test_first_reading_has_no_power(self):
        n = FakeNative(energy=[[("GPU Energy", "mJ", 5)]])
        s = telemetry.AppleSensors(n, clock=Clock(0.0, 1.0))
        self.assertIsNone(s.gpu_power())             # the construction call took the only sample

    def test_metal_reader_carries_the_readings_and_the_note(self):
        from serve.test_server import IOREG_FIXTURE
        n = FakeNative(temps=[("GPU MTR Temp Sensor1", 61.0)], energy=[[], [("GPU Energy", "mJ", 8000)]])
        g = telemetry._Metal(0, run=lambda: IOREG_FIXTURE, sensors=telemetry.AppleSensors(n, clock=Clock(0.0, 1.0)))
        r = g.read()
        self.assertEqual((r["temp"], r["power"]), (61.0, 8.0))
        self.assertNotIn("powermetrics", g.note)

    def test_without_sensors_the_reader_is_unchanged(self):
        g = telemetry._Metal(0, run=lambda: "", sensors=telemetry.AppleSensors(FakeNative(rep=False), clock=Clock(0)))
        self.assertIn("powermetrics", g.note)       # nothing readable: the class's old note stays

    def test_opt_in(self):
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry, "AppleSensors", return_value="on") as made:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STRATA_MACOS_SENSORS", None)
                self.assertIsNone(telemetry.apple_sensors())            # unset: a Mac's Monitor is what it was
            for value in ("0", "", "true"):
                with mock.patch.dict(os.environ, {"STRATA_MACOS_SENSORS": value}):
                    self.assertIsNone(telemetry.apple_sensors(), value)  # only "1" turns the private calls on
            self.assertFalse(made.called)
            with mock.patch.dict(os.environ, {"STRATA_MACOS_SENSORS": "1"}):
                self.assertEqual(telemetry.apple_sensors(), "on")
                self.assertTrue(made.called)

    def test_only_the_monitors_reader_gets_the_sensors(self):
        # The private calls start only where the Monitor reads: gpu_reader (the server's VRAM checks, the drafter's sizing) never starts them.
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry, "apple_sensors", return_value="S") as sensors, \
                mock.patch.object(telemetry, "_Metal") as metal:
            telemetry.gpu_reader(0)
            metal.assert_called_once_with(0)
            sensors.assert_not_called()
            metal.return_value.attach_sensors.assert_not_called()
            telemetry.monitor_reader(0)
            sensors.assert_called_once()
            metal.return_value.attach_sensors.assert_called_once_with("S")

    def test_the_monitor_reader_without_the_switch_attaches_nothing(self):
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry, "apple_sensors", return_value=None), \
                mock.patch.object(telemetry, "_Metal") as metal:
            telemetry.monitor_reader(0)
            metal.return_value.attach_sensors.assert_not_called()

    def test_the_monitor_reader_leaves_other_backends_alone(self):
        with mock.patch.object(telemetry.sys, "platform", "darwin"), \
                mock.patch.object(telemetry, "apple_sensors", return_value="S") as sensors, \
                mock.patch.object(telemetry, "_Amd") as amd:
            telemetry.monitor_reader(0, amd=True)
            amd.assert_called_once_with(0)
            sensors.assert_not_called()                                   # a Mac that runs the AMD tests does not start the Apple calls

    def test_off_by_default_the_reader_gains_nothing(self):
        from serve.test_server import IOREG_FIXTURE
        metal = telemetry._Metal(0, run=lambda: IOREG_FIXTURE)           # what gpu_reader gives without the variable
        r = metal.read()
        self.assertTrue(all(k not in r for k in ("temp", "power")))
        self.assertIn("powermetrics", metal.note)


if __name__ == "__main__":
    unittest.main()
