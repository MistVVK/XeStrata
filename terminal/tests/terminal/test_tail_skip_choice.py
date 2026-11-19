"""The route tail skip (STRATA_ROUTE_TAIL_SKIP) is a choice in the app's Expert settings and a row of the Terminal's Advanced table.
0.2.16: Automatic (no entry: the engine's own rule), Off (0) or a rank 1-9.  0.2.17 (owner 11-19): Off is the default (no entry; every engine start sets 0,
test_tail_skip_default_off.py), On is 7 and is written as STRATA_ROUTE_TAIL_SKIP=7, the other ranks 1-6, 8, 9 are numbers.  One list in
strata_control/speed_switches.py serves both."""
import unittest

from strata_control import app_policy, speed_switches as ss

CUDA = [{"name": "NVIDIA GeForce RTX 5070", "vendor": "nvidia", "arch": "120", "vram_gb": 12.0}]
R9700 = [{"name": "AMD Radeon AI PRO R9700", "vendor": "amd", "arch": "gfx1201", "vram_gb": 32.0}]
RX7900 = [{"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "arch": "gfx1100", "vram_gb": 24.0}]
NOTE = "Lower numbers skip more parts and are faster; answers can change more. Only 7 has been measured."


class TailSkipChoice(unittest.TestCase):
    def test_the_row_is_shown_where_the_engine_reads_the_setting(self):
        shows = lambda gpus, backend, cfg=None: "route_tail_skip" in [s.id for s in ss.applicable(cfg or {}, gpus, backend)]   # noqa: E731
        self.assertTrue(shows(CUDA, "cuda"))
        self.assertTrue(shows(CUDA, "cuda", {"arch": "qwen35moe"}))
        self.assertTrue(shows(R9700, "hip"))
        self.assertFalse(shows(RX7900, "hip"))
        for backend in ("metal", "sycl", "cpu", None):
            self.assertFalse(shows(CUDA, backend), backend)

    def test_off_is_the_default_and_on_writes_seven(self):
        sw = ss.BY_ID["route_tail_skip"]
        self.assertEqual(sw.default, "off")
        self.assertEqual(dict(sw.env), {"STRATA_ROUTE_TAIL_SKIP": "7"})
        self.assertFalse(ss.is_on({}, sw))                                            # no entry: Off (every start sets 0)
        self.assertFalse(ss.is_on({"STRATA_ROUTE_TAIL_SKIP": "0"}, sw))               # setup's 0 and the person's own 0 are Off
        self.assertTrue(ss.is_on({"STRATA_ROUTE_TAIL_SKIP": "7"}, sw))
        self.assertEqual(ss.set_switch({"OTHER": "1"}, sw, True), {"OTHER": "1", "STRATA_ROUTE_TAIL_SKIP": "7"})
        self.assertEqual(ss.set_switch({"STRATA_ROUTE_TAIL_SKIP": "7", "OTHER": "1"}, sw, False), {"OTHER": "1"})

    def test_the_choices_and_the_terminal_sentence(self):
        sw = ss.BY_ID["route_tail_skip"]
        self.assertEqual(sw.choices, (("7", "On"),) + tuple((str(r), str(r)) for r in (1, 2, 3, 4, 5, 6, 8, 9)))
        self.assertEqual(sw.terminal_default, "")                                     # the Default cell says "off", the switch's own default
        self.assertEqual(ss.env_words("STRATA_ROUTE_TAIL_SKIP"),
                         "Skip rarely used parts of the model (off by default; set 7 to turn it on, or 1 to 9 to choose how much is skipped). "
                         "Faster answers: parts of the model that matter least for a word are skipped when they are not on the graphics card yet. "
                         "Answers can change slightly. Off by default. " + NOTE)

    def test_the_app_table_carries_the_choices_in_order(self):
        row = next(r for r in app_policy.table() if r["id"] == "route_tail_skip")
        self.assertEqual(row["env"], {"STRATA_ROUTE_TAIL_SKIP": "7"})
        self.assertEqual(row["default"], "off")
        self.assertEqual(row["words"], {"on": "On", "off": "Off (default)"})
        self.assertEqual([v for v, _ in row["choices"]], ["7", "1", "2", "3", "4", "5", "6", "8", "9"])
        self.assertEqual(row["choiceNote"], NOTE)

    def test_the_other_switches_keep_their_two_state_sentence(self):
        self.assertEqual(ss.env_words("STRATA_KVA").split(". ")[0], "Faster long prompts (off by default; set 1 to turn on)")


if __name__ == "__main__":
    unittest.main()
