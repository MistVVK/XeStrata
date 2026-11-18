"""#4662: the server says, in one plain line, when a card in use is of an architecture the compiled engine has no code for.

    python -m unittest serve.test_engine_arch_note
"""
import json
import tempfile
import unittest
from pathlib import Path

import serve.server as server
from serve.server_parts import engines

CARDS = {0: 75, 1: 86}


class ArchNote(unittest.TestCase):
    def test_a_card_the_build_does_not_cover_is_said(self):
        line = server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75]})
        self.assertIn("sm_75", line)
        self.assertIn("sm_86", line)
        self.assertIn("update", line)

    def test_a_covered_pair_says_nothing(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75, 86]}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": ["sm_75", "sm_86"]}))

    def test_only_the_cards_the_run_uses_count(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0]}, CARDS, {"archs": [75]}))
        self.assertIsNotNone(server.engine_arch_note({}, CARDS, {"archs": [75]}))     # no "gpu": every card

    def test_a_build_with_ptx_covers_newer_cards(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75], "ptx": True}))

    # The real 0.1.62 Linux packages (UPLOAD2 of 0.2.16) write "archs" as ONE string.  It used to be walked character by character, and
    # p1004 (sm_60) was told "this engine was built for sm_0, sm_1, sm_2, sm_5, sm_6, sm_7, sm_8, sm_9".
    CUDA12 = json.loads('{"version": "0.1.62", "source": "tools/packaging/build_cuda_linux.sh", "archs": "60;61;70;75;80;86;120;89", '
                        '"cuda": "12.9", "portable": true, "vision_archs": "60;61;70;75;80;86;120;89"}')
    CUDA13 = json.loads('{"version": "0.1.62", "source": "tools/packaging/build_cuda_linux.sh", "archs": "75;86;89;120", '
                        '"cuda": "13.0", "portable": true, "vision_archs": "75;86;89;120"}')

    def test_the_real_cuda12_build_json_covers_a_pascal_card(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0]}, {0: 60}, self.CUDA12))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, {0: 60, 1: 86}, self.CUDA12))
        self.assertIsNone(server.engine_arch_note({}, {0: 61, 1: 70, 2: 75, 3: 80, 4: 89, 5: 120}, self.CUDA12))

    def test_the_real_cuda13_build_json_names_the_card_it_lacks(self):
        line = server.engine_arch_note({"gpu": [0]}, {0: 60}, self.CUDA13)
        self.assertIn("built for sm_75, sm_86, sm_89, sm_120", line)
        self.assertIn("card in use is sm_60", line)
        self.assertNotIn("sm_0", line)
        self.assertIsNone(server.engine_arch_note({"gpu": [0]}, {0: 86}, self.CUDA13))

    def test_every_written_form_reads_the_same(self):
        want = [60, 61, 70, 75, 80, 86, 89, 120]
        for form in ("60;61;70;75;80;86;120;89", "60,61,70,75,80,86,120,89", "60 61 70 75 80 86 120 89", " 60; 61 ,70;75;80;86;120;89;",
                     [60, 61, 70, 75, 80, 86, 120, 89], ["sm_60", "61", "70-real", "sm_75", 80, "compute_86", "sm_120", "sm_89"]):
            self.assertEqual(engines.build_arch_numbers(form), want, form)
        self.assertEqual(engines.build_arch_numbers(None), [])
        self.assertEqual(engines.build_arch_numbers(""), [])
        self.assertEqual(engines.build_arch_numbers("x"), [])

    def test_the_note_reads_the_build_json_beside_the_engine(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "BUILD.json").write_text(json.dumps(self.CUDA13), encoding="utf-8")
            exe = str(Path(d) / "strata")
            self.assertIsNotNone(engines.engine_arch_note({"exe": exe, "gpu": [0]}, {0: 60}))
            (Path(d) / "BUILD.json").write_text(json.dumps(self.CUDA12), encoding="utf-8")
            self.assertIsNone(engines.engine_arch_note({"exe": exe, "gpu": [0]}, {0: 60}))

    def test_unknown_things_say_nothing(self):
        self.assertIsNone(server.engine_arch_note({"backend": "hip"}, CARDS, {"archs": [75]}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": ["x"]}))
        self.assertIsNone(server.engine_arch_note({"exe": "component:strata-engine"}))


if __name__ == "__main__":
    unittest.main()
