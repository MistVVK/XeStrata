"""The draft vocabulary follows the computer's languages when the person chose none (strata_control/drafting.py vocab_for_languages, os_language_signals, auto_vocab, apply).

Measured on l620 (RESULTS-DRAFT-VOCAB.md): `cyrillic` Russian +26%, `latin` German / Spanish / French / Polish +8 to +12%, both within 0.3% of the default on English and code, and 12-16% slower on Chinese and
Japanese than the default. So any Chinese / Japanese / Korean signal anywhere keeps the default."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
_HARDWARE = mock.patch.object(drafting, "detect_hardware", return_value=("cuda", ["sm_120"]))


def setUpModule():
    _HARDWARE.start()


def tearDownModule():
    _HARDWARE.stop()


class TheMapping(unittest.TestCase):
    """Signals in, file name out (one pure function)."""

    TABLE = (
        # (ui, region, keyboards) -> choice
        ((["en-US"], ["ru-RU"], []), "cyrillic"),            # an English Windows with a Russian regional format
        ((["de-DE"], ["de-DE"], []), "latin"),
        ((["de-DE"], [], []), "latin"),
        ((["en-US"], ["zh-CN"], []), "cjk"),                 # Chinese region, English display: the default
        ((["de-DE"], [], ["ja-JP"]), "cjk"),                 # a Japanese keyboard beside a German display: the default
        ((["ru-RU"], ["zh-CN"], []), "cjk"),                 # CJK anywhere wins over everything
        ((["en-US"], ["en-US"], ["en-US", "ko-KR"]), "cjk"),
        ((["C"], [], []), "cjk"),
        ((["POSIX"], ["C.UTF-8"], []), "cjk"),
        (([], [], []), "cjk"),
        ((["en-US"], ["en-US"], ["en-US"]), "cjk"),          # English only: the default, untouched
        ((["ar-SA"], ["he-IL"], ["hi-IN", "th-TH"]), "cjk"),   # scripts the files do not cover
        ((["uk"], [], []), "cyrillic"),
        ((["bg"], [], []), "cyrillic"),
        ((["sr-Cyrl"], [], []), "cyrillic"),
        ((["sr-Cyrl-RS"], [], []), "cyrillic"),
        ((["sr"], [], []), "cyrillic"),
        ((["kk"], [], []), "cyrillic"),
        ((["be-BY"], [], []), "cyrillic"),
        ((["sr-Latn"], [], []), "latin"),
        ((["sr_RS@latin"], [], []), "latin"),
        ((["pt-BR"], [], []), "latin"),
        ((["tr"], [], []), "latin"),
        ((["vi"], [], []), "latin"),
        ((["cs"], [], []), "latin"),
        ((["pl_PL.UTF-8"], [], []), "latin"),
        ((["fr_FR@euro"], [], []), "latin"),
        # both kinds present: the display language's script wins, then the regional format, then the keyboards
        ((["de-DE"], ["ru-RU"], []), "latin"),
        ((["ru-RU"], ["de-DE"], []), "cyrillic"),
        ((["en-US"], ["de-DE"], ["ru-RU"]), "latin"),
        ((["en-US"], ["ru-RU"], ["de-DE"]), "cyrillic"),
        ((["en-US"], [], ["de-DE", "ru-RU"]), "latin"),
        ((["en-US", "ru-RU", "de-DE"], [], []), "cyrillic"),
    )

    def test_the_table(self):
        for signals, want in self.TABLE:
            self.assertEqual(drafting.vocab_for_languages(*signals)[0], want, signals)

    def test_the_language_that_decided(self):
        self.assertEqual(drafting.vocab_for_languages(["en-US"], ["ru-RU"]), ("cyrillic", "ru"))
        self.assertEqual(drafting.vocab_for_languages(["de_DE.UTF-8"]), ("latin", "de"))
        self.assertEqual(drafting.vocab_for_languages(["sr-Latn-RS"]), ("latin", "sr"))
        self.assertEqual(drafting.vocab_for_languages(["en-US"], ["zh-CN"]), ("cjk", ""))
        self.assertEqual(drafting.vocab_for_languages(), ("cjk", ""))

    def test_tags(self):
        for tag, want in (("en", ""), ("en-US", ""), ("en-Latn-US", ""), ("C", ""), ("POSIX", ""), ("C.UTF-8", ""), ("", ""), (None, ""), ("zh-Hans", "cjk"), ("zh_TW.UTF-8", "cjk"),
                          ("yue-HK", "cjk"), ("ja", "cjk"), ("ko_KR.eucKR", "cjk"), ("x-Jpan", "cjk"), ("ru", "cyrillic"), ("uz-Cyrl", "cyrillic"), ("uz", "latin"), ("de@euro", "latin"),
                          ("nonsense", "")):
            self.assertEqual(drafting.script_of_language(tag), want, tag)


class ReadingTheComputer(unittest.TestCase):
    def test_linux_variables(self):
        sig = drafting.os_language_signals({"LANG": "ru_RU.UTF-8"}, "Linux")
        self.assertEqual(drafting.vocab_for_languages(**sig)[0], "cyrillic")
        sig = drafting.os_language_signals({"LANG": "en_US.UTF-8", "LC_MESSAGES": "de_DE.UTF-8", "LANGUAGE": "de:en"}, "Linux")
        self.assertEqual(sig["ui"][0], "de")
        self.assertEqual(drafting.vocab_for_languages(**sig)[0], "latin")
        sig = drafting.os_language_signals({"LANG": "en_US.UTF-8", "LC_CTYPE": "ja_JP.UTF-8"}, "Linux")
        self.assertEqual(drafting.vocab_for_languages(**sig)[0], "cjk")
        for env in ({"LANG": "C"}, {"LC_ALL": "POSIX"}, {"LANG": "C.UTF-8"}, {}):
            self.assertEqual(drafting.vocab_for_languages(**drafting.os_language_signals(env, "Linux"))[0], "cjk", env)

    def test_a_failing_read_is_no_signal(self):
        class Broken(dict):
            def get(self, *a, **k):
                raise RuntimeError("no environment")
        self.assertEqual(drafting.os_language_signals(Broken(), "Linux"), {"ui": [], "region": [], "keyboards": []})
        with mock.patch.object(drafting, "_windows_languages", side_effect=OSError("no ctypes")):
            with self.assertRaises(OSError):
                drafting.os_language_signals({}, "Windows")       # (the caller of the whole read, auto_vocab, catches it: see below)

    def test_this_computers_own_read_returns_lists_and_never_raises(self):
        sig = drafting.os_language_signals()
        self.assertEqual(sorted(sig), ["keyboards", "region", "ui"])
        for v in sig.values():
            self.assertIsInstance(v, list)

    @unittest.skipUnless(sys.platform == "win32", "ctypes calls of Windows")
    def test_windows_names_look_like_locale_names(self):
        ui, region, keyboards = drafting._windows_languages()
        for name in ui + region + keyboards:
            self.assertRegex(name, r"^[A-Za-z]{2,3}(-[A-Za-z0-9]+)*$")

    @unittest.skipUnless(sys.platform == "darwin", "the Mac's preferences file")
    def test_mac_preferences(self):
        self.assertIsInstance(drafting.os_language_signals()["ui"], list)


class ThePickAtStart(unittest.TestCase):
    NAME = "qwen3.8-flash-next-iq3_s"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.mtp = self.data / "mtp" / "rt"
        self.mtp.mkdir(parents=True)
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.mtp / "draft_vocab.bin").write_bytes(b"v")
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))

    def start(self, signals, cfg_extra=None, args_extra=(), form=None):
        args = ["--pack", str(self.data / "packs" / "x"), "--native", "/d/m.gguf", "--mtp", str(self.mtp), "--spec", "4", "--spec-min-p", "0.5", *args_extra]
        cfg = {"model_name": self.NAME, "args": args, **(cfg_extra or {})}
        with mock.patch.object(drafting, "computer_language_signals", return_value={"ui": [], "region": [], "keyboards": [], **signals}):
            if form:
                with mock.patch.object(drafting, "engine_form", return_value=form):
                    decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"])
            else:
                decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"])
        return decision, cfg

    def vocab_of(self, cfg):
        a = cfg["args"]
        return Path(a[a.index("--mtp-draft-vocab") + 1])

    def test_nothing_readable_or_english_is_the_default_exactly_as_before(self):
        for signals in ({}, {"ui": ["en-US"], "region": ["en-US"], "keyboards": ["en-US"]}, {"ui": ["C"]}):
            decision, cfg = self.start(signals)
            self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))
            self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin", signals)
            self.assertEqual(decision.vocab, "draft_vocab: " if False else f"draft vocabulary: draft_vocab.bin (the default)")
            self.assertNotIn("draft_vocab", cfg, "nothing is written to the config")

    def test_a_russian_computer_gets_cyrillic_and_a_german_one_latin(self):
        decision, cfg = self.start({"ui": ["en-US"], "region": ["ru-RU"]})
        self.assertEqual(self.vocab_of(cfg), DATA / "draft_vocab_cyrillic.bin")
        self.assertEqual(decision.vocab, "draft vocabulary: draft_vocab_cyrillic.bin (by this computer's language: ru)")
        decision, cfg = self.start({"ui": ["de-DE"]})
        self.assertEqual(self.vocab_of(cfg), DATA / "draft_vocab_latin.bin")
        self.assertEqual(decision.vocab, "draft vocabulary: draft_vocab_latin.bin (by this computer's language: de)")
        self.assertNotIn("draft_vocab", cfg, "the pick is worked out at each start and never saved")
        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))

    def test_a_cjk_signal_keeps_the_default(self):
        for signals in ({"ui": ["en-US"], "region": ["zh-CN"]}, {"ui": ["de-DE"], "keyboards": ["ja-JP"]}, {"ui": ["ru-RU"], "keyboards": ["ko-KR"]}):
            decision, cfg = self.start(signals)
            self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin", signals)

    def test_a_saved_choice_always_wins_also_the_default_one(self):
        for saved in ("cjk", "en", "cyrillic", "fr", "latin"):
            decision, cfg = self.start({"ui": ["ru-RU"]}, {"draft_vocab": saved})
            self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin", saved)        # the copy setup put there; the pick does not touch it
            self.assertEqual(decision.vocab, "draft vocabulary: draft_vocab.bin (set by you)", saved)

    def test_the_persons_own_vocabulary_flag_wins(self):
        mine = self.data / "mine.bin"
        mine.write_bytes(struct.pack("<3i", 1, 2, 3))
        decision, cfg = self.start({"ui": ["ru-RU"]}, args_extra=("--mtp-draft-vocab", str(mine)))
        self.assertEqual(self.vocab_of(cfg), mine)
        self.assertEqual(decision.vocab, "draft vocabulary: mine.bin (set by you)")
        decision, cfg = self.start({"ui": ["ru-RU"]}, {"user_args": ["--mtp-draft-vocab", str(mine)]})
        self.assertEqual(decision.vocab, "draft vocabulary: draft_vocab.bin (set by you)")

    def test_an_engine_that_takes_a_folder_keeps_the_folder(self):
        decision, cfg = self.start({"ui": ["ru-RU"]}, form=drafting.FORM_DIR)
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], str(self.mtp))
        self.assertNotIn("--mtp-draft-vocab", cfg["args"])
        self.assertTrue(decision.vocab.endswith("(the default)"))

    def test_a_missing_or_wrong_file_is_the_default_silently(self):
        with mock.patch.object(drafting, "shipped_vocab", return_value=self.data / "gone.bin"):
            decision, cfg = self.start({"ui": ["ru-RU"]})
        self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin")
        self.assertEqual((decision.drafter, decision.note), ("dflash2", ""))
        wrong = self.data / "wrong.bin"
        wrong.write_bytes(struct.pack("<3i", 1, 2, 999999))       # an id past the model's vocabulary: another tokenizer's file
        with mock.patch.object(drafting, "shipped_vocab", return_value=wrong):
            decision, cfg = self.start({"ui": ["ru-RU"]})
        self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin")
        self.assertEqual(decision.note, "")

    def test_a_read_that_fails_is_the_default(self):
        with mock.patch.object(drafting, "computer_language_signals", side_effect=OSError("no registry")):
            args = ["--pack", "p", "--native", "/d/m.gguf", "--mtp", str(self.mtp)]
            cfg = {"model_name": self.NAME, "args": args}
            decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"])
        self.assertEqual(decision.drafter, "dflash2")
        self.assertEqual(self.vocab_of(cfg), self.mtp / "draft_vocab.bin")

    def test_the_mtp_and_the_other_models_are_not_touched(self):
        decision, cfg = self.start({"ui": ["ru-RU"]}, {"drafting": "mtp"})
        self.assertEqual((decision.drafter, decision.vocab), ("mtp", ""))
        self.assertNotIn("--mtp-draft-vocab", cfg["args"])


class TheShippedFiles(unittest.TestCase):
    def test_every_registered_choice_has_its_file_and_the_file_fits_the_vocabulary(self):
        for choice, name in drafting.VOCAB_FILES.items():
            path = DATA / name
            self.assertTrue(path.is_file(), f"{choice}: data/{name} is not in the repository")
            self.assertTrue(drafting.vocab_ids_fit(path), f"{choice}: data/{name} is not a draft vocabulary of the 248,320-token vocabulary")
        self.assertEqual(drafting.shipped_vocab("latin"), DATA / "draft_vocab_latin.bin")

    def test_ids_that_do_not_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.bin"
            for body, want in ((struct.pack("<3i", 0, 5, 248319), True), (struct.pack("<2i", 5, 248320), False), (struct.pack("<2i", -1, 5), False), (b"\x01\x00\x00", False), (b"", False)):
                p.write_bytes(body)
                self.assertEqual(drafting.vocab_ids_fit(p), want, body)
            self.assertFalse(drafting.vocab_ids_fit(Path(tmp) / "missing.bin"))

    def test_latin_holds_english_and_the_french_subset(self):
        import array

        def ids(name):
            a = array.array("i")
            a.frombytes((DATA / name).read_bytes())
            return set(a)
        latin = ids("draft_vocab_latin.bin")
        self.assertEqual(len(latin), 62102)
        self.assertLessEqual(ids("draft_vocab_en.bin"), latin)
        self.assertLessEqual(ids("draft_vocab_fr.bin"), latin)
        cjk_only = ids("draft_vocab.bin") - ids("draft_vocab_en.bin")                 # the 66,000 Chinese / Japanese / Korean tokens the default adds to English
        self.assertLess(len(latin & cjk_only), 20, "no CJK block in it (a few punctuation tokens that appear in Latin text too may be)")

    def test_setups_choices_name_the_same_files(self):
        sys.path.insert(0, str(ROOT))
        import setup as S                                                    # noqa: PLC0415
        for choice, name in S.DRAFT_VOCABS.items():
            self.assertEqual(drafting.VOCAB_FILES[choice], name, choice)


class ThePackagingLists(unittest.TestCase):
    """data/ goes into every package whole (packaging/package_terminal.py INSTALLER_DATA is the folder, not a list of files), so a choice's file is in the Terminal's installer bundle, the app's Control
    component (app/desktop/scripts/stage-control.py takes the same name) and the Mac's Control. The Mac ENGINE folder (metal/data) carries only the default file, by design: the other choices are read from Control's data/."""

    def test_the_installer_data_is_the_whole_folder_and_leaves_no_registered_file_out(self):
        sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))
        import package_terminal as pt                                        # noqa: PLC0415
        self.assertEqual(pt.INSTALLER_DATA, "data")
        stage = (ROOT.parent / "app" / "desktop" / "scripts" / "stage-control.py").read_text(encoding="utf-8")
        self.assertIn("pt.INSTALLER_DATA", stage)
        source = (ROOT.parent / "tools" / "packaging" / "package_terminal.py").read_text(encoding="utf-8")
        self.assertIn("shutil.copytree(source / INSTALLER_DATA, folder / INSTALLER_DATA", source)
        self.assertNotIn("draft_vocab_", source.replace("draft_vocab\"", ""), "a per-file list would need every registered file named: none exists, the folder is copied")
        for name in drafting.VOCAB_FILES.values():
            self.assertTrue((ROOT / pt.INSTALLER_DATA / name).is_file(), name)

    def test_the_mac_engine_folder_carries_the_default_only(self):
        mac = (ROOT.parent / "tools" / "packaging" / "mac" / "build_engine.sh").read_text(encoding="utf-8")
        self.assertIn("data/draft_vocab.bin", mac)
        self.assertEqual(drafting.VOCAB_FILES["cjk"], "draft_vocab.bin")


if __name__ == "__main__":
    unittest.main()
