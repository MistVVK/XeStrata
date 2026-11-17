"""The draft-vocabulary subset a person chose in setup (--draft-vocab en / fr / cyrillic) reaches DFlash2, which is the default drafter: on an install that has NO MTP folder
(a fresh DFlash2-only install) as well as on one that has it.  Pure file work in temporary folders, no GPU, no downloads, no prompts.

    python -m unittest tools.test_dflash2_vocab -v

What is checked, per choice (cjk = the default file, en, cyrillic, fr):
  - setup's vocab step (provide_dflash2_vocab / refresh_draft_vocab / refresh_shared_draft_vocab) and then strata_control.drafting.apply: the file `--mtp-draft-vocab` points at is the chosen subset
    byte for byte, for an explicit `dflash2` and for `auto`;
  - a later choice replaces a shipped subset, a subset made by hand stays;
  - an update (refresh_shared_draft_vocab) and the drafter's own fetch (provide_vocab without a choice) keep the person's choice.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import drafting  # noqa: E402

CHOICES = ("cjk", "en", "cyrillic", "fr")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shipped(choice: str) -> Path:
    return ROOT / "data" / setup.DRAFT_VOCABS[choice]


def vocab_arg(cfg: dict) -> Path:
    a = cfg["args"]
    return Path(a[a.index("--mtp-draft-vocab") + 1])


@unittest.skipUnless(all(shipped(c).exists() for c in CHOICES), "the shipped draft subsets are not in this checkout")
class Base(unittest.TestCase):
    with_mtp = False

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name) / "data"
        (self.data / "packs" / "m").mkdir(parents=True)
        (self.data / "dflash2").mkdir(parents=True)
        (self.data / "dflash2" / drafting.PACK_NAME).write_bytes(b"\0" * ((1 << 20) + 16))   # "downloaded" (size is all drafting.pack_ready looks at)
        self.rt = self.data / "mtp" / "rt"
        if self.with_mtp:
            self.rt.mkdir(parents=True)
            (self.rt / "experts.bin").write_bytes(b"e")
            (self.rt / "draft_vocab.bin").write_bytes(shipped("cjk").read_bytes())      # what an install of the MTP layer holds first

    # ---- what setup does for a model with this draft_vocab choice
    def setup_step(self, choice: str) -> None:
        """setup.py's vocab step on an install with a DFlash2 drafter: the DFlash2 copy (beside the pack) and, when there is an MTP folder, the folder's file."""
        takes_choice = "choice" in inspect.signature(setup.provide_dflash2_vocab).parameters
        if takes_choice:
            setup.provide_dflash2_vocab(self.data, choice)
        else:
            setup.provide_dflash2_vocab(self.data)               # the code before this branch
        if (self.rt / "experts.bin").exists():
            setup.refresh_draft_vocab(self.rt, choice)

    def config(self, mode: str = "dflash2", choice: str | None = None) -> dict:
        args = ["--pack", str(self.data / "packs" / "m"), "--native", "x.gguf", "--mtp", str(self.rt), "--spec", "4", "--spec-min-p", "0.5"]
        cfg = {"args": args, "model_name": "qwen3.8-flash-next-iq3_s", "drafting": mode}
        if choice:
            cfg["draft_vocab"] = choice
        return cfg

    def applied(self, mode: str = "dflash2") -> dict:
        cfg = self.config(mode)
        d = drafting.apply(cfg, data_dir=self.data, family="qwen", backend="cuda", archs=["sm_121"], slots=0)
        self.assertEqual(d.drafter, "dflash2", d)
        return cfg


class DFlash2Only(Base):
    with_mtp = False

    def test_each_choice_reaches_the_engine(self):
        for choice in CHOICES:
            with self.subTest(choice=choice):
                self.setup_step(choice)
                self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped(choice)), f"--mtp-draft-vocab is not the {choice} subset")

    def test_auto_uses_the_choice(self):
        for choice in CHOICES:
            with self.subTest(choice=choice):
                self.setup_step(choice)
                self.assertEqual(sha(vocab_arg(self.applied("auto"))), sha(shipped(choice)))

    def test_a_later_choice_replaces_a_shipped_subset_and_a_hand_made_one_stays(self):
        self.setup_step("fr")
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("fr")))
        self.setup_step("en")
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("en")))
        self.setup_step("cjk")
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("cjk")))
        mine = drafting.vocab_target(self.data)
        mine.write_bytes(bytes([1, 0, 0, 0]))
        self.setup_step("fr")
        self.assertEqual(mine.read_bytes(), bytes([1, 0, 0, 0]), "a subset made by hand is kept")

    def test_an_update_and_the_drafters_fetch_keep_the_choice(self):
        self.setup_step("fr")
        cfg = self.config(choice="fr")
        setup.refresh_shared_draft_vocab(cfg, set())                       # setup.py --update, the saved config names fr
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("fr")))
        drafting.provide_vocab(self.data)                                  # the fetch (Load, `strata drafting fetch`) never overwrites a copy that is there
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("fr")))

    def test_an_update_applies_the_saved_choice_to_an_older_copy(self):
        drafting.provide_vocab(self.data)                                  # an install made before the choice reached DFlash2: the default copy
        self.assertEqual(sha(drafting.vocab_target(self.data)), sha(shipped("cjk")))
        setup.refresh_shared_draft_vocab(self.config(choice="cyrillic"), set())
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("cyrillic")))

    def test_no_choice_means_the_default_file(self):
        setup.refresh_shared_draft_vocab(self.config(), set())             # no "draft_vocab" key: nothing to apply, nothing broken
        drafting.provide_vocab(self.data)
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("cjk")))


class WithMtpFolder(Base):
    with_mtp = True

    def test_each_choice_reaches_the_engine(self):
        for choice in CHOICES:
            with self.subTest(choice=choice):
                self.setup_step(choice)
                self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped(choice)))
                self.assertEqual(sha(self.rt / "draft_vocab.bin"), sha(shipped(choice)), "the MTP layer's file follows too")
                if drafting.vocab_target(self.data).exists():
                    self.assertEqual(sha(drafting.vocab_target(self.data)), sha(shipped(choice)), "the DFlash2 copy agrees with the folder")

    def test_auto_uses_the_choice(self):
        self.setup_step("fr")
        self.assertEqual(sha(vocab_arg(self.applied("auto"))), sha(shipped("fr")))

    def test_the_two_files_never_disagree_after_an_update(self):
        self.setup_step("en")
        setup.refresh_shared_draft_vocab(self.config(choice="fr"), set())
        self.assertEqual(sha(self.rt / "draft_vocab.bin"), sha(shipped("fr")))
        if drafting.vocab_target(self.data).exists():
            self.assertEqual(sha(drafting.vocab_target(self.data)), sha(shipped("fr")))
        self.assertEqual(sha(vocab_arg(self.applied())), sha(shipped("fr")))


if __name__ == "__main__":
    unittest.main()
