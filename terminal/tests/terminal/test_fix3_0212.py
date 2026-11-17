"""FIX3 (11-04): three launcher and planning bugs found in round 0211.

1. The draft-layer lookup walked up from the model file to a folder named "models" and looked only beside it. With the model folder outside the data
   folder the plan asked for the 5.2 GB layer although <data-dir>/mtp/rt was there (a second download).
2. The all-cards split engine's fit counted each card's size. Cards other programs hold (8 GB free of 96) still got "split + MTP" and the 5.2 GB fetch; the fit
   counts free memory minus the reserve now.
3. The plan's card line gave no way to tell the --gpu number (GPU 2 is card 1) and listed cards CUDA_VISIBLE_DEVICES hides. It counts from 1 like the screens and now adds the number --gpu takes:
   "GPU 2 (--gpu 1)".
All of it with fake cards and fake sysfs; nothing is started, probed or downloaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import cmd as launch_cmd
from strata_cli.launch import modelstate
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from strata_control import discovery, gpu_roles, split_engine as SE
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli
from tests.terminal.test_own_files import IQ3XXS, gguf_bytes


class Base(unittest.TestCase):
    one_card_pc = True                                  # the plan tests run on a fake one-card PC; the others hand their own hardware to catalog()

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        if self.one_card_pc:
            p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
            p.start()
            self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-fix3-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def model_files(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in IQ3XXS:
            (folder / n).write_bytes(gguf_bytes(["IQ3_XXS"]))
        return folder

    def layer_in(self, data: Path) -> Path:
        rt = data / "mtp" / "rt"
        rt.mkdir(parents=True, exist_ok=True)
        (rt / "experts.bin").write_bytes(b"x")
        return rt

    def plan_text(self, *args):
        code, out, err = run_cli(["launch", "--plan", *args])
        self.assertEqual(code, 0, err or out)
        return out


# ----------------------------------------------------------------------------------------------------------------------- 1. the draft layer

class DraftLayerInTheDataDir(Base):
    def test_prepared_beside_looks_in_the_data_dir_when_the_model_folder_is_elsewhere(self):
        data = self.base / "data"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertIsNone(discovery.prepared_beside(first)["mtp"], "the old lookup: only beside the models folder")
        got = discovery.prepared_beside(first, [data])
        self.assertEqual(got["mtp"], rt)
        self.assertEqual(got["data_dir"], self.base / "ext", "the folder the model sits in is still what data_dir says")

    def test_a_model_folder_with_no_models_ancestor_finds_it_too(self):
        data = self.base / "data"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "somewhere" / "IQ3_XXS") / IQ3XXS[0]
        got = discovery.prepared_beside(first, [None, self.base / "nope", data])
        self.assertEqual((got["mtp"], got["data_dir"]), (rt, None))

    def test_the_layer_beside_the_models_folder_wins_and_nothing_is_invented(self):
        beside, data = self.base / "ext", self.base / "data"
        mine = self.layer_in(beside)
        self.layer_in(data)
        first = self.model_files(beside / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertEqual(discovery.prepared_beside(first, [data])["mtp"], mine)
        none = discovery.prepared_beside(self.model_files(self.base / "lone" / "models" / "x") / IQ3XXS[0], [self.base / "empty"])
        self.assertEqual((none["mtp"], none["pack"]), (None, None))

    def test_a_pack_in_the_data_dir_is_found_by_the_models_tag(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        self.assertIsNone(discovery.prepared_beside(first)["pack"])
        self.assertEqual(discovery.prepared_beside(first, [data])["pack"], data / "packs" / "iq3_xxs")

    def test_assess_counts_no_draft_layer_when_data_dir_has_it(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        c = catalog(HW_12)
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "elsewhere"):
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=data)
            self.assertEqual([p.kind for p in st.parts], [])
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=self.base / "empty")
            self.assertEqual([p.kind for p in st.parts], [], "no pack anywhere: the draft layer is not counted either (as before)")
            (self.base / "empty" / "packs" / "iq3_xxs").mkdir(parents=True)
            st = modelstate.assess(c, "qwen", "IQ3_XXS", vision="off", model_path=str(folder), data_dir=self.base / "empty")
        self.assertEqual([p.kind for p in st.parts], ["mtp"], "a pack but no layer anywhere: it is still asked for")

    def test_assess_finds_the_picture_reader_in_the_data_dir_too(self):
        data = self.base / "data"
        (data / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "elsewhere"):
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="gpu", model_path=str(folder), data_dir=data)
            self.assertEqual([p.kind for p in st.parts], ["mmproj"], "a pack in the data dir but no reader anywhere: the reader is asked for")
            (data / "models").mkdir()
            (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="gpu", model_path=str(folder), data_dir=data)
        self.assertEqual([p.kind for p in st.parts], [])

    def test_the_remembered_data_dir_counts_when_none_is_given(self):
        remembered = self.base / "remembered"
        (remembered / "packs" / "iq3_xxs").mkdir(parents=True)
        self.layer_in(remembered)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=remembered):
            st = modelstate.assess(catalog(HW_12), "qwen", "IQ3_XXS", vision="off", model_path=str(folder))
        self.assertEqual([p.kind for p in st.parts], [])

    # the plan as `strata launch --plan` prints it (check (a))
    def test_plan_says_nothing_to_download_for_a_model_folder_outside_the_data_dir(self):
        data = self.base / "data"
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None), mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "off")
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_plan_with_a_models_ancestor_outside_the_data_dir_is_the_same(self):
        data = self.base / "data"
        self.layer_in(data)
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "off")
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_plan_still_asks_for_the_layer_when_no_data_dir_has_it(self):
        folder = self.model_files(self.base / "ext" / "models" / "iq3_xxs")
        with mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"),                 mock.patch.object(Runner, "intel_chosen", lambda self: False):       # an Intel Arc in the box that runs the test would plan its own engine
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(self.base / "data"), "--vision", "off")
        self.assertIn("About 5.2 GB to download: the draft layer (5.2 GB).", out)

    def test_the_picture_reader_in_the_data_dir_is_found_too(self):
        data = self.base / "data"
        self.layer_in(data)
        (data / "models").mkdir()
        (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
        folder = self.model_files(self.base / "ext" / "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None), mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--data-dir", str(data), "--vision", "on")
        self.assertIn("download  Nothing to download", out)

    def test_the_import_preview_sees_the_layer_in_the_data_folder_in_use(self):
        from strata_control import models
        data = self.base / "inuse"
        rt = self.layer_in(data)
        first = self.model_files(self.base / "ext" / "models" / "iq3_xxs") / IQ3XXS[0]
        with mock.patch.object(discovery, "remembered_data_dirs", return_value=[data]):
            got = models._already_set_up([first], {"family": "qwen", "model": "IQ3_XXS"})
        self.assertEqual(got["mtp"], str(rt))

    def test_remembered_data_dirs_reads_only_the_folders_that_exist(self):
        with mock.patch.object(discovery, "data_dir_choices", return_value=[(self.base, "settings"), (self.base / "gone", "app"), (self.base, "documents")]):
            self.assertEqual(discovery.remembered_data_dirs(), [self.base])

    def test_dflash2_pack_is_looked_for_in_the_data_dir_that_was_named(self):
        """The DFlash2 lookup was never tied to the model folder: it asks the data dir (the one named, else the run config's)."""
        from strata_control import drafting
        data = self.base / "data"
        pack = drafting.target_path(data)
        pack.parent.mkdir(parents=True)
        with pack.open("wb") as f:
            f.truncate((1 << 20) + 1)
        self.assertTrue(drafting.pack_ready(data))
        self.assertFalse(drafting.pack_ready(self.base / "ext"))
        r = Runner(S.LaunchSettings(model_path=str(self.model_files(self.base / "ext" / "IQ3_XXS")), data_dir=str(data), drafting="dflash2", vision="off"),
                   catalog=catalog(HW_12))
        with mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_86"])):
            want = r.drafting_inputs({})
        self.assertEqual(Path(want["data_dir"]), data)
        self.assertFalse(drafting.needs_pack(**want))


# ----------------------------------------------------------------------------------------------------------------------- 2. free memory

def card(i, total=98304, free=None, **kw):
    spec = dict(name="NVIDIA RTX PRO 6000", vram_mib=total, cc=120, free_mib=free)
    spec.update(kw)
    return SE.Card(i, **spec)


GIB = 1 << 30
UDQ4 = SE.ModelFacts(routed_types={"Q4_K": 97, "Q5_K": 2, "Q5_1": 43, "Q8_0": 5}, routed_bytes=int(71.7 * GIB), embd_type="Q8_0", shexp_types={"Q8_0": 144},
                     embd_ram_bytes=int(27.5 * GIB), layers=48, experts=512)


def decide(cards, model=UDQ4, **kw):
    return SE.decide(enabled=True, system="linux", cards=cards, ram_total_gib=500.0, ram_available_gib=400.0, model=model, **kw)


class SplitFitCountsFreeMemory(unittest.TestCase):
    def test_free_cards_decide_as_before(self):
        before = decide([card(0), card(1)])                                   # free not known: the size counts
        for free in (98304, 98000, 97500):                                    # an idle card shows a few hundred MiB in use
            d = decide([card(0, free=free), card(1, free=free)])
            self.assertEqual((d.use_split, d.reasons, d.need_vram_gib), (before.use_split, before.reasons, before.need_vram_gib), free)
        self.assertTrue(before.use_split)

    def test_cards_held_to_8_gb_free_do_not_fit_the_experts(self):
        d = decide([card(0, free=8192), card(1, free=8192)])
        self.assertFalse(d.use_split)
        self.assertTrue(any("do not fit in the cards' memory" in r for r in d.reasons), d.reasons)
        self.assertIn("has 7 GiB free)", " ".join(d.reasons), "free memory minus the 700 MiB reserve, and it says free")
        old = decide([card(0, total=4096), card(1, total=4096)])
        self.assertIn("has 4 GiB)", " ".join(old.reasons), "free memory not read: the old sentence, the size")

    def test_one_busy_card_is_enough(self):
        self.assertFalse(decide([card(0), card(1, free=8192)]).use_split)

    def test_the_limit_is_free_minus_the_reserve(self):
        need = decide([card(0), card(1)]).need_vram_gib * 1024
        edge = int(need) + SE.FREE_RESERVE_MIB
        self.assertTrue(decide([card(0, free=edge + 8), card(1, free=edge + 8)]).use_split)
        self.assertFalse(decide([card(0, free=edge - 64), card(1, free=edge - 64)]).use_split)

    def test_free_cannot_raise_a_card_above_its_size(self):
        c = card(0, total=1000, free=5000)
        self.assertEqual(c.room_mib, 1000 - SE.FREE_RESERVE_MIB)
        self.assertEqual(card(0, total=1000).room_mib, 1000, "not known: the size")

    def test_expected_is_false_on_busy_cards_and_true_on_free_ones(self):
        ram = lambda: (500.0, 400.0)                                            # noqa: E731
        probe = lambda cards_: (lambda idx: cards_)                              # noqa: E731
        self.assertTrue(SE.expected({}, probe=probe([card(0, free=90000), card(1, free=90000)]), ram=ram, system="linux"))
        self.assertFalse(SE.expected({}, probe=probe([card(0, free=8192), card(1, free=8192)]), ram=ram, system="linux"), "even a model not on disk needs the 9 GiB")
        self.assertTrue(SE.expected({}, probe=probe([card(0), card(1)]), ram=ram, system="linux"))

    def test_a_busy_start_is_quiet_in_auto_and_told_with_on(self):
        busy = [card(0, free=8192), card(1, free=8192)]
        for m, quiet in (("auto", True), ("on", False)):
            cfg = {"args": ["--native", "/m/x.gguf", "--mtp", "/m/mtp/rt"], "exe": "/e/strata", "split_engine": m}
            with mock.patch.object(SE, "scan_model", lambda p: UDQ4):
                d = SE.apply_to_config(cfg, exe_for_cards=lambda n, cc=None: "/e/strata-split-2", system="linux", probe=lambda i: busy, ram=lambda: (500.0, 400.0))
            self.assertEqual((d.use_split, d.quiet), (False, quiet), m)
            self.assertNotIn("split_fallback", cfg, "the run config stays the layer split")

    def test_the_nvidia_probe_reads_memory_free_in_the_same_call(self):
        out = "0, NVIDIA RTX PRO 6000, 97887, 12.0, 8123\n1, NVIDIA RTX PRO 6000, 97887, 12.0, [N/A]\n"
        seen = []

        def fake(argv, **kw):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        with mock.patch("subprocess.run", fake), mock.patch.dict(os.environ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            got = SE.probe_nvidia_cards()
        self.assertEqual(len(seen), 1, "one nvidia-smi call, as before")
        self.assertIn("memory.free", seen[0][1])
        self.assertEqual([(c.index, c.vram_mib, c.free_mib) for c in got], [(0, 97887, 8123), (1, 97887, None)])

    def test_an_old_four_column_answer_still_counts_the_size(self):
        out = "0, NVIDIA RTX PRO 6000, 97887, 12.0\n1, NVIDIA RTX PRO 6000, 97887, 12.0\n"
        with mock.patch("subprocess.run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")), mock.patch.dict(os.environ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            got = SE.probe_nvidia_cards()
        self.assertEqual([c.free_mib for c in got], [None, None])
        self.assertEqual(len(got), 2)

    def test_the_amd_probe_reads_the_render_nodes_use(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for n, (minor, used) in enumerate(((128, 30 * (1 << 30)), (129, 1 << 20))):
                node = root / f"class/kfd/kfd/topology/nodes/{n + 1}"
                (node / "mem_banks/0").mkdir(parents=True)
                (node / "properties").write_text(f"simd_count 128\ngfx_target_version 120001\nmarketing_name R9700\ndrm_render_minor {minor}\n")
                (node / "mem_banks/0/properties").write_text(f"size_in_bytes {32 * (1 << 30)}\n")
                dev = root / f"class/drm/renderD{minor}/device"
                dev.mkdir(parents=True)
                (dev / "mem_info_vram_used").write_text(str(used))
            (root / "class/kfd/kfd/topology/nodes/0").mkdir(parents=True)
            (root / "class/kfd/kfd/topology/nodes/0/properties").write_text("simd_count 0\n")
            with mock.patch.dict(os.environ):
                for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
                    os.environ.pop(k, None)
                got = SE.probe_amd_cards(root=str(root))
        self.assertEqual([(c.index, c.vram_mib, c.free_mib) for c in got], [(0, 32768, 2048), (1, 32768, 32767)])

    def test_the_amd_probe_without_a_render_node_leaves_free_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            node = Path(t) / "class/kfd/kfd/topology/nodes/0"
            (node / "mem_banks/0").mkdir(parents=True)
            (node / "properties").write_text("simd_count 128\ngfx_target_version 120001\n")
            (node / "mem_banks/0/properties").write_text(f"size_in_bytes {32 * (1 << 30)}\n")
            with mock.patch.dict(os.environ):
                for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
                    os.environ.pop(k, None)
                got = SE.probe_amd_cards(root=t)
        self.assertEqual([c.free_mib for c in got], [None])


class PlanOnBusyCards(Base):
    """The plan: busy cards get the layer split with DFlash2 and no 5.2 GB fetch; free cards still get the split with the MTP."""
    one_card_pc = False

    def runner(self, probe, **kw):
        s = S.LaunchSettings(**{"family": "qwen", "model": "UD-Q4_K_XL", "context": 131072, "tune": "skip", "vision": "off", "gpus": [0, 1], **kw})
        r = Runner(s, catalog=catalog({**HW_12, "ram_gb": 125.0, "engine_vendor": "nvidia",
                                       "gpus": [{"index": i, "name": "NVIDIA RTX PRO 6000", "vram_gb": 96.0, "usable": True, "vendor": "nvidia"} for i in range(2)]}))
        for p in (mock.patch.object(SE, "probe_cards", lambda idx=None: probe), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)),
                  mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"])),
                  mock.patch("sys.platform", "linux"),       # the all-cards split rule asks the host OS (Linux only): this fake PC is Linux
                  mock.patch.object(Runner, "intel_chosen", lambda self: False)):       # and NVIDIA: the real box's Intel Arc must not make the launch an Intel one
            p.start()
            self.addCleanup(p.stop)
        return r

    def test_busy_cards_resolve_to_dflash2_and_fetch_no_draft_layer(self):
        r = self.runner([card(0, free=8192), card(1, free=8192)])
        self.assertFalse(r.split_expected())
        self.assertEqual(r.resolved_drafting("qwen"), "dflash2")
        kinds = [p["kind"] for p in r.download_plan()["parts"]]
        self.assertNotIn("mtp", kinds)
        self.assertNotIn("MTP", r.drafting_plan_line())

    def test_free_cards_still_resolve_to_the_split_with_the_mtp(self):
        r = self.runner([card(0, free=97000), card(1, free=97000)])
        self.assertTrue(r.split_expected())
        self.assertEqual(r.resolved_drafting("qwen"), "mtp")
        self.assertIn("mtp", [p["kind"] for p in r.download_plan()["parts"]])
        self.assertEqual(r.drafting_plan_line(), "Drafting: MTP (the all-cards split engine uses the MTP draft layer, 5.2 GB)")

    def test_a_running_server_of_this_launch_is_not_busy_memory(self):
        """build_config runs before the old server is stopped: the memory it holds is given back, so it must not turn the split down on a restart."""
        busy = [card(0, free=8192), card(1, free=8192)]
        r = self.runner(busy)
        self.assertIsNone(r.card_probe(), "nothing of ours is running: the engine's own probe")
        with mock.patch.object(Runner, "serving_launches", lambda self: [{"profile": self.profile, "name": "x", "port": 1, "choices": None}]):
            r = self.runner(busy)
            probe = r.card_probe()
            self.assertTrue(r.split_expected())
        self.assertEqual([c.free_mib for c in probe(None)], [None, None])

    def test_another_launch_on_another_port_is_busy_memory(self):
        r = self.runner([card(0, free=8192), card(1, free=8192)])
        with mock.patch.object(Runner, "serving_launches", lambda self: [{"profile": "launch-other", "name": "other", "port": int(self.s.port) + 1, "choices": None}]):
            self.assertIsNone(r.card_probe())


class AppPlanWhileItsEngineRuns(Base):
    """The desktop app asks `drafting plan` before it unloads the running model: `--engines-unloading` says that engine's memory is given back, as Runner.card_probe does."""
    one_card_pc = False

    def plan(self, *extra):
        data = self.base / "d"
        data.mkdir(exist_ok=True)
        busy = [card(0, free=8192), card(1, free=8192)]
        with mock.patch.object(SE, "probe_cards", lambda idx=None: busy), mock.patch.object(SE, "ram_gib", lambda: (500.0, 400.0)),                 mock.patch("strata_control.drafting.detect_hardware", lambda: ("cuda", ["sm_120"])),                 mock.patch("sys.platform", "linux"),                 mock.patch.object(Runner, "intel_chosen", lambda self: False):       # the split rule asks the host OS (Linux only) and the make: this fake PC is Linux and NVIDIA
            code, out, err = run_cli(["drafting", "plan", "--family", "qwen", "--data-dir", str(data), "--mode", "auto", "--output", "json", *extra])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_busy_cards_say_dflash2_unless_the_running_engine_is_about_to_be_unloaded(self):
        p = self.plan()
        self.assertEqual((p["split"], p["choice"]), (False, "dflash2"))
        p = self.plan("--engines-unloading")
        self.assertEqual((p["split"], p["choice"]), (True, "mtp"))


# ----------------------------------------------------------------------------------------------------------------------- 3. the card numbers

def four(vendor="nvidia"):
    return [{"index": i, "name": "NVIDIA RTX A6000", "vram_gb": 48.0, "usable": True, "vendor": vendor} for i in range(4)]


class CardNumbers(Base):
    one_card_pc = False

    def line(self, env=None, vendor="nvidia", cards=None, **kw):
        s = S.LaunchSettings(**{"family": "qwen", "model": "UD-Q4_K_XL", **kw})
        c = catalog({**HW_12, "engine_vendor": vendor, "gpus": cards or four(vendor=vendor)})
        with mock.patch.dict(os.environ, env or {}):
            return launch_cmd.roles_line(s, c)

    def test_describe_keeps_the_screens_count_and_the_plan_adds_the_gpu_flag_number(self):
        cs = [{"index": 0, "name": "RTX 3060", "vram_gb": 12}, {"index": 1, "name": "RTX 4070 Ti", "vram_gb": 12}]
        self.assertEqual(gpu_roles.describe({0: "helper", 1: "main"}, cs), "Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.")
        self.assertEqual(gpu_roles.describe({0: "helper", 1: "main"}, cs, card_numbers=True), "Main: GPU 2 (--gpu 1) RTX 4070 Ti. Helper: GPU 1 (--gpu 0) RTX 3060.")

    def test_gpu_0_1_on_four_cards_with_cuda_visible_devices_0_1(self):
        got = self.line({"CUDA_VISIBLE_DEVICES": "0,1"}, gpus=[0, 1])
        self.assertEqual(got, "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000.", "no Off cards for the ones the environment hides")

    def test_without_the_variable_the_other_cards_are_off_with_their_own_numbers(self):
        got = self.line({}, gpus=[0, 1])
        self.assertEqual(got, "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000. Off: GPU 3 (--gpu 2) RTX A6000, GPU 4 (--gpu 3) RTX A6000.")

    def test_one_visible_card_named_by_number(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "1"}, gpus=[1]), "Main: GPU 2 (--gpu 1) RTX A6000.")

    def test_a_named_card_the_environment_hides_stays_because_the_server_sets_visibility_from_it(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "1"}, gpus=[0]), "Main: GPU 1 (--gpu 0) RTX A6000. Off: GPU 2 (--gpu 1) RTX A6000.")

    def test_roles_by_number(self):
        got = self.line({"CUDA_VISIBLE_DEVICES": "2,3"}, gpu_roles={"2": "helper", "3": "main"})
        self.assertEqual(got, "Main: GPU 4 (--gpu 3) RTX A6000. Helper: GPU 3 (--gpu 2) RTX A6000.")

    def test_amd_honours_hip_and_rocr(self):
        amd = four(vendor="amd")
        self.assertEqual(self.line({"HIP_VISIBLE_DEVICES": "0,1"}, vendor="amd", cards=amd, gpus=[0, 1]), "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000.")
        # ROCR first (cards 1 and 3 of the box), then HIP numbers those: HIP 0 is box card 1
        self.assertEqual(self.line({"ROCR_VISIBLE_DEVICES": "1,3", "HIP_VISIBLE_DEVICES": "0"}, vendor="amd", cards=amd, gpus=[1]), "Main: GPU 2 (--gpu 1) RTX A6000.")
        self.assertEqual(self.line({"ROCR_VISIBLE_DEVICES": "1,3"}, vendor="amd", cards=amd, gpus=[1, 3]), "Main: GPU 4 (--gpu 3) RTX A6000. Helper: GPU 2 (--gpu 1) RTX A6000.")

    def test_cuda_visible_devices_does_not_hide_amd_cards(self):
        self.assertEqual(self.line({"CUDA_VISIBLE_DEVICES": "0"}, vendor="amd", cards=four(vendor="amd"), gpus=[0, 1]),
                         "Main: GPU 2 (--gpu 1) RTX A6000. Helper: GPU 1 (--gpu 0) RTX A6000. Off: GPU 3 (--gpu 2) RTX A6000, GPU 4 (--gpu 3) RTX A6000.")


class EnvVisible(unittest.TestCase):
    def test_numbers_and_order(self):
        ev = lambda v, env, **kw: gpu_roles.env_visible([0, 1, 2, 3], v, env, **kw)             # noqa: E731
        self.assertIsNone(ev("nvidia", {}))
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": "1,0"}, alike=True), [1, 0])
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": ""}), [])
        self.assertEqual(ev("nvidia", {"CUDA_VISIBLE_DEVICES": "0,9,1"}, alike=True), [0], "an entry that is no card ends the list, as CUDA reads it")

    def test_a_uuid_or_an_ambiguous_order_is_not_guessed(self):
        self.assertIsNone(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "GPU-1234"}, alike=True))
        self.assertIsNone(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "1"}), "different cards, CUDA's own order: not the nvidia-smi numbers")
        self.assertEqual(gpu_roles.env_visible([0, 1], "nvidia", {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}), [1])
        self.assertIsNone(gpu_roles.env_visible([0, 1], "amd", {"ROCR_VISIBLE_DEVICES": "GPU-1234"}))
        self.assertIsNone(gpu_roles.env_visible([0, 1], "intel", {"CUDA_VISIBLE_DEVICES": "0"}))


class SplitProbeRespectsTheEnvironment(unittest.TestCase):
    def probe(self, env, indexes=None):
        smi = "".join(f"{i}, NVIDIA RTX A6000, 49140, 8.6, 48000\n" for i in range(4))
        with mock.patch("subprocess.run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=smi, stderr="")), mock.patch.dict(os.environ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.update(env)
            return [c.index for c in SE.probe_nvidia_cards(indexes)]

    def test_hidden_cards_are_not_counted_when_none_is_named(self):
        self.assertEqual(self.probe({"CUDA_VISIBLE_DEVICES": "0,1"}), [0, 1])
        self.assertEqual(self.probe({}), [0, 1, 2, 3])

    def test_named_cards_are_kept_as_named(self):
        self.assertEqual(self.probe({"CUDA_VISIBLE_DEVICES": "0,1"}, [2, 3]), [2, 3])


if __name__ == "__main__":
    unittest.main()
