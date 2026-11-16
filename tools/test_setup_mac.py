"""Tests for setup on a Mac (metal/setup_mac.py and setup.py's hand-over): the Mac check, the pack recommended by
unified memory, what a bigger choice risks, the engine arguments with the Metal defaults and the config. Runs on any
PC: sysctl / sw_vers / xcode-select are mocked. No GPU, no network.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup  # noqa: E402
import setup_mac as M  # noqa: E402


def fake_run(values: dict, clt_rc=0):
    def run(cmd, **kw):
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, values.get(cmd[2], ""), "")
        if cmd[0] == "sw_vers":
            return subprocess.CompletedProcess(cmd, 0, values.get("macos", ""), "")
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, clt_rc, "/Library/Developer/CommandLineTools" if clt_rc == 0 else "",
                                               "")
        raise AssertionError(cmd)
    return run


M5 = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.memsize": str(128 * 2 ** 30), "macos": "27.2",
      "hw.perflevel0.physicalcpu": "12"}


class MacCheck(unittest.TestCase):
    def test_reads_chip_memory_macos_and_tools(self):
        with mock.patch.object(M.platform, "machine", return_value="arm64"):
            info = M.mac_info(fake_run(M5))
        self.assertEqual(info["chip"], "Apple M5 Max")
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["macos"], "27.2")
        self.assertTrue(info["arm64"])
        self.assertTrue(info["clt"])
        self.assertEqual(M.mac_problems(info), [])

    def test_an_intel_python_on_apple_silicon_is_still_an_apple_silicon_mac(self):     # #3894
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "machdep.cpu.brand_string": "Apple M1 Max"}))
        self.assertTrue(info["arm64"])
        self.assertTrue(info["rosetta_python"])
        self.assertNotIn("Intel", " ".join(M.mac_problems(info)))

    def test_intel_old_macos_and_missing_tools_are_named(self):
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "macos": "14.6", "machdep.cpu.brand_string": "Intel Core i9"}, clt_rc=2))
        p = " | ".join(M.mac_problems(info))
        self.assertIn("Apple Silicon", p)
        self.assertIn("macOS 15", p)
        self.assertIn("xcode-select --install", p)


class Recommendation(unittest.TestCase):
    def test_by_memory(self):
        # IQ3_S and UD-Q4_K_XL are the recommended sizes; IQ3_S is the default wherever it stays resident
        self.assertEqual(M.RECOMMENDED, ("IQ3_S", "UD-Q4_K_XL"))
        rec, notes = M.recommend_model(128)
        self.assertEqual(rec, "IQ3_S")
        self.assertTrue(any("UD-Q4_K_XL" in n and "higher quality" in n for n in notes))   # named next to it
        rec96, notes96 = M.recommend_model(96)
        self.assertEqual(rec96, "IQ3_S")
        self.assertFalse(notes96)                       # UD-Q4_K_XL does not fit 96 GB resident: no note
        self.assertEqual(M.recommend_model(64)[0], "Q2_0")      # no recommended size fits: the largest that does
        self.assertEqual(M.recommend_model(80)[0], "IQ3_XXS")
        rec, notes = M.recommend_model(32)
        self.assertEqual(rec, "Q2_0")
        self.assertTrue(notes and "stream" in notes[0] and "~14 tok/s" in notes[0])
        self.assertTrue(M.streams("Q2_0", 32) and M.streams("Q2_0", 48) and not M.streams("Q2_0", 64))

    def test_a_bigger_choice_is_kept_with_a_note(self):
        notes = M.model_note("IQ3_XXS", 64)
        self.assertTrue(any("stream from the SSD" in n for n in notes))
        self.assertEqual(M.model_note("Q2_0", 128), [])
        self.assertEqual(M.model_note("IQ3_S", 128), [])

    def test_context(self):
        self.assertEqual(M.recommend_context(64), 32768)
        self.assertEqual(M.recommend_context(128), 65536)


class Config(unittest.TestCase):
    def test_engine_args_carry_the_metal_defaults(self):
        args = M.engine_args(Path("/p/packs/q2_0"), [Path("/m/a-00001.gguf"), Path("/m/a-00002.gguf")],
                             Path("/d/mtp/rt"), 32768)
        self.assertEqual(args[args.index("--spec") + 1], "2")
        self.assertEqual(args[args.index("--native") + 1], str(Path("/m/a-00001.gguf")))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(Path("/m/a-00002.gguf")))
        self.assertEqual(args[args.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertIn("--expert-profile", args)
        self.assertNotIn("--hc-q8", args)                         # Q8_0 HC is the engine's own default on Metal
        st = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True)
        self.assertNotIn("--mtp", st)                             # streaming: one token per step, no draft layer
        self.assertEqual(st[st.index("--suffix-draft") + 1], "0")
        self.assertEqual(st[st.index("--spec") + 1], "2")
        # a big Mac that streams (UD-Q5/Q6 on 128 GB) keeps the draft layer, with a longer window
        big = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 65536, stream=True,
                            ram_gb=128)
        self.assertEqual(big[big.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertEqual(big[big.index("--spec") + 1], "3")
        self.assertEqual(big[big.index("--suffix-draft") + 1], "0")
        small = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True,
                              ram_gb=32)
        self.assertNotIn("--mtp", small)
        self.assertTrue(M.stream_mtp(False, 16) and M.stream_mtp(True, 128) and not M.stream_mtp(True, 64))

    def test_iq3_s_runs_a_three_token_window_with_min_p(self):
        # M5 Max, MTP, 5 interleaved pairs, identical tokens: --spec 3 --spec-min-p 0.5 over --spec 2: IQ3_S -0.9% / +2.5% / +16.7% / +4.9%
        # (story / code / JSON / agent turn), Q2_0 +0.4% / +11.5% / +6.3% / +1.9%
        shards = [Path("/m/a-00001.gguf"), Path("/m/a-00002.gguf")]
        iq = M.engine_args(Path("/p/packs/iq3_s"), shards, Path("/d/mtp/rt"), 32768, model="IQ3_S")
        self.assertEqual(iq[iq.index("--spec") + 1], "3")
        self.assertEqual(iq[iq.index("--spec-min-p") + 1], "0.5")
        self.assertIn("--mtp", iq)
        q2 = M.engine_args(Path("/p/packs/q2_0"), shards, Path("/d/mtp/rt"), 32768, model="Q2_0")
        self.assertEqual(q2[q2.index("--spec") + 1], "3")
        self.assertEqual(q2[q2.index("--spec-min-p") + 1], "0.5")
        coder = M.engine_args(Path("/p/packs/coder"), shards, Path("/d/mtp/rt"), 32768, model="Coder")   # 5 pairs: story -2.5%, code +7.1%, JSON +9.3%, agent +12.5%
        self.assertEqual(coder[coder.index("--spec") + 1], "3")
        self.assertEqual(coder[coder.index("--spec-min-p") + 1], "0.5")
        other = M.engine_args(Path("/p/packs/iq3_xxs"), shards, Path("/d/mtp/rt"), 32768, model="IQ3_XXS")   # not measured: the old default
        self.assertEqual(other[other.index("--spec") + 1], "2")
        self.assertNotIn("--spec-min-p", other)
        none = M.engine_args(Path("/p"), shards, Path("/d/mtp/rt"), 32768)         # no model named: the old default
        self.assertEqual(none[none.index("--spec") + 1], "2")
        self.assertNotIn("--spec-min-p", none)
        # a streaming IQ3_S keeps its own rules: no draft layer on a small Mac (spec 2, nothing else), --spec 3 only on a big one
        small = M.engine_args(Path("/p"), shards, Path("/d/mtp/rt"), 32768, stream=True, ram_gb=32, model="IQ3_S")
        self.assertNotIn("--spec-min-p", small)
        self.assertEqual(small[small.index("--spec") + 1], "2")
        big = M.engine_args(Path("/p"), shards, Path("/d/mtp/rt"), 65536, stream=True, ram_gb=128, model="IQ3_S")
        self.assertEqual(big[big.index("--spec") + 1], "3")
        self.assertNotIn("--spec-min-p", big)

    def test_fp16_decode_is_the_engines_own_default(self):
        # STRATA_METAL_F16 for a resident IQ3_S / Q2_0 pack is decided by the engine (metal window.cpp f16_auto_for); setup writes nothing for it
        self.assertFalse(hasattr(M._fit, "engine_env"))
        self.assertFalse(hasattr(M._fit, "F16_MODELS"))

    def test_config_is_a_model_config_with_the_metal_backend(self):
        cfg = M.mac_config(Path("/r/build-metal/strata"), ["--pack", "x"], Path("/p"), "qwen3.8-flash-next-q2_0",
                           "Q2_0", 8080)
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["tokenizer"], str(Path("/p") / "tokenizer"))
        self.assertTrue(cfg["exe"] and isinstance(cfg["args"], list))


class Vision(unittest.TestCase):
    def test_the_projector_is_pinned_and_the_encoder_builds_with_metal(self):
        self.assertIn("mmproj-Qwen3.8-Flash-Next-BF16.gguf", M.VISION_SHA256)
        src = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"-DSTRATA_VISION_CUDA=OFF"', src)
        self.assertIn('cfg["args"] += ["--vision"]', src)


class ExpertsFile(unittest.TestCase):
    """The streaming Mac's <model>.experts: written only with room on the disk, else a warning and the GGUF."""

    def run_it(self, free_gb, done=False):
        import tempfile
        sys.path.insert(0, str(ROOT / "metal" / "tools"))
        import stream_experts
        d = Path(tempfile.mkdtemp())
        native = d / "m.gguf"
        native.write_bytes(b"")
        if done:
            (d / "m.gguf.experts").write_bytes(b"")
            (d / "m.gguf.experts.done").write_text("x")
        with mock.patch.object(stream_experts, "size_bytes", return_value=40e9), \
                mock.patch.object(M.S, "free_gb", return_value=free_gb), \
                mock.patch.object(M.S, "run") as run, mock.patch.object(M.S, "warn") as warn, \
                mock.patch.object(M.S, "mark") as mark, mock.patch.object(M.S, "say"), mock.patch.object(M.S, "ok"):
            M.experts_file(native)
        return run, warn, mark

    def test_fits_needs_the_margin(self):
        self.assertFalse(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB - 0.1))
        self.assertTrue(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB))

    def test_written_and_verified_with_room(self):
        run, warn, mark = self.run_it(100)
        self.assertIn("--verify", run.call_args[0][0])
        self.assertTrue(str(run.call_args[0][0][1]).endswith("stream_experts.py"))
        mark.assert_called_once()
        warn.assert_not_called()

    def test_too_little_disk_warns_and_goes_on(self):
        run, warn, mark = self.run_it(45)
        run.assert_not_called()
        mark.assert_not_called()
        self.assertIn("not written", warn.call_args[0][0])

    def test_done_is_not_written_again(self):
        run, warn, mark = self.run_it(100, done=True)
        run.assert_not_called()


class V31Sets(unittest.TestCase):
    """Round 0214: Unsloth's v3.1 sets on a Mac.  One switch (mac_fit.METAL_Q5_0_DOWN) lists the sets with Q5_0 expert downs; until the Metal kernel exists they carry "missing"."""
    NEW = ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31")
    Q5_0 = ("UD-IQ2_XS", "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-IQ4_XS-v31")

    def test_entries_and_files(self):
        for m in self.NEW:
            d = M.MAC_MODELS[m]
            self.assertEqual(d["family"], "unsloth")
            self.assertEqual(d["shards"], setup.MODELS[m]["shards"])
            self.assertAlmostEqual(d["shard1_gb"], setup.MODELS[m]["download_gb"] - 28.8, delta=0.15)     # the download less the n-gram table
            self.assertEqual(M.model_shard_names(m), [setup.model_file(setup.FAMILIES["unsloth"], m, i) for i in range(1, d["shards"] + 1)])
            self.assertTrue(all(M.shard_sha(n) for n in M.model_shard_names(m)), m)
            self.assertIn(m, M.MAC_ABOUT)

    def test_all_seven_are_listed_now_the_metal_kernel_exists(self):
        from strata_control import mac_fit
        self.assertTrue(mac_fit.METAL_Q5_0_DOWN)            # f-0214-metal: the Q5_0 down kernels
        for m in self.NEW:
            self.assertFalse(mac_fit.MAC_MODELS[m].get("missing"), m)
        self.assertNotIn("does not run on this Metal engine yet", "".join(mac_fit.model_note("UD-Q3_K_XL", 128)))

    def test_the_switch_lists_them_all(self):
        from strata_control import mac_fit
        ns = {"__name__": "mac_fit_on", "__file__": mac_fit.__file__}
        src = Path(mac_fit.__file__).read_text(encoding="utf-8").replace("METAL_Q5_0_DOWN = True", "METAL_Q5_0_DOWN = False")
        exec(compile(src, mac_fit.__file__, "exec"), ns)                 # the switch off: the four Q5_0-down sets are held back again
        self.assertEqual([m for m in self.NEW if ns["MAC_MODELS"][m].get("missing")], list(self.Q5_0))


class HandOver(unittest.TestCase):
    def test_darwin_hands_the_run_to_setup_mac(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--model", "Q2_0", "--yes"]):
            self.assertEqual(setup.main(), 0)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("metal") / "setup_mac.py")))
        self.assertEqual(cmd[2:], ["--model", "Q2_0", "--yes"])

    def test_other_platforms_never_reach_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin":', src)
        with mock.patch.object(setup, "mac_setup", side_effect=AssertionError("called")), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--inspect", "x"]), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(SystemExit):
                setup.main()

    def test_setup_sh_fetches_a_pinned_python_on_macos_only(self):
        sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('[ "$(uname -s)" = Darwin ]', sh)
        self.assertIn("83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8", sh)


class RepackGuard(unittest.TestCase):
    """No engine beside a repack: the locks, and the resident rule that matches the engine's arena_budget.h."""

    def test_resident_rule_matches_the_engine(self):
        # the resident cap: Metal's working set (3/4), and on a Mac of 64 GB or more the RAM less 30%
        self.assertAlmostEqual(M.gpu_budget_gb(128), 128 * 1.073741824 * 0.70, places=3)    # 96.2 GB, as arena_budget.h
        self.assertAlmostEqual(M.gpu_budget_gb(96), 96 * 1.073741824 * 0.70, places=3)
        self.assertAlmostEqual(M.gpu_budget_gb(32), 32 * 1.073741824 * 0.75, places=3)      # a small Mac: the working set
        self.assertFalse(M.streams("UD-Q4_K_XL", 128))          # 82.6 + 8 GB: resident (measured: nothing compressed)
        self.assertTrue(M.streams("UD-Q5_K_XL", 128))
        self.assertTrue(M.streams("UD-Q4_K_XL", 96))

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_a_live_lock_is_seen_a_stale_one_is_dropped(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(M.repack_running(d), [])
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))          # a live process that is not us
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / "m-metal.gguf.repack.lock").write_text("999999999")                # no such process
            self.assertEqual(M.repack_running(d), [])
            self.assertFalse((d / "m-metal.gguf.repack.lock").exists())             # the stale lock was removed
            (d / M.PREPARE_LOCK).write_text(str(os.getppid()))
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / M.PREPARE_LOCK).write_text(str(os.getpid()))                       # our own setup's lock does not stop us
            self.assertEqual(M.repack_running(d), [])

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_start_refuses_while_a_repack_runs(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gguf, exe = d / "m-metal.gguf", d / "strata"
            gguf.write_bytes(b"x")
            exe.write_bytes(b"x")
            cfg = d / "strata-x.json"
            cfg.write_text(json.dumps({"exe": str(exe), "args": ["--native", str(gguf)], "port": 8080}))
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))
            with mock.patch.object(M.subprocess, "call", side_effect=AssertionError("started")), \
                    mock.patch.object(M.S, "fail", side_effect=SystemExit("refused")) as fail:
                with self.assertRaises(SystemExit):
                    M.start(cfg, None, False)
            self.assertIn("repack is still running", fail.call_args[0][0])

    def test_only_this_users_engine_stops_setup(self):
        """On a shared Mac another user's engine is not ours to stop: pgrep is asked for this user's uid only."""
        seen = []

        def run(cmd, **kw):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "4242 ", "")
        with mock.patch.object(M.os, "getuid", return_value=501, create=True):
            self.assertEqual(M.engine_pids(run), [4242])
        self.assertEqual(seen, [["pgrep", "-u", "501", "-x", "strata"]])
        self.assertEqual(M.engine_pids(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "")), [])

    def test_the_repack_writes_in_bounded_pieces(self):
        src = (ROOT / "metal" / "tools" / "metal_repack.py").read_text(encoding="utf-8")
        self.assertIn("SYNC_EVERY = 1 << 30", src)                  # flushed every GiB
        self.assertIn("nocache(o)", src)                            # F_NOCACHE on the output
        self.assertIn("buffering=0", src)                           # write() is a write
        self.assertNotIn("mmap", src)                               # no dirty mapping of the output


M4 = {"machdep.cpu.brand_string": "Apple M4", "hw.memsize": str(32 * 2 ** 30), "macos": "27.2", "hw.perflevel0.physicalcpu": "4"}


class ControlMode(unittest.TestCase):
    """The app on a Mac (STRATA_CONTROL_INSTALL=1): model files and a run config naming the engine component. No source build, no
    metal/setup_mac.py, no Command Line Tools; the Mac rules are strata_control/mac_fit.py's, the module setup_mac.py uses too."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name).resolve() / "install"
        self.data = Path(self.tmp.name).resolve() / "data"
        self.models = Path(self.tmp.name).resolve() / "own"
        self.home.mkdir()
        self.models.mkdir()
        fam = setup.FAMILIES["qwen"]
        self.files = []
        for i in (1, 2):
            f = self.models / setup.model_file(fam, "Q2_0", i)
            f.write_bytes(b"x")
            f.with_name(f.name + ".done").write_text("whole")
            self.files.append(f)

    def run_setup(self, hw=M4, argv=("--yes", "--no-start", "--family", "qwen", "--model", "Q2_0", "--context", "8192"), clt_rc=0, tool=None, free=500.0):
        from strata_control import engine_component, mac_fit
        ran = []

        def fake_tool(cmd, **kw):
            ran.append(cmd)
            pack = Path(cmd[cmd.index("--out") + 1])
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            (pack / "native_experts.txt").write_text("x")
            (pack / "tokenizer" / "vocab.json").write_text("{}")
        patches = [
            mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "_CONTROL_HOME", self.home),
            mock.patch.object(setup.sys, "platform", "darwin"),
            mock.patch.object(setup.sys, "argv", ["setup.py", *argv, "--data-dir", str(self.data), "--gguf-dir", str(self.models)]),
            mock.patch.object(mac_fit, "mac_info", lambda run=None, _real=mac_fit.mac_info: _real(fake_run(hw, clt_rc))),
            mock.patch.object(setup, "load_settings", lambda: {}), mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: Path(self.tmp.name) / "llama"),
            mock.patch.object(setup, "whole_shard", lambda s: True), mock.patch.object(setup, "check_shards", lambda s: None),
            mock.patch.object(setup, "run", tool or fake_tool), mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "say", lambda *a, **k: self.said.append(" ".join(str(x) for x in a))),
            mock.patch.object(setup, "ok", lambda *a, **k: None),
            mock.patch.object(setup, "step", lambda *a, **k: None),
            mock.patch.object(setup, "warn", lambda *a, **k: self.warnings.append(" ".join(str(x) for x in a))),
            mock.patch.object(mac_fit.platform, "machine", return_value="arm64"),
            mock.patch.object(engine_component, "resolve", side_effect=engine_component.EngineMissing("no package here", None)),
            mock.patch.object(setup, "mac_setup", side_effect=AssertionError("setup_mac.py must not run in Control mode")),
            mock.patch.object(setup.subprocess, "call", side_effect=AssertionError("nothing is started or built")),
        ]
        self.warnings, self.said = [], []
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc = setup.main()
        return rc, ran

    def config(self, tag="q2_0"):
        import json
        return json.loads((self.home / f"strata-{tag}.json").read_text(encoding="utf-8"))

    def test_the_install_writes_a_run_config_for_the_engine_component(self):
        rc, ran = self.run_setup()
        self.assertEqual(rc, 0)
        cfg = self.config()
        self.assertEqual(cfg["exe"], "component:strata-engine")
        self.assertEqual(cfg["backend"], "metal")
        args = cfg["args"]
        # a 32 GB Mac streams Q2_0's experts from the SSD: one token per step, no draft layer, the GGUF mapped in place
        self.assertEqual(args[args.index("--native") + 1], str(self.files[0]))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(self.files[1]))
        self.assertIn("--mmap-experts", args)
        self.assertNotIn("--mtp", args)
        self.assertEqual(args[args.index("--suffix-draft") + 1], "0")
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "1"})
        self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / "q2_0" / "tokenizer"))
        self.assertEqual(len(ran), 1, "only the pack is made (tools/iq_pack.py): no draft layer, no build")
        self.assertTrue(ran[0][1].endswith("iq_pack.py"))
        self.assertFalse((ROOT / "build-metal").exists() and any((ROOT / "build-metal").glob("*.stamp")))

    def test_a_model_without_a_draft_layer_is_said_on_a_streaming_mac_too(self):
        # a feed model whose entry says "drafter: none" (mac_fit's no_mtp) on a Mac that streams (no draft layer there anyway): the model
        # list's line is said as on Windows and Linux (tools/feed_model_install_gate.py; GitHub's macOS runner did not get it)
        from strata_control import mac_fit
        with mock.patch.dict(mac_fit.MAC_MODELS["Q2_0"], {"no_mtp": True}):
            rc, _ = self.run_setup()
        self.assertEqual(rc, 0)
        self.assertIn("  The model list says this model has no draft layer (MTP), so none is fetched or used.", self.said)
        self.assertNotIn("--mtp", self.config()["args"])
        rc, _ = self.run_setup()                                  # the Flash-Next size itself says nothing of the kind
        self.assertFalse([s for s in self.said if "model list says" in s])

    def test_a_model_without_a_draft_layer_plans_and_starts_no_drafter(self):
        # the run config of such a model says what Windows and Linux write ("mtp": false, "drafting": "off"), so the plan and the start say
        # "Drafting: Off"; before, neither key was written and both said "Drafting: MTP" with nothing to load (TRIAGE-0215)
        from strata_control import drafting, mac_fit
        for hw in (M4, M5):                                       # a Mac that streams and one that keeps the model in memory
            with mock.patch.dict(mac_fit.MAC_MODELS["Q2_0"], {"no_mtp": True}):
                rc, _ = self.run_setup(hw=hw)
            self.assertEqual(rc, 0)
            cfg = self.config()
            self.assertIs(cfg.get("mtp"), False, hw)
            self.assertEqual(cfg.get("drafting"), "off", hw)
            self.assertNotIn("--mtp", cfg["args"])
            for name in (cfg["model_name"], "feedfam-q4_k_m"):    # the feed model's own name as well
                d = drafting.apply(dict(cfg, model_name=name), data_dir=self.data, backend="metal", archs=[])
                self.assertEqual((d.drafter, drafting.summary_line(d)), ("off", "Drafting: Off"), (hw, name))
        (self.home / "strata-q2_0.json").unlink()                 # (a fresh install: nothing carried over from the runs above)
        rc, _ = self.run_setup(hw=M5)                             # the Flash-Next size keeps its drafter: neither key is written
        cfg = self.config()
        self.assertNotIn("drafting", cfg)
        self.assertNotIn("mtp", cfg)

    def test_a_big_mac_keeps_the_pack_in_memory_and_adds_the_draft_layer(self):
        rc, ran = self.run_setup(hw=M5, argv=("--yes", "--no-start", "--model", "IQ3_XXS"))
        self.assertEqual(rc, 0)
        cfg = self.config("iq3_xxs")
        self.assertIn("--mtp", cfg["args"])
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "0"})
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "65536")      # the default context by memory

    def test_no_command_line_tools_needed_and_ud_iq1_s_is_not_offered(self):
        rc, _ = self.run_setup(clt_rc=2)
        self.assertEqual(rc, 0)
        # UD-IQ1_S: never offered.  A name the built-in tables do not know is looked up in the signed feed first (a feed model): no feed here
        with self.assertRaises(SystemExit), mock.patch.object(setup, "feed_models_setup", lambda *a, **k: None):
            self.run_setup(argv=("--yes", "--no-start", "--model", "UD-IQ1_S"))

    def test_gyro_is_offered_on_a_mac_as_one_file_with_a_compat_bf16_pack(self):
        fam = setup.FAMILIES["gyro"]
        g = self.models / setup.model_file(fam, "Gyro-S", 1)
        g.write_bytes(b"x")
        g.with_name(g.name + ".done").write_text("whole")
        rc, ran = self.run_setup(argv=("--yes", "--no-start", "--family", "gyro", "--model", "Gyro-S", "--context", "8192"))
        self.assertEqual(rc, 0)
        cfg = self.config("gyro-gyro-s")
        args = cfg["args"]
        self.assertEqual(args[args.index("--native") + 1], str(g), "the one file is the model")
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(g), "... and holds its n-gram table")
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["env"], {"STRATA_METAL_STREAM": "1"}, "29.7 GB of weights stream on a 32 GB Mac")
        self.assertNotIn("--mtp", args)
        self.assertTrue(ran[0][1].endswith("iq_pack.py"))
        self.assertIn("--compat-bf16", ran[0], "Gyro's pack converts the small Q6_K / Q5_K tensors (docs/GYRO.md)")
        self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / "gyro-gyro-s" / "tokenizer"))

    def test_gyro_m_on_a_small_mac_is_a_warning_not_a_refusal(self):
        fam = setup.FAMILIES["gyro"]
        g = self.models / setup.model_file(fam, "Gyro-M", 1)
        g.write_bytes(b"x")
        g.with_name(g.name + ".done").write_text("whole")
        rc, _ = self.run_setup(argv=("--yes", "--no-start", "--family", "gyro", "--model", "Gyro-M", "--context", "8192"))
        self.assertEqual(rc, 0)
        said = " ".join(self.warnings)
        self.assertIn("Gyro-M", said)
        self.assertIn("stream from the SSD", said)
        self.assertIn("not been measured on a Mac", said)

    def test_other_platforms_and_the_terminal_still_use_their_own_setup(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin" and not CONTROL:', src)
        self.assertIn('if sys.platform == "darwin" and CONTROL:', src)


class MacAppUnsloth(unittest.TestCase):
    """Round 0214: the Mac app installs Unsloth's sets as the Terminal's setup_mac does: the shards -> ONE GGUF for the GPU (metal_repack.py --verify, the n-gram table
    left out), the pack from that file, --native the repacked file, --ple-gguf the table's shard, --mmap-experts.  Header-only fixtures (tests/fixtures/unsloth_v31): no
    Mac, no weights; the repack and the pack tools are recorded, not run."""
    setUp = ControlMode.setUp
    run_setup = ControlMode.run_setup
    config = ControlMode.config
    NEW = ("UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-IQ4_XS-v31")

    def shards(self, model):
        sys.path.insert(0, str(ROOT / "tools"))
        import unsloth_v31_fixture as FX
        doc = FX.load()
        hub = "UD-IQ4_XS" if model == "UD-IQ4_XS-v31" else model
        for old in self.models.glob("*"):
            old.unlink()
        out = []
        for i, s in enumerate(doc["sets"][hub]["shards"]):
            h = s["header"] or {"split": [0, len(doc["sets"][hub]["shards"]), 1224], "tensors": []}
            p = FX.materialize({"header": h}, self.models / s["name"], first=(i == 0))
            p.with_name(p.name + ".done").write_text("whole")
            out.append(p)
        return out

    def tool(self, ran):
        def fake(cmd, **kw):
            ran.append([str(x) for x in cmd])
            if cmd[1].endswith("metal_repack.py"):
                Path(cmd[cmd.index("--out") + 1]).write_bytes(b"x")
            else:
                pack = Path(cmd[cmd.index("--out") + 1])
                (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
                (pack / "native_experts.txt").write_text("x")
                (pack / "tokenizer" / "vocab.json").write_text("{}")
        return fake

    def install(self, model, hw=M5, **kw):
        ran = []
        engines = kw.pop("engines", [])
        with mock.patch.object(setup, "mac_engine_pids", lambda: engines):
            rc, _ = self.run_setup(hw=hw, argv=("--yes", "--no-start", "--family", "unsloth", "--model", model, "--context", "8192", "--mtp", "off"),
                                   tool=self.tool(ran), **kw)
        return rc, ran

    def test_the_repack_the_pack_and_the_config(self):
        for model in self.NEW:
            with self.subTest(model=model):
                files = self.shards(model)
                shutil_native = self.data / "metal" / f"unsloth-{model.lower()}-metal.gguf"
                if shutil_native.exists():
                    shutil_native.unlink()
                    shutil_native.with_name(shutil_native.name + ".done").unlink(missing_ok=True)
                rc, ran = self.install(model)
                self.assertEqual(rc, 0)
                self.assertEqual(len(ran), 2, ran)
                repack, pack = ran
                self.assertTrue(repack[1].endswith("metal_repack.py"))
                self.assertEqual(repack[repack.index("--shard1") + 1], str(files[0]))
                self.assertEqual(repack[repack.index("--out") + 1], str(shutil_native))
                self.assertIn("--verify", repack)
                self.assertTrue(pack[1].endswith("iq_pack.py"))
                self.assertEqual(pack[pack.index("--gguf") + 1], str(shutil_native))        # the pack is read from the repacked file
                self.assertIn("--compat-bf16", pack)
                cfg = self.config("unsloth-" + model.lower())
                a = cfg["args"]
                self.assertEqual(cfg["exe"], "component:strata-engine")
                self.assertEqual(a[a.index("--native") + 1], str(shutil_native))
                self.assertEqual(a[a.index("--native-dense-gguf") + 1], str(shutil_native))
                self.assertEqual(a[a.index("--ple-gguf") + 1], str(files[-1]))               # the n-gram table: the LAST shard of these sets
                self.assertIn("--mmap-experts", a)
                self.assertEqual(cfg["env"]["STRATA_METAL_STREAM"], "1" if M.streams(model, 128.0) else "0")
                self.assertEqual(cfg["tokenizer"], str(self.data / "packs" / ("unsloth-" + model.lower()) / "tokenizer"))
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-" + model.lower())
                self.assertFalse((shutil_native.parent / ".preparing.lock").exists(), "the prepare lock is gone")

    def test_an_already_repacked_set_is_not_repacked_again(self):
        model = "UD-IQ1_M"
        self.shards(model)
        native = self.data / "metal" / "unsloth-ud-iq1_m-metal.gguf"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"x")
        setup.mark(native)
        rc, ran = self.install(model)
        self.assertEqual(rc, 0)
        self.assertEqual([str(r[1]).replace("\\", "/").rsplit("/", 1)[-1] for r in ran], ["iq_pack.py"])      # (the path is C:\...\iq_pack.py on Windows)

    def test_a_running_engine_stops_the_repack_before_anything_is_written(self):
        self.shards("UD-IQ1_M")
        with self.assertRaises(SystemExit):
            self.install("UD-IQ1_M", engines=[4242])
        self.assertFalse((self.data / "metal").exists() and any((self.data / "metal").iterdir()))

    def test_the_apps_hint_for_a_running_engine(self):
        self.shards("UD-IQ1_M")
        said = []

        def fail(msg, hint=None):
            said.append((msg, hint))
            raise SystemExit(1)
        with mock.patch.object(setup, "fail", fail):
            with self.assertRaises(SystemExit):
                self.install("UD-IQ1_M", engines=[4242])
        self.assertEqual(said[0][0], "a Strata engine is running (pid 4242): preparing a model beside it can freeze this Mac")
        self.assertEqual(said[0][1], "stop the model in Strata (unload it), then install again")

    def test_the_disk_check_counts_the_repacked_file(self):
        self.shards("UD-IQ1_M")
        self.install("UD-IQ1_M", free=30.0)
        said = [w for w in self.warnings if "are needed on the disk" in w]
        self.assertEqual(len(said), 1, self.warnings)
        self.assertIn("about 45 GB", said[0])          # nothing to download (the files are here), the 39.2 GB repacked file and 6 GB to work in

    def test_a_set_the_engine_cannot_run_is_not_offered(self):
        from strata_control import mac_fit
        self.shards("UD-Q3_K_XL")
        with mock.patch.dict(mac_fit.MAC_MODELS, {"UD-Q3_K_XL": {**mac_fit.MAC_MODELS["UD-Q3_K_XL"], "missing": "the Q5_0 expert kernels"}}):
            with self.assertRaises(SystemExit):
                self.install("UD-Q3_K_XL")

    def test_the_bundle_carries_the_repack_tools(self):
        sys.path.insert(0, str(ROOT / "packaging"))
        import package_terminal as pt
        self.assertEqual(pt.INSTALLER_METAL_TOOLS, ("metal_repack", "stream_experts"))
        for n in pt.INSTALLER_METAL_TOOLS:
            self.assertTrue((ROOT / "metal" / "tools" / f"{n}.py").is_file())
        with tempfile.TemporaryDirectory() as t:
            out = pt.add_installer(Path(t))
            self.assertIn("metal/tools/metal_repack.py", out["files"])
            self.assertTrue((Path(t) / "metal" / "tools" / "stream_experts.py").is_file())


class UpdateFlow(unittest.TestCase):
    """update.sh asks the same questions as a setup run but does not download model files: a model chosen there must
    be named, not dropped in silence (an M1 Max picked IQ3_S in update.sh and got no model and no message)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name).resolve() / "data"
        self.models = Path(self.tmp.name).resolve() / "own"
        self.data.mkdir()
        self.models.mkdir()
        self.said, self.warned = [], []

    def run_update(self, argv=("--update", "--model", "IQ3_S", "--vision", "no", "--yes")):
        with mock.patch.object(M, "mac_configs", lambda: []), \
                mock.patch.object(M.S, "data_folder_for", lambda a: (self.data, [])), \
                mock.patch.object(M, "mac_info", lambda run=None: {"chip": "Apple M1 Max", "ram_gb": 32.0,
                                                                   "macos": "26.6.2", "arm64": True, "clt": True}), \
                mock.patch.object(M, "mac_problems", lambda info: []), \
                mock.patch.object(M.S, "free_gb", lambda p: 200.0), \
                mock.patch.object(M.S, "step", lambda *a, **k: None), \
                mock.patch.object(M.S, "say", lambda *a, **k: self.said.append(" ".join(str(x) for x in a))), \
                mock.patch.object(M.S, "ok", lambda *a, **k: None), \
                mock.patch.object(M.S, "warn", lambda *a, **k: self.warned.append(" ".join(str(x) for x in a))), \
                mock.patch.object(M.S, "pip_install", lambda *a, **k: None), \
                mock.patch.object(M.S, "get_llama_cpp", lambda: self.data / "llama"), \
                mock.patch.object(M, "build_engine", lambda llama, jobs=None: self.data / "strata"), \
                mock.patch.object(M.S, "download", side_effect=AssertionError("an update must not download")):
            return M.main([*argv, "--data-dir", str(self.data), "--gguf-dir", str(self.models)])

    def test_a_model_chosen_in_an_update_is_named_not_dropped(self):
        self.assertEqual(self.run_update(), 0)
        self.assertIn("IQ3_S is not installed here", " ".join(self.warned))
        self.assertIn("does not download model files", " ".join(self.warned))
        self.assertIn("./setup.sh --setup", " ".join(self.said))

    def test_an_installed_model_keeps_the_update_quiet(self):
        for n in M.model_shard_names("IQ3_S"):
            f = self.models / n
            f.write_bytes(b"x")
            f.with_name(f.name + ".done").write_text("whole")
        tag = M.S.FAMILIES["qwen"]["tag"] + "IQ3_S"
        pack = self.data / "packs" / tag.lower()
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "native_experts.txt").write_text("x")
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        self.assertEqual(self.run_update(), 0)
        self.assertFalse([w for w in self.warned if "not installed here" in w], self.warned)


class Source(unittest.TestCase):
    """setup.py hands a Mac over before its own parser runs, so the Mac parser has to know --source:
    source_hint() recommends it on every platform (#2363)."""

    def test_the_source_flag_reaches_strata_source(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SOURCE", None)
        warnings = []
        # setup.data_folder() is mocked: the real one moves the model files of this checkout and of the data folder in
        # the user's settings.json into --data-dir, and saves that dir as the new data folder; with a temp dir that
        # deletes the models on exit and leaves settings.json pointing at a deleted directory
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup, "data_folder", lambda requested, *a, **k: (Path(tmp), [])), \
                mock.patch("subprocess.run", fake_run(M5)), \
                mock.patch.object(setup, "warn", lambda *a, **k: warnings.append(" ".join(str(x) for x in a))), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "ok", lambda *a, **k: None), \
                mock.patch.object(setup, "step", lambda *a, **k: None):
            M.main(["--source", "modelscope", "--yes", "--check", "--data-dir", tmp])
        self.assertEqual(os.environ.get("STRATA_SOURCE"), "modelscope")
        self.assertFalse([w for w in warnings if "--source" in w],
                         f"--source was not used: {warnings}")


if __name__ == "__main__":
    unittest.main()
