# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_lora_train.py - the browser's LoRA training (serve/lora_train.py), without a GPU: a fake trainer (a
short script that speaks the budget protocol and writes real-shaped safetensors versions), a fake preprocess and
fake samples.  The API's checks, the stages, the cancel, the restart, the versions and the registration, the
TRAINERS mapping, and the budget's give-and-take (pause, resume, move, the image jobs' yield)."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import lora_train as lt
from serve import loras as lora_api

FAKE_TRAINER = r'''#!/usr/bin/env python3
import json, os, struct, sys, time
def ckpt(step, kind, trigger):
    rank, dim = 16, 64
    t = {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight":
             {"dtype": "F32", "shape": [rank, dim], "data_offsets": [0, rank * dim * 4]},
         "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight":
             {"dtype": "F32", "shape": [dim, rank], "data_offsets": [rank * dim * 4, 2 * rank * dim * 4]},
         "diffusion_model.transformer_blocks.0.attn.to_q.alpha":
             {"dtype": "F32", "shape": [], "data_offsets": [2 * rank * dim * 4, 2 * rank * dim * 4 + 4]}}
    meta = {"ss_steps": str(step), "ss_max_steps": str(step), "ss_xestrata_kind": kind,
            "ss_network_module": "networks.lora", "ss_network_dim": str(rank)}
    if trigger:
        meta["modelspec.trigger_phrase"] = trigger
    h = json.dumps({"__metadata__": meta, **t}).encode()
    return struct.pack("<Q", len(h)) + h + b"\0" * (2 * rank * dim * 4 + 4)
a = sys.argv[2:]
f = {a[i][2:]: a[i + 1] for i in range(0, len(a), 2)}
mf = os.path.basename(sys.argv[0]).endswith("mf-train")
steps = int(f["steps"]); k = int(f.get("checkpoint" if mf else "save-every", "250"))
out = f["out"]; control = f.get("vram-control"); state = f.get("state")
sleep = float(os.environ.get("FAKE_STEP_SLEEP", "0.01"))
peak = int(os.environ.get("FAKE_PEAK", "14848")); least = int(os.environ.get("FAKE_MIN", "8192"))
if f.get("resume"):
    open(out + ".resumed", "w").write(f["resume"])
def budget():
    if not control or not os.path.exists(control):
        return int(f.get("vram-budget", "0")) or peak, False, False
    c = json.load(open(control))
    return int(c.get("budget_mib", 0)), bool(c.get("save_state")), bool(c.get("stop"))
def save(step):
    name = f"{out}.{step}.safetensors" if mf else f"{out}_{step}.safetensors"
    open(name, "wb").write(ckpt(step, f.get("kind", "other"), f.get("trigger", "")))
def read_control():
    b, ss, stop = budget()
    if ss and state:
        open(state, "wb").write(b"XSTRAIN1")
        print(f"state saved {state}", flush=True)
    if stop:
        open(out + ".stopped", "w").write("save=" + str(ss))
        sys.exit(0)
    return b
b = read_control()
print(f"vram 100 {peak} {least} running budget_mib={b} need_mib=8252", flush=True)
for step in range(1, steps + 1):
    b = read_control()
    if b <= 0:
        print(f"vram 0 {peak} {least} paused need_mib=8252", flush=True)
        while True:
            b = read_control()
            if b > 0:
                break
            time.sleep(0.05)
        print(f"vram 100 {peak} {least} running need_mib=8252", flush=True)
    if mf:
        print(f"step {step:5d} loss 0.100000 lr 1.0e-04", flush=True)
    else:
        print(f"step {step} loss 0.10000 (avg 0.10000) t 500.0 grad 1.000 1024x1024 0.01 s ({sleep:.2f} s/step)",
              flush=True)
    if step % k == 0:
        save(step)
    print(f"progress step={step} total={steps} loss=0.1", flush=True)
    time.sleep(sleep)
save(steps)
if mf:
    open(out + ".final.safetensors", "wb").write(ckpt(steps, f.get("kind", "other"), f.get("trigger", "")))
print("done", flush=True)
'''


class FakeGen:
    def __init__(self, tmp: Path, model="mage-flow", family="mage-flow"):
        self.model, self.family = model, family
        exe = tmp / "engine" / "xestrata-image"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        dit = tmp / "models" / "dit.safetensors"
        dit.parent.mkdir(parents=True, exist_ok=True)
        dit.write_bytes(b"dit")
        self.cfg = {"exe": str(exe), "diffusion_model": str(dit), "vae": str(dit), "llm": str(dit),
                    "model_name": model, "family": family, "lora_dir": str(tmp / "loras")}
        (tmp / "loras").mkdir(exist_ok=True)
        self.engine = type("E", (), {"env": {}})()
        self.loras = lora_api.Folder(self.cfg)
        self.job = None


class FakeSvc:
    def __init__(self, tmp: Path, gen: FakeGen):
        gens = {gen.model: gen}
        for fam, model in (("qwen-image", "qwen-image-2.1"), ("moss-image", "moss-image-sft"), ("anima", "anima")):
            g = FakeGen(tmp, model=model, family=fam)
            gens[model] = g
        self.images = type("M", (), {"gens": gens, "default": gen,
                                     "pick": lambda s, name: gens.get(name, gen)})()
        self.status_lock = threading.Lock()
        self.status = {"busy": False, "queued": 0}
        self.live_reqs = None
        self.port = 8095
        self.config_path = None
        self.engine = type("E", (), {"env": {}})()
        self.lora_jobs = None


class FakeBudget:
    """train_budget's figures and Decider contract, scripted by the test through `mode`."""

    def __init__(self):
        self.mode = "run"
        self.card = "0000:00:02.0"
        self.move_to = "0000:00:03.0"
        self.moved = False

    def figures(self, model, base):
        return 0, 8192, 14848

    def dataset_tokens(self, model, data):
        from serve.train_budget import dataset_tokens
        return dataset_tokens(model, data)

    def initial_vram(self, model, base, image_tokens, rank):
        return 6737, 6782

    def Decider(self, cfg, model, base, devices=None, image_tokens=None, rank=None):
        self.args = (model, base, image_tokens, rank)
        return self.DeciderCls(self)

    class DeciderCls:
        def __init__(self, outer):
            self.outer = outer

        def tick(self, st):
            o = self.outer
            if st["image_job"]:
                paused = st["train"].get("state") == "paused"
                if o.mode == "image-unconfirmed":
                    return {"pci": o.card, "budget_mib": 0, "save_state": False, "image_ready": False}
                known = st["image_need_mib"] is not None
                assert st["device_free"][o.card] == 20 * 1024
                alive = st["proc_alive"]
                return {"pci": o.card, "budget_mib": 0, "save_state": alive and paused and not known,
                        "image_ready": not alive or (paused and known), "reason": "an image job goes first"}
            if o.mode == "refuse":
                return {"pci": None, "reason": ("not enough free GPU memory: need 6782 MiB, "
                                                "available 6200 MiB after reserve")}
            if o.mode == "pause":
                return {"pci": o.card, "budget_mib": 0, "save_state": False, "reason": "the chat goes first"}
            if o.mode == "move" and not o.moved:
                o.moved = True
                return {"pci": o.move_to, "budget_mib": 0, "save_state": True, "reason": "a move is cheaper"}
            card = o.move_to if o.moved else o.card
            return {"pci": card, "budget_mib": 14848, "save_state": False, "reason": "a card of its own"}


def wait_for(pred, timeout=30.0, what="the condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = pred()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class Training(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lora-train-test-")
        self.root = Path(self.tmp.name)
        self.gen = FakeGen(self.root)
        self.svc = FakeSvc(self.root, self.gen)
        self.tb = FakeBudget()
        self.patches = [
            mock.patch.object(lt, "budget_mod", lambda: self.tb),
            mock.patch("serve.gpu_select.scope", lambda exe, env, pci: dict(env, XESTRATA_GPU_PCI=pci)),
            mock.patch.object(lt, "devices", lambda svc, **kwargs: [
                {"pci": "0000:00:02.0", "name": "Fake GPU A", "total_bytes": 24 << 30, "free_bytes": 20 << 30},
                {"pci": "0000:00:03.0", "name": "Fake GPU B", "total_bytes": 24 << 30, "free_bytes": 20 << 30}]),
            mock.patch.dict(lt.TRAINERS["mage-flow"], {"elastic": True}),
            mock.patch.dict(lt.TRAINERS["qwen-image"], {"elastic": True}),
            mock.patch.dict(os.environ, {"FAKE_STEP_SLEEP": "0.01"}),
        ]
        for p in self.patches:
            p.start()
        exe = self.root / "engine" / "xestrata-image-mf-train"
        exe.write_text(FAKE_TRAINER.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        exe.chmod(0o755)
        self.jobs = lt.TrainJobs(self.root / "lora-jobs", self.svc)
        self.svc.lora_jobs = self.jobs
        # the preprocess and the samples without a GPU: the markers and files they would write
        self.jobs.preprocess_argv = lambda rec: [
            [sys.executable, "-c",
             f"from pathlib import Path; d = Path({str(self.jobs.dir_of(rec['id']))!r}); "
             "(d / 'export').mkdir(exist_ok=True); "
             "(d / 'export' / 'manifest.tsv').write_text('001\\t48\\t48\\tx\\t29\\t29\\n')"]]
        self.jobs._samples = self._fake_samples

    def tearDown(self):
        for jid in list(self.jobs.cancels):
            self.jobs.cancel(jid)
        if self.jobs.worker is not None:
            self.jobs.worker.join(15)
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _fake_samples(self, rec, cancel):
        d = self.jobs.dir_of(rec["id"])
        for v in rec["versions"]:
            if not v.get("samples"):
                v["samples"] = [f"{v['step']}-0.png"]
                (d / "samples" / f"{v['step']}-0.png").write_bytes(b"\x89PNG fake")
        self.jobs._write(rec)

    def _draft(self, n=2, kind="character", trigger="tsumikaze"):
        from PIL import Image
        import io
        pics = []
        for i in range(n):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (i * 40, 80, 120)).save(buf, format="PNG")
            pics.append((f"pic{i}.png", buf.getvalue()))
        rec = self.jobs.create({"model": self.gen.model, "name": "my lora", "kind": kind, "trigger": trigger}, pics)
        caps = {im["file"]: {"caption": f"{trigger}, a girl in scene {i}", "note": ""}
                for i, im in enumerate(rec["images"])}
        return self.jobs.set_captions(rec["id"], caps)

    def _start(self, rec, **fields):
        return self.jobs.start(rec["id"], {"steps": 40, "save_every": 10, **fields})

    # ---- the checks
    def test_fields(self):
        for family in lt.FAMILIES:
            self.assertEqual(lt.PRESETS[family]["dropout"], 0.0)
            self.assertEqual(lt.check_settings({}, family)["dropout"], 0.0)
        self.assertEqual(lt.check_name("../../etc/passwd"), "passwd")   # folders taken off, as loras.file_name
        self.assertEqual(lt.check_name("a/b\\c d"), "c d")
        with self.assertRaises(lt.TrainError):
            lt.check_trigger("two, words")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"steps": 0}, "mage-flow")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"steps": 100, "save_every": 500}, "mage-flow")   # k past n
        s = lt.check_settings({"steps": 100}, "mage-flow")
        self.assertEqual(s["save_every"], 100)                    # the preset's k capped at n
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"rank": 999}, "mage-flow")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"lr": 5}, "mage-flow")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"method": "magic"}, "mage-flow")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"sample_prompts": []}, "mage-flow")
        with self.assertRaises(lt.TrainError):
            lt.check_caption("x" * (lt.MAX_CAPTION + 1))
        with self.assertRaises(KeyError):
            self.jobs.dir_of("lora_" + "0" * 31 + "/../etc")

    def test_images_limits(self):
        rec = self._draft(n=1)
        with self.assertRaises(lt.TrainError):
            self.jobs.add_image(rec["id"], "bad.txt", b"not a picture")
        with self.assertRaises(lt.TrainError):
            self.jobs.add_image(rec["id"], "bad.png", b"not a picture")
        with mock.patch.object(lt, "MAX_IMAGES", 1):
            with self.assertRaises(lt.TrainError):
                self.jobs.add_image(rec["id"], "ok.png", self._png())
        with mock.patch.object(lt, "MAX_IMAGE_BYTES", 4):
            with self.assertRaises(lt.TrainError):
                self.jobs.add_image(rec["id"], "big.png", self._png())
        got = self.jobs.remove_image(rec["id"], rec["images"][0]["file"])
        self.assertEqual(got["images"], [])

    @staticmethod
    def _png():
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    def test_start_needs_captions_and_trigger(self):
        rec = self._draft()
        self.jobs.set_captions(rec["id"], {rec["images"][0]["file"]: {"caption": "", "note": ""}})
        with self.assertRaises(lt.TrainError):
            self._start(rec)
        self.jobs.set_captions(rec["id"],
                               {i["file"]: {"caption": "no trigger here", "note": ""} for i in rec["images"]})
        with self.assertRaises(lt.TrainError):
            self._start(rec)

    def test_start_refuses_without_elastic(self):
        rec = self._draft()
        with mock.patch.dict(lt.TRAINERS["mage-flow"], {"elastic": False}):
            with self.assertRaises(lt.TrainError) as e:
                self._start(rec)
            self.assertIn("budget", str(e.exception))

    # ---- the whole path
    def test_pipeline_and_register(self):
        rec = self._draft()
        self._start(rec)
        done = wait_for(lambda: (r := self.jobs.get(rec["id"]))["status"] == "awaiting_choice" and r,
                        what="the training to end")
        self.assertEqual([v["step"] for v in done["versions"]], [10, 20, 30, 40])
        self.assertTrue(all(v["samples"] for v in done["versions"]))
        self.assertEqual(done["progress"]["step"], 40)
        self.assertEqual(done["training_shape"], {"tokens": 2333, "rank": 32, "min_mib": 6737, "need_mib": 6782})
        self.assertEqual(self.tb.args, ("mage-flow", "int8", 2333, 32))
        # the last version by default
        got = self.jobs.register(rec["id"], None, None)
        self.assertEqual(got["status"], "completed")
        self.assertEqual(got["registered"]["step"], 40)
        listing = self.gen.loras.listing()
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["steps"], 40)
        self.assertEqual(listing[0]["kind"], "character")
        self.assertEqual(listing[0]["trigger"], "tsumikaze")
        self.assertEqual(listing[0]["name"], "my lora")
        # an unselected version goes; the registered one stays
        got = self.jobs.delete_version(rec["id"], 10)
        self.assertEqual([v["step"] for v in got["versions"]], [20, 30, 40])
        self.assertFalse((self.jobs.dir_of(rec["id"]) / "out" / "lora.10.safetensors").exists())
        with self.assertRaises(lt.TrainError):
            self.jobs.delete_version(rec["id"], 40)   # registered

    def test_cancel_keeps_versions(self):
        with mock.patch.dict(os.environ, {"FAKE_STEP_SLEEP": "0.15"}):
            rec = self._draft()
            self._start(rec, steps=400, save_every=5)
            wait_for(lambda: len(self.jobs.get(rec["id"])["versions"]) >= 2, what="two versions")
            self.jobs.cancel(rec["id"])
            done = wait_for(lambda: (r := self.jobs.get(rec["id"]))["status"] == "cancelled" and r,
                            what="the cancel")
        self.assertGreaterEqual(len(done["versions"]), 2)
        self.assertIsNone(done["error"])
        self.assertIsNone(self.jobs._proc)
        self.assertEqual((self.jobs.dir_of(rec["id"]) / "out" / "lora.stopped").read_text(), "save=False")
        got = self.jobs.register(rec["id"], done["versions"][0]["step"], "early")
        self.assertEqual(got["registered"]["name"], "early")

    def test_restart_and_resume(self):
        rec = self._draft()
        self._start(rec)
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "training", what="the train stage")
        wait_for(lambda: len(self.jobs.get(rec["id"])["versions"]) >= 1, what="a version")
        # a server restart: the record says training, its pid is gone (a fake one)
        r = self.jobs.get(rec["id"])
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")
        r = self.jobs.get(rec["id"])
        r.update(status="training", pid=999999, pid_start=0.0, owner_pid=999999, owner_start=0.0)
        self.jobs._write(r)
        jobs2 = lt.TrainJobs(self.root / "lora-jobs", self.svc)
        jobs2.preprocess_argv = self.jobs.preprocess_argv
        jobs2._samples = self._fake_samples
        got = jobs2.get(rec["id"])
        self.assertEqual(got["status"], "interrupted")
        self.assertEqual(got["error"]["code"], "server_restarted")
        self.assertGreaterEqual(len(got["versions"]), 1)
        jobs2.resume(rec["id"])
        done = wait_for(lambda: (r := jobs2.get(rec["id"]))["status"] == "awaiting_choice" and r,
                        what="the resume to end")
        self.assertEqual(done["versions"][-1]["step"], 40)
        if jobs2.worker is not None:
            jobs2.worker.join(15)

    def test_cache_size_and_refusal_reason(self):
        rec = self._draft()
        rec["settings"] = lt.check_settings({"max_size": 768, "rank": 64}, "mage-flow")
        with mock.patch.object(self.jobs, "preprocess_argv", wraps=lt.TrainJobs.preprocess_argv.__get__(self.jobs)):
            pre = self.jobs.preprocess_argv(rec)[0]
        self.assertEqual(pre[pre.index("--budget") + 1], "512,768")
        with self.assertRaises(lt.TrainError):
            lt.check_settings({"max_size": 900}, "mage-flow")
        self.tb.mode = "refuse"
        self._start(rec, max_size=768, rank=64)
        failed = wait_for(lambda: (r := self.jobs.get(rec["id"]))["status"] == "failed" and r, what="refusal")
        self.assertEqual(failed["error"]["message"],
                         "not enough free GPU memory: need 6782 MiB, available 6200 MiB after reserve")
        self.assertEqual(self.tb.args, ("mage-flow", "int8", 2333, 64))
        self.assertEqual(failed["settings"]["max_size"], 768)

    def test_orphan_stop_control_precedes_any_signal(self):
        import subprocess
        from serve.vram_ledger import process_start
        rec = self._draft()
        d = self.jobs.dir_of(rec["id"])
        (d / "out").mkdir()
        self.jobs._write_control(d, 12000, False)
        proc = subprocess.Popen([str(self.root / "engine" / "xestrata-image-mf-train"), "train",
                                 "--steps", "300", "--checkpoint", "50", "--out", str(d / "out" / "lora"),
                                 "--vram-control", str(d / "vram.json"), "--state", str(d / "state.bin"),
                                 "--kind", "character", "--trigger", "test"], stdout=subprocess.DEVNULL)
        try:
            started = process_start(proc.pid)
            self.assertIsNotNone(started)
            rec.update(stage="train", pid=proc.pid, pid_start=started)

            def still_running(pid):
                return started if proc.poll() is None else None

            with mock.patch("serve.vram_ledger.process_start", side_effect=still_running), \
                    mock.patch.object(lt.os, "kill", side_effect=AssertionError("unexpected signal")):
                self.jobs._stop_orphan(rec)
            self.assertEqual(proc.wait(5), 0)
            self.assertEqual((d / "out" / "lora.stopped").read_text(), "save=False")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_real_cache_shapes_are_read_again(self):
        import struct
        from serve.train_budget import dataset_tokens
        d = self.root / "cache probe"   # a data folder can hold spaces: the items file is tab-separated
        d.mkdir()
        lat = d / "latent"
        text = d / "text"
        empty = d / "empty"
        lat.write_bytes(struct.pack("<ff", 48, 48))
        text.write_bytes(struct.pack("<f", 29))
        empty.write_bytes(struct.pack("<f", 35))
        items = d / "items.txt"
        items.write_text(f"empty\t{empty}\n{lat}\t{text}\n")
        self.assertEqual(dataset_tokens("qwen-image-2.1", items), 48 * 48 + 35)
        lat.write_bytes(struct.pack("<ff", 64, 64))
        self.assertEqual(dataset_tokens("qwen-image-2.1", items), 64 * 64 + 35)

    def test_stop_timeout_escalates_only_after_json(self):
        import subprocess
        import json
        rec = self._draft()
        rec["progress"] = {"s_per_step": 40}
        d = self.jobs.dir_of(rec["id"])
        self.jobs._write_control(d, 9000, False)
        proc = mock.Mock()
        proc.wait.side_effect = subprocess.TimeoutExpired("trainer", 120)
        with mock.patch.object(self.jobs, "_terminate") as terminate:
            self.jobs._stop_trainer(rec, proc, save_state=True)
        self.assertEqual(json.loads((d / "vram.json").read_text()),
                         {"budget_mib": 9000, "save_state": True, "stop": True})
        proc.wait.assert_called_once_with(120.0)
        terminate.assert_called_once_with(proc, 120.0)

    def test_stop_wait_scales_with_step(self):
        self.assertEqual(self.jobs._stop_wait({}), 120.0)
        self.assertEqual(self.jobs._stop_wait({"progress": {"s_per_step": 2}}), 30.0)
        self.assertEqual(self.jobs._stop_wait({"progress": {"s_per_step": 40}}), 120.0)

    # ---- the TRAINERS mapping
    def test_train_argv(self):
        rec = self._draft()
        rec["settings"] = lt.check_settings({"steps": 100, "save_every": 25, "accum": 4}, "mage-flow", "character")
        argv = self.jobs.train_argv(rec, 12000, False)
        self.assertIn("--checkpoint", argv)
        self.assertEqual(argv[argv.index("--steps") + 1], "400")   # 100 updates x accum 4
        self.assertEqual(argv[argv.index("--accum") + 1], "4")
        self.assertEqual(argv[argv.index("--vram-budget") + 1], "12000")
        self.assertIn("--vram-control", argv)
        self.assertIn("--state", argv)
        self.assertNotIn("--resume", argv)
        for fam, exe, model in (("qwen-image", "qi", "qwen-image-2.1"), ("moss-image", "moss", "moss-image-sft"),
                                ("anima", "anima", "anima")):
            r = {**rec, "family": fam, "model": model}
            r["settings"] = lt.check_settings({"steps": 100, "save_every": 25}, fam, "character")
            with mock.patch.object(lt, "train_base", lambda gen: (Path("dit"), "")):
                a = self.jobs.train_argv(r, 0, True)
            self.assertIn(f"xestrata-image-{exe}-train", a[0])
            self.assertEqual(a[a.index("--save-every") + 1], "25")
            self.assertEqual(a[a.index("--steps") + 1], "100")

    # ---- the budget's give-and-take
    def test_pause_and_resume_by_budget(self):
        rec = self._draft()
        self._start(rec, steps=300, save_every=50)
        wait_for(lambda: (self.jobs.get(rec["id"])["progress"] or {}).get("step", 0) >= 3, what="some steps")
        self.tb.mode = "pause"
        wait_for(lambda: (self.jobs.place or {}).get("mode") == "paused", what="the pause")
        wait_for(lambda: self.jobs._train_state.get("state") == "paused", what="the trainer paused")
        self.tb.mode = "run"
        wait_for(lambda: self.jobs._train_state.get("state") == "running", what="the trainer running again")
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")

    def test_image_job_goes_first(self):
        rec = self._draft()
        self._start(rec, steps=300, save_every=50)
        wait_for(lambda: (self.jobs.get(rec["id"])["progress"] or {}).get("step", 0) >= 3, what="some steps")
        t0 = time.monotonic()
        token = self.jobs.image_begin()  # unknown image need: pause, save and exit
        self.assertLess(time.monotonic() - t0, lt.IMAGE_YIELD_WAIT_S)
        self.assertIsNone(self.jobs._proc)   # the unknown image need required a state save and process exit
        self.assertTrue((self.jobs.dir_of(rec["id"]) / "state.bin").exists())
        self.assertEqual((self.jobs.dir_of(rec["id"]) / "out" / "lora.stopped").read_text(), "save=True")
        self.assertFalse((self.jobs.dir_of(rec["id"]) / "out" / "lora.resumed").exists())
        token2 = self.jobs.image_begin(4096)
        self.jobs.image_end(token)
        self.assertEqual(self.jobs.image_waiting, 1)
        self.assertIsNone(self.jobs._proc)  # the second image still holds the turn
        self.jobs.image_end(token2)
        wait_for(lambda: (self.jobs.dir_of(rec["id"]) / "out" / "lora.resumed").exists(), what="resume after image")
        wait_for(lambda: self.jobs._train_state.get("state") == "running", what="the training back")
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")

    def test_image_does_not_pass_on_pause_without_decider_approval(self):
        rec = self._draft()
        self._start(rec, steps=300, save_every=50)
        wait_for(lambda: (self.jobs.get(rec["id"])["progress"] or {}).get("step", 0) >= 3, what="some steps")
        self.tb.mode = "image-unconfirmed"
        with mock.patch.object(lt, "IMAGE_YIELD_WAIT_S", 2.0):
            with self.assertRaises(lt.TrainError):
                self.jobs.image_begin(4096)
        self.assertEqual(self.jobs.image_waiting, 0)
        self.assertIsNotNone(self.jobs._proc)
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")

    def test_known_image_need_can_keep_the_paused_process(self):
        rec = self._draft()
        self._start(rec, steps=300, save_every=50)
        wait_for(lambda: (self.jobs.get(rec["id"])["progress"] or {}).get("step", 0) >= 3, what="some steps")
        proc = self.jobs._proc
        token = self.jobs.image_begin(4096)
        self.assertIs(self.jobs._proc, proc)
        self.assertIsNone(proc.poll())
        self.assertEqual(self.jobs._train_state.get("state"), "paused")
        self.jobs.image_end(token)
        wait_for(lambda: self.jobs._train_state.get("state") == "running", what="resume after image")
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")

    def test_old_image_decision_does_not_release_a_new_request(self):
        self.jobs.image_needs["first"] = 4096
        self.jobs.image_seq = 1
        old = {"image_seq": 1}
        self.jobs.image_needs["second"] = None
        self.jobs.image_seq = 2
        self.jobs._accept_yield({"image_ready": True}, old)
        self.assertFalse(self.jobs.yielded.is_set())
        self.jobs._accept_yield({"image_ready": True}, {"image_seq": 2})
        self.assertTrue(self.jobs.yielded.is_set())
        self.jobs.image_needs.clear()

    def test_image_need_only_queries_an_engine_already_running(self):
        from contextlib import nullcontext
        from serve import images
        gen = mock.Mock()
        gen.engine.env = {}
        gen.need.return_value = {"least_mib": 4096}
        jobs = mock.Mock(current="lora_job", place={"pci": "a"})
        jobs.image_begin.return_value = "token"
        svc = mock.Mock(lora_jobs=jobs)
        with mock.patch.object(images, "_gpu_turn", return_value=nullcontext("ok")):
            gen.engine.alive.return_value = False
            with images.gpu_turn(svc, gen, {}) as value:
                self.assertEqual(value, "ok")
            gen.need.assert_not_called()
            jobs.image_begin.assert_called_with(None, None)
            jobs.image_end.assert_called_with("token")
            gen.engine.alive.return_value = True
            with images.gpu_turn(svc, gen, {}):
                pass
            gen.need.assert_called_once_with({}, start=False)
            jobs.image_begin.assert_called_with(4096, None)
        idle = images.ImageGen.__new__(images.ImageGen)
        idle.cfg = {}
        idle.tok = object()
        idle.engine = mock.Mock()
        idle.engine.alive.return_value = False
        with self.assertRaises(images.ImageJobFailed):
            idle._ready(start=False)
        idle.engine.start.assert_not_called()

    def test_move_saves_state_and_restarts(self):
        rec = self._draft()
        self._start(rec, steps=300, save_every=50)
        wait_for(lambda: (self.jobs.get(rec["id"])["progress"] or {}).get("step", 0) >= 3, what="some steps")
        self.tb.mode = "move"
        wait_for(lambda: (self.jobs.dir_of(rec["id"]) / "out" / "lora.resumed").exists(),
                 timeout=60, what="the restart with --resume")
        self.assertEqual(self.jobs.place["pci"], self.tb.move_to)
        self.jobs.cancel(rec["id"])
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "cancelled", what="the cancel")

    def test_vram_line_extra_fields(self):
        rec = self._draft()
        self._start(rec, steps=60, save_every=20)
        wait_for(lambda: self.jobs._train_state.get("state") == "running", what="a vram line")
        st = self.jobs._train_state
        self.assertEqual(st["used_mib"], 100)          # the trailing budget_mib= field ignored
        self.assertEqual(st["peak_est_mib"], 14848)
        self.assertEqual(st["need_mib"], 8252)
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "awaiting_choice", what="the end")

    # ---- the models row and the status
    def test_model_rows(self):
        rows = lt.model_rows(self.svc)
        self.assertEqual(len(rows), 4)               # the mage-flow one and the other families' (no trainer built)
        self.assertTrue(rows[0]["can_train"])
        self.assertEqual(rows[0]["family"], "mage-flow")
        self.assertFalse(any(r["can_train"] for r in rows[1:]))
        qwen = next(r for r in rows if r["model"] == "qwen-image-2.1")
        self.assertIn("not built", qwen["why"])
        self.assertIn("--image-lora-training", qwen["why"])
        (self.root / "engine" / "xestrata-image-mf-train").unlink()
        rows = lt.model_rows(self.svc)
        self.assertFalse(rows[0]["can_train"])
        self.assertIn("not built", rows[0]["why"])

    def test_distilled_models_use_full_step_base(self):
        from serve import images as img
        self.assertNotIn("moss-image-base-4-step", lt.DISTILLED)
        self.assertNotIn("moss-image-base-4-step-uncensored", lt.DISTILLED)
        self.assertEqual(lt.training_model("moss-image-sft-4-step-uncensored")[0], "moss-image-sft-uncensored")
        cases = [("mage-flow-turbo", "mage-flow"), ("moss-image-sft-4-step", "moss-image-sft"),
                 ("anima-turbo", "anima-base"),
                 ("qwen-image-2.1-turbo", "qwen-image-2.1")]
        families = {"mage-flow": "mage-flow", "moss-image-sft": "moss-image", "anima-base": "anima",
                    "qwen-image-2.1": "qwen-image"}
        for requested, base in cases:
            with self.subTest(model=requested):
                gen = FakeGen(self.root, model=base, family=families[base])
                self.svc.images.gens[base] = gen
                derivative = FakeGen(self.root, model=requested, family=families[base])
                derivative.cfg["few_step"] = {"lora": "BAD_LORA", "time_path": "BAD_TIME", "dataspace": "BAD_DATA"}
                self.svc.images.gens[requested] = derivative
                if gen.family == "moss-image":
                    moss = self.root / base
                    moss.mkdir(exist_ok=True)
                    (moss / "config.json").touch()
                    for i in range(1, 4):
                        (moss / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors").touch()
                    gen.cfg["diffusion_model"] = str(moss)
                if base == "anima-base":
                    gen.cfg["t5_tokenizer"] = gen.cfg["vae"]
                if base == "qwen-image-2.1":
                    (Path(gen.cfg["diffusion_model"]).parent / "qwen-image-2.1-F16.gguf").write_bytes(b"base")
                rec = self.jobs.create({"model": requested, "name": "x", "kind": "character", "trigger": "x"},
                                       [("x.png", self._png())])
                self.assertEqual(rec["model"], base)
                self.assertTrue(rec["model_switched"])
                rec["settings"] = lt.check_settings({}, rec["family"])
                argv = str(lt.TrainJobs.preprocess_argv(self.jobs, rec)) + str(self.jobs.train_argv(rec, 100, False))
                self.assertNotIn("BAD_", argv)
                rows = lt.model_rows(self.svc)
                self.assertFalse(any(r["model"] == requested for r in rows))
                self.assertTrue(any(r["model"] == base for r in rows))
                # The chat tool's choice is the same one exposed to the panel.
                tool = img.ImageTools._train_lora_tool(type("T", (), {"svc": self.svc})(),
                                                       {"model": requested}, time.monotonic())
                self.assertEqual(tool["training_model"], base)
                self.assertTrue(tool["model_switched"])
        del self.svc.images.gens["mage-flow"]
        self.svc.images.pick = lambda name: self.svc.images.gens.get(name, self.svc.images.gens["mage-flow-turbo"])
        with self.assertRaisesRegex(lt.TrainError, "--image-lora-training"):
            self.jobs.create({"model": "mage-flow-turbo"}, [("x.png", self._png())])
        row = next(r for r in lt.model_rows(self.svc) if r["model"] == "mage-flow")
        self.assertFalse(row["can_train"])
        self.assertIn("--image-lora-training", row["why"])

    def test_status_row(self):
        self.assertIsNone(lt.status_row(self.svc))
        rec = self._draft()
        self._start(rec, steps=60, save_every=20)
        row = wait_for(lambda: (r := lt.status_row(self.svc)) and r.get("place") and r, what="the status row")
        self.assertEqual(row["id"], rec["id"])
        self.assertIn(row["place"]["mode"], ("run", "shrunk", "paused"))
        wait_for(lambda: self.jobs.get(rec["id"])["status"] == "awaiting_choice", what="the end")


if __name__ == "__main__":
    unittest.main()
