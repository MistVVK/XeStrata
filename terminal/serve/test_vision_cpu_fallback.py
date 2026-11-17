"""The picture reader reads on the processor instead of ending the launch (owner, 2026-11-10): the NVIDIA reader (strata-vision) is the CPU reader too,
without --gpu, so a GPU reader that does not start - resident or on demand, whatever the cause - falls back to it, with one log line; when the cause was a
card with no room, the words for the Terminal's summary go with it (Vision.cpu_note, /metrics engine.vision_note, Runner.report_vision).

    python -m serve.test_vision_cpu_fallback
"""
import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from strata_cli.launch import runner as R

ALLOC = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory"
NOTE = ("Pictures are read on the processor, which is slower: graphics card 2 has no room for the picture reader "
        "(it needs about 1.5 GB; 0.1 GB are free). To read them on the card, close the other programs that use it and start again.")


def plain_popen(what, args, **kw):
    kw.pop("role", None)
    return subprocess.Popen(args, **kw)


class Construction(unittest.TestCase):
    def make(self, **vision):
        cfg = {"exe": "/engine/strata-vision", "mmproj": "/m/mmproj.gguf", "model": "/m/model.gguf", "gpu": True, **vision}
        with mock.patch.object(server.Vision, "work_dir", return_value=Path(tempfile.gettempdir())):
            return server.Vision(cfg, lazy=True)

    def test_the_nvidia_reader_has_the_cpu_reader_in_every_mode(self):
        for extra in ({}, {"elastic": True}, {"elastic": False}, {"cuda_device": 1}):
            with self.subTest(extra=extra):
                v = self.make(**extra)
                self.assertIn("--gpu", v.spawn[0])
                self.assertIsNotNone(v.fallback)
                self.assertNotIn("--gpu", v.fallback[0])
                self.assertTrue(v.fallback[0][0].endswith("strata-vision"))
                self.assertIsNone(v.cpu_note)

    def test_the_persons_image_limit_goes_with_it(self):
        v = self.make(max_tokens=1024)
        self.assertEqual(v.fallback[0][v.fallback[0].index("--max-tokens") + 1], "1024")

    def test_a_cpu_reader_has_nothing_to_fall_back_to(self):
        self.assertIsNone(self.make(gpu=False).fallback)

    def test_amd_readers_keep_their_own_cpu_reader(self):
        self.assertIsNone(self.make(exe="/engine/strata-vision-hip").fallback)                       # no cpu_exe next to it: nothing to fall back to
        v = self.make(exe="/engine/strata-vision-hip", cpu_exe="/engine/strata-vision")
        self.assertTrue(v.fallback[0][0].endswith("strata-vision"))


class Fallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.procs, self.logs = [], []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
            for stream in (p.stdin, p.stdout):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        for f in self.logs:
            with contextlib.suppress(OSError):
                f.close()

    def script(self, body):
        path = Path(self.tmp.name) / ("reader%d.py" % len(list(Path(self.tmp.name).glob("reader*.py"))))
        path.write_text("import os, sys, time\n" + textwrap.dedent(body), encoding="utf-8")
        return [sys.executable, "-u", str(path)]

    def vision(self, gpu_body, cpu_body, env=None):
        path = Path(self.tmp.name) / "engine.log"
        path.write_text("", encoding="utf-8")
        log = open(path, "a", encoding="utf-8")
        self.logs.append(log)
        v = server.Vision.__new__(server.Vision)
        v.spawn = (self.script(gpu_body), log, env)
        v.fallback = (self.script(cpu_body), log, env)
        v.gpu_spawn, v.cpu_spawn = v.spawn, v.fallback
        v.dir = Path(self.tmp.name)
        v.stopped, v.proc, v.on_cpu, v.cpu_note, v.elastic = True, None, False, None, False
        return v

    def start(self, v):
        out, error = io.StringIO(), None

        def keep(what, args, **kw):
            p = plain_popen(what, args, **kw)
            self.procs.append(p)
            return p

        with mock.patch.object(server, "popen", keep), mock.patch.object(server, "card_free_mib", lambda index, amd=False: 114), contextlib.redirect_stdout(out):
            try:
                v._start()
            except RuntimeError as e:
                error = e
        return error, out.getvalue()

    OOM = """
        sys.stderr.write(%r + "\\n")
        sys.stderr.flush()
        print("[New LWP 5]", flush=True)
        sys.exit(1)
    """ % ALLOC
    READY = """
        print("READY 2560", flush=True)
        sys.stdin.readline()
    """

    def test_a_full_card_reads_on_the_processor_and_says_so(self):
        v = self.vision(self.OOM, self.READY, env={"CUDA_VISIBLE_DEVICES": "2,3"})
        error, said = self.start(v)
        self.assertIsNone(error)
        self.assertTrue(v.on_cpu)
        self.assertTrue(v.alive())
        self.assertEqual(v.cpu_note, NOTE)                       # card 2: the reader's card 0 of "2,3"; 114 MiB free read after it failed
        self.assertIn("[strata] the GPU picture reader did not start", said)
        self.assertIn("pictures are read on the CPU", said)

    def test_the_need_in_the_note_follows_the_image_limit(self):
        v = self.vision(self.OOM, self.READY, env={"CUDA_VISIBLE_DEVICES": "2,3"})
        v.max_tokens = 16384
        self.start(v)
        self.assertIn("(it needs about 3.6 GB; 0.1 GB are free)", v.cpu_note)

    def test_a_reader_that_starts_changes_nothing(self):
        v = self.vision(self.READY, self.OOM)
        error, said = self.start(v)
        self.assertIsNone(error)
        self.assertFalse(v.on_cpu)
        self.assertIsNone(v.cpu_note)
        self.assertNotIn("did not start", said)

    def test_another_cause_falls_back_too_without_the_full_card_words(self):
        v = self.vision("""
            print("ERR cannot load the vision encoder x.gguf", flush=True)
            sys.exit(1)
        """, self.READY)
        error, said = self.start(v)
        self.assertIsNone(error)
        self.assertTrue(v.on_cpu)
        self.assertIn("ERR cannot load the vision encoder", said)
        self.assertEqual(v.cpu_note, "Pictures are read on the processor, which is slower: the picture reader could not start on graphics card 0. "
                                     "The reason is in the engine's log: " + self.logs[0].name + ".")

    def test_the_card_in_that_sentence_is_the_first_one_the_reader_is_shown(self):
        v = self.vision("""
            print("ERR cannot load the vision encoder x.gguf", flush=True)
            sys.exit(1)
        """, self.READY, env={"CUDA_VISIBLE_DEVICES": "3,1"})
        self.start(v)
        self.assertIn("could not start on graphics card 3. The reason", v.cpu_note)

    def test_on_a_mac_there_is_no_card_number(self):
        v = self.vision("""
            print("ERR cannot load the vision encoder x.gguf", flush=True)
            sys.exit(1)
        """, self.READY)
        with mock.patch.object(server.sys, "platform", "darwin"):
            self.start(v)
        self.assertTrue(v.cpu_note.startswith("Pictures are read on the processor, which is slower: the picture reader could not start. The reason is in"))

    def test_the_cpu_reader_failing_too_ends_it_with_its_own_error(self):
        v = self.vision(self.OOM, """
            print("ERR cannot open the text model x.gguf (no such file)", flush=True)
            sys.exit(1)
        """)
        error, _ = self.start(v)
        self.assertIsNotNone(error)
        self.assertIn("ERR cannot open the text model", str(error))

    def test_the_card_is_unknown_when_the_reader_saw_no_such_card(self):
        v = self.vision(self.OOM, self.READY, env={"CUDA_VISIBLE_DEVICES": ""})
        v.spawn = (v.spawn[0], v.spawn[1], {"CUDA_VISIBLE_DEVICES": "GPU-8c3f"})
        error, _ = self.start(v)
        self.assertIsNone(error)
        self.assertEqual(v.cpu_note, "Pictures are read on the processor, which is slower: the graphics card has no room for the picture reader "
                                     "(it needs about 1.5 GB). To read them on the card, close the other programs that use it and start again.")

    def test_an_on_demand_reader_that_goes_up_again_loses_the_note(self):
        v = self.vision(self.OOM, self.READY, env={"CUDA_VISIBLE_DEVICES": "2,3"})
        v.elastic = True
        self.start(v)
        self.assertEqual(v.cpu_note, NOTE)
        v.close = lambda: None
        v.unload()
        self.assertIsNone(v.cpu_note)


class ShownInTheSummary(unittest.TestCase):
    def runner(self):
        r = R.Runner.__new__(R.Runner)
        r.notices, r.events = [], []
        r.emit = lambda stage, text, pct=None, level="info", **data: r.events.append((stage, text, level))
        return r

    def test_the_note_from_the_server_is_a_notice_and_an_event(self):
        r = self.runner()
        r.report_vision({"vision_note": NOTE})
        self.assertEqual(r.notices, [NOTE])
        self.assertEqual(r.events, [("start", NOTE, "warn")])

    def test_it_is_said_once(self):
        r = self.runner()
        r.report_vision({"vision_note": NOTE})
        r.report_vision({"vision_note": NOTE})
        self.assertEqual(len(r.notices), 1)

    def test_no_note_no_line(self):
        r = self.runner()
        r.report_vision({})
        r.report_vision({"vision_note": None})
        self.assertEqual((r.notices, r.events), ([], []))


if __name__ == "__main__":
    unittest.main()
