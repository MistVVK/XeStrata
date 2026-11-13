# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The picture reader falls back to the processor in every mode instead of ending the launch (serve/server.py Vision).

A resident, lazy or elastic GPU reader that does not start - the card full, a kernel-image error, a card another
program holds - used to end the server's images and, at a launch, the launch.  The processor's reader (setup writes it
as the vision section's "fallback") reads instead, with the person's image limit, and the reason is kept for
/health (vision.fallback) and the About page (engine.vision_note).  The readers here are small Python scripts run as
real processes.

    python -m unittest serve.test_vision_cpu_fallback
"""
import contextlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.server as server  # noqa: E402


def plain_popen(what, args, **kw):                 # the real process, without the lifeline's anchor and watcher
    kw.pop("role", None)
    return subprocess.Popen(args, **kw)


class Fallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.procs = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                p.wait(timeout=10)

    def reader(self, body: str) -> str:
        script = self.dir / ("reader%d.py" % len(list(self.dir.glob("reader*.py"))))
        script.write_text("#!/usr/bin/env python3\nimport sys, time\n" + textwrap.dedent(body), encoding="utf-8")
        script.chmod(0o755)                         # the config's "exe" is the program itself, as setup's reader is
        return str(script)

    def vision(self, gpu_body: str, cpu_body: str | None = None, gpu: bool = True, **more) -> server.Vision:
        log = open(self.dir / "engine.log", "a", encoding="utf-8")
        self.addCleanup(log.close)
        cfg = {"exe": self.reader(gpu_body), "mmproj": str(self.dir / "mmproj.gguf"),
               "model": str(self.dir / "model.gguf"), "gpu": gpu, **more}
        if cpu_body is not None:
            cfg["fallback"] = {"exe": self.reader(cpu_body), "mmproj": cfg["mmproj"], "model": cfg["model"],
                               "gpu": False}
        v = server.Vision(cfg, log=log, lazy=True)
        self.addCleanup(v.shutdown)

        def keep(what, args, **kw):
            p = plain_popen(what, args, **kw)
            self.procs.append(p)
            return p

        patcher = mock.patch.object(server, "popen", keep)
        patcher.start()
        self.addCleanup(patcher.stop)
        return v

    OK_READER = """
        print("READY 2560", flush=True)
        sys.stdin.readline()
    """
    FAILS = """
        print("ERR cannot load the vision encoder x.gguf (no such file)", flush=True)
        sys.exit(1)
    """

    def test_a_gpu_reader_that_does_not_start_falls_back_to_the_processor(self):
        v = self.vision(self.FAILS, self.OK_READER)
        v.restart()
        self.assertTrue(v.on_cpu)
        self.assertEqual(v.label, "the CPU encoder")
        self.assertFalse(v.stopped)
        self.assertIsNone(v.proc.poll(), "the processor's reader runs")
        self.assertIn("ERR cannot load the vision encoder", v.fallback_why)
        self.assertIn("the picture reader could not start", v.cpu_note)     # not a full card: the log names the why
        self.assertIn("the engine's log", v.cpu_note)
        v.close()

    def test_a_full_card_says_so_in_the_persons_words(self):
        v = self.vision("""
            sys.stderr.write("alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736\\n")
            sys.stderr.flush()
            sys.exit(1)
        """, self.OK_READER, gpu=True, gpu_pci="0000:03:00.0")
        with mock.patch("serve.telemetry.free_vram_mib", lambda pci=None: 400):
            v.restart()
        self.assertIn("Pictures are read on the processor", v.cpu_note)
        self.assertIn("card 0000:03:00.0 has no room", v.cpu_note)
        self.assertIn("0.4 GB are free", v.cpu_note)
        self.assertIn("to read them on the card, close the other programs".replace("to read", "To read"),
                      v.cpu_note)
        v.close()

    def test_without_a_processor_reader_the_start_fails(self):
        v = self.vision(self.FAILS)
        with self.assertRaises(RuntimeError) as cm:
            v.restart()
        self.assertIn("did not start", str(cm.exception))

    def test_a_processor_reader_that_fails_too_raises_its_own_error(self):
        v = self.vision(self.FAILS, "sys.exit(3)\n")
        with self.assertRaises(RuntimeError) as cm:
            v.restart()
        self.assertIn("exit code 3", str(cm.exception))

    def test_an_elastic_reader_on_the_processor_tries_the_card_again(self):
        v = self.vision(self.FAILS, self.OK_READER, gpu=True, gpu_pci="0000:09:00.0", elastic=True)
        v.restart()
        self.assertTrue(v.on_cpu)
        v.close()                                       # the idle stop of an elastic reader
        self.assertFalse(v.on_cpu)
        self.assertIn("did not start", v.fallback_why)   # the reason stays for /health
        v.restart()                                     # the card fails again: the processor reads again
        self.assertTrue(v.on_cpu)
        v.close()


if __name__ == "__main__":
    unittest.main()
