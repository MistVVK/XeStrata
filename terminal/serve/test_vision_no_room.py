"""A picture reader that cannot start because its card has no room (serve/server.py vision_start_error, VisionNoRoom): the server names the cause
instead of the last line of whatever a debugger printed.  The readers are small Python scripts run as real processes; their stderr is a log file, as the
engine's log is in the product.

    python -m serve.test_vision_no_room
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

ALLOC = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory"
BUFFER = "alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736"
ASSERT = "ggml-backend.cpp:188: GGML_ASSERT(buffer) failed"


def plain_popen(what, args, **kw):
    kw.pop("role", None)
    return subprocess.Popen(args, **kw)


class NoRoom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.procs = []
        self.logs = []
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

    def log(self, text=""):
        path = Path(self.tmp.name) / ("engine%d.log" % len(self.logs))
        path.write_text(text, encoding="utf-8")
        f = open(path, "a", encoding="utf-8")
        self.logs.append(f)
        return f

    def reader(self, body: str, log):
        script = Path(self.tmp.name) / ("reader%d.py" % len(list(Path(self.tmp.name).glob("reader*.py"))))
        script.write_text("import os, sys, time\n" + textwrap.dedent(body), encoding="utf-8")
        v = server.Vision.__new__(server.Vision)
        v.spawn = ([sys.executable, "-u", str(script)], log, None)
        v.dir = Path(self.tmp.name)
        v.stopped, v.proc = True, None
        return v

    def start(self, v):
        error = None

        def keep(what, args, **kw):
            p = plain_popen(what, args, **kw)
            self.procs.append(p)
            return p

        with mock.patch.object(server, "popen", keep), contextlib.redirect_stdout(io.StringIO()):
            try:
                v._start_reader()
            except RuntimeError as e:
                error = e
        return error

    OOM_READER = """
        print("[New LWP 477206]", flush=True)
        sys.stderr.write(%r + "\\n" + %r + "\\n" + %r + "\\n")
        sys.stderr.flush()
        print("Python Exception <class 'gdb.error'>: Couldn't get registers: No such process.", flush=True)
        sys.exit(1)
    """ % (ALLOC, BUFFER, ASSERT)

    def test_the_v100_launch_in_miniature(self):
        v = self.reader(self.OOM_READER, self.log())
        error = self.start(v)
        self.assertIsInstance(error, server.VisionNoRoom)
        self.assertEqual(error.room.device, 0)
        self.assertAlmostEqual(error.room.need_mib, 865.48, places=2)
        text = str(error)
        self.assertIn("the vision encoder did not start: the graphics card has no room for it", text)
        self.assertIn("865 MiB", text)
        self.assertIn("its card 0", text)
        self.assertIn("cudaMalloc failed: out of memory", text)

    def test_a_reader_without_a_log_that_prints_it_on_its_output_is_found_too(self):
        v = self.reader("""
            print(%r, flush=True)
            sys.exit(1)
        """ % ALLOC, None)
        error = self.start(v)
        self.assertIsInstance(error, server.VisionNoRoom)

    def test_lines_an_earlier_start_left_in_the_shared_log_are_not_this_starts(self):
        log = self.log(ALLOC + "\n" + BUFFER + "\n" + ASSERT + "\n")           # the last launch's failure, still in the file
        v = self.reader("""
            print("ERR cannot open the text model x.gguf (no such file)", flush=True)
            sys.exit(1)
        """, log)
        error = self.start(v)
        self.assertNotIsInstance(error, server.VisionNoRoom)
        self.assertIn("ERR cannot open the text model", str(error))
        self.assertNotIn("no room", str(error))

    def test_a_card_that_is_not_there_is_not_a_full_card(self):
        v = self.reader("""
            sys.stderr.write("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device\\n")
            sys.exit(1)
        """, self.log())
        error = self.start(v)
        self.assertNotIsInstance(error, server.VisionNoRoom)
        self.assertNotIn("no room", str(error))

    def test_the_message_function_alone(self):
        log = self.log(ALLOC + "\n" + ASSERT + "\n")
        text = server.vision_start_error("", log, 134)
        self.assertIn("the graphics card has no room for it (it asked for 865 MiB on its card 0)", text)
        self.assertEqual(server.vision_start_error("ERR no mmproj\n"), "the vision encoder did not start: ERR no mmproj")      # unchanged

    def test_a_cause_the_server_names_already_keeps_its_own_words(self):
        busy = server.vision_start_error("cudaMalloc failed: CUDA-capable device(s) is/are busy or unavailable")
        self.assertIn("Exclusive_Process", busy)


if __name__ == "__main__":
    unittest.main()
