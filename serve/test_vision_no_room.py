# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""A picture reader that could not start because its card was full: the cause is named, with the card, the figures
and the next step (serve/reader_room.py, serve/server.py Vision._start_failure).

The person whose launch quietly lost the GPU reader read only the last line of the engine log before (a backtrace
frame, or the reader's bare ERR).  The reader's own out-of-memory lines say what happened; the words are one set for
the log and the About page.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.server as server  # noqa: E402
from serve import reader_room  # noqa: E402


class Lines(unittest.TestCase):
    def test_a_ggml_allocation_says_the_card_and_the_size(self):
        room = reader_room.find("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 1: "
                                "cudaMalloc failed: out of memory")
        self.assertEqual((room.device, round(room.need_mib)), (1, 865))

    def test_a_buffer_line_gives_the_size_in_mib(self):
        room = reader_room.find(["alloc_tensor_range: failed to allocate CUDA0 buffer of size 907524736"])
        self.assertEqual((room.device, round(room.need_mib)), (0, 865))

    def test_xestratas_runtime_names_count_too(self):
        for line in ("ERR cannot load the vision encoder x.gguf (vision: copy: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)",
                     "PI_ERROR_OUT_OF_RESOURCES", "Memory allocation failed"):
            self.assertIsNotNone(reader_room.find(line), line)

    def test_a_plain_error_is_not_a_full_card(self):
        self.assertIsNone(reader_room.find(["ERR cannot read the image x.png (no such file)"]))

    def test_the_words(self):
        text = reader_room.on_processor("0000:03:00.0", 2300, 400)
        self.assertIn("Pictures are read on the processor", text)
        self.assertIn("card 0000:03:00.0 has no room", text)
        self.assertIn("it needs about 2.2 GB; 0.4 GB are free", text)
        self.assertIn("close the other programs that use it and start again", text)
        self.assertNotIn("(", reader_room.on_processor())            # nothing known: no figures
        on_cpu = reader_room.not_started_on_processor("0000:03:00.0", "/tmp/e.log")
        self.assertIn("could not start on card 0000:03:00.0", on_cpu)
        self.assertIn("the engine's log: /tmp/e.log", reader_room.not_started_on_processor(None, "/tmp/e.log"))

    def test_the_start_error_carries_the_readers_own_line(self):
        room = reader_room.find("allocating 865.48 MiB on device 1: cudaMalloc failed: out of memory")
        text = reader_room.start_error(room, ["it exited (exit code 1)", "allocating 865.48 MiB on device 1: "
                                              "cudaMalloc failed: out of memory"])
        self.assertIn("no room for it (it asked for 865 MiB on card 1)", text)
        self.assertIn("cudaMalloc failed", text)


class AStartOnAFullCard(unittest.TestCase):
    def test_the_reader_that_died_of_a_full_card_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "reader.py"
            script.write_text("import sys\n"
                              "sys.stderr.write('alloc_tensor_range: failed to allocate CUDA0 buffer of size "
                              "907524736\\n')\n"
                              "sys.stderr.flush()\n"
                              "sys.exit(1)\n", encoding="utf-8")
            log = open(Path(d) / "engine.log", "w", encoding="utf-8")     # one log for every start, as the server's
            self.addCleanup(log.close)
            v = server.Vision.__new__(server.Vision)
            v.spawn = ([sys.executable, "-u", str(script)], log, None)
            v.dir, v.stopped, v.proc, v.dead, v.cache = Path(d), True, None, None, {}
            v.fallback = v.fallback_why = v.cpu_note = v.cpu_spawn = v.cpu_label = None
            v.gpu_spawn, v.on_cpu, v.elastic, v.label = v.spawn, False, False, "the GPU encoder"
            with mock.patch.object(server, "VISION_READY_S", 30):
                with self.assertRaises(server.VisionNoRoom) as cm:
                    v._start()
            text = str(cm.exception)
            self.assertIn("no room for it", text)
            self.assertIn("it asked for 865 MiB", text)
            self.assertIn("failed to allocate", text)


if __name__ == "__main__":
    unittest.main()
