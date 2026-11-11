"""Item 1 of lane NET: the silent wait in launch step 5. setup.download() against a server that accepts and never answers, a refusing
port and a working one (the rule's limits scaled down); what `strata launch` shows while setup waits (the waiting line, the heartbeat in
the download phase, the plain printer); setup's [X] reason in the launch error and the setup log."""

from __future__ import annotations

import contextlib
import io
import re
import time
import unittest
from unittest import mock

import setup
from strata_cli.launch import cmd
from strata_cli.launch.runner import Event, LaunchError, Runner, capital
from strata_control import netpolicy, progress_words
from tests.terminal.test_flow2_022 import Base as FlowBase
from tests.terminal.test_netpolicy import AT_ONCE_S, REFUSED_WAIT_LINE_S, REFUSED_WINDOW, BlackHole, Server, ok_body, refused_port


def scaled():
    """The rule with sub-second limits: 0.4 s, 0.1 s gap, 0.8 s; the waiting line after 0.2 s, then every 0.3 s."""
    return [mock.patch.object(netpolicy, "CONNECT_LIMITS", (0.4, 0.8)), mock.patch.object(netpolicy, "RETRY_GAP_S", 0.1),
            mock.patch.object(netpolicy, "WAIT_LINE_AFTER_S", 0.2), mock.patch.object(netpolicy, "WAIT_LINE_EVERY_S", 0.3),
            mock.patch.dict("os.environ", {"STRATA_SOURCE": "huggingface"})]


class SetupDownloadWaits(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        for p in scaled():
            p.start()
            self.addCleanup(p.stop)
        self.memory = netpolicy.HostMemory()
        p = mock.patch.object(netpolicy, "HOSTS", self.memory)
        p.start()
        self.addCleanup(p.stop)
        t = tempfile.TemporaryDirectory(prefix="strata-netwait-")
        self.addCleanup(t.cleanup)
        self.dir = Path(t.name)

    def download(self, url, name="model-00001-of-00002.gguf", what=None):
        buf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stdout(buf):
            try:
                setup.download(url, self.dir / name, what)
                code = 0
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue(), time.monotonic() - t0

    def test_a_black_hole_shows_the_host_soon_and_fails_fast_with_the_reason(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        code, out, took = self.download(f"http://127.0.0.1:{hole.port}/m/model-00001-of-00002.gguf")
        self.assertEqual(code, 1, out)
        self.assertLess(took, 2.5, out)                                  # 0.4 + 0.1 + 0.8, not 5 x 60 + 4 x 5 s
        self.assertRegex(out, r"(?m)^  Waiting for 127\.0\.0\.1 to answer \(0 s so far\) \.\.\.$")
        self.assertRegex(out, r"\[X\]  127\.0\.0\.1 did not answer within 1 seconds, so the model files could not be downloaded\.")
        self.assertIn("Check that this computer can open https://127.0.0.1, then run the same command again: what is already downloaded is kept. "
                      "Models that are already on this computer still start without the internet.", out)
        self.assertEqual(len(hole.held), 2)
        # the next file from the same server: at once, with the same sentence
        code, out, took = self.download(f"http://127.0.0.1:{hole.port}/m/model-00002-of-00002.gguf", "model-00002-of-00002.gguf")
        self.assertEqual(code, 1)
        self.assertLess(took, 0.3)
        self.assertIn("[X]  127.0.0.1 did not answer a moment ago either, so the model files could not be downloaded.", out)
        self.assertEqual(len(hole.held), 2)

    def test_a_short_model_list_read_does_not_cost_the_download_its_own_attempt(self):
        """The model-list read at the start of setup (5 s, optional) fails on a black hole; the engine or model download from the same host
        afterwards still gets its own full window and says W3, not W5 (coordinator 11-10)."""
        from strata_control import transfer
        hole = BlackHole()
        self.addCleanup(hole.close)
        with self.assertRaises(transfer.TransferError):
            transfer.read_small(f"http://127.0.0.1:{hole.port}/strata-feed.json", timeout=0.2)
        self.assertEqual(len(hole.held), 2)
        code, out, took = self.download(f"http://127.0.0.1:{hole.port}/m/model-00001-of-00002.gguf")
        self.assertEqual(code, 1)
        self.assertGreater(took, 1.0)
        self.assertEqual(len(hole.held), 4)
        self.assertIn("[X]  127.0.0.1 did not answer within 1 seconds, so the model files could not be downloaded.", out)
        self.assertNotIn("a moment ago", out)

    def test_the_picture_reader_is_named(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        code, out, _t = self.download(f"http://127.0.0.1:{hole.port}/m/mmproj.gguf", "mmproj.gguf", "vision encoder")
        self.assertIn("so the picture reader could not be downloaded.", out)

    def test_a_refusing_port_fails_at_once_with_no_waiting_line(self):
        port = refused_port()
        with mock.patch.object(netpolicy, "CONNECT_LIMITS", REFUSED_WINDOW), mock.patch.object(netpolicy, "WAIT_LINE_AFTER_S", REFUSED_WAIT_LINE_S):
            code, out, took = self.download(f"http://127.0.0.1:{port}/m/model-00001-of-00002.gguf")
        self.assertEqual(code, 1)
        self.assertLess(took, AT_ONCE_S)
        self.assertNotIn("Waiting for", out)
        self.assertIn("[X]  This computer could not connect to 127.0.0.1 (it looks offline), so the model files could not be downloaded.", out)

    def test_a_working_server_still_downloads_and_prints_no_waiting_line(self):
        srv = Server(ok_body(b"x" * 1000))
        self.addCleanup(srv.close)
        code, out, _t = self.download(srv.url("/m/small.bin"), "small.bin")
        self.assertEqual(code, 0, out)
        self.assertEqual((self.dir / "small.bin").read_bytes(), b"x" * 1000)
        self.assertNotIn("Waiting for", out)


class LaunchShowsTheWait(FlowBase):
    def run_lines(self, steps, heartbeat=0.3):
        r = self.runner(family="qwen", model="Q2_0", vision="off", context=32768)
        r.HEARTBEAT_S = heartbeat
        seen = []
        r.emit_cb = lambda ev: seen.append((time.monotonic(), ev.text))

        def slow():
            for delay, line in steps:
                time.sleep(delay)
                yield line + "\n"
        proc = mock.Mock()
        proc.stdout = slow()
        proc.wait.return_value = 1
        t0 = time.monotonic()
        with mock.patch.object(Runner, "_spawn", return_value=proc), mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(LaunchError) as cm:
                r.ensure_model()
        return [(t - t0, x) for t, x in seen], cm.exception

    def test_the_waiting_line_is_shown_and_the_reason_is_the_error(self):
        seen, err = self.run_lines([
            (0, "=== Step 5: downloading Qwen3.8-Flash-Next Q2_0 ==="),
            (0.2, "  Waiting for huggingface.co to answer (5 s so far) ..."),
            (0.2, "  Waiting for huggingface.co to answer (15 s so far) ..."),
            (0, "  (huggingface.co: TimeoutError: timed out)"),
            (0, ""),
            (0, "  [X]  huggingface.co did not answer within 44 seconds, so the model files could not be downloaded."),
            (0, "       Check that this computer can open https://huggingface.co, then run the same command again: what is already "
                "downloaded is kept. Models that are already on this computer still start without the internet."),
        ])
        texts = [x for _t, x in seen]
        self.assertIn("Waiting for huggingface.co to answer (5 s so far)", texts)
        self.assertIn("Waiting for huggingface.co to answer (15 s so far)", texts)
        # the step's own sentence comes first, before setup has said anything (an order, not a stopwatch: a slow or loaded machine only moves both)
        self.assertFalse(seen[0][1].startswith("Waiting for"), seen[0])
        self.assertLess(seen[0][0], next(t for t, x in seen if x.startswith("Waiting for")))
        self.assertIn("The model could not be installed. huggingface.co did not answer within 44 seconds, so the model files could not "
                      "be downloaded.", err.text)
        self.assertIn("Check that this computer can open https://huggingface.co", err.next)

    def test_a_silent_download_phase_gets_the_heartbeat(self):
        seen, _err = self.run_lines([(0, "=== Step 5: downloading Qwen3.8-Flash-Next Q2_0 ==="), (1.3, "[X] stopped for the test")])
        beats = [x for _t, x in seen if "still working" in x]
        self.assertGreaterEqual(len(beats), 2, seen)
        self.assertTrue(beats[0].startswith("Getting the model files (still working, "), beats)


class Words(unittest.TestCase):
    def test_the_waiting_phase(self):
        w = progress_words.describe(5, ["=== Step 5: downloading x ===", "  Waiting for github.com to answer (25 s so far) ..."])
        self.assertEqual((w.phase, w.label), ("waiting", "Waiting for github.com to answer (25 s so far)"))
        w = progress_words.describe(5, ["=== Step 5: downloading x ===", "  Waiting for github.com to answer (25 s so far) ...",
                                        "  model.gguf:   0.10 / 66.40 GB (0%)"])
        self.assertNotEqual(w.phase, "waiting")

    def test_the_app_string_is_the_same_sentence(self):
        import json
        from pathlib import Path
        en = json.loads((Path(__file__).resolve().parents[2] / "apps" / "host" / "src" / "locales" / "net-en.json").read_text(encoding="utf-8"))
        self.assertEqual(en["net01"], netpolicy.WORDS["waiting"].replace("{host}", "{0}").replace("{seconds}", "{1}"))

    def test_a_host_name_keeps_its_small_letter(self):
        self.assertEqual(capital("huggingface.co did not answer"), "huggingface.co did not answer")
        self.assertEqual(capital("the update feed could not be read"), "The update feed could not be read")
        self.assertEqual(capital("no engine. is here"), "No engine. is here")


class PlainPrinter(unittest.TestCase):
    def test_the_second_download_shows_its_first_figure(self):
        buf = io.StringIO()
        with mock.patch.object(cmd.sys, "stdout", buf):
            emit = cmd._printer()
            emit(Event("model", "Getting Q2_0 ready…", 0))
            emit(Event("model", "Downloading the model files: 30.0 of 66.4 GB", 45))
            emit(Event("model", "Downloading the model files: 66.4 of 66.4 GB", 100))
            emit(Event("model", "Downloading the picture reader: 0.0 of 0.9 GB", 0))
            emit(Event("model", "Downloading the picture reader: 0.0 of 0.9 GB", 1))
            emit(Event("model", "Downloading the picture reader: 0.1 of 0.9 GB", 12))
        text = buf.getvalue()
        self.assertIn("Downloading the picture reader: 0.0 of 0.9 GB", text)
        self.assertIn("Downloading the picture reader: 0.1 of 0.9 GB", text)
        self.assertEqual(text.count("picture reader"), 2)               # the same sentence is still thinned to 10% steps


if __name__ == "__main__":
    unittest.main()
