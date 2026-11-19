"""#879 conversation-cache clear logs, through local fixtures and HTTP only."""
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, ConvCacheLog, MockEngine, Service, serve


class ConversationCacheClear(unittest.TestCase):
    """Completed non-finite recovery clears the engine's parked conversation cache."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=4 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"
    RESTORED_CLEAR = ("strata serve: after non-finite logits (#879): the restored session replaces the live one; "
                      "dropped 1 parked conversation\n")
    LIVE_CLEAR = ("strata serve: after non-finite logits (#879): dropped the live session, 3 checkpoints, 0 parked "
                  "conversations and 2 idle slots (1 running slot not kept after their reply); reading this prompt from token 0\n")

    def test_clear_formats_preserve_history_for_zero_one_and_multiple_entries(self):
        lines = [self.RESTORED_CLEAR,
                 self.RESTORED_CLEAR.replace("1 parked conversation", "2 parked conversations"),
                 self.RESTORED_CLEAR.replace("1 parked conversation", "0 parked conversations"),
                 self.LIVE_CLEAR,
                 self.LIVE_CLEAR.replace("3 checkpoints", "1 checkpoint").replace("0 parked conversations", "1 parked conversation")
                 .replace("2 idle slots", "1 idle slot"),
                 self.LIVE_CLEAR.replace("0 parked conversations", "2 parked conversations")
                 .replace("1 running slot", "2 running slots")]
        for line in lines:
            with self.subTest(clear=line), tempfile.TemporaryDirectory() as d:
                log = Path(d) / "strata.log"
                # Zero parked entries can still have retained K/V bytes, which clear() also releases.
                log.write_text(self.PARK + self.RESTORE.replace("parked=2", "parked=0"), encoding="utf-8")
                cache = ConvCacheLog()
                before = cache.poll(str(log), 0)
                with log.open("a", encoding="utf-8") as f:
                    f.write(line)
                after = cache.poll(str(log), 0)
                self.assertEqual(after, {**before, "parked": 0, "bytes": 0})
                self.assertEqual(cache.poll(str(log), 0), after)

    def test_completed_clear_lines_reset_current_cache_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK + self.RESTORE, encoding="utf-8")
            c = ConvCacheLog()
            st = c.poll(str(log), 0)
            self.assertEqual((st["parked"], st["bytes"], st["evictions"], st["parks"], st["restores"],
                              st["last_event"], st["last_tokens"]), (2, 209715200, 4, 1, 1, "restored", 4000))
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: non-finite logits warning only\n")
            self.assertEqual(c.poll(str(log), 0)["parked"], 2)  # a warning alone is not an authoritative clear
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: after non-finite logits (#879): dropped the live session, 3 checkpoints, 0 parked")
            self.assertEqual(c.poll(str(log), 0)["parked"], 2)  # incomplete lines are deferred
            with open(log, "a", encoding="utf-8") as f:
                f.write(" conversations and 2 idle slots (1 running slot not kept after their reply); reading this prompt from token 0\n")
            st = c.poll(str(log), 0)
            self.assertEqual((st["parked"], st["bytes"], st["evictions"], st["parks"], st["restores"],
                              st["last_event"], st["last_tokens"]), (0, 0, 4, 1, 1, "restored", 4000))
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK + self.RESTORED_CLEAR + self.RESTORE)
            st = c.poll(str(log), 0)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"], st["last_event"], st["last_tokens"]),
                             (2, 209715200, 2, 2, "restored", 4000))

    def test_unrelated_lines_containing_clear_text_are_ignored(self):
        for clear in (self.RESTORED_CLEAR, self.LIVE_CLEAR):
            with self.subTest(clear=clear), tempfile.TemporaryDirectory() as d:
                log = Path(d) / "strata.log"
                log.write_text(self.PARK, encoding="utf-8")
                cache = ConvCacheLog()
                before = cache.poll(str(log), 0)
                with log.open("a", encoding="utf-8") as f:
                    f.write("strata serve: diagnostic: " + clear)
                    f.write(clear.rstrip("\n") + " (recovery pending)\n")
                self.assertEqual(cache.poll(str(log), 0), before)

    def test_metrics_reports_a_completed_clear(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK, encoding="utf-8")
            tok = ByteTokenizer()
            engine = MockEngine(tok, "Hello.", max_context=4096)
            engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
            engine.log_path, engine.log_start = str(log), 0
            svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
            httpd = serve(svc, port=0)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                    before = json.loads(r.read())["conversation_cache"]
                self.assertEqual((before["parked"], before["bytes"], before["evictions"]), (1, 104857600, 4))
                with open(log, "a", encoding="utf-8") as f:
                    f.write(self.RESTORED_CLEAR)
                with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                    after = json.loads(r.read())["conversation_cache"]
                self.assertEqual((after["parked"], after["bytes"], after["evictions"]), (0, 0, 4))
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
