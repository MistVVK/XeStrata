"""Native failure diagnostics survive EOF without recording the token stream."""
import io
import queue
import subprocess
from types import SimpleNamespace
import unittest

from serve import server


def engine(lines, log):
    result = object.__new__(server.StrataEngine)
    result.lines, result.slot_q = queue.Queue(), [queue.Queue()]
    result.slot_busy = [True]                       # the pump drops a frame for a slot no request has claimed (#4451): slot 0 is claimed
    result.log, result.ended, result.last_err = log, False, None
    result.terminated = []
    result.proc = SimpleNamespace(stdout=iter(lines), terminate=lambda: result.terminated.append(True))
    return result


class EngineErrorLog(unittest.TestCase):
    def test_error_survives_eof_but_tokens_are_not_written(self):
        log = io.StringIO()
        item = engine(['T 11\n', 'BT 0 12\n', 'ERR verify batch: graph failed\n'], log)
        item._pump()
        self.assertIn('ERR verify batch: graph failed', log.getvalue())
        self.assertEqual(log.getvalue().count('ERR verify batch: graph failed'), 1)
        self.assertNotIn('T 11', log.getvalue())
        self.assertNotIn('BT 0 12', log.getvalue())
        self.assertEqual([item.lines.get_nowait() for _ in range(3)],
                         ['T 11\n', 'ERR verify batch: graph failed\n', None])
        self.assertEqual([item.slot_q[0].get_nowait() for _ in range(2)], ['BT 0 12\n', None])

    def test_fatal_diagnostic_is_written_before_termination(self):
        log = io.StringIO()
        item = engine(['ERR verify: timed out at layer 8\n', 'DONE 0\n'], log)
        item.proc.terminate = lambda: item.terminated.append(log.getvalue())
        item._pump()
        self.assertEqual(len(item.terminated), 1)
        self.assertIn('ERR verify: timed out at layer 8', item.terminated[0])
        self.assertIsNone(item.lines.get_nowait())
        self.assertTrue(item.lines.empty())

    def test_original_pump_does_not_write_to_a_successor_log(self):
        original, successor = io.StringIO(), io.StringIO()
        item = engine([], original)

        def output():
            item.log = successor
            yield 'ERR verify batch: old process failed\n'

        item.proc.stdout = output()
        item._pump()
        self.assertIn('ERR verify batch: old process failed', original.getvalue())
        self.assertEqual(successor.getvalue(), '')

    def test_unavailable_log_does_not_block_eof_cleanup(self):
        closed = io.StringIO()
        closed.close()
        for log in (None, subprocess.DEVNULL, closed):
            with self.subTest(log=log):
                item = engine(['ERR verify batch: graph failed\n'], log)
                item._pump()
                self.assertTrue(item.ended)
                self.assertEqual(item.lines.get_nowait(), 'ERR verify batch: graph failed\n')
                self.assertIsNone(item.lines.get_nowait())
                self.assertIsNone(item.slot_q[0].get_nowait())


if __name__ == '__main__':
    unittest.main()
