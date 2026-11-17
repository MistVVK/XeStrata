"""After an aborted request the engine and the server are back in step: the next requests get their OWN answers.

A request that ends on an engine ERR (the DeepSeek NaN stop printed `ERR ...` and then also `DONE ...`), a client that
cancels mid-answer, and one that disconnects during the prompt: each is followed by two ordinary requests, and each of those
must receive the tokens meant for it - never the previous request's DONE or tokens (found on 2 x A4000 at 64K: the needle prompt
after a NaN-stopped request came back empty at once, and its answer went to the request after it).  No GPU or model is used.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server

FAKE = r'''import queue, sys, threading, time
mode = sys.argv[sys.argv.index('--mode') + 1]
cmds, stop = queue.Queue(), threading.Event()
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            stop.set()
        else:
            cmds.put(line)
    cmds.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print('READY 4096 stop', flush=True)
n = 0
while True:
    line = cmds.get()
    if line == 'QUIT':
        break
    if not line.startswith('GEN '):
        continue
    n += 1
    stop.clear()
    ids = line.split()[-1]
    if n == 1 and mode in ('err_done', 'err_only'):
        print('T 7', flush=True)
        print("ERR the model's numbers went non-finite (NaN reached the router), so the answer would be nonsense", flush=True)
        if mode == 'err_done':
            print('DONE 1 5 1 1 cancel 0 0 0', flush=True)
    elif n == 1 and mode == 'cancel':
        print('T 7', flush=True)
        stop.wait(10)
        time.sleep(.05)
        print('DONE 1 5 1 1 cancel 0 0 0', flush=True)
    else:
        # the answer of request n: its own marker token (100 + n) twice, so a late or early line is visible
        print(f'T {100 + n}', flush=True)
        print(f'T {100 + n}', flush=True)
        print('DONE 2 5 1 1 stop 0 0 0', flush=True)
'''


class AbortSync(unittest.TestCase):
    def engine(self, mode):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / 'engine.py'
        script.write_text(FAKE, encoding='utf-8')
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                               real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
            eng = server.StrataEngine('strata', ['--mode', mode])
        self.addCleanup(eng.close)
        return eng

    def ask(self, eng, cancel_after=None):
        cancel = threading.Event()
        got = []
        try:
            for t in eng.generate([1, 2, 3], 8, {}, cancel):
                got.append(t)
                if cancel_after is not None and len(got) >= cancel_after:
                    cancel.set()
                    break
        except ValueError as e:
            return got, str(e)
        return got, None

    def two_following(self, eng, first_n):
        a, err_a = self.ask(eng)
        b, err_b = self.ask(eng)
        self.assertEqual((a, err_a), ([100 + first_n + 1] * 2, None), "the first request after the abort got someone else's answer")
        self.assertEqual((b, err_b), ([100 + first_n + 2] * 2, None), "the second request after the abort got someone else's answer")

    def test_err_then_done_does_not_shift_the_next_answers(self):
        eng = self.engine('err_done')
        got, err = self.ask(eng)
        self.assertEqual(got, [7])
        self.assertIn('non-finite', err)
        self.two_following(eng, 1)

    def test_err_alone_is_over_with_the_err(self):
        eng = self.engine('err_only')
        got, err = self.ask(eng)
        self.assertIn('non-finite', err)
        self.two_following(eng, 1)

    def test_a_cancel_mid_answer_leaves_the_engine_in_step(self):
        eng = self.engine('cancel')
        got, err = self.ask(eng, cancel_after=1)
        self.assertEqual((got, err), ([7], None))
        self.two_following(eng, 1)


if __name__ == '__main__':
    unittest.main()
