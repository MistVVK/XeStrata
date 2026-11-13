# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream b3c633868 (#5044): a byte that is not UTF-8 on the engine's stdout must not end the reader.

The engine's stdout is decoded as text with errors="replace", so a log line with a byte such as 0x8e (a Windows console
code page, a path in another encoding) reaches the server as U+FFFD instead of raising UnicodeDecodeError in the reader
thread - which ended the thread, and with it the engine's output: the READY wait timed out and every reply lost its
DONE.  The protocol lines are ASCII, so nothing of them changes.
"""
import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server

# a diagnostic line with 0x8e before READY, and another one inside the reply
FAKE = r'''import sys
out = sys.stdout.buffer
out.write(b'INFO note=\x8e\n')
out.flush()
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        out.write(b'INFO token_note=\x8e\n')
        out.flush()
        print('T 72', flush=True)
        print('T 105', flush=True)
        print('DONE 2 16 1.0 2.0 stop', flush=True)
'''


class NonUtf8EngineOutput(unittest.TestCase):
    def test_a_byte_that_is_not_utf8_is_read_and_the_reader_lives(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'engine.py'
            script.write_text(FAKE, encoding='utf-8')
            real_popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                                   real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = server.StrataEngine('strata', [])
            service = server.Service(engine, server.ByteTokenizer(),
                                     server.ChatTemplate(Path(__file__).parent / 'test_chat_template.jinja'))
            http = server.serve(service, port=0)
            port = http.server_address[1]
            try:
                self.assertTrue(engine.alive(), 'the READY line after the byte arrived')
                body = {'model': 'm', 'max_tokens': 8, 'reasoning_effort': 'none',
                        'messages': [{'role': 'user', 'content': 'hi'}]}
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    reply = json.load(response)
                self.assertEqual(reply['choices'][0]['message']['content'], 'Hi', reply)
                self.assertEqual(reply['usage']['completion_tokens'], 2)
                self.assertTrue(engine.alive(), 'the byte inside the reply did not end the reader either')
                self.assertEqual(service.history[-1]['engine_generated'], 2)
            finally:
                http.shutdown()
                http.server_close()
                engine.close()


if __name__ == '__main__':
    unittest.main()
