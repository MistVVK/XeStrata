"""A client that has stopped does not own a handler thread, and does not own the engine's single turn.

The handler's connection has a deadline (PEER_STALL_S, STRATA_PEER_STALL_S): the longest a socket operation may block
with nothing moving either way.  Two peers that used to hold on for good:

  * one that connects and sends nothing at all - not even a request line;
  * one that asks for a streamed answer and then never reads it, which blocks the write while the generator is
    suspended inside the turn that holds the engine.

A client that is still working is never cut off: this is a deadline for one stalled operation, not a bound on a
request, and a peer that keeps sending or reading never reaches it.

    python -m unittest serve.test_connection_deadline -v
"""
import json
import socket
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent
STALL = 1.0                                     # the deadline under test: the tests must not wait for the default
LONG = "the quick brown fox jumps over the lazy dog. " * 4000   # a script whose SSE encoding outruns the socket buffers


def wire(body) -> bytes:
    b = json.dumps(body).encode()
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b)


class PeerStall(unittest.TestCase):
    def server(self, script: str):
        """A server whose handler was built with the short deadline.  PEER_STALL_S is read where the handler class is
        made, so it has to be patched around serve() rather than around the requests."""
        with mock.patch.object(server, "PEER_STALL_S", STALL):
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, script, max_context=65536), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
            httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    def test_a_client_that_says_nothing_is_let_go(self):
        port = self.server("hello there")
        s = socket.create_connection(("127.0.0.1", port), timeout=30)
        try:
            s.settimeout(20.0)
            t0 = time.monotonic()
            try:
                first = s.recv(1)              # nothing to read: the server ends the connection instead of waiting
            except socket.timeout:
                self.fail("the connection was still held: the handler thread is parked on a client that said nothing")
            took = time.monotonic() - t0
            self.assertEqual(first, b"", "the connection is ended, nothing is sent to a client that made no request")
            self.assertGreaterEqual(took, STALL * 0.5, "the deadline, not a race")
            self.assertLess(took, 10, "it has to be let go at the deadline")
        finally:
            s.close()

    def test_a_stream_nobody_reads_does_not_hold_the_turn(self):
        port = self.server(LONG)
        slow = socket.create_connection(("127.0.0.1", port), timeout=30)
        other = None
        try:
            slow.sendall(wire({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20000, "stream": True}))
            slow.settimeout(10.0)
            self.assertTrue(slow.recv(256).startswith(b"HTTP/1.0 200"), "the stream has to open")
            # from here the client reads nothing, so the server's write blocks with the turn held
            time.sleep(0.5)
            other = socket.create_connection(("127.0.0.1", port), timeout=30)
            other.sendall(wire({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}))
            other.settimeout(20.0)
            t0 = time.monotonic()
            answer = other.recv(256)
            took = time.monotonic() - t0
            self.assertTrue(answer.startswith(b"HTTP/1.0 200"), answer)
            self.assertLess(took, 15, "the second client waited for the turn a stalled stream was holding")
        finally:
            # closing them releases the server side whatever state it is in, so a failing run cannot park a thread
            slow.close()
            if other is not None:
                other.close()

    def test_a_slow_but_reading_client_is_served(self):
        # the deadline is per socket operation, so reading a little at a time is not a stall: this one reads the whole
        # stream, slowly, and has to get all of it
        port = self.server("the quick brown fox jumps over the lazy dog. " * 100)
        s = socket.create_connection(("127.0.0.1", port), timeout=30)
        try:
            s.sendall(wire({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20000, "stream": True}))
            s.settimeout(30.0)
            seen = b""
            while b"data: [DONE]" not in seen:
                piece = s.recv(8192)
                if not piece:
                    break
                seen += piece
                time.sleep(0.02)               # a reader slower than the writer, but always reading
            self.assertIn(b"data: [DONE]", seen)
        finally:
            s.close()

    def test_the_deadline_can_be_switched_off(self):
        # 0 means the old behaviour (wait for ever), which is what #1317 chose for the engine's own reads
        with mock.patch.object(server, "PEER_STALL_S", None):
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, "hello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
            httpd = serve(svc, port=0)
        try:
            self.assertIsNone(httpd.RequestHandlerClass.timeout)
            s = socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=30)
            try:
                s.sendall(wire({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}))
                s.settimeout(20.0)
                self.assertTrue(s.recv(256).startswith(b"HTTP/1.0 200"))
            finally:
                s.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
