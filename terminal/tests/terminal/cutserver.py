#!/usr/bin/env python3
"""cutserver.py PORT DIR [NAMES BYTES]: serve DIR over http on 127.0.0.1; each file called in NAMES (comma separated) is cut after BYTES bytes (the connection breaks part-way),
every other file is served whole.  For the real-package flow tests (update_model_flow.sh): a download that fails.  Standard library only."""
import functools
import socketserver
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Server(ThreadingHTTPServer):
    """http.server's server_bind also asks socket.getfqdn for the address, which stalled 35 s in every new process on GitHub's macOS runner."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class Handler(SimpleHTTPRequestHandler):
    cut_names = []
    cut_bytes = 0

    def do_GET(self):
        if self.path.rsplit("/", 1)[-1] in self.cut_names:
            path = self.translate_path(self.path)
            with open(path, "rb") as f:
                size = f.seek(0, 2)
                f.seek(0)
                data = f.read(self.cut_bytes)
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Type", "application/gzip")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            self.close_connection = True
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    port, directory = int(sys.argv[1]), sys.argv[2]
    Handler.cut_names = (sys.argv[3] if len(sys.argv) > 3 else "").split(",")
    Handler.cut_bytes = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    server = Server(("127.0.0.1", port), functools.partial(Handler, directory=directory))
    server.serve_forever()


if __name__ == "__main__":
    main()
