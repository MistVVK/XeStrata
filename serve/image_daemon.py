# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/image_daemon.py - the image engine outside the servers: one `strata-image --serve` that the servers of one
image model on one GPU share, so that it stays warm (its kernels built, the prompt's states, the layers it holds) when
a server stops - a model switch starts another server on the same port - and serves them one request at a time.

    python serve/image_daemon.py --socket ADDR [--idle S] [--log FILE] -- strata-image --dit ... --serve

serve/images.py starts it (its environment the server's engine's: the same GPU and libraries) and connects; another
server of the same image model and GPU finds it at the same address (images.daemon_address).  ADDR: a Unix socket's
path, or on Windows a file the daemon writes its loopback port and a random token to (a client sends the token
first).  The lines are strata-image's own (src/image/strata_image.cpp): on connecting a client gets READY (or ERROR,
and the daemon ends); a request line (IMG, NEED, FREE, FIT, BUDGET, RELEASE, RATIO, DETECT) waits for the engine and
gets its lines up to the answer (STEP lines before an IMAGE); STOP ends the client's running image after its step;
QUIT closes the connection; SHUTDOWN ends the engine (after a running request) and the daemon, and answers STOPPED once
the engine is gone.  Without a client and a request for --idle seconds (default 1800)
the daemon ends, and so it does when the engine ends.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time

ANSWERS = {"IMG": ("IMAGE",), "NEED": ("NEED",), "FREE": ("FREE",), "FIT": ("BUDGET",), "BUDGET": ("BUDGET",),
           "RELEASE": ("RELEASE",), "RATIO": ("RATIO",), "DETECT": ("DETECT",)}
START_TIMEOUT_S = 600


class Daemon:
    def __init__(self, cmd: list[str], idle: float, log):
        self.cmd, self.idle, self.log = cmd, idle, log
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.ready = ""
        self.lock = threading.Lock()            # one request at a time on the engine
        self.owner = None                       # the connection whose request runs
        self.clients = 0
        self.count_lock = threading.Lock()
        self.last = time.monotonic()
        self.done = threading.Event()

    def say(self, text: str) -> None:
        print(f"[image daemon] {text}", file=self.log, flush=True)

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, errors="replace", bufsize=1)

        def pump():
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\r\n"))
            self.lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        t0 = time.monotonic()
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, START_TIMEOUT_S - (time.monotonic() - t0)))
            except queue.Empty:
                raise RuntimeError(f"the image engine did not start in {START_TIMEOUT_S} s") from None
            if line is None:
                raise RuntimeError(f"the image engine ended (exit code {self.proc.poll()})")
            if line.startswith("READY"):
                self.ready = line
                self.say(f"image engine ready in {time.monotonic() - t0:.1f} s: {line}")
                return

    def send(self, line: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def request(self, conn, f, line: str) -> None:
        """One request line: the engine's lines to the client up to the answer.  A client gone meanwhile has its image
        stopped; the lines are still read to the answer, so the next request gets its own."""
        op = line.split(" ", 1)[0]
        ends = ANSWERS.get(op)
        if ends is None:
            write(f, f"ERROR unknown request {op}")
            return
        with self.lock:
            self.owner = conn
            gone = False
            try:
                self.send(line)
                while True:
                    got = self.lines.get()
                    if got is None:
                        write(f, "ERROR the image engine ended")
                        self.done.set()
                        return
                    if not gone:
                        try:
                            write(f, got)
                        except OSError:
                            gone = True
                            if op == "IMG":
                                with contextlib.suppress(OSError, ValueError):
                                    self.send("STOP")
                    if got.split(" ", 1)[0] in ends or got.startswith("ERROR"):
                        return
            finally:
                self.owner = None
                self.last = time.monotonic()

    def client(self, conn) -> None:
        with self.count_lock:
            self.clients += 1
        try:
            f = conn.makefile("rw", encoding="utf-8", errors="replace", newline="\n")
            write(f, self.ready)
            requests: queue.Queue = queue.Queue()

            def read():   # the client's lines: STOP at once (it targets the running image), the rest in turn
                try:
                    for raw in f:
                        line = raw.rstrip("\r\n")
                        if line == "STOP":
                            if self.owner is conn:
                                with contextlib.suppress(OSError, ValueError):
                                    self.send("STOP")
                            continue
                        requests.put(line)
                finally:
                    requests.put(None)
            threading.Thread(target=read, daemon=True).start()
            while True:
                line = requests.get()
                if line is None or line == "QUIT":
                    return
                if line == "SHUTDOWN":   # the engine ended (after a running request) before the answer, so that
                    self.done.set()      # the asker can take the GPU at once
                    with self.lock:
                        self.stop()
                    with contextlib.suppress(OSError, ValueError):
                        write(f, "STOPPED")
                    return
                if line:
                    self.request(conn, f, line)
        except OSError:
            pass
        finally:
            with self.count_lock:
                self.clients -= 1
            self.last = time.monotonic()
            with contextlib.suppress(OSError):
                conn.close()

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin is not None:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def write(f, line: str) -> None:
    f.write(line + "\n")
    f.flush()


def listen(addr: str):
    """The listening socket at `addr` -> (socket, token or None)."""
    if os.name != "nt":
        with contextlib.suppress(FileNotFoundError):
            os.unlink(addr)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)                   # the socket for this user only
        try:
            s.bind(addr)
        finally:
            os.umask(old)
        s.listen(8)
        return s, None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    token = secrets.token_hex(16)
    tmp = addr + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(f"{s.getsockname()[1]} {token}\n")
    os.replace(tmp, addr)
    return s, token


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--idle", type=float, default=1800)
    ap.add_argument("--log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    log = open(a.log, "a", encoding="utf-8", errors="replace") if a.log else sys.stderr
    d = Daemon(cmd, a.idle, log)
    try:
        d.start()
    except (OSError, RuntimeError) as e:
        d.say(f"the image engine did not start: {e}")
        d.stop()
        return 1
    srv, token = listen(a.socket)
    srv.settimeout(1.0)
    d.say(f"listening at {a.socket}")

    def accept():
        while not d.done.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if token is not None:   # loopback TCP: the client proves it read the address file
                conn.settimeout(10)
                try:
                    if conn.makefile("r", encoding="utf-8").readline().strip() != token:
                        conn.close()
                        continue
                except OSError:
                    conn.close()
                    continue
                conn.settimeout(None)
            threading.Thread(target=d.client, args=(conn,), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()
    while not d.done.is_set():
        if d.proc is None or d.proc.poll() is not None:
            d.say("the image engine ended")
            break
        if d.clients == 0 and d.owner is None and time.monotonic() - d.last > d.idle:
            d.say(f"no client for {d.idle:g} s: ending")
            break
        d.done.wait(1.0)
    with contextlib.suppress(OSError):
        srv.close()
    if os.name != "nt" or token is not None:
        with contextlib.suppress(OSError):
            os.unlink(a.socket)
    d.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
