# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/image_daemon.py - the image engine outside the servers: one `xestrata-image --serve` that the servers of one
image model on one GPU share, so that it stays warm (its kernels built, the prompt's states, the layers it holds) when
a server stops - a model switch starts another server on the same port - and serves them one request at a time.

    python serve/image_daemon.py --socket ADDR [--idle S] [--log FILE] -- xestrata-image --dit ... --serve

serve/images.py starts it (its environment the server's engine's: the same GPU and libraries) and connects; another
server of the same image model and GPU finds it at the same address (images.daemon_address).  ADDR: a Unix socket's
path, or on Windows a file the daemon writes its loopback port and a random token to (a client sends the token
first).  The lines are xestrata-image's own (src/image/strata_image.cpp): on connecting a client gets READY (or ERROR,
and the daemon ends); a request line (IMG, NEED, FREE, W8, FIT, BUDGET, RELEASE, RATIO, DETECT, UPS, UPSNEED) waits for
the engine and gets its lines up to the answer (STEP lines before an IMAGE); STOP ends the client's running image after
its step; RESUME goes on with the client's running video after its YIELD (the engine waits there for it); QUIT closes
the connection; SHUTDOWN ends the engine (after a running request) and the daemon, and answers
STOPPED once the engine is gone.  Without a client and a request for --idle seconds (default 1800)
the daemon ends, and so it does when the engine ends.  A request the engine says nothing for in --silence seconds
(default SILENCE_S; 0: wait for ever) has a stalled engine: the daemon stops listening, ends the engine, answers
"ERROR stalled: ..." and ends, so the next request starts a new daemon and engine.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import line_process  # noqa: E402
from serve import idle_watch  # noqa: E402
from serve import disk_watch  # noqa: E402

ANSWERS = {"IMG": ("IMAGE",), "NEED": ("NEED",), "FREE": ("FREE",), "W8": ("W8",), "FIT": ("BUDGET",),
           "BUDGET": ("BUDGET",), "RELEASE": ("RELEASE",), "RATIO": ("RATIO",), "DETECT": ("DETECT",),
           "UPS": ("UPSCALED",), "UPSNEED": ("NEED",), "SEGMENT": ("SEGMENT",), "FLORENCE": ("FLORENCE",),
           "POLYMASK": ("POLYMASK",), "MATTE": ("MATTE",), "AUXNEED": ("NEED",), "UPSV": ("UPSVIDEO",),
           "UPSVNEED": ("NEED",), "SEGV": ("SEGVIDEO",), "CONTROL": ("CONTROL",),
           "COLORMAP": ("COLORMAP",)}
START_TIMEOUT_S = 600
# A running request's engine silent this long is stalled, not slow: a step, a tile, a video's frame and a load's share
# each print a line, and the slowest seen take minutes (a layer read from a slow SSD, HunyuanImage 3.0's step).  The
# stall it is for (2026-11-24, RX 9060 XT on ROCm): a layer's copy that the driver kept waiting for 50 minutes while the
# engine's queues were evicted, its thread spinning on the CPU; the job's own limit (timeout_s, 3600 s) then left the
# engine as it was and every later job failed on it.
SILENCE_S = 900


class Daemon:
    def __init__(self, cmd: list[str], idle: float, log, silence: float = SILENCE_S):
        self.cmd, self.idle, self.log, self.silence = cmd, idle, log, silence
        self.srv: socket.socket | None = None
        self.addr: str | None = None
        self.ino: int | None = None              # the address file's, so that the daemon removes its own alone
        self.resuming = False                   # a YIELD waits for the client's RESUME: not the engine's silence
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
        self.proc = line_process.popen(
            "the XeStrata image engine", self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, errors="replace", bufsize=1)

        threading.Thread(target=line_process.pump_lines, args=(self.proc, self.lines),
                         kwargs={"strip": True}, daemon=True).start()
        t0 = time.monotonic()
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, START_TIMEOUT_S - (time.monotonic() - t0)))
            except queue.Empty:
                raise RuntimeError(f"the image engine did not start in {START_TIMEOUT_S} s") from None
            if line is None:
                raise RuntimeError(f"the image engine ended (exit code {self.proc.poll()})")
            if line.startswith("READY"):
                # the engine's pid: a server enters the engine's VRAM in the GPU's ledger under it (vram_ledger.py), so
                # the entry lasts while the engine holds the memory, whichever server asked for it
                self.ready = f"{line} pid={self.proc.pid}"
                self.say(f"image engine ready in {time.monotonic() - t0:.1f} s: {line}")
                return

    def send(self, line: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def request(self, conn, f, line: str, disconnected: threading.Event) -> None:
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
                heard = time.monotonic()
                while True:
                    if disconnected.is_set() and not gone:
                        gone = True
                        if op in ("IMG", "UPSV", "SEGV"):
                            with contextlib.suppress(OSError, ValueError):
                                self.send("STOP")
                    try:
                        got = self.lines.get(timeout=0.25)
                    except queue.Empty:
                        silent = time.monotonic() - heard
                        if self.resuming:
                            heard = time.monotonic()
                        elif 0 < self.silence < silent:
                            self.stalled(f, op, silent)
                            return
                        continue
                    heard = time.monotonic()
                    self.resuming = got == "YIELD"
                    if got is None:
                        write(f, "ERROR the image engine ended")
                        self.done.set()
                        return
                    if not gone:
                        try:
                            write(f, got)
                        except OSError:
                            gone = True
                            if op in ("IMG", "UPSV", "SEGV"):
                                with contextlib.suppress(OSError, ValueError):
                                    self.send("STOP")
                    if gone and got == "YIELD" and op not in ("IMG", "UPSV", "SEGV"):
                        # Requests without STOP still need to leave the yield.
                        self.resuming = False
                        with contextlib.suppress(OSError, ValueError):
                            self.send("RESUME")
                    if got.split(" ", 1)[0] in ends or got.startswith("ERROR"):
                        return
            finally:
                self.owner = None
                self.resuming = False
                self.last = time.monotonic()

    def stalled(self, f, op: str, silent: float) -> None:
        """The engine said nothing for --silence seconds in a request: no server finds this daemon again, the engine is
        ended, the client told, and the daemon ends (the next request starts a new one)."""
        why = f"the image engine said nothing for {silent:.0f} s during {op} (the config's silence_s)"
        self.say(f"{why}: ending it")
        self.retire()
        line_process.end_gpu_process(self.proc)
        with contextlib.suppress(OSError, ValueError):
            write(f, f"ERROR stalled: {why}; it was ended")
        self.done.set()

    def retire(self) -> None:
        """Stop listening and remove the address, when it is still this daemon's (a new daemon may have taken it)."""
        srv, self.srv = self.srv, None
        if srv is not None:
            with contextlib.suppress(OSError):
                srv.close()
        with contextlib.suppress(OSError):
            if self.addr is not None and os.stat(self.addr).st_ino == self.ino:
                os.unlink(self.addr)

    def client(self, conn) -> None:
        with self.count_lock:
            self.clients += 1
        try:
            f = conn.makefile("rw", encoding="utf-8", errors="replace", newline="\n")
            write(f, self.ready)
            requests: queue.Queue = queue.Queue()
            disconnected = threading.Event()

            def read():   # the client's lines: STOP and RESUME at once (for the running request), the rest in turn
                try:
                    for raw in f:
                        line = raw.rstrip("\r\n")
                        if line in ("STOP", "RESUME"):
                            if self.owner is conn:
                                if line == "RESUME":
                                    self.resuming = False
                                with contextlib.suppress(OSError, ValueError):
                                    self.send(line)
                            continue
                        requests.put(line)
                finally:
                    disconnected.set()
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
                    self.request(conn, f, line, disconnected)
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
        line_process.shutdown(proc, 10, 10)


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
    ap.add_argument("--silence", type=float, default=SILENCE_S)
    ap.add_argument("--log")
    ap.add_argument("--disk-paths", default="[]", help="JSON [[label, path], ...] from the image engine's config")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    log = open(a.log, "a", encoding="utf-8", errors="replace") if a.log else sys.stderr
    d = Daemon(cmd, a.idle, log, a.silence)
    try:
        given = json.loads(a.disk_paths)
        disk_paths = [(str(label), Path(p)) for label, p in given if isinstance(p, str)]
    except (ValueError, TypeError):
        disk_paths = []                         # a bad argument must not stop the engine
    disk_watch.start(disk_paths, say=lambda line: d.say(f"{disk_watch.severity(line)}: {line}"))
    try:
        d.start()
    except (OSError, RuntimeError) as e:
        d.say(f"the image engine did not start: {e}")
        d.stop()
        return 1
    srv, token = listen(a.socket)
    srv.settimeout(1.0)
    d.srv, d.addr, d.ino = srv, a.socket, os.stat(a.socket).st_ino
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

    def check_idle():
        if d.proc is None or d.proc.poll() is not None:
            d.say("the image engine ended")
            d.done.set()
        elif d.clients == 0 and d.owner is None and time.monotonic() - d.last > d.idle:
            d.say(f"no client for {d.idle:g} s: ending")
            d.done.set()
    check_idle()
    if not d.done.is_set():
        idle_watch.watch.add(1.0, check_idle, repeat=True)
        d.done.wait()
    d.retire()
    d.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
