#!/usr/bin/env python3
"""Run a command in a real terminal (a pty) and answer its questions, for the real-package flow tests (update_model_flow.sh).

    pty_drive.py [--timeout S] [--cols N --rows N] [--quit-after S --quit-with KEYS] [--answer 'REGEX=>TEXT' ...] -- command args

Each --answer is used once, in order, the first time its REGEX shows in what the command printed after the previous answer; TEXT is typed followed by
Enter (TEXT may be empty).  The whole transcript goes to stdout (control codes kept: the screens are drawn with them); the exit code is the command's,
or 124 when the time ran out.  Standard library only.
"""
import argparse
import fcntl
import os
import pty
import re
import select
import struct
import sys
import termios
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--cols", type=int, default=140)
    ap.add_argument("--rows", type=int, default=45)
    ap.add_argument("--answer", action="append", default=[])
    ap.add_argument("--quit-after", type=float, default=0, help="seconds after which --quit-with is typed (a screen that never ends by itself)")
    ap.add_argument("--quit-with", default="q")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    answers = []
    for item in a.answer:
        pattern, _, text = item.partition("=>")
        answers.append((re.compile(pattern), text))
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = os.environ.get("TERM", "xterm-256color")
        os.execvp(cmd[0], cmd)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", a.rows, a.cols, 0, 0))
    start = time.monotonic()
    seen = ""                      # what was printed since the last answer
    quit_sent = False
    code = None
    while True:
        if time.monotonic() - start > a.timeout:
            os.kill(pid, 15)
            code = 124
            break
        if a.quit_after and not quit_sent and time.monotonic() - start > a.quit_after:
            os.write(fd, a.quit_with.encode())
            quit_sent = True
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b""
            if not data:
                break
            text = data.decode("utf-8", "replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            seen += text
            plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", seen)
            if answers and answers[0][0].search(plain):
                os.write(fd, (answers[0][1] + "\n").encode())
                answers.pop(0)
                seen = ""
    try:
        _, status = os.waitpid(pid, 0)
        if code is None:
            code = os.waitstatus_to_exitcode(status)
    except ChildProcessError:
        code = code if code is not None else 0
    sys.stdout.write(f"\n[pty_drive] exit {code}; unanswered: {[p.pattern for p, _ in answers]}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
