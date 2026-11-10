# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Starting and ending the engines that communicate over lines."""
import subprocess


def popen(what: str, args: list, **kw):
    """Explain Windows' application-control refusal instead of a bare OSError."""
    try:
        return subprocess.Popen(args, **kw)
    except OSError as e:
        if getattr(e, "winerror", None) in (4551, 1260):
            raise OSError(f"Windows blocked {what} ({args[0]}): Smart App Control, or an application-control policy, "
                          "does not let a program that is not signed run. Turn Smart App Control off (Windows "
                          "Security > App & browser control > Smart App Control settings), or - for the image encoder "
                          "only - run setup with --vision no (see docs/TROUBLESHOOTING.md)") from e
        raise


def pump_lines(proc, out, *, strip: bool = False, ready_only: bool = False) -> None:
    """Read lines to a queue; in ready-only mode leave the rest to the engine's normal reader."""
    ready = False
    try:
        for line in proc.stdout:
            out.put(line.rstrip("\r\n") if strip else line)
            if ready_only and line.startswith("READY"):
                ready = True
                return
    except (OSError, ValueError):
        pass
    finally:
        if not ready:
            out.put(None)


def _terminate(proc, grace: float, kill_wait: float | None = None) -> bool:
    # SIGTERM lets the engine drain GPU queues; SIGKILL alone can leave copies using unmapped memory.
    try:
        proc.terminate()
        proc.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            if kill_wait is not None:
                proc.wait(timeout=kill_wait)
            return True
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return True
    except OSError:
        return True


def end_gpu_process(proc, grace: float = 10.0) -> None:
    """Stop a process immediately when a protocol failure makes QUIT unsafe."""
    try:
        if proc is not None and proc.poll() is None:
            _terminate(proc, grace)
    except OSError:
        pass


def shutdown(proc, quit_wait: float, term_wait: float, kill_wait: float | None = None,
             close_stdin: bool = False) -> bool:
    """QUIT, then SIGTERM, then SIGKILL; false only when the final wait times out."""
    if proc is None or proc.poll() is not None:
        return True
    try:
        if proc.stdin is not None:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            if close_stdin:
                proc.stdin.close()
        proc.wait(timeout=quit_wait)
        return True
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        pass
    return _terminate(proc, term_wait, kill_wait)
