"""Keep systemd-oomd from ending a model load on a Linux desktop (0.2.16, FIX-PLAT P4).

Ubuntu (24.04 and others) runs systemd-oomd and sets ManagedOOMMemoryPressure=kill (limit 50 %, 20 s) on every user's manager,
`user@UID.service`. A program started from a desktop terminal lives below it. While a model of tens of GB is read into RAM the kernel
reclaims memory all the time and the manager's memory pressure stays above 50 %: after 20 s oomd ends the cgroup that reclaims the most,
the engine's, and the person sees the window and the engine vanish (measured: USERTEST-0216-LIN F1 on p1004, 0.2.15 and 0.2.16 alike).
A login session (ssh, a text console) is not below `user@UID.service`, so the same launch worked there.

No scope or unit property helps (measured on p1004, FIX-PLAT): oomd must end something while the manager is under pressure, so an engine
excluded with ManagedOOMPreference=omit is spared and the person's terminal is ended instead. The only fix without root is to keep that
pressure under the limit while the engine loads: this pacer reads the manager's memory.pressure and stops the engine (SIGSTOP) for a few
seconds when it nears the limit, then continues it (SIGCONT) when the pressure has fallen. The load takes longer, and is not killed.

Limits: Linux only; only when systemd-oomd is active, the manager's limit is a kill limit and this server runs below `user@UID.service`; only
before READY; only the engine process this server started; at most PAUSE_BUDGET_S of pausing in all, then it stops pacing and says so; one
log line per pause. Everything is read-only except the two signals; any problem reading means "not applicable", never a failed start.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PAUSE_BUDGET_S = 600.0            # the most the engine is paused in all (10 minutes), then pacing ends and the load goes on unpaced
PAUSE_MAX_S = 30.0                # one pause never lasts longer, whatever the pressure
RUN_MIN_S = 3.0                   # after a continue the engine runs at least this long before it can be stopped again
STOP_FRACTION = 0.70              # stop at 70 % of oomd's limit (35 % of a 50 % limit) ...
RESUME_FRACTION = 0.30            # ... continue at 30 % of it (15 %): the 10 s average halves about every 7 s
POLL_S = 0.5
SIGSTOP = getattr(signal, "SIGSTOP", 19)         # Linux numbers; the module is imported (and tested) on other systems too
SIGCONT = getattr(signal, "SIGCONT", 18)
OOMD_FULL_SCALE = 4294967295.0    # systemd's percentage property: 100 % = 2**32 - 1 (ManagedOOMMemoryPressureLimit=2147483648 is 50 %)
CGROUP_ROOT = Path("/sys/fs/cgroup")
_USER_MANAGER = re.compile(r"/user@(\d+)\.service(?=/|$)")


def manager_cgroup(cgroup_text: str | None = None) -> str | None:
    """The cgroup path of `user@UID.service` this process runs below ('/user.slice/user-1000.slice/user@1000.service'), or None: not below it
    (a login session, a container, no systemd, no cgroup v2)."""
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8") if cgroup_text is None else cgroup_text
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            m = _USER_MANAGER.search(line[3:])
            if m:
                return line[3:][:m.end()]
    return None


def kill_limit_percent(show_text: str) -> float | None:
    """The kill limit in percent from `systemctl show user@UID.service -p ManagedOOMMemoryPressure -p ManagedOOMMemoryPressureLimit`,
    or None when the manager is not set to kill. A limit that is 0 (unset) is oomd's default, 60 %."""
    values = dict(line.split("=", 1) for line in show_text.splitlines() if "=" in line)
    if values.get("ManagedOOMMemoryPressure", "").strip() != "kill":
        return None
    try:
        raw = float(values.get("ManagedOOMMemoryPressureLimit", "0").strip() or 0)
    except ValueError:
        return None
    return 60.0 if raw <= 0 else round(raw / OOMD_FULL_SCALE * 100.0, 1)


def _run(args: list[str]) -> str | None:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def avg10(text: str) -> float | None:
    """'some avg10=12.34 ...' (the first line of a memory.pressure file) as a number."""
    m = re.match(r"\s*some\s+avg10=([0-9.]+)", text)
    return float(m.group(1)) if m else None


def applicable(environ=None) -> tuple[Path, float] | None:
    """(the memory.pressure file of the user manager, oomd's kill limit in percent) when pacing can matter here, else None."""
    environ = os.environ if environ is None else environ
    if not sys.platform.startswith("linux") or environ.get("STRATA_OOMD_PACER") == "0":
        return None
    cg = manager_cgroup()
    if cg is None:
        return None
    pressure = CGROUP_ROOT / cg.lstrip("/") / "memory.pressure"
    try:
        if avg10(pressure.read_text(encoding="utf-8")) is None:
            return None
    except OSError:
        return None
    if _run(["systemctl", "is-active", "systemd-oomd"]) is None:
        return None
    unit = cg.rsplit("/", 1)[-1]
    shown = _run(["systemctl", "show", unit, "-p", "ManagedOOMMemoryPressure", "-p", "ManagedOOMMemoryPressureLimit"])
    limit = kill_limit_percent(shown) if shown else None
    return (pressure, limit) if limit else None


class Pacer:
    """The loop, with its clock, its pressure reader and its signal sender as arguments so that a test can drive it."""

    def __init__(self, pid: int, read, limit: float, say, *, alive, kill=os.kill, now=time.monotonic, sleep=time.sleep,
                 budget: float = PAUSE_BUDGET_S):
        self.pid, self.read, self.say, self.alive = pid, read, say, alive
        self.kill, self.now, self.sleep, self.budget = kill, now, sleep, budget
        self.stop_at, self.resume_at = limit * STOP_FRACTION, limit * RESUME_FRACTION
        self.limit = limit
        self.pauses = 0
        self.paused_s = 0.0
        self.stopped = False

    def run(self, loading: threading.Event) -> None:
        last_cont = self.now() - RUN_MIN_S
        try:
            while self.alive() and not loading.is_set():
                value = self.read()
                if value is not None and value >= self.stop_at and self.now() - last_cont >= RUN_MIN_S:
                    if self.paused_s >= self.budget:
                        self.say(f"[strata] memory guard: the engine was paused for {self.paused_s:.0f} s in all and is not paused any more; "
                                 "the load goes on (this desktop may still end it)")
                        return
                    self._pause(value, loading)
                    last_cont = self.now()
                    continue
                self.sleep(POLL_S)
        finally:
            self._cont()
            if self.pauses and loading.is_set():
                self.say(f"[strata] memory guard: the load needed {self.pauses} short pause(s), {self.paused_s:.0f} s in all, so that this desktop did not end it")

    def _pause(self, value: float, loading: threading.Event) -> None:
        self.pauses += 1
        self.say(f"[strata] memory guard: this desktop (systemd-oomd) ends a program when memory has been under strain for 20 s "
                 f"(now {value:.0f} %, its limit {self.limit:.0f} %). The engine waits a few seconds while memory settles (pause {self.pauses}).")
        start = self.now()
        try:
            self.kill(self.pid, SIGSTOP)
        except OSError:
            return
        self.stopped = True
        try:
            while self.alive() and not loading.is_set() and self.now() - start < PAUSE_MAX_S and self.paused_s + (self.now() - start) < self.budget:
                v = self.read()
                if v is not None and v <= self.resume_at:
                    break
                self.sleep(POLL_S)
        finally:
            self._cont()
            self.paused_s += self.now() - start

    def _cont(self) -> None:
        if self.stopped:
            self.stopped = False
            try:
                self.kill(self.pid, SIGCONT)
            except OSError:
                pass


def killed_words(rc, environ=None) -> str:
    """One plain sentence for a start error when the engine was ended by SIGKILL (exit status -9) on a desktop whose memory guard is systemd-oomd
    and this server runs below its user manager: the likely reason, and what to do. "" in every other case."""
    try:
        if rc != -9 or applicable(environ) is None:
            return ""
    except Exception:                                    # noqa: BLE001
        return ""
    return ("\n  This desktop's memory guard (systemd-oomd) most likely ended it: it ends a program when memory has been under strain for 20 s, "
            "and a large model loading can do that. Close other programs and start again; a text console or an ssh login is not watched by it.")


def start(proc, loading: threading.Event, say=None) -> threading.Thread | None:
    """Pace `proc` (the engine this server just started) until `loading` is set (READY) or it ends. None when pacing does not apply here.
    Never raises."""
    try:
        found = applicable()
        if found is None:
            return None
        pressure, limit = found
        say = say or (lambda text: print(text, flush=True))

        def read() -> float | None:
            try:
                return avg10(pressure.read_text(encoding="utf-8"))
            except OSError:
                return None
        pacer = Pacer(proc.pid, read, limit, say, alive=lambda: proc.poll() is None)
        thread = threading.Thread(target=pacer.run, args=(loading,), name="strata-oomd-pacer", daemon=True)
        thread.start()
        return thread
    except Exception:                                    # noqa: BLE001 - a nicety, never a reason for a start to fail
        return None
