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

P4b (0.2.16 retest, p1004): the pause alone did not hold. The pressure that got the engine ended comes from its own 40 GB RAM placement (one thread in one
long system call: a stop signal is not acted on until it returns), while the user's cgroup still holds an earlier start's file cache that the kernel must
reclaim inside that call. So make_room() drops that cache of the model's files before the engine starts, and a stop that has not taken hold within 2 s is
taken back at once (_took_hold). serve/lastrun.py says at the next start what happened to a last run that did not end normally.

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
TRIM_FRACTION = 0.30              # from 30 % of oomd's limit (15 % of a 50 % limit) the model file's cache is given back as it loads ...
TRIM_EVERY_S = 1.0                # ... at most once a second
STOP_CONFIRM_S = 2.0              # a stop that has not taken hold after this long is taken back (see Pacer._took_hold)
STOP_CONFIRM_POLL_S = 0.1
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
        done = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=5, encoding="utf-8", errors="replace")
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


# ------------------------------------------------------------------------------------------------ make room before the load (P4b)
# Measured on p1004 (11-18, FIX2-PLAT; rel0216/fix2plat/): the pressure that gets an engine ended does not come from reading the model. It comes from the
# engine placing its 40 GB of experts in RAM in one go (anonymous memory, one thread, one long system call that SIGSTOP cannot interrupt) while the
# user's cgroup still holds the file cache of the PREVIOUS start's reads (14 GB): the kernel has to reclaim that cache page by page inside the
# allocation, and the manager's memory pressure goes to 80 % for 20-30 s (natural cache: peak 79.5 %, 18 s above 50 %; the same load with the model files'
# cache dropped first: peak 1 %).  So the server drops that cache itself, before it starts the engine: posix_fadvise(DONTNEED) on the files the engine
# is about to read - clean pages only, nothing is lost but a faster re-read.
ROOM_MIN_CACHE = 1 << 30            # the manager's file cache is at least 1 GiB ...
ROOM_FREE_FRACTION = 0.8            # ... and free memory is below 0.8 x the size of the model's files (else there is room without it)
ROOM_FREE_FRACTION_BOXED = 1.1      # in a container the engine's placement needs the model's size free (plus 10 %): MemFree is all the container can see
ROOM_MIN_FREED = 256 << 20          # say it only when at least this much was given back
ROOM_MAX_FILES = 200000


def size_words(nbytes: int) -> str:
    """An amount of memory for the person: whole GB from 1 GB up, whole MB below (0.25 to 0.99 GB was shown as "0 GB" or "1 GB")."""
    return f"{nbytes / 1e9:.0f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.0f} MB"
_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)


def _arg_files(args) -> list[Path]:
    """The files the engine is about to read: every existing absolute path among its arguments (a folder: the files under it)."""
    found: list[Path] = []
    folders: list[Path] = []
    for item in args:                                    # the named files first (the big model files), then the folders' files (a pack is ~24,000 small ones)
        text = str(item)
        if not os.path.isabs(text):
            continue
        p = Path(text)
        try:
            if p.is_file():
                found.append(p)
            elif p.is_dir():
                folders.append(p)
        except OSError:
            continue
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                found.append(Path(root) / name)
                if len(found) >= ROOM_MAX_FILES:
                    return found
    return found


def ple_table_file(args) -> Path | None:
    """The file named by --ple-gguf when the engine keeps its table in RAM or maps it (--ple-io ram / mmap / not given); None for --ple-io direct (random 4 KiB reads: it
    gains nothing from the cache) and when there is no --ple-gguf. Its cache is the one the engine needs again at its start: make_room leaves it."""
    items = [str(a) for a in args]
    io = None
    for i, a in enumerate(items):
        if a == "--ple-io" and i + 1 < len(items):
            io = items[i + 1]
    if io == "direct":
        return None
    for i, a in enumerate(items):
        if a == "--ple-gguf" and i + 1 < len(items) and os.path.isabs(items[i + 1]):
            return Path(items[i + 1])
    return None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a == b or os.path.samefile(a, b)
    except OSError:
        return False


def _drop_cache(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, _DONTNEED)
        return True
    except (OSError, AttributeError):
        return False
    finally:
        os.close(fd)


def _stat_value(cgroup_dir: Path, key: str) -> int | None:
    try:
        for line in (cgroup_dir / "memory.stat").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return None


def in_container(environ=None) -> bool:
    """A Docker or Podman container (Linux): the engine's memory guard is outside, on the host (0.2.17: rootless Podman on Ubuntu, where systemd-oomd ended the
    container during the load; the container sees neither the user manager's cgroup nor the pacer's conditions, only the host's /proc/meminfo)."""
    environ = os.environ if environ is None else environ
    return sys.platform.startswith("linux") and (os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or bool(environ.get("container")))


def _meminfo_cached() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("Cached:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _mem_free() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemFree:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def make_room(args, say=None, environ=None, *, found=None, drop=_drop_cache, cache_of=None, free_of=None, sizes=None) -> int:
    """Before the engine starts: drop the file cache the user's cgroup holds for the model's files, when this desktop's memory guard watches us and
    memory is tight (see above).  Returns the bytes given back (0: nothing done).  Never raises.  STRATA_OOMD_ROOM=0 turns it off."""
    environ = os.environ if environ is None else environ
    debug = environ.get("STRATA_OOMD_DEBUG") == "1"        # testers: one line saying why nothing was done

    def why(text: str) -> int:
        if debug:
            print(f"[strata] memory guard (debug): make_room: {text}", flush=True)
        return 0
    try:
        if environ.get("STRATA_OOMD_ROOM") == "0" or _DONTNEED is None:
            return why("turned off")
        guard = applicable(environ) if found is None else found
        boxed = guard is None and found is None and in_container(environ)      # 0.2.17: in a container the manager's cgroup cannot be seen, the host's can
        if guard is None and not boxed:
            return why("not below a user manager watched by systemd-oomd")
        if boxed:
            cache = cache_of or _meminfo_cached
            fraction = ROOM_FREE_FRACTION_BOXED
        else:
            cg = manager_cgroup()
            if cg is None and found is None:
                return why("no user manager cgroup")
            cgroup_dir = CGROUP_ROOT / (cg or "").lstrip("/")
            cache = (cache_of or (lambda: _stat_value(cgroup_dir, "file")))
            fraction = ROOM_FREE_FRACTION
        before = cache()
        if before is None or before < ROOM_MIN_CACHE:
            return why(f"file cache of the manager {before} < {ROOM_MIN_CACHE}")
        files = _arg_files(args)
        total = 0
        for p in files:
            try:
                total += (sizes or (lambda q: q.stat().st_size))(p)
            except OSError:
                continue                                   # a broken link or a file that went away: not counted
        free = (free_of or _mem_free)()
        if not files or total <= 0 or free is None or free >= fraction * total:
            return why(f"enough room: free {free}, files {len(files)}, bytes {total}")
        kept = ple_table_file(args)
        for p in files:
            if kept is not None and _same_file(p, kept):
                continue                                   # the engine reads the table through the cache at its start: dropping it costs a cold read of its whole size (GB10, 26.8 GiB: +48 s)
            drop(p)
        after = cache()
        freed = max(0, before - (after if after is not None else before))
        if freed < ROOM_MIN_FREED and free_of is None:     # the cgroup's counter lags a moment behind: the free memory shows it at once
            time.sleep(0.3)
            later = _mem_free()
            if later is not None:
                freed = max(freed, later - free)
        if debug:
            print(f"[strata] memory guard (debug): make_room: cache {before} -> {after}, {len(files)} files, free {free}", flush=True)
        if freed >= ROOM_MIN_FREED:
            (say or (lambda text: print(text, flush=True)))(
                f"[strata] memory guard: freed {size_words(freed)} of memory that still held copies of this model's files from an earlier start, "
                + ("so that loading does not run short of memory and a memory guard of this computer (such as systemd-oomd) does not end the container"
                   if boxed else "so that loading does not run short of memory and this desktop (systemd-oomd) does not end it"))
        return freed
    except Exception as exc:                             # noqa: BLE001 - a nicety, never a reason for a start to fail
        return why(f"error {type(exc).__name__}: {exc}")


def process_state(pid: int) -> str | None:
    """The state letter of a process from /proc/<pid>/stat ('R', 'S', 'D', 'T' ...), or None when it cannot be read."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        return text[text.rindex(")") + 2:].split()[0]
    except (OSError, ValueError, IndexError):
        return None


class Pacer:
    """The loop, with its clock, its pressure reader and its signal sender as arguments so that a test can drive it."""

    def __init__(self, pid: int, read, limit: float, say, *, alive, kill=os.kill, now=time.monotonic, sleep=time.sleep,
                 budget: float = PAUSE_BUDGET_S, state=None, trim=None):
        self.trim, self.last_trim, self.trimmed = trim, -TRIM_EVERY_S, False
        self.pid, self.read, self.say, self.alive = pid, read, say, alive
        self.kill, self.now, self.sleep, self.budget = kill, now, sleep, budget
        self.state = state                       # () -> the process's state letter ("T" = stopped), or None; None: a stop is taken for granted
        self.stop_at, self.resume_at = limit * STOP_FRACTION, limit * RESUME_FRACTION
        self.limit = limit
        self.pauses = 0
        self.unstoppable = 0
        self.paused_s = 0.0
        self.stopped = False

    def _maybe_trim(self, value) -> None:
        """While memory is tight, give back the model file's cache as the load goes (see trim_files): the engine reads the file once, so the pages behind
        it are of no use, and keeping them makes every new page a reclaim stall (measured: pressure 90 % with 10 GB free)."""
        if self.trim is None or value is None or value < self.limit * TRIM_FRACTION or self.now() - self.last_trim < TRIM_EVERY_S:
            return
        self.last_trim = self.now()
        try:
            self.trim()
        except Exception:                                  # noqa: BLE001
            return
        if not self.trimmed:
            self.trimmed = True
            self.say("[strata] memory guard: memory is tight, so the memory that holds copies of the model's file is given back as it loads "
                     "(this does not slow the load)")

    def run(self, loading: threading.Event) -> None:
        last_cont = self.now() - RUN_MIN_S
        try:
            while self.alive() and not loading.is_set():
                value = self.read()
                self._maybe_trim(value)
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

    def _took_hold(self) -> bool:
        """Did the stop take hold within STOP_CONFIRM_S?  A stop signal is acted on by ONE thread of the engine when it returns to user code; while that
        thread is inside one long system call (the engine placing its 40 GB of experts in RAM, 20-40 s under memory pressure: measured on p1004) the
        whole engine goes on running and the pressure keeps rising.  Then the stop is taken back at once (SIGCONT drops it), so that it cannot hit
        later, when the call returns and the pressure is already falling."""
        if self.state is None:
            return True
        end = self.now() + STOP_CONFIRM_S
        while True:
            letter = self.state()
            if letter is None or letter == "T":
                return True
            if not self.alive():
                return True
            if self.now() >= end:
                return False
            self.sleep(STOP_CONFIRM_POLL_S)

    def _pause(self, value: float, loading: threading.Event) -> None:
        start = self.now()
        try:
            self.kill(self.pid, SIGSTOP)
        except OSError:
            return
        self.stopped = True
        if not self._took_hold():
            self._cont()
            self.unstoppable += 1
            return
        self.pauses += 1
        self.say(f"[strata] memory guard: this desktop (systemd-oomd) ends a program when memory has been under strain for 20 s "
                 f"(now {value:.0f} %, its limit {self.limit:.0f} %). The engine waits a few seconds while memory settles (pause {self.pauses}).")
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


def trim_files(args) -> list[Path]:
    """The big model file the engine reads its experts from, once: the path after --native (not the PLE table, which is read at random later)."""
    items = [str(a) for a in args]
    return [Path(items[i + 1]) for i, a in enumerate(items[:-1]) if a == "--native" and os.path.isabs(items[i + 1]) and Path(items[i + 1]).is_file()]


def start(proc, loading: threading.Event, say=None, args=None) -> threading.Thread | None:
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
        targets = trim_files(args or []) if os.environ.get("STRATA_OOMD_TRIM") != "0" else []
        pacer = Pacer(proc.pid, read, limit, say, alive=lambda: proc.poll() is None, state=lambda: process_state(proc.pid),
                      trim=(lambda: [_drop_cache(p) for p in targets]) if targets else None)
        thread = threading.Thread(target=pacer.run, args=(loading,), name="strata-oomd-pacer", daemon=True)
        thread.start()
        return thread
    except Exception:                                    # noqa: BLE001 - a nicety, never a reason for a start to fail
        return None
