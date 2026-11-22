"""The component updater: the one explicit step that downloads and installs engines (and Control, and the app installer).

Everything that fetches software lives behind this module. Starting or serving never imports it (`acquisition.require`
refuses here when ``STRATA_NO_ACQUIRE`` is set), so a PC that only runs models never reaches the network.

What an engine update does, in order, and what keeps the old engine safe at each point:

1. read the signed feed (`feed.py`); a bad signature, an expired feed or an older sequence number ends here;
2. detect the hardware and pick the best package that fits it (`feed.select_engine`); nothing newer, nothing to do;
3. refuse to go on while an engine is serving (wait, or ask);
4. download the archive next to the component folder with resume and progress; the SHA-256 in the signed feed decides;
5. unpack it into a *staging* component folder (``.updates/staging/<id>/engine/<name>``), never into the live ``engine/``;
6. check the package: the manifest is the one the feed named (SHA-256), says what the feed says, carries a valid
   signature, every file matches its recorded hash, and it fits this PC;
7. **test-start** it from the staging folder through the real start path (owner runtime, a saved model, one token);
8. only if that passed, move it into ``engine/`` (a rename), re-resolve to confirm it is the one a start will pick, record the
   previous version for a manual rollback, and mark saved tuning presets for the short re-check.

If any step fails, the live ``engine/`` folder has not changed: the previous engine keeps working, and what failed is
named. Every older engine stays installed until the person deletes it (Settings > Engine, ``strata components remove``);
``strata components use`` / ``rollback`` switch between them.
Models and presets are never read, moved, deleted or downloaded here.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from strata_control import acquisition, components, engine_component as ec, feed as feedmod, folder_use, longpath, preset_marks, signing, transfer
from strata_control.targets import host_platform
from strata_control import atomic as _atomic_mod

STATE_DIR = ".updates"
STAGE_DIR = ".s"                 # staging is short on purpose: <components>\.s\<4 hex>\engine\<folder> (see longpath.py)
SKEW_FREE_BYTES = 1.15


def megabytes(n) -> int:
    """Decimal megabytes (1 MB = 1,000,000 bytes), rounded half up, at least 1: the one figure for an engine's size in the Terminal, the app
    (app/host/src/lib/format.ts megabytes) and the release page. Both sides are tested on the same numbers."""
    return max(1, int((n or 0) / 1_000_000 + 0.5))


class UpdateError(RuntimeError):
    """An update step stopped. `exit_code` is the CLI's; the old engine is untouched unless the message says otherwise."""

    def __init__(self, message: str, exit_code: int = 6):
        super().__init__(message)
        self.exit_code = exit_code


LAUNCH_NEXT_STEP = "Stop the model first (`strata launch --stop`), then run `strata update` again. Nothing was changed."


class Busy(UpdateError):
    """An engine is serving, so nothing was installed or switched."""

    def __init__(self, serving: list[dict]):
        if any(str(item.get("profile", "")).startswith("launch-") for item in serving):
            super().__init__("Strata's own launch is using the graphics card. " + LAUNCH_NEXT_STEP, 7)
            self.serving = serving
            return
        names = ", ".join(item.get("profile", "an engine") for item in serving) or "an engine"
        super().__init__(f"{names} is serving. Stop it (or wait until it is idle) and run the update again; "
                         "the update never replaces or tests an engine while one is serving.", 7)
        self.serving = serving


class TestStartFailed(UpdateError):
    __test__ = False                       # not a pytest class

    #: reasons that say the graphics card or memory was busy, not that the engine is broken
    BUSY = re.compile(r"timed?[ -]?out|out of memory|\boom\b|not enough (gpu |video |free )?memory|vram|memory allocation|"
                      r"cudaerrormemory|device (is )?(busy|in use)|resource busy|no free (gpu|memory)", re.I)

    @staticmethod
    def plain(label: str, reason: str, busy: bool, hint: str | None) -> str:
        """The one text of a failed test start (X01): what did not work in plain words, that nothing was changed, and one next step. The engine's
        own words stay in ``reason`` (the state file, the JSON output and the kept log), never in this sentence."""
        what = "Control" if label.startswith("Control") else "engine"
        head = (f"The new {label} did not start correctly on this computer, so it was not installed. Your current {what} is unchanged."
                if what == "Control" else
                f"The new engine {label} did not start correctly on this computer, so it was not installed. Your current engine is unchanged.")
        low = reason.lower()
        library = missing_library(reason)
        if what == "Control":
            cause = f"It answered `--version` with: {reason.split('--version said', 1)[1].strip()}" if "--version said" in low else \
                "It did not answer `--version` in time or at all."
        elif library:
            cause = f"The engine needs the system library {library}, and this computer does not have it."
        elif "one-token probe" in low:
            cause = "The engine started but stopped before it answered a test question."
        elif "--version exited" in low or "could not run" in low:
            cause = "The engine program did not run on this computer."
        elif "no installed model" in low:
            cause = "There is no installed model to test the engine with."
        elif low.startswith("it "):                      # Strata's own plain sentences ("it was still loading the model after 10 minutes")
            cause = reason[:1].upper() + reason[1:].rstrip(".") + "."
        elif busy:
            cause = "The graphics card or memory was in use."
        else:
            cause = "The engine reported a problem while it started."
        if hint is None:
            hint = ("Run `strata update --force` to try again; if it fails again, report it with the log." if not busy and what == "engine"
                    else "Run `strata update` again.")
        return f"{head} {cause} {hint}"

    def __init__(self, label: str, reason: str, *, busy: bool | None = None, hint: str | None = None):
        # busy: True/False says outright whether the card or memory was busy (the wait-expired reasons carry the engine's log tail, which
        # mentions VRAM on every healthy start and must not be read as a busy card, #3683); None reads it from the words of the reason
        self.busy = bool(self.BUSY.search(reason)) if busy is None else busy
        if hint is None and self.busy:
            hint = ("The graphics card or memory looks busy: close programs that use the card (games, video, other models) "
                    "and run `strata update` again.")
        super().__init__(self.plain(label, reason, self.busy, hint), 6)
        self.label = label
        self.reason = reason


#: The loader's own words when a program needs a shared library the system does not have (glibc's ld.so).
_MISSING_LIBRARY = re.compile(r"error while loading shared libraries:\s*([^\s:]+):\s*cannot open shared object file")

#: The package that holds a system library, per Linux family (debian = Debian, Ubuntu, Mint ...; fedora = Fedora, RHEL, Rocky ...; arch = Arch,
#: Manjaro ...). Only libraries an engine package can need from the system and that a family names differently are listed. Arch: gcc-libs pulls in
#: libatomic / libgomp / libstdc++ / libgcc both before and after Arch split them into their own packages.
LIBRARY_PACKAGES = {
    "libatomic.so.1": {"debian": "libatomic1", "fedora": "libatomic", "arch": "gcc-libs"},
    "libgomp.so.1": {"debian": "libgomp1", "fedora": "libgomp", "arch": "gcc-libs"},
    "libstdc++.so.6": {"debian": "libstdc++6", "fedora": "libstdc++", "arch": "gcc-libs"},
    "libgcc_s.so.1": {"debian": "libgcc-s1", "fedora": "libgcc", "arch": "gcc-libs"},
    "libnuma.so.1": {"debian": "libnuma1", "fedora": "numactl-libs", "arch": "numactl"},
    "libz.so.1": {"debian": "zlib1g", "fedora": "zlib", "arch": "zlib"},
}
_INSTALL = {"debian": "sudo apt install {}", "fedora": "sudo dnf install {}", "arch": "sudo pacman -S {}"}


def missing_library(text: str | None) -> str | None:
    """The shared library the loader could not find (``libatomic.so.1``), from a program's output; None when it says no such thing."""
    found = _MISSING_LIBRARY.search(text or "")
    return found.group(1) if found else None


def linux_family(os_release: str = "/etc/os-release") -> str | None:
    """debian, fedora or arch for this Linux (from /etc/os-release ID and ID_LIKE), or None when it is none of them or cannot be read."""
    try:
        text = Path(os_release).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    words = []
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("ID", "ID_LIKE"):
            words += value.strip().strip("\"'").lower().split()
    for family, names in (("debian", ("debian", "ubuntu")), ("fedora", ("fedora", "rhel", "centos")), ("arch", ("arch", "archlinux"))):
        if any(w in names for w in words):
            return family
    return None


def library_hint(library: str, family: str | None = None) -> str:
    """The one next step for a missing system library: the command that installs its package on this Linux, or the package names when the
    family is not known, then run the update again."""
    packages = LIBRARY_PACKAGES.get(library)
    if packages and family in packages:
        return f"Install it with `{_INSTALL[family].format(packages[family])}`, then run `strata update` again."
    if packages:
        return (f"Install it with your system's package manager ({packages['debian']} on Debian and Ubuntu, {packages['fedora']} on Fedora, "
                f"{packages['arch']} on Arch), then run `strata update` again.")
    return "Install the package that has it with your system's package manager, then run `strata update` again."


# ---------------------------------------------------------------------------------------------- state

def updates_dir(root: Path) -> Path:
    return root / STATE_DIR


@contextlib.contextmanager
def update_lock(root: Path):
    """One updater at a time per component folder."""
    directory = updates_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / "update.lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise UpdateError("another update is already running for this component folder", 7) from None
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise UpdateError("another update is already running for this component folder", 7) from None
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()


class State:
    """``<components>/.updates/state.json``: the last accepted feed sequence, which engine is the previous one, a pin set by a
    manual rollback, versions not to offer again. Plain JSON, replaced atomically; a damaged file reads as empty."""

    def __init__(self, root: Path):
        self.path = updates_dir(root) / "state.json"
        self.data = self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return {}

    def section(self, name: str) -> dict:
        value = self.data.get(name)
        if not isinstance(value, dict):
            value = self.data[name] = {}
        return value

    def history(self, entry: dict) -> None:
        log = self.data.setdefault("history", [])
        log.append(entry)
        del log[:-30]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["schema_version"] = 1
        descriptor, temp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=1, sort_keys=True)
                stream.write("\n")
            _atomic_mod.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


# ---------------------------------------------------------------------------------------------- hardware

def _gfx_names() -> list[str]:
    found: list[str] = []
    kfd = Path("/sys/class/kfd/kfd/topology/nodes")
    if kfd.is_dir():
        for node in sorted(kfd.iterdir()):
            try:
                text = (node / "properties").read_text()
            except OSError:
                continue
            match = re.search(r"^gfx_target_version\s+(\d+)", text, re.MULTILINE)
            if match and int(match.group(1)) > 0:
                value = int(match.group(1))
                found.append(f"gfx{value // 10000}{(value // 100) % 100:x}{value % 100:x}")
    if not found:
        tool = shutil.which("rocminfo") if os.name != "nt" else shutil.which("hipInfo")
        if tool:
            try:
                done = subprocess.run([tool], capture_output=True, text=True, errors="replace", timeout=10,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                found = sorted(set(re.findall(r"\bgfx[0-9a-f]{3,4}\b", done.stdout)))
            except (OSError, subprocess.SubprocessError):
                pass
    if not found and os.name != "nt":
        try:
            found = ec.hip_archs_from_pci()                  # no kfd topology and no rocminfo: the PCI ids of the AMD cards say (an AppImage, a container)
        except Exception:                                   # noqa: BLE001 - detection is best effort
            found = []
    if not found and os.name == "nt":
        # a normal Windows PC has no hipInfo (it ships with the HIP SDK): the display adapters' PCI ids / names say which gfx it is
        try:
            found = ec.hip_archs_present()
        except Exception:                                   # noqa: BLE001 - detection is best effort
            found = []
    return sorted(set(found))


def _amd_card_names() -> list[str]:
    if os.name != "nt":
        return []
    try:
        return [c["name"] for c in ec.windows_amd_cards() if c.get("name")]
    except Exception:                                       # noqa: BLE001
        return []


def detect_hardware(environ=None) -> dict:
    """What the package choice depends on. Read-only probes of tools that are already installed (nvidia-smi, rocminfo, sysfs);
    ``STRATA_ENGINE_BACKEND`` overrides the backend like it does for a start. Anything unreadable is left empty, which the
    selection treats as "cannot tell", never as a refusal."""
    environ = os.environ if environ is None else environ
    prefer = ""
    family, prefer = ec.split_backend(environ.get(ec.ENV_BACKEND, ""))
    if prefer:
        # "hip-rx9070" is how the feed labels a package; accept it as "the hip family, this variant"
        environ = {**environ, ec.ENV_BACKEND: family}
    backend, how = ec.detect_backend(environ)
    hw = {"platform": host_platform(), "backend": backend, "backend_source": how, "runtime": ec.runtime_version(),
          "nvidia_driver": None, "cuda_archs": [], "gfx": [], "gpu_names": [], "variant_wanted": prefer,
          "makes": sorted(ec.gpu_makes_present())}
    if backend == "cuda":
        hw["nvidia_driver"] = ec.nvidia_driver_version()
        hw["cuda_archs"] = ec.cuda_archs_present()
    elif backend == "hip":
        hw["gfx"] = _gfx_names()
        hw["gpu_names"] = _amd_card_names()
    return hw


# ---------------------------------------------------------------------------------------------- serving

def _recovered_stale(runtime, entry: dict) -> bool:
    """True when `entry` was a stale record (its owner host is gone) and it is cleared now."""
    if entry.get("state") != "stale":
        return False
    try:
        result = runtime.recover_lost(entry["profile"])
    except Exception:                                  # a record that cannot be read or cleared stays: the update says a model is running
        return False
    return bool(result) and result.get("ok") is True and result.get("state") == "stopped"


def serving_now() -> list[dict]:
    """Engines that are serving: managed instances of the owner runtime and anything holding the shared activity lease
    (`strata serve`, a runtime host). Empty when nothing is. The exclusive lease is only tried, never kept."""
    from strata_control import profiles, runtime
    try:
        managed = runtime.active_managed_instances()
    except (profiles.ProfileError, OSError):
        managed = []
    # #3977 / #6239: a record whose owner host is gone ("stale": the app was closed or crashed with a model loaded) is not an engine that serves.
    # It is recovered the way `strata runtime stop` does (only what Strata recorded, each process re-checked); only a record that cannot be
    # cleared stays in the answer. A record whose host is alive again is left as it is.
    managed = [entry for entry in managed if not _recovered_stale(runtime, entry)]
    if managed:
        return managed
    try:
        descriptor = runtime.acquire_activity_lease(shared=False)
    except profiles.ProfileError as exc:
        if "lease" in str(exc):
            return [{"profile": "a running Strata server", "state": "serving"}]
        return []
    except OSError:
        return []
    try:
        os.close(descriptor)
    except OSError:
        pass
    return []


def wait_until_idle(check: Callable[[], list[dict]], *, wait: float, emit: Callable, cancel=None,
                    poll: float = 3.0) -> None:
    """Return when nothing is serving; raise `Busy` if something still is after `wait` seconds (0 = do not wait)."""
    deadline = time.monotonic() + wait
    serving = check()
    while serving:
        if wait <= 0 or time.monotonic() >= deadline or (cancel is not None and cancel.is_set()):
            raise Busy(serving)
        emit({"event": "waiting", "message": "An engine is serving; waiting for it to stop before updating.",
              "serving": serving})
        time.sleep(poll)
        serving = check()


def idle_gate(ctx: "Context") -> None:
    """Called after a download, right before the first thing is staged, test-started or switched: with a gate attached, the models the person agreed to
    stop are stopped here (and a model nobody agreed to stop is refused); then, as always, nothing may still be serving."""
    gate = getattr(ctx, "gate", None)
    if gate is not None:
        gate.release(ctx.serving)
    wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)


# ---------------------------------------------------------------------------------------------- context

@dataclass
class Context:
    components: Path
    source: str
    trusted: dict
    hardware: dict
    emit: Callable[[dict], None] = lambda event: None
    cancel: threading.Event | None = None
    test_start: Callable | None = None           # (stage_root, component, ctx) -> dict; defaults to `default_test_start`
    serving: Callable[[], list[dict]] = serving_now
    wait: float = 0.0                             # seconds to wait for a serving engine to stop
    force: bool = False
    channel: str = "stable"
    app_version: str | None = None
    now: dt.datetime | None = None
    run_config: str | None = None                 # a run config for the test start (default: an installed model's)
    test_timeout: float = 600.0
    poll: float = 3.0
    environ: dict = field(default_factory=lambda: dict(os.environ))
    allow_no_model_smoke: bool = True
    state_base: Path | None = None                # where tuning/recheck.json goes (default: the per-user state folder)
    feed_timeout: float = 30.0                    # seconds a read of the feed may wait (the quiet daily check uses 5)
    automatic: bool = False                       # a read of the feed nobody asked for by name (netpolicy rule 4b: the refusal note)
    gate: object | None = None                    # strata_control.engine_update.ModelGate: what to do when a model is loaded (None: refuse before anything is downloaded)
    memory_check: Callable | None = None          # (run config) -> (free GB, needed GB) when memory is short for a test load, else None (#5400; tests)

    def say(self, phase: str, message: str, **extra) -> None:
        self.emit({"event": "phase", "phase": phase, "message": message, **extra})


def require_acquisition(what: str) -> None:
    """The update path acquires by definition; a process that has forbidden acquiring (a start, a serve) may not run it."""
    acquisition.require(what, "`strata update` from a terminal")


def downgrade_words(sequence, accepted, state_file: Path) -> str:
    """A feed older than the one already accepted (A01b/A05), in plain words: both numbers, the folder that remembers the newer one, and one next step.
    The sentence keeps "older than the one already accepted" so a log search for the old text still finds it."""
    return (f"this update source is older than the one already accepted on this computer (its sequence is {sequence}, the accepted one is "
            f"{accepted}; refusing a replayed feed). The newer one is remembered in {state_file.parent}. Use the newest release folder; "
            f"if you really want this older one, delete {state_file} and run this again.")


def open_feed(ctx: Context, state: State) -> feedmod.Feed:
    require_acquisition("the update feed")
    last = (state.data.get("feed") or {}).get("sequence")
    ctx.say("feed", "Reading the signed update feed")
    try:
        loaded = feedmod.fetch(ctx.source, ctx.trusted, timeout=ctx.feed_timeout, last_sequence=last, channel=ctx.channel, now=ctx.now,
                               automatic=ctx.automatic)
    except feedmod.FeedDowngrade as exc:
        raise feedmod.FeedDowngrade(downgrade_words(exc.sequence, exc.accepted, state.path)) from None
    for line in loaded.skipped:                       # an entry from a newer release that this Control cannot use: said once, never an error
        ctx.say("feed", line)
    try:                                              # the feed's model list, kept (signature and all) for a PC that is offline later; nothing when the feed has none
        from strata_control import feed_models
        feed_models.remember(loaded, state.path.parent.parent)
    except Exception:                                 # noqa: BLE001 - never part of an update
        pass
    info = state.section("feed")
    info.update({"sequence": loaded.sequence, "key_id": loaded.key_id, "source": ctx.source, "channel": loaded.channel,
                 "checked_at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()})
    return loaded


# ---------------------------------------------------------------------------------------------- archives

def _archive_size(archive: Path, fmt: str) -> int:
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            return sum(info.file_size for info in zf.infolist())
    mode = {"tar.gz": "r:gz", "tar.xz": "r:xz"}[fmt]
    with tarfile.open(archive, mode) as tar:
        return sum(m.size for m in tar.getmembers() if m.isfile())


def _extract_with_links(archive: Path, destination: Path, fmt: str) -> None:
    """Like `components.safe_extract`, but a relative symbolic link that stays inside `destination` is allowed. A Control bundle's
    Python needs them (libpython, terminfo). Every name is checked before anything is written; hard links, devices, absolute
    links and links that leave the folder are refused."""
    mode = {"tar.gz": "r:gz", "tar.xz": "r:xz"}.get(fmt)
    if mode is None:
        raise components.ComponentError(f"links are only unpacked from tar archives, not {fmt}")
    root = destination.resolve()
    with tarfile.open(archive, mode) as tar:
        members = tar.getmembers()
        for m in members:
            components._check_member_within(destination, m.name)
            if m.isdir() or m.isfile():
                continue
            if not m.issym():
                raise components.UnsafeArchive(f"the archive contains a member that is not a file or a symbolic link ({m.name!r})")
            link = m.linkname.replace("\\", "/")
            if not link or link.startswith("/") or re.match(r"^[A-Za-z]:", link):
                raise components.UnsafeArchive(f"the archive has an absolute symbolic link ({m.name!r})")
            resolved = (destination / m.name).parent.joinpath(link).resolve()
            if resolved != root and root not in resolved.parents:
                raise components.UnsafeArchive(f"the symbolic link {m.name!r} leaves the package")
        destination.mkdir(parents=True, exist_ok=True)
        links = []
        for m in members:
            target = components._check_member_within(destination, m.name)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(m)
                with target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if m.mode & 0o111:
                    target.chmod(0o755)
            else:
                links.append((target, m.linkname))
        for target, linkname in links:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(linkname, target)


def extract(archive: Path, destination: Path, fmt: str, installed_bytes: int, *, allow_links: bool = False) -> None:
    """Unpack into a fresh folder. The archive's own total size must not exceed what the signed feed declared, so a package
    cannot grow into a disk-filling bomb; names and links are checked by `components.safe_extract` before anything is written
    (an engine package may not contain links; a Control bundle may contain relative links that stay inside it)."""
    try:
        total = _archive_size(archive, fmt)
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
        raise UpdateError(f"the downloaded archive cannot be read ({type(exc).__name__})", 5) from None
    if total > installed_bytes:
        raise UpdateError(f"the archive holds {total} bytes, the signed feed says the package is {installed_bytes}; not unpacked", 5)
    destination = longpath.long_path(destination)        # extended-length form: a long user name must not stop the unpacking
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    try:
        if allow_links:
            _extract_with_links(archive, destination, fmt)
        else:
            components.safe_extract(archive, destination, fmt=fmt)
    except components.ComponentError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise UpdateError(f"the archive was refused: {exc}", 5) from None
    except OSError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise UpdateError(f"the package could not be unpacked: {longpath.describe_oserror(exc)}", 6) from None


def _free_space_check(root: Path, need: int, ctx: "Context | None" = None, sha256: str = "", component: str = "") -> None:
    """Stop only when the drive cannot hold the bytes that are still to be written (the package not yet downloaded plus its unpacked
    size). The 1.15 margin is a guess at working room, not a need: below it the update warns and goes on. A resumed download's
    partial file is already on the drive, so it is not counted a second time."""
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    have = 0
    if ctx is not None and sha256:
        try:
            have = (updates_dir(ctx.components) / "partial" / f"{sha256[:16]}.part").stat().st_size
        except OSError:
            have = 0
    real = max(need - have, 0)
    drive = root.anchor or str(root)
    if free < real:
        raise UpdateError(f"not enough free space on {drive}: {free // (1 << 20)} MiB free, {real // (1 << 20)} MiB needed. "
                          f"Free up some space on {drive} and run the update again", 6)
    if free < real * SKEW_FREE_BYTES and ctx is not None:
        ctx.say("warn", f"Little free space on {drive}: {free // (1 << 20)} MiB free for about {real // (1 << 20)} MiB. "
                        "The update goes on and should fit; free up space if it stops.", component=component or "update")


def _download(ctx: Context, state_name: str, url_ref: str, size: int, sha256: str, label: str, feed: feedmod.Feed) -> Path:
    work = updates_dir(ctx.components) / "partial"
    partial = work / f"{sha256[:16]}.part"
    start = time.monotonic()
    last = {"at": 0.0, "pct": -1}

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        pct = int(done * 100 / total) if total else 0
        if now - last["at"] >= 0.25 or pct != last["pct"] and pct in (0, 100):
            last["at"], last["pct"] = now, pct
            rate = done / max(now - start, 1e-6)
            ctx.emit({"event": "progress", "phase": "download", "component": state_name, "label": label,
                      "done": done, "total": total, "percent": pct, "bytes_per_second": int(rate)})

    ctx.say("download", f"Downloading {label} ({megabytes(size)} MB)", component=state_name)
    try:
        what = label if label.lower().startswith("the ") else "the " + label          # "the engine 0.1.59 (cuda)", in the network rule's sentences
        return transfer.download(feed.url(url_ref), partial, size=size, sha256=sha256, progress=progress, cancel=ctx.cancel, what=what,
                                 say=lambda line: ctx.say("download", line, component=state_name))
    except transfer.Cancelled:
        raise UpdateError("the update was stopped; the partial download is kept and the next update resumes it", 130) from None
    except transfer.ChecksumMismatch as exc:
        raise UpdateError(str(exc), 5) from None
    except transfer.TransferError as exc:
        raise UpdateError(str(exc), 6) from None


# ---------------------------------------------------------------------------------------------- engine

def folder_name(pkg: feedmod.EnginePackage) -> str:
    return f"{pkg.version}-{pkg.platform}-{pkg.backend}" + (f"-{pkg.variant}" if pkg.variant else "")


def _manifest_sha(folder: Path) -> str:
    return transfer.sha256_file(folder / ec.MANIFEST)


def installed_engine(root: Path, hardware: dict) -> tuple[ec.EngineComponent | None, str | None]:
    """(the engine a start would use, why not). A package that fails its integrity check is reported, not hidden."""
    try:
        return ec.resolve(root, backend=hardware.get("backend"), full=True, runtime=hardware.get("runtime")), None
    except ec.EngineMissing as exc:
        return None, str(exc)


def _engine_view(component: ec.EngineComponent | None) -> dict | None:
    if component is None:
        return None
    return {"version": component.version, "platform": component.platform, "backend": component.backend,
            "variant": component.variant, "build_id": component.build_id, "folder": component.root.name,
            "signed": bool(component.signature)}


def decide_engine(feed: feedmod.Feed, ctx: Context, state: State) -> dict:
    """What an update would do for the engine, without downloading anything."""
    return _choose_engine(feed, ctx, state)[0]


GPU_MAKES = ("cuda", "hip", "sycl")


def installed_gpu_makes(root: Path, platform: str | None) -> set[str]:
    """The GPU makes (cuda / hip / sycl) of the engines installed for this platform, whatever their state."""
    try:
        found, _ = ec.scan(root)
    except OSError:
        return set()
    return {item.backend for item in found if item.platform == platform and item.backend in GPU_MAKES}


def make_guard(ctx: Context) -> str | None:
    """Never cross GPU makes (#3688). An update replaces the engine; it does not convert the PC from one make of card to another,
    and a package for the wrong make can only fail its test start. When the engine already installed is for a different make
    than the one detected:
      - if this PC really has a card of the installed make too (the display adapters / PCI vendors say so), the update stays
        on the installed make (a PC with an NVIDIA card and an AMD card keeps the engine it was set up with);
      - otherwise nothing is chosen and the returned sentence says why.
    An explicit ask (``--backend``, ``STRATA_ENGINE_BACKEND``) is the person's decision and passes. Returns None when the update
    may go on."""
    hw = ctx.hardware
    want = hw.get("backend")
    if want not in GPU_MAKES or hw.get("backend_source") in ("environment", "requested", "installed engine", "saved choice"):
        return None
    have = installed_gpu_makes(ctx.components, hw.get("platform"))
    if not have or want in have:
        return None
    here = set(hw.get("makes") or [])
    keep = [make for make in GPU_MAKES if make in have and make in here]
    if keep:
        ctx.hardware = {**detect_hardware({**ctx.environ, ec.ENV_BACKEND: keep[0]}), "backend_source": "installed engine"}
        return None
    mine = " and ".join(ec.MAKE_NAMES[m] for m in GPU_MAKES if m in have)
    return (f"The engine installed here is for {mine} cards, but this update would give your PC {ec.MAKE_NAMES[want]} one. "
            f"Nothing was changed. If you really changed your graphics card, download the {ec.MAKE_NAMES[want]} version "
            f"from {ec.RELEASES_URL}.")


def _choose_engine(feed: feedmod.Feed, ctx: Context, state: State) -> tuple[dict, feedmod.EnginePackage | None]:
    refusal = make_guard(ctx)
    if refusal:
        installed, _ = installed_engine(ctx.components, {**ctx.hardware, "backend": next(iter(sorted(installed_gpu_makes(ctx.components, ctx.hardware.get("platform")))), None)})
        return {"installed": _engine_view(installed), "offered": None, "reasons": [refusal], "status": "wrong-make", "note": refusal,
                "download_bytes": 0, "previous": state.section("engine").get("previous"), "pinned": state.section("engine").get("pinned"),
                "problem": None}, None
    installed, missing_reason = installed_engine(ctx.components, ctx.hardware)
    sticky = {**ctx.hardware, "installed_variant": installed.variant if installed is not None and installed.backend in ("hip", "cuda") else ""}
    offered, reasons = feedmod.select_engine([p for p in feed.engine], sticky)
    section = state.section("engine")
    report = {"installed": _engine_view(installed), "offered": offered.to_dict() if offered else None,
              "reasons": reasons, "status": "none-for-this-pc", "download_bytes": 0,
              "previous": section.get("previous"), "pinned": section.get("pinned"),
              "problem": missing_reason if installed is None and missing_reason and "holds no engine" not in missing_reason else None}
    if offered is None:
        if not feed.engine:
            report["status"] = "not-in-feed"
        else:
            hip = [p for p in feed.engine if p.backend == "hip" and p.platform == ctx.hardware.get("platform")]
            if hip and ctx.hardware.get("backend") == "hip" and ctx.hardware.get("gfx") and not any(
                    feedmod.engine_fit_reasons(p, ctx.hardware) == [] for p in hip):
                # a build for another AMD card family: one plain line instead of the list of reasons
                report["note"] = ec.arch_notice(tuple(sorted({a for p in hip for a in p.archs if a.startswith("gfx")})), ctx.hardware["gfx"])
        return report, None
    report["download_bytes"] = offered.size
    if installed is None:
        report["status"] = "install"
        return report, offered
    have, want = feedmod.version_key(installed.version), feedmod.version_key(offered.version)
    if want < have:
        report["status"] = "current"
        report["note"] = f"the feed offers {offered.version}, older than the installed {installed.version}; an update never downgrades"
    elif want == have:
        report["status"] = "current"
    else:
        report["status"] = "available"
    # The offered engine is already installed (the person switched to an older one for a test): the update only makes it the engine in use again,
    # with nothing to download. A version the person switched away from is never hidden from the update (owner, 11-01).
    local = installed_copy(ctx.components, offered)
    if report["status"] == "available" and local:
        report["installed_locally"] = local
        report["download_bytes"] = 0
        if offered.version in (section.get("failed_unproven") or {}) and not ctx.force:
            # #5400: this engine was used without a test load (memory was short) and could not start the model; switching to it again by itself
            # would repeat that. The person can ask for it by name.
            report["status"] = "failed-before"
            report["note"] = (f"{offered.version} could not start your model after the last update, so the previous engine is in use; "
                              "run `strata update --force` to use it again")
        return report, offered
    bad = (section.get("failed") or {}).get(offered.version)
    if bad and _failed_mark_is_old(bad, ctx):
        bad = None                                           # a failure is held against a version for a day, not for ever
    if bad and _failed_by_an_older_control(bad):
        bad = None                                           # the Control that tried it could not start any engine here (DIAG-WINHOST-0216): try again with this one
    if report["status"] == "available" and not ctx.force:
        if bad and bad.get("manifest_sha256") == offered.manifest_sha256:
            report["status"] = "failed-before"
            report["note"] = (f"{offered.version} failed its test start on {bad.get('at', 'an earlier try')} ({bad.get('reason')}); "
                              "run `strata update --force` to try it again")
    return report, offered


#: The words of a test start that failed because the Control's own runtime host died, not because of the engine (Windows 0.2.15 and older on a PC where
#: WMI refuses the account: an access violation in CPython's `_wmi`, DIAG-WINHOST-0216; the host's answer is "runtime control is unavailable").
HOST_DEAD_WORDS = "runtime control is unavailable"


def _failed_by_an_older_control(mark: dict) -> bool:
    """A failure record made by a DIFFERENT Control (older: records of 0.2.15 and before carry no "control" field) whose reason is the dead host
    says nothing about the engine, because that Control's host could not start ANY engine on this PC. The first `strata update` from such a
    Control installs the new Control after it has already failed the engine's test start (its order is engine, then Control, and it cannot be
    changed); the next plain `strata update` therefore tries the engine again instead of asking for `--force`."""
    if HOST_DEAD_WORDS not in str(mark.get("reason", "")):
        return False
    mine = ec.runtime_version()
    return mark.get("control") != mine if mine else mark.get("control") is None


def installed_copy(root: Path, pkg: feedmod.EnginePackage) -> str | None:
    """The installed package folder that is byte for byte the feed's ``pkg`` (same name, same manifest hash), or None."""
    folder = ec.engine_dir(root) / folder_name(pkg)
    try:
        if (folder / ec.MANIFEST).is_file() and _manifest_sha(folder) == pkg.manifest_sha256:
            return folder.name
    except OSError:
        pass
    return None


def _serving_or_empty(ctx: Context) -> list[dict]:
    try:
        return ctx.serving()
    except Exception:                                   # noqa: BLE001 - informational only
        return []


def check(ctx: Context) -> dict:
    """Read the signed feed and say what an update would do for the engine, Control and the app. Downloads nothing but the feed."""
    with update_lock(ctx.components):
        state = State(ctx.components)
        feed = open_feed(ctx, state)
        report = {"feed": {"source": ctx.source, "sequence": feed.sequence, "channel": feed.channel, "key_id": feed.key_id,
                           "generated_at": feed.generated_at, "trusted_by": "test key file" if signing.extra_keys_in_effect(ctx.environ) else "built-in keys"},
                  "engine": decide_engine(feed, ctx, state), "control": decide_control(feed, ctx, state),
                  "app": decide_app(feed, ctx), "serving": _serving_or_empty(ctx), "hardware": ctx.hardware}
        state.data["last_check"] = _last_check(ctx, report)
        state.save()
        return report


def _last_check(ctx: Context, report: dict, control_version: str | None = None) -> dict:
    """What the feed offers that is newer than what is installed, kept for the notice a start or the app shows without touching the network
    (engine_component.cached_offers).  ``control_version``: the Control an update has just installed (the notice belongs to it, not to the one running)."""
    return {
        "at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(),
        "engine": report["engine"]["offered"]["version"] if report["engine"]["status"] in ("available", "install") and report["engine"]["offered"] else None,
        "app": report["app"]["offered"] if report["app"]["status"] == "available" else None,
        # the versions this result is for, so a later Control, engine or app change drops the cached notice
        "control": control_version or ec.runtime_version(), "engine_installed": ec.newest_installed_version(ctx.components)}


def new_stage(root: Path, prefix: str = "") -> Path:
    r"""A fresh, short staging folder: ``<components>\.s\<4 hex>``. Old leftovers (a crashed update; the update lock is held, so
    nothing else uses them) are swept first, along with the longer folder older versions used."""
    base = root / STAGE_DIR
    for old in [base, updates_dir(root) / "staging"]:
        if old.exists():
            shutil.rmtree(longpath.long_path(old), ignore_errors=True)
    while True:
        stage = base / (prefix + uuid.uuid4().hex[:4])
        if not stage.exists():
            return stage


def stage_engine(ctx: Context, feed: feedmod.Feed, pkg: feedmod.EnginePackage, archive: Path, *, component_id: str = ec.COMPONENT_ID) -> tuple[Path, ec.EngineComponent]:
    """Unpack and check a downloaded engine package in a staging component folder. Returns (staging root, component). `component_id`: the engine of a model shape
    (strata_control/shape_engine.py) is read and checked as its own component, never as the Strata engine."""
    stage_root = new_stage(ctx.components)
    folder = stage_root / "engine" / folder_name(pkg)
    ctx.say("verify", f"Unpacking {pkg.label}", component="engine")
    longpath.long_path(folder.parent).mkdir(parents=True, exist_ok=True)
    try:
        extract(archive, folder, pkg.format, pkg.installed_bytes)
        manifest = folder / ec.MANIFEST
        if not manifest.is_file():
            raise UpdateError(f"the package has no {ec.MANIFEST}", 5)
        if transfer.sha256_file(manifest) != pkg.manifest_sha256:
            raise UpdateError("the package's manifest is not the one the signed feed names; not installed", 5)
        try:
            component = ec.read_manifest(folder, component_id)
        except ec.ComponentProblem as exc:
            raise UpdateError(f"the package manifest is invalid: {exc}", 5) from None
        mismatches = [name for name, a, b in (
            ("version", component.version, pkg.version), ("platform", component.platform, pkg.platform), ("shape", component.shape, pkg.shape),
            ("backend", component.backend, pkg.backend), ("variant", component.variant, pkg.variant),
            ("build_id", component.build_id, pkg.build_id), ("archs", set(component.archs), set(pkg.archs)),
            ("min_driver", component.min_driver, pkg.min_driver), ("min_runtime", component.min_runtime, pkg.min_runtime))
            if a != b]
        if mismatches:
            raise UpdateError(f"the package says something different from the signed feed ({', '.join(mismatches)}); not installed", 5)
        ctx.say("verify", "Checking the package signature and every file", component="engine")
        outer = ec._verifier
        signing.install_component_verifier(ctx.trusted)
        try:
            ec.verify_signature(component, {**ctx.environ, ec.ENV_REQUIRE_SIGNED: "1"})
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            raise UpdateError(f"the package failed its check: {exc}", 5) from None
        finally:
            ec.register_signature_verifier(outer)
        why = feedmod.engine_fit_reasons(pkg, ctx.hardware)
        if why:
            raise UpdateError("; ".join(why), 5)
        _warn_if_deep(ctx, ctx.components / "engine" / folder_name(pkg), max((len(rel) for rel, _, _ in component.files), default=0) + 1)
        return stage_root, component
    except BaseException:
        shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)
        raise


_SHAPE_ONLY_FLAGS = frozenset({"--engram", "--dspark", "--dfly-override", "--vram-experts-gb", "--ram-experts-gb", "--ssd-experts"})


def _config_component_id(source) -> str:
    """Which engine component (`strata-engine`, `strata-hy4`, `strata-inkling`, ...) a saved run config targets (#6589, #6623)."""
    cfg = source
    if not isinstance(cfg, dict):
        try:
            cfg = json.loads(Path(source).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            return ec.COMPONENT_ID
    if not isinstance(cfg, dict):
        return ec.COMPONENT_ID
    exe = str(cfg.get("exe") or "").strip()
    if ec.is_reference(exe):
        comp = exe[len(ec.REFERENCE_PREFIX):].strip()
        if comp:
            return comp
    exe_base = exe.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if exe_base.startswith("strata-hy4"):
        return "strata-hy4"
    if exe_base.startswith("strata-inkling"):
        return "strata-inkling"
    arch = str(cfg.get("arch") or "").strip().lower()
    if arch in ("hy4flash", "hy4"):
        return "strata-hy4"
    if arch in ("inkling2", "inkling2_loop"):
        return "strata-inkling"
    args = cfg.get("args")
    if isinstance(args, list):
        arg_list = [str(a) for a in args]
        arg_set = set(arg_list)
        if "--dspark" in arg_set:
            return "strata-inkling"
        if arg_set & _SHAPE_ONLY_FLAGS:
            return "strata-hy4"
        for i, a in enumerate(arg_list):
            if a == "--native" and i + 1 < len(arg_list) and arg_list[i + 1].lower().endswith(".strata"):
                return "strata-inkling" if "inkling" in arg_list[i + 1].lower() else "strata-hy4"
    return ec.COMPONENT_ID


def _is_shape_engine_config(source) -> bool:
    """True when a saved run config targets a shape engine (Hy4-Flash, Inkling-2-Mini, or another `component:strata-<shape>`):
    updating the general `strata-engine` must never pass Hy4 (`--engram`) or Inkling (`--dspark`) flags to `strata` (#6589, #6623)."""
    return _config_component_id(source) != ec.COMPONENT_ID


def discover_run_config(environ=None, *, component_id: str = ec.COMPONENT_ID) -> Path | None:
    """A saved model's run config for the test start: the one a saved profile uses if its files are all present, otherwise any
    installed model whose files are present and match `component_id`. DeepSeek models and configs belonging to another engine
    component (e.g. Hy4-Flash `--engram` or Inkling `--dspark` when testing `strata-engine`) are left out (#6589, #6623).
    None when no matching model can be used (falls back to binary smoke)."""
    from strata_control import models, profiles, retired
    try:
        listed = models.list_installed()
    except Exception:                                   # noqa: BLE001 - discovery is best effort
        return None
    want = component_id or ec.COMPONENT_ID
    ready = [m for m in listed.get("models", []) if m.get("component_status") == "present" and m.get("source_config")
             and not retired.config_file_reason(m["source_config"])
             and _config_component_id(m["source_config"]) == want]
    if not ready:
        return None
    try:
        used = {profile.get("source_config") for profile in profiles.list_profiles() if profile.get("source_config")}
    except Exception:                                   # noqa: BLE001
        used = set()
    ready.sort(key=lambda m: (m["source_config"] not in used, m["id"]))
    return Path(ready[0]["source_config"])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def _patched_environ(values: dict):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _keep_text(ctx: "Context | None", component: ec.EngineComponent, text: str) -> Path | None:
    """What the engine printed at its test start, kept as ``.updates/logs/test-start-<version>.log`` (the file `_keep_log` writes for a full test
    start), so a failed update can still be read after the downloaded package is gone. None when there is no context or it cannot be written."""
    if ctx is None:
        return None
    try:
        target = updates_dir(ctx.components) / "logs"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"test-start-{component.version}.log"
        path.write_bytes(text.encode("utf-8", errors="replace")[-65536:])
        logs = sorted(target.glob("test-start-*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[:-5]:
            old.unlink(missing_ok=True)
        return path
    except OSError:
        return None


def hip_test_cards(component: ec.EngineComponent, env: dict, timeout: float = 60) -> tuple[str | None, list[dict]]:
    """AMD (#6015): the cards the test start of a HIP package lets the engine see, as HIP_VISIBLE_DEVICES, and the cards the package's own
    `strata-device --list-devices` found. A card the package has no code for is left out, and so is an integrated chip while a card of its own
    can run (the start's rule: serve/server.py pick_hip_devices): a Ryzen's integrated Radeon is HIP's device 0 on many PCs, and with it visible
    the engine's `--version` failed on a PC whose RX 9070 the package is built for, the same test on every retry. (None, cards) when nothing needs
    hiding, the environment already names the cards, or the probe is missing or fails: then the test runs as before."""
    if not str(getattr(component, "backend", "") or "").startswith("hip") or env.get("HIP_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES"):
        return None, []
    probe = component.root / ("strata-device.exe" if os.name == "nt" else "strata-device")
    if not probe.is_file():
        return None, []
    try:
        done = subprocess.run([str(probe), "--list-devices"], capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
                              cwd=str(component.root), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None, []
    if done.returncode != 0:
        return None, []
    from serve.server import parse_hip_devices
    devices = parse_hip_devices(done.stdout)
    runnable = [d for d in devices if not d["cannot_run"]]
    pool = [d for d in runnable if not d["integrated"]] or runnable
    if not pool or len(pool) == len(devices):
        return None, devices
    return ",".join(str(d["index"]) for d in pool), devices


def _cards_seen(devices: list[dict], visible: str | None) -> str:
    """The cards a test start ran with, for its failure reason: "GPU 1 (AMD Radeon RX 9070, gfx1201)"."""
    shown = [d for d in devices if visible is None or str(d["index"]) in visible.split(",")]
    return ", ".join(f"GPU {d['index']} ({d['name'] or 'graphics card'}, {d['arch'] or 'unknown'})" for d in shown)


def _same_failure_again(ctx: "Context | None", component: ec.EngineComponent, reason: str) -> bool:
    """Whether the last failed test start of this engine version failed with the same reason (#6015: "Continue" ran the same failing test
    again and again). Records this failure for the next call. False without a context or when nothing can be read or written."""
    if ctx is None:
        return False
    path = updates_dir(ctx.components) / "logs" / "test-start-last-failure.json"
    now = {"version": component.version, "reason": reason[:600]}
    try:
        before = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        before = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now), encoding="utf-8")
    except OSError:
        pass
    return isinstance(before, dict) and before.get("version") == now["version"] and before.get("reason") == now["reason"]


def binary_smoke(component: ec.EngineComponent, timeout: float = 60, ctx: "Context | None" = None) -> dict:
    """No model installed yet: the engine must at least load (its libraries resolve, the GPU runtime initialises the loader) and
    report its version. Weaker than a one-token answer, and reported as such. With ``ctx`` a failure keeps the engine's output in
    ``.updates/logs`` and says where; a missing system library is named, with the package that has it. AMD: the engine sees only the cards its
    package has code for (hip_test_cards), and a failure names the cards it ran with."""
    var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    env = dict(os.environ)
    env[var] = os.pathsep.join(ec.package_lib_dirs(component.root)) + os.pathsep + env.get(var, "")
    visible, devices = hip_test_cards(component, env)
    if visible is not None:
        env["HIP_VISIBLE_DEVICES"] = visible
    command = [str(component.exe), "--version"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
                              cwd=str(component.root), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        _keep_text(ctx, component, f"$ {' '.join(command)}\n{type(exc).__name__}: {exc}\n")
        raise TestStartFailed(component.version, f"the engine could not run ({type(exc).__name__}: {exc})") from None
    if done.returncode != 0:
        text = (done.stderr or "") + "\n" + (done.stdout or "")
        options = re.findall(r"^\s+--[a-z0-9][a-z0-9-]*", text, re.MULTILINE)
        if done.returncode == 2 and len(options) >= 5:
            # an engine built before it answered --version prints its usage (the list of its own options) and exits 2: it loaded, its libraries
            # resolved, the GPU runtime started and its argument parser ran. That is what this test is for; say so, do not fail it.
            return {"mode": "binary-only", "detail": "no model is installed yet; this engine does not answer --version, but it loaded and listed its options "
                                                     f"({len(options)} of them), so it runs", "output": ""}
        kept = _keep_text(ctx, component, f"$ {' '.join(command)}\nexit code {done.returncode}\n--- stderr\n{done.stderr or ''}\n--- stdout\n{done.stdout or ''}\n")
        library = missing_library(text)
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        reason = f"--version exited {done.returncode}: {tail[0][:200]}"
        if library and library not in reason:
            reason += f" (error while loading shared libraries: {library}: cannot open shared object file)"
        if devices:
            reason += f"; cards the engine saw: {_cards_seen(devices, visible)}"
        where = f" The engine's log from this test is in {kept}." if kept else ""
        if library:
            raise TestStartFailed(component.version, reason, busy=False, hint=library_hint(library, linux_family()) + where)
        if _same_failure_again(ctx, component, reason):
            # (#6015) the second identical failure says that trying again does not help, instead of asking for the same retry (coordinator-approved 11-16)
            hint = ("This is the same problem as last time, so trying again won't fix it. Please report it on GitHub"
                    + (f" and attach this log: {kept}." if kept else "."))
            raise TestStartFailed(component.version, reason, busy=False, hint=hint)
        hint = f"Run `strata update --force` to try again; if it fails again, report it with the log: {kept}" if kept else None
        raise TestStartFailed(component.version, reason, hint=hint)
    return {"mode": "binary-only", "detail": "no model is installed yet, so the engine was run with --version only",
            "output": done.stdout.strip()[:200]}


def _log_tail(sandbox: Path, component: ec.EngineComponent) -> str:
    log = sandbox / "engine.log"
    try:
        lines = log.read_text(errors="replace").splitlines()[-3:]
    except OSError:
        return ""
    # Keep the end: startup diagnostics and long model paths can otherwise push the actual error out of the retained reason.
    return " | " + " ".join(lines)[-1000:] if lines else ""


def _keep_log(sandbox: Path, component: ec.EngineComponent, ctx: Context) -> None:
    """The engine's log from the test start (the last 64 KiB), kept in ``.updates/logs`` so a failed update can be read afterwards."""
    log = sandbox / "engine.log"
    try:
        data = log.read_bytes()[-65536:]
        target = updates_dir(ctx.components) / "logs"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"test-start-{component.version}.log").write_bytes(data)
        logs = sorted(target.glob("test-start-*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[:-5]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


_RECORD_LIMIT = 65536                     # bytes read from any one kept file
_RECORDS_KEPT = 3                         # failed test starts whose records stay in .updates/logs


def _scrub(value, key: str = ""):
    """A kept JSON record without anything that could authorise a request: keys that name a token or a key keep only their first
    characters (enough to see that two records agree), the API key is dropped."""
    if isinstance(value, dict):
        return {k: _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, key) for v in value]
    low = key.lower()
    if isinstance(value, str) and value and ("api_key" in low or low in ("key", "authorization")):
        return "<removed>"
    if isinstance(value, str) and ("token" in low or "secret" in low or low.endswith("_key")):
        return value[:8] + "...(rest removed)"
    return value


def _keep_records(sandbox: Path, component: ec.EngineComponent, ctx: Context) -> None:
    """A failed test start keeps what the sandbox knew, because the sandbox is deleted right after (#3654): the run config the test used
    (without keys), the host, child and ready records of the runtime (their PIDs show a mismatch), and the host and server logs, next to the
    engine log in ``.updates/logs/test-start-<version>-failed``. The last few failed starts are kept. Best effort: it never hides the failure."""
    try:
        target = updates_dir(ctx.components) / "logs" / f"test-start-{component.version}-failed"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        wanted = [sandbox / "run.json", sandbox / "engine.log"]
        wanted += sorted((sandbox / "runtime").glob("*.json")) + sorted((sandbox / "runtime").glob("*.log"))
        for source in wanted:
            try:
                if not source.is_file():
                    continue
                data = source.read_bytes()[-_RECORD_LIMIT:]
                if source.suffix == ".json":
                    try:
                        data = (json.dumps(_scrub(json.loads(data.decode("utf-8-sig"))), indent=1) + "\n").encode("utf-8")
                    except ValueError:
                        pass                                                   # not JSON after all: kept as it is
                (target / source.name).write_bytes(data)
            except OSError:
                continue
        folders = sorted((d for d in target.parent.glob("test-start-*-failed") if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for old in folders[:-_RECORDS_KEPT]:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass


def _warn_if_deep(ctx: Context, folder: Path, longest_inside: int) -> None:
    """Tell the person, in plain words, when the engine will sit too deep for Windows' ordinary path limit and long paths are
    off. The install continues: Strata's own file work uses extended paths; only the engine's own loading may not."""
    if getattr(ctx, "_deep_warned", False):
        return
    words = longpath.warning_for(folder, longest_inside)
    if words and not longpath.plain_long_paths_work(ctx.components):
        ctx._deep_warned = True
        ctx.say("warn", words, component="engine")


def _test_progress(ctx: Context, component: ec.EngineComponent):
    """What the person sees while the staged engine is test-started: a line when the phase changes and one about every 30 seconds,
    so a long load or a stuck probe is never a silent wait (#3683: 10 minutes of nothing, then a timeout)."""
    began = time.monotonic()
    seen = {"phase": None, "at": 0.0}
    words = {"loading": "is loading the model",
             "verifying_generation": "has loaded the model; waiting for it to answer one token",
             "ready_file_not_matched": "says it is ready; checking that it is the engine Strata started"}

    def progress(status: dict) -> None:
        phase = status.get("phase") if status.get("state") == "starting" else None
        if phase not in words:
            return
        now = time.monotonic()
        if phase != seen["phase"] or now - seen["at"] >= 30:
            seen["phase"], seen["at"] = phase, now
            ctx.say("test", f"Engine {component.version} {words[phase]} ({int(now - began)} s)", component="engine")

    return progress


def _test_start_failure(component: ec.EngineComponent, started: dict, ctx: Context, tail: str) -> TestStartFailed:
    """The plain reason a test start did not pass, in the words of what the engine was doing when it ran out of time."""
    error = started.get("error")
    minutes = max(1, int(min(ctx.test_timeout, 600) // 60))
    log = updates_dir(ctx.components) / "logs" / f"test-start-{component.version}.log"
    if error == "startup_wait_expired":
        phase = started.get("phase")
        kept = f" The engine's log from this test is in {log}."
        if phase == "verifying_generation":
            return TestStartFailed(component.version, f"it loaded the model but did not answer one token within {minutes} minutes",
                                   busy=False, hint="Nothing was changed. Run `strata update` again; if it happens again, report it with the log." + kept)
        if phase == "ready_file_not_matched":
            return TestStartFailed(component.version, f"it reported that it was ready, but Strata could not match it to the engine process it "
                                   f"started", busy=False,
                                   hint="This looks like a problem in Strata, not in your engine or card. Nothing was changed; report it with the log." + kept)
        return TestStartFailed(component.version, f"it was still loading the model after {minutes} minutes", busy=True,
                               hint="Nothing was changed. A slow disk or a card that other programs are using can do this: close them and run "
                                    "`strata update` again." + kept)
    reason = f"it did not answer a one-token probe ({error or started.get('state')}){tail}"
    try:
        library = missing_library(tail) or missing_library(log.read_text(errors="replace"))
    except OSError:
        library = missing_library(tail)
    if library:                                          # the engine program never ran: the system lacks a library it needs
        if not missing_library(reason):
            reason += f" | error while loading shared libraries: {library}: cannot open shared object file"
        return TestStartFailed(component.version, reason, busy=False, hint=library_hint(library, linux_family()) + f" The engine's log from this test is in {log}.")
    # #4692: "report it with the log" named no log. The engine's words stay out of the sentence, but the person is told where the kept log is.
    hint = (None if TestStartFailed.BUSY.search(reason) else
            f"Run `strata update --force` to try again; if it fails again, report it with the log: {log}")
    return TestStartFailed(component.version, reason, hint=hint)


TEST_CONTEXT = 4096
_TEST_DROP_WITH_VALUE = ("--kv-resident", "--vision-tokens")
#: (#5400) "--ple-io ram" locks the n-gram (PLE) table in RAM (28.8 GB for IQ3_S, measured on l620): the test reads its rows from the file
#: ("direct"), which is all a one-token test needs. The engine also picks "ram" by itself when the RAM is free (its residency policy): the test
#: start's environment turns that off (TEST_ENV).
_TEST_PLE_IO = "direct"
TEST_ENV = {"STRATA_PLE_AUTO_RAM": "0"}
_TEST_DROP_FLAGS = ("--vision",)


def light_test_args(args: list) -> list:
    """The saved model's engine options made light for the test start (#3842): a short context, no KV kept in VRAM and no picture reader. The
    test asks whether the NEW engine loads and answers one token, not whether this card holds the model's 262,144-token settings next to the
    desktop and the running model; a 16 GB card refused good engines with 'out of memory' at the first weight upload."""
    out, i = [], 0
    values = [str(x) for x in args]
    while i < len(values):
        item = values[i]
        nxt = values[i + 1] if i + 1 < len(values) else None
        if item == "--max-context" and nxt is not None:
            try:
                out += [item, str(min(int(nxt), TEST_CONTEXT))]
            except ValueError:
                out += [item, nxt]
            i += 2
        elif item in _TEST_DROP_WITH_VALUE and nxt is not None:
            i += 2
        elif item == "--ple-io" and nxt == "ram":
            out += [item, _TEST_PLE_IO]
            i += 2
        elif item in _TEST_DROP_FLAGS:
            i += 2 if nxt is not None and not nxt.startswith("--") else 1
        else:
            out.append(item)
            i += 1
    return out


#: #5400: the result mode of a test start that did not load the model because memory was short (the engine's binary check only), and the
#: share of a model's RAM figure (setup's MODELS[size]["ram_gb"], which counts the whole PC) that is the operating system's.
LOW_MEMORY_MODE = "binary-low-memory"
TABLE_OS_SHARE_GB = 4.0
_SETUP_MODELS: list = []


def _setup_models() -> dict:
    """setup.py's model table (MODELS), read once, quietly; {} when it cannot be read."""
    if not _SETUP_MODELS:
        table: dict = {}
        try:
            import importlib.util
            path = Path(__file__).resolve().parents[1] / "setup.py"
            spec = importlib.util.spec_from_file_location("strata_setup_for_update", path)
            mod = importlib.util.module_from_spec(spec)
            real = sys.stdout
            sys.stdout = sys.stderr
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.stdout = real
            table = dict(getattr(mod, "MODELS", {}) or {})
        except (Exception, SystemExit):                  # noqa: BLE001 - no table: the files' sizes decide
            table = {}
        _SETUP_MODELS.append(table)
    return _SETUP_MODELS[0]


#: RAM the test start keeps free beside what it plans (#5400; the resident mode's own STRATA_RESIDENT_HEADROOM_GIB is the same 4 GB), and the
#: engine's own memory beside the expert arena and the PLE table: 84.4 GB peak (VmHWM) on l620 for IQ3_S = 50.3 arena + 28.8 PLE + 5.3
TEST_HEADROOM_GB = 4.0
ENGINE_BASE_GB = 5.3
_LOW_RAM_FLAGS = ("--mmap-experts", "--resident-experts")


def _flag_number(args: list[str], flag: str) -> float | None:
    if flag in args[:-1]:
        try:
            return float(args[args.index(flag) + 1])
        except ValueError:
            return None
    return None


def test_ram_need_gb(cfg: dict) -> float | None:
    """The RAM the update's test start (light_test_args) needs for this run config, in GB, from the engine's own plan of where the experts go
    (docs/DETAILS.md "Low-RAM modes"), plus TEST_HEADROOM_GB:
    - the normal mode copies every expert into RAM: the model's expert arena (setup's MODELS[size]["arena_gb"]);
    - --resident-budget-gib N: N GiB of experts in RAM;
    - --mmap-experts / --resident-experts: the experts are mapped, or copied only when they fit beside the headroom the engine keeps itself: nothing
      beyond the headroom;
    the light arguments leave no KV cache in RAM and no picture reader. Without the arena figure: setup's whole-PC figure less the system's share,
    else the size of the model files. None when nothing is known (then the full test runs, as before)."""
    args = [str(a) for a in cfg.get("args") or []]
    extra = ENGINE_BASE_GB + TEST_HEADROOM_GB
    if "--ple-io" in args[:-1] and args[args.index("--ple-io") + 1] == "ram" and "--ple-gguf" in args[:-1]:
        try:
            extra += Path(args[args.index("--ple-gguf") + 1]).stat().st_size / 1e9        # the PLE table locked in RAM (not after light_test_args)
        except OSError:
            pass
    budget = _flag_number(args, "--resident-budget-gib")
    if budget is not None:
        return budget * 1.073741824 + extra
    if any(f in args for f in _LOW_RAM_FLAGS):
        return extra
    name = str(cfg.get("model_name") or "").lower()
    best = None
    for size, entry in _setup_models().items():
        if name.endswith("-" + str(size).lower()) and isinstance(entry, dict) and (entry.get("arena_gb") or entry.get("ram_gb")):
            if best is None or len(str(size)) > len(best[0]):
                best = (str(size), entry)
    if best is not None:
        entry = best[1]
        if entry.get("arena_gb"):
            return float(entry["arena_gb"]) + extra
        return max(0.0, float(entry["ram_gb"]) - TABLE_OS_SHARE_GB)
    total = 0
    for flag in ("--native", "--ple-gguf"):
        if flag in args[:-1]:
            try:
                total += Path(args[args.index(flag) + 1]).stat().st_size
            except OSError:
                pass
    return total / 1e9 + extra if total else None


def free_ram_gb() -> float | None:
    try:
        from strata_control import tune_machine
        mib = tune_machine.memory().get("available_mib")
        return None if mib is None else mib * 1048576 / 1e9
    except Exception:                                    # noqa: BLE001 - unknown: the full test
        return None


def low_memory(cfg: dict) -> tuple[float, float] | None:
    """(free GB, needed GB) when the memory free now is less than a test load of this model needs; None when there is room or either is unknown."""
    need, free = test_ram_need_gb(cfg), free_ram_gb()
    if need is None or free is None or free >= need:
        return None
    return free, need


def default_test_start(stage_root: Path, component: ec.EngineComponent, ctx: Context) -> dict:
    """Start the staged engine through the real start path (owner runtime, `component:<id>`, a saved model), wait for
    the one-token generation probe, and stop it. Everything runs in an isolated runtime, profile and component folder, so a
    user's own instances, profiles and engines are not involved. With no matching model installed the engine is only run with --version."""
    target_comp = component.id or ec.COMPONENT_ID
    source = Path(ctx.run_config) if ctx.run_config else discover_run_config(ctx.environ, component_id=target_comp)
    if source is not None and _config_component_id(source) != target_comp:
        # #6589 / #6623: never test an engine against another engine component's run config (e.g. Hy4 `--engram` on `strata`)
        source = discover_run_config(ctx.environ, component_id=target_comp)
    if source is None:
        if not ctx.allow_no_model_smoke:
            raise TestStartFailed(component.version, "no installed model to test with")
        return binary_smoke(component, ctx=ctx)
    from strata_control import profiles, runtime
    sandbox = Path(tempfile.mkdtemp(prefix="t-", dir=stage_root))
    try:
        cfg = json.loads(source.read_text(encoding="utf-8-sig"))
        # paths setup wrote inside a Control folder that is gone (an AppImage's /tmp/.mount_<random>, a moved install): this Control's own files (QAGUI-LIN 0213)
        from strata_control import control_paths
        cfg = control_paths.repoint(cfg)
        # #5681: paths in a version folder an update removed (components/control/0.1.56/...) are resolved to the current version HERE, while
        # STRATA_COMPONENTS_DIR is still the real one: under the sandbox's (the stage, which has no control folder) the start's own repair
        # found no current Control and the test start failed with a ProfileError about the old file
        from strata_control import cfg_paths
        cfg, _moved, _lost = cfg_paths.resolve_config(cfg)
        if _config_component_id(cfg) != target_comp:
            if not ctx.allow_no_model_smoke:
                raise TestStartFailed(component.version, "no installed model matching this engine component")
            return binary_smoke(component, ctx=ctx)
        cfg["exe"] = ec.REFERENCE_PREFIX + target_comp
        cfg["backend"] = component.backend
        cfg.pop("lib_dirs", None)
        if isinstance(cfg.get("args"), list):
            cfg["args"] = light_test_args(cfg["args"])
        cfg["env"] = {**(cfg.get("env") if isinstance(cfg.get("env"), dict) else {}), **TEST_ENV}
        short = (ctx.memory_check or low_memory)(cfg)
        if short:
            # #5400: loading the model now would push this PC out of memory (other programs hold it). The engine gets the check an install with no
            # model gets; it becomes "unproven" (_finish_engine_install), and if its first real start fails the previous engine is used again by itself.
            free, need = short
            model = cfg.get("model_name") or source.stem
            if component.version in (State(ctx.components).section("engine").get("failed_unproven") or {}):
                raise TestStartFailed(component.version, f"it could not be tested with your model: only {free:.0f} GB of memory is free right now and {model} needs "
                                                         f"about {need:.0f} GB, and the last time this engine was used without that test it could not start your model",
                                      busy=True)
            ctx.say("test", f"Only {free:.0f} GB of memory is free right now and {model} needs about {need:.0f} GB, so the new engine was checked without "
                            "loading the model. It loads your model the next time it starts.", component="engine")
            out = binary_smoke(component, ctx=ctx)
            return {**out, "mode": LOW_MEMORY_MODE, "model": model, "free_gb": round(free, 1), "need_gb": round(need, 1),
                    "detail": f"checked without the model: not enough free memory for a test load ({free:.0f} of {need:.0f} GB)"}
        cfg["host"] = "127.0.0.1"
        cfg["port"] = _free_port()
        cfg["log"] = str(sandbox / "engine.log")
        config_path = sandbox / "run.json"
        config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        env = {"STRATA_COMPONENTS_DIR": str(stage_root), "STRATA_PROFILE_DIR": str(sandbox / "profiles"),
               "STRATA_RUNTIME_DIR": str(sandbox / "runtime"), "STRATA_NO_ACQUIRE": "1",
               "STRATA_ENGINE_BACKEND": component.backend}
        name = "updtest"
        with _patched_environ(env):
            os.environ.pop("STRATA_API_KEY", None)
            profiles.create(name, str(config_path))
            started = None
            try:
                started = runtime.start(name, wait=min(ctx.test_timeout, 600), shutdown_timeout=20,
                                        on_status=_test_progress(ctx, component))
                if started.get("ok") and started.get("state") == "ready" and started.get("generation_ready") is not True:
                    # a server that loads its model on the first request is "ready" without having generated anything:
                    # force the load and the one-token probe, because that is what the new engine has to be able to do
                    ctx.say("test", f"Engine {component.version} is up; asking it for one token", component="engine")
                    started = {**started, **runtime.verify(name, timeout=min(ctx.test_timeout, 600), load=True)}
                if not started.get("ok") or started.get("state") != "ready" or started.get("generation_ready") is not True:
                    _keep_log(sandbox, component, ctx)
                    _keep_records(sandbox, component, ctx)
                    raise _test_start_failure(component, started, ctx, _log_tail(sandbox, component))
                _keep_log(sandbox, component, ctx)
                return {"mode": "full", "model": cfg.get("model_name") or source.stem, "state": started["state"],
                        "generation_ready": True, "detail": "started, answered a one-token probe, stopped"}
            except TestStartFailed:
                _keep_log(sandbox, component, ctx)
                _keep_records(sandbox, component, ctx)
                raise
            except Exception as exc:                    # noqa: BLE001
                _keep_log(sandbox, component, ctx)
                _keep_records(sandbox, component, ctx)
                raise TestStartFailed(component.version, f"{type(exc).__name__}: {exc}") from None
            finally:
                try:
                    runtime.stop(name, timeout=60)
                except Exception:                       # noqa: BLE001 - stopping a sandbox is best effort
                    pass
    finally:
        shutil.rmtree(longpath.long_path(sandbox), ignore_errors=True)


#: How long a folder rename is retried before the copy is used instead (seconds). A 950 MB engine that was just written is scanned by
#: the virus scanner for longer than the old 8 seconds on a slow disk (#4853: unpacked and checked, then "the update didn't finish").
MOVE_PATIENCE_S = 40.0


def _move_dir(source: Path, destination: Path, *, patience: float = MOVE_PATIENCE_S, sleep=time.sleep, clock=time.monotonic) -> None:
    """Put a checked folder where it belongs. A rename is the normal way; on Windows it is refused for as long as any program (the virus
    scanner, the indexer, Explorer) holds one file inside the folder, so it is retried for `patience` seconds. If it is still refused,
    the folder is copied instead: a copy needs only to read the files, and the caller checks the copy again (every file's hash is
    verified when the engine is chosen), so a copy that is wrong is never used."""
    last = None
    deadline = clock() + patience
    attempt = 0
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            last = exc
            if clock() >= deadline:
                break
            attempt += 1
            sleep(min(0.4 * attempt, 3.0))
    try:
        shutil.copytree(longpath.long_path(source), longpath.long_path(destination))
    except OSError as exc:
        shutil.rmtree(longpath.long_path(destination), ignore_errors=True)
        raise UpdateError(f"the new engine could not be moved into place ({longpath.describe_oserror(last)}); copying it did not work "
                          f"either ({longpath.describe_oserror(exc)})", 6) from None
    shutil.rmtree(longpath.long_path(source), ignore_errors=True)         # the staging folder is removed again at the end; a held file waits


def prune_engines(root: Path, keep: set[str]) -> list[str]:
    removed = []
    folder = ec.engine_dir(root)
    if not folder.is_dir():
        return removed
    for entry in sorted(folder.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            try:
                shutil.rmtree(longpath.long_path(entry))
                removed.append(entry.name)
            except OSError:
                pass                                     # in use: it goes with the next update
    return removed


def update_engine(ctx: Context, feed: feedmod.Feed, state: State) -> dict:
    """Install the best engine package for this PC if it is newer than the installed one. See the module docstring."""
    decision, pkg = _choose_engine(feed, ctx, state)
    status = decision["status"]
    result = {"component": "engine", "status": status, "installed_before": decision["installed"], "changed": False}
    if status not in ("install", "available") and not (ctx.force and status in ("failed-before", "skipped")):
        result["note"] = decision.get("note") or {"current": "The engine is up to date.",
                                                   "none-for-this-pc": "The feed has no engine package for this PC.",
                                                   "not-in-feed": "The feed lists no engines."}.get(status, status)
        if status == "none-for-this-pc":
            result["reasons"] = decision["reasons"]
        return result
    root = ctx.components
    local = decision.get("installed_locally")
    if local:
        return _switch_to_installed(ctx, state, local, result)
    have_makes = installed_gpu_makes(root, pkg.platform)
    if pkg.backend in GPU_MAKES and have_makes and pkg.backend not in have_makes and ctx.hardware.get("backend_source") not in ("environment", "requested", "saved choice"):
        raise UpdateError(f"the {pkg.label} engine is for another make of graphics card than the installed one; nothing was staged", 5)
    _warn_if_deep(ctx, ec.engine_dir(root) / folder_name(pkg), longpath.LONGEST_IN_PACKAGE.get(pkg.backend, longpath.LONGEST_IN_PACKAGE_DEFAULT) + 1)
    _free_space_check(root, pkg.size + pkg.installed_bytes, ctx, pkg.sha256, "engine")
    archive = _download(ctx, "engine", pkg.url, pkg.size, pkg.sha256, f"engine {pkg.label}", feed)
    # busy check after the long download: a model may have been started meanwhile; the verified archive is kept for the retry
    idle_gate(ctx)
    try:
        stage_root, staged = stage_engine(ctx, feed, pkg, archive)
    except UpdateError as exc:
        if exc.exit_code == 5:
            archive.unlink(missing_ok=True)             # a refused package is not kept for a retry
        raise
    return _finish_engine_install(ctx, state, pkg, stage_root, staged, archive, result, delete_archive=True)


def _switch_to_installed(ctx: Context, state: State, folder: str, result: dict) -> dict:
    """An update to an engine that is already on this computer: nothing is downloaded; it becomes the engine in use and any pin is cleared."""
    from strata_control import engine_library
    idle_gate(ctx)
    ctx.say("switch", "This engine is already on this computer; switching to it", component="engine")
    out = engine_library.use_qwen_unlocked(ctx, state, folder)
    chosen, _ = installed_engine(ctx.components, ctx.hardware)
    section = state.section("engine")
    if chosen is not None and chosen.version in (section.get("failed_unproven") or {}) and section.get("previous"):
        # (#5400) asked for again with --force after it could not start the model: still not proven, so a failed first start goes back once more
        section["unproven"] = {"version": chosen.version, "folder": chosen.root.name, "previous": section["previous"],
                               "at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()}
    result.update({"changed": out.get("changed", False), "status": "updated" if out.get("changed") else "current", "installed_after": _engine_view(chosen),
                   "previous": state.section("engine").get("previous"), "removed": [], "presets_marked": bool(out.get("presets_marked")), "from_disk": True})
    state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "update-engine", "from": (result.get("installed_before") or {}).get("version"),
                   "to": chosen.version if chosen else None, "ok": True, "detail": "already installed"})
    state.save()
    ctx.say("done", f"Engine {chosen.version if chosen else ''} is in use from the next start", component="engine")
    return result


def _finish_engine_install(ctx: Context, state: State, pkg: feedmod.EnginePackage, stage_root: Path, staged: ec.EngineComponent,
                           archive: Path, result: dict, *, delete_archive: bool) -> dict:
    """The part of an engine install both ways share (a download from the feed, a file the person chose): test-start the staged engine,
    switch to it, keep the previous one, remove the staging folder. ``archive`` is only deleted when Strata downloaded it."""
    root = ctx.components
    before_installed, _ = installed_engine(root, ctx.hardware)
    before_identity = preset_marks.identity_of(before_installed, _manifest_sha(before_installed.root)) if before_installed else None
    try:
        ctx.say("test", f"Test-starting engine {pkg.label} before switching to it", component="engine")
        runner = ctx.test_start or default_test_start
        try:
            smoke = runner(stage_root, staged, ctx)
        except TestStartFailed as exc:
            try:                                             # a model of Strata's own launch holds the card: one next step, not the engine's words
                own = [s for s in ctx.serving() if str(s.get("profile", "")).startswith("launch-")]
            except Exception:                                # noqa: BLE001
                own = []
            if own:
                raise Busy(own) from None
            section = state.section("engine")
            at = (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()
            if not exc.busy:                                 # a busy card says nothing about the engine: the next update tries again
                section.setdefault("failed", {})[pkg.version] = {
                    "manifest_sha256": pkg.manifest_sha256, "reason": exc.reason[:300], "at": at, "control": ec.runtime_version()}
            state.history({"at": at, "action": "update-engine", "to": pkg.version,
                           "from": before_installed.version if before_installed else None, "ok": False,
                           "detail": f"test start failed: {exc.reason[:200]}"})
            state.save()
            raise
        # last look before the switch: an engine may have started during the test-start
        idle_gate(ctx)
        name = folder_name(pkg)
        live = ec.engine_dir(root)
        live.mkdir(parents=True, exist_ok=True)
        final = live / name
        staged_folder = stage_root / "engine" / name
        if final.exists():
            if (final / ec.MANIFEST).is_file() and _manifest_sha(final) == pkg.manifest_sha256:
                shutil.rmtree(longpath.long_path(staged_folder), ignore_errors=True)       # already installed byte for byte: only the state changes
            else:
                final = live / f"{name}-{pkg.manifest_sha256[:8]}"
                _move_dir(staged_folder, final)
        else:
            _move_dir(staged_folder, final)
        ctx.say("switch", f"Switching to engine {pkg.label}", component="engine")
        section = state.section("engine")
        old_pin = section.pop("pinned", None)
        state.save()                                      # a start reads the pin from the file, so the unpin is written before the check
        try:
            chosen = ec.resolve(root, backend=ctx.hardware.get("backend"), full=True, runtime=ctx.hardware.get("runtime"))
            if chosen.root.name != final.name:
                raise ec.EngineMissing(f"a start would pick {chosen.root.name}, not {final.name}", root)
        except ec.EngineMissing as exc:
            if old_pin:
                section["pinned"] = old_pin
                state.save()
            # the rename did not give a working engine: put it back where it cannot be picked
            shutil.rmtree(longpath.long_path(live / ".rejected"), ignore_errors=True)
            (live / ".rejected").mkdir(parents=True, exist_ok=True)
            if final.exists():
                _move_dir(final, live / ".rejected" / final.name)
            raise UpdateError(f"the new engine was not accepted after the switch ({exc}); the previous engine is unchanged", 6) from None
        previous = before_installed.root.name if before_installed and before_installed.root.name != final.name else section.get("previous")
        section["previous"] = previous
        section["active"] = final.name
        section.get("failed", {}).pop(pkg.version, None)
        if smoke.get("mode") == LOW_MEMORY_MODE and previous:
            section["unproven"] = {"version": pkg.version, "folder": final.name, "previous": previous,
                                   "at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat()}       # #5400: proven by its first real start
        else:
            section.pop("unproven", None)
            if smoke.get("mode") == "full":
                (section.get("failed_unproven") or {}).pop(pkg.version, None)
        skipped = [v for v in section.get("skipped", []) if v != pkg.version]
        section["skipped"] = skipped
        after_identity = preset_marks.identity_of(chosen, pkg.manifest_sha256)
        mark = preset_marks.mark_engine_changed(before_identity, after_identity, base=ctx.state_base)
        removed: list[str] = []                           # every older engine stays until the person deletes it (Settings > Engine, `strata components remove`)
        state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "update-engine",
                       "from": before_installed.version if before_installed else None, "to": pkg.version, "ok": True,
                       "detail": smoke.get("mode")})
        state.save()
        if delete_archive:
            archive.unlink(missing_ok=True)
        result.update({"changed": True, "status": "updated", "installed_after": _engine_view(chosen), "test_start": smoke,
                       "previous": previous, "removed": removed, "presets_marked": bool(mark)})
        ctx.say("done", f"Engine {pkg.version} is installed and will be used on the next start", component="engine")
        return result
    finally:
        shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)


def _archive_format(archive: Path) -> str:
    name = archive.name.lower()
    for suffix, fmt in ((".tar.gz", "tar.gz"), (".tgz", "tar.gz"), (".tar.xz", "tar.xz"), (".zip", "zip")):
        if name.endswith(suffix):
            return fmt
    raise UpdateError("this is not a Strata engine package (it should be a .tar.gz or .zip file from the release page)", 5)


def install_engine_file(ctx: Context, archive: Path) -> dict:
    """Install an engine package the person downloaded themselves (the offline path of the first-run card). No feed and no network:
    the package must carry the feed key's signature on its manifest, every file must match the manifest, and the package must fit this
    PC. After that it goes through the same test start and switch as a downloaded one. The file the person chose is left where it is."""
    archive = Path(archive)
    if not archive.is_file():
        raise UpdateError(f"{archive} is not a file", 5)
    fmt = _archive_format(archive)
    state_root = ctx.components
    with update_lock(state_root):
        state = State(state_root)
        stage_root = new_stage(state_root)
        keep_stage = False
        try:
            try:
                total = _archive_size(archive, fmt)
            except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
                raise UpdateError(f"this file cannot be read as a package ({type(exc).__name__}); download it again", 5) from None
            _free_space_check(state_root, total * 2, ctx, "", "engine")
            folder = stage_root / "engine" / "incoming"
            longpath.long_path(folder.parent).mkdir(parents=True, exist_ok=True)
            ctx.say("verify", "Unpacking the engine file", component="engine")
            extract(archive, folder, fmt, total)
            try:
                component = ec.read_manifest(folder)
            except ec.ComponentProblem as exc:
                raise UpdateError(f"this file is not a Strata engine package: {exc}", 5) from None
            ctx.say("verify", "Checking the package signature and every file", component="engine")
            outer = ec._verifier
            signing.install_component_verifier(ctx.trusted)
            try:
                ec.verify_signature(component, {**ctx.environ, ec.ENV_REQUIRE_SIGNED: "1"})
                ec.verify(component, full=True)
            except ec.ComponentProblem as exc:
                raise UpdateError(f"this file did not pass the security check, so it was not used ({exc})", 5) from None
            finally:
                ec.register_signature_verifier(outer)
            pkg = feedmod.EnginePackage(
                version=component.version, platform=component.platform, backend=component.backend, variant=component.variant or "",
                archs=tuple(component.archs), min_driver=dict(component.min_driver or {}), min_runtime=component.min_runtime,
                build_id=component.build_id, url="", format=fmt, size=archive.stat().st_size, sha256=transfer.sha256_file(archive),
                manifest_sha256=transfer.sha256_file(folder / ec.MANIFEST), installed_bytes=total)
            why = feedmod.engine_fit_reasons(pkg, ctx.hardware)
            if why:
                raise UpdateError("this file is not a Strata engine for this PC: " + "; ".join(why), 5)
            # the staged folder gets the name an installed package has
            final_stage = stage_root / "engine" / folder_name(pkg)
            if final_stage != folder:
                _move_dir(folder, final_stage)
            staged = ec.read_manifest(final_stage)
            before, _ = installed_engine(state_root, ctx.hardware)
            if before is not None and feedmod.version_key(before.version) > feedmod.version_key(pkg.version) and not ctx.force:
                return {"component": "engine", "status": "current", "changed": False, "installed_before": _engine_view(before),
                        "note": f"Engine {before.version} is already installed and is newer than this file ({pkg.version}); nothing was changed."}
            result = {"component": "engine", "status": "install" if before is None else "available",
                      "installed_before": _engine_view(before), "changed": False, "source": "file"}
            keep_stage = True
            return _finish_engine_install(ctx, state, pkg, stage_root, staged, archive, result, delete_archive=False)
        finally:
            if not keep_stage:
                shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)


def switch_engine(ctx: Context, state: State, name: str, target: ec.EngineComponent, *, unpin_if_automatic: bool = False,
                  what: str = "engine") -> dict:
    """Make the installed package folder ``name`` the engine a start uses. Nothing is downloaded or removed; the package was verified by the
    caller. A pin makes the package the one a start picks even though a newer one is installed. With ``unpin_if_automatic`` the pin is dropped
    when the package is what a start picks anyway (the newest one), so the engine follows updates again. One writer of the pin and of the
    previous / skipped bookkeeping: `strata components rollback` and `use_engine` (Settings > Engine, `strata components use`) both come here."""
    section = state.section("engine")
    section.pop("unproven", None)                     # (#5400) an engine the person chose, or went back to, is never switched away from by itself
    current, _ = installed_engine(ctx.components, ctx.hardware)
    leaving = current.root.name if current else None
    before_identity = preset_marks.identity_of(current, _manifest_sha(current.root)) if current else None
    old_pin = section.get("pinned")
    automatic = False
    if unpin_if_automatic:
        section.pop("pinned", None)
        state.save()
        try:
            picked = ec.resolve(ctx.components, backend=ctx.hardware.get("backend"), full=True, runtime=ctx.hardware.get("runtime"))
            automatic = picked.root.name == name
        except ec.EngineMissing:
            automatic = False
    if not automatic:
        section["pinned"] = name
    state.save()                                      # a start reads the pin from the file, so it is written before it is checked
    try:
        chosen = ec.resolve(ctx.components, backend=ctx.hardware.get("backend"), full=True, runtime=ctx.hardware.get("runtime"))
        if chosen.root.name != name:
            raise ec.EngineMissing(f"a start would pick {chosen.root.name}", ctx.components)
    except ec.EngineMissing as exc:
        if old_pin:
            section["pinned"] = old_pin
        else:
            section.pop("pinned", None)
        state.save()
        raise UpdateError(f"the {what} does not fit this PC any more: {exc}", 5) from None
    forward = bool(current) and feedmod.version_key(target.version) > feedmod.version_key(current.version)
    if forward:
        # going to the newer engine: it is no longer one the person left, so a later update may offer it
        section["skipped"] = [v for v in section.get("skipped", []) if v != target.version]
    elif current and current.version != target.version:
        skip = section.setdefault("skipped", [])
        if current.version not in skip:
            skip.append(current.version)
    section["previous"] = leaving
    section["active"] = name
    mark = preset_marks.mark_engine_changed(before_identity, preset_marks.identity_of(chosen, _manifest_sha(chosen.root)), base=ctx.state_base)
    state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "rollback-engine" if what != "engine" else "use-engine",
                   "from": current.version if current else None, "to": target.version, "ok": True})
    state.save()
    return {"component": "engine", "status": "rolled-forward" if forward else "rolled-back", "from": current.version if current else None,
            "to": target.version, "folder": name, "presets_marked": bool(mark), "automatic": automatic}


def rollback_engine(ctx: Context) -> dict:
    """Return to the previous engine version, which the last switch or update left. Nothing is downloaded. A pin makes the older
    package the one a start picks even though a newer one is installed; the next successful update clears it."""
    with update_lock(ctx.components):
        state = State(ctx.components)
        section = state.section("engine")
        previous = section.get("previous")
        if not previous:
            raise UpdateError("there is no previous engine to return to (the last update kept none, or none has happened)", 2)
        folder = ec.engine_dir(ctx.components) / previous
        wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
        try:
            target = ec.read_manifest(folder)
            signing.ensure_component_verifier()
            ec.verify(target, full=True)
        except (ec.ComponentProblem, OSError) as exc:
            raise UpdateError(f"the previous engine ({previous}) can no longer be used: {exc}", 5) from None
        current, _ = installed_engine(ctx.components, ctx.hardware)
        if current is not None and current.root.name == previous:
            # an earlier rollback is still in force (S3: the text said "0.1.45.3 -> 0.1.45.3"): there is nothing to leave behind
            raise UpdateError(f"the engine is already the previous one ({current.version}); there is nothing older to return to. "
                              "Run `strata update --force` to try the newer version again.", 2)
        return switch_engine(ctx, state, previous, target, what="previous engine")


#: #5400: what a start says after it went back to the previous engine by itself.
FALLBACK_NOTE = ("The new engine {new} could not start your model, so Strata went back to engine {old}. Your settings are unchanged; "
                 "the update will be offered again.")


def unproven_in_use(root: Path | None = None, hardware: dict | None = None) -> dict | None:
    """The "unproven" record (#5400) when the engine a start uses now is the one the update switched to without a test load; else None."""
    root = Path(root) if root else ec.components_dir()
    try:
        u = State(root).section("engine").get("unproven")
        if not isinstance(u, dict) or not u.get("folder"):
            return None
        current, _ = installed_engine(root, hardware if hardware is not None else detect_hardware())
        return u if current is not None and current.root.name == u["folder"] else None
    except Exception:                                    # noqa: BLE001 - a start never fails on this bookkeeping
        return None


def mark_proven(root: Path | None = None, hardware: dict | None = None) -> bool:
    """A start reached ready with the unproven engine: it is proven, the record goes. True when there was one."""
    root = Path(root) if root else ec.components_dir()
    if unproven_in_use(root, hardware) is None:
        return False
    try:
        with update_lock(root):
            state = State(root)
            state.section("engine").pop("unproven", None)
            state.save()
        return True
    except Exception:                                    # noqa: BLE001
        return False


def fall_back_from_unproven(root: Path | None = None, *, hardware: dict | None = None, emit: Callable[[dict], None] = lambda event: None,
                            serving: Callable[[], list[dict]] = serving_now) -> dict | None:
    """The first real start with an unproven engine failed (#5400): go back to the previous engine (rollback_engine: nothing downloaded), once.
    The version is remembered (failed_unproven) so the next offer of it needs the full test, and it is not hidden from the update (not "skipped").
    None when there is nothing to do or the switch is not possible now (another model serving, the previous engine gone)."""
    root = Path(root) if root else ec.components_dir()
    hardware = hardware if hardware is not None else detect_hardware()
    u = unproven_in_use(root, hardware)
    if u is None:
        return None
    ctx = Context(components=root, source="", trusted={}, hardware=hardware, emit=emit, serving=serving)
    try:
        out = rollback_engine(ctx)
    except (UpdateError, Busy, OSError):
        return None
    with update_lock(root):
        state = State(root)
        section = state.section("engine")
        section.pop("unproven", None)
        section.setdefault("failed_unproven", {})[u["version"]] = (dt.datetime.now(dt.timezone.utc)).isoformat()
        section["skipped"] = [v for v in section.get("skipped", []) if v != u["version"]]
        state.history({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "action": "fallback-engine", "from": u["version"], "to": out.get("to"),
                       "ok": True, "detail": "the first start with an engine switched to without a test load failed"})
        state.save()
    return {"from": u["version"], "to": out.get("to"), "message": FALLBACK_NOTE.format(new=u["version"], old=out.get("to"))}


# ---------------------------------------------------------------------------------------------- control

def control_dir(root: Path) -> Path:
    return root / "control"


def active_control(root: Path) -> dict | None:
    try:
        data = json.loads((control_dir(root) / "active.json").read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) and data.get("version") and data.get("folder") else None
    except (OSError, ValueError):
        return None


def source_tree_version(root: Path | None = None) -> str | None:
    """The version of a source checkout (it has no VERSION file): the project version in its CMakeLists.txt, the same rule `strata --version` uses.
    Used only to PRINT what a checkout runs (`update --check` said `Control  ?: available`); the decision to stage a Control copy does not use it."""
    root = root or Path(__file__).resolve().parents[1]
    try:
        found = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)", (root / "CMakeLists.txt" if (root / "CMakeLists.txt").is_file() else root.parent / "engine" / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    except OSError:
        return None
    return found.group(1) if found else None


def decide_control(feed: feedmod.Feed, ctx: Context, state: State) -> dict:
    here = ctx.hardware["platform"]
    pkg = next((p for p in feed.control if p.platform == here), None)
    installed = (active_control(ctx.components) or {}).get("version") or ec.runtime_version()
    report = {"installed": installed, "offered": pkg.version if pkg else None, "status": "not-in-feed", "download_bytes": 0}
    if pkg is None:
        return report
    report["download_bytes"] = pkg.size
    if not installed:
        report["status"] = "available"
    else:
        have, want = feedmod.version_key(installed), feedmod.version_key(pkg.version)
        report["status"] = "available" if want > have else "current"
    return report


CONTROL_VERSION_TIMEOUTS = (240, 360)       # seconds: antivirus scans a fresh python.exe on its first run, which can take minutes


def _control_version(interpreter: str, entry: Path, stage: Path, ctx: Context, version: str):
    """Run `--version` of the staged Control; once more, with more time, when the first try timed out, failed to start or said nothing."""
    last_error = None
    done = None
    for attempt, timeout in enumerate(CONTROL_VERSION_TIMEOUTS):
        try:
            done = subprocess.run([interpreter, str(entry), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(stage),
                                  env={**ctx.environ, "PYTHONNOUSERSITE": "1", "STRATA_NO_ACQUIRE": "1", "STRATA_HANDOVER": "1"},
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            last_error = None
            if done.returncode == 0 and version in done.stdout:
                return done
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = exc
        if attempt + 1 < len(CONTROL_VERSION_TIMEOUTS):
            ctx.say("test", f"Control {version} did not answer yet (a virus scanner may be checking it); trying once more", component="control")
    if last_error is not None:
        raise TestStartFailed(f"Control {version}", f"{type(last_error).__name__}: {last_error}") from None
    return done


def _failed_mark_is_old(mark: dict, ctx: Context, hours: float = 24.0) -> bool:
    try:
        then = dt.datetime.fromisoformat(str(mark.get("at")))
        if then.tzinfo is None:
            then = then.replace(tzinfo=dt.timezone.utc)
        return (ctx.now or dt.datetime.now(dt.timezone.utc)) - then > dt.timedelta(hours=hours)
    except (TypeError, ValueError):
        return False


def update_control(ctx: Context, feed: feedmod.Feed, state: State) -> dict:
    """Stage a newer Control bundle beside the running one, run it once (`--version`), and only then point the app and the
    terminal at it (``components/control/active.json``). The running Control is never modified."""
    decision = decide_control(feed, ctx, state)
    result = {"component": "control", "status": decision["status"], "changed": False, "installed_before": decision["installed"]}
    if decision["status"] != "available":
        return result
    pkg = next(p for p in feed.control if p.platform == ctx.hardware["platform"])
    root = ctx.components
    _free_space_check(root, pkg.size + pkg.installed_bytes, ctx, pkg.sha256, "control")
    archive = _download(ctx, "control", pkg.url, pkg.size, pkg.sha256, f"Control {pkg.version}", feed)
    idle_gate(ctx)                                       # a new Control never takes over beside a model that is loaded without the person's yes
    stage = new_stage(root, "c")
    try:
        ctx.say("verify", f"Unpacking Control {pkg.version}", component="control")
        extract(archive, stage, pkg.format, pkg.installed_bytes, allow_links=pkg.format != "zip")
        entry = stage / pkg.entry
        if not entry.is_file():
            raise UpdateError(f"the Control package has no {pkg.entry}", 5)
        python = stage / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
        interpreter = str(python) if python.is_file() else sys.executable
        ctx.say("test", f"Test-starting Control {pkg.version}", component="control")
        done = _control_version(interpreter, entry, stage, ctx, pkg.version)
        if done.returncode != 0 or pkg.version not in done.stdout:
            raise TestStartFailed(f"Control {pkg.version}", f"--version said {done.stdout.strip()[:120]!r} (exit {done.returncode})")
        final = control_dir(root) / pkg.version
        control_dir(root).mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(longpath.long_path(final), ignore_errors=True)
        _move_dir(stage, final)
        section = state.section("control")
        previous = active_control(root)
        section["previous"] = previous
        pointer = control_dir(root) / "active.json"
        temp = pointer.with_suffix(".tmp")
        temp.write_text(json.dumps({"version": pkg.version, "folder": pkg.version, "sha256": pkg.sha256}), encoding="utf-8")
        _atomic_mod.replace(temp, pointer)
        keep = {pkg.version} | ({previous["folder"]} if previous else set())
        for entry_dir in control_dir(root).iterdir():
            if entry_dir.is_dir() and entry_dir.name not in keep and not folder_use.in_use(entry_dir):      # (never the Control a running server runs from)
                shutil.rmtree(longpath.long_path(entry_dir), ignore_errors=True)
        state.history({"at": (ctx.now or dt.datetime.now(dt.timezone.utc)).isoformat(), "action": "update-control",
                       "from": decision["installed"], "to": pkg.version, "ok": True})
        archive.unlink(missing_ok=True)
        result.update({"changed": True, "status": "updated", "installed_after": pkg.version,
                       "note": "the Terminal hands over to it from the next command; the app uses it from its next start"})
        return result
    finally:
        shutil.rmtree(longpath.long_path(stage), ignore_errors=True)


# ---------------------------------------------------------------------------------------------- app

def app_public_key(environ=None) -> str | None:
    environ = os.environ if environ is None else environ
    explicit = environ.get("STRATA_APP_UPDATE_PUBKEY")
    if explicit:
        return explicit
    try:
        data = json.loads(signing.BUILTIN_KEYS.read_text(encoding="utf-8-sig"))
        value = data.get("app_minisign_public")
        return value if isinstance(value, str) and value else None
    except (OSError, ValueError):
        return None


def decide_app(feed: feedmod.Feed, ctx: Context) -> dict:
    report = {"installed": ctx.app_version, "offered": None, "status": "not-in-feed", "notes": "", "download_bytes": 0}
    if feed.app is None:
        return report
    entry = feed.app.for_platform(ctx.hardware["platform"])
    if entry is None:
        report["status"] = "none-for-this-pc"
        return report
    report.update({"offered": feed.app.version, "notes": feed.app.notes, "download_bytes": entry["size"]})
    if not ctx.app_version:
        report["status"] = "unknown-installed"
    else:
        report["status"] = "available" if feedmod.app_version_key(feed.app.version) > feedmod.app_version_key(ctx.app_version) else "current"
    return report


def update_app(ctx: Context, feed: feedmod.Feed) -> dict:
    """The Terminal's share of an app update: download the installer, check its SHA-256 (signed feed) and its minisign signature
    (the key the app trusts), and leave it in ``.updates/app``. The Desktop app installs itself through Tauri's updater; a
    terminal only fetches and verifies, because installing an app is the platform installer's job."""
    decision = decide_app(feed, ctx)
    result = {"component": "app", "status": decision["status"], "changed": False, "installed_before": decision["installed"]}
    if decision["status"] not in ("available", "unknown-installed"):
        return result
    entry = feed.app.for_platform(ctx.hardware["platform"])
    key = app_public_key(ctx.environ)
    if not key:
        raise UpdateError("this build has no app update key (update_keys.json app_minisign_public); the app installer cannot be verified, so it was not downloaded", 5)
    root = ctx.components
    _free_space_check(root, entry["size"], ctx, entry["sha256"], "app")
    archive = _download(ctx, "app", entry["url"], entry["size"], entry["sha256"], f"Strata app {feed.app.version}", feed)

    def chunks():
        with archive.open("rb") as stream:
            yield from iter(lambda: stream.read(1 << 20), b"")
    try:
        signing.minisign_verify(key, entry["signature"], chunks())
    except signing.SignatureError as exc:
        archive.unlink(missing_ok=True)
        raise UpdateError(f"the app installer failed its signature check: {exc}; it was deleted", 5) from None
    target_dir = updates_dir(root) / "app"
    target_dir.mkdir(parents=True, exist_ok=True)
    name = Path(transfer.resolve_ref(transfer.base_of(feed.source), entry["url"]).replace("\\", "/")).name or f"strata-app-{feed.app.version}"
    target = target_dir / name
    _atomic_mod.replace(archive, target)
    result.update({"changed": True, "status": "downloaded", "installed_after": None, "version": feed.app.version, "path": str(target),
                   "note": "Verified. Run the installer to finish; models and settings are kept."})
    return result


# ---------------------------------------------------------------------------------------------- the one command

def _installed_before(name: str, ctx: Context):
    """What was installed when a step failed, for the report: the engine's view, Control's version, or nothing."""
    try:
        if name == "engine":
            return _engine_view(installed_engine(ctx.components, ctx.hardware)[0])
        if name == "control":
            return (active_control(ctx.components) or {}).get("version") or ec.runtime_version()
    except Exception:                                   # noqa: BLE001 - reporting only
        pass
    return None


def pending_changes(feed: feedmod.Feed, ctx: Context, state: State, *, engine: bool = True, control: bool = True) -> list[str]:
    """What this update would install or switch, in words, from the feed alone (nothing is downloaded).  Empty: nothing to do."""
    out = []
    if engine:
        decision = decide_engine(feed, ctx, state)
        if decision["status"] in ("install", "available") or (ctx.force and decision["status"] in ("failed-before", "skipped")):
            offered = (decision.get("offered") or {}).get("version")
            have = (decision.get("installed") or {}).get("version")
            out.append(f"engine {have} -> {offered}" if have else f"engine {offered}")
    if control:
        decision = decide_control(feed, ctx, state)
        if decision["status"] == "available":
            out.append(f"Control {decision['installed']} -> {decision['offered']}")
    return out


def _model_check(ctx: Context, feed: feedmod.Feed, state: State, *, engine: bool, control: bool) -> None:
    """Before anything is downloaded: with a model loaded and something to install, either wait for it to stop (--wait), ask (a gate), or refuse.
    An update never starts by itself beside a loaded model, and never half-runs (the engine refused while Control moves on)."""
    changes = pending_changes(feed, ctx, state, engine=engine, control=control)
    if not changes:
        return
    if ctx.wait:
        wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
    serving = ctx.serving()
    if not serving:
        return
    gate = getattr(ctx, "gate", None)
    if gate is None:
        raise Busy(serving)
    gate.consent(serving, "It would update the " + " and the ".join(changes) + ".")


def _shape_steps(ctx: Context, feed: feedmod.Feed, engine: bool) -> list:
    """One step for each model-shape engine installed here (strata_control/shape_engine.py): the explicit update brings it up to the feed's, beside the older one; the Strata engine is
    not touched. A PC that never installed one has no such step."""
    if not engine:
        return []
    from strata_control import shape_engine
    return [(shape_engine.component_id(shape), (lambda shape=shape: shape_engine.update_one(ctx, feed, shape, install_missing=False)))
            for shape in shape_engine.installed_shapes(ctx.components)]


def run_update(ctx: Context, *, engine: bool = True, control: bool = True, app: bool = False) -> dict:
    """`strata update`: the feed once, then the engine, Control and (if asked) the app installer, each reported separately.
    A component that fails does not undo one that succeeded; the exit code is the worst one."""
    require_acquisition("the update")
    results, failure = [], None
    with update_lock(ctx.components):
        state = State(ctx.components)
        feed = open_feed(ctx, state)
        state.save()
        _model_check(ctx, feed, state, engine=engine, control=control)
        steps = ([("engine", lambda: update_engine(ctx, feed, state))] if engine else []) + \
                _shape_steps(ctx, feed, engine) + \
                ([("control", lambda: update_control(ctx, feed, state))] if control else []) + \
                ([("app", lambda: update_app(ctx, feed))] if app else [])
        for name, step in steps:
            try:
                results.append(step())
            except UpdateError as exc:
                results.append({"component": name, "status": "failed", "changed": False, "error": str(exc), "exit_code": exc.exit_code,
                                "installed_before": _installed_before(name, ctx)})
                failure = failure or exc
        if any(r.get("component") == "engine" and r.get("status") == "updated" for r in results):
            state.data.get("last_check", {}).pop("engine", None)
        try:                                              # the feed was read for this update: what it offers that is still newer is kept for the app's one card
            now_control = next((r.get("installed_after") for r in results if r.get("component") == "control" and r.get("status") == "updated"), None)
            state.data["last_check"] = _last_check(ctx, {"engine": decide_engine(feed, ctx, state), "app": decide_app(feed, ctx)}, now_control)
        except Exception:                                 # noqa: BLE001 - only a hint for the app
            pass
        state.save()
    return {"feed": {"source": ctx.source, "sequence": feed.sequence, "channel": feed.channel, "key_id": feed.key_id},
            "results": results, "exit_code": failure.exit_code if failure else 0}
