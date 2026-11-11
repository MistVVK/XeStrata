"""DeepSeek-V4.1's engine as its own package (feed member ``dsv41``), downloaded when DeepSeek is chosen.

It goes in ``<data>/engine-dsv41/<version>-<platform>-<backend>[-<variant>]/``, beside the Strata engine and never inside it, so the Qwen
engine's files are not touched.  The package is picked, downloaded (resumable), unpacked and checked with the same code as an engine update
(strata_control/updater.py: the signed feed, the archive's SHA-256, the manifest hash the feed names, the package signature, every file's hash);
there is no test start here (setup's own 16-token question follows the pack).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

NAME = "strata-dsv41"
DIR = "engine-dsv41"

#: A Mac engine that carries DeepSeek's picture tower on Metal has this text in its program (the same way drafting.engine_reads_selector_q4 reads its mark).  Pictures for
#: DeepSeek on a Mac are on exactly when the installed strata-dsv41 has it: capability, not a switch.
MARK_MAC_VISION = b"STRATA_DS4_METAL_VISION_V1"
MAC_TEXT_LINE = "DeepSeek on a Mac reads text only with this engine (its picture reader needs an engine update)."
_macvis: dict = {}


def reads_mac_vision(exe) -> bool:
    """Does the strata-dsv41 program at `exe` contain MARK_MAC_VISION?  Read from the file (never started), once per path, size and time; False when it cannot be read."""
    import mmap
    try:
        path = Path(exe)
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
    except (OSError, ValueError, TypeError):
        return False
    if key not in _macvis:
        try:
            with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _macvis[key] = data.find(MARK_MAC_VISION) >= 0
        except (OSError, ValueError):
            _macvis[key] = False
    return _macvis[key]


def mac_vision_state(exe) -> str:
    """"yes" (the engine has the marker), "no" (an engine is there without it: text only) or "unknown" (no engine installed yet: the first start fetches the newest, and
    setup looks again once it is there)."""
    if not exe or not Path(str(exe)).is_file():
        return "unknown"
    return "yes" if reads_mac_vision(exe) else "no"


def exe_name() -> str:
    return NAME + (".exe" if os.name == "nt" else "")


def version_key(folder_name: str) -> tuple:
    """The version at the front of a package folder's name ("0.1.49-windows-x86_64-cuda-cuda13" -> (0, 1, 49)), for choosing the newest."""
    head = str(folder_name).split("-")[0]
    return tuple(int(x) for x in head.split(".") if x.isdigit())


def find(dest: Path) -> Path | None:
    """The strata-dsv41 a start uses from `dest` (engine-dsv41/): the package the person chose in Settings > Engine (the `.chosen` file), else the newest
    installed one.  None when there is none."""
    try:
        pin = (Path(dest) / ".chosen").read_text(encoding="utf-8-sig").strip()
        if pin and "/" not in pin and "\\" not in pin and (Path(dest) / pin / exe_name()).is_file():
            return Path(dest) / pin / exe_name()
    except OSError:
        pass
    try:
        folders = sorted((d for d in Path(dest).iterdir() if d.is_dir() and (d / exe_name()).is_file()), key=lambda d: (version_key(d.name), d.name))
    except OSError:
        return None
    return folders[-1] / exe_name() if folders else None


def cards_words(archs) -> str:
    """The cards a package holds code for, in the words of the box they came in ("RX 9070 and R9700 (gfx1200/1201)")."""
    from strata_control import engine_component as ec
    archs = sorted(str(a) for a in archs or () if str(a).startswith("gfx"))
    names = {"gfx1200": "RX 9060", "gfx1201": "RX 9070 and R9700"}
    named = [names[a] for a in archs if a in names]
    who = ", ".join(dict.fromkeys(named)) if named and len(named) == len(archs) else " and ".join(sorted({ec.hip_family(a) for a in archs})) or "other"
    short = "/".join(archs[:1] + [a[3:] for a in archs[1:]]) if archs else ""
    return f"{who} ({short})" if short else who


def card_unknown_line(archs) -> str:
    """The one plain line when an AMD package is installed although no probe could name this PC's graphics card."""
    return (f"This DeepSeek engine runs on {cards_words(archs)}; your graphics card wasn't recognised, so Strata installs it anyway "
            "and it will say if the card is not one of those.")


def card_mismatch_line(archs, here) -> str:
    """The one plain line when an AMD package has no code for the card(s) this PC has."""
    from strata_control import engine_component as ec
    have = " and ".join(sorted({ec.hip_family(a) for a in here}))
    return (f"This DeepSeek engine runs on {cards_words(archs)}; this computer has an AMD {have} card ({', '.join(sorted(set(here)))}), "
            "which this release holds no DeepSeek engine code for, so DeepSeek cannot start on it yet.")


def package_archs(exe: str | os.PathLike) -> tuple[str, ...]:
    """The AMD architectures an installed or hand-made package holds code for: its manifest's list, else BUILD.json's "archs" beside the
    program (a package made by hand). Empty when the package names none (a build of the person's own)."""
    folder = Path(exe).parent
    try:
        from strata_control import engine_component as ec
        if (folder / ec.MANIFEST).is_file():
            found = tuple(a for a in ec.read_manifest(folder).hip_archs if str(a).startswith("gfx"))
            if found:
                return found
    except Exception:                                      # noqa: BLE001 - a package that cannot be read names nothing
        pass
    try:
        import json
        data = json.loads((folder / "BUILD.json").read_text(encoding="utf-8-sig"))
        return tuple(str(a) for a in data.get("archs") or () if str(a).startswith("gfx"))
    except (OSError, ValueError, AttributeError):
        return ()


def card_problem(archs, here) -> str | None:
    """The plain line when `archs` (the package's) is known, `here` (this PC's cards) is known and no card of this PC is among the package's
    code; None when it fits, or when either side is unknown (an empty list is never a refusal: the caller says the card is unknown)."""
    from strata_control import feed as feedmod
    if not archs or not here or feedmod.gfx_covers(archs, list(here)):
        return None
    return card_mismatch_line(archs, here)


ARCHIVE_SUFFIXES = (".tar.gz", ".tar.xz", ".zip")
MAX_UNPACKED = 8 << 30                  # a DeepSeek engine package is under a GB; this only stops an archive that unpacks into something huge

LIB_VAR = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"


def lib_dirs(exe: str | os.PathLike) -> list[str]:
    """The folders that must be on the library path when this engine starts: its own package folder, then the folders the package's
    BUILD.json names.  A Windows HIP package keeps the ROCm DLLs (hipBLAS / rocBLAS and their imports) in ``rocm/bin``, which is also
    where rocBLAS's kernel folders and ``../.kpack`` are laid out from (strata_control/engine_component.py package_lib_dirs).  A CUDA
    package keeps everything beside the program, so this is its folder alone."""
    from strata_control import engine_component as ec
    try:
        return ec.package_lib_dirs(Path(exe).parent)
    except OSError:
        return [str(Path(exe).parent)]


def start_env(exe: str | os.PathLike, environ: dict | None = None) -> dict:
    """`environ` (the process's own by default) with this package's folders first on PATH (Windows) / LD_LIBRARY_PATH."""
    env = dict(os.environ if environ is None else environ)
    dirs = [d for d in lib_dirs(exe) if Path(d).is_dir()]
    if dirs:
        env[LIB_VAR] = os.pathsep.join(dirs + ([env[LIB_VAR]] if env.get(LIB_VAR) else []))
    return env


def pending(dest: Path) -> list[Path]:
    """Packages the person put in `dest` (engine-dsv41/) by hand: strata-dsv41-*.tar.gz / .zip, newest first."""
    try:
        files = [f for f in Path(dest).iterdir() if f.is_file() and f.name.startswith("strata-dsv41") and f.name.endswith(ARCHIVE_SUFFIXES)]
    except OSError:
        return []
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def _remove_older(dest: Path, keep: str) -> None:
    """Remove the other package folders in `dest`, except any a running engine uses: removing those would take the files from under a loaded model."""
    from strata_control import dsv41_paths
    for old in Path(dest).iterdir():
        if old.is_dir() and old.name != keep and not dsv41_paths.in_use(old):
            shutil.rmtree(old, ignore_errors=True)


def install_package(dest: Path, package: str | os.PathLike, *, emit: Callable[[str], None] = print) -> Path | None:
    """Install a DeepSeek engine package the person downloaded from the release page (a .tar.gz / .zip, or the unpacked folder) into `dest`.
    Checked before anything is installed: it is a DeepSeek engine package for this system, its manifest is signed by a key Strata trusts, and every
    file matches the hash in the manifest.  Returns the new strata-dsv41, or None with a plain line through `emit`."""
    from strata_control import engine_component as ec, signing, updater
    from strata_control.targets import host_platform

    src, dest = Path(package).expanduser(), Path(dest)
    if not src.exists():
        emit(f"{src} is not there. Download the DeepSeek engine package for your system from the release page and give its path again.")
        return None
    work = dest.parent / (DIR + ".manual")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        if src.is_dir():
            root = work / "pkg"
            shutil.copytree(src, root)
        else:
            fmt = "zip" if src.name.endswith(".zip") else "tar.xz" if src.name.endswith(".tar.xz") else "tar.gz"
            if not src.name.endswith(ARCHIVE_SUFFIXES):
                emit(f"{src.name} is not a package file (.tar.gz or .zip). Download the DeepSeek engine package for your system from the release page.")
                return None
            root = work / "pkg"
            updater.extract(src, root, fmt, MAX_UNPACKED)
        if not (root / ec.MANIFEST).is_file():
            inner = next((d for d in root.iterdir() if d.is_dir() and (d / ec.MANIFEST).is_file()), None) if root.is_dir() else None
            if inner is None:
                emit(f"{src.name} is not an engine package (it has no {ec.MANIFEST}).")
                return None
            root = inner
        component = ec.read_manifest(root)
        if component.entrypoint != exe_name():
            emit(f"{src.name} is not the DeepSeek engine (its program is {component.entrypoint}, not {exe_name()}). "
                 "The Strata engine package is a different download.")
            return None
        if component.platform != host_platform():
            emit(f"{src.name} is for {component.platform}, and this computer is {host_platform()}. Download the one for your system.")
            return None
        if component.backend == "hip":                          # a package for other cards than this PC's can only fail at its first kernel
            from strata_control import updater as _updater
            here = _updater.detect_hardware().get("gfx") or []
            problem = card_problem(tuple(a for a in component.archs if str(a).startswith("gfx")), here)
            if problem:
                emit(problem + " Nothing was installed.")
                return None
            if not here:
                emit(card_unknown_line(component.archs))
        outer = ec._verifier
        signing.install_component_verifier(signing.load_trusted_keys())
        try:
            ec.verify_signature(component, {**os.environ, ec.ENV_REQUIRE_SIGNED: "1"})
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            emit(f"{src.name} did not pass its check ({exc}). It may be damaged or not from the Strata release: download it again from the release page.")
            return None
        finally:
            ec.register_signature_verifier(outer)
        name = f"{component.version}-{component.platform}-{component.backend}" + (f"-{component.variant}" if component.variant else "")
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(root), str(target))
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        _remove_older(dest, target.name)
        emit(f"DeepSeek engine {component.version} ({component.backend}) checked (signed, every file matches) and installed: {target}")
        return target / exe_name()
    except (updater.UpdateError, ec.ComponentProblem, OSError) as exc:
        emit(f"{src.name} could not be installed: {exc}")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install(dest: Path, source: str, *, hardware: dict | None = None, emit: Callable[[str], None] = print, cancel=None) -> Path | None:
    """Fetch the DeepSeek engine package for this PC from the signed feed at `source` (a URL or a folder) into `dest`.
    Returns the new strata-dsv41, or None with a plain line through `emit` when there is none (not in the feed, nothing for this PC)."""
    from strata_control import feed as feedmod, signing, updater

    dest = Path(dest)
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                          emit=lambda event: emit(event["message"]) if event.get("event") == "phase" and event.get("message") else None)
    state = updater.State(work)
    try:
        feed = updater.open_feed(ctx, state)
        pkg, reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None:
            hip = [p for p in feed.dsv41 if p.backend == "hip" and p.platform == hw.get("platform")]
            line = card_problem(tuple(sorted({a for p in hip for a in p.archs if str(a).startswith("gfx")})), hw.get("gfx") or []) if hw.get("backend") == "hip" else None
            emit(line or ("There is no DeepSeek engine package for this computer in the update feed"
                          + (": " + "; ".join(reasons[:2]) if reasons else "") + "."))
            return None
        if pkg.backend == "hip" and not hw.get("gfx"):
            emit(card_unknown_line(pkg.archs))                 # (feed.gfx_covers lets an unknown card through; say so, once)
        archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
        stage_root, component = updater.stage_engine(ctx, feed, pkg, archive)
        try:
            folder = stage_root / "engine" / updater.folder_name(pkg)
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / folder.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(folder), str(target))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        _remove_older(dest, target.name)                        # one generation: the older folders of this engine go (never one a running engine uses)
        emit(f"DeepSeek engine {pkg.version} installed: {target}")
        shutil.rmtree(work, ignore_errors=True)               # (on a failure the partial download stays in `work`: the next try resumes it)
        return target / exe_name()
    except (updater.UpdateError, feedmod.FeedError, signing.SignatureError) as e:
        emit(f"The DeepSeek engine could not be fetched: {e}")
        return None


# ---------------------------------------------------------------------------------------------- upgrade of an installed package

def source_reachable(source: str, timeout: float = 10.0) -> bool:
    """Can the feed `source` be reached within `timeout` seconds?  A folder or file: it exists.  A URL: any HTTP answer counts (a 404 is an answer);
    no answer, a refused or timed-out connection does not.  One quick check instead of minutes of retries."""
    import urllib.error
    import urllib.parse
    import urllib.request
    from strata_control import transfer
    kind = transfer.kind_of(source)
    if kind in ("path", "file"):
        try:
            return transfer._local_path(source).exists()
        except (OSError, transfer.TransferError):
            return False
    parts = urllib.parse.urlsplit(source)
    try:
        urllib.request.urlopen(urllib.request.Request(f"{parts.scheme}://{parts.netloc}/", method="HEAD", headers={"User-Agent": "strata-launch"}),
                               timeout=timeout).close()
        return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _test_start(exe: Path, timeout: float = 120.0) -> tuple[bool, str]:
    """The new engine starts: `strata-dsv41 --version` runs, loads its libraries and names itself (it needs no graphics card and no model).
    The package's own folders go on the library path first: a Windows HIP package carries the ROCm runtime it loads in rocm/bin."""
    import subprocess
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(Path(exe).parent),
                              env=start_env(exe))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"it could not be started ({exc})"
    first = ((done.stdout or "").strip().splitlines() or [""])[0]
    if done.returncode != 0 or not first.startswith(NAME):
        tail = ((done.stderr or "").strip().splitlines() or [first or f"exit {done.returncode}"])[-1]
        return False, f"it did not start correctly ({tail[:160]})"
    return True, first


def upgrade(dest: Path, source: str, *, installed: Path | None = None, hardware: dict | None = None,
            emit: Callable[[str], None] = print, cancel=None) -> Path | None:
    """The installed DeepSeek engine package in `dest` (engine-dsv41/) is older than the signed feed's `dsv41` member for this PC: fetch the newer one,
    check it as `install` does (signature, hashes, every file), put it NEXT TO the old one, start it once (`--version`), and only then use it (the older
    packages are kept).  Returns the new strata-dsv41, or None: nothing newer, the feed cannot be read (silent: a launch never waits on the
    network), or the newer one could not be installed (one plain line, the installed engine keeps running).  Versions compare numerically
    (0.1.49 < 0.1.49.1). Older packages stay installed until the person deletes them (Settings > Engine).  The caller does not call this for a --dsv41-package or a STRATA_DSV41_EXE."""
    from strata_control import feed as feedmod, signing, transfer, updater

    dest = Path(dest)
    current = installed or find(dest)
    if current is None:
        return None
    if (Path(dest) / ".chosen").is_file() and find(dest) == current:
        return None                                               # the person chose this package in Settings > Engine: a launch does not replace it
    have_name = Path(current).parent.name
    have = version_key(have_name)
    shown = have_name.split("-")[0]
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    try:
        (work / "state").mkdir(parents=True, exist_ok=True)
        ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                              emit=lambda event: None)
        state = updater.State(work)
        try:
            feed = updater.open_feed(ctx, state)
        except feedmod.FeedError as exc:
            if "could not be read" in str(exc) or isinstance(exc, feedmod.FeedUnreachable):
                return None                                       # no network, no feed here: silent, the launch goes on
            emit(f"A newer DeepSeek engine may be available but could not be checked: {exc}; still using {shown}.")
            return None
        except (updater.UpdateError, OSError):
            return None                                           # (offline mode and the like: silent)
        pkg, _reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None or version_key(pkg.version) <= have:
            return None
        target = None
        try:
            archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
            stage_root, _component = updater.stage_engine(ctx, feed, pkg, archive)
            try:
                folder = stage_root / "engine" / updater.folder_name(pkg)
                target = dest / folder.name
                shutil.rmtree(target, ignore_errors=True)
                shutil.move(str(folder), str(target))
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)
            if os.name != "nt":
                (target / exe_name()).chmod(0o755)
            ok, why = _test_start(target / exe_name())
            if not ok:
                raise updater.UpdateError(why)
        except (updater.UpdateError, feedmod.FeedError, signing.SignatureError, transfer.TransferError, OSError) as exc:
            if target is not None and target.name != have_name:
                shutil.rmtree(target, ignore_errors=True)         # (never the installed one)
            emit(f"A newer DeepSeek engine is available but could not be installed: {str(exc).rstrip('.')}; still using {shown}.")
            return None
        emit(f"DeepSeek engine updated from {shown} to {pkg.version}.")
        shutil.rmtree(work, ignore_errors=True)
        return target / exe_name()
    except Exception as exc:                                      # noqa: BLE001 - an upgrade check must never stop a launch
        emit(f"A newer DeepSeek engine may be available but could not be checked ({type(exc).__name__}); still using {shown}.")
        return None
