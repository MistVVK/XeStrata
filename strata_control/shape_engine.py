"""The engine of one Qwen4 model shape, as its own component (feed member ``shape_engines``), picked by the model.

An engine is compiled for ONE model geometry (``-DSTRATA_GEOMETRY=...``, cmake/strata_geometry.cmake) and refuses a GGUF of another one at load. The Strata engine is
built for Flash-Next. A Qwen4 model of a different shape therefore ships with its own engine package, and that package never replaces the Strata engine:

* name: the shape is the compiled geometry's name (``q4_lite``); its component is ``strata-q4-lite``; a feed model of that shape has the family ``qwen4exp_q4_lite``;
* place: ``<components>/shapes/<shape>/engine/<version>-<platform>-<backend>[-<variant>]/``, beside the Strata engine's ``<components>/engine/`` and never inside it. The
  manifest's id is the shape's component (``strata-q4-lite``) and it carries ``"shape"``, so the Strata engine's scan never reads it and an older Control (which reads only
  ``strata-engine``) refuses one that was copied into its engine folder;
* feed: member ``shape_engines`` (the same entry as ``engine``, plus ``shape``). A Control from before this member ignores it, and ignores the models of those shapes too (their
  family is one it does not know);
* use: the run config of a model of that shape says ``"exe": "component:strata-q4-lite"``. Every start asks ``engine_component.expand``, which resolves it in the shape's folder
  (checks every file's SHA-256 and the signature, as for the Strata engine). The Strata engine and every model that runs on it are not touched;
* fetch: setup (the app's Download and the Terminal's launch) fetches the shape's engine through the signed feed when the model is chosen, as it does for DeepSeek's. An explicit
  `strata update` brings an installed shape engine up to date. Starting a model never downloads anything.

Nothing in this module imports anything outside the standard library and ``strata_control`` at load time.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

from strata_control import engine_component as ec

DIR = "shapes"
FAMILY_PREFIX = "qwen4exp_"
SHAPE_RE = re.compile(r"[a-z][a-z0-9_]{1,21}[a-z0-9]")                    # q4_lite, q4_mid: the geometry header's name (include/strata/core/geometries/<name>.hpp)
#: component ids that already mean something else; a shape never becomes one of them
RESERVED = frozenset({"strata-engine", "strata-dsv41", "strata-control", "strata-vision", "strata-device", "strata-terminal", "strata-app", "strata-drafter"})
#: The reason, inside WORDS["start_failed"]'s "...did not start correctly ({why})", when a shape engine cannot say its shape (owner, 11-09)
NO_SHAPE_ANSWER = "it does not say which model shape it was built for"
KEEP_VERSIONS = 2                                                           # the installed shape engine and the one before it
#: What a person reads in the lines below where "{model}" stands: the model's title ("Q4 Lite Shape"), not the geometry's name (q4_lite). One switch: False puts the geometry
#: name back everywhere (the owner decides; `strata components engine` and the JSON always name the shape).
USE_TITLE = True

# ---- the words a person can see: approved by the owner 11-08 ("A ok"), pinned by tests/terminal/test_shape_engine.py ApprovedWords. {model} is the model's title (see USE_TITLE).
# Detail (reasons lists, platform-backend, folders) is in the result rows, the JSON and the log, never in these lines.
WORDS = {
    "checking": "Checking the engine for {model}\u2026",
    "fetching": "{model} needs its own engine. It downloads separately, and the Strata engine stays as it is.",
    "installed": "Engine for {model} {version} ({backend}) is installed.",
    "ready": "Engine for {model} {version} ({backend}) is installed.",
    "updated": "Engine for {model} updated from {old} to {new}.",
    "not_in_feed": "The update feed has no engine for {model} for this computer.",
    "start_failed": "The engine for {model} {version} was not installed because it did not start correctly ({why}). The Strata engine is unchanged.",
    "fetch_failed": "The engine for {model} could not be downloaded: {why}",
    "newer_failed": "A newer engine for {model} is available but could not be installed ({why}). Still using {have}.",
    "missing": ("{model} needs its own engine, which is not installed or does not fit this PC: {reason}. Run `strata launch` again, or Download / repair in the app, "
                "to fetch it. Starting a model never downloads an engine."),
    "current": "The engine for {model} is up to date.",
    # setup.py (the [X] / [ok] marks are setup's)
    "setup_failed": "the engine for {model} could not be installed",
    "setup_failed_hint": "See the line above. Nothing was downloaded for the model. Run this again when the update feed has it.",
    "setup_ok": "engine for {model}: {version} in {folder}",
    "setup_skip": "no engine for {model} is installed: the model is downloaded and prepared now; set it up again when the engine can be fetched",
    "update_label": "Model engine",
}


# ---------------------------------------------------------------------------------------------- names

def shown(shape: str, title: str | None = None) -> str:
    """The name a person reads for the engine of `shape`: the model's title when there is one (and USE_TITLE), else the shape's name."""
    return title.strip() if USE_TITLE and isinstance(title, str) and title.strip() else shape


def title_from_feed(feed, shape: str) -> str | None:
    """The title of the feed's model of `shape` (the first one), for a line of `strata update`, which does not know which model asked."""
    for model in getattr(feed, "models", ()) or ():
        if model.shape == shape and model.title:
            return model.title
    return None


def valid(shape) -> bool:
    """Is `shape` a name this Control accepts for a model shape?"""
    return isinstance(shape, str) and bool(SHAPE_RE.fullmatch(shape)) and "__" not in shape and component_id(shape) not in RESERVED


def component_id(shape: str) -> str:
    """``q4_lite`` -> ``strata-q4-lite``."""
    return "strata-" + str(shape).replace("_", "-")


def shape_of_component(component: str) -> str | None:
    """``strata-q4-lite`` -> ``q4_lite``; None for anything that is not a shape's component (``strata-engine`` included)."""
    if not isinstance(component, str) or not component.startswith("strata-"):
        return None
    shape = component[len("strata-"):].replace("-", "_")
    return shape if valid(shape) and component_id(shape) == component else None


def family_of(shape: str) -> str:
    """The feed model family of a shape: ``q4_lite`` -> ``qwen4exp_q4_lite`` (a family older Controls do not know, so they never offer the model)."""
    return FAMILY_PREFIX + shape


def shape_of_family(family) -> str | None:
    """``qwen4exp_q4_lite`` -> ``q4_lite``; None for ``qwen4exp`` and every other family."""
    if not isinstance(family, str) or not family.startswith(FAMILY_PREFIX):
        return None
    shape = family[len(FAMILY_PREFIX):]
    return shape if valid(shape) else None


def reference(shape: str) -> str:
    """The run config's ``exe`` for a model of this shape: ``component:strata-q4-lite``."""
    return ec.REFERENCE_PREFIX + component_id(shape)


def exe_name() -> str:
    return "strata.exe" if os.name == "nt" else "strata"


# ---------------------------------------------------------------------------------------------- where, what is installed

def shapes_dir(components: Path | None = None) -> Path:
    return (Path(components) if components else ec.components_dir()) / DIR


def root(shape: str, components: Path | None = None) -> Path:
    """The component folder of a shape (it holds ``engine/<package>/``, as the Strata engine's component folder does)."""
    if not valid(shape):
        raise ValueError(f"{shape!r} is not a model shape name")
    return shapes_dir(components) / shape


def installed(shape: str, components: Path | None = None) -> list[ec.EngineComponent]:
    """The engine packages of `shape` that can be read (nothing is hashed), oldest first."""
    found, _ = ec.scan(root(shape, components), component_id(shape))
    return sorted(found, key=lambda item: (ec._version_key(item.version), item.root.name))


def installed_shapes(components: Path | None = None) -> list[str]:
    """The shapes that have at least one readable engine package here."""
    try:
        names = sorted(d.name for d in shapes_dir(components).iterdir() if d.is_dir() and valid(d.name))
    except OSError:
        return []
    return [name for name in names if installed(name, components)]


def newest_version(shape: str, components: Path | None = None) -> str | None:
    found = installed(shape, components)
    return found[-1].version if found else None


def find(shape: str, components: Path | None = None, *, backend: str | None = None, full: bool = False, title: str | None = None) -> ec.EngineComponent:
    """The engine of `shape` for this PC (`ShapeEngineMissing` / `EngineIntegrityError` when there is none that can start)."""
    try:
        return ec.resolve(root(shape, components), backend=backend, full=full, component_id=component_id(shape))
    except ec.EngineIntegrityError:
        raise
    except ec.EngineMissing as exc:
        raise ShapeEngineMissing(shape, exc.reason, root(shape, components), problems=exc.problems, title=title) from None


class ShapeEngineMissing(ec.EngineMissing):
    """No usable engine of this model's shape on this PC. Nothing was fetched; the message says how to get it."""

    code = "shape_engine_missing"

    def __init__(self, shape: str, reason: str, where: Path | None = None, *, problems: list | None = None, title: str | None = None):
        self.shape = shape
        self.reason = reason
        self.root = where
        self.problems = problems or []
        self.plain_notice = None
        RuntimeError.__init__(self, WORDS["missing"].format(model=shown(shape, title), reason=reason))
        self.what = f"the engine for {shown(shape, title)}"
        self.command = "strata launch"


def status(components: Path | None = None) -> list[dict]:
    """The shape engines installed here, for `strata components engine` and the app: one row per shape with its packages. Empty when there are none (a PC that
    never had a model of another shape reports exactly what it did before)."""
    rows = []
    for shape in installed_shapes(components):
        row = {"shape": shape, "component": component_id(shape), "installed": [item.to_dict() for item in installed(shape, components)], "usable": False, "engine": None}
        try:
            chosen = find(shape, components)
            row["engine"], row["usable"] = chosen.to_dict(), True
        except ec.EngineMissing as exc:
            row["notice"] = str(exc)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------- fetching from the signed feed

def _test_start(exe: Path, timeout: float = 120.0, shape: str | None = None) -> tuple[bool, str]:
    """The new engine starts: ``strata --version`` runs, loads its libraries and names itself (it needs no graphics card and no model). The package's own folders go on the
    library path first, as the start path does. With `shape`, ``strata --geometry`` (engine 0.2.13 and later) must name that shape: an engine compiled for another geometry than
    the signed manifest says is refused here, not at its first load, and so is one that cannot answer the flag (NO_SHAPE_ANSWER, owner 11-09)."""
    import subprocess
    env = dict(os.environ)
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    dirs = [d for d in ec.package_lib_dirs(Path(exe).parent) if Path(d).is_dir()]
    if dirs:
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    try:
        done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(Path(exe).parent), env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"it could not be started ({exc})"
    first = ((done.stdout or "").strip().splitlines() or [""])[0]
    if done.returncode != 0 or not first.startswith("strata"):
        tail = ((done.stderr or "").strip().splitlines() or [first or f"exit {done.returncode}"])[-1]
        return False, tail[:160]
    if shape:
        ok, why = check_geometry(exe, shape, env, timeout)
        if not ok:
            return False, why
    return True, first


def check_geometry(exe: Path, shape: str, env: dict | None = None, timeout: float = 60.0) -> tuple[bool, str]:
    """Does ``strata --geometry`` name `shape`? (True, "") when it does; (False, plain reason) when it names another, or names none: every engine of a model shape is built
    after the flag existed (0.2.13, and the Intel engine since 0.2.14), so one that cannot say its shape is refused (owner, 11-09). The Strata engine is never asked."""
    import subprocess
    try:
        done = subprocess.run([str(exe), "--geometry"], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=str(Path(exe).parent), env=env)
    except (OSError, subprocess.SubprocessError):
        return False, NO_SHAPE_ANSWER
    named = ((done.stdout or "").strip().splitlines() or [""])[0].strip()
    if done.returncode != 0 or not named or named.startswith("strata") or " " in named:
        return False, NO_SHAPE_ANSWER                     # an engine that does not know the flag (or answers something else)
    if named != shape:
        return False, f"it was built for a different model shape ({named})"
    return True, ""


def choose(feed, shape: str, hardware: dict, title: str | None = None):
    """The feed's engine of `shape` that fits this PC: (package, reasons) as ``feed.select_engine``; package None when the feed has none for it."""
    from strata_control import feed as feedmod
    mine = [p for p in feed.shape_engines if p.shape == shape]
    if not mine:
        return None, [f"the feed lists no engine for {shown(shape, title)}"]
    return feedmod.select_engine(mine, hardware)


def _prune(shape_root: Path, shape: str, keep_name: str) -> list[str]:
    """Keep the newest `KEEP_VERSIONS` packages of the shape; remove the older ones unless a running engine uses them."""
    from strata_control import dsv41_paths, longpath
    found, _ = ec.scan(shape_root, component_id(shape))
    ordered = sorted(found, key=lambda item: (ec._version_key(item.version), item.root.name), reverse=True)
    removed = []
    for item in ordered[KEEP_VERSIONS:]:
        if item.root.name == keep_name or dsv41_paths.in_use(item.root):
            continue
        shutil.rmtree(longpath.long_path(item.root), ignore_errors=True)
        removed.append(item.root.name)
    return removed


def update_one(ctx, feed, shape: str, *, install_missing: bool = True, title: str | None = None) -> dict:
    """Bring the engine of `shape` up to the feed's: fetch the newest package that fits this PC (resumable download, archive SHA-256, the manifest hash the signed feed names,
    the package signature, every file's hash, the shape named in the manifest), put it NEXT TO the older ones, start it once (`--version`) and only then keep it. The Strata engine is
    never touched. `install_missing` False: nothing is done for a shape that has no engine installed yet (an explicit `strata update` only updates what is there).
    Returns the result row (``status``: installed, updated, current, none-installed, none-for-this-pc)."""
    from strata_control import feed as feedmod, longpath, updater
    comp = component_id(shape)
    name = shown(shape, title or title_from_feed(feed, shape))                   # what a person reads; `shape` and `comp` stay the keys
    have = newest_version(shape, ctx.components)
    result = {"component": comp, "shape": shape, "status": "current" if have else "none-installed", "changed": False, "installed_before": have}
    if not have and not install_missing:
        return result
    pkg, reasons = choose(feed, shape, ctx.hardware, title=name if name != shape else None)
    if pkg is None:
        result.update(status="none-for-this-pc", note=WORDS["not_in_feed"].format(model=name), reasons=reasons)
        return result
    if have and feedmod.version_key(pkg.version) <= feedmod.version_key(have):
        result["note"] = WORDS["current"].format(model=name)
        return result
    place = root(shape, ctx.components)
    updater._free_space_check(place, pkg.size + pkg.installed_bytes, ctx, pkg.sha256, comp)
    archive = updater._download(ctx, comp, pkg.url, pkg.size, pkg.sha256, f"the engine for {name} {pkg.version}", feed)
    stage_root = None
    try:
        try:
            stage_root, _staged = updater.stage_engine(ctx, feed, pkg, archive, component_id=comp)
        except updater.UpdateError as exc:
            if exc.exit_code == 5:
                archive.unlink(missing_ok=True)                 # a refused package is not kept for a retry
            raise
        folder = stage_root / "engine" / updater.folder_name(pkg)
        target = ec.engine_dir(place) / folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(longpath.long_path(target), ignore_errors=True)
        updater._move_dir(folder, target)
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        ctx.say("test", f"Test-starting the engine for {name} {pkg.version}", component=comp)
        ok, why = _test_start(target / exe_name(), shape=shape)
        if not ok:
            shutil.rmtree(longpath.long_path(target), ignore_errors=True)         # (never an older one)
            refused = updater.UpdateError(WORDS["start_failed"].format(model=name, version=pkg.version, why=why), 6)
            refused.refused_why = why                              # ensure() says the approved line as it is (never inside another line)
            raise refused
        removed = _prune(place, shape, target.name)
        archive.unlink(missing_ok=True)
        result.update(status="updated" if have else "installed", changed=True, installed_after=pkg.version, folder=target.name, removed=removed,
                      note=(WORDS["updated"].format(model=name, old=have, new=pkg.version) if have
                            else WORDS["installed"].format(model=name, version=pkg.version, backend=pkg.backend)))
        return result
    finally:
        if stage_root is not None:
            shutil.rmtree(longpath.long_path(stage_root), ignore_errors=True)


def ensure(shape: str, source: str, *, components: Path | None = None, hardware: dict | None = None, emit: Callable[[str], None] = print, cancel=None,
           quiet_when_current: bool = True, title: str | None = None) -> Path | None:
    """The engine of `shape` is installed and up to date, from the signed feed at `source` (a URL or a folder). Returns its program, or None with one plain line through `emit` when
    there is none (not in the feed, nothing for this PC, the feed cannot be read). A newer package is fetched only when the feed can be read; an installed one keeps running when it
    cannot. This is what setup and `strata launch` call when a model of that shape is chosen."""
    from strata_control import acquisition, feed as feedmod, signing, updater
    components = Path(components) if components else ec.components_dir()
    hw = hardware or updater.detect_hardware()
    have = newest_version(shape, components)
    name = shown(shape, title)
    if not have:
        emit(WORDS["fetching"].format(model=name))
    ctx = updater.Context(components=components, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                          feed_timeout=5.0 if have else 30.0,           # an installed engine: a launch must not wait long for a feed that is not there
                          emit=lambda event: emit(event["message"]) if event.get("event") == "phase" and event.get("message") and event.get("phase") != "feed" else None)
    try:
        with updater.update_lock(components):
            state = updater.State(components)
            try:
                feed = updater.open_feed(ctx, state)
            except feedmod.FeedError as exc:
                if have:                                          # offline with an engine installed: the engine runs, nothing to say
                    return find(shape, components, full=False, title=title).exe
                emit(WORDS["fetch_failed"].format(model=name, why=exc))
                return None
            state.save()
            result = update_one(ctx, feed, shape, title=title)
            state.save()
    except Exception as exc:                                      # noqa: BLE001 - with an engine installed, a check for a newer one never stops a launch
        if not isinstance(exc, (updater.UpdateError, feedmod.FeedError, signing.SignatureError, acquisition.AcquisitionRefused, OSError)) and not have:
            raise
        if have:
            if isinstance(exc, (updater.UpdateError, signing.SignatureError)):         # the feed was read and the newer package was refused or did not start: said, the old one runs
                emit(WORDS["newer_failed"].format(model=name, why=getattr(exc, "refused_why", None) or str(exc).rstrip("."), have=have))
            try:
                return find(shape, components, full=False, title=title).exe
            except ec.EngineMissing:
                pass
        emit(str(exc) if getattr(exc, "refused_why", None) else WORDS["fetch_failed"].format(model=name, why=str(exc).rstrip(".")))
        return None
    if result["status"] == "none-for-this-pc":
        emit(result["note"])
        return None
    if result.get("changed") or not quiet_when_current:
        emit(result.get("note") or WORDS["ready"].format(model=name, version=newest_version(shape, components), backend=""))
    try:
        return find(shape, components, full=False, title=title).exe
    except ec.EngineMissing as exc:
        emit(str(exc))
        return None
