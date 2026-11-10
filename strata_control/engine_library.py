"""The engines installed on this computer: list them, switch to any of them, delete one. Local only: nothing here downloads.

One truth for the app (Settings > Engine) and the Terminal (`strata components engines | use | remove | rollback`). Updates keep every older
engine (strata_control/updater.py), so a person can go back to an older engine to test something and return to the newest later.

Two families, two mechanisms, one list:

* the Qwen engine (and every model but DeepSeek): packages in ``<components>/engine/<version>-<platform>-<backend>[-<variant>]/``. What a start runs
  is the pinned package (``.updates/state.json`` engine.pinned), else the newest that fits this PC (engine_component.resolve).
* the DeepSeek engine (strata-dsv41): packages in ``engine-dsv41/``. A saved profile's run config names its program; the person's choice is the
  ``.chosen`` file in that folder (dsv41_paths.PIN), which every place that picks a package reads, and the saved configs are pointed at it.

The two never touch each other's files or state: switching or deleting one family leaves the other as it was.

A switch warns, it does not refuse, wherever the engine can run at all. What cannot be switched to is a package this computer cannot run (another
operating system or make of graphics card, a driver that is too old, damaged files, a Strata that is too old for it). The engine in use cannot be
deleted, nor one a running program uses.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from strata_control import dsv41_paths, engine_component as ec, longpath, shape_engine, signing
from strata_control import updater
from strata_control.targets import backend_supported, host_platform

QWEN, DEEPSEEK = "qwen", "deepseek"
FAMILIES = (QWEN, DEEPSEEK)
TITLES = {QWEN: "Qwen and other models", DEEPSEEK: "DeepSeek"}
#: The engine of a Qwen4 model shape (strata_control/shape_engine.py): family "shape:<shape>", one per installed shape, listed after the two above. The family
#: string is an id for the app and the Terminal, never shown: the app titles the group with the model's own catalog title ("title" here is None).
SHAPE_PREFIX = "shape:"


def shape_of_family(family) -> str | None:
    """shape:q4_lite -> q4_lite; None for anything else."""
    if not isinstance(family, str) or not family.startswith(SHAPE_PREFIX):
        return None
    shape = family[len(SHAPE_PREFIX):]
    return shape if shape_engine.valid(shape) else None


def _walk_bytes(folder: Path) -> int:
    total = 0
    try:
        for base, _dirs, files in os.walk(longpath.long_path(folder)):
            for name in files:
                try:
                    total += os.lstat(os.path.join(base, name)).st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _version_of(folder_name: str) -> str:
    return str(folder_name).split("-")[0]


def _label(version: str, backend: str, variant: str = "") -> str:
    return version + (f" ({backend}{'-' + variant if variant else ''})" if backend else "")


# ----------------------------------------------------------------------------------------------- Qwen engine

def _unfit(item: ec.EngineComponent, *, wanted: str | None, target: str, runtime: str | None, cuda_here: list[str], driver_here: str | None,
           hip_here: list[str]) -> tuple[str, str, list[str]] | None:
    """Why this computer cannot run the package, as (plain English, a code the app translates, its values), or None. The same checks as
    engine_component.resolve, one package at a time."""
    if item.platform != target:
        return ("It was made for a different operating system.", "os", [])
    ok, _why = backend_supported(item.backend, target)
    if not ok or (wanted and item.backend != wanted):
        return ("It was made for a different kind of graphics card than this computer has.", "gpu_make", [])
    if item.min_runtime and runtime and ec._version_key(runtime) < ec._version_key(item.min_runtime):
        return (f"It needs Strata {item.min_runtime} or newer; this is {runtime}.", "strata", [item.min_runtime, runtime])
    if item.cuda_archs and cuda_here and not ec.arch_covers(item.cuda_archs, cuda_here):
        return ("It does not include support for this computer's graphics card.", "gpu_arch", [])
    if item.hip_archs and hip_here and not set(item.hip_archs) & set(hip_here):
        return ("It does not include support for this computer's graphics card.", "gpu_arch", [])
    need = item.min_driver.get("nvidia")
    if need and driver_here and ec._version_key(driver_here) < ec._version_key(need):
        return (f"It needs NVIDIA driver {need} or newer; this computer has {driver_here}.", "driver", [need, driver_here])
    try:
        ec.verify(item, full=False)
    except ec.ComponentProblem:
        return ("Some of its files are missing or changed, so it is not safe to start.", "files", [])
    return None


def _process_uses(folder: Path) -> bool:
    try:
        return dsv41_paths.in_use(folder)
    except Exception:                                   # noqa: BLE001 - unknown means not blocked by this check
        return False


def _qwen_entries(ctx: "updater.Context", root: Path | None = None, component_id: str = ec.COMPONENT_ID, family: str = QWEN) -> list[dict]:
    """The Qwen engine's packages; with root, component_id and family those of a model shape's engine (the same package layout and checks)."""
    root = ctx.components if root is None else root
    found, problems = ec.scan(root, component_id)
    wanted = ctx.hardware.get("backend")
    runtime = ctx.hardware.get("runtime") or ec.runtime_version()
    target = ctx.hardware.get("platform") or host_platform()
    cuda_here = ec.cuda_archs_present() if wanted == "cuda" else []
    driver_here = ec.nvidia_driver_version() if wanted == "cuda" else None
    hip_here = ec.hip_archs_present() if wanted == "hip" else []
    try:
        active = ec.resolve(root, backend=wanted, full=False, runtime=ctx.hardware.get("runtime"), component_id=component_id).root.name
    except ec.EngineMissing:
        active = None
    pin = ec.pinned_folder(root)
    rows = []
    for item in found:
        why, code, args = _unfit(item, wanted=wanted, target=target, runtime=runtime, cuda_here=cuda_here, driver_here=driver_here,
                                 hip_here=hip_here) or (None, None, [])
        rows.append({"family": family, "folder": item.root.name, "version": item.version, "backend": item.backend, "platform": item.platform,
                     "variant": item.variant, "bytes": sum(f[2] for f in item.files), "signed": bool(item.signature), "fits": why is None,
                     "why_not": why, "why_code": code, "why_args": args, "active": item.root.name == active, "build_id": item.build_id, "path": longpath.plain(item.root)})
    for bad in problems:
        folder = Path(bad["folder"])
        rows.append({"family": family, "folder": folder.name, "version": _version_of(folder.name), "backend": "", "platform": "", "variant": "",
                     "bytes": _walk_bytes(folder), "signed": False, "fits": False, "active": False, "build_id": "", "path": longpath.plain(folder),
                     "why_not": "Its description is missing or damaged, so it cannot be started.", "why_code": "manifest", "why_args": [],
                     "detail": bad["problem"]})
    fitting = [r for r in rows if r["fits"]]
    newest = max(fitting, key=lambda r: (ec._version_key(r["version"]), r["folder"]), default=None)
    for row in rows:
        row["newest"] = row is newest
        row["pinned"] = row["folder"] == pin
    return rows


# ----------------------------------------------------------------------------------------------- DeepSeek engine

def _ds_roots(ctx: "updater.Context") -> list[Path]:
    try:
        roots, _configs = updater._dsv41_installs(ctx)
    except Exception:                                   # noqa: BLE001
        return []
    return [r for r in roots if Path(r).is_dir()]


def _ds_entries(ctx: "updater.Context") -> list[dict]:
    wanted = ctx.hardware.get("backend")
    target = ctx.hardware.get("platform") or host_platform()
    rows = []
    for root in _ds_roots(ctx):
        exes = dsv41_paths.installed(root)
        by_kind: dict[str, list[Path]] = {}
        for exe in exes:
            by_kind.setdefault(dsv41_paths.kind_of(exe.parent.name), []).append(exe)
        for kind, group in by_kind.items():
            parts = kind.split("-")
            platform = "-".join(parts[:2])
            backend = parts[2] if len(parts) > 2 else ""
            variant = "-".join(parts[3:])
            picked = dsv41_paths.chosen(root, kind)
            fits = (not platform or platform == target) and (not wanted or not backend or backend == wanted)
            code = None if fits else ("os" if platform and platform != target else "gpu_make")
            why = None if fits else ("It was made for a different operating system." if code == "os"
                                     else "It was made for a different kind of graphics card than this computer has.")
            newest = group[-1]
            for exe in group:
                folder = exe.parent
                rows.append({"family": DEEPSEEK, "folder": folder.name, "root": str(root), "version": _version_of(folder.name), "backend": backend,
                             "platform": platform, "variant": variant, "bytes": _walk_bytes(folder), "signed": False, "fits": fits, "why_not": why, "why_code": code, "why_args": [],
                             "active": picked == exe, "newest": exe == newest and fits, "pinned": dsv41_paths.pinned(root) == folder.name,
                             "build_id": "", "path": longpath.plain(folder)})
    return rows


# ----------------------------------------------------------------------------------------------- the list

def _decorate(rows: list[dict], serving: bool) -> None:
    for row in rows:
        busy = _process_uses(Path(row["path"]))
        row["running"] = busy
        if row["active"]:
            row["can_delete"], row["delete_blocked"], row["blocked_code"] = False, "This is the engine in use. Switch to another one first.", "active"
        elif busy:
            row["can_delete"], row["delete_blocked"], row["blocked_code"] = False, "A running program is using it. Stop the model first.", "running"
        else:
            row["can_delete"], row["delete_blocked"], row["blocked_code"] = True, None, None
        row["can_use"] = bool(row["fits"]) and not row["active"]


def list_engines(ctx: "updater.Context") -> dict:
    """Every installed engine package, by family: ``{"families": [{"family", "title", "engines": [...]}], "total_bytes", "serving"}``. A family with
    no package is left out, except the Qwen one, which is always there. Reads files only; nothing is changed or downloaded."""
    try:
        serving = bool(ctx.serving())
    except Exception:                                   # noqa: BLE001
        serving = False
    qwen, deepseek = _qwen_entries(ctx), _ds_entries(ctx)
    shapes = _shape_families(ctx)
    for rows in (qwen, deepseek, *(f["engines"] for f in shapes)):
        _decorate(rows, serving)
        rows.sort(key=lambda r: (ec._version_key(r["version"]), r["folder"]), reverse=True)
    families = [{"family": QWEN, "title": TITLES[QWEN], "engines": qwen}]
    if deepseek:
        families.append({"family": DEEPSEEK, "title": TITLES[DEEPSEEK], "engines": deepseek})
    families += shapes                                  # none on a PC that never had a model of another shape: the list is what it was
    return {"schema_version": 1, "components_dir": str(ctx.components), "families": families,
            "total_bytes": sum(r["bytes"] for fam in families for r in fam["engines"]), "serving": serving}


def _shape_families(ctx: "updater.Context") -> list[dict]:
    """One family per model shape whose engine is installed here (shape_engine.py: <components>/shapes/<shape>/engine/), each with its packages."""
    out = []
    try:
        shapes = shape_engine.installed_shapes(ctx.components)
    except Exception:                                   # noqa: BLE001 - a shapes folder that cannot be read lists nothing, as before
        return []
    for shape in shapes:
        rows = _qwen_entries(ctx, shape_engine.root(shape, ctx.components), shape_engine.component_id(shape), SHAPE_PREFIX + shape)
        if rows:
            out.append({"family": SHAPE_PREFIX + shape, "shape": shape, "title": None, "engines": rows})
    return out


def find_entry(listing: dict, spec: str, family: str | None = None) -> dict:
    """The one entry a person's words name: a folder name, a version, or ``newest``. Ambiguous or unknown is an UpdateError (exit 2) that says what to type."""
    entries = [e for fam in listing["families"] for e in fam["engines"] if family in (None, fam["family"])]
    spec = str(spec).strip()
    if spec.lower() in ("newest", "latest"):
        picks = [e for e in entries if e["newest"]]
    else:
        picks = [e for e in entries if e["folder"] == spec] or [e for e in entries if e["version"] == spec]
    if not picks:
        have = ", ".join(sorted({e["version"] for e in entries})) or "none"
        raise updater.UpdateError(f"no installed engine matches {spec!r} (installed: {have}); `strata components engines` lists them", 2)
    if len(picks) > 1:
        raise updater.UpdateError(f"{spec!r} matches more than one installed engine ({', '.join(p['folder'] for p in picks)}); use the folder name", 2)
    return picks[0]


# ----------------------------------------------------------------------------------------------- switch

class SwitchBusy(updater.Busy):
    """A model is running, so the engine is not switched. One plain answer; the running model keeps the engine it started with."""

    def __init__(self, serving: list[dict]):
        updater.UpdateError.__init__(self, "A model is running, so the engine was not switched. Stop the model first (`strata launch --stop`, or Unload in the app), "
                                           "then switch. The running model keeps the engine it started with. Nothing was changed.", 7)
        self.serving = serving


def use_engine(ctx: "updater.Context", family: str, folder: str, root: str | None = None) -> dict:
    """Make an installed engine the one that starts from now on. Nothing is downloaded or deleted. Waits for / refuses while a model is running
    (`updater.Busy`, exit 7); the running model keeps the engine it started with. The newest engine is selected without a pin, so it follows updates again."""
    shape = shape_of_family(family)
    if family not in FAMILIES and shape is None:
        raise updater.UpdateError("choose the Qwen or the DeepSeek engine", 2)
    if not folder or folder != Path(folder).name or folder.startswith("."):
        raise updater.UpdateError("that is not an installed engine", 2)
    with updater.update_lock(ctx.components):
        try:
            updater.wait_until_idle(ctx.serving, wait=ctx.wait, emit=ctx.emit, cancel=ctx.cancel, poll=ctx.poll)
        except updater.Busy as busy:
            raise SwitchBusy(busy.serving) from None
        if shape is not None:
            return _use_shape(ctx, shape, folder)
        if family == QWEN:
            return _use_qwen(ctx, folder)
        return _use_deepseek(ctx, folder, root)


def _use_shape(ctx: "updater.Context", shape: str, folder: str) -> dict:
    """The switch for a model shape's engine: the same pin as the Qwen engine's (``.updates/state.json`` engine.pinned), in the shape's own folder, which
    engine_component.resolve reads at every start of a model of that shape. The newest package is selected without a pin, so it follows updates again."""
    base, cid, family = shape_engine.root(shape, ctx.components), shape_engine.component_id(shape), SHAPE_PREFIX + shape
    found, _ = ec.scan(base, cid)
    target = next((item for item in found if item.root.name == folder), None)
    if target is None:
        raise updater.UpdateError("that engine is not installed (it may have been deleted already)", 2)
    backend, runtime = ctx.hardware.get("backend"), ctx.hardware.get("runtime")

    def picked() -> str | None:
        try:
            return ec.resolve(base, backend=backend, full=True, runtime=runtime, component_id=cid).root.name
        except ec.EngineMissing:
            return None
    before = picked()
    if before == folder:
        return {"component": cid, "family": family, "status": "already", "changed": False, "to": target.version, "folder": folder}
    try:
        signing.ensure_component_verifier()
        ec.verify_signature(target)
        ec.verify(target, full=True)
    except (ec.ComponentProblem, signing.SignatureError, OSError) as exc:
        raise updater.UpdateError(f"engine {target.version} can no longer be used: {exc}", 5) from None
    state = updater.State(base)
    section = state.section("engine")
    old_pin = section.pop("pinned", None)
    state.save()
    automatic = picked() == folder
    if not automatic:
        section["pinned"] = folder
        state.save()
    if picked() != folder:
        if old_pin:
            section["pinned"] = old_pin
        else:
            section.pop("pinned", None)
        state.save()
        raise updater.UpdateError(f"engine {target.version} can no longer be used: a start would not pick it", 5)
    state.history({"at": (ctx.now or updater.dt.datetime.now(updater.dt.timezone.utc)).isoformat(), "action": "use-shape-engine", "shape": shape,
                   "from": _version_of(before) if before else None, "to": target.version, "ok": True})
    state.save()
    return {"component": cid, "family": family, "status": "switched", "changed": True, "automatic": automatic,
            "from": _version_of(before) if before else None, "to": target.version, "folder": folder}


def _use_qwen(ctx: "updater.Context", folder: str) -> dict:
    return use_qwen_unlocked(ctx, updater.State(ctx.components), folder)


def use_qwen_unlocked(ctx: "updater.Context", state: "updater.State", folder: str) -> dict:
    """The switch itself, for a caller that already holds the update lock (the update that finds its engine already installed)."""
    found, _ = ec.scan(ctx.components)
    target = next((item for item in found if item.root.name == folder), None)
    if target is None:
        raise updater.UpdateError("that engine is not installed (it may have been deleted already)", 2)
    current, _why = updater.installed_engine(ctx.components, ctx.hardware)
    if current is not None and current.root.name == folder:
        return {"component": "engine", "family": QWEN, "status": "already", "changed": False, "to": target.version, "folder": folder}
    try:
        signing.ensure_component_verifier()
        ec.verify_signature(target)
        ec.verify(target, full=True)
    except (ec.ComponentProblem, signing.SignatureError, OSError) as exc:
        raise updater.UpdateError(f"engine {target.version} can no longer be used: {exc}", 5) from None
    out = updater.switch_engine(ctx, state, folder, target, unpin_if_automatic=True)
    return {**out, "family": QWEN, "changed": True, "status": "switched"}


def _use_deepseek(ctx: "updater.Context", folder: str, root: str | None) -> dict:
    roots = _ds_roots(ctx)
    chosen_root = next((r for r in roots if str(r) == root), None) if root else (roots[0] if len(roots) == 1 else None)
    if chosen_root is None:
        raise updater.UpdateError("that DeepSeek engine is not installed here", 2)
    exe = Path(chosen_root) / folder / dsv41_paths.exe_name()
    if not exe.is_file():
        raise updater.UpdateError("that engine is not installed (it may have been deleted already)", 2)
    kind = dsv41_paths.kind_of(folder)
    before = dsv41_paths.chosen(chosen_root, kind)
    if before == exe:
        return {"component": "dsv41", "family": DEEPSEEK, "status": "already", "changed": False, "to": _version_of(folder), "folder": folder}
    ok, why = dsv41_paths.test_start(exe)
    if not ok:
        raise updater.UpdateError(f"DeepSeek engine {_version_of(folder)} can no longer be used: {why}", 5)
    newest = dsv41_paths.installed(chosen_root, kind)[-1]
    automatic = exe == newest
    dsv41_paths.set_pin(chosen_root, None if automatic else folder)
    moved = dsv41_paths.repoint_all(updater._dsv41_installs(ctx)[1], exe)
    state = updater.State(ctx.components)
    state.history({"at": (ctx.now or updater.dt.datetime.now(updater.dt.timezone.utc)).isoformat(), "action": "use-dsv41",
                   "from": _version_of(before.parent.name) if before else None, "to": _version_of(folder), "ok": True})
    state.save()
    return {"component": "dsv41", "family": DEEPSEEK, "status": "switched", "changed": True, "automatic": automatic,
            "from": _version_of(before.parent.name) if before else None, "to": _version_of(folder), "folder": folder, "profiles_repointed": moved}


# ----------------------------------------------------------------------------------------------- delete

def delete_engine(ctx: "updater.Context", family: str, folder: str, root: str | None = None) -> dict:
    """Delete one installed engine package. Refused (exit 2, nothing changed) for the engine in use and for one a running program uses. Models, settings
    and the other family are not touched. The folder is renamed away first, so a package that is held open is left whole, not half removed."""
    shape = shape_of_family(family)
    if family not in FAMILIES and shape is None:
        raise updater.UpdateError("choose the Qwen or the DeepSeek engine", 2)
    if not folder or folder != Path(folder).name or folder.startswith("."):
        raise updater.UpdateError("that is not an installed engine", 2)
    with updater.update_lock(ctx.components):
        listing = list_engines(ctx)
        fam = next((f for f in listing["families"] if f["family"] == family), None)
        rows = [e for e in (fam["engines"] if fam else []) if e["folder"] == folder and (root is None or e.get("root") in (None, root))]
        if not rows:
            raise updater.UpdateError("that engine is not installed (it may have been deleted already)", 2)
        row = rows[0]
        if not row["can_delete"]:
            raise updater.UpdateError(row["delete_blocked"], 2)
        target = Path(row["path"])
        aside = target.with_name(f".deleting-{uuid.uuid4().hex[:8]}")
        try:
            os.replace(longpath.long_path(target), longpath.long_path(aside))
        except OSError as exc:
            raise updater.UpdateError(f"the engine could not be removed because something is using it ({longpath.describe_oserror(exc)}). "
                                      "Close programs that use Strata and try again; nothing was changed.", 6) from None
        shutil.rmtree(longpath.long_path(aside), ignore_errors=True)      # a file that is held stays in .deleting-*; the next delete sweeps it
        _sweep(target.parent)
        if family == QWEN:
            state = updater.State(ctx.components)
            section = state.section("engine")
            if section.get("previous") == folder:
                section["previous"] = None
            if section.get("pinned") == folder:
                section.pop("pinned", None)
            if row["version"] not in {e["version"] for e in rows_of(list_engines(ctx), QWEN)}:
                (section.get("failed") or {}).pop(row["version"], None)
            state.history({"at": (ctx.now or updater.dt.datetime.now(updater.dt.timezone.utc)).isoformat(), "action": "delete-engine",
                           "from": row["version"], "to": None, "ok": True})
            state.save()
        elif shape is not None:
            state = updater.State(shape_engine.root(shape, ctx.components))
            section = state.section("engine")
            if section.get("pinned") == folder:
                section.pop("pinned", None)
            state.history({"at": (ctx.now or updater.dt.datetime.now(updater.dt.timezone.utc)).isoformat(), "action": "delete-shape-engine", "shape": shape,
                           "from": row["version"], "to": None, "ok": True})
            state.save()
        elif dsv41_paths.pinned(Path(row["root"])) is None and (Path(row["root"]) / dsv41_paths.PIN).is_file():
            dsv41_paths.set_pin(Path(row["root"]), None)       # a stale pin
        return {"component": ("engine" if family == QWEN else shape_engine.component_id(shape) if shape is not None else "dsv41"), "family": family, "status": "deleted", "changed": True, "folder": folder,
                "version": row["version"], "freed_bytes": row["bytes"]}


def rows_of(listing: dict, family: str) -> list[dict]:
    return next((f["engines"] for f in listing["families"] if f["family"] == family), [])


def _sweep(engine_folder: Path) -> None:
    """Remove what an earlier delete could not (a file that was held open at the time)."""
    try:
        for entry in engine_folder.iterdir():
            if entry.is_dir() and entry.name.startswith(".deleting-"):
                shutil.rmtree(longpath.long_path(entry), ignore_errors=True)
    except OSError:
        pass
