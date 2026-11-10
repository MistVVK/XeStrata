"""`strata update`, `strata components update|rollback`: the one explicit step that downloads and installs.

    strata update                 the engine for this PC and Control, from the signed feed; switches only after a test start
    strata update --check         read the feed and say what would change; downloads nothing else
    strata update --app           also download and verify the desktop app's installer
    strata components update      the same as `strata update` (engine + Control)
    strata components rollback    return to the previous engine version the last update kept
    strata components engines     list the engines installed on this computer (sizes, which one is in use)
    strata components use X       switch to an installed engine (a version or folder name; `newest` follows updates again)
    strata components remove X    delete an installed engine (never the one in use)

Starting and serving never come here. A process that has forbidden acquiring (STRATA_NO_ACQUIRE) is refused with exit 5.
Exit codes: 0 done or nothing to do, 2 usage, 5 refused (signature, hash, downgrade, no key, acquiring forbidden),
6 failed and the old version is still in use (network, test start), 7 a model is loaded and nothing was changed, 8 the update is done (or
failed with the old engine in use) but a model that was stopped for it did not start again, 130 stopped.

A model that is loaded (strata_control/engine_update.py): in a terminal the update says which model, asks before stopping it, downloads while it keeps
serving, stops it cleanly, updates, and asks whether to start it again with the same settings. A script is never asked:
    --yes            ask nothing; a loaded model is NOT stopped by it (exit 7, nothing changed, the two next steps are printed)
    --stop-model     the update may stop the loaded model(s) (a script's yes); without --start-again they stay stopped and the start line is printed
    --start-again    start the stopped model(s) again with the same settings after the update (also after a failed update, on the old engine)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

#: Where the feed is read when nothing else is set: the files of the latest GitHub release of Niko1221/Strata (the same folder the
#: one-line installers use). Only a feed signed with a key in update_keys.json (or STRATA_UPDATE_TRUSTED_KEYS) is accepted.
DEFAULT_FEED = "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json"
ENV_FEED = "STRATA_UPDATE_FEED"

ENV_APP_VERSION = "STRATA_APP_VERSION"


def default_feed() -> str:
    """DEFAULT_FEED for this machine: a Linux ARM64 machine (the DGX Spark) reads strata-feed-linux-aarch64.json from the same release."""
    from strata_control import feed as feedmod
    return DEFAULT_FEED.rsplit("/", 1)[0] + "/" + feedmod.feed_name()


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--feed", default=None, help=f"where the signed update feed is (URL or folder). Default: ${ENV_FEED}, the feed used last, or the project's")
    p.add_argument("--components-dir", default=None, help="the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)")
    p.add_argument("--backend", default=None, help="choose an engine family instead of detecting this PC's GPU")
    p.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                   help="if an engine is serving, wait up to this long for it to stop instead of refusing")
    p.add_argument("--force", action="store_true", help="try a version again that failed its test start or was rolled back from")
    p.add_argument("--model-config", default=None, metavar="RUN_CONFIG.json",
                   help="the run config of the model to use for the test start (default: an installed model)")
    p.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def add_parsers(sp, components_sp) -> list:
    update = sp.add_parser("update", help="download and install the engine for this PC (and Control) from the signed feed; switches only after a test start")
    _common(update)
    update.add_argument("--check", action="store_true", help="only read the feed and say what would change")
    update.add_argument("--app", action="store_true", help="also download and verify the desktop app's installer")
    update.add_argument("--app-version", default=None, help=f"the installed app's version (default: ${ENV_APP_VERSION})")
    update.add_argument("--no-engine", action="store_true")
    update.add_argument("--no-control", action="store_true")
    update.add_argument("--yes", action="store_true", help="ask nothing (a script); a loaded model is not stopped by this alone, see --stop-model")
    update.add_argument("--stop-model", action="store_true", help="let the update stop a loaded model first (the script's yes to the question a terminal asks)")
    update.add_argument("--start-again", action="store_true", help="after the update, start the model(s) that were stopped for it again with the same settings")
    comp_update = components_sp.add_parser("update", help="the same as `strata update` (engine and Control)")
    _common(comp_update)
    comp_update.add_argument("--check", action="store_true")
    comp_update.add_argument("--app", action="store_true")
    comp_update.add_argument("--app-version", default=None)
    comp_update.add_argument("--no-engine", action="store_true")
    comp_update.add_argument("--no-control", action="store_true")
    comp_update.add_argument("--yes", action="store_true", help="ask nothing (a script); a loaded model is not stopped by this alone, see --stop-model")
    comp_update.add_argument("--stop-model", action="store_true", help="let the update stop a loaded model first (the script's yes to the question a terminal asks)")
    comp_update.add_argument("--start-again", action="store_true", help="after the update, start the model(s) that were stopped for it again with the same settings")
    rollback = components_sp.add_parser("rollback", help="return to the previous engine version the last update kept")
    _common(rollback)
    engines = components_sp.add_parser("engines", help="list the engines installed on this computer and which one is in use")
    _common(engines)
    use = components_sp.add_parser("use", help="switch to an installed engine (nothing is downloaded); `newest` goes back to following updates")
    use.add_argument("engine", help="a version (0.1.54), a folder name, or newest")
    use.add_argument("--family", choices=_ENGINE_FAMILIES, default=None, help="which engine family when both have that version (default: either)")
    _common(use)
    remove = components_sp.add_parser("remove", help="delete an installed engine (never the one in use); models and settings are not touched")
    remove.add_argument("engine", help="a version (0.1.54) or a folder name")
    remove.add_argument("--family", choices=_ENGINE_FAMILIES, default=None)
    _common(remove)
    install_file = components_sp.add_parser("install-file", help="install an engine package you downloaded yourself (no network; the package's signature is checked)")
    install_file.add_argument("file", help="the engine package (.tar.gz or .zip) from the release page")
    _common(install_file)
    return [update, comp_update, rollback, install_file, engines, use, remove]


class _EngineFamilies(tuple):
    """The --family values of `strata components use | remove`: qwen, deepseek, and the engine of a Qwen4 model shape as shape:<shape> (what the app
    passes, strata_control/engine_library.py SHAPE_PREFIX). Help and error texts list the two as before."""

    def __contains__(self, value) -> bool:
        if tuple.__contains__(self, value):
            return True
        from strata_control import engine_library
        return engine_library.shape_of_family(value) is not None


_ENGINE_FAMILIES = _EngineFamilies(("qwen", "deepseek"))


def _emit_factory(a, lines: list):
    machine = a.output in ("json", "jsonl")
    state = {"pct": -10}

    def emit(event: dict) -> None:
        if a.output == "jsonl":
            print(json.dumps({"schema_version": 1, **event}, ensure_ascii=False), flush=True)
            return
        if machine:
            return
        if event.get("event") == "progress":
            pct = event.get("percent", 0)
            if sys.stdout.isatty():
                print(f"\r  downloading {event.get('label', '')}: {pct}%  ({event['done'] // 1_000_000} of {event['total'] // 1_000_000} MB)   ",
                      end="" if pct < 100 else "\n", flush=True)
            elif pct >= state["pct"] + 10 or pct == 100:
                state["pct"] = pct
                print(f"  downloading {event.get('label', '')}: {pct}%", flush=True)
        elif event.get("event") == "phase" and event.get("phase") != "download":
            print(f"  {event['message']}", flush=True)
        elif event.get("event") == "waiting":
            print(f"  {event['message']}", flush=True)
    return emit


def _feed_source(a, root: Path) -> str:
    from strata_control import updater
    if a.feed:
        return a.feed
    if os.environ.get(ENV_FEED):
        return os.environ[ENV_FEED]
    remembered = (updater.State(root).data.get("feed") or {}).get("source")
    if remembered == DEFAULT_FEED:                    # the common feed remembered from before: a Linux ARM64 machine reads its own file
        return default_feed()
    return remembered if _may_be_remembered(remembered) else default_feed()


def _may_be_remembered(source) -> bool:
    """Only an https address is carried over to later runs. A loopback address or a folder is what a test of an update feed uses; once
    one such run had been made on a PC, every later check (the app's Settings included) tried that dead address and said "No internet
    connection". Those runs name their feed again each time (--feed or STRATA_UPDATE_FEED)."""
    return isinstance(source, str) and source.lower().startswith("https://")


def _source_version() -> str | None:
    """What a source checkout runs, for the line `Control  <version>: <status>` when no Control is installed (display only)."""
    try:
        from strata_control import updater
        return updater.source_tree_version()
    except Exception:                                        # noqa: BLE001 - a missing figure is printed as '?'
        return None


def _mib(n: int | None) -> str:
    from strata_control.updater import megabytes
    return f"{megabytes(n)} MB"


def _line(name: str, result: dict) -> str:
    status = result.get("status")
    if result.get("component") == "engine":
        before = (result.get("installed_before") or {}).get("version")
        after = (result.get("installed_after") or {}).get("version")
        if status == "updated":
            test = result.get("test_start") or {}
            detail = "tested: started, answered one token, stopped" if test.get("mode") == "full" else f"tested: {test.get('detail', 'engine ran')}"
            if result.get("from_disk"):
                detail = "already installed, nothing downloaded"
            return f"{name:<8}{before or 'none'} -> {after}   updated ({detail})"
        if status == "current":
            return f"{name:<8}{before}   up to date"
        if status == "failed":
            return f"{name:<8}{before or 'none'}   NOT changed: {result.get('error')}"
        return f"{name:<8}{before or 'none'}   {result.get('note') or status}"
    if result.get("component") == "dsv41":
        if status == "none-installed":
            return f"{name:<8}no DeepSeek engine installed"
        before, after = result.get("installed_before"), result.get("installed_after")
        if status == "updated":
            return f"{name:<8}{before} -> {after}   updated (tested: it starts). {result.get('note') or ''}".rstrip()
        if status == "current":
            return f"{name:<8}{before}   up to date"
        if status == "failed":
            return f"{name:<8}{before or '?'}   NOT changed: {result.get('error')}"
        return f"{name:<8}{before or '?'}   {result.get('note') or status}"
    if result.get("shape"):                              # the engine of a model shape (strata_control/shape_engine.py)
        from strata_control import shape_engine
        if status == "failed":
            return f"{shape_engine.WORDS['update_label']}  {result.get('installed_before') or '?'}   NOT changed: {result.get('error')}"
        return f"{shape_engine.WORDS['update_label']}  {result.get('note') or status}"
    if result.get("component") == "control":
        if status == "updated":
            return f"{name:<8}{result.get('installed_before') or '?'} -> {result.get('installed_after')}   updated. The next `strata` command already runs it; the app switches on its next start"
        if status == "current":
            return f"{name:<8}{result.get('installed_before')}   up to date"
        if status == "not-in-feed":
            return f"{name:<8}not part of this feed"
        if status == "failed":
            return f"{name:<8}{result.get('installed_before') or '?'}   NOT changed: {result.get('error')}"
        return f"{name:<8}{status}"
    if status == "downloaded":
        return f"{name:<8}{result.get('version')}   downloaded and verified: {result.get('path')}\n{'':<8}{result.get('note')}"
    if status == "failed":
        return f"{name:<8}NOT downloaded: {result.get('error')}"
    return f"{name:<8}{status}"


def _print_check(report: dict) -> None:
    feed, eng, ctl, app = report["feed"], report["engine"], report["control"], report["app"]
    print(f"Update feed  sequence {feed['sequence']}, {feed['channel']} channel, signed by {feed['key_id']} ({feed['trusted_by']})")
    installed = (eng["installed"] or {}).get("version")
    offered = (eng["offered"] or {}).get("version")
    text = {"available": (f"update available: {installed} -> {offered} (already installed; the update switches to it, nothing to download)"
                          if eng.get("installed_locally") else f"update available: {installed} -> {offered} ({_mib(eng['download_bytes'])})"),
            "install": f"no engine installed yet; {offered} is the one for this PC ({_mib(eng['download_bytes'])})",
            "current": f"{installed} is up to date", "none-for-this-pc": eng.get("note") or "the feed has no engine for this PC", "wrong-make": eng.get("note") or "nothing was changed",
            "not-in-feed": "the feed lists no engines"}.get(eng["status"], eng.get("note") or eng["status"])
    print(f"  Engine   {text}")
    pick = eng.get("offered")
    if pick and eng["status"] in ("available", "install", "current"):
        # round 8, N9: the list below names only the packages that were NOT chosen; say which one was
        label = pick.get("label") or (f"{pick.get('version')} {pick.get('platform')}-{pick.get('backend')}" + (f"-{pick['variant']}" if pick.get("variant") else ""))
        print(f"           selected for this PC: {label}")
    if eng.get("reasons"):
        print("           not used for this PC:")
    for reason in eng.get("reasons") or []:
        print(f"           - {reason}")
    if eng.get("note") and eng["status"] not in ("failed-before", "skipped", "current"):
        print(f"           {eng['note']}")
    if ctl["status"] == "not-in-feed":
        print("  Control  not part of this feed")
    else:
        print(f"  Control  {ctl['installed'] or _source_version() or '?'}: {ctl['status']}" + (f" ({ctl['offered']})" if ctl["offered"] else ""))
    ds = report.get("dsv41") or {}
    if ds.get("status") not in (None, "none-installed"):
        print(f"  DeepSeek {ds.get('installed')}: " + (f"update available -> {ds['offered']} ({_mib(ds['download_bytes'])})" if ds["status"] == "available" else ds["status"]))
    if app["status"] != "not-in-feed":
        if app["status"] == "unknown-installed":
            print(f"  App      latest is {app['offered']}; this terminal cannot tell which version of the app is installed (the app checks itself)")
        else:
            print(f"  App      {app['installed'] or 'unknown'} -> {app['offered']}: {app['status']}")
    if report.get("serving"):
        from strata_control import engine_update
        names = ", ".join(m.label for m in engine_update.running_models(report["serving"]))
        print(f"  {names} is running. An update asks before it stops it, downloads while it keeps serving, and asks whether to start it again.")
    if (eng.get("status") in ("available", "install") or ds.get("status") == "available") and not report.get("serving"):
        print("  To update: strata update")


def source_store(checkout: Path | None = None) -> Path | None:
    """The engine store a source install made for itself: `<checkout>/engine-feed` (setup.py fetches its engine there from the signed feed), or None
    when this is not a checkout that has one with an engine package in it."""
    here = checkout if checkout is not None else Path(__file__).resolve().parents[1]
    store = here / "engine-feed"
    try:
        from strata_control import engine_component as ec
        found, _ = ec.scan(store) if store.is_dir() else ([], [])
    except Exception:                                    # noqa: BLE001 - a store that cannot be read is not one
        return None
    return store if found else None


def default_components_dir(ec) -> Path:
    """The folder `strata update` works in when none is named. STRATA_COMPONENTS_DIR and --components-dir win (ec.components_dir reads the first). A source
    checkout that has its own engine store, and whose per-user folder holds no engine package, updates the checkout's: running `strata update` there used
    to download into the per-user folder instead, leave a half-written state.json and a part file nobody resumed, and then download everything again
    when the right folder was named (#4463)."""
    chosen = ec.components_dir()
    if os.environ.get(ec.ENV_DIR):
        return chosen
    own = source_store()
    if own is None:
        return chosen
    try:
        found, _ = ec.scan(chosen) if chosen.is_dir() else ([], [])
    except Exception:                                    # noqa: BLE001
        found = []
    return chosen if found else own


def run(a) -> int:
    from strata_control import acquisition, engine_component as ec, feed as feedmod, signing, updater
    machine = a.output in ("json", "jsonl")

    def fail(message: str, code: int) -> int:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
        else:
            print(f"strata: {message}", file=sys.stderr)
        return code

    root = Path(a.components_dir) if a.components_dir else default_components_dir(ec)
    try:
        hardware = updater.detect_hardware()
        if a.backend:
            hardware["backend"], variant = ec.split_backend(a.backend)    # --backend hip-rdna2 = the hip family, the rdna2 package (#4013)
            if variant:
                hardware["variant_wanted"] = variant
            hardware["backend_source"] = "requested"             # the person's own decision: the make guard lets it through
        lines: list = []
        cancel = threading.Event()
        ctx = updater.Context(components=root, source="", trusted={}, hardware=hardware, emit=_emit_factory(a, lines), cancel=cancel,
                              force=getattr(a, "force", False), run_config=a.model_config, wait=a.wait or 0.0,
                              app_version=getattr(a, "app_version", None) or os.environ.get(ENV_APP_VERSION))
        if getattr(a, "components_command", None) == "rollback":
            ctx.serving = updater.serving_now
            _maybe_wait(a, ctx, machine)
            out = updater.rollback_engine(ctx)
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **out}))
            else:
                forward = out["status"] == "rolled-forward"
                print(f"Engine moved to the newer version: {out['from']} -> {out['to']}." if forward else
                      f"Engine rolled back: {out['from']} -> {out['to']}.")
                print("Saved tuning presets are marked for the short re-check; none were deleted." if out["presets_marked"] else
                      "Saved tuning presets are unchanged.")
                print("Run `strata update --force` to try the newer version again, or `strata components rollback` to return to it." if not forward else
                      "Run `strata components rollback` again to go back to the older one.")
            return 0
        if getattr(a, "components_command", None) in ("engines", "use", "remove"):
            return _engines_command(a, ctx, machine, fail)
        if getattr(a, "components_command", None) == "install-file":
            ctx.trusted = signing.load_trusted_keys()
            ctx.serving = updater.serving_now
            _maybe_wait(a, ctx, machine)
            out = updater.install_engine_file(ctx, Path(a.file))
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **out}, ensure_ascii=False))
            else:
                print(out.get("note") or _line("Engine", out))
            return 0
        acquisition.require("the update", "`strata update` from a terminal")
        trusted = signing.load_trusted_keys()
        ctx.trusted = trusted
        ctx.source = _feed_source(a, root)
        if signing.extra_keys_in_effect() and not machine:
            print(f"note: trusting the extra keys in {os.environ.get(signing.ENV_EXTRA_KEYS)} (a test setup)", file=sys.stderr)
        if a.check:
            report = updater.check(ctx)
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, **report}, ensure_ascii=False))
            else:
                _print_check(report)
            return 0
        if not machine:
            print(f"Strata update ({ctx.source})")
        from strata_control import engine_update
        out = engine_update.update_with_models(ctx, _gate_for(a, ctx, machine), show=None if machine else _show_results,
                                               engine=not a.no_engine, control=not a.no_control, app=a.app)
        if machine:
            print(json.dumps({"schema_version": 1, "ok": out["exit_code"] == 0, **out}, ensure_ascii=False))
        elif out["exit_code"] == engine_update.EXIT_NOT_RESTARTED:
            print("The update is done. A model that was stopped for it did not start again (see above); the new engine is in use.")
        return out["exit_code"]
    except updater.UpdateError as exc:
        if exc.exit_code == 0:                           # the person said no to stopping the model: nothing was changed
            print(str(exc))
            return 0
        return fail(str(exc), exc.exit_code)
    except feedmod.FeedError as exc:
        return fail(str(exc), exc.exit_code)
    except signing.SignatureError as exc:
        return fail(str(exc), 5)
    except acquisition.AcquisitionRefused as exc:
        return fail(f"{exc} (the update is the one explicit step that downloads; run it from a shell where STRATA_NO_ACQUIRE is not set)", 5)
    except KeyboardInterrupt:
        return fail("stopped; a partial download is kept and the next update resumes it", 130)


def _show_results(out: dict) -> None:
    print(f"Feed sequence {out['feed']['sequence']}, signed by {out['feed']['key_id']}")
    for result in out["results"]:
        print(_line(result["component"].capitalize(), result))
    if any(r.get("presets_marked") for r in out["results"]):
        print("Saved tuning presets: marked for the short re-check (nothing was deleted).")
    kept = next((r.get("previous") for r in out["results"] if r.get("status") == "updated" and r["component"] == "engine"), None)
    if kept:                                             # a first install has no earlier engine to keep
        print("The previous engine version is kept; `strata components rollback` returns to it.")


def _ask_factory(a, machine: bool):
    """The question a person at a terminal is asked, or None when nobody can be: output for a program, --yes, or no keyboard."""
    if machine or getattr(a, "yes", False) or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    def ask(question: str, default: bool) -> bool:
        try:
            answer = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        except EOFError:
            return default
        return default if not answer else answer in ("y", "yes")
    return ask


def _gate_for(a, ctx, machine: bool):
    from strata_control import engine_update
    return engine_update.ModelGate(ask=_ask_factory(a, machine), say=(lambda text: None) if machine else (lambda text: print(text, flush=True)),
                                   emit=ctx.emit, allow_stop=bool(getattr(a, "stop_model", False)), start_again=bool(getattr(a, "start_again", False)))


def _gb(n: int | None) -> str:
    n = n or 0
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{max(1, round(n / 1e6))} MB"


def _engines_command(a, ctx, machine: bool, fail) -> int:
    """`strata components engines | use | remove`: the same list and the same switch / delete the app's Settings > Engine uses (strata_control/engine_library.py)."""
    from strata_control import engine_library as lib, updater
    ctx.serving = updater.serving_now
    command = a.components_command
    if command == "engines":
        listing = lib.list_engines(ctx)
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **listing}, ensure_ascii=False))
            return 0
        print(f"Engines on this computer ({_gb(listing['total_bytes'])} in total)")
        for family in listing["families"]:
            print(f"\n{family['title'] or family.get('shape') or family['family']}")      # a model shape's engine: its shape's name, as shape_engine.py names it here
            for e in family["engines"]:
                tags = ["in use" if e["active"] else "", "newest" if e["newest"] else "", "" if e["fits"] else "not for this computer"]
                print(f"  {e['version']:<12}{_gb(e['bytes']):>9}  {e['backend'] + ('-' + e['variant'] if e['variant'] else ''):<14}{', '.join(t for t in tags if t)}")
                if e["why_not"] and not e["fits"]:
                    print(f"      {e['why_not']}")
        print("\n`strata components use <version>` switches, `strata components remove <version>` deletes. Nothing was changed.")
        return 0
    entry = lib.find_entry(lib.list_engines(ctx), a.engine, a.family)
    if command == "use":
        _maybe_wait(a, ctx, machine)
        if not entry["fits"]:
            raise updater.UpdateError(f"engine {entry['version']} cannot run on this computer: {entry['why_not']}", 5)
        out = lib.use_engine(ctx, entry["family"], entry["folder"], entry.get("root"))
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **out}, ensure_ascii=False))
        elif out["status"] == "already":
            print(f"Engine {out['to']} is already in use. Nothing was changed.")
        else:
            print(f"Engine {out['to']} is now in use (was {out.get('from') or 'none'}). Models you start from now on run on it.")
            print("Saved tuning presets are marked for the short re-check; none were deleted." if out.get("presets_marked") else
                  "Saved tuning presets are unchanged.")
            print("`strata update` makes the newest engine the one in use again." if not out.get("automatic") else
                  "This is the newest engine, so it follows updates again.")
        return 0
    out = lib.delete_engine(ctx, entry["family"], entry["folder"], entry.get("root"))
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, **out}, ensure_ascii=False))
    else:
        print(f"Engine {out['version']} deleted ({_gb(out['freed_bytes'])} freed). Models and settings were not touched.")
    return 0


def _maybe_wait(a, ctx, machine: bool) -> None:
    """If an engine is serving and no --wait was given, a terminal user is asked whether to wait; a script is told no."""
    if ctx.wait:
        return
    from strata_control import updater
    serving = updater.serving_now()
    if not serving:
        return
    names = ", ".join(item.get("profile", "an engine") for item in serving)
    if any(str(item.get("profile", "")).startswith("launch-") for item in serving):
        return                                   # Strata's own launch: run_update says the one next step (stop the model first) instead of asking
    if machine or not sys.stdin.isatty():
        return                                   # run_update raises Busy (exit 7) with the same names
    answer = input(f"{names} is serving. Wait for it to stop and then update? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        ctx.wait = 6 * 3600.0
