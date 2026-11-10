"""An engine update beside a loaded model: the notice, the warning, the clean stop and the start again.

The rule (owner, 11-01): an engine update never starts by itself while a model is loaded, and nothing replaces or removes the engine a running
model uses.  An update that finds a model loaded does one of three things, and never a fourth:

* nobody can be asked (a script, the app without a dialog, a service): it stops BEFORE any download and changes nothing (exit 7), and says
  the two next steps;
* the person said yes (the question in a terminal, ``--stop-model`` in a script): the new engine is downloaded while the model keeps
  serving, then the model is stopped cleanly, the update is made, and the model is started again with the same settings when the person says so
  (the question in a terminal, ``--start-again`` in a script; otherwise it stays stopped and the way to start it is printed);
* the person said no: nothing is changed (exit 0).

``ModelGate`` is the one object the update (strata_control/updater.py) calls at its two points: ``consent`` before anything is downloaded, and
``release`` after the download, right before the first thing is staged or test-started.  ``update_with_models`` runs a whole update around it
and starts the models again afterwards, also when the update failed (the old engine is still in use).

``notice`` is the one quiet line the Terminal shows when the last check found a newer engine.  It reads the cached check only (a local
file) and never touches the network, so showing it breaks no rule.  ``refresh_cache`` is the daily check, written but not called by anything:
adding it to a screen is one line, and waits for the owner's yes (the Terminal learns of a new engine only from a check somebody made).
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from strata_control import engine_component as ec, updater

EXIT_BUSY = 7                                                       # a loaded model, nothing was changed (the updater's own code)
EXIT_NOT_RESTARTED = 8                                              # the update is done (or failed and the old engine is in use) but a model did not start again
NEXT_STEP = "`strata update`"


# ---------------------------------------------------------------------------------------------------- the notice

def _installed_words(root) -> dict:
    return {"engine": ec.newest_installed_version(root), "dsv41": ec._installed_dsv41_version()}


def notice(root=None) -> dict | None:
    """What the last check found newer, as {"engine": v|None, "dsv41": v|None, "have": {...}, "line": plain sentence}, or None.  Local files only."""
    offers = ec.cached_offers(root)
    engine, dsv = offers.get("engine"), offers.get("dsv41")
    if not engine and not dsv:
        return None
    have = _installed_words(root)
    parts = []
    if engine:
        parts.append(f"engine {engine}" + (f" (you have {have['engine']})" if have["engine"] else ""))
    if dsv:
        parts.append(f"DeepSeek engine {dsv}" + (f" (you have {have['dsv41']})" if have["dsv41"] else ""))
    head = "A new " + parts[0] + " is available" if len(parts) == 1 else "New engines are available: " + ", ".join(parts)
    return {"engine": engine, "dsv41": dsv, "have": have, "line": head}


def notice_line(root=None, how: str = f"Update with {NEXT_STEP}.") -> str | None:
    """The single quiet line: what is new and the one next step.  ``how`` is the screen's own way (the monitor says "Press u or run ...")."""
    found = notice(root)
    return None if found is None else f"{found['line']}. {how}"


ATTEMPT_FILE = "notice-attempt.json"
CHECK_TIMEOUT_S = 5.0                    # one read of the feed waits this long at most (twice: the transfer layer tries a web address once more)
CHECK_ENV = "STRATA_UPDATE_CHECK"        # "0" / "off": no daily check (a tester's or a privacy switch; the default is on in the launch monitor only)


def _hours_since(text) -> float | None:
    try:
        return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(str(text))).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def refresh_cache(check: Callable[[], object], root=None, *, max_age_hours: float = 24.0) -> bool:
    """The daily check, quiet by design: run ``check()`` (updater.check: it reads the signed feed, writes the cache, downloads no engine) when neither
    the last check nor the last ATTEMPT is younger than ``max_age_hours``.  The attempt is written BEFORE the network is touched, so a PC with no
    internet, a blocked DNS, a proxy, a captive portal that answers junk or a connection that never answers is asked once a day and not again until
    tomorrow.  Nothing is printed or logged, whatever happens.  True when it ran."""
    import os
    if os.environ.get(CHECK_ENV, "").strip().lower() in ("0", "off", "no", "false", "never"):
        return False
    base = Path(root) if root else ec.components_dir()
    try:
        last = json.loads((base / ec.UPDATE_STATE).read_text(encoding="utf-8-sig")).get("last_check", {}).get("at")
    except (OSError, ValueError, AttributeError):
        last = None
    try:
        tried = json.loads((base / updater.STATE_DIR / ATTEMPT_FILE).read_text(encoding="utf-8")).get("at")
    except (OSError, ValueError, AttributeError):
        tried = None
    for stamp in (last, tried):
        age = _hours_since(stamp)
        if age is not None and age < max_age_hours:
            return False
    try:
        folder = base / updater.STATE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ATTEMPT_FILE).write_text(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat()}), encoding="utf-8")
    except OSError:
        return False                                     # cannot remember the attempt: do not risk asking again and again
    try:
        check()
    except BaseException:                                # noqa: BLE001 - quiet: no network, no notice
        return False
    return True


def start_daily_check(on_done: Callable[[], None] | None = None, root=None, *, feed: str | None = None) -> "threading.Thread | None":
    """Run the daily check in a thread of its own (a daemon: nothing ever waits for it, and a connection that never answers is simply left behind
    when the program ends).  ``on_done`` is called when a check finished well.  Used by the launch monitor only."""
    import threading
    from strata_control import acquisition
    if acquisition.is_forbidden():
        return None

    def work() -> None:
        try:
            def check():
                import argparse
                from strata_cli import update_cmd
                from strata_control import signing
                base = Path(root) if root else ec.components_dir()
                ctx = updater.Context(components=base, source=update_cmd._feed_source(argparse.Namespace(feed=feed), base), trusted=signing.load_trusted_keys(),
                                      hardware=updater.detect_hardware(), feed_timeout=CHECK_TIMEOUT_S, serving=lambda: [])
                return updater.check(ctx)
            if refresh_cache(check, root) and on_done is not None:
                on_done()
        except BaseException:                            # noqa: BLE001 - never a message, never a trace
            pass
    thread = threading.Thread(target=work, name="strata-daily-check", daemon=True)
    thread.start()
    return thread


def monitor_note(root=None) -> str | None:
    """The short form of the notice for the monitor's one message line."""
    found = notice(root)
    if found is None:
        return None
    if found["engine"] and found["dsv41"]:
        return "New engines are available. Press u to update."
    return (f"New engine {found['engine']} is available. Press u to update." if found["engine"]
            else f"New DeepSeek engine {found['dsv41']} is available. Press u to update.")


# ---------------------------------------------------------------------------------------------------- models that are loaded

@dataclass
class RunningModel:
    profile: str
    name: str
    state: str = "ready"
    port: int | None = None
    kind: str = "launch"                 # launch (strata launch), profile (the app's or a saved profile), server (a Strata server nobody here started)
    can_stop: bool = True
    choices: dict | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.name + (f" (port {self.port})" if self.port else "")

    def to_dict(self) -> dict:
        return {"profile": self.profile, "name": self.name, "port": self.port, "kind": self.kind, "state": self.state}


def _launch_dir() -> Path | None:
    try:
        from strata_cli.launch.runner import launch_dir
        return launch_dir()
    except Exception:                                    # noqa: BLE001
        return None


def _saved_choices(profile: str) -> dict | None:
    folder = _launch_dir()
    if folder is None or not profile.startswith("launch-"):
        return None
    try:
        return json.loads((folder / f"{profile[7:]}.choices.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def running_models(serving: list[dict] | None = None) -> list[RunningModel]:
    """The models that are loaded, named in words a person knows.  ``serving`` is updater.serving_now()'s answer (called here when not given)."""
    from strata_control import profiles
    items = updater.serving_now() if serving is None else serving
    out = []
    for item in items:
        pid = str(item.get("profile", ""))
        state = str(item.get("state", "ready"))
        if not profiles.PROFILE_ID.fullmatch(pid):       # "a running Strata server": the shared lease, held by a process that is not Strata's own launch
            out.append(RunningModel(pid or "a running Strata server", pid or "A Strata server", state, None, "server", can_stop=False))
            continue
        name, port = pid[7:] if pid.startswith("launch-") else pid, None
        try:
            prof, _rev = profiles.load(pid)
            port = int((prof.get("overrides") or {}).get("port") or 0) or None
            cfg = json.loads(Path(prof["source_config"]).read_text(encoding="utf-8-sig")) if prof.get("source_config") else {}
            name = str(cfg.get("model_name") or name)
        except Exception:                                # noqa: BLE001 - the name is a nicety
            pass
        out.append(RunningModel(pid, name, state, port, "launch" if pid.startswith("launch-") else "profile", True, _saved_choices(pid)))
    return out


def _names(models: list[RunningModel]) -> str:
    labels = [m.label for m in models]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def stop_model(model: RunningModel, timeout: float = 180) -> None:
    from strata_control import runtime
    runtime.stop(model.profile, timeout=timeout)
    try:
        from strata_cli.launch import access
        access.stop_tunnel(model.profile)
    except Exception:                                    # noqa: BLE001 - no tunnel, or none to stop
        pass


def _start_wait(model: RunningModel) -> float:
    try:
        from strata_control import profiles
        prof, _rev = profiles.load(model.profile)
        cfg = json.loads(Path(prof["source_config"]).read_text(encoding="utf-8-sig"))
        return min(1800.0, float(cfg.get("engine_ready_s") or 900) + 60)          # (runtime.start takes 0..1800; a DeepSeek config asks for more)
    except Exception:                                    # noqa: BLE001
        return 960.0


def start_model(model: RunningModel) -> tuple[bool, str]:
    """Start the saved profile again: the same run config and overrides, so the same settings.  (ok, a plain sentence)."""
    from strata_control import runtime
    cfg_path = None
    try:
        from strata_control import profiles
        cfg_path = Path(profiles.load(model.profile)[0]["source_config"])
    except Exception:                                    # noqa: BLE001
        pass
    why = ""
    try:
        result = runtime.start(model.profile, wait=_start_wait(model))
        if result.get("ok"):
            if (model.choices or {}).get("access") == "cloudflare":
                try:
                    from strata_cli.launch import access
                    access.start_tunnel(model.profile, int(model.port or 8080))
                except Exception:                        # noqa: BLE001 - the model serves on this computer; the tunnel is said not to be back
                    return True, f"{model.label} is serving again; its Cloudflare tunnel did not open, start it again with `strata launch`."
            return True, f"{model.label} is serving again with the same settings."
        try:
            from strata_cli.launch.runner import explain_engine_exit
            why = explain_engine_exit(result, cfg_path)[0]
        except Exception:                                # noqa: BLE001
            why = str(result.get("error") or result.get("next_action") or result.get("state") or "it stopped at its start")
    except Exception as exc:                             # noqa: BLE001 - a refused or failed start is a sentence, never a trace
        why = str(exc)
    for call in (lambda: runtime.stop(model.profile, timeout=30), lambda: runtime.cleanup(model.profile)):
        try:
            call()
        except Exception:                                # noqa: BLE001 - nothing may stay half alive
            pass
    return False, f"{model.label} did not start again: {why.rstrip('.')}."


def start_hint(model: RunningModel) -> str:
    """The way to start a stopped model by hand: the launch line of its saved choices when they can be read, else the command."""
    if model.choices:
        try:
            from strata_cli.launch import settings as S
            return S.flags_text(S.LaunchSettings.from_dict({**model.choices, "name": model.name}))
        except Exception:                                # noqa: BLE001
            pass
    return "`strata launch` (the screens start from your saved presets)"


# ---------------------------------------------------------------------------------------------------- the gate

class Declined(updater.UpdateError):
    """The person said no: nothing was changed.  Exit 0."""

    def __init__(self, message: str):
        super().__init__(message, 0)


class ModelGate:
    """What the updater asks when a model is loaded.  ``ask(question, default) -> bool`` is a person (None: nobody can be asked);
    ``allow_stop`` is the explicit yes of a script; ``start_again`` the explicit yes to the start-again question of a script.  ``say`` shows a
    line; ``emit`` gets one event per step (the app's host reads them from ``strata update --output jsonl``)."""

    def __init__(self, *, ask: Callable[[str, bool], bool] | None = None, say: Callable[[str], None] = lambda text: None,
                 emit: Callable[[dict], None] = lambda event: None, allow_stop: bool = False, start_again: bool = False, settle_s: float = 120.0,
                 stop: Callable[[RunningModel], None] | None = None, start: Callable[[RunningModel], tuple[bool, str]] | None = None):
        self.ask, self.say, self.emit = ask, say, emit
        self.allow_stop, self.start_again, self.settle_s = allow_stop, start_again, settle_s
        self._stop, self._start = stop or stop_model, start or start_model          # (looked up now, so a test or a host can replace them)
        self.consented: dict[str, RunningModel] = {}
        self.stopped: list[RunningModel] = []
        self.restarts: list[dict] = []

    # -- before anything is downloaded
    def consent(self, serving: list[dict], plan: str) -> None:
        models = running_models(serving)
        loose = [m for m in models if not m.can_stop]
        if loose:
            raise updater.UpdateError(
                f"{_names(loose)} is serving, and Strata did not start it from here (a service, `strata serve` or another program), so it cannot be "
                f"stopped for you. Stop it yourself, then run {NEXT_STEP} again. Nothing was changed.", EXIT_BUSY)
        names = _names(models)
        self.emit({"event": "models", "phase": "consent", "models": [m.to_dict() for m in models], "message": f"{names} is running"})
        if not self.allow_stop:
            if self.ask is None:
                raise updater.UpdateError(
                    f"{names} {'is' if len(models) == 1 else 'are'} running, and the update needs "
                    f"{'it' if len(models) == 1 else 'them'} stopped, so nothing was changed. To let the update stop "
                    f"{'it' if len(models) == 1 else 'them'}: run {NEXT_STEP} again with --stop-model (add --start-again to start "
                    f"{'it' if len(models) == 1 else 'them'} again afterwards). Or stop {'it' if len(models) == 1 else 'them'} yourself first "
                    f"(`strata launch --stop`).", EXIT_BUSY)
            self.say(f"{names} {'is' if len(models) == 1 else 'are'} running. {plan} The new engine is downloaded while "
                     f"{'it keeps' if len(models) == 1 else 'they keep'} serving; then {'it is' if len(models) == 1 else 'they are'} stopped "
                     f"(answers in progress end), the engine is updated, and you choose whether to start "
                     f"{'it' if len(models) == 1 else 'them'} again with the same settings.")
            if not self.ask(f"Stop {names} for the update?", False):
                raise Declined("Nothing was changed. The model keeps running.")
        for m in models:
            self.consented[m.profile] = m

    # -- after the download, before the first thing is staged or test-started
    def release(self, serving_now: Callable[[], list[dict]]) -> None:
        serving = serving_now()
        if not serving:
            return
        models = running_models(serving)
        new = [m for m in models if m.profile not in self.consented]
        if new:                                          # started after the yes: not covered by it
            names = _names(new)
            if self.ask is not None and not self.allow_stop:
                self.say(f"{names} started while the update was downloading.")
                if self.ask(f"Stop {names} too?", False):
                    for m in new:
                        self.consented[m.profile] = m
                    new = []
            if new:
                raise updater.UpdateError(
                    f"{names} started while the update was running, and nobody agreed to stop it, so nothing more was changed. "
                    f"Stop it, then run {NEXT_STEP} again (the download is kept).", EXIT_BUSY)
        for m in models:
            self.emit({"event": "models", "phase": "stopping", "models": [m.to_dict()], "message": f"Stopping {m.label}"})
            self.say(f"Stopping {m.label} (answers in progress end first)...")
            try:
                self._stop(m)
            except Exception as exc:                     # noqa: BLE001
                raise updater.UpdateError(f"{m.label} could not be stopped ({exc}). Nothing was changed; it may still be running.", EXIT_BUSY) from None
            if m.profile not in {s.profile for s in self.stopped}:
                self.stopped.append(m)
            self.emit({"event": "models", "phase": "stopped", "models": [m.to_dict()], "message": f"{m.label} is stopped"})
        self._settle(serving_now)

    def _settle(self, serving_now: Callable[[], list[dict]]) -> None:
        """A stopped engine lets go of the card and of the shared lease a moment later (a big DeepSeek model takes a while to unload): wait for that,
        up to ``settle_s``, before the update goes on. Found by the real DeepSeek flow, where the very next check still saw "a running Strata server"."""
        import time
        deadline = time.monotonic() + self.settle_s
        while serving_now():
            if time.monotonic() >= deadline:
                raise updater.UpdateError("The model was stopped but its engine has not let go of the graphics card yet, so nothing more was changed. "
                                          f"Wait a minute and run {NEXT_STEP} again.", EXIT_BUSY)
            time.sleep(min(1.0, self.settle_s / 10 or 0.1))

    # -- after the update, whatever happened
    def finish(self, update_ok: bool, changed: bool | None = None) -> bool:
        """Start the stopped models again when the person says so.  True when every model that was meant to start did."""
        everyone = True
        for m in self.stopped:
            if update_ok or changed:
                question = f"Start {m.label} again with the same settings?"
            else:
                question = f"The old engine is unchanged. Start {m.label} again on it?"
            if self.start_again:
                yes = True
            elif self.ask is not None:
                yes = self.ask(question, True)
            else:
                yes = False
            if not yes:
                hint = start_hint(m)
                self.say(f"{m.label} is stopped. Start it when you want with {hint}")
                self.restarts.append({**m.to_dict(), "restarted": False, "asked": self.ask is not None or self.start_again})
                self.emit({"event": "models", "phase": "left-stopped", "models": [m.to_dict()], "message": f"{m.label} stays stopped", "start": hint})
                continue
            self.emit({"event": "models", "phase": "restarting", "models": [m.to_dict()], "message": f"Starting {m.label} again"})
            self.say(f"Starting {m.label} again (loading can take a few minutes)...")
            ok, text = self._start(m)
            self.say(text)
            if not ok:
                everyone = False
                self.say(f"The engine is not the problem if this is a one-off: start it with {start_hint(m)}")
            self.restarts.append({**m.to_dict(), "restarted": ok, "asked": True, "message": text})
            self.emit({"event": "models", "phase": "restarted" if ok else "restart-failed", "models": [m.to_dict()], "message": text})
        return everyone

    def report(self) -> dict:
        return {"stopped": [m.to_dict() for m in self.stopped], "restarts": list(self.restarts)}


def update_with_models(ctx: "updater.Context", gate: ModelGate, show: Callable[[dict], None] | None = None, **kwargs) -> dict:
    """``updater.run_update`` with the gate attached, and the stopped models started again afterwards (also when the update failed or was refused
    after a model had been stopped).  Returns run_update's answer plus ``models`` and, if a model did not start again, exit code 8.
    Raises what run_update raises (Busy, Declined, a refused feed) after the models have had their chance to start again."""
    ctx.gate = gate
    ok, out, error, changed = False, None, None, False
    try:
        out = updater.run_update(ctx, **kwargs)
        ok = out["exit_code"] == 0
        changed = any(r.get("changed") for r in out["results"])
        if show is not None:
            show(out)                                    # what the update did comes before the questions about the models
    except Exception as exc:                             # noqa: BLE001 - restarted below, then re-raised
        error = exc
    restarted = gate.finish(ok, changed) if gate.stopped else True
    if error is not None:
        raise error
    out["models"] = gate.report()
    if not restarted and out["exit_code"] in (0, 6):
        out["exit_code"] = EXIT_NOT_RESTARTED if out["exit_code"] == 0 else out["exit_code"]
    return out


def build_context(root=None, feed: str | None = None, emit: Callable[[dict], None] = lambda event: None) -> "updater.Context":
    """The update's context as `strata update` builds it: this computer's hardware, the trusted keys, the feed the person or the install uses."""
    import argparse
    from strata_cli import update_cmd
    from strata_control import signing
    base = Path(root) if root else ec.components_dir()
    return updater.Context(components=base, source=update_cmd._feed_source(argparse.Namespace(feed=feed), base), trusted=signing.load_trusted_keys(),
                           hardware=updater.detect_hardware(), emit=emit)


def summary(out: dict) -> str:
    """One plain sentence for what an update did, for a screen that has one line (the monitor)."""
    parts = []
    for r in out.get("results", []):
        name = {"engine": "Engine", "dsv41": "DeepSeek engine", "control": "Control"}.get(r.get("component"), str(r.get("component")))
        status = r.get("status")
        if status == "updated":
            parts.append(f"{name} is now {r.get('installed_after') if isinstance(r.get('installed_after'), str) else (r.get('installed_after') or {}).get('version', 'new')}")
        elif status == "failed":
            parts.append(f"{name} was not changed: {str(r.get('error', '')).rstrip('.')}")
    text = "; ".join(parts) or "Nothing needed updating."
    for r in (out.get("models") or {}).get("restarts", []):
        text += " " + (r.get("message") or (f"{r['name']} is stopped." if not r.get("restarted") else f"{r['name']} is serving again."))
    return text


def run_flow(*, ask, say, emit: Callable[[dict], None] = lambda event: None, allow_stop: bool = False, start_again: bool = False,
             feed: str | None = None, root=None, engine: bool = True, control: bool = True) -> dict:
    """The whole opt-in update for a screen that cannot print (the monitor) or a host (the app): build the context, attach the gate, run, start the
    models again.  Raises what the update raises (Declined, Busy, a refused feed): the caller shows ``str(exc)``."""
    ctx = build_context(root, feed, emit)
    gate = ModelGate(ask=ask, say=say, emit=emit, allow_stop=allow_stop, start_again=start_again)
    return update_with_models(ctx, gate, engine=engine, control=control)
