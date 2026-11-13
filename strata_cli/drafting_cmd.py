"""``strata drafting plan|fetch``: the DFlash2 drafter file (strata_control/drafting_fetch.py). The app calls it with ``--output jsonl``.

With ``--config`` (a model's run config) and ``--mode`` (the choice, default: the config's own ``drafting``) the command decides what this
model needs: ``needed`` is true only when the choice would use the drafter and its file is not here yet. ``fetch --if-needed`` then downloads
only in that case, so the app can call it before every load without asking twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from strata_control import acquisition, drafting
from strata_control.discovery import MTP_FETCH_GB


def add_parser(sp) -> None:
    drafting_p = sp.add_parser("drafting", help="the DFlash2 drafter file: what it would download, and fetch it")
    d_sp = drafting_p.add_subparsers(dest="drafting_command", required=True)
    for name, text in (("plan", "say what fetching would download; nothing is changed"),
                       ("fetch", "download the drafter (pinned, checked) and pack it for the engine")):
        one = d_sp.add_parser(name, help=text)
        one.add_argument("--data-dir", metavar="DIR", help="the model data folder (default: the one Strata uses)")
        one.add_argument("--config", metavar="FILE", help="a model's run config: its data folder, model and drafting choice decide what is needed")
        one.add_argument("--family", metavar="NAME", help="the model family when there is no run config yet (qwen, gyro, coder, ...)")
        one.add_argument("--model", metavar="NAME", help="the model (size) when there is no run config yet, with --family: so the plan names the drafter this model will really use and its size")
        one.add_argument("--mode", choices=drafting.MODES, help="the choice to decide for (default: the run config's own)")
        one.add_argument("--no-split", action="store_true",
                         help="decide without the all-cards split engine, as if it will not take the start (a person's choice; the app asks this only after the draft layer for the split could not be downloaded, so the plan is DFlash2's again)")
        one.add_argument("--engines-unloading", action="store_true",
                         help=argparse.SUPPRESS)
        if name == "fetch":
            one.add_argument("--if-needed", action="store_true", help="download only when the choice would use the drafter and it is not here")
            one.add_argument("--throttle", action="store_true",
                             help="a load the person did not ask the drafter for (the app's plain load, a restart, a tuning run): skip the feed for 12 hours after it had nothing, and say nothing when it cannot be reached")
        one.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def _hardware() -> tuple[str | None, list[str]]:
    return drafting.detect_hardware()


def _split_expected(cfg: dict, args: list, unloading: bool = False) -> bool:
    """Would the all-cards split engine take a start of this run config (strata_control/split_engine.expected, asked before the draft layer is on disk)?"""
    try:
        from strata_control import split_engine as SE
        flags = [str(x) for x in args]
        flag = lambda name: next((flags[i + 1] for i, x in enumerate(flags[:-1]) if x == name), None)   # noqa: E731
        gpus = cfg.get("gpu")
        cards = [int(g) for g in gpus] if isinstance(gpus, list) else ([int(gpus)] if isinstance(gpus, int) and not isinstance(gpus, bool) else None)
        probe = None
        if unloading:                                        # memory the app's own engines hold is given back before the new start: the cards count by their size (as Runner.card_probe does)
            import dataclasses
            probe = lambda idx: [dataclasses.replace(c, free_mib=None) for c in SE.probe_cards(idx)]   # noqa: E731
        return SE.expected({"split_engine": cfg.get("split_engine"), "drafting": cfg.get("drafting"), "user_args": cfg.get("user_args"),
                            "replicas": cfg.get("replicas")}, indexes=cards, native=flag("--native"),
                           context=int(flag("--max-context") or SE.TESTED_CONTEXT), kv=flag("--kv"), probe=probe)
    except Exception:                                        # noqa: BLE001 - an early question is never a reason to stop
        return False


def _planned_model_name(a) -> str | None:
    """The run config's model_name for a model that has no run config yet (``--family qwen --model IQ3_S`` -> ``qwen3.8-flash-next-iq3_s``); a full name is taken as it is. None: not known."""
    name = getattr(a, "model", None)
    if not name:
        return None
    if str(name).lower().startswith(drafting.MODEL_PREFIX):
        return str(name)
    try:
        from strata_cli.launch.catalog import Catalog
        cat = Catalog()
        family, model = cat.resolve(str(name), a.family)
        return cat.model_name_of(family, model)
    except Exception:                                        # noqa: BLE001 - an unknown model is only the old, generic plan
        return None


SEVERAL_MODELS = "Several models are set up here. Say which one: strata drafting fetch --model <name>, or --config <file>."


def _installed_dflash2_configs(a) -> list[dict]:
    """A bare ``strata drafting fetch`` (no run config, no model, no family): the installed models DFlash2 is for; [] for any other command line or when none can be read."""
    if a.config or getattr(a, "model", None) or a.family or a.drafting_command != "fetch":
        return []
    try:
        from strata_cli.launch.catalog import Catalog
        return [c for c in Catalog().installed_configs() if c.get("path") and drafting.supported_model(c.get("model_name"))]
    except Exception:                                        # noqa: BLE001 - no answer is the generic fetch
        return []


def _only_installed_config(a) -> str | None:
    """A bare ``strata drafting fetch``: the run config of the installed model when exactly one model DFlash2 is for is set up here, so the
    fetch takes the drafter that model would use (the tuned one where it applies) and not the stock one. Several: `run` asks which one (SEVERAL_MODELS); none: None, the fetch is the generic one as before."""
    found = _installed_dflash2_configs(a)
    return str(found[0]["path"]) if len(found) == 1 else None


def _inputs(a) -> dict:
    cfg: dict = {}
    if not a.config:
        a.config = _only_installed_config(a)
    if a.config:
        try:
            cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            cfg = {}
    if a.data_dir:
        data = Path(a.data_dir)
    else:
        data = drafting.data_dir_of(cfg.get("args") or [])
        if data is None:
            from strata_cli.launch.catalog import Catalog
            data = Catalog().data_dir()
    backend, archs = _hardware()
    args = cfg.get("args") or []
    model_name = cfg.get("model_name") or _planned_model_name(a)
    mode = a.mode if a.mode else cfg.get("drafting")
    if cfg.get("mtp") is False and drafting.normalize(mode) == "auto":
        mode = "off"                                         # the Draft layer switch is off and no drafter was chosen since
    elif drafting.normalize(mode) in ("auto", "mtp") and not getattr(a, "no_split", False) and _split_expected({**cfg, "drafting": mode}, args, bool(getattr(a, "engines_unloading", False))):
        a._split = True                                      # the plan says so (the app's Models screen and its load read it); asked with the choice in hand, as the Terminal does
        mode = "mtp"                                         # the all-cards split engine runs the MTP draft layer: the app fetches that, not DFlash2's drafter
    parallel = cfg.get("parallel") if isinstance(cfg.get("parallel"), int) else 0
    slot_drafts = False
    if str(backend or "").lower() == "cuda" and drafting.request_count(args, parallel) in drafting.SLOT_DRAFT_REQUESTS:     # (the card is only read for a start that could be in the class)
        sig = _signals(a, {"backend": backend, "archs": archs})
        slot_drafts = drafting.slot_drafts_by_default(mode, backend=backend, archs=archs, vram_mib=sig.get("vram_mib"), args=sig.get("args"), slots=parallel)   # the drafter is fetched before the start
    return {"mode": mode, "klass": drafting.card_class(archs), "model_name": model_name, "family": a.family, "data_dir": data,
            "backend": backend, "archs": archs, "mtp_here": drafting.mtp_ready(args) if cfg else None,
            "blocked": drafting.blocked_by(args, slots=parallel, named=drafting.normalize(mode) == "dflash2", backend=backend, slot_drafts=slot_drafts)}


def _signals(a, want: dict) -> dict:
    """The arguments and the one card's memory drafting.tuned_applies reads (as the server reads them for the start); {} when there is no run config to read them from."""
    try:
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}
        args = [str(x) for x in cfg.get("args") or []]
        if cfg.get("user_args"):
            from strata_control import user_args
            args = user_args.merge(args, cfg["user_args"])
        return {"args": args, "vram_mib": drafting.card_vram_mib(cfg, want["backend"], want["archs"])}
    except Exception:                                        # noqa: BLE001 - no signal: the tuned drafter is not offered
        return {"args": [], "vram_mib": None}


def run(a) -> int:
    from strata_control import drafting_fetch      # here, not at the top: it pulls in the network code, which a started engine's command line must not load
    machine = a.output in ("json", "jsonl")
    if len(_installed_dflash2_configs(a)) > 1:               # which model's drafter? the generic fetch would take the stock one for a set that has a tuned one
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "which_model", "message": SEVERAL_MODELS}}, ensure_ascii=False))
        else:
            print(f"strata: {SEVERAL_MODELS}", file=sys.stderr)
        return 2
    want = _inputs(a)
    folder = want["data_dir"]
    sig = _signals(a, want)
    needed = drafting.needs_pack(**want, **sig) or drafting.needs_tuned(**want, **sig)       # (the tuned drafter is a download of its own, where the set and the card use it)
    tuned_needed = drafting.needs_tuned(**want, **sig)
    mtp_needed = drafting.needs_mtp(**want, **sig)
    if a.drafting_command == "plan":
        decision = drafting.decide(**want, **sig)
        support = drafting.hardware_support(want["backend"], want["archs"], want["model_name"])
        # serves: this computer's engine can run DFlash2 (the build has the code and the card is one it is made for); the app offers the choice only then
        result = {**drafting_fetch.plan(folder, drafting.installed_engine_exe(), model_name=want["model_name"], backend=want["backend"], archs=want["archs"], **sig), "needed": needed, "tuned_needed": tuned_needed, "mtp_here": want["mtp_here"], "mtp_needed": mtp_needed, "mtp_gb": MTP_FETCH_GB if mtp_needed else 0.0,
                  "serves": drafting.ENGINE_SERVES_DFLASH2 and support.can_run, "drafter": decision.drafter,
                  "choice": drafting.choice_of(want["mode"], model_name=want["model_name"], family=want["family"], backend=want["backend"], archs=want["archs"], **sig),
                  "split": bool(getattr(a, "_split", False)), "note": decision.note, "line": drafting.summary_line(decision), "support": support.level, "support_line": support.line}
        if result["split"]:                                # the Terminal's words for the same plan (runner.drafting_plan_line)
            from strata_control import split_engine as SE
            result["line"] = SE.split_plan_line(result["mtp_gb"] or None)
        if (a.config or a.mode or a.model) and not needed:           # asked about a model: nothing to download unless the choice uses the drafter
            result["gb"] = 0.0
            if not result["ready"]:                       # (MTP, off, a model DFlash2 is not for): the size of a drafter nobody fetches is not quoted
                result["text"] = "Nothing to download."
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **result}, ensure_ascii=False))
        else:
            print(result["text"] if needed or not decision.note else result["line"])
        return 0
    if getattr(a, "if_needed", False) and not needed:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": None, "skipped": True}, ensure_ascii=False))
        else:
            print("Nothing to download.")
        return 0

    def show(step: str, done: int, total: int) -> None:
        if a.output == "jsonl":
            print(json.dumps({"event": "progress", "step": step, "done": done, "total": total}, ensure_ascii=False), flush=True)
        elif not machine and total:
            print(f"\r{step} {done / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)

    try:
        out = drafting_fetch.ensure_for(folder, model_name=want["model_name"], backend=want["backend"], progress=show, archs=want["archs"],
                                        throttle=bool(getattr(a, "throttle", False)), **sig)
        if out is None and getattr(a, "throttle", False):    # a load the person did not ask the drafter for: the feed was not reachable (or asked a short while ago): the model starts as it is, nothing is said
            if machine:
                print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": None, "skipped": True}, ensure_ascii=False))
            else:
                print("Nothing to download.")
            return 0
        if out is None:                                    # (a set with a tuned drafter, and nothing answered: an explicit fetch says so)
            raise drafting_fetch.OfflineError("There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as exc:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "drafting_failed", "message": str(exc)}}, ensure_ascii=False))
        else:
            print(f"\nstrata: {exc}", file=sys.stderr)
        return 5
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": str(out)}, ensure_ascii=False))
    else:
        print(f"\nThe DFlash2 drafter is ready: {out}")
    return 0
