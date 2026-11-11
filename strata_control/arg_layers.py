"""One precedence for the engine options of a launch (round 0213, #5178 #5132 #4985).

An engine option can come from four places. For each option the first one that has it wins, and a launch builds its
run config from them once (``strata_cli/launch/runner.py derive_config``):

1. **explicit launch values**: what this command line (``--set``, the typed flags, the Advanced table) says. They are
   written to the config's ``"user_args"`` as well as to ``"args"``, because the drafter policy (``drafting.apply``) and
   the server rebuild ``"args"`` at every start and keep only ``"user_args"``. ``"launch_set"`` names them, so a flag
   given once is not remembered as the person's standing choice: leave it out next time and it is gone.
2. **persisted overrides**: the person's own options, ``"user_args"`` of the installed config plus the hand edits found in
   the previous launch config (``user_args.capture`` against ``"args_derived"``, the copy a launch stamps when it writes).
3. **calibrated values**: ``engine calibrate --apply-result`` records them in the run config's ``"calibrated"`` entry
   (settings, model and context they were measured for); a launch puts them over the generated defaults again, never
   over 1 or 2, and drops them when the model or the context is another.
4. **Auto**: what setup and the policies (drafting, cards, memory fit) generate.

``"args"`` of the written config is the result, so the file, the start and ``strata status`` show the same value for
an option. ``"user_args"`` stays authoritative at every start (``serve/server.py engine_args``).
"""

from __future__ import annotations

from typing import Any, Mapping

from strata_control import tune_knobs, user_args

CALIBRATED_KEY = "calibrated"
EXPLICIT_KEY = "launch_set"
DERIVED_KEY = "args_derived"       # "args" as the launch wrote them: a difference found next time is a hand edit


def _max_context(args: list[str]) -> str | None:
    for i, a in enumerate(args):
        if a == "--max-context" and i + 1 < len(args):
            return str(args[i + 1])
    return None


def calibration_record(settings: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict:
    """What ``calibrate --apply-result`` writes into the run config next to the changed ``"args"``."""
    args = [str(a) for a in cfg.get("args") or []]
    return {"settings": {str(k): str(v) for k, v in settings.items()}, "model_name": cfg.get("model_name"),
            "max_context": _max_context(args), "images": "--vision" in args}


def _calibration_fits(rec: Any, cfg: Mapping[str, Any]) -> bool:
    """A calibration is valid for the model, context and image setting it was measured with (setup's hardware_key says the same)."""
    if not isinstance(rec, dict) or not isinstance(rec.get("settings"), dict):
        return False
    args = [str(a) for a in cfg.get("args") or []]
    return (rec.get("model_name") == cfg.get("model_name") and rec.get("max_context") == _max_context(args)
            and bool(rec.get("images")) == ("--vision" in args))


def _hand_edits(prev: Mapping[str, Any] | None) -> list[str]:
    """The person's options in the previous launch config: its ``user_args`` and the changes made to its ``args`` since the launch wrote
    them; the flags that launch was given on its command line are not the person's standing choice. A config from before ``args_derived``
    has nothing to compare with, so only its explicit ``user_args`` count (none: they come from the installed config anyway)."""
    if not isinstance(prev, Mapping) or not isinstance(prev.get(DERIVED_KEY), list):
        return []
    mine = user_args.capture({"args": prev.get("args"), user_args.USER_KEY: prev.get(user_args.USER_KEY),
                              user_args.WRITTEN_KEY: prev.get(user_args.WRITTEN_KEY)})
    given = {str(f) for f in prev.get(EXPLICIT_KEY) or []}
    return user_args.flat([g for g in user_args.groups(mine) if user_args.flag_of(g) not in given])


def _put(mine: list[str], flag: str, value: Any, bool_flags: frozenset[str]) -> list[str]:
    if value is None or value is False:
        return user_args.set_flag(mine, flag, None)
    rest = [g for g in user_args.groups(mine) if user_args.flag_of(g) != flag]
    rest.append([flag] if value is True or flag in bool_flags else [flag, *user_args.value_tokens(flag, value)])
    return user_args.flat(rest)


def layer(cfg: dict, explicit: Mapping[str, Any] | None = None, prev: Mapping[str, Any] | None = None,
          bool_flags: frozenset[str] = tune_knobs.BOOL_FLAGS) -> dict:
    """Put the four sources over each other in ``cfg`` (in place; ``cfg["args"]`` is the generated layer on entry).
    ``explicit``: the launch's own options as ``tune_knobs.apply_to_argv`` changes (value, True for present, None for off).
    ``prev``: the config this launch wrote last time (or None). ``bool_flags``: the engine's valueless switches. Returns {"calibrated": bool, "persisted": [...], "explicit": [...]}."""
    explicit = dict(explicit or {})
    args = [str(a) for a in cfg.get("args") or []]
    # 3. calibrated values over the generated ones
    rec = None
    for source in (prev, cfg):                                # the previous launch config first: apply-result wrote it there
        if _calibration_fits((source or {}).get(CALIBRATED_KEY), cfg):
            rec = source[CALIBRATED_KEY]
            break
    cfg.pop(CALIBRATED_KEY, None)
    if rec:
        args = tune_knobs.apply_to_argv(args, {k: v for k, v in rec["settings"].items()})
        cfg[CALIBRATED_KEY] = rec
    # 2. the person's standing options: the installed config's, then the previous launch's hand edits (the newer wins)
    mine = user_args.merge(user_args.flat(user_args.groups(cfg.get(user_args.USER_KEY))), _hand_edits(prev))
    # 1. this command line
    for flag, value in explicit.items():
        mine = _put(mine, flag, value, bool_flags)
    args = tune_knobs.apply_to_argv(args, explicit, bool_flags) if explicit else args
    cfg["args"] = user_args.merge(args, mine)
    if mine:
        cfg[user_args.USER_KEY] = mine
    else:
        cfg.pop(user_args.USER_KEY, None)
    if explicit:
        cfg[EXPLICIT_KEY] = sorted(explicit)
    else:
        cfg.pop(EXPLICIT_KEY, None)
    return {"calibrated": bool(rec), "persisted": [user_args.flag_of(g) for g in user_args.groups(mine) if user_args.flag_of(g) and user_args.flag_of(g) not in explicit],
            "explicit": sorted(explicit)}


def stamp(cfg: dict) -> None:
    """Called by the launch after its last change to ``cfg["args"]``: records them as written, so only a later hand edit differs."""
    written = [str(a) for a in cfg.get("args") or []]
    cfg[DERIVED_KEY] = list(written)                          # (the marker: this config was written by a launch, so "args_written" below is the launch's)
    cfg[user_args.WRITTEN_KEY] = list(written)                # replaces the installed config's copy, which describes other arguments


def record_calibration(candidate: dict, before: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """`engine calibrate --apply-result` on a run config (in place on the candidate that already carries the calibrated ``"args"``): records
    the calibrated values with the model and context they were measured for, so a launch that builds its config again keeps them (#5132);
    the hand edits so far are recorded as the person's first, except the options the calibration sets (a calibration the person asks for is
    their newer choice, as for ``tune_apply``), and ``args_written`` is brought up to date so the calibrated values are never taken for a hand edit."""
    candidate[CALIBRATED_KEY] = calibration_record(settings, candidate)
    if before.get(user_args.WRITTEN_KEY) is not None or before.get(user_args.USER_KEY):
        mine = user_args.capture(dict(before))
        mine = user_args.flat([g for g in user_args.groups(mine) if user_args.flag_of(g) not in settings])
        user_args.stamp(candidate, mine)
        if user_args.WRITTEN_KEY not in before:
            candidate.pop(user_args.WRITTEN_KEY, None)
    if isinstance(candidate.get(DERIVED_KEY), list):         # a launch's own config: its copies of "args" follow, or the calibrated values would read as hand edits
        stamp(candidate)
