"""The account of a tune (#5232): what your settings measured, what a setting had to beat, what was tried and what was left out, and one next step.

Built from the result the tuner already holds (``autotune.TuneResult.to_dict()``), so nothing is measured again; saved privately after every tune
(``last-tune-report.txt`` in the presets folder, no argument lists, no paths, no keys) and printed again by ``strata tune report``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

FILE_NAME = "last-tune-report.txt"
NONE_YET = "No tune has been run yet."
#: the last line of `strata tune auto`'s own answer
POINTER = "Why each setting was or was not chosen: strata tune report"


def _rate(value: Any) -> str | None:
    try:
        return f"{float(value):.1f}" if value else None
    except (TypeError, ValueError):
        return None


def _runs_text(runs: Any) -> str:
    vals = [f"{float(x):g}" for x in (runs or []) if isinstance(x, (int, float))]
    return f" (measured {len(vals)} times: {', '.join(vals)})" if len(vals) >= 2 else ""


def _verdict(arm: dict[str, Any]) -> str:
    status = arm.get("status")
    ratio = arm.get("ratio_decode")
    pairs, faster = int(arm.get("pairs") or 0), int(arm.get("faster") or 0)
    against = f"{(ratio - 1.0) * 100:+.1f}% against your settings, faster in {faster} of {pairs} rounds" if ratio and pairs else ""
    if status == "failed":
        return f"could not run: {arm.get('reason') or 'the engine stopped'}"
    if arm.get("won"):
        return f"faster, {against}" if against else "faster"
    if status == "dropped":
        return "slower, stopped after one round"
    return f"not faster by enough: {against}" if against else "not faster by enough"


def _next_step(r: dict[str, Any], name: str | None) -> str:
    outcome = r.get("outcome")
    if outcome == "improved":
        return (f"'{name}' is applied at every start. See it with 'strata presets show', or go back with 'strata presets undo'." if name
                else "The faster settings were not saved.")
    if outcome == "default_confirmed":
        return "Nothing was changed. Your settings are already the fastest of what was tried."
    if outcome == "inconclusive":
        return "Nothing was changed. Close other programs and tune again."
    if outcome == "base_failed":
        return "Nothing was changed. Fix the problem above and tune again."
    return "Nothing was changed."


def build(result: dict[str, Any], preset_name: str | None = None) -> str:
    """The account, as plain lines. ``result`` is ``TuneResult.to_dict()`` (also the dict the app reads)."""
    base = result.get("base") or {}
    noise = result.get("noise") or {}
    lines: list[str] = []
    decode, prompt = _rate(base.get("decode_tok_s")), _rate(base.get("pp_tok_s"))
    if decode:
        loaded = (result.get("concurrency") or 1) > 1 and _rate(base.get("load_tok_s"))
        # with --concurrency the tune's repeated measurements (noise.base_runs) are of the loaded rate, so they belong on the loaded line
        lines.append(f"Your settings: answers at {decode} tok/s" + (f", prompts read at {prompt} tok/s" if prompt else "") + ("" if loaded else _runs_text(noise.get("base_runs"))))
        if loaded:
            lines.append(f"With {result['concurrency']} requests at once: {_rate(base.get('load_tok_s'))} tok/s in all" + _runs_text(noise.get("base_runs")))
        if noise.get("threshold_pct") is not None:
            lines.append(f"To count, a setting had to be at least +{float(noise['threshold_pct']):.1f}% faster; identical runs on this PC differed by "
                         f"{float(noise.get('aa_gap') or 0.0) * 100:.1f}%.")
    elif result.get("message"):
        lines.append(str(result["message"]))
    arms = [a for a in result.get("arms") or [] if a.get("status") != "skipped"]
    if arms:
        lines.append("What was tried:")
        lines += [f"  {a.get('label') or a.get('id')}  {_verdict(a)}" for a in arms]
    skipped = [s for s in result.get("skipped") or [] if s.get("label")]
    if skipped:
        lines.append("Left out:")
        lines += [f"  {s['label']}: {s.get('reason', '')}".rstrip(": ") for s in skipped]
    lines.append("Next: " + _next_step(result, preset_name))
    return "\n".join(lines) + "\n"


def report_path() -> Path:
    from strata_control import presets
    return presets.preset_dir() / FILE_NAME


def save(text: str) -> Path | None:
    """Write the account privately (0600 where the system has modes), replacing the last one. None when the folder cannot be written."""
    try:
        from strata_control import presets
        path = report_path()
        presets._atomic(path, text.encode("utf-8"))      # noqa: SLF001 - the presets folder's own atomic write
        from strata_control import private_files
        private_files.restrict_to_current_user(path)      # 0600, or a DACL for this user only on Windows, as the other private files
        return path
    except OSError:
        return None


def read() -> str | None:
    try:
        return report_path().read_text(encoding="utf-8")
    except OSError:
        return None
