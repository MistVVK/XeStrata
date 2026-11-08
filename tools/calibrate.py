# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Tune the engine's hardware-dependent settings on this PC (setup's --calibrate).

These settings depend on the machine more than on the model, and the defaults are right for the PC they were
measured on (a Ryzen 5 7600 + RTX 5070 on PCIe 5):
  --pcie-frac     the share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of
                  on the CPU.  Without it the engine splits them by an expert's time on each side, measured at start
                  (after Project Maya); a fixed share is kept only when it beats that.
  --pool-workers  the CPU threads that compute experts.  Every physical core is not always best: on hybrid CPUs the
                  efficiency cores can make the whole window wait for them.
  --adapt-every / --adapt-swaps / --adapt-decay
                  how the VRAM expert tier follows the conversation: how often, how many experts it swaps in, how fast
                  its use counts fade.  A PC whose CPU reads the missed experts slowly (DDR3, ~24 GB/s) gains from
                  swapping more: 160 swaps every window measured +7.9% over the default tier on a Xeon E5-2673 v3
                  (DDR3) with an RTX 4060 Ti on PCIe 3.0 x8 (Q2_0; with the swaps taking effect a window later, #764).
The PCIe share is measured through one engine (a per-request `strata_tune` key); the worker count and the adaptive
tier need a restart per value.  The verify window's length is not measured: the engine sizes each window from its
measured window costs (--spec-window auto), and a request that names spec_min_p would run the older floor rule
instead, so the measurements here never send one and --spec-min-p is not written.  Decode speed only: the prompt
path streams every expert whatever these settings say.

A setting is kept only when it beats the default by more than MIN_GAIN in an interleaved re-measurement - the
adaptive expert tier and the OS make single measurements noisy by a few percent.

    python tools/calibrate.py xestrata-q2_0.json        # measure and print; setup.py --calibrate also saves it
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                    # a setting must beat the default by this much to be kept
PCIE_FRACS = (0.0, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0)   # 1.0: every miss over PCIe, the CPU pool gets no expert
SWEEP_ROUNDS = 3                   # how many times the sweep visits each value (see `sweep`)
# the adaptive tier's candidates (every, swaps, decay) against the engine's own (None): swapping more and remembering
# longer, the rest of the set as the engine has it
ADAPT_CANDIDATES = (None, ("1", "80", "0.97"), ("1", "160", "0.97"))
ADAPT_FLAGS = ("--adapt-every", "--adapt-swaps", "--adapt-decay")
MAX_NEW = 128
# the expert tier follows a text over some windows: 128-token answers end before it shows (on a Xeon E5-2673 v3 with
# DDR3, every 1 / 160 / 0.97 measured +0.7% with 128 tokens and +7.9% with 512-token answers), so its step uses these
TIER_MAX_NEW = 512
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def arg_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    """`args` with `flag value` set (replaced if present), or removed when value is None."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def worker_candidates(default: int, extra=()) -> list[int]:
    """The engine's own count, and fewer: two thirds, a half and a quarter (at least 2), plus `extra` (the P-cores
    minus one on a hybrid CPU, one socket's cores minus one on a 2-socket PC), without repeats.  Bench #780 #815
    #707 #674: 4 beat 15 by 28.7% on 8P + 16E; 17-18 beat 35 by ~20% on 2-socket Xeons."""
    c = [default]
    for w in (round(default * 2 / 3), round(default / 2), round(default / 4), *extra):
        if 2 <= w < default and w not in c:
            c.append(w)
    return c


def _cpulist(text: str) -> set:
    """A kernel cpulist ("0-7,16") as a set of CPU numbers."""
    out: set[int] = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def host_worker_extras(sys_root: str = "/sys") -> list[int]:
    """P-cores - 1 (hybrid CPU) and one socket's cores - 1 (several sockets) of this PC: more counts to try (upstream
    3772ded7).  From what the kernel reports: Intel's hybrid PMU lists (devices/cpu_core/cpus) name the P-cores' CPUs,
    and each CPU's topology its socket and core; nothing is assumed of a model."""
    try:
        cpus = {}
        for d in (Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"):
            top = d / "topology"
            if top.is_dir():
                cpus[int(d.name[3:])] = ((top / "physical_package_id").read_text(encoding="utf-8").strip(),
                                         (top / "core_id").read_text(encoding="utf-8").strip())
        out = []
        p_list = Path(sys_root) / "devices" / "cpu_core" / "cpus"
        if p_list.is_file() and (Path(sys_root) / "devices" / "cpu_atom" / "cpus").is_file():
            p_cores = {cpus[c] for c in _cpulist(p_list.read_text(encoding="utf-8")) if c in cpus}
            if p_cores:
                out.append(len(p_cores) - 1)
        sockets = {pkg for pkg, _ in cpus.values()}
        if len(sockets) >= 2:
            first = min(sockets)
            out.append(len({core for pkg, core in cpus.values() if pkg == first}) - 1)
        return out
    except (OSError, ValueError):
        return []


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=lambda k: med[k])
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


class Session:
    """One running engine: measure decode tok/s for a setting (the median of the prompts' rates)."""

    def __init__(self, engine, ids_list):
        self.engine = engine
        self.ids_list = ids_list

    def rate(self, tune: dict | None = None, max_new: int | None = None) -> float:
        rates = []
        for ids in self.ids_list:
            sampling: dict = {"temperature": 0}
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, max_new or MAX_NEW, sampling, threading.Event())
                    if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {flag: value}, "report": {...}}.
    `start_engine(args)` returns a started engine (serve.server.StrataEngine or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env

        def start_engine(args):
            return StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks: list = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    return measure(engine_args(cfg), ids_list, start_engine, say, host_worker_extras())


def engine_args(cfg: dict) -> list[str]:
    """The arguments the server starts this config's engine with (serve.server.engine_args), so the tuning measures
    the engine as it runs.  #447: this used to read any "gpu" list as a layer split and add --layer-split auto, which
    broke a one-card config with a helper card for the expert tier ("gpu": [0] + --expert-cache-device1: the engine
    refuses a split that leaves no card without a stage), missed the "0,2" spelling, and ignored split_skip_if_fits."""
    from serve.server import engine_args as server_args
    return server_args(cfg)


def engine_error(log: str | None, since: int = 0) -> str | None:
    """#447: the engine's own reason for a failed start - the last line of its log written after byte `since` that is
    its own ("strata ..." or "ERR ..."), else the last line there; None without a log or a new line."""
    if not log:
        return None
    try:
        with open(log, "rb") as f:
            f.seek(since)
            lines = [x.strip() for x in f.read()[-16384:].decode("utf-8", "replace").splitlines() if x.strip()]
    except OSError:
        return None
    return next((x for x in reversed(lines) if x.startswith(("xestrata", "ERR"))), lines[-1] if lines else None)


def tune_of(pcie) -> dict:
    """A request's tuning keys: the PCIe share, none when `pcie` is None (the engine's own split).  No spec_min_p: a
    request that names one runs the floor rule instead of the engine's --spec-window auto."""
    return {} if pcie is None else {"pcie_frac": pcie}


def share_name(pcie) -> str:
    return "split (the engine's own)" if pcie is None else f"share {pcie:.2f}"


def sweep(s, values: list, tune, name, label: str, say) -> dict:
    """Every value, `SWEEP_ROUNDS` times, one round visiting all of them and every other round in reverse order
    (upstream d1990da1).  `tune(v)`: the request's tuning keys for value v; `name(v)`: how it is printed.

    A single rate per value cannot order neighbours.  upstream #1332 has 0.75 to 0.95 inside each other's noise at
    72.7, 69.0, 71.0 and 71.9, and that box moved 8-28% between engine starts at an unchanged setting.  Rotating the
    values inside one engine start puts that drift on every value the same number of times, and the median over the
    rounds keeps a single bad sample from deciding.  The caller takes the winner from the medians.
    """
    out: dict = {v: [] for v in values}
    for r in range(SWEEP_ROUNDS):
        for v in (values if r % 2 == 0 else values[::-1]):
            out[v].append(s.rate(tune(v)))
        say(f"    round {r + 1}/{SWEEP_ROUNDS} {label}: " +
            "  ".join(f"{name(v)} {out[v][-1]:.1f}" for v in values))
    med = {v: statistics.median(out[v]) for v in values}
    say(f"    {label}, median of {SWEEP_ROUNDS}: " +
        ", ".join(f"{name(v)}: {med[v]:.1f} ({min(out[v]):.1f}-{max(out[v]):.1f})" for v in values) +
        f" -> best {name(max(med, key=lambda k: med[k]))}")
    return out


def measure(base_args: list[str], ids_list, start_engine, say=print, extra_workers=()) -> dict:
    t0 = time.time()
    report: dict = {}
    say("  Loading the model for the measurements ...")
    base_args = apply(base_args, {})                   # the product defaults: what the measurements must beat
    eng = start_engine(base_args)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_workers = int(info.get("pool_workers", 0)) or None
        s = Session(eng, ids_list)
        s.warm_up()
        # 1. the PCIe share against the engine's own split of the missed experts (None: no share given - the split it
        #    measured at start, or its link's share where it measures none)
        by_pcie = sweep(s, [None] + sorted(PCIE_FRACS), tune_of, share_name, "PCIe", say)
        best_pcie = max(by_pcie, key=lambda k: statistics.median(by_pcie[k]))
        # 2. the winner against the default, interleaved, the same number of rounds
        dflt, cand = None, best_pcie
        confirm: dict = {dflt: [], cand: []}
        if cand != dflt:
            for _ in range(SWEEP_ROUNDS):
                for k in (dflt, cand):
                    confirm[k].append(s.rate(tune_of(k)))
        chosen = pick(confirm, dflt) if cand != dflt else dflt
        report.update(default={"pcie_frac": share_name(None), "pool_workers": d_workers},
                      pcie_sweep={share_name(k): v for k, v in by_pcie.items()},
                      confirm={share_name(k): v for k, v in confirm.items()})
    finally:
        close(eng)
    settings = {}
    if chosen is not None:
        settings["--pcie-frac"] = f"{chosen:.2f}"
    base_rate = statistics.median(confirm[chosen]) if confirm.get(chosen) else None
    # 3. fewer CPU workers (a restart each), with the chosen share.  #1337's rule covers a death during a
    # measurement too: an engine that exits mid-rate is that candidate losing; the measurements so far stay.
    if d_workers and len(worker_candidates(d_workers, extra_workers)) > 1:
        tuned = with_arg(base_args, "--pcie-frac", None if chosen is None else f"{chosen:.2f}")
        by_workers: dict = {}
        for w in worker_candidates(d_workers, extra_workers):
            say(f"  Measuring with {w} CPU workers (restarts the engine) ...")
            measure_candidate(start_engine, with_arg(tuned, "--pool-workers", None if w == d_workers else str(w)),
                              f"{w} workers", say, report, by_workers, w, ids_list)
        report["workers"] = {str(k): v for k, v in by_workers.items()}
        w_best = pick(by_workers, d_workers) if by_workers else d_workers
        if w_best != d_workers:
            settings["--pool-workers"] = str(w_best)
            base_rate = statistics.median(by_workers[w_best])
        elif by_workers.get(d_workers):
            base_rate = statistics.median(by_workers[d_workers])
    # 4. the adaptive expert tier (a restart each), with everything chosen so far.  A candidate whose engine dies
    # during its own measurement is dropped like a failed start (#1337); the other candidates and every earlier
    # result survive.
    tuned = apply(base_args, settings)
    by_adapt: dict = {}
    for adapt in ADAPT_CANDIDATES:
        key = "default" if adapt is None else "/".join(adapt)
        args = tuned
        for flag, v in zip(ADAPT_FLAGS, adapt or (None,) * len(ADAPT_FLAGS)):
            args = with_arg(args, flag, v)
        which = "(the engine default)" if adapt is None else f"every {adapt[0]}, {adapt[1]} swaps, decay {adapt[2]}"
        say(f"  Measuring the expert tier {which} (restarts the engine) ...")
        measure_candidate(start_engine, args, "the expert tier " + key, say, report, by_adapt, key, ids_list,
                          TIER_MAX_NEW)
    a_best = pick(by_adapt, "default") if by_adapt else "default"
    report["adapt"] = by_adapt
    if a_best != "default":
        for flag, v in zip(ADAPT_FLAGS, a_best.split("/")):
            settings[flag] = v
        base_rate = statistics.median(by_adapt[a_best])
    elif by_adapt.get("default"):
        base_rate = statistics.median(by_adapt["default"])
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "report": report}


def restart(start_engine, args, what, say, report):
    """#1337: a later step's engine start that fails is a candidate that loses, not a failed run: what the earlier
    steps measured (and the settings they chose) must survive it.  Returns the engine, or None after saying why and
    noting it in report["failed_starts"].  (The first start, for steps 1-3, is not covered: with no engine there is
    nothing measured.)"""
    try:
        return start_engine(args)
    except Exception as e:  # noqa: BLE001 - whatever the start raised (RuntimeError, EngineDied, OSError ...)
        why = first_line(e)
        say(f"    the engine did not start for {what}: {why} - this candidate is dropped, the earlier results are kept")
        report.setdefault("failed_starts", {})[what] = why
        return None


def first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__


def measure_candidate(start_engine, args, what, say, report, by: dict, key, ids_list, max_new: int | None = None):
    """One restart-and-measure candidate (#1337; upstream 41188ec73, #3317): a start that fails and an engine that
    exits mid-measurement are both that candidate losing - `by[key]` gets its two rates, or report["failed_starts"]
    its reason.  Everything measured earlier and the other candidates survive."""
    e = restart(start_engine, args, what, say, report)
    if e is None:
        return
    try:
        s = Session(e, ids_list)
        s.warm_up(1)
        by[key] = [s.rate(max_new=max_new), s.rate(max_new=max_new)]
        say(f"    {what}: {statistics.median(by[key]):.1f} tok/s")
    except Exception as exc:  # noqa: BLE001 - the engine died mid-measurement: this candidate loses (#1337)
        why = first_line(exc)
        say(f"    the engine died while measuring {what}: {why} - this candidate is dropped, "
            "the earlier results are kept")
        report.setdefault("failed_starts", {})[what] = why
    finally:
        close(e)


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(60)
    except Exception:
        proc.kill()


DEFAULTS = {"--pcie-frac": None, "--spec-min-p": "0.5", "--pool-workers": None,   # None: the engine's own choice
            "--adapt-every": None, "--adapt-swaps": None, "--adapt-decay": None}


def apply(args: list[str], settings: dict) -> list[str]:
    """`args` with the calibrated settings; a setting the calibration did not change goes back to the product
    default (setup's --spec-min-p 0.5, the engine's own PCIe share and worker count), so an older calibration's
    values never linger."""
    out = list(args)
    for flag, default in DEFAULTS.items():
        out = with_arg(out, flag, settings.get(flag, default))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate.py <xestrata-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    print(json.dumps(res, indent=1))
