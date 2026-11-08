# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/tool_parallel.py - the image and video tool calls of one reply run side by side (server.py run_with_mcp).

The calls a reply writes together cannot use each other's results, so they may run at once.  Each call goes to a lane
by what it needs (the image tools' ImageTools.lane):

- "now": no GPU job to wait for (a video job started on the server, a plain Lanczos resize): run at once, beside the
  rest.
- a GPU's lane: the image jobs of that GPU, one after another (one image model holds a GPU at a time, and the GPU's
  turn orders them with the other requests' jobs, images.gpu_turn): those of the same model run together, in the
  order the first of each model came, so a model is loaded as few times as it can be.  Calls of generate_image that
  differ only in their seed (none, or seeds one after another) and plan step are one job of n images: each call still
  gets its own image and seed.
- the text model's GPU, lent to the images (use_text_gpu): a second lane beside the image GPU's, where the text model
  makes room for the job as on a shared card.  ImageTools.lane chooses it only where the setting and the request let
  it, and the text GPU is free to lend.

Calls whose results are joined later share a `group`: one that fails stops the rest of its group (those waiting are not
run, a running one is cancelled); the others run to their end.  Every call gets its own result, in the order the reply
wrote them.  Stopping the request stops them all.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass


class Cancel:
    """A call's cancel: set by itself, by its group's failure or by the request's stop (any of `parents`)."""

    def __init__(self, *parents):
        self.parents = [p for p in parents if p is not None]
        self.own = threading.Event()

    def is_set(self) -> bool:
        return self.own.is_set() or any(p.is_set() for p in self.parents)

    def set(self) -> None:
        self.own.set()

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if end is not None and time.monotonic() >= end:
                return False
            time.sleep(0.05)
        return True


LENT = "text-gpu"   # the lane of the text model's GPU, lent to the images


@dataclass
class Task:
    calls: list            # (index, call) of the reply's calls it runs: one, or generate_image's merged into n images
    name: str
    args: dict
    lane: str
    model: str
    group: str = ""
    lend: bool = False     # it may run on the text model's GPU (ImageTools.may_lend)


def _args(c) -> dict:
    a = c.arguments
    if isinstance(a, str):
        try:
            a = json.loads(a or "{}")
        except ValueError:
            return {}
    return a if isinstance(a, dict) else {}


def _merge_key(name: str, args: dict, lane: str, model: str):
    """What two generate_image calls must share to be one job of n images (their seeds and steps aside)."""
    if name != "generate_image" or lane == "now":
        return None
    rest = {k: v for k, v in args.items() if k not in ("seed", "step", "group", "plan")}
    return lane, model, json.dumps(rest, sort_keys=True, ensure_ascii=False)


def plan(calls: list, hub) -> list[Task]:
    """The reply's image and video calls as tasks, each with its lane and model (hub.lane), generate_image's that can
    be one job merged."""
    tasks: list[Task] = []
    merged: dict = {}
    for i, c in calls:
        args = _args(c)
        lane, model, lend = hub.lane(c.name, args)
        group = str(args.get("group") or "")
        key = _merge_key(c.name, args, lane, model)
        t = merged.get((key, group)) if key is not None else None
        if t is not None:
            seeds = [_args(x).get("seed") for _, x in t.calls] + [args.get("seed")]
            given = [s for s in seeds if s is not None]
            first = given[0] if given else None
            if not given or (len(given) == len(seeds) and isinstance(first, int)
                             and all(s == first + k for k, s in enumerate(seeds))):
                t.calls.append((i, c))
                continue
        t = Task([(i, c)], c.name, args, lane, model, group, lend)
        tasks.append(t)
        if key is not None:
            merged[(key, group)] = t
    # the text model's GPU lent: only with two GPU jobs or more, a model's jobs together on one of the two GPUs, the
    # models shared out between them by their jobs' count
    gpu = [t for t in tasks if t.lane != "now"]
    if len(gpu) > 1 and any(t.lend for t in gpu):
        load: dict[str, int] = {}
        where: dict[str, str] = {}
        for t in gpu:
            if t.model not in where:
                lent = all(x.lend for x in gpu if x.model == t.model) and load.get(LENT, 0) < load.get(t.lane, 0)
                where[t.model] = LENT if lent else t.lane
            t.lane = where[t.model]
            load[t.lane] = load.get(t.lane, 0) + 1
    return tasks


def lanes(tasks: list[Task]) -> dict[str, list[Task]]:
    """The tasks of each GPU lane, those of one model together (in the order each model first came)."""
    out: dict[str, list[Task]] = {}
    for t in tasks:
        if t.lane != "now":
            out.setdefault(t.lane, []).append(t)
    for lane, ts in out.items():
        first: dict[str, int] = {}
        for k, t in enumerate(ts):
            first.setdefault(t.model, k)
        out[lane] = sorted(ts, key=lambda t: first[t.model])   # sorted() keeps the order within a model
    return out


def _skipped(name: str, why: str) -> dict:
    text = f"error: not run: {why}"
    return {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None, "tool": name,
            "images": [], "skipped": True}


class Runner:
    """The tasks run by their lanes on threads; `events` gets ("start", index) as a call starts and ("done", index,
    result) as it ends, which run_with_mcp turns into the client's events while the threads run."""

    def __init__(self, hub, calls: list, cancel):
        self.hub, self.cancel = hub, cancel
        self.tasks = plan(calls, hub)
        self.results: dict[int, dict] = {}
        self.events: list = []
        self.lock = threading.Lock()
        self.groups: dict[str, Cancel] = {}
        self.failed: dict[str, str] = {}
        self.threads: list[threading.Thread] = []

    def _emit(self, *ev) -> None:
        with self.lock:
            self.events.append(ev)

    def take(self) -> list:
        with self.lock:
            out, self.events = self.events, []
        return out

    def _run(self, t: Task) -> None:
        group = self.groups.setdefault(t.group, Cancel()) if t.group else None
        if t.group and t.group in self.failed:
            for i, c in t.calls:
                self._done(i, _skipped(c.name, f"{self.failed[t.group]} in its group failed"))
            return
        if self.cancel.is_set():
            for i, c in t.calls:
                self._done(i, _skipped(c.name, "the request was stopped"))
            return
        for i, _ in t.calls:
            self._emit("start", i)
        cancel = Cancel(self.cancel, group)
        args = dict(t.args)
        if len(t.calls) > 1:
            args["_n"] = len(t.calls)
        if t.lane == LENT:
            args["_lend"] = True
        try:
            r = self.hub.call(t.name, args, cancel)
        except Exception as e:   # a tool's own failure: its call's result, the others go on
            text = f"error: {e}"
            r = {"ok": False, "text": text, "chars": len(text), "truncated": False, "ms": 0, "server": None,
                 "tool": t.name, "images": []}
        parts = r.pop("split", None) if isinstance(r, dict) else None
        if parts is None or len(parts) != len(t.calls):
            parts = [r] * len(t.calls)
        for (i, _), p in zip(t.calls, parts):
            self._done(i, p)
        # a failure stops its group; a download waited for is not one (the call is made again later)
        if t.group and not r.get("ok") and not str(r.get("text", "")).startswith("downloading") \
                and not r.get("skipped") and not self.cancel.is_set():
            with self.lock:
                self.failed.setdefault(t.group, t.name)
            if group is not None:
                group.set()

    def _done(self, i: int, r: dict) -> None:
        with self.lock:
            self.results[i] = r
        self._emit("done", i, r)

    def _lane(self, ts: list[Task]) -> None:
        for t in ts:
            self._run(t)

    def start(self) -> None:
        for t in self.tasks:
            if t.lane == "now":
                self.threads.append(threading.Thread(target=self._run, args=(t,), daemon=True, name="tool-now"))
        for lane, ts in lanes(self.tasks).items():
            self.threads.append(threading.Thread(target=self._lane, args=(ts,), daemon=True, name=f"tool-{lane}"))
        for th in self.threads:
            th.start()

    def alive(self) -> bool:
        return any(th.is_alive() for th in self.threads)

    def join(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        for th in self.threads:
            th.join(max(0.0, end - time.monotonic()))
