# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Split serving: one engine reads the prompt, another makes the answer (upstream 690172c46; issue #4000 there, idea and
manual setup by @MarcoTool).

Two engines on the same PC, one card each, the same model files and the same session settings.  A long prompt is read by
the prefill engine, which saves the session (the engine's own session file: SAVE <path>) into RAM; the decode engine
restores it (RESTORE <path>) and makes the answer.  The server keeps its one HTTP front, its one Service and its one set
of endpoints; `SplitEngine` stands in for the engine, as `ReplicaEngine` does for replicas.

Opt-in, in the run config (nothing changes without it).  XeStrata names its GPUs by PCI address, as "gpu_pci" does; the
two cards may be of different makers (an Intel Arc and an NVIDIA card) when the engine's build drives both:

    "split_serve": {"prefill_gpu": "0000:03:00.0", "decode_gpu": "0000:09:00.0"}

    prefill_gpu / decode_gpu   the card of each engine, by PCI address (required, different); they replace the config's
                               "gpu_pci" and "split_pci"
    min_prompt_tokens          hand a request over only when the decode engine would read at least this many new tokens
                               itself (default 2048); a shorter tail is read by the decode engine in place
    dir                        where the session files go (default /dev/shm when it has room, else the temp folder)
    shared_arena               true (default): one copy of the experts in RAM for both engines (Linux, a memory file
                               without a name, which goes when the last of the server and its engines ends); false:
                               each engine keeps its own copy; a path: that file
    prefill_args / decode_args extra engine arguments for one role (a bigger expert cache for the decode engine, say)

What must match between the two engines is what the session file checks (docs/MULTIGPU.md): the model files, the engine
version, the backend, --kv, --max-context, --kv-resident, --mtp-window, the rope settings and the arithmetic switches.
They start from one config, so they do.  Every failure on the way (a prefill engine that died, a refused file) is not an
error for the client: the decode engine reads the prompt itself, as it would without the option.

Honest limit of this first version: one request at a time (the Service's FIFO), one conversation resident on the decode
engine (a restore replaces what it held), no parallel slots (--batch), no layer split in an engine, no pictures (a
request with pictures goes to the decode engine alone). The prefill engine keeps no parked conversations on disk
("conversation_save" is the decode engine's alone: two engines would write the same folder).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

MIN_TOKENS = 2048                    # the decode engine's own tail below this is read in place
KNOWN_KEYS = {"prefill_gpu", "decode_gpu", "min_prompt_tokens", "dir", "shared_arena", "prefill_args", "decode_args"}
FORBIDDEN_ARGS = {                   # engine arguments that cannot be used in a split-serve engine, and why
    "--layer-split": "an engine of the split runs on one card",
    "--split-device": "an engine of the split runs on one card",
    "--batch": "session files do not support --batch (parallel requests)",
    "--slots": "session files do not support parallel slots",
}
HEAL_COOLDOWN_S = 60.0               # a prefill engine that died is started again at most this often
HEARTBEAT_S = 5.0                    # a long file step sends a keep-alive this often


class SplitConfigError(ValueError):
    pass


def _card(v):
    """A PCI address (domain:bus:device.function), lower case, or None."""
    if not isinstance(v, str):
        return None
    c = v.strip().lower()
    return c if c.count(":") == 2 and "." in c else None


def split_spec(cfg: dict) -> dict | None:
    """The checked "split_serve" section of the run config, or None when the option is off."""
    raw = cfg.get("split_serve")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise SplitConfigError('"split_serve" must be {"prefill_gpu": "0000:03:00.0", "decode_gpu": "0000:09:00.0"}, '
                               'not ' + repr(raw))
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if unknown:
        raise SplitConfigError(f'"split_serve" has no key {unknown[0]!r} (keys: {", ".join(sorted(KNOWN_KEYS))})')
    cards = []
    for k in ("prefill_gpu", "decode_gpu"):
        c = _card(raw.get(k))
        if c is None:
            raise SplitConfigError(f'"split_serve" needs "{k}": a card by PCI address (0000:03:00.0), not '
                                   f'{raw.get(k)!r}')
        cards.append(c)
    if cards[0] == cards[1]:
        raise SplitConfigError(f'"split_serve": prefill_gpu and decode_gpu are both card {cards[0]}; each engine '
                               "needs a card of its own")
    out = {"prefill_gpu": cards[0], "decode_gpu": cards[1]}
    n = raw.get("min_prompt_tokens", MIN_TOKENS)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise SplitConfigError(f'"split_serve": "min_prompt_tokens" must be a number of tokens (0 or more), not {n!r}')
    out["min_prompt_tokens"] = n
    d = raw.get("dir")
    if d is not None and (not isinstance(d, str) or not d.strip()):
        raise SplitConfigError(f'"split_serve": "dir" must be a folder name, not {d!r}')
    out["dir"] = d
    sa = raw.get("shared_arena", True)
    if not (isinstance(sa, bool) or (isinstance(sa, str) and sa.strip())):
        raise SplitConfigError(f'"split_serve": "shared_arena" must be true, false or a file name, not {sa!r}')
    out["shared_arena"] = sa
    for k in ("prefill_args", "decode_args"):
        a = raw.get(k, [])
        if not isinstance(a, list) or not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in a):
            raise SplitConfigError(f'"split_serve": "{k}" must be a list of engine arguments')
        out[k] = [str(x) for x in a]
    return out


def check_cfg(cfg: dict, spec: dict) -> None:
    """Refuse (with the reason, in plain words) a config the split cannot run."""
    if cfg.get("replicas") not in (None, False):
        raise SplitConfigError('"split_serve" and "replicas" cannot be used together: both start engines on the cards')
    args = [str(a) for a in cfg.get("args") or []] + spec["prefill_args"] + spec["decode_args"]
    for flag, why in FORBIDDEN_ARGS.items():
        if flag in args:
            raise SplitConfigError(f'"split_serve" cannot be used with {flag} in the engine arguments: {why}')
    for i, a in enumerate(args):
        if a == "--prompt-cache" and i + 1 < len(args) and args[i + 1].strip() == "0":
            raise SplitConfigError('"split_serve" needs the conversation cache (--prompt-cache 1 or more): the session '
                                   "file is made from it")
    if "--mtp" not in args:
        raise SplitConfigError('"split_serve" needs --mtp in the engine arguments: a session file carries the draft '
                               "layer's K/V")
    par = cfg.get("parallel")
    if isinstance(par, int) and not isinstance(par, bool) and par > 1:
        raise SplitConfigError('"split_serve" does not run parallel requests ("parallel"): remove it')


def _suffixed(path, tag: str):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.{tag}{ext}"


def role_cfg(cfg: dict, spec: dict, role: str) -> dict:
    """The config of one engine of the split: its own card, log and learned-profile file; its role's extra arguments."""
    import copy
    c = copy.deepcopy(cfg)
    c.pop("split_serve", None)
    c.pop("gpu", None)
    c["gpu_pci"] = spec[f"{role}_gpu"]
    for k in ("split_pci", "layer_split", "gpu_order"):    # a layer split's cards: an engine of the split has one
        c.pop(k, None)
    if role == "prefill":
        c.pop("conversation_save", None)
    c["log"] = _suffixed(cfg.get("log"), role)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], role)
    extra = spec[f"{role}_args"]
    if extra:
        c["args"] = list(c["args"]) + extra
    return c


def _arg_value(args: list, flag: str):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return None


def arena_path(cfg: dict, spec: dict) -> tuple[str | None, int | None, str]:
    """(file for --shared-expert-arena or None, the descriptor the engines inherit for it or None, a line for the log).
    Only Linux can share the arena.  By default it is a memfd the server holds and the engines open as
    /proc/self/fd/<n>: a named file in /dev/shm stayed in RAM, the size of the experts, whenever the server ended
    without its close() (SIGKILL, a crash), and a restarted engine still finds the arena through the server's
    descriptor."""
    args = [str(a) for a in cfg.get("args") or []]
    if "--shared-expert-arena" in args:
        return None, None, "the config already names a shared expert arena"
    sa = spec["shared_arena"]
    if sa is False:
        return None, None, "shared_arena is off: each engine keeps its own copy of the experts in RAM"
    if "--mmap-experts" in args:
        return None, None, "--mmap-experts reads the experts from the file (the engine cannot share an arena with it)"
    if os.name == "nt":
        return None, None, ("this OS cannot share the expert arena between two engines (Linux only): each engine "
                            "keeps its own copy of the experts in RAM, so the PC needs RAM for two")
    if isinstance(sa, str):
        return os.path.abspath(sa), None, f"shared expert arena in {os.path.abspath(sa)}"
    if not hasattr(os, "memfd_create"):
        return None, None, ("this OS has no memfd: each engine keeps its own copy of the experts; name a file in "
                            "\"shared_arena\" to share one")
    try:
        fd = os.memfd_create("xestrata-split-arena")      # close-on-exec: only the engines given it (pass_fds) get it
    except OSError as e:
        return None, None, f"no memory file for a shared arena ({e}): each engine keeps its own copy of the experts"
    return f"/proc/self/fd/{fd}", fd, "one shared expert arena in RAM"


def session_dir(spec: dict) -> str:
    d = spec.get("dir")
    if d:
        d = os.path.abspath(d)                          # the engines run in the config's cwd
        os.makedirs(d, exist_ok=True)
        return d
    shm = "/dev/shm"
    try:
        if os.name != "nt" and os.path.isdir(shm) and os.access(shm, os.W_OK) and \
                shutil.disk_usage(shm).free > (6 << 30):
            return tempfile.mkdtemp(prefix="xestrata-split-", dir=shm)
    except OSError:
        pass
    return tempfile.mkdtemp(prefix="xestrata-split-")


def common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class SplitEngine:
    """Stands in for one engine (the interface Service uses), over a prefill engine and a decode engine."""

    # not offered: one conversation file / VRAM resize / control line
    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}
    PROPAGATE = {"silence_s", "gpu_busy"}

    def __init__(self, prefill, decode, *, min_tokens: int = MIN_TOKENS, work_dir: str | None = None,
                 arena_fd: int | None = None, say=None):
        d = self.__dict__
        d["prefill"], d["decode"] = prefill, decode
        d["cur"] = decode                      # the engine the request is on now (progress, last)
        d["min_tokens"] = int(min_tokens)
        d["work_dir"] = work_dir or tempfile.mkdtemp(prefix="xestrata-split-")
        d["arena_fd"] = arena_fd
        d["say"] = say or (lambda s: print(s, flush=True))
        d["held"] = []                         # the tokens the decode engine's session holds (estimated: ids + answer)
        d["seq"] = 0
        d["heal_at"] = 0.0
        d["healing"] = False
        d["lock"] = threading.Lock()
        d["stats"] = {"requests": 0, "handed_over": 0, "fallbacks": 0, "short": 0, "bytes": 0, "prefill_ms": 0.0,
                      "save_ms": 0.0, "restore_ms": 0.0, "last": None}

    # ---- attribute plumbing
    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.decode, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            self.prefill.__setattr__(name, value)
            self.decode.__setattr__(name, value)
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return self.decode.last

    @last.setter
    def last(self, value):
        self.decode.last = value

    @property
    def progress(self):
        return self.cur.progress

    @property
    def prefill_tok_s_mean(self):
        return self.cur.prefill_tok_s_mean

    @property
    def batch(self) -> int:
        return 0

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in (self.prefill, self.decode) if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in (self.prefill, self.decode))

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in (self.prefill, self.decode))

    @property
    def info(self) -> dict:
        i = dict(self.decode.info)
        i["split_serve"] = True
        i["split_prefill_alive"] = bool(self.prefill.alive())
        return i

    @info.setter
    def info(self, value):
        pass

    def alive(self) -> bool:
        return bool(self.decode.alive())

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in (self.decode, self.prefill) if not e.alive()) if c is not None),
                    None)

    def death_note(self) -> str:
        return next((e.death_note() for e in (self.decode, self.prefill) if not e.alive()), "")

    def split_view(self) -> dict:
        """For /metrics: how the two engines are used."""
        s = dict(self.stats)
        s.update(prefill_alive=bool(self.prefill.alive()), decode_alive=bool(self.decode.alive()),
                 min_prompt_tokens=self.min_tokens)
        return s

    # ---- life cycle
    def restart(self, tries: int = 3):
        errors = {}
        for name, e in (("decode", self.decode), ("prefill", self.prefill)):      # one at a time: the arena fills once
            if e.alive():
                continue
            try:
                e.restart(tries)
            except BaseException as x:           # noqa: BLE001
                errors[name] = x
        if "decode" in errors:
            raise errors["decode"]
        for name, err in errors.items():
            self.say(f"[xestrata] the {name} engine of the split did not start ({err}); the other one serves")
        self.__dict__["held"] = []

    def unload(self):
        for e in (self.prefill, self.decode):
            e.unload()
        self.__dict__["held"] = []

    def close(self):
        err = None
        for e in (self.prefill, self.decode):
            try:
                e.close()
            except BaseException as x:           # noqa: BLE001
                err = err or x
        shutil.rmtree(self.work_dir, ignore_errors=True)
        if self.arena_fd is not None:
            try:
                os.close(self.arena_fd)          # the RAM goes once the engines that had it open have ended
            except OSError:
                pass
            self.__dict__["arena_fd"] = None
        if err:
            raise err

    def heal(self) -> None:
        """Start a prefill engine that died again, in the background (the decode engine serves meanwhile)."""
        e = self.prefill
        now = time.monotonic()
        if e.alive() or getattr(e, "unloaded", False) or getattr(e, "starting", False) or self.healing or \
                now < self.heal_at:
            return
        self.__dict__["heal_at"] = now + HEAL_COOLDOWN_S
        self.__dict__["healing"] = True

        def run():
            try:
                self.say("[xestrata] split: the prefill engine had stopped; starting it again")
                e.restart()
                self.say("[xestrata] split: the prefill engine is running again")
            except BaseException as x:           # noqa: BLE001
                self.say(f"[xestrata] split: the prefill engine did not start again: {x}")
            finally:
                self.__dict__["healing"] = False
        threading.Thread(target=run, daemon=True).start()

    # ---- the request
    def wants_handover(self, ids) -> tuple[bool, int]:
        """(hand the prompt to the prefill engine?, the tokens the decode engine would read itself)."""
        n = len(ids)
        if n < 2:
            return False, 0
        need = (n - 1) - min(common_prefix(self.held, ids), n - 1)
        return need >= max(1, self.min_tokens) and self.prefill.alive(), need

    def _wait(self, fn):
        """Run a blocking engine call on a thread; yields None as a keep-alive while it runs.  -> its result."""
        box = {}

        def run():
            try:
                box["r"] = fn()
            except BaseException as e:           # noqa: BLE001
                box["e"] = e
        th = threading.Thread(target=run, daemon=True)
        th.start()
        while th.is_alive():
            th.join(HEARTBEAT_S)
            if th.is_alive():
                yield None
        if "e" in box:
            raise box["e"]
        return box.get("r")

    def _hand_over(self, ids, cancel, rec: dict):
        """Prefill engine reads the prompt, saves the session; the decode engine restores it.  Yields a keep-alive now
        and then.
        rec['ok'] is True when the decode engine now holds the prompt's first n-1 tokens."""
        from serve.server import EngineDied, SessionRefused
        P, D = self.prefill, self.decode
        n = len(ids)
        self.__dict__["seq"] += 1
        path = os.path.join(self.work_dir, f"handover-{os.getpid()}-{self.seq}.session")
        rec.update(ok=False)
        t0 = time.monotonic()
        self.__dict__["cur"] = P
        try:
            try:
                for _ in P.generate(ids, 0, {}, cancel, prefill_only=True):
                    yield None
            except (EngineDied, ValueError) as e:
                self.say(f"[xestrata] split: the prefill engine failed ({e}); the decode engine reads the prompt "
                         "itself")
                self.heal()
                return
            if cancel.is_set():
                return
            p_last = dict(P.last or {})
            rec["prefill_ms"] = (time.monotonic() - t0) * 1000.0
            rec["p_last"] = p_last
            try:
                saved = yield from self._wait(lambda: P.session_file("save", path))
                self.__dict__["cur"] = D
                restored = yield from self._wait(lambda: D.session_file("restore", path))
            except SessionRefused as e:
                self.say(f"[xestrata] split: the session file was refused ({e}); the decode engine reads the prompt "
                         "itself")
                return
            except EngineDied as e:
                if not D.alive():
                    raise
                self.say(f"[xestrata] split: the prefill engine failed while saving ({e}); the decode engine reads the "
                         "prompt itself")
                self.heal()
                return
            rec.update(save_ms=float(saved["ms"]), restore_ms=float(restored["ms"]), bytes=int(saved["bytes"]),
                       tokens=int(restored["tokens"]), ok=restored["tokens"] == n - 1)
            if not rec["ok"]:
                self.say(f"[xestrata] split: the decode engine restored {restored['tokens']} tokens, not the {n - 1} "
                         "the prompt needs; it reads the rest itself")
                rec["ok"] = restored["tokens"] > 0
        finally:
            self.__dict__["cur"] = D
            try:
                os.unlink(path)
            except OSError:
                pass

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        D = self.decode
        ids = list(ids)
        st = self.stats
        st["requests"] += 1
        rec: dict = {}
        use, need = (False, 0) if embeddings is not None else self.wants_handover(ids)
        if not use:
            st["short"] += 1
        if use:
            if not self.prefill.alive():
                self.heal()
            yield from self._hand_over(ids, cancel, rec)
            if cancel.is_set():
                self.__dict__["held"] = []
                return
            if rec.get("ok"):
                st["handed_over"] += 1
                st["bytes"] += rec["bytes"]
                st["prefill_ms"] += rec["prefill_ms"]
                st["save_ms"] += rec["save_ms"]
                st["restore_ms"] += rec["restore_ms"]
                self.__dict__["held"] = ids[:len(ids) - 1]
            else:
                st["fallbacks"] += 1
        self.__dict__["cur"] = D
        produced: list[int] = []
        ok = False
        try:
            gen = D.generate(ids, max_new, sampling, cancel, embeddings=embeddings) if embeddings is not None \
                else D.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
            ok = True
        finally:
            self.__dict__["held"] = (ids + produced) if ok else []
            if rec.get("ok") and ok:
                self._account(ids, rec)

    def _account(self, ids, rec: dict) -> None:
        """The figures of the engine's last DONE, as the whole request: the prefill engine's read, the file steps and
        the decode engine's own start are the prompt time; what the prefill engine reused is the cached part."""
        D = self.decode
        last = dict(D.last or {})
        p = rec.get("p_last") or {}
        total_prompt_ms = rec["prefill_ms"] + rec["save_ms"] + rec["restore_ms"] + float(last.get("prompt_ms") or 0.0)
        p_read = p.get("prompt_read")
        if p_read is None:
            p_read = len(ids) - int(p.get("reused") or 0)
        last.update(prompt_ms=total_prompt_ms, reused=int(p.get("reused") or 0),
                    prompt_read=max(0, min(len(ids), int(p_read) + int(last.get("prompt_read") or 0))),
                    split_prefill_ms=round(rec["prefill_ms"], 1), split_save_ms=round(rec["save_ms"], 1),
                    split_restore_ms=round(rec["restore_ms"], 1), split_bytes=rec["bytes"])
        D.last = last
        self.stats["last"] = {k: last[k] for k in last if k.startswith("split_")}
        self.say(f"[xestrata] split: prompt {len(ids)} tokens read on the prefill card in "
                 f"{rec['prefill_ms'] / 1000:.1f} s, "
                 f"session {rec['bytes'] / 2**20:.0f} MiB saved in {rec['save_ms'] / 1000:.2f} s and restored in "
                 f"{rec['restore_ms'] / 1000:.2f} s; the decode card makes the answer")
