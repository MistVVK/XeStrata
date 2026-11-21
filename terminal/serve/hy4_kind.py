"""The Hy4-Flash engine as serve sees it: another engine program (`strata-hy4`) behind the same `--serve` line protocol, with its own short list of arguments.

Everything serve adds to a Flash-Next start (the drafter's flags, the card-fit rules, `--vram-reserve-mib`, `--resident-budget-gib`, `--pool-workers`, `--batch`, the expert profile,
the saved tuning) is a flag of the Strata engine that strata-hy4 does not have, and it refuses a flag it does not know.  So for this engine `engine_args` returns the config's own
arguments plus the few it owns, and every step that would rewrite them is skipped by one test, `is_hy4(cfg)`: a config of any other engine takes exactly the path it took before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARCH = "hy4flash"
EXE_PREFIX = "strata-hy4"
REFERENCE = "component:strata-hy4"
#: the ids that end a reply besides the tokenizer's eos: <hy_end:opensource> 120025 (config eos_token_id) and <hy_EOT> 120008
END_TEXTS = ("<｜hy_end:opensource｜>", "<｜hy_EOT｜>")
#: what the engine refuses to do for a request, with the reason a client reads (400 before anything is sent)
REFUSED = {"logprobs": "logprobs: not supported by the Hy4-Flash engine",
           "logit_bias": "logit_bias: not supported by the Hy4-Flash engine"}


def _base(exe) -> str:
    """The program's file name, whichever separator the config was written with (a Windows path read on Linux, or the other way round)."""
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_hy4(cfg: dict | None) -> bool:
    """A run config for the Hy4-Flash engine: its arch, or its program."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    exe = str(cfg.get("exe") or "")
    return exe == REFERENCE or _base(exe).startswith(EXE_PREFIX)


def is_hy4_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


NO_DRAFTER_NOTE = "Hy4-Flash: this engine has no drafter (DFly2) yet, so it starts without it"
NO_IMAGES_NOTE = "Hy4-Flash: this engine does not read images yet, so it starts without the image reader"
NO_IMAGES_REPLY = ("this Hy4-Flash engine does not read images yet; send the text of the picture, or update Strata (an engine that reads images says images=1 at its start)")
_PROBES: dict = {}


def wants_drafter(cfg: dict) -> bool:
    """The drafter (DFly2) is on unless the person said off: the family's default is on (hy4_family.DEFAULT_DRAFTER, owner 2026-11-19); `mtp` / `dflash2` / `auto` are Qwen's
    words and mean "the default" here."""
    from strata_control import hy4_family
    choice = str(cfg.get("drafting") or hy4_family.DEFAULT_DRAFTER).lower()
    return choice != "off"


def parse_info(text: str) -> dict:
    """The key=value facts of `strata-hy4 --info`'s line (arch=hy4flash drafter=dfly2 images=1 ...); {} when there is none."""
    for line in str(text or "").splitlines():
        if line.startswith("arch="):
            return dict(p.split("=", 1) for p in line.split() if "=" in p)
    return {}


def probe(exe: str | None, args: list, cfg: dict | None = None, timeout: float = 30.0) -> dict | None:
    """What the engine program says it can do: `exe --info <model file>` (no graphics card, no model load: it reads Section 0).  None when it cannot be asked (not a file, an
    older program with no --info, a failure): the caller then assumes the plain engine, with neither drafter nor images.  Asked once per program file and model."""
    import subprocess
    model = next((args[i + 1] for i, a in enumerate(args) if a in ("--native", "--model") and i + 1 < len(args)), None)
    if not exe or not model or not Path(str(exe)).is_file():
        return None
    try:
        st = Path(str(exe)).stat()
        key = (str(exe), st.st_mtime_ns, st.st_size, str(model))
    except OSError:
        return None
    if key in _PROBES:
        return _PROBES[key]
    from strata_control.childenv import quiet_crash_env
    env = quiet_crash_env(dict(os.environ))          # no debugger on a user's machine (the same rule every program of ours is started with)
    dirs = [d for d in ((cfg or {}).get("lib_dirs") or []) if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    try:
        done = subprocess.run([str(exe), "--info", str(model)], capture_output=True, text=True, errors="replace", timeout=timeout, env=env, cwd=str(Path(str(exe)).parent))
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    """`--draft dfly2` when the drafter is wanted AND the engine says it has one (`--info`: drafter=dfly2); otherwise nothing: an engine that cannot run it starts without."""
    return ["--draft", "dfly2"] if wants_drafter(cfg) and (info or {}).get("drafter") == "dfly2" else []


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """The arguments of a Hy4-Flash start: the config's (setup wrote --native, --engram, --vision, --max-context, ...), the options the person set by hand over them, the drafter
    (on by default, when the engine has it) and nothing else (the card is chosen by the environment: child_env).  What the engine says it cannot do (`--info`) is left out with
    one line: no drafter yet, no image reader yet: a start never fails because of them."""
    from strata_control import user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    wants_images = "--vision" in args
    need_draft = "--draft" not in args and wants_drafter(cfg)
    info = probe(exe, args, cfg) if (exe and (wants_images or need_draft)) else None
    if need_draft:
        add = drafter_args(cfg, info)
        args += add
        if not add and announce:
            print(f"[strata] {NO_DRAFTER_NOTE}", flush=True)
    if wants_images and (info or {}).get("images") != "1":
        at = args.index("--vision")
        del args[at:at + 2]                      # --vision <Vision shard>
        if announce:
            print(f"[strata] {NO_IMAGES_NOTE}", flush=True)
    return args


def encode_path(path) -> str:
    """A file path as the protocol's `image=` value: it ends at the next space, so a space is %20 and a percent sign %25 (the engine decodes the same two)."""
    return str(path).replace("%", "%25").replace(" ", "%20")


def decode_path(text: str) -> str:
    return str(text).replace("%20", " ").replace("%25", "%")


def write_images(sources, load, normalize, cancel=None, folder=None) -> list[str]:
    """The picture files of one request, in prompt order, for the engine to read: each source (a data: URL, an http(s) URL, a local path) is read by `load`, made one of the
    formats the engine decodes (JPEG, PNG, BMP, GIF: `normalize`, which turns an EXIF-rotated JPEG upright and converts WebP and the rest to PNG) and written under a unique
    per-request name (`<hash>-<nonce>-<idx>.<ext>`) so a cancelled request's `drop_embeddings()` never unlinks a follow-up request's copy of the same picture in multi-turn chat.
    Returns the paths.  A source that cannot be read is a ValueError (a 400 that says so)."""
    import hashlib
    import secrets
    import tempfile
    base = Path(folder) if folder else Path(tempfile.gettempdir()) / "strata-hy4-images"
    base.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(6)
    out: list[str] = []
    try:
        for idx, src in enumerate(sources):
            if cancel is not None and cancel.is_set():
                raise ValueError("the request was cancelled")
            data = normalize(src if isinstance(src, bytes) else load(src))
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg" if data[:3] == b"\xff\xd8\xff" else ".bmp" if data[:2] == b"BM" else ".gif" if data[:3] == b"GIF" else ".img"
            p = base / f"{hashlib.sha256(data).hexdigest()[:24]}-{nonce}-{idx}{ext}"
            p.write_bytes(data)
            out.append(str(p))
    except Exception:
        for p in out:
            Path(p).unlink(missing_ok=True)
        raise
    return out


def refuse(engine, req: dict) -> None:
    """A request for something this engine cannot do is a 400 with the reason, never a silent difference in the answer."""
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "hy4_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def literal_tags(tokenizer) -> list[str]:
    """The texts of Hy4's tagged tokens that are not special (the think and tool-call tags): written inside a message they stay text, as `<think>` does for Qwen."""
    types = getattr(tokenizer, "token_types", None) or []
    toks = getattr(tokenizer, "tokens", None) or []
    return [toks[i] for i, ty in enumerate(types) if ty == 4 and i < len(toks)]


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids


def model_path(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    for key in ("--native", "--model"):
        if key in args and args.index(key) + 1 < len(args):
            return Path(args[args.index(key) + 1])
    return None
