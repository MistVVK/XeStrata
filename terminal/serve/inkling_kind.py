"""The Inkling-2-Mini engine as serve sees it (`strata-inkling` behind Strata's `--serve` line protocol).

Mirrors `serve/hy4_kind.py` while adding native `1..64` continuous-batching parallelism (`--parallel 1|4|8|32|64`),
on-device `DSpark` Block-8 Markov drafting (`--draft dspark|off`), and in-process 23-layer Vision (`--vision`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARCH = "inkling2_loop"
EXE_PREFIX = "strata-inkling"
REFERENCE = "component:strata-inkling"
PARALLEL_MAX = 64
#: End-of-turn / EOS markers: <|content_model_end_sampling|> (200006) and <|endoftext|> (199999)
END_TEXTS = ("<|content_model_end_sampling|>", "<|endoftext|>")
IMAGE_PAD_ID = 200054
REFUSED = {
    "logprobs": "logprobs: not supported by the Inkling-2-Mini engine",
    "logit_bias": "logit_bias: not supported by the Inkling-2-Mini engine",
}

NO_DRAFTER_NOTE = "Inkling-2-Mini: this engine has no DSpark drafter, so it starts without it"
NO_IMAGES_NOTE = "Inkling-2-Mini: this engine does not read images yet, so it starts without the Vision shard"
NO_IMAGES_REPLY = (
    "this Inkling-2-Mini engine was started without Inkling-2-Mini-Vision.strata (images=0); "
    "attach the Vision shard with --vision Inkling-2-Mini-Vision.strata"
)
_PROBES: dict = {}


def _base(exe) -> str:
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_inkling(cfg: dict | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    arch = str(cfg.get("arch") or "").lower()
    if arch in (ARCH, "inkling2", "inkling2_loop_text"):
        return True
    exe = str(cfg.get("exe") or "")
    return exe == REFERENCE or _base(exe).startswith(EXE_PREFIX)


def is_inkling_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


def wants_drafter(cfg: dict) -> bool:
    choice = str(cfg.get("drafting") or "dspark").lower()
    return choice != "off"


def parse_info(text: str) -> dict:
    for line in str(text or "").splitlines():
        if line.startswith("arch="):
            return dict(p.split("=", 1) for p in line.split() if "=" in p)
    return {}


def probe(exe: str | None, args: list, cfg: dict | None = None, timeout: float = 30.0) -> dict | None:
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
    env = quiet_crash_env(dict(os.environ))
    dirs = [d for d in ((cfg or {}).get("lib_dirs") or []) if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    try:
        done = subprocess.run(
            [str(exe), "--info", str(model)],
            capture_output=True, text=True, errors="replace", timeout=timeout, env=env, cwd=str(Path(str(exe)).parent),
        )
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    if not wants_drafter(cfg):
        return ["--draft", "off"]
    return ["--draft", "dspark"] if (info or {}).get("drafter") == "dspark" else []


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    if any(k in args for k in ("--parallel", "--batch", "--slots")):
        return []
    n = cfg.get("parallel")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 1:
        return []
    n = min(n, PARALLEL_MAX)
    return ["--parallel", str(n)]


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    from strata_control import user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    args += parallel_args(cfg, args)
    wants_images = "--vision" in args
    need_draft = "--draft" not in args
    info = probe(exe, args, cfg) if (exe and (wants_images or need_draft)) else None
    if need_draft:
        add = drafter_args(cfg, info)
        args += add
        if not add and wants_drafter(cfg) and announce:
            print(f"[strata] {NO_DRAFTER_NOTE}", flush=True)
    if wants_images and (info or {}).get("images") != "1":
        at = args.index("--vision")
        del args[at:at + 2]
        if announce:
            print(f"[strata] {NO_IMAGES_NOTE}", flush=True)
    return args


def encode_path(path) -> str:
    return str(path).replace("%", "%25").replace(" ", "%20")


def write_images(sources, load, normalize, cancel=None, folder=None) -> list[str]:
    from serve import hy4_kind
    import tempfile
    base = Path(folder) if folder else Path(tempfile.gettempdir()) / "strata-inkling-images"
    return hy4_kind.write_images(sources, load, normalize, cancel=cancel, folder=base)


def refuse(engine, req: dict) -> None:
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "inkling_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def end_ids(tokenizer) -> set[int]:
    ids = set()
    for text in END_TEXTS:
        got = tokenizer.encode(text, parse_special=True)
        if len(got) == 1:
            ids.add(int(got[0]))
    return ids
