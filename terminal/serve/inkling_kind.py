"""The Inkling-2-Mini engine as serve sees it: its own shape engine program (`strata-inkling`, component `strata-inkling`),
with native DSpark speculative drafting, on-demand in-process Vision, `--gpus 2` dual-GPU EP, and `1..64` continuous-batching parallelism (`--parallel`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARCH = "inkling2_loop"
EXE_PREFIX = "strata-inkling"
REFERENCE = "component:strata-inkling"
END_TEXTS = ("<|content_model_end_sampling|>", "<|endoftext|>")
IMAGE_PAD_TOKEN = "<|unused_200054|>"
REFUSED = {
    "logprobs": "logprobs: not supported by the Inkling-2-Mini engine",
    "logit_bias": "logit_bias: not supported by the Inkling-2-Mini engine",
}


def _base(exe) -> str:
    return str(exe or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def is_inkling(cfg: dict | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    if cfg.get("arch") == ARCH:
        return True
    exe = str(cfg.get("exe") or "")
    return exe == REFERENCE or _base(exe).startswith(EXE_PREFIX)


def is_inkling_exe(exe) -> bool:
    return _base(exe).startswith(EXE_PREFIX)


NO_DRAFTER_NOTE = "Inkling-2-Mini: this engine has no DSpark drafter yet, so it starts without it"
NO_IMAGES_NOTE = "Inkling-2-Mini: this engine does not read images yet, so it starts without the image reader"
NO_IMAGES_REPLY = "this Inkling-2-Mini engine does not read images yet; update Strata or attach the Vision shard"
_PROBES: dict = {}


def wants_drafter(cfg: dict) -> bool:
    from strata_control import inkling_family
    choice = str(cfg.get("drafting") or inkling_family.DEFAULT_DRAFTER).lower()
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
        done = subprocess.run([str(exe), "--info", str(model)], capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env, cwd=str(Path(str(exe)).parent))
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    return ["--draft", "dspark"] if wants_drafter(cfg) and (info or {}).get("drafter") == "dspark" else []


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """Build start args for `strata-inkling`, preserving `--parallel 1..64`, `--max-context`, `--vision`, and `--draft dspark`."""
    from strata_control import inkling_family, user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    # Propagate top-level cfg["parallel"] (1..64) into `--parallel <N>` when not explicitly in args
    par = cfg.get("parallel")
    if isinstance(par, int) and 1 <= par <= inkling_family.MAX_PARALLEL and "--parallel" not in args:
        if par > 1:
            args += ["--parallel", str(par)]
    # Propagate 2-GPU selection (`cfg["gpu"] = [0, 1]`) into `--gpus 2` for dual-GPU Expert Parallelism
    gpu_cfg = cfg.get("gpu")
    if isinstance(gpu_cfg, (list, tuple)) and len(gpu_cfg) >= 2 and "--gpus" not in args:
        args += ["--gpus", str(min(2, len(gpu_cfg)))]
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
        del args[at:at + 2]
        if announce:
            print(f"[strata] {NO_IMAGES_NOTE}", flush=True)
    return args


def encode_path(path) -> str:
    return str(path).replace("%", "%25").replace(" ", "%20")


def decode_path(text: str) -> str:
    return str(text).replace("%20", " ").replace("%25", "%")


def write_images(sources, load, normalize, cancel=None, folder=None) -> list[str]:
    """Write normalized request images under unique per-request names (`strata-inkling-images`)."""
    import hashlib
    import secrets
    import tempfile
    base = Path(folder) if folder else Path(tempfile.gettempdir()) / "strata-inkling-images"
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
    from serve.server_parts.core import top_logprobs
    if getattr(engine, "inkling_engine", False) is True and top_logprobs(req) is not None:
        raise ValueError(REFUSED["logprobs"])


def literal_tags(tokenizer) -> list[str]:
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
