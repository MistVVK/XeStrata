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
    vision = next((args[i + 1] for i, a in enumerate(args) if a == "--vision" and i + 1 < len(args)), None)
    dspark = next((args[i + 1] for i, a in enumerate(args) if a == "--dspark" and i + 1 < len(args)), None)
    try:
        st = Path(str(exe)).stat()
        key = (str(exe), st.st_mtime_ns, st.st_size, str(model), str(vision or ""), str(dspark or ""))
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
    cmd = [str(exe), "--info", str(model)]
    if vision:
        cmd += ["--vision", str(vision)]
    if dspark:
        cmd += ["--dspark", str(dspark)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=timeout, env=env, cwd=str(Path(str(exe)).parent))
        info = parse_info(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info = None
    _PROBES[key] = info or None
    return _PROBES[key]


def drafter_args(cfg: dict, info: dict | None = None) -> list[str]:
    if not wants_drafter(cfg):
        return ["--draft", "off"]
    return ["--draft", "dspark"] if (info or {}).get("drafter") == "dspark" else []


def drafting_info(args: list, info: dict | None = None) -> dict[str, str]:
    """Metrics/status view of Inkling-2-Mini's built-in DSpark drafter (`drafting`: `"dspark"` | `"off"`)."""
    items = [str(a) for a in (args or ())]
    mode = None
    for i in range(len(items) - 1):
        if items[i] == "--draft":
            mode = items[i + 1].strip().lower()
    if mode == "off":
        return {"drafting": "off", "drafting_note": ""}
    if mode == "dspark" or (info or {}).get("drafter") == "dspark":
        return {"drafting": "dspark", "drafting_note": ""}
    return {"drafting": "off", "drafting_note": ""}


def _local_gpu_ordinals(n_gpus: int, max_g: int) -> str:
    """Map Helper..Main visibility order (`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`ZE_AFFINITY_MASK`)
    to Main-first local ordinals for `strata-inkling --gpus` (`gpu_ids[0]` is primary/Main)."""
    use = min(max_g, n_gpus)
    main = n_gpus - 1
    helpers = list(range(n_gpus - use, main))
    return ",".join(str(i) for i in [main, *helpers])


def _strip_flag(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else 1
            continue
        out.append(args[i])
        i += 1
    return out


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """Build start args for `strata-inkling`, preserving `--parallel 1..64`, `--max-context`, `--vision`,
    Main-first `--gpus` + `--layer-split`, and `--draft dspark|off`."""
    from strata_control import inkling_family, user_args
    args = list(cfg["args"])
    mine = user_args.capture(cfg)
    if mine:
        args = user_args.merge(args, mine)
    # #6497: honor a configured CPU thread limit (`--pool-workers N`) on Inkling by translating it to `--threads N`.
    if "--pool-workers" in args:
        pw_val = None
        out_args: list[str] = []
        i = 0
        while i < len(args):
            if args[i] == "--pool-workers":
                if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                    pw_val = str(args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                out_args.append(args[i])
                i += 1
        args = out_args
        if pw_val and pw_val.isdigit() and int(pw_val) > 0 and "--threads" not in args:
            args += ["--threads", pw_val]
    # Propagate top-level cfg["parallel"] (1..64) into `--parallel <N>` when not explicitly in args
    par = cfg.get("parallel")
    if isinstance(par, int) and 1 <= par <= inkling_family.MAX_PARALLEL and "--parallel" not in args:
        if par > 1:
            args += ["--parallel", str(par)]
    # Propagate 2..4 GPU selection (`cfg["gpu"]` or `gpu_roles`) into Main-first `--gpus` and `--layer-split` (#6592, #6631).
    # child_env() exposes GPUs in Helper..Main order, so local ordinal N-1 is Main and 0..N-2 are Helpers;
    # strata-inkling puts gpu_ids[0] as primary GPU, so pass Main first (`1,0`, `2,0,1`, `3,0,1,2`).
    from serve.server_parts.engines import gpu_list
    n_gpus = len(gpu_list(cfg))
    is_cuda = cfg.get("backend") not in ("hip", "sycl", "metal", "cpu")
    mine_flags = {str(a) for a in (mine or ()) if str(a).startswith("--")}
    if n_gpus >= 2:
        max_g = 4 if is_cuda else 2
        local_spec = _local_gpu_ordinals(n_gpus, max_g)
        if "--gpus" not in args:
            args += ["--gpus", local_spec]
        elif "--gpus" not in mine_flags:
            at = args.index("--gpus")
            if at + 1 < len(args) and "," not in str(args[at + 1]):
                args[at + 1] = local_spec
        if is_cuda and "--layer-split" not in args and "--ep-split" not in args:
            args += ["--layer-split", str(cfg["layer_split"]) if cfg.get("layer_split") else "auto"]
        # Drop single-GPU expert caps when running multi-GPU unless the user explicitly pinned them in user_args.
        if "--vram-experts-gb" not in mine_flags:
            args = _strip_flag(args, "--vram-experts-gb")
        if "--ram-experts-gb" not in mine_flags:
            args = _strip_flag(args, "--ram-experts-gb")
    wants_images = "--vision" in args
    want_draft = wants_drafter(cfg)
    need_draft = "--draft" not in args and want_draft
    info = probe(exe, args, cfg) if (exe and (wants_images or need_draft)) else None
    if "--draft" not in args and not want_draft:
        args += ["--draft", "off"]
    elif need_draft:
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
