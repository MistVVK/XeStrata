"""serve/images.py - image generation (Qwen-Image-2.1-Turbo) through stable-diffusion.cpp's sd-cli.

POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url, background)
plus Strata's own fields seed and steps.  Each request runs one `sd-cli` process with the arguments the model's card
gives for stable-diffusion.cpp (8 steps, CFG 1, euler, Turbo's sigmas, --diffusion-fa).  The model writes RGBA; by
default each picture is put on white and returned as an 8-bit RGB PNG with sd-cli's other chunks (its "parameters"
text) unchanged (flatten_png); background "transparent" returns the PNG exactly as sd-cli wrote it.

The GPU is shared with the text engine one job at a time: an image job takes the service's turn (the FIFO every chat
request takes), so no chat turn runs on the GPU meanwhile, and the config's "gpu_policy" says what happens to the
text engine during the job (gpu_turn):
  keep    - it stays loaded and idle (an APU's shared memory, a card with room for both)
  shrink  - an engine started with --vram-elastic gives the image job VRAM from its expert cache and takes it back
            afterwards (POST /v1/vram's mechanism; the conversation stays in the engine)
  unload  - its conversation is saved to a session file, the engine is stopped, and the next start restores it
  auto    - (default) keep when there is room for the image job next to it, else shrink when the engine can,
            else unload
The process per request keeps nothing on the GPU between images: when the job ends, all of its memory is free again.

The config's "images" section (written by setup --images, tools/image/install.py):
  {"exe": "engine-image/sd-cli", "backend": "vulkan", "diffusion_model": "...Q8_0.gguf", "vae": "...",
   "llm": "...Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "quant": "Q8_0", "args": ["--diffusion-fa"],
   "gpu_policy": "auto", "reload": "lazy"|"eager", "max_n": 4, "max_side": 2048, "timeout_s": 3600,
   "lib_dirs": [...], "uma": false, "need_mib": null, "out_dir": null, "keep_files": 64,
   "native": false}   (native: Strata's own DiT runner, tools/image/sdcpp_native; CUDA / HIP builds)
"""
from __future__ import annotations

import base64
import contextlib
import os
import random
import re
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from strata_control.childenv import quiet_crash_env

# Qwen-Image-2.1-Turbo's own schedule (model_index.json "sample_sigmas") plus the terminal 0, as Unsloth's card passes
# it to sd-cli; it is an 8-step schedule, so another step count leaves the schedule to sd.cpp
TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
TURBO_STEPS = 8
MODEL_NAME = "qwen-image-2.1-turbo"
POLICIES = ("auto", "keep", "shrink", "unload")
SIZE_STEP = 32                     # sd.cpp's Qwen-Image-2.1 docs: dimensions divisible by 32
MIN_SIDE = 256
FILE_NAME = re.compile(r"^[0-9a-f]{32}\.png$")
STEP_LINE = re.compile(r"\|\s*(\d+)/(\d+)\s*-")      # sd.cpp's progress bar: "|====    | 3/8 - 14.1s/it"


class ImageRequestError(ValueError):
    """A request the image endpoint refuses (400)."""


class ImageJobFailed(RuntimeError):
    """sd-cli ended without the images (500); the message carries the end of its log."""


def parse_size(value, max_side: int) -> tuple[int, int]:
    """OpenAI's size, "1024x1024" (or "auto": 1024x1024) -> (width, height); each side a multiple of 32 from 256 to
    max_side."""
    if value is None or value == "auto":
        return 1024, 1024
    if not isinstance(value, str):
        raise ImageRequestError("size: a string WIDTHxHEIGHT, e.g. \"1024x1024\"")
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*", value)
    if not m:
        raise ImageRequestError(f"size: WIDTHxHEIGHT, e.g. \"1024x1024\", not {value!r}")
    w, h = int(m.group(1)), int(m.group(2))
    for side in (w, h):
        if side < MIN_SIDE or side > max_side or side % SIZE_STEP:
            raise ImageRequestError(f"size {w}x{h}: each side a multiple of {SIZE_STEP} from {MIN_SIDE} to {max_side}")
    return w, h


def _int_field(req: dict, key: str, lo: int, hi: int, default):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ImageRequestError(f"{key}: a whole number from {lo} to {hi}")
    return v


def parse_request(req: dict, cfg: dict) -> dict:
    """The fields of one /v1/images/generations request, checked.  Unknown OpenAI fields (quality, style, user,
    moderation) are accepted and ignored; an output format other than PNG is refused.  OpenAI's background:
    "transparent" keeps the model's own alpha; "opaque", "auto" or none puts the picture on white (flatten_png)."""
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageRequestError("prompt: a non-empty string")
    if len(prompt) > 16000:
        raise ImageRequestError("prompt: at most 16000 characters")
    if "\0" in prompt:
        raise ImageRequestError("prompt: no NUL characters")
    fmt = req.get("response_format") or "b64_json"
    if fmt not in ("b64_json", "url"):
        raise ImageRequestError("response_format: b64_json or url")
    out_fmt = req.get("output_format")
    if out_fmt not in (None, "png"):
        raise ImageRequestError("output_format: only png")
    background = req.get("background") or "auto"
    if background not in BACKGROUNDS:
        raise ImageRequestError("background: transparent, opaque or auto")
    w, h = parse_size(req.get("size"), int(cfg.get("max_side") or 2048))
    n = _int_field(req, "n", 1, int(cfg.get("max_n") or 4), 1)
    steps = _int_field(req, "steps", 1, 100, int(cfg.get("steps") or TURBO_STEPS))
    seed = _int_field(req, "seed", 0, 2**62, None)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    return {"prompt": prompt, "width": w, "height": h, "n": n, "steps": steps, "seed": seed, "response_format": fmt,
            "background": background}


def write_prompt(work: Path, prompt: str) -> Path:
    """The job's prompt as a file for sd-cli's --prompt-file, readable by this user only (#4202).  Not on the command line:
    every local account can read a process's arguments (ps, /proc/<pid>/cmdline).  Written as UTF-8 without newline
    translation and read back by sd-cli as raw bytes, so the prompt is unchanged."""
    path = work / "prompt.txt"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8", newline="") as f:
        f.write(prompt)
    return path


def sd_cli_args(cfg: dict, job: dict, out_pattern: str, prompt_file: str) -> list[str]:
    """The sd-cli command for one job: the model card's stable-diffusion.cpp command (8 steps, CFG 1, euler, Turbo's
    sigmas, --diffusion-fa from the config's args), the prompt from `prompt_file` (write_prompt), the size, seed and
    batch count, and the config's other args."""
    cmd = [str(cfg["exe"]),
           "--diffusion-model", str(cfg["diffusion_model"]), "--vae", str(cfg["vae"]), "--llm", str(cfg["llm"]),
           "--prompt-file", prompt_file,
           "--steps", str(job["steps"]), "--cfg-scale", "1.0", "--sampling-method", "euler",
           "-W", str(job["width"]), "-H", str(job["height"])]
    if job["steps"] == TURBO_STEPS:
        cmd += ["--sigmas", str(cfg.get("sigmas") or TURBO_SIGMAS)]
    cmd += ["-s", str(job["seed"])]
    if job["n"] > 1:
        cmd += ["-b", str(job["n"])]
    cmd += ["-o", out_pattern]
    cmd += [str(a) for a in (cfg.get("args") or [])]
    if cfg.get("native") is True:                   # Strata's DiT runner inside sd-cli (CUDA / HIP builds)
        cmd += ["--model-args", "qwen_image_2_1_native=1"]
    return cmd


def image_env(cfg: dict, base: dict | None = None) -> dict:
    """sd-cli's environment: the libraries its backend needs (ROCm's, setup's lib_dirs) first on the search path, the
    GPU to use when the config names one, and the config's own "env"."""
    env = quiet_crash_env(dict(os.environ if base is None else base))      # ggml's backtrace starts gdb: never on a user's machine
    dirs = [str(d) for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    gpu = cfg.get("gpu")
    if gpu is not None:
        var = {"cuda": "CUDA_VISIBLE_DEVICES", "hip": "HIP_VISIBLE_DEVICES",
               "vulkan": "GGML_VK_VISIBLE_DEVICES"}.get(cfg.get("backend"))
        if var:
            # one card is "1"; a list ([1, 0], "1,0") is every one of them, in order - str() of a list is not a device list
            env[var] = ",".join(str(x).strip() for x in gpu if str(x).strip() != "") if isinstance(gpu, (list, tuple)) else str(gpu)
            if var == "CUDA_VISIBLE_DEVICES":
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


# ------------------------------------------------------------------------------- the model's transparency
# Qwen-Image-2.1's VAE decodes RGBA: the model leaves some backgrounds transparent, with a flat key colour (often
# purple) under alpha 0.  By default the picture is put on white and written as an 8-bit RGB PNG; every chunk other
# than IHDR / IDAT (sd-cli's "parameters" text above all) is copied byte for byte.  background: "transparent" keeps
# sd-cli's PNG unchanged.
BACKGROUNDS = ("auto", "opaque", "transparent")
PNG_SIG = b"\x89PNG\r\n\x1a\n"
FLATTEN_ZLIB_LEVEL = 6      # 1024x1024: 0.03 s, 1536x1024: 0.14 s to encode, the smallest file of levels 1/3/6


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """[(type, the whole chunk's bytes)] of a PNG; ValueError on anything malformed."""
    import struct
    import zlib
    if data[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    out, i = [], 8
    while i < len(data):
        if i + 12 > len(data):
            raise ValueError("truncated chunk")
        n = struct.unpack(">I", data[i:i + 4])[0]
        end = i + 12 + n
        if end > len(data):
            raise ValueError("truncated chunk")
        t = data[i + 4:i + 8]
        if struct.unpack(">I", data[end - 4:end])[0] != zlib.crc32(data[i + 4:end - 4]) & 0xFFFFFFFF:
            raise ValueError(f"bad CRC in {t!r}")
        out.append((t, data[i:end]))
        i = end
        if t == b"IEND":
            break
    if not out or out[0][0] != b"IHDR" or out[-1][0] != b"IEND":
        raise ValueError("no IHDR / IEND")
    return out


def _rgba_pixels(data: bytes, w: int, h: int, cs):
    """The RGBA pixels as a uint8 array [h, w, 4]: Pillow when it imports, else zlib + numpy (PNG filters undone
    row by row; Average / Paeth rows go pixel by pixel, about 1-3 s for a large picture)."""
    import numpy as np
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            if im.mode == "RGBA":
                a = np.asarray(im)
                if a.shape == (h, w, 4):
                    return a
    except Exception:  # noqa: BLE001 - a missing or broken Pillow: the fallback below
        pass
    import zlib
    raw = zlib.decompress(b"".join(c[8:-4] for t, c in cs if t == b"IDAT"))
    stride = w * 4
    if len(raw) != h * (1 + stride):
        raise ValueError("IDAT size")
    rows = np.frombuffer(raw, np.uint8).reshape(h, 1 + stride)
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    for y in range(h):
        f, line = int(rows[y, 0]), rows[y, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 1:
            cur = np.cumsum(line.reshape(w, 4), axis=0).reshape(-1) & 255
        elif f == 2:
            cur = (line + prev) & 255
        elif f in (3, 4):
            ln, pv = line.tolist(), prev.tolist()
            cl = [0] * stride
            for x in range(stride):
                a = cl[x - 4] if x >= 4 else 0
                b = pv[x]
                if f == 3:
                    cl[x] = (ln[x] + ((a + b) >> 1)) & 255
                else:
                    c = pv[x - 4] if x >= 4 else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    cl[x] = (ln[x] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))) & 255
            cur = np.array(cl, np.int32)
        else:
            raise ValueError(f"PNG filter {f}")
        out[y] = cur
        prev = cur
    return out.reshape(h, w, 4)


def png_has_alpha(data: bytes) -> bool:
    """An alpha channel by the IHDR's colour type (4 grey + alpha, 6 RGBA); False for anything unreadable."""
    return len(data) > 25 and data[:8] == PNG_SIG and data[12:16] == b"IHDR" and data[25] in (4, 6)


def flatten_png(data: bytes) -> bytes:
    """An 8-bit RGBA PNG -> the same picture on white as an 8-bit RGB PNG (round(rgb * a + 255 * (1 - a)), a in
    [0, 1]), every chunk other than IHDR / IDAT copied byte for byte.  Any other PNG comes back unchanged.
    Raises on a malformed PNG (the caller then keeps the original)."""
    import struct
    import zlib

    import numpy as np
    cs = _png_chunks(data)
    w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", cs[0][1][8:21])
    if ctype != 6 or depth != 8:
        return data                                  # RGB (or anything else): nothing to flatten
    if comp or filt or inter:
        raise ValueError("unsupported PNG layout")
    rgba = _rgba_pixels(data, w, h, cs)
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = np.floor(rgba[..., :3].astype(np.float64) * a + 255.0 * (1.0 - a) + 0.5)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()   # filter 0 rows

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    out, idat_done = [PNG_SIG], False
    for t, c in cs:
        if t == b"IHDR":
            out.append(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        elif t == b"IDAT":
            if not idat_done:
                out.append(chunk(b"IDAT", zlib.compress(raw, FLATTEN_ZLIB_LEVEL)))
                idat_done = True
        else:
            out.append(c)                            # tEXt "parameters", IEND, ...: byte for byte
    return b"".join(out)


TIMING = re.compile(r"(get_learned_condition|sampling|decode_first_stage) completed, taking ([0-9]+(?:\.[0-9]+)?) ?(ms|s)\b")


def log_figures(log: str) -> dict:
    """The job's stage times from sd-cli's log: text_encode_ms, sampling_s, vae_decode_s (the last of each line).
    stable-diffusion.cpp 2988060 (the pinned commit) prints seconds for all three ("taking 0.70s"); the "N ms" form is
    read as well."""
    out = {}
    for stage, num, unit in TIMING.findall(log):
        secs = float(num) / 1000 if unit == "ms" else float(num)
        key, val = {"get_learned_condition": ("text_encode_ms", secs * 1000), "sampling": ("sampling_s", secs),
                    "decode_first_stage": ("vae_decode_s", secs)}[stage]
        out[key] = round(val, 3)
    return out


def need_mib(cfg: dict, width: int, height: int) -> int:
    """The GPU memory an image job is planned with: the three model files plus the working buffers (about 2 GiB plus
    4 GiB per megapixel at Turbo's settings; the 780M baseline peaked 7.4 GB over the weights at 1024x1024 on HIP);
    the config's "need_mib" overrides it."""
    if cfg.get("need_mib"):
        return int(cfg["need_mib"])
    files = 0
    for k in ("diffusion_model", "vae", "llm"):
        try:
            files += Path(cfg[k]).stat().st_size
        except (OSError, KeyError, TypeError):
            pass
    work = (2 << 30) + int(4 * (1 << 30) * width * height / (1024 * 1024))
    return (files + work) >> 20


class ImageGen:
    """One sd-cli process per request, one request at a time (the service's turn, gpu_turn)."""

    def __init__(self, cfg: dict, log=None):
        self.cfg = dict(cfg)
        policy = self.cfg.get("gpu_policy") or "auto"
        if policy not in POLICIES:
            raise ValueError(f'images "gpu_policy" must be one of {", ".join(POLICIES)}, not {policy!r}')
        reload = self.cfg.get("reload") or "lazy"
        if reload not in ("lazy", "eager"):
            raise ValueError(f'images "reload" must be lazy or eager, not {reload!r}')
        for k in ("exe", "diffusion_model", "vae", "llm"):
            if not isinstance(self.cfg.get(k), str) or not self.cfg[k]:
                raise ValueError(f'the config\'s "images" section needs "{k}" (run setup with --images again)')
        self.policy, self.reload = policy, reload
        self.model = self.cfg.get("model_name") or MODEL_NAME
        self.log = log
        out = self.cfg.get("out_dir")
        self.dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="strata-images-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_files = int(self.cfg.get("keep_files") or 64)
        self.lock = threading.Lock()              # one sd-cli at a time even when the service's turn is not taken
        self.totals = {"images": 0, "jobs": 0, "seconds": 0.0, "failed": 0}
        self.last = None                           # the last job's figures (/v1/status)
        self.progress = None                       # (step, steps) of the running job

    def missing(self) -> list[str]:
        return [self.cfg[k] for k in ("exe", "diffusion_model", "vae", "llm") if not Path(self.cfg[k]).exists()]

    def _say(self, text: str) -> None:
        print(f"[strata] {text}", flush=True)
        if self.log is not None:
            try:
                self.log.write(f"[images] {text}\n")
                self.log.flush()
            except (OSError, ValueError):
                pass

    def _prune(self) -> None:
        """Keep the newest keep_files PNGs of response_format url (the oldest go)."""
        files = sorted((p for p in self.dir.glob("*.png") if FILE_NAME.match(p.name)), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - self.keep_files)]:
            with contextlib.suppress(OSError):
                p.unlink()

    def file(self, name: str) -> Path | None:
        """A stored image (response_format url) by its name, or None."""
        if not FILE_NAME.match(name or ""):
            return None
        p = self.dir / name
        return p if p.is_file() else None

    def run(self, job: dict, cancel: threading.Event | None = None) -> tuple[list[bytes], dict]:
        """Run sd-cli for `job` (parse_request's dict) -> (the PNG files' bytes in order, figures).  The caller holds
        the GPU turn.  ImageJobFailed when sd-cli fails, times out or is cancelled."""
        work = Path(tempfile.mkdtemp(prefix="job-", dir=self.dir))
        pattern = str(work / "img_%d.png")
        try:
            cmd = sd_cli_args(self.cfg, job, pattern, str(write_prompt(work, job["prompt"])))
        except BaseException:                       # the prompt file was not made: nothing else is in the folder yet
            for p in work.iterdir():
                with contextlib.suppress(OSError):
                    p.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
            raise
        timeout = float(self.cfg.get("timeout_s") or 3600)
        logf = work / "sd-cli.log"
        t0 = time.monotonic()
        self.progress = (0, job["steps"])
        with self.lock:
            with open(logf, "w", encoding="utf-8", errors="replace") as lf:
                try:
                    from serve import lifeline
                    proc = lifeline.popen(cmd, "image", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=image_env(self.cfg),
                                          cwd=str(work), text=True, errors="replace", bufsize=1)
                except OSError as e:
                    raise ImageJobFailed(f"sd-cli did not start ({cmd[0]}): {e}") from e
                why = None

                def pump():
                    for line in proc.stdout:
                        lf.write(line)
                        m = STEP_LINE.search(line)
                        if m:
                            self.progress = (int(m.group(1)), int(m.group(2)))
                reader = threading.Thread(target=pump, daemon=True)
                reader.start()
                while proc.poll() is None:
                    if cancel is not None and cancel.is_set():
                        why = "cancelled (the client left)"
                    elif time.monotonic() - t0 > timeout:
                        why = f"it ran past timeout_s ({timeout:g} s)"
                    if why:
                        proc.kill()
                        break
                    time.sleep(0.2)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                reader.join(timeout=10)
        secs = time.monotonic() - t0
        self.progress = None
        log = logf.read_text(encoding="utf-8", errors="replace")
        outs = [work / f"img_{i}.png" for i in range(job["n"])]
        try:
            if why or proc.returncode != 0 or not all(p.is_file() for p in outs):
                self.totals["failed"] += 1
                tail = "\n".join(log.strip().splitlines()[-12:])
                what = why or (f"sd-cli exited with code {proc.returncode}" if proc.returncode
                               else "sd-cli wrote no image")
                self._say(f"image job failed: {what}\n{tail}")
                raise ImageJobFailed(f"{what}; the end of its log:\n{tail}")
            images = [p.read_bytes() for p in outs]
        finally:
            for p in work.iterdir():
                with contextlib.suppress(OSError):
                    p.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
        figures = {"seconds": round(secs, 2), "steps": job["steps"], "n": job["n"],
                   "size": f"{job['width']}x{job['height']}", "seed": job["seed"],
                   "backend": self.cfg.get("backend"), "quant": self.cfg.get("quant")}
        figures.update(log_figures(log))
        if job.get("background", "auto") != "transparent":
            t1 = time.monotonic()
            flat = [self._flatten(png) for png in images]
            images = [png for png, _ in flat]
            job["alpha_kept"] = any(kept for _, kept in flat)     # a picture went out with its alpha after all
            figures["flatten_s"] = round(time.monotonic() - t1, 3)
        self.totals["jobs"] += 1
        self.totals["images"] += job["n"]
        self.totals["seconds"] += secs
        self.last = figures
        return images, figures

    def _flatten(self, png: bytes) -> tuple[bytes, bool]:
        """flatten_png, never failing the request -> (the picture, whether it still has an alpha channel).  On any
        error the picture goes out as sd-cli wrote it."""
        try:
            out = flatten_png(png)
            return out, png_has_alpha(out)
        except Exception as e:  # noqa: BLE001 - numpy missing, a malformed PNG, ...: the original picture
            self._say(f"image: the picture keeps its transparency (it could not be put on white: {e})")
            return png, png_has_alpha(png)

    def store(self, data: bytes) -> str:
        """Keep one PNG for response_format url -> its file name (32 random hex digits: the name is the access)."""
        name = secrets.token_hex(16) + ".png"
        (self.dir / name).write_bytes(data)
        self._prune()
        return name

    def response(self, images: list[bytes], job: dict, figures: dict, base_url: str) -> dict:
        data = []
        for i, png in enumerate(images):
            item = {"revised_prompt": job["prompt"], "seed": job["seed"] + i}
            if job["response_format"] == "url":
                item["url"] = f"{base_url}/v1/images/files/{self.store(png)}"
            else:
                item["b64_json"] = base64.b64encode(png).decode("ascii")
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": self.model,
                "output_format": "png", "size": f"{job['width']}x{job['height']}", "strata": figures,
                "background": "transparent" if job.get("background") == "transparent" or job.get("alpha_kept")
                else "opaque"}

    def status(self) -> dict:
        out = {"enabled": True, "model": self.model, "backend": self.cfg.get("backend"),
               "quant": self.cfg.get("quant"), "gpu_policy": self.policy, **self.totals}
        if self.progress:
            out["step"], out["steps"] = self.progress
        if self.last:
            out["last"] = self.last
        return out


# ------------------------------------------------------------------------------------------- the GPU turn
def resolve_policy(svc, gen: ImageGen, need: int) -> str:
    """auto -> keep / shrink / unload for this job (see the module's docstring); a fixed policy is kept, except that
    an engine that is not running has nothing to give back (keep)."""
    eng = svc.engine
    running = not hasattr(eng, "alive") or eng.alive()
    if not running or not hasattr(eng, "unload"):
        return "keep"
    if gen.policy != "auto":
        return gen.policy
    if gen.cfg.get("uma"):
        return "keep"                         # an APU: the image job and the engine share the RAM either way
    free = svc.free_vram_mib()
    if free is None or free >= need:
        return "keep"
    if engine_elastic(eng):
        return "shrink"
    return "unload"


def engine_elastic(eng) -> bool:
    """An engine that resizes its expert cache on request (--vram-elastic, #533)."""
    if not hasattr(eng, "vram"):
        return False
    if (getattr(eng, "info", None) or {}).get("vram_elastic"):
        return True
    spawn = getattr(eng, "spawn", None)
    args = spawn[1] if isinstance(spawn, (list, tuple)) and len(spawn) > 1 else []
    return "--vram-elastic" in (args or [])


def _client_gone(svc) -> Exception:
    """The server module's ClientGone (it may run as __main__, so not imported by name)."""
    import sys
    return getattr(sys.modules.get(type(svc).__module__), "ClientGone", RuntimeError)()


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used."""
    with svc.status_lock:
        svc.status["queued"] += 1
    queued = True
    try:
        with svc._turn(cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
            queued = False
            ctl = getattr(svc.engine, "ctl", None) if getattr(svc.engine, "batch", 0) else None
            if ctl is not None:
                while not ctl.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        raise _client_gone(svc)
            try:
                if ctl is not None:                   # running parallel requests finish first
                    while getattr(svc, "live_reqs", None):
                        if cancel is not None and cancel.is_set():
                            raise _client_gone(svc)
                        time.sleep(0.25)
                need = need_mib(gen.cfg, job["width"], job["height"])
                policy = resolve_policy(svc, gen, need)
                with svc.status_lock:
                    svc.status.update(busy=True, phase="generating an image", started=time.time(),
                                      first_token=None, prompt_tokens=None, generated=None, max_tokens=None)
                restore = _make_room(svc, gen, policy, need)
                try:
                    yield policy
                finally:
                    _give_back(svc, gen, policy, restore)
                    with svc.status_lock:
                        svc.status["busy"] = False
                    svc.last_request_at = time.time()
            finally:
                if ctl is not None:
                    ctl.release()
    finally:
        if queued:
            with svc.status_lock:
                svc.status["queued"] -= 1


def _make_room(svc, gen: ImageGen, policy: str, need: int):
    """Do what `policy` says before the job; -> what _give_back needs."""
    eng = svc.engine
    if policy == "shrink":
        try:
            out = eng.vram(need)
            gen._say(f"image job: the engine's expert cache gave VRAM back ({out.get('vram_free_mib')} MiB free)")
            return "shrunk"
        except Exception as e:                       # noqa: BLE001 - the job still runs, with what is free
            gen._say(f"image job: the engine could not give VRAM back ({e}); the job runs with what is free")
            return None
    if policy == "unload":
        saved = None
        if hasattr(eng, "session_file") and not getattr(eng, "batch", 0):
            path = str(gen.dir / f"llm-session-{secrets.token_hex(6)}.bin")
            try:
                r = eng.session_file("save", path)
                saved = path
                gen._say(f"image job: the conversation ({r.get('tokens')} tokens) saved; unloading the model")
            except Exception as e:                   # noqa: BLE001 - unloading still frees the GPU
                gen._say(f"image job: the conversation could not be saved ({e}); unloading the model")
                with contextlib.suppress(OSError):
                    os.unlink(path)
        eng.unload()
        if svc.vision is not None and hasattr(svc.vision, "unload"):
            svc.vision.unload()
        return saved or "unloaded"
    return None


def _give_back(svc, gen: ImageGen, policy: str, restore) -> None:
    eng = svc.engine
    if restore == "shrunk":
        try:
            eng.vram(svc.vram_reserve)               # the reserve asked for last (None: the start's)
        except Exception as e:                       # noqa: BLE001
            gen._say(f"image job: the expert cache did not grow back ({e}); POST /v1/vram can retry")
    elif policy == "unload" and restore:
        if restore != "unloaded":
            svc.restore_after_load = restore         # Service.ensure_loaded restores it after the next start
        if gen.reload == "eager":
            try:
                svc.ensure_loaded()                  # the caller still holds the FIFO
            except Exception as e:                   # noqa: BLE001 - the next request tries again
                gen._say(f"image job: the model did not load again yet ({e}); the next request loads it")


def restore_session(svc) -> None:
    """After an engine start: restore the conversation an image job saved before it unloaded the engine."""
    path = getattr(svc, "restore_after_load", None)
    if not path:
        return
    svc.restore_after_load = None
    try:
        r = svc.engine.session_file("restore", path)
        print(f"[strata] the conversation from before the image job is back ({r.get('tokens')} tokens)", flush=True)
    except Exception as e:                           # noqa: BLE001 - the engine runs, without the old cache
        print(f"[strata] the conversation from before the image job was not restored ({e}); the next request "
              "reads its prompt again", flush=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
