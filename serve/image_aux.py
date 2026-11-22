# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/image_aux.py - the image tools' auxiliary weights (a detector, an upscaler...), downloaded the first time a
tool needs one, or by setup ahead (fetch, setup_names: the upscale's and the parts' fix), never bundled.

The weights are tools/image/qimage.py's AUX entries: each names a FILES entry (its repository, path, size and SHA-256
at the revision REVISIONS pins), what converts it for the engine (AUX "convert": "yolo" runs tools/image/yolo_pt.py,
"esrgan" tools/image/esrgan.py, "state" tools/image/state_pt.py, in a process of its own, with a time limit) and the
file the engine reads, in AUX_DIR beside the image models.

    status = image_aux.ensure("yolo-face", aux_dir, say)

ensure() returns at once: {"status": "ready", "path": ...} when the file is there; else it starts the download on a
thread of its own (one per weight, however many requests ask) and returns {"status": "downloading", "name", "bytes",
"total", "license", "source", "retry_s"}; a download that failed is {"status": "failed", "error", ...} until it is
asked for again, which starts it over.  A weight whose license the user must agree to first (AUX "consent": SAM 3's)
is {"status": "consent", ...} and fetches nothing until accept() has recorded the agreement (<name>.accepted in the
folder, which setup writes too); license_text() fetches the license's own text to show before that.  The caller (a
tool) tells the chat so and does not wait: the chat's model calls it again later, in a new request.  Every step goes
to the server's log through `say` (start, progress about every 10 seconds, the check, the conversion, the end or the
reason it stopped).

The file is fetched into <name>.part (resumed from its length after an interruption), checked against the pinned size
and SHA-256 and only then converted, into a temporary file renamed into place: a reader never sees a part of one.  A
lock file beside it (<out>.lock, held with flock on Linux, msvcrt on Windows) keeps two servers, or setup, from
fetching the same file at once: the one that does not get it reports "downloading" until the other is done.  No URL or
path comes from a request: the names are AUX's keys, the host Hugging Face's (HF_ENDPOINT, as setup reads it).
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from serve import download as download_core
from serve import file_lock, model_terms
from serve.model_terms import _qimage

RETRY_S = 30            # what the tool's result suggests before the chat asks again
SEEDVR2_WORK_MIB = 4096   # what a 7B run takes besides its weights at a couple of megapixels (the DiT's activations)
CONVERT_TIMEOUT_S = 600
CHUNK = 1 << 20
CONVERTERS = {"yolo": "yolo_pt.py", "esrgan": "esrgan.py", "state": "state_pt.py"}   # AUX "convert" -> its tool
# Qwen-Image's ControlNet takes no color, normal or segmentation map: image_controls.mode_of uses the same list
CONTROLNET_REFUSED_KINDS = ("color", "normal", "segmentation")

_lock = threading.Lock()
_jobs: dict[str, dict] = {}   # name -> its download's state, while one runs or after it failed


def endpoint() -> str:
    return download_core.hf_endpoint()


def aux_dir(cfg: dict) -> Path:
    """Where an image model's config keeps the tools' weights: its "aux_dir", else AUX_DIR beside its folders."""
    if cfg.get("aux_dir"):
        return Path(cfg["aux_dir"])
    base = cfg.get("vae") or cfg.get("diffusion_model") or "."
    # not resolve(): a model file linked from another folder (mage-flow's VAE into comfy/vae) would move the weights
    # with the model in use, away from setup's models/image-tools
    return Path(os.path.abspath(base)).parent.parent / _qimage().AUX_DIR


def info(name: str) -> dict:
    """What a weight is: its file's source page, size and license."""
    q = _qimage()
    a = q.AUX[name]
    repo, path, size, _ = q.FILES[a["file"]]
    return {"name": name, "total": size, "license": a["license"],
            "source": f"https://huggingface.co/{repo}/blob/{q.REVISIONS[repo]}/{path}"}


def path(name: str, root: Path) -> Path:
    return Path(root) / _qimage().AUX[name]["out"]


def consent_mark(name: str, root: Path) -> Path:
    return Path(root) / f"{name}.accepted"


def needs_consent(name: str, root: Path) -> bool:
    """A weight whose license the user has not agreed to yet."""
    return bool(_qimage().AUX[name].get("consent")) and not consent_mark(name, root).is_file()


def accept(name: str, root: Path, who: str) -> None:
    """Records that the user agreed to `name`'s license (`who`: where, the web page or setup): ensure() fetches it."""
    if not _qimage().AUX[name].get("consent"):
        raise KeyError(name)
    Path(root).mkdir(parents=True, exist_ok=True)
    consent_mark(name, root).write_text(f"{time.strftime('%Y-%m-%d %H:%M')} {who}\n{info(name)['license']}\n",
                                        encoding="utf-8")


def license_text(name: str, root: Path, say: Callable[[str], None] | None = None) -> dict:
    """The text of the license `name` asks consent to: {"status": "ready", "text"}, or its file's download status."""
    lic = _qimage().AUX[name]["consent"]
    got = ensure(lic, root, say)
    if got["status"] != "ready":
        return got
    return {"status": "ready", "name": name, "license": info(name)["license"],
            "text": Path(got["path"]).read_text(encoding="utf-8", errors="replace")}


def ready(name: str, root: Path) -> bool:
    out = path(name, root)
    return out.is_file() and out.with_name(out.name + ".done").is_file()


def status(name: str, root: Path) -> dict:
    """A weight's state without starting anything."""
    if ready(name, root):
        return {"status": "ready", "name": name, "path": str(path(name, root))}
    with _lock:
        job = _jobs.get(name)
        if job is not None:
            return {**info(name), **{k: v for k, v in job.items() if k != "thread"}, "retry_s": RETRY_S}
    return {**info(name), "status": "missing", "bytes": 0}


def ensure(name: str, root: Path, say: Callable[[str], None] | None = None) -> dict:
    """The weight `name` (an AUX key) in `root`: ready, or its download started (or going on) -> its status."""
    q = _qimage()
    if name not in q.AUX:
        raise KeyError(name)
    if ready(name, root):
        return status(name, root)
    if needs_consent(name, root):
        return {**info(name), "status": "consent", "bytes": 0}
    with _lock:
        job = _jobs.get(name)
        if job is None or job["status"] == "failed":
            job = {"status": "downloading", "bytes": 0, "total": info(name)["total"], "started": time.time()}
            _jobs[name] = job
            t = threading.Thread(target=_run, args=(name, Path(root), job, say or (lambda _m: None)),
                                 name=f"image-aux-{name}", daemon=True)
            job["thread"] = t
            t.start()
    return status(name, root)


def fetch(name: str, root: Path, say: Callable[[str], None] | None = None) -> dict:
    """`name` fetched now, waiting for it (setup's --image-tools): ensure(), then its download's end -> its status
    (ready, failed with the reason, or consent)."""
    got = ensure(name, root, say)
    if got["status"] == "downloading":
        with _lock:
            t = (_jobs.get(name) or {}).get("thread")
        if t is not None:
            t.join()
    return status(name, root)


def setup_names(vram_mib: int, ram_mib: int, root: Path, families=(), versions: bool = False) -> list[str]:
    """The weights setup fetches ahead (--image-tools): those the editing tools take by default or fall back to on their
    own, for the image models of `families` (their configs' "family"); with `versions` (--image-tools all) also the
    versions a request chooses.
    - the upscale: SeedVR2's 7B in 8 bits where it fits the VRAM and the RAM together, else its 3B in 8 bits, its VAE,
      and RealESRGAN for auto's fallback; `versions`: the 7B sharp in 8 bits too, where the 7B fits.
    - the parts' fix: the face, hand and person detectors.
    - a region in words: Florence-2 large and base, and SAM 3 and CLIP's tokenizer once its license was agreed to in
      `root`.
    - the background's removal: BiRefNet general and lite (the smaller one it falls back to); `versions`: the others.
    - the control, where a model of `families` takes one: the networks of the kinds it takes (CNET_KINDS), and the
      model's own (Qwen-Image's ControlNet Union, MOSS-Image's adapter and control LoRAs).  Qwen-Image's ControlNet
      refuses normal and segmentation maps, so those two networks are fetched only for a family that takes them.
    serve/images.py seedvr2_file takes the 7B's 8-bit file when it is here and fits, so the server uses these; it would
    fetch the 3B's FP16 file instead only with 10 GiB of VRAM beside less than 2 GiB of RAM."""
    q = _qimage()
    fits_7b = vram_mib + ram_mib >= (q.FILES["seedvr2-7b-int8"][2] >> 20) + SEEDVR2_WORK_MIB
    out = ["seedvr2-7b-int8" if fits_7b else "seedvr2-3b-int8", "seedvr2-vae", "realesrgan"]
    if versions and fits_7b:
        out.append("seedvr2-7b-sharp-int8")
    out += ["yolo-face", "yolo-hand", "yolo-person"]
    if not needs_consent("sam3", root):
        out += ["sam3", "clip-tokenizer"]
    out += ["florence-2-large", "florence-2-base", "florence-2-tokenizer", "birefnet-general", "birefnet-lite"]
    if versions:
        out += [f"birefnet-{v}" for v in q.BIREFNET_VERSIONS if v not in ("general", "lite")]
    caps = [q.family_capabilities(f) for f in families]
    kinds: set[str] = set()
    for c in caps:
        if not (c.get("control") or c.get("control_reference") or c.get("control_loras")):
            continue
        refused = CONTROLNET_REFUSED_KINDS if c.get("control") == "controlnet" else ()
        kinds.update(k for k in q.CNET_KINDS if k not in refused)
    out += [w for k in q.CNET_KINDS if k in kinds for w in q.CNET_KINDS[k]]
    if any(c.get("control") == "controlnet" for c in caps):
        out.append("cnet-union")
    if any(c.get("control") == "moss" for c in caps):
        out.append("moss-control")
    out += [f"moss-control-{k}" for c in caps for k in sorted(set((c.get("control_loras") or {}).values()))]
    return list(dict.fromkeys(out))


def every() -> list[dict]:
    """The downloads running or failed, for /health and the web page."""
    with _lock:
        return [{**info(n), **{k: v for k, v in j.items() if k != "thread"}} for n, j in _jobs.items()]


@contextlib.contextmanager
def _file_lock(p: Path):
    """An exclusive lock on `p` (created), or None when another process holds it."""
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+b")
    try:
        try:
            file_lock.acquire(f)
        except OSError:
            yield None
            return
        yield f
    finally:
        f.close()


def _run(name: str, root: Path, job: dict, say: Callable[[str], None]) -> None:
    q = _qimage()
    a = q.AUX[name]
    repo, rpath, size, sha = q.FILES[a["file"]]
    out = path(name, root)
    # a file used as it is goes straight to its name (two versions' model.safetensors would otherwise meet)
    src = out if a.get("convert") not in CONVERTERS else root / Path(rpath).name
    try:
        root.mkdir(parents=True, exist_ok=True)
        while True:
            with _file_lock(out.with_name(out.name + ".lock")) as held:
                if held is not None:
                    if not ready(name, root):
                        _fetch(f"{endpoint()}/{repo}/resolve/{q.REVISIONS[repo]}/{rpath}", src, size, sha, job, say,
                               name)
                        _convert(a, src, out, sha, say, name)
                    break
            job["waiting"] = "another process is downloading it"
            if ready(name, root):
                break
            time.sleep(2)
        say(f"image tools: {name} is ready ({out})")
        model_terms.fetch([repo], say=lambda m: say(f"image tools: {m}"), timeout=30)   # About > Model terms
        with _lock:
            _jobs.pop(name, None)
    except Exception as e:  # noqa: BLE001 - every failure is the download's result, told to the next caller
        why = _reason(e)
        say(f"image tools: downloading {name} failed: {why}")
        with _lock:
            job.update(status="failed", error=why)


def _reason(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"the server answered HTTP {e.code}"
    if isinstance(e, urllib.error.URLError):
        return (f"no connection to {endpoint().split('/')[2]} ({e.reason}): this computer is offline or the host is "
                "blocked; set HF_ENDPOINT to a mirror, or put the file there by hand")
    if isinstance(e, PermissionError):
        return f"cannot write {e.filename}: no permission (a read-only models folder)"
    if isinstance(e, OSError) and e.errno == 28:
        return "the disk is full"
    return str(e) or type(e).__name__


def _fetch(url: str, part_of: Path, size: int, sha: str, job: dict, say, name: str) -> None:
    """`url` into part_of (through <name>.part), checked against `size` and `sha`."""
    if part_of.is_file() and part_of.stat().st_size == size and _sha256(part_of) == sha:
        return
    part = part_of.with_name(part_of.name + ".part")
    say(f"image tools: downloading {name} ({size / 2**20:.0f} MiB) from {url.split('/resolve/')[0]}")
    for attempt in range(5):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xestrata-serve", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r:
                have = download_core.resumed_at(r, have, size, strict=False)
                last = time.monotonic()
                with open(part, "ab" if have else "wb") as f:
                    def report(done):
                        nonlocal last
                        job["bytes"] = done
                        if time.monotonic() - last > 10:
                            last = time.monotonic()
                            say(f"image tools: {name}: {done / 2**20:.0f} of {size / 2**20:.0f} MiB")

                    have = download_core.copy_stream(r, f, have=have, chunk=CHUNK, limit=size, progress=report)
                    f.flush()
                    os.fsync(f.fileno())
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 4:
                raise
            say(f"image tools: {name}: interrupted ({_reason(e) if isinstance(e, urllib.error.URLError) else e}); "
                f"again in {2 ** attempt * 2} s")
            time.sleep(2 ** attempt * 2)
    if part.stat().st_size != size:
        raise OSError(f"{part.name}: {part.stat().st_size} bytes, not {size}")
    got = _sha256(part)
    if got != sha:
        part.unlink()
        raise OSError(f"{part_of.name}: SHA-256 {got}, not the pinned {sha} (the file was removed)")
    os.replace(part, part_of)


def _convert(a: dict, src: Path, out: Path, sha: str, say, name: str) -> None:
    """The downloaded file as the engine reads it, in a process of its own (yolo_pt.py, esrgan.py, state_pt.py: they
    read the pickle without running it), renamed into place when complete, then its .done mark."""
    if a.get("convert") in CONVERTERS:
        say(f"image tools: converting {name} to safetensors (the weights' values as they are)")
        tool = Path(__file__).resolve().parent.parent / "tools" / "image" / CONVERTERS[a["convert"]]
        r = subprocess.run([sys.executable, "-I", str(tool), str(src), str(out), "--sha256", sha],
                           capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S, check=False)
        if r.returncode != 0:
            raise OSError(f"the conversion failed: {(r.stderr or r.stdout).strip()[-300:]}")
    elif src != out:
        os.replace(src, out)
    digest = _sha256(out) if a.get("convert") in CONVERTERS else sha
    download_core.write_done(out, f"sha256 {digest}")


def _sha256(p: Path) -> str:
    return download_core.sha256_file(p, chunk=8 << 20)
