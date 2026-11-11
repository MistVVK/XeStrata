"""Getting the DFlash2 drafter onto this computer: a pinned download, checked against the repository's own SHA-256, packed for the engine.

    strata drafting plan  [--data-dir DIR]      what it would download (nothing is changed)
    strata drafting fetch [--data-dir DIR]      download, check, pack; progress in plain words

Only an explicit command reaches the network (strata_control/acquisition.py): a start never downloads. The drafter is
incoai/Qwen3.8-Flash-Next-DFlash2 at one revision (strata_control/drafting.py has the revision, the file sizes and hashes). The 1.2 GB
safetensors file is packed (tools/dflash2_pack.py --q4_0, numpy only) into the engine's 0.52 GB runtime file, which is checked against its pinned size and
SHA-256, and the download is then removed. A Q8 pack from an earlier Strata is used as it is.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from strata_control import acquisition, drafting, transfer
from strata_control import feed as feedmod

Progress = Callable[[str, int, int], None]    # (step in plain words, bytes done, bytes total)


class DraftingError(RuntimeError):
    """The drafter could not be fetched or packed. The message is plain words and says what to do."""


class OfflineError(DraftingError):
    """The download address did not answer (no internet, or a host that is down). Setup skips this one silently: the model keeps the MTP (strata-offline-users)."""


def plan(data_dir: str | Path, engine: str | os.PathLike[str] | None = None, *, model_name: str | None = None, backend: str | None = None, archs=None,
         args: list | None = None, vram_mib: float | None = None) -> dict:
    """What fetching would do, in numbers and words. ``gb`` is 0 when the drafter is already here. ``engine``: the installed engine program (None: not known); ``vram_gb`` is
    the graphics memory DFlash2 alone takes on this computer (0.72 with the Q4_0 selector, which the engine reads and setup makes, else 0.89) and ``selq4`` says which.
    For a set with a tuned drafter (drafting.SET_DEFAULTS) the download is that 0.34 GB file, and it counts as missing until it is here."""
    tuned = drafting.tuned_applies(backend, model_name, archs, args=args, vram_mib=vram_mib)
    variant = drafting.tuned_variant(backend, model_name)
    pin = drafting.tuned_pack(variant)
    ready = drafting.tuned_ok(data_dir, variant) if tuned else drafting.pack_ready(data_dir)
    gb = 0.0 if ready else (pin.gb if tuned else drafting.DOWNLOAD_GB)
    if ready:
        text = "The DFlash2 drafter is already on this computer."
    elif tuned:
        text = (f"The DFlash2 drafter is {pin.gb:.2f} GB to download, from Strata's signed update feed, Apache-2.0 (a copy of {drafting.REPO} that Strata tuned). "
                f"It needs about {drafting.vram_gb(data_dir, engine):.1f} GB of graphics memory, less than the MTP's layer, which is not loaded beside it.")
    else:
        text = (f"The DFlash2 drafter is {drafting.DOWNLOAD_GB:.1f} GB to download (it is packed to {drafting.PACK_GB:.2f} GB for the engine and the "
                f"download is removed), from {drafting.REPO}, {drafting.LICENSE}. It needs about {drafting.vram_gb(data_dir, engine):.1f} GB of graphics memory, "
                "less than the MTP's layer, which is not loaded beside it.")
    return {"ready": ready, "gb": gb, "drafter_gb": pin.gb if tuned else drafting.DOWNLOAD_GB, "tuned": tuned, "file": pin.name if tuned else drafting.PACK_NAME,
            "vram_gb": drafting.vram_gb(data_dir, engine), "selq4": drafting.uses_selector_q4(data_dir, engine), "pack": str(drafting.pack_path(data_dir, model_name=model_name, backend=backend)), "repo": drafting.REPO, "revision": drafting.REVISION, "text": text}


def _packer():
    """tools/dflash2_pack.py as a module: beside strata_control in a source checkout and in the packaged Control (TOOLS_SUBSET)."""
    path = Path(__file__).resolve().parents[1] / "tools" / "dflash2_pack.py"
    if not path.is_file():
        raise DraftingError("The packing tool is missing from this installation. Update Strata, then try again.")
    spec = importlib.util.spec_from_file_location("strata_dflash2_pack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("strata_dflash2_pack", module)
    spec.loader.exec_module(module)
    return module


REACH_SECONDS = 6.0


def online(url: str, timeout: float = REACH_SECONDS) -> bool:
    """Does the download address answer at all? One short HEAD request, no retry: with no internet (or a connection that hangs) the person gets a plain
    sentence in a few seconds instead of a stalled progress bar. Any answer from the server counts, even an error page."""
    try:
        transfer.check_url_policy(url)
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-drafting/1"})
        with transfer._opener.open(request, timeout=timeout):     # the download's own opener: same certificates, same redirect policy, same proxy settings
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, transfer.TransferError) as exc:
        from strata_control import tls
        return bool(tls.is_certificate_error(exc))        # a certificate problem is an answer: the download explains it in its own words


def _hub_snapshots() -> list[Path]:
    """Where the Hugging Face tools keep this repository's pinned revision, if the person already has it from them (nothing is created or changed)."""
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r / ("models--" + drafting.REPO.replace("/", "--")) / "snapshots" / drafting.REVISION for r in roots]


def _present(folders: list[Path], spec) -> dict[str, Path]:
    """The pinned files that are already on this computer, found in these folders and checked against the repository's size and SHA-256."""
    found: dict[str, Path] = {}
    for name, size, sha in spec:
        for folder in folders:
            candidate = folder / name
            try:
                if candidate.is_file() and candidate.stat().st_size == size and transfer.sha256_file(candidate) == sha:
                    found[name] = candidate
                    break
            except OSError:
                continue
    return found


def convert_selector(data_dir: str | Path, *, progress: Progress | None = None, source_bytes: int | None = None, source_sha256: str | None = None,
                     size: int | None = None, sha256: str | None = None) -> Path | None:
    """Make the pack with the Q4_0 selector (drafting.CONVERTED_NAME, 174 MiB less in graphics memory) from the verified Q4_0 pack, on this computer. No network, a second
    or two on a fast machine, about 120 MB of memory. Called only by setup, `strata drafting fetch` and Load (launch), never by a start or a serve. The Q4_0 pack must be
    the pinned bytes (PACK_BYTES / PACK_SHA256) and the result the pinned bytes (CONVERTED_BYTES / CONVERTED_SHA256); anything else is deleted and None is returned:
    the Q4_0 pack with the bf16 selector keeps being used, nothing is reported as an error. The idea and the bytes are nazeshinjite's
    (https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF). The keyword arguments are for tests, which convert a small synthetic pack."""
    report = progress or (lambda *_: None)
    want_size = drafting.CONVERTED_BYTES if size is None else size
    want_sha = drafting.CONVERTED_SHA256 if sha256 is None else sha256
    if drafting.converted_ok(data_dir, size=want_size, sha256=want_sha):
        return drafting.converted_path(data_dir)
    src = drafting.target_path(data_dir)
    out = drafting.converted_path(data_dir)
    tmp = out.with_suffix(".tmp")
    try:
        if not src.is_file():
            return None
        pinned_bytes = drafting.PACK_BYTES if source_bytes is None else source_bytes
        pinned_sha = drafting.PACK_SHA256 if source_sha256 is None else source_sha256
        if src.stat().st_size != pinned_bytes or transfer.sha256_file(src) != pinned_sha:
            return None                                  # not the pack we measured: leave it as it is
        report("Preparing the DFlash2 drafter's smaller selector…", 0, 1)
        _packer().convert_selector_q4(str(src), str(tmp), say=lambda *_: None)
        if tmp.stat().st_size != want_size or transfer.sha256_file(tmp) != want_sha:
            tmp.unlink(missing_ok=True)
            return None
        os.replace(tmp, out)
        out.with_name(out.name + ".ok").unlink(missing_ok=True)
        if not drafting.converted_ok(data_dir, size=want_size, sha256=want_sha):        # writes the marker
            return None
        report("The DFlash2 drafter's smaller selector is ready.", 1, 1)
        return out
    except Exception:                                    # noqa: BLE001 - never a reason to stop a setup or a launch: the bf16-selector pack runs
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


FEED_TIMEOUT = 8.0
RETRY_AFTER_S = drafting.RETRY_AFTER_S          # Load (strata launch) asks the feed for the tuned drafter at most this often when the answer was "nothing"; setup and `strata drafting fetch` always ask
MISS_MARK = drafting.MISS_MARK
_recent_miss = drafting.recent_miss              # (the start's "Run strata drafting fetch" line reads the same mark: strata_control/drafting.py fetch_hint)


def _note_miss(data_dir: str | Path, missed: bool) -> None:
    mark = Path(data_dir) / drafting.FOLDER / MISS_MARK
    try:
        if missed:
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        else:
            mark.unlink(missing_ok=True)
    except OSError:
        pass


def _feed_source(feed: str | None = None) -> str:
    """The signed feed to read: the one named, STRATA_UPDATE_FEED, else the project's release feed (strata_cli/update_cmd.py default_feed)."""
    if feed:
        return feed
    from strata_cli import update_cmd
    return os.environ.get(update_cmd.ENV_FEED) or update_cmd.default_feed()


def tuned_from_feed(*, feed: str | None = None, backend: str | None = None, size: int | None = None, sha256: str | None = None, name: str | None = None,
                    variant: str | None = None) -> tuple[str, int, str] | None:
    """(address, size, sha256) of the tuned drafter as the signed feed lists it (member ``drafters``), or None: the feed cannot be read (offline, the host is down: silent), has no
    such entry for this engine family, or lists other bytes than the ones this Strata was measured with (pinned size and SHA-256: then it is another drafter, not ours).
    A feed that does not verify is DraftingError: a bad signature, an expired feed. A build that trusts no key (a source checkout from before the keys) is silent."""
    from strata_control import signing
    pin = drafting.tuned_pack(variant)
    want_size = pin.size if size is None else size
    want_sha = pin.sha256 if sha256 is None else sha256
    want_name = name or pin.name
    source = _feed_source(feed)
    try:
        loaded = feedmod.fetch(source, signing.load_trusted_keys(), timeout=FEED_TIMEOUT, channel="stable")
    except feedmod.FeedError as exc:
        if isinstance(exc, feedmod.FeedUnreachable) or exc.exit_code == 6 or "trusts no update key" in str(exc):
            return None
        raise DraftingError(f"The update feed could not be trusted: {exc}") from None
    for entry in loaded.drafters:
        if entry.name == want_name and (backend or drafting.TUNED_BACKEND) in entry.backends:
            if entry.size != want_size or entry.sha256 != want_sha:
                return None
            return loaded.url(entry.url), entry.size, entry.sha256
    return None


def _reachable(location: str) -> bool:
    """A local source (a file: address or a path) is always there to try; a web address must answer one short HEAD request."""
    return transfer.kind_of(location) in ("path", "file") or online(location)


def provide_licence(data_dir: str | Path) -> bool:
    """Copy the Apache-2.0 text and the 'modified by Strata' notice (data/dflash2-tuned/) next to the tuned drafter file. False when Strata's copy is missing."""
    source = Path(__file__).resolve().parents[1] / drafting.TUNED_LICENCE_FOLDER
    target = Path(data_dir) / drafting.FOLDER
    try:
        for name, out in (("LICENSE", "LICENSE-dflash2-tuned"), ("NOTICE.md", "NOTICE-dflash2-tuned.md")):
            text = (source / name).read_bytes()
            target.mkdir(parents=True, exist_ok=True)
            if not (target / out).is_file() or (target / out).read_bytes() != text:
                (target / out).write_bytes(text)
        return True
    except OSError:
        return False


def ensure_tuned(data_dir: str | Path, *, progress: Progress | None = None, cancel: threading.Event | None = None, url: str | None = None,
                 size: int | None = None, sha256: str | None = None, name: str | None = None, feed: str | None = None, throttle: bool = False, backend: str | None = None, variant: str | None = None) -> Path | None:
    """Make the tuned drafter (drafting.TUNED_NAME, 340 MB) exist under ``data_dir`` and return its path. It is found in the signed feed (``feed``: a URL or folder; default
    STRATA_UPDATE_FEED or the project's release feed), unless ``url`` or STRATA_DFLASH2_TUNED_URL names the file directly. None when it cannot be had without anything being wrong:
    the feed or the address does not answer (no internet, the host is down), the feed does not list the file, or it answers with an error page. Resumable (transfer.download continues a kept partial file). A complete
    file that is not the pinned bytes is deleted and DraftingError says so; an interrupted download keeps its partial file and raises DraftingError. The licence text and the
    notice are copied beside it. The stock pack is never touched. The keyword arguments are for tests."""
    report = progress or (lambda *_: None)
    pin = drafting.tuned_pack(variant)
    want_size = pin.size if size is None else size
    want_sha = pin.sha256 if sha256 is None else sha256
    file_name = name or pin.name
    out = Path(data_dir) / drafting.FOLDER / file_name
    if drafting.converted_ok(data_dir, size=want_size, sha256=want_sha, name=file_name):
        provide_licence(data_dir)
        return out
    acquisition.require("the tuned DFlash2 drafter", "strata drafting fetch")
    location = url or os.environ.get(drafting.TUNED_URL_ENV, "").strip()
    if location:                                          # a direct address (tests, a mirror): the pinned size and SHA-256 still decide
        if not _reachable(location):
            return None
    else:                                                 # the normal way: the signed feed names the file; nothing answers offline
        if throttle and _recent_miss(data_dir):           # a launch asked a short while ago and got nothing: no wait, no second try
            return None
        found = tuned_from_feed(feed=feed, backend=backend, size=want_size, sha256=want_sha, name=file_name, variant=variant)
        _note_miss(data_dir, found is None)
        if found is None:
            return None
        location = found[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    try:
        transfer.download(location, partial, size=want_size, sha256=want_sha, cancel=cancel, what="the tuned DFlash2 drafter",
                          progress=lambda d, _s: report("Downloading the tuned DFlash2 drafter…", d, want_size))
    except transfer.Unreachable as exc:
        raise DraftingError(str(exc)) from None          # the network rule's sentence: the host, what to check, what still works
    except transfer.ChecksumMismatch:
        raise DraftingError("The tuned DFlash2 drafter did not pass the safety check, so it was deleted.") from None
    except transfer.Cancelled:
        raise
    except transfer.TransferError as exc:
        if exc.resumable:
            raise DraftingError(f"The tuned DFlash2 download did not finish: {exc}. Try again; it continues where it stopped.") from None
        return None                                      # an error page from the host (not published yet, moved): nothing to report, the standard drafter runs
    os.replace(partial, out)
    out.with_name(out.name + ".ok").unlink(missing_ok=True)
    if not drafting.converted_ok(data_dir, size=want_size, sha256=want_sha, name=file_name):       # writes the marker
        out.unlink(missing_ok=True)
        raise DraftingError("The tuned DFlash2 drafter did not pass the safety check, so it was deleted.")
    provide_licence(data_dir)
    drafting.provide_vocab(data_dir)
    report("The tuned DFlash2 drafter is ready.", want_size, want_size)
    return out


def ensure_for(data_dir: str | Path, *, model_name: str | None, backend: str | None, progress: Progress | None = None,
               cancel: threading.Event | None = None, feed: str | None = None, throttle: bool = False, archs=None, args: list | None = None,
               vram_mib: float | None = None) -> Path | None:
    """The drafter a set should start with, fetched if needed: the tuned pack where the set has one (drafting.SET_DEFAULTS), else, or when the tuned one cannot be had, the
    stock drafter (ensure). For such a set: None when nothing could be fetched because the network did not answer (the MTP runs: silent for the caller); DraftingError for a
    real fault (a safety check that failed, an interrupted download). A set with no tuned entry gets ensure() unchanged."""
    if drafting.tuned_applies(backend, model_name, archs, args=args, vram_mib=vram_mib):
        tuned_error: DraftingError | None = None
        try:
            got = ensure_tuned(data_dir, progress=progress, cancel=cancel, feed=feed, throttle=throttle, backend=backend, variant=drafting.tuned_variant(backend, model_name))
        except DraftingError as exc:
            got, tuned_error = None, exc
        if got is not None:
            return got
        if drafting.pack_ready(data_dir):
            if tuned_error is not None:
                raise tuned_error
            return drafting.pack_path(data_dir)
        try:
            return ensure(data_dir, progress=progress, cancel=cancel)
        except OfflineError:
            if tuned_error is not None:
                raise tuned_error from None
            return None
    return ensure(data_dir, progress=progress, cancel=cancel)       # a set with no tuned entry: the stock drafter exactly as before


def ensure(data_dir: str | Path, *, progress: Progress | None = None, cancel: threading.Event | None = None,
           base_url: str | None = None, files=None) -> Path:
    """Make the packed drafter exist under ``data_dir`` and return its path. A pack that is there is kept. Resumable: an interrupted
    download is continued, a file whose hash differs is deleted and the person is told. ``base_url`` / ``files`` are for tests."""
    report = progress or (lambda *_: None)
    out = drafting.target_path(data_dir)
    if drafting.pack_ready(data_dir):
        drafting.provide_vocab(data_dir)
        if files is None and drafting.engine_has_selector_q4():
            convert_selector(data_dir, progress=report)
        report("The DFlash2 drafter is ready.", 1, 1)
        return drafting.pack_path(data_dir)
    acquisition.require("the DFlash2 drafter", "strata drafting fetch")
    folder = out.parent
    src = folder / "src"
    spec = tuple(files if files is not None else drafting.FILES)
    have = _present([src, *(_hub_snapshots() if files is None else [])], spec)       # already here and checked: used where it is, never fetched again
    if len(have) < len(spec) and not online(base_url or drafting.BASE_URL):
        raise OfflineError("There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
    src.mkdir(parents=True, exist_ok=True)
    total = sum(size for _n, size, _h in spec)
    done_before = 0
    for name, size, sha in spec:
        if name not in have:
            base = done_before
            partial = src / (name + ".part")
            try:
                transfer.download((base_url or drafting.BASE_URL) + name, partial, size=size, sha256=sha, cancel=cancel, what="the DFlash2 drafter",
                                  progress=lambda d, _s, base=base: report("Downloading the DFlash2 drafter…", base + d, total))
            except transfer.Unreachable as exc:
                raise DraftingError(str(exc)) from None  # the network rule's sentence: the host, what to check, what still works
            except transfer.ChecksumMismatch:
                raise DraftingError("The downloaded DFlash2 file did not pass the safety check, so it was deleted. Try again; if it keeps "
                                    "happening, use MTP for now.") from None
            except transfer.TransferError as exc:
                raise DraftingError(f"The DFlash2 download did not finish: {exc}. Try again; it continues where it stopped.") from None
            os.replace(partial, src / name)
            have[name] = src / name
        done_before += size
    report("Packing the DFlash2 drafter for the engine…", total, total)
    tmp = out.with_suffix(".tmp")
    try:
        _packer().pack(str(have["model.safetensors"]), str(have["config.json"]), str(tmp), say=lambda *_: None, q4=True)
        if files is None:                                # the real pinned files: the pack must be the exact bytes we measured (tests pack a small synthetic file)
            if tmp.stat().st_size != drafting.PACK_BYTES or transfer.sha256_file(tmp) != drafting.PACK_SHA256:
                tmp.unlink(missing_ok=True)
                raise DraftingError("The DFlash2 drafter was prepared but did not pass the safety check, so it was deleted. The model keeps using the MTP. "
                                    "Try again; if it keeps happening, update Strata.")
        os.replace(tmp, out)
    except DraftingError:
        raise
    except Exception as exc:                         # noqa: BLE001 - a plain sentence for any packing failure; the cause stays in the chain
        tmp.unlink(missing_ok=True)
        raise DraftingError("The DFlash2 drafter could not be prepared on this computer. The model keeps using the MTP.") from exc
    for name, _size, _sha in spec:                   # the source files are 1.2 GB of nothing the engine reads (a copy kept by the Hugging Face tools is theirs: left alone)
        if have[name].parent == src:
            have[name].unlink(missing_ok=True)
    try:
        src.rmdir()
    except OSError:
        pass
    drafting.provide_vocab(data_dir)                 # the draft vocabulary comes with the pack: an install that never had the MTP layer needs it
    if files is None and drafting.engine_has_selector_q4():
        convert_selector(data_dir, progress=report)  # the smaller selector (174 MiB), made here from the pack just checked; the bf16-selector pack stays beside it
    report("The DFlash2 drafter is ready.", total, total)
    return out
