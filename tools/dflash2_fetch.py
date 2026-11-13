# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/dflash2_fetch.py - the DFlash2 drafter onto this computer: a pinned download, checked, packed for the engine.

    python tools/dflash2_fetch.py plan   --out DIR     what it would download (nothing is changed)
    python tools/dflash2_fetch.py fetch  --out DIR     download, check, pack; progress in plain words
    python tools/dflash2_fetch.py verify --out DIR     the pack on disk against its pinned size and SHA-256

Only `fetch` reaches the network. The drafter is incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) at one revision; the
1.2 GB safetensors file is packed (tools/dflash2_pack.py --q4_0, numpy only) into the engine's 0.52 GB runtime file,
checked against its pinned size and SHA-256, and the download is then removed. The selector's codebooks are converted
to Q4_0 rows on this computer too (174 MiB less VRAM, the same drafts); a conversion that does not match its pin is
deleted and the bf16-selector pack stays. A Q8 pack from an earlier run keeps working. LICENSE and NOTICE (Inco AI,
Apache-2.0; the selector idea is nazeshinjite's) are copied beside the pack.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the checkout, for serve/envvars.py
sys.path.insert(0, str(Path(__file__).resolve().parent))       # tools/, for dflash2_pack
from serve import download as download_core  # noqa: E402
from serve import envvars, file_lock  # noqa: E402

import dflash2_pack  # noqa: E402

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
LICENSE = "Apache-2.0"
BASE_URL = download_core.hf_endpoint() + "/%s/resolve/%s/" % (REPO, REVISION)
# the repository's own sizes and hashes for that revision: a download is refused if either differs
FILES = (
    ("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"),
)
PACK_NAME = "qwen3.8-flash-next-dflash2.q4_0.pack"
PACK_BYTES = 522458112
PACK_SHA256 = "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"
CONVERTED_NAME = "qwen3.8-flash-next-dflash2.q4_0-selq4.pack"
CONVERTED_BYTES = 339694592
CONVERTED_SHA256 = "4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd"
LEGACY_PACK_NAME = "qwen3.8-flash-next-dflash2.q8.pack"      # an earlier run's Q8 pack: still read by the engine
DOWNLOAD_GB = 1.2
PACK_GB = 0.52
BAD = 3                                             # `verify`'s exit code: the pack is missing or corrupt
UA = {"User-Agent": "xestrata-dflash2-fetch"}


def pack_ok(path, size, sha):
    try:
        return path.is_file() and path.stat().st_size == size and download_core.sha256_file(path) == sha
    except OSError:
        return False


def ready(out: Path):
    """The pack this install would use: (path, name) or None. The converted pack first, then the Q4_0, then a Q8."""
    for name, size, sha in ((CONVERTED_NAME, CONVERTED_BYTES, CONVERTED_SHA256), (PACK_NAME, PACK_BYTES, PACK_SHA256)):
        if pack_ok(out / name, size, sha):
            return out / name, name
    legacy = out / LEGACY_PACK_NAME
    if legacy.is_file() and legacy.stat().st_size > 0:
        return legacy, LEGACY_PACK_NAME
    return None


def plan(out: Path):
    got = ready(out)
    if got:
        text = "The DFlash2 drafter is already on this computer (%s)." % got[1]
        gb = 0.0
    else:
        text = ("The DFlash2 drafter is %.1f GB to download (it is packed to %.2f GB for the engine and the download "
                "is removed), from %s, %s." % (DOWNLOAD_GB, PACK_GB, REPO, LICENSE))
        gb = DOWNLOAD_GB
    print(json.dumps({"ready": got is not None, "gb": gb, "pack": str(got[0]) if got else str(out / CONVERTED_NAME),
                      "repo": REPO, "revision": REVISION, "text": text}, indent=1))


def online(url, timeout=6.0):
    """Does the download address answer at all? One short HEAD, no retry: with no internet the person gets a plain
    sentence in a few seconds instead of a stalled progress bar. Any answer counts, even an error page."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


def hub_snapshots():
    """Where the Hugging Face tools keep this repository's pinned revision, if the person already has it (nothing is
    created or changed)."""
    roots = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ[name]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION for r in roots]


def present(folders, spec):
    """The pinned files already on this computer, checked against the repository's size and SHA-256."""
    found = {}
    for name, size, sha in spec:
        for folder in folders:
            candidate = folder / name
            try:
                if (candidate.is_file() and candidate.stat().st_size == size
                        and download_core.sha256_file(candidate) == sha):
                    found[name] = candidate
                    break
            except OSError:
                continue
    return found


def download(url, path: Path, size, sha):
    """One file, resumable: a .part keeps what a stopped run wrote; the next run continues after it (a Range the
    server honours with a 206). The finished file is checked against the pinned size and SHA-256."""
    part = path.with_name(path.name + ".part")
    for attempt in range(1, 4):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            if download_core.sha256_file(part) == sha:
                break
            part.unlink()                # full but corrupt: read it again from the start
            have = 0
        headers = dict(UA)
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                at = download_core.resumed_at(r, have, total=size)
                if at == 0 and have:
                    part.unlink()
                mode = "ab" if at else "wb"
                last = [time.monotonic()]

                def progress(n):
                    if time.monotonic() - last[0] > 2.0:
                        last[0] = time.monotonic()
                        print("  DFlash2 drafter: %.2f / %.2f GB (%d%%)" % (n / 1e9, size / 1e9, 100 * n // size),
                              file=sys.stderr, flush=True)

                with open(part, mode) as f:
                    got = download_core.copy_stream(r, f, have=at, limit=size, progress=progress)
            if got != size:
                raise OSError("the download ended at %d of %d bytes" % (got, size))
            if download_core.sha256_file(part) != sha:
                part.unlink()
                raise OSError("the downloaded file is not the pinned revision's (SHA-256 differs)")
            break
        except (OSError, urllib.error.URLError) as e:
            if attempt == 3:
                raise SystemExit("the DFlash2 download keeps failing (%s): check the connection and run setup again "
                                 "- what is downloaded is kept" % e)
            print("  the download stopped (%s); trying again, it continues where it stopped" % e, file=sys.stderr)
            time.sleep(5)
    part.replace(path)


def notices(out: Path):
    """LICENSE (Apache-2.0) and NOTICE (Inco AI, the selector credit) beside the pack, from the checkout's data/."""
    src = Path(__file__).resolve().parents[1] / "data" / "dflash2"
    for name in ("LICENSE", "NOTICE.md"):
        try:
            shutil.copyfile(src / name, out / name)
        except OSError as e:
            print("  the drafter's %s could not be copied (%s)" % (name, e), file=sys.stderr)


def convert(out: Path):
    """The verified Q4_0 pack's selector codebooks as Q4_0 rows (no network). A conversion that does not match its
    pin is deleted and the bf16-selector pack stays."""
    print("  converting the selector's codebooks to Q4_0 (less graphics memory, the same drafts)")
    conv = out / CONVERTED_NAME
    try:
        dflash2_pack.convert_selector_q4(str(out / PACK_NAME), str(conv))
        if not pack_ok(conv, CONVERTED_BYTES, CONVERTED_SHA256):
            raise ValueError("the converted pack does not match the pinned bytes")
    except (OSError, ValueError) as e:
        conv.unlink(missing_ok=True)
        print("  the selector conversion failed (%s): the bf16-selector pack stays" % e,
              file=sys.stderr)


def fetch(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if pack_ok(out / CONVERTED_NAME, CONVERTED_BYTES, CONVERTED_SHA256):
        print("The DFlash2 drafter is already here: %s" % (out / CONVERTED_NAME))
        notices(out)
        return
    if pack_ok(out / PACK_NAME, PACK_BYTES, PACK_SHA256):
        convert(out)                     # the Q4_0 pack is here: the selector conversion needs no network
        notices(out)
        return
    legacy = out / LEGACY_PACK_NAME
    if legacy.is_file() and legacy.stat().st_size > 0:
        print("The DFlash2 drafter is already here (an earlier Q8 pack): %s" % legacy)
        notices(out)
        return
    if not online(BASE_URL + "config.json"):
        raise SystemExit("the DFlash2 drafter's download address did not answer (no internet, or the host is down): "
                         "the model keeps its MTP draft layer; run setup again when the connection is back")
    found = present(hub_snapshots(), FILES)
    work = out / "download"
    work.mkdir(exist_ok=True)
    src = {}
    for name, size, sha in FILES:
        if name in found:
            src[name] = found[name]
            print("  using the Hugging Face cache's %s" % name)
            continue
        dst = work / name
        if not (dst.is_file() and dst.stat().st_size == size and download_core.sha256_file(dst) == sha):
            if size > (1 << 20):
                print("  downloading %s (%.2f GB)" % (name, size / 1e9))
            else:
                print("  downloading %s" % name)
            download(BASE_URL + name, dst, size, sha)
        src[name] = dst
    print("  packing the drafter for the engine (Q4_0, numpy only)")
    q4 = out / PACK_NAME
    dflash2_pack.pack(str(src["model.safetensors"]), str(src["config.json"]), str(q4), q4=True)
    if not pack_ok(q4, PACK_BYTES, PACK_SHA256):
        q4.unlink(missing_ok=True)
        raise SystemExit("the packed drafter does not match the pinned bytes (size or SHA-256): it was deleted; "
                         "run setup again")
    convert(out)
    shutil.rmtree(work, ignore_errors=True)      # the 1.2 GB download is removed; the pack stays
    notices(out)
    got = ready(out)
    print("The DFlash2 drafter is ready: %s" % (got[0] if got else q4))


def verify(out: Path):
    got = ready(out)
    if got is None:
        print("no checked DFlash2 pack in %s" % out, file=sys.stderr)
        return BAD
    print("%s: %s" % (got[1], got[0]))
    return 0


def lock_folder(out: Path, wait=900.0):
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / ".dflash2_fetch.lock", "a+b")
    end = time.monotonic() + wait
    told = False
    while True:
        try:
            file_lock.acquire(f, rewind=True)
            return f
        except OSError:
            if time.monotonic() > end:
                sys.exit("another DFlash2 download is still writing %s: wait for it or end it, then try again" % out)
            if not told:
                print("another DFlash2 download is writing %s: waiting for it to end" % out,
                      file=sys.stderr, flush=True)
                told = True
            time.sleep(2)


def exit_with_parent():
    """XESTRATA_EXIT_WITH_PARENT=1 (setup sets it): end when the process that started this one ends."""
    if (envvars.get("EXIT_WITH_PARENT") or "").strip() != "1":
        return
    parent = os.getppid()
    if parent <= 1:
        return
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        h = k32.OpenProcess(0x00100000, False, parent)   # SYNCHRONIZE
        if not h:
            return

        def gone():
            return k32.WaitForSingleObject(h, 0) == 0
    else:
        def gone():
            return os.getppid() != parent

    def watch():
        while not gone():
            time.sleep(1.5)
        print("the process that started this download has ended: stopping", file=sys.stderr, flush=True)
        os._exit(1)
    import threading
    threading.Thread(target=watch, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "fetch", "verify"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "plan":
        plan(out)
        return
    if a.cmd == "verify":
        sys.exit(verify(out))
    exit_with_parent()
    lock = lock_folder(out)
    try:
        fetch(out)
    finally:
        file_lock.release(lock)
        lock.close()


if __name__ == "__main__":
    main()
