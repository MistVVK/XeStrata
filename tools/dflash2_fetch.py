# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/dflash2_fetch.py - a model's DFlash2 drafter onto this computer: a pinned download, checked, packed for the
engine.

    python tools/dflash2_fetch.py plan   --out DIR [--model KEY]   what it would download (nothing is changed)
    python tools/dflash2_fetch.py fetch  --out DIR [--model KEY]   download, check, pack; progress in plain words
    python tools/dflash2_fetch.py verify --out DIR [--model KEY]   the pack on disk against its pinned size and SHA-256

Only `fetch` reaches the network. Each model with a drafter (DRAFTERS: Qwen3.8-Flash-Next, the default; GLM-5.3-Flash;
Qwen3.6-35B-A3B) has one of Inco AI's (Apache-2.0) at one revision; its safetensors file is packed (tools/
dflash2_pack.py --q4_0, numpy only) into the engine's runtime file, checked against its pinned size and SHA-256, and
the download is then removed. The selector's codebooks are converted to Q4_0 rows on this computer too (less VRAM,
the same drafts); a conversion that does not match its pin is deleted and the bf16-selector pack stays. A Q8 pack from
an earlier run keeps working. LICENSE and NOTICE.md (Inco AI, Apache-2.0; the selector idea is nazeshinjite's) are
copied beside the pack.
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


class Drafter:
    """One model's drafter: its repository and revision, the files' sizes and SHA-256 there, the packs' names and pins
    (None: packed on this computer and kept without a pin), the NOTICE in data/dflash2/."""

    def __init__(self, key, repo, revision, files, pack_name, pack_pin, converted_name, converted_pin, download_gb,
                 pack_gb, notice, legacy=None):
        self.key, self.repo, self.revision, self.files = key, repo, revision, files
        self.pack_name, self.pack_pin = pack_name, pack_pin
        self.converted_name, self.converted_pin = converted_name, converted_pin
        self.download_gb, self.pack_gb, self.notice, self.legacy = download_gb, pack_gb, notice, legacy
        self.base_url = download_core.hf_endpoint() + "/%s/resolve/%s/" % (repo, revision)


DRAFTERS = {
    # the drafter of Qwen3.8-Flash-Next (the target's layers 5/15/25/35/45, the mean of the four hyper-connection
    # streams); the bf16 1.2 GB download becomes a 0.52 GB Q4_0 pack, 0.34 GB with the Q4_0 selector
    "qwen3.8-flash-next": Drafter(
        "qwen3.8-flash-next", "incoai/Qwen3.8-Flash-Next-DFlash2", "a22a1744e85c17364dcc54a6a420af7135d4c7f6",
        (("config.json", 1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
         ("model.safetensors", 1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f")),
        "qwen3.8-flash-next-dflash2.q4_0.pack",
        (522458112, "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1"),
        "qwen3.8-flash-next-dflash2.q4_0-selq4.pack",
        (339694592, "4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd"),
        1.2, 0.52, "NOTICE.md", legacy="qwen3.8-flash-next-dflash2.q8.pack"),
    # GLM-5.3-Flash's (base zai-org/GLM-5.3-Flash): hidden 4096, 5 layers, the target's layers 5/14/24/33/42, the mean
    # of its four mHC streams (sglang's capture, PR 36708); vocabulary 154,880
    "glm-5.3-flash": Drafter(
        "glm-5.3-flash", "incoai/GLM-5.3-Flash-DFlash2", "6cc6c481eee930a5139619ed919f78726371adbc",
        (("config.json", 1273, "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573"),
         ("model.safetensors", 2342169800, "b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac")),
        "glm-5.3-flash-dflash2.q4_0.pack",
        (773469184, "f2b80152f170e7fc037b101876a960aff6db1b7461a40066d15d10430eeff8b1"),
        "glm-5.3-flash-dflash2.q4_0-selq4.pack",
        (659477504, "960304755f7adc13a706f0f4ba6dc55bbe81dfd4c187c2352aad2277c9df5d3d"),
        2.3, 0.75, "NOTICE-GLM-5.3-Flash.md"),
    # Qwen3.6-35B-A3B's: hidden 2048, 6 layers, the target's layers 1/6/11/16/22/27/32/37 (its residual: one stream);
    # vocabulary 248,320
    "qwen3.6-35b-a3b": Drafter(
        "qwen3.6-35b-a3b", "incoai/Qwen3.6-35B-A3B-DFlash2", "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
        (("config.json", 1291, "86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59"),
         ("model.safetensors", 1052513456, "d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d")),
        "qwen3.6-35b-a3b-dflash2.q4_0.pack",
        (479229952, "12a69bedca64f9e1196f34192d39519e5eab08a4f56de658abbf7f05db405784"),
        "qwen3.6-35b-a3b-dflash2.q4_0-selq4.pack",
        (296466432, "a635ae4af9497e35eca46d78c01d3f95c87709f7d1d2829ffd1dec08084acbed"),
        1.1, 0.45, "NOTICE-Qwen3.6-35B-A3B.md"),
}
DEFAULT = "qwen3.8-flash-next"
# the Qwen3.8 drafter's names, for the callers that knew only it (setup's)
REPO = DRAFTERS[DEFAULT].repo
REVISION = DRAFTERS[DEFAULT].revision
LICENSE = "Apache-2.0"
PACK_NAME = DRAFTERS[DEFAULT].pack_name
CONVERTED_NAME = DRAFTERS[DEFAULT].converted_name
BAD = 3                                             # `verify`'s exit code: the pack is missing or corrupt
UA = {"User-Agent": "xestrata-dflash2-fetch"}


def pack_ok(path, pin):
    """The pack is there and, where it is pinned ((size, SHA-256)), is those bytes; an unpinned pack (a drafter whose
    pack has no pin yet) only has to be a whole DF2P file (fetch writes it last, under a temporary name)."""
    try:
        if not path.is_file():
            return False
        if pin is None:
            with open(path, "rb") as f:
                return f.read(4) == b"DF2P" and path.stat().st_size > 4096
        return path.stat().st_size == pin[0] and download_core.sha256_file(path) == pin[1]
    except OSError:
        return False


def ready(d: Drafter, out: Path):
    """The pack this install would use: (path, name) or None. The converted pack first, then the Q4_0, then a Q8."""
    for name, pin in ((d.converted_name, d.converted_pin), (d.pack_name, d.pack_pin)):
        if pack_ok(out / name, pin):
            return out / name, name
    if d.legacy:
        legacy = out / d.legacy
        if legacy.is_file() and legacy.stat().st_size > 0:
            return legacy, d.legacy
    return None


def plan(d: Drafter, out: Path):
    got = ready(d, out)
    if got:
        text = "The DFlash2 drafter is already on this computer (%s)." % got[1]
        gb = 0.0
    else:
        text = ("The DFlash2 drafter is %.1f GB to download (it is packed to %.2f GB for the engine and the download "
                "is removed), from %s, %s." % (d.download_gb, d.pack_gb, d.repo, LICENSE))
        gb = d.download_gb
    print(json.dumps({"ready": got is not None, "gb": gb, "pack": str(got[0]) if got else str(out / d.converted_name),
                      "repo": d.repo, "revision": d.revision, "model": d.key, "text": text}, indent=1))


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


def hub_snapshots(d: Drafter):
    """Where the Hugging Face tools keep this repository's pinned revision, if the person already has it (nothing is
    created or changed)."""
    roots = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r / ("models--" + d.repo.replace("/", "--")) / "snapshots" / d.revision for r in roots]


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


def notices(d: Drafter, out: Path):
    """LICENSE (Apache-2.0) and NOTICE.md (Inco AI, the selector credit) beside the pack, from the checkout's data/."""
    src = Path(__file__).resolve().parents[1] / "data" / "dflash2"
    for name, dst in (("LICENSE", "LICENSE"), (d.notice, "NOTICE.md")):
        try:
            shutil.copyfile(src / name, out / dst)
        except OSError as e:
            print("  the drafter's %s could not be copied (%s)" % (dst, e), file=sys.stderr)


def convert(d: Drafter, out: Path):
    """The verified Q4_0 pack's selector codebooks as Q4_0 rows (no network). A conversion that does not match its
    pin is deleted and the bf16-selector pack stays."""
    print("  converting the selector's codebooks to Q4_0 (less graphics memory, the same drafts)")
    conv = out / d.converted_name
    tmp = conv.with_name(conv.name + ".tmp")
    try:
        dflash2_pack.convert_selector_q4(str(out / d.pack_name), str(tmp))
        tmp.replace(conv)
        if not pack_ok(conv, d.converted_pin):
            raise ValueError("the converted pack does not match the pinned bytes")
    except (OSError, ValueError) as e:
        tmp.unlink(missing_ok=True)
        conv.unlink(missing_ok=True)
        print("  the selector conversion failed (%s): the bf16-selector pack stays" % e,
              file=sys.stderr)


def fetch(d: Drafter, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if pack_ok(out / d.converted_name, d.converted_pin):
        print("The DFlash2 drafter is already here: %s" % (out / d.converted_name))
        notices(d, out)
        return
    if pack_ok(out / d.pack_name, d.pack_pin):
        convert(d, out)                  # the Q4_0 pack is here: the selector conversion needs no network
        notices(d, out)
        return
    if d.legacy:
        legacy = out / d.legacy
        if legacy.is_file() and legacy.stat().st_size > 0:
            print("The DFlash2 drafter is already here (an earlier Q8 pack): %s" % legacy)
            notices(d, out)
            return
    if not online(d.base_url + "config.json"):
        raise SystemExit("the DFlash2 drafter's download address did not answer (no internet, or the host is down): "
                         "the model keeps its MTP draft layer; run setup again when the connection is back")
    found = present(hub_snapshots(d), d.files)
    work = out / "download"
    work.mkdir(exist_ok=True)
    src = {}
    for name, size, sha in d.files:
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
            download(d.base_url + name, dst, size, sha)
        src[name] = dst
    print("  packing the drafter for the engine (Q4_0, numpy only)")
    q4 = out / d.pack_name
    tmp = q4.with_name(q4.name + ".tmp")         # the whole file or none: an unpinned pack is taken as it is
    dflash2_pack.pack(str(src["model.safetensors"]), str(src["config.json"]), str(tmp), q4=True,
                      source="%s %s" % (d.repo, d.revision[:8]))
    tmp.replace(q4)
    if not pack_ok(q4, d.pack_pin):
        q4.unlink(missing_ok=True)
        raise SystemExit("the packed drafter does not match the pinned bytes (size or SHA-256): it was deleted; "
                         "run setup again")
    convert(d, out)
    shutil.rmtree(work, ignore_errors=True)      # the download is removed; the pack stays
    notices(d, out)
    got = ready(d, out)
    print("The DFlash2 drafter is ready: %s" % (got[0] if got else q4))


def verify(d: Drafter, out: Path):
    got = ready(d, out)
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
    ap.add_argument("--model", default=DEFAULT, choices=sorted(DRAFTERS),
                    help="whose drafter (default %s)" % DEFAULT)
    a = ap.parse_args()
    out = Path(a.out)
    d = DRAFTERS[a.model]
    if a.cmd == "plan":
        plan(d, out)
        return
    if a.cmd == "verify":
        sys.exit(verify(d, out))
    exit_with_parent()
    lock = lock_folder(out)
    try:
        fetch(d, out)
    finally:
        file_lock.release(lock)
        lock.close()


if __name__ == "__main__":
    main()
