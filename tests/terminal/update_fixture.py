"""Shared pieces for the update tests: stub engine packages, signed archives, a feed folder and a local HTTP server that
supports Range requests and can cut a transfer short. Nothing here touches the network beyond 127.0.0.1."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import sys
import threading
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))

import feed_tool  # noqa: E402
import package_engine  # noqa: E402
from strata_control import components, signing  # noqa: E402

WINDOWS = os.name == "nt"

def local_family() -> dict:
    """The engine family the machine this call runs on takes, so a stub package is one the resolver will actually pick.
    A stub built for a family this platform refuses is dropped before the test reaches what it is about ("cuda is not a
    supported combination on macos-arm64"). Read at call time, because `fx.cuda_pc` names the machine per case. The Mac
    row is the published feed's own: `backend metal`, `variant ""`, `archs ["apple-gpu"]`, `min_driver {}`
    (strata-feed.json, release 0.2.4.2)."""
    if components.host_platform().startswith("macos"):
        return {"backend": "metal", "variant": "", "archs": ("apple-gpu",), "min_driver": {},
                "library": "libggml-metal.dylib"}
    return {"backend": "cuda", "variant": "cuda13", "archs": ("sm_86", "compute_89"), "min_driver": {"nvidia": "580.65"},
            "library": "cudart64_13.dll" if WINDOWS else "libcudart.so.13"}

GOOD_ENGINE = (
    "import sys, os\nfrom pathlib import Path\n"
    "Path(os.environ['FAKE_ENGINE_MARK']).write_text(os.path.abspath(__file__) + '\\n' + ' '.join(sys.argv[1:]))\n"
    "if '--version' in sys.argv:\n    print('strata-engine stub'); sys.exit(0)\n"
    "print('READY 4096 stop', flush=True)\n"
    "for line in sys.stdin:\n"
    "    if line.strip() == 'QUIT': break\n"
    "    if line.startswith(('GEN ', 'BGEN ')):\n"
    "        slot = int(line.split()[1]) if line.startswith('BGEN ') else None\n"
    "        print('T 97', flush=True)\n"
    "        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)\n"
    "        if slot is not None: print(f'BADM {slot} 0', flush=True)\n")
BAD_ENGINE = "import sys\nif '--version' in sys.argv:\n    sys.exit(0)\nsys.exit(3)\n"      # starts, then dies before READY


def make_key() -> dict:
    return signing.generate_key()


def trusted_of(key: dict) -> dict:
    return {key["key_id"]: base64.b64decode(key["public"])}


def cuda_pc(case):
    """Name this machine a Linux PC with an NVIDIA card for the length of one test case.

    A test that is about the CUDA path builds a CUDA engine package, and `engine_component.resolve` refuses a CUDA
    package on a platform that does not take one — on a Mac that is the first thing it says, and the test never reaches
    what it is about. `host_platform()` reads `platform.system()` and `platform.machine()` every time it is called, so
    naming them here gives the packager, the resolver and the feed one and the same machine. The card itself is already
    a stand-in in these tests: `cuda_archs_present` and `nvidia_driver_version` are mocked the same way."""
    first = mock.patch.object(platform, "system", lambda: "Linux")
    second = mock.patch.object(platform, "machine", lambda: "x86_64")
    first.start()
    second.start()
    case.addCleanup(second.stop)
    case.addCleanup(first.stop)


def build_package(parent: Path, version: str, *, engine_source: str = GOOD_ENGINE, key: dict | None = None,
                  backend: str | None = None, variant: str | None = None, build_id: str | None = None,
                  archs: tuple[str, ...] | None = None, min_driver=None, padding: int = 0) -> Path:
    """A schema-2 package made by the real packager (a stub engine behind a manifest-declared entrypoint). `padding` bytes of
    an incompressible 'library' make the archive big enough for a transfer to be cut part-way. The family defaults to what
    this machine takes (`local_family()`); a test that is about another family says so, and `cuda_pc` names the platform it is for."""
    family = local_family()
    backend = backend or family["backend"]
    variant = family["variant"] if variant is None else variant
    archs = family["archs"] if archs is None else archs
    min_driver = family["min_driver"] if min_driver is None else min_driver
    src = parent / f"src-{version}-{variant or backend}"
    src.mkdir(parents=True)
    (src / "engine_main.py").write_text(engine_source, encoding="utf-8")
    if WINDOWS:
        entry = "strata.cmd"
        (src / entry).write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine_main.py" %*\r\n'.encode())
        library = "cudart64_13.dll" if backend == "cuda" else family["library"]
    else:
        entry = "strata"
        (src / entry).write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
        (src / entry).chmod(0o755)
        library = "libcudart.so.13" if backend == "cuda" else family["library"]
    if not WINDOWS and backend == "metal":
        # package_engine.py refuses a macOS package with no picture reader (#3697), so a Mac stub carries one
        reader = src / "strata-vision"
        reader.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/engine_main.py" "$@"\n')
        reader.chmod(0o755)
    (src / library).write_bytes(os.urandom(padding) if padding else b"a runtime library the stub never loads " * 50)
    out = parent / "built"
    out.mkdir(parents=True, exist_ok=True)
    folder = package_engine.build(src, backend, version, components.host_platform(), out, entry=entry, archs=tuple(archs),
                                  min_driver=min_driver, build_id=build_id or f"stub-{version}",
                                  variant=variant, include=["engine_main.py"])
    if key is not None:
        feed_tool.sign_package(folder, key)
    return folder


def minisign_pair() -> tuple[str, bytes]:
    """(Tauri-style public key value, 32-byte seed): a TEST minisign key, ED25519 with the BLAKE2b-prehash 'ED' mode."""
    seed = os.urandom(32)
    public = signing.public_from_seed(seed)
    key_id = os.urandom(8)
    text = "untrusted comment: minisign public key: TEST\n" + base64.b64encode(b"Ed" + key_id + public).decode() + "\n"
    return base64.b64encode(text.encode()).decode(), key_id + seed


def minisign_sign(blob: bytes, key_id_and_seed: bytes, comment: str = "timestamp:1") -> str:
    key_id, seed = key_id_and_seed[:8], key_id_and_seed[8:]
    digest = hashlib.blake2b(blob, digest_size=64).digest()
    sig = signing.sign_raw(seed, digest)
    trusted = comment.encode()
    global_sig = signing.sign_raw(seed, sig + trusted)
    text = ("untrusted comment: signature from minisign secret key\n" + base64.b64encode(b"ED" + key_id + sig).decode() + "\n"
            f"trusted comment: {comment}\n" + base64.b64encode(global_sig).decode() + "\n")
    return base64.b64encode(text.encode()).decode()


class FeedServer:
    """Serves `root` on 127.0.0.1 with Range support. `cut` = {path-suffix: bytes}: the first response for that path stops after
    that many bytes (once), closing the connection. `requests` records (method, path, range)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.requests: list[tuple[str, str, str | None]] = []
        self.cut: dict[str, int] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(("GET", self.path, rng))
                target = (outer.root / self.path.lstrip("/")).resolve()
                if outer.root.resolve() not in target.parents or not target.is_file():
                    self.send_error(404)
                    return
                data = target.read_bytes()
                start, status = 0, 200
                if rng:
                    match = re.fullmatch(r"bytes=(\d+)-", rng)
                    if match:
                        start, status = int(match.group(1)), 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                limit = next((n for suffix, n in list(outer.cut.items()) if self.path.endswith(suffix)), None)
                if limit is not None:
                    del outer.cut[next(s for s in outer.cut if self.path.endswith(s))]
                    self.wfile.write(body[:limit])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False
