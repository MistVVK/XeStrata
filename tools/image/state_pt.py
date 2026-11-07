# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/state_pt.py - a PyTorch checkpoint of a state dict (a .pth: a zip of a pickle and its tensors' bytes)
to a safetensors file the image engine reads (the control images' networks, src/image/cnet_*.cpp), without PyTorch.

    python -I tools/image/state_pt.py IN.pth OUT.safetensors [--sha256 HEX]

The pickle is read by yolo_pt.py's unpickler (nothing built but tensors' records and dicts; a global outside its list
stops the read), with inert stand-ins for the few globals a training checkpoint's bookkeeping carries beside its
weights (mmengine's message hub in DWPose's: numpy arrays and scalars, a history buffer): each returns an empty
placeholder, its arguments unread, so no code of the checkpoint's runs.  The weights are the checkpoint's
"state_dict", else its "model" when that is a dict of tensors, else the top dict itself; each tensor's bytes are
copied as stored (FP16 or FP32, made contiguous where a stride asks) under its name, with the source file's SHA-256 in
the header's __metadata__.  The output is written to a temporary file and renamed into place only when complete.
"""
from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_pt  # noqa: E402

# a training checkpoint's bookkeeping (not weights): read as placeholders
INERT = {("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"),
         ("numpy", "dtype"), ("_codecs", "encode"), ("mmengine.logging.history_buffer", "HistoryBuffer")}


class Inert:
    """A placeholder for a value the read does not need: built from anything, it keeps nothing."""

    def __init__(self, *args, **kw):
        pass

    def __setstate__(self, state):
        pass


def _inert(*args, **kw):
    return Inert()


class Reader(yolo_pt.Reader):
    def find_class(self, module, name):
        if (module, name) in INERT:
            return Inert if name in ("ndarray", "HistoryBuffer") else _inert
        return super().find_class(module, name)


def read(path: Path) -> dict:
    """The checkpoint's weights at `path`: name -> (dtype, shape, bytes)."""
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) > yolo_pt.MAX_MEMBERS or sum(i.file_size for i in infos) > yolo_pt.MAX_TOTAL:
            raise yolo_pt.Refused("an archive past the size limits")
        for i in infos:
            if i.filename.startswith("/") or ".." in i.filename.split("/") or "\\" in i.filename:
                raise yolo_pt.Refused(f"the member name {i.filename!r}")
        pkl = [i for i in infos if i.filename.endswith("/data.pkl") and i.filename.count("/") == 1]
        if len(pkl) != 1 or pkl[0].file_size > yolo_pt.MAX_PICKLE:
            raise yolo_pt.Refused("no data.pkl, or more than one, or one past the size limit")
        prefix = pkl[0].filename[:-len("data.pkl")]
        data = zf.read(pkl[0])
        yolo_pt._check_opcodes(data)
        top = Reader(data, zf, prefix).load()
        if not isinstance(top, dict):
            raise yolo_pt.Refused("a checkpoint that is not a dict")
        sd: dict = top
        for key in ("state_dict", "model"):
            if isinstance(top.get(key), dict):
                sd = top[key]
                break
        tensors = {k: v for k, v in sd.items() if isinstance(k, str) and isinstance(v, yolo_pt.Tensor)}
        if not tensors or len(tensors) > yolo_pt.MAX_TENSORS:
            raise yolo_pt.Refused("no weights, or too many")
        cache: dict = {}
        return {k: (t.storage.kind.dtype, list(t.shape), yolo_pt._bytes(zf, t, cache)) for k, t in tensors.items()}


def convert(src: Path, out: Path, want_sha: str | None = None) -> dict:
    """Converts `src` into `out`; the metadata written.  Raises Refused (or zipfile's errors) on a file it does not
    take, and ValueError when `want_sha` is given and the file's SHA-256 is another."""
    got = yolo_pt.sha256(src)
    if want_sha and got != want_sha.lower():
        raise ValueError(f"{src.name}: SHA-256 {got}, not the pinned {want_sha}")
    blobs = read(src)
    meta = {"source": src.name, "source_sha256": got}
    yolo_pt.write(out, blobs, meta)
    return {**meta, "tensors": len(blobs)}


def main(argv: list[str]) -> int:
    want = None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--sha256":
            want = next(it, None)
        elif a.startswith("--sha256="):
            want = a.split("=", 1)[1]
        else:
            args.append(a)
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        meta = convert(Path(args[0]), Path(args[1]), want)
    except (yolo_pt.Refused, ValueError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError) as e:
        print(f"state_pt: {e}", file=sys.stderr)
        return 1
    print(f"state_pt: {args[1]} ({meta['tensors']} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
