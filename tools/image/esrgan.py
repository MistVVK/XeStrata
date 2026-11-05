# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/esrgan.py - RealESRGAN x4plus's official weights (RealESRGAN_x4plus.pth by Xintao Wang, Liangbin Xie,
Chao Dong and Ying Shan; xinntao/Real-ESRGAN, BSD-3-Clause) to the safetensors file the image engine's upscaler reads
(src/upscale/esrgan.cpp), without PyTorch.

    python -I tools/image/esrgan.py IN.pth OUT.safetensors [--sha256 HEX]

The .pth is a PyTorch zip of a pickled state dict ({"params_ema": {name: tensor}}).  It is read by yolo_pt.py's reader,
which builds nothing but its own records (the tensors' storages, offsets, shapes and strides; any global outside a
state dict's few refuses the file), and only RRDBNet's tensors are taken: conv_first, body.<i>.rdb<1-3>.conv<1-5>,
conv_body, conv_up1, conv_up2, conv_hr and conv_last, each a weight and a bias.  What it writes changes no weight:
params_ema's tensors as the file stores them (FP32), with the source's SHA-256 in the header's __metadata__, written
to a temporary file and renamed into place when complete.  The server runs it when an upscale first needs the weights
(serve/image_aux.py, qimage.AUX "realesrgan").
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # yolo_pt beside it (run with -I: no script folder)
import yolo_pt  # noqa: E402

NAMES = re.compile(r"(conv_first|body\.\d+\.rdb[123]\.conv[1-5]|conv_body|conv_up[12]|conv_hr|conv_last)"
                   r"\.(weight|bias)")
DTYPES = ("F32", "F16", "BF16")


def read(path: Path) -> dict:
    """The state dict's params_ema (else params, else the dict itself) -> {name: (dtype, shape, bytes)}."""
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
        top = yolo_pt.Reader(data, zf, prefix).load()
        if not isinstance(top, dict):
            raise yolo_pt.Refused("a checkpoint that is not a dict")
        sd = next((v for v in (top.get("params_ema"), top.get("params")) if isinstance(v, dict)), top)
        out: dict = {}
        cache: dict = {}
        for name, t in sd.items():
            if not isinstance(t, yolo_pt.Tensor) or not NAMES.fullmatch(str(name)):
                raise yolo_pt.Refused(f"{name!r}: not one of RRDBNet's tensors")
            if t.storage.kind.dtype not in DTYPES:
                raise yolo_pt.Refused(f"{name}: a tensor of {t.storage.kind.dtype}")
            out[name] = (t.storage.kind.dtype, list(t.shape), yolo_pt._bytes(zf, t, cache))
        if "conv_first.weight" not in out or "conv_last.weight" not in out:
            raise yolo_pt.Refused("no conv_first or conv_last: not RRDBNet's weights")
        return out


def convert(src: Path, out: Path, want_sha: str | None = None) -> int:
    """Converts `src` into `out` -> the tensors written.  Raises yolo_pt.Refused (or zipfile's errors) on a file it
    does not take, and ValueError when `want_sha` is given and the file's SHA-256 is another."""
    got = yolo_pt.sha256(src)
    if want_sha and got != want_sha.lower():
        raise ValueError(f"{src.name}: SHA-256 {got}, not the pinned {want_sha}")
    blobs = read(src)
    yolo_pt.write(out, blobs, {"source": src.name, "source_sha256": got, "format": "realesrgan-rrdbnet",
                               "license": "BSD-3-Clause"})
    return len(blobs)


def main(argv: list[str]) -> int:
    args, want = [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--sha256" and i + 1 < len(argv):
            want = argv[i + 1]
            i += 2
            continue
        args.append(argv[i])
        i += 1
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        n = convert(Path(args[0]), Path(args[1]), want)
    except (yolo_pt.Refused, ValueError, zipfile.BadZipFile, EOFError) as e:
        print(f"esrgan: {e}", file=sys.stderr)
        return 1
    print(f"esrgan: {args[1]} ({n} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
