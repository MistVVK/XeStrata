"""tools/image/gate/png_pixels.py - TEST-ONLY: the 8-bit pixels of a PNG (8-bit grey / RGB / RGBA, not interlaced),
decoded with Python's zlib and the five PNG row filters (stb_image_write, which sd-cli uses, picks a filter per row);
and the comparison with a raw interleaved file.

    python -I -B png_pixels.py <file.png> <raw> [<raw> ...]     prints, per raw file, identical or the count of
                                                                differing bytes; exit 1 if the first raw differs"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path


def png_pixels(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    at, idat, ihdr, text = 8, [], None, {}
    while at < len(data):
        n, = struct.unpack(">I", data[at:at + 4])
        typ, body = data[at + 4:at + 8], data[at + 8:at + 8 + n]
        if typ == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"tEXt":
            k, _, v = body.partition(b"\x00")
            text[k.decode("latin-1")] = v.decode("latin-1")
        at += 12 + n
    w, h, depth, ctype, _, _, interlace = ihdr
    assert depth == 8 and interlace == 0
    bpp = {0: 1, 4: 2, 2: 3, 6: 4}[ctype]
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    out = bytearray(h * stride)
    prev = bytearray(stride)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            else:
                assert f == 0, f
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return (w, h, bpp), bytes(out), text


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    (w, h, c), px, _ = png_pixels(Path(a[0]).read_bytes())
    first_bad = False
    for i, p in enumerate(a[1:]):
        raw = Path(p).read_bytes()
        n = sum(1 for x, y in zip(raw, px) if x != y) if len(raw) == len(px) else -1
        print(f"{Path(p).name} vs {Path(a[0]).name} ({w} x {h} x {c}): " +
              ("identical pixels" if n == 0 else f"{n} bytes differ" if n > 0 else "different size"))
        if i == 0:
            first_bad = n != 0
    return 1 if first_bad else 0


if __name__ == "__main__":
    sys.exit(main())
