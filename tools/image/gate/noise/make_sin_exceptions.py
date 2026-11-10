"""tools/image/gate/noise/make_sin_exceptions.py - TEST-ONLY: sin_exc.bin from check_domain's exc_sin.txt.

glibc's double sin (x86-64, the library sd.cpp's Linux x86 build calls) is not correctly rounded for some inputs.
check_domain.cpp (mode glibc) lists every reachable v where it differs from our correctly rounded value.  For each,
the correctly rounded reference is recomputed here with Python's decimal module (50 digits) and the difference is
checked to be exactly one unit in the last place; then the list is packed:

  sin_exc.bin = b"OWN5SEX1", uint32 count (little endian), then per entry (sorted by v's float bits) one varint
                (LEB128) of ((bits - previous bits) << 1 | down), down = 1 when glibc's double is one unit below ours
                in the bit pattern (previous bits start at 0)

    python -I -B make_sin_exceptions.py exc_sin.txt         (writes sin_exc.bin beside this script)"""
import struct
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 50
MAGIC = b"OWN5SEX1"


def cr_sin(v: float) -> float:
    x = D(v)
    s, t, k = D(0), x, 1
    while abs(t) > D(10) ** -48:
        s += t
        t = -t * x * x / ((k + 1) * (k + 2))
        k += 2
    return float(s)                      # Decimal -> float is correctly rounded


def dbits(d: float) -> int:
    return struct.unpack("<q", struct.pack("<d", d))[0]


def fval(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def pack(entries) -> bytes:
    out, prev = bytearray(MAGIC + struct.pack("<I", len(entries))), 0
    for bits, down in entries:
        v = ((bits - prev) << 1) | down
        prev = bits
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                break
    return bytes(out)


def unpack(data: bytes):
    assert data[:8] == MAGIC
    n = struct.unpack("<I", data[8:12])[0]
    i, prev, out = 12, 0, []
    for _ in range(n):
        v = shift = 0
        while True:
            b = data[i]
            i += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        prev += v >> 1
        out.append((prev, v & 1))
    assert i == len(data)
    return out


def main():
    entries = []
    for line in open(sys.argv[1]):
        vb, gb = line.split()
        g = struct.unpack("<q", struct.pack("<Q", int(gb, 16)))[0]
        delta = g - dbits(cr_sin(fval(int(vb, 16))))
        if delta not in (1, -1):
            raise SystemExit(f"v bits {vb}: glibc differs by {delta} units")
        entries.append((int(vb, 16), 1 if delta < 0 else 0))
    entries.sort()
    data = pack(entries)
    assert unpack(data) == entries
    (Path(__file__).resolve().parent / "sin_exc.bin").write_bytes(data)
    print(len(entries), "entries,", len(data), "bytes")


if __name__ == "__main__":
    main()
