#!/usr/bin/env python3
"""logits_dump.py FILE [--rows] - read a STRATA_LOGITS_DUMP file record by record (#6030).

The engine (src/core/verify.cpp, sycl/src/core/verify.cpp) writes one record per verify window:
    int32 magic 0x53474F4C ("LOGS"), int32 T, int32 n_vocab, int32 record, int64 position   (24 bytes)
    then T * n_vocab float32 logits, row 0 = the window's first position.
Reading the file as one flat float32 array takes each header for logits: the magic alone reads as 856028872704.0,
the value #6030 reported as the largest "logit" at index 0.  This reader checks every header and prints, per row:
the position, the top-5 (id, logit), the number of non-finite and non-zero values, and the largest |logit|.

    python3 tools/logits_dump.py dump.bin            # one line per record (its first row)
    python3 tools/logits_dump.py dump.bin --rows     # every row of every record
Standard library only (struct, array)."""
import array
import math
import struct
import sys

MAGIC = 0x53474F4C
HEADER = struct.Struct("<iiiiq")


def records(path):
    """Yields (record, T, n_vocab, position, rows): rows is a list of T array('f') of n_vocab logits each."""
    with open(path, "rb") as f:
        at = 0
        while True:
            head = f.read(HEADER.size)
            if not head:
                return
            if len(head) < HEADER.size:
                raise ValueError(f"{path}: a truncated record header at byte {at}")
            magic, t, n_vocab, rec, pos = HEADER.unpack(head)
            if magic != MAGIC:
                raise ValueError(f"{path}: byte {at} holds {magic:#010x}, not the record magic {MAGIC:#010x} (\"LOGS\"): "
                                 "not a STRATA_LOGITS_DUMP file, or a record was cut")
            if t <= 0 or n_vocab <= 0 or t > 4096 or n_vocab > (1 << 24):
                raise ValueError(f"{path}: record {rec} at byte {at}: implausible T={t} n_vocab={n_vocab}")
            rows = []
            for _ in range(t):
                a = array.array("f")
                try:
                    a.fromfile(f, n_vocab)
                except EOFError:
                    raise ValueError(f"{path}: record {rec} at byte {at}: its logits are cut short") from None
                if sys.byteorder != "little":
                    a.byteswap()
                rows.append(a)
            at += HEADER.size + 4 * t * n_vocab
            yield rec, t, n_vocab, pos, rows


def row_stats(row):
    finite = [(v, i) for i, v in enumerate(row) if math.isfinite(v)]
    top = sorted(finite, reverse=True)[:5]
    return {"nonfinite": len(row) - len(finite), "nonzero": sum(1 for v in row if v != 0.0),
            "max_abs": max((abs(v) for v, _ in finite), default=float("nan")), "top5": [(i, round(v, 4)) for v, i in top]}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    every = "--rows" in argv
    n = 0
    for rec, t, n_vocab, pos, rows in records(argv[0]):
        for k, row in enumerate(rows if every else rows[:1]):
            s = row_stats(row)
            print(f"record {rec} T={t} n_vocab={n_vocab} position {pos + k}: top5 {s['top5']} max|x| {s['max_abs']:.4g} "
                  f"non-zero {s['nonzero']}/{n_vocab} non-finite {s['nonfinite']}")
        n += 1
    print(f"{n} record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
