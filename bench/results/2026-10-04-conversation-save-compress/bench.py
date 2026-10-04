# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compress the parts split.py wrote with blosc2, zstd and lz4; print one JSON row per setting.

Usage: bench.py PARTS OUT.json, where PARTS holds one folder per file (named after its KV format: fp16, int8, q4_0)
with split.py's gdn.bin, k.bin, v.bin and pooled.bin.  Needs the PyPI packages blosc2, zstandard, lz4 and numpy.
A ratio is compressed / original bytes; c_gbs and d_gbs are compression and decompression in GB/s (the best of
three decompressions, one compression).
"""
import json
import os
import sys
import time
from typing import Any, Callable

import blosc2
import lz4.frame
import numpy as np
import zstandard


def timed(fn: Callable[[], Any], runs: int) -> tuple[float, Any]:
    best, out = float("inf"), None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def byte_shuffle(data: bytes, typesize: int) -> bytes:
    """The bytes of each element position together (blosc's SHUFFLE, done by hand for zstd and lz4)."""
    if typesize == 1:
        return data
    return np.frombuffer(data, np.uint8).reshape(-1, typesize).T.copy().tobytes()


def main() -> None:
    root, out = sys.argv[1], sys.argv[2]
    rows: list[dict[str, Any]] = []
    for kv in sorted(os.listdir(root)):
        for part in ("gdn", "k", "v", "pooled"):
            with open(os.path.join(root, kv, part + ".bin"), "rb") as f:
                data = f.read()
            # FP32 running states and indexer rows; FP16, INT8 or Q4_0 K/V
            typesize = (2 if kv == "fp16" else 1) if part in ("k", "v") else 4

            def record(lib: str, cfg: str, ct: float, dt: float, size: int) -> None:
                rows.append({"kv": kv, "part": part, "mib": round(len(data) / 2**20, 1), "lib": lib, "cfg": cfg,
                             "ratio": round(size / len(data), 3), "c_gbs": round(len(data) / ct / 1e9, 2),
                             "d_gbs": round(len(data) / dt / 1e9, 2)})

            for threads in (4, 28):
                blosc2.set_nthreads(threads)
                for codec in ("LZ4", "LZ4HC", "ZSTD", "BLOSCLZ"):
                    for filt in ("NOFILTER", "SHUFFLE", "BITSHUFFLE"):
                        for level in (1, 5, 9):
                            params = {"codec": getattr(blosc2.Codec, codec), "clevel": level, "typesize": typesize,
                                      "filters": [getattr(blosc2.Filter, filt)], "nthreads": threads}
                            ct, packed = timed(lambda: blosc2.compress2(data, **params), 1)
                            dt, _ = timed(lambda: blosc2.decompress2(packed), 3)
                            record("blosc2", f"{codec}/{filt}/c{level}/t{threads}", ct, dt, len(packed))
            for shuffled in (False, True) if typesize > 1 else (False,):
                src = byte_shuffle(data, typesize) if shuffled else data
                tag = "+byteshuffle" if shuffled else ""
                for level in (-1, 1, 3):
                    for threads in (1, 4, 28):
                        z = zstandard.ZstdCompressor(level=level, threads=threads if threads > 1 else 0)
                        ct, packed = timed(lambda: z.compress(src), 1)
                        unz = zstandard.ZstdDecompressor()
                        dt, _ = timed(lambda: unz.decompress(packed, max_output_size=len(src)), 3)
                        name = "fast=1" if level < 0 else str(level)
                        record("zstd", f"{name}/T{threads}{tag}", ct, dt, len(packed))
                for name, level in (("-1", 0), ("fast", -1), ("-9", 9)):
                    ct, packed = timed(lambda: lz4.frame.compress(src, compression_level=level), 1)
                    dt, _ = timed(lambda: lz4.frame.decompress(packed), 3)
                    record("lz4", name + tag, ct, dt, len(packed))
            print(kv, part, "done", file=sys.stderr, flush=True)
    with open(out, "w") as w:
        json.dump(rows, w, indent=0)


if __name__ == "__main__":
    main()
