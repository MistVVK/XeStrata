# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Split a --conversation-save file (.xsc, format version 1, src/core/conversation_disk.cpp) into its parts by kind.

Usage: split.py FILE OUTDIR.  Writes OUTDIR/<kind>.bin, every part of one kind concatenated in file order: the
running states' gdn, ple, tails, dead and block_pos, then each K/V layer's k, v, k_scale, v_scale and pooled.
"""
import os
import struct
import sys


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    parts: dict[str, bytearray] = {}
    with open(src, "rb") as f:
        if f.read(4) != b"XSCV" or struct.unpack("<I", f.read(4))[0] != 1:
            sys.exit(f"{src}: not a version 1 conversation file")
        header = f.read(struct.unpack("<Q", f.read(8))[0])
        f.read(8)  # the header's checksum
        o = 4 + struct.unpack_from("<I", header, 0)[0] + 1 + 18 * 8  # fingerprint, cvec, geometry
        tokens = struct.unpack_from("<Q", header, o)[0]
        o += 8 + 4 * tokens
        o += 8 + 16 * struct.unpack_from("<Q", header, o)[0]  # image keys
        checkpoints = struct.unpack_from("<Q", header, o)[0]

        def part(kind: str) -> None:
            n, codec = struct.unpack("<QB", f.read(9))
            if codec != 0:
                sys.exit(f"{src}: a compressed part")
            parts.setdefault(kind, bytearray()).extend(f.read(n))

        for _ in range(1 + checkpoints):
            for kind in ("gdn", "ple", "tails", "dead", "block_pos"):
                part(kind)
        formats = set()
        layers = struct.unpack("<Q", f.read(8))[0]
        for _ in range(layers):
            formats.add(struct.unpack("<i", f.read(4))[0])
            f.read(6 * 8)  # cells, heads, head_dim, page_size, pooled_rows, idx_dim
            for kind in ("k", "v", "k_scale", "v_scale", "pooled"):
                part(kind)
        if f.read(1):
            sys.exit(f"{src}: bytes after the last layer")
    print(f"{src}: {tokens} tokens, {1 + checkpoints} running states, {layers} K/V layers, formats {sorted(formats)}")
    for kind, data in parts.items():
        with open(os.path.join(out, kind + ".bin"), "wb") as w:
            w.write(data)
        print(f"  {kind:10s} {len(data) / 2**20:9.1f} MiB")


if __name__ == "__main__":
    main()
