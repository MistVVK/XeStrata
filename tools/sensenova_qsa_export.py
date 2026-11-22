# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Append the trained SenseNova-Loop QSA indexers to an official GGUF."""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from gguf_reader import GGUFFile

VISITS = range(3, 56, 4)
SHAPES = {"q_proj.weight": [512, 4096], "k_proj.weight": [128, 4096],
          "q_norm": [128], "k_norm": [128], "k_block_norm": [128]}


def string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(source: Path, indexer: Path, output: Path) -> None:
    g = GGUFFile(source)
    if g.metadata.get("general.architecture") != "sensenova_loop":
        raise ValueError("expected an official sensenova_loop GGUF")
    entries = {}
    with indexer.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        base = f.tell()
        expected = {f"{v}.{n}" for v in VISITS for n in SHAPES}
        if set(header) != expected:
            raise ValueError("indexer tensor names do not match the 14 visits")
        for v in VISITS:
            for name, shape in SHAPES.items():
                item = header[f"{v}.{name}"]
                count = 1
                for dim in shape:
                    count *= dim
                start, end = item["data_offsets"]
                if item["dtype"] != "F32" or item["shape"] != shape or end - start != count * 4:
                    raise ValueError(f"incorrect tensor {v}.{name}")
                f.seek(base + start)
                data = f.read(end - start)
                if len(data) != end - start:
                    raise ValueError(f"truncated tensor {v}.{name}")
                if name.endswith("norm"):
                    values = array.array("f")
                    values.frombytes(data)
                    if sys.byteorder != "little":
                        values.byteswap()
                    values = array.array("f", (x + 1.0 for x in values))
                    if sys.byteorder != "little":
                        values.byteswap()
                    data = values.tobytes()
                suffix = name if name.endswith(".weight") else name + ".weight"
                entries[f"blk.{v}.indexer.{suffix}"] = (shape, data)
    if set(entries) & {t.name for t in g.tensors}:
        raise ValueError("source already has indexer tensors")
    metadata: dict[str, int | str] = {
        "sensenova_loop.qsa.head_count": 4,
        "sensenova_loop.qsa.key_length": 128,
        "sensenova_loop.qsa.rope_length": 64,
        "sensenova_loop.qsa.block_size": 4,
        "sensenova_loop.qsa.budget": 2048,
        "sensenova_loop.qsa.rope_table_dtype": "bf16",
        "sensenova_loop.qsa.indexer_sha256": digest(indexer),
    }
    if set(metadata) & g.metadata.keys():
        raise ValueError("source already has QSA metadata")
    with source.open("rb") as f:
        f.seek(24)
        g._pos = 24
        for _ in g.metadata:
            g._str(f)
            g._value(f)
        meta_end = f.tell()
        for _ in g.tensors:
            g._str(f)
            dims, = g._unpack(f, "<I")
            g._read(f, dims * 8 + 12)
        directory_end = f.tell()
        f.seek(24)
        old_meta = f.read(meta_end - 24)
        old_directory = f.read(directory_end - meta_end)
        original_data_size = source.stat().st_size - g.data_start
        next_offset = align(original_data_size, g.alignment)
        new_directory = bytearray()
        for name, (shape, data) in entries.items():
            dims = list(reversed(shape))
            new_directory += string(name) + struct.pack("<I", len(dims))
            new_directory += struct.pack(f"<{len(dims)}Q", *dims)
            new_directory += struct.pack("<IQ", 0, next_offset)
            next_offset = align(next_offset + len(data), g.alignment)
        new_meta = bytearray()
        for name, value in metadata.items():
            new_meta += string(name)
            if isinstance(value, int):
                new_meta += struct.pack("<II", 4, value)
            else:
                new_meta += struct.pack("<I", 8) + string(value)
        with output.open("xb") as out:
            out.write(struct.pack("<IIQQ", 0x46554747, 3, len(g.tensors) + len(entries),
                                  len(g.metadata) + len(metadata)))
            out.write(old_meta + new_meta + old_directory + new_directory)
            out.write(bytes(align(out.tell(), g.alignment) - out.tell()))
            f.seek(g.data_start)
            shutil.copyfileobj(f, out, 8 * 1024 * 1024)
            out.write(bytes(align(original_data_size, g.alignment) - original_data_size))
            for _, data in entries.values():
                out.write(data)
                out.write(bytes(align(len(data), g.alignment) - len(data)))
    print(f"{digest(output)}  {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("indexer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.indexer, args.output)


if __name__ == "__main__":
    main()
