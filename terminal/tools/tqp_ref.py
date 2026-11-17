"""TQP5 / TQP6 / TQP7 (pair trellis, group 128) - the exact numpy reference decoder.

Ported line-for-line from agentionai's llama.cpp fork, ggml/src/ggml-quants.c
(dequantize_row_tqp_impl) and ggml/src/ggml-common.h (tqp_state, tq2t_hyb_index,
block_tqp5/6/7, tq2t_lut_f16). This is the parity reference the Metal kernels are
checked against; the C harness in ref/tqp/tqp_ref_c.c is the same decoder straight
from the fork, and the two must agree bit-for-bit in float32.

Block layout (QK_TQP = 128 weights): fp16 scale d, then a circular bit stream of
64 steps x K bits (K = 5 / 6 / 7), i.e. 8*K bytes - 41 / 49 / 57 bytes per block.
Step t (weights 2t, 2t+1) has the 16-bit state window starting at stream bit
(63 - t)*K, LSB-first, circular modulo 64*K; the state is masked to 15 bits and
hashed into the shared TQ2_T "Hyb" codebook (2048 fp16 pairs): index e = 2*s,
pair j = tq2t_lut_f16[2*tq2t_hyb_index(e) + j]; weight = d * that fp16 value.

The codebook lives in tests/data/tqp/tq2t_lut_f16.hex (4096 fp16 bit patterns,
extracted from the fork's header; must stay bit-identical to the encoder's table).
"""
from __future__ import annotations

import pathlib
import struct

import numpy as np

QK_TQP = 128
TQP_BITS = {"TQP5": 5, "TQP6": 6, "TQP7": 7}
TQP_IDS = {"TQP5": 147, "TQP6": 148, "TQP7": 149}
# (block elements, bytes per block): fp16 scale + 8*K stream bytes
TQP_GEOMETRY = {name: (QK_TQP, 2 + 8 * k) for name, k in TQP_BITS.items()}

_LUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "tqp" / "tq2t_lut_f16.hex"

TQ_STATE_MASK = (1 << 15) - 1


def load_lut(path: pathlib.Path | None = None) -> np.ndarray:
    """The TQ2_T Hyb codebook as float32 values (fp16 bit patterns decoded), 4096 entries."""
    raw = (path or _LUT_PATH).read_text().split()
    if len(raw) != 4096:
        raise ValueError(f"{path or _LUT_PATH}: expected 4096 fp16 patterns, got {len(raw)}")
    bits = np.array([int(t, 16) for t in raw], dtype=np.uint16)
    return bits.view(np.float16).astype(np.float32)


_LUT: np.ndarray | None = None


def lut() -> np.ndarray:
    global _LUT
    if _LUT is None:
        _LUT = load_lut()
    return _LUT


def tq2t_hyb_index(e: int) -> int:
    """ggml-common.h: x = (s masked to 15) * 0x9e3779b1 (u32 wrap); p=0 -> x>>21, p=1 -> (x>>10)&2047."""
    x = ((e >> 1) & TQ_STATE_MASK) * 0x9E3779B1 & 0xFFFFFFFF
    return ((x >> 10) & 2047) if (e & 1) else (x >> 21)


def tqp_states(qs: np.ndarray, k: int) -> np.ndarray:
    """The 64 15-bit states of one block: stream bit i = (qs[i>>3] >> (i&7)) & 1, window at bit (63-t)*k."""
    bits = np.unpackbits(qs, bitorder="little")          # circular stream, 64*k bits
    n = 64 * k
    starts = (63 - np.arange(64)) * k
    idx = (starts[:, None] + np.arange(16)[None, :]) % n
    s = bits[idx].dot(1 << np.arange(16)).astype(np.uint32)
    return s & TQ_STATE_MASK


def dequant_block(block: bytes, k: int, table: np.ndarray | None = None) -> np.ndarray:
    """Dequantise one 128-weight TQP block (2 + 8*k bytes) to 128 float32, exactly as the fork's CPU decoder."""
    if len(block) != 2 + 8 * k:
        raise ValueError(f"block is {len(block)} B, expected {2 + 8 * k} B for K={k}")
    if table is None:
        table = lut()
    d = struct.unpack("<e", block[:2])[0]
    qs = np.frombuffer(block[2:], dtype=np.uint8)
    s = tqp_states(qs, k)
    x = s * 0x9E3779B1 & 0xFFFFFFFF
    idx = (x >> 21).astype(np.int64)                     # pair p = 0 of entry e = 2*s
    pair = np.stack([table[2 * idx], table[2 * idx + 1]], axis=1)
    return (d * pair).reshape(-1).astype(np.float32)


def dequant_row(blocks: bytes, k: int, table: np.ndarray | None = None) -> np.ndarray:
    """Dequantise a whole row of blocks (concatenated), as dequantize_row_tqp_impl does."""
    bsize = 2 + 8 * k
    if len(blocks) % bsize:
        raise ValueError(f"{len(blocks)} B is not a whole number of {bsize} B blocks")
    out = np.empty(len(blocks) // bsize * QK_TQP, dtype=np.float32)
    for i in range(len(blocks) // bsize):
        out[i * QK_TQP:(i + 1) * QK_TQP] = dequant_block(blocks[i * bsize:(i + 1) * bsize], k, table)
    return out


# ---------------------------------------------------------------- GGUF access
def _patch_reader(reader):
    """Add the TQP ids/geometry to tools/gguf_reader.py locally, until phase 1 lands them."""
    changed = []
    for name, tid in TQP_IDS.items():
        if reader.GGML_TYPES.get(tid) != name:
            reader.GGML_TYPES[tid] = name
            changed.append(tid)
    for name, geom in TQP_GEOMETRY.items():
        if reader.BLOCK_GEOMETRY.get(name) != geom:
            reader.BLOCK_GEOMETRY[name] = geom
            changed.append(name)
    return changed


def open_gguf(path, reader=None):
    import gguf_reader
    if reader is None:
        reader = gguf_reader
    _patch_reader(reader)
    return reader.GGUFFile(pathlib.Path(path))


def tensor_blocks(gguf, tensor, first_block: int, count: int) -> bytes:
    """Raw bytes of `count` consecutive blocks of a named tensor, starting at `first_block`."""
    info = next((t for t in gguf.tensors if t.name == tensor), None)
    if info is None:
        raise KeyError(tensor)
    geom = TQP_GEOMETRY[info.type_name]
    if info.elements % geom[0]:
        raise ValueError(f"{tensor}: {info.elements} elements not a whole number of {geom[0]}-weight blocks")
    start = gguf.data_start + info.offset + first_block * geom[1]
    with gguf.path.open("rb") as fh:
        fh.seek(start)
        data = fh.read(count * geom[1])
    if len(data) != count * geom[1]:
        raise ValueError(f"{tensor}: short read at block {first_block}")
    return data
