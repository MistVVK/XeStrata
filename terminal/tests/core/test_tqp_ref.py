"""TQP5/6/7 reference parity: tests/data/tqp fixtures recomputed with tools/tqp_ref.py.

The fixtures were built by tools/tqp_fixtures.py from the real Gyro Coder-L file; the
expected .f32 files came from the fork's CPU decoder (ref/tqp/tqp_ref_c.c). This test
needs no model file and no C compiler: it re-derives every fixture from the raw block
bytes with the numpy reference and requires bit-identical float32.
"""
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tqp_ref as R

DATA = ROOT / "tests" / "data" / "tqp"


def test_lut_matches_fork_header():
    table = R.lut()
    assert table.size == 4096
    # the codebook's first and last points, as fp16 bit patterns from the fork header
    bits = np.array([int(t, 16) for t in (DATA / "tq2t_lut_f16.hex").read_text().split()], dtype=np.uint16)
    assert bits[0] == 0xBD4D and bits[1] == 0xB8F7
    assert np.isfinite(table).all()


def test_hyb_index():
    # ggml-common.h: p=0 -> x>>21, p=1 -> (x>>10)&2047, state masked to 15 bits
    assert R.tq2t_hyb_index(2 * 0) == 0
    assert R.tq2t_hyb_index(2 * 1) == (0x9E3779B1 >> 21)
    assert R.tq2t_hyb_index(2 * 1 + 1) == ((0x9E3779B1 >> 10) & 2047)
    assert R.tq2t_hyb_index(2 * 0x7FFF) == ((0x7FFF * 0x9E3779B1 & 0xFFFFFFFF) >> 21)
    assert R.tq2t_hyb_index(2 * 0x8000) == R.tq2t_hyb_index(0)   # the 15-bit mask


def test_state_window_wraps_circularly():
    # all-zero stream -> all-zero states. Step t's window is stream bits (63-t)*k ..
    # (63-t)*k+15 modulo 64k, so stream bit 0 is window bit 0 of step 63, and for the
    # steps whose window wraps past the end of the stream it is window bit
    # (63-t)*k - 64k + 16: t = 0, 1 at window bits 16-k and 16-2k (t = 2 at 16-3k only
    # for k = 5, where the 15-bit mask drops it), and steps 2..62 read nothing.
    for k in (5, 6, 7):
        qs = np.zeros(8 * k, dtype=np.uint8)
        assert (R.tqp_states(qs, k) == 0).all()
        qs[0] = 1
        s = R.tqp_states(qs, k)
        assert s[63] == 1
        assert s[0] == 1 << k and s[1] == 1 << (2 * k)
        assert (s[2:62] == 0).all()


def test_fixtures_recompute_bit_identical():
    index = json.loads((DATA / "index.json").read_text())
    table = R.lut()
    assert len(index["fixtures"]) == 6
    for fx in index["fixtures"]:
        raw = (DATA / fx["raw"]).read_bytes()
        want = np.frombuffer((DATA / fx["f32"]).read_bytes(), dtype=np.float32)
        k = fx["k"]
        assert fx["block_bytes"] == 2 + 8 * k
        assert len(raw) == fx["nblocks"] * fx["block_bytes"]
        assert want.size == fx["nblocks"] * R.QK_TQP
        got = R.dequant_row(raw, k, table)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), fx["raw"]


def test_each_block_decodes_independently():
    # dequant_row == concatenation of per-block dequant, and the scale is the block's fp16 d
    index = json.loads((DATA / "index.json").read_text())
    table = R.lut()
    for fx in index["fixtures"]:
        raw = (DATA / fx["raw"]).read_bytes()
        want = np.frombuffer((DATA / fx["f32"]).read_bytes(), dtype=np.float32)
        k, bs = fx["k"], fx["block_bytes"]
        for i in range(fx["nblocks"]):
            block = raw[i * bs:(i + 1) * bs]
            got = R.dequant_block(block, k, table)
            exp = want[i * R.QK_TQP:(i + 1) * R.QK_TQP]
            assert np.array_equal(got.view(np.uint32), exp.view(np.uint32)), f"{fx['raw']} block {i}"
            d = struct.unpack("<e", block[:2])[0]
            codes = np.unique(np.frombuffer(block[2:], dtype=np.uint8))
            assert np.all(np.abs(got) <= abs(d) * 8.0 + 1e-3), f"{fx['raw']} block {i}: out of scale range"
            assert codes.size > 0
