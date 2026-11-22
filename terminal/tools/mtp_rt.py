"""tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.

    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt
    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q4_0.gguf --out <Strata>/mtp-bf16/rt-q4

Writes
  experts.bin   512 routed experts:
                  * Q2_0 GGUF: the engine's S2 blob layout (`include/strata/kernels/cpu/expert.hpp`), gate/up rows
                    interleaved (2r = gate r, 2r+1 = up r), then down rows; codes plane then fp16 scales plane.
                    Backward-compatible with existing `rt/` packs (no `experts.txt` written).
                  * Native GGUF (Q4_0, IQ4_NL, Q8_0, etc.): raw GGUF blocks in `NativeExpertLayout` order
                    ([gate rows | up rows | down rows] per expert), paired with `experts.txt` so `MtpDrafter`
                    dispatches `native_expert_grouped`.
  experts.txt   written for native expert packs: format, gu_type, d_type, n_expert, n_embd, n_ff, blob_bytes.
  dense.bin     every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding) so the
                engine's multi-column MMVQ can run them; the hyper-connection and router weights kept BF16; the
                RMSNorm weights as F32 with the Gemma "+1" applied (vLLM GemmaRMSNorm scales by 1 + w).
  dense.txt     one line per tensor: name kind rows cols offset bytes   (kind = q8_0 | bf16 | f32)
"""
from __future__ import annotations
import sys
if sys.platform == "win32":
    # CPython 3.12's WMI helper (_wmi, behind platform.system() / uname()) takes an access violation in a worker thread when WMI refuses the
    # account (0x80041003) and the caller has already gone (DIAG-WINHOST-0216). Without _wmi, platform uses the registry and the environment.
    sys.modules["_wmi"] = None

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import gguf_lib  # noqa: E402
gguf = gguf_lib()

H, FF, NE = 2560, 640, 512
BLOB = 3 * (H * FF * 18 // 64)
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """gu: (1280, 720) Q2_0 rows (gate rows 0..639, up rows 640..1279); dn: (2560, 180)."""
    gub = gu.reshape(2 * FF, H // 64, 18)
    inter = np.empty_like(gub)
    inter[0::2] = gub[:FF]
    inter[1::2] = gub[FF:]
    dnb = dn.reshape(H, FF // 64, 18)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, len(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    gu_t = tens["mtp.layers.0.mlp.experts.gate_up_proj"]
    dn_t = tens["mtp.layers.0.mlp.experts.down_proj"]
    gu_type = int(gu_t.tensor_type)
    d_type = int(dn_t.tensor_type)
    gu = np.asarray(gu_t.data)
    dn = np.asarray(dn_t.data)
    exp_meta = out / "experts.txt"
    q2_0_id = int(gguf.GGMLQuantizationType.Q2_0)
    if gu_type == q2_0_id and d_type == q2_0_id:
        assert gu.shape == (NE, 2 * FF, 720) and dn.shape == (NE, H, 180), (gu.shape, dn.shape)
        with open(out / "experts.bin", "wb") as f:
            for e in range(NE):
                f.write(blob_of(gu[e], dn[e]))
        if exp_meta.exists():
            exp_meta.unlink()
        exp_bytes = NE * BLOB
        exp_desc = "q2_0 (s2)"
    else:
        gu_enum = gguf.GGMLQuantizationType(gu_type)
        d_enum = gguf.GGMLQuantizationType(d_type)
        gu_blk, gu_bsz = gguf.GGML_QUANT_SIZES[gu_enum]
        d_blk, d_bsz = gguf.GGML_QUANT_SIZES[d_enum]
        if H % gu_blk != 0 or FF % d_blk != 0:
            raise ValueError(f"expert dims ({H}, {FF}) incompatible with {gu_enum.name}/{d_enum.name} blocks ({gu_blk}, {d_blk})")
        gu_row = (H // gu_blk) * gu_bsz
        d_row = (FF // d_blk) * d_bsz
        assert gu.shape == (NE, 2 * FF, gu_row) and dn.shape == (NE, H, d_row), (
            f"unexpected shapes {gu.shape}, {dn.shape}; expected ({NE}, {2 * FF}, {gu_row}), ({NE}, {H}, {d_row})"
        )
        # NativeExpertLayout: [gate rows (FF, gu_row) | up rows (FF, gu_row) | down rows (H, d_row)] in raw GGUF blocks.
        blob_bytes = 2 * FF * gu_row + H * d_row
        with open(out / "experts.bin", "wb") as f:
            for e in range(NE):
                part = np.ascontiguousarray(gu[e]).tobytes() + np.ascontiguousarray(dn[e]).tobytes()
                assert len(part) == blob_bytes, len(part)
                f.write(part)
        exp_meta.write_text(
            f"format native\n"
            f"gu_type {gu_type}\n"
            f"d_type {d_type}\n"
            f"n_expert {NE}\n"
            f"n_embd {H}\n"
            f"n_ff {FF}\n"
            f"blob_bytes {blob_bytes}\n",
            encoding="utf-8",
        )
        exp_bytes = NE * blob_bytes
        exp_desc = f"native {gu_enum.name}/{d_enum.name}"
    lines = []
    off = 0
    with open(out / "dense.bin", "wb") as f:
        for name, t in tens.items():
            if "experts." in name:
                continue
            short = name[len("mtp."):]
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            data = np.asarray(t.data)
            if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
                arr = (data.astype(np.float32) + 1.0)
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:                              # BF16
                u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
                u16 = u16.reshape(data.shape[0], -1)
                rows, cols = u16.shape
                if short in Q8:
                    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                    raw, kind = q8_0(f32), "q8_0"
                else:
                    raw, kind = u16.tobytes(), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {exp_bytes} B [{exp_desc}], dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
