"""tools/image/gate/vae_ref.py - TEST-ONLY (OWN-5 gate harness): a numpy reference of Qwen-Image-2.1's VAE decoder, as
stable-diffusion.cpp 2988060 runs it for one picture (src/model/vae/wan_vae.hpp, VERSION_QWEN_IMAGE_2_1, 2D weights).

It is the oracle for our own GPU decoder (src/image/vae): the per-stage values, the activation ranges (what must fit
the f16 operands), and the f32 picture our decode is measured against besides sd.cpp's own.

The decoder, pixel-major (HWC, float32): latent (64 ch) -> post_quant_conv 1x1 -> conv_in 3x3 (1152) -> mid block
(resnet, attention, resnet) -> up_blocks 0..4 (1152, 1152, 576, 288, 144 channels; 3 resnets each; blocks 0-3 end in
nearest-2x + 3x3 conv and add a channel-to-space "DupUp" shortcut of the block's input) -> norm_out -> SiLU -> conv_out
3x3 (4 channels: RGBA in [-1, 1]).  A resnet is  h = shortcut(x);  x = conv2(silu(norm2(conv1(silu(norm1(x)))))) + h;
the norm is an RMS norm over the channels of each pixel (eps 1e-12) times gamma.  The attention is one head over all
latent pixels, dim 1152, scale 1/sqrt(1152), q/k/v = channels [0,C), [C,2C), [2C,3C) of to_qkv.
The temporal parts of the Wan decoder (time_conv, the frame cache) do nothing for one picture and are left out.

Two arithmetic modes:
  "f32"   weights bf16 -> f32 (exact), every operation in float32 (matmuls in float64, rounded to float32).
  "f16op" the operands of every convolution as sd.cpp and our GPU decoder see them: weights bf16 -> f16 (round to
          nearest even), inputs x * 2^-7 rounded to f16, the product accumulated exactly (float64) and x 2^7; the rest
          float32.  This is "f16 operands, exact accumulation": our GPU decode (f32 accumulation) must be close to it.

    python -I -B vae_ref.py <vae.safetensors> <latent.bin> <out.npy> [--mode f32|f16op] [--stats]
        latent.bin: float32 [64, H, W] as run_set.py --dump writes it (the diffusion-space latent; shape file says
        "W H 64 1"); out.npy: float32 [H*16, W*16, 4] in [-1, 1] before sd.cpp's 0-1 scaling
    python -I -B vae_ref.py --png <out.npy> <out.png>      sd.cpp's 8-bit conversion, RGBA PNG (needs Pillow)"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

# wan_vae.hpp get_latents_mean_std, VERSION_QWEN_IMAGE_2_1, 64 channels
LATENT_MEAN = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
               1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
               2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
               -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
               -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
               -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
               -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
               0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304]
LATENT_STD = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
              3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
              3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
              3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
              3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
              3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
              3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
              3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161]

CONV_SCALE = np.float32(1.0 / 128.0)   # wan_vae.hpp: "Keep large VAE activations within the FP16 convolution range"
F16_MAX = 65504.0
# up block i: (in, out, DupUp factor_t, has upsampler); wan_vae.hpp temperal_upsample {true, true, true, false}
UP_BLOCKS = [(1152, 1152, 2, True), (1152, 1152, 2, True), (1152, 576, 2, True), (576, 288, 1, True), (288, 144, 1, False)]


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_safetensors(path, prefix="") -> dict:
    """name -> float32 array (bf16 and f32 tensors only), for names starting with prefix."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, e in hdr.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            a, b = e["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if e["dtype"] == "BF16":
                arr = bf16_to_f32(raw)
            elif e["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                raise ValueError(f"{name}: dtype {e['dtype']}")
            out[name] = arr.reshape(e["shape"])
    return out


class Decoder:
    def __init__(self, weights: dict, mode: str = "f32", stats: bool = False):
        assert mode in ("f32", "f16op")
        self.w, self.mode = weights, mode
        self.stats = {} if stats else None
        self.cache = {}

    # --- building blocks ----------------------------------------------------------------------------------------
    def _conv_w(self, name):
        """[OC, IC, KH, KW] -> [KH*KW*IC, OC] (row index (ky, kx, ic)), float64, with the mode's weight rounding."""
        if name not in self.cache:
            w = self.w[name + ".weight"]
            if self.mode == "f16op":
                w = w.astype(np.float16).astype(np.float32)
            oc, ic, kh, kw = w.shape
            self.cache[name] = (w.transpose(2, 3, 1, 0).reshape(kh * kw * ic, oc).astype(np.float64), kh,
                                self.w[name + ".bias"].astype(np.float32))
        return self.cache[name]

    def _note(self, name, x):
        if self.stats is not None:
            self.stats[name] = float(np.abs(x).max())

    def conv(self, name, x, rows_per_chunk=None):
        """3x3 (pad 1) or 1x1 conv of x [H, W, IC] -> [H, W, OC]; the input scaled by 2^-7 like sd.cpp."""
        wm, k, bias = self._conv_w(name)
        H, W, C = x.shape
        xs = x * CONV_SCALE
        self._note(name + ":in_scaled", xs)
        if self.mode == "f16op":
            if np.abs(xs).max() > F16_MAX:
                raise OverflowError(f"{name}: conv input exceeds the f16 range after the 2^-7 scale")
            xs = xs.astype(np.float16).astype(np.float32)
        oc = wm.shape[1]
        out = np.empty((H, W, oc), dtype=np.float32)
        if k == 1:
            out[:] = (xs.reshape(-1, C).astype(np.float64) @ wm).reshape(H, W, oc)
        else:
            pad = np.zeros((H + 2, W + 2, C), dtype=np.float32)
            pad[1:-1, 1:-1] = xs
            step = rows_per_chunk or max(1, (1 << 26) // (W * 9 * C))
            for y0 in range(0, H, step):
                y1 = min(H, y0 + step)
                cols = np.concatenate([pad[y0 + dy:y1 + dy, dx:dx + W] for dy in range(3) for dx in range(3)], axis=2)
                out[y0:y1] = (cols.reshape(-1, 9 * C).astype(np.float64) @ wm).reshape(y1 - y0, W, oc)
        out = out / CONV_SCALE          # x 2^7: exact
        out += bias
        return out

    def rms(self, name, x):
        """RMS norm over channels, eps 1e-12, times gamma (ggml_rms_norm: mean of squares in float32 terms)."""
        g = self.w[name + ".gamma"].reshape(-1).astype(np.float32)
        ms = np.mean(x.astype(np.float64) ** 2, axis=2, keepdims=True)
        return (x * (1.0 / np.sqrt(ms + 1e-12)).astype(np.float32)) * g

    @staticmethod
    def silu(x):
        return x / (np.float32(1.0) + np.exp(-x))

    def resnet(self, p, x):
        h = self.conv(p + ".conv_shortcut", x) if (p + ".conv_shortcut.weight") in self.w else x
        t = self.conv(p + ".conv1", self.silu(self.rms(p + ".norm1", x)))
        t = self.conv(p + ".conv2", self.silu(self.rms(p + ".norm2", t)))
        self._note(p, t + h)
        return t + h

    def attention(self, p, x):
        H, W, C = x.shape
        qkv = self.conv(p + ".to_qkv", self.rms(p + ".norm", x)).reshape(H * W, 3 * C).astype(np.float64)
        q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]
        s = (q @ k.T) / np.sqrt(C)
        s -= s.max(axis=1, keepdims=True)
        e = np.exp(s)
        a = ((e / e.sum(axis=1, keepdims=True)) @ v).astype(np.float32).reshape(H, W, C)
        return self.conv(p + ".proj", a) + x

    @staticmethod
    def dup_up(x, out_c, ft, fs=2):
        """wan_vae.hpp DupUp3D for one frame (first_chunk): output channel oc at sub-position (i, j) of each 2x2 cell
        takes input channel (oc*ft*fs*fs + (ft-1)*fs*fs + i*fs + j) // repeats."""
        H, W, C = x.shape
        repeats = out_c * ft * fs * fs // C
        y = np.empty((H * fs, W * fs, out_c), dtype=np.float32)
        oc = np.arange(out_c)
        for i in range(fs):
            for j in range(fs):
                src = (oc * ft * fs * fs + (ft - 1) * fs * fs + i * fs + j) // repeats
                y[i::fs, j::fs] = x[:, :, src]
        return y

    @staticmethod
    def up2(x):
        return x.repeat(2, axis=0).repeat(2, axis=1)

    # --- the decoder --------------------------------------------------------------------------------------------
    def decode(self, latent_chw: np.ndarray, dump=None) -> np.ndarray:
        """latent_chw: the diffusion-space latent [64, h, w] -> float32 [16h, 16w, 4] in about [-1, 1]."""
        mean = np.array(LATENT_MEAN, dtype=np.float32).reshape(64, 1, 1)
        std = np.array(LATENT_STD, dtype=np.float32).reshape(64, 1, 1)
        z = (latent_chw.astype(np.float32) * std) / np.float32(1.0) + mean    # diffusion_to_vae_latents
        x = np.ascontiguousarray(z.transpose(1, 2, 0))
        d = dump if dump is not None else {}
        x = self.conv("post_quant_conv", x)
        x = self.conv("decoder.conv_in", x)
        d["conv_in"] = x
        x = self.resnet("decoder.mid_block.resnets.0", x)
        x = self.attention("decoder.mid_block.attentions.0", x)
        d["mid_attn"] = x
        x = self.resnet("decoder.mid_block.resnets.1", x)
        d["mid"] = x
        for i, (_, _, ft, up) in enumerate(UP_BLOCKS):
            p = f"decoder.up_blocks.{i}"
            x0 = x
            for j in range(3):
                x = self.resnet(f"{p}.resnets.{j}", x)
            if up:
                x = self.conv(f"{p}.upsampler.resample.1", self.up2(x)) + self.dup_up(x0, x.shape[2], ft)
            d[f"up{i}"] = x
        x = self.conv("decoder.conv_out", self.silu(self.rms("decoder.norm_out", x)))
        d["out"] = x
        return x


def to_u8(img: np.ndarray) -> np.ndarray:
    """sd.cpp: v = clamp((x + 1) * 0.5, 0, 1); u8 = v <= 0 ? 0 : v >= 1 ? 255 : (uint8)(v * 255 + 0.5)."""
    v = np.clip((img.astype(np.float32) + np.float32(1)) * np.float32(0.5), 0, 1)
    u = (v * np.float32(255) + np.float32(0.5)).astype(np.uint8)
    u[v <= 0] = 0
    u[v >= 1] = 255
    return u


def read_latent(path) -> np.ndarray:
    shape_file = Path(str(path)[:-4] + ".shape")
    w, h, c = [int(s) for s in shape_file.read_text().split()[:3]]
    return np.fromfile(path, dtype=np.float32).reshape(c, h, w)


def main(argv):
    if argv[:1] == ["--png"]:
        from PIL import Image
        Image.fromarray(to_u8(np.load(argv[1])), "RGBA").save(argv[2])
        return 0
    if len(argv) < 3:
        print(__doc__)
        return 2
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "f32"
    dec = Decoder(load_safetensors(argv[0]), mode=mode, stats="--stats" in argv)
    out = dec.decode(read_latent(argv[1]))
    np.save(argv[2], out)
    print(f"decoded {out.shape} mode {mode}: min {out.min():.4f} max {out.max():.4f}")
    if dec.stats is not None:
        worst = max(dec.stats.items(), key=lambda kv: kv[1] if kv[0].endswith(":in_scaled") else -1)
        print(f"largest conv input after the 2^-7 scale: {worst[1]:.1f} at {worst[0]} (f16 max {F16_MAX})")
        for k, v in dec.stats.items():
            print(f"  {k} {v:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
