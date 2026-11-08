# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/esrgan_ref.py - RealESRGAN x4plus in PyTorch, the reference xestrata-image's (src/image/esrgan.cpp) is
checked against: RRDBNet as basicsr defines it (written from its paper and its published structure, the official
weights loaded into it strictly, every key and shape), in FP32 with TF32 off, the whole image at once; and the
comparison of xestrata-image's output with it.

  python tools/image/esrgan_ref.py RealESRGAN_x4plus.pth IMAGE OUT.png [--device cuda|xpu|cpu]
  python tools/image/esrgan_ref.py --compare REF.png ENGINE.rgba WIDTH HEIGHT

A PyTorch with numpy and Pillow runs it (not the server's venv).
"""
from __future__ import annotations

import sys

import numpy as np


def model(pth: str):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf: int, gc: int):
            super().__init__()
            for k in range(5):
                setattr(self, f"conv{k + 1}", nn.Conv2d(nf + k * gc, gc if k < 4 else nf, 3, 1, 1))

        def forward(self, x):
            feats = [x]
            for k in range(4):
                feats.append(F.leaky_relu(getattr(self, f"conv{k + 1}")(torch.cat(feats, 1)), 0.2))
            return self.conv5(torch.cat(feats, 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf: int, gc: int):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, nf: int = 64, nb: int = 23, gc: int = 32):
            super().__init__()
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
            feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
            return self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))

    sd = torch.load(pth, weights_only=True, map_location="cpu")
    sd = sd.get("params_ema", sd.get("params", sd))
    net = RRDBNet()
    net.load_state_dict(sd, strict=True)
    return net.eval()


def run(pth: str, image: str, out: str, device: str = "cpu") -> None:
    import torch
    from PIL import Image
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = model(pth).to(device)
    rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
    with torch.inference_mode():
        y = net(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device))[0].clamp(0, 1)
    arr = np.rint(y.permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(out)


def compare(ref: str, engine: str, width: int, height: int) -> None:
    from PIL import Image
    a = np.asarray(Image.open(ref).convert("RGB"), dtype=np.int32)
    b = np.fromfile(engine, dtype=np.uint8).reshape(height, width, 4)[..., :3].astype(np.int32)
    d = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"max {d.max()} mean {d.mean():.4f} p99.9 {np.percentile(d, 99.9):.1f} over 2: {np.mean(d > 2) * 100:.4f}% "
          f"psnr {psnr:.2f} dB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--compare":
        compare(a[1], a[2], int(a[3]), int(a[4]))
    else:
        dev = a[a.index("--device") + 1] if "--device" in a else "cpu"
        run(a[0], a[1], a[2], dev)
