# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# emb.py DIR: the SVE1 embeddings of the Vulkan and SYCL encoders against the CPU encoder, and run to run
import pathlib
import sys

import numpy as np


def load(p):
    b = p.read_bytes()
    h = np.frombuffer(b[:20], np.int32)
    assert h[0] == 0x31455653, p
    return tuple(h[1:4]), np.frombuffer(b[20:], np.float32).reshape(h[1], h[4]).astype(np.float64)


def rel(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


d = pathlib.Path(sys.argv[1])
for cap, img, i in [(300, "landscape", 1), (300, "portrait", 3), (1024, "landscape-large", 1)]:
    shp, c = load(d / f"cpu-{cap}-{img}-{i}.sve")
    for v in ["sycl", "vulkan"]:
        s2, x = load(d / f"{v}-{cap}-{img}-{i}.sve")
        rows = np.linalg.norm(x - c, axis=1) / np.maximum(np.linalg.norm(c, axis=1), 1e-30)
        cos = (x * c).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(c, axis=1))
        print(f"{img:16s} cap {cap:4d} {v:6s} vs cpu: shape {'same' if s2 == shp else s2} "
              f"finite {np.isfinite(x).all()} max|d| {np.abs(x - c).max():.4f} relL2 {rel(x, c):.4f} "
              f"row relL2 median {np.median(rows):.4f} max {rows.max():.3f}  cos min {cos.min():.4f}")
    _, x = load(d / f"sycl-{cap}-{img}-{i}.sve")
    _, y = load(d / f"vulkan-{cap}-{img}-{i}.sve")
    print(f"{img:16s} cap {cap:4d} vulkan vs sycl: max|d| {np.abs(x - y).max():.4f} relL2 {rel(y, x):.4f}")
    for v in ["cpu", "sycl", "vulkan"]:
        _, x = load(d / f"{v}-{cap}-{img}-{i}.sve")
        _, y = load(d / f"{v}-{cap}-{img}-{i + 1}.sve")
        print(f"{img:16s} cap {cap:4d} {v} run-to-run max|d| {np.abs(x - y).max():.3g}")
