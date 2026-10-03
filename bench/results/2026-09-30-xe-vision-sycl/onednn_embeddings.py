# emb.py OUTDIR: the SVE1 embeddings of each encoder variant against the CPU encoder (same image, same cap)
import sys, numpy as np, pathlib
d = pathlib.Path(sys.argv[1])
def load(p):
    b = p.read_bytes(); h = np.frombuffer(b[:20], np.int32)
    assert h[0] == 0x31455653, p
    return tuple(h[1:4]), np.frombuffer(b[20:], np.float32).reshape(h[1], h[4]).astype(np.float64)
for cap, img, i in [(300, "landscape", 1), (300, "portrait", 3), (1024, "landscape-large", 1)]:
    shp, c = load(d / f"cpu-{cap}-{img}-{i}.sve")
    for v in ["nodnn", "dnn", "dnnoff"]:
        s2, x = load(d / f"{v}-{cap}-{img}-{i}.sve")
        rows = np.linalg.norm(x - c, axis=1) / np.maximum(np.linalg.norm(c, axis=1), 1e-30)
        cos = (x * c).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(c, axis=1))
        print(f"{img:16s} cap {cap:4d} {v:7s} vs cpu: shape {'same' if s2 == shp else s2} finite {np.isfinite(x).all()} "
              f"max|d| {np.abs(x - c).max():.4f} relL2 {np.linalg.norm(x - c) / np.linalg.norm(c):.4f} "
              f"row relL2 median {np.median(rows):.4f} max {rows.max():.3f}  cos min {cos.min():.4f}")
    for a, b in [("nodnn", "dnnoff"), ("nodnn", "dnn")]:
        _, x = load(d / f"{a}-{cap}-{img}-{i}.sve"); _, y = load(d / f"{b}-{cap}-{img}-{i}.sve")
        print(f"{img:16s} cap {cap:4d} {a} vs {b}: max|d| {np.abs(x - y).max():.3g} relL2 {np.linalg.norm(x - y) / np.linalg.norm(x):.4g}")
    for v in ["dnn", "nodnn"]:
        _, x = load(d / f"{v}-{cap}-{img}-{i}.sve"); _, y = load(d / f"{v}-{cap}-{img}-{i + 1}.sve")
        print(f"{img:16s} cap {cap:4d} {v} run-to-run max|d| {np.abs(x - y).max():.3g}")
