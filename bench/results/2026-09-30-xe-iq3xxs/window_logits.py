import sys, numpy as np
D = sys.argv[1]
def load(p):
    h = np.fromfile(p, np.int32, 2); return np.fromfile(p, np.float32, offset=8).reshape(h[1], h[0])
for cpu in sys.argv[2:] or ["ggml", "default"]:
    a, b = load(f"{D}/{cpu}-nodraft.bin"), load(f"{D}/{cpu}-oracle.bin")
    same = next((i for i in range(len(a)) if a[i].argmax() != b[i].argmax()), len(a))
    d = np.abs(a[:same] - b[:same]).max(1)
    top2 = np.sort(a[:same], 1)[:, -2:]; margin = top2[:, 1] - top2[:, 0]
    print(f"{cpu}: argmax equal on rows 0..{same - 1}; T=1 vs T=4 max|diff| per row: row0 {d[0]:.3g}, median {np.median(d):.3g}, "
          f"max {d.max():.3g} (row {d.argmax()}); rows bit-identical {int((d == 0).sum())} of {same}; smallest top-2 margin {margin.min():.3g} (row {margin.argmin()})")
    print("   per-row max|diff| (first 12):", " ".join(f"{x:.3g}" for x in d[:12]))
a, b = load(f"{D}/ggml-nodraft.bin"), load(f"{D}/default-nodraft.bin")
n = next((i for i in range(len(a)) if a[i].argmax() != b[i].argmax()), len(a))
print(f"T=1: ggml CPU vs default CPU: argmax equal through row {n - 1}; row0 max|diff| {np.abs(a[0] - b[0]).max():.3g}")
