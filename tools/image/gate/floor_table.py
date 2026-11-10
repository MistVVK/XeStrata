"""tools/image/gate/floor_table.py - TEST-ONLY (OWN-5 gate harness): the floor table (sd.cpp against itself) from kept
picture sets, by running compare.py for each pair and printing one markdown row per pair.

    python -I -B floor_table.py <pics root> [--check]

<pics root> holds the sets run_set.py wrote (one folder per tag).  --check compares the result with FLOOR below (the
survey's numbers, 2026-11-10) and exits 1 if any pair differs by more than rounding."""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# (label, set A, set B, the survey's numbers: byte-identical, max abs, mean abs, PSNR min, PSNR mean, SSIM min, SSIM mean)
FLOOR = [
    ("same build twice, stock, GB10 CUDA Q8_0 1024", "cuda_q8_stock_A", "cuda_q8_stock_B", (16, 0, 0, math.inf, math.inf, 1, 1)),
    ("same build twice, native, GB10 CUDA", "cuda_q8_native_A", "cuda_q8_native_B", (16, 0, 0, math.inf, math.inf, 1, 1)),
    ("same build twice, stock, R9700 HIP", "hip_q8_stock_A", "hip_q8_stock_B", (16, 0, 0, math.inf, math.inf, 1, 1)),
    ("same build twice, native, R9700 HIP", "hip_q8_native_A", "hip_q8_native_B", (16, 0, 0, math.inf, math.inf, 1, 1)),
    ("same build twice, A750 Vulkan Q4_K_M", "vk_q4_stock_A", "vk_q4_stock_B", (16, 0, 0, math.inf, math.inf, 1, 1)),
    ("same build twice, CPU 512", "cpu_q8_512_A", "cpu_q8_512_B", (4, 0, 0, math.inf, math.inf, 1, 1)),
    ("CPU 4 vs 8 threads", "cpu_q8_512_A", "cpu_q8_512_t4", (2, 0, 0, math.inf, math.inf, 1, 1)),
    ("attention kernel only (native, ours vs ggml)", "cuda_q8_native_A", "cuda_q8_native_ggmlattn", (0, 251, 1.57, 20.8, 35.7, 0.914, 0.983)),
    ("stock vs native, GB10", "cuda_q8_stock_A", "cuda_q8_native_A", (0, 250, 1.63, 21.3, 34.3, 0.923, 0.982)),
    ("stock vs native, R9700", "hip_q8_stock_A", "hip_q8_native_A", (0, 255, 2.45, 18.6, 34.2, 0.900, 0.974)),
    ("noise only: x86 vs aarch64, 512", "cuda_q8_512_A", "cuda_q8_512_noise_x86", (0, 183, 0.80, 36.6, 41.0, 0.991, 0.994)),
    ("noise only: Linux vs Windows", "cuda_q8_noise_x86", "cuda_q8_noise_msvc", (0, 248, 1.06, 25.4, 37.7, 0.961, 0.990)),
    ("HIP vs CUDA, same noise", "hip_q8_stock_A", "cuda_q8_noise_x86", (0, 255, 7.10, 17.1, 25.0, 0.825, 0.919)),
    ("HIP native vs CUDA, same noise", "hip_q8_native_A", "cuda_q8_noise_x86", (0, 255, 6.47, 18.9, 25.4, 0.826, 0.923)),
    ("HIP vs CUDA, same noise, 512", "hip_q8_512_A", "cuda_q8_512_noise_x86", (0, 202, 3.00, 23.8, 32.1, 0.901, 0.965)),
    ("CPU vs HIP, 512", "cpu_q8_512_A", "hip_q8_512_A", (0, 226, 9.09, 15.2, 25.3, 0.785, 0.884)),
    ("CPU vs CUDA, same noise, 512", "cpu_q8_512_A", "cuda_q8_512_noise_x86", (0, 225, 7.83, 15.3, 25.4, 0.789, 0.904)),
    ("CUDA vs Vulkan, Q4_K_M", "cuda_q4_stock_A", "vk_q4_stock_A", (0, 255, 7.68, 12.1, 27.3, 0.792, 0.925)),
]


def close(a, b, tol):
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol


def main():
    root = Path(sys.argv[1])
    bad = 0
    print("| comparison | identical | max abs | mean abs | PSNR min / mean dB | SSIM min / mean |")
    print("|---|---|---|---|---|---|")
    for label, a, b, want in FLOOR:
        if not (root / a).is_dir() or not (root / b).is_dir():
            print(f"| {label} | not here ({a} / {b}) | | | | |")
            continue
        r = subprocess.run([sys.executable, "-I", "-B", str(HERE / "compare.py"), str(root), a, b],
                           capture_output=True, text=True, check=True)
        s = json.loads(r.stdout.strip().splitlines()[-1])
        got = (s["bytes_equal"], s.get("max_abs", 0), s.get("mean_abs", 0), s.get("psnr_min", math.inf),
               s.get("psnr_mean", math.inf), s.get("ssim_min", 1), s.get("ssim_mean", 1))
        print(f"| {label} | {got[0]}/{s['pairs']} | {got[1]:.0f} | {got[2]:.2f} | {got[3]:.1f} / {got[4]:.1f} | "
              f"{got[5]:.3f} / {got[6]:.3f} |")
        tols = (0, 0.5, 0.006, 0.06, 0.06, 0.0006, 0.0006)
        if "--check" in sys.argv and not all(close(g, w, t) for g, w, t in zip(got, want, tols)):
            print(f"  DIFFERS from the survey: {want}")
            bad += 1
    if "--check" in sys.argv:
        print("FLOOR-OK" if not bad else f"FLOOR-DIFFERS in {bad} pairs")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
