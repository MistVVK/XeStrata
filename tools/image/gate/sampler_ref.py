"""tools/image/gate/sampler_ref.py - TEST-ONLY (OWN-5 gate harness): the sampler contract for Qwen-Image-2.1-Turbo, written
without any fused multiply-add and checked bit for bit against stable-diffusion.cpp 2988060's own final latent.

What sd-cli does for one picture (sample_euler in src/runtime/denoiser.hpp, the denoise callback in
src/pipeline/diffusion_engine.cpp, DiscreteFlowDenoiser): with the schedule `sigmas` (serve/images.py TURBO_SIGMAS, 9
values ending in 0), CFG 1, the flow denoiser's scalings c_skip = 1, c_out = -sigma, c_in = 1, and the noise as x:
    for each step i:  denoised = v_i * c_out + x * c_skip      (two tensor products, then one sum: three float loops)
                      d        = (x - denoised) / sigma_i      (one subtraction loop, one division loop)
                      x        = x + d * (sigma_i+1 - sigma_i) (one product loop, one sum loop)
    the final x is the latent (inverse noise scaling at sigma 0 is x * (1 / (1 - 0)) = x).
Every elementwise operation of sd.cpp's Tensor class is its own loop with one float operation in it (src/core/tensor.hpp),
so nothing can be fused; numpy float32 does the same.  All values are float32, every operation rounds to float32.

    python -I -B sampler_ref.py <stage root> <tag>          e.g. stage st_cuda_stock_A (run_set.py --dump: noise, vel_0..7,
                                                            latent per picture); prints identical / total
    python -I -B sampler_ref.py --self-test"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# serve/images.py TURBO_SIGMAS, as sd-cli reads the --sigmas list (strtof: the nearest float32 of each decimal)
TURBO_SIGMAS = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"


def sigmas_f32(text: str = TURBO_SIGMAS) -> np.ndarray:
    return np.array([float(x) for x in text.split(",")], dtype=np.float32)


def euler_flow(noise: np.ndarray, velocities: list, sigmas: np.ndarray) -> np.ndarray:
    """The final latent from the starting noise and the DiT's output of each step (float32 in, float32 out)."""
    f = np.float32
    assert noise.dtype == np.float32 and len(velocities) == len(sigmas) - 1
    x = noise.copy()
    for i, v in enumerate(velocities):
        sigma, nxt = sigmas[i], sigmas[i + 1]
        if sigma == nxt:
            continue
        c_skip, c_out = f(1.0), f(-sigma)
        denoised = (v * c_out) + (x * c_skip)
        d = (x - denoised) / sigma
        x = x + d * f(nxt - sigma)
    return x


def load(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def check(root: Path, tag: str) -> tuple[int, int]:
    sig = sigmas_f32()
    ok = total = 0
    for d in sorted((root / tag).glob("dump_p*_s*")):
        try:
            noise = load(d / "noise.bin")
            vels = [load(d / f"vel_{k}.bin") for k in range(len(sig) - 1)]
            want = load(d / "latent.bin")
        except OSError:
            continue
        got = euler_flow(noise, vels, sig)
        same = bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
        total += 1
        ok += same
        if not same:
            diff = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
            print(f"{d.name}: {len(diff)} of {got.size} values differ, max abs {float(np.abs(got - want).max()):.3g}")
        else:
            print(f"{d.name}: identical ({got.size} values)")
    print(f"SAMPLER-REF {tag}: identical {ok} of {total}")
    return ok, total


def self_test() -> None:
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(1000).astype(np.float32)
    vels = [rng.standard_normal(1000).astype(np.float32) for _ in range(8)]
    sig = sigmas_f32()
    a = euler_flow(noise, vels, sig)
    assert a.dtype == np.float32 and a.shape == noise.shape
    # one step by hand: denoised = v*-s + x ; d = (x - denoised)/s ; x + d*(0-s)  ->  x - (x - denoised) = denoised (+-1 ulp)
    s = sig[0]
    one = euler_flow(noise, [vels[0]] + [np.zeros(1000, np.float32)] * 7, np.array([sig[0], sig[1]] + [sig[1]] * 7, np.float32))
    assert one.shape == noise.shape
    assert sig[-1] == 0 and len(sig) == 9 and float(s) == 1.0
    print("SELF-TEST-OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        sys.exit(0)
    ok, total = check(Path(sys.argv[1]), sys.argv[2])
    sys.exit(0 if total and ok == total else 1)
