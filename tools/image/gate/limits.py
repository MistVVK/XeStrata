"""tools/image/gate/limits.py - TEST-ONLY (OWN-5 gate harness): the gate's limits as constants, each with the sd.cpp
floor it was set from (measured 2026-11-10, Spark GB10 CUDA / aarch64 CPU, Qwen-Image-2.1-Turbo Q8_0, 1024x1024,
8 pictures per arm unless noted).  A limit is about 1.2x the worst floor, rounded; the coordinator accepted them on
11-10.  Rule 3's end-to-end picture limits (E2E_* below) still decide; these are the stage checks that keep a
real bug from hiding in the 8-step amplification.

    python -I -B limits.py            prints the table (limit, floor, margin)

Each entry: name -> {"limit": {...}, "floor": {...}, "why": text}.  A "min" key is a lower bound, a "max" key an upper
bound.  check_limits.py applies them to a compare.py / stage_cmp.py result."""
from __future__ import annotations

LIMITS = {
    # --- exact: where we control the arithmetic -------------------------------------------------------------------
    "run_to_run": {
        "limit": {"identical_all": True},
        "floor": {"identical_all": True},
        "why": "same build, card and OS twice: noise, cond, velocity, latent and picture bytes identical (sd.cpp: 16/16 "
               "on CUDA, HIP, Vulkan; 4/4 CPU at any thread count)",
    },
    # --- VAE decode of a fixed latent -------------------------------------------------------------------------------
    "vae_same_tiling": {
        "limit": {"psnr_min": 45.0, "ssim_min": 0.998},
        "floor": {"psnr_min": 56.58, "ssim_min": 0.9988},
        "why": "CPU (aarch64) vs CUDA, same latent, 5 pictures; untiled vs untiled, or tiled vs tiled with the same tile "
               "size and overlap",
    },
    "vae_across_tiling": {
        "limit": {"psnr_min": 25.0},
        "floor": {"psnr_min": 27.7, "psnr_mean": 35.7, "ssim_min": 0.958},
        "why": "256-px tiles at 0.25 overlap vs untiled, CUDA: a different algorithm, a sanity floor only",
    },
    # --- one DiT step ------------------------------------------------------------------------------------------------
    "velocity": {
        "limit": {"rel_l2_max": 1.2e-2, "cos_min": 0.9999},
        "floor": {"rel_l2_max": 0.0090, "cos_min": 0.99996},
        "why": "step 0, fixed cond and noise: native vs stock 0.0079, our attention vs ggml's 0.0071, CPU vs CUDA 0.0090",
    },
    # --- text encoder hidden states (cond) ---------------------------------------------------------------------------
    "text_encoder_same_backend": {
        "limit": {"rel_l2_max": 0.07, "cos_min": 0.997, "token_cos_min": 0.985},
        "floor": {"rel_l2_max": 0.0603, "cos_min": 0.9982, "token_cos_min": 0.9909},
        "why": "CUDA, cuBLAS-forced GEMM vs stock (same card): the GEMM path alone moves cond by 5-6% (rel L2). To be "
               "tightened when our own kernel's number exists; token ids must be identical",
    },
    "text_encoder_cross_backend": {
        "limit": {"rel_l2_max": 0.11, "cos_min": 0.995, "token_cos_min": 0.96},
        "floor": {"rel_l2_max": 0.0934, "cos_min": 0.9956, "token_cos_min": 0.9668},
        "why": "CPU (aarch64) vs CUDA, 6 pictures",
    },
    # --- end to end (rule 3 and rule 5 of the survey; the pictures, not a stage) -----------------------------------
    "e2e_same_card": {
        "limit": {"psnr_mean_min": 34.0, "ssim_mean_min": 0.97, "psnr_min_min": 18.0},
        "floor": {"psnr_mean_min": 34.2, "ssim_mean_min": 0.974, "psnr_min_min": 18.6},
        "why": "our engine vs sd.cpp on the same card and noise, 16 pictures at 1024 (sd.cpp stock vs native: CUDA 34.3 / "
               "0.982 / 21.3, HIP 34.2 / 0.974 / 18.6; attention swap alone 35.7 / 0.983 / 20.8)",
    },
    "e2e_gpu_vs_gpu": {
        "limit": {"psnr_mean_min": 25.0, "ssim_mean_min": 0.91},
        "floor": {"psnr_mean_min": 25.0, "ssim_mean_min": 0.919},
        "why": "no worse than sd.cpp's own HIP vs CUDA pair with the same noise (16 pictures at 1024)",
    },
    "e2e_cpu_vs_gpu_512": {
        "limit": {"psnr_mean_min": 25.0, "ssim_mean_min": 0.88},
        "floor": {"psnr_mean_min": 25.3, "ssim_mean_min": 0.884},
        "why": "no worse than sd.cpp's CPU vs HIP pair (4 pictures at 512)",
    },
}


def _bound(key: str):
    """-> ("min"|"max", metric) from a limit key like psnr_min or rel_l2_max."""
    if key.endswith("_max"):
        return "max", key[:-4]
    if key.endswith("_min"):
        return "min", key[:-4]
    raise KeyError(key)


def passes(name: str, got: dict) -> list[str]:
    """The limit violations of `got` (metric -> value) against LIMITS[name]; empty means it holds.  Keys of the form
    psnr_min mean 'psnr_min >= limit' (the worst picture's PSNR); rel_l2_max means 'rel_l2_max <= limit'."""
    bad = []
    for key, limit in LIMITS[name]["limit"].items():
        if isinstance(limit, bool):
            if got.get(key) is not limit:
                bad.append(f"{key}: {got.get(key)} != {limit}")
            continue
        v = got.get(key)
        if v is None:
            bad.append(f"{key}: missing")
        elif key.startswith(("rel_l2",)) and v > limit:
            bad.append(f"{key}: {v:.5g} > {limit}")
        elif not key.startswith(("rel_l2",)) and v < limit:
            bad.append(f"{key}: {v:.5g} < {limit}")
    return bad


def floor_margin(name: str) -> dict:
    """metric -> (limit, floor): the floor must lie on the passing side of the limit."""
    return {k: (LIMITS[name]["limit"][k], LIMITS[name]["floor"].get(k)) for k in LIMITS[name]["limit"]}


if __name__ == "__main__":
    for n, e in LIMITS.items():
        print(f"{n}:")
        for k, (lim, fl) in floor_margin(n).items():
            print(f"    {k}: limit {lim}   floor {fl}")
        print(f"    ({e['why']})")
