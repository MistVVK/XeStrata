"""Unit test verifying Tencent Hy4-Flash (.strata v1.0) rank-cut memory fit rule across all 7 hardware classes
(docs/10-GUI-TUI.md §2 & tests/data/hy4_squant_fit_cases.json).
"""
import json
from pathlib import Path


def compute_hy4_fit(model_info: dict, consts: dict, vram_gib: float, ram_gib: float) -> dict:
    g2g = consts["GIB_TO_GB"]
    vram_gb = vram_gib * g2g
    ram_gb = ram_gib * g2g
    gpu_room = max(0.0, vram_gb - model_info["dense_gb"] - consts["VRAM_NON_EXPERT_GB"])
    ram_room = max(0.0, ram_gb - consts["OS_RESERVE_GB"] - consts["HOT_ENGRAM_GB"])
    vram_exp = min(model_info["bodies_gb"], gpu_room)
    ram_exp = min(model_info["bodies_gb"] - vram_exp, ram_room)
    ssd_exp = max(0.0, model_info["bodies_gb"] - vram_exp - ram_exp)
    return {"vram_gb": round(vram_exp, 2), "ram_gb": round(ram_exp, 2), "ssd_gb": round(ssd_exp, 2)}


def test_hy4_squant_fit_cases():
    data_path = Path(__file__).resolve().parents[1] / "data" / "hy4_squant_fit_cases.json"
    spec = json.loads(data_path.read_text(encoding="utf-8"))
    consts = spec["constants"]
    models = spec["models"]
    for c in spec["cases"]:
        res = compute_hy4_fit(models[c["model"]], consts, c["vram_gib"], c["ram_gib"])
        assert abs(res["ssd_gb"] - c["expected_ssd_gb"]) <= 0.05, (
            f"Fit mismatch for {c}: got {res}, expected ssd_gb={c['expected_ssd_gb']}"
        )


if __name__ == "__main__":
    test_hy4_squant_fit_cases()
    print("test_hy4_squant_fit_cases: ALL 7 FIT CASES PASSED!")
