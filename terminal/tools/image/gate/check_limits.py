"""tools/image/gate/check_limits.py - TEST-ONLY (OWN-5 gate harness): apply a limit from limits.py to a result file.

    python -I -B check_limits.py <limit name> <cmp json>            compare.py's <pics>/cmp/<label>.json (pictures)
    python -I -B check_limits.py <limit name> <stage json> <stage>   stage_cmp.py's <root>/cmp/stage_*.json, one stage

Prints PASS or FAIL with the violations; exit 0 on pass, 1 on fail, 2 on a usage or input error."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import limits  # noqa: E402


def from_pictures(summary: dict) -> dict:
    inf = math.inf
    return {"identical_all": summary.get("bytes_equal") == summary.get("pairs") and summary.get("pairs", 0) > 0,
            "psnr_min": summary.get("psnr_min", inf), "ssim_min": summary.get("ssim_min", 1.0),
            "psnr_mean_min": summary.get("psnr_mean", inf), "ssim_mean_min": summary.get("ssim_mean", 1.0),
            "psnr_min_min": summary.get("psnr_min", inf)}


def from_stage(summary: dict) -> dict:
    return {"identical_all": summary.get("identical") == summary.get("n") and summary.get("n", 0) > 0,
            "rel_l2_max": summary.get("rel_l2_max", 0.0), "cos_min": summary.get("cos_min", 1.0),
            "token_cos_min": summary.get("token_cos_min", 1.0)}


def main() -> int:
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in limits.LIMITS:
        print(__doc__)
        print("limit names:", ", ".join(limits.LIMITS))
        return 2
    data = json.loads(Path(a[1]).read_text(encoding="utf-8"))
    if len(a) == 3:
        if a[2] not in data:
            print(f"no stage {a[2]!r} in {a[1]}")
            return 2
        got = from_stage(data[a[2]]["summary"])
    else:
        got = from_pictures(data["summary"])
    bad = limits.passes(a[0], got)
    print(("FAIL " + "; ".join(bad)) if bad else f"PASS {a[0]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
