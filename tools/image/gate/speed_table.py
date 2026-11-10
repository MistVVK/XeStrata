"""OWN-5: item 4 table from pics/speed_*/results.jsonl (one clean run per setting).   python -I -B speed_table.py <pics>"""
import json
import sys
from pathlib import Path

rows = []
for d in sorted(Path(sys.argv[1]).glob("speed_*")):
    for ln in (d / "results.jsonl").read_text().splitlines():
        r = json.loads(ln)
        log = (d / f"p{r['prompt']}_s{r['seed']}.log").read_text(errors="replace")
        import re
        te = re.findall(r"get_learned_condition completed, taking ([0-9.]+) ?s", log)
        r["text_encode_s"] = float(te[-1]) if te else None
        rows.append(r)
print("| backend | mode | quant | size | s/picture | first step at s | TE s | sampling s | VAE s | max RSS MiB | GPU MiB peak (start) | MemAvailable drop MiB | rc |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    gpu = f"{r['gpu_mib_peak']} ({r['gpu_mib_start']})" if r.get("gpu_mib_peak") is not None else "n/a"
    drop = (r["memavail_mib_start"] - r["memavail_mib_min"]) if r.get("memavail_mib_start") and r.get("memavail_mib_min") else None
    print(f"| {r['backend']} | {'native' if r['native'] else 'stock'} | {r['quant']} | {r['size']} | {r['seconds']} | "
          f"{r.get('first_step_s')} | {r.get('text_encode_s')} | {r.get('sampling_s')} | {r.get('vae_decode_s')} | "
          f"{r['maxrss_mib']} | {gpu} | {drop} | {r['rc']} |")
