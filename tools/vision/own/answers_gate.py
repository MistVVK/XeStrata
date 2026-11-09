#!/usr/bin/env python3
"""The answers gate of the own vision tower: greedy picture answers of the whole Strata server, one run per picture reader, compared.

    python3 answers_gate.py --box <box.json> --helper <strata-vision> --out <dir> [--towers mtmd,own] [--max-tokens 1024] [--threads N]
                            [--gpu] [--n-tokens 48] [--question "..."] [--port N] <image> [<image> ...]

For each tower a fresh `serve/server.py` is started (engine + the picture reader `<helper> ... --tower <tower>`, through a one-line wrapper script made
in <dir>), every picture is sent with each question at temperature 0 (max --n-tokens tokens), and the server is stopped (its own process group only).
Then the answers of the towers are compared picture by picture: how many are identical, and the differing ones are printed side by side (not
argued away).  The box file is the one tools of the 145-vis runs use (src, exe, pack, native, ple, profile, mtp, tokenizer, mmproj, lib_dirs, gpu, port).
Exit 0 when every answer is identical, 1 otherwise.  Standard library plus the repo's tools/strata_bench.py.
"""
import argparse
import base64
import json
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

QUESTIONS = ["Describe this picture in two short sentences.", "What text is written in the picture? Answer with the text only."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--helper", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towers", default="mtmd,own")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="the picture reader on the GPU (default: the CPU reader)")
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--question", action="append", default=[])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    box = json.load(open(a.box))
    out = Path(a.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    src = Path(box["src"])
    sys.path.insert(0, str(src / "tools"))
    sys.path.insert(0, str(src))
    import strata_bench as SB  # noqa: E402

    port = a.port or int(box.get("port", 8195))
    base = "http://127.0.0.1:%d" % port
    questions = a.question or QUESTIONS
    ctx = 65536

    def get(path, timeout=20):
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return json.loads(r.read())

    results = {}
    for spec in a.towers.split(","):
        # a reader spec: name[:gpu][@tag]  (mtmd, own, mtmd:gpu, own:gpu, mtmd@2 for a second run of the same reader)
        name, _, mode = spec.split("@")[0].partition(":")
        gpu = a.gpu or mode == "gpu"
        tower = spec.replace(":", "-").replace("@", "-")        # the label of this run in file names and in the report
        wrap = out / ("reader-%s.sh" % tower)
        wrap.write_text('#!/bin/sh\nexec "%s" "$@" --tower %s\n' % (a.helper, name))
        wrap.chmod(wrap.stat().st_mode | stat.S_IXUSR)
        vis = {"exe": str(wrap), "mmproj": box["mmproj"], "model": box["native"], "gpu": gpu, "max_tokens": a.max_tokens}
        if a.threads and not gpu:
            vis["threads"] = a.threads
        cfg = {"exe": box["exe"], "cwd": str(src), "tokenizer": box["tokenizer"], "model_name": "qwen3.8-flash-next-iq3",
               "log": str(out / "logs" / ("engine-%s.log" % tower)), "lib_dirs": box.get("lib_dirs", []), "port": port, "gpu": box.get("gpu", 0),
               "gpus_asked": True, "vision": vis,
               "args": ["--pack", box["pack"], "--native", box["native"], "--ple-gguf", box["ple"], "--expert-profile", box["profile"],
                        "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", box["mtp"],
                        "--max-context", str(ctx), "--kv", "int8", "--vision", "--vram-reserve-mib", "700"]}
        cfgp = out / "logs" / ("cfg-%s.json" % tower)
        cfgp.write_text(json.dumps(cfg, indent=1))
        log = open(out / "logs" / ("server-%s.log" % tower), "w")
        p = subprocess.Popen([sys.executable, str(src / "serve" / "server.py"), "--engine", "strata", "--config", str(cfgp), "--port", str(port)],
                             stdout=log, stderr=subprocess.STDOUT, cwd=str(src), start_new_session=True)
        answers = {}
        try:
            t0, ok = time.time(), False
            while time.time() - t0 < 1500:
                if p.poll() is not None:
                    raise SystemExit("%s: the server exited (rc %s) while loading, see %s" % (tower, p.returncode, log.name))
                try:
                    if get("/v1/status", 5).get("loaded"):
                        ok = True
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(3)
            if not ok:
                raise SystemExit("%s: the server did not load in 1500 s" % tower)
            print("%s: server up after %.0f s" % (tower, time.time() - t0), flush=True)
            time.sleep(3)
            for img in a.images:
                b64 = base64.b64encode(Path(img).read_bytes()).decode()
                mime = "image/jpeg" if img.lower().endswith((".jpg", ".jpeg")) else "image/png"
                for qi, q in enumerate(questions):
                    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
                                                          {"type": "text", "text": q}]}]
                    t1 = time.time()
                    r = SB.chat(base, "", msgs, a.n_tokens)
                    key = "%s|q%d" % (os.path.basename(img), qi)
                    answers[key] = {"text": r.get("text") or "", "err": r.get("err"), "s": round(time.time() - t1, 1)}
                    print("%s %-28s q%d (%.1f s): %r%s" % (tower, os.path.basename(img), qi, time.time() - t1, answers[key]["text"][:90],
                                                          "  ERR " + str(r.get("err")) if r.get("err") else ""), flush=True)
        finally:
            try:
                os.killpg(p.pid, signal.SIGTERM)       # our own process group only
            except ProcessLookupError:
                pass
            for _ in range(60):
                if p.poll() is not None:
                    break
                time.sleep(1)
        results[tower] = answers
        (out / "answers.json").write_text(json.dumps(results, indent=1))

    towers = list(results)
    keys = list(results[towers[0]])
    ok = lambda t, k: results[t].get(k) and not results[t][k]["err"]  # noqa: E731
    print("ANSWERS, pair by pair (identical / compared):")
    differing = []
    for i, ta in enumerate(towers):
        for tb in towers[i + 1:]:
            same = sum(1 for k in keys if ok(ta, k) and ok(tb, k) and results[ta][k]["text"] == results[tb][k]["text"])
            print("  %-14s vs %-14s %d / %d" % (ta, tb, same, len(keys)))
    for k in keys:
        texts = {t: (results[t][k]["text"] if ok(t, k) else "ERR " + str(results[t].get(k, {}).get("err"))) for t in towers}
        if len(set(texts.values())) > 1:
            differing.append(k)
    print("%d of %d answers differ between some readers; all of them:" % (len(differing), len(keys)))
    for k in differing:
        print("DIFFERS %s" % k)
        for t in towers:
            print("  %-14s: %r" % (t, results[t][k]["text"] if ok(t, k) else "ERR " + str(results[t].get(k, {}).get("err"))))
    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
