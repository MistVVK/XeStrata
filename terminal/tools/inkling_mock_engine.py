"""A stand-in for `strata-inkling --serve` speaking Strata's native line protocol (including 1..64 concurrent batch slots
`BGEN`/`BADM`/`BT`/`BDONE`/`BSTOP` and single-stream `GEN`/`T`/`DONE`/`STOP`, plus `--info`, `--version`, and in-process Vision).

Used by unit/E2E tests and the QA suite to exercise `serve/server.py`, `inkling_dialect.py`, `inkling_kind.py`,
`reasoning_effort` levels, XML/JSON tool loops, vision attachments, and 4/8/32/64 concurrent SSE requests!
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

KNOWN = {
    "--version": 0, "--info": 1, "--serve": 0, "--native": 1, "--model": 1, "--vision": 1,
    "--max-context": 1, "--parallel": 1, "--batch": 1, "--slots": 1, "--kv-dtype": 1,
    "--chunk": 1, "--gpu": 1, "--draft": 1, "--vram-experts-gb": 1, "--ram-experts-gb": 1,
    "--threads": 1, "--script": 1, "--record": 1, "--delay": 1,
}
SAMPLING = {"temperature", "top_p", "top_k", "min_p", "penalty_repeat", "penalty_freq", "penalty_present", "penalty_last_n", "seed"}
IGNORED = {"pcie_frac", "spec_min_p", "prefill_pipe", "ckpt", "hold"}
IMAGE_PAD_ID = int(os.environ.get("INKLING_MOCK_PAD_ID") or 200054)
IMAGE_TOKENS = 256
EOS_ID = 200006

_OUT_LOCK = threading.Lock()


def say(line: str) -> None:
    with _OUT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def parse_args(argv):
    opts, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a not in KNOWN:
            say(f"ERR unknown flag {a}")
            sys.exit(2)
        if KNOWN[a]:
            if i + 1 >= len(argv):
                say(f"ERR {a} needs a value")
                sys.exit(2)
            opts[a] = argv[i + 1]
            i += 1
        else:
            opts[a] = True
        i += 1
    return opts


def caps_of(env) -> dict:
    return dict(p.split("=", 1) for p in str(env.get("INKLING_MOCK_CAPS") or "drafter=dspark images=1 parallel=64").split() if "=" in p)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--version"]:
        say("strata-inkling 0.1.65")
        return 0
    if args[:1] == ["--info"]:
        caps = caps_of(os.environ)
        say(f"arch=inkling2_loop layers=24 visits=48 hidden=2304 vocab=200058 experts=128 eos={EOS_ID} "
            f"drafter={caps.get('drafter', 'dspark')} images={caps.get('images', '1')} image_pad={IMAGE_PAD_ID} "
            f"parallel={caps.get('parallel', '64')}")
        return 0
    opts = parse_args(args)
    if "--serve" not in opts:
        say("ERR this mock only serves")
        return 2
    ctx = int(opts.get("--max-context", 32768))
    par = int(opts.get("--parallel") or opts.get("--batch") or opts.get("--slots") or 1)
    delay = float(opts.get("--delay", 0))
    scripts = []
    if "--script" in opts:
        data = json.loads(open(opts["--script"], encoding="utf-8").read())
        scripts = data if data and isinstance(data[0], list) else [data]
    record = open(opts["--record"], "a", encoding="utf-8") if "--record" in opts else None
    caps = caps_of(os.environ)
    batch_info = f" batch_slots={par} slot_read=1" if par > 1 else ""
    say(f"INFO engine=mock-0 arch=inkling2_loop backend=mock build=SQ-20 layers=24 visits=48 "
        f"images={1 if caps.get('images') == '1' else 0} "
        f"drafter={'dspark' if opts.get('--draft', 'dspark') == 'dspark' else 'off'}{batch_info}")
    say(f"READY {ctx} stop" + (" hold holdb" if par > 1 else ""))

    req_counter = [0]
    req_lock = threading.Lock()
    slot_stops: dict[int, threading.Event] = {}
    single_stop = threading.Event()

    def next_script():
        with req_lock:
            idx = req_counter[0]
            req_counter[0] += 1
        if not scripts:
            return [EOS_ID]
        return list(scripts[min(idx, len(scripts) - 1)])

    def run_bgen(slot: int, max_new: int, ids: list[int], images: list[str], stop_ev: threading.Event):
        seq = next_script()
        ptok = len(ids) + sum(IMAGE_TOKENS - 1 for _ in images)
        gen = 0
        reason = "stop"
        for tid in seq:
            if stop_ev.is_set():
                reason = "stop"
                break
            if gen >= max_new:
                reason = "length"
                break
            if delay > 0:
                time.sleep(delay)
            say(f"BT {slot} {tid}")
            gen += 1
            if tid == EOS_ID:
                break
        else:
            if gen >= max_new:
                reason = "length"
        say(f"BDONE {slot} {gen} {reason} 1.0 read=1.0")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("QUIT"):
            return 0
        if line.startswith("STOP"):
            single_stop.set()
            continue
        if line.startswith("BSTOP "):
            try:
                sl = int(line.split()[1])
                if sl in slot_stops:
                    slot_stops[sl].set()
            except (ValueError, IndexError):
                pass
            continue
        if line.startswith("BGEN "):
            if record:
                record.write(raw)
                record.flush()
            parts = line.split()
            try:
                slot = int(parts[1])
                max_new = int(parts[2])
                ids = [int(x) for x in parts[-1].split(",")]
                pairs = [p.split("=", 1) for p in parts[3:-1] if "=" in p]
                images = [v.replace("%20", " ").replace("%25", "%") for k, v in pairs if k == "image"]
            except (ValueError, IndexError):
                say("ERR bad BGEN line")
                continue
            ev = threading.Event()
            slot_stops[slot] = ev
            ptok = len(ids) + sum(IMAGE_TOKENS - 1 for _ in images)
            say(f"DONE 0 {ptok} 0.0 0.0 admit")
            say(f"BADM {slot} 1")
            threading.Thread(target=run_bgen, args=(slot, max_new, ids, images, ev), daemon=True).start()
            continue
        if line.startswith("GEN "):
            if record:
                record.write(raw)
                record.flush()
            single_stop.clear()
            parts = line.split()
            try:
                max_new = int(parts[1])
                ids = [int(x) for x in parts[-1].split(",")]
                pairs = [p.split("=", 1) for p in parts[2:-1] if "=" in p]
                keys = dict(pairs)
                images = [v.replace("%20", " ").replace("%25", "%") for k, v in pairs if k == "image"]
            except (ValueError, IndexError):
                say("ERR bad GEN line")
                continue
            bad = [k for k in keys if k not in SAMPLING and k not in IGNORED and not (k == "image" and caps.get("images") == "1")]
            if bad:
                say(f"ERR unsupported key {bad[0]}")
                continue
            pads = ids.count(IMAGE_PAD_ID)
            if images and pads != len(images):
                say(f"ERR image: {len(images)} files for {pads} image placeholders")
                continue
            ptok = len(ids) + sum(IMAGE_TOKENS - 1 for _ in images)
            say(f"PP {ptok} 0 1.0")
            seq = next_script()
            gen = 0
            reason = "stop"
            for tid in seq:
                if single_stop.is_set():
                    reason = "stop"
                    break
                if gen >= max_new:
                    reason = "length"
                    break
                if delay > 0:
                    time.sleep(delay)
                say(f"T {tid}")
                gen += 1
                if tid == EOS_ID:
                    break
            else:
                if gen >= max_new:
                    reason = "length"
            say(f"DONE {gen} {ptok} 1.0 1.0 {reason}")
            continue
        say(f"ERR unknown command {line.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
