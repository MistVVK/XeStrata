"""A stand-in for `strata-hy4 --serve` that speaks the engine line protocol (phaseA-tests/hy4/HY4-INTEGRATION-DESIGN.md section 1) and answers with canned token ids.

For tests of serve, Control and the app against this engine's protocol before (and beside) the real runner: it checks the request lines the way the runner must (a
bad flag or key is an ERR), records every GEN line it got in --record FILE, and replays the token ids of --script FILE (a JSON list of ids, or a list of such lists, one per
request in turn, the last one repeating).  --delay S waits between tokens (cancel tests), --max-context N is the READY value.

    hy4_mock_engine.py --serve --native M.strata --engram E.strata --max-context 8192 --script ids.json --record gens.txt
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

KNOWN = {"--serve": 0, "--native": 1, "--model": 1, "--engram": 1, "--vision": 1, "--max-context": 1, "--chunk": 1, "--gpu": 1, "--draft": 1,
         "--vram-experts-gb": 1, "--session-reuse": 1, "--ram-experts-gb": 1, "--engram-cache-mib": 1, "--no-engram": 0, "--kv": 1,
         "--script": 1, "--record": 1, "--delay": 1}
SAMPLING = {"temperature", "top_p", "top_k", "min_p", "penalty_repeat", "penalty_freq", "penalty_present", "penalty_last_n", "seed"}
IGNORED = {"pcie_frac", "spec_min_p", "prefill_pipe"}
END_IDS = (120025, 120008)


def say(line: str) -> None:
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


IMAGE_PAD_ID = int(os.environ.get("HY4_MOCK_PAD_ID") or 120120)   # <hy_image▁pad>: one per picture in the prompt; the engine expands it
IMAGE_TOKENS = 256


def caps_of(env) -> dict:
    """What this mock says it can do: HY4_MOCK_CAPS="drafter=dfly2 images=1" (default: neither, like the first runner)."""
    return dict(p.split("=", 1) for p in str(env.get("HY4_MOCK_CAPS") or "").split() if "=" in p)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--info"]:                                  # `exe --info FILE`: the facts serve reads before a start
        caps = caps_of(os.environ)
        say(f"arch=hy4flash layers=42 hidden=3072 vocab=120832 experts=512 eos=120025 drafter={caps.get('drafter', 'none')} images={caps.get('images', '0')}")
        return 0
    opts = parse_args(args)
    if "--serve" not in opts:
        say("ERR this mock only serves")
        return 2
    ctx = int(opts.get("--max-context", 8192))
    delay = float(opts.get("--delay", 0))
    scripts = []
    if "--script" in opts:
        data = json.loads(open(opts["--script"], encoding="utf-8").read())
        scripts = data if data and isinstance(data[0], list) else [data]
    record = open(opts["--record"], "a", encoding="utf-8") if "--record" in opts else None
    caps = caps_of(os.environ)
    say(f"INFO engine=mock-0 arch=hy4flash backend=mock build=SQ-64 layers=42 images={1 if caps.get('images') == '1' else 0} "
        f"drafter={'dfly2' if '--draft' in opts and opts['--draft'] == 'dfly2' else 'off'}")
    say(f"READY {ctx} stop")
    stop = threading.Event()
    lines: list[str] = []
    cv = threading.Condition()
    eof = []

    def reader():
        for line in sys.stdin:
            if line.startswith("STOP"):
                stop.set()
                continue
            with cv:
                lines.append(line)
                cv.notify()
        with cv:
            eof.append(True)
            cv.notify()
    threading.Thread(target=reader, daemon=True).start()
    n_req = 0
    while True:
        with cv:
            while not lines and not eof:
                cv.wait()
            if not lines:
                return 0
            line = lines.pop(0)
        if line.startswith("QUIT"):
            return 0
        if not line.startswith("GEN "):
            say(f"ERR expected: GEN <max_new> [key=value ...] <ids>, got {line.split(' ', 1)[0].strip()}")
            continue
        stop.clear()
        if record:
            record.write(line)
            record.flush()
        parts = line.split()
        try:
            max_new = int(parts[1])
            ids = [int(x) for x in parts[-1].split(",")]
            pairs = [p.split("=", 1) for p in parts[2:-1]]
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
        sizes = []
        for i, p in enumerate(images):
            try:
                sizes.append(os.path.getsize(p))
            except OSError:
                say(f"ERR image: cannot read {p}")
                break
        if len(sizes) != len(images):
            continue
        ptok = len(ids) + sum(IMAGE_TOKENS - 1 for _ in images)           # the engine expands each placeholder to IMAGE_TOKENS tokens
        for i in range(len(images)):
            say(f"IMG {i} {IMAGE_TOKENS}")
        if ptok >= ctx:
            say("ERR prompt exceeds context")
            continue
        script = scripts[min(n_req, len(scripts) - 1)] if scripts else [ord("o"), ord("k"), END_IDS[0]]
        n_req += 1
        t0 = time.monotonic()
        say(f"PP {ptok} {ptok} {(time.monotonic() - t0) * 1000:.0f} 1000.0")
        prompt_ms = (time.monotonic() - t0) * 1000
        t1 = time.monotonic()
        made, finish = 0, "length"
        for tok in script:
            if stop.is_set():
                finish = "cancel"
                break
            if made >= max_new:
                break
            say(f"T {tok}")
            made += 1
            if tok in END_IDS:
                finish = "stop"
                break
            if delay:
                time.sleep(delay)
        else:
            finish = "stop" if made < max_new else "length"
        if stop.is_set() and finish != "stop":
            finish = "cancel"
        say(f"DONE {made} {ptok} {prompt_ms:.1f} {(time.monotonic() - t1) * 1000:.1f} {finish}")


if __name__ == "__main__":
    sys.exit(main())
