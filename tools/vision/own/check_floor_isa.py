#!/usr/bin/env python3
"""Which code of a linked strata-vision (or strata-vision-own) holds instructions an older CPU lacks (RESULTS-OWN-3V.md PART H).

The CPU tower compiles each kernel tier in a file of its own with that tier's flags and calls it only after a CPU check (cpu_caps.h); everything else
must stay at the floor of the build.  An instruction-set flag on a file that uses the standard library also reaches its inline and template copies
(weak / COMDAT symbols), and the linker may keep such a copy for the whole program, so the check looks at the LINKED binary, function by function:

  floor none  (SSE4.2, Nehalem / Westmere): no VEX / EVEX instruction (no AVX of any kind), no BMI1/2, no MOVBE;
  floor avx   (Sandy Bridge): AVX allowed; no AVX2 (256-bit integer ops, broadcasts / permutes / gathers of AVX2), no FMA, no F16C, no EVEX, no BMI;
  floor avx_f16c (Ivy Bridge): as avx, F16C allowed.

A function with such an instruction passes only if it is a LOCAL symbol of a kernel tier (the anonymous namespace `kn` of cpu_kernels_impl.h) or a tier's
table getter (strata_vit::cpu_ops_<tier>), i.e. code reached only behind the CPU check.  Anything else is printed, weak / COMDAT copies marked.

    python3 check_floor_isa.py <binary> --floor none|avx|avx_f16c        exit 0 clean, 1 a violation, 77 no objdump / nm
Linux (GNU binutils) only.
"""
import argparse
import re
import shutil
import subprocess
import sys

FMA = re.compile(r"^vf(n?m(add|sub)|maddsub|msubadd)")
F16C = re.compile(r"^vcvt(ph2ps|ps2ph)")
AVX2_ONLY = re.compile(r"^(vpbroadcast|vbroadcasti128|vperm2i128|vpermd|vpermq|vpermps|vpermpd|v(p)?gather|vinserti128|vextracti128|vpmaskmov|vps[lr]lv|vpsrav|vpblendd)")
BMI = re.compile(r"^(andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx|shrx|movbe)\b")
AVX1_YMM_INT_OK = ("vptest", "vpermilps", "vpermilpd", "vperm2f128")      # AVX (1) instructions whose names start with "vp" and take ymm


def bad_reason(mn, ops, floor):
    if "zmm" in ops or re.search(r"%k[0-7]|\{%k", ops) or re.search(r"[xy]mm(1[6-9]|2\d|3[01])\b", ops):
        return "AVX-512"
    if BMI.match(mn):
        return "BMI/MOVBE"
    if floor == "none":
        if mn.startswith("v") and mn not in ("verr", "verw"):
            return "AVX"
        return None
    if FMA.match(mn):
        return "FMA"
    if F16C.match(mn) and floor != "avx_f16c":
        return "F16C"
    if AVX2_ONLY.match(mn):
        return "AVX2"
    if mn.startswith("vp") and "ymm" in ops and not mn.startswith(AVX1_YMM_INT_OK):
        return "AVX2"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--floor", required=True, choices=["none", "avx", "avx_f16c"])
    ap.add_argument("--show", type=int, default=3, help="instructions shown per function")
    a = ap.parse_args()
    if not shutil.which("objdump") or not shutil.which("nm"):
        print("objdump / nm not found")
        return 77
    kind = {}
    for line in subprocess.run(["nm", "--defined-only", a.binary], capture_output=True, text=True).stdout.splitlines():
        p = line.split()
        if len(p) >= 3:
            kind[int(p[0], 16)] = p[1]
    names = {}
    for line in subprocess.run(["nm", "--defined-only", "-C", a.binary], capture_output=True, text=True).stdout.splitlines():
        p = line.split(None, 2)
        if len(p) == 3:
            names.setdefault(int(p[0], 16), p[2])
    proc = subprocess.Popen(["objdump", "-d", "-w", "--no-show-raw-insn", a.binary], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    head = re.compile(r"^([0-9a-f]+) <(.+)>:$")
    ins = re.compile(r"^\s*[0-9a-f]+:\s+(\S+)\s*(.*)$")
    cur, cur_addr, hits = None, 0, {}
    for line in proc.stdout:
        m = head.match(line)
        if m:
            cur_addr = int(m.group(1), 16)
            cur = names.get(cur_addr, m.group(2))
            continue
        m = ins.match(line)
        if not m or cur is None:
            continue
        mn, ops = m.group(1), m.group(2)
        if mn in ("data16", "rex", "lock", "rep", "repz", "repnz", "notrack", "bnd") and ops:
            mn, _, ops = ops.partition(" ")
        r = bad_reason(mn, ops, a.floor)
        if r:
            h = hits.setdefault((cur_addr, cur), {"reasons": set(), "ex": []})
            h["reasons"].add(r)
            if len(h["ex"]) < a.show:
                h["ex"].append(mn + " " + ops.strip())
    proc.wait()
    ok_tier, bad = [], []
    for (addr, name), h in sorted(hits.items(), key=lambda x: x[0][1]):
        k = kind.get(addr, "?")
        local = k in ("t", "d", "b", "r")
        tier = ("(anonymous namespace)::kn::" in name and local) or re.match(r"^strata_vit::cpu_ops_(sse41|avx|avx2|avx512)\(\)$", name)
        (ok_tier if tier else bad).append((name, k, h))
    print("floor %s: %d functions with instructions above the floor; %d are kernel-tier code behind the CPU check, %d are NOT" % (
        a.floor, len(hits), len(ok_tier), len(bad)))
    for name, k, h in bad:
        weak = " WEAK/COMDAT" if k in ("W", "w", "V", "v", "u") else ""
        print("  NOT GUARDED [%s]%s %s: %s | e.g. %s" % (k, weak, name[:200], ",".join(sorted(h["reasons"])), " ; ".join(h["ex"])))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
