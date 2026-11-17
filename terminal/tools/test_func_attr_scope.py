"""cudaFuncSetAttribute / hipFuncSetAttribute must never run inside a once-per-PROCESS initializer.

The dynamic shared memory opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize) and the carveout belong to the DEVICE, not to the process.  A layer split or a
tensor-parallel run launches the same kernel on several cards; a function-local `static const bool once = [] { cudaFuncSetAttribute(...); return true; }();`
(or a std::call_once) sets it on the first card that gets there and leaves the others at the 48 KB default: the second card's launch fails with "invalid
argument", and inside a CUDA graph capture that shows up as "operation failed due to a previous error during capture" (the int3 line, 2 x RTX PRO 6000, 0 tokens).

Allowed: a per-device flag (`static bool done[64]; ... done[dev]`), a per-device table, or calling it on every launch.  A site that really is process-wide
(a single-device tool) can say so with a comment `// attr-scope: process` on the line of the call or the line before it.

Run: python -m pytest tools/test_func_attr_scope.py   (CPU only, reads src/ and include/)
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"\b(?:cuda|hip)FuncSetAttribute\s*\(")
# a function-local static initialised by an immediately invoked lambda: `static [const] <type> name = [..](..) [-> T] {`
STATIC_LAMBDA = re.compile(r"\bstatic\s+(?:const\s+|constexpr\s+)?(?:\w+(?:::\w+)*)\s+\w+\s*=\s*\[[^\]]*\]\s*(?:\([^)]*\))?\s*(?:->\s*[\w:<>\s*&]+)?\s*\{")
CALL_ONCE = re.compile(r"\bstd::call_once\s*\(")
SCAN_DIRS = ("src", "include")
SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp", ".h", ".cc"}


def _strip(text: str) -> str:
    """Comments and string literals blanked out (same length), so braces inside them do not count."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _body_end(code: str, open_brace: int) -> int:
    depth = 0
    for k in range(open_brace, len(code)):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(code)


def find_violations(text: str) -> list[tuple[int, str]]:
    """[(line, why)] for every attribute call inside a once-per-process initializer."""
    code = _strip(text)
    lines = text.splitlines()
    bad: list[tuple[int, str]] = []
    spans: list[tuple[int, int, str]] = []
    for m in STATIC_LAMBDA.finditer(code):
        o = m.end() - 1
        spans.append((o, _body_end(code, o), "a function-local static initialised by a lambda"))
    for m in CALL_ONCE.finditer(code):
        o = code.find("{", m.end())
        if o >= 0:
            spans.append((o, _body_end(code, o), "std::call_once"))
    for c in CALL.finditer(code):
        line = text.count("\n", 0, c.start()) + 1
        ctx = " ".join(lines[max(0, line - 2):line])
        if "attr-scope: process" in ctx:
            continue
        for lo, hi, why in spans:
            if lo < c.start() < hi:
                bad.append((line, why))
                break
    return bad


def scan() -> list[str]:
    found = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix in SUFFIXES and p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if not CALL.search(text):
                    continue
                for line, why in find_violations(text):
                    found.append(f"{p.relative_to(ROOT).as_posix()}:{line}: cudaFuncSetAttribute inside {why} (per PROCESS, but the attribute is per DEVICE)")
    return found


def test_the_tree_has_no_process_wide_attribute_setup():
    assert scan() == []


# ---- the detector itself: the two sites that failed on two RTX PRO 6000 cards, and the forms that must stay allowed
OLD_PF = """
template <int K, int PF>
static void gr_down_pf_launch(const GrMulti& m, float* part, float* ssg, cudaStream_t st) {
    constexpr size_t sm = (size_t) K * N * sizeof(float);
    static const bool once = [] {
        cudaFuncSetAttribute(gr_down_v3_kernel<1, K, true, 1, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
        return true;
    }();
    (void) once;
}
"""
OLD_RB = """
static void f(cudaStream_t st) {
    static const bool once = []() -> bool {
        cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
}
"""
CALL_ONCE_SAMPLE = """
void g() { static std::once_flag f; std::call_once(f, [] { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); }); }
"""
PER_DEVICE_OK = """
static bool attrs[64] = {};
int dev = 0; cudaGetDevice(&dev);
if (!attrs[dev]) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); attrs[dev] = true; }
static const bool env = [] { const char* e = getenv("X"); return e != nullptr; }();   // a once-lambda without the call is fine
"""
HELPER_OK = """
template <class F> static void set_attr_once_per_device(bool (&done)[64], F&& set) {
    static std::mutex mu; int dev = 0; cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mu);
    if (!done[dev]) { set(); done[dev] = true; }
}
static void h() { static bool done[64] = {}; set_attr_once_per_device(done, [] { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); }); }
"""
MARKED_OK = """
static void tool() {
    // attr-scope: process (a single-device test tool)
    static const bool once = [] { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536); return true; }();
}
"""
COMMENT_ONLY = """
// static const bool once = [] { cudaFuncSetAttribute(k, a, 1); return true; }();
const char* s = "static const bool once = [] { cudaFuncSetAttribute(k, a, 1); return true; }();";
"""


def test_detector_flags_the_failing_forms():
    assert find_violations(OLD_PF), "the gr_down_pf_launch form must be flagged"
    assert find_violations(OLD_RB), "a lambda with a trailing return type must be flagged"
    assert find_violations(CALL_ONCE_SAMPLE), "std::call_once must be flagged"


def test_detector_keeps_the_allowed_forms():
    assert find_violations(PER_DEVICE_OK) == []
    assert find_violations(HELPER_OK) == []
    assert find_violations(MARKED_OK) == []
    assert find_violations(COMMENT_ONLY) == []
