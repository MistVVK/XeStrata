"""List the top-level function blocks of a C++ file (a block = a line at column 0 that opens a function, up to the next `}` at column 0),
the namespace nesting they sit in, and the file-scope statics / anonymous-namespace objects. A reading aid for the split plan, not a parser.

    python -I cpp_funcs.py <file.cpp> [--min N]       (--min: only blocks of N lines or more; default 60)
"""
import re
import sys

path = sys.argv[1]
mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 60
lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
ns_stack = []
blocks = []
i = 0
FUNC = re.compile(r"^(?:template\s*<.*>\s*)?(?:static\s+|inline\s+|extern\s+|constexpr\s+|__global__\s+|__device__\s+)*[A-Za-z_][\w:<>,\*&\s~]*?\b([A-Za-z_~][\w:~]*)\s*\(")
NS = re.compile(r"^\s*namespace\s*([\w:]*)\s*\{")
depth = 0
while i < len(lines):
    l = lines[i]
    m = NS.match(l)
    if m and l.rstrip().endswith("{"):
        ns_stack.append(m.group(1) or "(anon)")
    elif re.match(r"^\}\s*(//.*)?$", l) and ns_stack and re.search(r"namespace|//", l) and "namespace" in l:
        ns_stack.pop()
    elif re.match(r"^\}\s*(//\s*namespace.*)?$", l) and l.startswith("}") and "namespace" in l and ns_stack:
        ns_stack.pop()
    elif not l.startswith((" ", "\t", "#", "//", "}", "{")) and "(" in l and FUNC.match(l) and not l.rstrip().endswith(";"):
        # a definition: find the opening brace line and the closing `}` at column 0
        j = i
        while j < len(lines) and "{" not in lines[j] and not lines[j].rstrip().endswith(";") and j - i < 12:
            j += 1
        if j < len(lines) and "{" in lines[j]:
            k = j
            if lines[j].rstrip().endswith("}") and lines[j].count("{") == lines[j].count("}"):
                end = j
            else:
                k = j + 1
                while k < len(lines) and not lines[k].startswith("}"):
                    k += 1
                end = k
            blocks.append((i + 1, end + 1, FUNC.match(l).group(1), "::".join(ns_stack), l.strip()[:90]))
            i = end
    i += 1
tot = sum(e - s + 1 for s, e, *_ in blocks)
print("%s: %d lines, %d function blocks (%d lines in them)" % (path, len(lines), len(blocks), tot))
for s, e, name, ns, head in blocks:
    if e - s + 1 >= mn:
        print("%6d-%6d (%5d)  %-34s ns=%-28s %s" % (s, e, e - s + 1, name[:34], ns[:28], head[:60]))
print("-- file-scope statics / objects in anonymous or file namespace (static or constexpr or thread_local at column 0):")
cnt = 0
for n, l in enumerate(lines, 1):
    if re.match(r"^(static|thread_local|constexpr|const)\s", l) and "(" not in l.split("=")[0]:
        cnt += 1
        if cnt <= 40:
            print("%6d  %s" % (n, l.strip()[:110]))
print("   ... %d in all" % cnt)
