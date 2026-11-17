"""Inside one huge C++ function: count its top-level (2-space indent) declarations and list its top-level blocks of N lines or more.
    python -I big_blocks.py <file.cpp> <first line> <last line> [--min N]
A reading aid for the split plan: the declarations are the locals every stage would share; the big blocks are the stage candidates."""
import re
import sys

path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 150
L = open(path, encoding="utf-8", errors="replace").read().split("\n")
decl = re.compile(r"^    (?:const\s+|static\s+|constexpr\s+)*(?:auto|int|int32_t|int64_t|uint32_t|uint64_t|size_t|bool|double|float|std::[\w:<>,\* ]+|strata::[\w:<>,\* ]+|[A-Z]\w+)[&\*]?\s+[a-z_]\w*\s*(=|\{|\(|;|\[)")
decls = []
blocks = []
i = a
while i <= b:
    l = L[i - 1]
    if re.match(r"^    \S", l) and not l.startswith("    //") and not l.startswith("    #"):
        if decl.match(l):
            decls.append(i)
        if l.rstrip().endswith("{") or (("{" in l) and not l.rstrip().endswith(";")):
            j = i
            while j <= b and not re.match(r"^    \}", L[j - 1]):
                j += 1
            if j - i + 1 >= mn:
                blocks.append((i, j, l.strip()[:100]))
            i = j
    i += 1
print("%s %d-%d: %d top-level declarations (locals shared by every stage), %d blocks of >= %d lines" % (path, a, b, len(decls), len(blocks), mn))
tot = 0
for s, e, h in blocks:
    tot += e - s + 1
    print("  %6d-%6d (%5d) %s" % (s, e, e - s + 1, h))
print("lines in those blocks: %d of %d" % (tot, b - a + 1))
