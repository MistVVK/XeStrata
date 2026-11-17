"""Per failing pytest test: the first frame in a repository file and the first E line.  usage: python -I fail_sites.py <log>..."""
import io, re, sys

HEAD = re.compile(r"^_{3,} (\S.*?) _{3,}$")
FRAME = re.compile(r"^((?:tests|tools|serve|strata_\w+|setup\.py|engine|terminal|\.\.)\S*?\.py):(\d+): ")
for path in sys.argv[1:]:
    text = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    cur, frames, err = None, [], None
    out = []
    def flush():
        if cur:
            out.append((cur, frames[:3], err))
    for line in text:
        m = HEAD.match(line)
        if m:
            flush()
            cur, frames, err = m.group(1), [], None
            continue
        if cur is None:
            continue
        f = FRAME.match(line)
        if f:
            frames.append(f.group(1) + ":" + f.group(2))
        if err is None and line.startswith("E  ") and line[1:].strip():
            err = line[1:].strip()[:170]
        if line.startswith("=") and "short test summary" in line:
            flush()
            cur = None
    sys.stdout.reconfigure(encoding="utf-8")
    for name, fr, e in out:
        print("%s | %s | %s" % (name, " > ".join(fr), e))
