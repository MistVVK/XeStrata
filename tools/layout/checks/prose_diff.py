"""New unedited path mentions since MAP-0215: entries of the new prose report whose (file, token) pair the old report did not have
as often.  usage: python -I prose_diff.py <old prose.tsv> <new prose.tsv> <out.tsv>"""
import collections, io, sys

def load(p):
    rows = []
    with io.open(p, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                rows.append(parts)
    return rows

old, new, out = sys.argv[1:4]
o = collections.Counter((r[0], r[2]) for r in load(old))
seen = collections.Counter()
fresh = []
for r in load(new):
    k = (r[0], r[2])
    seen[k] += 1
    if seen[k] > o.get(k, 0):
        fresh.append(r)
with io.open(out, 'w', encoding='utf-8') as f:
    f.write('file\tline\tnames\twould be\n' + ''.join('\t'.join(r) + '\n' for r in fresh))
by = collections.Counter(r[0] for r in fresh)
print(len(fresh), 'new mentions in', len(by), 'files')
for p, n in by.most_common(60):
    print(n, p)
