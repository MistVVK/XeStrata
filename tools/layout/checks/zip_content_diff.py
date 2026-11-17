"""python -I zip_content_diff.py <before.zip> <after.zip>: members (outside python/) whose bytes differ (CRC)."""
import sys
import zipfile

crc = []
for p in sys.argv[1:3]:
    with zipfile.ZipFile(p) as z:
        crc.append({i.filename.split("/", 1)[1]: i.CRC for i in z.infolist() if "/" in i.filename and not i.filename.split("/", 1)[1].startswith("python/")})
diff = sorted(n for n in crc[0] if n in crc[1] and crc[0][n] != crc[1][n])
print("%d members differ in content:" % len(diff))
for n in diff:
    print("  " + n)
