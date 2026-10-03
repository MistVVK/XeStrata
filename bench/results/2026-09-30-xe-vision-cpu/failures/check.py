"""The Vision class against fake encoders (enc_*.sh): a start that fails, a start that never answers, then an ERR
reply, an encoder that exits, and one that hangs.  The waits are cut to 3 s.
    python check.py   (from anywhere; it finds the repository two levels up)"""
import base64, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from serve.server import Vision
D = str(HERE) + "/"
png = (HERE.parent / "landscape.png").read_bytes()
def img(tag):   # a distinct PNG per call (the cache is keyed by content): a tag in a trailing chunk-free suffix
    return "data:image/png;base64," + base64.b64encode(png + tag.encode()).decode()
def cfg(exe): return {"exe": D + exe, "mmproj": "m", "model": "t"}
Vision.READY_TIMEOUT_S = 3
Vision.ENC_TIMEOUT_S = 3
for exe in ("enc_err_start.sh", "enc_silent.sh"):
    t0 = time.time()
    try:
        Vision(cfg(exe)); print(exe, "started?!")
    except RuntimeError as e:
        print(f"{exe}: RuntimeError after {time.time() - t0:.1f} s: {e}")
v = Vision(cfg("enc_ok.sh"))
print("ok image:", v.encode(img("a"))[1])
# the fake decides by the image FILE name, which is the content hash: rename via monkeypatching the key is not possible,
# so drive the three failure lines by writing ENC lines directly through encode() with names the fake recognises
import hashlib
orig = hashlib.sha256
for tag in ("bad", "die", "after", "hang"):
    class H:
        def __init__(self, data): self.data = data
        def hexdigest(self): return tag + orig(self.data).hexdigest()
    hashlib.sha256 = H
    t0 = time.time()
    try:
        print(tag, "->", v.encode(img(tag))[1])
    except ValueError as e:
        print(f"{tag}: ValueError after {time.time() - t0:.1f} s: {e}; dead={v.dead!r}")
hashlib.sha256 = orig
v2 = Vision(cfg("enc_ok.sh"))
hashlib.sha256 = lambda d, tag="hang": type("H", (), {"hexdigest": lambda self: "hang" + orig(d).hexdigest()})()
t0 = time.time()
try:
    v2.encode(img("h2"))
except ValueError as e:
    print(f"hang: ValueError after {time.time() - t0:.1f} s: {e}; dead={v2.dead!r}; alive={v2.proc.poll() is None}")
hashlib.sha256 = orig
v.close(); v2.close()
