"""A stand-in for stable-diffusion.cpp's sd-cli (serve/test_images.py): reads the arguments serve/images.py passes and
writes one small PNG per image whose pixels encode the seed, size and prompt, so a test can tell the images apart.
FAKE_SD_FAIL=1: exits 1 after a log line.  FAKE_SD_SLEEP=<s>: waits that long first (cancel tests).
FAKE_SD_ARGS=<file>: writes its argument list there (JSON).  FAKE_SD_RGBA=1: RGBA pictures like Qwen-Image-2.1's
(rgba_png); with FAKE_SD_BROKEN=1 their IDAT CRC is wrong."""
import hashlib
import json
import os
import struct
import sys
import time
import zlib


def png(w: int, h: int, seed: int, text: str) -> bytes:
    digest = hashlib.sha256(f"{seed}|{w}x{h}|{text}".encode()).digest()
    row = b"\x00" + digest[:3] * 4
    raw = row * 4
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
            + chunk(b"tEXt", f"parameters\x00{text}, Seed: {seed}, Size: {w}x{h}".encode("latin-1", "replace"))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


KEY = (160, 50, 200)          # the flat colour Qwen-Image-2.1 leaves under alpha 0 (a purple)


def rgba_png(w: int, h: int, seed: int, text: str, broken: bool = False) -> bytes:
    """A 4x4 8-bit RGBA PNG like Qwen-Image-2.1's: row 0 alpha 0 over the key colour, row 1 alpha 128, row 2 alpha
    255, row 3 alpha 1 / 64 / 254 / 255; the "parameters" chunk carries non-ASCII bytes.  broken: a bad IDAT CRC."""
    digest = hashlib.sha256(f"{seed}|{w}x{h}|{text}".encode()).digest()
    r, g, b = digest[0], digest[1], digest[2]
    rows = [bytes(KEY) + b"\x00", bytes((r, g, b, 128)), bytes((r, g, b, 255))]
    raw = b"".join(b"\x00" + px * 4 for px in rows)
    raw += b"\x00" + b"".join(bytes((r, g, b, a)) for a in (1, 64, 254, 255))
    def chunk(t, d, bad=False):
        crc = zlib.crc32(t + d) & 0xFFFFFFFF
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc ^ (1 if bad else 0))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0))
            + chunk(b"tEXt", f"parameters\x00{text}, Seed: {seed}, Size: {w}x{h} é".encode("latin-1", "replace"))
            + chunk(b"IDAT", zlib.compress(raw), broken) + chunk(b"IEND", b""))


def main(argv):
    a = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            a[argv[i]] = True
            i += 1
    if os.environ.get("FAKE_SD_ARGS"):
        with open(os.environ["FAKE_SD_ARGS"], "w", encoding="utf-8") as f:
            json.dump(argv, f)
    time.sleep(float(os.environ.get("FAKE_SD_SLEEP") or 0))
    print("[INFO ] loading tensors completed, taking 0.10s", flush=True)
    if os.environ.get("FAKE_SD_FAIL"):
        print("[ERROR] ggml_vulkan: ErrorDeviceLost", flush=True)
        return 1
    steps = int(a.get("--steps", 20))
    for s in range(1, steps + 1):
        print(f"  |{'=' * s}| {s}/{steps} - 0.01s/it", flush=True)
    # stable-diffusion.cpp 2988060 prints seconds here (pipeline/image.cpp: "taking %.2fs")
    print("[I] get_learned_condition completed, taking 0.70s --- image.cpp:529", flush=True)
    print("[INFO ] sampling completed, taking 0.08s", flush=True)
    print("[INFO ] decode_first_stage completed, taking 0.02s", flush=True)
    w, h, seed, n = int(a["-W"]), int(a["-H"]), int(a["-s"]), int(a.get("-b", 1))
    with open(a["--prompt-file"], encoding="utf-8", newline="") as f:     # the prompt is never an argument (#4202)
        prompt = f.read()
    for k in range(n):
        with open(a["-o"] % k, "wb") as f:
            if os.environ.get("FAKE_SD_RGBA"):
                f.write(rgba_png(w, h, seed + k, prompt, broken=bool(os.environ.get("FAKE_SD_BROKEN"))))
            else:
                f.write(png(w, h, seed + k, prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
