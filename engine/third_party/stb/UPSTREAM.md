# third_party/stb: stb_image.h, frozen

The image decoder of Strata's own Qwen vision tower (`tools/vision/own/`, own-stack phase 3): JPEG, PNG, BMP, GIF (first frame), PSD, TGA, HDR, PIC and PNM
to 8-bit RGB. The same decoder llama.cpp's mtmd uses in the `strata-vision` helper today, so the pixels that reach the tower are the same ones.
Phase 3 deletes `third_party/strata-llama`, whose `vendor/stb/stb_image.h` this file is a copy of; this folder is what stays.

pin: 3cf03257f219afbe7334045ff7c6a06ac68c627d

Source: `vendor/stb/stb_image.h` of https://github.com/ggml-org/llama.cpp at that commit (the same pin as `third_party/strata-ggml`,
`third_party/strata-gguf-py` and `third_party/strata-llama`). It is stb_image **v2.30** by Sean Barrett (https://github.com/nothings/stb). Licence: MIT or
public domain, your choice; the licence text is at the end of `stb_image.h` (kept, unchanged) and in `LICENSE`. There are no Strata edits: `stb_image.h`
is byte-identical to the upstream file (LF line ends).

`python tools/check_vendored.py` verifies the list below (with `--upstream <llama.cpp tree at the pin>` it also compares the hash with the upstream file,
`stb_image.h` being `vendor/stb/stb_image.h`).

<!-- vendored-files:begin -->
```
# kind sha256 upstream-sha256 path (relative to third_party/stb)
added 76ae95c8dc44cc872b34c8e20302eaaa5b63a15488da48fceaeb1cee8376e52c - LICENSE
upstream 594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3 594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3 stb_image.h
```
<!-- vendored-files:end -->
