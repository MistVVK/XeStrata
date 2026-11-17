// include/strata/image/png_write.hpp - a small PNG writer of Strata's own (step 5): 8-bit grey / RGB / RGBA, every row
// with filter 0, the pixel data in deflate STORED blocks inside a zlib stream (no compression yet: no dependency), and
// the "parameters" tEXt chunk right after IHDR, where stable-diffusion.cpp's stb_image_write puts it (sd-cli writes
// the generation parameters there).  The file bytes are not stb's (stb compresses); the decoded pixels are the same.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

// CRC-32 (ISO-HDLC / zlib / PNG: reflected polynomial 0xEDB88320, init and final xor 0xFFFFFFFF); crc = a previous
// result to continue over more bytes
uint32_t png_crc32(const uint8_t* p, size_t n, uint32_t crc = 0);

// Adler-32 (zlib); a = a previous result to continue
uint32_t png_adler32(const uint8_t* p, size_t n, uint32_t a = 1);

// the PNG file: pixels row-major, `channels` (1, 3 or 4) bytes a pixel, w x h; parameters "" = no text chunk
std::vector<uint8_t> png_encode(const uint8_t* pixels, int w, int h, int channels, const std::string& parameters);

// png_encode into a file; "" on success
std::string png_write_file(const std::string& path, const uint8_t* pixels, int w, int h, int channels,
                           const std::string& parameters);

}  // namespace strata::image
