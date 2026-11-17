// src/image/png/png_write.cpp - see include/strata/image/png_write.hpp.
#include "strata/image/png_write.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace strata::image {
namespace {

const std::array<uint32_t, 256>& crc_table() {
    static const std::array<uint32_t, 256> t = [] {
        std::array<uint32_t, 256> r{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            r[i] = c;
        }
        return r;
    }();
    return t;
}

void put32(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back((uint8_t) (v >> 24));
    o.push_back((uint8_t) (v >> 16));
    o.push_back((uint8_t) (v >> 8));
    o.push_back((uint8_t) v);
}

// one chunk: length, type, data, CRC over type + data
void chunk(std::vector<uint8_t>& o, const char type[4], const uint8_t* data, size_t n) {
    put32(o, (uint32_t) n);
    const size_t at = o.size();
    o.insert(o.end(), type, type + 4);
    if (n) o.insert(o.end(), data, data + n);
    put32(o, png_crc32(o.data() + at, n + 4));
}

}  // namespace

uint32_t png_crc32(const uint8_t* p, size_t n, uint32_t crc) {
    const auto& t = crc_table();
    uint32_t c = crc ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = t[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

uint32_t png_adler32(const uint8_t* p, size_t n, uint32_t a) {
    uint32_t s1 = a & 0xFFFFu, s2 = a >> 16;
    while (n > 0) {
        const size_t k = n < 5552 ? n : 5552;   // the largest run before the sums can overflow 32 bits
        for (size_t i = 0; i < k; ++i) {
            s1 += p[i];
            s2 += s1;
        }
        s1 %= 65521u;
        s2 %= 65521u;
        p += k;
        n -= k;
    }
    return (s2 << 16) | s1;
}

std::vector<uint8_t> png_encode(const uint8_t* pixels, int w, int h, int channels, const std::string& parameters) {
    if (w <= 0 || h <= 0 || (channels != 1 && channels != 3 && channels != 4)) return {};
    // the raw stream: per row the filter byte 0, then the row's bytes
    const size_t row = (size_t) w * channels;
    std::vector<uint8_t> raw;
    raw.reserve((row + 1) * h);
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), pixels + (size_t) y * row, pixels + (size_t) (y + 1) * row);
    }
    // zlib: CMF 0x78 (deflate, 32 KB window), FLG 0x01 (fastest; 0x7801 % 31 == 0), stored blocks of <= 65535 bytes,
    // Adler-32 of the raw stream, big-endian
    std::vector<uint8_t> z{0x78, 0x01};
    for (size_t at = 0;;) {   // raw holds at least one row
        const size_t len = std::min<size_t>(65535, raw.size() - at);
        const bool last = at + len == raw.size();
        z.push_back(last ? 1 : 0);                 // BFINAL, BTYPE 00 (stored); the rest of the byte is padding
        z.push_back((uint8_t) len);
        z.push_back((uint8_t) (len >> 8));
        z.push_back((uint8_t) ~len);
        z.push_back((uint8_t) (~len >> 8));
        z.insert(z.end(), raw.begin() + (ptrdiff_t) at, raw.begin() + (ptrdiff_t) (at + len));
        at += len;
        if (last) break;
    }
    const uint32_t ad = png_adler32(raw.data(), raw.size());
    put32(z, ad);

    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    static const uint8_t ctype[5] = {0, 0, 4, 2, 6};   // by channel count: grey, grey+alpha, RGB, RGBA
    std::vector<uint8_t> o(sig, sig + 8);
    uint8_t ihdr[13];
    for (int i = 0; i < 4; ++i) {
        ihdr[i] = (uint8_t) ((uint32_t) w >> (24 - 8 * i));
        ihdr[4 + i] = (uint8_t) ((uint32_t) h >> (24 - 8 * i));
    }
    ihdr[8] = 8;                      // bit depth
    ihdr[9] = ctype[channels];
    ihdr[10] = ihdr[11] = ihdr[12] = 0;   // deflate, adaptive filtering (type 0 per row), no interlace
    chunk(o, "IHDR", ihdr, 13);
    if (!parameters.empty()) {        // as stb_image_write in sd.cpp: keyword "parameters", a NUL, the text
        std::vector<uint8_t> t((const uint8_t*) "parameters", (const uint8_t*) "parameters" + 10);
        t.push_back(0);
        t.insert(t.end(), parameters.begin(), parameters.end());
        chunk(o, "tEXt", t.data(), t.size());
    }
    chunk(o, "IDAT", z.data(), z.size());
    chunk(o, "IEND", nullptr, 0);
    return o;
}

std::string png_write_file(const std::string& path, const uint8_t* pixels, int w, int h, int channels,
                           const std::string& parameters) {
    const std::vector<uint8_t> png = png_encode(pixels, w, h, channels, parameters);
    if (png.empty()) return "png_write: bad image shape";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return "png_write: cannot create " + path;
    const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    return std::fclose(f) == 0 && ok ? std::string() : "png_write: cannot write " + path;
}

}  // namespace strata::image
