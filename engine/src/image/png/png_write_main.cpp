// src/image/png/png_write_main.cpp - test program strata-png-write (not shipped) for include/strata/image/png_write.hpp.
//
//   strata-png-write --vectors                              CRC-32 / Adler-32 against published check values
//   strata-png-write --pattern W H C OUT.png [PARAMETERS]   a fixed test pattern (byte = (7x + 13y + 29c) mod 256)
//   strata-png-write --raw IN W H C OUT.png [PARAMETERS]    IN: W*H*C bytes, row-major
#include "strata/image/png_write.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::image;

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--vectors") == 0) {
        struct V { const char* s; uint32_t crc, adler; };
        // CRC-32/ISO-HDLC check value (the CRC catalogue), Adler-32 of "Wikipedia" (RFC 1950's algorithm), empty input
        const V vs[] = {{"123456789", 0xCBF43926u, 0x091E01DEu}, {"Wikipedia", 0xADAAC02Eu, 0x11E60398u}, {"", 0u, 1u},
                        {"The quick brown fox jumps over the lazy dog", 0x414FA339u, 0x5BDC0FDAu}};
        int bad = 0;
        for (const V& v : vs) {
            const uint32_t c = png_crc32((const uint8_t*) v.s, std::strlen(v.s));
            const uint32_t a = png_adler32((const uint8_t*) v.s, std::strlen(v.s));
            std::printf("\"%s\": crc32 %08x (%s), adler32 %08x (%s)\n", v.s, c, c == v.crc ? "ok" : "WRONG", a,
                        a == v.adler ? "ok" : "WRONG");
            bad += c != v.crc || a != v.adler;
        }
        // continuing over two parts gives the whole
        const char* s = "123456789";
        const uint32_t c2 = png_crc32((const uint8_t*) s + 4, 5, png_crc32((const uint8_t*) s, 4));
        const uint32_t a2 = png_adler32((const uint8_t*) s + 4, 5, png_adler32((const uint8_t*) s, 4));
        std::printf("split: crc32 %s, adler32 %s\n", c2 == 0xCBF43926u ? "ok" : "WRONG", a2 == 0x091E01DEu ? "ok" : "WRONG");
        bad += c2 != 0xCBF43926u || a2 != 0x091E01DEu;
        // Adler-32 over 1 MB of 0xFF (the sums' modulo path)
        std::vector<uint8_t> ff(1 << 20, 0xFF);
        std::printf("adler32 of 1 MiB of 0xff: %08x\n", png_adler32(ff.data(), ff.size()));
        std::printf("VECTORS %s\n", bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }
    if (argc >= 6 && std::strcmp(argv[1], "--pattern") == 0) {
        const int w = std::atoi(argv[2]), h = std::atoi(argv[3]), c = std::atoi(argv[4]);
        std::vector<uint8_t> px((size_t) w * h * c);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int k = 0; k < c; ++k) px[((size_t) y * w + x) * c + k] = (uint8_t) ((7 * x + 13 * y + 29 * k) & 255);
        const std::string e = png_write_file(argv[5], px.data(), w, h, c, argc > 6 ? argv[6] : "");
        if (!e.empty()) std::fprintf(stderr, "%s\n", e.c_str());
        return e.empty() ? 0 : 1;
    }
    if (argc >= 7 && std::strcmp(argv[1], "--raw") == 0) {
        const int w = std::atoi(argv[3]), h = std::atoi(argv[4]), c = std::atoi(argv[5]);
        std::ifstream f(argv[2], std::ios::binary);
        std::vector<uint8_t> px((std::istreambuf_iterator<char>(f)), {});
        if (px.size() != (size_t) w * h * c) {
            std::fprintf(stderr, "raw size %zu != %d x %d x %d\n", px.size(), w, h, c);
            return 1;
        }
        const std::string e = png_write_file(argv[6], px.data(), w, h, c, argc > 7 ? argv[7] : "");
        if (!e.empty()) std::fprintf(stderr, "%s\n", e.c_str());
        return e.empty() ? 0 : 1;
    }
    std::fprintf(stderr, "usage: strata-png-write --vectors | --pattern W H C OUT [PARAMS] | --raw IN W H C OUT [PARAMS]\n");
    return 2;
}
