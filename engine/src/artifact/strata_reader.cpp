// src/artifact/strata_reader.cpp - Native .strata v1.0 container reader (docs/FORMAT-SPEC.md §2)
#include "strata/artifact/strata_reader.hpp"

#include <zstd.h>

#include <cstring>

namespace strata::artifact {
namespace {

constexpr uint8_t kStrataMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};

bool seek_abs(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

}  // namespace

void StrataFileReader::close() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
    manifest_json_.clear();
    experts_.clear();
}

bool StrataFileReader::open(const std::string& path, std::string& err) {
    close();
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
        err = "Failed to open .strata file: " + path;
        return false;
    }
    uint8_t frame[STRATA_PAGE_SIZE];
    if (std::fread(frame, 1, STRATA_PAGE_SIZE, fp_) != STRATA_PAGE_SIZE) {
        err = "Short read on .strata 4 KiB header frame";
        close();
        return false;
    }
    if (std::memcmp(frame, kStrataMagic, 8) != 0) {
        err = "Invalid .strata magic (expected STRATA\\x00\\x01)";
        close();
        return false;
    }
    std::memcpy(&hdr_.flags, frame + 8, 4);
    uint16_t prof = 0;
    std::memcpy(&prof, frame + 12, 2);
    hdr_.profile = static_cast<StrataProfile>(prof);
    std::memcpy(&hdr_.num_sections, frame + 14, 2);
    if (hdr_.num_sections != STRATA_NUM_SECTIONS) {
        err = "Unexpected .strata num_sections: " + std::to_string(hdr_.num_sections);
        close();
        return false;
    }
    for (int i = 0; i < STRATA_NUM_SECTIONS; ++i) {
        std::memcpy(&hdr_.sections[i], frame + 0x10 + i * sizeof(StrataSectionEntry), sizeof(StrataSectionEntry));
        if (hdr_.sections[i].byte_length > 0 && (hdr_.sections[i].file_offset % STRATA_PAGE_SIZE) != 0) {
            err = "Section " + std::to_string(i) + " is not 4 KiB aligned";
            close();
            return false;
        }
    }
    std::memcpy(hdr_.engram_shard_sha256, frame + 0x190, 32);

    // Decompress ZSTD Section 0 manifest
    std::vector<uint8_t> s0;
    if (!read_section(0, s0, err)) {
        close();
        return false;
    }
    if (!s0.empty()) {
        unsigned long long const rsize = ZSTD_getFrameContentSize(s0.data(), s0.size());
        if (rsize == ZSTD_CONTENTSIZE_ERROR) {
            err = "Section 0 manifest is not valid ZSTD";
            close();
            return false;
        }
        if (rsize != ZSTD_CONTENTSIZE_UNKNOWN) {
            manifest_json_.resize(static_cast<size_t>(rsize));
            size_t const dsize = ZSTD_decompress(manifest_json_.data(), manifest_json_.size(), s0.data(), s0.size());
            if (ZSTD_isError(dsize)) {
                err = std::string("ZSTD decompress failed: ") + ZSTD_getErrorName(dsize);
                close();
                return false;
            }
            manifest_json_.resize(dsize);
        } else {
            // Streaming decompress fallback
            ZSTD_DCtx* dctx = ZSTD_createDCtx();
            ZSTD_inBuffer in_b{s0.data(), s0.size(), 0};
            std::vector<char> chunk(1 << 20);
            while (in_b.pos < in_b.size) {
                ZSTD_outBuffer out_b{chunk.data(), chunk.size(), 0};
                size_t ret = ZSTD_decompressStream(dctx, &out_b, &in_b);
                if (ZSTD_isError(ret)) {
                    err = std::string("ZSTD stream decompress failed: ") + ZSTD_getErrorName(ret);
                    ZSTD_freeDCtx(dctx);
                    close();
                    return false;
                }
                manifest_json_.append(chunk.data(), out_b.pos);
            }
            ZSTD_freeDCtx(dctx);
        }
    }

    // Load Section 3 expert directory (if present)
    if (hdr_.sections[3].byte_length > 0) {
        std::vector<uint8_t> s3;
        if (!read_section(3, s3, err)) {
            close();
            return false;
        }
        if (s3.size() % sizeof(StrataExpertEntry) != 0) {
            err = "Section 3 byte_length not a multiple of 32 B StrataExpertEntry";
            close();
            return false;
        }
        const size_t n = s3.size() / sizeof(StrataExpertEntry);
        experts_.resize(n);
        std::memcpy(experts_.data(), s3.data(), s3.size());
    }

    return true;
}

bool StrataFileReader::read_at(uint64_t abs_offset, void* dst, size_t bytes, std::string& err) const {
    if (!fp_) {
        err = "File not open";
        return false;
    }
    if (!seek_abs(fp_, abs_offset)) {
        err = "Seek failed at offset " + std::to_string(abs_offset);
        return false;
    }
    if (std::fread(dst, 1, bytes, fp_) != bytes) {
        err = "Short read at offset " + std::to_string(abs_offset);
        return false;
    }
    return true;
}

bool StrataFileReader::read_section(uint32_t section_id, std::vector<uint8_t>& out, std::string& err) const {
    if (section_id >= STRATA_NUM_SECTIONS) {
        err = "Invalid section_id";
        return false;
    }
    const auto& sec = hdr_.sections[section_id];
    out.resize(static_cast<size_t>(sec.byte_length));
    if (sec.byte_length == 0) return true;
    return read_at(sec.file_offset, out.data(), out.size(), err);
}

bool StrataFileReader::read_expert_body(size_t dir_index, std::vector<uint8_t>& out, std::string& err) const {
    if (dir_index >= experts_.size()) {
        err = "Expert directory index out of bounds";
        return false;
    }
    const auto& e = experts_[dir_index];
    out.resize(e.body_bytes);
    return read_at(hdr_.sections[4].file_offset + e.body_offset, out.data(), e.body_bytes, err);
}

}  // namespace strata::artifact
