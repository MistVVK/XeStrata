// include/strata/artifact/strata_reader.hpp - Native .strata v1.0 container reader (docs/FORMAT-SPEC.md §2)
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::artifact {

constexpr size_t STRATA_PAGE_SIZE = 4096;
constexpr uint16_t STRATA_NUM_SECTIONS = 6;

enum class StrataProfile : uint16_t {
    Tiered      = 0,
    Unified     = 1,
    EngramShard = 2,
    VisionShard = 3,
};

#pragma pack(push, 1)
struct StrataSectionEntry {
    uint64_t file_offset;
    uint64_t byte_length;
    uint8_t  sha256[32];
    uint32_t section_id;
    uint32_t flags;
    uint64_t reserved;
};
static_assert(sizeof(StrataSectionEntry) == 64, "StrataSectionEntry must be 64 B");

struct StrataExpertEntry {
    uint16_t expert_id;
    uint8_t  gu_code_id;
    uint8_t  dn_code_id;
    uint16_t cluster_id;
    uint16_t flags;
    uint64_t body_offset;
    uint32_t body_bytes;
    uint32_t dn_offset;
    uint32_t aux_offsets[2];
};
static_assert(sizeof(StrataExpertEntry) == 32, "StrataExpertEntry must be 32 B");
#pragma pack(pop)

struct StrataHeader {
    uint32_t flags = 0;
    StrataProfile profile = StrataProfile::Tiered;
    uint16_t num_sections = 0;
    StrataSectionEntry sections[STRATA_NUM_SECTIONS]{};
    uint8_t engram_shard_sha256[32]{};
};

class StrataFileReader {
public:
    StrataFileReader() = default;
    ~StrataFileReader() { close(); }
    StrataFileReader(const StrataFileReader&) = delete;
    StrataFileReader& operator=(const StrataFileReader&) = delete;

    bool open(const std::string& path, std::string& err);
    void close();
    bool is_open() const noexcept { return fp_ != nullptr; }

    const StrataHeader& header() const noexcept { return hdr_; }
    const std::string& manifest_json() const noexcept { return manifest_json_; }
    const std::vector<StrataExpertEntry>& expert_directory() const noexcept { return experts_; }

    bool read_section(uint32_t section_id, std::vector<uint8_t>& out, std::string& err) const;
    bool read_at(uint64_t abs_offset, void* dst, size_t bytes, std::string& err) const;

    // Convenience: read one expert body from Section 4 by [moe_layer * n_experts + rank]
    bool read_expert_body(size_t dir_index, std::vector<uint8_t>& out, std::string& err) const;

private:
    std::FILE* fp_ = nullptr;
    StrataHeader hdr_{};
    std::string manifest_json_;
    std::vector<StrataExpertEntry> experts_;
};

}  // namespace strata::artifact
