// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 3, 3, 5, 5, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 3, 3, 5, 5, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 3, 3, 5, 5, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 3, 3, 5, 5, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 0, 0, 5, 5, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
    const std::vector<uint64_t> sizes{3, 7, 2, 9, 4, 5};
    require(make_cache_complement_plan(2, 3, sizes, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 18 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 5, 14, kNoCacheComplement},
            "wrong compact offsets for per-expert sizes");
    require(append_stage_lend_regions(2, 3, sizes, offsets, bytes, 25, {{{1, 2}, {0, 1}}}, error) == 1 &&
                bytes == 23 && offsets[5] == 18 && offsets[1] == kNoCacheComplement,
            "mixed-size lend region did not respect the budget");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {}, {}, offsets, bytes, error), "short size table accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 3, 3, 5, 5, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

// #4051: the wait for the RAM an arena needs (an engine still exiting gives it back a few seconds late)
void test_wait_for_ram() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    double waited = -1.0;
    uint64_t avail = 0;
    int slept = 0, noted = 0;
    const auto sleeper = [&](double) { ++slept; };
    const auto note = [&](double, uint64_t) { ++noted; };
    // enough at once: no wait, no sleep, no note
    require(wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t& a) { a = 50 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 0.0 && slept == 0 && noted == 0 && avail == 50 * GiB, "RAM that was there was waited for");
    // 10 GiB free for a 40 GiB arena, the exiting engine gives its 50 GiB back after 3 s (12 steps of 0.25 s)
    int reads = 0;
    slept = noted = 0;
    require(wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [&](uint64_t& a) { a = ++reads > 12 ? 60 * GiB : 10 * GiB; return true; },
                         sleeper, note, waited, avail) && waited == 3.0 && slept == 12 && noted == 1 && avail == 60 * GiB,
            "RAM that came back after 3 s was not waited for");
    // it never comes: the wait ends at the limit and says so (the load then goes on and warns)
    slept = noted = 0;
    require(!wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t& a) { a = 10 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 20.0 && slept == 80 && noted == 4 && avail == 10 * GiB, "the wait did not end at its limit");
    // no wait asked for: one reading
    slept = 0;
    require(!wait_for_ram(40 * GiB, 0.0, 0.25, 5.0, [](uint64_t& a) { a = 10 * GiB; return true; }, sleeper, note, waited, avail) &&
                waited == 0.0 && slept == 0, "a zero wait slept");
    // the RAM cannot be read: the wait ends at once
    slept = 0;
    require(!wait_for_ram(40 * GiB, 20.0, 0.25, 5.0, [](uint64_t&) { return false; }, sleeper, note, waited, avail) && slept == 0,
            "an unreadable RAM was waited for");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    // the kernel keeps min(half the clean cache, limit / 64) of it: 56 GiB / 64 = 0.875 GiB here
    const uint64_t keep56 = 56 * GiB / 64;
    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB - keep56, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB - keep56, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB - 12 * GiB / 64, "inactive-file accounting exceeded charged current usage");

    // 149: the ACTIVE page cache is reclaimable too (memory.stat "file" = active + inactive), as MemAvailable counts
    // it: a 32 GiB group whose own reads left 26 GiB of cache, 25 GiB of it active, and 4.5 GiB of anonymous memory
    CgroupMemoryStat active_cache{30 * GiB + GiB / 2, 1 * GiB, 0, 0, true, 26 * GiB};
    require(cgroup_available_bytes(32 * GiB, active_cache, bytes), "a memory.stat with file was rejected");
    require(bytes == 32 * GiB - (4 * GiB + GiB / 2) - 32 * GiB / 64, "the active page cache was not credited");
    // without "file" (an old kernel's stat): inactive_file alone, as before
    CgroupMemoryStat no_file{30 * GiB + GiB / 2, 1 * GiB, 0, 0, true};
    require(cgroup_available_bytes(32 * GiB, no_file, bytes) && bytes == 32 * GiB - (29 * GiB + GiB / 2) - GiB / 2,
            "a stat without file did not fall back to inactive_file");
    // dirty and writeback pages of the active cache stay charged; a small cache keeps half of itself
    CgroupMemoryStat dirty_active{20 * GiB, 1 * GiB, 2 * GiB, 1 * GiB, true, 10 * GiB};
    require(cgroup_available_bytes(32 * GiB, dirty_active, bytes) && bytes == 32 * GiB - (20 * GiB - 7 * GiB) - GiB / 2,
            "dirty / writeback pages of the active cache were counted as reclaimable");
    CgroupMemoryStat tiny_cache{8 * GiB, 0, 0, 0, true, 256ull << 20};
    require(cgroup_available_bytes(32 * GiB, tiny_cache, bytes) && bytes == 32 * GiB - 8 * GiB + (128ull << 20),
            "a small clean cache was not kept at half");
    // "file" larger than what is charged (a racy read) is bounded by memory.current
    CgroupMemoryStat racy{4 * GiB, 1 * GiB, 0, 0, true, 9 * GiB};
    require(cgroup_available_bytes(32 * GiB, racy, bytes) && bytes == 32 * GiB - GiB / 2,
            "a file counter above memory.current escaped the bound");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

// upstream 8c80a3ba (and ed5536ea): the file cache named in a "not enough RAM" start failure
void test_page_cache_hint() {
    using namespace strata::core::detail;
    const std::string info =
        "MemTotal:       65011484 kB\nMemFree:         1284096 kB\nMemAvailable:   33463296 kB\nCached:         54507520 kB\n";
    require(page_cache_hint(info).find("drop_caches") != std::string::npos &&
            page_cache_hint(info).find("XESTRATA_DROP_CACHE_ON_EXIT") != std::string::npos &&
            page_cache_hint(info).find("52.0 GiB") != std::string::npos, "the cache hint is incomplete");
    require(page_cache_hint("MemFree: 100 kB\nCached: 1000 kB\n").empty(), "a small cache was reported");
}

// STRATA_EXCHANGE_ROTATE (upstream 1e6cec80): 32 rounds of two swaps through a GPU cache, with and without the
// rotation and with an unpinned copy (the rotation then stays off); the bytes, the residency, the RAM and device
// aliases and the accounting after each
void test_rotating_source(bool rotate, bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("XESTRATA_EXCHANGE_ROTATE", rotate ? "1" : "0");
#else
    setenv("XESTRATA_EXCHANGE_ROTATE", rotate ? "1" : "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // Keep expert 4 on the file tier to check that the fallback survives rotation.
    require(src.pin_cache_complement(cache, err, pin ? FileExpertSource::Pin::on : FileExpertSource::Pin::off, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(2, err), err);
    require(src.exchange_rotation() == (rotate && pin), "rotation activation/fallback wrong");
    require(src.reserve_exchanges(1, err), "smaller capacity rejected");
    if (rotate && pin) require(!src.reserve_exchanges(3, err), "live arena growth accepted");
    int incoming[2] = {0, 1}, outgoing[2] = {2, 3};
    std::vector<uint8_t> actual(bytes);
    double commit_ms = 0;
    for (int round = 0; round < 32; ++round) {
        const uint8_t* prior[2]{};
        uint8_t* eviction[2]{};
        for (int q = 0; q < 2; ++q) {
            prior[q] = src.blob(0, incoming[q]); eviction[q] = src.exchange_buffer(q);
            require(prior[q] && !std::memcmp(prior[q], truth[incoming[q]].data(), bytes), "wrong incoming bytes");
            require(strata::gpu::copy(eviction[q], cache.device_slot(q), bytes), "D2H eviction failed");
            require(src.stage_exchange(0, incoming[q], outgoing[q], q), "stage rejected");
            require(!std::memcmp(src.blob(0, outgoing[q]), truth[outgoing[q]].data(), bytes), "staged override wrong");
            require(cache.fill_slot_blocking(q, prior[q], err), err); // H2D completed before commit
        }
        require(!src.stage_exchange(0, incoming[0], outgoing[0], 0), "duplicate stage accepted");
        require(!src.stage_exchange(0, incoming[0], 5, 0), "invalid expert accepted");
        const auto t0 = std::chrono::steady_clock::now();
        require(src.commit_exchanges() == 2, "commit count wrong");
        commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        require(src.commit_exchanges() == 0, "empty commit not empty");
        for (int q = 0; q < 2; ++q) {
            require(!src.has_resident(0, incoming[q]) && src.has_resident(0, outgoing[q]), "residency wrong");
            const uint8_t* held = src.blob(0, outgoing[q]);
            require(!std::memcmp(held, truth[outgoing[q]].data(), bytes), "evicted bytes corrupted");
            require(held == ((rotate && pin) ? eviction[q] : prior[q]), "wrong storage selected");
            if (rotate && pin) require(src.exchange_buffer(q) == prior[q], "old input not recycled");
            require(src.copy_blob(0, outgoing[q], actual.data()) && actual == truth[outgoing[q]], "copy_blob mismatch");
            require(!src.transient(0, outgoing[q]), "resident became transient");
            if (pin) {
                require(src.pinned(0, outgoing[q]) && src.device_alias(0, outgoing[q]), "mapping lost");
                require(strata::gpu::copy(actual.data(), src.device_alias(0, outgoing[q]), bytes),
                        "device alias read failed");
                require(actual == truth[outgoing[q]], "device alias points at stale expert");
            }
            require(cache.verify_slot(q, truth[incoming[q]].data(), err), err);
            std::swap(incoming[q], outgoing[q]);
        }
        require(!src.has_resident(0, 4) && src.copy_blob(0, 4, actual.data()) && actual == truth[4], "file fallback changed");
    }
    require(src.exchanges() == 64, "exchange accounting wrong");
    require(src.rotated_exchanges() == ((rotate && pin) ? 64u : 0u), "rotation accounting wrong");
    require(src.avoided_exchange_copy_bytes() == ((rotate && pin) ? 64 * bytes : 0u), "avoided bytes wrong");
    std::cout << "rotation integration: rotate=" << rotate << " pinned=" << pin
              << " exchanges=64 commit_ms=" << commit_ms << " exact_bytes=PASS\n";
    src.close();
    require(!src.exchange_rotation() && !src.exchange_buffer(0), "close retained storage");
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(!src.has_resident(0, 0) && !src.exchange_rotation(), "reopen retained residency");
}

void test_replace_resident(bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("XESTRATA_EXCHANGE_ROTATE", "0");
#else
    setenv("XESTRATA_EXCHANGE_ROTATE", "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // experts 2 and 3 are in VRAM, the RAM budget holds 0 and 1, expert 4 stays on the file tier
    require(src.pin_cache_complement(cache, err, pin ? FileExpertSource::Pin::on : FileExpertSource::Pin::off, {}, -1, 0,
                                     2 * bytes, &rank), err);
    require(src.reserve_exchanges(1, err), err);
    require(src.has_resident(0, 0) && src.has_resident(0, 1) && !src.has_resident(0, 4), "fixture residency wrong");
    std::vector<uint8_t> actual(bytes);
    auto holds = [&](int e) {
        const uint8_t* b = src.blob(0, e);
        return b != nullptr && !std::memcmp(b, truth[(size_t)e].data(), bytes);
    };
    require(!src.replace_resident(0, 1, 0), "an expert already in RAM accepted");
    require(!src.replace_resident(0, 4, 2), "a place the RAM does not hold accepted");
    require(!src.replace_resident(0, 4, 4) && !src.replace_resident(0, 5, 0) && !src.replace_resident(1, 4, 0),
            "invalid arguments accepted");
    require(src.ram_adapts() == 0 && holds(0) && holds(1) && holds(4), "a refused call changed the copy");
    // a staged exchange owns the tables until it commits
    require(strata::gpu::copy(src.exchange_buffer(0), cache.device_slot(0), bytes), "D2H eviction failed");
    require(src.stage_exchange(0, 0, 2, 0), "stage rejected");
    require(!src.replace_resident(0, 4, 1), "accepted beside a staged exchange");
    require(cache.fill_slot_blocking(0, truth[0].data(), err), err);
    require(src.commit_exchanges() == 1, "commit count wrong");   // RAM: 2 (in 0's place) and 1; VRAM: 0 and 3
    const uint8_t* place = src.blob(0, 1);
    require(src.replace_resident(0, 4, 1), "replace rejected");
    require(src.ram_adapts() == 1 && src.has_resident(0, 4) && !src.has_resident(0, 1), "residency wrong");
    require(src.blob(0, 4) == place && holds(4), "the incoming expert is not in the outgoing one's place");
    require(holds(1) && holds(2), "another expert's bytes changed");
    require(src.copy_blob(0, 1, actual.data()) && actual == truth[1], "the outgoing expert is not read from the file");
    require(!src.transient(0, 4), "resident became transient");
    if (pin) {
        require(src.pinned(0, 4) && src.device_alias(0, 4), "mapping lost");
        require(strata::gpu::copy(actual.data(), src.device_alias(0, 4), bytes), "device alias read failed");
        require(actual == truth[4], "the device alias points at the old expert");
    }
    // and back, then an exchange with the VRAM tier still works on the moved place
    require(src.replace_resident(0, 1, 4) && holds(1) && src.blob(0, 1) == place && !src.has_resident(0, 4),
            "replace back wrong");
    require(strata::gpu::copy(src.exchange_buffer(0), cache.device_slot(1), bytes), "D2H eviction failed");
    require(src.stage_exchange(0, 1, 3, 0), "stage after replace rejected");
    require(cache.fill_slot_blocking(1, truth[1].data(), err), err);
    require(src.commit_exchanges() == 1 && src.has_resident(0, 3) && holds(3) && !src.has_resident(0, 1),
            "exchange after replace wrong");
    require(src.ram_adapts() == 2, "accounting wrong");
    std::cout << "replace_resident: pinned=" << pin << " exact_bytes=PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--rotation-gpu") {
            test_rotating_source(false, true);
            test_rotating_source(true, true);
            test_rotating_source(true, false);
            test_replace_resident(true);
            test_replace_resident(false);
            std::cout << "file_expert_source_test --rotation-gpu: PASS\n";
            return 0;
        }
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_wait_for_ram();
        test_cgroup_memory_budget();
        test_page_cache_hint();
        test_canonical_layout();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
