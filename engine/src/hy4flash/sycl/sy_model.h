// sy_model.h - residency of the Hy4-Flash trunk (Section 1) on the Arc GPU, in slabs of < 4 GiB (Level Zero max single allocation on the A750 is 4 GiB).
// Tensors never straddle a slab. S4_K / S8_K weight matrices are repacked to the RP layout (sy_rp.h) on the host while loading; everything else is copied as is.
// The routed experts (Section 4) stay in host memory and are computed on the CPU (hy4s_cpuexp.h).
#pragma once
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hy4_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#include "sy_common.h"
#include "sy_rp.h"

namespace hy4s {

inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct SyModel {
    hy4::Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    struct Slab { uint8_t* d = nullptr; uint64_t bytes = 0; };
    std::vector<Slab> slabs;
    std::unordered_map<std::string, const uint8_t*> ptr;
    std::unordered_set<std::string> rp;  // tensors stored in the RP layout
    std::vector<uint8_t> seed;           // Section 2 (engram hot seed), host
    uint64_t trunk_bytes = 0, dev_bytes_total = 0;
    double t_load = 0;
    bool use_rp = true;

    const uint8_t* tw(const std::string& name) const {
        auto it = ptr.find(name);
        if (it == ptr.end()) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            exit(2);
        }
        return it->second;
    }
    bool is_rp(const std::string& name) const { return rp.count(name) != 0; }
    const hy4::TInfo& ti(const std::string& name) const { return man.get(name); }

    static bool wants_rp(const hy4::TInfo& t) {
        if (t.shape.size() != 2) return false;
        if (t.name.find("kv_b_proj") != std::string::npos || t.name.find("embed_tokens") != std::string::npos) return false;  // consumed raw (split at init / row gather)
        return rp_supported(t.fmt, (int)t.cols());
    }

    // slab_mb: target slab size (must stay < 4096 MiB incl. one tensor)
    bool load(const std::string& model_path, std::string& err, size_t slab_mb = 1400) {
        path = model_path;
        double t0 = now_s();
        if (!rd.open(path, err)) return false;
        hy4::parse_manifest(rd.manifest_json(), man);
        const auto& h = rd.header();
        trunk_bytes = h.sections[1].byte_length;
        const uint64_t sec1 = h.sections[1].file_offset;
        std::vector<const hy4::TInfo*> ts;
        for (auto& kv : man.T) ts.push_back(&kv.second);
        std::sort(ts.begin(), ts.end(), [](const hy4::TInfo* a, const hy4::TInfo* b) { return a->off < b->off; });
        std::vector<char> isrp(ts.size());
        std::vector<uint64_t> dsz(ts.size());
        uint64_t max_raw = 0, max_out = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
            isrp[i] = use_rp && wants_rp(*ts[i]);
            const uint64_t db = isrp[i] ? rp_bytes(ts[i]->fmt, (int)ts[i]->rows(), (int)ts[i]->cols()) : ts[i]->bytes;
            dsz[i] = (db + 255) & ~255ull;
            max_raw = std::max<uint64_t>(max_raw, ts[i]->bytes);
            max_out = std::max<uint64_t>(max_out, dsz[i]);
        }
        const uint64_t slab_max = (uint64_t)slab_mb << 20;
        std::vector<std::pair<size_t, size_t>> groups;  // [first, last] tensor index per slab
        size_t i0 = 0;
        uint64_t acc = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
            if (acc + dsz[i] > slab_max && i > i0) {
                groups.push_back({i0, i - 1});
                i0 = i;
                acc = 0;
            }
            acc += dsz[i];
        }
        if (!ts.empty()) groups.push_back({i0, ts.size() - 1});
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open failed"; return false; }
        uint8_t* raw = halloc<uint8_t>(max_raw + 4096);
        uint8_t* out = halloc<uint8_t>(max_out + 4096);
        for (auto& g : groups) {
            uint64_t sz = 0;
            for (size_t i = g.first; i <= g.second; ++i) sz += dsz[i];
            Slab s;
            s.bytes = sz;
            s.d = dalloc<uint8_t>(sz);
            uint64_t pos = 0;
            for (size_t i = g.first; i <= g.second; ++i) {
                const hy4::TInfo& t = *ts[i];
                size_t got = 0;
                while (got < t.bytes) {
                    ssize_t r = pread(fd, raw + got, t.bytes - got, (off_t)(sec1 + t.off + got));
                    if (r <= 0) { err = "short read of trunk tensor " + t.name; return false; }
                    got += (size_t)r;
                }
                posix_fadvise(fd, (off_t)(sec1 + t.off), (off_t)t.bytes, POSIX_FADV_DONTNEED);
                if (isrp[i]) {
                    rp_repack_host(t.fmt, raw, out, (int)t.rows(), (int)t.cols());
                    Q().memcpy(s.d + pos, out, rp_bytes(t.fmt, (int)t.rows(), (int)t.cols())).wait();
                    rp.insert(t.name);
                } else {
                    Q().memcpy(s.d + pos, raw, t.bytes).wait();
                }
                ptr[t.name] = s.d + pos;
                pos += dsz[i];
            }
            slabs.push_back(s);
            dev_bytes_total += sz;
        }
        ::close(fd);
        sycl::free(raw, Q());
        sycl::free(out, Q());
        if (!rd.read_section(2, seed, err)) return false;
        t_load = now_s() - t0;
        fprintf(stderr, "[load] trunk %.2f GB file -> %.2f GB device in %zu slabs, %zu tensors (%zu repacked), %.1f s\n", trunk_bytes / 1e9, dev_bytes_total / 1e9, slabs.size(), man.T.size(), rp.size(), t_load);
        return true;
    }
};

}  // namespace hy4s
