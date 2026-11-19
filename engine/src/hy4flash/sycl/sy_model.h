// sy_model.h - residency of the Hy4-Flash trunk (Section 1) on the Arc GPU, in slabs of < 4 GiB (Level Zero max single allocation on the A750 is 4 GiB).
// Tensors never straddle a slab. The routed experts (Section 4) stay in host memory and are computed on the CPU (hy4s_cpuexp.h).
#pragma once
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <vector>

#include "hy4_manifest.h"
#include "strata/artifact/strata_reader.hpp"
#include "sy_common.h"

namespace hy4s {

inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct SyModel {
    hy4::Manifest man;
    strata::artifact::StrataFileReader rd;
    std::string path;
    struct Slab { uint8_t* d = nullptr; uint64_t start = 0, bytes = 0; };
    std::vector<Slab> slabs;
    std::unordered_map<std::string, const uint8_t*> ptr;
    std::vector<uint8_t> seed;  // Section 2 (engram hot seed), host
    uint64_t trunk_bytes = 0;
    double t_load = 0;

    const uint8_t* tw(const std::string& name) const {
        auto it = ptr.find(name);
        if (it == ptr.end()) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            exit(2);
        }
        return it->second;
    }
    const hy4::TInfo& ti(const std::string& name) const { return man.get(name); }

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
        const uint64_t slab_max = (uint64_t)slab_mb << 20;
        // slab layout: contiguous file ranges [start, start+bytes), cut at tensor boundaries
        std::vector<std::pair<size_t, size_t>> groups;  // [first tensor, last tensor] indices
        size_t i0 = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
            uint64_t end_i = ts[i]->off + ts[i]->bytes;
            if (end_i - ts[i0]->off > slab_max && i > i0) {
                groups.push_back({i0, i - 1});
                i0 = i;
            }
        }
        if (!ts.empty()) groups.push_back({i0, ts.size() - 1});
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open failed"; return false; }
        const size_t CH = 64ull << 20;
        uint8_t* stg[2] = {halloc<uint8_t>(CH), halloc<uint8_t>(CH)};
        sycl::event ev[2];
        bool have_ev[2] = {false, false};
        uint64_t total = 0;
        for (auto& g : groups) {
            const uint64_t start = ts[g.first]->off;
            uint64_t end = 0;
            for (size_t i = g.first; i <= g.second; ++i) end = std::max(end, ts[i]->off + ts[i]->bytes);
            Slab s;
            s.start = start;
            s.bytes = end - start;
            s.d = dalloc<uint8_t>(s.bytes);
            uint64_t done = 0;
            int b = 0;
            while (done < s.bytes) {
                size_t n = (size_t)std::min<uint64_t>(CH, s.bytes - done);
                if (have_ev[b]) ev[b].wait();
                size_t got = 0;
                while (got < n) {
                    ssize_t r = pread(fd, stg[b] + got, n - got, (off_t)(sec1 + start + done + got));
                    if (r <= 0) { err = "short read of trunk"; return false; }
                    got += (size_t)r;
                }
                posix_fadvise(fd, (off_t)(sec1 + start + done), (off_t)n, POSIX_FADV_DONTNEED);
                ev[b] = Q().memcpy(s.d + done, stg[b], n);
                have_ev[b] = true;
                done += n;
                b ^= 1;
            }
            for (size_t i = g.first; i <= g.second; ++i) ptr[ts[i]->name] = s.d + (ts[i]->off - start);
            slabs.push_back(s);
            total += s.bytes;
        }
        wait();
        ::close(fd);
        sycl::free(stg[0], Q());
        sycl::free(stg[1], Q());
        if (!rd.read_section(2, seed, err)) return false;
        t_load = now_s() - t0;
        fprintf(stderr, "[load] trunk %.2f GB in %zu slabs, %zu tensors, %.1f s\n", total / 1e9, slabs.size(), man.T.size(), t_load);
        return true;
    }
};

}  // namespace hy4s
