// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/core/latent_pack_test.cpp - the latent MoE pack's readers (strata/core/latent.hpp), without a model or a GPU.
//
//   1. native_experts.txt: the published Edge0 manifests (v6, their header lines verbatim) parse to the
//      widths, types, offsets and blob sizes the packs have; a v3 header without expert_embd reads 0;
//      versions and rows the reader does not know are refused.
//   2. labels.bin: exact size, the group range, refusals.
//   3. scales/l*.f32: all-or-none, the reciprocals, a non-positive scale reads as 1.
//   4. LatentDense: the Q8_0 / Q4_0 layouts, the layer offsets, the format from the environment, the size
//      refusals (both candidate sizes named), and the published packs' file sizes from the layout alone.
//   5. LatentA: the blob offsets and the size refusal.
//   6. The blob and row layouts against ggml itself: the types' row sizes, one expert's section offsets at
//      both latent widths, and the decoders on crafted blocks (Q8_0 / Q4_0 exact, Q3_K / IQ4_NL finite).
#include "strata/core/latent.hpp"

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() : path(fs::temp_directory_path() /
               ("strata-latent-pack-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_text(const fs::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }
void write_bytes(const fs::path& p, const std::vector<uint8_t>& b) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*) b.data(), (std::streamsize) b.size());
}

void env_set(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// ---- 1. the manifest
void manifest() {
    // the published Edge0 manifests' header lines, verbatim (their rows are layer gu_type d_type offset blob_bytes,
    // the offset layer * 512 * blob; the real files are the published Edge0 HF manifests)
    struct Want { const char* head; int64_t embd, blob; };
    const Want wants[] = {
        {"# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)",
         1792, 1630720},
        {"# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1280, expert_ff 640)",
         1280, 1164800},
    };
    TempDir d;
    for (const Want& w : wants) {
        std::string text = std::string(w.head) + "\n";
        for (int l = 0; l < 48; ++l)
            text += std::to_string(l) + " 11 20 " + std::to_string((long long) l * 512 * w.blob) + " " +
                    std::to_string(w.blob) + "\n";
        write_text(d.path / "edge0.txt", text);
        LatentManifest m;
        std::string err;
        const bool ok = latent_manifest_read((d.path / "edge0.txt").string(), m, err);
        check(ok, std::string("Edge0 manifest (ELL ") + std::to_string(w.embd) + "): reads" + (ok ? "" : ": " + err));
        if (!ok) continue;
        bool same = m.version == 6 && m.expert_embd == w.embd && m.expert_ff == 640 && m.layers.size() == 48;
        for (size_t l = 0; l < m.layers.size(); ++l) {
            const LatentLayerFmt& f = m.layers[l];
            same = same && f.gu_type == 11 && f.d_type == 20 && f.blob_bytes == (uint64_t) w.blob &&
                   f.offset == (uint64_t) l * 512 * (uint64_t) w.blob;
        }
        check(same, std::string("Edge0 manifest: v6, expert_embd ") + std::to_string(w.embd) +
                        ", 48 layers of Q3_K/IQ4_NL, blob " + std::to_string(w.blob));
    }
    LatentManifest m;
    std::string err;
    write_text(d.path / "v3.txt", "# strata native experts v3: layer gu_type d_type offset blob_bytes\n"
                                  "0 12 8 0 100\n1 12 8 100 100\n");
    check(latent_manifest_read((d.path / "v3.txt").string(), m, err) && m.version == 3 && m.expert_embd == 0 &&
              m.layers.size() == 2 && m.layers[1].offset == 100,
          "v3 without expert_embd: reads, the width stays 0");
    write_text(d.path / "ne.txt", "# strata native experts v3: (n_expert 512, total 1; x)\n0 11 20 0 100\n");
    check(latent_manifest_read((d.path / "ne.txt").string(), m, err) && m.n_expert == 512,
          "the header's (n_expert N) reads");
    const char* bad[] = {
        "# strata native experts v2: x\n0 12 8 0 100\n",
        "# strata native experts v7: x\n0 12 8 0 100\n",
        "# xestrata native experts xs1: x\n0 12 8 0 100\n",
        "# strata native experts v6: x\n1 11 20 0 100\n",
        "# strata native experts v6: x\n0 11 20 0 0\n",
        "# strata native experts v6: x\n0 11 20\n",
    };
    const char* names[] = {"v2 refused", "v7 refused", "xs1 refused (expert_layout_load's)", "layer out of order refused",
                           "zero blob refused", "short row refused"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        write_text(d.path / "bad.txt", bad[i]);
        check(!latent_manifest_read((d.path / "bad.txt").string(), m, err), names[i]);
    }
}

// ---- 2. labels
void labels() {
    TempDir d;
    std::vector<int32_t> got;
    std::string err;
    std::vector<int32_t> want(64);
    for (size_t i = 0; i < want.size(); ++i) want[i] = (int32_t) (i % 8);
    write_bytes(d.path / "labels.bin", std::vector<uint8_t>(reinterpret_cast<uint8_t*>(want.data()), reinterpret_cast<uint8_t*>(want.data() + want.size())));
    check(latent_labels_read((d.path / "labels.bin").string(), 2, 32, 8, got, err) && got == want,
          "labels: 2 layers x 32 experts read back");
    want[37] = 8;
    write_bytes(d.path / "labels.bin", std::vector<uint8_t>(reinterpret_cast<uint8_t*>(want.data()), reinterpret_cast<uint8_t*>(want.data() + want.size())));
    check(!latent_labels_read((d.path / "labels.bin").string(), 2, 32, 8, got, err), "label 8 of 0..7 refused");
    want[37] = -1;
    write_bytes(d.path / "labels.bin", std::vector<uint8_t>(reinterpret_cast<uint8_t*>(want.data()), reinterpret_cast<uint8_t*>(want.data() + want.size())));
    check(!latent_labels_read((d.path / "labels.bin").string(), 2, 32, 8, got, err), "label -1 refused");
    write_bytes(d.path / "short.bin", std::vector<uint8_t>(252, 0));
    check(!latent_labels_read((d.path / "short.bin").string(), 2, 32, 8, got, err), "a short file refused");
    write_bytes(d.path / "long.bin", std::vector<uint8_t>(260, 0));
    check(!latent_labels_read((d.path / "long.bin").string(), 2, 32, 8, got, err), "a long file refused");
}

// ---- 3. the column scales
void colscales() {
    TempDir d;
    std::vector<float> inv;
    std::string err;
    check(latent_colscale_read(d.path.string(), 2, 4, inv, err) && inv.empty(), "no scales dir: empty, not an error");
    fs::create_directories(d.path / "scales");
    const float s0[4] = {2.0f, 0.5f, 1.0f, 4.0f}, s1[4] = {1.0f, 3.0f, 0.0f, -2.0f};
    write_bytes(d.path / "scales/l0.f32", std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s0), reinterpret_cast<const uint8_t*>(s0 + 4)));
    check(!latent_colscale_read(d.path.string(), 2, 4, inv, err), "l0 without l1 refused (all-or-none)");
    write_bytes(d.path / "scales/l1.f32", std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s1), reinterpret_cast<const uint8_t*>(s1 + 4)));
    const bool ok = latent_colscale_read(d.path.string(), 2, 4, inv, err);
    check(ok && inv.size() == 8 && inv[0] == 0.5f && inv[1] == 2.0f && inv[2] == 1.0f && inv[3] == 0.25f &&
              inv[4] == 1.0f && inv[5] == 1.0f / 3.0f && inv[6] == 1.0f && inv[7] == 1.0f,
          "the reciprocals, layer-major; a scale <= 0 reads as 1");
    write_bytes(d.path / "scales/l1.f32", std::vector<uint8_t>(12, 0));
    check(!latent_colscale_read(d.path.string(), 2, 4, inv, err), "a short scale file refused");
}

// ---- 4. the D/U file
void dense() {
    TempDir d;
    std::string err;
    // the published packs' sizes from the layout alone (the files HF serves, verified at fetch)
    check(LatentDense::bytes_for(48, 8, 2560, 1792, 8) == 3743416320ull &&
              LatentDense::bytes_for(48, 8, 2560, 1792, 4) == 1981808640ull &&
              LatentDense::bytes_for(48, 8, 2560, 1280, 8) == 2673868800ull &&
              LatentDense::bytes_for(48, 8, 2560, 1280, 4) == 1415577600ull,
          "the four published D/U files' sizes from the layout");
    const int64_t L = 2, G = 2, H = 256, ELL = 128;
    const size_t db = (size_t) G * ELL * (H / 32) * 34, ub = (size_t) G * H * (ELL / 32) * 34;
    std::vector<uint8_t> file((size_t) L * (db + ub));
    for (size_t i = 0; i < file.size(); ++i) file[i] = (uint8_t) (i * 7 + 3);
    write_bytes(d.path / "du8.bin", file);
    env_set("XESTRATA_LATENT_DENSE_FMT", nullptr);
    LatentDense du;
    bool ok = du.open((d.path / "du8.bin").string(), L, G, H, ELL, err);
    check(ok && du.fmt() == 8 && du.bytes() == file.size(),
          "the Q8_0 file opens at fmt 8 by default (auto: its size)" + (ok ? "" : ": " + err));
    if (ok) {
        bool same = true;
        for (int64_t l = 0; l < L; ++l) {
            same = same && du.u(l) - du.d(l) == (ptrdiff_t) db &&
                   std::memcmp(du.d(l), file.data() + (size_t) l * (db + ub), 64) == 0 &&
                   std::memcmp(du.u(l), file.data() + (size_t) l * (db + ub) + db, 64) == 0;
            if (l + 1 < L) same = same && du.d(l + 1) - du.d(l) == (ptrdiff_t) (db + ub);
        }
        same = same && du.d(-1) == nullptr && du.d(L) == nullptr && du.u(L) == nullptr;
        check(same, "the layers' D then U, in place; out-of-range layers are null");
        check(du.d_row_bytes() == (H / 32) * 34 && du.u_row_bytes() == (ELL / 32) * 34, "the row bytes");
    }
    du.close();
    check(!du.mapped() && du.d(0) == nullptr, "close unmaps");
    env_set("XESTRATA_LATENT_DENSE_FMT", "4");
    check(!du.open((d.path / "du8.bin").string(), L, G, H, ELL, err) && err.find("the other fmt wants") != std::string::npos,
          "fmt 4 against the Q8_0 file refused, both sizes named");
    const size_t db4 = (size_t) G * ELL * (H / 32) * 18, ub4 = (size_t) G * H * (ELL / 32) * 18;
    write_bytes(d.path / "du4.bin", std::vector<uint8_t>((size_t) L * (db4 + ub4), 0x5A));
    ok = du.open((d.path / "du4.bin").string(), L, G, H, ELL, err);
    check(ok && du.fmt() == 4 && du.d_bytes() == db4 && du.u_bytes() == ub4, "Q4_0 opens at fmt 4");
    env_set("XESTRATA_LATENT_DENSE_FMT", "8");
    check(!du.open((d.path / "du4.bin").string(), L, G, H, ELL, err) &&
              err.find("the other fmt wants") != std::string::npos,
          "fmt 8 against the Q4_0 file refused, both sizes named");
    env_set("XESTRATA_LATENT_DENSE_FMT", "9");
    check(!du.open((d.path / "du4.bin").string(), L, G, H, ELL, err), "fmt 9 refused");
    // auto (unset, or `auto`): the format the file's size is
    for (const char* v : {(const char*) nullptr, "auto"}) {
        env_set("XESTRATA_LATENT_DENSE_FMT", v);
        const std::string how = v == nullptr ? "by default" : "at auto";
        ok = du.open((d.path / "du4.bin").string(), L, G, H, ELL, err);
        check(ok && du.fmt() == 4 && du.d_bytes() == db4, "the Q4_0 file opens at fmt 4 " + how);
        ok = du.open((d.path / "du8.bin").string(), L, G, H, ELL, err);
        check(ok && du.fmt() == 8 && du.d_bytes() == db, "the Q8_0 file opens at fmt 8 " + how);
    }
    env_set("XESTRATA_LATENT_DENSE_FMT", nullptr);
    write_bytes(d.path / "du_odd.bin", std::vector<uint8_t>((size_t) L * (db4 + ub4) + 1, 0x5A));
    check(!du.open((d.path / "du_odd.bin").string(), L, G, H, ELL, err), "a file of neither format's size refused");
    check(!du.open((d.path / "du8.bin").string(), L, G, H, 127, err), "a latent width that is not whole blocks refused");
    check(!du.open((d.path / "missing.bin").string(), L, G, H, ELL, err), "a missing file refused");
}

// ---- 5. the A file
void experts_a() {
    TempDir d;
    std::string err;
    const int64_t L = 2, E = 4, B = 1024;
    std::vector<uint8_t> file((size_t) L * E * B);
    for (size_t i = 0; i < file.size(); ++i) file[i] = (uint8_t) (i % 251);
    write_bytes(d.path / "a.bin", file);
    LatentA a;
    const bool ok = a.open((d.path / "a.bin").string(), L, E, B, err);
    check(ok && a.blob_bytes() == B, "the A file opens" + (ok ? "" : ": " + err));
    if (ok) {
        bool same = true;
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 0; e < E; ++e) {
                same = same && a.blob(l, e) - a.blob(0, 0) == (ptrdiff_t) (((size_t) l * E + (size_t) e) * B);
                same = same && std::memcmp(a.blob(l, e), file.data() + ((size_t) l * E + (size_t) e) * B, 64) == 0;
            }
        same = same && a.blob(-1, 0) == nullptr && a.blob(L, 0) == nullptr && a.blob(0, E) == nullptr;
        check(same, "the blobs layer-major, expert-major; out of range is null");
    }
    a.close();
    write_bytes(d.path / "short.bin", std::vector<uint8_t>((size_t) L * E * B - 1, 0));
    check(!a.open((d.path / "short.bin").string(), L, E, B, err), "a short A file refused");
}

// ---- 6. the layouts against ggml
void ggml_layouts() {
    check(ggml_blck_size(GGML_TYPE_Q3_K) == 256 && ggml_row_size(GGML_TYPE_Q3_K, 256) == 110 &&
              ggml_blck_size(GGML_TYPE_IQ4_NL) == 32 && ggml_row_size(GGML_TYPE_IQ4_NL, 32) == 18 &&
              ggml_row_size(GGML_TYPE_Q8_0, 32) == 34 && ggml_row_size(GGML_TYPE_Q4_0, 32) == 18,
          "the block sizes and row bytes the readers assume are ggml's");
    // one expert's sections at both latent widths (native_expert_layout(11, 20, ELL, 640)'s arithmetic)
    const int64_t I = 640;
    for (int64_t ELL : {1792, 1280}) {
        const size_t gu_row = ggml_row_size(GGML_TYPE_Q3_K, ELL);
        const size_t up_off = (size_t) I * gu_row;
        const size_t down_off = up_off + (size_t) I * gu_row;
        const size_t bytes = down_off + (size_t) ELL * ggml_row_size(GGML_TYPE_IQ4_NL, I);
        const size_t want = ELL == 1792 ? 1630720 : 1164800;
        check(bytes == want, "one expert's blob at ELL " + std::to_string(ELL) + ": " + std::to_string(bytes) + " B");
    }
    // the decoders on crafted blocks: Q8_0 and Q4_0 exact values (the centred codes the D/U readers assume),
    // Q3_K and IQ4_NL random blocks decode finite (any byte pattern is a valid block of theirs)
    std::mt19937 rng(20261113);   // NOLINT(bugprone-random-generator-seed): reproducible blocks
    std::uniform_int_distribution<int> byte(0, 255);
    {
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, 32));
        const ggml_fp16_t one = ggml_fp32_to_fp16(1.0f);
        std::memcpy(q.data(), &one, 2);
        for (int i = 0; i < 32; ++i) q[2 + i] = (uint8_t) (int8_t) (i - 16);
        std::vector<float> f(32);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), f.data(), 32);
        bool ok = true;
        for (int i = 0; i < 32; ++i) ok = ok && f[i] == (float) (i - 16);
        check(ok, "Q8_0: d=1, codes i-16 decode to i-16");
    }
    {
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q4_0, 32));
        const ggml_fp16_t one = ggml_fp32_to_fp16(1.0f);
        std::memcpy(q.data(), &one, 2);
        for (int j = 0; j < 16; ++j) q[2 + j] = (uint8_t) (j | (j << 4));
        std::vector<float> f(32);
        ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(q.data(), f.data(), 32);
        bool ok = true;
        for (int j = 0; j < 16; ++j) ok = ok && f[j] == (float) (j - 8) && f[j + 16] == (float) (j - 8);
        check(ok, "Q4_0: element j in byte j's low nibble, j+16 in its high, value (code-8)*d");
    }
    for (ggml_type t : {GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL}) {
        const int64_t n = 2560;
        std::vector<uint8_t> q(ggml_row_size(t, n));
        for (uint8_t& b : q) b = (uint8_t) byte(rng);
        for (int64_t b = 0; b < n / ggml_blck_size(t); ++b) {   // a finite scale in every block
            const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
            // Q3_K's d sits at the block's end (hmask 32 + qs 64 + scales 12, then d); IQ4_NL's at its start
            const size_t d_off = t == GGML_TYPE_Q3_K ? 108 : 0;
            std::memcpy(q.data() + (size_t) b * ggml_row_size(t, ggml_blck_size(t)) + d_off, &d, 2);
        }
        std::vector<float> f((size_t) n);
        ggml_get_type_traits(t)->to_float(q.data(), f.data(), n);
        bool ok = true;
        for (float v : f) ok = ok && std::isfinite(v);
        check(ok, std::string("ggml decodes random ") + ggml_type_name(t) + " blocks finite");
    }
}
}  // namespace

int main() {
    std::printf("latent_pack_test\n");
    manifest();
    labels();
    colscales();
    dense();
    experts_a();
    ggml_layouts();
    std::printf("latent_pack_test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}