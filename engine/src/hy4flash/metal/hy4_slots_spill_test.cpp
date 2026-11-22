// hy4_slots_spill_test.cpp - standalone unit tests for hy4_metal_slots.h and hy4_metal_spill.h (no GPU required).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hy4_metal_slots.h"
#include "hy4_metal_spill.h"

struct FakeBuf {
    std::vector<uint8_t> storage;
    void* ptr = nullptr;
    size_t bytes = 0;

    FakeBuf() = default;
    explicit FakeBuf(size_t n) : storage(n, 0), ptr(storage.data()), bytes(n) {}
    FakeBuf(FakeBuf&& o) noexcept : storage(std::move(o.storage)), ptr(storage.data()), bytes(o.bytes) {
        o.ptr = nullptr;
        o.bytes = 0;
    }
    FakeBuf& operator=(FakeBuf&& o) noexcept {
        if (this != &o) {
            storage = std::move(o.storage);
            ptr = storage.data();
            bytes = o.bytes;
            o.ptr = nullptr;
            o.bytes = 0;
        }
        return *this;
    }
    FakeBuf(const FakeBuf&) = delete;
    FakeBuf& operator=(const FakeBuf&) = delete;
};

static HyKvSet<FakeBuf> make_set(const std::vector<char>& is_swa, size_t swa_bytes, int max_ctx, uint8_t seed) {
    HyKvSet<FakeBuf> s;
    s.kc.reserve(is_swa.size());
    s.iv.reserve(is_swa.size());
    s.pl.reserve(is_swa.size());
    for (size_t il = 0; il < is_swa.size(); ++il) {
        size_t kc_b = is_swa[il] ? swa_bytes : (size_t) (max_ctx + 8) * 1152;
        FakeBuf kc(kc_b);
        for (size_t i = 0; i < kc_b; ++i) kc.storage[i] = (uint8_t) ((seed + il * 17 + i * 31) & 0xff);
        s.kc.push_back(std::move(kc));
        if (!is_swa[il]) {
            size_t iv_b = (size_t) (max_ctx + 8) * 512;
            size_t pl_b = (size_t) (max_ctx / 4 + 8) * 256;
            FakeBuf iv(iv_b), pl(pl_b);
            for (size_t i = 0; i < iv_b; ++i) iv.storage[i] = (uint8_t) ((seed + il * 23 + i * 13) & 0xff);
            for (size_t i = 0; i < pl_b; ++i) pl.storage[i] = (uint8_t) ((seed + il * 29 + i * 7) & 0xff);
            s.iv.push_back(std::move(iv));
            s.pl.push_back(std::move(pl));
        } else {
            s.iv.emplace_back();
            s.pl.emplace_back();
        }
    }
    return s;
}

static std::vector<int32_t> make_seq(int n, int base) {
    std::vector<int32_t> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = base + i;
    return v;
}

static void test_slots() {
    // 1. hy4_reuse_start basics
    auto a = make_seq(200, 1000);
    auto b = make_seq(200, 1000);
    // exact match capped at T0 - 1
    assert(hy4_reuse_start(a, 200, b, 200, 512) == 199);
    // diverges at index 120
    b[120] = 999999;
    assert(hy4_reuse_start(a, 200, b, 200, 512) == 120);
    // rollback too large for ring slack: fwd + 64 - L = 200 + 64 - 120 = 144 > 100
    assert(hy4_reuse_start(a, 200, b, 200, 100) == 0);
    assert(hy4_reuse_start(a, 200, b, 200, 144) == 120);
    // no common prefix
    auto c = make_seq(50, 5000);
    assert(hy4_reuse_start(a, 200, c, 50, 512) == 0);

    // 2. hy4_pick_slot: single slot mode (plain LCP, no floor)
    std::vector<HyPickIn> s1 = {{&a, 200, 200, 1}};
    int start = -1;
    assert(hy4_pick_slot(s1, 0, b, 200, 1024, 1, 1024, false, start) == 0);
    assert(start == 120);
    // no_reuse forces start = 0
    assert(hy4_pick_slot(s1, 0, b, 200, 1024, 1, 1024, true, start) == 0);
    assert(start == 0);

    // 3. Multi-slot: longest prefix wins, new slot allocated when below floor, LRU evicted when full
    auto sA = make_seq(2000, 10000);
    auto sB = make_seq(2000, 20000);
    auto qA = sA;
    qA.push_back(777);
    auto qNew = make_seq(2000, 30000);
    std::vector<HyPickIn> ms = {{&sA, 2000, 2000, 1}, {&sB, 2000, 2000, 2}};
    // Match slot 0
    assert(hy4_pick_slot(ms, 1, qA, (int) qA.size(), 8192, 3, 1024, false, start) == 0);
    assert(start == 2000);
    // New unrelated prompt allocates slot 2 (n < max_slots=3)
    assert(hy4_pick_slot(ms, 1, qNew, (int) qNew.size(), 8192, 3, 1024, false, start) == 2);
    assert(start == 0);
    // When full (max_slots=2), evicts LRU (slot 0, last_used=1 < 2)
    assert(hy4_pick_slot(ms, 1, qNew, (int) qNew.size(), 8192, 2, 1024, false, start) == 0);
    assert(start == 0);
    // Free slot preferred over allocating a new one
    std::vector<HyPickIn> ms_free = {{&sA, 2000, 2000, 5}, {&sB, 0, 0, 1}};
    assert(hy4_pick_slot(ms_free, 0, qNew, (int) qNew.size(), 8192, 4, 1024, false, start) == 1);
    assert(start == 0);
    // hw > fwd tightens effective slack
    std::vector<HyPickIn> ms_hw = {{&sA, 2000, 2500, 1}};
    // slack = 600 - (2500-2000) = 100; fwd + 64 - 1900 = 164 > 100 -> rejected
    auto qShort = sA;
    qShort[1900] = 88888;
    assert(hy4_pick_slot(ms_hw, 0, qShort, 2000, 600, 2, 1024, false, start) == 1);
    assert(start == 0);
}

static void test_spill() {
    // 1. hy4_spill_root
    assert(hy4_spill_root("/custom/dir", "/a/models/m.strata", false) == "/custom/dir");
    assert(hy4_spill_root("", "/data/strata/models/Hy4.strata", true) == "/data/strata/conversation-spill");
    assert(hy4_spill_root("", "/data/strata/models/Hy4.strata", false).empty());
    assert(hy4_spill_root("", "/tmp/Hy4.strata", true).empty());

    // 2. hy4_spill_fp sensitivity
    uint64_t fp1 = hy4_spill_fp("/nonexistent/Hy4.strata", "", 4096, 1024, 4112, 4);
    uint64_t fp2 = hy4_spill_fp("/nonexistent/Hy4.strata", "", 8192, 1024, 4112, 4);
    assert(fp1 != fp2);

    const std::string root = "/tmp/hy4_slots_spill_test_" + std::to_string((long) getpid());
    hy4_rm_dir(root);
    hy4_mkdirs(root);

    // Create a fake dead-pid sibling dir to verify hy4_spill_clean_dead
    const std::string stale = root + "/hy4-0000000000000001.p999999";
    hy4_mkdirs(stale);

    const std::vector<char> is_swa = {1, 0, 1, 0};
    const int max_ctx = 64;
    const size_t swa_bytes = 32 * 1152;
    const uint64_t fp = fp1;

    // 3. Submit, drain, best, load & byte-exact verify (persist = true)
    auto tok1 = make_seq(32, 100);
    auto tok2 = make_seq(40, 200);
    auto ref1 = make_set(is_swa, swa_bytes, max_ctx, 11);
    auto ref2 = make_set(is_swa, swa_bytes, max_ctx, 22);
    std::string persist_dir;
    {
        HySpill<FakeBuf> sp;
        assert(sp.init(root, fp, true, 64, is_swa, swa_bytes));
        persist_dir = sp.dir();
        // Stale dir cleaned
        struct stat st;
        assert(stat(stale.c_str(), &st) != 0);

        auto s1 = make_set(is_swa, swa_bytes, max_ctx, 11);
        assert(sp.submit(s1, tok1, 32, 32));
        // Duplicate/prefix submit returns false
        auto s1_dup = make_set(is_swa, swa_bytes, max_ctx, 11);
        assert(!sp.submit(s1_dup, tok1, 24, 24));
        sp.drain();
        assert(sp.entries() == 1);

        // Free set returned by writer
        HyKvSet<FakeBuf> recycled;
        assert(sp.take_free(recycled));

        // Load back and verify payload bytes match ref1
        HySpill<FakeBuf>::Cand cand;
        auto q1 = tok1;
        q1.push_back(999);
        assert(sp.best(q1, (int) q1.size(), 1024, cand));
        assert(cand.start == 32);

        auto dst = make_set(is_swa, swa_bytes, max_ctx, 0);
        std::vector<int32_t> lt;
        int lf = 0, lh = 0;
        assert(sp.load(cand.seq, dst, lt, lf, lh));
        assert(lf == 32 && lh == 32 && lt == tok1);
        for (size_t il = 0; il < is_swa.size(); ++il) {
            if (is_swa[il]) {
                assert(std::memcmp(dst.kc[il].ptr, ref1.kc[il].ptr, swa_bytes) == 0);
            } else {
                assert(std::memcmp(dst.kc[il].ptr, ref1.kc[il].ptr, 32 * 1152) == 0);
                assert(std::memcmp(dst.iv[il].ptr, ref1.iv[il].ptr, 32 * 512) == 0);
                assert(std::memcmp(dst.pl[il].ptr, ref1.pl[il].ptr, (32 / 4) * 256) == 0);
            }
        }

        // write_now for a second conversation before shutdown
        auto s2 = make_set(is_swa, swa_bytes, max_ctx, 22);
        assert(sp.write_now(s2, tok2, 40, 44));
        assert(sp.entries() == 2);
        sp.shutdown();
    }

    // 4. Restart with persist=true: adopts both entries
    {
        HySpill<FakeBuf> sp2;
        assert(sp2.init(root, fp, true, 64, is_swa, swa_bytes));
        assert(sp2.adopted == 2);
        HySpill<FakeBuf>::Cand cand;
        auto q2 = tok2;
        q2.push_back(777);
        assert(sp2.best(q2, (int) q2.size(), 1024, cand));
        assert(cand.start == 40);
        auto dst = make_set(is_swa, swa_bytes, max_ctx, 0);
        std::vector<int32_t> lt;
        int lf = 0, lh = 0;
        assert(sp2.load(cand.seq, dst, lt, lf, lh));
        assert(lf == 40 && lh == 44 && lt == tok2);
        for (size_t il = 0; il < is_swa.size(); ++il) {
            if (!is_swa[il]) assert(std::memcmp(dst.kc[il].ptr, ref2.kc[il].ptr, 40 * 1152) == 0);
        }
        sp2.shutdown();
    }

    // 5. Ephemeral mode (persist=false) deletes its directory on shutdown
    std::string eph_dir;
    {
        HySpill<FakeBuf> sp3;
        assert(sp3.init(root, fp, false, 64, is_swa, swa_bytes));
        eph_dir = sp3.dir();
        auto s1 = make_set(is_swa, swa_bytes, max_ctx, 33);
        assert(sp3.submit(s1, tok1, 32, 32));
        sp3.drain();
        sp3.shutdown();
    }
    struct stat st;
    assert(stat(eph_dir.c_str(), &st) != 0);

    // Cleanup
    hy4_rm_dir(persist_dir);
    hy4_rm_dir(root);
}

int main() {
    test_slots();
    test_spill();
    printf("[hy4_slots_spill_test] ALL PASS\n");
    return 0;
}
