// #6075: two savers of the background checkpoint copy (STRATA_CKPT_ASYNC=1) on two threads, as a layer split's stages
// save their parts, forced into the order that broke the split: saver B passes its join, saver A takes the staging
// buffer and starts its copy, then B refills or resizes the buffer (and its pinned twin and stream) before A's copy
// has read it.  Each checkpoint must hold its own session's bytes, and every copy must succeed.
//
// Cases: the same size on one card (B's bytes would land in A's checkpoint), B larger on one card (B frees A's buffer
// and stream: "invalid resource handle"), and, with two cards, the same size on card 0 and card 1 (B frees A's buffer
// on A's card).  The order is forced through conversation_checkpoint_test_hook; a saver that is held back by the fix
// (it cannot pass while the other owns the buffer) makes the hook's wait time out after 1.5 s, and the test goes on.
// Without STRATA_CKPT_ASYNC=1, or without a card, the test is skipped (77).
#include "strata/core/conversation_snapshot.hpp"
#include "conversation_copy.hpp"
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& label) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label.c_str()); }
}
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA (%s): %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

struct Gate {
    std::mutex m;
    std::condition_variable cv;
    bool open_ = false;
    void open() { { std::lock_guard<std::mutex> lk(m); open_ = true; } cv.notify_all(); }
    bool wait() {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::milliseconds(1500), [&] { return open_; });
    }
    void reset() { std::lock_guard<std::mutex> lk(m); open_ = false; }
};
Gate b_at_stage, a_copying, b_filled;
std::atomic<int> copies{0};
std::atomic<int> held_back{0};   // hook waits that timed out: a saver the lock kept out (the fixed order)
thread_local int role = 0;       // 1 = saver A, 2 = saver B; 0 = the background copy threads

void hook(int point) {
    if (point == 1 && role == 2) {          // B has passed its join and is about to take the buffer
        b_at_stage.open();
        if (!a_copying.wait()) ++held_back;  // ... until A's copy runs (never, while B holds the buffer)
    } else if (point == 2 && copies.fetch_add(1) == 0) {   // the first background copy is about to read the buffer
        a_copying.open();
        if (!b_filled.wait()) ++held_back;  // ... after B has refilled it (base) or at once (B's own copy)
    } else if (point == 3 && role == 2) {
        b_filled.open();
    }
}

struct Session {
    int dev = 0;
    SessionState ss;
    size_t bytes = 0;
    std::vector<uint8_t> want;
    Session(const ModelGeometry& g, int device, int64_t rows, uint8_t salt) : dev(device) {
        cuda_check(cudaSetDevice(dev), "set device");
        ss.max_cells = 96; ss.layer_hi = g.n_layers;
        ss.gdn_alloc = rows; ss.qsa_alloc = 0; ss.qsa_ord0 = 0;   // GDN rows only: the part the background copy moves
        ConversationStateSizes z;
        std::string err;
        if (!conversation_session_sizes(g, ss, z, err)) { std::fprintf(stderr, "sizes: %s\n", err.c_str()); std::exit(1); }
        bytes = z.gdn;
        cuda_check(cudaMalloc(&ss.gdn_state, bytes), "session state");
        want.resize(bytes);
        for (size_t i = 0; i < bytes; ++i) want[i] = (uint8_t) (salt + i * 7 + i / 4093);
        cuda_check(cudaMemcpy(ss.gdn_state, want.data(), bytes, cudaMemcpyHostToDevice), "fill");
        cuda_check(cudaDeviceSynchronize(), "fill sync");
    }
    ~Session() { cudaSetDevice(dev); cudaFree(ss.gdn_state); }
};

void race(const ModelGeometry& g, const char* name, int dev_a, int64_t rows_a, int dev_b, int64_t rows_b) {
    b_at_stage.reset(); a_copying.reset(); b_filled.reset();
    copies = 0; held_back = 0;
    Session A(g, dev_a, rows_a, 11), B(g, dev_b, rows_b, 173);
    ConversationCheckpoint ca, cb;
    ca.ids = cb.ids = {1, 2, 3};
    bool ok_a = false, ok_b = false;
    std::string err_a, err_b;
    std::thread tb([&] {
        role = 2;
        cuda_check(cudaSetDevice(B.dev), "B device");
        ok_b = conversation_checkpoint_save(cb, B.ss, g, err_b);
    });
    b_at_stage.wait();   // B is past its join
    std::thread ta([&] {
        role = 1;
        cuda_check(cudaSetDevice(A.dev), "A device");
        ok_a = conversation_checkpoint_save(ca, A.ss, g, err_a);
    });
    ta.join(); tb.join();
    conversation_checkpoint_join(nullptr);
    const std::string n(name);
    check(ok_a, n + ": saver A's save (" + err_a + ")");
    check(ok_b, n + ": saver B's save (" + err_b + ")");
    check(ca.gdn == A.want, n + ": A's checkpoint holds A's running state");
    check(cb.gdn == B.want, n + ": B's checkpoint holds B's running state");
    for (int d : {dev_a, dev_b}) {
        cudaSetDevice(d);
        const cudaError_t e = cudaDeviceSynchronize();
        check(e == cudaSuccess, n + ": the card is still healthy (" + cudaGetErrorString(e) + ")");
    }
    std::printf("%s: %d hook wait(s) timed out (0 = the forced overlap happened; the fix holds B back)\n", name, held_back.load());
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no card"); return 77; }
    const char* on = std::getenv("STRATA_CKPT_ASYNC");
    if (on == nullptr || on[0] != '1') { std::puts("SKIP: STRATA_CKPT_ASYNC=1 not set"); return 77; }
    ModelGeometry g;
    g.n_layers = 4; g.n_expert = 256;
    g.ssm_state_size = 64; g.ssm_v_heads = 64; g.ssm_conv_channels = 8;   // a GDN row just over 1 MiB: the background path
    conversation_checkpoint_test_hook(hook);
    race(g, "same size, one card", 0, 2, 0, 2);
    race(g, "B larger, one card", 0, 1, 0, 3);
    if (devices >= 2) race(g, "same size, two cards", 0, 2, 1, 2);
    else std::puts("two-card case skipped: one card");
    conversation_checkpoint_test_hook(nullptr);
    std::printf("conversation_ckpt_race_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
