// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/graph.hpp - P2.S5: the GraphRegistry.
//
// `GraphRegistry`: capture/replay wrappers keyed by `(layer_type, n_tokens)`, with static input/output
// buffers at fixed addresses.
//
// THREE THINGS ARE BAKED INTO THIS INTERFACE, and each one is a measurement rather than a preference.  All
// three come from `bench/micro/graph_capture.cu`, which was written to answer the phase's own instruction -
// "verify on the installed CUDA version with a minimal test before building on it".
//
// 1. **THE WAIT MUST POLL THE DRIVER, NOT MEMORY.**  The phase's host loop says "spin on
//    doorbell_seq == expected ... no cudaStreamSynchronize anywhere".  A spin that only reads memory NEVER
//    RUNS THE KERNEL: 5,907,703 spins over 500 ms, and the datum flips to 1 the instant anything calls into
//    the driver and never before. Hence `wait_ms`, which polls an event -
//    a QUERY and not a blocking sync, so P2.X3's "zero synchronization calls in the layer loop" still holds,
//    and it costs 0.021 ms.
//
// 2. **A GRAPH RE-READS ITS INPUT BUFFERS AT REPLAY, BUT NOT ITS KERNEL ARGUMENTS.**  Verified: eight
//    replays with a different pinned input each time all tracked correctly, so the data path is live; but a
//    kernel argument is copied into the node at capture and is never re-read.  **Anything that changes per
//    step - a position, a page-table base, a token id - must be DATA in a device buffer, never an argument.**
//    This is the phase's "graph capture with pointers that change between steps silently replays stale
//    addresses" hazard, and it now has a measurement behind it.
//
// 3. **FIXED ADDRESSES ARE THE CALLER'S OBLIGATION.**  The registry cannot check them.  Every buffer a
//    captured body touches must be allocated before `record` and never reallocated; the registry holds the
//    graph, not the memory.
//
// What the registry is FOR: a layer's ops are the same every token, so recording them once and replaying
// removes the per-call host cost of submitting them individually.  Measured, that cost is 10-22 us PER CALL
// into the driver on this machine, and a layer with 8 submissions is ~0.1 ms of round trip.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "strata/core/gpu.hpp"

namespace strata::core {

/// The two attention families.  `docs/semantics.md`: 36 GDN layers and 12 QSA, at indices 3, 7, ... 47.
enum class LayerType : int { GDN = 0, QSA = 1 };
inline const char* to_string(LayerType t) { return t == LayerType::GDN ? "GDN" : "QSA"; }

/// One recorded graph.  Move-only: it owns an executable graph and an event, and two owners would
/// double-destroy.
class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    CapturedGraph(CapturedGraph&& o) noexcept { *this = std::move(o); }
    CapturedGraph& operator=(CapturedGraph&& o) noexcept;

    /// Start recording everything the caller launches into `stream`.
    bool begin(void* stream, std::string& err);

    /// Stop recording, instantiate, and record the completion event.  A capture that produced ZERO nodes is
    /// reported as an error: it means the body did nothing, which is a wiring mistake and NOT an empty graph
    /// to be replayed happily forever.
    bool end(void* stream, std::string& err);

    /// Stops a recording in progress and discards it.
    void abandon(void* stream);

    /// Replay.  Does not synchronise.
    bool launch(void* stream, std::string& err) const;

    /// The host loop's wait: poll `done_` until it completes or `timeout_ms` elapses.  Returns false on
    /// timeout or a failed query, `err` saying which, which the caller must treat as a hang and not as "probably
    /// fine".  See NOTE 1 above for why this exists instead of a memory spin.
    bool wait_ms(int timeout_ms, std::string& err) const;

    size_t nodes() const { return nodes_; }
    bool valid() const { return exec_ != nullptr; }

private:
    void reset();
    gpu::Graph* exec_ = nullptr;
    gpu::Event* done_ = nullptr;
    bool capturing_ = false;
    void* capture_stream_ = nullptr;   // the stream a recording in progress belongs to
    size_t nodes_ = 0;
};

/// Capture-once, replay-many, keyed by `(layer_type, n_tokens)`.
///
/// `record` is given a CALLABLE that launches the layer's ops.  On the first call for a key it captures;
/// afterwards it returns the cached graph and does NOT run the callable - so a caller that mutates state
/// inside the body will be surprised, which is why the body must only launch.
class GraphRegistry {
public:
    explicit GraphRegistry(void* stream) : stream_(stream) {}

    /// Capture `body` for `key` if it is not already recorded.  Returns false and fills `err` on failure.
    /// Recording the same key twice is a no-op and is NOT an error - that is the point of a registry, and a
    /// regression test asserts that the body runs exactly once across many calls.
    bool record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err);

    const CapturedGraph* find(LayerType type, int n_tokens) const;

    /// Launch and wait, in that order, the two things the host loop does per layer.
    bool launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const;

    size_t size() const { return graphs_.size(); }
    /// How many times a BODY was actually captured.  With a working cache this equals `size()`.
    size_t captures() const { return captures_; }

private:
    using Key = std::pair<int, int>;
    void* stream_ = nullptr;
    std::map<Key, CapturedGraph> graphs_;
    size_t captures_ = 0;
};

/// A count raised wherever a buffer shared between calls (a library's workspace, a kernel's buffer kept per queue) is
/// freed and allocated anew: a graph recorded before it may hold the old address, so StepForm drops its graphs.
uint64_t graph_epoch();
void graph_epoch_bump();

/// A step the same every time (what changes per step DATA in device buffers, NOTE 2), run with a recorded graph or
/// straight onto the stream, whichever is faster on this GPU.  The graph is not always faster (Qwen3.5-9B's decode,
/// 2026-10-27: the B70 46.9 tok/s with it, 52.7 without; the RTX 4070 79.9 and 77.1, the RX 9060 XT 47.5 and 41.4),
/// so the first steps take the two in turns, then the faster stays: one of each to warm up, then 4 of each counted
/// (each step timed in the slower form costs the run, and the two differ by a tenth or so).  A key's first step goes
/// straight, so what a library sets up at its first call (a plan, a scratch buffer, a wait) is done before a recording.
/// An environment variable may fix it: "0" a graph, "1" straight (a profiler that cannot take the graphs sees the
/// kernels).  Optionally it keeps the graphs too, by a key the caller makes of everything the step bakes in (its
/// buffers' addresses and sizes): a recording that fails leaves the steps straight.
class StepForm {
public:
    explicit StepForm(const char* env = nullptr);
    ~StepForm() { clear(); }
    StepForm(const StepForm&) = delete;
    StepForm& operator=(const StepForm&) = delete;

    /// -1 while timed, 0 with a graph, 1 straight
    int form() const { return form_; }
    void fix(int form) { form_ = form; }
    /// Whether this step goes straight (while timed, in turns).
    bool straight() const { return form_ == 1 || (form_ == -1 && (trial_ & 1) == 0); }
    /// After a step taken while timed, its milliseconds; the tenth counted picks the form and says so (`who`, `what`).
    /// A step of run()'s that ran a key the first time or recorded it is not counted.
    void timed(double ms, const char* who, const char* what);

    /// The step: `body` onto `stream` straight, or recorded once for `key` and replayed.  Does not synchronise.
    /// With `outputs` (the step's results read back, the stream synchronised) and `clear` (which puts something else
    /// in them onto the stream), a recording is checked once: the step straight, then its results cleared and its graph
    /// replayed on the same inputs, must give the same bytes, else the steps go straight.  A library that runs outside
    /// a recording (oneMath's rocBLAS backend before xestrata-13) did its work once, at the recording, and is left out
    /// of the replays; the clearing keeps what it wrote then from passing for the replay's results.  That needs a step
    /// that gives the same results when run again on its inputs.
    bool run(uint64_t key, void* stream, const std::function<void()>& body, std::string& err,
             const std::function<std::vector<uint8_t>()>& outputs = nullptr,
             const std::function<void()>& clear = nullptr);
    /// Drops the graphs (before their buffers go).
    void clear();

private:
    int form_ = -1;
    int trial_ = 0;
    double ms_[2] = {};                    ///< the timed steps' milliseconds, with a graph and straight
    bool setup_ = false;                   ///< the last run() set a key up
    std::map<uint64_t, gpu::Graph*> graphs_;   ///< null: the key's first step, run straight
    uint64_t epoch_ = 0;                   ///< graph_epoch() when the graphs were recorded
};

}  // namespace strata::core
