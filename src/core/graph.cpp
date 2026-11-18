// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/graph.cpp - P2.S5: the GraphRegistry implementation, on SYCL command graphs (strata/core/gpu.hpp).
#include "strata/core/graph.hpp"
#include "strata/platform/env.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::core {
namespace {

bool fail(std::string& err, const char* what) {
    err = std::string(what) + ": " + gpu::last_error();
    return false;
}

}  // namespace

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        exec_ = o.exec_;
        done_ = o.done_;
        nodes_ = o.nodes_;
        capturing_ = o.capturing_;
        capture_stream_ = o.capture_stream_;
        o.exec_ = nullptr;
        o.done_ = nullptr;
        o.nodes_ = 0;
        o.capturing_ = false;
        o.capture_stream_ = nullptr;
    }
    return *this;
}

void CapturedGraph::reset() {
    if (capturing_) { gpu::abandon_capture(capture_stream_); capturing_ = false; capture_stream_ = nullptr; }
    // graph_destroy drains the queues first, so a replay in flight completes before its graph goes
    if (exec_) { gpu::graph_destroy(exec_); exec_ = nullptr; }
    if (done_) { gpu::event_destroy(done_); done_ = nullptr; }
    nodes_ = 0;
}

bool CapturedGraph::begin(void* stream, std::string& err) {
    if (exec_ || capturing_) { err = "begin: this CapturedGraph is already recorded"; return false; }
    if (!gpu::begin_capture(stream)) return fail(err, "begin_capture");
    capturing_ = true;
    capture_stream_ = stream;
    return true;
}

bool CapturedGraph::end(void* stream, std::string& err) {
    capturing_ = false;
    capture_stream_ = nullptr;
    // an empty recording is a wiring mistake: a graph that replays nothing produces no error and no output
    if (!gpu::end_capture(stream, &exec_)) return fail(err, "end_capture");
    nodes_ = gpu::graph_nodes(exec_);
    // the completion event is re-recorded after each replay, which makes `wait_ms` a query rather than a sync
    if (!gpu::event_create(&done_)) return fail(err, "event_create");
    return true;
}

void CapturedGraph::abandon(void* stream) {
    gpu::abandon_capture(stream);
    capturing_ = false;
    capture_stream_ = nullptr;
    reset();
}

bool CapturedGraph::launch(void* stream, std::string& err) const {
    if (!exec_) { err = "launch: not recorded"; return false; }
    if (!gpu::graph_launch(exec_, stream)) return fail(err, "graph_launch");
    if (!gpu::event_record(done_, stream)) return fail(err, "event_record");
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms, std::string& err) const {
    if (!done_) { err = "launch: not recorded"; return false; }
    // A BOUNDED wait on a real clock: an unbounded spin turns a protocol bug into a hung run.  The event query is a
    // driver call that does not block.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const gpu::Status st = gpu::event_status(done_);
        if (st == gpu::Status::Done) return true;
        if (st == gpu::Status::Failed) return fail(err, "launch: the graph");
        if (std::chrono::steady_clock::now() >= deadline) {
            err = "launch: timed out waiting for the graph to complete";
            return false;
        }
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;              // already recorded: the point of a registry

    CapturedGraph g;
    if (!g.begin(stream_, err)) return false;
    try {
        body();                                    // the caller submits into the recorded stream
    } catch (const std::exception& e) {
        g.abandon(stream_);                        // leave the stream out of recording mode
        err = std::string("the capture body failed: ") + e.what();
        return false;
    }
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    return g->wait_ms(timeout_ms, err);
}

namespace {
std::atomic<uint64_t> g_epoch{0};
}

uint64_t graph_epoch() { return g_epoch.load(std::memory_order_acquire); }
void graph_epoch_bump() { g_epoch.fetch_add(1, std::memory_order_acq_rel); }

StepForm::StepForm(const char* env) {
    const char* e = env != nullptr ? strata::env_get(env) : nullptr;
    if (e != nullptr && (e[0] == '0' || e[0] == '1')) form_ = e[0] - '0';
}

void StepForm::timed(double ms, const char* who, const char* what) {
    if (form_ != -1) return;
    if (setup_) {   // a step that set up or recorded: not counted
        setup_ = false;
        return;
    }
    if (trial_ >= 2) ms_[(trial_ & 1) == 0 ? 1 : 0] += ms;
    if (++trial_ == 10) {
        form_ = ms_[1] < ms_[0] ? 1 : 0;
        std::fprintf(stderr, "%s: %s %s (timed: %.2f ms with a graph, %.2f ms without)\n", who, what,
                     form_ == 1 ? "without a graph" : "with a graph", ms_[0] / 4, ms_[1] / 4);
    }
}

bool StepForm::run(uint64_t key, void* stream, const std::function<void()>& body, std::string& err,
                   const std::function<std::vector<uint8_t>()>& outputs, const std::function<void()>& clear) {
    if (straight()) {
        body();
        return true;
    }
    if (epoch_ != graph_epoch()) {
        this->clear();
        epoch_ = graph_epoch();
    }
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {   // a key's first step straight: what its libraries set up at a first call is done
        if (graphs_.size() >= 16) this->clear();   // the keys of buffers since grown or shapes since gone
        graphs_.emplace(key, nullptr);
        setup_ = true;
        body();
        return true;
    }
    if (it->second == nullptr) {
        std::vector<uint8_t> want;
        if (outputs) {
            body();
            want = outputs();
        }
        if (!gpu::begin_capture(stream)) return fail(err, "begin_capture");
        gpu::Graph* g = nullptr;
        try {
            body();
        } catch (const std::exception& e) {
            gpu::abandon_capture(stream);
            std::fprintf(stderr, "xestrata: a step could not be recorded (%s); its steps go straight\n", e.what());
            form_ = 1;
            body();
            return true;
        }
        if (!gpu::end_capture(stream, &g)) {
            std::fprintf(stderr, "xestrata: a step could not be recorded (%s); its steps go straight\n",
                         gpu::last_error());
            form_ = 1;
            body();
            return true;
        }
        it->second = g;
        setup_ = true;
        if (outputs) {
            if (clear) clear();
            if (!gpu::graph_launch(g, stream)) return fail(err, "graph_launch");
            const std::vector<uint8_t> got = outputs();
            if (got != want) {
                size_t differ = 0;
                for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) differ += got[i] != want[i];
                std::fprintf(stderr, "xestrata: a recorded step gave other results than the step itself (%zu of %zu "
                                     "bytes); its steps go straight\n", differ, want.size());
                form_ = 1;
                body();
            }
            return true;
        }
    }
    if (!gpu::graph_launch(it->second, stream)) return fail(err, "graph_launch");
    return true;
}

void StepForm::clear() {
    for (auto& [k, g] : graphs_)
        if (g != nullptr) gpu::graph_destroy(g);
    graphs_.clear();
}

}  // namespace strata::core
