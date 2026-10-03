// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/graph.cpp - P2.S5: the GraphRegistry implementation, on SYCL command graphs (strata/core/gpu.hpp).
#include "strata/core/graph.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>

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

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    // A BOUNDED wait on a real clock: an unbounded spin turns a protocol bug into a hung run.  The event query is a
    // driver call that does not block.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (gpu::event_query(done_)) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
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
    if (!g->wait_ms(timeout_ms)) { err = "launch: timed out waiting for the graph to complete"; return false; }
    return true;
}

}  // namespace strata::core
