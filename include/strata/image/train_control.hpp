// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_control.hpp - a training run's VRAM control (src/image/train_control.cpp), the one
// form serve's training job and the four trainers share:
//
//   --vram-budget <MiB>    the first budget (0: pause before the first step; the open still needs its room)
//   --vram-control <path>  the control file serve writes (tmp + rename): {"budget_mib": N, "save_state": bool, "stop": bool}
//   --state <path>         where save_state writes the continuation state (default <out>.state.bin)
//   --resume <path>        continue from a state file (the step count, the loop's rng and order, the LoRA, AdamW)
//
// The trainer reads the file at every micro-step's boundary (its mtime and size; a paused run polls it every
// XESTRATA_TRAIN_CONTROL_MS, 200 ms).  budget_mib 0 pauses there: the trainer's allocations go to the host as they
// are and the VRAM is given back (Trainer::pause); a budget above 0 resumes at the same values.  save_state true
// writes the state file at the next micro-step boundary (including accumulated gradients) and then pauses; serve
// may then stop the process and start it on another card with --resume. stop true ends normally at the next boundary
// (after a requested save). SIGTERM and SIGINT request the same stop, without an asynchronous GPU teardown.
// The stdout lines:
//
//   vram <used_mib> <peak_est_mib> <min_mib> <running|paused> [budget_mib= resident=k/nb tapes=k inputs=... free_mib= need_mib=]
//   state saved <path>
//
// The first four fields are the form serve reads; the key=value extras are for the log and the tests.  The vram
// line comes when its figures change (every step's boundary, and at a pause/resume). used is the trainer's own
// allocation accounting; process_mib includes the runtime, from process_source. need_mib is the estimated
// least-form process peak plus a sampling margin; the run starts or resumes only when the budget reaches it.
#pragma once

#include "strata/image/train_trainer.hpp"

#include <cstdint>
#include <functional>
#include <ios>
#include <iosfwd>
#include <string>

namespace strata::image::train {

/// Initial least-form estimate from the shared 1024-class table, scaled up for larger shapes and ranks.
/// These observations are estimates, not a hard bound on a new device or workload.
/// Install before starting any thread: SIGTERM/SIGINT request a normal stop at the next step or pause boundary.
void install_stop_signals();
bool stop_requested();

struct InitialVram { uint64_t min_mib, need_mib; };
InitialVram initial_vram(const std::string& model, const std::string& base, int64_t image_tokens, int64_t rank);

class Control {
public:
    struct Opts {
        std::string path;          // --vram-control (empty: the control is off, the CLI as it was)
        std::string state;         // --state (the state file's path)
        uint64_t budget_mib = 0;   // --vram-budget
        InitialVram initial = {};  // largest cached shape's least-form estimates
    };
    Control(Trainer& tr, Opts o);
    bool enabled() const { return !opts_.path.empty(); }

    /// Before each micro-step: read the control file, shrink/grow (the budget), pause (budget 0) or write the
    /// state and pause (save_state, even in a gradient accumulation). `save_loop` writes the loop's own state
    /// (the step count, the rng, the data's order) into the file.
    /// A paused run waits here, polling the file, until a budget above 0 resumes it.
    /// False means stop normally; the caller must return so the trainer's destructors drain and free their queues.
    bool boundary(bool update_done, const std::function<void(std::ostream&)>& save_loop);
    /// The vram line, when its figures changed (after a step, and at a pause/resume).
    void report();
    /// --resume: the state file into the trainer and the loop (`load_loop` reads what save_loop wrote).  The
    /// completed step count comes back through the loop's own state.  False where the path is empty.
    bool load(const std::string& path, const std::function<void(std::istream&)>& load_loop);

private:
    void read();                 // the control file, when its mtime/size changed (or a second passed)
    void save(const std::function<void(std::ostream&)>& save_loop);
    bool wait_resume(const std::function<void(std::ostream&)>& save_loop);
    void emit(bool force);
    Trainer& tr_;
    Opts opts_;
    uint64_t budget_mib_ = 0;
    uint64_t paused_min_mib_ = 0, paused_need_mib_ = 0;
    bool save_state_ = false, saved_ = false, stop_ = false;
    long long mtime_ = -1;
    std::streamoff size_ = -1;
    double last_read_ = 0;
    int poll_ms_ = 200;
    std::string last_line_;
};

}  // namespace strata::image::train