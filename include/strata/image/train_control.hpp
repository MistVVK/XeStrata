// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_control.hpp - a training run's VRAM control (src/image/train_control.cpp), the one
// form serve's training job and the four trainers share:
//
//   --vram-budget <MiB>    the first budget (0: pause before the first step; the open still needs its room)
//   --vram-control <path>  the control file serve writes (tmp + rename): {"budget_mib": N, "save_state": bool}
//   --state <path>         where save_state writes the continuation state (default <out>.state.bin)
//   --resume <path>        continue from a state file (the step count, the loop's rng and order, the LoRA, AdamW)
//
// The trainer reads the file at every micro-step's boundary (its mtime and size; a paused run polls it every
// XESTRATA_TRAIN_CONTROL_MS, 200 ms).  budget_mib 0 pauses there: the trainer's allocations go to the host as they
// are and the VRAM is given back (Trainer::pause); a budget above 0 resumes at the same values.  save_state true
// writes the state file at the next optimizer update's boundary (the gradients zero there) and then pauses; serve
// may then end the process and start it on another card with --resume.  The stdout lines:
//
//   vram <used_mib> <peak_est_mib> <min_mib> <running|paused> [budget_mib= resident=k/nb tapes=k inputs=... free_mib=]
//   state saved <path>
//
// The first four fields are the form serve reads; the key=value extras are for the log and the tests.  The vram
// line comes when its figures change (every step's boundary, and at a pause/resume).  used is the device's report
// (total - free) where it is had: what the runtime still holds after a pause is counted, not what was freed.
#pragma once

#include "strata/image/train_trainer.hpp"

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>

namespace strata::image::train {

class Control {
public:
    struct Opts {
        std::string path;          // --vram-control (empty: the control is off, the CLI as it was)
        std::string state;         // --state (the state file's path)
        uint64_t budget_mib = 0;   // --vram-budget
    };
    Control(Trainer& tr, Opts o);
    bool enabled() const { return !opts_.path.empty(); }

    /// Before each micro-step: read the control file, shrink/grow (the budget), pause (budget 0) or write the
    /// state and pause (save_state, only where `update_done`: the last optimizer update finished, the gradients
    /// zero).  `save_loop` writes the loop's own state (the step count, the rng, the data's order) into the file.
    /// A paused run waits here, polling the file, until a budget above 0 resumes it.
    void boundary(bool update_done, const std::function<void(std::ostream&)>& save_loop);
    /// The vram line, when its figures changed (after a step, and at a pause/resume).
    void report();
    /// --resume: the state file into the trainer and the loop (`load_loop` reads what save_loop wrote).  The
    /// completed step count comes back through the loop's own state.  False where the path is empty.
    bool load(const std::string& path, const std::function<void(std::istream&)>& load_loop);

private:
    void read();                 // the control file, when its mtime/size changed (or a second passed)
    void save(const std::function<void(std::ostream&)>& save_loop);
    void wait_resume(const std::function<void(std::ostream&)>& save_loop);
    void emit(bool force);
    Trainer& tr_;
    Opts opts_;
    uint64_t budget_mib_ = 0;
    bool save_state_ = false, saved_ = false;
    long long mtime_ = -1;
    std::streamoff size_ = -1;
    double last_read_ = 0;
    int poll_ms_ = 200;
    std::string last_line_;
};

}  // namespace strata::image::train