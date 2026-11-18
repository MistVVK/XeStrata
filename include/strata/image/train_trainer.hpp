// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/train_trainer.hpp - what every image model's LoRA trainer holds and does the same way
// (src/image/train_trainer.cpp): the device memory it owns, the trained factors and their clipped AdamW update, the
// blocks' weights through a LayerStream with their VRAM budget, the blocks' shared scratch (an arena), the last blocks'
// tapes kept from the forward to the backward, and the form the blocks' inputs are kept in.  A model's trainer
// (qi_trainer.cpp, moss_trainer.cpp, anima_trainer.cpp, mf_trainer.cpp) derives from it and holds what is its own:
// its DiT's edges, its blocks and its file's names.
#pragma once

#include "strata/image/qi_stream.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_ops.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <deque>
#include <iosfwd>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::image::train {

/// A trained factor pair on one product [N][K]: FP32 masters A [rank][K] and B [N][rank], and `ad`, what the block
/// reads (the FP16 views a and b, their transposed views where the GPU takes no transposed products, the gradients dA
/// and dB).  A one-row LoRA (a modulation's, row_lora_fwd) reads the masters as they are and has no FP16 views.
struct Lora {
    Adapter ad;
    float *A = nullptr, *B = nullptr;
    int64_t N = 0, K = 0;
    bool views = true;
};

class Trainer {
public:
    /// `what` names the trainer in its messages ("Qwen-Image-2.1 training").
    explicit Trainer(std::string what);
    virtual ~Trainer();
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    /// The step's stream and the products' Gemm (its workspace `ws_bytes`).
    bool init(void* stream, size_t ws_bytes, std::string& err);

    // ---- device memory the trainer owns, freed with it
    /// n values, zeroed on the stream and waited for: the host's values are copied in through the compute queue
    /// (gpu::copy), which is not ordered against the stream, and a zero that ran late wiped parts of the weights
    /// (2026-11-09).
    template <class T> T* get(int64_t n) { return static_cast<T*>(take((size_t) n * sizeof(T))); }
    void release(void* p);
    float* dev_vec(const std::vector<float>& v);
    uint16_t* dev16(const void* v, int64_t n);   // n FP16 values
    std::vector<float> read(const float* p, int64_t n) const;

    // ---- the trained tensors
    /// A plain LoRA of `rank` (alpha = rank: scale 1) on a product [N][K]: A drawn from rng (lora_init), B zero, both
    /// trained; `views` false for a one-row LoRA.  The reference stays valid for the trainer's life.
    Lora& add_lora(int64_t N, int64_t K, int64_t rank, std::mt19937_64& rng, bool views = true);
    /// A tensor of n values the update trains (w its FP32 master, g its gradient), its AdamW moments made here.  The
    /// global norm and the updates run in the order the tensors were added.
    void add_param(float* w, float* g, int64_t n);
    /// The FP16 views of a LoRA's masters (written after each update).
    void sync_views(const Adapter& ad, const float* A, const float* B, int64_t N, int64_t K);
    /// The optimizer's step: the global norm of every gradient times `gscale` (a gradient summed over micro-steps
    /// averaged), torch's clip_grad_norm_ at 1 (the gradients scaled by 1 / (norm + 1e-6) past it), AdamW on every
    /// tensor (the gradients zeroed), then the LoRAs' views; the norm before the clip.
    double update(const AdamW& o, float gscale = 1.f);
    /// The trained LoRAs, in the order add_lora made them.
    const std::deque<Lora>& loras() const { return loras_; }

    /// The scratch dgrad_weights writes for a block's products p[0 .. n) (every block's the same shapes), `int8` an int8
    /// base's: none where the products read a 16-bit weight's rows as they are.
    void dgrad_scratch(const Product* p, int n, bool int8);
    /// The weights a block's backward reads for its products' dgrads, p[0 .. n) as the block is bound: where the GPU's
    /// products read a transposed operand (Gemm::f16_batch_ex_ok), the rows as they are (Product::wr: w itself, an
    /// int8 base's straight-through weight rotated back) and no transposed copy, a block each step; else transposed
    /// (Product::wt).
    void dgrad_weights(Product* p, int n);

    // ---- the memory of a step
    /// The VRAM the blocks' weights (ws) take, set when a step brings more rows than any before and once after the
    /// first step (its reserve for the kernels' first runs given back): the free memory, what the weights and the
    /// tapes hold now, less the step's own beside them - `keep_row` bytes a row it takes after the budget is set (the
    /// blocks' inputs in BF16 at least), the arena's growth (scratch_growth, `guess_row` bytes a row before a step has
    /// shown it), 1.5 GiB of transients and 3 GiB more before the first step (the kernels' first runs: free memory
    /// fell 2.7 GB more than the arenas took on the B70, 2026-11-09) - all of them where that takes them, else as many
    /// as it does and the rest streamed.  XESTRATA_TRAIN_WEIGHTS_MIB sets it.  The tapes are let go where the weights
    /// need their memory.
    void set_budget(int64_t rows, uint64_t keep_row, uint64_t guess_row);
    /// Whether this step keeps its blocks' inputs in BF16 (XESTRATA_TRAIN_INPUTS16=0 or 1, or by the memory): when the
    /// free memory and the tapes' buffer fall short of `f32_bytes` (the inputs in FP32 and what else the caller counts
    /// beside them), the arena's growth to this step's rows (`guess_row` bytes a row before a step has shown it),
    /// 1.5 GiB of transients and 3 GiB more before the first step.  The forward then rounds each input to BF16 as it
    /// keeps it, so the backward reads what the forward used.
    bool inputs_half(int64_t rows, uint64_t f32_bytes, uint64_t guess_row);
    /// How many of the last of `nb` blocks keep their tapes this step (a tape `per` bytes, `step_bytes` what the step
    /// takes of its own after this): XESTRATA_TRAIN_TAPES, or as many as the device memory takes once a step of as
    /// many rows or more has shown the arena's size - none the first step, nor a step of more rows than any before (a
    /// guess of the arena's growth ran the RX 9060 XT out of memory): the free memory and the tapes' own buffer, less
    /// the arena's growth to this step's rows, step_bytes and 1.5 GiB of transients.  The buffer grows when a step
    /// keeps more tapes than it holds, and shrinks only when the memory does not allow it (a smaller step's tapes fit
    /// in it as they are).
    int64_t keep_tapes(int64_t nb, int64_t rows, size_t per, size_t step_bytes);
    /// Block b's kept tape (keep_tapes), null where it keeps none.
    Arena* tape(int64_t b) const;
    /// A step of `rows` ran: its kernels compiled and the arena grown to it (`row_bytes`: one R×C FP32 input).
    void stepped_rows(int64_t rows, uint64_t row_bytes = 0);
    /// The last step's rows and input bytes, and the arena's per-row guess (the VRAM report's figures).
    int64_t last_rows() const { return last_rows_; }
    uint64_t last_row_bytes() const { return last_row_bytes_; }
    uint64_t guess_row() const { return guess_row_; }

    // ---- the VRAM budget: an explicit cap, the step-boundary shrink/grow, pause and resume
    /// The VRAM (bytes) this trainer may hold on its GPU, everything of its own counted (0: the automatic form,
    /// the free memory less the reserves).  Taken at the next step's planning; a change asks the step's buffers
    /// to be planned again (replan_due).
    void set_vram_budget(uint64_t bytes);
    uint64_t vram_budget() const { return budget_; }
    /// Whether a budget change asks the cached step buffers to be taken again (qi's ensure).
    bool replan_due() const { return replan_; }
    /// What the trainer holds now: its own allocations, the arena, the tapes and the weights in VRAM.
    uint64_t held() const;
    /// The least-form estimate (one streaming slot, no tape, host inputs): the live baseline before a step has run,
    /// then the last step's measured process use with its reclaimable allocations removed. Not a guaranteed peak.
    uint64_t min_bytes(int64_t rows, uint64_t row_bytes, uint64_t guess_row) const;
    bool min_measured() const { return min_rows_ > 0; }
    /// The peak a step of `rows` is estimated to take: what is held, the arena's growth and the transients' reserve.
    uint64_t est_bytes(int64_t rows, uint64_t guess_row) const;
    /// Pause at a step's boundary: every allocation of the trainer's copied to the host as it is (the values kept
    /// byte for byte) and freed, the weights' regions, the tapes and the arena given back (the streamed layers'
    /// pinned homes stay: the resume is a copy, not a read of the file).  The model's own pointers are fixed in its
    /// remap() through moved().  resume() takes the same values back; nothing of the computation changes.
    void pause();
    void resume();
    bool paused() const { return paused_; }
    /// The new address of an allocation pause() took (the model's remap hook).  A pointer that was the trainer's
    /// and is not in the map is a missed remap: it throws, not a dangling read.  A pointer it never owned comes
    /// back as it was (a weights region's, bound again at the next step).
    void* moved(const void* old) const;
    /// The inputs' form this step (the VRAM report's): "fp32", "host" or "bf16".
    void set_inputs_mode(const char* m) { inputs_mode_ = m; }
    const char* inputs_mode() const { return inputs_mode_; }
    /// The host form taken (the pinned memory for `slots` rows of `row_bytes`): false where
    /// XESTRATA_TRAIN_INPUTS16=1 forces BF16, XESTRATA_TRAIN_INPUTS_HOST=0 forbids the host, or the pinned
    /// memory is not had (the caller falls back to BF16).
    bool host_inputs(int64_t slots, int64_t row_bytes);
    int64_t tapes_kept() const { return tapes_kept_; }
    /// The trained tensors as the state file holds them (train::Control writes the header and the loop's state
    /// around this): every parameter's FP32 master, gradient and moments m and v, in add_param order.  A paused
    /// trainer reads its saved host allocations; a running trainer reads them from the GPU.
    void save_state(std::ostream& os) const;
    void load_state(std::istream& is);

    /// The blocks' inputs kept on the host as FP32 (the low-VRAM form that keeps the values byte for byte):
    /// (slots) rows of `bytes` in pinned memory, copied out as the forward writes them and read back before the
    /// backward's block, one block ahead (HostInputs).  open() false where the pinned memory is not had: the
    /// caller falls back to BF16 (inputs_half).
    class HostInputs {
    public:
        ~HostInputs();
        bool open(int64_t slots, int64_t bytes, void* compute);
        void close();
        bool ok() const { return h_ != nullptr; }
        /// The forward's write of `src` copied to slot `slot` (the copy waits for the compute stream's write).
        void keep(int64_t slot, const float* src, int64_t bytes);
        /// The compute stream waits for slot `slot`'s keep copy: the buffer it read may be written again.
        void kept_wait(int64_t slot);
        /// Slot `slot` read into `dst` ahead of its use: the copy waits for `used_parity`'s last reader of dst
        /// (-1: none), the compute stream waits for the copy at wait(slot).
        void fetch(int64_t slot, float* dst, int64_t bytes, int used_parity);
        void wait(int64_t slot);
        /// The compute stream is past the reads of the buffer parity `p` (it may be a fetch's destination again).
        void used(int p);

    private:
        uint8_t* h_ = nullptr;
        int64_t slot_bytes_ = 0;
        void* s_ = nullptr;
        void* cs_ = nullptr;
        gpu::Event* wr_ = nullptr;
        gpu::Event* used_[2] = {};
        std::vector<gpu::Event*> done_;
        std::vector<gpu::Event*> kept_;
    };
    HostInputs host;

    void* s = nullptr;
    prefill::Gemm gemm;
    LayerStream ws;                              // the blocks' weights
    std::vector<std::vector<uint8_t>> whost;     // their rows as the products read them, made at open: ws's source
    Arena arena;                                 // the blocks' scratch, kept from block to block and step to step
    int64_t steps = 0;                           // the optimizer's steps
    int64_t scratch_rows = 0;                    // the most rows a step has had: the arena holds what they took
    bool stepped = false;                        // a step ran (its kernels compiled, the arena grown)

protected:
    const std::string& what() const { return what_; }
    /// The model fixes its own persistent pointers after resume(), through moved() (the trainer's own are fixed).
    virtual void remap() {}
    virtual void sync_loaded_views() {}

private:
    void* take(size_t bytes);
    void drop_tapes();
    struct Param {
        float *w, *g, *m, *v;
        int64_t n;
    };
    std::string what_;
    std::vector<std::pair<void*, size_t>> owned_;
    uint64_t owned_bytes_ = 0;
    std::deque<Lora> loras_;
    std::vector<uint16_t*> dgrad_tmp_;   // dgrad_weights' [N][K] or [K][N] FP16 a product
    uint16_t* w8tmp_ = nullptr;          // an int8 base's rows before their transpose
    std::vector<Param> params_;
    int64_t budget_rows_ = 0;
    bool budget_stepped_ = false;
    int64_t last_rows_ = 0;
    uint64_t last_row_bytes_ = 0, guess_row_ = 0;
    int64_t min_rows_ = 0;
    uint64_t measured_min_ = 0;
    std::vector<std::unique_ptr<Arena>> tapes_;
    void* tape_buf_ = nullptr;
    size_t tape_cap_ = 0;
    int64_t tapes_kept_ = 0;
    uint64_t budget_ = 0;                // the explicit VRAM cap (0: automatic)
    bool replan_ = false;                // a budget change asks the step's planning again
    bool paused_ = false;
    const char* inputs_mode_ = "fp32";
    struct Hosted {
        void* old;
        size_t bytes;
        std::vector<uint8_t> data;
    };
    std::vector<Hosted> host_;           // pause()'s copies (freed at resume; the addresses kept for moved())
    std::vector<void*> old_owned_;
    std::unordered_map<void*, void*> moved_;
};

}  // namespace strata::image::train
