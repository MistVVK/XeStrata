// The DFlash2 drafter for the all-cards split engine. strata::core::Dflash2Drafter (src/core/dflash2.cpp) is compiled in the strata world
// (src/split/CMakeLists.txt, library strata_split_df2) and this header is all the fnx world sees of it: raw pointers, plain C++, no strata header.
//
// One drafter per lane, all on ONE card (the pack and the draft head live there once; a lane owns a K/V ring of about 100 MiB). The taps it
// reads (the mean of the 4 hyper-connection streams after layers 5, 15, 25, 35, 45) are made by tap() inside the engine's window graphs.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fnx {
namespace df2 {

constexpr int kRows = 8;      // block rows: the anchor + 7 masks; also the most context rows a draft() takes
constexpr int kDrafts = 7;
constexpr int kTaps = 5;      // tapped layers
constexpr int kNEmbd = 2560;
constexpr int kRing = 2048;   // positions of taps the drafter looks back over

class Bridge {
public:
    Bridge();
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    // pack: tools/dflash2_pack.py's file (Q8_0 or Q4_0 matrices); device: the card; lanes: drafters (one per engine lane);
    // vocab: the draft vocabulary's token ids, head_rows: those tokens' rows of output.weight (Q8_0 blocks, vocab.size() * row_bytes, host);
    // embed(tok, out): the token's embedding row (N floats), host side. Everything is allocated here, before the engine sizes its K/V pool
    bool load(const std::string& pack, int device, int lanes, const std::vector<int32_t>& vocab, const void* head_rows, size_t row_bytes,
              int head_type, std::function<void(int, float*)> embed, std::string& err);
    int device() const;
    int lanes() const;
    const std::vector<int>& tap_layers() const;   // {5, 15, 25, 35, 45}
    uint64_t vram_bytes() const;                  // everything it holds on its card

    // window taps: R (stride D = 4 N), bo (stride N) and inj (stride 4) of the card the drafter sits on, `rows` rows -> out rows (stride out_stride)
    // for tap slot `slot`. Enqueued on `stream`, so it can be captured into a graph
    static void tap(const float* R, const float* bo, const float* inj, int rows, int slot, float* out, int64_t out_stride, void* stream);

    float* landing(int lane) const;   // [kRows][kTaps][kNEmbd] on the drafter's card: the rows a draft() reads (its graphs are captured on it)
    void* stream(int lane) const;
    // the committed rows' taps (device, [n][kTaps * kNEmbd]) for positions pos0 ... into lane's ring; waits
    bool context(int lane, const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    // one round: rows 0..n_new-1 of landing(lane) are the committed rows at pos0 ..., anchor the last committed token (at pos0 + n_new)
    bool launch(int lane, int n_new, int64_t pos0, int anchor, std::string& err);
    bool finish(int lane, int32_t* drafts, float* probs, std::string& err);   // kDrafts of each
    void reset(int lane);                      // the lane's ring is empty (a new sequence)
    int64_t ring_hi(int lane) const;           // one past the highest position its ring holds in this epoch
    void set_max_drafts(int n);
    bool prepare(std::string& err);            // every lane's round graphs captured now
    double ms_gpu(int lane) const;
    long rounds(int lane) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace df2
}  // namespace fnx
