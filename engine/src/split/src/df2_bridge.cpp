// See df2_bridge.hpp. Compiled with the strata tree's headers (not the fnx ones).
#include "df2_bridge.hpp"

#include "strata/core/dflash2.hpp"
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace fnx {
namespace df2 {

using strata::core::Dflash2Drafter;

struct Bridge::Impl {
    int device = -1;
    std::vector<std::unique_ptr<Dflash2Drafter>> d;
    std::vector<float*> stage;              // per lane: the block's embeddings [kRows][N] on the card (row 0 the anchor, the rest the mask token)
    std::vector<std::vector<float>> anchor; // per lane, host: the anchor's embedding row
    std::vector<float> mask;                // the mask token's embedding row
    std::function<void(int, float*)> embed;
    uint8_t* head = nullptr;
    uint64_t extra = 0;
    std::vector<int> taps;
    int lane_of(cudaStream_t cs) const {
        for (size_t i = 0; i < d.size(); ++i)
            if (d[i]->stream() == cs) return (int) i;
        return -1;
    }
};

Bridge::Bridge() : p_(new Impl) {}

Bridge::~Bridge() {
    if (!p_) return;
    if (p_->device >= 0) cudaSetDevice(p_->device);
    for (auto& dr : p_->d) dr.reset();   // (their streams and buffers first)
    for (float* s : p_->stage) if (s) cudaFree(s);
    if (p_->head) cudaFree(p_->head);
}

bool Bridge::load(const std::string& pack, int device, int lanes, const std::vector<int32_t>& vocab, const void* head_rows, size_t row_bytes, int head_type,
                  std::function<void(int, float*)> embed, std::string& err) {
    Impl& I = *p_;
    I.device = device;
    I.embed = std::move(embed);
    // the ring log (host copies of the activation rows for a rebuild) is the main engine's way of resuming a conversation; the split starts a lane's ring over instead
    ::setenv("STRATA_DF2_RING_LOG", "0", 0);
    if (cudaSetDevice(device) != cudaSuccess) { err = "dflash2: cannot select the card"; return false; }
    I.d.emplace_back(new Dflash2Drafter);
    Dflash2Drafter& parent = *I.d[0];
    if (!parent.load(pack, device, err)) return false;
    for (int i = 1; i < lanes; ++i) {
        I.d.emplace_back(new Dflash2Drafter);
        if (!I.d.back()->clone_alloc(parent, err)) return false;
    }
    const size_t bytes = vocab.size() * row_bytes;
    if (cudaMalloc((void**) &I.head, bytes) != cudaSuccess) { err = "dflash2: the draft head's rows do not fit"; return false; }
    if (cudaMemcpy(I.head, head_rows, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "dflash2: the draft head's upload"; return false; }
    I.extra += bytes;
    const int N = parent.n_embd();
    I.mask.assign((size_t) N, 0.f);
    I.embed(parent.mask_token(), I.mask.data());
    Impl* ip = &I;
    auto fn = [ip](const int32_t*, int n, float* out, cudaStream_t cs) {
        const int lane = ip->lane_of(cs);
        const int L = lane < 0 ? 0 : lane;
        cudaMemcpyAsync(out, ip->stage[(size_t) L], (size_t) n * (size_t) (ip->mask.size()) * 4, cudaMemcpyDeviceToDevice, cs);
    };
    for (int i = 0; i < lanes; ++i) {
        float* s = nullptr;
        if (cudaMalloc((void**) &s, (size_t) kRows * (size_t) N * 4) != cudaSuccess) { err = "dflash2: the embedding staging does not fit"; return false; }
        I.extra += (size_t) kRows * (size_t) N * 4;
        std::vector<float> rows((size_t) kRows * (size_t) N);
        for (int r = 0; r < kRows; ++r) std::memcpy(rows.data() + (size_t) r * N, I.mask.data(), (size_t) N * 4);
        cudaMemcpy(s, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice);
        I.stage.push_back(s);
        I.anchor.emplace_back((size_t) N, 0.f);
    }
    if (!parent.bind_raw(fn, I.head, head_type, (int64_t) vocab.size(), vocab, err)) return false;
    for (int i = 1; i < lanes; ++i)
        if (!I.d[(size_t) i]->clone_bind(parent, err)) return false;
    for (auto& dr : I.d) dr->set_max_drafts(kDrafts);
    I.taps = parent.tap_layers();
    return true;
}

int Bridge::device() const { return p_->device; }
int Bridge::lanes() const { return (int) p_->d.size(); }
const std::vector<int>& Bridge::tap_layers() const { return p_->taps; }

uint64_t Bridge::vram_bytes() const {
    uint64_t v = p_->extra;
    for (const auto& d : p_->d) v += d->vram_bytes();
    return v;
}

void Bridge::tap(const float* R, const float* bo, const float* inj, int rows, int slot, float* out, int64_t out_stride, void* stream) {
    strata::kernels::df2_tap_gr(R, 4 * kNEmbd, bo, kNEmbd, inj, 4, out + (size_t) slot * kNEmbd, out_stride, rows, kNEmbd, 4, stream);
}

float* Bridge::landing(int lane) const { return p_->d[(size_t) lane]->landing(); }
void* Bridge::stream(int lane) const { return p_->d[(size_t) lane]->stream(); }

bool Bridge::context(int lane, const float* taps_dev, int64_t n, int64_t pos0, std::string& err) {
    return p_->d[(size_t) lane]->context(taps_dev, n, pos0, err);
}

bool Bridge::launch(int lane, int n_new, int64_t pos0, int anchor, std::string& err) {
    Impl& I = *p_;
    cudaSetDevice(I.device);
    std::vector<float>& a = I.anchor[(size_t) lane];
    I.embed(anchor, a.data());
    // row 0 of the lane's block staging (the rest stays the mask row); ordered before the graph on the drafter's own stream
    if (cudaMemcpyAsync(I.stage[(size_t) lane], a.data(), a.size() * 4, cudaMemcpyHostToDevice, I.d[(size_t) lane]->stream()) != cudaSuccess) {
        err = "dflash2: the anchor's embedding upload";
        return false;
    }
    Dflash2Drafter& d = *I.d[(size_t) lane];
    return d.draft_launch(d.landing(), n_new, pos0, anchor, err);
}

bool Bridge::finish(int lane, int32_t* drafts, float* probs, std::string& err) { return p_->d[(size_t) lane]->draft_finish(drafts, probs, err); }
void Bridge::reset(int lane) { p_->d[(size_t) lane]->reset(); }
int64_t Bridge::ring_hi(int lane) const { return p_->d[(size_t) lane]->ring_hi(); }
void Bridge::set_max_drafts(int n) { for (auto& d : p_->d) d->set_max_drafts(n); }
double Bridge::ms_gpu(int lane) const { return p_->d[(size_t) lane]->ms_gpu; }
long Bridge::rounds(int lane) const { return (long) p_->d[(size_t) lane]->rounds; }

bool Bridge::prepare(std::string& err) {
    for (auto& d : p_->d)
        if (!d->prepare(d->landing(), err)) return false;
    return true;
}

}  // namespace df2
}  // namespace fnx
