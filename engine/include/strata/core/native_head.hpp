#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {
struct BitplaneHeadHandle;
}

namespace strata::core {

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (strata::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err, int peer_device = -1);
    void unload();
    /// Moves the loaded weights from VRAM to mapped pinned host memory (a card too short for it: every logits row then reads them over PCIe).  False: not loaded, already
    /// on the host, or a repacked layout that stays in VRAM.
    bool to_host(std::string& err);
    bool on_host() const { return host_ != nullptr; }
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    const void* scratch() const { return scratch_; }   ///< the activation scratch the head's GEMV reads (captured with the weights)
    int type() const { return type_; }
    /// The device whose memory holds the weights (the one current when `load` ran; -1 before).
    int device() const { return device_; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }
    /// Exact 2-stage bit-plane LM head handle (Q8_0 / Q6_K on sm_80+; nullptr if unavailable or disabled).
    strata::kernels::BitplaneHeadHandle* bitplane() const { return bitplane_; }
    /// 50/50 P2P output head split across primary `device()` and `peer_device()` (STRATA_PEER_HEAD_SPLIT=1).
    bool is_split() const { return peer_weights_ != nullptr; }
    int peer_device() const { return peer_device_; }
    int split_rows0() const { return split_rows0_; }
    int split_rows1() const { return split_rows1_; }
    int max_split_t() const { return max_split_t_; }
    const void* peer_weights() const { return peer_weights_; }
    bool gather_rows(const int32_t* vocab_host, int64_t n_rows, int dst_dev, uint8_t* dst, std::string& err) const;
    void run_verify_split(const void* xq, void* xil, float* head_logits_dev0, int h0, int HT, bool use_il,
                          void* arg_scratch0, int32_t* m_out_row, uint32_t* m_nf_row, void* stream0) const;
    void materialize_logits(float* head_logits_dev0, int T, void* stream0) const;

private:
    void* weights_ = nullptr;
    void* host_ = nullptr;   // the pinned host buffer weights_ aliases when the head did not fit in VRAM
    void* scratch_ = nullptr;
    strata::kernels::BitplaneHeadHandle* bitplane_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
    int device_ = -1;
    int peer_device_ = -1;
    int split_rows0_ = 0, split_rows1_ = 0, max_split_t_ = 0;
    void* peer_weights_ = nullptr;
    void* peer_stream_ = nullptr;
    void* ev_in_ = nullptr;
    void* ev_done_ = nullptr;
    uint8_t *peer_xq_ = nullptr, *peer_xil_ = nullptr;
    float* peer_logits_ = nullptr;
    uint8_t* peer_arg_scratch_ = nullptr;
    float *max_val0_ = nullptr, *max_val1_ = nullptr, *peer_max_val_ = nullptr;
    int32_t *max_idx0_ = nullptr, *max_idx1_ = nullptr, *peer_max_idx_ = nullptr;
    uint32_t *nf1_ = nullptr, *peer_nf_ = nullptr;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token, so the table costs no VRAM.  `embed_row`, the verify window and the MTP drafter
/// use it instead of the canonical table when it is set (`set_native_embed`).  `STRATA_EMBED_VRAM=1` places the
/// same bytes in VRAM instead (slow PCIe links); the gathers read `dev_` either way.
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    /// The table was placed in VRAM (STRATA_EMBED_VRAM=1, or the pinned / alias fallbacks) rather than mapped.
    bool in_vram() const { return dev_ != nullptr && host_ == nullptr; }

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace strata::core
