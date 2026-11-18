// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/image_lora.hpp - the LoRA adapters the image engine applies at run time: an adapter file read
// into the host tensors of the adapter forms (LoRA, LoHa, LoKr, GLoRA, OFT, BOFT, DoRA and the whole deltas), and the
// delta a DiT's products add.
//
// A file's keys are brought to the DiT's own names (the prefix dropped, the suffix that says which tensor of the pair
// it is), then grouped: one entry a weight, with the form it holds.  The DiT says which of its weights an entry
// changes and over which rows (a fused weight such as Qwen-Image-2.1's img_mlp.gate_up takes a row range, or the
// separate gate_layer and proj of diffusers' form).  The delta, for the plain form: y[t] += scale * (x[t] A^T) B^T,
// A [rank][k] and B [n][rank], in FP16 with FP32 sums (LoraDeviceEntry::add).  LoKr's delta is its two factors'
// Kronecker product, w1 [out_l][in_m] and w2 [out_k][in_n]: the input's rows read in in_m groups of in_n, w2's product
// of each group, then w1's over the groups (two products, the Kronecker product never made).
//
// A DoRA (the weight decomposed into its rows' norms and directions) keeps the weight it changes.  Its magnitude m is
// the file's `dora_scale`, the same vector under PEFT's name (`lora_magnitude_vector`, its `.default` form too) or
// ai-toolkit's (`.magnitude`), or LyCORIS's rows' vector shaped [rows][1]: c = m / (|W| a row + eps) over the
// weight's output rows, and W' = W + s (c (W + alpha/rank * BA) - W), so a product's output is
// y = [1 + s (c - 1)] (W x) + s c (alpha/rank BA x): the base product's rows scaled by 1 + s (c - 1) (LoraPart::base,
// the DiT's product applies it where its output is in the input's units) and the delta's rows by c (folded into B when
// the entry goes to the GPU).  eps is FP32's, whatever the file's tensors are (LP64_IEC_559's 1.1920929e-07).
//
// A whole delta (diff, set_weight, w_norm) is the weight's own matrix: y[t] += s * Delta x[t] (LoraEntry::delta, on
// the GPU as FP16, a product of its own).  A diff patches the weight as W += strength * diff; a set_weight, read as
// written, replaces the weight whatever the strength, and the engine interpolates instead - W' = W + s (S - W), the
// base product's rows scaled by 1 - s (the same base factors a DoRA's are) and the delta's by s - so the strength 1
// is the replacement and 0 leaves the weight as it was.  A LoHa's is the same kind of delta, made from its four
// factors when it is bound ((w1a . w1b) * (w2a . w2b), the Hadamard product of two low ranks) rather than kept: the
// weight's size is not held on the host a second time, the file's tensors are a few MiB.
//
// A bias's delta (x.diff_b, or x.b_norm beside the weight's w_norm) is a second patch of the same layer, on the
// weight's bias: b' = b + s * Db, the request's strength only - a diff patch has no alpha - added where
// the DiT adds the weight's bias (after the product's row scales and whatever its own bias's value is).  A file may
// hold one alone (LoraKind::bias: it changes no weight) or beside the weight's own delta.
//
// A GLoRA (a1/a2/b1/b2) writes its delta as a low rank of the weight on each side:
// W' = W + s (alpha / rank) (W a1 a2 + b1 b2) - LyCORIS's older form, whose a1/a2 and b1/b2 are the other way round
// (W a2 a1 + b2 b1), is the same two products with the roles swapped, and the tensors' shapes say which one a file
// holds.  So a product's output is y = W x + s c (W v + B (A x)) with v = a1 (a2 x): the file's b1 b2 is a plain
// LoRA's pair (B = b1, A = b2) and adds to the delta as the plain form does, and the weight is applied a second time,
// to another input (v), rather than the weight's delta being made (which is the weight's size, for a form whose two
// sides are a few MiB).  A weight's own path is the weight's, so a GLoRA keeps to its rows as a DoRA and an OFT do
// (another patch's rows must not overlap them: the files' order would decide how the two compose).
//
// An OFT (oft_blocks) turns a weight's output rows: the rows are cut into blocks of `bs` rows, each block's rotation
// r = (I + Q) (I - Q)^-1 of its skew part Q = blocks - blocks^T (the file's alpha clamped as Q's Frobenius norm where
// it is above 0), and the weight becomes r W: a product's output y is turned, (1 - s) y + s * rescale * (r y) -
// interpolated once, so the strength 1 is the whole rotation r W (a patch that interpolates the rotation and then
// scales the turned weight by s again applies s twice: the engine takes the one interpolation, as its set_weight's)
// and 0 leaves the weight as it was.  `rescale`, where the file holds one, scales the turned rows (LyCORIS's form;
// some readers drop it for an OFT and take it for a BOFT).  A bias is not turned (the rotation is the weight's).  A
// BOFT (four dimensions) applies `stages` of such rotations, each stage's blocks over the rows their own way (a
// butterfly), composed before the one interpolation.  The rows are the weight's own, so an OFT keeps to them:
// another adapter's rows must not overlap its own (the files' order would decide how the two compose).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::prefill {
class Gemm;
}

namespace strata::image {

/// DoRA's eps: the weight dtype's rounding added to a row's norm (torch.finfo(weight.dtype).eps); the engine's
/// weights are FP32 on the host whatever their file holds, so FP32's, the same for every file and GPU.
constexpr float kLoraDoraEps = 1.1920929e-07f;

/// The delta forms a file may hold, by the names of their tensors' roles.
enum class LoraKind : int {
    lora = 0,   // lora_A/B, lora.down/up, lora_down/up: y += scale * (x A^T) B^T
    lokr = 1,   // lokr_w1 (x) lokr_w2, or their a/b factors (Tucker's lokr_t2 too)
    loha = 2,   // (hada_w1_a hada_w1_b) * (hada_w2_a hada_w2_b)
    glora = 3,  // a1/a2/b1/b2: the plain form's pair beside the weight's own product on the input a1 a2 x
    oft = 4,    // oft_blocks: the output's rows turned by a block-diagonal orthogonal matrix (BOFT: 4 dimensions)
    diff = 5,   // a whole delta (.diff, .set_weight, .w_norm): y += scale * x Delta^T
    bias = 6,   // a bias's delta alone (.diff_b, .b_norm): y += scale * Db, no weight changed
};

/// The kind's name, as the reports and the errors write it.
const char* lora_kind_name(LoraKind k);

/// LoKr's geometry: w1 [out_l][in_m] and w2 [out_k][in_n], the file's matrices or its two factors' products (in
/// `extra`), the delta their Kronecker product is at the weight's rows out_l * out_k and input columns in_m * in_n.
struct LokrShape {
    int64_t out_l = 0;   // w1's rows
    int64_t in_m = 0;    // w1's columns, the input's groups
    int64_t out_k = 0;   // w2's rows
    int64_t in_n = 0;    // w2's columns
    bool empty() const { return out_l <= 0 || in_m <= 0 || out_k <= 0 || in_n <= 0; }
};

/// OFT's geometry, a BOFT's too: the output's rows are turned by `blocks` block-diagonal rotations of `bs` rows each,
/// `stages` of them one after another (an OFT file's one).  The weight's rows are blocks * bs.
struct OftShape {
    int64_t blocks = 0;   // the rotations of one stage
    int64_t bs = 0;       // a rotation's rows
    int64_t stages = 0;   // a BOFT's stages; an OFT file's one
    bool boft = false;    // the file's crossing of the rows is a BOFT's (four dimensions), not an OFT's
    bool empty() const { return blocks <= 0 || bs <= 0 || stages <= 0; }
    int64_t rows() const { return empty() ? 0 : blocks * bs; }
    /// A stage's rows in the order it turns them: the block's rows are read and written `bs` at a time from the
    /// weight's rows m1 * group * k + j * r_b + i / group + (i % group) * k, with m1 = block / steps, j = block %
    /// steps and steps = k / r_b (the butterfly's reshapes of the rows into four dimensions; an OFT's group 1, k = r_b
    /// = bs reads them in place).  A BOFT's m = 1 crosses them too: it is not an OFT.
    int64_t group() const { return boft ? 2 : 1; }
    int64_t r_b() const { return boft ? bs / 2 : bs; }
    int64_t k(int64_t stage) const { return boft ? (int64_t(1) << stage) * (bs / 2) : bs; }
};

/// A GLoRA's tensors in one of the form's two readings: the plain form's pair (a [rank][cols], b [rows][rank]) and the
/// input's two factors as the file writes them - `in` [rank][cols], the one applied to x first, and `out` [cols][rank]
/// (both the rows of the products that take them: v = out (in x)) - FP16, with the shapes the reading gives.
struct GloraReading {
    int64_t rows = 0, cols = 0, rank = 0;
    std::vector<uint16_t> a, b, in, out;
    bool empty() const { return rows <= 0 || cols <= 0 || rank <= 0; }
};

/// One weight's delta, on the host: the tensors the file gave, A and B in FP16 (half the RAM of FP32, which is what
/// the GPU takes), the scales in FP32.  The other forms' tensors are kept in `extra` by their role (`hada_w1_a`,
/// `oft_blocks`, `diff`, ...) and FP32, for the forms the engine learns to apply; a LoKr's materialized w1 and w2 are
/// in `extra` under "lokr_w1" and "lokr_w2".
struct LoraEntry {
    std::string key;                  // the DiT's name, e.g. "transformer_blocks.0.attn.to_q"
    LoraKind kind = LoraKind::lora;
    int64_t rows = 0, cols = 0, rank = 0;
    bool alpha_set = false;
    float alpha = 0.0f;               // from ".alpha" (1.0 where the file has none); an OFT's: its Q's largest norm
    LokrShape lokr;                   // LoraKind::lokr: the factors' shape, `rows` and `cols` their product's
    OftShape oft;                     // LoraKind::oft: the blocks' shape, `rows` their rows * bs and `cols` 0
    /// alpha / rank, the file's own scale (the caller's strength multiplies it).  LoKr's rank is the factorization's
    /// (the file's alpha is divided by it, and by nothing where both of its matrices are whole).
    float scale() const { return alpha_set && rank > 0 ? alpha / static_cast<float>(rank) : 1.0f; }
    std::vector<uint16_t> a, b;       // lora: a [rank][cols], b [rows][rank], row-major
    /// glora: the reading the file's shapes said (the older of the two where both fit), `rows`, `cols` and `rank` its
    /// shapes; `a` and `b` above are the plain form's only.
    GloraReading glora;
    /// glora: the other reading, where the four tensors fit both - a GLoRA's two readings differ only in which of them
    /// is the input's path and which the plain pair, and a square weight's tensors fit each, so the weight's rows and
    /// columns choose between them (the older reading where its shape is the weight's, else this one; done by
    /// `lora_glora_for_weight`).
    /// Empty: the file's shapes said one reading.
    GloraReading glora_other;
    std::vector<uint16_t> delta;      // diff: the weight's own delta [rows][cols], row-major, FP16 (a set_weight's:
                                      // the weight it sets, what the strength interpolates towards); a LoHa's is
                                      // made from the factors in `extra` when it is bound (its delta is the weight's
                                      // size and is not kept on the host)
    bool replaces = false;            // the delta is a set_weight's (the weight it sets), not a value added to it
    std::vector<float> dora;          // dora_scale (DoRA): over the rows, else over the columns
    bool dora_over_rows = true;
    std::vector<float> bias;          // a bias's delta (.diff_b/.b_norm): `rows` values, FP32; empty: none.  A
                                      // LoraKind::bias entry's whole delta is this (its `cols` is 0)
    std::vector<float> rescale;       // an OFT's rescale: the turned rows' factors, FP32; empty: none
    std::map<std::string, std::vector<float>> extra;   // the other forms' tensors, FP32, by role
    std::map<std::string, std::vector<int64_t>> shapes;   // their shapes, the file's order
};

/// An adapter a request names that the engine does not take - its file, its forms, its rows against the DiT's, a model
/// that takes none: the request's own fault, which strata_image says as `ERROR lora: ...` and the server answers 400.
struct LoraRejected : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// An adapter file, read: its entries, its metadata.
class LoraFile {
public:
    /// Reads `path` (a safetensors or a GGUF file, by its first bytes).  Throws std::runtime_error saying what is
    /// wrong (the file, the key), as the image engine's other readers do.
    static std::unique_ptr<LoraFile> open(const std::string& path);

    const std::vector<LoraEntry>& entries() const { return entries_; }
    const std::string& path() const { return path_; }
    /// The container's string metadata (safetensors' __metadata__; a GGUF's string fields), for the listing.
    const std::map<std::string, std::string>& metadata() const { return meta_; }
    std::string meta(const std::string& key) const;
    /// What the entries hold on the host.
    uint64_t host_bytes() const;
    /// A copy holding the text encoder's entries (`text`) or the others, the same path and metadata.
    std::shared_ptr<const LoraFile> share(bool text) const;

private:
    std::string path_;
    std::vector<LoraEntry> entries_;
    std::map<std::string, std::string> meta_;
};

/// One adapter a request applies: its file, read once and shared while its request runs, and the strength its delta is
/// scaled by (the file's own alpha / rank is the entry's).  `blocks`: the strength's factor a DiT block (block b's
/// weights take scale * blocks[b]; the weights outside the blocks take scale), as A1111's LoRA Block Weight gives it,
/// one a block of the DiT; empty: 1 for every block.
struct LoraUse {
    std::shared_ptr<const LoraFile> file;
    float scale = 1.0f;
    std::vector<float> blocks;
};

/// Whether a file's key names this weight: the name as the DiT writes it, or as sd-scripts' `lora_unet_` keys and the
/// LyCORIS form's `lycoris_` keys write it - the DiT's name with every dot an underscore and its own underscores kept
/// (`attn.to_q` reaches `attn_to_q`).  One table answers both: the DiT's own names, each also written in the underscore
/// form.  serve/loras.py's kohya_name brings the same keys to the same
/// names for the listing, so a file the listing shows is a file the DiT binds.
bool lora_key_names(const std::string& key, const std::string& name);
/// A key a text encoder's weight has: sd-scripts' lora_te (lora_te1_, lora_te2_), diffusers' text_encoder., te.
bool lora_is_text_encoder_key(const std::string& key);
/// The text encoder's own name of such a key with its dots for underscores ("lora_te_layers_0_self_attn_q_proj" and
/// "text_encoder.model.layers.0.self_attn.q_proj" -> "layers.0.self.attn.q.proj"): the prefixes off, sd-scripts'
/// underscores and the file's dots alike, which a text encoder compares with its own names written the same way.
std::string lora_text_encoder_name(const std::string& key);
/// The adapters split by the part they change: `dit` the files' DiT entries, `te` their text encoder's (a file with
/// none of one part is left out of that list; a file of both, a copy of each share).  The copies are kept for the same
/// files, so a request's split is the last one's when its files are.
void lora_split(const std::vector<LoraUse>& in, std::vector<LoraUse>& dit, std::vector<LoraUse>& te);
/// A set's key: its files' paths, strengths and block factors in order ("" for none), what a part compares to know the
/// set changed.
std::string lora_key_of(const std::vector<LoraUse>& loras);

/// The bytes the adapters take on the GPU: their entries' tensors (FP16), and the products' scratch of T rows (T times
/// the most values an entry's delta holds a row, FP32 and FP16 each): what a request's budget puts beside the model.
uint64_t lora_gpu_bytes(const std::vector<LoraUse>& loras, int64_t T);

/// The values an entry's delta holds for each row of the product's scratch (FP32): a plain LoRA's rank, LoKr's
/// in_m * out_k (its product's intermediate [T][in_m][out_k]), a whole delta's (or a LoHa's) cols where the input's
/// rows must be scaled before its product (a W8A8 product's row scale, or a strength other than 1).  A GLoRA's is the
/// more of its rank and its columns (its input's two products, then the weight's own product's input).  A DiT sizes
/// its deltas' scratch by the largest.
int64_t lora_scratch_cols(const LoraEntry& e);

/// The tokens a GLoRA's own path takes at a time: the weight's product of its input is made a band of the product's
/// rows at a time, so that the table it is added from stays a few MiB whatever the weight's rows (128 of the largest
/// DiT weights' 24576 rows is 12 MiB).
constexpr int64_t kLoraGloraRows = 128;

/// One entry's tensors on the GPU (FP16: a plain LoRA's A and B, a LoKr's w1 and w2), and the entry's delta applied
/// to a product's output.
class LoraDeviceEntry {
public:
    LoraDeviceEntry() = default;
    ~LoraDeviceEntry();
    LoraDeviceEntry(const LoraDeviceEntry&) = delete;
    LoraDeviceEntry& operator=(const LoraDeviceEntry&) = delete;

    /// Puts the entry's A and B on the stream's GPU.  `row_coef` (null: 1), one a row of the entry's, folds into B
    /// (the plain form's DoRA: the rows' c, a product's output scaled there; a LoKr's rows are the Kronecker product's
    /// and the caller refuses its DoRA).  False with err on a failure (the memory, a form the GPU cannot take: the
    /// shapes are the file's own and were checked when it was read).
    bool upload(const LoraEntry& e, void* stream, std::string& err, const float* row_coef = nullptr);
    void release();
    /// A plain LoRA's (A and B): the form a weight's bake folds in.
    bool plain() const { return kind_ == LoraKind::lora && rank_ > 0 && a_ != nullptr && b_ != nullptr; }
    const uint16_t* a() const { return a_; }
    const uint16_t* b() const { return b_; }
    int64_t rank() const { return rank_; }
    int64_t cols() const { return cols_; }
    int64_t rows() const { return rows_; }
    uint64_t bytes() const { return bytes_; }

    /// y[t][row0 + c] (row stride ldy) += scale * (x16[t] A^T) B^T, t < T, for the entry's rows
    /// [src_row, src_row + n) into the weight's rows [row0, row0 + n) (a fused weight's half, or the whole).  An
    /// LoKr's rows are its Kronecker product's; the delta is the same product's, its own way.  A whole delta's (and a
    /// LoHa's, its factors' product made when it was uploaded) is scale * (x16[t] Delta^T) over the entry's rows, the
    /// input scaled into `tmp16` (T * cols values) where the strength is not 1 or a row scale is passed.  A W8A8
    /// product's output is its weights' and activations' scales applied (no row scale of the input): `row_scale`
    /// (null: none) is the input rows' scale, which the base product's sum does not carry and this delta must.
    /// `tmp_f32` and `tmp16` hold T * lora_scratch_cols(e) values each (grown by the caller).
    void add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
             float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16,
             void* stream) const;

    /// A GLoRA's own path: v16[t] = scale * row_scale[t] * (glora_out (glora_in x16[t])), [T][cols], the input the
    /// weight's product is run on a second time (the caller's own product of it: the weight's path, added to y).
    /// `tmp_f32` and `tmp16` hold T * lora_scratch_cols(e) values each; the input is left in `tmp16`'s first T * cols.
    void glora_input(const uint16_t* x16, int64_t T, float scale, const float* row_scale, prefill::Gemm& gemm,
                     float* tmp_f32, uint16_t* tmp16, void* stream) const;

    /// y[t][row0 + c] (row stride ldy) += scale * bias[src_row + c], t < T: the bias's delta (a "diff" patch
    /// of the weight's bias), added where the DiT adds the weight's own bias.  No bias: nothing.  A
    /// LoraKind::bias entry has no weight's delta: this is its whole patch.
    void add_bias(float* y, int64_t T, int64_t ldy, int64_t src_row, int64_t row0, int64_t n, float scale,
                  void* stream) const;
    bool has_bias() const { return bias_ != nullptr; }

    /// y[t][row0 + c] (row stride ldy) = (1 - scale) y[t][row0 + c] + scale * rescale[src_row + c] * (r y)[t][row0
    /// + c], t < T, over the entry's own rows (an OFT's, or a BOFT's stages one after another): the product's output
    /// turned, the rotation of the weight interpolated once.  `tmp` holds T * rows values (the turned rows' own
    /// table: the engine's scratch).  The part's rows must be the entry's (n == rows()).
    void rotate(float* y, int64_t T, int64_t ldy, int64_t src_row, int64_t row0, int64_t n, float scale, float* tmp,
                void* stream) const;

private:
    void add_lokr(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
                  float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, void* stream) const;
    void add_diff(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
                  float scale, const float* row_scale, prefill::Gemm& gemm, uint16_t* tmp16, void* stream) const;

    LoraKind kind_ = LoraKind::lora;
    uint16_t* a_ = nullptr;   // lora: A [rank][cols]
    uint16_t* b_ = nullptr;   // lora: B [rows][rank]
    uint16_t* w1_ = nullptr;  // lokr: w1 [out_l][in_m]
    uint16_t* w2_ = nullptr;  // lokr: w2 [out_k][in_n]
    uint16_t* glora_in_ = nullptr;    // glora: glora_in [rank][cols]
    uint16_t* glora_out_ = nullptr;   // glora: glora_out [cols][rank]
    uint16_t* diff_ = nullptr;   // diff: Delta [rows][cols]
    float* bias_ = nullptr;      // a bias's delta [rows], FP32
    float* rot_ = nullptr;       // oft: the rotations [stages][blocks][bs][bs], FP32
    float* rescale_ = nullptr;   // oft: the turned rows' factors [rows], FP32
    LokrShape lokr_;
    OftShape oft_;
    int64_t rank_ = 0, cols_ = 0, rows_ = 0;
    uint64_t bytes_ = 0;
};

/// One share of a weight's delta: the entry's rows [src_row, src_row + n) at the weight's rows [row0, row0 + n), times
/// `scale` (the file's and the request's).  Two files may change one weight: the weight takes a part each.
struct LoraPart {
    std::shared_ptr<const LoraDeviceEntry> entry;
    int64_t row0 = 0, n = 0, src_row = 0;
    float scale = 1.0f;
    float bias_scale = 0.0f;   // the entry's bias's delta (none: 0), times the request's strength only (no alpha)
    bool dora = false;   // a DoRA's: its base factors are c's, which the weight's other patches do not know
    bool oft = false;    // an OFT's: it turns the product's output over the part's whole rows
    bool glora = false;  // a GLoRA's: the weight's own product of its input, over the part's rows
    bool replaces = false;   // a set_weight's: the weight it sets, which a patch before or after it would not keep
    /// DoRA: the base product's factor of each of the part's rows, 1 + strength (c - 1); a set_weight's: 1 - the
    /// strength.  Empty: the rows as the weight has them.  A row of the weight takes one part's factors.
    std::vector<float> base;
};

/// The weight's own product of a GLoRA's input, added to the product's output: `output(v16, t0, rows, row0, n)` puts
/// the weight's rows [row0, row0 + n) product of v16's [rows][cols] (the tokens t0 .. t0 + rows) into the product's
/// output at those rows: the DiT writes it, its weight's own path (an FP16 or a quantized weight's) and its own table
/// of the rows.
using LoraGloraOutput = std::function<void(const uint16_t* v16, int64_t t0, int64_t rows, int64_t row0, int64_t n)>;

/// The deltas one weight adds to its products, as the DiT bound them.  A weight without one has no layer.
class LoraLayer {
public:
    LoraLayer() = default;
    LoraLayer(const LoraLayer&) = delete;
    LoraLayer& operator=(const LoraLayer&) = delete;

    std::vector<LoraPart> parts;
    bool empty() const { return parts.empty(); }
    /// True where a part turns the product's output (its bias is then added after the deltas: a rotation of the
    /// weight's rows is not one of the weight's bias).
    bool has_oft() const;
    /// True where a part is a GLoRA's (the weight's product is run a second time, on its input).
    bool has_glora() const;
    /// The GPU memory the parts' entries hold (and the base product's factors).
    uint64_t bytes() const;

    /// True where every part is a plain LoRA's (no DoRA, OFT, GLoRA, set_weight or bias delta): their deltas can be
    /// folded into the weight once (bake) instead of added to each product.
    bool bakeable() const;
    /// The weight [n][k] at `w` on the stream's GPU plus the parts' deltas (scale B A over each part's rows), in place:
    /// `type` 1 FP16, -8 W8A8's (each group of kQiRotate of a row times the normalized Hadamard matrix, the rows'
    /// int8 values then their n scales: the delta turned the same way, the rows quantized again), 8 Q8_0 (its blocks
    /// quantized again); the 8-bit ones rounded with a fixed dither, as a delta under half a step would round away.  `tmp` holds n * k floats, `at16` k * the largest rank values.  bakeable() layers only.
    void bake(int type, void* w, int64_t n, int64_t k, prefill::Gemm& gemm, float* tmp, uint16_t* at16,
              void* stream) const;

    /// The base product's factors over all of the weight's rows, 1 where no DoRA part changes the row: made from the
    /// parts (`rows`: the weight's rows), on the GPU once the DiT has bound them all.  null: no part is a DoRA's.
    bool build_base(int64_t rows, void* stream, std::string& err);
    bool has_base() const { return base_ != nullptr; }
    const float* base() const { return base_; }

    /// y[t][ldy] += the parts that cover the weight's rows [w_row0, w_row0 + w_rows), each into its own rows there,
    /// and the parts' bias deltas.  `tmp_f32` and `tmp16` hold T * rank values each (grown by the caller).
    void add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t w_row0, int64_t w_rows,
             const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream) const;

    /// The GLoRA parts' own path (their plain pair is `add`'s), a band of the tokens at a time (`kLoraGloraRows`): the
    /// input v = scale * row_scale * (glora_out^T (glora_in^T x)), and `output` puts the weight's product of it over
    /// the part's rows the product wrote (the caller's y, which its own callback adds to).  `tmp_f32` and `tmp16` as
    /// `add`'s (the input is left in tmp16); `w_row0` and `w_rows` are the rows the product wrote.
    void add_glora(const uint16_t* x16, int64_t T, int64_t w_row0, int64_t w_rows, const float* row_scale,
                   prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream,
                   const LoraGloraOutput& output) const;

    ~LoraLayer();

private:
    float* base_ = nullptr;
    int64_t base_rows_ = 0;
    bool base_built_ = false;
};

/// The GLoRA entry as the reading a weight of `rows` x `cols` takes it: `&e` where its shapes are the weight's, or
/// `&alt` filled with the other reading where the four tensors fit both (a square weight's), or `&e` where neither is
/// (the caller's check of the shapes refuses it).  The entry's alpha divides the reading's own rank.
const LoraEntry* lora_glora_for_weight(const LoraEntry& e, int64_t rows, int64_t cols, LoraEntry& alt);

/// y[t][row0 + c] += src[t][c], t < T (row stride ldy): the weight's product of a GLoRA's input, made into a table of
/// the product's rows, added to the product's output.
void lora_output_add(float* y, int64_t T, int64_t ldy, int64_t row0, const float* src, int64_t n, void* stream);

/// One DoRA part's rows: `norms` (the part's n rows' composed norms ||W_r + (alpha/rank) (BA)_r||, FP32,
/// dit_dora_norms) -> the delta's factor of each of the entry's rows [src_row, src_row + n),
/// c = dora_scale / (norm + eps) in `coef`, and the base product's 1 + strength (c - 1) in `base` (PEFT's DoRA at
/// strength 1: the base rows times c, the delta's times c and the file's alpha / rank; the strength blends the
/// base's factor between 1 and c as it always has).  False with err for the DoRA the engine does not take: a scale
/// over the columns (the other axis, whose norm is of the weight with the delta in it, and the columns are an
/// input's), or a form whose rows a product's output does not have (a LoKr's are its Kronecker product's).
bool lora_dora_rows(const LoraEntry& e, int64_t src_row, int64_t n, const std::vector<float>& norms,
                    float strength, std::vector<float>& coef, std::vector<float>& base, std::string& err);

/// The bound part whose rows overlap [row0, row0 + n) of a weight's, where one of the two is a DoRA's, an OFT's, a
/// GLoRA's or a set_weight's: the two cannot be composed (patched in the files' order, a DoRA's
/// c, an OFT's turning, a GLoRA's read of the weight before its patch, and whether a set_weight keeps the other patch
/// would depend on that order), so the caller refuses the set.  null: no such part.
const LoraPart* lora_dora_overlap(const LoraLayer* layer, int64_t row0, int64_t n, bool dora, bool oft = false,
                                  bool glora = false, bool replaces = false);

/// Why a part's rows take no other adapter's (`lora_dora_overlap`'s part, and the forms of the entry being bound
/// against it): the sentence a caller's error ends with.
std::string lora_patch_clash_why(const LoraPart& p, bool oft = false, bool glora = false, bool replaces = false);

/// The host tensors of `e` as FP16 (a file's BF16, F16 or F32), for a test or a form built by hand.
void lora_to_f16(const std::vector<float>& f32, std::vector<uint16_t>& out);

}  // namespace strata::image