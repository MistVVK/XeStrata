// metal/tests/tqp_parity.cpp - the TQP5 / TQP6 / TQP7 pair trellis (ggml types 147-149, Gyro-L) against the
// agention fork's CPU reference: real blocks of each type from a Gyro-L GGUF, dequantized by Strata's own
// reference (ref::dequant, metal/src/ref_quant.cpp) and by the fork's dequantize_row_tqp* (ggml-quants.c),
// compared bit for bit.  The fork's ggml is not linked here: ggml-quants.c's tqp section is compiled straight
// out of the checkout (STRATA_FORK_GGML), with the fork's own codebook table extracted from its ggml-common.h.
//
//   tqp_parity <model.gguf> [fork-ggml-src-dir]      fork dir: the ggml/src of the agention-llama checkout
//
// Without a GGUF it still checks the decoder against the format's definition (a third transcription, per
// expert_parity --decode-check) on random blocks.
#include "ref_quant.h"
#include "ggml-common-tq.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_lite.h"

// the fork's reference, compiled from its own source (see CMakeLists.txt): the declarations
extern "C" void ggml_ref_dequantize_row_tqp5(const void* x, float* y, int64_t k);
extern "C" void ggml_ref_dequantize_row_tqp6(const void* x, float* y, int64_t k);
extern "C" void ggml_ref_dequantize_row_tqp7(const void* x, float* y, int64_t k);

namespace {

// the fork's decoder written a third time, straight from the format's definition (ggml-common.h): step t's
// state = the 16 stream bits from bit (63 - t) * K, LSB first, circular modulo 64*K, masked to 15 bits;
// weights 2t, 2t+1 = the fp16 pair lut[(state * 0x9e3779b1) >> 21] times d.
void decode_def(int K, const uint8_t* b, float* y) {
    const float d = ref::f16_to_f32((uint16_t) (b[0] | (b[1] << 8)));
    const uint8_t* qs = b + 2;
    for (int t = 0; t < 64; ++t) {
        const int off = ((63 - t) * K) % (64 * K);
        uint32_t state = 0;
        for (int bit = 0; bit < 16; ++bit) {
            const int pos = (off + bit) % (64 * K);
            state |= (uint32_t) ((qs[pos / 8] >> (pos % 8)) & 1) << bit;
        }
        const uint32_t pa = ((state & 0x7fffu) * 0x9e3779b1u) >> 21;
        y[2 * t + 0] = ref::f16_to_f32(tq_lut_f16[2 * pa]) * d;
        y[2 * t + 1] = ref::f16_to_f32(tq_lut_f16[2 * pa + 1]) * d;
    }
}

int check_type(int ty, const uint8_t* base, size_t nblocks, size_t stride, int K) {
    const int bb = ref::block_bytes(ty);
    long bad_ref = 0, bad_def = 0, n = 0;
    double max_err = 0.0;
    float want[128], got[128], def[128];
    for (size_t i = 0; i < nblocks; i += stride) {
        const uint8_t* b = base + i * bb;
        ref::dequant(ty, b, got, 128);
        switch (ty) {
            case ref::TQP5: ggml_ref_dequantize_row_tqp5(b, want, 128); break;
            case ref::TQP6: ggml_ref_dequantize_row_tqp6(b, want, 128); break;
            default:        ggml_ref_dequantize_row_tqp7(b, want, 128); break;
        }
        decode_def(K, b, def);
        for (int j = 0; j < 128; ++j) {
            if (got[j] != want[j]) ++bad_ref;
            if (def[j] != want[j]) ++bad_def;
            const double e = std::abs((double) got[j] - (double) want[j]);
            if (e > max_err) max_err = e;
        }
        ++n;
    }
    std::printf("  %-5s %8ld blocks  vs fork ref: %s (max |err| %.3g)  vs definition: %s\n",
                ref::type_name(ty), n, bad_ref ? "FAIL" : "bit-exact", max_err, bad_def ? "FAIL" : "bit-exact");
    return (bad_ref || bad_def) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        // no model: random blocks, still against the fork's compiled reference
        int bad = 0;
        for (int ty : {ref::TQP5, ref::TQP6, ref::TQP7}) {
            const int K = ty == ref::TQP5 ? 5 : ty == ref::TQP6 ? 6 : 7, bb = ref::block_bytes(ty);
            std::vector<uint8_t> raw((size_t) 4096 * bb);
            ref::random_blocks(ty, raw.data(), 4096, 128, 147 + ty);
            bad += check_type(ty, raw.data(), 4096, 1, K);
        }
        std::printf("tqp_parity: %s\n", bad ? "FAIL" : "PASS");
        return bad;
    }
    std::string err;
    gguf::File gf;
    if (!gf.open(argv[1], err)) { std::fprintf(stderr, "%s: %s\n", argv[1], err.c_str()); return 2; }
    if (argc > 2) std::printf("  fork reference: %s\n", argv[2]);
    int bad = 0, seen = 0;
    // the expert tensors of each type, wherever they live: Gyro-L mixes TQP6 (blk.0), TQP7 and TQP5
    // (the later blocks) across the same roles
    for (int ty : {ref::TQP5, ref::TQP6, ref::TQP7}) {
        const int K = ty == ref::TQP5 ? 5 : ty == ref::TQP6 ? 6 : 7;
        for (const auto& [name, t] : gf.tensors) {
            if (t.type != ty || name.find("ffn_") == std::string::npos || name.find("_exps.weight") == std::string::npos) continue;
            const size_t nb = t.bytes / ref::block_bytes(ty);
            const size_t stride = std::max<size_t>(1, nb / 4000);   // up to 4k blocks per tensor, spread over it
            std::printf("  %-5s %s\n", ref::type_name(ty), name.c_str());
            bad += check_type(ty, (const uint8_t*) gf.map + t.offset, nb, stride, K);
            ++seen;
        }
    }
    if (!seen) { std::fprintf(stderr, "%s: no TQP5/6/7 expert tensor found\n", argv[1]); return 2; }
    std::printf("tqp_parity: %s (%d tensors)\n", bad ? "FAIL" : "PASS", seen);
    return bad;
}
