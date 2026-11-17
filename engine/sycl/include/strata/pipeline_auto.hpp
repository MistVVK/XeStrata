// The default for --pipeline-windows when it is not given (host-only logic, no SYCL runtime: tested by
// src/core/pipeline_auto_test.cpp).  On for a layer split of exactly two Arc Battlemage (Xe2) cards, with --serve and the
// MTP drafter; off for everything else.  The caller finds `battlemage_*` from the devices' architecture ids
// (strata::intel_gpu_gen), never from a name.
#pragma once

namespace strata {

struct PipelineAutoInputs {
    bool serve = false;            ///< --serve
    bool multi_gpu = false;        ///< a layer split across GPUs
    bool split_same = false;       ///< the split puts every stage on one GPU
    unsigned later_stages = 0;     ///< GPUs after the first one: 1 = a split of exactly two stages
    bool auto_off = false;         ///< STRATA_PIPELINE_AUTO=0
    bool has_mtp = false;          ///< --mtp given
    int spec = 0;                  ///< --spec after the engine's own widening (>= 2 means a draft window)
    int batch = 0;                 ///< --batch slots
    bool peer_device = false;      ///< --peer-device
    bool helper_caches = false;    ///< --expert-cache-device1..3
    bool first_is_battlemage = false;   ///< stage 0's GPU is a Battlemage card (architecture id)
    bool later_is_battlemage = false;   ///< the second stage's GPU is a Battlemage card
};

/// True when --pipeline-windows 2 is the default for this run.  Single card, an Arc A-series card (Alchemist), any other
/// vendor, three or more stages (the SYCL pipeline runs exactly two), --batch, --peer-device and the helper caches stay serial.
inline bool pipeline_auto_on(const PipelineAutoInputs& in) {
    return in.serve && in.multi_gpu && !in.split_same && in.later_stages == 1 && !in.auto_off && in.has_mtp &&
           in.spec >= 2 && in.batch == 0 && !in.peer_device && !in.helper_caches && in.first_is_battlemage &&
           in.later_is_battlemage;
}

}  // namespace strata
