// SPDX-FileCopyrightText: 2026 recutita, MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/ep_kernels.hpp - expert parallelism over two GPUs (--expert-parallel-device).
// For each layer card 0 pushes the routed ids and q8_1 rows into the peer's inbox (ep_push); the peer copies them
// (ep_copy_peer), plans and computes the entries card 0 does not hold and sends their rows back (ep_send_rows) while
// card 0 computes its own, then card 0 waits for them (ep_wait) and merges both (ep_merge_rows) before its combine.
// A flag holds the window epoch each graph bumps first (a value baked in at capture would never change).  Memory the
// other card wrote is read with system-scope atomics: on the B70 a spin on a volatile load never saw it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// *ctr += 1 (device memory).
void ep_bump(uint32_t* ctr, void* stream);
/// Copy the ids and q8_1 rows (xq_bytes a multiple of 4) into the peer's inbox, then set the peer's flag to *ctr.
void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream);
/// Spin until *flag reaches *ctr; after `spin_max` reads give up and store `code` in *err (host memory).
void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream);
/// Copy `bytes` (a multiple of 4) the other card wrote.
void ep_copy_peer(void* dst, const void* src, size_t bytes, void* stream);
/// The peer: copy the rows of entries card 0 does not own (ids[r] not in `res0`) into card 0's memory, then set
/// card 0's flag to *ctr.  The last work-group to count itself on `done` sets the flag: with the flag in a later
/// kernel card 0 sometimes read a row before all of it had arrived (upstream Strata, on the B70).
void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream);
/// Card 0: parts[r] = 0 + hit_out[r] for its own entries (moe_hit_add's bits), peer_rows[r] for the others.
void ep_merge_rows(float* parts, const float* peer_rows, const float* hit_out, const int32_t* ids, const int32_t* res0,
                   int n, int64_t row, void* stream);

}  // namespace strata::kernels
