// include/strata/prefill/equal_cut_rule.hpp - how a long prompt segment is cut into chunks (no GPU needed to check).
//
// A segment longer than its chunk was read in full chunks and a remainder: 20,931 tokens at a 10,240-token chunk as
// 10,240 + 10,240 + 451.  On one GPU every chunk at or above the stream floor streams each expert the card does not
// hold, and a remainder under the floor runs the staged walk, which on a card that streams most of its experts costs
// close to a full pass.  Reading the same segment in n = ceil(tokens / chunk) chunks of EQUAL size (3 x 6,977) keeps
// the number of chunks and gives every chunk the streamed walk.  Measured on an RTX 5070 12 GB, Q2_0 plain pack, build
// without MMQ: 0.923-1.001 of the time of today's cut at 14,000-32,000 tokens (0.937 at 20,932, 0.930 at 32,000); one
// chunk more than needed was slower (1.054-1.145), so the count stays ceil(tokens / chunk).
//
// The equal chunk must not be short (kEqualMinChunk).  A streamed chunk of a few thousand tokens cannot hide the copy of
// the experts it streams, and every chunk after the first stages its K/V from RAM.  RTX 4090 24 GB, IQ3_S, chunk 8,192,
// build without MMQ (pod, 11-14, warm text): 8,279 tokens as 2 x 4,140 took 3,517 ms against 2,579-2,686 for 8,192 + 87
// (wait copy 1,178 against 470-480 ms, K/V stage 450 against 4 ms, 34,852 experts streamed against 20,963); the equal cut
// against today's: 1.34 / 1.25 / 1.14 at 8,279 / 8,448 / 9,192 (equal chunks 4,140-4,596), 1.01-1.03 at 11,192 (5,596), 1.13 at
// 16,884 (5,628), 0.98 at 20,000 (6,667), 0.99 at 32,000 (8,000).  The RTX 5070 rows above all have equal chunks of
// 6,978 or more; its one smaller split (10,100 as 2 x 5,050) was 1.009.  So with a configured chunk of kEqualMinApplies
// or more, an equal chunk under kEqualMinChunk keeps today's cut; a smaller configured chunk keeps the rule as before.
//
// The rule applies where a segment's chunk is chosen (generate.cpp: the serve lend and generate's prompt) and only on
// one GPU: a layer split keeps today's cut (prefill.cpp pipeline_chunk: evening out a short tail added a full pass
// over the experts there, two RTX 4090: 25 % slower).  STRATA_PREFILL_EQUAL=0 keeps today's cut everywhere;
// STRATA_PREFILL_EQUAL=1 on a layer split keeps the older opt-in rule (#693, `opt_in_chunk`).
#pragma once

#include <cstdint>
#include <limits>

namespace strata::prefill::equal_cut {

/// Which cut a segment gets.
enum class Cut {
    today,    // full chunks and a remainder (the chunk of the request, rounded up to 256, at most the configured chunk)
    opt_in,   // the older opt-in (#693): equal chunks rounded up to 256, but a last chunk under the floor keeps today's cut
    equal,    // n = ceil(tokens / chunk) chunks of equal size, every one of them at or above the floor
};

/// `env`: STRATA_PREFILL_EQUAL as set (nullptr: unset); `one_gpu`: the segment is read by one GPU (no layer split, no peer
/// card, no --batch above one slot, no head alias).  "0" keeps today's cut; one GPU otherwise gets the equal cut; a
/// layer split gets the older opt-in rule only when it is asked for with "1".
constexpr Cut pick(const char* env, bool one_gpu) {
    if (env != nullptr && env[0] == '0') return Cut::today;
    if (one_gpu) return Cut::equal;
    return env != nullptr && env[0] == '1' ? Cut::opt_in : Cut::today;
}

/// Today's chunk for a segment of `tokens` (generate.cpp request_chunk): rounded up to the 256-token grid, at most `max_chunk`.
constexpr int64_t today_chunk(int64_t tokens, int64_t max_chunk) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255 ? tokens : ((tokens + 255) / 256) * 256;
    return rounded < max_chunk ? rounded : max_chunk;
}

/// The shortest equal chunk taken with a configured chunk of kEqualMinApplies or more (see the top of this file).
inline constexpr int64_t kEqualMinChunk = 6144;
inline constexpr int64_t kEqualMinApplies = 8192;
constexpr bool short_equal(int64_t per, int64_t max_chunk) {
    return max_chunk >= kEqualMinApplies && per < kEqualMinChunk;
}

/// The number of chunks a segment of `tokens` is read in at chunk `chunk` (prefill.cpp run_impl: c0 += C).
constexpr int64_t chunk_count(int64_t tokens, int64_t chunk) {
    return tokens <= 0 || chunk <= 0 ? 0 : tokens / chunk + (tokens % chunk != 0);
}

/// The older opt-in rule (#693), unchanged: n = ceil(tokens / max_chunk) chunks of ceil(tokens / n) rounded up to 256 (at most
/// `max_chunk`); one chunk, or a last chunk of today's cut under `floor`, keeps today's chunk.
constexpr int64_t opt_in_chunk(int64_t tokens, int64_t max_chunk, int64_t floor) {
    const int64_t plain = today_chunk(tokens, max_chunk);
    if (tokens <= 0 || max_chunk <= 0) return plain;
    const int64_t n = chunk_count(tokens, max_chunk);
    const int64_t last = tokens - (n - 1) * max_chunk;
    if (n <= 1 || last < floor) return plain;
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
    return rounded < max_chunk ? rounded : max_chunk;
}

/// The equal cut: a segment longer than `max_chunk` is read in n = ceil(tokens / max_chunk) chunks of E = ceil(tokens / n)
/// tokens.  The reading loop takes E at a time, so the first n - 1 chunks hold E tokens and the last one tokens - (n - 1) * E,
/// between E - (n - 1) and E: the remainder of the division is taken off the last chunk, at most n - 1 tokens.  E is not
/// rounded to the 256 grid (rounding would leave the last chunk up to (n - 1) * 255 tokens short).  E <= max_chunk, so every
/// buffer stays within the size laid out for max_chunk, and exactly n chunks are read (never more than today's cut).  A
/// segment that fits one chunk keeps today's chunk; so does a cut whose last chunk would fall under `floor` (every chunk of the
/// equal cut takes the streamed walk, as today's full chunks do).
constexpr int64_t equal_chunk(int64_t tokens, int64_t max_chunk, int64_t floor) {
    const int64_t plain = today_chunk(tokens, max_chunk);
    if (tokens <= max_chunk || max_chunk <= 0) return plain;
    const int64_t n = chunk_count(tokens, max_chunk);
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t last = tokens - (n - 1) * per;
    if (per > max_chunk || last < floor || last <= 0) return plain;
    if (short_equal(per, max_chunk)) return plain;
    return per;
}

/// The equal chunk that only the kEqualMinChunk condition turned down (today's cut was handed back for it), 0 otherwise.
constexpr int64_t declined_equal(int64_t tokens, int64_t max_chunk, int64_t floor) {
    if (tokens <= max_chunk || max_chunk <= 0) return 0;
    const int64_t n = chunk_count(tokens, max_chunk);
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t last = tokens - (n - 1) * per;
    if (per > max_chunk || last < floor || last <= 0) return 0;
    return short_equal(per, max_chunk) ? per : 0;
}

/// Today's cut was handed back for a short equal chunk (declined_equal() > 0), but its loan does not fit while the equal
/// chunk's does: keep the equal cut rather than halving the chunk (#2090).  `today_need` / `equal_need`: the slots each
/// loan takes (0 or less: it cannot be laid out); `room`: the slots that can be lent (the cache minus what is kept).
constexpr bool keep_equal(int64_t today_need, int64_t equal_need, int64_t room) {
    const bool today_fits = today_need > 0 && today_need <= room;
    const bool equal_fits = equal_need > 0 && equal_need <= room;
    return !today_fits && equal_fits;
}

/// The chunk a segment of `tokens` is read in under `cut`, at most `max_chunk`; `floor`: stream_all_min_tokens().
constexpr int64_t chunk(Cut cut, int64_t tokens, int64_t max_chunk, int64_t floor) {
    return cut == Cut::equal ? equal_chunk(tokens, max_chunk, floor)
         : cut == Cut::opt_in ? opt_in_chunk(tokens, max_chunk, floor)
                              : today_chunk(tokens, max_chunk);
}

/// The equal cut's loan of expert-cache slots must not be larger than today's (a chunk at or below the auto scan's small_max
/// lays out a bigger ring than a larger chunk, generate.cpp #2090): `equal_slots` / `today_slots` for the two chunks; 0 or less
/// means "does not fit".  False: the segment keeps today's cut.
constexpr bool loan_ok(int64_t equal_slots, int64_t today_slots) {
    return equal_slots > 0 && today_slots > 0 && equal_slots <= today_slots;
}

}  // namespace strata::prefill::equal_cut
