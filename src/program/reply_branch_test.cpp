// (asserts stay on in a Release build: the test is its own check)
#undef NDEBUG
#include "strata/program/reply_branch.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using strata::program::live_tail_is_replaced_reply;

namespace {
constexpr int64_t S = 900;   // <|im_start|>
constexpr int64_t E = 901;   // <|im_end|>
constexpr int64_t SYS = 10, USER = 11, ASSIST = 12;

std::vector<int64_t> cat(std::initializer_list<std::vector<int64_t>> parts) {
    std::vector<int64_t> v;
    for (const auto& p : parts) v.insert(v.end(), p.begin(), p.end());
    return v;
}
std::vector<int32_t> to32(const std::vector<int64_t>& v) { return std::vector<int32_t>(v.begin(), v.end()); }
}  // namespace

int main() {
    const std::vector<int64_t> sys = {S, SYS, 1, 2, 3, E};
    const std::vector<int64_t> user1 = {S, USER, 4, 5, 6, E};
    const std::vector<int64_t> head = {S, ASSIST, 7};                 // the generation header (<|im_start|>assistant\n...)
    const std::vector<int64_t> reasoning = {20, 21, 22, 23};
    const std::vector<int64_t> answer = {30, 31, 32};
    const std::vector<int64_t> user2 = {S, USER, 8, 9, E};
    const auto history = cat({sys, user1});
    const int64_t turn = static_cast<int64_t>(history.size());       // the turn checkpoint: [0, turn)
    // live: the last request's prompt + its reply (reasoning and answer), the last emitted token not fed
    const auto live = to32(cat({history, head, reasoning, answer}));
    int cases = 0;

    // (b) reasoning dropped: the history renders the assistant turn with the answer only, then the next user turn
    assert(live_tail_is_replaced_reply(live, cat({history, {S, ASSIST}, answer, {E}, user2, head}), turn, S)); ++cases;
    // (c) one word of the answer changed
    assert(live_tail_is_replaced_reply(live, cat({history, {S, ASSIST}, {30, 99, 32}, {E}, user2, head}), turn, S)); ++cases;
    // regenerate: the same history again, a new assistant turn opens where the dropped one was
    assert(live_tail_is_replaced_reply(live, cat({history, head}), turn, S)); ++cases;
    // the live branch was only a prompt read (prefill), nothing generated past the header
    assert(live_tail_is_replaced_reply(to32(cat({history, head})), cat({history, {S, ASSIST}, answer, {E}, user2, head}), turn,
                                       S)); ++cases;

    // a new USER turn where the reply was (the client left the reply out entirely): another kind of request, park as before
    assert(!live_tail_is_replaced_reply(live, cat({history, user2, head}), turn, S)); ++cases;
    // the live tail holds more than one turn (a tool result, a user turn after the reply): not just one reply
    assert(!live_tail_is_replaced_reply(to32(cat({history, head, answer, {E}, user2, head, answer})),
                                        cat({history, {S, ASSIST}, answer, {E}, user2, head}), turn, S)); ++cases;
    // resume not at a turn opening of the live branch (a periodic or root checkpoint in the middle of a message)
    assert(!live_tail_is_replaced_reply(live, cat({history, head}), turn - 2, S)); ++cases;
    // the prompt differs before resume (another conversation): never
    {
        auto other = cat({history, head});
        other[2] = 77;
        assert(!live_tail_is_replaced_reply(live, other, turn, S)); ++cases;
    }
    // the live branch ends at the checkpoint (nothing past it) or one token past it: nothing to say about a reply
    assert(!live_tail_is_replaced_reply(to32(history), cat({history, head}), turn, S)); ++cases;
    assert(!live_tail_is_replaced_reply(to32(cat({history, {S}})), cat({history, head}), turn, S)); ++cases;
    // the prompt ends right at the turn opening (one token past resume): no role to compare
    assert(!live_tail_is_replaced_reply(live, cat({history, {S}}), turn, S)); ++cases;
    // no turn token configured, resume 0
    assert(!live_tail_is_replaced_reply(live, cat({history, head}), turn, -1)); ++cases;
    assert(!live_tail_is_replaced_reply(live, cat({history, head}), 0, S)); ++cases;
    std::cout << "SUMMARY reply_branch cases=" << cases << " passed\n";
    return 0;
}
