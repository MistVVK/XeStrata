#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

// --serve: is the live conversation's tail past `resume` only the reply this prompt replaces?
//
// A request that resumes from a checkpoint (not from the live session) parks the live branch first, so a later request
// can come back to it.  When the checkpoint is the live branch's own turn checkpoint - the history before its last
// assistant turn - and the prompt opens an assistant turn at that same place again, the client has re-sent the history
// with that reply dropped, edited or about to be regenerated (an OpenAI-style client sends every earlier reply back
// without its reasoning, so this is every next turn of such a client).  The branch it leaves holds nothing but that one
// reply, which no request of the same client sends back as it was: parking it copies the whole conversation to host
// memory on the request's path (~130-290 ms on an R9700) for a copy that only ever gets dropped as superseded.
//
// True when all of these hold:
//   - `live` continues past `resume` (live[resume] exists) with a turn opening there (live[resume] == turn_token) and
//     no other turn after it: what follows is one assistant turn, its header and the reply generated after it;
//   - the prompt opens a turn at the same place with the same role token (ids[resume] == turn_token,
//     ids[resume + 1] == live[resume + 1]): the same assistant turn, re-sent or opened again;
//   - live and the prompt agree on [0, resume) (the caller resumes from a checkpoint of exactly those tokens; checked
//     here again so the rule stands alone).
//   - the live branch has no picture at or past `resume` (`live_pictures`: where each of its pictures starts): a branch
//     whose tail holds a picture is parked as before (a chat request's reply holds none, so this is a guard).
template<class LiveTok, class PromptTok>
bool live_tail_is_replaced_reply(const std::vector<LiveTok>& live, const std::vector<PromptTok>& ids, int64_t resume,
                                 int64_t turn_token, const std::vector<int64_t>& live_pictures = {}) {
    if (turn_token < 0 || resume < 1) return false;
    for (const int64_t at : live_pictures)
        if (at >= resume) return false;
    const auto r = static_cast<std::size_t>(resume);
    if (live.size() < r + 2 || ids.size() < r + 2) return false;
    if (static_cast<int64_t>(live[r]) != turn_token || static_cast<int64_t>(ids[r]) != turn_token) return false;
    if (static_cast<int64_t>(live[r + 1]) != static_cast<int64_t>(ids[r + 1])) return false;
    for (std::size_t i = r + 1; i < live.size(); ++i)
        if (static_cast<int64_t>(live[i]) == turn_token) return false;
    for (std::size_t i = 0; i < r; ++i)
        if (static_cast<int64_t>(live[i]) != static_cast<int64_t>(ids[i])) return false;
    return true;
}

} // namespace strata::program
