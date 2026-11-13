#pragma once
#include <chrono>
#include <cstdlib>
// Only the pipeline owner thread calls these chain methods. The state belongs
// to its current MtpDrafter and is reset before every chain and request. No
// class definition or layout differs from the normal engine's archives.
namespace strata::private_demand {
struct State {
    const void* owner = nullptr;
    bool enabled = false, final = false;
    int submitted = 0, target = 0, limit = 0;
    long long chains = 0, requested = 0, actual = 0, continuations = 0, dead = 0;
    bool metrics = false;
    long long last_ready_ns = 0, reaction_ns = 0, submit_ns = 0, max_reaction_ns = 0;
    int ordinal = 0, T = 0, a = 0, n_force = 0;
    long long p = 0;
    int tokens[8] = {}, force[8] = {};
};
inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline State& state() { static thread_local State s; return s; }
inline State* get(const void* p) {
    auto& s = state(); return s.enabled && s.owner == p ? &s : nullptr;
}
inline void begin(const void* p, bool on) {
    auto& s = state(); const int ordinal = s.ordinal + int(on); s = State{};
    s.owner = p; s.enabled = on; s.ordinal = ordinal;
    s.metrics = std::getenv("STRATA_PRIVATE_MTP_DEMAND_LOG") || std::getenv("STRATA_PRIVATE_MTP_DEMAND_MICRO");
}
inline bool can_enqueue_s1(const void* p) {
    auto* s = get(p); return !s || s->submitted == 0 || s->final;
}
inline void drain(const void* p) {
    if (auto* s = get(p)) { s->target = s->limit; s->final = true; }
}
} // namespace strata::private_demand
