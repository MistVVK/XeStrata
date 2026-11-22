#include "bmg_expert_window_plan.hpp"
#include "../bmg_prefill_options.hpp"
#include <cstdio>

using namespace strata::prefill::bmg_window;
int main(int argc, char** argv) {
    if (argc == 3) {
        using strata::bmg::Mode;
        const bool experts = std::strcmp(argv[1], "experts") == 0;
        const auto window = std::strcmp(argv[2], "auto") == 0 ? Mode::Auto
                          : std::strcmp(argv[2], "on") == 0   ? Mode::Enabled : Mode::Disabled;
        if (strata::bmg::experts_enabled() != experts || strata::bmg::window_mode() != window) {
            std::fprintf(stderr, "unexpected expert or window setting\n");
            return 1;
        }
        return 0;
    }

    struct Threshold {
        int rows, gate;
        bool down;
    };
    for (const auto c : {Threshold{31, 4, false},
                         {32, 4, false},
                         {33, 2, false},
                         {95, 2, false},
                         {96, 2, false},
                         {97, 4, false},
                         {127, 4, false},
                         {128, 4, false},
                         {129, 4, true},
                         {191, 4, true},
                         {192, 4, true},
                         {193, 2, false},
                         {767, 2, false},
                         {768, 2, false},
                         {769, 1, false}}) {
        if (gate_order(c.rows) != c.gate || down_order(c.rows) != c.down) {
            std::fprintf(stderr, "unexpected reduction order at %d rows\n", c.rows);
            return 1;
        }
    }
    struct Pair {
        int left, right;
        bool gate, down;
    };
    for (const auto p : {Pair{16, 16, true, true},
                         {16, 17, false, true},
                         {64, 64, false, true},
                         {64, 65, false, false},
                         {100, 30, true, false},
                         {129, 129, false, false},
                         {384, 384, true, true},
                         {384, 385, false, true},
                         {770, 771, true, true},
                         {0, 64, false, false}}) {
        const auto result = merge_products(p.left, p.right);
        if (result.gate != p.gate || result.down != p.down) {
            std::fprintf(stderr, "unexpected merge for %d + %d rows\n", p.left, p.right);
            return 1;
        }
    }
    return 0;
}
