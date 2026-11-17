// The kernel self-test of a card (HIP only; selftest.cu, rules and lines in hip/selftest_policy.hpp).
#pragma once
#include <string>

namespace fnx {
// runs the tests on device dev: "" when they all passed (covered: what ran, for the pass line), else the first failure in a few words
std::string selftest_card(int dev, std::string& covered);
}  // namespace fnx
