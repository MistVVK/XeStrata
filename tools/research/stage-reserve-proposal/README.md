# Layer-split reserve: the unused window-term proposal

Moved here from `include/strata/program/stage_reserve.hpp` and `tests/program/stage_reserve_test.cpp` in cleanup step 4.2.
Nothing in the engine calls it; the default placement uses `reserve_mib` and `room_bytes` in the header, which are unchanged.

## What it is

Issue #3702: on an unequal pair (RTX 3060 + RTX 5060 Ti) the last card was left with 68 MiB after loading although 700 + 96 MiB were reserved for it, and the verify windows' graphs (captured on first use, 4-6 MiB each and once 42 MiB) then ran it out of memory. The proposal makes the window term of a later stage grow with the number of window sizes:

- `proposed_window_mib(max_window_t)` = `kWindowMib` + (sizes - 1) * `kGraphMib` (6 MiB) + `kInstantiateBurstMib` (48 MiB)
- `proposed_reserve_mib(in, max_window_t)` = `reserve_mib(in)` with that term on a later stage

For `--spec 4` (windows of 1..6 tokens) the term is 174 MiB instead of 96. It changes cache capacity, so it can change speed and it is not a cleanup change.

## What it needs before it could become the default

The model gate `phaseA-tests/gh027/gates/3702-later-reserve.sh` (private lab folder, not in this repository), and an A/B on a mixed-card pair.

## Files

- `proposal.hpp`: the two functions and their two constants (text unchanged).
- `proposal_test.cpp`: the checks that used to sit in `stage_reserve_test.cpp`.

Run the checks (from the repository root, any C++17 compiler):

    c++ -std=c++17 -Iinclude -Iresearch/stage-reserve-proposal research/stage-reserve-proposal/proposal_test.cpp -o proposal_test && ./proposal_test
