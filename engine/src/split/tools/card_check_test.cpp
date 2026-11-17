// src/card_check.hpp and src/peer_probe.hpp's CPU side: a card's compute capability against the build's arch;
// CUDA's and nvidia-smi's PCI addresses in sysfs's spelling, BAR1's size from real cards'
// /sys/bus/pci/devices/<bdf>/resource text, the text it refuses, and the BAR1-against-memory check the engine makes as
// it loads: prod's V100s pass, an RTX 3090 with a pre-ReBAR VBIOS (256 MiB) is refused, one at full ReBAR passes; and
// the start-up link probe's report and failure message. Exits 1 after reporting every difference.
//   bin/card_check_test
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../src/card_check.hpp"
#include "../src/model.hpp"
#include "../src/peer_probe.hpp"

namespace {

int fails = 0;
void fail(const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    ++fails;
}

// prod's card 0 (0000:3b:00.0), a Tesla V100-PCIE-32GB: the first four lines of its resource file
const char* V100 =
    "0x00000000ab000000 0x00000000abffffff 0x0000000000040200\n"
    "0x00000cb000000000 0x00000cb7ffffffff 0x000000000014220c\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x00000cb800000000 0x00000cb801ffffff 0x000000000014220c\n";
// read from an RTX 3090 in a desktop of this box's owner (0000:01:00.0, driver 580), whose VBIOS offers no Resizable
// BAR: BAR1 256 MiB. Not the tester's card, a look-alike of it: theirs had the same 256 MiB BAR1 under a pre-ReBAR
// VBIOS and hung their ring. The whole file
const char* RTX3090_PRE_REBAR =
    "0x00000000de000000 0x00000000deffffff 0x0000000000040200\n"
    "0x00000000c0000000 0x00000000cfffffff 0x000000000014220c\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x00000000d0000000 0x00000000d1ffffff 0x000000000014220c\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x000000000000e000 0x000000000000e07f 0x0000000000040101\n"
    "0x00000000000c0000 0x00000000000dffff 0x0000000000000212\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";
// the same card with BAR1 at 32 GiB, as the tester's cards were after a ReBAR VBIOS: made from the lines above, not
// read from a card
const char* RTX3090_REBAR =
    "0x00000000de000000 0x00000000deffffff 0x0000000000040200\n"
    "0x0000007800000000 0x0000007fffffffff 0x000000000014220c\n"
    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";

constexpr unsigned long long MiB = 1ull << 20, GiB = 1ull << 30;
// nvidia-smi's memory.total on these cards: at least CUDA's totalGlobalMem, so the hardest case for a pass
constexpr unsigned long long V100_VRAM = 32768 * MiB, RTX3090_VRAM = 24576 * MiB;

void bdf(const std::string& in, const std::string& want) {
    try {
        const std::string got = fnx::sysfs_bdf(in);
        if (got != want) fail("sysfs_bdf('" + in + "') = '" + got + "', want '" + want + "'");
    } catch (const std::exception& e) {
        fail("sysfs_bdf('" + in + "') threw: " + e.what());
    }
}
void bdf_refused(const std::string& in) {
    try {
        fail("sysfs_bdf('" + in + "') = '" + fnx::sysfs_bdf(in) + "', want it refused");
    } catch (const std::runtime_error&) {
    }
}
void bar1(const char* name, const std::string& text, unsigned long long want) {
    try {
        const unsigned long long got = fnx::bar1_bytes(text, name);
        if (got != want) fail(std::string(name) + ": BAR1 " + std::to_string(got) + " bytes, want " + std::to_string(want));
    } catch (const std::exception& e) {
        fail(std::string(name) + " threw: " + e.what());
    }
}
// text bar1_bytes must refuse, with an error that says `says`
void bar1_refused(const char* name, const std::string& text, const std::string& says) {
    try {
        fail(std::string(name) + ": BAR1 " + std::to_string(fnx::bar1_bytes(text, name)) + " bytes, want it refused");
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find(says) == std::string::npos)
            fail(std::string(name) + ": refused with '" + e.what() + "', want it to say '" + says + "'");
    }
}
void covers(const char* name, unsigned long long bar, unsigned long long vram, bool ok) {
    try {
        fnx::require_bar1_covers(0, "0000:01:00.0", name, bar, vram, 3);
        if (!ok) fail(std::string(name) + ": BAR1 " + std::to_string(bar >> 20) + " MiB for " + std::to_string(vram >> 20) + " MiB passed");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (ok) fail(std::string(name) + ": refused: " + msg);
        else if (msg.find(std::to_string(bar >> 20) + " MiB BAR1") == std::string::npos || msg.find("FNX_SKIP_BAR1_CHECK") == std::string::npos)
            fail(std::string(name) + ": the refusal does not name the BAR1 size and the skip: " + msg);
    }
}

// require_card_arch on a card (major.minor) against a build (arch); refused when `says` is not empty, saying it
void arch(const char* name, int major, int minor, int build, const std::string& says) {
    try {
        fnx::require_card_arch(0, name, major, minor, build);
        if (!says.empty()) fail(std::string(name) + " on an sm_" + std::to_string(build) + " build passed, want it refused");
    } catch (const std::runtime_error& e) {
        if (says.empty()) fail(std::string(name) + " on an sm_" + std::to_string(build) + " build refused: " + e.what());
        else if (std::string(e.what()).find(says) == std::string::npos)
            fail(std::string(name) + ": refused with '" + e.what() + "', want it to say '" + says + "'");
    }
}

// peer_report on rates (from, to, us) for ncard cards: want lines in all, each `says` found in the line it indexes
void report(const char* name, const std::vector<fnx::PeerRate>& r, int ncard, size_t lines,
            const std::vector<std::pair<size_t, std::string>>& says) {
    try {
        const std::vector<std::string> got = fnx::peer_report(r, ncard);
        if (got.size() != lines) fail(std::string(name) + ": " + std::to_string(got.size()) + " lines, want " + std::to_string(lines));
        for (const auto& [i, text] : says)
            if (i >= got.size() || got[i].find(text) == std::string::npos)
                fail(std::string(name) + ": line " + std::to_string(i) + " does not say '" + text + "'" + (i < got.size() ? ": " + got[i] : ""));
    } catch (const std::exception& e) {
        fail(std::string(name) + " threw: " + e.what());
    }
}
// peer_report must refuse, with an error that says `says`
void report_refused(const char* name, const std::vector<fnx::PeerRate>& r, int ncard, const std::string& says) {
    try {
        fnx::peer_report(r, ncard);
        fail(std::string(name) + ": reported, want it refused");
    } catch (const std::logic_error& e) {
        if (std::string(e.what()).find(says) == std::string::npos)
            fail(std::string(name) + ": refused with '" + e.what() + "', want it to say '" + says + "'");
    }
}
// a failure to report, its fields by name so no two numbers can trade places
fnx::PeerFailure failure(int from, int to, int ncard, bool copy, unsigned missing, size_t words, unsigned long long timeout_ms) {
    fnx::PeerFailure f{};
    f.from = from;
    f.to = to;
    f.ncard = ncard;
    f.copy = copy;
    f.missing = missing;
    f.words = words;
    f.timeout_ms = timeout_ms;
    return f;
}

}  // namespace

int main() {
    arch("Tesla V100-PCIE-32GB", 7, 0, 70, "");
    arch("Tesla PG500-216", 7, 0, 70, "");                     // 72 SMs: the hc read walks its 80 blocks of work on them
    arch("NVIDIA GeForce RTX 3090", 8, 6, 86, "");
    arch("NVIDIA GeForce RTX 3090", 8, 6, 70, "make ARCH=86");    // prod's build on a 3090: the PTX compiled at load
    arch("Tesla V100-PCIE-32GB", 7, 0, 86, "make ARCH=70");
    arch("NVIDIA GeForce RTX 4090", 8, 9, 86, "make ARCH=89");
    arch("NVIDIA A100-PCIE-40GB", 8, 0, 86, "make ARCH=80");
    arch("NVIDIA GeForce RTX 4090", 8, 9, 89, "");             // a build for its arch: refused only by what else it lacks
    arch("NVIDIA GeForce RTX 3080", 8, 6, 86, "");             // 68 SMs

    bdf("0000:3B:00.0", "0000:3b:00.0");        // cudaDeviceGetPCIBusId
    bdf("00000000:3B:00.0", "0000:3b:00.0");    // nvidia-smi
    bdf("0000:d8:00.0", "0000:d8:00.0");
    bdf("0001:01:1f.7", "0001:01:1f.7");
    bdf_refused("");
    bdf_refused("3B:00.0");
    bdf_refused("0000:3B:00");
    bdf_refused("0000:3B:00.0x");
    bdf_refused("10000:3B:00.0");
    bdf_refused("0000:3B:20.0");   // device numbers end at 0x1f
    bdf_refused("0000:3B:00.8");   // functions at 7
    // what sscanf's %x took: a sign, a value too wide for the field, a 0x, a space
    bdf_refused("0000:-ffffffffffffffff:00.0");
    bdf_refused("100000001:3b:00.0");
    bdf_refused("0000:0x3b:00.0");
    bdf_refused("0000: 3b:00.0");
    bdf_refused("0000:+3b:00.0");
    bdf_refused("0000:100:00.0");   // a bus number is two digits
    {
        unsigned long long v = 0;
        if (fnx::parse_hex("1", 17, true, v)) fail("parse_hex took a width of 17 digits, past what 64 bits hold");
    }

    bar1("V100", V100, 32 * GiB);
    bar1("RTX 3090, pre-ReBAR", RTX3090_PRE_REBAR, 256 * MiB);
    bar1("RTX 3090, ReBAR", RTX3090_REBAR, 32 * GiB);
    bar1("no 0x, upper case", "0 ff 40200\nC0000000 CFFFFFFF 14220C\n", 256 * MiB);
    bar1("no newline at the end", "0x0 0xff 0x40200\n0x0000000100000000 0x00000001ffffffff 0x14220c", 4 * GiB);

    bar1_refused("empty", "", "0 lines");
    bar1_refused("BAR0 alone", "0x00000000de000000 0x00000000deffffff 0x0000000000040200\n", "1 lines");
    bar1_refused("BAR1 unassigned", "0x00000000de000000 0x00000000deffffff 0x0000000000040200\n"
                                    "0x0000000000000000 0x0000000000000000 0x0000000000000000\n", "no address");
    bar1_refused("BAR1 I/O ports", "0x0 0xff 0x40200\n0x000000000000e000 0x000000000000e07f 0x0000000000040101\n", "not a memory range");
    bar1_refused("BAR1 backwards", "0x0 0xff 0x40200\n0x00000000d0000000 0x00000000c0000000 0x000000000014220c\n", "not a memory range");
    bar1_refused("two fields", "0x0 0xff 0x40200\n0x00000000c0000000 0x00000000cfffffff\n", "cannot read BAR1");
    bar1_refused("four fields", "0x0 0xff 0x40200\n0x00000000c0000000 0x00000000cfffffff 0x14220c 0x1\n", "cannot read BAR1");
    bar1_refused("not hex", "0x0 0xff 0x40200\nBAR1 at c0000000\n", "cannot read BAR1");
    // what sscanf's %llx took: signs (this one read as exactly 24 GiB and passed), 17 digits, a range from 0 that
    // wraps the size to 0
    bar1_refused("signed", "0x0 0xff 0x40200\n-0x600000000 -0x1 0x14220c\n", "cannot read BAR1");
    // without 0x a sign-permissive parser reads this as a plausible 4 GiB BAR1
    bar1_refused("signed, no 0x", "0x0 0xff 0x40200\n-1 -ffffffff 14220c\n", "cannot read BAR1");
    bar1_refused("17 digits", "0x0 0xff 0x40200\n0x10000000000000000 0x1ffffffffffffffff 0x14220c\n", "cannot read BAR1");
    bar1_refused("all of memory from 0", "0x0 0xff 0x40200\n0x0 0xffffffffffffffff 0x14220c\n", "no address");

    covers("Tesla V100-PCIE-32GB", 32 * GiB, V100_VRAM, true);
    covers("NVIDIA GeForce RTX 3090, pre-ReBAR", 256 * MiB, RTX3090_VRAM, false);
    covers("NVIDIA GeForce RTX 3090, ReBAR", 32 * GiB, RTX3090_VRAM, true);
    covers("a byte short", 24 * GiB - 1, 24 * GiB, false);
    covers("exactly", 24 * GiB, 24 * GiB, true);

    // the start-up link probe's report: ~47 us per 512 KiB as a tester's 3090s measured (tools/p2p_store.cu); prod's
    // V100s take 68-97 us
    const std::vector<fnx::PeerRate> three = {{0, 1, 47}, {0, 2, 47}, {1, 0, 47}, {1, 2, 47}, {2, 0, 47}, {2, 1, 47}};
    report("three cards, all ~11 GB/s", three, 3, 1, {{0, "kernel stores between the cards: 6 pairs, 11.2 to 11.2 GB/s (512 KiB a pair)"}});
    std::vector<fnx::PeerRate> one_slow = three;
    one_slow[3].us = 1310;   // 0.40 GB/s, as a link far down its lanes might
    report("one slow pair", one_slow, 3, 2, {{0, "6 pairs, 0.4 to 11.2 GB/s"}, {1, "card 1 -> card 2 stores at 0.40 GB/s"}, {1, "the engine runs"}});
    std::vector<fnx::PeerRate> at_floor = three;
    at_floor[0].us = (double) fnx::PEER_PROBE_BYTES / (fnx::PEER_SLOW_GBS * 1e3);   // exactly the floor: not slow
    report("a pair at the floor", at_floor, 3, 1, {});
    report("two cards", {{0, 1, 47}, {1, 0, 47}}, 2, 1, {{0, "2 pairs"}});
    report_refused("no pairs", {}, 3, "0 pairs probed for 3 cards");
    report_refused("a pair left out", {three.begin(), three.end() - 1}, 3, "5 pairs probed for 3 cards");
    report_refused("one card", {}, 1, "for 1 cards");
    report_refused("a count no card number makes", three, 65536, "6 pairs probed for 65536 cards");
    std::vector<fnx::PeerRate> no_time = three;
    no_time[2].us = 0;
    report_refused("a pair that took no time", no_time, 3, "took no time");
    // the pair checks, each wrong pair in the last slot so no pair after it can be what refuses
    const auto last_is = [&](fnx::PeerRate p) {
        std::vector<fnx::PeerRate> r = three;
        r.back() = p;
        return r;
    };
    report_refused("a pair twice, another missing", last_is({0, 1, 47}), 3, "card 0 -> 1 is not a new pair");
    report_refused("a card to itself", last_is({1, 1, 47}), 3, "card 1 -> 1 is not a new pair");
    report_refused("a card past the count", last_is({3, 0, 47}), 3, "card 3 -> 0 is not a new pair");
    report_refused("a card past the count, receiving", last_is({0, 3, 47}), 3, "card 0 -> 3 is not a new pair");
    report_refused("a negative card", last_is({-1, 0, 47}), 3, "card -1 -> 0 is not a new pair");
    report_refused("a negative card, receiving", last_is({2, -1, 47}), 3, "card 2 -> -1 is not a new pair");
    for (const fnx::PeerFailure& bad : {failure(1, 2, 3, false, 4096, 65536, 0), failure(1, 2, 3, false, 0, 65536, 1000),
                                        failure(1, 2, 3, false, 65537, 65536, 1000)})
        try {
            fail("peer_probe_failure reported " + std::to_string(bad.missing) + " of " + std::to_string(bad.words) + " in " +
                 std::to_string(bad.timeout_ms) + " ms: " + fnx::peer_probe_failure(bad));
        } catch (const std::logic_error& e) {
            if (std::string(e.what()).find("nothing to report") == std::string::npos) fail(std::string("peer_probe_failure refused with: ") + e.what());
        }
    {
        const std::string stores = fnx::peer_probe_failure(failure(1, 2, 3, false, 4096, 65536, 1000));
        const std::string copy = fnx::peer_probe_failure(failure(0, 2, 3, true, 1, 65536, 1000));
        for (const char* want : {"card 1 -> card 2", "4096 of 65536 words stored by a kernel", "within 1000 ms", "under 0.5 MB/s",
                                 "leave card 1 or card 2 out of the config's \"gpu\"", "build for 2 (make FNX_NCARD=2)"})
            if (stores.find(want) == std::string::npos) fail("peer_probe_failure does not say '" + std::string(want) + "': " + stores);
        if (copy.find("1 of 65536 words copied by the copy engines from card 0") == std::string::npos)
            fail("peer_probe_failure for a copy does not name it: " + copy);
    }
    // the remedy on every count: from 2 there is no smaller build; from 6 and 8 the next one leaves out two cards
    const std::pair<int, const char*> remedies[] = {
        {2, "replace card 1: a two-card build has no smaller count"},
        {3, "build for 2 (make FNX_NCARD=2)"},
        {4, "build for 3 (make FNX_NCARD=3)"},
        {6, "build for 4 (make FNX_NCARD=4): 4 is the next count below 6, so leave out 2 cards in all"},
        {8, "build for 6 (make FNX_NCARD=6): 6 is the next count below 8, so leave out 2 cards in all"}};
    for (const auto& [n, want] : remedies) {
        const std::string got = fnx::leave_card_out("card 1", n);
        if (got.find(want) == std::string::npos) fail("leave_card_out on " + std::to_string(n) + " cards: '" + got + "', want '" + want + "'");
    }

    if (fails) {
        std::printf("card_check_test: %d failed\n", fails);
        return 1;
    }
    std::printf("card_check_test: ok\n");
    return 0;
}
