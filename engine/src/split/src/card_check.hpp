#pragma once
// What the engine checks of every card as the model loads, before anything is placed on one (model.cpp,
// check_cards): the compute capability the build needs, the PCI address and BAR1 as sysfs spells them. Kept
// apart from CUDA and the files so a CPU test holds the rules to real cards' numbers and sysfs text
// (tools/card_check_test.cpp). The links between the cards are tried later, by the engine (src/peer_probe.hpp)
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fnx {

// The kernels are compiled for one compute capability (make ARCH=...). On any other card the driver compiles their
// PTX at load without a word, a different program from the one the checks ran, so a card must be the build's arch
// exactly. Any SM count runs (the hc read sizes its grid to the card, src/hc.cuh). arch: the build's (FNX_ARCH); major,
// minor: the card's
inline void require_card_arch(int dev, const std::string& name, int major, int minor, int arch) {
    const int cc = major * 10 + minor;
    if (cc != arch)
        throw std::runtime_error("card " + std::to_string(dev) + " (" + name + ") is compute capability " + std::to_string(major) + "." +
                                 std::to_string(minor) + " and this build is for sm_" + std::to_string(arch) +
                                 ": rebuild with make ARCH=" + std::to_string(cc) + " (the Makefile says if it builds for that one)");
}

// s as 1 to max_digits hex digits (after a 0x when prefix allows one) and nothing else, max_digits at most 16 (more would
// wrap the value). sscanf's %x takes a sign and wraps what is too wide, so "-0x1" would read as all ones and
// "100000001" as 1
inline bool parse_hex(const std::string& s, size_t max_digits, bool prefix, unsigned long long& v) {
    if (max_digits > 16) return false;
    const size_t i0 = prefix && s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') ? 2 : 0;
    if (s.size() == i0 || s.size() - i0 > max_digits) return false;
    v = 0;
    for (size_t i = i0; i < s.size(); ++i) {
        const char c = s[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return false;
        v = v * 16 + (unsigned) d;
    }
    return true;
}

// CUDA's spelling of a PCI address ("0000:3B:00.0"; nvidia-smi's "00000000:3B:00.0") in sysfs's: lower case, a 4-digit
// domain
inline std::string sysfs_bdf(const std::string& bus) {
    const size_t npos = std::string::npos, c1 = bus.find(':'), c2 = c1 == npos ? npos : bus.find(':', c1 + 1),
                 dot = c2 == npos ? npos : bus.find('.', c2 + 1);
    unsigned long long dom = 0, b = 0, d = 0, fn = 0;
    if (dot == npos || bus.find_first_of(":.", dot + 1) != npos || !parse_hex(bus.substr(0, c1), 8, false, dom) ||
        !parse_hex(bus.substr(c1 + 1, c2 - c1 - 1), 2, false, b) || !parse_hex(bus.substr(c2 + 1, dot - c2 - 1), 2, false, d) ||
        !parse_hex(bus.substr(dot + 1), 1, false, fn) || dom > 0xffff || d > 0x1f || fn > 7)
        throw std::runtime_error("cannot read the PCI address '" + bus + "'");
    char s[16];
    std::snprintf(s, sizeof s, "%04x:%02x:%02x.%x", (unsigned) dom, (unsigned) b, (unsigned) d, (unsigned) fn);
    return s;
}

// IORESOURCE_MEM in a sysfs resource line's flags (include/linux/ioport.h): memory, not I/O ports
constexpr unsigned long long PCI_RES_MEM = 0x200;

// BAR1's size in bytes from the text of /sys/bus/pci/devices/<bdf>/resource (path, for the errors): one "start end
// flags" line in hex per resource, BAR i on line i + 1. A BAR1 left unassigned (start 0) or not memory is a card the
// ring cannot write into, so it throws rather than read as a size
inline unsigned long long bar1_bytes(const std::string& text, const std::string& path) {
    std::istringstream in(text);
    std::string line;
    for (int i = 0; i < 2; ++i)
        if (!std::getline(in, line)) throw std::runtime_error(path + ": " + std::to_string(i) + " lines, BAR1 is on the second");
    std::istringstream fields(line);
    std::string f[4];
    int n = 0;
    while (n < 4 && fields >> f[n]) ++n;
    unsigned long long start = 0, end = 0, flags = 0;
    if (n != 3 || !parse_hex(f[0], 16, true, start) || !parse_hex(f[1], 16, true, end) || !parse_hex(f[2], 16, true, flags))
        throw std::runtime_error(path + ": cannot read BAR1 from '" + line + "'");
    if (start == 0)
        throw std::runtime_error(path + ": BAR1 has no address (the BIOS left it unassigned: Above 4G Decoding off?)");
    if (!(flags & PCI_RES_MEM) || end < start) throw std::runtime_error(path + ": BAR1 is not a memory range: '" + line + "'");
    return end - start + 1;
}

// what to do about a card the engine cannot use, on a build for ncard cards. A build runs exactly its count, and counts
// are 2, 3, 4, 6 or 8: leaving the card out means building for the next count below, which from 6 or 8 leaves out a
// second card too, and from 2 there is none. which: "card 2", "card 1 or card 2". Card numbers count the config's "gpu"
// list in order (CUDA_VISIBLE_DEVICES), not nvidia-smi's: card 0 of "gpu": [1, 2, 3] is nvidia-smi's GPU 1
inline std::string leave_card_out(const std::string& which, int ncard) {
    if (ncard <= 2) return "replace " + which + ": a two-card build has no smaller count to fall back to (one card is Strata's)";
    const int m = ncard == 8 ? 6 : ncard == 6 ? 4 : ncard - 1;
    std::string s = "leave " + which + " out of the config's \"gpu\" (card n is its list's n-th entry, counting from 0) and build for " +
                    std::to_string(m) + " (make FNX_NCARD=" + std::to_string(m) + ")";
    if (ncard - m > 1)
        s += ": " + std::to_string(m) + " is the next count below " + std::to_string(ncard) + ", so leave out " + std::to_string(ncard - m) +
             " cards in all";
    return s;
}

// The ring's stores into a peer go through the peer's BAR1. With the GeForce P2P patch (aikitoria's
// open-gpu-kernel-modules) the driver maps a pair only when both cards' BAR1 covers all their memory ("static BAR1",
// Resizable BAR at full size); otherwise it falls back to mailbox P2P, which fails to set up on RTX 3090s (dmesg:
// "remoteWMBoxLocalAddr != ~0ULL @ kern_bus_gm200.c") while cudaDeviceCanAccessPeer still says 1. The stores are then
// lost or land elsewhere: the ring hangs in warm_graphs, and a tester's host reset itself after a probe through such a
// mapping. A V100-32GB's BAR1 is 32 GiB, all of it. vram: the card's memory as CUDA reports it (totalGlobalMem)
inline void require_bar1_covers(int dev, const std::string& bdf, const std::string& name, unsigned long long bar1,
                                unsigned long long vram, int ncard) {
    if (bar1 >= vram) return;
    throw std::runtime_error("card " + std::to_string(dev) + " (" + bdf + ", " + name + ") has a " + std::to_string(bar1 >> 20) +
                             " MiB BAR1 for " + std::to_string(vram >> 20) +
                             " MiB of memory: the ring's stores into it need BAR1 to cover all of it, or they are lost and "
                             "the ring hangs. Turn on Resizable BAR at full size (the BIOS, and a ReBAR VBIOS from the "
                             "card's maker if `lspci -vv` lists no size that large), or " + leave_card_out("card " + std::to_string(dev), ncard) +
                             ". FNX_SKIP_BAR1_CHECK=1 skips this for cards whose peer access needs no BAR1 that size");
}

}  // namespace fnx
