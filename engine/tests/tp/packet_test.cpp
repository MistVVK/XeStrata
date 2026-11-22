#include "strata/tp/packet.hpp"
#include <bit>
#include <cstdio>
#include <cstdint>
#include <limits>

int main() try {
    using namespace strata::tp;
    enum class Kind : uint8_t { generate = 7 };
    Packet p;
    write(p, Kind::generate); write(p, int64_t(-17)); write(p, -0.0); write(p, true);
    const std::string text = std::string("Привет") + '\0' + "мир";
    write(p, text);
    std::vector<int> ids(1048576 - 64);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = int(i % 248320);
    write(p, ids); write(p, std::vector<int>{});
    write(p, std::vector<std::pair<float,int>>{{1.25f,2},{-0.5f,4}});
    PacketReader r(p.bytes());
    if (r.scalar<Kind>() != Kind::generate || r.scalar<int64_t>() != -17 ||
        std::bit_cast<uint64_t>(r.scalar<double>()) != std::bit_cast<uint64_t>(-0.0) || !r.boolean() || r.string() != text ||
        r.vector<int>() != ids || !r.vector<int>().empty() || r.count() != 2 || r.scalar<float>() != 1.25f ||
        r.scalar<int>() != 2 || r.scalar<float>() != -0.5f || r.scalar<int>() != 4)
        throw std::runtime_error("packet changed values, row order or string bytes");
    r.finish();
    auto refuse = [](auto operation) {
        bool failed = false;
        try { operation(); } catch (const std::runtime_error&) { failed = true; }
        if (!failed) throw std::runtime_error("malformed MPI packet accepted");
    };
    refuse([&] { PacketReader q(std::string_view(p.bytes()).substr(0,2));q.scalar<uint64_t>(); });
    refuse([&] { PacketReader q(p.bytes());q.finish(); });
    Packet bad;write(bad,std::numeric_limits<uint32_t>::max());
    refuse([&] { PacketReader q(bad.bytes());q.vector<int>(); });
    Packet boolean;write(boolean,uint8_t(2));
    refuse([&] { PacketReader q(boolean.bytes());q.boolean(); });
    refuse([&] { Packet large;write(large,std::vector<int>(max_prompt_bytes/sizeof(int),1)); });
    std::puts("MPI packet scalars, million token IDs, strings, pairs and malformed frames passed");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
