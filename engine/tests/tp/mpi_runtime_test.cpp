#include "strata/tp/mpi_runtime.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

int main(int argc, char** argv) try {
    strata::tp::MpiRuntime runtime(argc, argv, std::chrono::seconds(2));
    const bool mismatch = argc > 1 && std::string(argv[1]) == "mismatch";
    runtime.agree(mismatch && runtime.rank() == 1 ? "different-job" : "test-model-and-settings");
    std::array<std::byte, 128> id{};
    if (runtime.rank() == 0) for (size_t i = 0; i < id.size(); ++i) id[i] = std::byte(i);
    runtime.broadcast(id);
    for (size_t i = 0; i < id.size(); ++i) if (id[i] != std::byte(i)) throw std::runtime_error("bootstrap ID changed");
    std::array<uint32_t, 3> work{};
    if (runtime.rank() == 0) work = {2, 41, 512};
    runtime.broadcast_control(work);
    if (work != std::array<uint32_t, 3>{2, 41, 512}) throw std::runtime_error("control message changed");
    if (runtime.minimum(uint64_t(runtime.rank() * 7 + 5)) != 5) throw std::runtime_error("minimum did not include every rank");
    std::vector<int> sizes(runtime.size());
    for (int r = 0; r < runtime.size(); ++r) sizes[r] = r + 1;
    std::vector<std::byte> local(runtime.rank() + 1, std::byte(runtime.rank())), gathered(runtime.size() * (runtime.size() + 1) / 2);
    runtime.gather(local, gathered, sizes);
    size_t at = 0;
    for (int r = 0; r < runtime.size(); ++r)
        for (int j = 0; j <= r; ++j) if (gathered[at++] != std::byte(r)) throw std::runtime_error("uneven rank gather changed payload");
    if (argc > 1 && std::string(argv[1]) == "inplace") {
        size_t offset = 0;
        for (int r = 0; r < runtime.rank(); ++r) offset += sizes[r];
        std::fill(gathered.begin(), gathered.end(), std::byte(255));
        std::copy(local.begin(), local.end(), gathered.begin() + offset);
        runtime.gather(std::span<const std::byte>(gathered).subspan(offset, local.size()), gathered, sizes);
        at = 0;
        for (int r = 0; r < runtime.size(); ++r)
            for (int j = 0; j <= r; ++j) if (gathered[at++] != std::byte(r)) throw std::runtime_error("in-place gather changed payload");
    }
    // Idle exceeds the active-operation timeout; waiting ranks must remain usable.
    if (runtime.rank() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2250));
    if (runtime.command(runtime.rank() == 0 ? "GEN 17 1,2,3" : "") != "GEN 17 1,2,3")
        throw std::runtime_error("GEN command changed");
    if (argc > 1 && std::string(argv[1]) == "long") {
        const std::string command = "GEN 64 " + std::string(7 * 1048576 - 448, '1');
        if (runtime.command(runtime.rank() == 0 ? command : "") != command)
            throw std::runtime_error("MPI truncated a million-token-sized command");
    }
    if (runtime.command(runtime.rank() == 0 ? "QUIT" : "") != "QUIT") throw std::runtime_error("QUIT lost");
    std::printf("MPI runtime rank=%d world=%d correct=1\n", runtime.rank(), runtime.size());
    runtime.finish();
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
