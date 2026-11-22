#include "hy4_remote_experts.hip.h"

#include <array>
#include <limits>
#include <vector>

using namespace hy4;

template <class T>
T* upload(const T* data, size_t count) {
    T* device = nullptr;
    HCK(cudaMalloc(&device, count * sizeof(T)));
    HCK(cudaMemcpy(device, data, count * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

int main() {
    constexpr size_t rows = 3;
    constexpr size_t slots = rows * MOE_TOPK;
    constexpr float big = 16777216.f;
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    std::array<int32_t, slots> ids{};
    std::array<ExpertDev, MOE_E> experts{};
    experts[1].pad = 1;
    std::vector<float> local(slots * MOE_D, invalid);
    std::vector<float> remote(local.size(), invalid);
    for (size_t slot = 0; slot < slots; ++slot) {
        const size_t row = slot / MOE_TOPK;
        const size_t top = slot % MOE_TOPK;
        // In row 0, (big + 1) - big is zero. Summing rank 0 first yields 1.
        const bool worker = row == 1 || (row == 0 && top == 1);
        ids[slot] = worker ? 1 : 0;
        float value = float(row);
        if (row == 0) {
            constexpr std::array<float, 3> mixed{big, 1.f, -big};
            value = top < mixed.size() ? mixed[top] : 0.f;
        }
        auto& owner = worker ? remote : local;
        for (size_t column = 0; column < MOE_D; ++column)
            owner[slot * MOE_D + column] = value * (column % 2 == 0 ? 1.f : 2.f);
    }
    auto* device_local = upload(local.data(), local.size());
    auto* device_remote = upload(remote.data(), remote.size());
    auto* device_experts = upload(experts.data(), experts.size());
    auto* device_ids = upload(ids.data(), ids.size());
    float* device_output = nullptr;
    HCK(cudaMalloc(&device_output, rows * MOE_D * sizeof(float)));
    k_moe_reduce_remote<<<rows, 256>>>(device_local, device_remote, device_output, device_experts, device_ids);
    std::vector<float> output(rows * MOE_D);
    HCK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool equal = true;
    for (size_t i = 0; i < output.size(); ++i) {
        const float expected = float(i / MOE_D * MOE_TOPK) * (i % 2 == 0 ? 1.f : 2.f);
        equal &= output[i] == expected;
    }
    cudaFree(device_output);
    cudaFree(device_ids);
    cudaFree(device_experts);
    cudaFree(device_remote);
    cudaFree(device_local);
    fprintf(stderr, "Hy4 slot order: %s\n", equal ? "passed" : "failed");
    return equal ? 0 : 1;
}
