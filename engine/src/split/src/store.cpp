#include "store.hpp"

#if FNX_HIP
// The lab's shared model memory is CUDA's virtual memory API with file descriptors; the HIP build has no holder process (the
// launcher never uses one: the engine loads its own weights), so these refuse.
#include <stdexcept>
namespace fnx {
static void no_store() { throw std::runtime_error("store: the shared model memory (holder / worker) is not built on HIP"); }
void Store::reserve(int, size_t) { no_store(); }
void* Store::take(int, size_t) { no_store(); return nullptr; }
void* Store::dev(int, size_t) { no_store(); return nullptr; }
void* Store::host(int, size_t) { no_store(); return nullptr; }
void Store::free_dev(int, void*, size_t) {}
size_t Store::bytes(int) const { return 0; }
void Store::attach(const std::vector<Chunk>&) { no_store(); }
const void* Store::move(const void* p) const { if (p) no_store(); return p; }
}  // namespace fnx
#else
#include <cuda.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "model.hpp"

namespace fnx {

namespace {

constexpr size_t VA_RESERVE = (size_t) 512 << 30;   // a location's range in the holder
// small enough that the last, part-used chunk of a location wastes little (cards: <256 MiB)
constexpr size_t DEV_CHUNK = (size_t) 256 << 20, HOST_CHUNK = (size_t) 1 << 30;

void cu(CUresult r, const std::string& what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error("store: " + what + ": " + (s ? s : "unknown error"));
}
bool is_host(int loc) { return loc >= NCARD; }
int card_of(int loc) { return loc % NCARD; }
CUmemAllocationProp prop_of(int loc) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (is_host(loc)) {
        CUdevice d;
        cu(cuDeviceGet(&d, card_of(loc)), "device");
        int numa = 0;
        cu(cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, d), "host NUMA id");
        p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        p.location.id = numa < 0 ? 0 : numa;
    } else {
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = loc;
    }
    return p;
}
// device chunks: their card; host chunks: every card and the CPU. A worker maps the holder's chunks read only: the
// weights are shared by every later worker, and a stray write by a faulting kernel (CUDA traps none inside an
// allocation) would poison them for all of them; read only, it faults in the worker that made it
void grant(int loc, uint64_t va, size_t size, bool read_only) {
    const CUmemAccess_flags prot = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUmemAccessDesc a[NCARD + 1] = {};
    int n = 0;
    for (int c = 0; c < NCARD; ++c) {
        if (!is_host(loc) && c != loc) continue;
        a[n].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a[n].location.id = c;
        a[n].flags = prot;
        ++n;
    }
    if (is_host(loc)) {
        a[n].location = prop_of(loc).location;
        a[n].flags = prot;
        ++n;
    }
    cu(cuMemSetAccess((CUdeviceptr) va, size, a, n), "access");
}

}  // namespace

// a range wherever CUDA puts it (a requested address is honoured only inside its own window)
void Store::reserve(int loc, size_t bytes) {
    if (reserved_[loc]) return;
    cu(cuInit(0), "init");
    CUdeviceptr got = 0;
    cu(cuMemAddressReserve(&got, bytes, 0, 0, 0), "reserve");
    base_[loc] = next_[loc] = end_[loc] = got;
    reserved_[loc] = true;
}

void* Store::take(int loc, size_t bytes) {
    reserve(loc, VA_RESERVE);
    bytes = (bytes + 255) / 256 * 256;
    if (next_[loc] + bytes > end_[loc]) {
        const CUmemAllocationProp p = prop_of(loc);
        size_t gran = 0;
        cu(cuMemGetAllocationGranularity(&gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
        // a new chunk right after the last, big enough for the rest of this allocation
        const size_t need = next_[loc] + bytes - end_[loc];
        size_t size = std::max(need, is_host(loc) ? HOST_CHUNK : DEV_CHUNK);
        size = (size + gran - 1) / gran * gran;
        if (end_[loc] + size > base_[loc] + VA_RESERVE) throw std::runtime_error("store: location " + std::to_string(loc) + " past its range");
        CUmemGenericAllocationHandle h;
        cu(cuMemCreate(&h, size, &p, 0), "create (" + std::to_string(size >> 20) + " MiB at location " + std::to_string(loc) + ")");
        cu(cuMemMap((CUdeviceptr) end_[loc], size, 0, h, 0), "map");
        grant(loc, end_[loc], size, false);
        Chunk c{loc, end_[loc], size};
        cu(cuMemExportToShareableHandle(&c.fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "export");
        cu(cuMemRelease(h), "release");   // the mapping keeps the memory
        chunks_.push_back(c);
        end_[loc] += size;
    }
    void* at = (void*) next_[loc];
    next_[loc] += bytes;
    return at;
}

void* Store::dev(int card, size_t bytes) { return take(card, bytes); }
void* Store::host(int card, size_t bytes) { return take(NCARD + card, bytes); }
void Store::free_dev(int card, void* p, size_t bytes) {
    if ((uint64_t) p + (bytes + 255) / 256 * 256 == next_[card]) next_[card] = (uint64_t) p;
}

size_t Store::bytes(int loc) const {
    size_t n = 0;
    for (const Chunk& c : chunks_) n += c.loc == loc ? c.size : 0;
    return n;
}

// each location's chunks at their offsets from the location's first one, in a range of this process's
void Store::attach(const std::vector<Chunk>& from) {
    for (int loc = 0; loc < 2 * NCARD; ++loc) {
        uint64_t lo = UINT64_MAX, hi = 0;
        for (const Chunk& c : from)
            if (c.loc == loc) lo = std::min(lo, c.va), hi = std::max(hi, c.va + c.size);
        if (hi == 0) continue;
        reserve(loc, hi - lo);
        hbase_[loc] = lo;
        end_[loc] = hi;
    }
    for (const Chunk& c : from) {
        if (c.loc < 0 || c.loc >= 2 * NCARD) throw std::runtime_error("store: a chunk of an unknown location");
        const uint64_t at = base_[c.loc] + (c.va - hbase_[c.loc]);
        CUmemGenericAllocationHandle h;
        cu(cuMemImportFromShareableHandle(&h, (void*) (uintptr_t) c.fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), "import");
        cu(cuMemMap((CUdeviceptr) at, c.size, 0, h, 0), "map");
        grant(c.loc, at, c.size, true);
        cu(cuMemRelease(h), "release");
        chunks_.push_back(c);
    }
}

const void* Store::move(const void* p) const {
    if (!p) return p;
    const uint64_t a = (uint64_t) p;
    for (int loc = 0; loc < 2 * NCARD; ++loc)
        if (reserved_[loc] && a >= hbase_[loc] && a < end_[loc]) return (const void*) (a - hbase_[loc] + base_[loc]);
    throw std::runtime_error("store: a holder's address outside its chunks");
}

}  // namespace fnx
#endif  // FNX_HIP
