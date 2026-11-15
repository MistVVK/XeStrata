// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU core_device
#include "strata/kernels/device_arch_tag.hpp"
#include "strata/platform/memory.hpp"
#include "strata/platform/env.hpp"


#if defined(__has_include)
#if __has_include(<level_zero/ze_api.h>) && __has_include(<sycl/ext/oneapi/backend/level_zero.hpp>)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#define STRATA_HAVE_ZE_API 1
#endif
#endif

#ifndef _WIN32
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<drm/xe_drm.h>)
#include <drm/drm.h>
#include <drm/xe_drm.h>
#endif
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <fstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {
namespace {
#ifdef STRATA_HAVE_ZE_API
// A function of the Level Zero loader the SYCL runtime has loaded (libze_loader.so.1, Windows' ze_loader.dll); `load`:
// load it where it has not.  Null where there is none.
void* ze_function(const char* name, bool load) {
#if defined(_WIN32)
    HMODULE lib = GetModuleHandleA("ze_loader.dll");
    if (lib == nullptr && load) lib = LoadLibraryA("ze_loader.dll");
    return lib != nullptr ? reinterpret_cast<void*>(GetProcAddress(lib, name)) : nullptr;
#else
    void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr && load) lib = dlopen("libze_loader.so.1", RTLD_NOW);
    return lib != nullptr ? dlsym(lib, name) : nullptr;
#endif
}
#endif

// The backends the engine runs on: Level Zero for Intel GPUs, CUDA and HIP where the build has their targets.  The
// same Intel GPU is listed by OpenCL as well, which the engine does not use.
bool engine_backend(const sycl::device& d) {
    const auto b = d.get_backend();
    return b == sycl::backend::ext_oneapi_level_zero || b == sycl::backend::ext_oneapi_cuda ||
           b == sycl::backend::ext_oneapi_hip;
}

// A kernel with no work, built for the device to learn whether the binary carries code for it: a CUDA or HIP GPU
// is listed whenever the runtime has the backend, also by a build without that target (STRATA_CUDA_ARCHS).
struct ProbeKernel;

// Why a GPU cannot run the engine, from what it reports (empty: it can).  The kernels take 32-wide sub-groups, FP16
// and device and host USM; nothing is chosen by device ID or name (AGENTS.md).
std::string unusable(const sycl::device& d) {
    if (!d.has(sycl::aspect::usm_device_allocations) || !d.has(sycl::aspect::usm_host_allocations))
        return "no device and host USM";
    if (!d.has(sycl::aspect::fp16)) return "no FP16";
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), size_t(32)) == sg.end()) return "no 32-wide sub-groups";
    try {
        const sycl::context ctx(d);
        (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {d},
                                                                       {sycl::get_kernel_id<ProbeKernel>()});
    } catch (const sycl::exception& e) {
        return std::string("this build has no code for it (") + e.what() + ")";
    }
    return {};
}

bool integrated(const sycl::device& d) { return integrated_gpu(d); }

std::string pci_of(const sycl::device& d) {
    if (!d.has(sycl::aspect::ext_intel_pci_address)) return "?";
    std::string s = d.get_info<sycl::ext::intel::info::device::pci_address>();
    for (auto& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// The sysfs files are present only for devices whose driver reports a CPU-visible VRAM window.
bool visible_vram(const std::string& pci, uint64_t& total, uint64_t& used) {
#if defined(_WIN32)
    (void) pci; (void) total; (void) used;
    return false;
#else
    if (pci == "?") return false;
    const std::string base = "/sys/bus/pci/devices/" + pci + "/";
    if (!(std::ifstream(base + "mem_info_vis_vram_total") >> total) || total == 0) return false;
    if (!(std::ifstream(base + "mem_info_vis_vram_used") >> used)) used = UINT64_MAX;
    return true;
#endif
}

// No GPU on Level Zero, CUDA or HIP at all: the GPU's driver is missing, or the SYCL runtime did not load its adapter
// for the backend.  A user who started the engine without the libraries setup gave it (the config's lib_dirs) read
// only "no usable GPU" and looked for the fault in the card; say what the runtime lists and which libsycl it is.
std::string no_gpu_listed() {
    std::string seen;
    for (const auto& p : sycl::platform::get_platforms())
        seen += (seen.empty() ? "" : ", ") + p.get_info<sycl::info::platform::name>();
    std::string lib = "?";
#ifndef _WIN32
    // the loaded object by its name: dladdr on one of its functions names the engine itself, which is not PIE (the
    // address is the engine's PLT entry)
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void* out) {
        if (info->dlpi_name == nullptr || std::strstr(info->dlpi_name, "/libsycl.so") == nullptr) return 0;
        *static_cast<std::string*>(out) = info->dlpi_name;
        return 1;
    }, &lib);
#endif
    return "the SYCL runtime lists no GPU on Level Zero, CUDA or HIP (platforms: " + (seen.empty() ? "none" : seen) +
           "; libsycl: " + lib + "): the GPU driver is missing, or the runtime did not load its adapter for the GPU. "
           "Started by hand, the engine needs the lib_dirs of its config in LD_LIBRARY_PATH (the server sets it); "
           "SYCL_UR_TRACE=1 shows which adapters load";
}

// The GPUs the engine can drive, device 0 first: the GPU chosen below, then the others in the same order of
// preference.  Read once.
std::vector<sycl::device> find_devices() {
    std::vector<sycl::device> candidates;
    std::string refused;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (!engine_backend(d)) continue;
        if (const std::string why = unusable(d); !why.empty()) {
            refused += "; " + d.get_info<sycl::info::device::name>() + " (" + pci_of(d) + "): " + why;
            continue;
        }
        candidates.push_back(d);
    }
    if (candidates.empty()) throw DeviceError(refused.empty() ? no_gpu_listed() : "no usable GPU" + refused);
    // setup names the card by PCI address: Level Zero's own numbering can include GPUs that setup does not list
    if (const char* want = strata::env_get("GPU_PCI"); want != nullptr && *want != '\0') {
        std::string w = want;
        for (auto& c : w) c = (char) std::tolower((unsigned char) c);
        std::vector<sycl::device> match;
        for (const auto& d : candidates)
            if (pci_of(d) == w) match.push_back(d);
        if (match.size() != 1)
            throw DeviceError(std::string("XESTRATA_GPU_PCI=") + want + ": no usable GPU at that PCI address" +
                              refused);
    }
    // A discrete card before the processor's own graphics, then the most memory, then the most compute units (whose
    // size differs between GPU makers, so they only break a tie); the GPU STRATA_GPU_PCI names goes first.
    const char* want = strata::env_get("GPU_PCI");
    std::string w = want ? want : "";
    for (auto& c : w) c = (char) std::tolower((unsigned char) c);
    std::stable_sort(candidates.begin(), candidates.end(), [&w](const sycl::device& x, const sycl::device& y) {
        if (!w.empty() && (pci_of(x) == w) != (pci_of(y) == w)) return pci_of(x) == w;
        const bool ix = integrated(x), iy = integrated(y);
        if (ix != iy) return iy;
        const auto mx = x.get_info<sycl::info::device::global_mem_size>();
        const auto my = y.get_info<sycl::info::device::global_mem_size>();
        if (mx != my) return mx > my;
        return x.get_info<sycl::info::device::max_compute_units>() > y.get_info<sycl::info::device::max_compute_units>();
    });
    return candidates;
}

// IGC 2.27 to 2.30 (Ubuntu 26.04's and Debian's 2.28.4 among them) miscompile, for Xe2 and Xe3 GPUs, reads from one
// pointer at a runtime index N and at constant multiples of it: the byte offsets of x[2N] and x[3N] are taken for
// element indices, and the kernel reads x[8N] and x[9N] (a fault past the buffer, or wrong values inside it).  Their
// GEP simplification does it, and IGC_EnableGEPSimplification=0 turns that pass off: on the B70 the decode of a 1K
// prompt reads 5 to 10% slower with it off and 32K the same, so it is off only where both are true: the installed IGC
// is one of those (2.24 and 2.32 on compile the reads right) and a GPU the xe driver runs is Xe2 or later (graphics IP
// 20 and on; Alchemist, IP 12.55, compiles the reads right; AGENTS.md, "Intel Graphics Compiler").  IGC reads its
// IGC_ variables when Level Zero loads it, as the SYCL runtime lists the devices, so the GPUs are asked of the kernel
// driver here, before that; a value the user set stays.  Linux only: Windows keeps its driver's compiler as it is.
struct CompilerWorkaround {
    bool set = false;      // the engine set IGC_EnableGEPSimplification=0
    std::string version;   // IGC's, when known
};
std::string g_workaround_version;   // set by engine_devices when the workaround applies to a listed GPU

#if !defined(_WIN32) && defined(__has_include)
#if __has_include(<drm/xe_drm.h>)
#define STRATA_XE_QUERY 1
// The kernel's GT-list entry gained ip_ver_major with Linux 6.16 (the IP version a GMD_ID GPU has); a header from
// before it (Ubuntu 24.04's linux-libc-dev, a 6.8 UAPI) has no such member, and no kernel of that header's time
// reports one, so the workaround cannot be decided there and stays off.
template <typename T, typename = void>
struct has_ip_ver_major : std::false_type {};
template <typename T>
struct has_ip_ver_major<T, std::void_t<decltype(std::declval<T>().ip_ver_major)>> : std::true_type {};
template <typename List>
bool gt_list_is_xe2(const List* list) {
    if constexpr (has_ip_ver_major<drm_xe_gt>::value) {
        for (uint32_t i = 0; i < list->num_gt; ++i)
            if (list->gt_list[i].type == DRM_XE_QUERY_GT_TYPE_MAIN && list->gt_list[i].ip_ver_major >= 20) return true;
    } else {
        (void) list;
    }
    return false;
}
#endif
#endif

// Whether a GPU of the xe kernel driver has a graphics IP of 20 or later (DRM_XE_DEVICE_QUERY_GT_LIST, the main GT's
// ip_ver_major; 0 on a GPU without GMD_ID, Alchemist and older, which is not one).  False when no xe GPU can be opened.
#if !defined(_WIN32)
bool xe_gpu_from_xe2() {
#if defined(STRATA_XE_QUERY)
    for (int minor = 128; minor < 192; ++minor) {
        const std::string node = "/dev/dri/renderD" + std::to_string(minor);
        const int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        char name[16] = {};
        drm_version ver{};
        ver.name_len = sizeof(name) - 1;
        ver.name = name;
        bool found = false;
        if (ioctl(fd, DRM_IOCTL_VERSION, &ver) == 0 && std::strcmp(name, "xe") == 0) {
            drm_xe_device_query q{};
            q.query = DRM_XE_DEVICE_QUERY_GT_LIST;
            if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0 && q.size >= sizeof(drm_xe_query_gt_list)) {
                std::vector<uint64_t> buf((q.size + 7) / 8);
                q.data = (uint64_t) (uintptr_t) buf.data();
                if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0) {
                    const auto* list = reinterpret_cast<const drm_xe_query_gt_list*>(buf.data());
                    found = gt_list_is_xe2(list);
                }
            }
        }
        close(fd);
        if (found) return true;
    }
#endif
    return false;
}
#endif

CompilerWorkaround intel_compiler_workaround() {
    static const CompilerWorkaround done = [] {
        CompilerWorkaround w;
#if !defined(_WIN32)
        if (std::getenv("IGC_EnableGEPSimplification") != nullptr || !xe_gpu_from_xe2()) return w;
        void* lib = dlopen("libigc.so.2", RTLD_LAZY | RTLD_LOCAL);   // the driver loads the same one later
        if (lib == nullptr) return w;                                 // no Intel compiler: nothing to set
        bool affected = true;                                         // (a file name with no version: as affected)
        link_map* map = nullptr;
        char path[PATH_MAX];
        if (dlinfo(lib, RTLD_DI_LINKMAP, static_cast<void*>(&map)) == 0 && map != nullptr && map->l_name != nullptr &&
            realpath(map->l_name, path) != nullptr) {
            // libigc.so.2.28.4+0: the version after "libigc.so.", up to the build number
            const char* name = std::strrchr(path, '/');
            name = name ? name + 1 : path;
            if (std::strncmp(name, "libigc.so.", 10) == 0) {
                const char* v = name + 10;
                size_t n = 0;
                while (std::isdigit((unsigned char) v[n]) || v[n] == '.') ++n;
                w.version.assign(v, n);
                char* rest = nullptr;
                const long major = std::strtol(w.version.c_str(), &rest, 10);
                const long minor = rest != nullptr && *rest == '.' ? std::strtol(rest + 1, nullptr, 10) : -1;
                if (minor >= 0) affected = major == 2 && minor >= 25 && minor < 32;
            }
        }
        if (affected) setenv("IGC_EnableGEPSimplification", "0", 0);
        w.set = affected;
#endif
        return w;
    }();
    return done;
}

// The SYCL runtime's on-disk cache of the device code it builds (SYCL_CACHE_PERSISTENT), on unless the user set it:
// without it every start builds each kernel module it uses from the SPIR-V again, linking the device libraries in, and
// the GPU driver's own cache does not spare that.  GLM-5.3-Flash on the B70 (2026-10-19): a 1K-token prompt took 54.7 s
// (17 s of it the first use of the q8_1 kernels' module) and the first decode tokens 7.6 s; from the cache 27.4 s and
// 0.19 s, the model loaded in 1.3 s instead of 11.8.  The cache's keys hold the device, the driver's version and the
// build options, so a new driver builds anew.
static void sycl_code_cache() {
    if (std::getenv("SYCL_CACHE_PERSISTENT") != nullptr) return;
#if defined(_WIN32)
    _putenv_s("SYCL_CACHE_PERSISTENT", "1");
#else
    setenv("SYCL_CACHE_PERSISTENT", "1", 0);
#endif
}

// The HIP runtime's hardware queues a process takes (GPU_MAX_HW_QUEUES, ROCclr's; 4 by default), 2 unless the user
// set it.  With 4, the expert streaming of a prompt ran at one of two speeds from start to start, the same binary and
// settings (RX 9060 XT, ROCm 7.1, Linux, 2026-11-13: Qwen3.8-Flash-Next IQ3_XXS, a 4K prompt 692-727 or 913-977
// tok/s, about half the starts each; a 1K prompt 429 or 632; a 32K one 1138-1151), the slow starts losing 10-40 ms
// at random layers to the copies and the products around them; another GPU program run between two starts moved
// which start was slow, and nothing the engine logs differed.  With 2: 945-949, 666-683 and 1210-1211, three starts
// each, the decode alike; Qwen3.6-35B-A3B's 4K prompt 1541-1545 -> 1575-1579.  Only the HIP runtime reads it.
// Linux only: Windows's HIP runtime was not measured.
static void hip_hw_queues() {
#if !defined(_WIN32)
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
#endif
}

// What the RAM and the kernel's memory compaction did between the engine's start and a stop on a GPU fault, said with
// the fault: the B70's copy engine faulted (Fault response -EBUSY) while the page cache turned over, and the xe driver
// mapped the registered host memory again after each move of its pages (#42; 2026-10-22's trace: 83k to 134k
// invalidations where a healthy run has one).  The start's numbers are taken as the engine lists its GPUs.  Linux only.
struct HostCounters {
    long long compact_stall = -1, compact_success = 0, pgmigrate_success = 0;
};
HostCounters host_counters() {
    HostCounters c;
#if !defined(_WIN32)
    if (std::FILE* f = std::fopen("/proc/vmstat", "r")) {
        char line[128];
        c.compact_stall = 0;
        while (std::fgets(line, sizeof line, f) != nullptr) {
            char* sp = std::strchr(line, ' ');
            if (sp == nullptr) continue;
            *sp = '\0';
            const long long v = std::strtoll(sp + 1, nullptr, 10);
            if (std::strcmp(line, "compact_stall") == 0) c.compact_stall = v;
            else if (std::strcmp(line, "compact_success") == 0) c.compact_success = v;
            else if (std::strcmp(line, "pgmigrate_success") == 0) c.pgmigrate_success = v;
        }
        std::fclose(f);
    }
#endif
    return c;
}
HostCounters g_start_counters;

const std::vector<sycl::device>& engine_devices() {
    static const std::vector<sycl::device> devices = [] {
        sycl_code_cache();
        hip_hw_queues();
        g_start_counters = host_counters();
        const CompilerWorkaround w = intel_compiler_workaround();
        std::vector<sycl::device> found = find_devices();
        if (w.set && std::any_of(found.begin(), found.end(), [](const sycl::device& d) {
                return d.get_backend() == sycl::backend::ext_oneapi_level_zero;
            })) {
            g_workaround_version = w.version.empty() ? "unknown" : w.version;
            std::fprintf(stderr, "Xe runtime: IGC's GEP simplification is off (IGC_EnableGEPSimplification=0; IGC %s): "
                                 "2.27 to 2.30 miscompile some reads for Xe2 and later GPUs. The decode of short "
                                 "prompts is up to 10%% slower. For full speed, use Fedora 44 (its IGC is 2.32 or "
                                 "later), or install with Podman (Fedora 44 inside)\n", g_workaround_version.c_str());
        }
        return found;
    }();
    return devices;
}

thread_local int t_current = 0;

// Instantiates ProbeKernel for the device compilers; never run.
[[maybe_unused]] void probe_kernel_instance(sycl::queue& q) {
    q.single_task<ProbeKernel>([] {});
}

// The start's counters against the fault's, with the RAM's state then (HostCounters).
void report_host_at_fault() {
#if !defined(_WIN32)
    long long free_kb = -1, cached_kb = -1, dirty_kb = -1;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[128];
        // NOLINTNEXTLINE(clang-analyzer-unix.BlockInCriticalSection): /proc never blocks, and the process ends next
        while (std::fgets(line, sizeof line, f) != nullptr) {
            char* colon = std::strchr(line, ':');
            if (colon == nullptr) continue;
            *colon = '\0';
            const long long v = std::strtoll(colon + 1, nullptr, 10);
            if (std::strcmp(line, "MemFree") == 0) free_kb = v;
            else if (std::strcmp(line, "Cached") == 0) cached_kb = v;
            else if (std::strcmp(line, "Dirty") == 0) dirty_kb = v;
        }
        std::fclose(f);
    }
    const HostCounters now = host_counters();
    if (free_kb < 0 || now.compact_stall < 0 || g_start_counters.compact_stall < 0) return;
    std::fprintf(stderr, "Xe runtime: at the fault: RAM free %lld MiB, page cache %lld MiB, dirty %lld MiB; since the "
                         "start the kernel compacted memory %lld times (%lld stalls) and moved %lld pages\n",
                 free_kb >> 10, cached_kb >> 10, dirty_kb >> 10, now.compact_success - g_start_counters.compact_success,
                 now.compact_stall - g_start_counters.compact_stall,
                 now.pgmigrate_success - g_start_counters.pgmigrate_success);
#endif
}

[[noreturn]] void fatal(const char* message) {
    std::fprintf(stderr, "Xe runtime: %s; terminating without releasing in-flight allocations\n", message);
    report_host_at_fault();
    std::fflush(stderr);
    std::_Exit(1);
}

// The xe driver resets a job after its job_timeout_ms (5 s here, at most 10 s), which completes the event with an
// error, so a healthy wait - even a drain of many queued jobs - returns well inside this.  A wait that outlives it
// is stuck in the driver or the host, and only a thread other than the waiter can report that.  #59: other drivers
// (Windows', CUDA's, HIP's) may not end a job that long, so XESTRATA_GPU_WAIT_LIMIT_S sets it (seconds; 0: no limit).
std::chrono::seconds hang_limit() {
    static const std::chrono::seconds limit = [] {
        const char* v = strata::env_get("GPU_WAIT_LIMIT_S");
        const long s = v != nullptr ? std::strtol(v, nullptr, 10) : 120;
        return std::chrono::seconds(s > 0 ? s : 0);
    }();
    return limit;
}
}  // namespace

std::string compiler_workaround() {
    (void) engine_devices();
    return g_workaround_version;
}

int current_device() { return t_current; }

void set_current_device(int device) {
    if (device < 0 || device >= Runtime::count())
        throw DeviceError("no engine device " + std::to_string(device) + " (" + std::to_string(Runtime::count()) +
                          " usable)");
    t_current = device;
}

void restore_current_device(int device) noexcept { t_current = device; }

namespace {
std::mutex g_runtimes_mutex;
std::vector<std::unique_ptr<Runtime>>& runtimes() {
    static std::vector<std::unique_ptr<Runtime>> all;
    return all;
}
}  // namespace

int Runtime::count() { return (int) engine_devices().size(); }

Runtime& Runtime::get() { return at(t_current); }

Runtime& Runtime::at(int ordinal) {
    if (ordinal < 0 || ordinal >= count())
        throw DeviceError("no engine device " + std::to_string(ordinal) + " (" + std::to_string(count()) +
                          " usable)");
    std::lock_guard<std::mutex> lock(g_runtimes_mutex);
    auto& all = runtimes();
    if (all.size() < (size_t) count()) all.resize((size_t) count());
    auto& r = all[(size_t) ordinal];
    if (!r) r.reset(new Runtime(ordinal, engine_devices()[(size_t) ordinal]));
    return *r;
}

Runtime* Runtime::owner(const void* pointer) {
    if (!pointer) return nullptr;
    std::lock_guard<std::mutex> lock(g_runtimes_mutex);
    for (auto& r : runtimes()) {
        if (!r) continue;
        // ROCm (RX 9060 XT) sometimes refuses the query (UR_RESULT_ERROR_UNKNOWN) with a large streamed KV host
        // copy beside the expert arena; treat that as "not this runtime's" rather than letting it abort the start.
        try {
            if (sycl::get_pointer_type(pointer, r->context_) != sycl::usm::alloc::unknown) return r.get();
        } catch (const std::exception&) { continue; }
    }
    return nullptr;
}

Runtime* Runtime::owner_of_stream(const void* stream) {
    std::lock_guard<std::mutex> lock(g_runtimes_mutex);
    for (auto& r : runtimes())
        if (r && r->owns(static_cast<const sycl::queue*>(stream))) return r.get();
    return nullptr;
}

sycl::async_handler Runtime::handler() {
    return [this](sycl::exception_list errors) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_ && errors.size()) error_ = *errors.begin();
    };
}

namespace {
void settle_vram(const sycl::device& d);
}  // namespace

Runtime::Runtime(int ordinal, const sycl::device& device) : ordinal_(ordinal), pci_(pci_of(device)), device_(device), context_(device_),
    compute_(context_, device_, handler(), sycl::property::queue::in_order{}),
    transfer_(context_, device_, handler(), sycl::property::queue::in_order{}), watchdog_([this] { watch(); }) {
    uint64_t visible = 0, used = 0, vram = 0;
#if !defined(_WIN32)
    if (visible_vram(pci_, visible, used) &&
        (std::ifstream("/sys/bus/pci/devices/" + pci_ + "/mem_info_vram_total") >> vram) &&
        vram > visible) {
        std::fprintf(stderr, "Xe runtime: GPU %d CPU-visible VRAM is %llu of %llu MiB; copies can be slower, "
                     "AMD GPUs can appear stalled and LoRA training can fail. Enable Above 4G Decoding and "
                     "Resizable BAR in the BIOS.\n", ordinal_, (unsigned long long) (visible >> 20),
                     (unsigned long long) (vram >> 20));
    }
#else
    (void) visible; (void) used; (void) vram;
#endif
    settle_vram(device_);
    marks_by_barrier_ = measure_marks();
}

// A timestamp (submit_profiling_tag, the timing event's record) costs the host its submission and, while its event is
// held, the next wait; the prompt path holds 384, its streamed ring's.  On the B70 with Level Zero the cost moved
// between the runtime's versions: with compute-runtime 26.05 the wait, about 90 us a timestamp (the prompt at 1,248
// against 1,594 tok/s with barriers); with 26.35 the submission, about 11 us (the wait 0.05 us; the prompt at 1,270
// against 1,748 tok/s, issue #37), and each one maps the event's host memory for the GPU and frees it again, which
// fragments the driver's address ranges over hours of use.  On the RTX 4070 with CUDA a timestamp costs about 4 us more
// than a barrier, and the prompt reads 1% faster with timestamps (the 2026-10-09-xe-event-record and
// 2026-10-21-issue37-event-mark records).  So 64 timestamps and 64 barriers are each submitted and waited for on a
// queue of its own, and events without timing are barriers where a timestamp costs more than 6 us over a barrier: a
// wrong barrier costs the prompt about 1%, a wrong timestamp a quarter.  This measure reads 90 to 115 us a timestamp
// on the B70 with 26.05, 22 to 25 us with 26.35 and 4.1 to 4.3 us on the RTX 4070 (an i7-14700's submissions).
// STRATA_EVENT_MARK=tag or barrier chooses instead.
bool Runtime::measure_marks() {
    if (const char* v = strata::env_get("EVENT_MARK")) return std::strcmp(v, "barrier") == 0;
    constexpr int kMarks = 64;
    constexpr double kCostUs = 6.0;
    try {
        sycl::queue q(context_, device_, sycl::property::queue::in_order{});
        std::vector<sycl::event> held(kMarks);
        auto cost_us = [&q, &held](bool tag) {
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (auto& e : held)
                e = tag ? sycl::ext::oneapi::experimental::submit_profiling_tag(q) : q.ext_oneapi_submit_barrier();
            q.ext_oneapi_submit_barrier().wait();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        };
        double tag = 1e30, barrier = 1e30;
        for (int round = 0; round < 3; ++round) {
            tag = std::min(tag, cost_us(true));
            barrier = std::min(barrier, cost_us(false));
        }
        return tag - barrier > kMarks * kCostUs;
    } catch (const std::exception&) {
        return false;
    }
}

sycl::queue* Runtime::create_stream(bool high_priority) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
#if defined(SYCL_EXT_ONEAPI_QUEUE_PRIORITY)
    if (high_priority)
        return &streams_.emplace_back(context_, device_, handler(),
                                      sycl::property_list{sycl::property::queue::in_order{},
                                                          sycl::ext::oneapi::property::queue::priority_high{}});
#else
    (void) high_priority;
#endif
    return &streams_.emplace_back(context_, device_, handler(), sycl::property::queue::in_order{});
}

// The prompt path's expert copies on the GPU's copy engine (upstream 01fe68dd, in its sycl/ port): Level Zero's v2
// adapter runs an in-order queue's memcpy on the compute engine.  A device that reports a copy-only queue group gets an
// immediate command list on it, wrapped as a SYCL queue; any other device (CUDA, HIP, no such group, a build without
// the Level Zero headers) an ordinary engine stream.  On the B70 the prompt read as fast either way (4K 1,504 tok/s,
// 32K 2,312 and 2,305, 2026-10-12): the copies were already behind the kernels there.  STRATA_COPY_ENGINE=0: off.
sycl::queue* Runtime::create_copy_stream() {
#ifdef STRATA_HAVE_ZE_API
    static const bool off = [] {
        const char* v = strata::env_get("COPY_ENGINE");
        return v != nullptr && std::strtol(v, nullptr, 10) == 0;
    }();
    if (!off && device_.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        try {
            using GroupsFn = decltype(&zeDeviceGetCommandQueueGroupProperties);
            using ImmFn = decltype(&zeCommandListCreateImmediate);
            auto groups = reinterpret_cast<GroupsFn>(ze_function("zeDeviceGetCommandQueueGroupProperties", true));
            auto imm = reinterpret_cast<ImmFn>(ze_function("zeCommandListCreateImmediate", true));
            const auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device_);
            const auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context_);
            uint32_t n = 0;
            std::vector<ze_command_queue_group_properties_t> g;
            if (groups != nullptr && imm != nullptr && groups(zdev, &n, nullptr) == ZE_RESULT_SUCCESS && n > 0) {
                ze_command_queue_group_properties_t blank;
                std::memset(&blank, 0, sizeof blank);
                blank.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
                g.assign(n, blank);
                if (groups(zdev, &n, g.data()) != ZE_RESULT_SUCCESS) g.clear();
            }
            for (uint32_t i = 0; i < (uint32_t) g.size(); ++i) {
                if (!(g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ||
                    (g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) || g[i].numQueues == 0)
                    continue;
                ze_command_queue_desc_t d;
                std::memset(&d, 0, sizeof d);
                d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
                d.ordinal = i;
                d.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
                d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
                d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
                ze_command_list_handle_t cl = nullptr;
                if (imm(zctx, zdev, &d, &cl) != ZE_RESULT_SUCCESS || cl == nullptr) break;
                sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> in(
                    cl, device_, sycl::ext::oneapi::level_zero::ownership::transfer,
                    sycl::property_list{sycl::property::queue::in_order{}});
                std::lock_guard<std::mutex> lock(streams_mutex_);
                sycl::queue* q = &streams_.emplace_back(
                    sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(in, context_, handler()));
                static std::atomic<bool> said{false};
                if (!said.exchange(true))
                    std::fprintf(stderr, "xestrata: the prompt's expert copies on the GPU's copy engine (queue group %u; "
                                         "XESTRATA_COPY_ENGINE=0 turns it off)\n", i);
                return q;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata: no copy-engine queue (%s): an ordinary stream\n", e.what());
        }
    }
#endif
    return create_stream();
}

std::string Runtime::device_status() {
#ifdef STRATA_HAVE_ZE_API
    if (device_.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        try {
            using StatusFn = decltype(&zeDeviceGetStatus);
            auto status = reinterpret_cast<StatusFn>(ze_function("zeDeviceGetStatus", false));
            if (status == nullptr) return "unknown";
            const ze_result_t r = status(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device_));
            if (r == ZE_RESULT_SUCCESS) return "answers";
            if (r == ZE_RESULT_ERROR_DEVICE_LOST) return "lost";
            char buf[48];
            std::snprintf(buf, sizeof buf, "Level Zero result 0x%x", (unsigned) r);
            return buf;
        } catch (const std::exception&) {
            return "unknown";
        }
    }
#endif
    // no status query here (CUDA, HIP, a build without Level Zero's headers): whether the driver still answers a
    // question about the device's free memory, which CUDA refuses once a kernel has faulted its context (its errors
    // stick).  A failure report on these said "the GPU: unknown" every time (#36, #39).
    if (device_.has(sycl::aspect::ext_intel_free_memory)) {
        try {
            (void) device_.get_info<sycl::ext::intel::info::device::free_memory>();
            return "answers a memory query";
        } catch (const std::exception& e) {
            return std::string("does not answer a memory query (") + e.what() + ")";
        }
    }
    return "unknown";
}

void Runtime::destroy_stream(sycl::queue* stream) {
    if (!stream) return;
    if (!owns(stream) || stream == &compute_ || stream == &transfer_)
        throw DeviceError("destroy_stream: not an engine stream of the Xe runtime");
    finish(*stream);
    std::lock_guard<std::mutex> lock(streams_mutex_);
    streams_.remove_if([stream](const sycl::queue& q) { return &q == stream; });
}

bool Runtime::owns(const sycl::queue* queue) {
    if (queue == &compute_ || queue == &transfer_) return true;
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (const auto& q : streams_)
        if (&q == queue) return true;
    return false;
}

sycl::queue& Runtime::stream(void* stream) {
    if (!stream) return compute_;
    auto* q = static_cast<sycl::queue*>(stream);
    // a queue of another device's runtime is that device's: a kernel goes where its stream is, as with CUDA streams
    if (!owns(q) && owner_of_stream(q) == nullptr) throw DeviceError("the stream is not a queue of the Xe runtime");
    return *q;
}

void Runtime::preload_kernels() {
    if (preload_.joinable()) return;
    preload_ = std::thread([this] {
        int built = 0, skipped = 0;
        for (const sycl::kernel_id& id : sycl::get_kernel_ids()) {
            if (preload_stop_.load(std::memory_order_relaxed)) return;
            try {
                (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(context_, {device_}, {id});
                ++built;
            } catch (const std::exception&) {   // none for this device, or a shape it does not run
                ++skipped;
            }
        }
        if (strata::env_get("TRACE"))
            std::fprintf(stderr, "xestrata trace: %d kernels built ahead, %d not for this GPU\n", built, skipped);
    });
}

Runtime::~Runtime() {
    preload_stop_.store(true, std::memory_order_relaxed);
    if (preload_.joinable()) preload_.join();
    try { finish(); }
    catch (const std::exception& e) { fatal(e.what()); }
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        stop_ = true;
    }
    watch_cv_.notify_all();
    watchdog_.join();
}

Runtime::CopyWatch::CopyWatch(Runtime& runtime, const char* what, uint64_t bytes, int layer) : runtime_(runtime) {
    std::lock_guard<std::mutex> lock(runtime_.watch_mutex_);
    entry_ = &*runtime_.waiters_.insert(runtime_.waiters_.end(),
        Waiter{std::chrono::steady_clock::now(), what, bytes, {}, true, layer});
}

Runtime::CopyWatch::~CopyWatch() {
    std::lock_guard<std::mutex> lock(runtime_.watch_mutex_);
    for (auto it = runtime_.waiters_.begin(); it != runtime_.waiters_.end(); ++it) {
        if (&*it == entry_) { runtime_.waiters_.erase(it); break; }
    }
}

void Runtime::watch() {
    std::unique_lock<std::mutex> lock(watch_mutex_);
    const std::chrono::seconds limit = hang_limit();
    // A healthy copy of even a large layer is far shorter than a minute; 60 s leaves room for slow storage and
    // paging. The observed blocked submission lasted 19 minutes. Repeat only every 5 minutes if it remains blocked.
    constexpr auto copy_limit = std::chrono::seconds(60);
    constexpr auto repeat = std::chrono::minutes(5);
    while (!watch_cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_; })) {
        const auto now = std::chrono::steady_clock::now();
        for (Waiter& w : waiters_) {
            const auto elapsed = now - w.since;
            if (w.copy) {
                if (elapsed < copy_limit || (w.reported != decltype(w.reported){} && now - w.reported < repeat)) continue;
                w.reported = now;
                uint64_t total = 0, used = 0;
                std::fprintf(stderr, "Xe runtime: GPU %d copy submission %s", ordinal_, w.what);
                if (w.layer >= 0) std::fprintf(stderr, " layer %d", w.layer);
                std::fprintf(stderr, " (%llu bytes) has taken %lld s", (unsigned long long) w.bytes,
                             (long long) std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                if (visible_vram(pci_, total, used)) {
                    std::fprintf(stderr, "; CPU-visible VRAM %llu MiB", (unsigned long long) (total >> 20));
                    if (used != UINT64_MAX) std::fprintf(stderr, ", %llu MiB used", (unsigned long long) (used >> 20));
                }
                std::fprintf(stderr, "\n");
            } else if (limit.count() && elapsed >= limit) {
                const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                std::fprintf(stderr, "Xe runtime: waiting for %s for %lld s (%zu waits in progress)\n", w.what,
                             (long long) seconds, waiters_.size());
                fatal("GPU wait exceeded the hang limit (XESTRATA_GPU_WAIT_LIMIT_S)");
            }
        }
    }
}

void Runtime::check_async() {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (error_) std::rethrow_exception(error_);
}

sycl::event Runtime::copy_async(void* dst, const void* src, uint64_t bytes,
                               const std::vector<sycl::event>& dependencies) {
    check_async();
    CopyWatch watch(*this, "transfer queue", bytes);
    return transfer_.memcpy(dst, src, bytes, dependencies);
}

sycl::event Runtime::compute_after(const std::vector<sycl::event>& dependencies) {
    check_async();
    return compute_.ext_oneapi_submit_barrier(dependencies);
}

void Runtime::wait(sycl::event event, const char* what) {
    // Blocking in the driver rather than polling keeps the wake-up latency off decode's per-token waits.  A timeout
    // cannot safely throw and free USM that an unfinished command owns, so a stuck wait ends the process instead.
    std::list<Waiter>::iterator self;
    {
        std::lock_guard<std::mutex> lock(watch_mutex_);
        self = waiters_.insert(waiters_.end(), Waiter{std::chrono::steady_clock::now(), what});
    }
    struct Done {
        Runtime& r;
        std::list<Waiter>::iterator it;
        ~Done() { std::lock_guard<std::mutex> lock(r.watch_mutex_); r.waiters_.erase(it); }
    } done{*this, self};
    event.wait_and_throw();
    check_async();
}

void Runtime::finish(sycl::queue& queue) {
    if (!owns(&queue)) throw DeviceError("cannot drain a queue not owned by the Xe runtime");
    wait(queue.ext_oneapi_submit_barrier(), &queue == &compute_ ? "compute queue drain"
                                            : &queue == &transfer_ ? "transfer queue drain" : "engine stream drain");
}
void Runtime::finish() {
    finish(compute_);
    finish(transfer_);
    std::vector<sycl::queue*> streams;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        for (auto& q : streams_) streams.push_back(&q);
    }
    for (auto* q : streams) finish(*q);
}

void Runtime::finish_all() noexcept {
    std::vector<Runtime*> made;
    {
        std::lock_guard<std::mutex> lock(g_runtimes_mutex);
        for (auto& r : runtimes())
            if (r) made.push_back(r.get());
    }
    for (Runtime* r : made) {
        try {
            r->finish();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Xe runtime: GPU %d not drained before the exit: %s\n", r->ordinal(), e.what());
        }
    }
}

// SIGTERM and SIGINT end the engine at once, as their default action does, but on a thread of their own.  While the GPU
// compiler (IGC) translates a kernel it takes them over and longjmps to the translating thread, so one delivered to any
// other thread aborted the engine ("longjmp causes uninitialized stack frame"; the server's SIGTERM after a QUIT, or a
// stop while the kernels are built).  Blocked before any thread exists, so every thread inherits the mask.
// The GPUs' queues are drained first, for at most 5 s: on the B70 (xe driver) a copy-engine job still queued when the
// process ended read memory the exit had unmapped (an engine memory fault, and a reset).  Of 7 engines ended as everyday
// use's bench ends them (the server's process group at once) 4 faulted, with STRATA_COPY_ENGINE=0 as well (the
// runtime's own copies), and none once drained, 2026-10-15.
void route_stop_signals() {
#if defined(_WIN32)
    // Windows has no signal masks; Ctrl+C reaches a thread of its own already, and the server ends the engine with
    // taskkill, not a signal
#else
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) return;
    std::thread([set] {
        int sig = 0;
        while (sigwait(&set, &sig) != 0) {}
        std::thread([sig] {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            _exit(128 + sig);
        }).detach();
        Runtime::finish_all();
        _exit(128 + sig);
    }).detach();
#endif
}

void Runtime::free(void* pointer) noexcept {
    if (!pointer) return;
    try {
        finish();
        sycl::free(pointer, context_);
    } catch (const std::exception& e) { fatal(e.what()); }
}

namespace {
uint64_t env_mib(const char* name) {
    const char* v = strata::env_get(name);
    return v ? (uint64_t) std::strtoull(v, nullptr, 10) << 20 : 0;
}
}  // namespace

uint64_t vram_limit_bytes() {
    static const uint64_t v = env_mib("VRAM_LIMIT_MIB");
    return v;
}
uint64_t max_alloc_limit_bytes() {
    static const uint64_t v = env_mib("MAX_ALLOC_MIB");
    return v;
}

// Whether the GPU is the processor's own graphics (its memory is the system RAM), as the SYCL runtime reports it.
// Not asked of Level Zero directly: the engine links no maker's driver library, so a contrib build starts on a PC
// with only Intel's or only NVIDIA's (the runtime opens each backend's library when it is there).  Only Level Zero
// answers the integrated-GPU aspect: the CUDA and HIP adapters print an error each time it is asked ("<CUDA>[ERROR]:
// Unsupported ParamName", UR_DEVICE_INFO_IS_INTEGRATED_GPU) and report it absent.  The HIP runtime says the same as
// unified host memory (hipDeviceAttributeIntegrated), which is asked in its place.
bool integrated_gpu(const sycl::device& d) {
    const sycl::backend b = d.get_backend();
    if (b == sycl::backend::ext_oneapi_level_zero) return d.has(sycl::aspect::ext_oneapi_is_integrated_gpu);
    if (b != sycl::backend::ext_oneapi_hip) return false;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return d.get_info<sycl::info::device::host_unified_memory>();
#pragma clang diagnostic pop
}

namespace {
// The VRAM a GPU of the xe kernel driver has free, asked of the driver (DRM_XE_DEVICE_QUERY_MEM_REGIONS: its VRAM
// regions' size and what is used of them) through the render node of the card at the PCI address `pci`.  False on
// another driver, and where the driver gives no use: it may give it only to a process with CAP_PERFMON and 0 to
// others (Linux 7.0 gives a user the VRAM's use, the system memory's 0), and its own allocations are never none.
bool xe_vram(const std::string& pci, uint64_t& total, uint64_t& used) {
#if defined(STRATA_XE_QUERY)
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/sys/bus/pci/devices/" + pci + "/drm", ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("renderD", 0) != 0) continue;
        const int fd = open(("/dev/dri/" + name).c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        char drv[16] = {};
        drm_version ver{};
        ver.name_len = sizeof(drv) - 1;
        ver.name = drv;
        uint64_t size = 0, in_use = 0;
        if (ioctl(fd, DRM_IOCTL_VERSION, &ver) == 0 && std::strcmp(drv, "xe") == 0) {
            drm_xe_device_query q{};
            q.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
            if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0 && q.size >= sizeof(drm_xe_query_mem_regions)) {
                std::vector<uint64_t> buf((q.size + 7) / 8);
                q.data = (uint64_t) (uintptr_t) buf.data();
                if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) == 0) {
                    const auto* r = reinterpret_cast<const drm_xe_query_mem_regions*>(buf.data());
                    for (uint32_t i = 0; i < r->num_mem_regions; ++i)
                        if (r->mem_regions[i].mem_class == DRM_XE_MEM_REGION_CLASS_VRAM) {
                            size += r->mem_regions[i].total_size;
                            in_use += r->mem_regions[i].used;
                        }
                }
            }
        }
        close(fd);
        if (size == 0 || in_use == 0) return false;
        total = size;
        used = in_use;
        return true;
    }
#else
    (void) pci; (void) total; (void) used;
#endif
    return false;
}

// Whether the render node of the card at the PCI address `pci` is the i915 driver's.
bool i915_gpu(const std::string& pci) {
#if defined(STRATA_XE_QUERY)
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/sys/bus/pci/devices/" + pci + "/drm", ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("renderD", 0) != 0) continue;
        const int fd = open(("/dev/dri/" + name).c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        char drv[16] = {};
        drm_version ver{};
        ver.name_len = sizeof(drv) - 1;
        ver.name = drv;
        const bool i915 = ioctl(fd, DRM_IOCTL_VERSION, &ver) == 0 && std::strcmp(drv, "i915") == 0;
        close(fd);
        return i915;
    }
#else
    (void) pci;
#endif
    return false;
}

// A card of the xe driver whose VRAM a process that just ended is still giving back: the driver clears and frees it with
// jobs of its own on the copy engine, and a start that took tens of GB meanwhile was refused its memory, then the
// driver's own copy job faulted again and again and the GPU was wedged until a reboot (the B70, Linux 7.0, 2026-10-21:
// an engine started right after the one before it ended on SIGTERM).  The start waits while the driver's count of the
// VRAM in use falls, at most 20 s; where nothing is being given back that is one reading 100 ms after the first.
void settle_vram(const sycl::device& d) {
    if (integrated_gpu(d) || !d.has(sycl::aspect::ext_intel_pci_address)) return;
    std::string pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
    for (auto& c : pci) c = (char) std::tolower((unsigned char) c);
    uint64_t total = 0, used = 0;
    if (!xe_vram(pci, total, used)) return;
    constexpr uint64_t kStill = 64ull << 20;   // less than this given back in 100 ms: settled
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t first = used;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        uint64_t now = 0;
        if (!xe_vram(pci, total, now)) return;
        const bool falling = now + kStill < used;
        used = now;
        if (!falling || std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) break;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (s > 0.5)
        std::fprintf(stderr, "Xe runtime: GPU %s: waited %.1f s while the driver gave back %llu MiB of VRAM\n",
                     pci.c_str(), s, (unsigned long long) ((first > used ? first - used : 0) >> 20));
}

bool driver_free_memory(const sycl::device& d, uint64_t& free) {
    if (integrated_gpu(d) || !d.has(sycl::aspect::ext_intel_pci_address)) return false;
    std::string pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
    for (auto& c : pci) c = (char) std::tolower((unsigned char) c);
    uint64_t total = 0, used = 0;
    if (!xe_vram(pci, total, used)) return false;
    total = std::min(total, (uint64_t) d.get_info<sycl::info::device::global_mem_size>());
    free = total > used ? total - used : 0;
    return true;
}
}  // namespace

// The SYCL runtime's free memory (ext_intel_free_memory) comes on Level Zero from the driver's usable-memory property,
// or, from a driver without it (compute-runtime 26.05), from sysman, whose zesInit fails for every GPU when one of
// them is not the driver's (a Coffee Lake UHD 630 beside an Arc Pro B60: the B60's free VRAM read 0, and the warm-up
// kept the 0 in the config; technicalife.net, 2026-11-01).  There the xe driver is asked; capped at the size the SYCL
// runtime reports, the B60's 23256 MiB of the driver's 24480.
bool device_free_memory(const sycl::device& d, uint64_t& free) {
    if (d.has(sycl::aspect::ext_intel_free_memory)) {
        try {
            free = d.get_info<sycl::ext::intel::info::device::free_memory>();
            // ROCm's figure can come back below zero as an unsigned value: 140 MiB short read as 17592186044276 MiB
            // free (RX 9060 XT, ROCm 7.14, 2026-11-15).  More than the device has is none left.
            if (free > d.get_info<sycl::info::device::global_mem_size>()) free = 0;
            return true;
        } catch (const sycl::exception&) {
            return driver_free_memory(d, free);
        }
    }
    return driver_free_memory(d, free);
}

// upstream 004b15cd (#409): an integrated GPU's memory is the system RAM, and what its runtime reports free need not
// subtract the CPU's allocations (an AMD APU's figure counts its whole shared pool).  The engine takes the smaller of
// that figure and what the OS can give back, less STRATA_UMA_HEADROOM_GIB (default 6) for the system.
void apply_os_budget(const std::string& pci, uint64_t& free) {
    uint64_t budget = 0, usage = 0;
    std::string why;
    if (strata::platform::gpu_local_memory_budget(pci, budget, usage, why))
        free = std::min(free, budget > usage ? budget - usage : 0);
}

void apply_uma_limit(uint64_t& free, uint64_t total) {
    static const uint64_t headroom = [] {
        const char* v = strata::env_get("UMA_HEADROOM_GIB");
        const long long g = v != nullptr ? std::strtoll(v, nullptr, 10) : 6;
        return (uint64_t) std::max(0LL, g) << 30;
    }();
    uint64_t os_total = 0, avail = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return;
    os_total = (uint64_t) ms.ullTotalPhys;
    avail = (uint64_t) ms.ullAvailPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t kb = 0;
    while ((os_total == 0 || avail == 0) && meminfo >> key >> kb) {
        if (key == "MemTotal:")
            os_total = kb << 10;
        else if (key == "MemAvailable:")
            avail = kb << 10;
        else
            meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (avail == 0) return;   // no figure to cut to: the device's own stands
#endif
    // A BIOS carve-out is a pool of its own: the device reports more memory than the OS has, so what the OS can give
    // back says nothing about it (the Strix Halo's 64 GiB carve-out beside the 62 GiB the OS sees; the free figure is
    // the carve-out's).  Where the device's total is within the OS's RAM, the two are the same pool and the OS's
    // figure binds (an Intel iGPU, an APU with a small carve-out).
    if (os_total != 0 && total > os_total) return;
    free = std::min(free, avail > headroom ? avail - headroom : 0);
}

// The used memory is the larger of the report's and what the process holds: the B70 reports its global_mem_size
// (31023 MiB) less than its 32 GiB and its free memory as the smaller of that and what is free, so the first 1.4 GiB a
// process takes do not show (2026-10-19); llm-jp-4.1 Q4_K_M under 6144 MiB took 7.3 GiB.
void apply_vram_limit(uint64_t& free, uint64_t& total, uint64_t held) {
    const uint64_t lim = vram_limit_bytes();
    if (lim == 0 || lim >= total) return;
    const uint64_t used = std::max(total > free ? total - free : 0, held);
    total = lim;
    free = lim > used ? lim - used : 0;
}

// The room no plan counts, of any device's memory: see the declaration.  A caller that cannot read the device's
// memory (total 0) gets its own floor, the room measured on the smallest card.
uint64_t vram_room(uint64_t total_bytes, uint64_t parts, uint64_t least_bytes, uint64_t most_bytes) {
    const uint64_t floor_b = std::min(least_bytes, most_bytes);
    const uint64_t room = std::clamp<uint64_t>(total_bytes / 32, floor_b, std::max(floor_b, most_bytes));
    return parts > 1 ? room / parts : room;
}

uint64_t unreported_vram_margin(const std::string& pci, uint64_t total) {
    static std::mutex mu;
    static std::vector<std::pair<std::string, bool>> seen;   // the driver of each card asked once
    std::lock_guard<std::mutex> lock(mu);
    auto it = std::find_if(seen.begin(), seen.end(), [&](const auto& e) { return e.first == pci; });
    if (it == seen.end()) it = seen.insert(seen.end(), {pci, i915_gpu(pci)});
    // Measured on an A380 (6 GB, i915, compute-runtime 26.05, 2026-11-03): with the engine's own allocations counted, a
    // cache that left 320 MiB of the 5783 lost its contents and one that left 440 MiB ran, and a probe filled 5120 MiB
    // in 256 MiB blocks and read all of it back.  A sixteenth (361 MiB there) beside the engines' own reserves (the
    // plain engine's 1/32, Qwen3.8's 700 MiB); a larger card's runtime takes more (its scratch grows with its units)
    return it->second ? total / 16 : 0;
}

uint64_t max_alloc_bytes() {
    static const uint64_t v = [] {
        const uint64_t dev = Runtime::get().device().get_info<sycl::info::device::max_mem_alloc_size>();
        const uint64_t lim = max_alloc_limit_bytes();
        const uint64_t claim = lim != 0 && lim < dev ? lim : dev;
        // HIP's adapter can report a largest allocation far below what the device gives: 32 MiB for a gfx1151 that
        // hands out 24 GiB in one call (Strix Halo, 2026-11-13).  A claim that small splits the expert tier into
        // pieces too small for the prompt path to borrow its chunk across (74 MiB for 16 tokens), so it is measured:
        // allocations twice the claim up to 1 GiB, the largest that succeeds.  Where the claim is the device's
        // limit the first try fails and the claim stands; the tier's own allocation is what reports a refusal, and
        // its pieces fallback covers that.
        const uint64_t total = Runtime::get().device().get_info<sycl::info::device::global_mem_size>();
        if (claim >= (1ull << 30) || total < (8ull << 30)) return claim;
        uint64_t good = claim;
        for (uint64_t want = claim * 2; want <= (1ull << 30); want *= 2) {
            void* p = strata::gpu::alloc_device((size_t) want);
            if (p == nullptr) break;
            strata::gpu::free(p);
            good = want;
        }
        return good;
    }();
    return v;
}

DeviceInfo device_info(int ordinal) {
    if (ordinal < 0 || ordinal >= Runtime::count())
        throw DeviceError("no engine device " + std::to_string(ordinal));
    const auto& d = engine_devices()[(size_t) ordinal];
    DeviceInfo out;
    out.ordinal = ordinal;
    out.name = d.get_info<sycl::info::device::name>();
    out.backend = d.get_backend() == sycl::backend::ext_oneapi_level_zero ? "Level Zero"
                  : d.get_backend() == sycl::backend::ext_oneapi_cuda  ? "CUDA"
                  : d.get_backend() == sycl::backend::ext_oneapi_hip   ? "HIP"
                                                                       : "other";
    out.vendor_id = d.get_info<sycl::info::device::vendor_id>();
    out.device_id = d.has(sycl::aspect::ext_intel_device_id) ? d.get_info<sycl::ext::intel::info::device::device_id>() : 0;
    out.driver_version = d.get_info<sycl::info::device::driver_version>();
    out.platform_version = d.get_platform().get_info<sycl::info::platform::version>();
    out.total_bytes = d.get_info<sycl::info::device::global_mem_size>();
    out.free_bytes_known = device_free_memory(d, out.free_bytes);   // never zero for unknown
    uint64_t free = out.free_bytes_known ? out.free_bytes : out.total_bytes;
    if (out.free_bytes_known && !integrated(d)) apply_os_budget(pci_of(d), free);
    apply_vram_limit(free, out.total_bytes);
    if (out.free_bytes_known) out.free_bytes = free;
    out.compute_units = d.get_info<sycl::info::device::max_compute_units>();
    out.pci = pci_of(d);
    if (d.get_backend() == sycl::backend::ext_oneapi_hip) {   // gcnArchName: "gfx1030:sramecc-:xnack-" on some
        const std::string v = d.get_info<sycl::info::device::version>();
        out.arch = v.substr(0, v.find(':'));
    }
    out.integrated = integrated(d);
    out.subgroup_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    out.fp64 = d.has(sycl::aspect::fp64);
    out.host_usm = d.has(sycl::aspect::usm_host_allocations);
    out.device_usm = d.has(sycl::aspect::usm_device_allocations);
    return out;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison) : capacity_(bytes), ordinal_(ordinal) {
    if (!bytes || bytes > std::numeric_limits<size_t>::max())
        throw DeviceError("DeviceArena requires nonzero bytes");
    auto& runtime = Runtime::at(ordinal);
    runtime_ = &runtime;
    // 4 KB aligned by hand: the CUDA adapter refuses an alignment its driver does not promise (cuMemAlloc gives
    // 256 bytes; UNSUPPORTED_ALIGNMENT whenever the pointer comes back unaligned).  Level Zero returns pages, so
    // the offset is 0 there.
    raw_ = sycl::malloc_device(bytes + 4095, runtime.device(), runtime.context());
    if (!raw_) throw DeviceError("device USM allocation failed for " + std::to_string(bytes) + " bytes");
    base_ = (void*) (((uintptr_t) raw_ + 4095) & ~(uintptr_t) 4095);
    try {
        // in 1 GiB pieces: a single memset past 4 GiB never completes on the B70 (see gpu.cpp)
        if (poison) {
            sycl::event e;
            for (uint64_t o = 0; o < bytes; o += 1ull << 30)
                e = runtime.compute().memset((uint8_t*) base_ + o, 0xff, std::min<uint64_t>(1ull << 30, bytes - o));
            runtime.wait(e, "arena poison fill");
        }
    } catch (...) { runtime.free(raw_); raw_ = base_ = nullptr; throw; }
}

DeviceArena::~DeviceArena() { runtime_->free(raw_); }

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (!align || (align & (align - 1))) throw DeviceError("alignment must be a power of two");
    const uintptr_t address = reinterpret_cast<uintptr_t>(base_) + used_;
    const uint64_t padding = (align - (address & (align - 1))) & (align - 1);
    if (padding > capacity_ - used_ || bytes > capacity_ - used_ - padding)
        throw DeviceError("DeviceArena capacity exceeded");
    const uint64_t offset = used_ + padding;
    used_ = offset + bytes;
    return static_cast<uint8_t*>(base_) + offset;
}
}  // namespace strata::core
