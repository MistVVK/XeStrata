#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::prefill::hipblaslt {

enum class InputType : uint8_t { f16, bf16 };

struct TuningRow {
    InputType type = InputType::bf16;
    int n = 0;
    int k = 0;
    int ldy = 0;
    int t_bucket = 0;
    int solution_id = -1;
};

class TuningTable {
public:
    bool load(const std::string& path, const std::string& expected_arch, int expected_version, std::string& err) {
        std::ifstream input(path);
        if (!input) {
            err = "cannot open tuning file";
            return false;
        }

        bool saw_header = false;
        std::string line;
        size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;

            std::istringstream row(line.substr(first));
            if (!saw_header) {
                std::string magic;
                std::string arch;
                int version = 0;
                std::string extra;
                if (!(row >> magic >> arch >> version) || (row >> extra) || magic != "STRATA_HIPBLASLT_TUNING_V1") {
                    err = "invalid tuning header at line " + std::to_string(line_number);
                    return false;
                }
                // A table's solution IDs are scoped to its hipBLASLt version and its architecture FAMILY:
                // gfx1200/gfx1201 (RDNA4) share them, as gfx1100/gfx1101/gfx1102 (RDNA3) do.  A card can report
                // its sibling - an R9700 (gfx1201) under HSA_OVERRIDE_GFX_VERSION=12.0.0 reports gfx1200 - and
                // must still load its own calibration.  A solution ID that does not exist on the running card
                // fails the per-solution gate in prefill/gemm.cu (getAlgosFromIndex / matmulIsAlgoSupported) and
                // falls back to hipBLASEx, so the header gate does not need to be exact.
                const auto is_rdna4 = [](const std::string& a) { return a == "gfx1200" || a == "gfx1201"; };
                const auto is_rdna3 = [](const std::string& a) {
                    return a == "gfx1100" || a == "gfx1101" || a == "gfx1102";
                };
                if (arch != expected_arch &&
                    !(is_rdna4(arch) && is_rdna4(expected_arch)) &&
                    !(is_rdna3(arch) && is_rdna3(expected_arch))) {
                    err = "tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                if (version != expected_version) {
                    err = "hipBLASLt version mismatch: file=" + std::to_string(version) +
                          " runtime=" + std::to_string(expected_version);
                    return false;
                }
                arch_ = std::move(arch);
                version_ = version;
                saw_header = true;
                continue;
            }

            std::string type_name;
            int n = 0, k = 0, ldy = 0, t_bucket = 0, solution_id = -1;
            std::string extra;
            if (!(row >> type_name >> n >> k >> ldy >> t_bucket >> solution_id) || (row >> extra) ||
                (type_name != "f16" && type_name != "bf16") || n <= 0 || k <= 0 || ldy < n || t_bucket <= 0 ||
                solution_id < 0) {
                err = "invalid tuning row at line " + std::to_string(line_number);
                return false;
            }
            TuningRow parsed;
            parsed.type = type_name == "f16" ? InputType::f16 : InputType::bf16;
            parsed.n = n;
            parsed.k = k;
            parsed.ldy = ldy;
            parsed.t_bucket = t_bucket;
            parsed.solution_id = solution_id;
            for (const auto& existing : rows_) {
                if (existing.type == parsed.type && existing.n == parsed.n && existing.k == parsed.k &&
                    existing.ldy == parsed.ldy && existing.t_bucket == parsed.t_bucket) {
                    err = "duplicate tuning key at line " + std::to_string(line_number);
                    return false;
                }
            }
            rows_.push_back(parsed);
        }
        if (!saw_header) {
            err = "tuning file has no header";
            return false;
        }
        return true;
    }

    const TuningRow* closest(InputType type, int n, int k, int ldy, int t) const {
        const TuningRow* best = nullptr;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (const auto& row : rows_) {
            if (row.type != type || row.n != n || row.k != k || row.ldy != ldy) continue;
            const int64_t distance = row.t_bucket >= t ? int64_t(row.t_bucket) - t : int64_t(t) - row.t_bucket;
            if (distance < best_distance ||
                (distance == best_distance && (!best || row.t_bucket < best->t_bucket))) {
                best = &row;
                best_distance = distance;
            }
        }
        return best;
    }

    const std::vector<TuningRow>& rows() const { return rows_; }
    const std::string& arch() const { return arch_; }
    int version() const { return version_; }

private:
    std::string arch_;
    int version_ = 0;
    std::vector<TuningRow> rows_;
};

/// Parse `/sys/module/amdgpu/parameters/cwsr_enable`: returns 0 when explicitly disabled ("0"), 1 when enabled
/// (non-zero integer such as "1" or "-1"), or -1 when empty/unparseable.
inline int parse_cwsr_enable_value(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return -1;
    const auto last = text.find_last_not_of(" \t\r\n");
    const std::string tok = text.substr(first, last - first + 1);
    std::istringstream is(tok);
    int v = 0;
    char extra = 0;
    if (!(is >> v) || (is >> extra)) return -1;
    return v == 0 ? 0 : 1;
}

/// True when kernel cmdline contains the exact parameter token `amdgpu.cwsr_enable=0`.
inline bool cmdline_has_cwsr_disabled(const std::string& cmdline) {
    std::istringstream is(cmdline);
    std::string tok;
    while (is >> tok) {
        if (tok == "amdgpu.cwsr_enable=0") return true;
    }
    return false;
}

/// #6574 / #6132: On Linux gfx1103 (Radeon 780M), enabling hipBLASLt tuning tables while amdgpu CWSR is active
/// (`cwsr_enable != 0`) can trigger KFD/MES `REMOVE_QUEUE` warm-up crashes. Returns false only on Linux gfx1103
/// when `cwsr_enable` is known non-zero and `amdgpu.cwsr_enable=0` is absent from `/proc/cmdline`.
inline bool gfx1103_hipblaslt_cwsr_safe(const std::string& arch, bool linux_os, int cwsr_enable,
                                        bool cmdline_disabled) {
    if (!linux_os || arch != "gfx1103") return true;
    if (cmdline_disabled || cwsr_enable == 0) return true;
    return cwsr_enable < 0;   // unreadable sysfs: do not falsely block
}

/// hipBLAS status codes that fail before launching a kernel on the device (safe to fall back to plain hipBLAS even
/// when `beta != 0`, since the output buffer `Y` has not been modified).
/// Values match `hipblasStatus_t`: 1=NOT_INITIALIZED, 2=ALLOC_FAILED, 3=INVALID_VALUE, 4=MAPPING_ERROR,
/// 8=ARCH_MISMATCH, 10=NOT_SUPPORTED.
inline bool hipblaslt_prelaunch_error(int status) {
    return status == 1 || status == 2 || status == 3 || status == 4 || status == 8 || status == 10;
}

}  // namespace strata::prefill::hipblaslt
