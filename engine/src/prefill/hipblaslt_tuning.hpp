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
    std::string kernel_name;   ///< V2 (ROCm 10.2+ hipBLASLt >= 100500): stable getAlgoName() kernel identifier
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
        int format_v = 1;
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
                if (!(row >> magic >> arch >> version) || (row >> extra) ||
                    (magic != "STRATA_HIPBLASLT_TUNING_V1" && magic != "STRATA_HIPBLASLT_TUNING_V2")) {
                    err = "invalid tuning header at line " + std::to_string(line_number);
                    return false;
                }
                format_v = (magic == "STRATA_HIPBLASLT_TUNING_V2") ? 2 : 1;
                // A table's solution IDs are scoped to its architecture FAMILY:
                // gfx1200/gfx1201 (RDNA4), gfx1100/gfx1101/gfx1102 (RDNA3), and gfx1150..gfx1153 (RDNA3.5).
                // V1 numeric solution IDs require an exact hipBLASLt version match; V2 carries getAlgoName()
                // kernel names, which ROCm 10.2.0 (hipBLASLt 1.5.0 = 100500) guarantees stable across >= 100500.
                const auto is_rdna4 = [](const std::string& a) { return a == "gfx1200" || a == "gfx1201"; };
                const auto is_rdna3 = [](const std::string& a) {
                    return a == "gfx1100" || a == "gfx1101" || a == "gfx1102";
                };
                const auto is_rdna35 = [](const std::string& a) {
                    return a == "gfx1150" || a == "gfx1151" || a == "gfx1152" || a == "gfx1153";
                };
                if (arch != expected_arch &&
                    !(is_rdna4(arch) && is_rdna4(expected_arch)) &&
                    !(is_rdna3(arch) && is_rdna3(expected_arch)) &&
                    !(is_rdna35(arch) && is_rdna35(expected_arch))) {
                    err = "tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                const bool v2_cross_ok = (format_v == 2 && version >= 100500 && expected_version >= 100500);
                if (version != expected_version && !v2_cross_ok) {
                    err = "hipBLASLt version mismatch: file=" + std::to_string(version) +
                          " runtime=" + std::to_string(expected_version);
                    return false;
                }
                arch_ = std::move(arch);
                version_ = version;
                format_version_ = format_v;
                saw_header = true;
                continue;
            }

            std::string type_name;
            int n = 0, k = 0, ldy = 0, t_bucket = 0, solution_id = -1;
            std::string kname, extra;
            const bool have_cols = (format_v == 2)
                                       ? (static_cast<bool>(row >> type_name >> n >> k >> ldy >> t_bucket >> solution_id >> kname) &&
                                          !(row >> extra) && !kname.empty())
                                       : (static_cast<bool>(row >> type_name >> n >> k >> ldy >> t_bucket >> solution_id) &&
                                          !(row >> extra));
            if (!have_cols ||
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
            parsed.kernel_name = std::move(kname);
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
    int format_version() const { return format_version_; }

private:
    std::string arch_;
    int version_ = 0;
    int format_version_ = 1;
    std::vector<TuningRow> rows_;
};

}  // namespace strata::prefill::hipblaslt
