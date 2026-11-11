// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/control_vector.cpp - see control_vector.hpp.  Moved from generate.cpp and plain_generate.cpp, the
// loading and the tables one; the wording of the two engines' warnings kept as their arguments.

#include "strata/core/control_vector.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

bool load_control_vectors(const std::vector<std::pair<std::string, float>>& files, int64_t L, int64_t N, int64_t hc,
                          int first, int last, int mode, int single, const std::string& hint_expected,
                          const std::string& who, bool hint_named, ControlVectorLoad& out, std::string& err) {
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != hint_expected) {
                if (hint_named)
                    std::fprintf(stderr, "%s: %s was made for '%s', not %s\n", who.c_str(), path.c_str(),
                                 hint->s.c_str(), hint_expected.c_str());
                else
                    std::fprintf(stderr, "%s: %s was made for '%s'\n", who.c_str(), path.c_str(), hint->s.c_str());
            }
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    first = first <= 0 ? 1 : first;
    last = last <= 0 || last >= L ? (int) L - 1 : last;
    single = mode == 0 ? single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (std::ptrdiff_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) {
        err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last);
        return false;
    }
    if (!strata::kernels::cvec_upload(dir, s, mode, first, last, N, hc, err)) return false;
    out.dir = std::move(dir);
    out.scale = std::move(s);
    out.first = first;
    out.last = last;
    out.single = single;
    out.steered = steered;
    out.summary = std::string(mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" +
                  std::to_string(last) + (single >= 0 ? ":single" + std::to_string(single) : "");
    return true;
}

}  // namespace strata::core
