// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// src/image/sam3_video.cpp - see include/strata/image/sam3_video.hpp.  A frame's steps, in the order of
// vLLM-Omni's step: detect; propagate each bucket from its memory; associate; hot start;
// recondition (every recondition_every frames); suppress the recently occluded; encode the frame's memory; add the new
// objects and remove the dropped ones; record the frame's outputs, shown hot_start frames later.  The masks the rules
// compare are the 4x level's (288 x 288), binary, on the host; the network's work is the GPU's.

#include "strata/image/sam3_video.hpp"

#include "strata/core/gpu.hpp"
#include "strata/image/aux_kernels.hpp"
#include "strata/kernels/elementwise.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr int64_t kPad = -1, kGone = -2;   // a slot never used, a removed object's (not used again)
constexpr float kNoObj = -1024.f;          // a mask's logits where the decoder saw no object
constexpr int kStageMask = 100;            // Sam3's upload_async stages of the slots' masks (100 + slot)

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    explicit Dev(size_t n) { grow(n); }
    ~Dev() { gpu::free(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void grow(size_t n) {
        if (n <= bytes) return;
        gpu::free(p);
        p = nullptr;
        if (!gpu::alloc_device(&p, n)) throw std::runtime_error("SAM 3 video: GPU memory");
        bytes = n;
    }
    template<class T> T* as() const { return static_cast<T*>(p); }
};

using Binary = std::vector<uint8_t>;

Binary binary(const std::vector<float>& logits) {
    Binary b(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) b[i] = logits[i] > 0.f;
    return b;
}

int64_t area(const Binary& a) {
    int64_t n = 0;
    for (uint8_t v : a) n += v;
    return n;
}

int64_t intersection(const Binary& a, const Binary& b) {
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += a[i] & b[i];
    return n;
}

// the share of the smaller mask the two have in common (0 when one is empty), and the intersection over the union
float iom(const Binary& a, const Binary& b, int64_t aa, int64_t ab) {
    const int64_t small = std::min(aa, ab);
    return small > 0 ? (float) intersection(a, b) / (float) small : 0.f;
}

float iou(const Binary& a, const Binary& b, int64_t aa, int64_t ab) {
    const int64_t i = intersection(a, b), u = aa + ab - i;
    return u > 0 ? (float) i / (float) u : 0.f;
}

}  // namespace

struct Sam3Video::Impl {
    Sam3& sam;
    Config cfg;
    int64_t W, H, frames;
    std::vector<Sam3VideoWords> words;
    std::vector<Sam3VideoClick> clicks;
    int64_t d = 0, G = 0, P = 0, P0 = 0, S = 0;
    static constexpr int64_t M = Sam3::kMux;

    // a bucket's frame: its memory rows [P][2d] (the frame's image features | the memory, FP16), its objects'
    // pointers [M][d], the slots given as clean masks on it (its cond channels)
    struct FrameMem {
        bool cond = false;
        Dev mem16, ptr;
        std::array<float, M> cond_slot{};
    };
    struct Bucket {
        std::array<int64_t, M> slot;
        std::map<int64_t, std::unique_ptr<FrameMem>> mem;
        Bucket() { slot.fill(kPad); }
        bool any() const {
            return std::any_of(slot.begin(), slot.end(), [](int64_t s) { return s >= 0; });
        }
    };
    // what found an object: its words' index, or -1 - its click's (a click's object matches no detection)
    struct Obj {
        int64_t id = 0, group = 0, bucket = 0, slot = 0, first = 0, last_occluded = -1;
        int unmatched = 0, consecutive = 0;
        bool confirmed = false, user = false;   // user: a click's (shown at once, never removed)
        float score = 0;          // its first detection's
        float track_logit = 0;    // this frame's tracker score
        std::vector<float> low;   // this frame's mask logits [P0]
    };
    struct Det {
        int64_t query = 0, group = 0;
        float prob = 0;
        float box[4] = {0, 0, 0, 0};   // cx, cy, w, h on [0, 1]
        std::vector<float> low;
        Binary bin;
        int64_t area = 0;
    };
    // a frame's record until it is shown
    struct Pending {
        int64_t frame = 0;
        std::vector<std::pair<int64_t, std::vector<float>>> masks;
        std::map<int64_t, float> score, track;   // the first detection's, this frame's tracker probability
    };

    std::vector<std::unique_ptr<Bucket>> buckets;
    std::vector<Obj> objs;   // the objects tracked, in their ids' order
    std::map<std::pair<int64_t, int64_t>, int> overlap;   // (earlier id, later id) -> frames sharing a detection
    int64_t next_id = 0, frame = 0;
    std::deque<Pending> held;
    std::set<int64_t> removed_unconfirmed;
    std::map<int64_t, std::set<int64_t>> unconfirmed;   // by frame: the objects not yet confirmed after it
    std::map<int64_t, float> first_score;               // every object's, removed ones' too
    std::map<int64_t, int64_t> group_of;                // every object's group
    // the GPU's buffers
    Dev prev_dev, low_dev, mem_rows, tpos_rows, ptr_pos, pix_mem, ptr_tmp, ptr16, masks1008, tokens, img16;

    Impl(Sam3& s, int64_t w, int64_t h, int64_t n, std::vector<Sam3VideoWords> wd, std::vector<Sam3VideoClick> ck,
         Config c)
        : sam(s), cfg(c), W(w), H(h), frames(n), words(std::move(wd)), clicks(std::move(ck)) {}

    void* stream() const { return sam.s_; }

    // ------------------------------------------------------------------------------------------- the frame's parts
    // each of the words looked for on frame f: the detector's queries past the threshold away from the edges, by IoM
    // NMS, in the queries' order (a new object's id follows it); the words in order
    std::vector<Det> detect(int64_t f) {
        std::vector<Det> out;
        int64_t active = 0, done = 0;
        for (const Sam3VideoWords& w : words) active += w.from <= f;
        for (size_t g = 0; g < words.size(); ++g) {
            if (words[g].from > f) continue;
            if (done++ == 0) sam.detect_neck(active > 1);
            else sam.detect_restore();
            sam.text(words[g].ids);
            std::vector<Det> got = detect_words(g);
            for (Det& dd : got) out.push_back(std::move(dd));
        }
        return out;
    }

    std::vector<Det> detect_words(size_t g) {
        const Sam3::Detection det = sam.detect(cfg.score_threshold);
        std::vector<int64_t> keep;
        // best first; a box whose center (in the frame's [0, 1]) is not more than half the frame less the margin away
        // from its middle, on both axes, is left out
        const float reach = 0.5f - cfg.edge;
        for (int64_t q : det.keep) {
            const float* box = det.box.data() + q * 4;
            if (std::fabs(box[0] - 0.5f) < reach && std::fabs(box[1] - 0.5f) < reach) keep.push_back(q);
        }
        std::vector<Det> out;
        if (keep.empty()) return out;
        sam.mask_prep();
        low_dev.grow((size_t) (keep.size() * P0) * 4);
        sam.mask_logits(keep.data(), (int64_t) keep.size(), low_dev.as<float>());
        sam.sync();
        std::vector<float> all((size_t) (keep.size() * P0));
        gpu::copy(all.data(), low_dev.p, all.size() * 4);
        for (size_t i = 0; i < keep.size(); ++i) {
            Det dd;
            dd.query = keep[i];
            dd.group = (int64_t) g;
            dd.prob = det.prob[(size_t) keep[i]];
            std::copy(det.box.begin() + keep[i] * 4, det.box.begin() + keep[i] * 4 + 4, dd.box);
            dd.low.assign(all.begin() + (int64_t) i * P0, all.begin() + (int64_t) (i + 1) * P0);
            dd.bin = binary(dd.low);
            dd.area = area(dd.bin);
            bool clear = true;
            for (const Det& k : out)
                if (iom(dd.bin, k.bin, dd.area, k.area) > cfg.nms_iom) {
                    clear = false;
                    break;
                }
            if (clear) out.push_back(std::move(dd));
        }
        std::stable_sort(out.begin(), out.end(), [](const Det& a, const Det& b) { return a.query < b.query; });
        return out;
    }

    // the clicks of frame f: each one's mask (the interactive decoder's; an object's redone with its mask before)
    struct Clicked {
        size_t click = 0;
        Obj* obj = nullptr;   // the object redone, or none: a new one
        std::vector<float> low;
    };
    std::vector<Clicked> click(int64_t f) {
        std::vector<Clicked> out;
        for (size_t c = 0; c < clicks.size(); ++c) {
            const Sam3VideoClick& k = clicks[c];
            if (k.frame != f || k.points.empty()) continue;
            Clicked r;
            r.click = c;
            r.obj = k.object >= 0 ? find(k.object) : nullptr;
            std::vector<std::array<float, 3>> pts = k.points;
            for (auto& p : pts) {   // to the picture's [0, 1] (the frame is stretched to its square)
                p[0] = std::clamp(p[0] / (float) W, 0.f, 1.f);
                p[1] = std::clamp(p[1] / (float) H, 0.f, 1.f);
            }
            const float* prev = nullptr;
            if (r.obj != nullptr) {
                prev_dev.grow((size_t) P0 * 4);
                gpu::copy(prev_dev.p, r.obj->low.data(), (size_t) P0 * 4);
                prev = prev_dev.as<float>();
            }
            sam.point_prompt(pts, prev, r.low, nullptr);
            sam.sync();
            out.push_back(std::move(r));
        }
        return out;
    }

    // a bucket's memory for frame f: the closest cond frames (up to max_cond_frames), the frames before it, the
    // pointers of the cond frames before it and of the frames up to max_pointers back
    void propagate(Bucket& b, int64_t f, std::map<int64_t, std::vector<float>>& masks, std::map<int64_t, float>& logit,
                   FrameMem& now) {
        const int64_t nmem = sam.mem_frames();
        std::vector<int64_t> cond;
        for (const auto& [t, m] : b.mem)
            if (m->cond) cond.push_back(t);
        std::set<int64_t> selected;
        if ((int64_t) cond.size() <= cfg.max_cond_frames) {
            selected.insert(cond.begin(), cond.end());
        } else {
            int64_t before = -1, after = -1;
            for (int64_t t : cond) {
                if (t < f) before = std::max(before, t);
                if (t >= f && (after < 0 || t < after)) after = t;
            }
            if (before >= 0) selected.insert(before);
            if (after >= 0) selected.insert(after);
            std::vector<int64_t> rest;
            for (int64_t t : cond)
                if (!selected.count(t)) rest.push_back(t);
            std::stable_sort(rest.begin(), rest.end(), [&](int64_t a, int64_t c) { return std::llabs(a - f) < std::llabs(c - f); });
            for (int64_t t : rest)
                if ((int64_t) selected.size() < cfg.max_cond_frames) selected.insert(t);
        }
        std::vector<std::pair<int64_t, int64_t>> spatial;   // (frame, its time position's row)
        for (int64_t t : selected) {
            const int64_t dist = f - t;
            spatial.emplace_back(t, dist > 0 && dist < nmem ? nmem - dist - 1 : nmem - 1);
        }
        for (int64_t tp = 1; tp < nmem; ++tp) {
            const int64_t t = f - (nmem - tp);
            const auto it = b.mem.find(t);
            if (it != b.mem.end() && !selected.count(t)) spatial.emplace_back(t, nmem - tp - 1);
        }
        std::vector<std::pair<int64_t, int64_t>> ptrs;   // (frame, its distance)
        const int64_t maxp = std::min<int64_t>(std::max<int64_t>(frames, f + 1), cfg.max_pointers);
        for (int64_t t : selected)
            if (t <= f) ptrs.emplace_back(t, f - t);
        for (int64_t td = 1; td < maxp; ++td) {
            const int64_t t = f - td;
            if (t < 0) break;
            if (b.mem.count(t) && !selected.count(t)) ptrs.emplace_back(t, td);
        }
        const int64_t nsp = (int64_t) spatial.size() * P, N = nsp + (int64_t) ptrs.size() * M;
        mem_rows.grow((size_t) (N * 2 * d) * 2);
        auto* rows = mem_rows.as<uint16_t>();
        void* s = stream();
        for (size_t i = 0; i < spatial.size(); ++i)
            gpu::copy_async(rows + (int64_t) i * P * 2 * d, b.mem.at(spatial[i].first)->mem16.p, (size_t) (P * 2 * d) * 2, s);
        if (!ptrs.empty()) {
            gpu::memset_async(rows + nsp * 2 * d, 0, (size_t) (ptrs.size() * M * 2 * d) * 2, s);
            ptr16.grow((size_t) (M * d) * 2);
            for (size_t i = 0; i < ptrs.size(); ++i) {
                kernels::f32_to_f16_bulk(b.mem.at(ptrs[i].first)->ptr.as<float>(), ptr16.as<uint16_t>(), M * d, s);
                gpu::copy2d_async(rows + nsp * 2 * d + (int64_t) i * M * 2 * d + d, (size_t) (2 * d) * 2,
                                  ptr16.p, (size_t) d * 2, (size_t) d * 2, (size_t) M, s);
            }
        }
        tpos_rows.grow((size_t) (std::max<size_t>(spatial.size(), 1) * d) * 4);
        for (size_t i = 0; i < spatial.size(); ++i)
            gpu::copy_async(tpos_rows.as<float>() + (int64_t) i * d, sam.mem_tpos() + spatial[i].second * d,
                            (size_t) d * 4, s);
        ptr_pos.grow((size_t) (std::max<size_t>(ptrs.size(), 1) * d) * 4);
        std::vector<int64_t> pos;
        pos.reserve(ptrs.size());
        for (const auto& p : ptrs) pos.push_back(p.second);
        sam.ptr_positions(pos, maxp, ptr_pos.as<float>());
        pix_mem.grow((size_t) (P * d) * 4);
        sam.mem_attention(sam.track_level(2), sam.sine_positions(), rows, 1, N, nsp, sam.sine_positions(),
                          tpos_rows.as<float>(), ptr_pos.as<float>(), M, pix_mem.as<float>());
        // the multiplex decoder: each object's best mask by its IoU, its score, its pointer
        std::vector<uint8_t> valid((size_t) M);
        for (int64_t i = 0; i < M; ++i) valid[(size_t) i] = b.slot[(size_t) i] >= 0;
        const Sam3::MuxOut r = sam.mux_decode(pix_mem.as<float>(), sam.track_level(0), sam.track_level(1), valid, 1);
        sam.sync();
        std::vector<float> iou((size_t) (M * r.T)), score((size_t) M);
        gpu::copy(iou.data(), r.iou, iou.size() * 4);
        gpu::copy(score.data(), r.score, score.size() * 4);
        tokens.grow((size_t) (M * d) * 4);
        gpu::memset(tokens.p, 0, (size_t) (M * d) * 4);
        for (int64_t sl = 0; sl < M; ++sl) {
            const int64_t id = b.slot[(size_t) sl];
            if (id < 0) continue;
            int64_t best = 0;
            for (int64_t t = 1; t < r.T; ++t)
                if (iou[(size_t) (sl * r.T + t)] > iou[(size_t) (sl * r.T + best)]) best = t;
            std::vector<float> m((size_t) P0);
            if (score[(size_t) sl] > 0.f) gpu::copy(m.data(), r.masks + (sl * r.T + best) * P0, (size_t) P0 * 4);
            else std::fill(m.begin(), m.end(), kNoObj);
            masks[id] = std::move(m);
            logit[id] = score[(size_t) sl];
            gpu::copy(tokens.as<float>() + sl * d, r.tokens + (sl * r.T + best) * d, (size_t) d * 4);
        }
        now.ptr.grow((size_t) (M * d) * 4);
        sam.pointers(tokens.as<float>(), M, false, score, now.ptr.as<float>());
        for (int64_t sl = 0; sl < M; ++sl)   // the empty slots' pointers: none
            if (b.slot[(size_t) sl] < 0) gpu::memset_async(now.ptr.as<float>() + sl * d, 0, (size_t) d * 4, s);
        sam.sync();
    }

    // the frame's memory of bucket b from its objects' masks [P0] (the rest of the slots empty)
    void encode(Bucket& b, FrameMem& fm, const std::map<int64_t, const std::vector<float>*>& masks) {
        // the slots' masks to the GPU, the empty ones left as they are (the encoder reads only the valid)
        std::vector<float> valid((size_t) M, 0.f), is_obj((size_t) M, 0.f);
        std::vector<float> cond(fm.cond_slot.begin(), fm.cond_slot.end());
        low_dev.grow((size_t) (M * P0) * 4);
        for (int64_t sl = 0; sl < M; ++sl) {
            const int64_t id = b.slot[(size_t) sl];
            if (id < 0) continue;
            const auto it = masks.find(id);
            if (it == masks.end()) continue;
            const std::vector<float>& m = *it->second;
            sam.upload_async(low_dev.as<float>() + sl * P0, m.data(), (size_t) P0 * 4, kStageMask + (int) sl);
            valid[(size_t) sl] = 1.f;
            is_obj[(size_t) sl] = std::any_of(m.begin(), m.end(), [](float v) { return v > 0.f; }) ? 1.f : 0.f;
        }
        fm.mem16.grow((size_t) (P * 2 * d) * 2);
        // the frame's image features in the rows' first half, the memory in the second
        img16.grow((size_t) (P * d) * 2);
        kernels::f32_to_f16_bulk(sam.track_level(2), img16.as<uint16_t>(), P * d, stream());
        gpu::copy2d_async(fm.mem16.p, (size_t) (2 * d) * 2, img16.p, (size_t) d * 2, (size_t) d * 2, (size_t) P,
                          stream());
        sam.mem_encode(sam.track_level(2), low_dev.as<float>(), M, 4 * G, 4 * G, false, cond, is_obj, 1, nullptr,
                       fm.mem16.as<uint16_t>(), 2 * d, d, &valid);
    }

    // clean masks (detections' logits [n][P0]) given to slots: their pointers (the interactive decoder's) into fm
    void condition(FrameMem& fm, const std::vector<int64_t>& slots, const std::vector<const std::vector<float>*>& lows) {
        const int64_t n = (int64_t) slots.size();
        if (n == 0) return;
        void* s = stream();
        low_dev.grow((size_t) (n * P0) * 4);
        masks1008.grow((size_t) (n * S * S) * 4);
        for (int64_t i = 0; i < n; ++i) gpu::copy(low_dev.as<float>() + i * P0, lows[(size_t) i]->data(), (size_t) P0 * 4);
        for (int64_t i = 0; i < n; ++i) {
            aux_resize_into(low_dev.as<float>() + i * P0, 4 * G, 4 * G, 1, masks1008.as<float>() + i * S * S, S, S, 1, 0,
                            false, false, s);
        }
        aux_binarize(masks1008.as<float>(), n * S * S, masks1008.as<float>(), s);
        tokens.grow((size_t) (n * d) * 4);
        std::vector<float> score;
        sam.interactive(masks1008.as<float>(), n, tokens.as<float>(), score);
        ptr_tmp.grow((size_t) (n * d) * 4);
        sam.pointers(tokens.as<float>(), n, true, score, ptr_tmp.as<float>());
        fm.ptr.grow((size_t) (M * d) * 4);
        for (int64_t i = 0; i < n; ++i) {
            gpu::copy_async(fm.ptr.as<float>() + slots[(size_t) i] * d, ptr_tmp.as<float>() + i * d, (size_t) d * 4, s);
            fm.cond_slot[(size_t) slots[(size_t) i]] = 1.f;
        }
        sam.sync();
    }

    // a bucket's frame memory, its buffers those of a frame let go where there is one (a frame's 5 MB of rows
    // allocated and freed each frame cost the B70 about 2 ms of the memory's 4, 2026-11-07)
    std::vector<std::unique_ptr<FrameMem>> spare;
    std::unique_ptr<FrameMem> fresh_mem() {
        if (spare.empty()) return std::make_unique<FrameMem>();
        std::unique_ptr<FrameMem> m = std::move(spare.back());
        spare.pop_back();
        m->cond = false;
        m->cond_slot.fill(0.f);
        return m;
    }

    Obj* find(int64_t id) {
        for (Obj& o : objs)
            if (o.id == id) return &o;
        return nullptr;
    }

    // ------------------------------------------------------------------------------------------------ a frame
    void step(const QiRef& picture, Times& t) {
        const int64_t f = frame++;
        auto t0 = std::chrono::steady_clock::now();
        sam.trunk(picture, Filter::cv_cubic);
        sam.sync();
        t.trunk += since(t0);
        t0 = std::chrono::steady_clock::now();
        std::vector<Det> dets = detect(f);
        t.detect += since(t0);
        t0 = std::chrono::steady_clock::now();
        // the objects carried to this frame, bucket by bucket (their memory written after the rules)
        bool tracked = false;   // the propagation's neck, made once a frame where a bucket or a new object needs it
        const auto features = [&] {
            if (!tracked) sam.track_features();
            tracked = true;
        };
        if (std::any_of(buckets.begin(), buckets.end(), [](const auto& b) { return b->any(); })) features();
        std::map<int64_t, std::vector<float>> trk;
        std::map<int64_t, float> trk_logit;
        std::map<int64_t, std::unique_ptr<FrameMem>> now;   // by bucket
        for (size_t bi = 0; bi < buckets.size(); ++bi) {
            if (!buckets[bi]->any()) continue;
            auto fm = fresh_mem();
            propagate(*buckets[bi], f, trk, trk_logit, *fm);
            now[(int64_t) bi] = std::move(fm);
        }
        sam.sync();
        for (Obj& o : objs) {
            o.low = std::move(trk[o.id]);
            o.track_logit = trk_logit[o.id];
        }
        // the clicks: an object redone takes their mask (as sure of it as can be; reconditioned below), the rest are
        // new objects
        std::vector<Clicked> clicked = click(f);
        std::set<int64_t> redone;
        for (Clicked& c : clicked) {
            if (c.obj == nullptr) continue;
            Obj& o = *c.obj;
            o.low = c.low;
            o.track_logit = 20.f;
            o.user = o.confirmed = true;
            o.score = first_score[o.id] = 1.f;
            redone.insert(o.id);
        }
        t.track += since(t0);
        t0 = std::chrono::steady_clock::now();
        // association: by IoM, the detections against the masklets
        const size_t nd = dets.size(), no = objs.size();
        std::vector<Binary> ob(no);
        std::vector<int64_t> oa(no);
        for (size_t j = 0; j < no; ++j) {
            ob[j] = binary(objs[j].low);
            oa[j] = area(ob[j]);
        }
        std::vector<std::vector<float>> im(nd, std::vector<float>(no, 0.f));
        for (size_t i = 0; i < nd; ++i)
            for (size_t j = 0; j < no; ++j)   // a detection against its own words' masklets only
                im[i][j] = dets[i].group == objs[j].group ? iom(dets[i].bin, ob[j], dets[i].area, oa[j]) : 0.f;
        std::vector<char> matched_trk(no, 0), is_new(nd, 0);
        for (size_t i = 0; i < nd; ++i) {
            bool any = false;
            for (size_t j = 0; j < no; ++j) {
                if (im[i][j] >= cfg.trk_assoc) matched_trk[j] = 1;
                if (im[i][j] >= cfg.assoc) any = true;
            }
            is_new[i] = static_cast<char>(dets[i].prob >= cfg.new_det && !any);
        }
        // the pairs the reconditioning may use: a sure detection (high_conf, not a new object) and a masklet that are
        // each the other's only overlap of recondition_iom or more
        std::map<int64_t, size_t> high;   // object id -> detection
        {
            std::vector<int> det_hits(nd, 0), trk_hits(no, 0);
            std::vector<size_t> det_pair(nd, 0);
            for (size_t i = 0; i < nd; ++i)
                for (size_t j = 0; j < no; ++j)
                    if (im[i][j] >= cfg.recondition_iom) {
                        ++det_hits[i];
                        ++trk_hits[j];
                        det_pair[i] = j;
                    }
            for (size_t i = 0; i < nd; ++i)
                if (det_hits[i] == 1 && trk_hits[det_pair[i]] == 1 && dets[i].prob >= cfg.high_conf && !is_new[i])
                    high[objs[det_pair[i]].id] = i;
        }
        // hot start: the unmatched frames, the masklets sharing a detection (keep-alive is not counted: it would hide
        // nothing)
        std::set<int64_t> to_remove;
        {
            for (size_t j = 0; j < no; ++j)
                if (oa[j] > 0 && !matched_trk[j] && !objs[j].user) ++objs[j].unmatched;
            for (size_t i = 0; i < nd; ++i) {
                std::vector<size_t> js;
                for (size_t j = 0; j < no; ++j)
                    if (im[i][j] >= cfg.assoc) js.push_back(j);
                if (js.size() < 2) continue;
                for (size_t a = 0; a < js.size(); ++a)
                    for (size_t c = a + 1; c < js.size(); ++c) ++overlap[{objs[js[a]].id, objs[js[c]].id}];
            }
            const int64_t diff = f - cfg.hot_start;
            for (size_t j = 0; j < no; ++j) {
                const Obj& o = objs[j];
                const bool within = o.first > diff && !o.user;
                bool remove = within && o.unmatched >= cfg.hot_unmatched;
                if (!remove && within) {
                    int most = 0;
                    for (size_t a = 0; a < no; ++a) {
                        if (!(objs[a].first < o.first)) continue;
                        const auto it = overlap.find({std::min(objs[a].id, o.id), std::max(objs[a].id, o.id)});
                        if (it != overlap.end() && objs[a].id < o.id) most = std::max(most, it->second);
                    }
                    remove = most >= cfg.hot_duplicate;
                }
                if (remove) to_remove.insert(o.id);
            }
        }
        // reconditioning, every recondition_every frames: a masklet sure of itself and its sure detection agree
        std::map<int64_t, std::vector<int64_t>> recond_slots;   // by bucket
        std::map<int64_t, std::vector<const std::vector<float>*>> recond_masks;
        std::vector<std::vector<float>> recond_store;
        recond_store.reserve(high.size() + clicked.size());
        for (const Clicked& c : clicked) {   // the objects the clicks redid: their masks as clean ones
            if (c.obj == nullptr) continue;
            recond_store.push_back(c.low);
            recond_slots[c.obj->bucket].push_back(c.obj->slot);
            recond_masks[c.obj->bucket].push_back(&recond_store.back());
        }
        if (cfg.recondition_every > 0 && f % cfg.recondition_every == 0) {
            for (const auto& [id, di] : high) {
                Obj* o = find(id);
                if (o == nullptr || redone.count(id) || !(sigmoid(o->track_logit) > cfg.high_conf)) continue;
                const Det& dd = dets[di];
                for (int64_t p = 0; p < P0; ++p)   // where the two disagree, the detection's
                    if ((dd.low[(size_t) p] > 0.f) != (o->low[(size_t) p] > 0.f)) o->low[(size_t) p] = dd.low[(size_t) p];
                recond_store.push_back(dd.low);
                recond_slots[o->bucket].push_back(o->slot);
                recond_masks[o->bucket].push_back(&recond_store.back());
            }
        }
        // the recently occluded masklets overlapping another: hidden
        if (no > 1 && cfg.occlusion > 0.f) {
            std::vector<int64_t> last(no);
            for (size_t j = 0; j < no; ++j) last[j] = to_remove.count(objs[j].id) ? 100000 : objs[j].last_occluded;
            std::vector<char> sup(no, 0);
            for (size_t a = 0; a < no; ++a)
                for (size_t c = a + 1; c < no; ++c) {
                    if (objs[a].group != objs[c].group || iou(ob[a], ob[c], oa[a], oa[c]) < cfg.occlusion) continue;
                    if (last[a] > last[c] && last[c] > -1) sup[a] = 1;
                    if (last[c] > last[a] && last[a] > -1) sup[c] = 1;
                }
            for (size_t j = 0; j < no; ++j) {
                if (oa[j] == 0 || sup[j]) objs[j].last_occluded = f;
                if (sup[j]) std::fill(objs[j].low.begin(), objs[j].low.end(), -10.f);
            }
        } else {
            for (size_t j = 0; j < no; ++j)
                if (oa[j] == 0) objs[j].last_occluded = f;
        }
        double mem_s = 0;   // the memory's share of this section: its encoder, the clean masks' pointers
        auto m0 = std::chrono::steady_clock::now();
        for (const auto& [bi, sl] : recond_slots) condition(*now.at(bi), sl, recond_masks.at(bi));
        mem_s += since(m0);
        // the frame's memory: each pixel to the object of its words scoring it highest, an object shrunk below 30% by
        // that left out (its masks as they are otherwise)
        std::map<int64_t, std::vector<float>> mem_masks;
        {
            std::vector<int64_t> before(no, 0), after(no, 0);
            std::vector<size_t> gi(no);
            std::vector<int64_t> gs;
            for (size_t j = 0; j < no; ++j) {
                const auto it = std::find(gs.begin(), gs.end(), objs[j].group);
                gi[j] = (size_t) (it - gs.begin());
                if (it == gs.end()) gs.push_back(objs[j].group);
            }
            std::vector<size_t> best(gs.size());
            for (int64_t p = 0; p < P0; ++p) {
                std::fill(best.begin(), best.end(), no);
                for (size_t j = 0; j < no; ++j) {
                    const float v = objs[j].low[(size_t) p];
                    if (v > 0.f) ++before[j];
                    size_t& b = best[gi[j]];
                    if (b == no || v > objs[b].low[(size_t) p]) b = j;
                }
                for (size_t b : best)
                    if (b < no && objs[b].low[(size_t) p] > 0.f) ++after[b];
            }
            for (size_t j = 0; j < no; ++j) {
                std::vector<float> m = objs[j].low;
                if (no > 1 && (float) after[j] / (float) std::max<int64_t>(before[j], 1) < 0.3f)
                    for (float& v : m) v = std::min(v, -10.f);
                mem_masks[objs[j].id] = std::move(m);
            }
        }
        m0 = std::chrono::steady_clock::now();
        for (auto& [bi, fm] : now) {
            std::map<int64_t, const std::vector<float>*> mm;
            for (const auto& [id, m] : mem_masks) mm[id] = &m;
            encode(*buckets[(size_t) bi], *fm, mm);
        }
        sam.sync();
        mem_s += since(m0);
        // the outputs as tracked (before this frame's removals), then the new objects
        Pending out;
        out.frame = f;
        for (const Obj& o : objs) {
            out.masks.emplace_back(o.id, o.low);
            out.track[o.id] = sigmoid(o.track_logit);
        }
        // new objects: the detections (in their order), then the clicks', up to max_objects in all
        struct Fresh {
            const std::vector<float>* low = nullptr;
            float prob = 0;
            int64_t group = 0;
            bool user = false;
            Binary bin;
        };
        std::vector<Fresh> fresh;
        for (size_t i = 0; i < nd; ++i)
            if (is_new[i]) fresh.push_back(Fresh{&dets[i].low, dets[i].prob, dets[i].group, false, dets[i].bin});
        for (const Clicked& c : clicked)
            if (c.obj == nullptr) fresh.push_back(Fresh{&c.low, 1.f, -1 - (int64_t) c.click, true, binary(c.low)});
        const int64_t room = cfg.max_objects - (int64_t) objs.size();
        if ((int64_t) fresh.size() > room) fresh.resize((size_t) std::max<int64_t>(room, 0));
        if (!fresh.empty()) {
            // the first bucket in use with room for them all, else a new one
            int64_t target = -1;
            for (size_t bi = 0; bi < buckets.size() && target < 0; ++bi)
                if (buckets[bi]->any() &&
                    std::count(buckets[bi]->slot.begin(), buckets[bi]->slot.end(), kPad) >= (int64_t) fresh.size())
                    target = (int64_t) bi;
            const bool created = target < 0;
            if (created) {
                buckets.push_back(std::make_unique<Bucket>());
                target = (int64_t) buckets.size() - 1;
                now[target] = fresh_mem();
                now[target]->cond = true;
                now[target]->ptr.grow((size_t) (M * d) * 4);
                gpu::memset(now[target]->ptr.p, 0, (size_t) (M * d) * 4);
            }
            Bucket& b = *buckets[(size_t) target];
            FrameMem& fm = *now.at(target);
            std::vector<int64_t> slots;
            std::vector<const std::vector<float>*> lows;
            for (const Fresh& n : fresh) {
                Obj o;
                o.id = next_id++;
                o.group = n.group;
                o.bucket = target;
                o.slot = std::find(b.slot.begin(), b.slot.end(), kPad) - b.slot.begin();
                b.slot[(size_t) o.slot] = o.id;
                o.first = f;
                o.score = n.prob;
                o.user = o.confirmed = n.user;
                o.low = *n.low;
                first_score[o.id] = o.score;
                group_of[o.id] = o.group;
                out.masks.emplace_back(o.id, *n.low);
                out.track[o.id] = n.prob;
                slots.push_back(o.slot);
                lows.push_back(n.low);
                objs.push_back(std::move(o));
            }
            if (!tracked) {   // the new objects' memory needs the frame's propagation features: the tracker's time
                const auto n0 = std::chrono::steady_clock::now();
                features();
                sam.sync();
                t.track += since(n0);
            }
            m0 = std::chrono::steady_clock::now();
            condition(fm, slots, lows);
            // the bucket's memory again: the objects there before give way where the new ones are
            std::map<int64_t, const std::vector<float>*> mm;
            std::vector<std::vector<float>> kept;
            kept.reserve(objs.size());
            for (Obj& o : objs) {
                if (o.bucket != target) continue;
                if (o.first == f) {
                    mm[o.id] = &o.low;
                    continue;
                }
                std::vector<float> m = mem_masks.count(o.id) ? mem_masks.at(o.id) : o.low;
                for (const Fresh& n : fresh)
                    for (int64_t p = 0; p < P0; ++p)
                        if (n.bin[(size_t) p]) m[(size_t) p] = kNoObj;
                kept.push_back(std::move(m));
                mm[o.id] = &kept.back();
            }
            encode(b, fm, mm);
            sam.sync();
            mem_s += since(m0);
        }
        // the dropped objects: their slots not used again, a bucket with none left gone
        for (int64_t id : to_remove) {
            Obj* o = find(id);
            if (o == nullptr) continue;
            if (!o->confirmed) removed_unconfirmed.insert(id);
            buckets[(size_t) o->bucket]->slot[(size_t) o->slot] = kGone;
        }
        objs.erase(std::remove_if(objs.begin(), objs.end(), [&](const Obj& o) { return to_remove.count(o.id) > 0; }),
                   objs.end());
        // the frame's memory kept with its bucket; the frames past the pointers' and memories' reach let go
        for (auto& [bi, fm] : now) buckets[(size_t) bi]->mem[f] = std::move(fm);
        for (auto& b : buckets) {
            if (!b->any()) {
                for (auto& [t, m] : b->mem) spare.push_back(std::move(m));
                b->mem.clear();
                continue;
            }
            for (auto it = b->mem.begin(); it != b->mem.end();) {
                if (!it->second->cond && it->first < f - std::max<int64_t>(cfg.max_pointers, sam.mem_frames())) {
                    spare.push_back(std::move(it->second));
                    it = b->mem.erase(it);
                } else {
                    ++it;
                }
            }
        }
        // confirmation: the frames in a row a masklet was matched in (its first frame counts as one); confirm_after of
        // them confirm it for good
        {
            std::set<int64_t> hit;   // the objects a detection matched (out.masks' first no are this frame's tracked)
            for (size_t i = 0; i < nd; ++i)
                for (size_t j = 0; j < no && j < ob.size(); ++j)
                    if (im[i][j] >= cfg.assoc) hit.insert(out.masks[j].first);
            for (Obj& o : objs) {
                o.consecutive = o.first == f || hit.count(o.id) ? o.consecutive + 1 : 0;
                o.confirmed = o.confirmed || o.consecutive >= cfg.confirm_after;
            }
            auto& un = unconfirmed[f];
            for (const Obj& o : objs)
                if (!o.confirmed) un.insert(o.id);
        }
        for (const auto& [id, s] : first_score) out.score[id] = s;
        held.push_back(std::move(out));
        t.memory += mem_s;
        t.rules += since(t0) - mem_s;
    }

    // a held frame shown: its masks at the clip's size, the hidden ones left out, each pixel to one object
    Frame show(Pending& p) {
        Frame fr;
        fr.index = p.frame;
        const int64_t last = frame - 1;
        const auto un = unconfirmed.find(std::min(p.frame + cfg.confirm_after - 1, last));
        std::vector<std::pair<int64_t, const std::vector<float>*>> keep;
        for (const auto& [id, m] : p.masks) {
            // the masklets not yet confirmed are hidden; the removed and the kept-alive-out ones stay shown
            if (removed_unconfirmed.count(id)) continue;
            if (frame >= cfg.confirm_after && un != unconfirmed.end() && un->second.count(id)) continue;
            keep.emplace_back(id, &m);
        }
        if (keep.empty()) return fr;
        const int64_t n = (int64_t) keep.size();
        low_dev.grow((size_t) (n * P0) * 4);
        for (int64_t i = 0; i < n; ++i)
            gpu::copy(low_dev.as<float>() + i * P0, keep[(size_t) i].second->data(), (size_t) P0 * 4);
        masks1008.grow((size_t) (n * W * H));
        aux_mask_resize(low_dev.as<float>(), n, 4 * G, 4 * G, H, W, true, 0.f, masks1008.as<uint8_t>(), stream());
        sam.sync();
        std::vector<std::vector<uint8_t>> m((size_t) n, std::vector<uint8_t>((size_t) (W * H)));
        for (int64_t i = 0; i < n; ++i) gpu::copy(m[(size_t) i].data(), masks1008.as<uint8_t>() + i * W * H, (size_t) (W * H));
        // overlaps among one words' objects: the pixel to the one with the best tracker score there
        std::vector<size_t> gi((size_t) n);
        std::vector<float> ts((size_t) n);
        std::vector<int64_t> gs;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = group_of.at(keep[(size_t) i].first);
            const auto it = std::find(gs.begin(), gs.end(), g);
            gi[(size_t) i] = (size_t) (it - gs.begin());
            if (it == gs.end()) gs.push_back(g);
            ts[(size_t) i] = p.track[keep[(size_t) i].first];
        }
        std::vector<int64_t> best(gs.size());
        for (int64_t px = 0; px < W * H; ++px) {
            std::fill(best.begin(), best.end(), -1);
            for (int64_t i = 0; i < n; ++i) {
                int64_t& b = best[gi[(size_t) i]];
                if (m[(size_t) i][(size_t) px] && (b < 0 || ts[(size_t) i] > ts[(size_t) b])) b = i;
            }
            for (int64_t i = 0; i < n; ++i)
                if (m[(size_t) i][(size_t) px] && best[gi[(size_t) i]] != i) m[(size_t) i][(size_t) px] = 0;
        }
        for (int64_t i = 0; i < n; ++i) {
            Object o;
            o.id = keep[(size_t) i].first;
            const int64_t g = group_of.at(o.id);
            (g >= 0 ? o.words : o.click) = g >= 0 ? g : -1 - g;
            o.score = p.score.count(o.id) ? p.score[o.id] : 0.f;
            o.x0 = W;
            o.y0 = H;
            bool any = false;
            for (int64_t y = 0; y < H; ++y)
                for (int64_t x = 0; x < W; ++x)
                    if (m[(size_t) i][(size_t) (y * W + x)]) {
                        any = true;
                        o.x0 = std::min(o.x0, x);
                        o.y0 = std::min(o.y0, y);
                        o.x1 = std::max(o.x1, x + 1);
                        o.y1 = std::max(o.y1, y + 1);
                    }
            if (!any) continue;
            for (uint8_t& v : m[(size_t) i]) v = v ? 255 : 0;
            o.mask = std::move(m[(size_t) i]);
            fr.objects.push_back(std::move(o));
        }
        return fr;
    }
};

Sam3Video::Sam3Video(Sam3& sam, int64_t width, int64_t height, int64_t frames, std::vector<Sam3VideoWords> words,
                     std::vector<Sam3VideoClick> clicks, Config c)
    : m_(std::make_unique<Impl>(sam, width, height, frames, std::move(words), std::move(clicks), c)) {
    if (m_->words.size() > 16) throw std::invalid_argument("SAM 3 video: at most 16 words");
    sam.track_load();
    Impl& m = *m_;
    m.d = sam.d_;
    m.G = sam.grid_;
    m.P = m.G * m.G;
    m.P0 = 16 * m.P;
    m.S = sam.image_;
}

Sam3Video::~Sam3Video() = default;

std::vector<Sam3Video::Frame> Sam3Video::push(const QiRef& frame) {
    Impl& m = *m_;
    m.step(frame, t_);
    std::vector<Frame> out;
    const auto t0 = std::chrono::steady_clock::now();
    while ((int) m.held.size() >= std::max(m.cfg.confirm_after, 1)) {   // a frame's confirmation is known then
        out.push_back(m.show(m.held.front()));
        m.held.pop_front();
    }
    t_.output += since(t0);
    return out;
}

std::vector<Sam3Video::Frame> Sam3Video::finish() {
    Impl& m = *m_;
    std::vector<Frame> out;
    const auto t0 = std::chrono::steady_clock::now();
    while (!m.held.empty()) {
        out.push_back(m.show(m.held.front()));
        m.held.pop_front();
    }
    t_.output += since(t0);
    return out;
}

}  // namespace strata::image
