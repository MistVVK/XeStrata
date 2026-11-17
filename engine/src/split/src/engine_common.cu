// The parts of the Engine (engine.hpp) that do not depend on the model, for every model's build: see engine_parts.hpp.
#include "engine_parts.hpp"
#include "env.hpp"

#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fnx {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

namespace {
// a state's pieces between a card and pageable host memory (laid end to end there), through the card's two
// page-locked halves in blocks of STAGE_BYTES: the copy engine fills (or empties) one half while the host copies the
// other. Pageable, because page-locked memory is mapped into every card: 2 MiB of each card's VRAM per GiB (64 MiB for
// 32 GiB, tools/pin_probe), 256 MiB for the 128 GiB cache, about what cards 0 and 1 have left
constexpr size_t STAGE_BYTES = (size_t) 32 << 20;
}  // namespace

void install_graph(CardBuf::GraphSlot& s, cudaGraph_t g, const std::vector<int>& key) {
    if (s.exec) {
        cudaGraphExecUpdateResultInfo info{};
        if (cudaGraphExecUpdate(s.exec, g, &info) == cudaSuccess) {
            s.key = key;
            ck(cudaGraphDestroy(g), "graph destroy");
            return;
        }
        cudaGetLastError();   // a different shape: a new executable in its place
        ck(cudaGraphExecDestroy(s.exec), "graph exec destroy");
        s.exec = nullptr;
    }
    ck(cudaGraphInstantiate(&s.exec, g, 0), "instantiate");
    s.key = key;
    ck(cudaGraphDestroy(g), "graph destroy");
}

long host_available_mib() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb < 0 ? -1 : kb / 1024;
}

struct CardPool {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable go, done;
    const std::function<void(int)>* job = nullptr;
    unsigned gen = 0;
    int left = 0;
    bool quit = false;
    std::exception_ptr err[NCARD];
    int dev0 = 0;
    explicit CardPool(const std::vector<int>& devs) : dev0(devs[0]) {
        for (int c = 1; c < NCARD; ++c)
            th.emplace_back([this, c, d = devs[(size_t) c]] {
                const cudaError_t set = cudaSetDevice(d);
                unsigned seen = 0;
                for (;;) {
                    const std::function<void(int)>* f;
                    {
                        std::unique_lock<std::mutex> l(mu);
                        go.wait(l, [&] { return quit || gen != seen; });
                        if (quit) return;
                        seen = gen;
                        f = job;
                    }
                    try {
                        if (set != cudaSuccess) throw std::runtime_error(std::string("card pool: cudaSetDevice: ") + cudaGetErrorString(set));
                        (*f)(c);
                    } catch (...) { err[c] = std::current_exception(); }
                    std::lock_guard<std::mutex> l(mu);
                    if (--left == 0) done.notify_one();
                }
            });
    }
    ~CardPool() {
        {
            std::lock_guard<std::mutex> l(mu);
            quit = true;
        }
        go.notify_all();
        for (auto& t : th) t.join();
    }
    void run(const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> l(mu);
            job = &f;
            left = NCARD - 1;
            ++gen;
            for (auto& e : err) e = nullptr;
        }
        go.notify_all();
        std::exception_ptr e0;
        try {
            ck(cudaSetDevice(dev0), "cudaSetDevice");
            f(0);
        } catch (...) { e0 = std::current_exception(); }
        {
            std::unique_lock<std::mutex> l(mu);
            done.wait(l, [&] { return left == 0; });
        }
        if (e0) std::rethrow_exception(e0);
        for (auto& e : err)
            if (e) std::rethrow_exception(e);
    }
};

void Engine::on_cards(const std::function<void(int)>& f) {
    if (!pool_) {
        std::vector<int> devs;
        for (CardBuf* b : cb_) devs.push_back(b->dev);
        pool_ = std::make_shared<CardPool>(devs);
    }
    pool_->run(f);
}

unsigned Engine::ring_tag() {
    tags_room(1);
    return ++tag_max_;
}

unsigned Engine::take_tags(int n) {
    tags_room((unsigned) n + 1u);   // the parity step below can take one tag more than n
    unsigned base = tag_max_ + 1u;
    if ((base & 1u) != ((tag_ + 1u) & 1u)) ++base;
    tag_ = base + (unsigned) n - 1u;
    tag_max_ = tag_;
    return base;
}

void Engine::release_pages() {
    LaneHost& L = lanes_[(size_t) cur_];
    for (int p : L.pages) free_pages_.push_back(p);
    L.pages.clear();
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
}

void Engine::use_lane(int i) {
    if (i < 0 || i >= (int) lanes_.size()) throw std::runtime_error("use_lane: no lane " + std::to_string(i));
    if (i == cur_) return;
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("use_lane: a window is not committed");
    LaneHost& was = lanes_[(size_t) cur_];
    was.pos = pos_;
    was.tag = tag_;
    was.history = std::move(history_);
    cur_ = i;
    LaneHost& now = lanes_[(size_t) i];
    pos_ = now.pos;
    tag_ = now.tag;
    history_ = std::move(now.history);
    now.history.clear();
    for (CardBuf* b : cb_) card_lane(*b, b->lanes[(size_t) i]);
}

void Engine::set_images(const std::vector<Image>& images) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("set_images: a window is not committed");
    int end = 0;
    for (const Image& im : images) {
        if (im.nx < 1 || im.ny < 1 || im.n != im.nx * im.ny) throw std::runtime_error("set_images: an image's cells are not its grid");
        if (im.cell < end || im.cell + im.n > max_ctx_) throw std::runtime_error("set_images: images out of order or past the context");
        end = im.cell + im.n;
    }
    LaneHost& L = lanes_[(size_t) cur_];
    const bool same = L.images.size() == images.size() &&
                      std::equal(images.begin(), images.end(), L.images.begin(), [](const Image& a, const Image& b) {
                          return a.cell == b.cell && a.n == b.n && a.nx == b.nx && a.ny == b.ny;
                      });
    L.images = images;
    if (same) return;   // the rows change; the positions do not
    if (images.empty()) {
        for (CardBuf* b : cb_) b->mrope = b->lanes[(size_t) cur_].mrope = nullptr;
        return;
    }
    std::vector<int> tab((size_t) max_ctx_ * 3);
    int p = 0;
    size_t k = 0;
    for (int c = 0; c < max_ctx_;) {
        if (k < images.size() && c == images[k].cell) {
            const Image& im = images[k++];
            for (int j = 0; j < im.n; ++j) {
                tab[(size_t) (c + j) * 3] = p;
                tab[(size_t) (c + j) * 3 + 1] = p + j / im.nx;
                tab[(size_t) (c + j) * 3 + 2] = p + j % im.nx;
            }
            c += im.n;
            p += std::max(im.nx, im.ny);
        } else {
            tab[(size_t) c * 3] = tab[(size_t) c * 3 + 1] = tab[(size_t) c * 3 + 2] = p++;
            ++c;
        }
    }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        LaneDev& D = b->lanes[(size_t) cur_];
        if (!D.mrope_buf) D.mrope_buf = dalloc<int>(tab.size());
        ck(cudaMemcpy(D.mrope_buf, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice), "rotary table");
        b->mrope = D.mrope = D.mrope_buf;
    }
}

bool Engine::precap() const {
#if FNX_HIP
    static const bool on = env_int("FNX_PRECAP", 1) != 0;   // AMD: on (measured +18% with FNX_CAPTURE_AFTER=2 at 8 users, same bits; RESULTS-HIP.md section 8)
#else
    static const bool on = env_int("FNX_PRECAP", 0) != 0;   // NVIDIA: off in 0.2.9 (frozen)
#endif
    return on;
}

namespace {
void swap_slot(CardBuf& b, int T, CardBuf::GraphEntry& e) {
    CardBuf::GraphSlot& s = b.batch_graphs[(size_t) T];
    std::swap(s.exec, e.exec);
    std::swap(s.key, e.key);
    std::swap(b.lane_rows[(size_t) T], e.lane_rows);
    std::swap(b.lane_gdn[(size_t) T], e.lane_gdn);
    std::swap(b.seg_ctl[(size_t) T], e.seg_ctl);
    std::swap(b.seg_row0[(size_t) T], e.seg_row0);
    std::swap(b.batch_slot_used[(size_t) T], e.used);
}
}  // namespace

bool Engine::batch_graph_ready(int T, const std::vector<int>& key) {
    CardBuf& b0 = *cb_[0];
    if (b0.batch_graphs[(size_t) T].exec && b0.batch_graphs[(size_t) T].key == key) {
        if (precap()) for (CardBuf* b : cb_) b->batch_slot_used[(size_t) T] = ++graph_tick_;
        return true;
    }
    if (!precap()) return false;
    std::vector<CardBuf::GraphEntry>& cache = b0.batch_cache[(size_t) T];
    for (size_t i = 0; i < cache.size(); ++i)
        if (cache[i].key == key) {
            for (CardBuf* b : cb_) {
                swap_slot(*b, T, b->batch_cache[(size_t) T][i]);
                b->batch_slot_used[(size_t) T] = ++graph_tick_;
            }
            return true;
        }
    return false;
}

bool Engine::lazy_capture(const std::vector<int>& key) {
#if FNX_HIP
    static const long long after = env_int("FNX_CAPTURE_AFTER", 2);
#else
    static const long long after = env_int("FNX_CAPTURE_AFTER", 0);
#endif
    if (after < 2 || warming_) return false;
    if (seen_.size() > 8192) seen_.clear();
    return ++seen_[key] < after;
}

void Engine::batch_graph_stash(int T) {
    if (!precap() || !cb_[0]->batch_graphs[(size_t) T].exec) return;
    static const size_t max_graphs = (size_t) std::max<long long>(1, env_int("FNX_PRECAP_GRAPHS", 64));
    size_t total = 0;
    for (const auto& v : cb_[0]->batch_cache) total += v.size();
    int vt = -1;
    size_t vi = 0;
    if (total >= max_graphs) {   // the least recently used entry goes; its tables are the next capture's
        unsigned long best = ~0UL;
        for (int t = 0; t <= MAXT; ++t)
            for (size_t i = 0; i < cb_[0]->batch_cache[(size_t) t].size(); ++i)
                if (cb_[0]->batch_cache[(size_t) t][i].used < best) best = cb_[0]->batch_cache[(size_t) t][i].used, vt = t, vi = i;
    }
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        CardBuf::GraphEntry nu;
        CardBuf::GraphSlot& s = b->batch_graphs[(size_t) T];
        nu.exec = s.exec;
        nu.key = std::move(s.key);
        nu.lane_rows = b->lane_rows[(size_t) T];
        nu.lane_gdn = b->lane_gdn[(size_t) T];
        nu.seg_ctl = b->seg_ctl[(size_t) T];
        nu.seg_row0 = b->seg_row0[(size_t) T];
        nu.used = b->batch_slot_used[(size_t) T];
        s.exec = nullptr;
        s.key.clear();
        b->lane_rows[(size_t) T] = nullptr, b->lane_gdn[(size_t) T] = nullptr, b->seg_ctl[(size_t) T] = nullptr, b->seg_row0[(size_t) T] = nullptr;
        if (vt >= 0) {
            CardBuf::GraphEntry& v = b->batch_cache[(size_t) vt][vi];
            if (v.exec) cudaGraphExecDestroy(v.exec);
            b->lane_rows[(size_t) T] = v.lane_rows, b->lane_gdn[(size_t) T] = v.lane_gdn;
            b->seg_ctl[(size_t) T] = v.seg_ctl, b->seg_row0[(size_t) T] = v.seg_row0;
            b->batch_cache[(size_t) vt].erase(b->batch_cache[(size_t) vt].begin() + (long) vi);
        }
        b->batch_cache[(size_t) T].push_back(std::move(nu));
    }
}

void Engine::warm_graphs() {
    const int was = cur_;
    warming_ = true;
    std::vector<int> lanes;
    for (int i = 0; i < (int) lanes_.size(); ++i) lanes.push_back(i);
    const int tok = 248045;   // any token: every lane is reset after
    use_lane(0);
    reset();
    // the prompt path's buffers (and cuBLAS's) with a read of two chunks, every lane's rotary table
    std::fprintf(stderr, "fnx: warm: prompt path (1100 tokens) ...\n");
    prefill(std::vector<int>(1100, tok), tok);
    std::fprintf(stderr, "fnx: warm: prompt path done\n");
    reset();
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        for (LaneDev& D : b->lanes)
            if (!D.mrope_buf) D.mrope_buf = dalloc<int>((size_t) max_ctx_ * 3);
    }
    std::fprintf(stderr, "fnx: warm: verify windows ...\n");
    for (int T = 1; T <= MAXT; ++T) {
        verify(std::vector<int>((size_t) T, tok));
        commit(1);
        if (T == 1 || T == MAXT) std::fprintf(stderr, "fnx: warm: verify %d rows done\n", T);
    }
    if (mtp_)
        for (int T = 1; T <= MAXT; ++T) draft(T, true, 0, std::vector<int>((size_t) T, tok), 0);
    for (int n = 1; n <= std::min((int) lanes.size(), MAXT); ++n)
        step(std::vector<int>(lanes.begin(), lanes.begin() + n), std::vector<int>((size_t) n, tok));
    // batched windows with drafts (they share step()'s graph slots, one per row count): 2 .. MAXT rows over as many lanes
    // as there are, up to the rows, dealt out evenly
    if (batch_drafts() != BatchDrafts::off && precap()) {
        // FNX_PRECAP: every uniform layout (k rows a lane) of the first n lanes, n = 1 .. lanes, kept as graphs: a window of the lanes
        // in order with the same drafts each (or none) then never captures. The mean capture is what the planner is charged
        const auto t0 = std::chrono::steady_clock::now();
        unsigned long n_cap = 0;
        double ms_cap = 0.0;
        std::fprintf(stderr, "fnx: warm: precapture of the uniform batched windows ...\n");
        for (int n = 1; n <= (int) lanes.size(); ++n)
            for (int k = 1; n * k <= MAXT; ++k) {
                if (n == 1 && k == 1) continue;   // the one-row window is step()'s, warmed above
                const std::vector<int> ids(lanes.begin(), lanes.begin() + n), counts((size_t) n, k);
                verify_rows(ids, counts, std::vector<int>((size_t) (n * k), tok));
                commit_rows(std::vector<int>((size_t) n, 1));
                if (last_capture_ms > 0.0) ++n_cap, ms_cap += last_capture_ms;
            }
        if (n_cap) warm_capture_ms = ms_cap / (double) n_cap;
        long total = 0;
        for (const auto& v : cb_[0]->batch_cache) total += (long) v.size();
        std::fprintf(stderr, "fnx: warm: precapture done: %lu captures, %.1f ms each, %ld graphs held, %.1f s\n", n_cap, warm_capture_ms, total,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else if (batch_drafts() != BatchDrafts::off)
        for (int T = 2; T <= MAXT; ++T) {
            const int n = std::min((int) lanes.size(), T);
            std::vector<int> counts((size_t) n, T / n);
            for (int i = 0; i < T % n; ++i) ++counts[(size_t) i];
            verify_rows(std::vector<int>(lanes.begin(), lanes.begin() + n), counts, std::vector<int>((size_t) T, tok));
            commit_rows(std::vector<int>((size_t) n, 1));
        }
    for (int i : lanes) {
        use_lane(i);
        reset();
    }
    use_lane(was);
    warming_ = false;
}

std::vector<long> Engine::free_vram_mib() const {
    std::vector<long> v;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        size_t f = 0, t = 0;
        ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
        v.push_back((long) (f >> 20));
    }
    return v;
}

namespace {
std::mutex owned_mu;
std::vector<std::function<void()>>& owned_list() {
    static std::vector<std::function<void()>> list;
    return list;
}
}  // namespace

void own(std::function<void()> release) {
    std::lock_guard<std::mutex> l(owned_mu);
    owned_list().push_back(std::move(release));
}

std::vector<std::function<void()>> take_owned() {
    std::lock_guard<std::mutex> l(owned_mu);
    return std::move(owned_list());
}

// every card idle, the window graphs destroyed, then what the engine made released, newest first (errors ignored: a
// destructor must not throw, and the process may be on its way out)
Engine::~Engine() {
    for (CardBuf* b : cb_) {
        cudaSetDevice(b->dev);
        cudaDeviceSynchronize();
        for (auto* slots : {&b->win_graphs, &b->mtp_graphs, &b->batch_graphs})
            for (CardBuf::GraphSlot& s : *slots)
                if (s.exec) cudaGraphExecDestroy(s.exec);
        for (auto& v : b->batch_cache)
            for (CardBuf::GraphEntry& e : v)
                if (e.exec) cudaGraphExecDestroy(e.exec);
    }
    std::vector<std::function<void()>> o = take_owned();
    for (auto it = o.rbegin(); it != o.rend(); ++it) (*it)();
    cudaGetLastError();
    for (CardBuf* b : cb_) delete b;
}

void Engine::embed_token(int tok, int cell, float* out) const {
    if (tok >= IMAGE_TOK) {
        for (const Image& im : lanes_[(size_t) cur_].images)
            if (cell >= im.cell && cell < im.cell + im.n) {
                if (!im.rows) break;
                std::memcpy(out, im.rows + (size_t) (cell - im.cell) * N, N * sizeof(float));
                return;
            }
        throw std::runtime_error("embed: image cell " + std::to_string(cell) + " without its embedding row");
    }
    if (tok < 0 || tok >= m_.vocab) throw std::runtime_error("embed: token " + std::to_string(tok) + " outside the vocabulary");
    const Tensor& te = *m_.tok_embd;
    const uint8_t* row = te.data + (size_t) tok * te.row_bytes();
    for (int b = 0; b < N / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, row + b * 34, 2);
        const float d = h2f(dh);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * (float) (int8_t) row[b * 34 + 2 + j];
    }
}

void Engine::dump_dev(const std::string& name, int layer, const float* p, int T, int n) {
    if (!dump) return;
    std::vector<float> h((size_t) T * n);
    ck(cudaSetDevice(cb_[0]->dev), "set");
    ck(cudaStreamSynchronize(cb_[0]->st), ("dump sync before " + name + " of layer " + std::to_string(layer)).c_str());
    ck(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost), "dump copy");
    dump(name, layer, T, n, h);
}

std::vector<int> Engine::window(const std::vector<int>& tokens) {
    std::vector<int> out = verify(tokens);
    commit((int) tokens.size());
    return out;
}

size_t Engine::SeqState::total() const {
    size_t t = 0;
    for (size_t b : bytes) t += b;
    return t;
}

Engine::SeqState::~SeqState() {
    for (void* p : host) std::free(p);
}

void stage_copy(CardBuf& b, const std::vector<std::pair<char*, size_t>>& pieces, char* host, size_t total, bool to_host) {
    if (!b.stage[0]) {
        for (int k = 0; k < 2; ++k) {
            const cudaError_t e = cudaHostAlloc((void**) &b.stage[k], STAGE_BYTES, cudaHostAllocPortable);
            if (e != cudaSuccess) {
                cudaGetLastError();
                b.stage[k] = nullptr;
                throw std::runtime_error("state staging: page-locked host memory for 32 MiB: " + std::string(cudaGetErrorString(e)));
            }
            ck(cudaEventCreateWithFlags(&b.stage_ev[k], cudaEventDisableTiming), "event");
            own_pinned(b.stage[k]);
            own_event(b.stage_ev[k]);
        }
    }
    const size_t nblk = (total + STAGE_BYTES - 1) / STAGE_BYTES;
    size_t pi = 0, poff = 0;   // the next piece byte to move
    // the DMAs of block k between the device pieces and half k % 2
    auto dma = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        char* half = b.stage[k % 2];
        for (size_t done = 0; done < blk;) {
            const size_t take = std::min(blk - done, pieces[pi].second - poff);
            char* dev = pieces[pi].first + poff;
            ck(cudaMemcpyAsync(to_host ? half + done : dev, to_host ? dev : half + done, take,
                               to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, b.st), "state copy");
            done += take;
            poff += take;
            if (poff == pieces[pi].second) {
                ++pi;
                poff = 0;
            }
        }
        ck(cudaEventRecord(b.stage_ev[k % 2], b.st), "event");
    };
    auto host_part = [&](size_t k) {
        const size_t blk = std::min(STAGE_BYTES, total - k * STAGE_BYTES);
        if (to_host) std::memcpy(host + k * STAGE_BYTES, b.stage[k % 2], blk);
        else std::memcpy(b.stage[k % 2], host + k * STAGE_BYTES, blk);
    };
    for (size_t k = 0; k < nblk; ++k) {
        if (to_host) {
            dma(k);
            if (k > 0) {
                ck(cudaEventSynchronize(b.stage_ev[(k - 1) % 2]), "state copy");
                host_part(k - 1);
            }
        } else {
            if (k > 1) ck(cudaEventSynchronize(b.stage_ev[k % 2]), "state copy");   // the half's last DMA is done
            host_part(k);
            dma(k);
        }
    }
    if (to_host && nblk > 0) {
        ck(cudaEventSynchronize(b.stage_ev[(nblk - 1) % 2]), "state copy");
        host_part(nblk - 1);
    }
    ck(cudaStreamSynchronize(b.st), "state copy");
}

// f(card) on a thread per card, the first failure rethrown after all of them end
void each_card_thread(int n, const std::function<void(int)>& f) {
    std::vector<std::exception_ptr> err((size_t) n);
    std::vector<std::thread> th;
    for (int c = 0; c < n; ++c)
        th.emplace_back([&, c] {
            try {
                f(c);
            } catch (...) {
                err[(size_t) c] = std::current_exception();
            }
        });
    for (std::thread& t : th) t.join();
    for (const std::exception_ptr& e : err)
        if (e) std::rethrow_exception(e);
}

std::shared_ptr<Engine::SeqState> Engine::save_state(bool kv) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("save_state: a window is not committed");
    auto s = std::make_shared<SeqState>();
    s->tokens = history_;
    for (const Image& im : lanes_[(size_t) cur_].images)
        if (im.cell + im.n <= pos_) s->images.push_back(Image{im.cell, im.n, im.nx, im.ny, nullptr});
    if (kv) ensure_pages(pos_);
    s->kv = kv;
    s->host.assign(NCARD, nullptr);
    s->bytes.assign(NCARD, 0);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos_, kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        s->host[c] = std::aligned_alloc(4096, (n + 4095) / 4096 * 4096);
        if (!s->host[c])
            throw std::runtime_error("save_state: host memory for " + std::to_string(n >> 20) + " MiB (" + std::to_string(host_available_mib()) +
                                     " MiB available)");
        s->bytes[c] = n;
    }
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s->host[c], s->bytes[c], true);
    });
    return s;
}

void Engine::load_state(const SeqState& s) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("load_state: a window is not committed");
    if ((int) s.tokens.size() + 1 > max_ctx_) throw std::runtime_error("load_state: the state does not fit the context");
    if (!s.kv && (history_.size() < s.tokens.size() || !std::equal(s.tokens.begin(), s.tokens.end(), history_.begin())))
        throw std::runtime_error("load_state: a checkpoint of another sequence (the engine does not hold its K/V)");
    const int pos = (int) s.tokens.size();
    if (s.kv) ensure_pages(pos);
    const unsigned half = (tag_ + 1u) & 1u;
    std::vector<std::vector<std::pair<char*, size_t>>> pieces(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        pieces[(size_t) c] = state_pieces(*cb_[c], m_.cards[c], mtp_ != nullptr, pos, s.kv, half, lanes_[(size_t) cur_].pages);
        size_t n = 0;
        for (const auto& p : pieces[(size_t) c]) n += p.second;
        if (n != s.bytes[c]) throw std::runtime_error("load_state: a state saved with another layout");
    }
    each_card_thread(NCARD, [&](int c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        stage_copy(*cb_[c], pieces[(size_t) c], (char*) s.host[c], s.bytes[c], false);
    });
    history_ = s.tokens;
    pos_ = pos;
    if (df2_) df2_forget(cur_);
    set_images(s.images);
    // pages past the state's length go back to the pool
    LaneHost& L = lanes_[(size_t) cur_];
    const size_t keep = (size_t) (pos / KVP_POS + 1);
    if (L.pages.size() > keep) {
        free_pages_.insert(free_pages_.end(), L.pages.begin() + (long) keep, L.pages.end());
        L.pages.resize(keep);
        std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    }
}

void Engine::reset() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("reset: a window is not committed");
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) continue;
            ck(cudaMemsetAsync(b.S[l], 0, (size_t) w.nv() * HS * HS * 4, b.st), "reset state");
            ck(cudaMemsetAsync(b.chist[l], 0, (size_t) 2 * 3 * w.conv_ch() * 4, b.st), "reset conv history");
        }
        ck(reset_hists(b), "reset histories");
        ck(cudaStreamSynchronize(b.st), "reset sync");
    }
    release_pages();
    history_.clear();
    pos_ = 0;
    if (df2_) df2_forget(cur_);
    set_images({});
}

void Engine::commit(int n) {
    const int T = (int) pending_.size();
    if (T == 0) throw std::runtime_error("commit: no window to commit");
    if (n < 1 || n > T) throw std::runtime_error("commit: keep 1..T rows");
    if (n < T) {
        // the window read history parity tag & 1 and wrote the other: that one becomes the kept count's history. The
        // kept row's states and histories in one copy_many per card (Flash-Next's as cudaMemcpyAsync calls: 73 a card,
        // ~0.5 ms of host time a commit)
        const unsigned wrote = (tag_ & 1u) ^ 1u;
        constexpr int CJ = 2 * NL + HIST_JOBS;
        if (!h_.commit_jobs) h_.commit_jobs = halloc<CopyJob>((size_t) NCARD * CJ);
        for (CardBuf* b : cb_)
            if (!b->commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->commit_jobs = dalloc<CopyJob>(CJ);
                ck(cudaEventCreateWithFlags(&b->commit_ev, cudaEventDisableTiming), "commit event");
                own_event(b->commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& w = m_.cards[c];
            const size_t sbytes = (size_t) w.nv() * HS * HS * 4, hfloats = (size_t) 3 * w.conv_ch();
            // the page-locked table again only once the last commit's upload from it is done
            ck(cudaEventSynchronize(b.commit_ev), "commit table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.commit_jobs) + (size_t) c * CJ;
            int k = 0;
            for (int l = 0; l < NL; ++l) {
                if (is_attn(l)) continue;
                jobs[k++] = {b.S[l], b.snap[l] + (size_t) (n - 1) * w.nv() * HS * HS, sbytes};
                jobs[k++] = {b.chist[l] + wrote * hfloats, b.hsnap[l] + (size_t) (n - 1) * hfloats, hfloats * 4};
            }
            k += hist_jobs(jobs + k, b, nullptr, (size_t) (n - 1));
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit jobs");
            ck(cudaEventRecord(b.commit_ev, b.st), "commit event");
            copy_many(b.commit_jobs, k, b.st);
        });
    }
    history_.insert(history_.end(), pending_.begin(), pending_.begin() + n);
    pos_ += n;
    pending_.clear();
}

std::vector<float> Engine::last_logits(int T) const {
    const int vocab = m_.cards[NCARD - 1].v1;
    std::vector<float> out((size_t) T * vocab);
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(cb_[c]->dev), "set");
        const int nv = w.v1 - w.v0;
        for (int t = 0; t < T; ++t)
            ck(cudaMemcpy(out.data() + (size_t) t * vocab + w.v0, cb_[c]->head_logits + (size_t) t * nv, (size_t) nv * 4,
                          cudaMemcpyDeviceToHost), "logits");
    }
    return out;
}

std::vector<std::vector<std::pair<float, int>>> Engine::top_tokens(int T, const Penalties& pen) {
    if (T < 1 || T > MAXT) throw std::runtime_error("top_tokens: 1..MAXT rows");
    if (!pen.rows.empty() && (int) pen.rows.size() != T) throw std::runtime_error("top_tokens: one penalty list per row");
    if (!tk_.pen_tok) {
        tk_.pen_tok = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_cnt = halloc<int>((size_t) MAXT * PEN_CAP);
        tk_.pen_n = halloc<int>(MAXT);
        tk_.v = halloc<float>((size_t) NCARD * MAXT * TOPK_N);
        tk_.i = halloc<int>((size_t) NCARD * MAXT * TOPK_N);
        tk_.err = halloc<unsigned>(NCARD);
        tk_.par = halloc<float>((size_t) MAXT * 3);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            b->tk_pen_tok = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_cnt = dalloc<int>((size_t) MAXT * PEN_CAP);
            b->tk_pen_n = dalloc<int>(MAXT);
            b->tk_v = dalloc<float>((size_t) MAXT * TOPK_N);
            b->tk_i = dalloc<int>((size_t) MAXT * TOPK_N);
            b->tk_err = dalloc<unsigned>(1);
            b->tk_par = dalloc<float>((size_t) MAXT * 3);
        }
    }
    // the rows' penalty lists at the stride of the longest (the copies carry only what is used)
    static const Penalties::Row none;
    int stride = 1;
    for (int t = 0; t < T; ++t) stride = std::max(stride, (int) (pen.rows.empty() ? none : pen.rows[(size_t) t]).counts.size());
    if (stride > PEN_CAP) throw std::runtime_error("top_tokens: more penalized tokens than " + std::to_string(PEN_CAP));
    for (int t = 0; t < T; ++t) {
        const Penalties::Row& row = pen.rows.empty() ? none : pen.rows[(size_t) t];
        const auto& r = row.counts;
        tk_.par[3 * t] = row.repeat;
        tk_.par[3 * t + 1] = row.freq;
        tk_.par[3 * t + 2] = row.present;
        tk_.pen_n[t] = (int) r.size();
        for (size_t k = 0; k < r.size(); ++k) {
            tk_.pen_tok[(size_t) t * stride + k] = r[k].first;
            tk_.pen_cnt[(size_t) t * stride + k] = r[k].second;
        }
    }
    on_cards([&](int c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaMemcpyAsync(b.tk_pen_tok, tk_.pen_tok, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_cnt, tk_.pen_cnt, (size_t) T * stride * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_pen_n, tk_.pen_n, (size_t) T * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemcpyAsync(b.tk_par, tk_.par, (size_t) T * 3 * 4, cudaMemcpyHostToDevice, b.st), "penalties");
        ck(cudaMemsetAsync(b.tk_err, 0, 4, b.st), "top err");
        fnx::top_tokens(b.head_logits, w.v1 - w.v0, T, w.v0, b.tk_pen_tok, b.tk_pen_cnt, b.tk_pen_n, stride, b.tk_par,
                        b.tk_v, b.tk_i, b.tk_err, b.st);
        ck(cudaMemcpyAsync(tk_.v + (size_t) c * MAXT * TOPK_N, b.tk_v, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top v");
        ck(cudaMemcpyAsync(tk_.i + (size_t) c * MAXT * TOPK_N, b.tk_i, (size_t) T * TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "top i");
        ck(cudaMemcpyAsync(tk_.err + c, b.tk_err, 4, cudaMemcpyDeviceToHost, b.st), "top err");
        ck(cudaStreamSynchronize(b.st), "top sync");
    });
    for (int c = 0; c < NCARD; ++c)
        if (tk_.err[c] & 16u) throw std::runtime_error("top_tokens: the head's logits hold a NaN or inf");
    std::vector<std::vector<std::pair<float, int>>> out(T);
    for (int t = 0; t < T; ++t) {
        auto& r = out[t];
        r.reserve((size_t) NCARD * TOPK_N);
        for (int c = 0; c < NCARD; ++c)
            for (int k = 0; k < TOPK_N; ++k) {
                const size_t at = ((size_t) c * MAXT + t) * TOPK_N + k;
                r.emplace_back(tk_.v[at], tk_.i[at]);
            }
        std::sort(r.begin(), r.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        r.resize(TOPK_N);
    }
    return out;
}

std::vector<int> Engine::verify(const std::vector<int>& tokens) {
    const int T = (int) tokens.size();
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify: the last window is not committed");
    if (T < 1 || T > MAXT) throw std::runtime_error("window: 1..MAXT tokens");
    if (pos_ + T > max_ctx_) throw std::runtime_error("window: context full");
    arena_churn();
    ensure_pages(pos_ + T);
    const auto t0 = std::chrono::steady_clock::now();
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {   // shared with the draft layer, which can run first (over a prompt read by prefill)
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    const int h0 = (int) history_.size();
    embed_into(0, T, pos_, [&](int q) -> int { return q < h0 ? history_[(size_t) q] : tokens[(size_t) (q - h0)]; }, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    h_.ctl[0] = pos_;
    h_.ctl[1] = (int) take_tags(1);
    float* bv = h_.bv;
    int* bi = h_.bi;
    auto upload = [&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
    };
    auto download = [&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    };
    if (dump) {
        each(upload);
        dump_dev("R0", -1, cb_[0]->R, T, ROW_W);
        enqueue(T, true);
        each(download);
    } else {
        const std::vector<int> key = graph_key({cur_});
        if (cb_[0]->win_graphs[T].key != key || !cb_[0]->win_graphs[T].exec) {   // capturing queues no work
            const auto tc = std::chrono::steady_clock::now();
            each([&](int, CardBuf& b, const CardW&) {
                ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
            });
            enqueue(T, false);
            each([&](int, CardBuf& b, const CardW&) {
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                install_graph(b.win_graphs[T], g, key);
            });
            ++cap_stat[2].n;
            cap_stat[2].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
        }
        const auto l0 = std::chrono::steady_clock::now();
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            upload(c, b, m_.cards[c]);
            ck(cudaGraphLaunch(b.win_graphs[T].exec, b.st), "graph launch");
            download(c, b, m_.cards[c]);
        });
        last_launch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count();
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "window sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
        last_card_ms[c] = ms;
    });
    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("window: ") + e);
    std::vector<int> out(T);
    for (int t = 0; t < T; ++t) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (bv[c * T + t] > bv[best * T + t]) best = c;
        out[t] = bi[best * T + t];
    }
    pending_ = tokens;
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

std::vector<int> Engine::step(const std::vector<int>& lanes, const std::vector<int>& tokens) {
    const int T = (int) lanes.size();
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("step: 1..MAXT lanes, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("step: the current lane's window is not committed");
    for (int r = 0; r < T; ++r) {
        if (lanes[r] < 0 || lanes[r] >= (int) lanes_.size()) throw std::runtime_error("step: no lane " + std::to_string(lanes[r]));
        for (int q = 0; q < r; ++q)
            if (lanes[q] == lanes[r]) throw std::runtime_error("step: a lane twice in one window");
    }
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    std::vector<int> lane_ctl((size_t) 2 * T);
    try {
        for (int r = 0; r < T; ++r) {
            LaneHost& L = lanes_[(size_t) lanes[r]];
            if (L.pos + 1 > max_ctx_) throw std::runtime_error("step: lane " + std::to_string(lanes[r]) + "'s context is full");
            ensure_pages_of(lanes[r], L.pos + 1);
            const std::vector<int>& h = L.history;
            const int tok = tokens[r], h0 = (int) h.size();
            embed_into(r, 1, L.pos, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tok; }, "step");
            // the lane's recurrent history half flips, as a window of its own would
            L.tag = L.tag + 1u;
            lane_ctl[2 * r] = L.pos;
            lane_ctl[2 * r + 1] = (int) L.tag;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int r = 0; r < T; ++r)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[r]].ctl, &lane_ctl[(size_t) 2 * r], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
    });
    const std::vector<int> bkey = graph_key(lanes);
    if (!batch_graph_ready(T, bkey)) {
        batch_graph_stash(T);
        const auto tc = std::chrono::steady_clock::now();
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
            std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                rows[(size_t) r] = LD.ctl;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            for (int l = 0; l < NL; ++l)
                for (int r = 0; r < T; ++r) {
                    gdn[(size_t) l * MAXT + r] = b.lanes[(size_t) lanes[r]].S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + r] = b.lanes[(size_t) lanes[r]].chist[(size_t) l];
                }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        enqueue_batch(lanes);
        each([&](int, CardBuf& b, const CardW&) {
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b.st, &g), "end capture");
            install_graph(b.batch_graphs[T], g, bkey);
        });
        ++cap_stat[3].n;
        cap_stat[3].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    each([&](int, CardBuf& b, const CardW&) { ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch"); });
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "step sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("step: ") + e);
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
        LaneHost& L = lanes_[(size_t) lanes[r]];
        L.history.push_back(tokens[r]);
        ++L.pos;
    }
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    return out;
}

double rope_scale() {
    const char* e = std::getenv("FNX_ROPE_SCALE");
    if (!e) return 0.0;
    // 1 is off, so a profile's setting can be overridden without unsetting it
    const double S = env_num("FNX_ROPE_SCALE", 0.0);
    if (S == 1.0) return 0.0;
    if (!(S > 1.0)) throw std::runtime_error(std::string("FNX_ROPE_SCALE: want 1 (off) or a scale above 1, got ") + e);
    return S;
}

bool ring_f32() {
    // fp16 pairs by default on every count: fp32 costs ~30% of one request's decode on two cards. fp32's KL is lower
    // (two cards' code prompt 5.73e-4 against fp16's 7.48e-4, a bar of 6.06e-4, so fp16 is over it there; three cards
    // 4.38e-4 against 5.75e-4, both within)
    const long long v = env_int("FNX_RING_F32", 0);
    if (v != 0 && v != 1) throw std::runtime_error("FNX_RING_F32: 0 or 1, got " + std::to_string(v));
    return v == 1;
}

BatchDrafts batch_drafts() {
    static const BatchDrafts mode = [] {
        const char* e = std::getenv("FNX_BATCH_DRAFTS");
        if (!e || !std::strcmp(e, "1")) return BatchDrafts::mtp;
        if (!std::strcmp(e, "0")) return BatchDrafts::off;
        if (!std::strcmp(e, "lookup")) return BatchDrafts::lookup;
        throw std::runtime_error(std::string("FNX_BATCH_DRAFTS: 0, lookup or 1, not '") + e + "'");
    }();
    return mode;
}

RowSegs row_segs(const std::vector<int>& counts) {
    RowSegs sg;
    const int nl = (int) counts.size();
    for (int i = 0; i < nl; ++i) {
        sg.r0.push_back(sg.T);
        for (int j = 0; j < counts[(size_t) i]; ++j) sg.row_lane.push_back(i);
        sg.T += counts[(size_t) i];
        sg.order.push_back(i);
    }
    std::stable_sort(sg.order.begin(), sg.order.end(), [&](int a, int b) { return counts[(size_t) a] < counts[(size_t) b]; });
    for (int s = 0; s < nl; ++s) {
        const int k = counts[(size_t) sg.order[(size_t) s]];
        if (sg.groups.empty() || sg.groups.back().count != k) sg.groups.push_back({k, s, 0});
        ++sg.groups.back().n;
    }
    return sg;
}

std::vector<int> Engine::verify_rows(const std::vector<int>& lanes, const std::vector<int>& counts, const std::vector<int>& tokens) {
    const int nl = (int) lanes.size();
    if (nl < 1 || (int) counts.size() != nl) throw std::runtime_error("verify_rows: one lane or more, a row count each");
    int T = 0;
    for (int k : counts) {
        if (k < 1) throw std::runtime_error("verify_rows: a lane with no rows");
        T += k;
    }
    if (T > MAXT || (int) tokens.size() != T) throw std::runtime_error("verify_rows: at most MAXT rows in all, a token each");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("verify_rows: the last window is not committed");
    for (int i = 0; i < nl; ++i) {
        if (lanes[i] < 0 || lanes[i] >= (int) lanes_.size()) throw std::runtime_error("verify_rows: no lane " + std::to_string(lanes[i]));
        for (int q = 0; q < i; ++q)
            if (lanes[q] == lanes[i]) throw std::runtime_error("verify_rows: a lane twice in one window");
    }
    arena_churn();
    const auto t0 = std::chrono::steady_clock::now();
    // the current lane's scalars to its slot, so every lane reads from lanes_
    LaneHost& cur = lanes_[(size_t) cur_];
    cur.pos = pos_;
    cur.tag = tag_;
    cur.history = std::move(history_);
    if (!h_.R0) {
        h_.R0 = halloc<float>((size_t) MAXT * ROW_W);
        h_.ctl = halloc<int>(2);
    }
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
    for (CardBuf* b : cb_)
        if (!b->row_pos) {
            ck(cudaSetDevice(b->dev), "set");
            b->row_pos = dalloc<int>(MAXT);
        }
    const RowSegs sg = row_segs(counts);
    std::vector<int> lane_ctl((size_t) 2 * nl), row_pos((size_t) T);
    std::vector<unsigned> tags((size_t) nl);
    try {
        for (int i = 0; i < nl; ++i) {
            const LaneHost& L = lanes_[(size_t) lanes[i]];
            const int k = counts[(size_t) i], r = sg.r0[(size_t) i];
            if (L.pos + k > max_ctx_) throw std::runtime_error("verify_rows: lane " + std::to_string(lanes[i]) + "'s context is full");
            ensure_pages_of(lanes[i], L.pos + k);
            const std::vector<int>& h = L.history;
            const int h0 = (int) h.size();
            embed_into(r, k, L.pos, [&](int q) -> int { return q < h0 ? h[(size_t) q] : tokens[(size_t) (r + q - h0)]; }, "verify_rows");
            // the lane's recurrent history half flips, as a window of its own would
            tags[(size_t) i] = L.tag + 1u;
            lane_ctl[(size_t) 2 * i] = L.pos;
            lane_ctl[(size_t) 2 * i + 1] = (int) tags[(size_t) i];
            for (int j = 0; j < k; ++j) row_pos[(size_t) (r + j)] = L.pos + j;
        }
    } catch (...) {
        history_ = std::move(cur.history);
        throw;
    }
    for (int i = 0; i < nl; ++i) lanes_[(size_t) lanes[i]].tag = tags[(size_t) i];
    h_.ctl[0] = 0;
    h_.ctl[1] = (int) ring_tag();
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    each([&](int, CardBuf& b, const CardW&) {
        ck(cudaEventRecord(b.ev0, b.st), "event");
        rows_upload(b, T);
        ck(cudaMemcpyAsync(b.ctl, h_.ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "ctl");
        for (int i = 0; i < nl; ++i)
            ck(cudaMemcpyAsync(b.lanes[(size_t) lanes[i]].ctl, &lane_ctl[(size_t) 2 * i], 2 * sizeof(int), cudaMemcpyHostToDevice, b.st),
               "lane ctl");
        ck(cudaMemcpyAsync(b.row_pos, row_pos.data(), (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "row positions");
    });
    // the graph slot step() uses for T rows; the layout (each lane's count) is part of the key, and -1 tells it from step()'s
    std::vector<int> key = graph_key(lanes);
    key.push_back(-1);
    key.insert(key.end(), counts.begin(), counts.end());
    last_capture_ms = 0.0;
    last_kind_ = 0;
    bool eager = false;
    if (!batch_graph_ready(T, key)) {
        eager = lazy_capture(key);
        last_kind_ = eager ? 2 : 1;
        if (!eager) batch_graph_stash(T);
        const auto tc = std::chrono::steady_clock::now();
        auto swap_eager = [&] {   // an eager window reads its tables from the eager set, not the slot's (a held graph points at those)
            for (CardBuf* b : cb_) {
                std::swap(b->lane_rows[(size_t) T], b->eager_tab.lane_rows);
                std::swap(b->lane_gdn[(size_t) T], b->eager_tab.lane_gdn);
                std::swap(b->seg_ctl[(size_t) T], b->eager_tab.seg_ctl);
                std::swap(b->seg_row0[(size_t) T], b->eager_tab.seg_row0);
            }
        };
        if (eager) swap_eager();
        each([&](int, CardBuf& b, const CardW&) {
            if (!b.lane_rows[(size_t) T]) b.lane_rows[(size_t) T] = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
            std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
            for (int r = 0; r < T; ++r) {
                const LaneDev& LD = b.lanes[(size_t) lanes[(size_t) sg.row_lane[(size_t) r]]];
                rows[(size_t) r] = b.row_pos + r;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
            }
            ck(cudaMemcpy(b.lane_rows[(size_t) T], rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "lane rows");
            if (!b.lane_gdn[(size_t) T]) b.lane_gdn[(size_t) T] = dalloc<float*>((size_t) 2 * NL * MAXT);
            if (!b.seg_ctl[(size_t) T]) b.seg_ctl[(size_t) T] = dalloc<const int*>(MAXT);
            if (!b.seg_row0[(size_t) T]) b.seg_row0[(size_t) T] = dalloc<int>(MAXT);
            std::vector<float*> gdn((size_t) 2 * NL * MAXT, nullptr);
            std::vector<const int*> sctl((size_t) MAXT, nullptr);
            std::vector<int> srow0((size_t) MAXT, 0);
            for (int s = 0; s < nl; ++s) {
                const int i = sg.order[(size_t) s];
                const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                for (int l = 0; l < NL; ++l) {
                    gdn[(size_t) l * MAXT + s] = LD.S[(size_t) l];
                    gdn[(size_t) (NL + l) * MAXT + s] = LD.chist[(size_t) l];
                }
                sctl[(size_t) s] = LD.ctl;
                srow0[(size_t) s] = sg.r0[(size_t) i];
            }
            ck(cudaMemcpy(b.lane_gdn[(size_t) T], gdn.data(), gdn.size() * sizeof(float*), cudaMemcpyHostToDevice), "lane recurrent rows");
            ck(cudaMemcpy(b.seg_ctl[(size_t) T], sctl.data(), sctl.size() * sizeof(const int*), cudaMemcpyHostToDevice), "segment ctl");
            ck(cudaMemcpy(b.seg_row0[(size_t) T], srow0.data(), srow0.size() * sizeof(int), cudaMemcpyHostToDevice), "segment rows");
            if (!eager) ck(cudaStreamBeginCapture(b.st, cudaStreamCaptureModeRelaxed), "begin capture");
        });
        enqueue_rows(lanes, counts);
        if (eager) {
            swap_eager();
            last_capture_ms = 0.0;
        } else {
            each([&](int, CardBuf& b, const CardW&) {
                cudaGraph_t g;
                ck(cudaStreamEndCapture(b.st, &g), "end capture");
                install_graph(b.batch_graphs[T], g, key);
            });
            last_capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            ++cap_stat[0].n;
            cap_stat[0].ms += last_capture_ms;
        }
        static const bool trace = env_int("FNX_PRECAP_TRACE", 0) != 0;
        if (trace) {
            std::string t;
            for (size_t i = 0; i < lanes.size(); ++i) t += " " + std::to_string(lanes[i]) + ":" + std::to_string(counts[i]);
            std::fprintf(stderr, "fnx: %s verify_rows T=%d lanes:rows%s (%.1f ms)\n", eager ? "eager" : "capture", T, t.c_str(),
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count());
        }
    }
    {   // FNX_BATCH_STATS=1: windows, graph captures, and the time between one window's start and the next's by the first one's row count; one block per 200 windows
        static const bool stats = env_flag("FNX_BATCH_STATS");
        if (stats) {
            static unsigned long nw = 0, ncap = 0, rows_sum = 0, cnt_t[MAXT + 1] = {};
            static double cap_ms = 0, ms_t[MAXT + 1] = {};
            static int prev_t = 0;
            static std::chrono::steady_clock::time_point t_first{}, t_prev{};
            const auto now = std::chrono::steady_clock::now();
            if (nw == 0) t_first = now;
            if (prev_t > 0) {
                const double gap = std::chrono::duration<double, std::milli>(now - t_prev).count();
                if (gap < 400.0) cnt_t[prev_t]++, ms_t[prev_t] += gap;   // a gap past 400 ms is an idle server, not a window
            }
            prev_t = T;
            t_prev = now;
            ++nw;
            rows_sum += (unsigned long) T;
            if (last_capture_ms > 0.0) ++ncap, cap_ms += last_capture_ms;
            if (nw % 200 == 0) {
                std::fprintf(stderr, "fnx: batch stats: %lu windows, %lu captured (%.1f ms each), %.1f rows a window, %.2f ms a window since the first\n", nw,
                             ncap, ncap ? cap_ms / (double) ncap : 0.0, (double) rows_sum / (double) nw,
                             std::chrono::duration<double, std::milli>(now - t_first).count() / (double) nw);
                std::fprintf(stderr, "fnx: captures since start (n, host ms): verify_rows %lu %.0f, draft %lu %.0f, window %lu %.0f, batch step %lu %.0f\n",
                             cap_stat[0].n, cap_stat[0].ms, cap_stat[1].n, cap_stat[1].ms, cap_stat[2].n, cap_stat[2].ms, cap_stat[3].n,
                             cap_stat[3].ms);
                std::fprintf(stderr, "fnx: verify_rows windows since start (n, mean host ms, mean gpu ms): held graph %lu %.1f %.1f, captured %lu %.1f %.1f, eager %lu %.1f %.1f\n",
                             win_stat[0].n, win_stat[0].ms / (double) std::max(1UL, win_stat[0].n), win_stat[0].gpu_ms / (double) std::max(1UL, win_stat[0].n),
                             win_stat[1].n, win_stat[1].ms / (double) std::max(1UL, win_stat[1].n), win_stat[1].gpu_ms / (double) std::max(1UL, win_stat[1].n),
                             win_stat[2].n, win_stat[2].ms / (double) std::max(1UL, win_stat[2].n), win_stat[2].gpu_ms / (double) std::max(1UL, win_stat[2].n));
                {
                    const char* nm[3] = {"held graph", "captured", "eager"};
                    for (int kd = 0; kd < 3; ++kd) {
                        std::string row = std::string("fnx: verify_rows host ms by rows, ") + nm[kd] + " (rows:count:ms):";
                        for (int k = 1; k <= MAXT; ++k)
                            if (win_stat_t[kd][k].n) {
                                char buf[48];
                                std::snprintf(buf, sizeof buf, " %d:%lu:%.1f", k, win_stat_t[kd][k].n, win_stat_t[kd][k].ms / (double) win_stat_t[kd][k].n);
                                row += buf;
                            }
                        std::fprintf(stderr, "%s\n", row.c_str());
                    }
                }
                std::string tab = "fnx: batch stats by rows (rows:count:ms):";
                for (int k = 1; k <= MAXT; ++k)
                    if (cnt_t[k]) {
                        char buf[48];
                        std::snprintf(buf, sizeof buf, " %d:%lu:%.1f", k, cnt_t[k], ms_t[k] / (double) cnt_t[k]);
                        tab += buf;
                    }
                std::fprintf(stderr, "%s\n", tab.c_str());
                nw = ncap = rows_sum = 0;
                cap_ms = 0;
                for (int k = 0; k <= MAXT; ++k) cnt_t[k] = 0, ms_t[k] = 0;
            }
        }
    }
    if (!eager) each([&](int, CardBuf& b, const CardW&) { ck(cudaGraphLaunch(b.batch_graphs[T].exec, b.st), "graph launch"); });
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaMemcpyAsync(&h_.bv[c * T], b.best_v, T * 4, cudaMemcpyDeviceToHost, b.st), "best v");
        ck(cudaMemcpyAsync(&h_.bi[c * T], b.best_i, T * 4, cudaMemcpyDeviceToHost, b.st), "best i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        ck(cudaEventRecord(b.ev1, b.st), "event");
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    each([&](int c, CardBuf& b, const CardW&) {
        ck(cudaStreamSynchronize(b.st), "verify_rows sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, b.ev0, b.ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    });
    pos_ = cur.pos;
    tag_ = cur.tag;
    history_ = std::move(cur.history);
    cur.history.clear();
    // the window ran: its rows wait for commit_rows even when an error below ends the caller (the lanes' states moved)
    rows_pending_ = RowsWindow{lanes, counts, tokens};
    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("verify_rows: ") + e);
    std::vector<int> out(T);
    for (int r = 0; r < T; ++r) {
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (h_.bv[c * T + r] > h_.bv[best * T + r]) best = c;
        out[r] = h_.bi[best * T + r];
    }
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_gpu_ms = gpu_ms;
    ++win_stat[last_kind_].n;
    win_stat[last_kind_].ms += last_ms;
    win_stat[last_kind_].gpu_ms += gpu_ms;
    ++win_stat_t[last_kind_][T].n;
    win_stat_t[last_kind_][T].ms += last_ms;
    return out;
}

std::vector<std::vector<int>> Engine::held_layouts(const std::vector<int>& lanes) const {
    std::vector<int> pre = graph_key(lanes);
    pre.push_back(-1);
    std::vector<std::vector<int>> out;
    for (int T = (int) lanes.size(); T <= MAXT; ++T) {
        const CardBuf::GraphSlot& s = cb_[0]->batch_graphs[(size_t) T];
        if (s.exec && s.key.size() == pre.size() + lanes.size() && std::equal(pre.begin(), pre.end(), s.key.begin()))
            out.emplace_back(s.key.begin() + (long) pre.size(), s.key.end());
        for (const CardBuf::GraphEntry& e : cb_[0]->batch_cache[(size_t) T])
            if (e.exec && e.key.size() == pre.size() + lanes.size() && std::equal(pre.begin(), pre.end(), e.key.begin()))
                out.emplace_back(e.key.begin() + (long) pre.size(), e.key.end());
    }
    return out;
}

void Engine::commit_rows(const std::vector<int>& keep) {
    const RowsWindow& w = rows_pending_;
    const int nl = (int) w.lanes.size();
    if (nl == 0) throw std::runtime_error("commit_rows: no window to commit");
    if ((int) keep.size() != nl) throw std::runtime_error("commit_rows: a kept count per lane");
    for (int i = 0; i < nl; ++i)
        if (keep[i] < 1 || keep[i] > w.counts[(size_t) i]) throw std::runtime_error("commit_rows: keep 1 .. its rows of each lane");
    const RowSegs sg = row_segs(w.counts);
    bool cut = false;
    for (int i = 0; i < nl; ++i) cut = cut || keep[i] < w.counts[(size_t) i];
    if (cut) {
        // a lane cut back takes its kept row's states and histories from the window's snapshots at its rows, as commit()
        // does (the history half its window wrote: the parity after its tag's)
        constexpr int CJ = 2 * NL + HIST_JOBS;
        if (!h_.rows_commit_jobs) h_.rows_commit_jobs = halloc<CopyJob>((size_t) NCARD * MAXT * CJ);
        for (CardBuf* b : cb_)
            if (!b->rows_commit_jobs) {
                ck(cudaSetDevice(b->dev), "set");
                b->rows_commit_jobs = dalloc<CopyJob>((size_t) MAXT * CJ);
                ck(cudaEventCreateWithFlags(&b->rows_commit_ev, cudaEventDisableTiming), "commit_rows event");
                own_event(b->rows_commit_ev);
            }
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            const CardW& cw = m_.cards[c];
            const size_t sfloats = (size_t) cw.nv() * HS * HS, hfloats = (size_t) 3 * cw.conv_ch();
            // the page-locked table again only once the last commit_rows' upload from it is done
            ck(cudaEventSynchronize(b.rows_commit_ev), "commit_rows table");
            CopyJob* jobs = static_cast<CopyJob*>(h_.rows_commit_jobs) + (size_t) c * MAXT * CJ;
            int k = 0;
            for (int i = 0; i < nl; ++i) {
                if (keep[i] == w.counts[(size_t) i]) continue;
                const int lane = w.lanes[(size_t) i];
                const LaneDev& LD = b.lanes[(size_t) lane];
                const unsigned wrote = ((lane == cur_ ? tag_ : lanes_[(size_t) lane].tag) & 1u) ^ 1u;
                const size_t at = (size_t) (sg.r0[(size_t) i] + keep[i] - 1);
                for (int l = 0; l < NL; ++l) {
                    if (is_attn(l)) continue;
                    jobs[k++] = {LD.S[(size_t) l], b.snap[l] + at * sfloats, sfloats * 4};
                    jobs[k++] = {LD.chist[(size_t) l] + wrote * hfloats, b.hsnap[l] + at * hfloats, hfloats * 4};
                }
                k += hist_jobs(jobs + k, b, &LD, at);
            }
            for (int i = 0; i < k; ++i)
                if (((uintptr_t) jobs[i].dst | (uintptr_t) jobs[i].src | jobs[i].bytes) & 15u)
                    throw std::runtime_error("commit_rows: a copy not 16-byte aligned");
            ck(cudaMemcpyAsync(b.rows_commit_jobs, jobs, (size_t) k * sizeof(CopyJob), cudaMemcpyHostToDevice, b.st), "commit_rows jobs");
            ck(cudaEventRecord(b.rows_commit_ev, b.st), "commit_rows event");
            copy_many(b.rows_commit_jobs, k, b.st);
        });
    }
    for (int i = 0; i < nl; ++i) {
        const int lane = w.lanes[(size_t) i];
        const auto from = w.tokens.begin() + sg.r0[(size_t) i];
        std::vector<int>& h = lane == cur_ ? history_ : lanes_[(size_t) lane].history;
        h.insert(h.end(), from, from + keep[i]);
        (lane == cur_ ? pos_ : lanes_[(size_t) lane].pos) += keep[i];
    }
    rows_pending_ = RowsWindow{};
}

int Engine::draft(int T, bool from_main, int src_row, const std::vector<int>& tokens, int pos0, const Penalties::Row* pen,
                  bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (!h_.R0) throw std::runtime_error("draft: no window has run (it drafts from a window's rows)");
    if (src_row < 0 || src_row + T > MAXT) throw std::runtime_error("draft: source rows out of range");
    std::vector<const float*> src;
    for (CardBuf* b : cb_) src.push_back(draft_src(*b, from_main) + (size_t) src_row * ROW_W);
    return draft_rows(src, T, tokens, pos0, pen, cands);
}

int Engine::draft_rows(const std::vector<const float*>& src, int T, const std::vector<int>& tokens, int pos0,
                       const Penalties::Row* pen, bool cands) {
    if (!mtp_) throw std::runtime_error("draft: no draft layer attached");
    if (T < 1 || T > MAXT || (int) tokens.size() != T) throw std::runtime_error("draft: 1..MAXT rows, one token each");
    if (pos0 < 0 || pos0 + T > max_ctx_) throw std::runtime_error("draft: cells out of the context");
    ensure_pages(pos0 + T);
    if (!h_.mtp_emb) throw std::runtime_error("draft: mtp_attach first");
    const auto t0 = std::chrono::steady_clock::now();
    // the tokens' embeddings (Q8_0 rows of the main table; the token after row t sits at cell pos0 + t + 1)
    for (int t = 0; t < T; ++t) embed_token(tokens[t], pos0 + t + 1, h_.mtp_emb + (size_t) t * N);
    h_.mtp_ctl[0] = pos0;
    h_.mtp_ctl[1] = (int) ring_tag();
    // the penalized tokens in each card's slice of the draft vocabulary: {k, indexes, counts}
    int pen_k[NCARD] = {};
    if (pen) {
        if ((int) pen->counts.size() > PEN_CAP) throw std::runtime_error("draft: more penalized tokens than " + std::to_string(PEN_CAP));
        h_.mtp_pen_par[0] = pen->repeat, h_.mtp_pen_par[1] = pen->freq, h_.mtp_pen_par[2] = pen->present;
        for (int c = 0; c < NCARD; ++c) {
            const MtpCard& mc = mtp_->cards[c];
            int* hp = h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP);
            int k = 0;
            for (const auto& [tok, cnt] : pen->counts) {
                const int di = tok >= 0 && tok < (int) draft_index_.size() ? draft_index_[(size_t) tok] : -1;
                if (di >= mc.dv0 && di < mc.dv1) hp[1 + k] = di - mc.dv0, hp[1 + PEN_CAP + k] = cnt, ++k;
            }
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);   // counts right after the indexes
            hp[0] = pen_k[c] = k;
        }
    } else {
        for (int c = 0; c < NCARD; ++c) h_.mtp_pen[(size_t) c * (1 + 2 * PEN_CAP)] = 0;
    }
    const std::vector<int> key = graph_key({cur_});
    if (cb_[0]->mtp_graphs[T].key != key || !cb_[0]->mtp_graphs[T].exec) {   // capturing queues no work
        const auto tc = std::chrono::steady_clock::now();
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            ck(cudaStreamBeginCapture(b->st, cudaStreamCaptureModeRelaxed), "begin capture");
        }
        mtp_enqueue(T);
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            cudaGraph_t g;
            ck(cudaStreamEndCapture(b->st, &g), "end capture");
            install_graph(b->mtp_graphs[T], g, key);
        }
        ++cap_stat[1].n;
        cap_stat[1].ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    }
    on_cards([&](int c) {
        CardBuf& b = *cb_[c];
        ck(cudaMemcpyAsync(b.mtp_src, src[c], (size_t) T * ROW_W * 4, cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        ck(cudaMemcpyAsync(b.mtp_emb, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
        ck(cudaMemcpyAsync(b.mtp_ctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
        ck(cudaMemcpyAsync(b.mtp_pen, h_.mtp_pen + (size_t) c * (1 + 2 * PEN_CAP), (size_t) (1 + 2 * pen_k[c]) * 4,
                           cudaMemcpyHostToDevice, b.st), "draft penalties");
        if (pen_k[c]) ck(cudaMemcpyAsync(b.mtp_pen_par, h_.mtp_pen_par, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
        ck(cudaEventRecord(b.ev0, b.st), "event");
        ck(cudaGraphLaunch(b.mtp_graphs[T].exec, b.st), "graph launch");
        const MtpCard& mc = mtp_->cards[c];
        const int nv = mc.dv1 - mc.dv0, last = MTP_HEAD_EVERY_ROW ? T - 1 : 0;   // the last row's head row
        if (cands && nv > 0) {   // the last row's logits are penalized already
            ck(cudaMemsetAsync(b.mtp_tk_err, 0, 4, b.st), "draft top err");
            fnx::top_tokens(b.mtp_logits + (size_t) last * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0, b.mtp_pen_par,
                       b.mtp_tk_v, b.mtp_tk_i, b.mtp_tk_err, b.st);
        }
        ck(cudaEventRecord(b.ev1, b.st), "event");
        ck(cudaMemcpyAsync(&h_.bv[c], b.mtp_bv + last, 4, cudaMemcpyDeviceToHost, b.st), "draft v");
        ck(cudaMemcpyAsync(&h_.bi[c], b.mtp_bi + last, 4, cudaMemcpyDeviceToHost, b.st), "draft i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        if (cands && nv > 0) {
            ck(cudaMemcpyAsync(h_.mtp_tk_v + (size_t) c * TOPK_N, b.mtp_tk_v, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
            ck(cudaMemcpyAsync(h_.mtp_tk_i + (size_t) c * TOPK_N, b.mtp_tk_i, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
            ck(cudaMemcpyAsync(h_.mtp_tk_err + c, b.mtp_tk_err, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
        }
    });
    unsigned err = 0;
    float gpu_ms = 0.f;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c]->ev0, cb_[c]->ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    }
    last_draft_gpu_ms = gpu_ms;
    if (const char* e = window_error(err, true)) throw std::runtime_error(std::string("draft: ") + e);
    int best = 0;
    for (int c = 1; c < NCARD; ++c)
        if (mtp_->cards[c].dv1 > mtp_->cards[c].dv0 && h_.bv[c] > h_.bv[best]) best = c;
    if (cands) {
        last_draft_cands.clear();
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            if (h_.mtp_tk_err[c] & 16u) throw std::runtime_error("draft: the draft layer's logits hold a NaN or inf");
            for (int k = 0; k < TOPK_N; ++k)
                last_draft_cands.emplace_back(h_.mtp_tk_v[(size_t) c * TOPK_N + k], mtp_->vocab[h_.mtp_tk_i[(size_t) c * TOPK_N + k]]);
        }
        std::sort(last_draft_cands.begin(), last_draft_cands.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        last_draft_cands.resize(TOPK_N);
    }
    last_draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return mtp_->vocab[h_.bi[best]];
}


std::vector<int> Engine::draft_lanes(const std::vector<DraftReq>& reqs) {
    if (!mtp_) throw std::runtime_error("draft_lanes: no draft layer attached");
    if (!h_.R0 || !h_.mtp_emb) throw std::runtime_error("draft_lanes: no window has run (it drafts from a window's rows)");
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("draft_lanes: a window is not committed");
    const int nl = (int) reqs.size();
    int T = 0;
    for (const DraftReq& q : reqs) {
        if (q.T < 1 || (int) q.tokens.size() != q.T) throw std::runtime_error("draft_lanes: a lane's rows and tokens differ");
        if (q.lane < 0 || q.lane >= (int) lanes_.size()) throw std::runtime_error("draft_lanes: no lane " + std::to_string(q.lane));
        if (q.src_row < 0 || q.src_row + q.T > MAXT) throw std::runtime_error("draft_lanes: source rows out of range");
        if (q.pos0 < 0 || q.pos0 + q.T > max_ctx_ || q.pos0 + q.T > lane_pages(q.lane) * KVP_POS)
            throw std::runtime_error("draft_lanes: cells out of the lane's pages");
        T += q.T;
    }
    if (nl < 1 || T > MAXT) throw std::runtime_error("draft_lanes: 1.." + std::to_string(MAXT) + " rows in all");
    const auto t0 = std::chrono::steady_clock::now();
    const int was = cur_;
    std::vector<int> off((size_t) nl);
    int o = 0;
    for (int i = 0; i < nl; ++i) {
        const DraftReq& q = reqs[(size_t) i];
        off[(size_t) i] = o;
        use_lane(q.lane);   // the embeddings read the lane's images
        for (int t = 0; t < q.T; ++t) {
            embed_token(q.tokens[(size_t) t], q.pos0 + t + 1, h_.mtp_emb + (size_t) (o + t) * N);
            h_.mtp_row_pos[o + t] = q.pos0 + t;
        }
        h_.mtp_lctl[2 * i] = q.pos0;
        h_.mtp_lctl[2 * i + 1] = 0;
        o += q.T;
    }
    use_lane(was);
    h_.mtp_ctl[0] = 0;
    h_.mtp_ctl[1] = (int) ring_tag();
    // the penalized tokens of the lanes that penalize their drafts, in each card's slice of the draft vocabulary: {k, indexes, counts}, as draft() lays them out
    std::vector<std::array<int, NCARD>> pen_k((size_t) nl);
    for (int i = 0; i < nl; ++i) {
        pen_k[(size_t) i].fill(0);
        const Penalties::Row* pen = reqs[(size_t) i].pen;
        if (!pen) continue;
        if ((int) pen->counts.size() > PEN_CAP) throw std::runtime_error("draft_lanes: more penalized tokens than " + std::to_string(PEN_CAP));
        h_.mtp_pen_par_l[3 * i] = pen->repeat, h_.mtp_pen_par_l[3 * i + 1] = pen->freq, h_.mtp_pen_par_l[3 * i + 2] = pen->present;
        for (int c = 0; c < NCARD; ++c) {
            const MtpCard& mc = mtp_->cards[c];
            int* hp = h_.mtp_pen_l + ((size_t) i * NCARD + c) * (1 + 2 * PEN_CAP);
            int k = 0;
            for (const auto& [tok, cnt] : pen->counts) {
                const int di = tok >= 0 && tok < (int) draft_index_.size() ? draft_index_[(size_t) tok] : -1;
                if (di >= mc.dv0 && di < mc.dv1) hp[1 + k] = di - mc.dv0, hp[1 + PEN_CAP + k] = cnt, ++k;
            }
            std::memmove(hp + 1 + k, hp + 1 + PEN_CAP, (size_t) k * 4);   // counts right after the indexes
            hp[0] = pen_k[(size_t) i][c] = k;
        }
    }
    last_lane_cands.assign((size_t) nl, {});
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        std::vector<const int*> rows((size_t) LANE_ROW_COLS * MAXT, nullptr);
        for (int i = 0; i < nl; ++i)
            for (int t = 0; t < reqs[(size_t) i].T; ++t) {
                const int r = off[(size_t) i] + t;
                const LaneDev& LD = b.lanes[(size_t) reqs[(size_t) i].lane];
                rows[(size_t) r] = b.mtp_row_pos + r;
                rows[(size_t) (MAXT + r)] = LD.pt;
                lane_row_extra(rows.data(), r, LD);
            }
        ck(cudaMemcpy(b.mtp_lane_rows, rows.data(), rows.size() * sizeof(const int*), cudaMemcpyHostToDevice), "draft lane rows");
        for (int i = 0; i < nl; ++i) {
            const DraftReq& q = reqs[(size_t) i];
            ck(cudaMemcpyAsync(b.mtp_src + (size_t) off[(size_t) i] * ROW_W, draft_src(b, q.from_main) + (size_t) q.src_row * ROW_W, (size_t) q.T * ROW_W * 4,
                               cudaMemcpyDeviceToDevice, b.st), "draft source rows");
        }
        ck(cudaMemcpyAsync(b.mtp_emb, h_.mtp_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "draft embeddings");
        ck(cudaMemcpyAsync(b.mtp_ctl, h_.mtp_ctl, 2 * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft ctl");
        ck(cudaMemcpyAsync(b.mtp_lctl, h_.mtp_lctl, (size_t) 2 * nl * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft lane ctl");
        ck(cudaMemcpyAsync(b.mtp_row_pos, h_.mtp_row_pos, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, b.st), "draft row positions");
        for (int i = 0; i < nl; ++i)
            if (reqs[(size_t) i].pen) {
                ck(cudaMemcpyAsync(b.mtp_pen_l + (size_t) i * (1 + 2 * PEN_CAP), h_.mtp_pen_l + ((size_t) i * NCARD + c) * (1 + 2 * PEN_CAP),
                                   (size_t) (1 + 2 * pen_k[(size_t) i][c]) * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
                ck(cudaMemcpyAsync(b.mtp_pen_par_l + 3 * i, h_.mtp_pen_par_l + 3 * i, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
            } else if (reqs[(size_t) i].cands) {
                h_.mtp_pen_par_l[3 * i] = 1.f, h_.mtp_pen_par_l[3 * i + 1] = 0.f, h_.mtp_pen_par_l[3 * i + 2] = 0.f;
                ck(cudaMemcpyAsync(b.mtp_pen_par_l + 3 * i, h_.mtp_pen_par_l + 3 * i, 3 * 4, cudaMemcpyHostToDevice, b.st), "draft penalties");
            }
        ck(cudaEventRecord(b.ev0, b.st), "event");
    }
    mtp_enqueue_lanes(reqs, T);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaEventRecord(b.ev1, b.st), "event");
        ck(cudaMemcpyAsync(h_.bv + (size_t) c * MAXT, b.mtp_bv, (size_t) T * 4, cudaMemcpyDeviceToHost, b.st), "draft v");
        ck(cudaMemcpyAsync(h_.bi + (size_t) c * MAXT, b.mtp_bi, (size_t) T * 4, cudaMemcpyDeviceToHost, b.st), "draft i");
        ck(cudaMemcpyAsync(&h_.err[c], err_word(b), 4, cudaMemcpyDeviceToHost, b.st), "window err");
        const MtpCard& mc = mtp_->cards[c];
        for (int i = 0; i < nl; ++i)
            if (reqs[(size_t) i].cands && mc.dv1 > mc.dv0) {
                const size_t at = (size_t) i * NCARD + c;
                ck(cudaMemcpyAsync(h_.mtp_tk_v_l + at * TOPK_N, b.mtp_tk_v_l + (size_t) i * TOPK_N, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top v");
                ck(cudaMemcpyAsync(h_.mtp_tk_i_l + at * TOPK_N, b.mtp_tk_i_l + (size_t) i * TOPK_N, TOPK_N * 4, cudaMemcpyDeviceToHost, b.st), "draft top i");
                ck(cudaMemcpyAsync(h_.mtp_tk_err_l + at, b.mtp_tk_err_l + i, 4, cudaMemcpyDeviceToHost, b.st), "draft top err");
            }
    }
    unsigned err = 0;
    float gpu_ms = 0.f;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaStreamSynchronize(cb_[c]->st), "draft sync");
        err |= h_.err[c];
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, cb_[c]->ev0, cb_[c]->ev1), "event time");
        gpu_ms = std::max(gpu_ms, ms);
    }
    last_draft_gpu_ms = gpu_ms;
    if (const char* e = window_error(err, true)) throw std::runtime_error(std::string("draft_lanes: ") + e);
    for (int i = 0; i < nl; ++i) {
        if (!reqs[(size_t) i].cands) continue;
        std::vector<std::pair<float, int>>& cd = last_lane_cands[(size_t) i];
        for (int c = 0; c < NCARD; ++c) {
            if (mtp_->cards[c].dv1 <= mtp_->cards[c].dv0) continue;
            const size_t at = (size_t) i * NCARD + c;
            if (h_.mtp_tk_err_l[at] & 16u) throw std::runtime_error("draft_lanes: the draft layer's logits hold a NaN or inf");
            for (int k = 0; k < TOPK_N; ++k) cd.emplace_back(h_.mtp_tk_v_l[at * TOPK_N + k], mtp_->vocab[h_.mtp_tk_i_l[at * TOPK_N + k]]);
        }
        std::sort(cd.begin(), cd.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        cd.resize(TOPK_N);
    }
    std::vector<int> out((size_t) nl);
    for (int i = 0; i < nl; ++i) {
        const int row = off[(size_t) i] + reqs[(size_t) i].T - 1;
        int best = 0;
        for (int c = 1; c < NCARD; ++c)
            if (mtp_->cards[c].dv1 > mtp_->cards[c].dv0 && h_.bv[(size_t) c * MAXT + row] > h_.bv[(size_t) best * MAXT + row]) best = c;
        out[(size_t) i] = mtp_->vocab[h_.bi[(size_t) best * MAXT + row]];
    }
    last_draft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ---- DFlash2 ------------------------------------------------------------------------------------------------------------------------------------
#if FNX_DF2
void Engine::df2_forget(int lane) {
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace) std::fprintf(stderr, "df2: lane %d forgotten (drafter ring_hi %ld, host ring [%ld, %ld))\n", lane, (long) df2_->ring_hi(lane), df2_lane_[(size_t) lane].lo, df2_lane_[(size_t) lane].hi);
    df2_->reset(lane);
    df2_lane_[(size_t) lane].lo = df2_lane_[(size_t) lane].hi = 0;
}

Engine::Df2Info Engine::df2_info(int lane) const {
    Df2Info i;
    if (!df2_) return i;
    i.lo = df2_lane_[(size_t) lane].lo, i.hi = df2_lane_[(size_t) lane].hi, i.ring_hi = (long) df2_->ring_hi(lane);
    return i;
}

// the lane's drafter brought up to position pos0 (the first row of the window about to be drafted from) with what the prompt reads tapped: its ring holds
// the positions below pos0 in the host ring [lo, hi) that it has not got yet, the last 2047 at most. A ring that is ahead of pos0 (the sequence went back) or
// does not touch the host ring's run starts over; whatever is missing is left out (the drafts are poorer for a while, never wrong: the verify decides)
void Engine::df2_catch_up(int lane, int pos0) {
    Df2Lane& L = df2_lane_[(size_t) lane];
    df2::Bridge& B = *df2_;
    long have = (long) B.ring_hi(lane);
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace && have != pos0) std::fprintf(stderr, "df2: lane %d catch-up at %d: drafter ring_hi %ld, host ring [%ld, %ld)\n", lane, pos0, have, L.lo, L.hi);
    if (have == pos0) return;
    if (have > pos0) {
        B.reset(lane);
        have = 0;
    }
    if (!L.ring || L.hi <= L.lo) return;
    long from = std::max({L.lo, have, (long) pos0 - df2::kRing + 1});
    const long to = std::min<long>(L.hi, pos0);
    if (from >= to) return;
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    std::string err;
    while (from < to) {   // (the ring wraps at kRing positions: at most two runs)
        const long slot = from % df2::kRing, n = std::min<long>(to - from, df2::kRing - slot);
        if (!B.context(lane, L.ring + (size_t) slot * rowf, n, from, err)) throw std::runtime_error("dflash2: " + err);
        from += n;
    }
}

void Engine::df2_defer(const Df2Req& r) {
    if (!df2_) throw std::runtime_error("df2_defer: no DFlash2 attached");
    if (r.used < 1 || r.row0 < 0 || r.row0 + r.used > MAXT) throw std::runtime_error("df2_defer: rows outside the window");
    Df2Lane& L = df2_lane_[(size_t) r.lane];
    if (!L.ring) {
        ck(cudaSetDevice(cb_[0]->dev), "set");
        ck(cudaHostAlloc((void**) &L.ring, (size_t) df2::kRing * df2::kTaps * df2::kNEmbd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "df2 tap ring");
        own_pinned(L.ring);
    }
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    ck(cudaSetDevice(cb_[0]->dev), "set");
    long pos = r.pos0;
    for (int done = 0; done < r.used;) {
        const int at = (int) (pos % df2::kRing), k = std::min(r.used - done, df2::kRing - at);
        ck(cudaMemcpy(L.ring + (size_t) at * rowf, df2_tapw_ + (size_t) (r.row0 + done) * rowf, (size_t) k * rowf * 4, cudaMemcpyDeviceToHost), "df2 defer");
        done += k;
        pos += k;
    }
    const long e = (long) r.pos0 + r.used;
    if (L.hi == r.pos0 && L.hi > L.lo) L.hi = e, L.lo = std::max<long>(L.lo, e - df2::kRing);
    else L.lo = r.pos0, L.hi = e;
}

void Engine::df2_draft(std::vector<Df2Req>& reqs) {
    if (!df2_) throw std::runtime_error("df2_draft: no DFlash2 attached");
    const auto t0 = std::chrono::steady_clock::now();
    df2::Bridge& B = *df2_;
    ck(cudaSetDevice(cb_[0]->dev), "set");
    constexpr size_t rowf = (size_t) df2::kTaps * df2::kNEmbd;
    std::string err;
    for (Df2Req& r : reqs) {
        if (r.used < 1 || r.row0 < 0 || r.row0 + r.used > MAXT) throw std::runtime_error("df2_draft: rows outside the window");
        if (r.lane < 0 || r.lane >= B.lanes()) throw std::runtime_error("df2_draft: no drafter for this lane");
        df2_catch_up(r.lane, r.pos0);
        const float* src = df2_tapw_ + (size_t) r.row0 * rowf;
        int n = r.used;
        long p0 = r.pos0;
        if (n > df2::kRows) {   // a lookup window kept more rows than a round takes: the first ones only update the ring
            if (!B.context(r.lane, src, n - df2::kRows, p0, err)) throw std::runtime_error("dflash2: " + err);
            src += (size_t) (n - df2::kRows) * rowf;
            p0 += n - df2::kRows;
            n = df2::kRows;
        }
        ck(cudaMemcpyAsync(B.landing(r.lane), src, (size_t) n * rowf * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) B.stream(r.lane)), "df2 taps");
        if (!B.launch(r.lane, n, p0, r.anchor, err)) throw std::runtime_error("dflash2: " + err);
    }
    for (Df2Req& r : reqs)
        if (!B.finish(r.lane, r.tokens, r.probs, err)) throw std::runtime_error("dflash2: " + err);
    last_df2_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static const bool trace = env_flag("FNX_DF2_TRACE");
    if (trace) {   // the mean wall time of a call by the number of lanes in it, every 400 calls
        static double ms[MAXT + 1] = {};
        static long n[MAXT + 1] = {}, calls = 0;
        const size_t k = std::min<size_t>(reqs.size(), MAXT);
        ms[k] += last_df2_ms, ++n[k];
        if (++calls % 400 == 0) {
            std::fprintf(stderr, "df2: drafter calls by lanes (mean ms):");
            for (int i = 1; i <= MAXT; ++i)
                if (n[i]) std::fprintf(stderr, " %d lanes %.2f (%ld)", i, ms[i] / (double) n[i], n[i]);
            std::fprintf(stderr, "%c", 10);
        }
    }
}
#else
void Engine::df2_forget(int) {}
Engine::Df2Info Engine::df2_info(int) const { return {}; }
void Engine::df2_catch_up(int, int) {}
void Engine::df2_draft(std::vector<Df2Req>&) { throw std::runtime_error("this engine was built without DFlash2"); }
void Engine::df2_defer(const Df2Req&) {}
#endif

}  // namespace fnx
