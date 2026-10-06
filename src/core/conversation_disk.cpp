// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/conversation_disk.hpp"
#include "conversation_checked.hpp"
#include "strata/kernels/kv_stream.hpp"

#ifdef STRATA_HAVE_BLOSC2
#include <blosc2.h>
#endif

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>

namespace strata::core {
namespace {
namespace fs = std::filesystem;
using conversation_detail::add;
using conversation_detail::product;

constexpr char kMagic[4] = {'X', 'S', 'C', 'V'};
constexpr uint32_t kVersion = 1;
constexpr uint8_t kCodecNone = 0, kCodecBlosc2 = 1;
// A compressed part is a run of chunks of this many raw bytes (the last one shorter), each its compressed size and
// one c-blosc2 chunk.  It is a ConversationBuffer segment, so a freshly sized buffer takes a chunk in one piece.
constexpr size_t kChunk = ConversationBuffer::segment_bytes;
constexpr const char* kExtension = ".xsc";
// a header is the conversation's token IDs and a few hundred bytes more: anything far larger is a damaged file
constexpr uint64_t kMaxHeader = uint64_t{1} << 30;
constexpr uint64_t kMaxLayers = 4096;   // K/V layers in a file: far more than any model has
constexpr uint64_t kFnvOffset = 1469598103934665603ull, kFnvPrime = 1099511628211ull;

// FNV-1a over 64-bit words in four independent lanes, so a multi-GB file hashes at memory speed rather than at
// one multiply per byte.  Each step is a bijection of its lane, so any one changed word changes the digest.  It
// detects damage; it is not a defence against a crafted file (the folder is the user's own, mode 0700).
class Hash {
public:
    void update(const void* data, size_t n) {
        const auto* p = static_cast<const uint8_t*>(data);
        total_ += n;
        if (pending_) {
            const size_t take = std::min(n, sizeof block_ - pending_);
            std::memcpy(block_ + pending_, p, take);
            pending_ += take; p += take; n -= take;
            if (pending_ < sizeof block_) return;
            mix(block_);
            pending_ = 0;
        }
        for (; n >= sizeof block_; p += sizeof block_, n -= sizeof block_) mix(p);
        std::memcpy(block_, p, n);
        pending_ = n;
    }
    uint64_t digest() const {
        uint64_t h = kFnvOffset;
        for (uint64_t lane : lanes_) h = (h ^ lane) * kFnvPrime;
        for (size_t i = 0; i < pending_; ++i) h = (h ^ block_[i]) * kFnvPrime;
        h = (h ^ total_) * kFnvPrime;
        // a multiply carries a word's high bits only upward: splitmix64's finalizer spreads them over the digest,
        // so two names that differ in one setting do not share their low half
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
        return h ^ (h >> 31);
    }

private:
    void mix(const uint8_t* p) {
        for (size_t i = 0; i < 4; ++i) {
            uint64_t w;
            std::memcpy(&w, p + 8 * i, 8);
            lanes_[i] = (lanes_[i] ^ w) * kFnvPrime;
        }
    }
    uint64_t lanes_[4] = {kFnvOffset, kFnvOffset ^ 1, kFnvOffset ^ 2, kFnvOffset ^ 3};
    uint8_t block_[32] = {};
    size_t pending_ = 0;
    uint64_t total_ = 0;
};

#ifdef STRATA_HAVE_BLOSC2
// The CPUs the process may use, taken before main: the serve loop pins its own thread to one core
// (SessionLoopScratch), and c-blosc2's threads, started from that thread, would all inherit that one core.
struct StartAffinity {
    cpu_set_t set;
    bool ok;
    StartAffinity() noexcept {
        CPU_ZERO(&set);
        ok = sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0;
    }
};
const StartAffinity g_start_affinity;

// While c-blosc2 runs, the calling thread (and the threads it starts) may use those CPUs; then its pin comes back.
class WideAffinity {
public:
    WideAffinity() {
        restore_ = g_start_affinity.ok && pthread_getaffinity_np(pthread_self(), sizeof saved_, &saved_) == 0 &&
                   pthread_setaffinity_np(pthread_self(), sizeof g_start_affinity.set, &g_start_affinity.set) == 0;
    }
    WideAffinity(const WideAffinity&) = delete;
    WideAffinity& operator=(const WideAffinity&) = delete;
    ~WideAffinity() { if (restore_) pthread_setaffinity_np(pthread_self(), sizeof saved_, &saved_); }

private:
    cpu_set_t saved_;
    bool restore_ = false;
};

// Every CPU the process may use: the expert pool is idle while a conversation is written or read.
int16_t blosc2_threads() {
    const int n = g_start_affinity.ok ? CPU_COUNT(&g_start_affinity.set) : (int) std::thread::hardware_concurrency();
    return (int16_t) std::clamp(n, 1, 256);
}

struct Blosc2Context {
    blosc2_context* ctx = nullptr;
    Blosc2Context() = default;
    Blosc2Context(const Blosc2Context&) = delete;
    Blosc2Context& operator=(const Blosc2Context&) = delete;
    ~Blosc2Context() { if (ctx != nullptr) blosc2_free_ctx(ctx); }
};
#endif

// ---- the header, built in memory (little-endian, as every machine this engine runs on)
struct Header {
    std::string fingerprint;
    bool cvec = true;
    std::array<int64_t, 18> geometry{};
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    struct Checkpoint { uint64_t tokens = 0, images = 0, used = 0; };
    std::vector<Checkpoint> checkpoints;
    uint64_t payload = 0, body_hash = 0;
};

template<class T> void put(std::vector<uint8_t>& out, const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof v);
}

std::vector<uint8_t> encode(const Header& h) {
    std::vector<uint8_t> out;
    put(out, (uint32_t) h.fingerprint.size());
    out.insert(out.end(), h.fingerprint.begin(), h.fingerprint.end());
    put(out, (uint8_t) h.cvec);
    for (int64_t v : h.geometry) put(out, v);
    put(out, (uint64_t) h.ids.size());
    for (int32_t v : h.ids) put(out, v);
    put(out, (uint64_t) h.imgs.size());
    for (const auto& k : h.imgs) { put(out, k.start); put(out, k.hash); }
    put(out, (uint64_t) h.checkpoints.size());
    for (const auto& c : h.checkpoints) { put(out, c.tokens); put(out, c.images); put(out, c.used); }
    put(out, h.payload);
    put(out, h.body_hash);
    return out;
}

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    template<class T> bool get(T& v) {
        if (sizeof v > n_ - at_) return false;
        std::memcpy(&v, p_ + at_, sizeof v);
        at_ += sizeof v;
        return true;
    }
    bool bytes(void* dst, size_t n) {
        if (n > n_ - at_) return false;
        std::memcpy(dst, p_ + at_, n);
        at_ += n;
        return true;
    }
    size_t left() const { return n_ - at_; }

private:
    const uint8_t* p_;
    size_t n_, at_ = 0;
};

bool decode(const std::vector<uint8_t>& raw, Header& h) {
    Reader r(raw.data(), raw.size());
    uint32_t fp = 0;
    uint8_t cvec = 0;
    uint64_t n = 0;
    if (!r.get(fp) || fp > r.left()) return false;
    h.fingerprint.resize(fp);
    if (!r.bytes(h.fingerprint.data(), fp) || !r.get(cvec) || cvec > 1) return false;
    h.cvec = cvec != 0;
    for (int64_t& v : h.geometry) if (!r.get(v)) return false;
    if (!r.get(n) || n > r.left() / sizeof(int32_t)) return false;
    h.ids.resize((size_t) n);
    for (int32_t& v : h.ids) if (!r.get(v)) return false;
    if (!r.get(n) || n > r.left() / 16) return false;
    h.imgs.resize((size_t) n);
    for (auto& k : h.imgs) if (!r.get(k.start) || !r.get(k.hash)) return false;
    if (!r.get(n) || n > r.left() / 24) return false;
    h.checkpoints.resize((size_t) n);
    for (auto& c : h.checkpoints) {
        if (!r.get(c.tokens) || !r.get(c.images) || !r.get(c.used) || c.tokens == 0 || c.tokens > h.ids.size() ||
            c.images > h.imgs.size() || (c.images < h.imgs.size() && (uint64_t) h.imgs[c.images].start < c.tokens) ||
            (c.images > 0 && (uint64_t) h.imgs[c.images - 1].start >= c.tokens))
            return false;
    }
    return r.get(h.payload) && r.get(h.body_hash) && r.left() == 0 && !h.ids.empty();
}

// The checkpoints a file keeps: the deepest (where the next turn of this chat resumes; the next request does not
// always render this reply's header or thinking the same), the chain's root (the system prompt's end, where a new
// chat of the same client starts), then the most recently used.  In their chain order.
std::vector<size_t> kept_checkpoints(const std::vector<ConversationCheckpoint>& all, size_t limit) {
    std::vector<size_t> order;
    if (all.empty() || limit == 0) return order;
    size_t deepest = 0, root = 0;
    for (size_t i = 1; i < all.size(); ++i) {
        if (all[i].ids.size() >= all[deepest].ids.size()) deepest = i;
        if (all[i].ids.size() < all[root].ids.size()) root = i;
    }
    order.push_back(deepest);
    if (root != deepest) order.push_back(root);
    std::vector<size_t> rest;
    for (size_t i = 0; i < all.size(); ++i) if (i != deepest && i != root) rest.push_back(i);
    std::stable_sort(rest.begin(), rest.end(), [&](size_t a, size_t b) { return all[a].used > all[b].used; });
    order.insert(order.end(), rest.begin(), rest.end());
    if (order.size() > limit) order.resize(limit);
    std::sort(order.begin(), order.end());
    return order;
}

std::array<const std::vector<uint8_t>*, 5> state_parts(const ConversationCheckpoint& c) {
    return {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
}

// ---- the body: every running state (the live one first), then every K/V layer.  Each part is its size, its codec
// and its bytes.  With --conversation-save-compress the floating-point parts that compress (the DeltaNet states,
// the indexer rows, FP16 K/V) go through c-blosc2's ZSTD at level 1 after a byte shuffle; 1-byte K/V codes gain
// 4-7% and the small parts nothing, so they stay as they are (bench/results/2026-10-04-conversation-save-compress).
struct Writer {
    std::FILE* f = nullptr;
    Hash hash;
    bool ok = true;
    bool compress = false;
    std::vector<uint8_t> stage, packed;   // a chunk gathered from a buffer's segments, and its compressed form
#ifdef STRATA_HAVE_BLOSC2
    Blosc2Context cctx[5];                // by element size
#endif
    void write(const void* p, size_t n) {
        if (!ok || n == 0) return;
        ok = std::fwrite(p, 1, n, f) == n;
        hash.update(p, n);
    }
    template<class T> void value(const T& v) { write(&v, sizeof v); }
    // typesize: the element size the shuffle groups bytes by (2 or 4), 0 for a part stored as it is
    bool compressing(size_t n, int typesize) const { return compress && typesize > 0 && n > 0; }
    void part(const void* p, size_t n, int typesize = 0) {
        value((uint64_t) n);
        if (!compressing(n, typesize)) {
            value(kCodecNone);
            write(p, n);
            return;
        }
        value(kCodecBlosc2);
        for (size_t at = 0; at < n && ok; at += kChunk)
            chunk(static_cast<const uint8_t*>(p) + at, std::min(kChunk, n - at), typesize);
    }
    void part(const ConversationBuffer& b, int typesize = 0) {
        value((uint64_t) b.size());
        if (!compressing(b.size(), typesize)) {
            value(kCodecNone);
            b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) { write(p, n); return ok; });
            return;
        }
        value(kCodecBlosc2);
        for (size_t at = 0; at < b.size() && ok; at += kChunk) {
            const size_t len = std::min(kChunk, b.size() - at);
            const uint8_t* src = nullptr;
            b.visit(at, len, [&](const uint8_t* p, size_t k, size_t) { if (k == len) src = p; return true; });
            if (src == nullptr) {   // the chunk spans segments: gather it
                stage.resize(len);
                ok = b.read(stage.data(), at, len);
                src = stage.data();
            }
            if (ok) chunk(src, len, typesize);
        }
    }
    void chunk(const uint8_t* src, size_t len, int typesize) {
#ifdef STRATA_HAVE_BLOSC2
        Blosc2Context& c = cctx[typesize];
        if (c.ctx == nullptr) {
            blosc2_cparams cp = BLOSC2_CPARAMS_DEFAULTS;   // its one filter is the byte shuffle
            cp.compcode = BLOSC_ZSTD;
            cp.clevel = 1;
            cp.typesize = typesize;
            cp.nthreads = blosc2_threads();
            c.ctx = blosc2_create_cctx(cp);
        }
        packed.resize(len + BLOSC2_MAX_OVERHEAD);
        const int n = c.ctx == nullptr ? -1 : blosc2_compress_ctx(c.ctx, src, (int32_t) len, packed.data(),
                                                                   (int32_t) packed.size());
        if (n <= 0) {
            ok = false;
            return;
        }
        value((uint64_t) n);
        write(packed.data(), (size_t) n);
#else
        (void) src; (void) len; (void) typesize;
        ok = false;
#endif
    }
};

constexpr size_t kPartHeader = sizeof(uint64_t) + sizeof(uint8_t);
constexpr size_t kLayerHeader = sizeof(int32_t) + 6 * sizeof(int64_t);

// a whole read, with the stream still good for the next one
bool read_exact(std::FILE* f, void* p, size_t n) {
    return std::fread(p, 1, n, f) == n && !std::ferror(f) && !std::feof(f);
}

// The checksum covers the stored bytes, so a compressed chunk is decompressed before the file's checksum is known:
// its sizes are checked first, and c-blosc2 checks the rest of a chunk itself.
struct FileReader {
    std::FILE* f = nullptr;
    Hash hash;
    uint64_t left = 0;            // payload bytes the header promised and the body has not used yet
    const char* why = nullptr;    // a reason more useful than "damaged", when there is one
    std::vector<uint8_t> stage, packed;
#ifdef STRATA_HAVE_BLOSC2
    Blosc2Context dctx;
#endif
    bool read(void* p, size_t n) {
        if (n == 0) return true;
        if (!read_exact(f, p, n)) return false;
        hash.update(p, n);
        return true;
    }
    template<class T> bool value(T& v) { return read(&v, sizeof v); }
    bool size(uint64_t& n, uint8_t& codec) {
        if (!value(n) || !value(codec) || n > left) return false;
        if (codec == kCodecBlosc2 && !conversation_disk_can_compress()) {
            why = "compressed, and this build has no c-blosc2";
            return false;
        }
        if (codec != kCodecNone && codec != kCodecBlosc2) return false;
        left -= n;
        return true;
    }
    bool part(std::vector<uint8_t>& v) {
        uint64_t n = 0;
        uint8_t codec = 0;
        if (!size(n, codec)) return false;
        v.resize((size_t) n);
        if (codec == kCodecNone) return read(v.data(), v.size());
        for (size_t at = 0; at < v.size(); at += kChunk)
            if (!chunk(v.data() + at, std::min(kChunk, v.size() - at))) return false;
        return true;
    }
    bool part(ConversationBuffer& b) {
        uint64_t n = 0;
        uint8_t codec = 0;
        if (!size(n, codec)) return false;
        b.resize((size_t) n);
        if (codec == kCodecNone) return b.visit(0, b.size(), [&](uint8_t* p, size_t k, size_t) { return read(p, k); });
        for (size_t at = 0; at < b.size(); at += kChunk) {
            const size_t len = std::min(kChunk, b.size() - at);
            uint8_t* dst = nullptr;
            b.visit(at, len, [&](uint8_t* p, size_t k, size_t) { if (k == len) dst = p; return true; });
            if (dst != nullptr) {
                if (!chunk(dst, len)) return false;
                continue;
            }
            stage.resize(len);   // the chunk spans segments: decompress, then scatter
            if (!chunk(stage.data(), len)) return false;
            b.visit(at, len, [&](uint8_t* p, size_t k, size_t from) {
                std::memcpy(p, stage.data() + (from - at), k);
                return true;
            });
        }
        return true;
    }
    bool chunk(uint8_t* dst, size_t len) {
#ifdef STRATA_HAVE_BLOSC2
        uint64_t bytes = 0;
        if (!value(bytes) || bytes < BLOSC_MIN_HEADER_LENGTH || bytes > len + BLOSC2_MAX_OVERHEAD) return false;
        packed.resize((size_t) bytes);
        if (!read(packed.data(), packed.size())) return false;
        int32_t raw = 0, stored = 0, block = 0;
        if (blosc2_cbuffer_sizes(packed.data(), &raw, &stored, &block) < 0 || (size_t) raw != len ||
            (uint64_t) stored != bytes) return false;
        if (dctx.ctx == nullptr) {
            blosc2_dparams dp = BLOSC2_DPARAMS_DEFAULTS;
            dp.nthreads = blosc2_threads();
            dctx.ctx = blosc2_create_dctx(dp);
        }
        return dctx.ctx != nullptr &&
               blosc2_decompress_ctx(dctx.ctx, packed.data(), (int32_t) bytes, dst, (int32_t) len) == (int) len;
#else
        (void) dst; (void) len;
        return false;
#endif
    }
};

std::string hex(uint64_t v) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long) v);
    return s;
}

struct Found {
    fs::path path;
    fs::file_time_type time;
    uint64_t bytes = 0;
};

std::vector<Found> scan(const fs::path& dir) {
    std::vector<Found> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e;
        if (!it->is_regular_file(e) || it->path().extension() != kExtension) continue;
        const auto time = it->last_write_time(e);
        const auto bytes = e ? 0 : it->file_size(e);
        if (!e) out.push_back({it->path(), time, bytes});
    }
    std::sort(out.begin(), out.end(), [](const Found& a, const Found& b) { return a.time < b.time; });
    return out;
}

bool read_header(std::FILE* f, Header& h) {
    char magic[4];
    uint32_t version = 0;
    uint64_t length = 0, stored = 0;
    if (!read_exact(f, magic, 4) || std::memcmp(magic, kMagic, 4) != 0 ||
        !read_exact(f, &version, sizeof version) || version != kVersion ||
        !read_exact(f, &length, sizeof length) || length == 0 || length > kMaxHeader) return false;
    // the header and its checksum, in one read
    std::vector<uint8_t> raw((size_t) length + sizeof stored);
    if (!read_exact(f, raw.data(), raw.size())) return false;
    std::memcpy(&stored, raw.data() + length, sizeof stored);
    raw.resize((size_t) length);
    Hash hash;
    hash.update(raw.data(), raw.size());
    return hash.digest() == stored && decode(raw, h);
}

ConversationCheckpoint prefix_of(const Header& h, const Header::Checkpoint& c) {
    ConversationCheckpoint out;
    out.ids.assign(h.ids.begin(), h.ids.begin() + (std::ptrdiff_t) c.tokens);
    out.imgs.assign(h.imgs.begin(), h.imgs.begin() + (std::ptrdiff_t) c.images);
    out.used = c.used;
    return out;
}
} // namespace

bool conversation_disk_can_compress() {
#ifdef STRATA_HAVE_BLOSC2
    return true;
#else
    return false;
#endif
}

std::string conversation_file_identity(const std::string& path) {
    if (path.empty()) return "-";
    std::error_code ec;
    const fs::path p = fs::weakly_canonical(path, ec);
    if (ec) return "?" + path;
    auto stamp = [](const fs::path& file) {
        std::error_code e;
        const auto bytes = fs::file_size(file, e);
        if (e) return std::string("?");
        const auto time = fs::last_write_time(file, e);
        if (e) return std::string("?");
        return std::to_string(bytes) + "@" + std::to_string((long long) time.time_since_epoch().count());
    };
    if (!fs::is_directory(p, ec)) return p.string() + ":" + stamp(p);
    std::vector<std::string> files;
    for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e;
        if (it->is_regular_file(e)) files.push_back(it->path().filename().string() + ":" + stamp(it->path()));
    }
    if (ec) return "?" + p.string();
    std::sort(files.begin(), files.end());
    std::string out = p.string() + "/{";
    for (const auto& f : files) out += f + ",";
    return out + "}";
}

bool ConversationDisk::open(ConversationDiskOptions options, std::string& error) {
    enabled_ = false;
    entries_.clear();
    options_ = std::move(options);
    std::error_code ec;
    if (!fs::exists(options_.dir, ec)) {
        if (!fs::create_directories(options_.dir, ec) || ec) {
            error = "cannot create " + options_.dir.string() + ": " + ec.message();
            return false;
        }
        // the files hold what was computed from the conversations: the owner's only
        fs::permissions(options_.dir, fs::perms::owner_all, fs::perm_options::replace, ec);
    }
    if (!fs::is_directory(options_.dir, ec)) {
        error = options_.dir.string() + " is not a folder";
        return false;
    }
    // a write cut short by a crash leaves its .tmp; nothing reads one
    for (fs::directory_iterator it(options_.dir, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.size() > 8 && name.compare(name.size() - 8, 8, ".xsc.tmp") == 0) {
            std::error_code e;
            fs::remove(it->path(), e);
        }
    }
#ifdef STRATA_HAVE_BLOSC2
    static const bool blosc2_ready = (blosc2_init(), true);   // once per process; the contexts do the rest
    (void) blosc2_ready;
#endif
    if (options_.compress && !conversation_disk_can_compress()) {
        error = "compression needs c-blosc2, which this build lacks";
        return false;
    }
    enabled_ = true;
    expire();
    for (const Found& found : scan(options_.dir)) {
        std::FILE* f = std::fopen(found.path.c_str(), "rb");
        if (f == nullptr) continue;
        Header h;
        const bool ok = read_header(f, h);
        std::fclose(f);
        // another model, engine or setting (kept until it expires: its own engine may still use it), or damaged
        if (!ok || h.fingerprint != options_.fingerprint) continue;
        Entry e;
        e.path = found.path;
        for (const auto& c : h.checkpoints) e.checkpoints.push_back(prefix_of(h, c));
        e.live.ids = std::move(h.ids);
        e.live.imgs = std::move(h.imgs);
        e.cvec = h.cvec;
        e.payload = (size_t) h.payload;
        entries_.push_back(std::move(e));
    }
    return true;
}

bool ConversationDisk::remove_file(const fs::path& path) {
    std::error_code ec;
    const bool removed = fs::remove(path, ec);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.path == path; }),
                   entries_.end());
    return removed;
}

size_t ConversationDisk::expire() {
    if (!enabled_) return 0;
    const auto now = fs::file_time_type::clock::now();
    size_t removed = 0;
    auto found = scan(options_.dir);
    folder_bytes_ = 0;
    std::vector<Found> kept;
    for (const Found& f : found) {
        if (options_.max_age.count() > 0 && now - f.time > options_.max_age) {
            removed += remove_file(f.path);
            continue;
        }
        kept.push_back(f);
        folder_bytes_ += (size_t) f.bytes;
    }
    // oldest first, every file in the folder (another model's too: the budget is the folder's)
    for (size_t i = 0; i < kept.size() && folder_bytes_ > options_.budget; ++i) {
        folder_bytes_ -= (size_t) kept[i].bytes;
        removed += remove_file(kept[i].path);
    }
    return removed;
}

void ConversationDisk::touch(size_t index) {
    Entry e = std::move(entries_.at(index));
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    std::error_code ec;
    fs::last_write_time(e.path, fs::file_time_type::clock::now(), ec);
    entries_.push_back(std::move(e));
}

bool ConversationDisk::load(size_t index, SavedConversation& image, std::string& error) {
    const fs::path path = entries_.at(index).path;
    auto reject = [&](const char* why) {
        error = path.filename().string() + ": " + why + " (deleted)";
        remove_file(path);
        return false;
    };
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return reject("cannot open");
    struct Close { std::FILE* f; ~Close() { std::fclose(f); } } close{f};
    Header h;
    if (!read_header(f, h) || h.fingerprint != options_.fingerprint) return reject("unreadable header");
#ifdef STRATA_HAVE_BLOSC2
    const WideAffinity wide;
#endif
    FileReader r;
    r.f = f;
    r.left = h.payload;
    SavedConversation out;
    out.geometry = h.geometry;
    out.cvec = h.cvec;
    out.live.ids = h.ids;
    out.live.imgs = h.imgs;
    for (const auto& c : h.checkpoints) out.checkpoints.push_back(prefix_of(h, c));
    auto one = [&](ConversationCheckpoint& c) {
        return r.part(c.gdn) && r.part(c.ple) && r.part(c.tails) && r.part(c.dead) && r.part(c.block_pos);
    };
    auto state = [&](ConversationCheckpoint& c) {   // its own, then the later GPUs' of a layer split
        bool good = one(c);
        c.stage_parts.resize(options_.stage_parts);
        for (auto& part : c.stage_parts) {
            part.ids = c.ids;
            part.imgs = c.imgs;
            good = good && one(part);
        }
        return good;
    };
    bool ok = state(out.live);
    for (auto& c : out.checkpoints) ok = ok && state(c);
    uint64_t layers = 0;
    ok = ok && r.value(layers) && layers <= kMaxLayers;
    if (ok) out.kv.resize((size_t) layers);
    for (auto& kv : out.kv) {
        ok = ok && r.value(kv.format) && r.value(kv.cells) && r.value(kv.heads) && r.value(kv.head_dim) &&
             r.value(kv.page_size) && r.value(kv.pooled_rows) && r.value(kv.idx_dim) &&
             r.part(kv.k) && r.part(kv.v) && r.part(kv.k_scale) && r.part(kv.v_scale) && r.part(kv.pooled);
        if (!ok) break;
    }
    if (!ok) return reject(r.why != nullptr ? r.why : "short or damaged body");
    char extra;
    if (r.left != 0 || std::fread(&extra, 1, 1, f) != 0) return reject("unexpected length");
    if (r.hash.digest() != h.body_hash) return reject("checksum differs");
    image = std::move(out);
    return true;
}

bool ConversationDisk::save(const SavedConversation& image, size_t& written, std::string& error) {
    written = 0;
    if (!enabled_) return true;
    const auto kept = kept_checkpoints(image.checkpoints, options_.checkpoints);
    Header h;
    h.fingerprint = options_.fingerprint;
    h.cvec = image.cvec;
    h.geometry = image.geometry;
    h.ids = image.live.ids;
    h.imgs = image.live.imgs;
    // the payload: what a load allocates, and with the part and layer headers what the body holds uncompressed
    size_t payload = 0, body = 0;
    auto count_one = [&](const ConversationCheckpoint& c) {
        for (const auto* p : state_parts(c))
            if (!add(payload, p->size()) || !add(body, kPartHeader)) return false;
        return true;
    };
    auto count_state = [&](const ConversationCheckpoint& c) {
        if (c.stage_parts.size() != options_.stage_parts) return false;
        bool good = count_one(c);
        for (const auto& part : c.stage_parts) good = good && count_one(part);
        return good;
    };
    if (image.live.stage_parts.size() != options_.stage_parts) {
        error = "the conversation's layer-split parts differ from the folder's";
        return false;
    }
    bool sized = count_state(image.live) && add(body, sizeof(uint64_t));
    for (size_t i : kept) {
        const auto& c = image.checkpoints[i];
        if (c.ids.empty() || c.ids.size() > h.ids.size() || !std::equal(c.ids.begin(), c.ids.end(), h.ids.begin())) {
            error = "a checkpoint is not a prefix of the conversation";
            return false;
        }
        size_t images = 0;
        while (images < h.imgs.size() && (uint64_t) h.imgs[images].start < c.ids.size()) ++images;
        h.checkpoints.push_back({c.ids.size(), images, c.used});
        sized = sized && count_state(c);
    }
    for (const auto& kv : image.kv) {
        for (const ConversationBuffer* b : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled})
            sized = sized && add(payload, b->size()) && add(body, kPartHeader);
        sized = sized && add(body, kLayerHeader);
    }
    sized = sized && add(body, payload);
    h.payload = payload;
    std::vector<uint8_t> header = encode(h);
    size_t total = 4 + 4 + 8 + header.size() + 8;
    if (!sized || !add(total, body)) {
        error = "conversation size overflow";
        return false;
    }
    if (total > options_.budget) {
        error = "the conversation (" + std::to_string(total >> 20) + " MiB) exceeds the folder's budget";
        return false;
    }
    Hash name;
    name.update(options_.fingerprint.data(), options_.fingerprint.size());
    name.update(h.ids.data(), h.ids.size() * sizeof(int32_t));
    for (const auto& k : h.imgs) { name.update(&k.start, 8); name.update(&k.hash, 8); }
    const uint8_t cvec = h.cvec;
    name.update(&cvec, 1);
    const fs::path path = options_.dir / (hex(name.digest()) + kExtension);
    const fs::path tmp = path.string() + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    std::FILE* f = fd < 0 ? nullptr : ::fdopen(fd, "wb");
    if (f == nullptr) {
        if (fd >= 0) ::close(fd);
        error = "cannot create " + tmp.string();
        return false;
    }
    const uint64_t length = header.size();
    auto prefix = [&]() {
        Hash hh;
        hh.update(header.data(), header.size());
        const uint64_t header_hash = hh.digest();
        return std::fwrite(kMagic, 1, 4, f) == 4 && std::fwrite(&kVersion, 4, 1, f) == 1 &&
               std::fwrite(&length, 8, 1, f) == 1 && std::fwrite(header.data(), 1, header.size(), f) == header.size() &&
               std::fwrite(&header_hash, 8, 1, f) == 1;
    };
    // the body's checksum is in the header: write the header, the body, then the header again with it
#ifdef STRATA_HAVE_BLOSC2
    std::optional<WideAffinity> wide;
    if (options_.compress) wide.emplace();
#endif
    Writer w;
    w.f = f;
    w.compress = options_.compress;
    w.ok = prefix();
    auto write_one = [&](const ConversationCheckpoint& c) {
        w.part(c.gdn.data(), c.gdn.size(), sizeof(float));   // the DeltaNet states; the rest is about 1 MiB
        for (const auto* p : state_parts(c))
            if (p != &c.gdn) w.part(p->data(), p->size());
    };
    auto write_state = [&](const ConversationCheckpoint& c) {   // its own, then the later GPUs' (count_state)
        write_one(c);
        for (const auto& part : c.stage_parts) write_one(part);
    };
    write_state(image.live);
    for (size_t i : kept) write_state(image.checkpoints[i]);
    w.value((uint64_t) image.kv.size());
    for (const auto& kv : image.kv) {
        w.value(kv.format); w.value(kv.cells); w.value(kv.heads); w.value(kv.head_dim);
        w.value(kv.page_size); w.value(kv.pooled_rows); w.value(kv.idx_dim);
        // the format carries +16 for a rotated K/V; FP16 K/V is 2-byte elements, the others 1-byte codes
        const int kv_type = (kv.format & 15) == strata::kernels::kKvF16 ? 2 : 0;
        w.part(kv.k, kv_type); w.part(kv.v, kv_type); w.part(kv.k_scale); w.part(kv.v_scale);
        w.part(kv.pooled, sizeof(float));
    }
    const long end = std::ftell(f);   // the file's length: the body ends it
    bool ok = w.ok && end > 0;
    if (ok) {
        h.body_hash = w.hash.digest();
        header = encode(h);
        ok = header.size() == length && std::fseek(f, 0, SEEK_SET) == 0 && prefix();
    }
    ok = std::fflush(f) == 0 && ok;
    ok = ::fsync(::fileno(f)) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) fs::rename(tmp, path, ec);
    if (!ok || ec) {
        fs::remove(tmp, ec);
        error = "cannot write " + path.string();
        return false;
    }
    written = (size_t) end;
    // #342 as the RAM cache applies it: a file whose deepest checkpoint this conversation's chain holds is the same
    // chat a turn (or a restart) back
    for (size_t i = 0; i < entries_.size();) {
        const Entry& e = entries_[i];
        const ConversationCheckpoint* deepest = nullptr;
        for (const auto& c : e.checkpoints)
            if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
        if (e.path != path && e.cvec == image.cvec && deepest &&
            conversation_chain_holds(*deepest, image.live.ids, image.live.imgs, image.checkpoints)) {
            remove_file(e.path);
            continue;
        }
        ++i;
    }
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.path == path; }),
                   entries_.end());
    Entry e;
    e.path = path;
    e.live.ids = h.ids;
    e.live.imgs = h.imgs;
    for (const auto& c : h.checkpoints) e.checkpoints.push_back(prefix_of(h, c));
    e.cvec = h.cvec;
    e.payload = payload;
    entries_.push_back(std::move(e));
    expire();
    return true;
}

} // namespace strata::core
