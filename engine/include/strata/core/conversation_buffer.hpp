#pragma once

#include "strata/core/conversation_segment_pool.hpp"
#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace strata::core {

namespace conversation_buffer_detail {
// Where a segment's bytes come from: the process-wide segment pool (STRATA_PARK_SEGPOOL_MIB, conversation_segment_pool.hpp)
// when one is installed and has a free run for it, else ::operator new as always.  The pool decides by pointer on the way
// back, so allocate and deallocate always agree.
inline void* segment_alloc(size_t bytes) {
    if (SegmentPool* pool = segment_pool())
        if (void* p = pool->allocate(bytes)) return p;
    return ::operator new(bytes);
}
inline void segment_free(void* p, size_t bytes) noexcept {
    if (SegmentPool* pool = segment_pool())
        if (pool->deallocate(p, bytes)) return;
    ::operator delete(p);
}
}   // namespace conversation_buffer_detail

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t segment_count() const { return segments_.size(); }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) { resize_impl(n, &value); }
    void resize_impl(size_t n, const uint8_t* value) {
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                if (value) segments_.back().resize(segments_.back().size() + extend, *value);
                else segments_.back().resize(segments_.back().size() + extend);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                if (value) segment.resize(count, *value);
                else segment.resize(count);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }
    // resize() without the zero fill of what it adds (the fresh pages are not touched here): only for a caller that
    // writes every added byte before anything reads it (STRATA_PARK_ASYNC's background copy)
    void resize_for_overwrite(size_t n) { resize_impl(n, nullptr); }

    // Allocation ranges include the spare capacity in the final segment.
    template<class Fn> bool visit_allocations(Fn fn) const {
        for (size_t i = 0; i < segments_.size(); ++i)
            if (!fn(segments_[i].data(), segments_[i].capacity(), i)) return false;
        return true;
    }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + (at - offset), p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    // value-initializes only when asked (resize's explicit fill value): resize_for_overwrite leaves the bytes alone
    template<class T> struct DefaultInit : std::allocator<T> {
        template<class U> struct rebind { using other = DefaultInit<U>; };
        DefaultInit() = default;
        template<class U> DefaultInit(const DefaultInit<U>&) noexcept {}
        T* allocate(size_t n) { return static_cast<T*>(conversation_buffer_detail::segment_alloc(n * sizeof(T))); }
        void deallocate(T* p, size_t n) noexcept { conversation_buffer_detail::segment_free(p, n * sizeof(T)); }
        template<class U> void construct(U* p) { ::new ((void*) p) U; }
        template<class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
    };
    using Segment = std::vector<uint8_t, DefaultInit<uint8_t>>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + (offset - begin), n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

// The caller has validated non-overlapping CPU ranges and grown `parked` to
// max(active_bytes, parked_bytes).  No allocation or address change occurs.
// `written` becomes true before the first destination byte is changed.
inline bool conversation_buffer_exchange(ConversationBuffer& parked, uint8_t* active,
                                         size_t active_bytes, size_t parked_bytes,
                                         uint8_t* scratch, size_t block_bytes, bool& written) {
    if (!scratch || !block_bytes || (!active && (active_bytes || parked_bytes)) ||
        parked.size() < std::max(active_bytes, parked_bytes)) return false;
    auto copy = [](void* dst, const void* src, size_t bytes) {
        constexpr size_t step = size_t(1) << 20;
        auto* d = static_cast<uint8_t*>(dst);
        const auto* s = static_cast<const uint8_t*>(src);
        for (size_t at = 0; at < bytes;) {
            const size_t n = std::min(step, bytes - at);
            std::memcpy(d + at, s + at, n);
            at += n;
            progress_beat();
        }
    };
    const size_t common = std::min(active_bytes, parked_bytes);
    for (size_t at = 0; at < common;) {
        const size_t n = std::min(block_bytes, common - at);
        copy(scratch, active + at, n);
        if (!parked.visit(at, n, [&](uint8_t* p, size_t count, size_t pos) {
                written = true;
                copy(active + pos, p, count);
                copy(p, scratch + (pos - at), count);
                return true;
            })) return false;
        at += n;
    }
    const size_t end = std::max(active_bytes, parked_bytes);
    for (size_t at = common; at < end;) {
        const size_t n = std::min(block_bytes, end - at);
        if (!parked.visit(at, n, [&](uint8_t* p, size_t count, size_t pos) {
                written = true;
                if (active_bytes > parked_bytes) copy(p, active + pos, count);
                else copy(active + pos, p, count);
                return true;
            })) return false;
        at += n;
    }
    return true;
}

} // namespace strata::core
