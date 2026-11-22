#pragma once

#include "strata/tp/limits.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::tp {
class Packet {
    std::string bytes_;
public:
    void append(const void* data, size_t bytes) {
        if (bytes > max_prompt_bytes - bytes_.size()) throw std::runtime_error("MPI operation exceeds the command limit");
        if (bytes) bytes_.append(static_cast<const char*>(data), bytes);
    }
    const std::string& bytes() const { return bytes_; }
};

template<class T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
void write(Packet& p, const T& value) { p.append(&value, sizeof value); }
inline void write(Packet& p, const std::string& value) {
    if (value.size() > max_prompt_bytes) throw std::runtime_error("MPI string exceeds the command limit");
    write(p, uint32_t(value.size())); p.append(value.data(), value.size());
}
template<class A, class B> void write(Packet& p, const std::pair<A, B>& value);
template<class T> void write(Packet& p, const std::vector<T>& values) {
    if (values.size() > max_prompt_bytes) throw std::runtime_error("MPI vector exceeds the command limit");
    write(p, uint32_t(values.size()));
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) p.append(values.data(), values.size() * sizeof(T));
    else for (const auto& value : values) write(p, value);
}
template<class A, class B> void write(Packet& p, const std::pair<A, B>& value) {
    write(p, value.first); write(p, value.second);
}

class PacketReader {
    std::string_view bytes_;
public:
    explicit PacketReader(std::string_view bytes) : bytes_(bytes) {
        if (bytes.size() > max_prompt_bytes) throw std::runtime_error("MPI command exceeds the packet limit");
    }
    void take(void* out, size_t bytes) {
        if (bytes > bytes_.size()) throw std::runtime_error("truncated MPI operation");
        if (bytes) std::memcpy(out, bytes_.data(), bytes);
        bytes_.remove_prefix(bytes);
    }
    template<class T, std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, int> = 0>
    T scalar() {
        static_assert(!std::is_same_v<T, bool>, "read a wire bool with boolean()");
        T value; take(&value, sizeof value); return value;
    }
    bool boolean() {
        const auto value = scalar<uint8_t>();
        if (value > 1) throw std::runtime_error("invalid MPI boolean");
        return value != 0;
    }
    uint32_t count() {
        const auto n = scalar<uint32_t>();
        if (n > bytes_.size()) throw std::runtime_error("MPI element count exceeds the packet");
        return n;
    }
    template<class T> std::vector<T> vector() {
        static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);
        const auto n = count();
        if (size_t(n) > bytes_.size() / sizeof(T)) throw std::runtime_error("truncated MPI vector");
        std::vector<T> values(n); take(values.data(), size_t(n) * sizeof(T)); return values;
    }
    std::string string() {
        const auto n = count(); std::string value(bytes_.substr(0, n)); bytes_.remove_prefix(n); return value;
    }
    void finish() const { if (!bytes_.empty()) throw std::runtime_error("extra data after MPI operation"); }
};
}
