// Numbers parsed whole, for sFNX's settings from the environment and its request lines: a value that is not a number
// is an error naming the variable and the value (std::atoi and friends read garbage as 0, which turned a setting off
// without a word, and std::stoi / stof stop at the first character they cannot read, so "0.7x" was 0.7).
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fnx {

// s as a whole number in base 10: an optional sign, digits, nothing else, within long long
inline bool parse_whole(const char* s, long long& x) {
    if (!(s[0] == '-' || s[0] == '+' || (s[0] >= '0' && s[0] <= '9'))) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtoll(s, &end, 10);
    return *end == '\0' && end != s && errno != ERANGE;
}

// s as a finite decimal number (strtod also reads hex floats, inf and nan, and skips leading spaces: none of them pass)
inline bool parse_number(const char* s, double& x) {
    if (!(s[0] == '-' || s[0] == '+' || s[0] == '.' || (s[0] >= '0' && s[0] <= '9')) || std::strpbrk(s, "xX")) return false;
    char* end = nullptr;
    errno = 0;
    x = std::strtod(s, &end);
    return *end == '\0' && end != s && errno != ERANGE && std::isfinite(x);
}

// the variable as a whole number (parse_whole); def when it is not set
inline long long env_int(const char* name, long long def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    long long x = 0;
    if (!parse_whole(v, x)) throw std::runtime_error(std::string(name) + ": '" + v + "' is not a whole number");
    return x;
}

// the variable as a finite decimal number (parse_number); def when it is not set
inline double env_num(const char* name, double def) {
    const char* v = std::getenv(name);
    if (!v) return def;
    double x = 0;
    if (!parse_number(v, x)) throw std::runtime_error(std::string(name) + ": '" + v + "' is not a number");
    return x;
}

// an on/off switch: a whole number, on when not 0; off when it is not set
inline bool env_flag(const char* name) { return env_int(name, 0) != 0; }

}  // namespace fnx
