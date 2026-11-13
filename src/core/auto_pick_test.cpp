// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/auto_pick_test.cpp - the settings of a speed switch and the choice its first windows make (header-only, CPU).
//
//   1. XESTRATA_<name>: unset, "auto", "0"/"off", "1"/"on", and a value that is none of them (auto);
//   2. the windows alternate ABBA, the default first, and the choice is the faster time a token;
//   3. the other form has to be 3% faster: a run that cannot tell them apart keeps the default;
//   4. windows where the switch changes nothing are not counted, and a run that never sees one never decides.
#include "strata/core/auto_pick.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace core = strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-86s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void clear_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}
}  // namespace

int main() {
    std::printf("auto_pick_test\n");
    {   // the three settings (`env_get` adds the prefix; the test names the variable whole)
        const char* name = "AUTO_PICK_TEST";
        const char* key = "XESTRATA_AUTO_PICK_TEST";
        clear_env(key);
        check(core::pick_from_env(name) == core::Pick::Auto, "unset: auto");
        set_env(key, "auto");
        check(core::pick_from_env(name) == core::Pick::Auto, "\"auto\": auto");
        set_env(key, "0");
        check(core::pick_from_env(name) == core::Pick::Off, "\"0\": off");
        set_env(key, "off");
        check(core::pick_from_env(name) == core::Pick::Off, "\"off\": off");
        set_env(key, "no");
        check(core::pick_from_env(name) == core::Pick::Off, "\"no\": off");
        set_env(key, "1");
        check(core::pick_from_env(name) == core::Pick::On, "\"1\": on");
        set_env(key, "on");
        check(core::pick_from_env(name) == core::Pick::On, "\"on\": on");
        set_env(key, "yes");
        check(core::pick_from_env(name) == core::Pick::On, "\"yes\": on");
        set_env(key, "5");
        check(core::pick_from_env(name) == core::Pick::Auto, "a value that is none of them: auto");
        clear_env(key);
    }
    {   // ABBA, the default first, then the faster form
        core::WindowChoice c("test", "TEST", "the test form");
        check(!c.on() && c.choosing() && c.blocks() == 4, "the first window is the default, four blocks of two rounds");
        bool order[32];
        for (int i = 0; i < 32; ++i) {
            order[i] = c.on();
            c.took(10.0, 1);   // one ms a window under either form: the default wins a tie
        }
        bool abba = true;
        for (int i = 0; i < 32; ++i) {   // 8 windows a block, A B B A
            const int b = i / 8;
            abba = abba && order[i] == (((b % 2) != ((b / 2) % 2)));
        }
        check(abba, "the windows run default, other, other, default");
        check(!c.choosing() && !c.on(), "a tie keeps the default (off)");
    }
    {   // the other form 10% faster: on
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 32; ++i) c.took(c.on() ? 9.0 : 10.0, 1);
        check(!c.choosing() && c.on(), "the other form 10% faster: on");
    }
    {   // the other form 1% faster: within the margin, the default
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 32; ++i) c.took(c.on() ? 9.9 : 10.0, 1);
        check(!c.choosing() && !c.on(), "the other form 1% faster: off (3% is the margin)");
    }
    {   // the default 10% faster: off
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 32; ++i) c.took(c.on() ? 11.0 : 10.0, 1);
        check(!c.choosing() && !c.on(), "the default 10% faster: off");
    }
    {   // no informative window: it keeps waiting, and the default stays
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 64; ++i) c.took(c.on() ? 1.0 : 10.0, 1, false);
        check(c.choosing() && !c.on(), "no window where the switch matters: off, still waiting");
    }
    {   // a difference smaller than the windows' own spread: within twice the standard error, the default
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 32; ++i) c.took(c.on() ? (i % 2 ? 11.25 : 7.75) : (i % 2 ? 11.5 : 8.5), 1);
        check(!c.choosing() && !c.on(), "5% apart but noisy: off (twice the standard error)");
    }
    {   // tokens are weighted, not windows: a long window's milliseconds count over its rows
        core::WindowChoice c("test", "TEST", "the test form");
        for (int i = 0; i < 32; ++i) c.took(c.on() ? 9.0 : 10.0, c.on() ? 4 : 1);
        check(!c.choosing() && c.on(), "the time a token: 9 ms over four tokens beats 10 ms over one");
    }
    std::printf("%s\n", g_fail == 0 ? "auto_pick_test: all passed" : "auto_pick_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
