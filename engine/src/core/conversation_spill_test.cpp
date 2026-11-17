// The conversation cache's SSD tier: its folder per engine process and its budget (conversation_spill.hpp).  No GPU, no model.
#include "strata/core/conversation_spill.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
constexpr uint64_t GiB = uint64_t(1) << 30;
void touch(const fs::path& p) { std::ofstream(p) << "x"; }
}  // namespace

int main() {
    // the budget: min(32 GiB, 25% of the free space), never into the floor
    check(spill_default_budget(400 * GiB, 4 * GiB) == 32 * GiB, "400 GiB free: 32 GiB");
    check(spill_default_budget(40 * GiB, 4 * GiB) == 10 * GiB, "40 GiB free: a quarter, 10 GiB");
    check(spill_default_budget(5 * GiB, 4 * GiB) == 1 * GiB, "5 GiB free, floor 4: the 1 GiB above the floor");
    check(spill_default_budget(4 * GiB, 4 * GiB) == 0 && spill_default_budget(3 * GiB, 4 * GiB) == 0, "at or under the floor: none");
    SpillBudget b{10 * GiB, 0, 0};
    check(b.reserve(6 * GiB) && !b.reserve(5 * GiB) && b.reserve(4 * GiB), "reservations count until the writes end");
    b.finish(6 * GiB, 5 * GiB);
    check(b.used == 5 * GiB && b.reserved == 4 * GiB && !b.over(), "a write ends: its reservation goes, its bytes stay");
    b.finish(4 * GiB, 0);
    check(b.used == 5 * GiB && b.reserved == 0, "a failed write leaves nothing");
    check(!b.reserve(6 * GiB) && b.reserve(5 * GiB), "the cap includes what is written");
    b.finish(5 * GiB, 6 * GiB);
    check(b.over(), "a file larger than its reservation shows as over the cap");
    b.drop(6 * GiB);
    check(!b.over() && b.used == 5 * GiB, "a dropped entry gives its bytes back");
    check(!SpillBudget{0, 0, 0}.reserve(1), "no budget: nothing");

    // the names a clean may touch
    check(spill_detail::is_namespace_name("123-0123456789abcdef"), "pid-16 hex");
    for (const char* n : {"123-0123456789ABCDEF", "123-0123456789abcde", "-0123456789abcdef", "12a-0123456789abcdef",
                          "notes", "123-0123456789abcdef0", "123_0123456789abcdef"})
        check(!spill_detail::is_namespace_name(n), n);

    // the folders: two live engines on one root keep each other's folder; a dead one's goes; nothing else is touched
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(std::random_device{}()));
    fs::remove_all(root, ec);
    {
        SpillNamespace a, b2;
        std::string err;
        check(a.open(root, err) && a.is_open() && fs::is_directory(a.dir()), "the first engine's folder");
        touch(a.file_for(7));
        const fs::path dead = root / "4242-0123456789abcdef";
        fs::create_directories(dead);
        touch(dead / "conv-1.session");                               // a crashed engine's file, no lock file left
        const fs::path dead2 = root / "4243-fedcba9876543210";
        fs::create_directories(dead2);
        touch(dead2 / ".lock");                                       // a crashed engine's lock file, nobody holds it
        fs::create_directories(root / "keepme");
        touch(root / "keepme" / "conv-1.session");
        touch(root / "notes.txt");
        size_t cleaned = 99;
        check(b2.open(root, err, &cleaned), "a second engine on the same root");
        check(cleaned == 2 && !fs::exists(dead) && !fs::exists(dead2), "the two dead folders went");
        check(fs::exists(a.file_for(7)), "the first engine's live folder and file stay");
        check(fs::exists(root / "keepme" / "conv-1.session") && fs::exists(root / "notes.txt"), "nothing else under the root is touched");
        check(a.dir() != b2.dir() && a.file_for(7) != b2.file_for(7), "each engine writes its own files");
#if !defined(_WIN32)
        const fs::path outside = fs::temp_directory_path() / ("strata-spill-outside-" + std::to_string(std::random_device{}()));
        fs::create_directories(outside);
        touch(outside / "keep.txt");
        fs::create_directory_symlink(outside, root / "4244-0123456789abcdef", ec);
        check(!ec && SpillNamespace::clean_dead(root) == 0 && fs::exists(outside / "keep.txt"), "a link named like a folder is never followed");
        fs::remove(root / "4244-0123456789abcdef", ec);
        fs::remove_all(outside, ec);
#endif
        const fs::path adir = a.dir();
        a.close();
        check(!fs::exists(adir) && fs::exists(b2.dir()), "an orderly end removes its own folder only");
    }
    check(fs::exists(root) && !fs::exists(root / "4242-0123456789abcdef"), "both engines ended");
    std::string err;
    check(!SpillNamespace().open(root / "notes.txt" / "sub", err) && !err.empty(), "a root that cannot be a folder is refused with a reason");

    // the entries: written as session files, the oldest go first, matched like the RAM cache, read back exactly
    {
        auto img = [](std::vector<int32_t> ids, uint8_t fill, std::vector<int32_t> ckpt = {}) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 20, fill);                  // 1 MiB of running state
            s.cvec = true;
            if (!ckpt.empty()) { ConversationCheckpoint c; c.ids = ckpt; c.gdn.assign(4096, (uint8_t) (fill + 1)); s.checkpoints.push_back(c); }
            return s;
        };
        const SessionFileIdentity id{11, 22};
        ConversationSpill sp;
        // room for two entries of ~2 MiB each (1 MiB payload + the 1 MiB margin per write)
        check(sp.open(root, (uint64_t(7) << 19), err) && sp.enabled(), "the tier opens");
        check(sp.put(img({1, 2, 3, 4}, 7, {1, 2}), id, err) && sp.put(img({5, 6, 7}, 9), id, err) && sp.size() == 2, "two entries written");
        const fs::path first = sp.entry(0).file;
        check(fs::exists(first) && fs::exists(sp.entry(1).file), "their files exist");
        check(sp.bytes() == sp.entry(0).bytes + sp.entry(1).bytes && sp.budget().reserved == 0, "the budget counts what is written");
        check(sp.put(img({8, 9}, 3), id, err) && sp.size() == 2 && !fs::exists(first), "a third pushes out the oldest, its file deleted");
        auto m = sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, true);
        check(m.tokens == 3 && m.live && sp.entry(m.index).live.ids == std::vector<int32_t>({5, 6, 7}), "a prompt finds the entry it continues");
        check(sp.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a dropped entry is not found");
        check(sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, false).tokens == 0, "other steering: not found");
        SavedConversation back;
        const fs::path taken = sp.entry(m.index).file;
        check(sp.take(m.index, id, {}, back, err) && back.live.ids == std::vector<int32_t>({5, 6, 7}) &&
              back.live.gdn == std::vector<uint8_t>(1 << 20, 9), "taken back byte for byte");
        check(sp.size() == 1 && !fs::exists(taken) && sp.bytes() == sp.entry(0).bytes, "a taken entry leaves the tier");
        // a checkpoint match
        check(sp.put(img({20, 21, 22, 23}, 5, {20, 21}), id, err), "an entry with a checkpoint");
        auto mc = sp.best(std::vector<int32_t>{20, 21, 99}, {}, true);
        check(mc.tokens == 2 && !mc.live, "a prompt that shares only the checkpoint finds it");
        // another model's identity: refused, the entry goes, out unchanged
        SavedConversation keep = img({1}, 1);
        check(!sp.take(mc.index, SessionFileIdentity{11, 23}, {}, keep, err) && keep.live.ids == std::vector<int32_t>({1}), "another identity: refused");
        check(sp.size() == 1, "the refused entry is gone");
        // a cut file: refused by the bounded read
        {
            const fs::path f = sp.entry(0).file;
            fs::resize_file(f, fs::file_size(f) / 2, ec);
            SavedConversation x;
            check(!ec && !sp.take(0, id, {}, x, err) && sp.size() == 0 && !fs::exists(f), "a cut file is refused and removed");
        }
        // larger than the whole tier: refused, nothing written
        ConversationSpill small;
        std::string e2;
        check(small.open(root, 1 << 20, e2) && !small.put(img({1, 2}, 4), id, e2) && !e2.empty(), "an image larger than the tier is refused");
        size_t files = 0;
        for (auto& x : fs::directory_iterator(small.dir())) if (x.path().filename() != ".lock") ++files;
        check(files == 0, "nothing of a refused put is left on the disk");
        small.close();
        // #879: clear() drops every entry and its file, the tier stays open
        check(sp.put(img({30, 31}, 2), id, err) && sp.put(img({40, 41}, 2), id, err) && sp.size() == 2, "two more entries");
        const fs::path c0 = sp.entry(0).file, c1 = sp.entry(1).file;
        check(sp.clear() == 2 && sp.size() == 0 && sp.bytes() == 0 && !fs::exists(c0) && !fs::exists(c1), "clear() drops them and their files");
        check(sp.enabled() && fs::is_directory(sp.dir()) && sp.put(img({50}, 1), id, err) && sp.size() == 1, "the tier stays open after clear()");
        sp.close();
    }
    {   // the cost rule, with l620's measured speeds (a 236 MB image written in 120 ms, read back in 220 ms; 5 prompt tokens per ms)
        SpillCost c;
        check(c.reread_tokens_per_ms == 2.0 && c.round_trip_ms(2000000) == 4.0, "the defaults: 2 tokens per ms, 1 GB/s each way");
        for (int i = 0; i < 40; ++i) {
            c.saw_prompt(8000, 1600.0);
            c.saw_write(236000000, 120.0);
            c.saw_read(236000000, 220.0);
        }
        check(c.reread_tokens_per_ms > 4.99 && c.reread_tokens_per_ms < 5.01, "the prompt speed converges to the measured one");
        check(c.round_trip_ms(236000000) > 339.0 && c.round_trip_ms(236000000) < 341.0, "the round trip converges to 120 + 220 ms");
        check(!c.worth(51, 236000000), "a 51-token conversation is not worth the disk (re-read about 10 ms of tokens)");
        check(!c.worth(2000, 264000000), "a 2K one is not either here (400 ms against about 1.25 x 380 ms)");
        check(c.worth(8000, 350000000), "an 8K one is (1600 ms against about 1.25 x 504 ms)");
        const SpillCost before = c;
        c.saw_prompt(100, 1.0);       // fewer than kMinTokens: a request's fixed costs, not the prompt speed
        c.saw_prompt(4000, 0.0);      // no time: ignored
        c.saw_write(0, 5.0);          // nothing written: ignored
        c.saw_read(1000, -1.0);       // a clock that went back: ignored
        check(c.reread_tokens_per_ms == before.reread_tokens_per_ms && c.write_bytes_per_ms == before.write_bytes_per_ms &&
              c.read_bytes_per_ms == before.read_bytes_per_ms, "samples that say nothing about the speeds are ignored");
        SpillCost slow;   // a card that reads prompts at 1 token per ms: the same 2K conversation is worth spilling
        for (int i = 0; i < 40; ++i) { slow.saw_prompt(2000, 2000.0); slow.saw_write(264000000, 130.0); slow.saw_read(264000000, 250.0); }
        check(slow.worth(2000, 264000000), "on a slower card a 2K conversation is worth the disk");
        check(!c.worth(0, 1) && !slow.worth(0, 1), "an empty conversation never is");
    }
    {   // 2c: the background writer
        using namespace std::chrono_literals;
        struct Gate {   // holds the writer inside a write until opened
            std::mutex mu; std::condition_variable cv; bool open = false; int entered = 0;
            void enter() { std::unique_lock<std::mutex> lk(mu); ++entered; cv.notify_all(); cv.wait(lk, [&] { return open; }); }
            void release() { { std::lock_guard<std::mutex> lk(mu); open = true; } cv.notify_all(); }
            void wait_entered(int n) { std::unique_lock<std::mutex> lk(mu); cv.wait_for(lk, 5s, [&] { return entered >= n; }); }
        };
        Gate gate;
        std::vector<int> order;
        std::mutex order_mu;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t {
                gate.enter();
                std::lock_guard<std::mutex> lk(order_mu);
                order.push_back(v);
                return v == 3 ? 0 : (uint64_t) v * 10;   // item 3: a failed write
            });
            w.push(1);
            gate.wait_entered(1);
            w.push(2);   // one waits behind the one in flight
            std::atomic<bool> third_in{false};
            std::thread t([&] { w.push(3); third_in = true; });
            std::this_thread::sleep_for(150ms);
            check(!third_in.load(), "a third push waits while one is in flight and one is queued");
            int held = 0;
            w.for_each_held([&](const int&) { ++held; });
            check(held == 2 && !w.idle(), "the queued one and the one in flight are held");
            gate.release();
            t.join();
            w.wait_idle();
            check(third_in.load() && w.idle(), "the queue drains");
            auto done = w.take_done();
            check(done.size() == 3 && done[0].item == 1 && done[0].serial == 10 && done[1].item == 2 && done[1].serial == 20 &&
                  done[2].item == 3 && done[2].serial == 0, "written in order, each handed back with its serial (0: failed)");
            check(order == std::vector<int>({1, 2, 3}) && w.take_done().empty(), "handed back once");
        }
        Gate g2;
        std::vector<int> written;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t { g2.enter(); written.push_back(v); return 1; });
            w.push(7);
            g2.wait_entered(1);
            w.push(8);
            auto queued = w.take_queued();
            check(queued.size() == 1 && queued[0] == 8, "take_queued gives the queued one back unwritten");
            w.push(9);   // and the writer takes more after it
            g2.release();
        }   // the destructor: 7 finishes, 9 is not written (queued at the end)
        check(written == std::vector<int>({7}) || written == std::vector<int>({7, 9}),
              "the destructor finishes the write in flight and does not hang");
        // a SavedConversation through the real SSD tier, while the serve thread matches and takes by serial
        auto img = [](std::vector<int32_t> ids, uint8_t fill) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 18, fill);
            s.cvec = true;
            return s;
        };
        const SessionFileIdentity id{31, 41};
        ConversationSpill sp;
        std::string err;
        check(sp.open(root, uint64_t(64) << 20, err), "a tier for the writer");
        {
            BackgroundWriter<SavedConversation> w([&](const SavedConversation& s) -> uint64_t {
                uint64_t serial = 0;
                std::string e;
                return sp.put(s, id, e, &serial) ? serial : 0;
            });
            for (int k = 0; k < 6; ++k) w.push(img({k, k + 1, k + 2}, (uint8_t) k));
            for (int k = 0; k < 200; ++k) {   // the serve thread matching while the writer puts
                const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
                if (m.tokens > 0) (void) sp.find(m.serial);   // (may be gone or not yet listed: it must only not crash or race)
            }
            w.wait_idle();
            auto done = w.take_done();
            check(done.size() == 6 && sp.size() == 6, "six written, six listed");
            const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
            check(m.tokens == 3, "the entry a prompt continues is found");
            const auto e = sp.find(m.serial);
            check(e && e->live.ids == std::vector<int32_t>({2, 3, 4}), "found by its serial");
            SavedConversation back;
            check(sp.take_serial(m.serial, id, {}, back, err) && back.live.gdn == std::vector<uint8_t>(1 << 18, 2) &&
                  !sp.find(m.serial) && sp.size() == 5, "taken by its serial, byte for byte; it leaves the list");
            check(!sp.take_serial(m.serial, id, {}, back, err) && !sp.drop_serial(m.serial), "a gone serial: nothing");
            check(sp.drop_serial(done[0].serial) && sp.size() == 4, "dropped by its serial");
            // a written image still in RAM: the prefix it gives, by the RAM cache's rule
            const auto p = conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7, 1}, {}, true);
            check(p.first == 3 && p.second, "a held image's live prefix");
            check(conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7}, {}, false).first == 0, "other steering: none");
        }
        sp.close();
    }
    fs::remove_all(root, ec);
    std::printf("conversation_spill_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
