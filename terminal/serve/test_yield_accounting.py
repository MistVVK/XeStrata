"""Real serving flow over a fake native engine with accurate partial-read DONE fields."""

import threading, time, unittest
from unittest import mock
from serve import test_parallel as tp


def accurate_fake():
    source = tp.FAKE_BATCH
    source = source.replace("def reused(ids):", "partial = {}\ndef reused(ids):", 1)
    old = "    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3"
    assert source.count(old) == 1
    source = source.replace(
        old,
        """    if not REUSE: return 0
    kept = max((len(h) for h in partial.values() if ids[:len(h)] == h), default=0)
    return max(kept, len(ids) - 1 if continued(ids)[1] else 3)""",
    )
    old = "        given = None\n        for q in range(CH, len(ids) - 1, CH):"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "        read_from = reused(ids)\n        given = None\n        for q in range((read_from // CH + 1) * CH, len(ids) - 1, CH):",
    )
    old = '            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {reused(ids)}", flush=True)'
    assert source.count(old) == 1
    source = source.replace(
        old,
        """            partial[given[0]] = ids[:given[1]]
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {read_from}" +
                  (f" 0 0 0 0 0.0 {given[1] - read_from}" if READF else ""), flush=True)""",
    )
    source = source.replace("{len(ids) - reused(ids)}", "{len(ids) - read_from}")
    source = source.replace("{reused(ids)}", "{read_from}")
    compile(source, "accurate_fake", "exec")
    return source


class YieldAccounting(tp.FakeEngineCase):
    def start_accurate(self, slots):
        with mock.patch.object(tp, "FAKE_BATCH", accurate_fake()):
            self.start(slots, reuse=True, more=["--readfield"])

    def race_long(self):
        replies = []
        errors = []

        def long():
            try:
                replies.append(self.chat("long " * 600, max_tokens=64))
            except Exception as error:
                errors.append(repr(error))

        thread = threading.Thread(target=long)
        thread.start()
        time.sleep(0.15)
        self.chat("short", max_tokens=64)
        thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(replies), 1)
        return replies[0]

    def check_full_input(self, response):
        n = response["usage"]["prompt_tokens"]
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        legs = [x for x in log if x.startswith(("GEN ", "BGEN ")) and int(x.split()[2]) == n]
        self.assertGreaterEqual(len(legs), 2)
        self.assertEqual(response["usage"]["prompt_tokens_details"]["cached_tokens"], 3)
        self.assertEqual((response["timings"]["cache_n"], response["timings"]["prompt_n"]), (3, n - 3))
        self.assertEqual(response["timings"]["prompt_ms"], 5.0 * len(legs))
        row = max(self.svc.history, key=lambda x: x["prompt_tokens"])
        self.assertEqual((row["reused"], row["prompt_read"]), (3, n - 3))

    def test_solo_yield_keeps_original_cache_and_sums_prompt_work(self):
        self.start_accurate(2)
        self.check_full_input(self.race_long())

    def test_admission_yield_keeps_original_cache_and_sums_prompt_work(self):
        self.start_accurate(3)
        errors = []

        def background():
            try:
                self.chat("LONGREPLY background", max_tokens=60)
            except Exception as error:
                errors.append(repr(error))

        bg = threading.Thread(target=background)
        bg.start()
        time.sleep(0.05)
        response = self.race_long()
        bg.join(20)
        self.assertFalse(bg.is_alive())
        self.assertEqual(errors, [])
        self.check_full_input(response)

    def cancel_after_yields(self, count):
        self.start_accurate(2)
        original = self.engine._take_control
        seen = 0

        def take(cancel, plen, after_epoch=None, **kwargs):
            nonlocal seen
            if after_epoch is not None:
                seen += 1
                if seen == count:
                    cancel.set()
                    return False
                return (yield from original(cancel, plen, **kwargs))
            return (yield from original(cancel, plen, **kwargs))

        with mock.patch.object(self.engine, "_shorter_waiting", return_value=True), mock.patch.object(
            self.engine, "_take_control", side_effect=take
        ):
            self.chat("long " * 600, max_tokens=64)
        positions = [int(x.split()[2]) for x in self.log.read_text().splitlines() if x.startswith("YIELD ")]
        self.assertEqual(len(positions), count)
        self.assertEqual(positions, sorted(set(positions)))
        row = self.svc.history[-1]
        self.assertEqual(row["finish"], "cancel")
        self.assertEqual(
            (row["prompt_tokens"], row["reused"], row["prompt_read"]), (positions[-1], 3, positions[-1] - 3)
        )
        self.assertEqual(row["prompt_ms"], 5.0 * count)

    def test_cancel_after_one_yield_counts_that_done_once(self):
        self.cancel_after_yields(1)

    def test_cancel_after_two_yields_counts_each_done_once(self):
        self.cancel_after_yields(2)
