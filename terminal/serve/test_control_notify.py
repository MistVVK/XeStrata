"""Control fairness notifications, without a process, GPU or wall-clock sleeps."""
import threading
import unittest
from unittest import mock
from types import SimpleNamespace
from serve import server
from serve.server import EngineDied, QueueWaitExpired, StrataEngine


class Condition(threading.Condition):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.exits = 0
        self.after_check = None
        self.on_wait = None
        self.waits = []
        self.notifications = 0

    def __exit__(self, *args):
        result = super().__exit__(*args)
        self.exits += 1
        # The first exit follows registration; the second follows the fairness
        # predicate read. Inject the producer after that lock was released.
        if self.exits == 2 and self.after_check is not None:
            callback, self.after_check = self.after_check, None
            callback()
        return result

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.on_wait:
            self.on_wait()
        return True

    def notify_all(self):
        self.notifications += 1
        return super().notify_all()


class Control:
    def __init__(self, owner):
        self.owner = owner
        self.held = False
        self.on_acquire = None

    def acquire(self, timeout=None):
        if self.owner.slot_cv._is_owned():
            raise AssertionError("blocking control acquire holds slot_cv")
        if self.on_acquire:
            return self.on_acquire()
        self.held = True
        return True

    def release(self):
        self.held = False


def owner():
    e = SimpleNamespace(waiting=1, wait_lens=[], arrivals=[], ctl_epoch=0, starting=False, live=True)
    e.alive = lambda: e.live
    e._admit_next = lambda: StrataEngine._admit_next(e)
    e.slot_cv = Condition(e)
    e.ctl = Control(e)
    return e


def finish(generator):
    try:
        next(generator)
    except StopIteration as result:
        return result.value
    raise AssertionError("unexpected heartbeat")


class ControlNotify(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(server, "ADMIT_SHORT_FIRST", 0)
        patch.start()
        self.addCleanup(patch.stop)

    def test_epoch_notify_between_check_and_wait_does_not_wait(self):
        e = owner()
        def advance():
            with e.slot_cv:
                e.ctl_epoch += 1
                e.slot_cv.notify_all()
        e.slot_cv.after_check = advance
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 10, after_epoch=0)))
        self.assertEqual(e.slot_cv.waits, [], "eligible waiter lost its notification and entered timeout wait")
        self.assertTrue(e.ctl.held)
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])

    def test_cancel_removal_notifies_remaining_fairness_waiter(self):
        e, cancel = owner(), threading.Event()
        e.slot_cv.on_wait = cancel.set
        self.assertFalse(finish(StrataEngine._take_control(e, cancel, 10, after_epoch=0)))
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])
        self.assertGreater(e.slot_cv.notifications, 0)
        self.assertFalse(e.ctl.held)

    def test_fairness_waits_until_epoch_advances(self):
        e = owner()
        e.slot_cv.on_wait = lambda: setattr(e, "ctl_epoch", 1)
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 10, after_epoch=0)))
        self.assertEqual(e.slot_cv.waits, [0.5])
        self.assertEqual(e.ctl_epoch, 2)

    def test_dead_engine_cleans_registration(self):
        e = owner()
        e.live = False
        with self.assertRaises(EngineDied):
            finish(StrataEngine._take_control(e, threading.Event(), 10))
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])
        self.assertFalse(e.ctl.held)

    def test_restart_wait_does_not_spin_or_hold_control(self):
        e = owner()
        e.live, e.starting = False, True
        e.slot_cv.on_wait = lambda: setattr(e, "live", True)
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 10)))
        self.assertEqual(e.slot_cv.waits, [0.5])
        self.assertTrue(e.ctl.held)

    def test_control_timeout_observes_cancel(self):
        e, cancel = owner(), threading.Event()
        def timeout():
            cancel.set()
            return False
        e.ctl.on_acquire = timeout
        self.assertFalse(finish(StrataEngine._take_control(e, cancel, 10)))
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])

    def test_closing_heartbeat_generator_cleans_and_notifies(self):
        e = owner()
        g = StrataEngine._take_control(e, threading.Event(), 10, after_epoch=0)
        with mock.patch("serve.server.time.monotonic", side_effect=[0, 0, 11, 11]):
            self.assertIsNone(next(g))
            g.close()
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])
        self.assertGreater(e.slot_cv.notifications, 0)

    def test_uncontended_acquire_never_waits(self):
        e = owner()
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 10)))
        self.assertEqual(e.slot_cv.waits, [])
        self.assertTrue(e.ctl.held)


class OrderedControlNotify(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.multiple(server, ADMIT_SHORT_FIRST=4, ADMIT_MAX_WAIT_S=10**12)
        patch.start()
        self.addCleanup(patch.stop)

    def other(self, e, length=1):
        entry = [length, 0, True]
        e.wait_lens.append(entry)
        return entry

    def remove(self, e, entry):
        with e.slot_cv:
            e.wait_lens.remove(entry)
            e.waiting -= 1
            e.slot_cv.notify_all()

    def test_shorter_removal_between_check_and_wait_rechecks_order(self):
        e = owner()
        shorter = self.other(e)
        e.slot_cv.after_check = lambda: self.remove(e, shorter)
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 100, new=True)))
        self.assertEqual(e.slot_cv.waits, [])
        self.assertEqual(e.waiting, 0)

    def test_still_shorter_waiter_is_not_bypassed_by_recheck(self):
        e = owner()
        shorter = self.other(e)
        e.slot_cv.on_wait = lambda: self.remove(e, shorter)
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 100, new=True)))
        self.assertEqual(e.slot_cv.waits, [0.5])

    def test_shorter_arriving_during_acquire_keeps_priority(self):
        e = owner()
        calls = []
        def acquire():
            calls.append(1)
            e.ctl.held = True
            if len(calls) == 1:
                e.waiting += 1
                shorter = self.other(e)
                e.slot_cv.on_wait = lambda: self.remove(e, shorter)
            return True
        e.ctl.on_acquire = acquire
        self.assertTrue(finish(StrataEngine._take_control(e, threading.Event(), 100, new=True)))
        self.assertEqual(len(calls), 2)
        self.assertEqual(e.slot_cv.waits, [0.5])
        self.assertTrue(e.ctl.held)

    def test_three_waiters_cancel_notifies_new_ordered_winner(self):
        e, cancel = owner(), threading.Event()
        e.waiting = 2
        self.other(e, 100)
        self.other(e, 200)
        e.slot_cv.on_wait = cancel.set
        self.assertFalse(finish(StrataEngine._take_control(e, cancel, 1, after_epoch=0, new=True)))
        self.assertEqual(e.waiting, 2)
        self.assertEqual(len(e.wait_lens), 2)
        self.assertGreater(e.slot_cv.notifications, 0)

    def test_three_waiters_generator_close_notifies_new_ordered_winner(self):
        e = owner()
        e.waiting = 2
        self.other(e, 100)
        self.other(e, 200)
        g = StrataEngine._take_control(e, threading.Event(), 1, after_epoch=0, new=True)
        with mock.patch("serve.server.time.monotonic", side_effect=[0, 0, 0, 11, 11]):
            self.assertIsNone(next(g))
            g.close()
        self.assertEqual(e.waiting, 2)
        self.assertGreater(e.slot_cv.notifications, 0)

    def test_expired_queue_deadline_cleans_without_acquire(self):
        e, cancel = owner(), threading.Event()
        cancel._strata_queue_deadline = 1
        with mock.patch("serve.server.time.monotonic", return_value=2):
            with self.assertRaises(QueueWaitExpired):
                finish(StrataEngine._take_control(e, cancel, 100, new=True))
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])
        self.assertFalse(e.ctl.held)
        self.assertEqual(e.slot_cv.waits, [])

    def test_queue_deadline_after_acquire_releases_control(self):
        e, cancel = owner(), threading.Event()
        cancel._strata_queue_deadline = 1
        now = [0]
        def acquire():
            e.ctl.held = True
            now[0] = 2
            return True
        e.ctl.on_acquire = acquire
        with mock.patch("serve.server.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaises(QueueWaitExpired):
                finish(StrataEngine._take_control(e, cancel, 100, new=True))
        self.assertEqual(e.waiting, 1)
        self.assertEqual(e.wait_lens, [])
        self.assertFalse(e.ctl.held)


if __name__ == "__main__":
    unittest.main()
