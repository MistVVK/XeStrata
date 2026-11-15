"""#5812: on the launch's Access step, moving the highlight past "This computer only" made an API key, and nothing took it back when the
highlight went back: a launch that ended on this computer served with a key while the screen said none is needed there."""

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from strata_cli.launch import screens
except ImportError:                                   # the optional Textual dependency is absent
    screens = None


class FakeAccess:
    """The parts of the Access screen its key logic uses: the highlighted choice, the launch settings and the status line."""

    def __init__(self, api_key=None, no_key=False):
        self.ctx = SimpleNamespace(settings=SimpleNamespace(access="local", api_key=api_key, no_key=no_key))
        self.key = "local"
        self.said = []

    def menu(self):
        return SimpleNamespace(current=SimpleNamespace(key=self.key))

    def say(self, text, warn=False):
        self.said.append(text)

    def drop_cursor_key(self):
        return screens.AccessScreen.drop_cursor_key(self)

    def build(self):
        pass

    def paint(self):
        pass


@unittest.skipIf(screens is None, "optional Textual dependency absent")
class CursorKey(unittest.TestCase):
    def setUp(self):
        for name, value in (("kept_key", None), ("new_key", "sk-made-by-the-cursor")):
            patch = mock.patch.object(screens.acc, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def move(self, screen, kind):
        screen.key = kind
        screens.AccessScreen.on_cursor(screen)

    def test_reading_the_choices_and_ending_on_this_computer_leaves_no_key(self):
        screen = FakeAccess()
        screens.AccessScreen.on_mount(screen)
        self.move(screen, "lan")
        self.assertEqual(screen.ctx.settings.api_key, "sk-made-by-the-cursor")      # beyond this computer: a key, as before
        self.move(screen, "tailscale")
        self.move(screen, "local")
        self.assertIsNone(screen.ctx.settings.api_key)
        screens.AccessScreen.commit(screen)
        self.assertEqual((screen.ctx.settings.access, screen.ctx.settings.api_key), ("local", None))

    def test_a_key_the_launch_already_had_stays(self):
        screen = FakeAccess(api_key="sk-given-with-api-key")
        screens.AccessScreen.on_mount(screen)
        self.move(screen, "lan")
        self.move(screen, "local")
        screens.AccessScreen.commit(screen)
        self.assertEqual(screen.ctx.settings.api_key, "sk-given-with-api-key")

    def test_a_key_the_person_made_on_the_step_stays(self):
        screen = FakeAccess()
        screens.AccessScreen.on_mount(screen)
        self.move(screen, "lan")
        with mock.patch.object(screens.acc, "new_key", return_value="sk-pressed-k"):
            screens.AccessScreen.action_newkey(screen)
        self.move(screen, "local")
        screens.AccessScreen.commit(screen)
        self.assertEqual(screen.ctx.settings.api_key, "sk-pressed-k")

    def test_a_launch_beyond_this_computer_still_gets_its_key(self):
        screen = FakeAccess()
        screens.AccessScreen.on_mount(screen)
        self.move(screen, "lan")
        self.move(screen, "local")
        self.move(screen, "lan")
        screens.AccessScreen.commit(screen)
        self.assertEqual((screen.ctx.settings.access, screen.ctx.settings.api_key), ("lan", "sk-made-by-the-cursor"))


if __name__ == "__main__":
    unittest.main()
