"""#5940: on the Advanced step every cursor move re-derived the whole engine command (the card choice, the engine's backend and the GPU's
free memory: four nvidia-smi runs per key here, about 1 s per key on the reporter's Windows PC), and every edit ran the engine's --help again.
The command is worked out once per set of choices and the engine's list once per engine; an edit still shows its new value at once."""

from __future__ import annotations

from unittest import mock

from strata_cli.launch import runner
from strata_cli.launch.app import Ctx
from tests.terminal.test_launch_tui import Base, make_ctx, press, text_of, typed

HELP = "  --prefill N          prompt chunk size (default 2048)\n  --kv fp16|int8       KV storage\n  --native-rope        experimental pinned rotary\n"


class AdvancedLatency(Base):
    def reach(self):
        return ("enter", "enter", "enter", "enter", "enter", "enter", "enter")

    def ctx_with_engine(self):
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "qwen", "IQ3_XXS"
        base = {"exe": "component:strata-engine", "args": ["--max-context", "32768", "--kv", "int8", "--prefill", "auto"]}
        self.help = mock.Mock(return_value=HELP)
        self.derive = mock.Mock(side_effect=runner.derive_config)
        for p in (mock.patch.object(Ctx, "base_config", lambda self: base),
                  mock.patch.object(Ctx, "engine_exe", lambda self: "/engine/strata"),
                  mock.patch("strata_cli.launch.params.engine_help_text", self.help),
                  mock.patch.object(runner, "derive_config", self.derive)):
            p.start()
            self.addCleanup(p.stop)
        return ctx

    def test_cursor_moves_do_not_work_out_the_command_again(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            self.assertEqual(app.current, "advanced")
            before = self.derive.call_count
            self.assertGreaterEqual(before, 1)
            await press(pilot, "down", "down", "down", "up", "down", "end", "home", pause=0.05)
            self.assertEqual(self.derive.call_count, before)
            self.assertIn("Command that will run", text_of(app))
        self.run_app(case, ctx=ctx)

    def test_an_edit_shows_the_new_value_and_reads_the_engine_list_once(self):
        ctx = self.ctx_with_engine()

        async def case(app, pilot, ctx):
            await press(pilot, *self.reach())
            names = [it.key for it in app.screen.menu().items]
            for _ in range(names.index("--prefill")):
                await press(pilot, "down", pause=0.05)
            derived = self.derive.call_count
            await press(pilot, "e")
            await typed(pilot, "65536")
            await press(pilot, "enter")
            self.assertEqual(ctx.settings.extra, {"--prefill": "65536"})
            self.assertGreater(self.derive.call_count, derived)          # the choices changed: the command is worked out again
            self.assertIn("--prefill 65536", text_of(app).replace("\n", " "))
            await press(pilot, "delete")
            self.assertEqual(ctx.settings.extra, {})
            self.assertEqual(self.help.call_count, 1)                    # the same engine: its list is read once
        self.run_app(case, ctx=ctx)
