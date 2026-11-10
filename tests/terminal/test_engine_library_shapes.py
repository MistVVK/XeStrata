"""Settings > Engine and `strata components engines | use | remove` for the engine of a Qwen4 model shape (APP-WIN item 6, owner 11-08 "later, after 0.2.13").

A shape's engine lives in <components>/shapes/<shape>/engine/ (strata_control/shape_engine.py). The list shows one family per installed shape ("shape:<shape>",
an id the app never shows; the title is None and the app names the group with the model's catalog title), after the Qwen and DeepSeek families. Switching
pins within the shape's own folder (the same ``.updates/state.json`` engine.pinned the Qwen engine uses, which engine_component.resolve reads at every start of
a model of that shape); the newest goes back to no pin. Deleting refuses the one in use and one a running program uses with the existing lines. The Strata
engine's files and state are never touched, and a PC with no shape engine lists exactly what it did before.
"""
from __future__ import annotations

import io
import json
import contextlib
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec, engine_library as lib, shape_engine, signing, updater
from tests.terminal.test_shape_engine import SHAPE, Base, tree

FAMILY = "shape:" + SHAPE


class ShapeEngines(Base):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(lib, "_process_uses", return_value=False), mock.patch.object(lib, "_ds_entries", return_value=[])):
            p.start()
            self.addCleanup(p.stop)

    def ctx(self):
        return updater.Context(components=self.comp, source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw, serving=lambda: [], poll=0.05)

    def install(self, *versions):
        for i in range(len(versions)):
            self.publish(*versions[: i + 1], sequence=i + 1)
            with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
                self.assertIsNotNone(self.ensure())

    def rows(self):
        return {e["version"]: e for e in lib.rows_of(lib.list_engines(self.ctx()), FAMILY)}

    def picked(self):
        return ec.resolve(shape_engine.root(SHAPE, self.comp), backend="cuda", component_id=shape_engine.component_id(SHAPE)).version

    def test_no_shape_engine_lists_what_it_did_before(self):
        listing = lib.list_engines(self.ctx())
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen"])
        self.assertEqual(set(listing["families"][0]), {"family", "title", "engines"})

    def test_an_installed_shape_engine_is_its_own_family_after_the_strata_engine(self):
        self.install("0.1.59", "0.1.60")
        listing = lib.list_engines(self.ctx())
        self.assertEqual([f["family"] for f in listing["families"]], ["qwen", FAMILY])
        fam = listing["families"][1]
        self.assertEqual((fam["shape"], fam["title"]), (SHAPE, None), "the app names the group from the catalog; the family id is not a title")
        rows = self.rows()
        self.assertEqual(sorted(rows), ["0.1.59", "0.1.60"])
        self.assertTrue(rows["0.1.60"]["active"] and rows["0.1.60"]["newest"])
        self.assertTrue(rows["0.1.59"]["can_use"] and rows["0.1.59"]["can_delete"])
        self.assertEqual(rows["0.1.60"]["delete_blocked"], "This is the engine in use. Switch to another one first.")
        self.assertEqual({r["family"] for r in rows.values()}, {FAMILY})
        self.assertEqual(listing["total_bytes"], sum(r["bytes"] for f in listing["families"] for r in f["engines"]))
        qwen = lib.rows_of(listing, lib.QWEN)
        self.assertEqual([r["version"] for r in qwen], ["0.1.58"], "the Strata engine is listed as before, alone in its family")

    def test_switch_to_the_older_one_and_back_to_the_newest(self):
        self.install("0.1.59", "0.1.60")
        old = self.rows()["0.1.59"]["folder"]
        out = lib.use_engine(self.ctx(), FAMILY, old)
        self.assertEqual((out["status"], out["from"], out["to"], out["automatic"]), ("switched", "0.1.60", "0.1.59", False))
        self.assertEqual(self.picked(), "0.1.59")
        self.assertEqual(ec.pinned_folder(shape_engine.root(SHAPE, self.comp)), old)
        self.assertIsNone(ec.pinned_folder(self.comp), "the Strata engine's pin is not touched")
        self.assertTrue(self.rows()["0.1.59"]["active"], "a fresh read (a restart) sees it")
        back = lib.use_engine(self.ctx(), FAMILY, self.rows()["0.1.60"]["folder"])
        self.assertEqual((back["status"], back["automatic"]), ("switched", True))
        self.assertIsNone(ec.pinned_folder(shape_engine.root(SHAPE, self.comp)), "the newest follows updates again: no pin")
        self.assertEqual(lib.use_engine(self.ctx(), FAMILY, self.rows()["0.1.60"]["folder"])["status"], "already")
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_delete_the_older_one_and_never_the_one_in_use_or_one_a_program_uses(self):
        self.install("0.1.59", "0.1.60")
        rows = self.rows()
        with self.assertRaises(updater.UpdateError) as e:
            lib.delete_engine(self.ctx(), FAMILY, rows["0.1.60"]["folder"])
        self.assertEqual(str(e.exception), "This is the engine in use. Switch to another one first.")
        with mock.patch.object(lib, "_process_uses", side_effect=lambda folder: folder.name == rows["0.1.59"]["folder"]):
            with self.assertRaises(updater.UpdateError) as e:
                lib.delete_engine(self.ctx(), FAMILY, rows["0.1.59"]["folder"])
        self.assertEqual(str(e.exception), "A running program is using it. Stop the model first.")
        out = lib.delete_engine(self.ctx(), FAMILY, rows["0.1.59"]["folder"])
        self.assertEqual((out["status"], out["version"], out["component"]), ("deleted", "0.1.59", shape_engine.component_id(SHAPE)))
        self.assertEqual(sorted(self.rows()), ["0.1.60"])
        self.assertEqual(self.picked(), "0.1.60")
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_a_deleted_pinned_engine_leaves_no_pin(self):
        self.install("0.1.59", "0.1.60")
        old = self.rows()["0.1.59"]["folder"]
        lib.use_engine(self.ctx(), FAMILY, old)
        lib.use_engine(self.ctx(), FAMILY, self.rows()["0.1.60"]["folder"])
        state = updater.State(shape_engine.root(SHAPE, self.comp))
        state.section("engine")["pinned"] = old                 # as if a pin was left behind
        state.save()
        self.assertEqual(self.picked(), "0.1.59")
        lib.use_engine(self.ctx(), FAMILY, self.rows()["0.1.60"]["folder"])
        lib.delete_engine(self.ctx(), FAMILY, old)
        self.assertIsNone(ec.pinned_folder(shape_engine.root(SHAPE, self.comp)))

    def test_a_family_that_is_not_a_shape_is_refused(self):
        for bad in ("shape:", "shape:X", "shape:engine", "shapes:q4_lite", "strata-q4-lite"):
            with self.assertRaises(updater.UpdateError):
                lib.use_engine(self.ctx(), bad, "0.1.59-x")
        self.assertEqual(lib.shape_of_family(FAMILY), SHAPE)
        self.assertIsNone(lib.shape_of_family("qwen"))

    def test_the_terminal_lists_the_shape_by_name_and_takes_the_family_the_app_passes(self):
        from strata_cli import update_cmd
        self.assertIn(FAMILY, update_cmd._ENGINE_FAMILIES)
        self.assertIn("qwen", update_cmd._ENGINE_FAMILIES)
        self.assertNotIn("shape:Nope", update_cmd._ENGINE_FAMILIES)
        self.assertEqual(list(update_cmd._ENGINE_FAMILIES), ["qwen", "deepseek"], "help and errors list the two as before")
        self.install("0.1.59")
        a = mock.Mock(components_command="engines")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            update_cmd._engines_command(a, self.ctx(), False, None)
        text = out.getvalue()
        self.assertIn("\n" + SHAPE + "\n", text)
        self.assertNotIn(FAMILY, text)
        self.assertNotIn("None", text)
