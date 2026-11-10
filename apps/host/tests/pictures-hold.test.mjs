/**
 * QAGUI-STRIX 11-09 finding 4: "Read pictures" unticked and loaded did not hold across an app restart on a model set up without the picture files:
 * the box came back ticked and Load fetched the 866 MB picture reader. The off was saved (ui_vision_enabled false), but the page honoured a saved off
 * only when the picture files were prepared (a guard for the off 0.2.9 saved while the engine had no reader). A choice made with a reader present now
 * carries a marker (ui_vision_chosen) and holds; an old off without the marker behaves exactly as before.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { openingVision } from '../src/lib/pictures-choice.ts';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

test('the page: an off with the marker holds without the picture files; without the marker nothing changed', () => {
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: false, visionChosen: true, images: 'off' }), 'no', 'the Strix case');
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: false, images: 'off' }), 'yes', 'an old off (no marker, no files): on, as before');
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: false, visionChosen: false, images: 'off' }), 'yes');
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: true }), 'no', 'off with the files: off, as before');
  assert.equal(openingVision({ visionEnabled: true, visionPrepared: false, visionChosen: true, images: 'off' }), 'yes', 'on stays on');
  assert.equal(openingVision({ visionEnabled: true, visionPrepared: true, visionChosen: true, visionDevice: 'cpu' }), 'cpu');
  assert.equal(openingVision(undefined), 'yes');
});

test('the host: a load with Read pictures off on a model without picture files saves the off with the marker, and reports it after a restart', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const path = join(f.repo, 'strata-iq2_xs.json');
    const { vision, ...noFiles } = JSON.parse(readFileSync(path, 'utf8'));
    assert.ok(vision, 'the fixture had picture files');
    writeFileSync(path, JSON.stringify(noFiles));                       // set up without the picture files (the Strix model)
    await f.control.refresh(true);
    const row = () => f.control.state().installed.models.find(m => m.model === 'iq2_xs');
    assert.equal(row().visionPrepared, false);
    assert.equal(row().visionChosen, false, 'no marker before a choice');
    const off = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', vision: 'no' }).id);
    assert.equal(off.state, 'done', off.error);
    const saved = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(saved.ui_vision_enabled, false);
    assert.equal(saved.ui_vision_chosen, true, 'the marker is saved with the choice');
    await f.control.refresh(true);
    assert.equal(row().visionChosen, true);
    assert.equal(openingVision(row()), 'no', 'after a restart the box opens unticked');
  } finally { await f.cleanup(); }
});

test('the host: a load while the engine has no reader still saves nothing, the marker neither; an old file without the marker reads as before', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh();
    const path = join(f.repo, 'strata-iq2_xs.json');
    const none = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', vision: 'no', visionUnavailable: true }).id);
    assert.equal(none.state, 'done', none.error);
    const saved = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(saved.ui_vision_enabled, undefined);
    assert.equal(saved.ui_vision_chosen, undefined);
    const { vision, ...noFiles } = saved;
    writeFileSync(path, JSON.stringify({ ...noFiles, ui_vision_enabled: false, ui_vision_device: 'yes' }));   // a 0.2.9 file: off, no files, no marker
    await f.control.refresh(true);
    const row = f.control.state().installed.models.find(m => m.model === 'iq2_xs');
    assert.equal(row.visionChosen, false);
    assert.equal(openingVision(row), 'yes');
  } finally { await f.cleanup(); }
});
