// #6048: after 0.2.14 the DeepSeek models of an earlier Strata (strata_control/retired.py) disappeared from Settings -> Storage, so their
// files (138 GB and 155 GB for the reporter) could no longer be seen or freed there; the Models page only says "no longer supported".
// Storage lists them again, with Control's own line (no new text), their size when it is known, and the usual Delete, which opens the same
// confirmation as any model (nothing is deleted without it). The files are never touched otherwise.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const storage = readFileSync(fileURLToPath(new URL('../src/components/SettingsStorage.tsx', import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('retired models are listed in Storage, counted, with Control\'s line and the usual Delete', () => {
  assert.match(storage, /const retiredModels = \(native\?\.installed\?\.models \?\? \[\]\)\.filter\(m => m\.retired\);/);
  assert.match(storage, /const rows = nativeModels\.length \+ retiredModels\.length \+ versions\.length;/, 'a PC with only DeepSeek files still shows the list');
  const row = storage.slice(storage.indexOf('{tr(retiredModels.map('), storage.indexOf('</li>))}', storage.indexOf('{tr(retiredModels.map(')));
  assert.ok(row.length > 0, 'a row per retired model');
  assert.match(row, /<small className="muted">\{m\.retired\?\.line\}<\/small>/, 'Control\'s own words, nothing new');
  assert.match(row, /\(m\.diskGb \?\? 0\) > 0 \? sizeGb\(m\.diskGb \?\? 0\) : ''/, 'a size only when it is known (never a made-up 0.0 GB)');
  assert.match(row, /onClick=\{\(\) => setNativeDeleting\(m\.model\)\}>\{tr\("Delete"\)\}/, 'Delete opens the same confirmation');
});

test('the models that run are listed as before, and retired ones stay out of that part', () => {
  assert.match(storage, /const nativeModels = \(native\?\.installed\?\.models \?\? \[\]\)\.filter\(m => !m\.retired && \(m\.ready \|\| \(m\.diskGb \?\? 0\) > 0\)\);/);
});
