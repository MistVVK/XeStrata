import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {resolve, join} from 'node:path';

const root = resolve(import.meta.dirname, '..');
const read = (p) => readFileSync(resolve(root, p), 'utf8');

test('the window title is "Strata" on every page', () => {
  assert.match(read('index.html'), /<title>Strata<\/title>/);
  assert.match(read('server/index.mjs'), /<title>Strata<\/title>/);
  assert.match(read('../desktop/src-tauri/src/main.rs'), /WebviewWindowBuilder::new\(app, "main"[^\n]*\.title\("Strata"\)/);
});

test('no screen sets document.title to anything else', () => {
  const hits = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const full = join(dir, name);
      if (statSync(full).isDirectory()) walk(full);
      else if (/\.(ts|tsx)$/.test(name) && /document\.title\s*=/.test(readFileSync(full, 'utf8'))) hits.push(full);
    }
  };
  walk(resolve(root, 'src'));
  assert.deepEqual(hits, []);
});
