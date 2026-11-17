// #5996: a helper's row says so once it has worked more than 30 minutes (whatever the time limit), and shows the host's "stopped by the limit" sentence.
// The real SubagentRow is rendered (react-dom/server) with the page's buttons and client stubbed out.
import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { build } from 'esbuild';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';

const root = resolve(import.meta.dirname, '..');
const entry = [
  "import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server';",
  "import { SubagentRow } from './src/components/SubagentRow.tsx';",
  'export const html = (agent, now) => renderToStaticMarkup(React.createElement(SubagentRow, { agent, now }));',
].join('\n');
const stubs = { './ui': 'export const Btn = () => null; export const Dialog = () => null; export const Icon = () => null;', '../lib/client': 'export const api = {};' };
const bundle = await build({
  stdin: { contents: entry, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'esm', jsx: 'automatic', external: ['react', 'react-dom', 'react-dom/server', 'react/jsx-runtime'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^(\.\/ui|\.\.\/lib\/client)$/ }, args => ({ path: args.path, namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, args => ({ contents: stubs[args.path], loader: 'js' }));
  } }],
});
const dir = mkdtempSync(resolve(root, 'node_modules', '.guie1117-row-'));
const file = resolve(dir, 'bundle.mjs');
writeFileSync(file, bundle.outputFiles[0].text);
const { html } = await import(`file://${file.replaceAll('\\', '/')}`);
test.after(() => rmSync(dir, { recursive: true, force: true }));

const NOTE = 'This helper has been working for over 30 minutes.';
const row = (status, durationMs, extra = {}) => ({ id: 'h', name: 'Check the code', description: 'Reading files', status, durationMs, tokens: 100, updatedAt: 1000, ...extra });

test('a helper that is working shows the note after 30 minutes and not before; it is not shown for one that finished', () => {
  assert.ok(!html(row('working', 29 * 60_000), 1000).includes(NOTE));
  assert.ok(!html(row('working', 30 * 60_000), 1000).includes(NOTE), 'exactly 30 minutes: not yet');
  assert.ok(html(row('working', 30 * 60_000 + 1000), 1000).includes(NOTE));
  assert.ok(html(row('working', 20 * 60_000), 1000 + 11 * 60_000).includes(NOTE), 'the time since the last frame counts while it works');
  assert.ok(!html(row('completed', 5 * 3600_000), 1000).includes(NOTE));
  assert.ok(!html(row('cancelled', 5 * 3600_000), 1000).includes(NOTE));
  assert.ok(!html(row('queued', 0), 1000).includes(NOTE));
});

test('the sentence the host gives a helper the limit stopped is on its row', () => {
  const out = html(row('cancelled', 31 * 60_000, { description: 'Stopped: it reached the helper time limit.' }), 1000);
  assert.ok(out.includes('Stopped: it reached the helper time limit.'));
});
