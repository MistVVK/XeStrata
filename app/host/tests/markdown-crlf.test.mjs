// Stale-preview OOM (QAGUI final pass): once the sidebar showed the file's CURRENT content, a file written on Windows (the agent's
// `Add-Content` writes \r\n) reached the Markdown renderer with a \r at the end of every line. `.` does not match \r, so a list line passed the
// list start test but not the item pattern, the list loop took no line and the outer loop never advanced: an endless loop pushing nodes, 150 MB to
// 4.4 GB in 20 s on a 60 byte file. 0.2.10 never showed the changed content, so it never met this. The render runs in a child process with a
// memory cap and a time limit, so a regression fails the test instead of hanging the suite.
import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
// the text arrives as JSON in a file named by argv[2], so no shell or platform touches its line ends (and a long text is not cut by
// the command-line length limit: Windows allows about 32 KB, which the 1500-section case below passes)
const entry = [
  "import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server';",
  "import { Markdown } from './src/components/Markdown.tsx'; import { splitPreview } from './src/lib/preview-render.ts';",
  "import { readFileSync } from 'node:fs';",
  "const text = JSON.parse(readFileSync(process.argv[2], 'utf8'));",
  'console.log(JSON.stringify({ html: renderToStaticMarkup(React.createElement(Markdown, { text })), slices: splitPreview(text).map(s => s.text) }));',
].join('\n');
const bundle = await build({
  stdin: { contents: entry, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null; export const Icon = () => null;' }));
  } }],
});
const dir = mkdtempSync(join(tmpdir(), 'strata-crlf-'));
const file = join(dir, 'render.cjs');
writeFileSync(file, bundle.outputFiles[0].text);
let renders = 0;
function render(text) {
  const input = join(dir, `text-${renders++}.json`);
  writeFileSync(input, JSON.stringify(text));
  const run = spawnSync(process.execPath, ['--max-old-space-size=200', file, input], { cwd: root, encoding: 'utf8', timeout: 20000, maxBuffer: 20 * 1024 * 1024 });
  return { ok: run.status === 0 && !run.error, out: run.stdout, err: (run.stderr || '') + (run.error?.message ?? ''), status: run.status, signal: run.signal };
}

test('QAGUI case: the file with a CRLF bullet renders (it used to run the page out of memory)', () => {
  const r = render('# Version Two\r\n\r\n- the file was rewritten on disk\r\n- appended\r\n');
  assert.ok(r.ok, `the render did not finish: status ${r.status} signal ${r.signal} ${r.err.slice(0, 200)}`);
  const { html } = JSON.parse(r.out);
  assert.equal((html.match(/<li>/g) ?? []).length, 2);
  assert.ok(html.includes('appended') && html.includes('Version Two'));
  assert.ok(!html.includes('\r'));
});

test('CRLF, CR and LF give the same page, for lists, ordered lists, tables, quotes, code and paragraphs', () => {
  const lf = '# T\n\n- a\n- b\n\n1. x\n2. y\n\n| h | i |\n| --- | --- |\n| 1 | 2 |\n\n> quote\n\n```js\nconst a = 1;\n```\n\nline one\nline two\n';
  const want = JSON.parse(render(lf).out).html;
  for (const sep of ['\r\n', '\r']) {
    const r = render(lf.replace(/\n/g, sep));
    assert.ok(r.ok, `sep ${JSON.stringify(sep)}: ${r.err.slice(0, 200)}`);
    assert.equal(JSON.parse(r.out).html, want, 'sep ' + JSON.stringify(sep));
  }
});

test('the preview slices are cut the same on CRLF text, and carry no \\r', () => {
  const lf = Array.from({ length: 1500 }, (_, i) => `# S${i}\n\n- item ${i}\n- another ${i}\n`).join('\n');
  const a = JSON.parse(render(lf).out).slices, b = JSON.parse(render(lf.replace(/\n/g, '\r\n')).out).slices;
  assert.ok(a.length > 3);
  assert.deepEqual(b, a);
});

test('a list line the item pattern rejects is shown and passed, never an endless loop', () => {
  for (const text of ['-  \n', '- x\n', '- a b\n', '1. a\u0085\n', '-   　\n- ok\n', '- \n', '1. \n']) {
    const r = render(text);
    assert.ok(r.ok, JSON.stringify(text) + ' ' + r.err.slice(0, 120));
  }
});
