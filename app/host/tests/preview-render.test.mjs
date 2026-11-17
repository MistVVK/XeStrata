// #5180: the Markdown preview in the sidebar is keyed by the text, drawn in bounded slices, and does no work for an unchanged file
// while unrelated store updates (streamed tokens) re-render the chat. Pure text work plus server rendering; the browser run is in the report.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const bundle = await build({
  stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { Markdown } from './src/components/Markdown.tsx'; export * from './src/lib/preview-render.ts'; export const html = text => renderToStaticMarkup(React.createElement(Markdown, { text }));`, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null; export const Icon = () => null;' }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
const { html, textRevision, splitPreview, previewSlices, previewStats, previewTier, slicesAfter, slicesAtStart,
  PREVIEW_SLICE_CHARS, PREVIEW_FIRST_SLICES, PREVIEW_BLOCK_LIMIT, PREVIEW_RENDER_LIMIT } = mod.exports;

/** A long document the way the reports describe it: headings, paragraphs, lists, tables and fenced code with blank lines in them. */
function longDoc(sections) {
  const out = [];
  for (let n = 0; n < sections; n++) {
    out.push(`# Section ${n}`, '', `Paragraph ${n} with **bold**, *italic*, \`code\` and a [link](https://example.com/${n}).`, 'It continues on a second line.', '');
    out.push('- first item', '- second item', '', '1. step one', '2. step two', '');
    out.push('| Name | Value |', '| --- | --- |', ...Array.from({ length: 6 }, (_, r) => `| row ${r} | value ${n}-${r} |`), '');
    out.push('```js', `const a${n} = 1;`, '', '', `console.log(a${n});`, '```', '');
    out.push('> a quote', '');
  }
  return out.join('\n');
}

test('the revision key follows the text and nothing else', () => {
  assert.equal(textRevision('abc'), textRevision('abc'));
  assert.notEqual(textRevision('abc'), textRevision('abd'));
  assert.notEqual(textRevision('abc'), textRevision('abc '));
  assert.notEqual(textRevision(''), textRevision(' '));
  assert.match(textRevision('x'), /^[0-9a-z]+-[0-9a-z]+$/);
});

test('slices drawn one after the other are the same elements as the whole file', () => {
  const docs = [
    longDoc(40),
    'one\n\ntwo\n\n\n\nthree',
    '```\nfenced\n\nwith a blank line\n\n```\n\nafter',
    '$$\nx = 1\n\ny = 2\n$$\n\ntext',
    '\\[\na\n\nb\n\\]\n\ntext',
    '| a | b |\n| --- | --- |\n| 1 | 2 |\n\n| c | d |\n| --- | --- |\n| 3 | 4 |',
    '- a\n- b\n\n- c\n\n1. x\n\n2. y',
    '$$ open and never closed\n\nnext paragraph\n\nmore',
    '',
    '\n\n\n',
  ];
  for (const doc of docs)
    for (const target of [1, 40, 500, PREVIEW_SLICE_CHARS]) {
      const whole = html(doc);
      const sliced = splitPreview(doc, target).map(s => html(s.text)).join('');
      assert.equal(sliced, whole, `target ${target}, doc starting ${JSON.stringify(doc.slice(0, 30))}`);
    }
});

test('a slice never ends inside a fenced block or a display-math block', () => {
  const doc = ['intro', '', '```', 'a', '', 'b', '```', '', '$$', 'x', '', 'y', '$$', '', 'outro'].join('\n');
  const slices = splitPreview(doc, 1);
  for (const s of slices) {
    const fences = (s.text.match(/^\s*```/gm) ?? []).length;
    assert.equal(fences % 2, 0, 'balanced fences in ' + JSON.stringify(s.text));
    const math = (s.text.match(/^\s*\$\$/gm) ?? []).length;
    assert.equal(math % 2, 0, 'balanced math in ' + JSON.stringify(s.text));
  }
  assert.ok(slices.length >= 3);
});

test('a long file is cut into slices of about the target size, each with its own key', () => {
  const doc = longDoc(160);
  assert.ok(doc.length > 60000, 'a long file: ' + doc.length);
  const slices = splitPreview(doc);
  assert.ok(slices.length > 8);
  for (const s of slices.slice(0, -1))
    assert.ok(s.text.length < PREVIEW_SLICE_CHARS * 2, 'bounded slice ' + s.text.length);
  assert.equal(new Set(slices.map(s => s.key)).size, slices.length);
});

test('changing one place changes one slice key; the others keep theirs and are not rendered again', () => {
  const a = longDoc(60);
  const b = a.replace('Paragraph 3 with', 'Paragraph 3 changed with');
  const ka = splitPreview(a).map(s => s.key), kb = splitPreview(b).map(s => s.key);
  assert.equal(ka.length, kb.length);
  const changed = ka.filter((k, i) => k !== kb[i]);
  assert.equal(changed.length, 1);
});

test('render count: an unchanged file is cut once however many store updates arrive', () => {
  const doc = longDoc(80);
  const before = previewStats.cuts;
  let first;
  for (let tick = 0; tick < 500; tick++) {          // 500 streamed-token updates of the chat, each asking for the same file
    const slices = previewSlices(doc);
    first ??= slices;
    assert.equal(slices, first, 'the same array: nothing downstream sees a change');
  }
  assert.equal(previewStats.cuts - before, 1);
  // the file changing is one more cut, and going back to the old text is still known
  const changed = doc + '\nmore';
  previewSlices(changed); previewSlices(changed);
  assert.equal(previewStats.cuts - before, 2);
  previewSlices(doc);
  assert.equal(previewStats.cuts - before, 2);
});

test('the cache keeps a few texts only', () => {
  const before = previewStats.cuts;
  for (let i = 0; i < 20; i++) previewSlices('text number ' + i);
  assert.equal(previewStats.cuts - before, 20);
  previewSlices('text number 0');                 // long gone from a cache of four
  assert.equal(previewStats.cuts - before, 21);
});

test('a very large file does not produce unbounded work', () => {
  const huge = longDoc(2000);
  assert.ok(huge.length > PREVIEW_RENDER_LIMIT);
  assert.equal(previewTier(huge), 'plain');
  assert.equal(previewTier(longDoc(100)), 'markdown');
  // one table or code block that cannot be cut is plain text, not parsed
  const bigTable = '| a | b |\n| --- | --- |\n' + Array.from({ length: 5000 }, (_, i) => `| ${i} | ${'x'.repeat(10)} |`).join('\n');
  assert.ok(bigTable.length > PREVIEW_BLOCK_LIMIT);
  const slices = splitPreview(bigTable);
  assert.equal(slices.length, 1);
  assert.equal(slices[0].plain, true);
  assert.equal(splitPreview('| a | b |\n| --- | --- |\n| 1 | 2 |')[0].plain, false);
});

test('the first paint and every step are bounded; a new text never takes back slices already shown', () => {
  assert.equal(slicesAtStart(0, 100), PREVIEW_FIRST_SLICES);
  assert.equal(slicesAtStart(0, 1), 1);
  assert.equal(slicesAtStart(0, 0), 0);
  assert.equal(slicesAtStart(30, 100), 30);
  assert.equal(slicesAtStart(30, 10), 10);
  // steps reach the end in a bounded number of steps and never pass it
  let shown = slicesAtStart(0, 100), steps = 0;
  while (shown < 100) { const next = slicesAfter(shown, 100); assert.ok(next > shown && next - shown <= 2); shown = next; steps++; }
  assert.equal(steps, 49);
  assert.equal(slicesAfter(100, 100), 100);
});

test('the sidebar uses the bounded component and the component cancels superseded work', () => {
  const panel = readFileSync(resolve(root, 'src/components/ArtifactPanel.tsx'), 'utf8');
  assert.match(panel, /<PreviewMarkdown text=\{text\}\/>/);
  assert.doesNotMatch(panel, /<Markdown text=\{text\}\/>/);
  const view = readFileSync(resolve(root, 'src/components/PreviewMarkdown.tsx'), 'utf8');
  assert.match(view, /export const PreviewMarkdown = memo\(/);
  assert.match(view, /const Slice = memo\(/);
  assert.match(view, /return \(\) => clearTimeout\(timer\)/);
  assert.match(view, /key=\{s\.key\}/);
});
