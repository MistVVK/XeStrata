/** Second batch of the owner's chat feedback: reasoning that opens smoothly, code blocks, the preview icon button, the sidebar preview's motion. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const read = p => readFile(new URL(`../${p}`, import.meta.url), 'utf8');
const { code } = await transform((await read('src/lib/code-preview.ts')).replace("import { useSyncExternalStore } from 'react';", 'const useSyncExternalStore = () => null;').replace("import type { PreviewArtifact } from './artifacts';", ''), { loader: 'ts', format: 'esm' });
const lib = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const liquid = await read('src/styles/liquid.css');

test('only HTML and Markdown blocks have a preview; each block has its own stable id', () => {
  assert.equal(lib.codePreviewKind('html'), 'website');
  assert.equal(lib.codePreviewKind('HTML'), 'website');
  assert.equal(lib.codePreviewKind('md'), 'markdown');
  assert.equal(lib.codePreviewKind('markdown'), 'markdown');
  assert.equal(lib.codePreviewKind('bash'), null);
  const a = lib.codePreview('m1', 'html', '<p>a</p>'), b = lib.codePreview('m1', 'html', '<p>b</p>');
  assert.notEqual(a.id, b.id, 'two blocks in one answer are two previews');
  assert.equal(a.id, lib.codePreview('m1', 'html', '<p>a</p>').id, 'the same block keeps its id');
  assert.equal(lib.codePreview('m1', 'bash', 'ls'), null);
});

test('a finished answer with exactly one whole HTML page goes to the sidebar by itself; snippets and several blocks do not', () => {
  const page = '<!doctype html>\n<html><body>hi</body></html>';
  assert.equal(lib.mainCodePreview('m', `Here:\n\n\`\`\`html\n${page}\n\`\`\`\n`)?.kind, 'website');
  assert.equal(lib.mainCodePreview('m', 'A snippet:\n\n```html\n<b>x</b>\n```\n'), null);
  assert.equal(lib.mainCodePreview('m', `\`\`\`html\n${page}\n\`\`\`\n\n\`\`\`html\n${page}\n\`\`\`\n`), null);
  assert.equal(lib.mainCodePreview('m', 'Notes:\n\n```markdown\n# a\n```\n'), null, 'Markdown blocks keep the button only');
});

test('the code block has an icon preview button left of Copy, with an open state, and no text button', async () => {
  const md = await read('src/components/Markdown.tsx'), btn = await read('src/components/CodePreviewBtn.tsx');
  assert.ok(!/<Btn[^>]*>\{tr\("Open preview"\)/.test(md), 'no text button');
  assert.ok(md.indexOf('<CodePreviewBtn') < md.indexOf('<CopyBtn'), 'preview first, copy after it');
  assert.match(btn, /aria-pressed=\{open\}/);
  assert.match(btn, /tr\(open \? 'Close preview' : 'Open preview'\)/);
  assert.match(liquid, /\.st-code \.st-btn--icon\[data-open\]/);
});

test('reasoning opens by growing and fading, the line above never moves, and it has hover and press feedback', () => {
  assert.match(liquid, /\.rz::details-content \{[^}]*block-size: 0;[^}]*opacity: 0;/);
  assert.match(liquid, /\.rz\[open\]::details-content \{[^}]*block-size: auto;[^}]*opacity: 1;/);
  assert.match(liquid, /\.rz > summary:hover \{[^}]*--st-fill-hover/);
  assert.match(liquid, /\.rz > summary:active \{[^}]*--st-fill-press/);
  assert.match(liquid, /\.rz__body \{[^}]*border-left: 2px solid/);
});

test('code blocks have room after them and are lighter than the text; the 12px floor of pass 37 is untouched', async () => {
  assert.match(liquid, /\.st-bubble \.st-code \{\s*margin: 16px 0 20px;/);
  assert.match(liquid, /\.st-code__head \{\s*min-height: 34px;/);
  const app = await read('src/styles/app.css');
  assert.match(app, /--st-fs-read-sm: max\(12px, calc\(var\(--st-content-fs, 12px\) \* \.8125\)\);/);
});

test('the sidebar preview comes in and goes out with motion, from the one motion scale, and the reduced-motion gate zeroes the distance', () => {
  assert.match(liquid, /\.chatsplit \.artifact-panel \{\s*animation: st-panel-in var\(--st-t-choreo\) var\(--st-ease-enter\) both;/);
  assert.match(liquid, /\[data-leaving\] \{[^}]*animation: st-panel-out var\(--st-t-choreo\) var\(--st-ease-exit\) both;/);
  assert.match(liquid, /@media \(prefers-reduced-motion: reduce\) \{\s*:root \{ --st-press-scale: 1; --st-panel-shift: 0px; \}/);
});

test('only the layer-safe properties move: no transition on width, height or margin in the motion section', () => {
  const section = liquid.slice(liquid.indexOf('17. the motion system'));
  assert.ok(!/transition:[^;]*\b(width|height|margin|top|left)\b/.test(section));
});
