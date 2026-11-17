// #3818: a single newline inside a paragraph is a space; a blank line starts a new paragraph; two trailing spaces or a backslash is a line break.
import assert from 'node:assert/strict';
import test from 'node:test';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const bundle = await build({
  stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { Markdown } from './src/components/Markdown.tsx'; export const html = text => renderToStaticMarkup(React.createElement(Markdown, { text }));`, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null; export const Icon = () => null;' }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
const { html } = mod.exports;
const paragraphs = out => (out.match(/<p>/g) ?? []).length;

test('a single newline is a soft break, a blank line a new paragraph', () => {
  const soft = html('Line one\nLine two');
  assert.equal(paragraphs(soft), 1); assert.match(soft, /Line one Line two/);
  assert.equal(paragraphs(html('One\n\nTwo')), 2);
});
test('two trailing spaces or a backslash keep a line break', () => {
  assert.match(html('One  \nTwo'), /One<br\/>Two/);
  assert.match(html('One' + String.fromCharCode(92) + String.fromCharCode(10) + 'Two'), /One<br\/>Two/);
});
test('a list, heading or code block right after a line still starts its own block', () => {
  const out = html('Intro\n- a\n- b\n## Head\ntext\n```\ncode\n```');
  assert.match(out, /<p>Intro<\/p><ul>/); assert.match(out, /<h2>Head<\/h2><p>text<\/p>/);
});
