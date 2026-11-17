// #6000: a web address the answer writes as plain text was never a link (only [text](url) was). Plain text now links http(s) addresses, with
// CJK punctuation, sentence punctuation and unbalanced brackets left outside; code spans and blocks, other schemes and existing links unchanged.
import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const entry = [
  "import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server';",
  "import { Markdown } from './src/components/Markdown.tsx'; export { splitLinks } from './src/lib/autolink.ts';",
  'export const html = text => renderToStaticMarkup(React.createElement(Markdown, { text }));',
].join('\n');
const bundle = await build({
  stdin: { contents: entry, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'esm', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css', 'react', 'react-dom', 'react-dom/server', 'react/jsx-runtime'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null; export const Icon = () => null;' }));
  } }],
});
const { writeFileSync, mkdtempSync, rmSync } = await import('node:fs');
const dir = mkdtempSync(resolve(root, 'node_modules', '.gui1116-autolink-'));
const file = resolve(dir, 'bundle.mjs');
writeFileSync(file, bundle.outputFiles[0].text);
const { splitLinks, html } = await import(`file://${file.replaceAll('\\', '/')}`);
test.after(() => rmSync(dir, { recursive: true, force: true }));

const links = text => splitLinks(text).filter(p => p.href).map(p => p.href);

test('plain addresses become links; the sentence around them stays text', () => {
  assert.deepEqual(links('See https://example.com/docs for more.'), ['https://example.com/docs']);
  assert.deepEqual(links('Two: http://a.example/x?q=1&r=2#top, and https://b.example.'), ['http://a.example/x?q=1&r=2#top', 'https://b.example']);
  assert.deepEqual(splitLinks('go to https://example.com.').map(p => p.text), ['go to ', 'https://example.com', '.']);
});

test('CJK punctuation ends an address', () => {
  assert.deepEqual(links('文档在https://example.com/文档，请看。'), ['https://example.com/文档']);
  assert.deepEqual(links('「https://example.jp/a」を開く'), ['https://example.jp/a']);
  assert.deepEqual(links('（https://example.cn/path）'), ['https://example.cn/path']);
});

test('brackets: a balanced pair stays in the address, an unbalanced closing one is the sentence\'s', () => {
  assert.deepEqual(links('https://en.wikipedia.org/wiki/Rust_(programming_language) is it'), ['https://en.wikipedia.org/wiki/Rust_(programming_language)']);
  assert.deepEqual(links('(see https://example.com/a)'), ['https://example.com/a']);
  assert.deepEqual(links('[https://example.com/b]'), ['https://example.com/b']);
});

test('other schemes, no host, and plain words are never links', () => {
  for (const t of ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,hi', 'https://', 'http:// spaced', 'example.com', 'ftp://example.com']) assert.deepEqual(links(t), [], t);
});

test('rendered: a plain address is an <a>; code spans and blocks, and existing links, are as before', () => {
  const out = html('Read https://example.com/guide, then `curl https://example.com/api` and [the docs](https://example.com/docs).\n\n```\nhttps://example.com/in-a-block\n```');
  assert.match(out, /<a href="https:\/\/example\.com\/guide" target="_blank" rel="noreferrer">https:\/\/example\.com\/guide<\/a>,/);
  assert.match(out, /<code class="md-code">curl https:\/\/example\.com\/api<\/code>/);
  assert.match(out, /<a href="https:\/\/example\.com\/docs" target="_blank" rel="noreferrer">the docs<\/a>/);
  assert.ok(!/href="https:\/\/example\.com\/in-a-block"/.test(out), 'nothing inside a code block is linked');
  assert.ok(!/href="https:\/\/example\.com\/api"/.test(out), 'nothing inside a code span is linked');
});
