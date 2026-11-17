// A tool call the model wrote as text that nothing ran (a <tool_call> block that reached the answer) is one collapsed line that opens to the raw text.
// Only a block that starts its line counts: inside a code block or in the middle of a sentence the tag stays as written, so talk about the format
// still shows it. A block that is not closed yet (the reply is streaming) is collapsed too.
import assert from 'node:assert/strict';
import test from 'node:test';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const bundle = await build({
  stdin: { contents: `import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { Markdown, continuesParagraph, failedCall } from './src/components/Markdown.tsx'; export { continuesParagraph, failedCall }; export const html = text => renderToStaticMarkup(React.createElement(Markdown, { text }));`, resolveDir: root, loader: 'ts' },
  bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['katex', 'katex/dist/katex.min.css'],
  plugins: [{ name: 'stub', setup(b) {
    b.onResolve({ filter: /^\.\/ui$/ }, () => ({ path: 'ui', namespace: 'test-stub' }));
    b.onLoad({ filter: /.*/, namespace: 'test-stub' }, () => ({ contents: 'export const IconBtn = () => null; export const Btn = () => null; export const Icon = () => null;' }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
const { html, continuesParagraph, failedCall } = mod.exports;
const count = (out, re) => (out.match(re) ?? []).length;
const COLLAPSED = /<details class="md-failedcall"><summary>A tool call that could not run<\/summary><pre class="md-failedcall__raw">/g;

test('the owner\'s case: prose, then a failed call right under it, then prose: the call is one collapsed line, the prose stays', () => {
  const out = html('I will run it.\n<tool_call>\n<function=nosuch>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n\nAll done after the tool step.');
  assert.equal(count(out, COLLAPSED), 1);
  assert.match(out, /<p>I will run it\.<\/p>/);
  assert.match(out, /<p>All done after the tool step\.<\/p>/);
  // the raw text is inside the disclosure, escaped, and nowhere else
  assert.match(out, /<pre class="md-failedcall__raw">&lt;tool_call&gt;\n&lt;function=nosuch&gt;\n&lt;parameter=command&gt;\nls\n&lt;\/parameter&gt;\n&lt;\/function&gt;\n&lt;\/tool_call&gt;<\/pre>/);
  assert.equal(count(out, /&lt;tool_call&gt;/g), 1);
  assert.doesNotMatch(out.replace(/<details.*?<\/details>/s, ''), /tool_call/);
  assert.doesNotMatch(out, /<details[^>]* open/);
});

test('a block inside a code fence stays raw, a fence that talks about the format still shows it', () => {
  const text = 'The format is:\n```xml\n<tool_call>\n<function=read>\n<parameter=file_path>\nx.txt\n</parameter>\n</function>\n</tool_call>\n```\nThat is all.';
  const out = html(text);
  assert.equal(count(out, COLLAPSED), 0);
  assert.match(out, /st-code/);
  assert.match(out, /tool_call/);
});

test('a tag in the middle of a sentence stays as written, also with a function tag in it', () => {
  const out = html('Models write <tool_call><function=read></function></tool_call> when they call a tool.');
  assert.equal(count(out, COLLAPSED), 0);
  assert.match(out, /<p>Models write &lt;tool_call&gt;&lt;function=read&gt;&lt;\/function&gt;&lt;\/tool_call&gt; when they call a tool\.<\/p>/);
});

test('a line that starts with the tag but holds no function or parameter tag is only text', () => {
  const out = html('<tool_call> is the tag the model server looks for.');
  assert.equal(count(out, COLLAPSED), 0);
  assert.match(out, /<p>&lt;tool_call&gt; is the tag/);
});

test('streaming: a block that is not closed yet is collapsed to the end of the text, and its raw text follows what has arrived', () => {
  const part = 'Reading it.\n<tool_call>\n<function=read>\n<parameter=file_path>\n';
  const out = html(part);
  assert.equal(count(out, COLLAPSED), 1);
  assert.match(out, /<p>Reading it\.<\/p>/);
  assert.match(out, /&lt;parameter=file_path&gt;/);
  // a few tokens in, before any function tag has arrived, it is still text (and collapses as soon as the tag is there)
  assert.equal(count(html('Reading it.\n<tool_call>\n'), COLLAPSED), 0);
  assert.equal(count(html(part + 'x.txt\n</parameter>\n</function>\n</tool_call>\n\nDone.'), COLLAPSED), 1);
});

test('text after the closing tag on its line is kept, and two blocks are two lines', () => {
  const out = html('<tool_call><function=a></function></tool_call> and then some words\n\n<tool_call>\n<function=b>\n</function>\n</tool_call>');
  assert.equal(count(out, COLLAPSED), 2);
  assert.match(out, /<\/details><p>and then some words<\/p><details/);
});

test('the paragraph above a failed call ends where the call starts', () => {
  const lines = ['I will run it.', '<tool_call>', '<function=x>', '</function>', '</tool_call>'];
  assert.equal(continuesParagraph(lines, 1), false);
  assert.equal(continuesParagraph(['one', '<tool_call> is a tag'], 1), true);
  assert.deepEqual(failedCall(lines, 1), { raw: '<tool_call>\n<function=x>\n</function>\n</tool_call>', end: 4, rest: '' });
  assert.equal(failedCall(lines, 0), null);
});

test('plain answers are unchanged', () => {
  assert.equal(count(html('Hello\n\n- a\n- b\n\n```js\nconst x = 1;\n```'), COLLAPSED), 0);
  assert.match(html('One\n\nTwo'), /<p>One<\/p><p>Two<\/p>/);
});
