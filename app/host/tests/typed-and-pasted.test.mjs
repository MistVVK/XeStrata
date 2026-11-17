import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const load = async name => import(`data:text/javascript;base64,${Buffer.from((await transform(await readFile(new URL(`../src/lib/${name}.ts`, import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' })).code).toString('base64')}`);
const { parseDecimal, partialDecimal } = await load('decimal');
const { pastedFiles } = await load('pasted-files');

test('a decimal point can be typed step by step (#3736)', () => {
  for (const step of ['', '.', '0', '0.', '0.9', '0.95']) assert.equal(partialDecimal(step), true, step);
  assert.equal(parseDecimal('.'), null); assert.equal(parseDecimal('0.'), null);
  assert.equal(parseDecimal('0.95'), 0.95); assert.equal(parseDecimal('.95'), 0.95); assert.equal(parseDecimal('0,95'), 0.95); assert.equal(parseDecimal('20'), 20);
  assert.equal(partialDecimal('0.9x'), false); assert.equal(parseDecimal('1e3'), null);
});

test('a screenshot that is only in clipboard items is attached (#3748)', () => {
  const png = new File(['x'], 'image.png', { type: 'image/png' });
  const data = { files: [], items: [{ kind: 'file', type: 'image/png', getAsFile: () => png }, { kind: 'string', type: 'text/plain', getAsFile: () => null }] };
  const out = pastedFiles(data);
  assert.equal(out.length, 1); assert.equal(out[0].type, 'image/png'); assert.match(out[0].name, /^Pasted picture .*\.png$/);
  assert.equal(pastedFiles({ files: [png], items: [{ kind: 'file', type: 'image/png', getAsFile: () => png }] }).length, 1, 'never twice');
  assert.deepEqual(pastedFiles({ files: [], items: [{ kind: 'string', type: 'text/plain', getAsFile: () => null }] }), []);
});
