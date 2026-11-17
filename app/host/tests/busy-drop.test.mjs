// #5960: a picture pasted or dropped while an answer is being written was thrown away without a word.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const TITLE = 'Picture not added', BODY = 'Strata is still busy with this chat. Add the picture again when it is done.';

test('what came in decides what the notice says', async () => {
  const { busyDropKind } = await load('lib/busy-drop.ts');
  assert.equal(busyDropKind([{ type: 'image/png' }]), 'picture');
  assert.equal(busyDropKind([{ type: 'text/plain' }, { type: 'image/jpeg' }]), 'picture');
  assert.equal(busyDropKind([{ type: 'text/plain' }]), 'file');
  assert.equal(busyDropKind([{ type: '' }]), 'file');
  assert.equal(busyDropKind([]), null);
  assert.equal(busyDropKind(undefined), null);
});

test('the composer says so when it cannot take a drop, and still takes it when idle', () => {
  const c = src('components/ComposerControls.tsx');
  assert.match(c, /import \{ busyDropKind \} from '\.\.\/lib\/busy-drop';/);
  assert.match(c, /if \(!streaming && !reading\) \{ void addFilesLatest\.current\(incoming\); return; \}/);
  assert.ok(c.includes(`pushToast('warn', tr('${TITLE}'), tr('${BODY}'))`));
  assert.doesNotMatch(c, /const dropped = \(e: Event\) => \{ if \(!streaming && !reading\) void/);
});

test('both sentences are in the catalog in all ten languages, with the language rules', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  const key = text => Object.keys(cat.en).find(k => cat.en[k] === text);
  for (const text of [TITLE, BODY]) {
    const k = key(text);
    assert.ok(k, `${text} is in guic-en.json`);
    for (const l of LANGS) assert.ok(cat[l][k] && cat[l][k].trim(), `${l} has ${k}`);
  }
  const body = key(BODY);
  assert.match(cat.de[body], /Fügen Sie/, 'German uses Sie');
  // French: a no-break space (U+00A0) before ; : ? ! (none of these sentences has one, so none may be glued or use a plain space)
  for (const k of [key(TITLE), body]) assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[k]), false, 'French: ; : ? ! only after a no-break space');
});
