// #5071, the other text fields: the Enter that commits an IME conversion runs no field's action (the command palette, the font name,
// the model folder, a custom model ID, the number fields), and the web page of `strata serve` (serve/web/ime.js) follows the same rule.
// The guard itself and each engine's order of events are in ime-enter.test.mjs. VERIFIED BY REPLAY ONLY, like that file.
import assert from 'node:assert/strict';
import test from 'node:test';
import ts from 'typescript';
import { createRequire } from 'node:module';
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import { build } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const load = async entry => {
  const built = await build({ entryPoints: [resolve(root, entry)], bundle: true, write: false, platform: 'node', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const { imeKeys } = await load('src/lib/ime-keys.ts');
const { createImeGuard, IME_COMMIT_WINDOW_MS } = await load('src/lib/ime-enter.ts');
const page = createRequire(import.meta.url)(resolve(root, '../../serve/web/ime.js'));

// each engine's commit, then the person's own Enter (t: the event's timeStamp in ms)
const key = (t, k, keyCode = 13, isComposing = false) => ({ type: 'keydown', t, key: k, keyCode, isComposing });
const ORDERS = {
  'Chrome / Edge / Firefox': [{ type: 'start', t: 0 }, key(100, 'Enter', 229, true), { type: 'end', t: 101 }, key(900, 'Enter')],
  'Safari / WKWebView': [{ type: 'start', t: 0 }, { type: 'end', t: 100 }, key(101, 'Enter', 229), key(900, 'Enter')],
  'WebKitGTK': [{ type: 'start', t: 0 }, { type: 'end', t: 100 }, key(101, 'Enter'), key(900, 'Enter')],
  'an Escape that cancels': [{ type: 'start', t: 0 }, key(100, 'Escape', 229, true), { type: 'end', t: 101 }, key(900, 'Enter')],
  'Latin input': [key(100, 'Enter'), key(200, 'Enter')],
};

test('a field that spreads ime.events and asks ime.owns(e) first acts once, on the Enter after the conversion', () => {
  for (const [name, events] of Object.entries(ORDERS)) {
    const ime = imeKeys(), acted = [];
    for (const e of events) {
      if (e.type === 'start') ime.events.onCompositionStart();
      else if (e.type === 'end') ime.events.onCompositionEnd({ timeStamp: e.t });
      else if (!ime.owns({ key: e.key, timeStamp: e.t, nativeEvent: { isComposing: e.isComposing, keyCode: e.keyCode } }) && e.key === 'Enter') acted.push(e.t);
    }
    assert.deepEqual(acted, name === 'Latin input' ? [100, 200] : [900], name);
  }
});

test('the web page of strata serve uses the same rule as the app (serve/web/ime.js)', () => {
  assert.equal(page.WINDOW_MS, IME_COMMIT_WINDOW_MS);
  const late = [0, 1, IME_COMMIT_WINDOW_MS - 1, IME_COMMIT_WINDOW_MS, IME_COMMIT_WINDOW_MS + 1, -1];
  const orders = { ...ORDERS, ...Object.fromEntries(late.map(d => [`Enter ${d} ms after compositionend`, [{ type: 'end', t: 100 }, key(100 + d, 'Enter')]])) };
  for (const [name, events] of Object.entries(orders)) {
    const app = createImeGuard(), web = page.createGuard();
    for (const e of events) {
      if (e.type === 'start') { app.compositionStart(); web.compositionStart(); }
      else if (e.type === 'end') { app.compositionEnd(e.t); web.compositionEnd(e.t); }
      else assert.equal(web.keyDown(e, e.t), app.keyDown(e, e.t), `${name}, key at ${e.t}`);
    }
  }
});

/** The text of a JSX onKeyDown handler: inline, or the function a name in the same file points to. */
function handlerText(sf, expression) {
  if (!ts.isIdentifier(expression)) return expression.getText(sf);
  let found = '';
  const visit = node => {
    if ((ts.isVariableDeclaration(node) || ts.isFunctionDeclaration(node)) && node.name?.getText(sf) === expression.text) found = node.getText(sf);
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return found;
}

test('every text field that acts on Enter asks the IME guard first (a new field cannot forget it)', async () => {
  const fields = [], missing = [];
  const scan = async dir => {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) { await scan(path); continue; }
      if (!path.endsWith('.tsx')) continue;
      const sf = ts.createSourceFile(path, await readFile(path, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
      const visit = node => {
        if ((ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) && ['input', 'textarea'].includes(node.tagName.getText(sf))) {
          const attrs = node.attributes.properties;
          const onKeyDown = attrs.find(a => ts.isJsxAttribute(a) && a.name.getText(sf) === 'onKeyDown');
          const handler = onKeyDown?.initializer && ts.isJsxExpression(onKeyDown.initializer) && onKeyDown.initializer.expression ? handlerText(sf, onKeyDown.initializer.expression) : '';
          if (/['"]Enter['"]/.test(handler)) {
            const at = `${relative(root, path).replace(/\\/g, '/')}:${sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1}`;
            fields.push(at);
            const asks = /\bime\.owns\(|\bime\.current\.keyDown\(/.test(handler);
            const hears = attrs.some(a => (ts.isJsxSpreadAttribute(a) && a.expression.getText(sf) === 'ime.events') || (ts.isJsxAttribute(a) && a.name.getText(sf) === 'onCompositionEnd'));
            if (!asks || !hears) missing.push(`${at}${asks ? '' : ' (onKeyDown does not ask ime.owns(e) first)'}${hears ? '' : ' (no {...ime.events})'}`);
          }
        }
        ts.forEachChild(node, visit);
      };
      visit(sf);
    }
  };
  await scan(resolve(root, 'src'));
  assert.ok(fields.length >= 8, `found ${fields.length} fields: ${fields.join(', ')}`);
  assert.deepEqual(missing, []);
});
