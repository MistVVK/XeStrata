// #6060: Hebrew reads right to left. The page gets dir="rtl" for it (app: store.ts, web page: i18n.js), and the composer's send arrow
// is mirrored there so it points the way the text runs. The other languages keep dir="ltr" and the arrow as it is.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = rel => readFileSync(new URL(rel, import.meta.url), 'utf8');

test('the app sets dir=rtl for Hebrew only', () => {
  const store = read('../src/lib/store.ts');
  assert.match(store, /root\.dir = state\.language === 'he' \? 'rtl' : 'ltr'/);
});

test('the web page sets dir=rtl for Hebrew only', () => {
  const js = read('../../../serve/web/i18n.js');
  assert.match(js, /RTL\.includes\(lang\) \? "rtl" : "ltr"/);
  assert.match(js, /RTL\s*=\s*\[\s*"he"\s*\]/);
});

test('the send arrow is mirrored in a right-to-left page, in the app and on the web page', () => {
  const app = read('../src/styles/app.css');
  const web = read('../../../serve/web/app.css');
  assert.match(app, /\[dir="rtl"\]\s+\.st-icon:has\(> use\[href\$="#i-send"\]\)\s*\{\s*transform:\s*scaleX\(-1\);?\s*\}/);
  assert.match(web, /\[dir="rtl"\]\s+\.st-btn--send\s+\.st-icon\s*\{\s*transform:\s*scaleX\(-1\);?\s*\}/);
  // nothing mirrors it for left-to-right pages
  assert.equal([...app.matchAll(/i-send[^{]*\{[^}]*scaleX\(-1\)/g)].length, 1, 'one rule in the app, the right-to-left one');
  assert.equal([...web.matchAll(/\.st-btn--send[^{]*\{[^}]*scaleX\(-1\)/g)].length, 1, 'one rule on the web page, the right-to-left one');
});
