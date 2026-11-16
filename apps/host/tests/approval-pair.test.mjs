/** #5077: Deny and Approve on an approval are one pair on one line, and the main action has no shadow that falls below its
 *  box (only the fill marks it). The focus ring (a box-shadow on button:focus-visible) must still win on the main action. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const css = await readFile(new URL('../src/styles/app.css', import.meta.url), 'utf8');

/** The last value of `prop` among the declaration blocks whose selector list names `selector` exactly. */
function last(selector, prop) {
  let value;
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(css))) {
    const sels = m[1].replace(/\/\*[\s\S]*?\*\//g, '').split(',').map(s => s.trim());
    if (!sels.includes(selector)) continue;
    for (const decl of m[2].split(';')) {
      const i = decl.indexOf(':');
      if (i > 0 && decl.slice(0, i).trim() === prop) value = decl.slice(i + 1).trim();
    }
  }
  return value;
}

for (const box of ['.askcard__acts', '.toolcard__quick-actions']) {
  test(`${box}: the main action casts no shadow below its box, and keeps its focus ring`, () => {
    const shadow = last(`${box} > .st-btn--primary:not(:focus-visible)`, 'box-shadow');
    assert.ok(shadow, 'a pair rule sets the main action\'s shadow');
    for (const layer of shadow.split(/,(?![^(]*\))/)) assert.match(layer.trim(), /^inset\b/, `outer shadow layer: ${layer}`);
    // No pair rule may set box-shadow on the main action while it has focus.
    assert.equal(last(`${box} > .st-btn--primary`, 'box-shadow'), undefined);
  });
}

test('the approval pair sits on one line, centred', () => {
  assert.equal(last('.askcard__acts', 'align-items'), 'center');
  assert.equal(last('.toolcard__quick-actions', 'align-items'), 'center');
});
