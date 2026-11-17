import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const liquid = readFileSync(resolve(import.meta.dirname, '..', 'src/styles/liquid.css'), 'utf8');
const rule = (selector) => {
  const start = liquid.indexOf(`${selector} {`);
  assert.ok(start >= 0, `${selector} exists`);
  return liquid.slice(start, liquid.indexOf('}', start));
};

test('light New chat is the shared primary: tinted surface, green text, no near-black ink', () => {
  const light = rule('.nav-new.st-btn--primary');
  assert.match(light, /background:\s*var\(--st-accent-surface\)/);
  assert.match(light, /color:\s*var\(--st-accent-surface-ink\)/);
  assert.ok(!/--st-accent-ink/.test(light), 'not the dark ink on a saturated fill');
  assert.ok(!/background:\s*var\(--st-accent\)\s*;/.test(light), 'no full-strength #10b981 fill');
});

test('its hover also uses the shared primary token', () => {
  assert.match(rule('.nav-new.st-btn--primary:hover:not(:disabled)'), /var\(--st-accent-surface-hover\)/);
});
