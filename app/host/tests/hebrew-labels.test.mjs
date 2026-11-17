// Hebrew: the screen title "Serving" and eleven icon / part names (answer-speed, bolt, check, clock ...) kept their English text in he.json
// while every other language translates them (found by the 0.2.16 all-languages check). They are Hebrew now.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = name => JSON.parse(readFileSync(new URL(`../src/locales/${name}.json`, import.meta.url), 'utf8'));
const en = read('en');
const he = read('he');
const KEYS = ['s1611', 's2185', 's2187', 's2196', 's2209', 's2210', 's2222', 's2226', 's2306', 's2346', 's2356', 's2367'];

test('these Hebrew labels are not the English text', () => {
  for (const key of KEYS) {
    assert.notEqual(he[key], en[key], `${key} (${en[key]})`);
    assert.match(he[key], /[֐-׿]/, `${key} has Hebrew letters`);
  }
});
