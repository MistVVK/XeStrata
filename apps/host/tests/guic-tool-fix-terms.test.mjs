// The two tool-call strings of the agent (guic23: asking the model to fix its tool call; guic24: stopped after two tool calls that could not run) use
// the catalog's own words: German "Tool-Aufruf" and "Runde" (s1178, s1997; never "Werkzeug…" / "Durchgang"), Portuguese "turno" (s1997).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = lang => JSON.parse(readFileSync(new URL(`../src/locales/guic-${lang}.json`, import.meta.url), 'utf8'));

test('German says Tool-Aufruf and Runde, like the main catalog', () => {
  const de = read('de');
  assert.match(de.guic23, /Tool-Aufruf/);
  assert.match(de.guic24, /Tool-Aufrufe/);
  assert.match(de.guic24, /diese Runde/);
  for (const key of ['guic23', 'guic24']) assert.doesNotMatch(de[key], /Werkzeug|Durchgang/, key);
});

test('Portuguese says turno, like the main catalog', () => {
  assert.match(read('pt').guic24, /interrompeu este turno/);
});
