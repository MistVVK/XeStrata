// Owner 11-16: 0.2.16 ships with every language checked. Three older English sentences that this round's fixes show had no catalog key, so
// they stayed English in every language: the answer an approval card ends with when nothing waits for it any more (#5308), the toast when an
// answer was not delivered, and the context-size error whose number #4249 changed. Each now has a key and translations (drafts on the
// lane's [DRAFT] list), and the approval card passes an error result through tr().
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/${l}.json`, import.meta.url), 'utf8'))]));
const src = p => readFileSync(new URL(p, import.meta.url), 'utf8');

const sentences = [
  ['That question is no longer waiting for an answer.', '../server/agent/agent-bridge.mjs', "new AgentUnavailable('That question is no longer waiting for an answer.', 'approval-gone')"],
  ['That answer was not delivered', '../src/lib/store.ts', "pushToast('error', 'That answer was not delivered',"],
  ['ctx.length: enter a whole context size from 1K to 1024K.', '../server/engine.mjs', "addError('ctx.length', 'enter a whole context size from 1K to 1024K.')"],
];

test('each sentence is the one the code produces, has a key, and is translated in all eight other languages', () => {
  assert.match(src('../server/engine.mjs'), /const addError = \(key, message\) => issues\.push\(\{ level: 'error', key, message: `\$\{key\}: \$\{message\}` \}\);/, 'the settings error is "<key>: <message>"');
  for (const [text, file, use] of sentences) {
    assert.ok(src(file).includes(use), `${file} still says it: ${use}`);
    const id = Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
    assert.ok(id, `"${text}" has a key`);
    for (const l of langs.slice(1)) assert.ok(catalogs[l][id] && catalogs[l][id] !== text, `${l}: "${text}"`);
  }
});

test('the approval card shows an error result in the person\'s language; other results as the tool wrote them', () => {
  assert.ok(src('../src/components/ApprovalCard.tsx').includes("{t.result ? <p className=\"small\">{t.state === 'error' ? tr(t.result) : t.result}</p> : null}"));
});
