import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

/* The direct chat path (no agent): the composer's Reasoning effort menu reaches the server as reasoning_effort ("Off" -> "none", which the
   server turns into enable_thinking=false), and the Thinking budget as reasoning_budget_tokens, never when thinking is off. */
const host = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
const composer = readFileSync(new URL('../src/components/ComposerControls.tsx', import.meta.url), 'utf8');
const store = readFileSync(new URL('../src/lib/store.ts', import.meta.url), 'utf8');

test('Off is sent as none and Max as high for Qwen (Max as max for Hy4-Flash); the budget only with thinking on', () => {
  assert.match(host, /\(\{off:'none',max:'high'\}\[String\(thinking\)\.toLowerCase\(\)\]/);
  assert.match(host, /\(\{off:'none',(?:minimal:'high',)?low:'high',medium:'high',high:'high',max:'max'\}\[String\(thinking\)\.toLowerCase\(\)\]/);
  assert.match(host, /thinkingBudget>0&&String\(thinking\)\.toLowerCase\(\)!=='off'\?\{reasoning_budget_tokens:thinkingBudget\}/);
});

test('the budget travels from the menu to the request', () => {
  assert.match(composer, /THINKING_BUDGETS/);
  assert.match(composer, /thinkingBudget: value/);
  assert.match(store, /thinkingBudget:state\.sessions\.find/);
  assert.match(host, /thinkingBudget:Number\.isInteger\(body\.thinkingBudget\)/);
});
