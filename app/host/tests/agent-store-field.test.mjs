// #5903: a Gemini model added as an OpenAI-compatible connection failed with 400 "Invalid JSON payload received. Unknown name \"store\"":
// the Harness's OpenAI client (pi-ai) sends store: false to every OpenAI-style server it does not know as non-standard, and Google's
// endpoint refuses unknown fields. "store" is OpenAI's own; the profile now tells pi-ai to leave it out for every other address.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProfile } from '../server/agent/profile.mjs';

const providersOf = profile => {
  const out = {};
  const walk = v => { if (Array.isArray(v)) v.forEach(walk); else if (v && typeof v === 'object') { if (typeof v.api === 'string' && 'baseURL' in v) out[v.baseURL] = v; Object.values(v).forEach(walk); } };
  walk(JSON.parse(profile.patch.split('\n').slice(1).join('\n')));
  return out;
};
const conn = (id, baseUrl, protocol = 'openai-chat') => ({ id, name: id, baseUrl, protocol, apiKey: 'k', models: [{ id: 'm', name: 'm', contextMax: 131072 }] });

test('Gemini and other OpenAI-style servers get no "store"; OpenAI keeps it', () => {
  const p = providersOf(buildProfile({ local: [], connections: [
    conn('gemini', 'https://generativelanguage.googleapis.com/v1beta/openai'),
    conn('openai', 'https://api.openai.com/v1'),
    conn('other', 'https://api.example-llm.com/v1'),
    conn('router', 'https://openrouter.ai/api/v1'),
  ] }));
  assert.equal(p['https://generativelanguage.googleapis.com/v1beta/openai'].compat?.supportsStore, false, 'the reporter\'s Gemini endpoint');
  assert.equal(p['https://api.example-llm.com/v1'].compat?.supportsStore, false);
  assert.equal(p['https://api.openai.com/v1'].compat?.supportsStore, undefined, 'OpenAI: pi-ai\'s own choice, as before');
  assert.deepEqual(p['https://openrouter.ai/api/v1'].compat, { thinkingFormat: 'openrouter', supportsStore: false }, 'OpenRouter keeps its thinking format');
});

test('the Responses and Anthropic protocols are left as they were', () => {
  const p = providersOf(buildProfile({ local: [], connections: [
    conn('resp', 'https://api.example-llm.com/v1', 'openai-responses'),
    conn('anth', 'https://api.anthropic.com/v1', 'anthropic'),
  ] }));
  for (const v of Object.values(p)) assert.equal(v.compat, undefined, v.baseURL);
});
