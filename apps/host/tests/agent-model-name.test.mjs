// #5971: asked which model it runs on, the assistant answered "Claude": its prompt said "You are running as {{model}}", and the Harness's
// {{model}} is the route's model id (for a model on this computer, a quantization file name). The persona now says {{model_name}}, the name
// the person sees in the model picker, registered per chat by plugins/strata-models.mjs from the agent's route; the id when no route fits.
// Without model routes the plugin is not in the profile, so the persona keeps {{model}} (an unregistered variable would fail the prompt).
import test from 'node:test';
import assert from 'node:assert/strict';
import { apply, modelNameOf } from '../server/agent/plugins/strata-models.mjs';
import { agentPromptFor } from '../server/agent/preset.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

const routes = [
  { provider: 'strata-local', model: 'iq3_s', name: 'Qwen3.8 Flash Next', connection: 'this computer', local: true },
  { provider: 'api-own', model: 'deepseek/deepseek-v3.2', name: 'DeepSeek V3.2', connection: 'OpenRouter', local: false },
];

test('the name is found by the chat\'s route and model; the id when no route fits', () => {
  assert.equal(modelNameOf(routes, { provider: 'strata-local', model: 'iq3_s' }), 'Qwen3.8 Flash Next');
  assert.equal(modelNameOf(routes, { provider: 'api-own', model: 'deepseek/deepseek-v3.2' }), 'DeepSeek V3.2');
  assert.equal(modelNameOf(routes, { provider: 'api-other', model: 'iq3_s' }), 'iq3_s', 'the same id on another route is not this name');
  assert.equal(modelNameOf([], { provider: 'x', model: 'm-1' }), 'm-1');
  assert.equal(modelNameOf(undefined, undefined), 'an AI model');
});

test('the plugin registers {{model_name}}, resolved from the agent the prompt is for', () => {
  const vars = new Map();
  const ctx = { systemPrompt: { variable: (name, fn) => vars.set(name, fn) }, tools: { get: () => undefined }, get: () => ({ list: () => [] }), on: () => {} };
  apply(ctx, { routes });
  assert.ok(vars.has('model_name'));
  assert.equal(vars.get('model_name')({ agent: { options: { provider: 'strata-local', model: 'iq3_s' } } }), 'Qwen3.8 Flash Next');
  assert.equal(vars.get('model_name')({}), 'an AI model', 'an assembly with no agent still gets text');
});

test('the persona says {{model_name}} only when asked to, and keeps the computer line', () => {
  assert.match(agentPromptFor('win32', { modelNames: true }), /You are running as \{\{model_name\}\}\./);
  assert.match(agentPromptFor('win32'), /You are running as \{\{model\}\}\./);
  assert.equal(agentPromptFor('win32', { modelNames: true }).replace('{{model_name}}', '{{model}}'), agentPromptFor('win32'));
});

test('the profile: the persona names {{model_name}} exactly when the plugin that registers it is in the profile', () => {
  const rows = p => JSON.parse(p.patch.slice(p.patch.indexOf('[')));
  const flat = list => list.flatMap(r => r.insert ? r.insert : [r]);
  const persona = list => flat(list).find(r => r.id === 'preset-standard')?.config?.plugins?.find(x => x.id === 'persona')?.config?.prefix ?? '';
  const withModel = rows(buildProfile({ local: [{ id: 'a', modelId: 'iq3_s', label: 'Qwen3.8 Flash Next', baseUrl: 'http://127.0.0.1:9001', context: 65536 }], connections: [] }));
  assert.ok(flat(withModel).some(r => r.id === 'strata-models'));
  assert.match(persona(withModel), /\{\{model_name\}\}/);
  const none = rows(buildProfile({ local: [], connections: [] }));
  assert.ok(!flat(none).some(r => r.id === 'strata-models'));
  assert.ok(!/\{\{model_name\}\}/.test(JSON.stringify(none)), 'no plugin, no {{model_name}}');
});
