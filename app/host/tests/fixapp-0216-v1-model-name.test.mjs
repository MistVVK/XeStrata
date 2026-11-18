// FIX-APP-0216 A4: the app host's local /v1 answered HTTP 400 to the 0.2.16 test harness's chat3 and agent1 checks (AppImage cell on p1004, 11-18 08:21; the 0.2.15 cell the same).
// Cause: the harness (rel0216/relnotes/app/t16_scen.py) sends "model": "x"; with a model loaded the host's gateway resolved the name exactly and answered
// 400 {"error":{"message":"Choose a loaded model ID.","type":"invalid_request_error","param":"model","code":null}}. `strata serve` serves any name with its one model, and the host's own
// /v1/messages route already did (#5312), so other apps that send a model name of their own (gpt-4o, default, x) worked against the Terminal's server but not the app.
// This test runs the host's real native control (fixture) and the real gateway with the exact request bodies of chat3 turn 1 and agent1, and prints the request and response bodies
// ("A4-REPRO" lines) so the failure before the fix and the answer after it can be read from the log.
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';
import { nativeGateway } from '../server/native-gateway.mjs';

/** The two bodies of t16_scen.py, as it sends them. */
const CHAT3_TURN1 = { model: 'x', messages: [{ role: 'user', content: 'My name is Rada, my budget is 1840 euros and I travel with my dog Pim. Suggest one region in the Alps for a week, in three sentences.' }], max_tokens: 3000, temperature: 0.7 };
const AGENT1 = { model: 'x', messages: [{ role: 'system', content: 'You are a coding agent. Use the read_file tool when asked about a file.' }, { role: 'user', content: 'Read note07.txt with the tool and tell me the one word in it.' }],
  tools: [{ type: 'function', function: { name: 'read_file', description: 'Read a text file of the project', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } }], max_tokens: 2000, temperature: 0 };

/** One request through the real gateway: what the client sent, what reached the engine, what the client got back. */
async function viaGateway(control, path, body, label) {
  const toEngine = [], records = [];
  const serving = { open: v => { const r = { ...v, id: String(records.length) }; records.push(r); return r; }, mark: (id, v) => Object.assign(records[Number(id)], v), finish: (id, v) => Object.assign(records[Number(id)], v) };
  const fetcher = async (url, options) => { toEngine.push({ url, body: JSON.parse(options.body) }); return Response.json({ id: 'c1', object: 'chat.completion', choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 1 } }); };
  let status = 0, headers = {}, text = '';
  const res = { headersSent: false, writableEnded: false, destroyed: false, on() {}, writeHead(s, h) { status = s; headers = h; this.headersSent = true; }, write(chunk) { text += chunk; return true; }, end(chunk) { if (chunk) text += chunk; this.writableEnded = true; } };
  await nativeGateway({ req: { headers: { 'user-agent': 'python-urllib' } }, res, path, body, control, serving, fetcher });
  console.log(`A4-REPRO ${label} request=${JSON.stringify({ path, model: body.model, keys: Object.keys(body) })} response=${status} ${text.slice(0, 300)}`);
  return { status, text, toEngine, records };
}

async function loaded(f, models) {
  await f.control.refresh(true);
  for (const [model, gpus] of models) {
    const done = await finishJob(f.control, f.control.operation('load', { model, gpus, multiple: true }).id);
    assert.equal(done.state, 'done', done.error);
  }
  return f.control.state().instances;
}

test('one model loaded: the harness\'s chat3 and agent1 requests (model "x") are answered by that model, with its own ID sent to the engine', async () => {
  const f = nativeFixture();
  try {
    const [instance] = await loaded(f, [['iq2_xs', [0]]]);
    for (const [label, body] of [['chat3-turn1', CHAT3_TURN1], ['agent1', AGENT1]]) {
      const r = await viaGateway(f.control, '/v1/chat/completions', body, label);
      assert.equal(r.status, 200, `${label}: ${r.text}`);
      assert.equal(r.toEngine.length, 1);
      assert.equal(r.toEngine[0].body.model, instance.apiModelId, 'the engine is asked with the loaded model\'s own ID');
      assert.deepEqual(r.toEngine[0].body.messages, body.messages, 'the conversation is passed on unchanged');
      if (body.tools) assert.deepEqual(r.toEngine[0].body.tools, body.tools, 'the tools are passed on unchanged');
    }
    const r = await viaGateway(f.control, '/v1/responses', { model: 'gpt-4o', input: 'Hi' }, 'responses-gpt-4o');
    assert.equal(r.status, 200, r.text);
    assert.equal(r.toEngine[0].body.model, instance.apiModelId);
    const exact = await viaGateway(f.control, '/v1/chat/completions', { ...CHAT3_TURN1, model: instance.apiModelId }, 'exact-id');
    assert.equal(exact.status, 200);
  } finally { await f.cleanup(); }
});

test('two models loaded: an unknown name is still a 400 and names the IDs that work; an exact ID or an alias routes', async () => {
  const f = nativeFixture();
  try {
    const instances = await loaded(f, [['iq2_xs', [0]], ['iq3_s', [1]]]);
    assert.equal(instances.length, 2);
    const refused = await viaGateway(f.control, '/v1/chat/completions', CHAT3_TURN1, 'two-models-x');
    assert.equal(refused.status, 400);
    const error = JSON.parse(refused.text).error;
    assert.equal(error.param, 'model');
    assert.equal(error.type, 'invalid_request_error');
    assert.equal(refused.toEngine.length, 0, 'nothing reached an engine');
    assert.match(error.message, /^The model "x" is not loaded\. Loaded models: /);
    for (const i of instances) assert.ok(error.message.includes(i.apiModelId), `${i.apiModelId} is listed`);
    const exact = await viaGateway(f.control, '/v1/chat/completions', { ...CHAT3_TURN1, model: instances[1].apiModelId }, 'two-models-exact');
    assert.equal(exact.status, 200);
    assert.equal(exact.toEngine[0].body.model, instances[1].apiModelId);
    const noName = await viaGateway(f.control, '/v1/chat/completions', { messages: CHAT3_TURN1.messages }, 'two-models-no-name');
    assert.equal(noName.status, 400, 'no model named and two loaded: still not guessed');
  } finally { await f.cleanup(); }
});

test('the sentence for a control with nothing to list stays the engine\'s own', async () => {
  const { unknownModelMessage } = await import('../server/native-gateway.mjs');
  assert.equal(unknownModelMessage(new Error('Choose a loaded model ID.'), 'x', {}), 'Choose a loaded model ID.');
  assert.equal(unknownModelMessage(new Error('This model is not ready.'), 'x', { state: () => ({ instances: [{ status: 'ready', apiModelId: 'a' }, { status: 'ready', apiModelId: 'b' }] }) }), 'This model is not ready.');
  assert.equal(unknownModelMessage(new Error('Choose a loaded model ID.'), 'x', { state: () => ({ instances: [{ status: 'ready', apiModelId: 'a' }] }) }), 'Choose a loaded model ID.');
});
