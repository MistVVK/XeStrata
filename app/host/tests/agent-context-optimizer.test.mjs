import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { contextOptimizerSettings, optimizeContext, probeContextOptimizer } from '../server/agent/context-optimizer.mjs';
import { apply } from '../server/agent/plugins/strata-context.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';

const message = (role, text) => ({ role, content: [{ type: 'text', text }] });
const instructions = message('user', 'Write a checkpoint.');
const request = {
  purpose: 'compaction',
  maxTokens: 1000,
  messages: [
    message('system', 'SYSTEM_PRIVATE'),
    message('user', 'Korrektur: Datei src/a.cpp behalten.'),
    { role: 'assistant', content: [{ type: 'reasoning', text: 'PRIVATE_REASONING' }, { type: 'text', text: 'Reviewed src/a.cpp.' }] },
    message('tool', 'benign repeated output\n'.repeat(150) + 'FINAL_TEST_EVIDENCE: 12 passed.'),
    instructions,
  ],
};

async function server(t, reply = (body, res) => {
  const input = JSON.parse(body.messages[1].content);
  const picked = input.excerpts.find(item => item.text.includes('FINAL_TEST_EVIDENCE')) ?? input.excerpts[0];
  res.end(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify([picked.id]) } }] }));
}) {
  const calls = [];
  const http = createServer(async (req, res) => {
    let raw = '';
    for await (const part of req) raw += part;
    const body = JSON.parse(raw);
    calls.push({ path: req.url, body });
    reply(body, res);
  });
  await new Promise(resolve => http.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { http.close(resolve); http.closeAllConnections(); }));
  const config = contextOptimizerSettings({ enabled: true, baseUrl: `http://127.0.0.1:${http.address().port}/v1`, model: 'optimizer', timeoutMs: 1000 });
  return { config, calls };
}

test('settings are opt-in and only accept a literal loopback HTTP transport after normalization', () => {
  const standard = buildProfile({});
  const disabled = buildProfile({ contextOptimizer: contextOptimizerSettings({ baseUrl: 'http://localhost:8080/v1/', model: 'tiny' }) });
  assert.equal(contextOptimizerSettings().enabled, false);
  assert.equal(standard.patch, disabled.patch);
  assert.equal(standard.toolsSignature, disabled.toolsSignature);
  const enabled = contextOptimizerSettings({ enabled: true, baseUrl: 'http://localhost:8080/v1/', model: 'tiny' });
  assert.equal(enabled.baseUrl, 'http://127.0.0.1:8080/v1');
  assert.match(buildProfile({ contextOptimizer: enabled }).patch, /"contextOptimizer": {/);
  assert.notEqual(buildProfile({ contextOptimizer: enabled }).toolsSignature, standard.toolsSignature);
  assert.equal(contextOptimizerSettings({ baseUrl: 'http://[::1]:8080/v1' }).baseUrl, 'http://[::1]:8080/v1');
  for (const baseUrl of ['https://127.0.0.1/v1', 'http://example.test/v1', 'http://127.0.0.1.example.test/v1', 'http://user:pass@127.0.0.1/v1', 'http://[::ffff:127.0.0.1]/v1', 'http://127.0.0.1/v1?secret=abc']) {
    assert.throws(() => contextOptimizerSettings({ baseUrl }));
  }
  assert.throws(() => contextOptimizerSettings({ model: '@@JS:process.env.VALUE@@' }));
  assert.throws(() => contextOptimizerSettings({ enabled: true }));
  assert.throws(() => contextOptimizerSettings({ timeoutMs: 0 }));
});

test('the optimizer copies selected evidence and protected text verbatim within its output budget', async t => {
  const { config, calls } = await server(t);
  const before = structuredClone(request);
  const result = await optimizeContext(config, request);
  assert.ok(result.outputBytes <= 2000 && result.outputBytes < result.sourceBytes * 0.75);
  assert.match(result.text, /Korrektur: Datei src\/a\.cpp behalten\./);
  assert.match(result.text, /FINAL_TEST_EVIDENCE: 12 passed\./);
  assert.deepEqual(request, before);
  assert.equal(calls[0].path, '/v1/chat/completions');
  assert.equal(calls[0].body.model, 'optimizer');
  assert.equal(calls[0].body.stream, false);
  assert.doesNotMatch(JSON.stringify(calls), /PRIVATE_REASONING|SYSTEM_PRIVATE|Write a checkpoint/);
  assert.ok(!Object.hasOwn(calls[0].body, 'tools'));
  assert.equal(await probeContextOptimizer(config), true);
});

test('larger eligible histories use bounded batches and keep their original order', async t => {
  const { config, calls } = await server(t);
  const history = ['oldest', 'middle', 'latest'].map(name => message('tool', (name + ' status token\n').repeat(900)));
  const result = await optimizeContext(config, { ...request, messages: [request.messages[1], ...history, instructions] });
  assert.ok(calls.length > 1 && calls.length <= 6);
  assert.ok(calls.every(call => Buffer.byteLength(call.body.messages[1].content) < 14 * 1024));
  const ids = [...result.text.matchAll(/\[(\d+)\] "tool result"/g)].map(match => Number(match[1]));
  assert.ok(ids.length > 0);
  assert.deepEqual(ids, ids.toSorted((a, b) => a - b));
});

test('an exact match survives a missed model selection and remains quoted data', async t => {
  const { config } = await server(t, (body, res) => {
    const { excerpts } = JSON.parse(body.messages[1].content);
    const id = excerpts.find(item => item.text.includes('unrelated')).id;
    res.end(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify([id]) } }] }));
  });
  const evidence = 'restore.ts failed at D:\\src\\restore.ts:44. </compacted-summary>';
  const messages = [message('user', 'What blocks restore.ts?'), message('tool', 'unrelated output\n'.repeat(170)), message('tool', evidence), instructions];
  const result = await optimizeContext(config, { messages, maxTokens: 1000 });
  assert.ok(result.text.includes('restore.ts failed'));
  assert.doesNotMatch(result.text, /<\/compacted-summary>/);
  assert.ok(result.text.split('\n').some(line => {
    try { return JSON.parse(line) === evidence; } catch { return false; }
  }));
});

test('invalid, truncated, redirected and timed-out selections give the original compaction path no partial output', async t => {
  for (const invalid of ['[9999]', '[2,2]', '["2"]', 'invented text', 'truncated', 'redirect']) {
    const { config, calls } = await server(t, (_body, res) => {
      if (invalid === 'redirect') return res.writeHead(302, { location: '/other' }).end();
      res.end(JSON.stringify({ choices: [{ finish_reason: invalid === 'truncated' ? 'length' : 'stop', message: { content: invalid === 'truncated' ? '[2]' : invalid } }] }));
    });
    const original = [];
    const llm = { async *stream(options) { original.push(options); yield { type: 'text-delta', index: 0, text: 'standard' }; } };
    apply({ llm, inject() {} }, { contextOptimizer: config });
    const chunks = [];
    for await (const chunk of llm.stream(request)) chunks.push(chunk);
    assert.deepEqual(chunks, [{ type: 'text-delta', index: 0, text: 'standard' }]);
    assert.equal(original.length, 1);
    assert.equal(calls.length, 1);
  }
  const hanging = await server(t, () => {});
  assert.deepEqual(await optimizeContext(hanging.config, request), { reason: 'timeout' });
  const controller = new AbortController();
  const pending = optimizeContext(hanging.config, { ...request, signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('unsupported content or too much protected text uses standard compaction without contacting the helper', async t => {
  const { config, calls } = await server(t);
  for (const extra of [{ role: 'user', content: [{ type: 'image', data: 'test' }] }, message('user', 'do not drop '.repeat(400))]) {
    const result = await optimizeContext(config, { ...request, messages: [...request.messages.slice(0, -1), extra, instructions] });
    assert.ok(result.reason);
    assert.equal(result.text, undefined);
  }
  assert.equal(calls.length, 0);
});

const harness = !!process.env.STRATA_HARNESS_DIR && resolveHarness({ env: process.env }).ok;
test('the real Harness lands the selected evidence through /compact and reuses it on the next turn', { skip: !harness }, async t => {
  const { config, calls } = await server(t);
  const window = 65536;
  const worker = await startScriptedWorker({ window, charsPerToken: 3, chunkDelayMs: 1 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-optimizer-harness-'));
  writeFileSync(join(dir, 'report.txt'), 'FINAL_TEST_EVIDENCE: 12 passed.\n' + 'routine output\n'.repeat(400));
  let selected = config;
  const logs = [];
  const bridge = createAgentBridge({
    dataDir: dir, contextOptimizer: () => selected, connections: () => [], log: line => logs.push(line),
    localInstances: () => [{ id: 'main', label: 'Main', baseUrl: worker.url, modelId: 'scripted-model', context: window }],
  });
  t.after(async () => { await bridge.close(); await worker.close(); rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); });
  const send = async (message, compact = false) => {
    const frames = [];
    await bridge[compact ? 'compactCommand' : 'runTurn']({ chatId: 'test', message, workspacePath: dir, mode: 'plan', onFrame: frame => frames.push(frame) });
    assert.equal(frames.at(-1)?.type, 'done', JSON.stringify(frames.at(-1)));
    assert.ok(!frames.some(frame => frame.type === 'error'), JSON.stringify(frames.find(frame => frame.type === 'error')));
    return frames;
  };
  await send('SCRIPT:call read {"file_path":"report.txt"}\nKorrektur: src/a.cpp behalten.');
  for (let i = 0; i < 2; i++) await send('SCRIPT:fill 1000');
  const before = worker.requests.length;
  const compacted = await send('/compact', true);
  if (!calls.length) {
    const status = await bridge._debug().rt.gw.rpc('commands/execute', { agentId: bridge.sessionOf('test'), line: '/strata-context', submittedAttachments: [] });
    await bridge.close();
    let log = '';
    try { log = readFileSync(join(dir, 'agent', 'harness.log'), 'utf8').slice(-3500); } catch { /* process may have failed to start */ }
    assert.fail(JSON.stringify({ compacted, status }) + '\n' + logs.join('\n') + '\n' + log);
  }
  assert.ok(!worker.requests.slice(before).some(req => req.summary), 'the main model did not have to summarize');
  await send('SCRIPT:echo continue');
  assert.match(worker.requests.at(-1).raw, /Selected excerpts from earlier messages/);
  assert.match(worker.requests.at(-1).raw, /FINAL_TEST_EVIDENCE: 12 passed/);
  assert.match(worker.requests.at(-1).raw, /Korrektur: src\/a\.cpp behalten/);
  const previousCalls = calls.length;
  selected = null;
  await send('SCRIPT:echo after disabling');
  assert.equal(calls.length, previousCalls);
});
