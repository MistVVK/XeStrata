import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { apply, workingBrief } from '../server/agent/plugins/strata-working-memory.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

const event = (type, seq, data) => ({ type, seq, time: Date.UTC(2026, 10, 19) + seq, data });
const user = text => ({ role: 'user', source: { kind: 'user', rpcId: 'request' }, content: [{ type: 'text', text }] });
const recall = text => ({ ...user(text), source: { kind: 'strata-working-memory' } });
const sample = [
  event('user/message', 1, user('Update the report after approval.')),
  event('tool/call', 2, { callId: 'r1', name: 'read', arguments: '{"file_path":"report.md"}' }),
  event('tool/result', 3, { message: { toolCallId: 'r1', isError: false, content: [{ type: 'text', text: 'private file contents' }] } }),
  event('todo/write', 4, { todos: [{ content: 'Wait for approval', status: 'pending' }] }),
  event('user/message', 5, user('Keep the current heading.')),
  event('user/message', 6, recall('Do not remember this memory.')),
  event('tool/call', 7, { callId: 'j1', name: 'pwsh', arguments: { description: 'Run report checks', command: 'private command text' } }),
  event('turn/end', 8, { reason: { kind: 'aborted' } }),
];

test('the brief uses time-stamped prior work and corrections without replaying the tool data', () => {
  const brief = workingBrief(sample);
  assert.match(brief, /Update the report after approval/);
  assert.match(brief, /Keep the current heading/);
  assert.match(brief, /Open task .*Wait for approval/);
  assert.match(brief, /Last recorded turn .*aborted/);
  assert.match(brief, /Tool call without a recorded result .*Run report checks/);
  assert.match(brief, /Earlier tool outcome .*report\.md.*"seq":3.*2026-11-19/);
  assert.doesNotMatch(brief, /private|Do not remember this memory/);
  assert.equal(workingBrief([sample[0]]), '', 'ordinary conversation has no working brief');
});

test('the brief stays bounded for noisy, multibyte and legacy event logs', () => {
  const work = [];
  for (let n = 0; n < 1200; n++) {
    work.push(event('tool/call', n * 2, { callId: String(n), name: 'edit', arguments: { file_path: '測'.repeat(390) + n } }));
    work.push(event('tool/result', n * 2 + 1, { message: { toolCallId: String(n), isError: n % 2 === 0 } }));
  }
  work.push(event('tool/call', 10_000, { callId: 'legacy', name: 'read', arguments: '[malformed' }));
  const brief = workingBrief([sample[0], sample[3], ...work]);
  assert.match(brief, /Update the report after approval/);
  assert.match(brief, /Wait for approval/);
  assert(Buffer.byteLength(brief, 'utf8') <= 3000);
  assert.match(brief, /"result":"returned"/);
  assert.match(brief, /Earlier tool outcome/);
});

const fileHistory = (...paths) => paths.flatMap((path, i) => [
  event('tool/call', i * 2 + 1, { callId: String(i), name: 'read', arguments: { file_path: path } }),
  event('tool/result', i * 2 + 2, { message: { toolCallId: String(i), isError: false } }),
]);

test('the current request can recall an older file and omit newer unrelated results', () => {
  const files = fileHistory('docs/active spec.md', ...Array.from({ length: 8 }, (_, i) => 'notes/noise-' + i + '.txt'));
  const previous = workingBrief(files);
  const focused = workingBrief(files, [], ['Check @"docs/active spec.md" again.']);
  assert.doesNotMatch(previous, /active spec/);
  assert.match(focused, /Earlier tool outcome .*docs\/active spec\.md/);
  assert.doesNotMatch(focused, /notes\/noise/);
  assert(Buffer.byteLength(focused) < Buffer.byteLength(previous));
});

test('an active task supplies context for a continuation, with ambiguous names left unassigned', () => {
  const files = fileHistory('src/policy.mjs', 'app/main.mjs', 'tests/main.mjs', 'other.txt');
  const checklist = event('todo/write', 20, { todos: [
    { content: 'Later review other.txt', status: 'pending' },
    { content: 'Finish src/policy.mjs', status: 'in_progress' },
  ] });
  const continued = workingBrief([...files, checklist], [], ['Continue.']);
  assert.match(continued, /Earlier tool outcome .*src\/policy\.mjs/);
  assert.doesNotMatch(continued, /Earlier tool outcome .*other\.txt/);
  assert.match(workingBrief(files, [], ['Check main.mjs.']), /Earlier tool outcome .*other\.txt/);
  const explicit = workingBrief([...files, checklist], [], ['Focus on tests/main.mjs instead.']);
  assert.match(explicit, /Earlier tool outcome .*tests\/main\.mjs/);
  assert.doesNotMatch(explicit, /Earlier tool outcome .*src\/policy\.mjs/);
});

test('the plugin only adds a brief to an existing human turn in its own session', async () => {
  let hook;
  const jobs = { list: id => [
    { id: 'job-own', owner: id, status: 'running', label: 'Build reports' },
    { id: 'job-other', owner: 'different-chat', status: 'running', label: 'Foreign work' },
    { id: 'job-done', owner: id, status: 'done', label: 'Finished work' },
  ] };
  apply({ jobs, on: (name, fn) => { assert.equal(name, 'agent/pre-step'); hook = fn; } });
  const request = user('Continue.');
  const input = { agent: { session: { id: 'one-chat', header: {}, log: sample } }, messages: [request], step: 1, signal: new AbortController().signal };
  const allow = messages => async () => ({ kind: 'enter', messages });
  const projected = await hook(input, allow([request]));
  assert.equal(projected.messages.length, 2);
  assert.equal(projected.messages[0].source.kind, 'strata-working-memory');
  assert.equal(projected.messages[1], request);
  assert.match(projected.messages[0].content[0].text, /Active job in this session .*job-own/);
  assert.doesNotMatch(projected.messages[0].content[0].text, /job-other|job-done/);
  assert.deepEqual(await hook({ ...input, step: 2 }, allow([])), { kind: 'enter', messages: [] });
  assert.deepEqual(await hook(input, allow([recall('old')])), { kind: 'enter', messages: [recall('old')] });
  assert.deepEqual(await hook({ ...input, agent: { session: { header: { parentSession: 'parent' }, log: sample } } }, allow([request])), { kind: 'enter', messages: [request] });
});

test('the unchanged default profile mounts the plugin only when explicitly enabled', () => {
  const baseline = buildProfile({});
  assert.deepEqual(baseline, buildProfile({ workingMemory: false }));
  assert.doesNotMatch(baseline.patch, /strata-working-memory/);
  assert.doesNotMatch(baseline.patch, /strata-edit-recovery/);
  assert.match(buildProfile({ workingMemory: true }).patch, /strata-working-memory/);
  assert.match(buildProfile({ workingMemory: true }).patch, /strata-edit-recovery/);
});


const hasHarness = process.env.STRATA_HARNESS_DIR && resolveHarness({ env: process.env }).ok;

test('the packaged Harness recalls work across restarts and requires a fresh Read after a stale edit', { skip: !hasHarness, timeout: 360_000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-working-memory-'));
  const workspace = join(dir, 'workspace');
  const target = join(workspace, 'target.txt');
  mkdirSync(workspace);
  writeFileSync(target, 'owner=first\nstatus=waiting\n');
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const newBridge = () => createAgentBridge({
    dataDir: join(dir, 'data'),
    localInstances: () => [{ id: 'local', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }],
    connections: () => [],
    env: { ...process.env, STRATA_AGENT_WORKING_MEMORY: '1' },
  });
  let bridge = newBridge();
  t.after(async () => {
    await bridge.close();
    await worker.close();
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  });
  const run = async message => {
    const frames = [];
    await bridge.runTurn({ chatId: 'workspace-chat', message, workspacePath: workspace, mode: 'bypass', onFrame: frame => frames.push(frame) });
    assert.equal(frames.at(-1)?.type, 'done');
    return frames.filter(frame => frame.type === 'agent-tool').map(frame => frame.tool);
  };
  assert((await run('SCRIPT:read target.txt')).some(tool => tool.name === 'read' && tool.state === 'done'));
  const sessionId = bridge.sessionOf('workspace-chat');
  const pid = bridge.status().runtime.pid;
  await bridge.close();
  writeFileSync(target, 'owner=externally-changed\nstatus=waiting\n');
  bridge = newBridge();

  // The original anchor is deliberately still present. A literal match alone
  // would otherwise accept this edit without observing the new file.
  const edit = 'SCRIPT:call edit {"file_path":"target.txt","old_string":"status=waiting","new_string":"status=done"}';
  const requestMark = worker.requests.length;
  const firstEdit = await run(edit);
  assert.equal(bridge.sessionOf('workspace-chat'), sessionId);
  assert.notEqual(bridge.status().runtime.pid, pid);
  assert(worker.requests.slice(requestMark).some(request => request.raw.includes('Strata working state') && request.raw.includes('Earlier tool outcome')));
  assert(firstEdit.some(tool => tool.name === 'edit' && tool.state === 'error'));
  assert.equal(readFileSync(target, 'utf8'), 'owner=externally-changed\nstatus=waiting\n');

  assert((await run('SCRIPT:read target.txt')).some(tool => tool.name === 'read' && tool.state === 'done'));
  assert((await run(edit)).some(tool => tool.name === 'edit' && tool.state === 'done'));
  assert.equal(readFileSync(target, 'utf8'), 'owner=externally-changed\nstatus=done\n');

  const stale = 'SCRIPT:call edit {"file_path":"target.txt","old_string":"status=waiting","new_string":"status=approved"}';
  const recovered = 'SCRIPT:call edit {"file_path":"target.txt","old_string":"status=done","new_string":"status=approved"}';
  const beforeMiss = worker.requests.length;
  assert((await run(stale)).some(tool => tool.name === 'edit' && tool.state === 'error'));
  assert(worker.requests.slice(beforeMiss).some(request => request.raw.includes('old_string was not found') && request.raw.includes('Use Read on this file now')));
  await bridge.close();
  bridge = newBridge();
  const beforeRetry = worker.requests.length;
  assert((await run(recovered)).some(tool => tool.name === 'edit' && !['done', 'running'].includes(tool.state)));
  assert(worker.requests.slice(beforeRetry).some(request => request.raw.includes('A previous edit could not find its original text')));
  assert.equal(readFileSync(target, 'utf8'), 'owner=externally-changed\nstatus=done\n');
  assert((await run('SCRIPT:read target.txt')).some(tool => tool.name === 'read' && tool.state === 'done'));
  assert((await run(recovered)).some(tool => tool.name === 'edit' && tool.state === 'done'));
  assert.equal(readFileSync(target, 'utf8'), 'owner=externally-changed\nstatus=approved\n');
});
