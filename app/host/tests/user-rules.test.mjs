// #5084: the person's own rules for every chat. Settings saves them to <data>/agent/home/AGENTS.md (the file the Harness reads in agent
// mode); the plain chat paths add the same text to the system prompt. With no rules the prompt is byte-identical to the one before.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:net';
import { createUserRules, rulesFile, RULES_MAX_BYTES } from '../server/user-rules.mjs';
import { createMockModelApi } from './mock-model-api.mjs';
import { ROOMY_HOST } from './helpers/hardware.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

function scratch(t) { const dir = mkdtempSync(join(tmpdir(), 'strata-user-rules-')); t.after(() => rmSync(dir, { recursive: true, force: true })); return dir; }

test('the rules live in the AGENTS.md the Harness reads, and a save round-trips', t => {
  const data = scratch(t), file = rulesFile(data);
  assert.equal(file, join(data, 'agent', 'home', 'AGENTS.md'));
  const rules = createUserRules({ file });
  const empty = rules.read();
  assert.equal(empty.text, ''); assert.equal(empty.maxBytes, RULES_MAX_BYTES);
  const saved = rules.write('Answer in German.\r\nKeep answers short.', empty.revision);
  assert.equal(readFileSync(file, 'utf8'), 'Answer in German.\nKeep answers short.');
  assert.deepEqual(rules.read(), saved);
  assert.notEqual(saved.revision, empty.revision);
});

test('a save over a file another window changed is refused and changes nothing', t => {
  const data = scratch(t), file = rulesFile(data), rules = createUserRules({ file });
  const first = rules.read();
  rules.write('One', first.revision);                                    // window A saves
  assert.throws(() => rules.write('Two', first.revision), e => e.status === 409 && e.code === 'changed');   // window B still holds the old revision
  assert.equal(readFileSync(file, 'utf8'), 'One');
  writeFileSync(file, 'edited by hand');                                 // a change outside Strata counts too
  assert.throws(() => rules.write('Three', rules.read().revision.replace(/./, c => (c === '0' ? '1' : '0'))), e => e.status === 409);
  assert.equal(rules.read().text, 'edited by hand');
});

test('the 64 KB limit counts UTF-8 bytes, not characters', t => {
  const rules = createUserRules({ file: rulesFile(scratch(t)) });
  const fits = 'x'.repeat(RULES_MAX_BYTES), rev = rules.read().revision;
  const next = rules.write(fits, rev);
  assert.equal(next.bytes, RULES_MAX_BYTES);
  const han = '中'.repeat(21846);                                        // 21,846 characters, 65,538 bytes
  assert.ok(han.length < RULES_MAX_BYTES);
  assert.throws(() => rules.write(han, next.revision), e => e.status === 413 && e.code === 'too-long');
  assert.equal(rules.read().text, fits);
});

test('no rules add nothing to the prompt; rules add one block', t => {
  const data = scratch(t), file = rulesFile(data), rules = createUserRules({ file });
  const compose = () => ['SYSTEM', rules.promptBlock(), 'SKILLS'].filter(Boolean).join('\n\n');
  assert.equal(compose(), 'SYSTEM\n\nSKILLS');                         // no file
  mkdirSync(join(data, 'agent', 'home'), { recursive: true }); writeFileSync(file, ' \n\t\n');
  assert.equal(compose(), 'SYSTEM\n\nSKILLS');                         // only blank lines
  writeFileSync(file, '﻿Answer in German.\n');
  assert.equal(rules.promptBlock(), 'The person you are helping set these rules for every answer:\nAnswer in German.');
});

test('every plain chat path of the host adds the rules block', () => {
  const src = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.equal((src.match(/userRules\.promptBlock\(\)/g) ?? []).length, 3);
  assert.equal((src.match(/\[SYSTEM_PROMPT, *skillInstructions/g) ?? []).length, 0);
});

async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

test('the host route saves the rules and the next chat request carries them; cleared rules give the old prompt byte for byte', { timeout: 60000 }, async t => {
  const mock = await createMockModelApi(), home = mkdtempSync(join(tmpdir(), 'strata-user-rules-host-'));
  let child, exited;
  t.after(async () => { if (child && child.exitCode === null) { child.kill(); await exited; } await mock.close(); rmSync(home, { recursive: true, force: true, maxRetries: 3 }); });
  mkdirSync(join(home, 'data'));
  writeFileSync(join(home, 'data', 'model-providers.json'), JSON.stringify([{ id: 'fixture', kind: 'custom', name: 'Fixture', baseUrl: mock.baseUrl, protocol: 'openai-chat', apiKey: 'fixture-key', models: [{ id: 'plain', reasoning: false }] }]));
  const port = await freePort(), base = `http://127.0.0.1:${port}`;
  const env = { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_WORKSPACE: join(home, 'work'), STRATA_SETTINGS_FILE: join(home, 'settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'), STRATA_CONTROL_DIR: join(home, 'no-control'), STRATA_HARNESS_DIR: join(home, 'no-harness'), STRATA_NATIVE_REPO: join(home, 'no-repo'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_BIND: '127.0.0.1', STRATA_REAL_ENGINE: '0', STRATA_TEST_HARDWARE: ROOMY_HOST, STRATA_AGENT: 'off', STRATA_NO_ACQUIRE: '1' };
  child = spawn(process.execPath, ['server/index.mjs'], { cwd: join(import.meta.dirname, '..'), windowsHide: true, stdio: 'ignore', env });
  exited = once(child, 'exit');
  let up = false; for (let n = 0; n < 150 && !up; n++) { try { up = (await fetch(base + '/api/health')).ok; } catch { /* starting */ } if (!up) await new Promise(r => setTimeout(r, 100)); }
  assert.ok(up, 'fixture host started');
  const ask = async () => {
    const res = await fetch(base + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message: 'Hello', history: [], agent: false, useTools: false, mode: 'bypass', remoteModel: { connectionId: 'fixture', modelId: 'plain' } }) });
    assert.equal(res.status, 200); await res.text();
    return mock.requests.filter(r => r.path === '/v1/chat/completions').at(-1).body.messages[0];
  };
  const put = (text, revision) => fetch(base + '/api/rules', { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text, revision }) });
  const before = await ask();
  assert.equal(before.role, 'system');
  const first = await (await fetch(base + '/api/rules')).json();
  assert.equal(first.text, '');
  const saved = await put('Answer in German.', first.revision);
  assert.equal(saved.status, 200);
  assert.equal(readFileSync(join(home, 'data', 'agent', 'home', 'AGENTS.md'), 'utf8'), 'Answer in German.');
  const withRules = await ask();
  assert.ok(withRules.content.startsWith(before.content + '\n\n'), withRules.content);
  assert.ok(withRules.content.includes('Answer in German.'));
  const stale = await put('Something else', first.revision);
  assert.equal(stale.status, 409); assert.equal((await stale.json()).code, 'changed');
  const tooLong = await put('x'.repeat(RULES_MAX_BYTES + 1), (await saved.json()).revision);
  assert.equal(tooLong.status, 413);
  const now = await (await fetch(base + '/api/rules')).json();
  assert.equal((await put('', now.revision)).status, 200);
  assert.ok(existsSync(join(home, 'data', 'agent', 'home', 'AGENTS.md')));
  assert.deepEqual(await ask(), before);                                  // cleared rules: the same system message, byte for byte
});

test('the real Harness puts the saved rules into the request it sends to the model (agent mode)', { skip, timeout: 180000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-user-rules-agent-')), ws = join(dir, 'work');
  mkdirSync(ws, { recursive: true });
  const worker = await startScriptedWorker({ chunkDelayMs: 0 });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 65536, slots: 1 }], connections: () => [], env: process.env, startTimeoutMs: 120_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const rules = createUserRules({ file: rulesFile(dir) });
  rules.write('RULE-5084: always end with the word Ahoy.', rules.read().revision);
  const frames = [];
  await bridge.runTurn({ chatId: 'c', message: 'SCRIPT:echo hello', workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
  assert.equal(frames.at(-1).type, 'done', JSON.stringify(frames.at(-1)));
  const sent = worker.requests.filter(r => r.directive === 'echo');
  assert.ok(sent.length, 'the turn reached the model');
  assert.ok(sent.every(r => r.raw.includes('RULE-5084: always end with the word Ahoy.')), 'every model request of the turn carries the rules');
});
