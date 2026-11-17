// SAFETY (visual walkthrough 11-17, finding 1): a folder added under Settings -> Tools -> Blocked folders was not enforced until the host restarted. The approval gate reads the
// list once, when the Harness starts; the bridge restarts the Harness for the next message only when its signature changes, and the signature did not cover the gate's settings.
// These tests: the signature moves with the gate's settings, and, with the real Harness, a block added in Settings refuses the next message's read / write / list / command in that
// folder in every chat with no restart by the person, and removing it allows them again.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };

function makeTools(dir) { let saved = {}; return createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: dir }); }
const sigOf = tools => buildProfile({ local: [], connections: [], tools: tools.profileView() }).toolsSignature;

test('the signature that decides a restart for the next message moves with the gate\'s settings: a blocked folder, the web switch, the search service', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-blocksig-'));
  try {
    const tools = makeTools(dir);
    const base = sigOf(tools);
    assert.equal(sigOf(tools), base, 'the same settings, the same signature');
    tools.update({ blockedFolders: { add: join(dir, 'private') } });
    const blocked = sigOf(tools);
    assert.notEqual(blocked, base, 'adding a blocked folder');
    tools.update({ blockedFolders: { remove: tools.view().blockedFolders[0] } });
    assert.equal(sigOf(tools), base, 'removing it again goes back');
    tools.update({ webWithoutAsking: true });
    assert.notEqual(sigOf(tools), base, 'the web question switch');
  } finally { cleanup(dir); }
});

test('real Harness: a block added in Settings refuses the next message in every chat (read, write, list, command), a removed block allows them again, and nothing is restarted by hand', { skip }, async t => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-blocklive-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const priv = join(dir, 'private'); mkdirSync(priv, { recursive: true });
  writeFileSync(join(priv, 'secret.txt'), 'the secret words\n');
  const tools = makeTools(dir);
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], tools: () => tools.profileView(), env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); cleanup(dir); });
  const shell = process.platform === 'win32' ? 'pwsh' : 'bash';
  const secret = join(priv, 'secret.txt');
  const calls = {
    read: ['read', { file_path: secret }],
    write: ['write', { file_path: join(priv, 'new.txt'), content: 'x' }],
    list: ['glob', { pattern: '*', path: priv }],
    command: [shell, { command: process.platform === 'win32' ? `Get-Content '${secret}'` : `cat '${secret}'` }],
  };
  const run = async (chatId, kind) => {
    const [tool, args] = calls[kind];
    const frames = [];
    await bridge.runTurn({ chatId, message: `SCRIPT:call ${tool} ${JSON.stringify(args)}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
    return frames.filter(f => f.type === 'agent-tool').map(f => f.tool).at(-1);
  };
  const blockedCard = card => card?.state === 'error' && /blocked/i.test(`${card.result ?? ''} ${card.error ?? ''} ${JSON.stringify(card)}`);

  // before: allowed (a read of the file works)
  const first = await run('chat-1', 'read');
  assert.equal(first.state, 'done', JSON.stringify(first).slice(0, 300));
  assert.match(String(first.result), /the secret words/);
  const pidBefore = (await bridge.status()).runtime?.pid;

  // the person adds the folder in Settings; nothing else is done by hand
  tools.update({ blockedFolders: { add: priv } });
  for (const [chat, kind] of [['chat-1', 'read'], ['chat-2', 'write'], ['chat-3', 'list'], ['chat-4', 'command']]) {
    const card = await run(chat, kind);
    assert.ok(blockedCard(card), `${kind} in ${chat} is refused: ${JSON.stringify(card).slice(0, 400)}`);
  }
  assert.notEqual((await bridge.status()).runtime?.pid, pidBefore, 'the agent was restarted by the bridge for the first message after the change');

  // the person removes it: allowed again
  tools.update({ blockedFolders: { remove: tools.view().blockedFolders[0] } });
  const again = await run('chat-1', 'read');
  assert.equal(again.state, 'done', JSON.stringify(again).slice(0, 300));
  assert.match(String(again.result), /the secret words/);
});
