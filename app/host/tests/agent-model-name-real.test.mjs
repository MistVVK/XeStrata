// #5971 on the real Harness (STRATA_HARNESS_DIR): the system prompt names the model the chat runs on, by the name the person knows, and
// follows a switch to another model. Strix, 11-16, packaged app: the request went to the chosen API model while the prompt said
// "You are running as deepseek-flash" (the agent's creation default); this runs the same two routes against a scripted model server.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('the prompt says the chosen model by name, on a local model and after a switch to an API model', { skip, timeout: 240_000 }, async (t) => {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-model-name-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({
    dataDir: dir, env: process.env, startTimeoutMs: 90_000,
    localInstances: () => [{ id: 'i', label: 'Qwen3.8 Flash Next', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }],
    connections: () => [{ id: 'smoke-conn', name: 'Smoke API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'smoke-id', name: 'Smoke Model 7B', contextMax: 32768 }] }],
    tools: () => ({ off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: true }),
  });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  // the persona sentence inside the request's JSON: it ends at its full stop before an escaped newline, a quote or a space
  const runningAs = () => /You are running as (.+?)\.(?=\\n|\\"|"| )/.exec(worker.requests.at(-1)?.raw ?? '')?.[1] ?? null;

  await bridge.runTurn({ chatId: 'c1', message: 'SCRIPT:hello', workspacePath: ws, mode: 'manual', onFrame: () => {} });
  assert.equal(worker.requests.at(-1)?.model, 'scripted-model');
  assert.equal(runningAs(), 'Qwen3.8 Flash Next', 'the local model by its name');

  await bridge.runTurn({ chatId: 'c1', message: 'SCRIPT:hello', workspacePath: ws, mode: 'manual', choice: { connectionId: 'smoke-conn', modelId: 'smoke-id' }, onFrame: () => {} });
  assert.equal(worker.requests.at(-1)?.model, 'smoke-id', 'the request went to the chosen API model');
  assert.equal(runningAs(), 'Smoke Model 7B', 'and the prompt says so after the switch');
});
