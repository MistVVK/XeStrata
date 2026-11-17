// #5661: the Serving switch "Tools for apps on this computer", one per model row, off by default. It writes serving.mcpTools through
// PATCH /api/serving; turning it off removes the model; the list stays at 64 models; without an API key it is off, disabled and says
// "Create an API key first.".
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const home = mkdtempSync(join(tmpdir(), 'strata-mcp-switch-'));
process.env.STRATA_HOME = home;
const serving = await import('../server/serving.mjs');
test.after(() => { try { rmSync(home, { recursive: true, force: true }); } catch { /* temp folder */ } });
const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/mcp-tools-switch.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' });
const { mcpToolsSwitch, mcpToolsPatch } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const MODEL = 'qwen3.8-flash';
serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });

test('without an API key the switch is off and disabled and asks for a key first', () => {
  serving.patch({ apiKey: null });
  assert.deepEqual(mcpToolsSwitch(serving.state(), MODEL), { checked: false, disabled: true, hint: 'needs-key' });
  assert.deepEqual(mcpToolsSwitch(null, MODEL), { checked: false, disabled: true, hint: 'needs-key' });
  serving.patch({ mcpTools: { model: MODEL, on: true } });   // turned on earlier, then the key was removed: shown off (the guard refuses too)
  assert.equal(mcpToolsSwitch(serving.state(), MODEL).checked, false);
  serving.patch(mcpToolsPatch(MODEL, false));
});

test('with a key it is off by default; the switch writes serving.mcpTools and turning it off removes the model', () => {
  serving.patch({ apiKey: 'sk-strata-switch-test' });
  assert.deepEqual(mcpToolsSwitch(serving.state(), MODEL), { checked: false, disabled: false, hint: 'help' });
  assert.deepEqual(mcpToolsPatch(MODEL, true), { mcpTools: { model: MODEL, on: true } });
  let s = serving.patch(mcpToolsPatch(MODEL, true));
  assert.deepEqual(s.mcpTools, [MODEL]);
  assert.equal(mcpToolsSwitch(s, MODEL).checked, true);
  assert.equal(mcpToolsSwitch(s, 'other').checked, false, 'per model');
  s = serving.patch(mcpToolsPatch('other', true));
  s = serving.patch(mcpToolsPatch(MODEL, false));
  assert.deepEqual(s.mcpTools, ['other']);
  assert.equal(mcpToolsSwitch(s, MODEL).checked, false);
  serving.patch(mcpToolsPatch('other', false));
});

test('the list holds at most 64 models', () => {
  for (let i = 0; i < 70; i++) serving.patch(mcpToolsPatch(`m${i}`, true));
  const list = serving.state().mcpTools;
  assert.equal(list.length, 64);
  assert.equal(serving.MCP_TOOL_MODELS_MAX, 64);
  assert.ok(list.includes('m69') && !list.includes('m0'), 'the newest choices are kept');
});

test('each Serving model row has the switch, wired to setMcpTools, with the approved words', () => {
  const screen = readFileSync(fileURLToPath(new URL('../src/screens/Serving.tsx', import.meta.url)), 'utf8');
  assert.match(screen, /const toolsSwitch = \(apiModelId: string, bare = false\) => \{ const sw = mcpToolsSwitch\(serving, apiModelId\);/);
  assert.match(screen, /\{toolsSwitch\(i\.apiModelId\)\}/);
  assert.match(screen, /<input type="checkbox" role="switch" aria-label=\{tr\("Tools for apps on this computer"\)\} checked=\{sw\.checked\} disabled=\{sw\.disabled \|\| pending\} onChange=\{e => void operation\(\(\) => setMcpTools\(apiModelId, e\.target\.checked\)\)\}\/>\{bare \? null : tr\("Tools for apps on this computer"\)\}/);
  assert.match(screen, /sw\.hint === 'help' \? tr\("When this is on, an app on this computer that sends your API key can let this model use its tools, such as reading and changing files\. Apps on other devices or through the internet link never can\."\) : tr\("Create an API key first\."\)/);
  const store = readFileSync(fileURLToPath(new URL('../src/lib/store.ts', import.meta.url)), 'utf8');
  assert.match(store, /export async function setMcpTools\(model: string, on: boolean\) \{\s*const result = await api\.patch<ServingState>\('\/api\/serving', mcpToolsPatch\(model, on\)\);/);
});
