/**
 * #5968: "Run as administrator" (Windows only, off by default). Bypass mode never gave Windows administrator rights and the agent had no
 * way to ask for them. The tool asks Windows for one command at a time (the consent prompt every call); Manual asks first, Plan refuses,
 * choosing No at the Windows prompt is reported as refused (and the reply says so), and a command whose PowerShell errors were only
 * non-terminating is not reported as done. Nothing here starts an elevated process: the launcher is replaced in the call tests.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { CANCELLED, NOT_STARTED, REFUSED, TOOL, innerScript, launcherScript, resultText, runAsAdministrator } from '../server/agent/admin-tool-mcp.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { buildProfile, ADMIN_TOOL_SCRIPT } from '../server/agent/profile.mjs';
import { manualDecision, planDecision } from '../server/agent/plugins/strata-gate.mjs';
import { TurnProjector } from '../server/agent/events.mjs';

const NAME = `mcp__strata-admin__${TOOL}`;
const entriesOf = profile => JSON.parse(profile.patch.split('\n').slice(1).join('\n'));
const base = { servers: {}, hasInstructionTools: false, hasCommandTools: false, toolFile: null, browserEnabled: false, search: { provider: 'builtin', key: '' }, off: new Set() };
const adminRow = profile => JSON.stringify(entriesOf(profile)).includes('mcp-strata-admin');

test('the setting is off by default, offered on Windows only, and reaches the agent only when on there', () => {
  const make = (platform, state = {}) => { let saved = state; return createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: mkdtempSync(join(tmpdir(), 'strata-admin-set-')), platform }); };
  assert.deepEqual(make('win32').view().runAsAdmin, { enabled: false, available: true });
  assert.equal(make('win32').profileView().runAsAdminEnabled, false, 'off unless the person turned it on');
  const win = make('win32');
  assert.deepEqual(win.update({ runAsAdmin: { enabled: true } }).runAsAdmin, { enabled: true, available: true });
  assert.equal(win.profileView().runAsAdminEnabled, true);
  assert.deepEqual(make('linux', { runAsAdmin: true }).view().runAsAdmin, { enabled: true, available: false });
  assert.equal(make('linux', { runAsAdmin: true }).profileView().runAsAdminEnabled, false, 'never outside Windows');
  assert.equal(make('win32', { runAsAdmin: 'yes' }).profileView().runAsAdminEnabled, false, 'only a real true turns it on');
});

test('the profile adds the tool server only when the setting is on', () => {
  assert.equal(adminRow(buildProfile({ local: [], connections: [], tools: { ...base, runAsAdminEnabled: false } })), false);
  assert.equal(adminRow(buildProfile({ local: [], connections: [] })), false, 'no settings: no tool');
  const on = entriesOf(buildProfile({ local: [], connections: [], tools: { ...base, runAsAdminEnabled: true } }));
  const row = JSON.stringify(on);
  assert.ok(row.includes('mcp-strata-admin') && row.includes('"serverName":"strata-admin"'), 'the server is there');
  assert.ok(ADMIN_TOOL_SCRIPT.endsWith('admin-tool-mcp.mjs') && row.includes('admin-tool-mcp.mjs'), 'it runs the app\'s own admin tool script');
});

test('Manual asks first with the command in the card; Plan refuses; the Windows prompt still comes in Bypass', () => {
  const asked = manualDecision({ name: NAME, arguments: { command: 'New-Item HKLM:\\SOFTWARE\\X', reason: 'a machine-wide value' } });
  assert.equal(asked.kind, 'ask');
  assert.match(asked.reason, /^Run as administrator \(Windows will ask too\): New-Item HKLM:\\SOFTWARE\\X \(a machine-wide value\)$/);
  assert.equal(planDecision({ name: NAME, arguments: {} }).kind, 'deny');
});

test('the command reaches the elevated PowerShell in a file, never on a command line, and non-terminating errors count as errors', () => {
  const inner = innerScript("Set-ItemProperty -Path 'HKLM:\\X' -Name Y -Value 1", 'C:\\t\\output.txt', 'C:\\t\\status.json');
  assert.match(inner, /\$Error\.Clear\(\)/);
  assert.match(inner, /\$failed = \(\$Error\.Count -gt 0\) -or \(\$exit -ne 0\)/, 'an error PowerShell only wrote out still makes it fail');
  assert.ok(inner.includes("Set-ItemProperty -Path 'HKLM:\\X' -Name Y -Value 1"));
  assert.ok(inner.includes("Set-Content -LiteralPath 'C:\\t\\output.txt'"));
  const launcher = launcherScript("C:\\Users\\o'neil\\AppData\\Local\\Temp\\strata-admin-1\\command.ps1");
  assert.match(launcher, /-Verb RunAs/);
  assert.match(launcher, new RegExp(`NativeErrorCode -eq ${CANCELLED}\\) \\{ exit ${CANCELLED} \\}`));
  assert.ok(launcher.includes("'\"C:\\Users\\o''neil\\AppData\\Local\\Temp\\strata-admin-1\\command.ps1\"'"), 'the path is quoted for PowerShell');
  assert.ok(!launcher.includes('Set-ItemProperty'));
});

test('what the model reads: done, errors, No at the prompt, not started, still running', () => {
  assert.deepEqual(resultText({ code: 0, output: 'wrote it\r\n', status: { failed: false, exit: 0 } }), { ok: true, text: 'Ran as administrator (exit code 0).\nwrote it' });
  const errors = resultText({ code: 1, output: 'Access denied', status: { failed: true, exit: 0 } });
  assert.equal(errors.ok, false); assert.match(errors.text, /^Ran as administrator, but the command reported errors/);
  const no = resultText({ code: CANCELLED });
  assert.equal(no.ok, false); assert.ok(no.text.startsWith(REFUSED));
  assert.match(resultText({ code: NOT_STARTED, stderr: 'no consent service' }).text, /^Windows did not start the command as administrator\. no consent service$/);
  assert.match(resultText({ code: null, timedOut: true }).text, /^Still running after 2 minutes/);
  assert.match(resultText({ code: 0, output: 'x'.repeat(30000), status: { failed: false, exit: 0 } }).text, /\[cut at 20000 characters\]$/);
});

test('a call writes the two scripts, asks once, and reads back the result (the launcher replaced)', async () => {
  const seen = [];
  const fake = (code, write) => async launcher => {
    const text = readFileSync(launcher, 'utf8'); seen.push(text);
    const inner = /'"(.+command\.ps1)"'/.exec(text)[1];
    const script = readFileSync(inner, 'utf8');
    assert.ok(script.startsWith('\ufeff'), 'a byte-order mark for Windows PowerShell 5.1');
    if (write) {
      const out = /Set-Content -LiteralPath '([^']+output\.txt)'/.exec(script)[1], status = /Set-Content -LiteralPath '([^']+status\.json)'/.exec(script)[1];
      writeFileSync(out, write.output); writeFileSync(status, JSON.stringify(write.status));
    }
    return { code };
  };
  const done = await runAsAdministrator({ command: 'whoami /groups', reason: 'check' }, { platform: 'win32', run: fake(0, { output: 'BUILTIN\\Administrators', status: { failed: false, exit: 0 } }) });
  assert.deepEqual(done, { ok: true, text: 'Ran as administrator (exit code 0).\nBUILTIN\\Administrators' });
  const refused = await runAsAdministrator({ command: 'whoami', reason: 'check' }, { platform: 'win32', run: fake(CANCELLED) });
  assert.ok(refused.text.startsWith(REFUSED) && refused.ok === false);
  assert.equal(seen.length, 2, 'one prompt per call');
  assert.deepEqual(await runAsAdministrator({ command: 'whoami' }, { platform: 'linux', run: () => { throw new Error('must not run'); } }), { ok: false, text: 'Run as administrator works on Windows only.' });
  assert.deepEqual(await runAsAdministrator({ command: '  ' }, { platform: 'win32', run: () => { throw new Error('must not run'); } }), { ok: false, text: 'Give the PowerShell command to run.' });
});

test('No at the Windows prompt adds one line to the reply, in the person\'s language; other results do not', () => {
  const ev = (type, data, seq = 1) => ({ type: 'event', event: { type, seq, time: 1000, data } });
  const run = text => {
    const p = new TurnProjector({ startedAt: 0 });
    p.consume(ev('tool/call', { callId: 'c1', name: NAME, arguments: JSON.stringify({ command: 'x', reason: 'y' }) }));
    return p.consume(ev('tool/result', { message: { toolCallId: 'c1', isError: true, content: [{ type: 'text', text }] } }, 2));
  };
  const out = run(`${REFUSED} Do not try again unless the person asks for it.`);
  const line = out.find(f => f.type === 'delta');
  assert.deepEqual({ own: line.own, key: line.key, text: line.text }, { own: true, key: 'Administrator rights were refused at the Windows prompt.', text: '\n\nAdministrator rights were refused at the Windows prompt.' });
  assert.equal(run('Ran as administrator, but the command reported errors (exit code 1).').some(f => f.type === 'delta'), false);
  const en = JSON.parse(readFileSync(new URL('../src/locales/en.json', import.meta.url), 'utf8'));
  assert.ok(Object.values(en).includes(line.key), 'the line has a catalog key, so it is translated');
});

test('the tool server speaks MCP: one tool, and a call outside Windows says so', { skip: process.platform === 'win32' ? 'a real call here would ask Windows' : false }, async () => {
  const child = spawn(process.execPath, [new URL('../server/agent/admin-tool-mcp.mjs', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')], { stdio: ['pipe', 'pipe', 'inherit'] });
  const lines = createInterface({ input: child.stdout });
  const replies = [];
  lines.on('line', l => replies.push(JSON.parse(l)));
  const ask = async (id, method, params) => { child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'); for (let i = 0; i < 100 && !replies.find(r => r.id === id); i++) await new Promise(r => setTimeout(r, 20)); return replies.find(r => r.id === id); };
  try {
    assert.equal((await ask(1, 'initialize', {})).result.serverInfo.name, 'strata-admin');
    const tools = (await ask(2, 'tools/list', {})).result.tools;
    assert.deepEqual(tools.map(t => t.name), [TOOL]);
    assert.deepEqual(tools[0].inputSchema.required, ['command', 'reason']);
    const call = (await ask(3, 'tools/call', { name: TOOL, arguments: { command: 'whoami', reason: 'test' } })).result;
    assert.deepEqual(call, { isError: true, content: [{ type: 'text', text: 'Run as administrator works on Windows only.' }] });
  } finally { child.kill(); }
});
