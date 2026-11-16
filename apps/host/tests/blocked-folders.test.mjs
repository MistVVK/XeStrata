/**
 * #5737: blocked folders (Settings -> Tools). Every tool call that names a file or folder inside one is refused by the approval gate, in
 * every mode, before anything runs; the reply says which folder; the person adds and removes folders in Settings.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { blockedForms, checkFolder } from '../server/agent/blocked-folders.mjs';
import { BLOCKED_REASON, apply, blockedFolderIn, pathForm } from '../server/agent/plugins/strata-gate.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { TurnProjector } from '../server/agent/events.mjs';

const win = folders => blockedForms(folders, { platform: 'win32', home: 'C:\\Users\\ana', realpath: p => p, short: p => (p === 'D:\\Private Files' ? 'D:\\PRIVAT~1' : null) });
const call = (name, args) => ({ name, arguments: args });

test('what the person enters: a drive, a folder on one, a share; anything else is refused in words', () => {
  assert.deepEqual(checkFolder('d:', 'win32'), { folder: 'D:\\' });
  assert.deepEqual(checkFolder(' "d:/Private Files/" ', 'win32'), { folder: 'D:\\Private Files' });
  assert.deepEqual(checkFolder('\\\\nas\\share\\x', 'win32'), { folder: '\\\\nas\\share\\x' });
  for (const bad of ['', 'Private', '..\\x', 'C', 'C:x']) assert.ok(checkFolder(bad, 'win32').error, bad);
  assert.deepEqual(checkFolder('/home/ana/secret/', 'linux'), { folder: '/home/ana/secret' });
  assert.ok(checkFolder('secret', 'linux').error);
});

test('the spellings that reach the same folder', () => {
  const [p] = win(['D:\\Private Files']);
  assert.deepEqual(p.shown, 'D:\\Private Files');
  for (const f of ['d:/private files', 'd:/privat~1', '/mnt/d/private files', '/d/private files']) assert.ok(p.forms.includes(f), f);
  const [h] = win(['C:\\Users\\ana\\Secret']);
  for (const f of ['c:/users/ana/secret', '~/secret', '%userprofile%/secret', '$env:userprofile/secret', '$home/secret']) assert.ok(h.forms.includes(f), f);
  const [d] = win(['E:\\']);
  assert.deepEqual(d.forms.sort(), ['/mnt/e', 'e:']);
  assert.equal(pathForm('C:\\Work\\..\\Private\\.\\a.txt'), 'c:/private/a.txt');
});

test('a call that names a blocked folder in any argument is found; near misses are not', () => {
  const b = win(['D:\\Private Files', 'C:\\Users\\ana\\Secret', 'E:\\']);
  const hit = [
    call('read', { file_path: 'D:\\Private Files\\notes.txt' }),
    call('read', { file_path: 'd:/private files' }),
    call('pwsh', { command: 'Get-Content "D:\\PRIVAT~1\\a.txt"' }),
    call('bash', { command: 'cat /mnt/d/Private\\ Files/a.txt; cat "/mnt/d/private files/b"' }),
    call('pwsh', { command: 'dir $env:USERPROFILE\\Secret' }),
    call('bash', { command: 'ls ~/Secret' }),
    call('write', { path: 'C:\\Work\\..\\Users\\ana\\Secret\\x.md', content: 'x' }),
    call('mcp__custom-x__tool', { nested: { deep: ['E:\\movies'] } }),
    call('pwsh', { command: 'Set-Location E:' }),
  ];
  for (const c of hit) assert.ok(blockedFolderIn(c, b), JSON.stringify(c.arguments));
  const miss = [
    call('read', { file_path: 'D:\\Private Filesystem\\a' }),
    call('read', { file_path: 'D:\\Public\\a' }),
    call('pwsh', { command: 'Write-Output "he said: hello"' }),
    call('read', { file_path: 'C:\\Users\\ana\\Secrets.txt' }),
    call('pwsh', { command: 'Get-Date' }),
  ];
  for (const c of miss) assert.equal(blockedFolderIn(c, b), null, JSON.stringify(c.arguments));
  assert.equal(blockedFolderIn(hit[0], []), null, 'nothing blocked: nothing found');
});

test('the gate refuses it in every mode, helpers and run_code included, before the mode decides', async () => {
  const blocked = win(['D:\\Private Files']);
  for (const preset of ['strata-manual', 'strata-plan', 'strata-bypass']) {
    let handler = null;
    apply({ on: (name, fn) => { if (name === 'tools/pre-execute') handler = fn; }, inject: () => {}, sessions: { get: () => null },
            permissionPresets: { current: () => preset }, approval: { overrideOf: () => null } }, { blocked });
    let ran = false;
    const next = async () => { ran = true; return { kind: 'allow' }; };
    const session = { header: {}, append: () => {} };
    for (const name of ['read', 'run_code']) {
      const d = await handler({ name, arguments: { file_path: 'D:\\Private Files\\a.txt', code: 'open("D:/Private Files/a")' }, agent: { session } }, next);
      assert.equal(d.kind, 'deny', `${preset} ${name}`);
      assert.equal(d.reason, `${BLOCKED_REASON} (D:\\Private Files)`);
    }
    assert.equal(ran, false, `${preset}: nothing further decided or ran`);
    const ok = await handler({ name: 'read', arguments: { file_path: 'D:\\Public\\a.txt' }, agent: { session } }, next);
    assert.notEqual(ok.kind, 'deny', `${preset}: other folders as before`);
  }
});

test('Settings: add, refuse a non-path, keep no doubles, remove; the gate gets the spellings', () => {
  let saved = {};
  const tools = createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: mkdtempSync(join(tmpdir(), 'strata-blocked-')), platform: 'win32' });
  assert.deepEqual(tools.view().blockedFolders, [], 'none by default');
  assert.deepEqual(tools.update({ blockedFolders: { add: 'd:/Private Files' } }).blockedFolders, ['D:\\Private Files']);
  assert.deepEqual(tools.update({ blockedFolders: { add: 'D:\\private files\\' } }).blockedFolders, ['D:\\Private Files'], 'the same folder once');
  assert.throws(() => tools.update({ blockedFolders: { add: 'Private' } }), /full path/);
  const forms = tools.profileView().blockedFolders;
  assert.equal(forms[0].shown, 'D:\\Private Files'); assert.ok(forms[0].forms.includes('d:/private files'));
  const gate = JSON.parse(buildProfile({ local: [], connections: [], tools: { ...tools.profileView(), off: new Set() } }).patch.split('\n').slice(1).join('\n'));
  assert.ok(JSON.stringify(gate).includes('"blocked":[{"shown":"D:\\\\Private Files"'), 'the gate is told');
  assert.deepEqual(tools.update({ blockedFolders: { remove: 'D:\\Private Files' } }).blockedFolders, []);
});

test('the reply says which folder was blocked, once, in the person\'s language', () => {
  const p = new TurnProjector({ startedAt: 0 });
  const ev = (type, data, seq) => ({ type: 'event', event: { type, seq, time: 1000, data } });
  p.consume(ev('tool/call', { callId: 'c1', name: 'read', arguments: JSON.stringify({ file_path: 'D:\\Private Files\\a' }) }, 1));
  const out = p.consume(ev('tool/result', { message: { toolCallId: 'c1', isError: true, content: [{ type: 'text', text: `Error: ${BLOCKED_REASON} (D:\\Private Files)` }] } }, 2));
  const line = out.filter(f => f.type === 'delta');
  assert.equal(line.length, 1);
  assert.deepEqual({ key: line[0].key, values: line[0].values, text: line[0].text },
    { key: 'Blocked: {0} is in your blocked folders (Settings → Tools).', values: ['D:\\Private Files'], text: '\n\nBlocked: D:\\Private Files is in your blocked folders (Settings → Tools).' });
});
