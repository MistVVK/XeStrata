import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { endTree, endTreeThenKill, groupAlive, ownGroup } from '../server/process-tree.mjs';
import { runCommand } from '../server/agent/custom-command-run.mjs';

const isWin = process.platform === 'win32';
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Whether the process is there. A process that has ended but whose parent has not collected it yet (a zombie: Node reaps its own children when the event
// loop gets to it, init reaps the orphans when it gets to them) answers signal 0 and is not running: on a loaded machine that gap is long.
function alive(pid) {
  try { process.kill(pid, 0); } catch (error) { return error.code === 'EPERM'; }
  if (process.platform === 'linux') {
    try { return readFileSync(`/proc/${pid}/stat`, 'utf8').replace(/^.*\) /, '')[0] !== 'Z'; } catch { return false; }
  }
  return true;
}
// Every wait is on the real condition, with a long limit: it returns the moment the condition holds, so the limit costs nothing when the machine is fine.
async function until(condition, ms = 30_000) { for (let waited = 0; waited < ms; waited += 50) { if (condition()) return true; await sleep(50); } return condition(); }
const gone = (pid, ms) => until(() => !alive(pid), ms);

// A program that starts a program of its own and reports its pid, the way Stop meets a job and its revision command (#4529 #4570).
const parentScript = `const { spawn } = require('node:child_process'); const k = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore' }); console.log(k.pid); setTimeout(()=>{}, 120000);`;

async function tree() {
  const child = spawn(process.execPath, ['-e', parentScript], { detached: ownGroup(), stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true });
  const kid = await new Promise(resolve => child.stdout.once('data', b => resolve(Number(String(b).trim()))));
  return { child, kid };
}

test('endTree ends the program and the program it started, and nothing else (#4529 #4570)', async () => {
  const outsider = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore', windowsHide: true });   // same program, not started by the job
  const { child, kid } = await tree();
  try {
    assert.ok(alive(kid));
    endTree(child);
    assert.ok(await gone(child.pid), 'the job survived');
    assert.ok(await gone(kid), 'what the job started survived');
    assert.ok(alive(outsider.pid), 'a process of the same program that the job did not start was ended');
  } finally { outsider.kill(); try { process.kill(kid, 'SIGKILL'); } catch { /* gone */ } child.kill('SIGKILL'); }
});

test('endTreeThenKill follows a TERM that is ignored with a KILL of the group', { skip: isWin && 'POSIX signals' }, async () => {
  // Both programs ignore SIGTERM, and the test goes on only when both have said that the handler is in place (no fixed delay for a slow start).
  const stubborn = `const { spawn } = require('node:child_process'); process.on('SIGTERM', () => {});
    const k = spawn(process.execPath, ['-e', "process.on('SIGTERM',()=>{});console.log('up');setTimeout(()=>{},120000)"], { stdio: ['ignore', 'pipe', 'ignore'] });
    k.stdout.once('data', () => { console.log(k.pid); }); setTimeout(()=>{}, 120000);`;
  const child = spawn(process.execPath, ['-e', stubborn], { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
  const kid = await new Promise(resolve => child.stdout.once('data', b => resolve(Number(String(b).trim()))));
  try {
    assert.ok(alive(kid));
    endTreeThenKill(child, 500);
    assert.ok(await gone(kid, 60_000), 'the group survived SIGTERM and was not killed');
    assert.ok(await gone(child.pid, 60_000));
    assert.ok(await until(() => !groupAlive(child), 60_000), 'a member of the group is still there');
  } finally { try { process.kill(-child.pid, 'SIGKILL'); } catch { /* gone */ } }
});

test('a child that leads no group of its own is still ended, alone, as before', async () => {
  const child = spawn(process.execPath, ['-e', 'setTimeout(()=>{}, 120000)'], { stdio: 'ignore', windowsHide: true });
  endTree(child);
  assert.ok(await gone(child.pid));
});

test('a custom command that runs out of time takes the programs it started with it', { skip: isWin && 'POSIX shell' }, async () => {
  const marker = join(tmpdir(), `strata-cc-${process.pid}.pid`);
  rmSync(marker, { force: true });
  // The limit is far above what the shell needs to start and write its marker, so a slow machine cannot end the command before it did.
  const result = await runCommand({ command: `sleep 120 & echo $! > ${marker}; wait`, timeoutSec: 8 });
  assert.equal(result.ok, false);
  assert.ok(await until(() => existsSync(marker) && /^\d+/.test(readFileSync(marker, 'utf8'))), 'the command never wrote its marker');
  const pid = Number(readFileSync(marker, 'utf8').trim());
  rmSync(marker, { force: true });
  assert.ok(await gone(pid), 'the command\'s own child outlived the time limit');
});

test('ending a child that never started leaves the caller and its process group alone (macOS signalled the whole group)', { skip: isWin && 'POSIX process groups' }, async () => {
  // A program that fails to start a child and ends it before the "error" event arrives, the way a Stop can meet a job whose Python is missing.
  // It leads a group of its own, so if the signal reaches its group it dies by SIGTERM and the test fails; this process is never in the line of fire.
  const body = `import { spawn } from 'node:child_process'; import { endTree, endTreeThenKill } from ${JSON.stringify(new URL('../server/process-tree.mjs', import.meta.url).href)};
    const c = spawn('strata-no-such-program-xyz', [], { stdio: 'ignore' }); c.on('error', () => {}); endTree(c); endTreeThenKill(c, 10); setTimeout(() => console.log('survived'), 300);`;
  const child = spawn(process.execPath, ['--input-type=module', '-e', body], { detached: true, stdio: ['ignore', 'pipe', 'ignore'] });
  let out = ''; child.stdout.on('data', b => { out += b; });
  const code = await new Promise(resolve => child.on('close', (c, s) => resolve(s ?? c)));
  assert.equal(code, 0, `the caller was signalled (${code})`);
  assert.match(out, /survived/);
});
