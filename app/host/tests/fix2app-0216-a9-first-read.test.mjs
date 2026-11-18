// FIX2-APP-0216 A9 (retest WIN owner-PC F2): the first-run "Checking what your computer can run..." line lasted 23 s.
// Measured: the host's first read of the PC is the hardware probe (about 2 s of PowerShell and nvidia-smi calls), Control's own calls (about 1 s) and strata_status,
// which waited the full 2 s HTTP timeout for EVERY port nobody listens on (the default port and one per prepared model; Windows never answers "refused" on a
// closed loopback port). Fixed in Control (terminal/tools/strata_mcp.py: port_accepts, probe_many; terminal/tests/desktop/test_status_probe.py). Here: the host starts
// that read when the packaged app starts, and the line says it is still working after 12 s.
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = p => readFileSync(new URL(p, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

test('the packaged host starts its first read of the PC when it listens, not when the first page asks', () => {
  const index = read('../server/index.mjs');
  const listen = index.slice(index.indexOf('server.listen(PORT, HOST, () => {'));
  const body = listen.slice(0, listen.indexOf('\n});') + 4);
  assert.match(body, /if \(control && process\.env\.STRATA_PACKAGED === '1'\) Promise\.resolve\(native\.refresh\(false\)\)\.catch\(/, 'a not-forced read, packaged app only, errors left to the page');
  assert.ok(!/native\.refresh\(true\)/.test(body), 'not forced: a read already running is joined, not repeated');
});

test('the line says it is still working after 12 seconds, and goes back when the list is here', () => {
  const page = read('../src/components/GuidedSetup.tsx');
  assert.match(page, /const readingPc = catalogLoading\(native\), \[readingSlow, setReadingSlow\] = useState\(false\);/);
  assert.match(page, /if \(!readingPc\) \{ setReadingSlow\(false\); return; \}\s+const timer = setTimeout\(\(\) => setReadingSlow\(true\), 12000\);\s+return \(\) => clearTimeout\(timer\);/);
  assert.match(page, /\{readingSlow \? tr\("Still checking\. On the first start this can take up to a minute\."\) : tr\("This takes a few seconds\."\)\}/);
});

test('the new line is in the catalog in all ten languages', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(read(`../src/locales/fixa-${l}.json`))]));
  const text = 'Still checking. On the first start this can take up to a minute.';
  const key = Object.keys(cat.en).find(k => cat.en[k] === text);
  assert.equal(key, 'fixa05');
  for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), l);
  for (const l of LANGS) assert.deepEqual(Object.keys(cat[l]), Object.keys(cat.en), `${l} has the same keys as English`);
});

test('a read that has started is joined by the page, not repeated (the host answers every caller from the one read)', async () => {
  const { createNativeControl } = await import('../server/native-control.mjs');
  let calls = 0;
  const control = {
    control: { dir: '/none' }, version: '0.1.62',
    async installedModels() { calls++; await new Promise(r => setTimeout(r, 80)); return { install_dir: '/none', models: [] }; },
    async engine() { return { usable: false }; },
  };
  const native = createNativeControl({ control, directory: process.env.TEMP + '/strata-a9-' + process.pid, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  try {
    const first = native.refresh(false), second = native.refresh(false);
    await Promise.allSettled([first, second]);
    assert.ok(calls <= 1, `the PC was read ${calls} times for two asks`);
  } finally { await native.close(); }
});
