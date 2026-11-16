// #5964: one folder the person may not read blanked a whole glob / grep in the agent ("glob search failed (exit 2): rg: ...\locked: IO
// error ... (os error 5)"). The staged Harness's search module is patched at staging (apps/desktop/scripts/harness-search-partial.mjs).
// The fixture is the exact MIT-licensed @deepseek-ai/dsh-tool-fs-search 0.2.1-alpha.3 lib/index.js (fixtures/dsh-tool-fs-search-LICENSE),
// run here with its Harness imports stubbed and a fake ripgrep subprocess: the module's own run, parse and render code is what is tested.
import test from 'node:test';
import assert from 'node:assert/strict';
import { linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import path, { join } from 'node:path';
import { runInNewContext } from 'node:vm';
import { HARNESS_SEARCH_PATCH, patchStagedHarnessSearch, patchedSearchSource } from '../../desktop/scripts/harness-search-partial.mjs';

const original = readFileSync(new URL('./fixtures/dsh-tool-fs-search-0.2.1-alpha.3-index.js.fixture', import.meta.url), 'utf8').replaceAll('\r\n', '\n');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

/** The module's code in a fresh context: imports become stubs, the exports are returned. */
function load(source) {
  const chain = new Proxy(function () {}, { get: () => chain, apply: () => chain });
  const exported = source.match(/^export \{([^}]*)\};?\s*$/m)[1].split(',').map(s => s.trim()).filter(Boolean);
  const code = source.replace(/^import .*;\n/gm, '').replace(/^export \{[^}]*\};?\s*$/m, '') + `\n({ ${exported.join(', ')} });`;
  class HarnessError extends Error { constructor(message, code, options) { super(message, options); this.code = code; } }
  // the head-retention contract the module reads (dsh-output-retention): text / truncated, items / kept / seen / truncated
  class TextRetainer { constructor({ maxBytes }) { this.max = maxBytes; this.text = ''; } push(t) { this.text += t; } finish() { const cut = Buffer.byteLength(this.text) > this.max; return { text: cut ? this.text.slice(0, this.max) : this.text, truncated: cut }; } }
  class ItemRetainer { constructor({ maxItems }) { this.max = maxItems; this.all = []; } push(i) { this.all.push(i); } finish() { const items = this.all.slice(0, this.max); return { items, kept: items.length, seen: this.all.length, truncated: this.all.length > this.max }; } }
  // the single-file runtime's way to ripgrep ("pkg" in process and an `-rg` sidecar): no dynamic import, and the fake spawn runs nothing
  const fakeProcess = { pkg: {}, execPath: process.execPath, platform: process.platform, versions: process.versions, env: {}, cwd: () => process.cwd() };
  return runInNewContext(code, {
    z: chain, MAX_TIMER_DELAY_MS: 2 ** 31 - 1, defineTool: options => options, HarnessError, TextRetainer, ItemRetainer,
    existsSync: () => true, isAbsolute: path.isAbsolute, join: path.join, parse: path.parse, relative: path.relative, sep: path.sep,
    process: fakeProcess, Buffer, console, URL, TextEncoder, TextDecoder,
  });
}

const CAPS = { maxResults: 100, sampleOverCapGlobResults: false, maxMatches: 250, maxLineBytes: 2000, maxMetaBytes: 65536, rawOutputMaxBytes: 2e7, graceMs: 3000, stderrMaxBytes: 65536, timeoutMs: 30000 };

/** A glob and a grep tool of the module, wired to a fake ripgrep that ends as `run` says. */
function tools(mod, run) {
  const registered = {};
  const ctx = {
    systemPrompt: { section() {}, getSectionOrder: () => 0 },
    tools: { register: t => { registered[t.name] = t; }, get: () => undefined },
    on() {},
    workingDirectory: { ensure: async () => process.cwd() },
    subprocess: {
      spawn() {
        const r = typeof run === 'function' ? run() : run;
        if (r.throws) throw r.throws;
        return { done: Promise.resolve({ exitCode: r.exitCode ?? 0, signal: r.signal ?? null }),
          collected: { stdout: { readFrom: () => ({ text: r.stdout ?? '', lossy: false }) }, stderr: { readFrom: () => ({ text: r.stderr ?? '', lossy: !!r.stderrLossy }) } } };
      },
    },
  };
  mod.applyGlobTool(ctx, CAPS);
  mod.applyGrepTool(ctx, CAPS);
  return registered;
}
async function call(tool, args, { aborted = false } = {}) {
  const exec = { agent: undefined, signal: { aborted }, arguments: args };
  const value = await tool.execute(args, exec);
  return { value, text: tool.output.render(args, JSON.parse(JSON.stringify(value)))[0].text };   // the Harness snapshots the value
}

const cwd = process.cwd();
const denied = dir => `rg: ${dir}: IO error for operation on ${dir}: Access is denied. (os error 5)`;
const grepLine = (file, n, text) => JSON.stringify({ type: 'match', data: { path: { text: file }, lines: { text: text + '\n' }, line_number: n, submatches: [] } });

test('the pinned source and the patch: one exact place for every edit, and a mismatch refuses', () => {
  assert.equal(sha256(Buffer.from(original)), HARNESS_SEARCH_PATCH.beforeSha256, 'the fixture is the pinned file');
  const patched = patchedSearchSource(original);
  assert.notEqual(patched, original);
  assert.throws(() => patchedSearchSource(patched), /expected exactly one/, 'a second application refuses');
});

test('unpatched (the bug): one unreadable folder throws the found paths away', async () => {
  const glob = tools(load(original), { exitCode: 2, stdout: join(cwd, 'target.txt') + '\n', stderr: denied(join(cwd, 'locked')) }).glob;
  await assert.rejects(call(glob, { pattern: '**/*.txt' }), /glob search failed \(exit 2\)/);
});

test('patched: every stderr line is an access error -> the found paths and matches, and the partial note', async () => {
  const mod = load(patchedSearchSource(original));
  const locked = join(cwd, 'locked');
  const glob = tools(mod, { exitCode: 2, stdout: join(cwd, 'target.txt') + '\n', stderr: denied(locked) + '\n' }).glob;
  const g = await call(glob, { pattern: '**/*.txt' });
  assert.deepEqual([...g.value.paths], ['target.txt']);
  assert.match(g.text, /^target\.txt\n\n\(Partial result: 1 folder could not be read and was skipped: .*locked\. Matches inside it are not listed\.\)$/);
  const grep = tools(mod, { exitCode: 2, stdout: grepLine(join(cwd, 'target.txt'), 1, 'match me') + '\n', stderr: `rg: ./a: Permission denied (os error 13)\nrg: ./b: Permission denied (os error 13)\n` }).grep;
  const r = await call(grep, { pattern: 'match me' });
  assert.equal(r.value.matches.length, 1);
  assert.match(r.text, /match me[\s\S]*\(Partial result: 2 folders could not be read and were skipped: \.\/a, \.\/b\. Matches inside them are not listed\.\)$/);
  const none = await call(tools(mod, { exitCode: 2, stdout: '', stderr: denied(locked) }).glob, { pattern: '*.md' });
  assert.match(none.text, /^No files found\n\n\(Partial result: 1 folder/, 'nothing found outside the folders it could not read: still partial');
  const clean = await call(tools(mod, { exitCode: 0, stdout: join(cwd, 'x.txt') + '\n' }).glob, { pattern: '*.txt' });
  assert.equal(clean.text, 'x.txt', 'a complete search has no note');
});

test('more than five unreadable folders: five are named, then ", ..."', async () => {
  const mod = load(patchedSearchSource(original));
  const stderr = Array.from({ length: 7 }, (_, i) => denied(`C:\\d${i}`)).join('\r\n');
  const r = await call(tools(mod, { exitCode: 2, stdout: join(cwd, 'f.txt') + '\n', stderr }).glob, { pattern: '*.txt' });
  assert.match(r.text, /7 folders could not be read and were skipped: C:\\d0, C:\\d1, C:\\d2, C:\\d3, C:\\d4, \.\.\.\. Matches inside them/);
});

test('patched: everything else fails as before', async () => {
  const mod = load(patchedSearchSource(original));
  const glob = run => tools(mod, run).glob, out = join(cwd, 't.txt') + '\n';
  await assert.rejects(call(glob({ exitCode: 2, stdout: out, stderr: denied('x') + '\nrg: something else went wrong' }), { pattern: '*' }), /search failed \(exit 2\)/, 'mixed stderr');
  await assert.rejects(call(glob({ exitCode: 2, stdout: '', stderr: 'rg: regex parse error:\n    (\n    ^\nerror: unclosed group' }), { pattern: '(' }), /pattern rejected by ripgrep/, 'pattern error');
  await assert.rejects(call(glob({ exitCode: 2, stdout: '', stderr: "rg: error parsing glob '[': unclosed character class" }), { pattern: '[' }), /pattern rejected by ripgrep/, 'glob error');
  await assert.rejects(call(glob({ exitCode: 2, stdout: out, stderr: denied('x'), stderrLossy: true }), { pattern: '*' }), /search failed \(exit 2\)/, 'stderr cut short');
  await assert.rejects(call(glob({ exitCode: 2, stdout: out, stderr: '' }), { pattern: '*' }), /search failed \(exit 2\)/, 'exit 2 with no reason');
  await assert.rejects(call(glob({ exitCode: null, signal: 'SIGTERM', stdout: out, stderr: denied('x') }), { pattern: '*' }), /killed by signal SIGTERM/, 'signal');
  await assert.rejects(call(glob({ exitCode: 2, stdout: out, stderr: denied('x') }), { pattern: '*' }, { aborted: true }), /aborted/, 'abort');
  await assert.rejects(call(glob({ exitCode: 3, stdout: out, stderr: denied('x') }), { pattern: '*' }), /search failed \(exit 3\)/, 'another exit code');
});

test('the installer\'s Harness staging applies the patch and records it in BUNDLE.json', () => {
  const stage = readFileSync(new URL('../../desktop/scripts/stage-harness.mjs', import.meta.url), 'utf8');
  assert.match(stage, /import \{ patchStagedHarnessSearch \} from '\.\/harness-search-partial\.mjs';/);
  assert.match(stage, /const patches = \[patchStagedHarnessBudget\(out\), patchStagedHarnessSearch\(out\)\];/);
  assert.match(stage, /\bpatches,/, 'the provenance list goes into BUNDLE.json');
});

function staged(t, { source = original, manifest = { name: '@deepseek-ai/dsh-tool-fs-search', version: '0.2.1-alpha.3' } } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-harness-search-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, HARNESS_SEARCH_PATCH.file);
  mkdirSync(join(file, '..'), { recursive: true });
  writeFileSync(file, source); writeFileSync(join(root, 'node_modules', '@deepseek-ai', 'dsh-tool-fs-search', 'package.json'), JSON.stringify(manifest));
  return { root, file };
}

test('staging patches the copy once and records it; another version, other bytes, a link or a second run refuse', t => {
  const f = staged(t), provenance = patchStagedHarnessSearch(f.root);
  assert.deepEqual(provenance, { ...HARNESS_SEARCH_PATCH, afterSha256: sha256(readFileSync(f.file)) });
  assert.equal(readFileSync(f.file, 'utf8'), patchedSearchSource(original));
  assert.throws(() => patchStagedHarnessSearch(f.root), /unexpected lib\/index\.js SHA-256/, 'a second run');
  assert.throws(() => patchStagedHarnessSearch(staged(t, { manifest: { name: '@deepseek-ai/dsh-tool-fs-search', version: '0.2.2-rc.3' } }).root), /expected @deepseek-ai\/dsh-tool-fs-search 0\.2\.1-alpha\.3/);
  assert.throws(() => patchStagedHarnessSearch(staged(t, { source: original + '\n' }).root), /unexpected lib\/index\.js SHA-256/);
  const l = staged(t);
  linkSync(l.file, l.file + '.other');
  assert.throws(() => patchStagedHarnessSearch(l.root), /expected an unlinked staged module/);
  assert.equal(readFileSync(l.file, 'utf8'), original, 'nothing written when it refuses');
});
