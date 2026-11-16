import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, ({ ...JSON.parse(await readFile(`src/locales/${l}.json`, 'utf8')), ...JSON.parse(await readFile(`src/locales/engine-${l}.json`, 'utf8')), ...JSON.parse(await readFile(`src/locales/gui-${l}.json`, 'utf8')) })])));
const built = await build({ stdin: { contents: "export * from './src/lib/plain-errors'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { plainFailure, plainFailureText, chatFailure, setTranslationLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) await loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()

const cases = [
  ['Strata’s tunnel for this address is already running (PID 31), so Strata did not start a second one. Stop it first (or use it as it is), then connect again here.', 'tunnel-own-running', /tunnel for this address is already running/],
  ['GPU 2 is already used by qwen3.8-flash-next-q2_0, so this model did not fit there. Load it on another GPU, or unload qwen3.8-flash-next-q2_0 first.', 'gpu-shared', /no room for this one/],
  ['This model needs a newer engine: Gyro-S needs engine 0.1.45.1 or newer, and this one is 0.1.45. Update the engine in Settings, then try again.', 'engine-too-old', /needs a newer engine/],
  ['The settings file for this model has a mistake, so the model was not started. Line 4: The settings file is not valid JSON; look near line 4.', 'settings-file', /settings file has a mistake/],
  ['The Cloudflare connector (cloudflared) is not installed here, and the app never downloads it while it runs. Install it from the optional connector component.', 'connector-missing', /could not find Cloudflare’s connector/],
  ['The cloudflared at /usr/bin/cloudflared is version 2021.3.0, which is too old. Strata needs 2022.1.0 or newer. Update cloudflared, then retry.', 'connector-old', /connector on this computer is too old/],
  ['The file at /x/cloudflared did not run as cloudflared. Reinstall cloudflared, then retry.', 'connector-broken', /did not run/],
  ['lost host child is still present; retry after it exits', 'previous-session', /last Strata session is still closing down/],
  ['server_exited', 'start-failed', /model stopped while starting\./],
  ['server_exited (exit code 0xC0000135)', 'dll-missing', /program files is missing/],
  ['server_exited (exit code 0xC000007B)', 'dll-bad', /does not fit this computer or is damaged/],
  ['server_exited (exit code 0xC0000005)', 'crash', /crashed while starting/],
  ['server_exited (exit code 0xC000001D)', 'cpu-unsupported', /newer processor feature/],
  ['spawn C:/x/strata.exe ENOENT', 'file-missing', /could not find a file/],
  ['server_exited (exit code 0xC0000022)', 'blocked', /security software stopped/],
  ['server_exited: memory guard: 5.1 GiB of RAM free is under the 6 GiB limit', 'memory', /ran low on memory/],
  ['server_exited (exit code 137): Killed', 'memory', /ran low on memory/],
  ['Engine stopped (exit 3). Open diagnostics or reload.', 'start-failed', /model stopped while starting\./],
  ['Loading took more than five minutes. Check the engine log and free memory.', 'slow-load', /taking too long/],
  ['No engine for this PC was found in C:/x/components: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', 'engine-missing', /engine for this computer isn’t installed yet/],
  ['cannot reach huggingface.co (<urlopen error [Errno 11001] getaddrinfo failed>)', 'network', /couldn’t reach the internet/],
  ['sha256 mismatch for Qwen3.8-Flash-Next-00001-of-00002.gguf', 'integrity', /safety check/],
  ['The agent component (DeepSeek Harness) is not installed.', 'assistant-missing', /coding assistant isn’t installed/],
  ['component-missing', 'assistant-missing', /coding assistant/],
  ['Strata Control did not answer within 60 s (models list).', 'control', /couldn’t read its own components/],
  ['Prepare this model’s vision files before enabling picture reading.', 'picture-reading', /Picture reading isn’t available/],
  ['server_exited (exit code 1): strata generate: GPU 0 (AMD Radeon 760M Graphics, gfx1103) is not an architecture this Strata engine was compiled for (gfx1200,gfx1201); compile it for this card', 'gpu-no-code', /no code for the graphics card/],
  ['server_exited (exit code 1): [strata] none of this computer\'s graphics cards has code in this Strata engine (it is built for gfx1200,gfx1201). Cards found: AMD Radeon RX 6800 XT (gfx1030).', 'gpu-no-code', /no code for the graphics card/],
  ['Traceback (most recent call last):\n  File "x.py", line 3', 'technical', /Something went wrong/],
];

for (const [raw, kind, message] of cases) {
  test(`plain words for: ${raw.slice(0, 60)}`, () => {
    const f = plainFailure(raw);
    assert.equal(f.kind, kind);
    assert.match(f.message, message);
    assert.ok(f.next, 'a next step');
    assert.equal(f.detail, raw.trim(), 'the original text stays for Details');
    assert.doesNotMatch(f.message, /server_exited|exit \d|component-missing|Traceback|0x[0-9A-F]{8}/);
  });
}

test('a DeepSeek model of an earlier Strata: Control’s own words, the folder and size kept, the next step apart (English only until the owner approves translations)', () => {
  const d1 = 'DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started.';
  const files = ' Its files are still on this computer: 257 GB in /home/me/Strata-data/models/deepseek-v4.1-flash-ist-das-2.00bpw. Nothing was changed or deleted.';
  const next = 'Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that folder to free the space.';
  for (const message of [d1, d1 + files]) {
    const f = plainFailure(message + ' ' + next);
    assert.equal(f.kind, 'deepseek-retired');
    assert.equal(f.message, message);
    assert.equal(f.next, next);
    assert.equal(plainFailure(JSON.stringify({ error: message + ' ' + next })).message, message, 'a wrapped error body too');
  }
  assert.notEqual(plainFailure('server_exited').kind, 'deepseek-retired');
  assert.notEqual(plainFailure('The agent component (DeepSeek Harness) is not installed.').kind, 'deepseek-retired');
});

test('an ordinary sentence is left as it is and gets no Details', () => {
  const f = plainFailure('Some model files are missing. Use Download / repair to restore them.');
  assert.equal(f.kind, 'plain');
  assert.equal(f.detail, undefined);
});

test('every plain message and next step is translated in all nine languages', () => {
  const en = catalogs.en;
  const ids = Object.fromEntries(Object.entries(en).map(([id, text]) => [text, id]));
  for (const [raw] of cases) {
    const f = plainFailure(raw);
    for (const text of [f.message, f.next]) {
      const id = ids[text];
      assert.ok(id, `"${text}" is in the catalog`);
      for (const l of languages) if (l !== 'en') assert.notEqual(catalogs[l][id], text, `${l}: "${text}" is translated`);
    }
  }
  setTranslationLanguage('de');
  assert.notEqual(plainFailureText('server_exited').message, plainFailure('server_exited').message);
  // a missing DLL is never described as low memory
  assert.doesNotMatch(plainFailure('server_exited (exit code 0xC0000135)').message + plainFailure('server_exited (exit code 0xC0000135)').next, /memory|Leave 20/i);
  setTranslationLanguage('en');
});

test('missing model files are named, in every language, with a next step (#3478)', async () => {
  const { missingFilesError, missingNames } = await import('../server/missing-files.mjs');
  const names = missingNames([{ part: 'engine', file: 'strata.exe', status: 'missing' }, { part: 'native', file: 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf', status: 'missing' }, { part: 'vision/mmproj', file: 'mmproj-F16.gguf', status: 'missing' }],
    [['C:', 'models', 'Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf'].join(String.fromCharCode(92)), '/data/models/mmproj-F16.gguf']);
  assert.deepEqual(names, ['Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf', 'mmproj-F16.gguf', 'the engine program', 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf']);
  const error = missingFilesError('qwen36-q4', names);
  assert.match(error.message, /missing: Qwen3\.6-35B-A3B-UD-Q4_K_XL-00002-of-00003\.gguf, mmproj-F16\.gguf, the engine program, Qwen3\.6-35B-A3B-UD-Q4_K_XL\.gguf\. Use Download/);
  assert.match(error.details[0], /Missing for qwen36-q4: /);
  const more = missingFilesError('m', ['a.gguf', 'b.gguf', 'c.gguf', 'd.gguf', 'e.gguf', 'f.gguf']);
  assert.match(more.message, /a\.gguf, b\.gguf, c\.gguf, d\.gguf and 2 more\./);
  const f = plainFailure(error.message);
  assert.equal(f.kind, 'model-files-missing');
  assert.equal(f.message, 'These model files are missing: Qwen3.6-35B-A3B-UD-Q4_K_XL-00002-of-00003.gguf, mmproj-F16.gguf, the engine program, Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf.');
  assert.ok(f.next && f.detail === error.message);
  for (const language of languages) {
    setTranslationLanguage(language);
    const shown = plainFailureText(error.message);
    if (language !== 'en') {
      assert.notEqual(shown.message, f.message, `${language}: the sentence is translated`);
      assert.notEqual(shown.next, f.next, `${language}: the next step is translated`);
    }
    assert.ok(shown.message.includes('mmproj-F16.gguf'), `${language}: the file names survive`);
    assert.doesNotMatch(shown.message, /\{0\}/);
  }
  setTranslationLanguage('en');
  assert.equal(plainFailure(missingFilesError('m', []).message).kind, 'plain', 'with no names the old sentence is left as it was');
});

test('the engine component is named in words, once (#3900)', async () => {
  const { missingNames } = await import('../server/missing-files.mjs');
  assert.deepEqual(missingNames([{ part: 'engine', file: 'component:strata-engine' }], ['component:strata-engine']), ['the engine program']);
});

// An OpenAI-shaped error body ({"error": {"type", "message"}}, what strata serve and the Responses API answer) showed as raw JSON: the
// wrapper is peeled the same way as a plain {"error": "..."} (GUI-1116, from the engine-error review).
test('an error body whose error is an object shows its message, not the JSON', () => {
  for (const raw of ['HTTP 500: {"error":{"type":"server_error","message":"dequant launch: unspecified launch failure"}}',
                     '{"error":{"type":"invalid_request_error","message":"messages is required","code":null}}']) {
    const f = plainFailure(raw);
    assert.doesNotMatch(f.message, /[{}]|server_error|invalid_request_error/, raw);
  }
  assert.equal(plainFailure('{"error":{"type":"server_error","message":"dequant launch: unspecified launch failure"}}').message, 'Dequant launch: unspecified launch failure');
  assert.equal(plainFailure('{"error":{"type":"x"}}').message, '{"error":{"type":"x"}}', 'no message inside: left as it was');
});

// The engine's errors in a chat (coordinator's review of 11-16, wording approved). strata serve answers a stop it restarts from with
// "...; the next request restarts it" (EngineDied, a fatal ERR line) and a non-fatal ERR with the engine's bare line.
const STOPPED = 'The model stopped while it was answering.', PROBLEM = 'The model ran into a problem while answering.';
test('an engine that stopped while answering is said so, not as a download, a crash at start or a stop at start', () => {
  for (const raw of ['Unrecoverable native verification failure: verify: timed out at layer 12; the next request restarts it',
                     'the engine stopped unexpectedly (exit code 3221225477); the next request restarts it',
                     'the engine stopped while this request waited; it was not sent; the next request restarts it',
                     'HTTP 503: {"error":{"type":"server_error","message":"the engine stopped unexpectedly (exit code 3); the next request restarts it"}}']) {
    const f = plainFailure(raw);
    assert.equal(f.kind, 'engine-stopped-answering', raw);
    assert.equal(f.message, STOPPED);
    assert.match(f.next, /^Send your message again\. Strata starts the model again by itself\./);
    assert.match(f.detail, /the next request restarts it/, 'the engine line stays under Details');
  }
  const again = plainFailure(STOPPED);    // the chat keeps the sentence; ErrorNotice reads it again and finds the next step
  assert.equal(again.kind, 'engine-stopped-answering');
  assert.ok(again.next);
});

test('a real network failure is still a download without internet', () => {
  for (const raw of ['cannot reach github.com (<urlopen error [Errno 11001] getaddrinfo failed>) - check your internet connection and run it again',
                     'could not resolve host: huggingface.co', 'connection refused while downloading the model']) {
    assert.equal(plainFailure(raw).kind, 'network', raw);
  }
});

test('in a chat, an engine line no rule knows becomes one plain sentence with the line under Details; a sentence for people stays', () => {
  for (const [raw, line] of [["prefill: the PLE rows' upload failed: unspecified launch failure", "prefill: the PLE rows' upload failed: unspecified launch failure"],
                             ['HTTP 500: {"error":{"type":"server_error","message":"dequant launch: unspecified launch failure"}}', 'dequant launch: unspecified launch failure'],
                             ['500 dequant launch: unspecified launch failure', '500 dequant launch: unspecified launch failure']]) {
    const f = chatFailure(raw);
    assert.equal(f.kind, 'engine-error-answering', raw);
    assert.equal(f.message, PROBLEM);
    assert.match(f.next, /^Send your message again\. If it keeps happening, open Details/);
    assert.equal(f.detail, line);
  }
  assert.equal(plainFailure(PROBLEM).kind, 'engine-error-answering', 'ErrorNotice finds the next step for the kept sentence');
  for (const raw of ['Insufficient balance. Top up your account to continue.', 'Invalid API key']) {
    assert.equal(chatFailure(raw).kind, 'plain', raw);
    assert.equal(chatFailure(raw).message, raw);
  }
  assert.equal(chatFailure('the engine stopped unexpectedly (exit code 3); the next request restarts it').kind, 'engine-stopped-answering', 'rule (a) first');
  const store = readFileSync(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /const failure = typeof frame\.key === 'string' \? null : chatFailure\(String\(frame\.message \?\? 'Engine error'\)\);/);
  assert.match(store, /\.\.\.\(detail \? \{ errorDetail: detail \} : \{\}\)/, 'the engine line is kept for Details');
});

// GUI-1116 (owner 11-16: every language checked for 0.2.16): the engine-failure sentences of 7bcb5a69c have catalog keys, so the chat shows
// them in the person's language instead of English. The translations are drafts on the lane's [DRAFT] list for the release review.
test('the engine-failure sentences and next steps are translated in all nine languages', () => {
  const sentences = [chatFailure('the model stopped; the next request restarts it'), chatFailure('HTTP 500: weird engine text')].flatMap(f => [f.message, f.next]);
  assert.equal(sentences.length, 4);
  for (const text of sentences) {
    const id = Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
    assert.ok(id, `"${text}" has a catalog key`);
    for (const l of languages) if (l !== 'en') assert.ok(catalogs[l][id] && catalogs[l][id] !== text, `${l}: "${text}" is translated`);
  }
});
