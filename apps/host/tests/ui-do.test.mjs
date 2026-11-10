// The 0.2.10 interface changes (branch f-0291-ui): one section per issue. Source-level checks for what only a browser can show are
// backed by scripts/ui-do-drive.mjs, which drives the built app.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build, transform } from 'esbuild';

// a Windows checkout with core.autocrlf (GitHub's runners) has CRLF line ends: the source checks below match LF
const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const locale = name => JSON.parse(readFileSync(new URL(`../src/locales/${name}.json`, import.meta.url), 'utf8'));
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const withI18n = async entry => {
  const built = await build({ stdin: { contents: `export * from './src/lib/i18n'; export * from './src/lib/${entry}';`, resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
  const mod = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
  for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) await mod.loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()
  return mod;
};

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];

test('the ui catalogs have the same keys and placeholders in all nine languages, none still English, none in another catalog', () => {
  const en = locale('ui-en');
  const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
  const others = new Set();
  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en']) for (const v of Object.values(locale(name))) others.add(v);
  for (const l of languages) {
    const catalog = locale(`ui-${l}`);
    assert.deepEqual(Object.keys(catalog), Object.keys(en), l);
    for (const [id, source] of Object.entries(en)) {
      assert.ok(catalog[id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(catalog[id]), slots(source), `${l}:${id}`);
      if (l !== 'en') assert.notEqual(catalog[id], source, `${l}:${id} is still English`);
    }
  }
  for (const [id, text] of Object.entries(en)) assert.ok(!others.has(text), `already in another catalog: ${id} ${text}`);
});

// ---------------------------------------------------------------------------------------------------------------- #4725
test('#4725 the pending approvals of a chat are listed oldest first, and nothing else is', async () => {
  const { pendingApprovals } = await load('lib/approvals.ts');
  const tool = (id, state) => ({ id, name: 'write', state });
  const session = { messages: [
    { id: 'm1', tools: [tool('a', 'done'), tool('b', 'proposed')] },
    { id: 'm2', tools: [tool('c', 'denied'), tool('d', 'running')] },
    { id: 'm3' },
    { id: 'm4', tools: [tool('e', 'proposed')] },
  ] };
  assert.deepEqual(pendingApprovals(session).map(x => [x.messageId, x.tool.id]), [['m1', 'b'], ['m4', 'e']]);
  assert.deepEqual(pendingApprovals(null), []);
  assert.deepEqual(pendingApprovals({ messages: [] }), []);
});

test('#4725 the Approve and Deny buttons live in the dock above the message box, and the reply only says that it waits', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /<ApprovalDock sessionId=\{session\?\.id\}\/>/);
  const toolCard = chat.slice(chat.indexOf('function ToolCard'));
  assert.doesNotMatch(toolCard, /\{tr\("(Approve|Deny)"\)\}|resolveTool/,'the card in the reply no longer carries the buttons');
  assert.match(toolCard, /Waiting for your answer below/);
  const dock = src('components/ApprovalCard.tsx');
  assert.match(dock, /resolveTool\(messageId, t\.id, true\)/);
  assert.match(dock, /resolveTool\(messageId, t\.id, false\)/);
  assert.equal((dock.match(/tr\("Approve"\)/g) ?? []).length, 2, 'one Approve and Deny pair per card style, nowhere else');
});

test('#4725 the dock and the waiting line are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) {
    setTranslationLanguage(l);
    for (const text of ['Waiting for your answer', 'Waiting for your answer below']) assert.notEqual(tr(text), text, `${l}: ${text}`);
  }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4748
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
const walk = dir => readdirSync(dir).flatMap(name => { const p = join(dir, name); return statSync(p).isDirectory() ? (name === 'locales' ? [] : walk(p)) : /\.tsx?$/.test(name) ? [p] : []; });

test('#4748 every Copy control goes through one shared behaviour: no bare clipboard call, no copy icon button of its own', () => {
  const files = walk(new URL('../src', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
  const bare = [], own = [];
  for (const file of files) {
    if (/lib[\/]clipboard\.ts$/.test(file)) continue;
    const text = readFileSync(file, 'utf8');
    if (/navigator\.clipboard\.writeText/.test(text)) bare.push(file);
    if (/<IconBtn name="copy"/.test(text)) own.push(file);
  }
  assert.deepEqual(bare, [], 'a bare clipboard call reports nothing when the clipboard is refused');
  assert.deepEqual(own, [], 'a copy icon button that is not CopyBtn shows nothing on itself');
});

test('#4748 the shared copy shows its state on the control for about two seconds, then goes back; a menu item says Copied and then closes', () => {
  const hook = src('lib/use-copy.ts');
  assert.match(hook, /COPIED_MS = 1800/);
  assert.match(hook, /COPIED_MENU_MS = 900/);
  assert.match(hook, /if \(ok\) \{\s*setCopied\(true\);/, 'only a copy that worked is shown');
  const btn = src('components/CopyBtn.tsx');
  assert.match(btn, /data-copied=\{copied \|\| undefined\}/);
  assert.match(btn, /name=\{copied \? 'check' : 'copy'\}/);
  assert.match(btn, /role="status">\{copied \? tr\('Copied'\) : ''\}/, 'a screen reader hears Copied');
  assert.match(btn, /e\.stopPropagation\(\)/, 'the menu does not close before the person sees it');
  const css = src('styles/app.css');
  assert.match(css, /\.st-msg__actions:has\(\[data-copied\]\)\s*\{\s*opacity:\s*1/, 'the hover-only action row stays while it says Copied');
});

test('#4748 Copied is translated in every language', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) { setTranslationLanguage(l); assert.notEqual(tr('Copied'), 'Copied', l); }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4804
test('#4804 every condition that greys out Start tuning has a name, and the ones that last have a sentence', async () => {
  const { tuneStartBlock, TUNE_BLOCK_TEXT } = await withI18n('native-tune.ts');
  const status = { ok: true };
  assert.equal(tuneStartBlock({ busy: false, otherRunning: false, status, error: '' }), null);
  assert.equal(tuneStartBlock({ busy: true, otherRunning: false, status, error: '' }), 'busy', 'a click being carried out: a moment, no sentence');
  assert.equal(tuneStartBlock({ busy: false, otherRunning: true, status, error: '' }), 'other');
  assert.equal(tuneStartBlock({ busy: false, otherRunning: false, status: null, error: '' }), 'loading');
  assert.equal(tuneStartBlock({ busy: false, otherRunning: false, status: null, error: 'HTTP 500' }), 'unreadable');
  assert.equal(tuneStartBlock({ busy: true, otherRunning: true, status: null, error: '' }), 'other', 'the first reason in the order wins');
  assert.deepEqual(Object.keys(TUNE_BLOCK_TEXT).sort(), ['loading', 'other', 'unreadable']);
  // It is exactly the old rule: disabled when busy, another operation runs, or there is no status.
  for (const busy of [false, true]) for (const otherRunning of [false, true]) for (const s of [null, status]) assert.equal(tuneStartBlock({ busy, otherRunning, status: s, error: '' }) !== null, busy || otherRunning || !s);
});

test('#4804 the dialog shows the sentence, and does not say "No saved tuning" while it is still reading', () => {
  const dialog = src('components/NativeTune.tsx');
  assert.match(dialog, /disabled=\{block !== null\}/);
  assert.match(dialog, /tune-why/);
  assert.match(dialog, /block === 'loading' \? <p className="small muted">\{tr\("Checking for saved tunings\\u2026"\)\}/);
  assert.doesNotMatch(dialog, /disabled=\{busy \|\| !!other \|\| !status\}/);
});

test('#4804 the new sentences are translated, and the existing one is reused', async () => {
  const { tr, setTranslationLanguage, TUNE_BLOCK_TEXT } = await withI18n('native-tune.ts');
  for (const l of languages.slice(1)) { setTranslationLanguage(l); for (const text of Object.values(TUNE_BLOCK_TEXT)) assert.notEqual(tr(text), text, `${l}: ${text}`); assert.notEqual(tr('Checking for saved tunings\u2026'), 'Checking for saved tunings\u2026', l); }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4721
test('#4721 a plain click on a web address goes to the default browser; the app window asks the host, a browser tab opens it itself', async () => {
  const { openExternal } = await withI18n('open-external.ts');
  const seen = [];
  const deps = mode => ({ known: () => mode, load: async () => mode, post: async (...a) => { seen.push(['post', ...a]); }, open: u => seen.push(['open', u]) });
  await openExternal('https://example.com/a?b=1', deps({ mode: 'window' }));
  assert.deepEqual(seen.splice(0), [['post', '/api/artifacts/open', { action: 'open', url: 'https://example.com/a?b=1' }]]);
  await openExternal('https://example.com/a', deps({ mode: 'browser' }));
  assert.deepEqual(seen.splice(0), [['open', 'https://example.com/a']]);
  await openExternal('https://example.com/a', deps({ mode: null }));
  assert.deepEqual(seen.splice(0), [['open', 'https://example.com/a']], 'a host that does not say: the browser way');
  // The answer has not arrived yet: the click waits for it, so the app window is not left with a window.open it drops.
  let release;
  const late = { known: () => null, load: () => new Promise(r => { release = () => r({ mode: 'window' }); }), post: async (...a) => { seen.push(['post', ...a]); }, open: u => seen.push(['open', u]) };
  const pending = openExternal('https://example.com/late', late);
  release();
  await pending;
  assert.equal(seen.splice(0)[0][0], 'post');
  // The host cannot open it: the page tries too.
  await openExternal('https://example.com/x', { ...deps({ mode: 'window' }), post: async () => { throw new Error('no'); } });
  assert.deepEqual(seen.splice(0), [['open', 'https://example.com/x']]);
});

test('#4721 only a website goes to the browser; a file keeps the side panel', async () => {
  globalThis.location = { origin: 'http://127.0.0.1:1' };
  const { artifactFromUrl } = await withI18n('artifacts.ts');
  for (const url of ['https://example.com/docs', 'https://github.com/Niko1221/Strata/issues/4721', 'http://localhost:3000/']) assert.equal(artifactFromUrl(url).kind, 'website', url);
  for (const [url, kind] of [['https://example.com/a.pdf', 'pdf'], ['https://example.com/a.png', 'image'], ['https://example.com/a.csv', 'table'], ['https://example.com/a.ts', 'code'], ['https://example.com/readme.md', 'markdown']]) assert.equal(artifactFromUrl(url).kind, kind, url);
  assert.equal(artifactFromUrl('javascript:alert(1)'), null);
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /if \(item\.kind === 'website'\) \{\s*void openExternal\(url\);\s*return;\s*\}\s*openPreview\(item\);/);
});

// ---------------------------------------------------------------------------------------------------------------------- #4802
test('#4802 / GUI-FB2 item 3: the progress card shows the step and the numbers only, with Stop as a plain word', () => {
  const models = src('components/NativeModels.tsx');
  assert.match(models, /className="native-progress__ring"/, 'a soft ring beside the step, no spinner box');
  assert.match(models, /className="native-progress__stop"/, 'Stop is a text button');
  assert.doesNotMatch(models, /Still working for/, 'no clock sentence');
  assert.doesNotMatch(models, /The buttons below wait until this finishes/, 'no sentence about the buttons');
  assert.doesNotMatch(models, /native-wait-note/);
  assert.match(models, /role="progressbar"/, 'the bar says its value to a screen reader');
  const css = src('styles/app.css');
  assert.match(css, /\.native-progress__bar > i \{[^}]*transform: scaleX\(0\)/, 'the bar fills by transform, not by layout');
  assert.match(css, /@media \(prefers-reduced-motion: reduce\) \{\s*\.native-progress__ring/, 'reduced motion holds the ring and the bar still');
  assert.match(css, /html\[data-motion='reduced'\] \.native-progress__ring/);
  assert.match(css, /\.native-progress__stop \{[^}]*border: 0; background: transparent;/);
});

test('#4802 the elapsed time is counted from the job start and never goes below zero', async () => {
  const hook = src('lib/use-elapsed.ts');
  assert.match(hook, /Math\.max\(0, now - start\)/);
  assert.match(hook, /setInterval\(\(\) => setNow\(Date\.now\(\)\), 1000\)/);
  assert.match(hook, /return \(\) => clearInterval\(timer\)/);
});

test('#4802 the progress card has no new sentence of its own: Stop and Stopping are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) {
    setTranslationLanguage(l);
    assert.notEqual(tr('Stop'), 'Stop', l);
    assert.notEqual(tr('Stopping…'), 'Stopping…', l);
  }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4798
test('#4798 while sharing is on, Serving shows the address on this computer and the one for other devices, with one sentence', () => {
  const page = src('screens/Serving.tsx');
  assert.match(page, /const twoAddresses = !!serving\.sharing\?\.active && !!serving\.sharing\.url && !!serving\.localUrl && serving\.localUrl !== serving\.sharing\.url;/);
  assert.match(page, /Address for other devices/);
  assert.equal((page.match(/Other devices use this address\./g) ?? []).length, 1);
  const host = readFileSync(new URL('../server/serving.mjs', import.meta.url), 'utf8');
  assert.match(host, /localUrl: baseUrl\(\),/);
  assert.match(host, /baseUrl: sharing\.active \? sharing\.url : baseUrl\(\),/, 'the existing field keeps its meaning for the connection guide');
});

test('#4798 the three new Serving sentences are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) {
    setTranslationLanguage(l);
    for (const text of ['API base URL on this computer', 'Address for other devices', "Other devices use this address. It has its own port, separate from the one Strata's window uses, because only the model's API is shared on the network, not the app."]) assert.notEqual(tr(text), text, `${l}: ${text}`);
  }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4722
test('#4722 paragraphs, lists and quotes share one measure; tables and code blocks keep the full column', () => {
  const css = src('styles/app.css');
  assert.match(css, /\.st-msg--assistant \.st-bubble :is\(p, ul, ol, blockquote\) \{\s*max-width: var\(--st-measure\);\s*\}/);
  assert.doesNotMatch(css, /\.st-bubble \.md-li/, 'no rule names the class that the renderer never emits');
  assert.doesNotMatch(css, /\.st-bubble[^{]*(table|\.md-table|\.st-code)[^{]*\{[^}]*max-width: var\(--st-measure\)/);
  // The renderer really emits bare lists, which is what the rule now targets.
  const md = src('components/Markdown.tsx');
  assert.match(md, /ordered \? <ol key=\{key\}>\{tr\(items\)\}<\/ol> : <ul key=\{key\}>/);
  assert.doesNotMatch(md, /className="md-li"/);
});

// ---------------------------------------------------------------------------------------------------------------------- #3723
test('#3723 a first start opens in the first of the computer\'s languages that Strata has', async () => {
  const { detectLanguage } = await withI18n('language-detect.ts');
  assert.equal(detectLanguage(['zh-CN', 'en']), 'zh');
  assert.equal(detectLanguage(['zh-TW']), 'zh');
  assert.equal(detectLanguage(['pt-BR']), 'pt');
  assert.equal(detectLanguage(['sv-SE', 'de-DE', 'en']), 'de', 'the first one that is offered, not the first one');
  assert.equal(detectLanguage(['ja']), 'ja');
  assert.equal(detectLanguage(['sv-SE', 'fi']), null);
  assert.equal(detectLanguage([]), null);
  assert.equal(detectLanguage(undefined), null);
  assert.equal(detectLanguage(['KO_kr']), 'ko');
});

test('#3723 only a first start is detected: a saved language, even English, is kept; the first screens carry the choice', () => {
  const store = src('lib/store.ts');
  assert.match(store, /language: validLanguage\(parsed\.language\) \? parsed\.language : 'en'/, 'a saved value is read as before');
  assert.match(store, /return \{ \.\.\.PREFS_DEFAULTS, language: detectLanguage\(tags\) \?\? 'en' \};/);
  assert.match(src('components/EngineFirstRun.tsx'), /<FirstRunLanguage\/>/);
  assert.match(src('components/GuidedSetup.tsx'), /step === 1 \? <FirstRunLanguage\/> : null/);
  assert.match(src('components/FirstRunLanguage.tsx'), /setLanguage\(value as Language\)/);
});

test('#3723 Context memory has one line per format, from the documented memory figures', () => {
  const models = src('components/NativeModels.tsx');
  assert.match(models, /KV_CHOICES\.map\(c => \(\{ value: c\.value, label: c\.label, title: c\.hint \}\)\)/);
  for (const text of ['INT8 is the standard: good quality and a medium amount of memory.', 'Q4 uses about half the memory of INT8, but is less precise on very long documents.', 'K8 / V4 uses about a quarter less memory than INT8 and is more precise than Q4. Prompts are 2 to 5% slower.']) assert.ok(models.includes(text), text);
  // The figures are the ones of docs/DETAILS.md and of native-fit.ts: q4_0 576 and k8v4 816 against int8 1056 bytes per token and layer.
  const fit = src('lib/native-fit.ts');
  assert.match(fit, /int8: 1056, q4_0: 576, k8v4: 816/);
  assert.ok(Math.abs(576 / 1056 - 0.55) < 0.06 && Math.abs(1 - 816 / 1056 - 0.23) < 0.01);
});

test('#3723 a start that ran out of graphics memory says so, with the engine\'s own words under Details', async () => {
  const { plainFailure } = await withI18n('plain-errors.ts');
  const reported = "server_exited (exit code 1): EngineDied: the engine exited before it was ready | the engine process ended: exit code 1: \nthe engine log's last lines:\n  strata generate: expert cache auto: 1.92 GiB free, 2048 MiB reserved (+184 MiB for the draft head) -> 0 slots\n  strata generate: no VRAM is left for the expert cache: it needs at least 332 slots (737 MiB), about 1000 MiB more than this card has free.";
  const r = plainFailure(reported);
  assert.equal(r.kind, 'vram');
  assert.equal(r.message, 'The model did not fit in your graphics card’s memory.');
  assert.match(r.next, /Context memory to Q4/);
  assert.equal(r.detail, reported, 'the original text stays for Details');
  // Other stops are not taken for it.
  assert.equal(plainFailure('server_exited (exit code 1): EngineDied: the engine exited before it was ready').kind, 'start-failed');
  assert.equal(plainFailure('out of memory').kind, 'memory');
  assert.equal(plainFailure('0xC0000135 DLL not found').kind, 'dll-missing');
});

test('#3723 the new sentences are translated in every language', async () => {
  const { tr, setTranslationLanguage, plainFailure } = await withI18n('plain-errors.ts');
  const lines = ['INT8 is the standard: good quality and a medium amount of memory.', 'Q4 uses about half the memory of INT8, but is less precise on very long documents.', 'K8 / V4 uses about a quarter less memory than INT8 and is more precise than Q4. Prompts are 2 to 5% slower.', 'The model did not fit in your graphics card’s memory.', 'Lower the conversation context, or set Context memory to Q4 in Expert settings, then try again.'];
  for (const l of languages.slice(1)) { setTranslationLanguage(l); for (const text of lines) assert.notEqual(tr(text), text, `${l}: ${text}`); }
  setTranslationLanguage('en');
  void plainFailure;
});

// ---------------------------------------------------------------------------------------------------------------------- #4749
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

test('#4749 a message for a helper is checked before anything is sent, and an agent that is not running says so', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-helper-msg-'));
  try {
    const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [], connections: () => [], env: process.env });
    const code = async args => { try { await bridge.messageHelper(args); return 'sent'; } catch (e) { return e.code; } };
    assert.equal(await code({ chatId: 'c', helperId: 'child-1', text: '   ' }), 'helper-empty');
    assert.equal(await code({ chatId: 'c', helperId: '', text: 'hi' }), 'helper-unknown');
    assert.equal(await code({ chatId: 'c', helperId: 'a b; rm -rf /', text: 'hi' }), 'helper-unknown');
    assert.equal(await code({ chatId: 'c', helperId: 'child-1', text: 'use ISO dates' }), 'helper-idle', 'no running agent, no live parent');
  } finally { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); }
});

test('#4749 the bridge uses the Harness own subagents/prompt with the documented request, steer for a working helper, queue for a finished one', () => {
  const bridge = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  assert.match(bridge, /rt\.gw\.rpc\('subagents\/prompt', \{ request: \{ requestId: `helper-\$\{randomUUID\(\)\}`, parentSessionId: parent, childSessionId: String\(helperId\), mode: 'continuable', delivery: delivery === 'steer' \? 'steer' : 'queue', content: \[\{ type: 'text', text: clean \}\] \} \}\)/);
  const index = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(index, /path === '\/api\/chat\/helper' && method === 'POST'/);
  const row = src('components/SubagentRow.tsx');
  assert.match(row, /delivery: a\.status === 'working' \? 'steer' : 'queue'/);
  assert.match(row, /sessionId && a\.status !== 'queued' && a\.status !== 'cancelled'/);
  assert.match(src('components/ChatToolbar.tsx'), /sessionId=\{session && !session\.showcase \? session\.id : undefined\}/, 'the example chat has no real helpers to write to');
});

test('#4749 the helper strings are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) {
    setTranslationLanguage(l);
    for (const [text, values] of [['Message this helper'], ['Message to {0}', ['X']], ['Write to {0}…', ['X']], ['Sent to {0}.', ['X']]]) { const out = tr(text, values); assert.notEqual(out, text.replace('{0}', 'X'), `${l}: ${text}`); }
  }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4783
test('#4783 the three choices while an answer is written, as the web chat page has them, and only those three are valid', async () => {
  const { busyPlan, validWhileBusy, WHILE_BUSY } = await load('lib/while-busy.ts');
  assert.deepEqual([...WHILE_BUSY], ['wait', 'queue', 'interject']);
  assert.equal(busyPlan('wait'), 'refuse');
  assert.equal(busyPlan('queue'), 'goal', 'a running goal has its own queue in the host; with none, the message is parked');
  assert.equal(busyPlan('interject'), 'interrupt');
  assert.equal(validWhileBusy('queue'), true);
  for (const bad of ['', 'steer', 'QUEUE', null, undefined, 3]) assert.equal(validWhileBusy(bad), false, String(bad));
});

test('#4783 the composer acts on the choice, the default keeps today\'s behaviour, and the parked message is sent when the chat is quiet', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /if \(plan === 'interrupt'\) \{\s*updateDraft\(''\);\s*clearQuotes\(session\.id\);\s*parkMessage\(session\.id, out\);\s*stopStreaming\(\);\s*return;\s*\}/);
  assert.match(chat, /if \(plan === 'goal'\) \{ clearQuotes\(session\.id\); parkMessage\(session\.id, out\); return; \}/);
  assert.match(chat, /pushToast\('info', tr\('The assistant is still answering'\)/, 'Wait still says so');
  assert.match(chat, /if \(!session \|\| streaming \|\| !parked\?\.length\)\s*return;\s*const next = takeParked\(session\.id\);/);
  const store = src('lib/store.ts');
  assert.match(store, /whileBusy: validWhileBusy\(parsed\.whileBusy\) \? parsed\.whileBusy : 'wait'/, 'an old or invalid saved value reads as Wait');
  assert.match(store, /\n  whileBusy: 'wait',\n/);
  assert.match(src('screens/Settings.tsx'), /setWhileBusy\(value as typeof prefs\.whileBusy\)/);
});

test('#4783 the new sentences are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  const lines = ['While an answer is being written', 'What Enter does when you send a message while the assistant is still answering.', 'Wait for the answer', 'Send after the answer', 'Interrupt the answer', 'Enter sends it when the answer is done.', 'Enter stops the answer here and sends it now.', 'Waiting to send', 'Take back'];
  for (const l of languages.slice(1)) { setTranslationLanguage(l); for (const text of lines) assert.notEqual(tr(text), text, `${l}: ${text}`); }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4785
test('#4785 a quoted message carries the quotes in one fixed form and reads back the same, apart from the reply', async () => {
  const { composeQuoted, parseQuoted, cleanQuote, addToQuotes, MAX_QUOTES, MAX_QUOTE_CHARS } = await load('lib/quotes.ts');
  const out = composeQuoted(['The first passage.', 'A second\npassage.'], 'my answer');
  assert.equal(out, 'Quoting:\n「The first passage.」\n「A second\npassage.」\n→ Reply: my answer');
  assert.deepEqual(parseQuoted(out), { quotes: ['The first passage.', 'A second\npassage.'], reply: 'my answer' });
  // A reply that itself looks like the form is still the reply.
  assert.deepEqual(parseQuoted(composeQuoted(['q'], '」\n→ Reply: x')), { quotes: ['q'], reply: '」\n→ Reply: x' });
  // A quote cannot end itself early.
  assert.equal(cleanQuote('he said 「hi」'), 'he said 『hi』');
  assert.deepEqual(parseQuoted(composeQuoted(['he said 「hi」'], 'ok')).quotes, ['he said 『hi』']);
  assert.equal(composeQuoted([], 'plain'), 'plain');
  assert.equal(composeQuoted(['   '], 'plain'), 'plain');
  assert.equal(parseQuoted('hello'), null);
  assert.equal(parseQuoted('Quoting:\nnothing'), null);
  assert.ok(cleanQuote('x'.repeat(MAX_QUOTE_CHARS + 50)).length <= MAX_QUOTE_CHARS);
  // The list: no empty one, no repeat of the same passage, no more than the limit.
  let list = [];
  list = addToQuotes(list, { text: 'a', messageId: 'm1', role: 'assistant' });
  list = addToQuotes(list, { text: 'a', messageId: 'm1', role: 'assistant' });
  list = addToQuotes(list, { text: '  ', messageId: 'm1', role: 'assistant' });
  assert.equal(list.length, 1);
  list = addToQuotes(list, { text: 'a', messageId: 'm2', role: 'user' });
  assert.equal(list.length, 2, 'the same words from another message is another quote');
  for (let i = 0; i < 20; i++) list = addToQuotes(list, { text: `n${i}`, messageId: 'm3', role: 'assistant' });
  assert.equal(list.length, MAX_QUOTES);
});

test('#4785 quoting is wired: the menu, the selection, the box above the composer, the sending and the sent bubble', () => {
  const chat = src('screens/Chat.tsx');
  assert.equal((chat.match(/addQuote\(activeSession, m\.content, m\.id, '(user|assistant)'\)/g) ?? []).length, 2, 'a Quote item in the menu of both kinds of message');
  assert.match(chat, /document\.addEventListener\('selectionchange', read\)/);
  assert.match(chat, /data-message-id=\{m\.id\}/);
  assert.match(chat, /return quotes\.length && !text\.startsWith\('\/'\) \? composeQuoted\(quotes\.map\(q => q\.text\), text\) : text;/, 'a command goes out as typed');
  assert.match(chat, /void sendMessage\(session \? quotedMessage\(quotes, text\) : text\);/);
  assert.match(chat, /const sentQuote = m\.role === 'user' \? parseQuoted\(m\.content\) : null;/);
});

test('#4785 the new words are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  const lines = ['Quote', 'Quoting the assistant', 'Quoting you', 'Remove this quote', 'Go to the quoted message', 'Quoted', 'Show or hide the rest of the quote'];
  for (const l of languages.slice(1)) { setTranslationLanguage(l); for (const text of lines) assert.notEqual(tr(text), text, `${l}: ${text}`); }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4244
test('#4244 the question of the Harness is read into something the card can draw, and what cannot be shown is left out', async () => {
  const { parseQuestions, buildAnswer } = await import(new URL('../server/agent/user-questions.mjs', import.meta.url));
  const raw = [
    { id: 'q1', question: ' Which colour? ', detail: 'Why', header: 'Colour', options: [{ label: 'Red', description: 'Warm' }, { label: '' }, { label: 'Blue' }] },
    { id: '', question: 'no id' }, { id: 'q3', question: '' }, null, 'x',
    { id: 'q2', question: 'Extras?', multiSelect: true, options: [{ label: 'A' }, { label: 'B' }] },
  ];
  const items = parseQuestions(raw);
  assert.deepEqual(items.map(q => q.id), ['q1', 'q2']);
  assert.deepEqual(items[0].options, [{ label: 'Red', description: 'Warm' }, { label: 'Blue', description: undefined }]);
  assert.equal(items[0].question, 'Which colour?'); assert.equal(items[1].multiSelect, true); assert.equal(items[0].multiSelect, false);
  assert.deepEqual(parseQuestions(undefined), []);
  assert.equal(parseQuestions(Array.from({ length: 20 }, (_, i) => ({ id: `q${i}`, question: 'x' }))).length, 6);
  // The answer is checked against what was asked, and has the shape of AskUserQuestionAnswer.
  assert.deepEqual(buildAnswer(items, [{ id: 'q1', selected: ['Blue'] }, { id: 'q2', selected: ['A', 'B', 'A'] }]), { answers: [{ id: 'q1', selected: ['Blue'] }, { id: 'q2', selected: ['A', 'B'] }] });
  assert.deepEqual(buildAnswer(items, [{ id: 'q1', selected: [], custom: ' my own ' }, { id: 'q2', selected: ['B'], custom: 'and more' }]), { answers: [{ id: 'q1', selected: [], custom: 'my own' }, { id: 'q2', selected: ['B'], custom: 'and more' }] });
  assert.throws(() => buildAnswer(items, [{ id: 'q1', selected: ['Blue'] }]), /Answer every question/, 'one question left open');
  assert.throws(() => buildAnswer(items, [{ id: 'q1', selected: ['Green'] }, { id: 'q2', selected: ['A'] }]), /Answer every question/, 'a label that was not offered is no answer');
  assert.throws(() => buildAnswer(items, [{ id: 'q1', selected: ['Red', 'Blue'] }, { id: 'q2', selected: ['A'] }]), /one answer/, 'single-select takes one');
});

test('#4244 the page side: the card needs every question answered, and sends the offered labels and the own words', async () => {
  const { readAsk, answerComplete, toAnswers } = await load('lib/agent-ask.ts');
  const ask = readAsk({ callId: 'c1', questions: [{ id: 'q1', question: 'Colour?', options: [{ label: 'Red' }], multiSelect: false }, { id: 'q2', question: 'Extras?', options: [{ label: 'A' }], multiSelect: true }, { id: '', question: 'bad' }] });
  assert.equal(ask.questions.length, 2);
  assert.equal(readAsk({ callId: '', questions: [{ id: 'q', question: 'x' }] }), null);
  assert.equal(readAsk({ callId: 'c', questions: [] }), null);
  assert.equal(answerComplete(ask.questions, {}), false);
  assert.equal(answerComplete(ask.questions, { q1: { selected: ['Red'], custom: '' } }), false);
  assert.equal(answerComplete(ask.questions, { q1: { selected: [], custom: '  ' }, q2: { selected: ['A'], custom: '' } }), false);
  assert.equal(answerComplete(ask.questions, { q1: { selected: [], custom: 'mine' }, q2: { selected: ['A'], custom: '' } }), true);
  assert.deepEqual(toAnswers(ask.questions, { q1: { selected: [], custom: ' mine ' }, q2: { selected: ['A'], custom: '' } }), [{ id: 'q1', selected: [], custom: 'mine' }, { id: 'q2', selected: ['A'] }]);
});

test('#4244 the tool is offered again, as the group "Questions for you", and the bridge answers through the Harness result', () => {
  const preset = readFileSync(new URL('../server/agent/preset.mjs', import.meta.url), 'utf8');
  assert.match(preset, /export const REMOVED_PLUGIN_IDS = \['tool-schedule', 'planning'\];/);
  assert.match(preset, /\{ id: 'tool-ask-user', name: '@deepseek-ai\/dsh-tool-ask-user' \}/);
  const catalog = readFileSync(new URL('../server/agent/tools-catalog.mjs', import.meta.url), 'utf8');
  assert.match(catalog, /id: 'questions'[^\n]*plugins: \['tool-ask-user'\]/);
  const bridge = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  assert.match(bridge, /await rt\.gw\.answerEvent\(\{ clientId: entry\.clientId \?\? rt\.clientId, eventId: entry\.eventId, outcome \}\);/);
  assert.match(bridge, /if \(!turn \|\| turn\.finished \|\| turn\.stopping \|\| turn\.goalRun \|\| turn\.plan \|\| !turn\.announceQuestion \|\| !items\.length\) \{\s*void runtime\.gw\.answerEvent\(\{ clientId: runtime\.clientId, eventId: value\.eventId, outcome: 'next' \}\)/, 'a helper question, a stopping chat or an empty request is still declared unanswerable');
  const gateway = readFileSync(new URL('../server/agent/gateway-client.mjs', import.meta.url), 'utf8');
  assert.match(gateway, /const wire = outcome === 'next' \? \{ kind: 'next' \} : \{ kind: 'result', value: outcome \};/, 'a structured answer travels as the result value');
  assert.match(src('components/ApprovalCard.tsx'), /asks\.map\(a => <QuestionCard key=\{a\.callId\} sessionId=\{sessionId!\} ask=\{a\}\/>\)/);
});

test('#4244 the new words are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  const lines = ['The assistant has a question for you', 'Type your own answer', 'Send answer', 'Skip', 'Questions for you', 'Lets the assistant stop and ask you a question, with answers to pick from, when it needs your decision.'];
  for (const l of languages.slice(1)) { setTranslationLanguage(l); for (const text of lines) assert.notEqual(tr(text), text, `${l}: ${text}`); }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #3818
test('#3818 the transcript lets go of the newest token while the reader has words selected in it, and offers the way back', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /function readerHasSelection\(scroller: HTMLElement\): boolean \{\s*const sel = window\.getSelection\(\);\s*return !!sel && !sel\.isCollapsed && !!sel\.anchorNode && scroller\.contains\(sel\.anchorNode\) && sel\.toString\(\)\.trim\(\)\.length > 0;/);
  assert.match(chat, /if \(pinned\.current && readerHasSelection\(el\)\) \{\s*\/\/[^\n]*\n\s*pinned\.current = false;\s*setAwayFromBottom\(distanceFromBottom\(el\) > PILL_DISTANCE\);\s*\}\s*if \(pinned\.current\)\s*el\.scrollTop = el\.scrollHeight;/, 'the observer that follows a growing reply');
  assert.match(chat, /if \(pinned\.current && !sent && readerHasSelection\(el\)\) \{\s*pinned\.current = false;/, 'and the effect that follows each token; sending a message still brings the reader to it');
});

test('#4244 a goal, a plan step and a loop run without anyone there, so a question is declined at once instead of waiting', () => {
  const bridge = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  assert.match(bridge, /if \(!turn \|\| turn\.finished \|\| turn\.stopping \|\| turn\.goalRun \|\| turn\.plan \|\| !turn\.announceQuestion \|\| !items\.length\) \{\s*void runtime\.gw\.answerEvent\(\{ clientId: runtime\.clientId, eventId: value\.eventId, outcome: 'next' \}\)/);
  assert.match(src('lib/store.ts'), /if \(ask && loopScheduler\.isRunning\(sessionId\)\) \{ void answerAgentAsk\(sessionId, ask\.callId, null\)\.catch\(\(\) => \{\}\); return; \}/);
});
