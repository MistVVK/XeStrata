#!/usr/bin/env node
/**
 * Drives the built app in a headless Chrome against a real host with a clean temporary profile (no engine, no model), for the 0.2.10
 * interface changes (branch f-0291-ui). Each case seeds the chat history it needs into the page's local storage, so the real screens
 * draw real data without a model.
 *
 *   npx vite build && node scripts/ui-do-drive.mjs <shots-dir> [case ...]
 *
 * Never point it at a real install: the host it starts uses a fresh STRATA_HOME in the temp folder and a random port.
 * Light and dark, at a normal and a narrow window. Prints "ok" / "FAIL" lines and exits 1 on a failure.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { startScriptedWorker } from '../tests/helpers/scripted-worker.mjs';

const HOST_DIR = resolve(import.meta.dirname, '..');
const SHOTS = resolve(process.argv[2] ?? 'ui-shots');
const ONLY = process.argv.slice(3);
mkdirSync(SHOTS, { recursive: true });
const { chromium } = await import(pathToFileURL(join(HOST_DIR, 'node_modules/playwright-core/index.mjs')).href);
const CHROME = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/usr/bin/google-chrome'].find(existsSync);
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let failed = false;
export const check = (ok, label, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${extra ? '  ' + extra : ''}`); if (!ok) failed = true; };

const home = mkdtempSync(join(tmpdir(), 'strata-ui-do-'));
const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
const port = await freePort();
const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components'), STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'), XDG_CONFIG_HOME: join(home, 'xdg-config'), XDG_STATE_HOME: join(home, 'xdg-state') };
delete env.STRATA_HARNESS_DIR;
const child = spawn(process.execPath, [join(HOST_DIR, 'server', 'index.mjs')], { cwd: HOST_DIR, env, stdio: 'ignore', windowsHide: true });
export const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/health`)).ok) break; } catch { /* starting */ } await sleep(200); }
// A saved API connection to a scripted model gives the app a model to talk to, so it stays on the chat screen (no model loaded would send it to Models).
const worker = await startScriptedWorker({ model: 'cloud-model' });
const saved = await (await fetch(`${base}/api/model-providers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ kind: 'custom', name: 'Scripted API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'sk-ui-do-test-0123456789', models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] }) })).json();
const remoteModel = { connectionId: saved.connections?.[0]?.id, modelId: 'cloud-model' };
check(!!remoteModel.connectionId, 'a scripted API model is connected');
// "Go straight to the app": the first-run screens are not what these cases look at.
await fetch(`${base}/api/setup/step`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ step: 1, dismissed: true, flowVersion: 2 }) });
const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });

const iso = (n = 0) => new Date(Date.UTC(2026, 9, 31, 9, 0, n)).toISOString();
export const user = (id, content, n = 0) => ({ id, role: 'user', content, at: iso(n) });
export const reply = (id, content, extra = {}) => ({ id, role: 'assistant', content, at: iso(5), tokens: 120, tps: 40, durationMs: 3000, ...extra });
export const chat = (id, title, messages) => ({ id, title, defaultTitle: false, updatedAt: iso(9), createdAt: iso(0), messages, workspace: ws, tokensIn: 0, tokensOut: 0, preferences: { useTools: true, mode: 'manual', remoteModel } });

/** A page with the given chats already in local storage. `prefs` are merged into the saved preferences. */
export async function open(scheme, width, height, sessions, { hash = '#/chat', prefs = {}, origin = base, before = null, locale = 'en-US', keepSetup = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, locale, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  pages.push(page);
  const problems = [];
  page.on('pageerror', e => problems.push(e.message));
  page.problems = problems;
  await page.addInitScript(({ sessions, prefs }) => {
    try {
      if (sessionStorage.getItem('ui-do-seeded')) return;
      sessionStorage.setItem('ui-do-seeded', '1');
      // The chats are files on the host: put the seed there (a synchronous request, so it is in place before the page starts).
      for (const chat of sessions ?? []) { const request = new XMLHttpRequest(); request.open('PUT', '/api/chats/' + encodeURIComponent(chat.id), false); request.setRequestHeader('x-strata-client', 'ui'); request.send(JSON.stringify({ messages: [], ...chat })); }
      if (Object.keys(prefs).length) localStorage.setItem('strata.prefs.v1', JSON.stringify(prefs));
    } catch { /* storage refused */ }
  }, { sessions, prefs });
  if (before) await before(page);
  await page.goto(`${origin}/${hash}`);
  await page.waitForTimeout(1800);
  const skip = page.getByText('Go straight to the app');
  if (!keepSetup && await skip.count()) { await skip.click(); await page.waitForTimeout(600); }
  return page;
}
export const shot = (page, name) => page.screenshot({ path: join(SHOTS, `${name}.png`) });
export const sizes = { wide: [1280, 860], narrow: [390, 800] };

export const cases = {};
const pages = [];

/* ------------------------------------------------------------------------------------------------------------------- #4725 */
const longReply = Array.from({ length: 70 }, (_, i) => `Paragraph ${i + 1}. This is part of a long answer from the assistant that goes on for many screens before it asks to change a file in the project.`).join('\n\n');
const askTool = (id, state = 'proposed', file = 'notes.txt') => ({ id, name: 'write', args: { file_path: file }, preview: `write ${file}\n\na\nb\nc`, effect: 'Creates a file.', scope: 'write', verb: 'Create a file', target: file, state, ask: { id: 'create-file', params: [file], why: 'replaces-if-exists' }, live: false });
const plainTool = (id, state = 'proposed') => ({ id, name: 'bash', args: {}, preview: 'npm install left-pad', effect: 'Runs a command on your computer.', scope: 'shell', verb: 'Run a command', target: 'npm install left-pad', state });
cases.approval = async () => {
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const sessions = [chat('s1', 'Long reply', [user('u1', 'Please write notes.txt'), reply('a1', longReply, { tools: [askTool('t1')] })])];
    const page = await open(scheme, w, h, sessions);
    const dock = page.locator('.approvaldock');
    await dock.waitFor({ timeout: 15000 });
    const view = page.viewportSize();
    const inView = async sel => { const b = await page.locator(sel).first().boundingBox(); return !!b && b.y >= 0 && b.y + b.height <= view.height + 1 && b.x >= 0 && b.x + b.width <= view.width + 1; };
    check(await page.locator('.askcard').count() === 1, `${tag}: one Approve/Deny card on the page`);
    check(await page.getByRole('button', { name: 'Approve' }).count() === 1 && await page.getByRole('button', { name: 'Deny' }).count() === 1, `${tag}: one Approve and one Deny`);
    check(await inView('.approvaldock'), `${tag}: the card is inside the window at the end of the reply`);
    const scroller = page.locator('.main__body.chat');
    const atEnd = () => scroller.evaluate(el => el.scrollHeight - el.scrollTop - el.clientHeight);
    check(await atEnd() < 4, `${tag}: the reply is still scrolled to its end`, String(await atEnd()));
    check(await page.locator('.approval-wait').count() === 1, `${tag}: the reply says it waits for an answer`);
    await shot(page, `4725-${tag}-end`);
    await scroller.evaluate(el => { el.scrollTop = 0; });
    await page.waitForTimeout(300);
    check(await inView('.approvaldock'), `${tag}: the card stays in the window when the reader is at the top`);
    const composerTop = await page.locator('.composer .st-composer').boundingBox();
    const dockBox = await dock.boundingBox();
    check(dockBox.y + dockBox.height <= composerTop.y, `${tag}: the card sits above the message box`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    await shot(page, `4725-${tag}-top`);
    await page.getByRole('button', { name: 'Approve' }).click();
    await page.waitForTimeout(500);
    check(await page.locator('.approvaldock').count() === 0, `${tag}: the card goes away after Approve`);
    check(await page.locator('.approval-wait').count() === 0, `${tag}: and so does the waiting line`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
  // Two waiting at once, one in each card style; Deny works from the dock.
  const sessions = [chat('s1', 'Two waiting', [user('u1', 'Do two things'), reply('a1', longReply.slice(0, 1500), { tools: [askTool('t1'), plainTool('t2')] })])];
  const page = await open('light', 1280, 860, sessions);
  await page.locator('.approvaldock').waitFor({ timeout: 15000 });
  check(await page.locator('.approvaldock > section').count() === 2, 'two waiting: both cards are in the dock');
  await shot(page, '4725-two-waiting');
  await page.getByRole('button', { name: 'Deny' }).first().click();
  await page.waitForTimeout(400);
  check(await page.locator('.approvaldock > section').count() === 1, 'two waiting: Deny on the first leaves the second');
  await page.context().close();
};

/* ------------------------------------------------------------------------------------------------------------------- #4748 */
const codeReply = 'Here is a command.\n\n```bash\nnpm install left-pad\n```\n\nThat is all.';
cases.copy = async () => {
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const sessions = [chat('s1', 'Copy', [user('u1', 'Show me a command'), reply('a1', codeReply), user('u2', 'Thanks'), reply('a2', 'You are welcome. Second reply.')])];
    const page = await open(scheme, w, h, sessions);
    await page.locator('.st-msg--assistant').last().waitFor({ timeout: 15000 });
    const flag = async loc => loc.evaluate(el => ({ copied: el.hasAttribute('data-copied'), icon: el.querySelector('use')?.getAttribute('href') ?? '' }));
    const read = () => page.evaluate(() => navigator.clipboard.readText());
    // The last reply's copy button (an icon in the action row).
    const copyResponse = page.getByRole('button', { name: 'Copy response' });
    await copyResponse.last().click();
    await page.mouse.move(2, 2);
    await page.waitForTimeout(250);
    let f = await flag(copyResponse.last());
    check(f.copied && f.icon.endsWith('i-check'), `${tag}: Copy response turns into a check mark`, JSON.stringify(f));
    check((await read()) === 'You are welcome. Second reply.', `${tag}: the reply is on the clipboard`);
    const row = page.locator('.st-msg--assistant').last().locator('.st-msg__actions');
    check(await row.evaluate(el => getComputedStyle(el).opacity) === '1', `${tag}: the action row stays visible while it says so, mouse away`);
    check(/Copied/.test(await page.locator('.st-msg--assistant').last().locator('[role=status]').first().innerText()), `${tag}: a screen reader hears Copied`);
    await shot(page, `4748-${tag}-reply`);
    await page.waitForTimeout(2200);
    f = await flag(copyResponse.last());
    check(!f.copied && f.icon.endsWith('i-copy'), `${tag}: after about two seconds it is a Copy button again`, JSON.stringify(f));
    // The code block.
    const code = page.locator('.st-code__head').getByRole('button', { name: 'Copy code' });
    await code.click();
    await page.waitForTimeout(250);
    f = await flag(code);
    check(f.copied && f.icon.endsWith('i-check'), `${tag}: the code block's Copy turns into a check mark`, JSON.stringify(f));
    check((await read()) === 'npm install left-pad', `${tag}: the code is on the clipboard`);
    await shot(page, `4748-${tag}-code`);
    await page.waitForTimeout(2200);
    // The menu item of an earlier reply: says Copied, then the menu closes.
    await page.locator('.st-msg--assistant').first().hover();
    await page.locator('.st-msg--assistant').first().getByRole('button', { name: 'More response actions' }).click();
    const item = page.getByRole('menuitem', { name: 'Copy response' });
    await item.click();
    await page.waitForTimeout(250);
    check(/Copied/.test(await page.locator('.menu [data-copied]').innerText()), `${tag}: the menu item says Copied`);
    check((await read()).startsWith('Here is a command'), `${tag}: the earlier reply is on the clipboard`);
    await shot(page, `4748-${tag}-menu`);
    await page.waitForTimeout(1300);
    check(await page.locator('.menu').count() === 0, `${tag}: and the menu closes by itself`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
  // A refused clipboard: nothing says Copied on the button, and the person is told in a toast.
  {
    const page = await open('light', 1280, 860, [chat('s1', 'Refused', [user('u1', 'Hi'), reply('a1', 'Hello there.')])]);
    await page.locator('.st-msg--assistant').last().waitFor({ timeout: 15000 });
    await page.evaluate(() => { Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.reject(new Error('denied')) }, configurable: true }); document.execCommand = () => false; });
    const b = page.getByRole('button', { name: 'Copy response' }).last();
    await b.click();
    await page.waitForTimeout(400);
    check(await b.evaluate(el => !el.hasAttribute('data-copied')), 'refused clipboard: the button does not claim success');
    check(/Couldn.t copy/.test(await page.locator('.st-toast').first().innerText()), 'refused clipboard: a toast says it did not work');
    await page.context().close();
  }
  // The Serving screen: the address and the request template.
  for (const scheme of ['light', 'dark']) {
    const page = await open(scheme, 1280, 860, [chat('s1', 'Serving', [user('u1', 'Hi'), reply('a1', 'Hello there.')])], { hash: '#/serving' });
    const c = page.getByRole('button', { name: 'Copy API base URL' });
    await c.waitFor({ timeout: 15000 });
    await c.click();
    await page.waitForTimeout(250);
    check(await c.evaluate(el => el.hasAttribute('data-copied') && (el.querySelector('use')?.getAttribute('href') ?? '').endsWith('i-check')), `${scheme}: Serving, Copy API base URL turns into a check mark`);
    check(/^http:\/\/127\.0\.0\.1:\d+\/v1$/.test(await page.evaluate(() => navigator.clipboard.readText())), `${scheme}: Serving, the address is on the clipboard`);
    await shot(page, `4748-serving-${scheme}`);
    await page.getByRole('button', { name: 'Connection guide' }).click();
    const t = page.getByRole('button', { name: 'Copy request template' });
    await t.click();
    await page.waitForTimeout(250);
    check(await t.evaluate(el => el.hasAttribute('data-copied')), `${scheme}: Serving, Copy request template turns into a check mark`);
    check((await page.evaluate(() => navigator.clipboard.readText())).startsWith('curl http'), `${scheme}: Serving, the template is on the clipboard`);
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4804 */
let tuneHostInfo = null;
async function tuneHost() {
  if (tuneHostInfo) return tuneHostInfo;
  const tport = await freePort();
  const proc = spawn(process.execPath, [join(HOST_DIR, 'tests/fixtures/tune-ui-host.mjs')], { cwd: HOST_DIR, env: { ...process.env, TUNE_UI_PORT: String(tport), TUNE_UPSTREAM: base, TUNE_SHOT_DIR: SHOTS }, stdio: 'ignore', windowsHide: true });
  const origin = `http://127.0.0.1:${tport}`;
  for (let i = 0; i < 100; i++) { try { if ((await fetch(`${origin}/api/native`)).ok) break; } catch { /* starting */ } await sleep(200); }
  tuneHostInfo = { origin, proc, mode: m => fetch(`${origin}/__fixture/status-mode`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ mode: m }) }) };
  return tuneHostInfo;
}
cases.tune = async () => {
  const t = await tuneHost();
  const sessions = [chat('s1', 'Tune', [user('u1', 'Hi'), reply('a1', 'Hello there.')])];
  const openDialog = async (scheme, w, h, mode, inject = false) => {
    await t.mode(mode);
    const page = await open(scheme, w, h, sessions, { hash: '#/models', origin: t.origin });
    if (inject) await page.route(/\/api\/native(\?.*)?$/, async route => {
      if (!page.injectJob) return route.continue();
      const r = await route.fetch(); const j = await r.json();
      j.jobs = [...(j.jobs ?? []), { id: 'job-x', kind: 'download', state: 'running', startedAt: new Date().toISOString(), summary: 'Downloading the model', error: null, result: null, progress: { text: 'Downloading', percent: 12 } }];
      return route.fulfill({ response: r, json: j });
    });
    const tune = page.getByRole('button', { name: 'Tune model' });
    await tune.waitFor({ timeout: 20000 });
    for (let i = 0; i < 40 && await tune.isDisabled(); i++) await page.waitForTimeout(250);
    await tune.click();
    await page.getByRole('dialog').waitFor({ timeout: 10000 });
    return page;
  };
  const start = page => page.getByRole('button', { name: 'Start tuning' });
  // The sentence must be on screen without scrolling the dialog, beside the button (above it when the window is narrow).
  const seen = async (page, w, h) => {
    const s = await page.locator('.tune-why').boundingBox(), b = await start(page).boundingBox();
    return !!s && !!b && s.y >= 0 && s.y + s.height <= h && s.x >= 0 && s.x + s.width <= w && Math.abs((s.y + s.height / 2) - (b.y + b.height / 2)) < 110;
  };
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    // 1. The saved tunings are still being read: greyed out, with the reason, and not "No saved tuning".
    let page = await openDialog(scheme, w, h, 'slow');
    await page.waitForTimeout(1200);
    check(await start(page).isDisabled(), `${tag}: loading: Start tuning is greyed out`);
    let why = await page.locator('.tune-why').innerText().catch(() => '');
    check(/still checking this model for saved tunings/.test(why), `${tag}: loading: one sentence says why`, why);
    check(await page.locator('.tune-why').count() === 1, `${tag}: loading: only one such sentence`);
    check(!/No saved tuning for these choices yet/.test(await page.getByRole('dialog').innerText()), `${tag}: loading: it does not claim there is no saved tuning`);
    check(await seen(page, w, h), `${tag}: loading: the sentence is on screen next to the button`);
    await shot(page, `4804-${tag}-loading`);
    await page.context().close();
    // 2. Reading them failed.
    page = await openDialog(scheme, w, h, 'fail');
    await page.waitForTimeout(1200);
    check(await start(page).isDisabled(), `${tag}: unreadable: Start tuning is greyed out`);
    why = await page.locator('.tune-why').innerText().catch(() => '');
    check(/could not read the tuning details/.test(why), `${tag}: unreadable: one sentence says why`, why);
    check(await seen(page, w, h), `${tag}: unreadable: the sentence is on screen next to the button`);
    check(!/No saved tuning/.test(await page.getByRole('dialog').innerText()), `${tag}: unreadable: it does not claim there is no saved tuning`);
    await shot(page, `4804-${tag}-unreadable`);
    await page.context().close();
    // 3. Another operation is running.
    page = await openDialog(scheme, w, h, 'ok', true);
    await page.waitForTimeout(800);
    check(await start(page).isEnabled() && await page.locator('.tune-why').count() === 0, `${tag}: ready: Start tuning works and shows no sentence`);
    page.injectJob = true;
    await page.waitForTimeout(3500);
    check(await start(page).isDisabled(), `${tag}: another operation: Start tuning is greyed out`);
    why = await page.locator('.tune-why').innerText().catch(() => '');
    check(/Finish the current operation first/.test(why), `${tag}: another operation: one sentence says why`, why);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    check(await seen(page, w, h), `${tag}: other: the sentence is on screen next to the button`);
    await shot(page, `4804-${tag}-other`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4721 */
cases.links = async () => {
  const sessions = () => [chat('s1', 'Links', [user('u1', 'Show links'), reply('a1', `Read [the docs](https://example.com/docs) or open [the logo](${base}/icons/sprite.svg).`)])];
  for (const scheme of ['light', 'dark']) {
    // A browser tab: the browser opens the address in a new tab; a file stays in the side panel.
    let page = await open(scheme, 1280, 860, sessions());
    await page.getByRole('link', { name: 'the docs' }).waitFor({ timeout: 15000 });
    await page.evaluate(() => { window.__opened = []; window.open = (...a) => { window.__opened.push(a); return null; }; });
    await page.getByRole('link', { name: 'the docs' }).click();
    await page.waitForTimeout(300);
    let opened = await page.evaluate(() => window.__opened);
    check(JSON.stringify(opened) === JSON.stringify([['https://example.com/docs', '_blank', 'noopener,noreferrer']]), `${scheme}: a browser tab: a plain click on a web address opens it in a new tab`, JSON.stringify(opened));
    check(await page.locator('.artifact-panel').count() === 0, `${scheme}: a browser tab: and no side panel opens`);
    await page.getByRole('link', { name: 'the logo' }).click();
    await page.locator('.artifact-panel').waitFor({ timeout: 8000 });
    check((await page.evaluate(() => window.__opened)).length === 1, `${scheme}: a file link keeps the side panel and opens no tab`);
    await shot(page, `4721-${scheme}-file`);
    await page.context().close();
    // The app window: the host hands the address to the default browser; window.open is not used.
    const posts = [];
    page = await open(scheme, 1280, 860, sessions(), { before: async p => {
      await p.route('**/api/app/mode', r => r.fulfill({ json: { mode: 'window', reason: null, canQuit: true, appUpdate: 'unavailable' } }));
      await p.route('**/api/artifacts/open', r => { posts.push(r.request().postDataJSON()); return r.fulfill({ json: { ok: true } }); });
    } });
    await page.getByRole('link', { name: 'the docs' }).waitFor({ timeout: 15000 });
    await page.evaluate(() => { window.__opened = []; window.open = (...a) => { window.__opened.push(a); return null; }; });
    await page.getByRole('link', { name: 'the docs' }).click();
    await page.waitForTimeout(400);
    check(JSON.stringify(posts) === JSON.stringify([{ action: 'open', url: 'https://example.com/docs' }]), `${scheme}: the app window: the host is asked to open it in the default browser`, JSON.stringify(posts));
    check((await page.evaluate(() => window.__opened)).length === 0 && await page.locator('.artifact-panel').count() === 0, `${scheme}: the app window: nothing else happens`);
    check(page.problems.length === 0, `${scheme}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4802 */
cases.progress = async () => {
  const t = await tuneHost();
  await t.mode('ok');
  const sessions = [chat('s1', 'Progress', [user('u1', 'Hi'), reply('a1', 'Hello there.')])];
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const page = await open(scheme, w, h, sessions, { hash: '#/models', origin: t.origin });
    const started = new Date(Date.now() - (3 * 60 + 12) * 1000).toISOString();
    await page.route(/\/api\/native(\?.*)?$/, async route => {
      const r = await route.fetch(); const j = await r.json();
      j.jobs = [...(j.jobs ?? []), { id: 'job-x', kind: 'install', model: 'iq2_xs', state: 'running', startedAt: started, summary: 'Preparing', error: null, result: { install: { step: '6/7', label: 'Preparing the model data for the engine (one time)', parts: [{ label: 'Model files', done_gb: 0.6, total_gb: 0.6, state: 'done' }], download: { file: 'model files', done_gb: 0.6, total_gb: 0.6, remaining_gb: 0, percent: 99 } } }, progress: null }];
      return route.fulfill({ response: r, json: j });
    });
    await page.reload();
    const card = page.locator('.native-progress');
    await card.waitFor({ timeout: 20000 });
    await page.waitForTimeout(1500);
    check(await card.locator('.spinner').count() === 1, `${tag}: a spinner shows that it is working`);
    const alive = () => card.locator('.native-progress__alive').innerText();
    const a1 = await alive();
    check(/^Still working for 3m \d+s\. You don't need to do anything\.$/.test(a1), `${tag}: it says how long and that nothing is needed`, a1);
    await page.waitForTimeout(2200);
    const a2 = await alive();
    check(a1 !== a2, `${tag}: the time keeps counting`, `${a1} -> ${a2}`);
    check(/The buttons below wait until this finishes/.test(await page.locator('.native-wait-note').innerText()), `${tag}: a sentence says why the buttons are greyed out`);
    const border = await card.evaluate(el => getComputedStyle(el).borderTopColor);
    const other = await page.locator('.st-card:not(.native-progress)').first().evaluate(el => getComputedStyle(el).borderTopColor);
    check(border !== other, `${tag}: the card has its own border colour`, `${border} vs ${other}`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    await card.scrollIntoViewIfNeeded();
    await shot(page, `4802-${tag}`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4798 */
cases.serving = async () => {
  const sessions = [chat('s1', 'Serving', [user('u1', 'Hi'), reply('a1', 'Hello there.')])];
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    // Sharing off: one address, no extra line. Sharing on: the app's own address and the one for other devices, with one sentence.
    for (const sharing of [false, true]) {
      const page = await open(scheme, w, h, sessions, { hash: '#/serving', before: async p => {
        if (!sharing) return;
        await p.route(/\/api\/serving$/, async route => {
          if (route.request().method() !== 'GET') return route.continue();
          const r = await route.fetch(); const j = await r.json();
          return route.fulfill({ response: r, json: { ...j, localUrl: 'http://127.0.0.1:8079/v1', baseUrl: 'http://100.76.61.73:8080/v1', sharing: { active: true, url: 'http://100.76.61.73:8080/v1' } } });
        });
      } });
      await page.getByText('API base URL', { exact: false }).first().waitFor({ timeout: 15000 });
      await page.waitForTimeout(800);
      const text = await page.locator('.serving-overview').innerText();
      if (sharing) {
        check(/API base URL on this computer/.test(text) && text.includes('http://127.0.0.1:8079/v1'), `${tag}: sharing on: the app's own address is shown`);
        check(/Address for other devices/.test(text) && text.includes('http://100.76.61.73:8080/v1'), `${tag}: sharing on: and the address for other devices`);
        check(await page.locator('.serving-two-addresses').count() === 1 && /its own port/.test(await page.locator('.serving-two-addresses').innerText()), `${tag}: sharing on: one sentence says why they differ`);
        const t = page.getByRole('button', { name: 'Copy local network URL' });
        await t.click(); await page.waitForTimeout(250);
        check((await page.evaluate(() => navigator.clipboard.readText())) === 'http://100.76.61.73:8080/v1', `${tag}: sharing on: the copy button gives the network address`);
        check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: sharing on: no sideways scroll`);
        await page.locator('.serving-overview').scrollIntoViewIfNeeded();
        await shot(page, `4798-${tag}-sharing`);
      } else {
        check(!/Address for other devices/.test(text) && await page.locator('.serving-two-addresses').count() === 0, `${tag}: sharing off: one address, no extra line`);
      }
      check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
      await page.context().close();
    }
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4722 */
cases.width = async () => {
  const text = '用中文写一段九十字左右的段落，说明本地模型为什么适合处理私人文件，并且不需要把任何内容上传到网络上去，这样可以让人放心。\n\n- 第一条要点同样写得比较长，用来观察列表项在气泡里到底会排到多宽才换行，看看它是否与上面的段落一样。\n- 第二条要点也一样长，看看两条要点是不是排得整齐，不会比段落更宽也不会更窄。\n\nAnd an English paragraph that is long enough to wrap onto at least two lines so that the measure of the prose can be read off it by the test.\n\n1. A numbered point in English that is also long enough to wrap onto a second line in the same bubble as the paragraph above it.\n2. Another numbered point.\n\n| Name | Value |\n| --- | --- |\n| a long cell that is wide | another long cell that is wide too |\n\n```bash\nnpm install a-very-long-package-name another-very-long-package-name yet-another-long-package-name --save-dev --legacy-peer-deps\n```';
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const page = await open(scheme, w, h, [chat('s1', 'Width', [user('u1', 'Write'), reply('a1', text)])]);
    await page.locator('.st-bubble ul').first().waitFor({ timeout: 15000 });
    const m = await page.evaluate(() => {
      const b = document.querySelector('.st-msg--assistant .st-bubble'), r = el => Math.round(el.getBoundingClientRect().width);
      return { bubble: r(b), p: r(b.querySelector('p')), ul: r(b.querySelector('ul')), ol: r(b.querySelector('ol')), table: r(b.querySelector('.md-table-wrap')), code: r(b.querySelector('.st-code')) };
    });
    const note = JSON.stringify(m);
    if (w > 700) {
      check(Math.abs(m.ul - m.p) <= 2 && Math.abs(m.ol - m.p) <= 2, `${tag}: bullets and numbers have the paragraph's width`, note);
      check(m.table > m.p + 20 && m.code > m.p + 20, `${tag}: the table and the code block keep the full width`, note);
    } else {
      check(Math.abs(m.ul - m.p) <= 2 && m.p <= m.bubble, `${tag}: narrow window: everything fits the column`, note);
    }
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    await shot(page, `4722-${tag}`);
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #3723 */
cases.firstrun = async () => {
  const setupStep = dismissed => fetch(`${base}/api/setup/step`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ step: 1, dismissed, flowVersion: 2 }) });
  await setupStep(false);
  // A computer set to a language Strata does not have: English.
  const page = await open('light', 1280, 860, null, { hash: '#/setup', locale: 'sv-SE', keepSetup: true });
  await page.locator('.onboard-language').waitFor({ timeout: 20000 });
  check(/English/.test(await page.locator('.onboard-language').innerText()), 'Swedish computer: English');
  await page.context().close();
  // A first start (nothing saved) on a computer set to Chinese opens in Chinese, with a language choice on the first screen.
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const page = await open(scheme, w, h, null, { hash: '#/setup', locale: 'zh-CN', keepSetup: true });
    await page.locator('.onboard-language').waitFor({ timeout: 20000 });
    await page.waitForTimeout(500);
    const pick = page.locator('.onboard-language [role=combobox], .onboard-language button').first();
    const shown = await pick.innerText();
    check(/中文/.test(shown), `${tag}: a computer set to Chinese starts in Chinese`, shown);
    check(/[\u4e00-\u9fff]/.test(await page.locator('.guided-onboarding h1').first().innerText()), `${tag}: the first screen is in Chinese`);
    const box = await page.locator('.onboard-language').boundingBox();
    check(!!box && box.x >= 0 && box.x + box.width <= w && box.y >= 0, `${tag}: the choice is on screen`, JSON.stringify(box));
    const brand = await page.locator('.guided-onboarding .onboard-header .brand').first().boundingBox();
    check(!brand || box.x >= brand.x + brand.width - 1 || box.x + box.width <= brand.x + 1 || box.y >= brand.y + brand.height || box.y + box.height <= brand.y, `${tag}: the choice does not cover the logo`, JSON.stringify({ brand, box }));
    await shot(page, `3723-${tag}-first-zh`);
    // Choosing German changes the screen at once, and it is kept.
    await pick.click();
    await page.getByRole('option', { name: 'Deutsch' }).click();
    await page.waitForTimeout(500);
    check(/[A-Za-zäöü]/.test(await page.locator('.guided-onboarding h1').first().innerText()) && !/[\u4e00-\u9fff]/.test(await page.locator('.guided-onboarding h1').first().innerText()), `${tag}: choosing Deutsch changes the first screen`);
    await shot(page, `3723-${tag}-first-de`);
    await page.reload();
    await page.locator('.onboard-language').waitFor({ timeout: 20000 });
    check(/Deutsch/.test(await page.locator('.onboard-language').innerText()), `${tag}: the choice is kept after a restart, whatever the computer says`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
  await setupStep(true);
};

cases.kv = async () => {
  const t = await tuneHost();
  await t.mode('ok');
  const sessions = [chat('s1', 'KV', [user('u1', 'Hi'), reply('a1', 'Hello there.')])];
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const page = await open(scheme, w, h, sessions, { hash: '#/models', origin: t.origin });
    await page.route(/\/api\/native(\?.*)?$/, async route => {
      const r = await route.fetch(); const j = await r.json();
      j.jobs = [...(j.jobs ?? []), { id: 'job-f', kind: 'load', model: 'iq2_xs', state: 'error', startedAt: new Date().toISOString(), summary: 'Load failed', error: "server_exited (exit code 1): EngineDied: the engine exited before it was ready | the engine process ended: exit code 1: \nthe engine log's last lines:\n  strata generate: expert cache auto: 1.92 GiB free, 2048 MiB reserved -> 0 slots\n  strata generate: no VRAM is left for the expert cache: it needs at least 332 slots (737 MiB), about 1000 MiB more than this card has free.", result: null, progress: null }];
      return route.fulfill({ response: r, json: j });
    });
    await page.reload();
    const expert = page.getByRole('button', { name: 'Expert mode' });
    await expert.waitFor({ timeout: 20000 });
    await page.waitForTimeout(1500);
    await expert.click();
    const sel = page.getByRole('combobox', { name: 'Native KV format' });
    if (!await sel.isVisible().catch(() => false)) { await page.waitForTimeout(800); if (!await sel.isVisible().catch(() => false)) await page.getByRole('button', { name: /expert mode/i }).first().click(); }
    await sel.waitFor({ timeout: 10000 });
    const hint = page.locator('.native-kv-hint');
    check(/INT8 is the standard/.test(await hint.innerText()), `${tag}: Context memory: one line says what the chosen format is`);
    await sel.click();
    check(await page.getByRole('option').count() === 3, `${tag}: Context memory: three options`);
    const titles = await page.locator('.select__option-label').evaluateAll(els => els.map(e => e.getAttribute('title')));
    check(titles.length === 3 && titles.every(x => x && x.length > 20), `${tag}: Context memory: each option carries its line`, JSON.stringify(titles));
    await page.getByRole('option', { name: 'Q4' }).click();
    await page.waitForTimeout(300);
    check(/Q4 uses about half the memory/.test(await hint.innerText()), `${tag}: Context memory: the line follows the choice`);
    await hint.scrollIntoViewIfNeeded();
    await shot(page, `3723-${tag}-kv`);
    const notice = page.locator('.error-notice');
    await notice.first().waitFor({ timeout: 10000 });
    const said = await notice.first().innerText();
    check(/did not fit in your graphics card.s memory/.test(said) && /Context memory to Q4/.test(said), `${tag}: a start that ran out of graphics memory says so in plain words`, said.replace(/\n+/g, ' | ').slice(0, 200));
    await notice.first().scrollIntoViewIfNeeded();
    await shot(page, `3723-${tag}-vram`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4749 */
cases.helper = async () => {
  const agents = [
    { id: 'child-1', name: 'Check the tests', description: 'All 12 tests pass.', status: 'completed', durationMs: 42000, tokens: 1800, updatedAt: Date.now() },
    { id: 'child-2', name: 'Write the notes', description: 'Could not write notes.txt', status: 'failed', durationMs: 9000, tokens: 400, updatedAt: Date.now() },
  ];
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const sessions = [{ ...chat('s1', 'Helpers', [user('u1', 'Do two things'), reply('a1', 'Both helpers are on it.')]), agents }];
    const posts = []; let refuse = false;
    const page = await open(scheme, w, h, sessions, { before: async p => {
      await p.route('**/api/chat/helper', async r => { posts.push(r.request().postDataJSON()); if (refuse) return r.fulfill({ status: 409, json: { error: 'This helper cannot take a message any more. Ask the assistant in the chat to start a new one.' } }); return r.fulfill({ json: { ok: true, messageId: 'm1' } }); });
    } });
    await page.getByRole('button', { name: /Subagents/ }).first().click();
    const rows = page.locator('.subagent-row');
    await rows.first().waitFor({ timeout: 15000 });
    // The finished helper: queue.
    await rows.nth(0).getByRole('button', { name: 'Message this helper' }).click();
    const input = rows.nth(0).getByRole('textbox', { name: 'Message to Check the tests' });
    await input.fill('use ISO dates in the file you just wrote');
    await shot(page, `4749-${tag}-writing`);
    await input.press('Enter');
    await rows.nth(0).getByRole('status').waitFor({ timeout: 5000 });
    check(JSON.stringify(posts[0]) === JSON.stringify({ sessionId: 's1', helperId: 'child-1', message: 'use ISO dates in the file you just wrote', delivery: 'queue' }), `${tag}: a finished helper gets the message as a new turn (queue)`, JSON.stringify(posts[0]));
    check(/Sent to Check the tests\./.test(await rows.nth(0).innerText()) && await input.inputValue() === '', `${tag}: the row says it was sent and the box is empty again`);
    // A second helper (a failed one is listed too).
    await rows.nth(1).getByRole('button', { name: 'Message this helper' }).click();
    await rows.nth(1).getByRole('textbox').fill('also add the date');
    await rows.nth(1).getByRole('button', { name: 'Send' }).click();
    await rows.nth(1).getByRole('status').waitFor({ timeout: 5000 });
    check(posts[1]?.delivery === 'queue' && posts[1]?.helperId === 'child-2', `${tag}: the message goes to the helper that was chosen`, JSON.stringify(posts[1]));
    // A refusal says so in plain words.
    refuse = true;
    await rows.nth(0).getByRole('textbox').fill('one more thing');
    await rows.nth(0).getByRole('button', { name: 'Send' }).click();
    await rows.nth(0).getByRole('alert').waitFor({ timeout: 5000 });
    check(/cannot take a message any more/.test(await rows.nth(0).getByRole('alert').innerText()), `${tag}: a refused message says why in plain words`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    await shot(page, `4749-${tag}-sent`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4783 */
cases.busy = async () => {
  const fresh = (scheme, w, h, whileBusy) => open(scheme, w, h, [chat('s1', 'Busy', [])], { prefs: whileBusy ? { whileBusy } : {} });
  const send = async (page, text) => { await page.locator('textarea').first().fill(text); await page.keyboard.press('Enter'); };
  const lastReply = page => page.locator('.st-msg--assistant .st-bubble').last();
  for (const scheme of ['light', 'dark']) {
    const w = 1280, h = 860, tag = scheme;
    // Wait (the default): Enter while it answers sends nothing, the message stays in the box, and a note says why.
    let page = await fresh(scheme, w, h, null);
    await send(page, 'SCRIPT:slow 80');
    await page.locator('.st-msg--assistant').first().waitFor({ timeout: 15000 });
    await page.locator('textarea').first().fill('my second message');
    await page.keyboard.press('Enter');
    await page.waitForTimeout(600);
    check(await page.locator('textarea').first().inputValue() === 'my second message', `${tag}: Wait: the message stays in the box`);
    check(await page.locator('.parked').count() === 0, `${tag}: Wait: nothing is parked`);
    await page.getByText('The assistant is still answering').first().waitFor({ timeout: 4000 }).catch(() => {});
    check(await page.getByText('The assistant is still answering').count() > 0, `${tag}: Wait: a note says the assistant is still answering`);
    await page.context().close();
    // Send after: it waits (shown), and goes when the answer is done, after the whole first answer.
    page = await fresh(scheme, w, h, 'queue');
    await send(page, 'SCRIPT:slow 80');
    await page.locator('.st-msg--assistant').first().waitFor({ timeout: 15000 });
    await page.locator('textarea').first().fill('SCRIPT:echo the second answer');
    check(/Enter sends it when the answer is done/.test(await page.locator('.composer__busy-hint').innerText()), `${tag}: Send after: the hint says what Enter will do`);
    await page.keyboard.press('Enter');
    await page.locator('.parked').waitFor({ timeout: 3000 });
    check(/Waiting to send/.test(await page.locator('.parked').innerText()) && /second answer/.test(await page.locator('.parked').innerText()), `${tag}: Send after: the waiting message is shown`);
    if (scheme === 'light') await shot(page, `4783-${tag}-waiting`);
    await page.locator('.st-msg--assistant').nth(1).waitFor({ timeout: 30000 });
    await page.waitForFunction(() => document.querySelectorAll('.parked').length === 0, null, { timeout: 15000 });
    const firstText = await page.locator('.st-msg--assistant .st-bubble').first().innerText();
    check(/w79/.test(firstText), `${tag}: Send after: the first answer was written to its end`, firstText.slice(-30));
    check(/the second answer/.test(await lastReply(page).innerText()), `${tag}: Send after: then the second message was answered`);
    await page.context().close();
    // Interrupt: the answer stops where it is, what is written stays, and the message goes at once.
    page = await fresh(scheme, w, h, 'interject');
    await send(page, 'SCRIPT:slow 600');
    await page.locator('.st-msg--assistant').first().waitFor({ timeout: 15000 });
    await page.waitForTimeout(800);
    await page.locator('textarea').first().fill('SCRIPT:echo interrupted and answered');
    check(/Enter stops the answer here and sends it now/.test(await page.locator('.composer__busy-hint').innerText()), `${tag}: Interrupt: the hint says what Enter will do`);
    await page.keyboard.press('Enter');
    const t0 = Date.now();
    await page.locator('.st-msg--assistant').nth(1).waitFor({ timeout: 30000 });
    check(Date.now() - t0 < 12000, `${tag}: Interrupt: the new message was answered long before the first answer would have ended`, `${Date.now() - t0} ms`);
    const cut = await page.locator('.st-msg--assistant .st-bubble').first().innerText();
    check(cut.includes('w0') && !cut.includes('w599'), `${tag}: Interrupt: the first answer stopped where it was and stays`, `${cut.length} characters`);
    await page.waitForFunction(() => /interrupted and answered/.test([...document.querySelectorAll('.st-msg--assistant .st-bubble')].at(-1)?.innerText ?? ''), null, { timeout: 15000 }).catch(() => {});
    check(/interrupted and answered/.test(await lastReply(page).innerText()), `${tag}: Interrupt: the second message was answered`);
    if (scheme === 'light') await shot(page, `4783-${tag}-interrupted`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4785 */
cases.quote = async () => {
  const long = 'First point: local models keep your files on your own computer. Second point: nothing has to be uploaded to a service. Third point: you can work without a network connection.';
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const page = await open(scheme, w, h, [chat('s1', 'Quote', [user('u1', 'Why run a model locally?'), reply('a1', long), user('u2', 'Thanks, one more question'), reply('a2', 'Ask away.')])]);
    await page.locator('.st-msg--assistant').first().waitFor({ timeout: 15000 });
    // A whole turn, from the message menu.
    await page.locator('.st-msg--assistant').first().hover();
    await page.locator('.st-msg--assistant').first().getByRole('button', { name: 'More response actions' }).click();
    await page.getByRole('menuitem', { name: 'Quote' }).click();
    await page.locator('.quote').first().waitFor({ timeout: 5000 });
    check(/Quoting the assistant/.test(await page.locator('.quote').first().innerText()) && /First point/.test(await page.locator('.quote').first().innerText()), `${tag}: Quote in the message menu puts the turn above the box`);
    // Just a passage: select the second sentence and quote it.
    await page.evaluate(() => {
      const p = document.querySelector('.st-msg--assistant .st-bubble p');
      const node = p.firstChild; const at = node.textContent.indexOf('Second point'); const end = node.textContent.indexOf('Third point');
      const r = document.createRange(); r.setStart(node, at); r.setEnd(node, end);
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
    });
    await page.locator('.quote-selection').waitFor({ timeout: 5000 });
    const sbox = await page.locator('.quote-selection').boundingBox();
    check(!!sbox && sbox.x >= 0 && sbox.y >= 0 && sbox.x + sbox.width <= w, `${tag}: a Quote button appears by the selected words, on screen`, JSON.stringify(sbox));
    await page.locator('.quote-selection').click();
    check(await page.locator('.quote').count() === 2 && /^Second point: nothing has to be uploaded to a service\.$/.test((await page.locator('.quote .quote__text').nth(1).innerText()).trim()), `${tag}: the second quote is just the selected passage, listed after the first`);
    check(await page.locator('.quote-selection').count() === 0, `${tag}: the button goes away`);
    await shot(page, `4785-${tag}-quotes`);
    // Remove one, quote it again, jump to the message.
    await page.locator('.quote').first().getByRole('button', { name: 'Remove this quote' }).click();
    check(await page.locator('.quote').count() === 1 && /Second point/.test(await page.locator('.quote').first().innerText()), `${tag}: one quote can be removed without touching the others`);
    await page.locator('.quote__body').first().click();
    await page.waitForTimeout(400);
    // Send an answer: the model gets the fixed form, the transcript shows the quote apart from the reply.
    await page.locator('textarea').first().fill('SCRIPT:echo thanks for the quote');
    await page.keyboard.press('Enter');
    await page.locator('.sent-quote').first().waitFor({ timeout: 15000 });
    await page.waitForTimeout(300);
    const sent = await page.locator('.st-msg--user').last().innerText();
    check(/Second point: nothing has to be uploaded/.test(sent) && /SCRIPT:echo thanks for the quote/.test(sent) && !/Quoting:/.test(sent) && !/「/.test(sent), `${tag}: the sent message shows the quote apart from the reply, without the markup`, sent.replace(/\n+/g, ' | ').slice(0, 160));
    const got = worker.requests.at(-1)?.raw ?? '';
    check(got.includes('Quoting:\\n「Second point: nothing has to be uploaded to a service.」\\n→ Reply: SCRIPT:echo thanks for the quote'), `${tag}: the model receives the quote in its fixed form`, got.slice(-220));
    check(await page.locator('.quote').count() === 0, `${tag}: the quotes are cleared after sending`);
    await page.locator('.sent-quote').first().click();
    check(await page.locator('.sent-quotes[data-open]').count() === 1, `${tag}: a click opens the rest of a sent quote`);
    await shot(page, `4785-${tag}-sent`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
};

/* ------------------------------------------------------------------------------------------------------------------- #4244 */
cases.ask = async () => {
  const t = await tuneHost();
  await t.mode('ok');
  const asked = () => fetch(`${t.origin}/__fixture/asked`).then(r => r.json());
  await fetch(`${t.origin}/__fixture/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ on: true }) });
  for (const scheme of ['light', 'dark']) for (const [sizeName, [w, h]] of Object.entries(sizes)) {
    const tag = `${scheme}-${sizeName}`;
    const before = (await asked()).length;
    const page = await open(scheme, w, h, [chat('s1', 'Ask', [])], { origin: t.origin });
    await page.locator('textarea').first().fill('Build me a page');
    await page.keyboard.press('Enter');
    const card = page.locator('.askq');
    await card.waitFor({ timeout: 15000 });
    const text = await card.innerText();
    check(/Which colour should the page be\?/.test(text) && /Which extras do you want\?/.test(text) && /Warm and strong/.test(text) && /It sets the whole theme/.test(text), `${tag}: the question, its detail and the option descriptions are shown`);
    check(await page.locator('.askcard').count() === 1 && await page.locator('.approvaldock').count() === 1, `${tag}: it is one card in the dock above the box`);
    const send = page.getByRole('button', { name: 'Send answer' });
    check(await send.isDisabled(), `${tag}: Send answer waits until every question has an answer`);
    await page.getByRole('radio', { name: /Red/ }).click();
    await page.getByRole('radio', { name: /Blue/ }).click();
    check(await page.getByRole('radio', { name: /Red/ }).getAttribute('aria-checked') === 'false' && await page.getByRole('radio', { name: /Blue/ }).getAttribute('aria-checked') === 'true', `${tag}: a single-select question takes one choice`);
    await page.getByRole('checkbox', { name: 'Dark mode' }).click();
    await page.getByRole('checkbox', { name: 'Animations' }).click();
    check(await send.isEnabled(), `${tag}: Send answer works once every question has an answer`);
    await page.waitForTimeout(300);
    const sb = await send.boundingBox();
    const top = await page.evaluate(({ x, y }) => { const e = document.elementFromPoint(x, y); return e?.closest('button')?.textContent ?? ''; }, { x: sb.x + sb.width / 2, y: sb.y + sb.height / 2 });
    check(sb.y >= 0 && sb.y + sb.height <= h && /Send answer/.test(top), `${tag}: the answer buttons are in view and reachable without scrolling the card`, JSON.stringify(sb));
    await shot(page, `4244-${tag}`);
    check((await page.evaluate(() => document.documentElement.scrollWidth)) <= w, `${tag}: no sideways scroll`);
    const box = await page.locator('.approvaldock').boundingBox();
    check(!!box && box.y >= 0 && box.y + box.height <= h + 1, `${tag}: the dock with the card is inside the window`, JSON.stringify(box));
    await send.click();
    await page.waitForFunction(() => document.querySelectorAll('.askq').length === 0, null, { timeout: 8000 });
    const got = (await asked()).slice(before);
    check(JSON.stringify(got.at(-1)) === JSON.stringify({ sessionId: 's1', callId: 'call-ask-1', answers: [{ id: 'q1', selected: ['Blue'] }, { id: 'q2', selected: ['Dark mode', 'Animations'] }] }), `${tag}: the answer goes to the host as chosen`, JSON.stringify(got.at(-1)));
    check(await page.locator('.askq').count() === 0, `${tag}: the card goes away, and the turn ends`);
    // The person's own words, and a skip.
    await page.locator('textarea').first().fill('Again');
    await page.keyboard.press('Enter');
    await card.waitFor({ timeout: 15000 });
    await page.getByRole('textbox', { name: 'Type your own answer' }).first().fill('Green, like the logo');
    await page.getByRole('checkbox', { name: 'Animations' }).click();
    await send.click();
    await page.waitForFunction(() => document.querySelectorAll('.askq').length === 0, null, { timeout: 8000 });
    const own = (await asked()).slice(before).at(-1);
    check(JSON.stringify(own?.answers) === JSON.stringify([{ id: 'q1', selected: [], custom: 'Green, like the logo' }, { id: 'q2', selected: ['Animations'] }]), `${tag}: the person's own words are sent as such`, JSON.stringify(own));
    await page.locator('textarea').first().fill('Once more');
    await page.keyboard.press('Enter');
    await card.waitFor({ timeout: 15000 });
    await page.getByRole('button', { name: 'Skip' }).click();
    await page.waitForFunction(() => document.querySelectorAll('.askq').length === 0, null, { timeout: 8000 });
    check((await asked()).slice(before).at(-1)?.skip === true, `${tag}: Skip is sent as a skip`);
    check(page.problems.length === 0, `${tag}: no page error`, page.problems.join('; '));
    await page.context().close();
  }
  // A /loop runs without anyone there: the question is skipped at once, no card, and the loop is not left waiting.
  {
    const before = (await asked()).length;
    const page = await open('light', 1280, 860, [chat('s1', 'Loop', [])], { origin: t.origin });
    await page.locator('textarea').first().fill('/loop 10m Build me a page stop after 1 answer');
    await page.keyboard.press('Enter');
    await page.waitForFunction(async () => (await (await fetch('/__fixture/asked')).json()).length > 0, null, { timeout: 15000 }).catch(() => {});
    const got = (await asked()).slice(before).at(-1);
    check(got?.skip === true && got?.callId === 'call-ask-1', 'a loop: the question is skipped at once', JSON.stringify(got));
    check(await page.locator('.askq').count() === 0, 'a loop: no question card is shown');
    await page.context().close();
  }
  await fetch(`${t.origin}/__fixture/ask`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ on: false }) });
};

/* ------------------------------------------------------------------------------------------------------------------- #3818 (a probe, no fix) */
cases.select = async () => {
  const paras = Array.from({ length: 40 }, (_, i) => `Paragraph ${i + 1}. This is a sentence of a long answer that goes on for a while so that the reply overflows the window.`).join('\n\n');
  const pick = async (page, fromText, toText) => {
    const a = await page.getByText(fromText, { exact: false }).first().boundingBox(), b = await page.getByText(toText, { exact: false }).first().boundingBox();
    await page.mouse.move(a.x + 2, a.y + a.height / 2);
    await page.mouse.down();
    await page.mouse.move(b.x + b.width - 4, b.y + b.height / 2, { steps: 12 });
    await page.mouse.up();
    return page.evaluate(() => window.getSelection().toString());
  };
  const page = await open('light', 1280, 860, [chat('s1', 'Select', [user('u1', 'Write a lot'), reply('a1', paras)])]);
  await page.locator('.st-bubble p').first().waitFor({ timeout: 15000 });
  await page.locator('.main__body.chat').evaluate(el => { el.scrollTop = el.scrollHeight; });
  await page.waitForTimeout(400);
  // After the reply has finished: a drag across several paragraphs.
  const visible = await page.evaluate(() => [...document.querySelectorAll('.st-bubble p')].filter(p => { const r = p.getBoundingClientRect(); return r.top > 100 && r.bottom < innerHeight - 220; }).map(p => p.textContent.slice(0, 12)));
  const first = visible[0], last = visible.at(-1);
  const got = await pick(page, first, last);
  check(got.includes(first) && got.includes(last) && got.split('\n').filter(Boolean).length >= visible.length, `finished reply: a drag across ${visible.length} paragraphs selects all of them`, `${got.length} characters`);
  await page.context().close();
  // While a reply streams: select a passage of the older reply, and see whether the selection and the view stay.
  const page2 = await open('light', 1280, 860, [chat('s1', 'Select', [user('u1', 'Write a lot'), reply('a1', paras)])]);
  await page2.locator('.st-bubble p').first().waitFor({ timeout: 15000 });
  await page2.locator('textarea').first().fill('SCRIPT:slow 400');
  await page2.keyboard.press('Enter');
  await page2.waitForTimeout(1500);
  const v2 = await page2.evaluate(() => [...document.querySelector('.st-msg--assistant').querySelectorAll('.st-bubble p')].filter(p => { const r = p.getBoundingClientRect(); return r.top > 100 && r.bottom < innerHeight - 220; }).map(p => p.textContent.slice(0, 12)));
  if (v2.length >= 2) {
    // The selection is made in code here (a drag moves with the layout and made this probe flaky): words of the older reply, as a drag would select them.
    const sel0 = await page2.evaluate(([a, b]) => {
      const ps = [...document.querySelector('.st-msg--assistant').querySelectorAll('.st-bubble p')], from = ps.find(p => p.textContent.startsWith(a)), to = ps.find(p => p.textContent.startsWith(b));
      const r = document.createRange(); r.setStart(from.firstChild, 0); r.setEnd(to.lastChild, to.lastChild.textContent.length);
      const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); return s.toString();
    }, [v2[0], v2.at(-1)]);
    const top0 = await page2.locator('.main__body.chat').evaluate(el => el.scrollTop);
    await page2.waitForTimeout(3000);
    const sel1 = await page2.evaluate(() => window.getSelection().toString());
    const top1 = await page2.locator('.main__body.chat').evaluate(el => el.scrollTop);
    console.log(`INFO streaming: selection ${sel0.length} -> ${sel1.length} characters, scrollTop ${Math.round(top0)} -> ${Math.round(top1)}`);
    console.log(`INFO before: ${JSON.stringify(sel0.slice(0, 40))} ... ${JSON.stringify(sel0.slice(-40))}
INFO after:  ${JSON.stringify(sel1.slice(0, 40))} ... ${JSON.stringify(sel1.slice(-40))}`);
    check(sel1 === sel0, 'streaming: a selection in the older reply stays as it was while the new reply is written (a selection inside the paragraph that is being written does not: React replaces its text node on every token)', `${sel0.length} -> ${sel1.length}`);
    check(Math.abs(top1 - top0) < 40, 'streaming: the view is not pulled away from the selection', `${Math.round(top0)} -> ${Math.round(top1)}`);
  } else console.log('INFO streaming: the older reply was not on screen, nothing to select');
  await page2.context().close();
};

/* ------------------------------------------------------------------------------------------------------------------ run */
let ran = 0;
for (const [name, fn] of Object.entries(cases)) {
  if (ONLY.length && !ONLY.includes(name)) continue;
  console.log(`--- ${name}`);
  ran += 1;
  try { await fn(); } catch (e) {
    check(false, `${name} threw`, String(e?.stack ?? e).split('\n').slice(0, 4).join(' | '));
    const last = pages.at(-1);
    if (last && !last.isClosed()) {
      await last.screenshot({ path: join(SHOTS, `debug-${name}.png`) }).catch(() => {});
      console.log('PAGE TEXT:', ((await last.evaluate(() => document.body.innerText).catch(() => '')) || '').slice(0, 600).replace(/\n+/g, ' | '));
      console.log('PAGE ERRORS:', last.problems?.join('; '));
    }
  }
}
check(ran > 0, 'at least one case ran');
await browser.close();
await worker.close();
for (const proc of [child, tuneHostInfo?.proc]) if (proc) { if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { windowsHide: true }); else proc.kill('SIGTERM'); }
try { rmSync(home, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ }
process.exit(failed ? 1 : 0);
