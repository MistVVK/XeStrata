// TOOLCALL-B: a reply whose tool call could not run (strata_rejected_tool_calls / strata_unread_tool_markup from the Strata server) does not
// end the agent's turn: the model gets a tool error and goes on, twice in a row at most, then the turn ends with one plain sentence.
// Pure parts always run; the end-to-end ones need a built Harness (STRATA_HARNESS_DIR) and use the scripted model server.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLlmProxy, MarkerTap, markersOf, requestFacts } from '../server/agent/llm-shape.mjs';
import { MarkerStore, MAX_TOOL_FIXES, MARKERS_KEPT, FIXING_STATUS, END_NOTE, correctionText, carries, endNote } from '../server/agent/tool-fix.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge, agentProvider } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

// ------------------------------------------------------------------------------------------------ the texts

test('the model reads what went wrong, which tools it has, and a close name only as a hint', () => {
  const declared = ['pwsh', 'read', 'write'];
  assert.equal(correctionText({ rejected: [{ name: 'exec', reason: 'undeclared' }], declared }),
    'Tool error: no such tool: exec. Nothing was run. The tools you can call are: pwsh, read, write.');
  assert.equal(correctionText({ rejected: [{ name: 'psh', reason: 'undeclared', closest_declared: 'pwsh' }], declared }),
    'Tool error: no such tool: psh. Nothing was run. The tools you can call are: pwsh, read, write. Did you mean pwsh?');
  assert.equal(correctionText({ rejected: [{ name: 'exec' }, { name: 'gread', closest_declared: 'read' }, { name: 'exec' }], declared }),
    'Tool error: no such tool: exec. Nothing was run. The tools you can call are: pwsh, read, write.\nTool error: no such tool: gread. Nothing was run. Did you mean read?', 'each name once, the list once');
  const unread = 'Tool error: your tool call could not be read (it was cut off or not complete), so nothing was run. Write the whole call again, or answer in plain text.';
  assert.equal(correctionText({ unread: { reason: 'cut_off', chars: 40 }, declared }), unread);
  assert.equal(correctionText({ rejected: [{ name: 'exec' }], unread: { reason: 'unreadable', chars: 9 }, declared }).split('\n')[1], unread);
  assert.equal(MAX_TOOL_FIXES, 2);
  const note = endNote();
  assert.equal(note.type, 'delta'); assert.equal(note.own, true); assert.equal(note.key, END_NOTE); assert.equal(note.text, `\n\n${END_NOTE}`);
});

test('the status line and the closing sentence are in the catalog in all nine languages', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
  const cat = Object.fromEntries(langs.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of [FIXING_STATUS, END_NOTE]) {
    const id = Object.entries(cat.en).find(([, v]) => v === text)?.[0];
    assert.ok(id, `"${text}" has a key`);
    for (const l of langs.slice(1)) assert.ok(cat[l][id]?.trim() && cat[l][id] !== text, `${l}: ${text}`);
  }
  assert.match(cat.de[Object.entries(cat.en).find(([, v]) => v === END_NOTE)[0]], /\bSie\b/, 'German uses "Sie"');
});

// ------------------------------------------------------------------------------------------------ which turn a marker belongs to

test('a marker belongs to the turn whose text its request carries, recorded after the turn\'s newest own event, and is used once', () => {
  let t = 1000;
  const s = new MarkerStore({ now: () => t });
  const users = ['The approval policy changed.', 'fix the build', 'Time sampled while preparing turn 1, step 1: ...'];
  s.add({ key: 'k', at: 1100, users, rejected: [{ name: 'exec' }], declared: ['read'] });
  assert.equal(s.take({ texts: new Set(['something else']), since: 0 }), null, 'another chat\'s text: not this turn\'s');
  assert.equal(s.take({ texts: new Set(['fix the build']), since: 1200 }), null, 'recorded before the turn\'s newest event (a call ran after it)');
  assert.equal(s.take({ texts: new Set(['fix the build']), since: 1000 }).marker.rejected[0].name, 'exec');
  assert.equal(s.take({ texts: new Set(['fix the build']), since: 1000 }), null, 'used once');
  s.add({ key: 'k', at: 1300, users, unread: { reason: 'cut_off', chars: 3 } });
  const both = s.take({ texts: new Set(['fix the build']), since: 1000 }, [{ texts: new Set(['fix the build']), since: 1000 }]);
  assert.ok(both.ambiguous && !both.marker, 'the same text in two running chats: acted on by none');
  assert.equal(s.size, 0, 'and it is gone');
  s.add({ key: 'k', at: 1400, users: ['other'] });
  assert.equal(s.size, 0, 'a reply with neither field is not kept');
  for (let i = 0; i < MARKERS_KEPT + 10; i++) s.add({ key: 'k', at: 2000 + i, users: [`m${i}`], unread: { reason: 'cut_off', chars: 1 } });
  assert.equal(s.size, MARKERS_KEPT, 'at most 64 kept');
  assert.equal(s.take({ texts: new Set(['m0']), since: 0 }), null, 'the oldest went first');
  t = 2000 + 31 * 60_000;
  assert.equal(s.take({ texts: new Set(['m70']), since: 0 }), null, 'an old marker expires');
  assert.equal(carries(['  hello\r\nworld '], new Set(['hello\nworld'])), true);
  assert.equal(carries(['x'], new Set([''])), false, 'an empty text matches nothing');
});

// ------------------------------------------------------------------------------------------------ reading the reply

test('the markers are read from every API shape, whatever the chunk boundaries, and an ordinary reply gives nothing', () => {
  const rej = [{ name: 'exec', reason: 'undeclared', closest_declared: 'pwsh' }];
  const chat = `data: {"choices":[{"delta":{"content":"héllo ✓"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"strata_rejected_tool_calls":${JSON.stringify(rej)}}\n\ndata: [DONE]\n\n`;
  const bytes = Buffer.from(chat);
  for (const size of [1, 2, 3, 7, 64, bytes.length]) {
    const got = [];
    const tap = new MarkerTap({ found: m => got.push(m) });
    for (let i = 0; i < bytes.length; i += size) tap.push(bytes.subarray(i, i + size));
    tap.end();
    assert.deepEqual(got, [{ rejected: rej, unread: null }], `pieces of ${size}`);
  }
  const unread = { reason: 'cut_off', chars: 12 };
  assert.deepEqual(markersOf({ type: 'message_delta', delta: { stop_reason: 'end_turn' }, strata_unread_tool_markup: unread }), { rejected: null, unread }, 'Anthropic message_delta');
  assert.deepEqual(markersOf({ type: 'response.completed', response: { status: 'completed', strata_rejected_tool_calls: rej } }), { rejected: rej, unread: null }, '/v1/responses');
  const body = []; const t2 = new MarkerTap({ sse: false, found: m => body.push(m) });
  t2.push(Buffer.from(JSON.stringify({ choices: [{ message: { content: '<tool_call>' } }], strata_unread_tool_markup: unread }))); t2.end();
  assert.deepEqual(body, [{ rejected: null, unread }], 'a non-stream body');
  const none = []; const t3 = new MarkerTap({ found: m => none.push(m) });
  t3.push(Buffer.from('data: {"choices":[{"delta":{"content":"strata_rejected_tool_calls is a word here"}}]}\n\n')); t3.end();
  assert.deepEqual(none, [], 'the name inside the answer text is not a marker');
  assert.deepEqual(requestFacts(Buffer.from(JSON.stringify({ messages: [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: [{ type: 'text', text: 'c' }, { type: 'image_url' }] }], tools: [{ type: 'function', function: { name: 'read' } }] }))),
    { users: ['a', 'c'], declared: ['read'] });
});

// ------------------------------------------------------------------------------------------------ the forwarder

/** A model server stand-in that answers with fixed bytes in fixed pieces; `seen` keeps every request's raw body. */
async function upstreamOf(pieces, { gapMs = 5, type = 'text/event-stream' } = {}) {
  const seen = []; const closed = [];
  const server = http.createServer((req, res) => {
    const chunks = []; req.on('data', c => chunks.push(c));
    req.on('end', async () => {
      seen.push(Buffer.concat(chunks));
      res.on('close', () => closed.push({ ended: res.writableEnded, at: Date.now() }));
      res.writeHead(200, { 'content-type': type });
      for (const p of pieces) { if (res.destroyed) return; res.write(p); await new Promise(r => setTimeout(r, gapMs)); }
      res.end();
    });
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  return { server, seen, closed, url: `http://127.0.0.1:${server.address().port}/v1` };
}
async function frontOf(t, targetUrl, onMarker = null) {
  const proxy = createLlmProxy({ targetOf: key => (key === 'k' ? targetUrl : null), samplingNow: () => null, onMarker });
  const front = http.createServer((req, res) => { const m = /^\/api\/internal\/agent-llm\/([^/]+)(\/.*)$/.exec(req.url); proxy(req, res, m[1], m[2]); });
  await new Promise(r => front.listen(0, '127.0.0.1', r));
  t.after(() => front.close());
  return front.address().port;
}
const post = (port, path, body, onChunk = null) => new Promise((resolve, reject) => {
  const r = http.request({ host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json' } }, res => {
    const parts = []; res.on('data', c => { parts.push(c); onChunk?.(c, r); }); res.on('end', () => resolve(Buffer.concat(parts))); res.on('error', reject);
  });
  r.on('error', reject); r.end(body);
});

test('the forwarded reply and request are byte for byte what the model server and the assistant wrote, and a marker is heard', async (t) => {
  const reply = ['data: {"choices":[{"delta":{"content":"caf', Buffer.from('é ✓"}}]}\n\n'), 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],', '"strata_unread_tool_markup":{"reason":"cut_off","chars":11}}\r\n\r\n', 'data: [DONE]\n\n'];
  // a multi-byte character split between two pieces
  const raw = Buffer.concat(reply.map(p => Buffer.isBuffer(p) ? p : Buffer.from(p)));
  const pieces = [raw.subarray(0, 40), raw.subarray(40, 41), raw.subarray(41, 120), raw.subarray(120)];
  const up = await upstreamOf(pieces);
  t.after(() => up.server.close());
  const heard = [];
  const port = await frontOf(t, up.url, m => heard.push(m));
  const body = '{"model":"m",  "stream":true, "temperature":1.50, "messages":[{"role":"user","content":"fix it"},{"role":"user","content":"Time sampled ..."}],"tools":[{"type":"function","function":{"name":"read"}}]}';
  const direct = await post(Number(new URL(up.url).port), '/v1/chat/completions', body);
  const forwarded = await post(port, '/api/internal/agent-llm/k/v1/chat/completions', body);
  assert.ok(direct.equals(raw), 'the stand-in answers the same bytes');
  assert.ok(forwarded.equals(direct), 'the forwarded stream is byte-identical');
  assert.ok(up.seen.at(-1).equals(Buffer.from(body)), 'the request body went on untouched (no re-serialising at the defaults)');
  assert.equal(heard.length, 1);
  assert.deepEqual(heard[0].users, ['fix it', 'Time sampled ...']); assert.deepEqual(heard[0].declared, ['read']);
  assert.deepEqual(heard[0].unread, { reason: 'cut_off', chars: 11 }); assert.equal(heard[0].key, 'k');
});

test('stopping a reply mid-stream frees the model server\'s request at once', async (t) => {
  const pieces = Array.from({ length: 200 }, (_, i) => `data: {"choices":[{"delta":{"content":"w${i} "}}]}\n\n`);
  const up = await upstreamOf(pieces, { gapMs: 20 });
  t.after(() => up.server.close());
  const port = await frontOf(t, up.url, () => {});
  const stoppedAt = await new Promise(resolve => {
    let n = 0;
    post(port, '/api/internal/agent-llm/k/v1/chat/completions', '{"stream":true,"messages":[]}', (c, r) => { if (++n === 3) { r.destroy(); resolve(Date.now()); } }).catch(() => {});
  });
  for (let i = 0; i < 100 && !up.closed.length; i++) await new Promise(r => setTimeout(r, 20));
  assert.equal(up.closed.length, 1, 'the model server saw the request go');
  assert.equal(up.closed[0].ended, false, 'before it had finished the reply');
  assert.ok(up.closed[0].at - stoppedAt < 1000, `within a second (${up.closed[0].at - stoppedAt} ms)`);
});

test('a local model and a server on this network go through the forwarder always; a cloud route never; the rollback restores the old choice', () => {
  const profile = buildProfile({
    local: [{ id: 'a', modelId: 'm', baseUrl: 'http://127.0.0.1:9001', context: 65536 }],
    connections: [
      { id: 'lan', name: 'LAN', baseUrl: 'http://192.168.1.9:8080/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'x' }] },
      { id: 'cloud', name: 'Cloud', baseUrl: 'https://api.deepseek.com', protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'y' }] },
    ],
    proxyBase: 'http://127.0.0.1:8791/api/internal/agent-llm',
  });
  const [local, lan, cloud] = ['strata-local', 'api-lan', 'api-cloud'].map(r => profile.routes.find(x => x.route === r));
  assert.equal(agentProvider(local), 'strata-local~s0', 'defaults: still forwarded');
  assert.equal(agentProvider(local, { budget: 8192 }), 'strata-local~s8192');
  assert.equal(agentProvider(lan), lan.shaped[0]);
  assert.equal(agentProvider(cloud), 'api-cloud', 'a cloud route goes straight to the service');
  assert.equal(agentProvider(cloud, { budget: 8192, sampling: { temperature: 0.1 } }), 'api-cloud');
  assert.equal(agentProvider(local, { forwardAlways: false }), 'strata-local', 'STRATA_AGENT_FORWARD=0: defaults go straight to the model, as before');
  assert.equal(agentProvider(local, { forwardAlways: false, sampling: { temperature: 0.1 } }), 'strata-local~s0');
});

// ------------------------------------------------------------------------------------------------ with the real Harness

async function rig(t, { env = process.env } = {}) {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-toolfix-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  writeFileSync(join(ws, 'notes.txt'), 'the notes file\n');
  const logs = []; let bridge; let forwarded = 0;
  const proxy = createLlmProxy({ targetOf: key => bridge.llmTarget(key), samplingNow: () => null, onMarker: m => bridge.noteToolMarker(m) });
  const front = http.createServer((req, res) => { forwarded++; const m = /^\/api\/internal\/agent-llm\/([^/]+)(\/.*)$/.exec(req.url); proxy(req, res, m[1], m[2]); });
  await new Promise(r => front.listen(0, '127.0.0.1', r));
  bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 4 }], connections: () => [],
    env, startTimeoutMs: 90_000, proxyBase: `http://127.0.0.1:${front.address().port}/api/internal/agent-llm`, log: l => logs.push(l) });
  t.after(async () => { await bridge.close(); await worker.close(); front.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const run = async (chatId, message) => { const out = []; await bridge.runTurn({ chatId, message, workspacePath: ws, mode: 'bypass', onFrame: f => out.push(f) }); return out; };
  return { worker, bridge, ws, logs, run, file: join(ws, 'notes.txt'), forwarded: () => forwarded };
}
const ofScript = (w, pattern) => w.requests.filter(q => q.directive === 'badseq' && q.toolNames.length && q.raw.includes(pattern));
const cards = (out, name) => out.filter(f => f.type === 'agent-tool' && f.tool.name === name && f.tool.state === 'done');
const keys = out => out.filter(f => f.type === 'delta' && f.key).map(f => f.key);
const statuses = out => out.filter(f => f.type === 'status').map(f => f.text);

for (const [label, step, expect] of [
  ['an undeclared name', 'rej:exec', /Tool error: no such tool: exec\. Nothing was run\. The tools you can call are: [^\n]*\bread\b[^\n]*\.(?! Did you mean)/],
  ['an undeclared name with a close declared one', 'rej:psh>read', /Tool error: no such tool: psh\. Nothing was run\. The tools you can call are: [^\n]*\. Did you mean read\?/],
  ['a call cut off inside its markup', 'cut', /Tool error: your tool call could not be read \(it was cut off or not complete\), so nothing was run\./],
  ['a closed block that reads as no call', 'junk', /Tool error: your tool call could not be read/],
]) {
  test(`${label}: the turn goes on, the next request carries the tool error, the corrected call runs`, { skip }, async (t) => {
    const r = await rig(t);
    const out = await r.run('c1', `SCRIPT:badseq ${step},read ${r.file}`);
    const reqs = ofScript(r.worker, step);
    assert.equal(reqs.length, 3, `bad reply, corrected call, answer (${reqs.length} requests)`);
    assert.doesNotMatch(reqs[0].raw, /Tool error/);
    assert.match(JSON.parse(reqs[1].raw).messages.filter(m => m.role === 'user').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n'), expect);
    assert.equal(cards(out, 'read').length, 1, 'the corrected call ran');
    assert.ok(statuses(out).includes(FIXING_STATUS), 'the status line said so');
    assert.equal(out.filter(f => f.type === 'error').length, 0);
    const done = out.find(f => f.type === 'done'); assert.ok(done && !done.cancelled);
    assert.ok(!keys(out).includes(END_NOTE));
    assert.match(out.filter(f => f.type === 'delta').map(f => f.text).join(''), /All done after the tool step\./);
    assert.ok(r.forwarded() >= 3, 'the requests went through the forwarder');
    assert.ok(r.logs.some(l => /correction 1 of 2/.test(l)));
  });
}

test('after two corrections in a row the third bad reply ends the turn with the plain sentence; nothing was run', { skip }, async (t) => {
  const r = await rig(t);
  const out = await r.run('cap', `SCRIPT:badseq rej:exec,rej:exec,cut,read ${r.file}`);
  const reqs = ofScript(r.worker, 'rej:exec,rej:exec,cut');
  assert.equal(reqs.length, 3, 'asked twice, then stopped');
  assert.equal(cards(out, 'read').length, 0, 'nothing ran');
  assert.equal(keys(out).at(-1), END_NOTE);
  assert.ok(out.findIndex(f => f.key === END_NOTE) < out.findIndex(f => f.type === 'done'));
  assert.equal(out.filter(f => f.type === 'error').length, 0);
  assert.ok(r.logs.some(l => /after 2 corrections in a row; the turn ends/.test(l)));
});

test('a call that runs starts the count again', { skip }, async (t) => {
  const r = await rig(t);
  const out = await r.run('reset', `SCRIPT:badseq rej:a,rej:b,read,rej:c,rej:d,read ${r.file}`);
  assert.equal(ofScript(r.worker, 'rej:a,rej:b').length, 7);
  assert.equal(cards(out, 'read').length, 2);
  assert.ok(!keys(out).includes(END_NOTE));
  assert.equal(r.logs.filter(l => /correction \d of 2/.test(l)).length, 4);
});

test('parallel chats: a marker in one chat does not continue the other', { skip }, async (t) => {
  const r = await rig(t);
  const [a, b] = await Promise.all([r.run('pa', `SCRIPT:badseq rej:exec,say ${r.file}`), r.run('pb', 'SCRIPT:slow 40')]);
  assert.equal(ofScript(r.worker, 'rej:exec,say').length, 2, 'chat A was corrected once');
  const slow = r.worker.requests.filter(q => q.directive === 'slow' && q.toolNames.length);
  assert.equal(slow.length, 1, 'chat B asked the model once');
  assert.ok(slow.every(q => !/Tool error/.test(q.raw)));
  assert.ok(!statuses(b).includes(FIXING_STATUS));
  assert.ok(statuses(a).includes(FIXING_STATUS));
  assert.ok(b.find(f => f.type === 'done') && a.find(f => f.type === 'done'));
});

test('no markers: one request through the forwarder and the turn ends as before', { skip }, async (t) => {
  const r = await rig(t);
  const out = await r.run('plain', 'SCRIPT:echo hello there');
  assert.equal(r.worker.requests.filter(q => q.directive === 'echo' && q.toolNames.length).length, 1);
  assert.ok(r.forwarded() >= 1);
  assert.equal(out.filter(f => f.type === 'delta').map(f => f.text).join(''), 'hello there');
  assert.ok(!statuses(out).includes(FIXING_STATUS));
  assert.ok(out.find(f => f.type === 'done' && !f.cancelled));
});

test('Stop during a forwarded reply frees the model server\'s request', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  const running = r.bridge.runTurn({ chatId: 'stop', message: 'SCRIPT:slow 4000', workspacePath: r.ws, mode: 'bypass', onFrame: f => {
    out.push(f);
    if (f.type === 'delta' && out.filter(x => x.type === 'delta').length === 3) void r.bridge.stop('stop');
  } });
  await running;
  for (let i = 0; i < 100 && r.worker.aborted < 1; i++) await new Promise(res => setTimeout(res, 20));
  assert.ok(r.worker.aborted >= 1, 'the worker saw its request closed before the reply ended');
  assert.ok(out.find(f => f.type === 'done')?.cancelled);
});

test('the rollback (STRATA_AGENT_FORWARD=0) goes straight to the model at the defaults, and nothing is corrected', { skip }, async (t) => {
  const r = await rig(t, { env: { ...process.env, STRATA_AGENT_FORWARD: '0' } });
  const out = await r.run('rb', `SCRIPT:badseq rej:exec,read ${r.file}`);
  assert.equal(r.forwarded(), 0);
  assert.equal(ofScript(r.worker, 'rej:exec,read').length, 1, 'the turn ended as before');
  assert.ok(!statuses(out).includes(FIXING_STATUS));
});
