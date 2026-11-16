// #5661: an API client could not use a model's MCP tools at all (the gateway forces strata_mcp off), and PR #5721 would have let any client
// on any listener turn them on. They now reach the engine only when every guard holds: the app's own listener (not LAN sharing, not the
// tunnel), a loopback address, no Origin of another site, an API key set and sent, and the person's opt-in for that model in Serving
// (default off). Each guard alone refuses; all of them together allow it; without an opt-in nothing changes.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const home = mkdtempSync(join(tmpdir(), 'strata-mcp-guard-'));
process.env.STRATA_HOME = home;
const serving = await import('../server/serving.mjs');
const { nativeGateway } = await import('../server/native-gateway.mjs');
test.after(() => { try { rmSync(home, { recursive: true, force: true }); } catch { /* temp folder */ } });

const KEY = 'sk-strata-test-key';
const MODEL = 'qwen3.8-flash';
const req = ({ address = '127.0.0.1', origin, key = KEY } = {}) => ({ socket: { remoteAddress: address }, headers: { ...(origin === undefined ? {} : { origin }), ...(key ? { authorization: `Bearer ${key}` } : {}) } });
const setUp = ({ key = KEY, on = true } = {}) => {
  serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });
  serving.patch(key ? { apiKey: key } : { apiKey: null });
  serving.patch({ mcpTools: { model: MODEL, on } });
};

test('default: no model is opted in, so a request with every other guard holding is refused', () => {
  serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });
  serving.patch({ apiKey: KEY });
  assert.deepEqual(serving.state().mcpTools, []);
  assert.equal(serving.mcpToolsRefusal(req(), 'app', MODEL), 'off');
});

test('all guards together allow it; each guard alone refuses', () => {
  setUp();
  assert.deepEqual(serving.state().mcpTools, [MODEL]);
  assert.equal(serving.mcpToolsRefusal(req(), 'app', MODEL), null);
  assert.equal(serving.mcpToolsRefusal(req({ address: '::1' }), 'app', MODEL), null);
  assert.equal(serving.mcpToolsRefusal(req({ address: '::ffff:127.0.0.1', origin: 'http://localhost:5173' }), 'app', MODEL), null);
  // the listener: LAN sharing and the tunnel never, also from a loopback address (cloudflared connects from 127.0.0.1)
  assert.equal(serving.mcpToolsRefusal(req(), 'share', MODEL), 'listener');
  assert.equal(serving.mcpToolsRefusal(req(), 'tunnel', MODEL), 'listener');
  assert.equal(serving.mcpToolsRefusal(req(), undefined, MODEL), 'listener');
  // another device reaching the app's own port (a host started with a network listen address)
  assert.equal(serving.mcpToolsRefusal(req({ address: '192.168.1.5' }), 'app', MODEL), 'not-loopback');
  assert.equal(serving.mcpToolsRefusal(req({ address: '::ffff:192.168.1.5' }), 'app', MODEL), 'not-loopback');
  // a web page of another site, a sandboxed page, a broken header
  for (const origin of ['https://evil.example', 'http://127.0.0.1.evil.example', 'null', '']) assert.equal(serving.mcpToolsRefusal(req({ origin }), 'app', MODEL), 'foreign-origin', origin);
  // the key: not sent, wrong
  assert.equal(serving.mcpToolsRefusal(req({ key: null }), 'app', MODEL), 'no-key');
  assert.equal(serving.mcpToolsRefusal(req({ key: 'sk-strata-wrong' }), 'app', MODEL), 'no-key');
  // another model, or the opt-in turned off again
  assert.equal(serving.mcpToolsRefusal(req(), 'app', 'other-model'), 'off');
  serving.patch({ mcpTools: { model: MODEL, on: false } });
  assert.equal(serving.mcpToolsRefusal(req(), 'app', MODEL), 'off');
});

test('no API key set: refused even with the opt-in on', () => {
  setUp({ key: null });
  assert.equal(serving.mcpToolsRefusal(req({ key: null }), 'app', MODEL), 'no-key');
  assert.equal(serving.mcpToolsRefusal(req(), 'app', MODEL), 'no-key');
});

test('the opt-in list takes only well-formed choices and stays bounded', () => {
  setUp({ on: false });
  for (const bad of [{ model: MODEL, on: 'yes' }, { model: '', on: true }, { model: 'x'.repeat(300), on: true }, { on: true }, 'x']) serving.patch({ mcpTools: bad });
  assert.deepEqual(serving.state().mcpTools, []);
  for (let i = 0; i < 80; i++) serving.patch({ mcpTools: { model: `m${i}`, on: true } });
  assert.equal(serving.state().mcpTools.length, serving.MCP_TOOL_MODELS_MAX);
  serving.patch({ mcpTools: { model: 'm79', on: true } });
  assert.equal(serving.state().mcpTools.filter(m => m === 'm79').length, 1);
});

const listen = server => new Promise(r => server.listen(0, '127.0.0.1', () => r(server.address().port)));
const close = server => new Promise(r => server.close(r));

test('the gateway: strata_mcp reaches the engine only when the client asks and the guards allow it', { timeout: 15000 }, async () => {
  const received = [];
  const engine = http.createServer(async (rq, rs) => { let t = ''; for await (const c of rq) t += c; received.push(JSON.parse(t)); rs.writeHead(200, { 'content-type': 'application/json' }); rs.end('{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'); });
  const enginePort = await listen(engine);
  const control = { resolveInstance: () => ({ id: 'x', apiModelId: MODEL, port: enginePort, key: 'k', gpus: [0] }) };
  const record = { open: v => ({ ...v, id: '1' }), mark() {}, finish() {} };
  // a real listener, so the loopback address and the Origin are what a client really sends; `via` as the host passes it
  const gateway = http.createServer(async (rq, rs) => {
    let t = ''; for await (const c of rq) t += c;
    const via = rq.headers['x-test-via'];
    const opts = { req: rq, res: rs, path: '/v1/chat/completions', body: JSON.parse(t), control, serving: record };
    await nativeGateway(via === 'none' ? opts : { ...opts, mcpAllowed: model => serving.mcpToolsAllowed(rq, via, model) });
  });
  const port = await listen(gateway);
  const call = async (body, headers = {}) => { const r = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${KEY}`, ...headers }, body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: 'list my files' }], ...body }) }); assert.equal(r.status, 200); await r.text(); return received.at(-1).strata_mcp; };
  try {
    setUp();
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'app' }), true, 'every guard holds');
    assert.equal(await call({}, { 'x-test-via': 'app' }), false, 'the client did not ask');
    assert.equal(await call({ strata_mcp: 'true' }, { 'x-test-via': 'app' }), false, 'only a boolean true asks');
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'share' }), false);
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'tunnel' }), false);
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'app', origin: 'https://evil.example' }), false);
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'none' }), false, 'a caller that passes no guard: off, as before');
    serving.patch({ mcpTools: { model: MODEL, on: false } });
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'app' }), false, 'opt-in off');
    setUp({ key: null });
    assert.equal(await call({ strata_mcp: true }, { 'x-test-via': 'app' }), false, 'no key set');
  } finally { await close(gateway); await close(engine); }
});

test('the host passes the listener a request came in on', () => {
  const src = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(src, /return await openai\(req, res, apiPath\(path\), method, 'app'\);/);
  assert.match(src, /const localShare = createLocalShare\(\{ handle: \(req, res, path, method\) => openai\(req, res, path, method, 'share'\),/);
  assert.match(src, /const tunnelOrigin = createLocalShare\(\{ handle: \(req, res, path, method\) => openai\(req, res, path, method, 'tunnel'\),/);
  assert.match(src, /nativeGateway\(\{req,res,path,body:await readBody\(req,12\*1024\*1024\),control:native,serving,mcpAllowed:model=>serving\.mcpToolsAllowed\(req,via,model\)\}\)/);
  assert.equal((src.match(/[^.\w]openai\(req, res/g) ?? []).length, 4, 'the definition and three callers: no caller without a listener');
  assert.equal((src.match(/openai\(req, res, [^\n]*?, '(?:app|share|tunnel)'\)/g) ?? []).length, 3);
});
