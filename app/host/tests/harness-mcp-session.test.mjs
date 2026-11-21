// #6014, the desktop agent's side: an MCP server at an address that was restarted (the UE5 editor) forgot the session, and every tool
// call failed with "-32600 Unknown session id ...; client should reinitialize" until the model was unloaded. The staged Harness's MCP
// client is patched at staging (app/desktop/scripts/harness-mcp-session.mjs). The fixture is the exact MIT-licensed
// @deepseek-ai/dsh-mcp-client 0.2.1-alpha.3 lib/index.js (the same DeepSeek licence as fixtures/dsh-tool-fs-search-LICENSE), run here with
// its imports stubbed and a fake MCP server behind a fake SDK client: the module's own connection, reconnect and tool code is tested.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';
import { HARNESS_MCP_SESSION_PATCH, patchStagedHarnessMcpSession, patchedMcpSource } from '../../desktop/scripts/harness-mcp-session.mjs';

const original = readFileSync(new URL('./fixtures/dsh-mcp-client-0.2.1-alpha.3-index.js.fixture', import.meta.url), 'utf8').replaceAll('\r\n', '\n');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const UNKNOWN = sid => `Error POSTing to endpoint: {"jsonrpc":"2.0","id":6,"error":{"code":-32600,"message":"Unknown session id '${sid}' for 'tools/call'; client should reinitialize"}}`;

/** A fake MCP server: sessions it knows, the calls it ran, and whether a call errors otherwise. */
function fakeServer() {
  return { sessions: new Set(), next: 1, ran: [], initialized: 0, restart() { this.sessions.clear(); }, failWith: null };
}

/** The module's code in a fresh context with its imports stubbed; returns its internal startConnection and the reconnect policy helper. */
function load(source, server) {
  const chain = new Proxy(function () {}, { get: () => chain, apply: () => chain });
  class Client {
    constructor() { this.onclose = undefined; this.transport = undefined; }
    async connect(transport) { this.transport = transport; this.session = `s-${server.next++}`; server.sessions.add(this.session); server.initialized++; }
    getInstructions() { return ''; }
    getServerCapabilities() { return { tools: {} }; }
    async listTools() { return { tools: [{ name: 'echo', inputSchema: { type: 'object', properties: { text: { type: 'string' } } } }] }; }
    async callTool({ name, arguments: args }) {
      if (!server.sessions.has(this.session)) throw new Error(UNKNOWN(this.session));
      if (server.failWith) throw server.failWith;
      server.ran.push({ name, args, session: this.session });
      return { content: [{ type: 'text', text: `ran ${args.text}` }] };
    }
    async close() { const t = this.transport; this.transport = undefined; if (t) this.onclose?.(); }
  }
  class StreamableHTTPClientTransport { constructor(url) { this.url = String(url); } async close() {} }
  const code = source.replace(/^import .*;\n/gm, '').replace(/^export \{[^}]*\};?\s*$/m, '') + '\n({ startConnection, resolveReconnectPolicy });';
  return runInNewContext(code, {
    z: chain, scopeOf: () => undefined, MAX_TIMER_DELAY_MS: 2 ** 31 - 1, Client, StreamableHTTPClientTransport,
    specTypeSchemas: { CallToolResult: { '~standard': { validate: value => ({ value }) } } },
    assertNever: v => { throw new Error(String(v)); }, StdioClientTransport: class {}, scrubbedParentEnv: () => ({}), createHash,
    isDeepStrictEqual: (a, b) => JSON.stringify(a) === JSON.stringify(b), isImageAdmissionError: () => false, assertSupportedJsonSchema: () => {},
    Buffer, URL, console, setTimeout, clearTimeout, Promise, Date,
  });
}

async function connect(t, source) {
  const server = fakeServer(), mod = load(source, server), tools = new Map(), log = [];
  const ctx = { logger: { info: m => log.push(m), warn: m => log.push(m), error: m => log.push(m) }, tools: { register: d => { tools.set(d.name, d); return () => tools.delete(d.name); } } };
  const conn = mod.startConnection(ctx, { serverName: 'ue5', transport: 'streamable-http', url: 'http://127.0.0.1:9/mcp', toolCallTimeoutMs: 60000 },
    mod.resolveReconnectPolicy({ initialDelayMs: 20, maxDelayMs: 200, maxAttempts: 3 }, 'test'));
  t.after(() => conn.dispose());
  await conn.ready;
  const call = async text => { const tool = tools.get('mcp__ue5__echo'); assert.ok(tool, 'the tool is registered'); return tool.execute({ text }, { signal: AbortSignal.timeout(5000) }); };
  return { server, tools, log, call };
}
const until = async (check, ms = 2000) => { const end = Date.now() + ms; while (!check()) { if (Date.now() > end) return false; await new Promise(r => setTimeout(r, 10)); } return true; };

test('the pinned source and the patch: one exact place for every edit, and a second run refuses', () => {
  assert.equal(sha256(Buffer.from(original)), HARNESS_MCP_SESSION_PATCH.beforeSha256, 'the fixture is the pinned file');
  assert.throws(() => patchedMcpSource(patchedMcpSource(original)), /expected exactly one/);
});

test('unpatched (the bug): after the server restarts, every call fails and the session is never renewed', async t => {
  const { server, call } = await connect(t, original);
  assert.equal((await call('a')).content[0].text, 'ran a');
  server.restart();
  for (const text of ['b', 'c']) await assert.rejects(call(text), /Unknown session id/);
  await new Promise(r => setTimeout(r, 200));
  await assert.rejects(call('d'), /Unknown session id/);
  assert.equal(server.initialized, 1);
});

test('patched: the refused call fails once, the client starts a new session, the next call runs; nothing runs twice', async t => {
  const { server, call, log, tools } = await connect(t, patchedMcpSource(original));
  assert.equal((await call('a')).content[0].text, 'ran a');
  server.restart();
  await assert.rejects(call('b'), /Unknown session id/, 'the refused call is not sent again');
  assert.ok(await until(() => server.initialized === 2 && tools.has('mcp__ue5__echo')), 'a new initialize and a new tool list');
  assert.equal((await call('c')).content[0].text, 'ran c');
  assert.deepEqual(server.ran.map(r => r.args.text), ['a', 'c'], 'b never ran, c ran once');
  assert.notEqual(server.ran[0].session, server.ran[1].session);
  assert.ok(log.some(m => /no longer knows this session/.test(m)), log.join('\n'));
});

test('patched: a 404 answer counts as an expired session too', async t => {
  const { server, call } = await connect(t, patchedMcpSource(original));
  server.failWith = Object.assign(new Error('Error POSTing to endpoint (HTTP 404): Session not found'), { code: 404 });
  await assert.rejects(call('x'), /HTTP 404/);
  server.failWith = null;
  assert.ok(await until(() => server.initialized === 2));
  assert.equal((await call('y')).content[0].text, 'ran y');
});

test('patched: other failures keep the session (a tool error, a timeout-like failure)', async t => {
  const { server, call } = await connect(t, patchedMcpSource(original));
  server.failWith = new Error('Request timed out');
  await assert.rejects(call('x'), /timed out/);
  server.failWith = Object.assign(new Error('MCP error -32602: Invalid params for tool echo'), { code: -32602 });
  await assert.rejects(call('y'), /Invalid params/);
  await new Promise(r => setTimeout(r, 150));
  assert.equal(server.initialized, 1, 'no reconnect');
  server.failWith = null;
  assert.equal((await call('z')).content[0].text, 'ran z');
});

test('patched: two calls refused at once start one new session', async t => {
  const { server, call } = await connect(t, patchedMcpSource(original));
  server.restart();
  const results = await Promise.allSettled([call('p'), call('q')]);
  assert.deepEqual(results.map(r => r.status), ['rejected', 'rejected']);
  assert.ok(await until(() => server.initialized === 2));
  await new Promise(r => setTimeout(r, 150));
  assert.equal(server.initialized, 2, 'one reconnect, not two');
  assert.equal((await call('r')).content[0].text, 'ran r');
});

function staged(t, { source = original, manifest = { name: '@deepseek-ai/dsh-mcp-client', version: '0.2.1-alpha.3' } } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-harness-mcp-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, HARNESS_MCP_SESSION_PATCH.file);
  mkdirSync(join(file, '..'), { recursive: true });
  writeFileSync(file, source); writeFileSync(join(root, 'node_modules', '@deepseek-ai', 'dsh-mcp-client', 'package.json'), JSON.stringify(manifest));
  return { root, file };
}

test('staging: patched once and recorded; another version, other bytes or a second run refuse; the installer applies it', t => {
  const f = staged(t);
  assert.deepEqual(patchStagedHarnessMcpSession(f.root), { ...HARNESS_MCP_SESSION_PATCH, afterSha256: sha256(readFileSync(f.file)) });
  assert.throws(() => patchStagedHarnessMcpSession(f.root), /unexpected lib\/index\.js SHA-256/);
  assert.throws(() => patchStagedHarnessMcpSession(staged(t, { manifest: { name: '@deepseek-ai/dsh-mcp-client', version: '0.2.1-alpha.4' } }).root), /expected @deepseek-ai\/dsh-mcp-client 0\.2\.1-alpha\.3/);
  assert.throws(() => patchStagedHarnessMcpSession(staged(t, { source: original + ' ' }).root), /unexpected lib\/index\.js SHA-256/);
  const stage = readFileSync(new URL('../../desktop/scripts/stage-harness.mjs', import.meta.url), 'utf8');
  assert.match(stage, /import \{ patchStagedHarnessMcpSession \} from '\.\/harness-mcp-session\.mjs';/);
  assert.match(stage, /const patches = \[[^\]]*patchStagedHarnessMcpSession\(out\)[^\]]*\];/);
});
