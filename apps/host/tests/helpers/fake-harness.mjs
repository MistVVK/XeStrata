/**
 * A stand-in for the DeepSeek Harness's `dsh web` gateway, for tests that need the agent bridge end to end without the real Harness or a model.
 * It speaks the two carriers the bridge uses (HTTP unary RPC and the `/api/remote.mux` WebSocket) on loopback and plays scripted sessions.
 *
 * `SCRIPT:bg` answers, ends its turn, and then, the way the Harness's job plugin does when a background command finishes, wakes the idle agent
 * with a completion notice and a turn of its own (no request id of the person's). That reply is streamed and committed at once, before anyone
 * could follow it; the turn ends a moment later, so a page can still take it over while it runs.
 * `FAKE_HARNESS_WAKE_MS` sets the pause before the wake (default 300).
 * Modelled on DeepSeek Harness (dsh) commit fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159.
 */
import { createServer } from 'node:http';
import { createHash, randomUUID } from 'node:crypto';

const token = randomUUID().replace(/-/g, '');
const wakeMs = Number(process.env.FAKE_HARNESS_WAKE_MS ?? 300);
const sessions = new Map();   // sessionId -> { events, followers:Set, permission, turn }
const sleep = ms => new Promise(r => setTimeout(r, ms));

function session(id) { return sessions.get(id); }
function append(s, type, data) {
  const event = { seq: s.events.length + 1, type, data, time: Date.now() };
  s.events.push(event);
  for (const f of s.followers) f({ type: 'event', event });
  return event;
}
function stream(s, frame) {
  for (const f of s.followers) f({ type: 'assistant-stream', frame });
}
function snapshot(s) {
  return {
    type: 'snapshot', cursor: s.events.length, hasMore: false, header: {}, projections: {},
    records: s.events.map(event => ({ type: 'event', event })),
    assistantStream: { revision: 1 },
  };
}

/** One model reply: the stream frames, then the committed message. */
async function reply(s, text) {
  const attemptId = `att-${randomUUID()}`;
  stream(s, { type: 'start', attemptId, turn: s.turn, step: 1 });
  const words = text.split(/(?<= )/);
  for (const [i, w] of words.entries()) {
    stream(s, { type: 'chunk', attemptId, index: i, chunk: { type: 'text-delta', index: 0, text: w } });
  }
  stream(s, { type: 'end', attemptId });
  append(s, 'assistant/message', { turn: s.turn, step: 1, message: { role: 'assistant', content: [{ type: 'text', text }] } });
}

async function turn(s, message, body) {
  s.turn++;
  append(s, 'turn/start', { turn: s.turn });
  append(s, 'user/message', message);
  append(s, 'step/start', { turn: s.turn, step: 1 });
  await body();
  append(s, 'step/end', { turn: s.turn, step: 1 });
  append(s, 'turn/end', { turn: s.turn, reason: { kind: 'completed' } });
}

async function play(s, requestId, text) {
  const bg = /^SCRIPT:bg\b/.test(text);
  await turn(s, { source: { kind: 'user', rpcId: requestId }, content: [{ type: 'text', text }] }, () => reply(s, bg ? 'Started the job in the background. ' : `Echo: ${text}`));
  if (!bg) return;
  await sleep(wakeMs);
  const notice = { source: { kind: 'tool-jobs', form: 'notice', summary: 'job 1 completed' }, content: [{ type: 'text', text: 'Job 1 completed with exit code 0. Output: 42' }] };
  await turn(s, notice, async () => { await reply(s, 'The background job finished and printed 42. '); await sleep(800); });
}

function rpc(method, args) {
  const r = args.request ?? {};
  switch (method) {
    case 'session/create': {
      const sessionId = randomUUID();
      sessions.set(sessionId, { events: [], followers: new Set(), permission: null, turn: 0 });
      return { sessionId };
    }
    case 'session/projections': return { values: { permissions: { currentValue: session(r.sessionId)?.permission ?? null } } };
    case 'commands/execute': {
      const m = /^\/permission\s+(\S+)/.exec(args.line ?? '');
      if (m && session(args.agentId)) session(args.agentId).permission = m[1];
      return { result: { kind: 'success', text: /strata-gate/.test(args.line ?? '') ? 'strata-gate active' : '' } };
    }
    case 'session/prompt': {
      const s = session(r.sessionId);
      if (!s) throw Object.assign(new Error('no such session'), { code: 'session/not-found' });
      void play(s, r.requestId, (r.content ?? []).map(p => p.text ?? '').join(''));
      return {};
    }
    default: return null;   // session/selectModel, session/cancel, goals/get, $events/result: nothing to do
  }
}

// -- a minimal RFC 6455 server: text frames, close and ping, which is all the bridge sends ---------------------------------------

function sendText(socket, text) {
  const data = Buffer.from(text);
  const head = data.length < 126 ? Buffer.from([0x81, data.length])
    : data.length < 65536 ? Buffer.from([0x81, 126, data.length >> 8, data.length & 255])
    : Buffer.concat([Buffer.from([0x81, 127]), (() => { const b = Buffer.alloc(8); b.writeBigUInt64BE(BigInt(data.length)); return b; })()]);
  if (!socket.destroyed) socket.write(Buffer.concat([head, data]));
}
function frames(socket, onText) {
  let buf = Buffer.alloc(0);
  socket.on('data', chunk => {
    buf = Buffer.concat([buf, chunk]);
    for (;;) {
      if (buf.length < 2) return;
      const op = buf[0] & 15; let len = buf[1] & 127; let at = 2;
      if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); at = 4; }
      else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); at = 10; }
      const masked = (buf[1] & 128) !== 0;
      if (buf.length < at + (masked ? 4 : 0) + len) return;
      const mask = masked ? buf.subarray(at, at + 4) : null; at += masked ? 4 : 0;
      const payload = Buffer.from(buf.subarray(at, at + len));
      if (mask) for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
      buf = buf.subarray(at + len);
      if (op === 1) onText(payload.toString());
      else if (op === 8) { socket.end(Buffer.from([0x88, 0])); return; }
      else if (op === 9) socket.write(Buffer.concat([Buffer.from([0x8a, payload.length]), payload]));
    }
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (req.method === 'GET' && url.pathname === '/') {
    if (url.searchParams.get('token') !== token) { res.writeHead(403).end(); return; }
    res.writeHead(302, { 'set-cookie': 'dsh-auth-fake=1; Path=/', location: '/' }).end();
    return;
  }
  if (req.method === 'POST' && url.pathname.startsWith('/api/')) {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      const method = url.pathname.slice(5);
      let result;
      try { result = { ok: true, value: rpc(method, JSON.parse(body).payload?.args ?? {}) }; }
      catch (e) { result = { ok: false, error: { code: e.code ?? 'fake/error', message: e.message } }; }
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ result }));
    });
    return;
  }
  res.writeHead(404).end();
});

server.on('upgrade', (req, socket) => {
  if (!String(req.headers.cookie ?? '').includes('dsh-auth-fake=1')) { socket.destroy(); return; }
  const accept = createHash('sha1').update(`${req.headers['sec-websocket-key']}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64');
  socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  const open = new Map();   // streamId -> stop
  socket.on('close', () => { for (const stop of open.values()) stop(); open.clear(); });
  socket.on('error', () => {});
  frames(socket, text => {
    let m; try { m = JSON.parse(text); } catch { return; }
    if (m.type === 'cancel') { open.get(m.streamId)?.(); open.delete(m.streamId); return; }
    if (m.type !== 'open') return;
    const send = value => sendText(socket, JSON.stringify({ type: 'item', streamId: m.streamId, value }));
    if (m.endpoint !== 'session/follow') { open.set(m.streamId, () => {}); return; }
    const s = session(m.payload?.args?.request?.address?.sessionId);
    if (!s) { sendText(socket, JSON.stringify({ type: 'error', streamId: m.streamId, error: { code: 'session/not-found', message: 'no such session' } })); return; }
    send(snapshot(s));
    s.followers.add(send);
    open.set(m.streamId, () => s.followers.delete(send));
  });
});

server.listen(0, '127.0.0.1', () => { process.stdout.write(`http://127.0.0.1:${server.address().port}/?token=${token}\n`); });
