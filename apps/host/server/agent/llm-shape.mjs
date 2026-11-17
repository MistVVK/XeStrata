/**
 * Response settings for agent turns.
 *
 * The chat's own path sends the Expert sampling (temperature, top-p, top-k, repeat penalty, seed) and the composer's thinking budget with
 * every request. The agent's requests are made by the Harness, which has no place for either, so they never reached the model. The agent's
 * model route goes through the small forwarding endpoint below (on this computer only), which adds exactly the fields that differ from the
 * defaults and passes everything else through untouched; with every setting at its default the request and the reply pass byte for byte.
 * Since TOOLCALL-B (0.2.16) the route is forwarded always, not only while a setting differs: the endpoint also reads, without changing
 * anything, whether the reply carries a tool call the server could not run (`strata_rejected_tool_calls`, `strata_unread_tool_markup`), so
 * the agent bridge can tell the model and let it go on (tool-fix.mjs). `STRATA_AGENT_FORWARD=0` restores the older choice (agent-bridge.mjs).
 *
 * Only a local model, or an OpenAI-style server on this computer or this network, is shaped: a cloud service would reject fields it does not know.
 */

import http from 'node:http';
import https from 'node:https';

/** The thinking budgets the composer offers (0 = no limit is not a budget). */
export const BUDGETS = Object.freeze([2048, 8192, 32768]);
/** The offered budget nearest to what a chat asks for (0 stays 0). */
export function nearestBudget(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return BUDGETS.reduce((best, b) => (Math.abs(b - n) < Math.abs(best - n) ? b : best), BUDGETS[0]);
}

/** Expert setting id -> request field, and the default each one has (the catalog's recommended value). */
const SAMPLING = Object.freeze({
  'sample.temp': ['temperature', 0.6],
  'sample.topP': ['top_p', 0.95],
  'sample.topK': ['top_k', 40],
  'sample.repeat': ['repeat_penalty', 1.05],
  'sample.seed': ['seed', -1],
});

/** The request fields for the Expert settings that are not at their default, or null when all are. */
export function samplingFromExpert(expert) {
  const out = {};
  for (const [id, [field, recommended]] of Object.entries(SAMPLING)) {
    const v = expert?.[id];
    if (typeof v === 'number' && Number.isFinite(v) && v !== recommended) out[field] = v;
  }
  if (expert?.['sample.speedProjection'] === false) out.experimental_speed_projection = false;
  const cap = Number(expert?.['sample.maxTokens']);
  if (Number.isFinite(cap) && cap > 0) out.max_tokens = Math.floor(cap);
  return Object.keys(out).length ? out : null;
}

/**
 * The request body with the person's settings added. A field the request already carries is left alone (the Harness's own wins), the
 * thinking budget is only added while thinking is on, and an output cap only ever lowers the request's own.
 */
export function shapeBody(body, { sampling = null, budget = 0 } = {}) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return body;
  const out = { ...body };
  for (const [field, value] of Object.entries(sampling ?? {})) {
    if (field === 'max_tokens' || field === 'max_completion_tokens') continue;
    if (out[field] === undefined) out[field] = value;
  }
  if (sampling?.max_tokens) {
    for (const key of ['max_tokens', 'max_completion_tokens']) if (Number.isFinite(out[key])) out[key] = Math.min(out[key], sampling.max_tokens);
    if (out.max_tokens === undefined && out.max_completion_tokens === undefined) out.max_tokens = sampling.max_tokens;
  }
  const thinkingOff = out.chat_template_kwargs?.enable_thinking === false;
  if (budget > 0 && !thinkingOff && out.reasoning_budget_tokens === undefined) out.reasoning_budget_tokens = budget;
  return out;
}

/** An address on this computer or on a private network: where the person's own model servers live. */
export function isPrivateHost(url) {
  let host;
  try { host = new URL(url).hostname.toLowerCase(); } catch { return false; }
  host = host.replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host === '::1' || host.endsWith('.local') || host.endsWith('.lan')) return true;
  const m = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(host);
  if (!m) return false;
  const [a, b] = [Number(m[1]), Number(m[2])];
  return a === 127 || a === 10 || (a === 192 && b === 168) || (a === 172 && b >= 16 && b <= 31) || (a === 169 && b === 254) || (a === 100 && b >= 64 && b <= 127);
}

const HOP = new Set(['host', 'connection', 'content-length', 'transfer-encoding', 'keep-alive', 'upgrade', 'proxy-connection']);

/** The two fields the Strata server adds to a reply whose tool call could not run (serve/server.py; RESULTS-SERVE-1116 "Interface"). */
export const MARKER_FIELDS = Object.freeze(['strata_rejected_tool_calls', 'strata_unread_tool_markup']);
const MARKER_HINT = /strata_(?:rejected_tool_calls|unread_tool_markup)/;

/** The two fields of one reply object, wherever the server puts them: top level (chat chunk or body, Anthropic message_delta or message), or the response object (/v1/responses). */
export function markersOf(obj) {
  if (!obj || typeof obj !== 'object') return null;
  for (const o of [obj, obj.response, obj.message]) {
    if (!o || typeof o !== 'object') continue;
    const rejected = Array.isArray(o.strata_rejected_tool_calls) && o.strata_rejected_tool_calls.length ? o.strata_rejected_tool_calls : null;
    const unread = o.strata_unread_tool_markup && typeof o.strata_unread_tool_markup === 'object' ? o.strata_unread_tool_markup : null;
    if (rejected || unread) return { rejected, unread };
  }
  return null;
}

/** The text of a chat message's content (a string, or text parts). */
const contentText = c => (typeof c === 'string' ? c : Array.isArray(c) ? c.map(p => (typeof p?.text === 'string' ? p.text : '')).join('') : '');

/** What a request asked: its user-role texts (newest last) and the tool names it declared. Read only when a reply carried a marker. */
export function requestFacts(body) {
  let doc;
  try { doc = JSON.parse(Buffer.isBuffer(body) ? body.toString('utf8') : String(body ?? '')); } catch { return { users: [], declared: [] }; }
  const users = (Array.isArray(doc?.messages) ? doc.messages : []).filter(m => m?.role === 'user').map(m => contentText(m.content));
  if (Array.isArray(doc?.input)) for (const m of doc.input) if (m?.role === 'user') users.push(contentText(m.content));
  const declared = (Array.isArray(doc?.tools) ? doc.tools : []).map(t => t?.function?.name ?? t?.name).filter(n => typeof n === 'string' && n);
  return { users, declared };
}

/**
 * Reads a reply as it passes, without changing it: SSE lines (`data: {...}`) or one JSON body. Only a line that names one of the two fields is
 * parsed, so an ordinary reply costs a substring search per line. `found(markers)` is called the moment a marker is read (before those bytes are
 * passed on), at most once per reply.
 */
export class MarkerTap {
  constructor({ found, sse = true, limit = 8 << 20 }) {
    this.found = found; this.sse = sse; this.limit = limit;
    this.decoder = new TextDecoder('utf-8'); this.line = ''; this.body = ''; this.done = false; this.overflow = false;
  }
  #take(text) {
    if (this.done || !MARKER_HINT.test(text)) return;
    let obj = null;
    try { obj = JSON.parse(text); } catch { return; }
    const m = markersOf(obj);
    if (m) { this.done = true; this.found(m); }
  }
  push(chunk) {
    if (this.done) return;
    const text = this.decoder.decode(chunk, { stream: true });
    if (!this.sse) { if (!this.overflow) { this.body += text; if (this.body.length > this.limit) { this.overflow = true; this.body = ''; } } return; }
    this.line += text;
    let at;
    while ((at = this.line.indexOf('\n')) >= 0) {
      const one = this.line.slice(0, at).replace(/\r$/, '');
      this.line = this.line.slice(at + 1);
      if (one.startsWith('data:')) this.#take(one.slice(5).trim());
    }
    if (this.line.length > this.limit) this.line = '';
  }
  end() {
    if (this.done) return;
    const rest = this.decoder.decode();
    if (!this.sse) { if (!this.overflow) this.#take(this.body + rest); return; }
    const one = (this.line + rest).replace(/\r$/, '');
    if (one.startsWith('data:')) this.#take(one.slice(5).trim());
  }
}

/**
 * The forwarding endpoint. `targetOf(key)` gives the real base address of a shaped route (or null), `samplingNow()` the Expert fields that
 * differ from the defaults right now. Called for `/api/internal/agent-llm/<key>/<path>`; this computer's own address only.
 * `onMarker({key, at, users, declared, rejected, unread})` hears of a reply whose tool call could not run (the agent bridge turns it into a tool
 * error for the model, TOOLCALL-B). The reply's bytes are passed on unchanged either way.
 */
export function createLlmProxy({ targetOf, samplingNow, onMarker = null }) {
  return function handle(req, res, key, rest) {
    const remote = req.socket?.remoteAddress ?? '';
    if (!/^(::1|127\.|::ffff:127\.)/.test(remote)) { res.writeHead(403, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'This endpoint is for the assistant on this computer only.' })); return; }
    const base = targetOf(key);
    if (!base) { res.writeHead(404, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: 'Unknown model route.' })); return; }
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('error', () => res.destroy());
    req.on('end', () => {
      let body = Buffer.concat(chunks);
      const sent = body;
      const budget = nearestBudget(req.headers['x-strata-budget']);
      const sampling = samplingNow();
      // Nothing to add (every setting at its default): the request goes on byte for byte as the assistant wrote it.
      if (req.method === 'POST' && /\/chat\/completions\/?$/.test(rest) && body.length && (budget || sampling)) {
        try { body = Buffer.from(JSON.stringify(shapeBody(JSON.parse(body.toString('utf8')), { sampling, budget }))); } catch { /* not JSON: pass it on untouched */ }
      }
      // The route's address is the model's own base (it ends in /v1): the assistant's /v1 is that same prefix.
      const target = new URL(base.replace(/\/+$/, '') + rest.replace(/^\/v1(?=\/|$|\?)/, ''));
      const headers = {};
      for (const [k, v] of Object.entries(req.headers)) if (!HOP.has(k) && !k.startsWith('x-strata-')) headers[k] = v;
      headers['content-length'] = String(body.length);
      const lib = target.protocol === 'https:' ? https : http;
      const upstream = lib.request(target, { method: req.method, headers }, reply => {
        const out = { ...reply.headers }; delete out['transfer-encoding']; delete out.connection;
        res.writeHead(reply.statusCode ?? 502, out);
        // Read the reply for the two markers as it passes (this listener runs before pipe's own, so a marker is known before the assistant
        // has the reply's end). An encoded body (gzip) is not read.
        const type = String(reply.headers['content-type'] ?? '');
        if (onMarker && req.method === 'POST' && !reply.headers['content-encoding'] && (reply.statusCode ?? 0) < 300 && /event-stream|json/i.test(type)) {
          const tap = new MarkerTap({ sse: /event-stream/i.test(type), found: m => {
            try { onMarker({ key, at: Date.now(), ...requestFacts(sent), ...m }); } catch { /* the reply goes on regardless */ }
          } });
          reply.on('data', c => tap.push(c));
          reply.on('end', () => tap.end());
        }
        reply.pipe(res);
        reply.on('error', () => res.destroy());
      });
      upstream.on('error', e => { if (!res.headersSent) { res.writeHead(502, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: `The model server could not be reached: ${e.message}` })); } else res.destroy(); });
      // The assistant stopped waiting (Stop): stop the model too.
      res.on('close', () => { if (!res.writableEnded) upstream.destroy(); });
      upstream.end(body);
    });
  };
}
