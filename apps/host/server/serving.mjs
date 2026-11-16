/** Serving surface: API keys, access policy, and the request log. */
import { randomBytes } from 'node:crypto';
import os from 'node:os';
import { get as getStore, update } from './store.mjs';
import { nowIso, round, sameSecret, uid } from './util.mjs';

const requests = [];
const MAX_REQUESTS = 60;
const cancellations = new Map();
let revision = Date.now();

let hubRef = null;
let tunnel = { status: 'idle', mode: null, url: null, error: null, installed: false, originUrl: null };
export function setTunnel(value) { tunnel = value; emit(); }
const tunnelActive = () => ['starting', 'connected', 'stopping'].includes(tunnel.status);
let sharing = { active: false, url: null };
export function setSharing(value) { sharing = value; emit(); }
let port = 8080;
let running = false;
let bindHost = '127.0.0.1';
const isLocal = () => ['127.0.0.1', 'localhost', '::1'].includes(bindHost);

export function init(hub, opts = {}) {
  hubRef = hub;
  port = opts.port ?? port;
  running = true;
  bindHost = opts.host ?? '127.0.0.1';
}

export function baseUrl() {
  const host = isLocal() ? '127.0.0.1' : localIp();
  return `http://${host}:${port}/v1`;
}

export function state() {
  const s = getStore();
  const active = requests.filter((r) => r.state === 'generating' || r.state === 'queued');
  return {
    revision,
    running,
    baseUrl: sharing.active ? sharing.url : baseUrl(),
    // The address on this computer, the one the app's own port serves, kept apart from the sharing address (#4798).
    localUrl: baseUrl(),
    sharing,
    tunnel,
    port,
    endpoints: { openai: '/chat/completions', anthropic: '/messages', models: '/models' },
    access: {
      otherDevices: !isLocal() || sharing.active || tunnelActive(),
      apiKeySet: !!s.serving?.apiKey,
      apiKeyHint: s.serving?.apiKey ? `Ends in ${s.serving.apiKey.slice(-4)}` : null,
      allowOrigin: '*',
    },
    stats: {
      running: active.filter((r) => r.state === 'generating').length,
      waiting: active.filter((r) => r.state === 'queued').length,
      completed: requests.filter((r) => r.state === 'done').length,
      failed: requests.filter((r) => r.state === 'error').length,
      tokensOut: requests.reduce((sum, r) => sum + (r.tokensOut || 0), 0),
    },
    requests: [...requests].reverse(),
    pauseServing: !!s.serving?.pauseServing,
    mcpTools: Array.isArray(s.serving?.mcpTools) ? [...s.serving.mcpTools] : [],
    trustedOrigins: Array.isArray(s.serving?.trustedOrigins) ? [...s.serving.trustedOrigins] : [],
  };
}

function emit() {
  revision = Math.max(Date.now(), revision + 1);
  hubRef?.send('serving', state());
}

export function patch(body) {
  const s = getStore();
  const serving = { ...(s.serving ?? {}) };
  if (typeof body.otherDevices === 'boolean' && body.otherDevices !== !isLocal()) throw new Error('Change the host listen address and restart to update network access.');
  // The key is optional (owner decision, 0.2.2): removing it while sharing or the tunnel runs is allowed; the app warns in plain words (local-share.mjs, cloudflare.mjs).
  if (body.apiKey === null) serving.apiKey = null;
  else if (typeof body.apiKey === 'string' && body.apiKey.trim()) serving.apiKey = body.apiKey.trim();
  else if (body.generateKey) serving.apiKey = `sk-strata-${randomBytes(12).toString('hex')}`;
  if (typeof body.pauseServing === 'boolean') serving.pauseServing = body.pauseServing;
  // The origins of pages that may use /v1 without an API key besides this app's own (as strata serve's trusted_origins): exact
  // http(s) origins, at most 32. No UI; set through PATCH /api/serving.
  if (Array.isArray(body.trustedOrigins)) serving.trustedOrigins = [...new Set(body.trustedOrigins.map(trustedOrigin).filter(Boolean))].slice(0, 32);
  // #5661: the models whose MCP tools an API client of this computer may use (default: none).
  const choice = body.mcpTools;
  if (choice && typeof choice.model === 'string' && choice.model.trim() && choice.model.length <= 256 && typeof choice.on === 'boolean') {
    const rest = (Array.isArray(serving.mcpTools) ? serving.mcpTools : []).filter(m => m !== choice.model);
    serving.mcpTools = choice.on ? [...rest, choice.model].slice(-MCP_TOOL_MODELS_MAX) : rest;
  }
  update({ serving });
  emit();
  return state();
}

export function authorized(req) {
  const key = getStore().serving?.apiKey;
  if (!key) return true;
  const header = req.headers?.authorization;
  const bearer = typeof header === 'string' && header.startsWith('Bearer ') ? header.slice(7) : '';
  const alt = req.headers?.['x-api-key'];
  // #4535: constant time over fixed-length digests, and both headers are always compared so the time does not show which one was sent.
  const viaBearer = sameSecret(bearer, key);
  const viaHeader = sameSecret(alt, key);
  return viaBearer || viaHeader;
}

/**
 * #5661: may this API request use the model's MCP tools (strata_mcp)? Those tools act on this computer (files, programs), so a request may
 * only when every guard holds: it came in on the app's own listener (not LAN sharing, not the tunnel), from this computer (a loopback
 * address), with no Origin of another site (a web page in a browser cannot reach them), with an API key set and sent, and the person
 * turned it on for this model in Serving (default off). The answer is the first guard that refuses, or null.
 */
export const MCP_TOOL_MODELS_MAX = 64;
const LOOPBACK = /^(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|::1|::ffff:127\.\d{1,3}\.\d{1,3}\.\d{1,3})$/;
export function foreignOrigin(origin) {
  if (origin === undefined) return false;
  try { return !['localhost', '127.0.0.1', '[::1]'].includes(new URL(String(origin)).hostname); } catch { return true; }   // "null", "", garbage
}
export function mcpToolsRefusal(req, via, model) {
  if (via !== 'app') return 'listener';
  if (!LOOPBACK.test(String(req?.socket?.remoteAddress ?? ''))) return 'not-loopback';
  if (foreignOrigin(req?.headers?.origin)) return 'foreign-origin';
  if (!getStore().serving?.apiKey || !authorized(req)) return 'no-key';
  const on = getStore().serving?.mcpTools;
  if (!Array.isArray(on) || typeof model !== 'string' || !on.includes(model)) return 'off';
  return null;
}
export const mcpToolsAllowed = (req, via, model) => mcpToolsRefusal(req, via, model) === null;

/**
 * The /v1 foreign-Origin rule (owner 11-16), the same contract as strata serve's /v1: while NO API key is set, a request whose Origin
 * names a page of another site is refused, because any site can make a visitor's browser POST to this computer. A request without
 * an Origin header (curl, the SDKs, editors, server-side apps) passes; so does this app's own page (a loopback name, or - on the app's
 * own listener only, `via` 'app' - the request's own Host; LAN sharing and the tunnel serve no page), a page whose origin is in
 * serving.trustedOrigins, and an origin of another scheme (a browser extension: no web site can send one). "null", an empty Origin and
 * one that cannot be read count as another site. With a key set the key is the check.
 */
const LOOPBACK_NAMES = new Set(['localhost', '127.0.0.1', '[::1]']);
export function trustedOrigin(value) {
  try { const u = new URL(String(value ?? '').trim()); return /^https?:$/.test(u.protocol) && u.host ? `${u.protocol}//${u.host}` : null; } catch { return null; }
}
export function v1OriginRefused(req, via = 'app') {
  if (getStore().serving?.apiKey) return false;
  const origin = req?.headers?.origin;
  if (origin === undefined) return false;
  let url;
  try { url = new URL(String(origin).trim()); } catch { return true; }            // "null", "", garbage
  if (!/^https?:$/.test(url.protocol)) return false;                              // chrome-extension:// and the like
  if (LOOPBACK_NAMES.has(url.hostname) || url.hostname.endsWith('.localhost')) return false;
  if (via === 'app' && url.host && url.host === String(req?.headers?.host ?? '').toLowerCase()) return false;
  const trusted = getStore().serving?.trustedOrigins;
  return !(Array.isArray(trusted) && trusted.includes(`${url.protocol}//${url.host}`));
}

export function open({ client, endpoint, tokensIn = null, model, instanceId, gpuIds, cancel }) {
  const entry = {
    id: uid('req'),
    at: nowIso(),
    client,
    endpoint,
    state: 'queued',
    tokensIn,
    tokensOut: null,
    durationMs: 0,
    model: model ?? 'local',
    instanceId, gpuIds,
    cancellable: typeof cancel === 'function',
    cancelRequested: false,
    usageSource: 'unavailable',
  };
  if (entry.cancellable) cancellations.set(entry.id, cancel);
  requests.push(entry);
  if (requests.length > MAX_REQUESTS) {
    const oldestCompleted = requests.findIndex(r => !['generating', 'queued'].includes(r.state));
    if (oldestCompleted >= 0) requests.splice(oldestCompleted, 1);
  }
  emit();
  return entry;
}

export function mark(id, patchObj) {
  const entry = requests.find((r) => r.id === id);
  if (!entry) return;
  Object.assign(entry, patchObj);
  emit();
  return entry;
}

export function finish(id, { tokensOut, durationMs, error }) {
  const entry = requests.find((r) => r.id === id);
  if (!entry) return;
  entry.state = error ? 'error' : 'done';
  entry.tokensOut = tokensOut ?? entry.tokensOut;
  entry.durationMs = durationMs ?? entry.durationMs;
  if (error) entry.error = error;
  entry.cancellable = false;
  cancellations.delete(id);
  emit();
}

export function cancel(id) {
  const entry = requests.find((r) => r.id === id);
  if (!entry) throw new Error('This request is no longer retained.');
  if (!['generating', 'queued'].includes(entry.state)) return state();
  const abort = cancellations.get(id);
  if (!abort) throw new Error('This request cannot be cancelled by this host.');
  if (!entry.cancelRequested) { entry.cancelRequested = true; abort(); }
  emit();
  return state();
}

export function rateSample() {
  const done = requests.filter((r) => r.state === 'done' && r.tokensOut > 0);
  if (!done.length) return 0;
  const tps = done.filter((r) => r.tps > 0).map((r) => r.tps);     // answer speeds (decode time), not tokens over the whole request
  if (!tps.length) return 0;
  return round(tps.reduce((a, b) => a + b, 0) / tps.length, 1);
}

function localIp() {
  try {
    const nets = os.networkInterfaces();
    for (const list of Object.values(nets)) {
      for (const net of list ?? []) {
        if (net.family === 'IPv4' && !net.internal) return net.address;
      }
    }
  } catch {
    /* fall through */
  }
  return '127.0.0.1';
}
