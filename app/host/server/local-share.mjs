/** A separate, key-protected listener for inference only. Never serves the UI or host APIs. */
import http from 'node:http';
import os from 'node:os';
import { apiPath } from './api-paths.mjs';
import { listenProblem } from './ports.mjs';
import { badRequest, requestUrl } from './util.mjs';
/** Sharing without a key is allowed (owner decision, 0.2.2): the person is told once, in plain words, and the app keeps offering to make a key. */
export const NO_KEY_WARNING = 'No API key is set, so any device on this network can use your model. Create an API key to protect it.';
export function createLocalShare({ handle, hasKey, changed = () => {}, host = '0.0.0.0', port = 8792, allowResponses = false }) {
  let listener = null, active = false;
  const address = () => Object.values(os.networkInterfaces()).flat().find(n => n?.family === 'IPv4' && !n.internal)?.address ?? '127.0.0.1';
  const state = () => ({ active, url: active ? `http://${host === '127.0.0.1' ? host : address()}:${listener.address().port}/v1` : null, warning: active && !hasKey() ? NO_KEY_WARNING : null });
  async function start() {
    if (active) return state();
    if (listener) throw new Error('Sharing is already starting.');
    listener = http.createServer(async (req, res) => {
      const url = requestUrl({ url: req.url, headers: {} });   // the Host header is not read here; a request target that is not a path is a 400 (#4397)
      if (!url) return badRequest(res);
      const path = apiPath(url.pathname);
      const method = req.method ?? 'GET';
      // #3208: with an API key set, a browser page of any site can call the shared API - the key decides, not the
      // origin. The browser's preflight (OPTIONS) never carries the key (the spec says so), so it is answered here,
      // before the path check and the key check; the real request that follows still goes through both. Without a
      // key no CORS headers go out, as before: the listener stays "a device on this network" (SECURITY.md), and a
      // web page of another site on the internet cannot read the model through it.
      if (hasKey()) {
        res.setHeader('access-control-allow-origin', '*');
        res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
        // the headers the preflight asks for (SDKs add their own; Anthropic browser clients send anthropic-version)
        res.setHeader('access-control-allow-headers', String(req.headers['access-control-request-headers'] ?? '') || 'authorization, content-type, x-api-key, x-strata-client, anthropic-version, anthropic-beta');
        res.setHeader('access-control-max-age', '600');
        if (method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
      }
      const isResponsesPost = allowResponses && path === '/v1/responses' && method === 'POST';
      if (!isResponsesPost && !['/v1/models','/v1/chat/completions','/v1/messages'].includes(path)) { res.writeHead(404); res.end('Not found'); return; }
      try { await handle(req, res, path, method); }
      catch { if (!res.headersSent) res.writeHead(500); res.end('Request failed'); }
    });
    try { await new Promise((resolve,reject) => { listener.once('error',reject); listener.listen(port,host,resolve); }); }
    catch(e) { listener.close(); listener=null; throw new Error(listenProblem(e, port, host)); }
    active=true; changed(); return state();
  }
  async function stop() {
    if (!listener) return state();
    const current=listener; listener=null; active=false;
    current.closeAllConnections(); await new Promise(resolve=>current.close(resolve)); changed(); return state();
  }
  return { state, start, stop };
}
