/** Shared helpers for the Strata host. */
import { createHash, randomUUID, timingSafeEqual } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { whichCommand } from './platform-paths.mjs';

export const exec = promisify(execFile);
export const uid = (prefix = 'id') => `${prefix}_${randomUUID().slice(0, 8)}`;
export const nowIso = () => new Date().toISOString();
export const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
export const round = (n, d = 1) => Number(n.toFixed(d));

export function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  res.end(payload);
}

export function error(res, status, message, extra = {}) {
  json(res, status, { error: message, ...extra });
}

/**
 * Two secrets compared in constant time (#4535). Both sides are hashed first, so the comparison always runs over two
 * 32-byte digests: neither the content nor the length of the given value can end it early. Anything that is not a
 * non-empty string never matches (an empty secret is "not set", not "equal to an empty header").
 */
export function sameSecret(given, wanted) {
  const ok = typeof given === 'string' && typeof wanted === 'string' && given.length > 0 && wanted.length > 0;
  const digest = value => createHash('sha256').update(typeof value === 'string' ? value : '', 'utf8').digest();
  return timingSafeEqual(digest(given), digest(wanted)) && ok;
}

/**
 * The request's URL, or null when the request cannot be read as one (#4397). The Host header is the caller's: an empty
 * value, one with a space, a path, credentials or an unclosed bracket used to make `new URL` throw before the routing
 * try/catch and end the whole process. A missing Host (HTTP/1.0) is read as localhost; a good one keeps its port and IPv6 form.
 */
export function requestUrl(req) {
  const host = req.headers?.host;
  try {
    let authority = 'localhost';
    if (host !== undefined) {
      if (typeof host !== 'string' || !host.trim() || /[\s/?#@\\]/.test(host)) return null;
      const base = new URL(`http://${host}`);
      if (!base.host || base.username || base.password || base.search || base.hash || base.pathname !== '/') return null;
      authority = host;
    }
    return new URL(req.url ?? '/', `http://${authority}`);
  } catch {
    return null;
  }
}

/** The answer to a request that could not be read as a URL: bounded, and the connection closes. */
export function badRequest(res) {
  if (res.headersSent) return res.end();
  const payload = '{"error":"bad_host"}';
  res.writeHead(400, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(payload), 'cache-control': 'no-store', connection: 'close' });
  res.end(payload);
}

export class InvalidJsonError extends Error {}

export async function readBody(req, limit = 4 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('Request body too large');
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  const raw = Buffer.concat(chunks).toString('utf8');
  try {
    return JSON.parse(raw);
  } catch {
    throw new InvalidJsonError('Body must be JSON');
  }
}

/** Run a command, returning stdout or '' when it is missing or fails. */
export async function probe(cmd, args = [], timeout = 4000) {
  try {
    const { stdout } = await exec(cmd, args, { timeout, maxBuffer: 4 * 1024 * 1024 });
    return stdout.trim();
  } catch {
    return '';
  }
}

/** Where `binary` is on PATH, or null. `/usr/bin/env which` on POSIX, System32\where.exe on Windows (first match). */
export async function which(binary, { env = process.env, platform = process.platform, run = probe } = {}) {
  const { command, args } = whichCommand(binary, env, platform);
  const out = await run(command, args);
  return out.split(/\r?\n/)[0].trim() || null;
}

/** Minimal SSE hub. */
export class EventHub {
  #clients = new Set();

  add(res) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });
    res.write(': connected\n\n');
    this.#clients.add(res);
    const ping = setInterval(() => res.write(': ping\n\n'), 15000);
    res.on('close', () => {
      clearInterval(ping);
      this.#clients.delete(res);
    });
  }

  send(event, payload) {
    const frame = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
    for (const res of this.#clients) {
      try {
        res.write(frame);
      } catch {
        this.#clients.delete(res);
      }
    }
  }

  get size() {
    return this.#clients.size;
  }
}

/** The raw bytes of a request body (an upload), up to `limit`. */
export async function readRaw(req, limit = 16 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('That upload is too big.');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
