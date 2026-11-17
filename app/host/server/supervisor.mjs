/**
 * Engine supervisor (what is left of it): the engine status the Monitor and the engine page read, its log ring buffer, the parser of the
 * engine's Prometheus metrics, and the translation of the tuning configuration into flags for the command preview.
 *
 * It starts nothing: the llama.cpp `llama-server` it used to find and start is gone (see start() below). Models run on the Strata engine.
 */
import { CONTEXT_TOKENS } from './catalog.mjs';

const LOG_LIMIT = 400;

const state = {
  status: 'stopped', // stopped | starting | running | stopping | error
  binary: null,
  binarySource: null,
  pid: null,
  port: 8080,
  modelId: null,
  modelPath: null,
  args: [],
  command: '',
  startedAt: null,
  error: null,
  logs: [],
  metrics: {
    source: 'none',
    promptTokens: 0,
    predictedTokens: 0,
    promptTps: 0,
    outputTps: 0,
    kvUsage: 0,
    kvTokens: 0,
    requestsProcessing: 0,
    requestsDeferred: 0,
    sampledAt: null,
  },
};

let child = null;
let pollTimer = null;
let hubRef = null;
let mcpAt=0,mcpStatus=null,mcpPending=false;

export function init(hub) {
  hubRef = hub;
}

/* --------------------------------------------------------------- arguments --- */

const KV_TYPES = { INT8: 'q8_0', FP16: 'f16', Q8_0: 'q8_0', Q4_0: 'q4_0' };

/**
 * Turn the tuning configuration into a real llama-server command line.
 * Exported so the UI can show exactly what will run.
 */
export function buildArgs({ modelPath, config, hardware, port = state.port }) {
  const beginner = config.beginner ?? {};
  const params = new Map((config.expert ?? []).map((p) => [p.id, p.value]));
  const num = (id, fallback) => {
    const v = Number(params.get(id));
    return Number.isFinite(v) ? v : fallback;
  };
  const ctxLabel = beginner.contextLength === 'recommended' ? '128K' : String(beginner.contextLength).toUpperCase();
  const ctx = CONTEXT_TOKENS[ctxLabel.toLowerCase()] ?? 32768;

  const args = [
    '-m', modelPath,
    '--host', '127.0.0.1',
    '--port', String(port),
    '-c', String(ctx),
    '--metrics',
    '--slots',
    '--jinja',
    '--no-webui',
  ];

  // GPU offload — on unified memory "all layers" is the right default.
  const split = String(params.get('gpu.layers') ?? 'Automatic');
  const ngl = split === 'CPU only' ? 0 : split === 'Quarter' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 4)) : split === 'Half' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 2)) : 999;
  args.push('-ngl', String(ngl));

  const threads = num('perf.threads', Math.max(4, (hardware?.cpu?.cores ?? 8) - 2));
  args.push('-t', String(threads));

  const kv = KV_TYPES[String(params.get('ctx.kvMode') ?? 'INT8')] ?? 'q8_0';
  args.push('-ctk', kv, '-ctv', kv);

  const batch = num('perf.batch', 2048);
  args.push('-b', String(batch), '-ub', String(Math.max(128, Math.min(512, Math.round(batch / 4)))));

  if (params.get('perf.flashAttn') === false) args.push('-fa', 'off');
  else args.push('-fa', 'on');

  if (params.get('mem.mmap') === false) args.push('--no-mmap');
  if (params.get('mem.lock') === true) args.push('--mlock');

  const parallel = Math.max(1, num('server.parallel', 1));
  args.push('-np', String(parallel));

  // Sampling defaults applied to every request by the server.
  args.push(
    '--temp', String(params.get('sample.temp') ?? 0.6),
    '--top-p', String(params.get('sample.topP') ?? 0.95),
    '--top-k', String(params.get('sample.topK') ?? 40),
    '--repeat-penalty', String(params.get('sample.repeat') ?? 1.05),
  );
  const seed = Number(params.get('sample.seed') ?? -1);
  if (seed >= 0) args.push('--seed', String(seed));

  if (config.serving?.apiKey) args.push('--api-key', config.serving.apiKey);

  return args;
}

export function describeCommand() {
  return [state.binary, ...state.args].filter(Boolean).join(' ');
}

/* ---------------------------------------------------------------- lifecycle --- */

function log(stream, line) {
  const entry = { at: new Date().toISOString(), stream, line };
  state.logs.push(entry);
  if (state.logs.length > LOG_LIMIT) state.logs.shift();
  hubRef?.send('engine-log', entry);
}

export function logs(limit = 200) {
  return state.logs.slice(-limit);
}

/**
 * The app host starts no inference process of its own: models run on the Strata engine (native control). This used to look for a llama.cpp
 * `llama-server` (STRATA_LLAMA_SERVER, <models>/../engine, /opt/homebrew/bin, /usr/local/bin, PATH) and start it; a packaged app with
 * STRATA_REAL_ENGINE=1 did so for an imported GGUF, also at start-up (0.2.14). start() now refuses, and nothing in this file spawns a process.
 */
export async function start() {
  state.status = 'error';
  state.error = 'Strata runs models on its own engine only.';
  hubRef?.send('engine-status', status());
  throw new Error(state.error);
}

export async function stop() {
  if (!child) {
    state.status = 'stopped';
    state.pid = null;
    stopPolling();
    hubRef?.send('engine-status', status());
    return status();
  }
  state.status = 'stopping';
  hubRef?.send('engine-status', status());
  const proc = child;
  proc.kill('SIGTERM');
  const exited = await Promise.race([
    new Promise((r) => proc.once('exit', () => r(true))),
    new Promise((r) => setTimeout(() => r(false), 4000)),
  ]);
  if (!exited) {
    log('supervisor', 'engine ignored SIGTERM — killing');
    proc.kill('SIGKILL');
  }
  child = null;
  state.status = 'stopped';
  state.pid = null;
  state.startedAt = null;
  stopPolling();
  hubRef?.send('engine-status', status());
  return status();
}

export async function restart(opts) {
  await stop();
  await new Promise((r) => setTimeout(r, 400));
  return start(opts);
}

/* ------------------------------------------------------------------ metrics --- */

function startPolling() {
  stopPolling();
  pollTimer = setInterval(() => void pollMetrics(), 1000);
  pollTimer.unref?.();
  void pollMetrics();
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function parsePrometheus(text) {
  const out = {};
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const match = /^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$/.exec(line.trim());
    if (!match) continue;
    out[match[1]] = Number(match[3]);
  }
  return out;
}

/** Original Strata serves JSON; llama.cpp serves Prometheus. Preserve the original dashboard facts. */
export function parseEngineMetrics(text) {
  if(text.trim().startsWith('{')) {
    const dashboard=JSON.parse(text);
    if(!dashboard.engine||!dashboard.live)throw new Error('Unrecognized engine metrics');
    const live=dashboard.live;
    return {source:'engine',dashboard,promptTokens:live.prompt_tokens??0,predictedTokens:live.generated??0,promptTps:live.prefill_tok_s_mean??0,outputTps:live.tok_s??0,kvUsage:0,kvTokens:0,requestsProcessing:live.running??(['reading','generating'].includes(live.state)?1:0),requestsDeferred:live.queued??0};
  }
  const m=parsePrometheus(text);
  if(!Object.keys(m).some(key=>/prompt_tokens|tokens_predicted|predicted_tokens|kv_cache|requests_processing/.test(key)))throw new Error('Unrecognized engine metrics');
  const p=name=>m[`llamacpp:${name}`]??m[name]??0;
  return {source:'engine',promptTokens:p('prompt_tokens_total'),predictedTokens:p('tokens_predicted_total'),promptTps:p('prompt_tokens_seconds'),outputTps:p('predicted_tokens_seconds'),kvUsage:p('kv_cache_usage_ratio'),kvTokens:p('kv_cache_tokens'),requestsProcessing:p('requests_processing'),requestsDeferred:p('requests_deferred')};
}

export async function pollMetrics() {
  if (state.status !== 'running') return state.metrics;
  try {
    const res = await fetch(`http://127.0.0.1:${state.port}/metrics${state.metrics.dashboard?'?requests=all':''}`, {
      signal: AbortSignal.timeout(1500),
      headers: state.apiKey ? { authorization: `Bearer ${state.apiKey}` } : undefined,
    });
    if (!res.ok) throw new Error(String(res.status));
    state.metrics = { ...parseEngineMetrics(await res.text()), sampledAt: new Date().toISOString() };
    if(state.metrics.dashboard) {
      state.metrics.dashboard.mcp=mcpStatus;
      if(!mcpPending&&Date.now()-mcpAt>10000) {
        mcpPending=true;mcpAt=Date.now();
        void fetch(`http://127.0.0.1:${state.port}/mcp`,{signal:AbortSignal.timeout(1500),headers:state.apiKey?{authorization:`Bearer ${state.apiKey}`}:undefined})
          .then(async response=>{mcpStatus=response.ok?await response.json():null;})
          .catch(()=>{mcpStatus=null;}).finally(()=>{mcpPending=false;});
      }
    }
  } catch {
    state.metrics = { ...state.metrics, source: 'unavailable', sampledAt: new Date().toISOString() };
  }
  return state.metrics;
}

/* ------------------------------------------------------------------- status --- */

export function status() {
  const binary = state.binary ? { path: state.binary, source: state.binarySource } : null;
  return {
    status: state.status,
    running: state.status === 'running',
    binary,
    pid: state.pid,
    port: state.port,
    modelId: state.modelId,
    modelPath: state.modelPath,
    startedAt: state.startedAt,
    error: state.error,
    args: state.args,
    command: state.command,
    metrics: state.metrics,
    logCount: state.logs.length,
  };
}

export function currentPort() {
  return state.port;
}

export function isRunning() {
  return state.status === 'running';
}
