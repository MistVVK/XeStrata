import { AsyncLocalStorage } from 'node:async_hooks';

export const name = 'strata-compaction';
export const inject = ['compaction', 'tokenMeter'];

const wrapped = new WeakSet();

export function apply(ctx) {
  // This plugin lives beside the isolated compaction service. Its pre-step
  // listener calls the original Cordis instance, including its internal methods.
  const compaction = ctx.compaction;
  const log = (level, text) => { try { ctx.logger?.[level]?.(text); } catch { /* logging never blocks */ } };
  guardPressureCompaction(compaction[Symbol.for('cordis.original')] ?? compaction, ctx.tokenMeter, log);
}

// A checkpoint includes framing and instructions. Tiny prefixes are cheaper
// to keep verbatim, especially when a summary would displace a cached prompt.
const MIN_PRESSURE_SOURCE_TOKENS = 128;
const SKIP_PRESSURE_COMPACTION = Symbol('skip pressure compaction');

export function guardPressureCompaction(compaction, meter, log) {
  if (!compaction || !meter || typeof compaction.compactIfNeeded !== 'function'
      || typeof compaction.compactRegion !== 'function' || wrapped.has(compaction)) return;
  wrapped.add(compaction);
  const compactIfNeeded = compaction.compactIfNeeded;
  const compactRegion = compaction.compactRegion;
  const pressure = new AsyncLocalStorage();
  const failed = new WeakMap();

  compaction.compactIfNeeded = async function (agent, trigger, signal) {
    if (trigger !== 'pressure') return compactIfNeeded.call(this, agent, trigger, signal);
    try {
      return await pressure.run(agent, () => compactIfNeeded.call(this, agent, trigger, signal));
    } catch (error) {
      if (error === SKIP_PRESSURE_COMPACTION) return null;
      throw error;
    }
  };

  compaction.compactRegion = async function (start, end, agent, signal) {
    if (pressure.getStore() !== agent) return compactRegion.call(this, start, end, agent, signal);
    signal?.throwIfAborted();
    const session = agent.session;
    const { nodes } = meter.measure(session);
    const first = nodes.findIndex(node => node.seq === start);
    const last = nodes.findIndex(node => node.seq === end);
    // The Harness still validates stale or unbalanced ranges itself.
    if (first < 0 || last < first) return compactRegion.call(this, start, end, agent, signal);
    const range = nodes.slice(first, last + 1);
    const tokens = range.reduce((total, node) => total + node.tokens, 0);
    const config = session.requestHeader?.()?.config;
    const key = JSON.stringify([config?.provider, config?.model, range.map(node => node.seq)]);
    if (tokens < MIN_PRESSURE_SOURCE_TOKENS || failed.get(session) === key) throw SKIP_PRESSURE_COMPACTION;
    try {
      const result = await compactRegion.call(this, start, end, agent, signal);
      failed.delete(session);
      return result;
    } catch (error) {
      // A failed model call may be transient. Remember only a checkpoint that
      // demonstrably did not fit or shrink this exact source range.
      if (signal?.aborted || !(error?.code === 'MAX_TOKENS'
          || /^summary is not smaller than the shadowed content \(/.test(error?.message ?? ''))) throw error;
      failed.set(session, key);
      log('info', 'skipping further pressure compaction of an unchanged source range');
      throw SKIP_PRESSURE_COMPACTION;
    }
  };
}
