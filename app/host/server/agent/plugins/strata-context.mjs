/**
 * Strata's context plugin for the Harness (plain ESM, loaded by absolute path from the profile, like strata-gate and strata-models).
 *
 * What it fixes (2026-10-22 hands-on test, Gyro-S at 64K):
 *   1. Compaction decided from the Harness's own price of the conversation, 4 characters a token. Code, logs and JSON are about twice as
 *      dense, so a chat the engine had read as 59,048 tokens was priced at 31,794 and compaction started far too late. Here the token meter
 *      is calibrated by what the model server reports for each answer (its real prompt size): the measured size is exact, and what was
 *      added since is priced with the density that measurement showed. Automatic compaction and the manual /compact both measure through
 *      the token meter, so both read the same number as the context meter in the app.
 *   2. The summarising request itself could be bigger than the window ("/compact" sent a 66,507-token prompt to a 65,536 window). The
 *      summary is now asked for in pieces that fit: a message bigger than a piece is cut for the request (the conversation keeps it until
 *      the summary replaces it), the pieces are summarised one after the other, each building on the summary before it, and a piece the
 *      server still refuses is halved and tried again.
 *
 * How. It wraps `tokenMeter.measure` and `llm.stream` in place, once each (the same way strata-models wraps the subagent tools). Only a
 * request the Harness marks `purpose: 'compaction'` is touched in `llm.stream`. Nothing in the Harness is rewritten: a Harness update
 * that renames either leaves compaction as it was, and the test that runs the real Harness fails.
 */
import { replyCap } from '../reply-cut.mjs';
import { densityOf, estimateMessage, estimateTools, isContextOverflow, MAX_DENSITY, pressureNow, promptTokensOf, splitMessages, summaryBudget, usageOfEvent, WINDOW_SLACK, withSummaryLanguage } from '../context-budget.mjs';

export const name = 'strata-context';

/** The routes the host writes for models on this computer (profile.mjs): `strata-local`, `strata-local-<id>`, and their `~s<budget>` copies. */
const isLocalRoute = provider => typeof provider === 'string' && provider.startsWith('strata-local');
export const inject = ['tokenMeter', 'llm'];

/** Real prompt measurements per session: { real, estimated } for the last answer the model server reported. */
const samples = new WeakMap();
const bySessionId = new Map();
const wrapped = new WeakSet();
/** Densities that made a summary request fail on this server, per session id (the next request starts from a more careful one). */
const careful = new Map();

export function apply(ctx) {
  const meter = ctx.tokenMeter;
  const llm = ctx.llm;
  const log = (level, text) => { try { ctx.logger?.[level]?.(`strata-context: ${text}`); } catch { /* logging never blocks */ } };

  // 1. The measured size of the prompt, from the model server's own report for each answer.
  if (meter && typeof meter.measure === 'function' && !wrapped.has(meter)) {
    wrapped.add(meter);
    const original = meter.measure;
    meter.measure = function measure(session, requestHeader) {
      const m = original.call(this, session, requestHeader);
      const sample = samples.get(session);
      if (!sample) return m;
      const now = m.surfaceTokens + estimateTools(requestHeader ?? session.requestHeader?.());
      const total = Math.max(m.totalTokens, pressureNow({ sample, now }));
      if (total <= m.totalTokens) return m;
      const density = densityOf(sample.real, sample.estimated);
      return { ...m, totalTokens: total, nodes: m.nodes.map(n => ({ ...n, tokens: Math.ceil(n.tokens * density) })) };
    };
    ctx.on('session/event', (session, event) => {
      if (event.type !== 'assistant/message') return;
      const real = promptTokensOf(usageOfEvent(event));
      if (!(real > 0)) return;
      try {
        // The request that was just answered saw the conversation without this answer: its price is the surface before the answer's node.
        const m = original.call(meter, session);
        const heuristic = m.nodes.reduce((n, x) => n + (x.seq === event.seq ? 0 : x.heuristicTokens), 0);
        const estimated = heuristic + estimateTools(session.requestHeader?.());
        if (!(estimated > 0)) return;
        const sample = { real, estimated };
        samples.set(session, sample);
        if (session.id) bySessionId.set(String(session.id), sample);
      } catch (e) { log('warn', `could not record a measurement: ${e.message}`); }
    });
  }

  // 2. A summary request that fits the window.
  if (llm && typeof llm.stream === 'function' && !wrapped.has(llm)) {
    wrapped.add(llm);
    const original = llm.stream;
    llm.stream = function stream(options) {
      if (options?.purpose !== 'compaction') return original.call(this, options);
      return summaryStream.call(this, options, original);
    };
  }

  // 3. A reply that has room to finish (AGENT-LOST, 11-01). The output budget in the profile is what the Harness's compaction sets aside for a
  // reply (30 percent of the window; it reads it from the request header), and the same number was also the ceiling of every request: a thinking
  // model that wrote a 15K-token file after 9K tokens of thinking was cut inside the tool call at 19,660, the unfinished call was dropped and the
  // turn ended with no file and no word. The header (and so compaction) stays as it is; the request that goes to the model server may produce up
  // to half the window, at most 64K tokens (replyCap), and pi-ai still clamps it to what the window has left after the prompt. The agent loop
  // dispatches through the adapter's `streamWithSnapshot`, after the loop has checked the request against its header, so that is where the ceiling
  // is raised, once per adapter, the first time a local route is prepared. A Harness that renames it leaves the ceiling as it was (the test that
  // runs the real Harness fails).
  if (llm && typeof llm.prepareCall === 'function' && !wrapped.has(llm.prepareCall)) {
    const original = llm.prepareCall;
    const raised = new WeakSet();
    const prepare = async function prepareCall(config, signal, configure) {
      try { raiseReplyCeiling(config?.provider, raised); } catch (e) { log('warn', `could not raise the reply ceiling: ${e.message}`); }
      return original.call(this, config, signal, configure);
    };
    wrapped.add(prepare);
    llm.prepareCall = prepare;
  }
  function raiseReplyCeiling(provider, raised) {
    if (!isLocalRoute(provider)) return;
    const adapter = llm.adapters?.get?.(provider)?.adapter;
    if (!adapter || raised.has(adapter) || typeof adapter.streamWithSnapshot !== 'function' || typeof adapter.modelOf !== 'function') return;
    raised.add(adapter);
    const stream = adapter.streamWithSnapshot;
    adapter.streamWithSnapshot = function streamWithSnapshot(options, snapshot) {
      let next = options;
      try {
        if (isLocalRoute(options?.provider)) {
          const model = this.modelOf(snapshot, options.provider, options.model);
          const asked = options.maxTokens ?? model.maxTokens ?? 0;
          // Only the agent's own reply budget is raised: a summary or a title asks for far less and keeps what it asked.
          if (model.contextWindow > 0 && asked >= Math.floor(model.contextWindow * 0.25)) {
            const cap = replyCap(model.contextWindow, asked);
            if (cap > asked) next = { ...options, maxTokens: cap };
          }
        }
      } catch { next = options; }
      return stream.call(this, next, snapshot);
    };
  }

  // The host asks, with /strata-context, how much of the window the chat holds right now: the same measured number compaction decides on
  // (JSON in the command's text: { tokens, window }). Before a message goes out, and after /compact, so the person is told true figures.
  ctx.inject(['commands'], (commandCtx) => {
    commandCtx.commands.register({
      definitionId: 'strata-context',
      name: 'strata-context',
      description: 'Reports how many tokens the conversation holds now and the size of the model window',
      handler: async (invocation) => {
        try {
          const session = invocation.agent.session;
          const tokens = meter.measure(session).totalTokens;
          const config = session.requestHeader?.()?.config;
          let window = 0;
          if (config) window = (await llm.resolveModelInfo(config.provider, config.model))?.context?.contextWindow ?? 0;
          return { kind: 'success', text: JSON.stringify({ tokens, window }) };
        } catch (e) { return { kind: 'error', text: `strata-context: ${e.message}` }; }
      },
    });
  });

  /** A finished stream: its chunks, the text it wrote and why it ended. */
  async function drain(iterable) {
    let text = '', finish = null;
    const chunks = [];
    for await (const chunk of iterable) {
      chunks.push(chunk);
      if (chunk.type === 'text-delta') text += chunk.text ?? '';
      else if (chunk.type === 'finish') finish = chunk.reason;
    }
    return { text, finish, chunks };
  }
  const refusedForSize = finish => finish?.kind === 'error' && isContextOverflow(finish.failure ?? finish);

  async function* summaryStream(options, original) {
    options = { ...options, messages: withSummaryLanguage(options.messages) };   // #4750: the summary follows the conversation's language
    let window = 0;
    try { window = (await llm.resolveModelInfo(options.provider, options.model, options.signal))?.context?.contextWindow ?? 0; } catch { window = 0; }
    const messages = options.messages ?? [];
    const key = String(options.sessionId ?? '');
    const summaryTokens = options.maxTokens ?? 4096;
    if (!(window > 0) || messages.length < 2) { yield* original.call(this, options); return; }
    const own = bySessionId.get(key);
    let density = Math.max(careful.get(key) ?? 1, own ? densityOf(own.real, own.estimated) : 1);
    const price = list => list.reduce((n, m) => n + estimateMessage(m), 0) * density;
    const limit = window - summaryTokens - WINDOW_SLACK;
    // The whole request fits by this price: ask as the Harness would (the model server's cache of the conversation is reused).
    if (price(messages) + estimateTools({ tools: options.tools }) * density <= limit) {
      const first = await drain(original.call(this, options));
      if (!refusedForSize(first.finish)) { for (const chunk of first.chunks) yield chunk; return; }
      density = Math.min(MAX_DENSITY, density * 1.6);
      careful.set(key, density);
      log('warn', 'the summary request did not fit the window; asking in pieces');
    }
    // In pieces. The instruction is the last message; the system prompt, when there is one, the first.
    const instruction = messages[messages.length - 1];
    const system = messages[0]?.role === 'system' ? messages[0] : null;
    const body = messages.slice(system ? 1 : 0, -1);
    let tools = options.tools;
    if (tools && estimateTools({ tools }) * density > window / 3) tools = undefined;   // a long tool list only describes tools: the summary does not need it
    const fixed = (system ? estimateMessage(system) : 0) * density + estimateTools({ tools }) * density + estimateMessage(instruction) * density;
    let budget = summaryBudget({ window, summaryTokens: summaryTokens * 2, systemTokens: fixed, instructionTokens: 0 });
    for (let attempt = 0; ; attempt++) {
      const runs = splitMessages(body, Math.max(1024, budget), density);
      log('info', `summarising ${body.length} messages in ${runs.length} piece(s) for a ${window}-token window (density ${density.toFixed(2)}, about ${Math.round(budget)} tokens a piece)`);
      let failed = null, summary = '';
      for (let i = 0; i < runs.length && !failed; i++) {
        options.signal?.throwIfAborted?.();
        const carried = summary ? [{ role: 'user', content: [{ type: 'text', text: `<compacted-summary>\n${summary}\n</compacted-summary>` }] }] : [];
        const { tools: _drop, ...rest } = options;
        const piece = { ...rest, ...(tools ? { tools } : {}), messages: [...(system ? [system] : []), ...carried, ...runs[i], instruction] };
        const { text, finish, chunks } = await drain(original.call(this, piece));
        if (refusedForSize(finish)) { failed = finish; break; }
        // The last piece's answer is the summary the Harness reads; a piece that ended in another error ends the request the same way.
        if (i === runs.length - 1 || finish?.kind === 'error' || finish?.kind === 'aborted') { for (const chunk of chunks) yield chunk; return; }
        summary = text;
      }
      if (!failed || attempt >= 3) throw Object.assign(new Error(failed?.failure?.message ?? 'the summary request does not fit the window'), { code: 'CONTEXT_WINDOW_EXCEEDED' });
      density = Math.min(MAX_DENSITY, density * 1.6);
      careful.set(key, density);
      budget = Math.max(1024, budget / 2);
      log('warn', `a piece did not fit; asking again with pieces of about ${Math.round(budget)} tokens`);
    }
  }
}
