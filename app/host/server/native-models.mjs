/** The native model list for GET /v1/models. meta.n_ctx is the context window in tokens, the key serve/server.py uses; it is left out when unknown, never 0 (#351, #3931). */
const tokens = value => Number.isInteger(value) && value > 0 ? value : undefined;

export function nativeModelEntries(instances) {
  return (instances ?? []).filter(i => i.status === 'ready').map(i => {
    const n_ctx = tokens(i.metrics?.engine?.max_context) ?? tokens(i.context);
    // #5893: also as context_length, the field OpenAI-style clients read; a client that needs the window then has no reason to use the
    // engine's private port, whose key is new at every load
    return { id: i.apiModelId, object: 'model', owned_by: 'strata', ...(n_ctx ? { context_length: n_ctx } : {}), meta: { instanceId: i.id, gpus: i.gpus, label: i.label, aliases: i.aliases, ...(n_ctx ? { n_ctx } : {}) } };
  });
}
