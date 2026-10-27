/** The native model list for GET /v1/models. meta.n_ctx is the context window in tokens, the key serve/server.py uses; it is left out when unknown, never 0 (#351, #3931). */
const tokens = value => Number.isInteger(value) && value > 0 ? value : undefined;

export function nativeModelEntries(instances) {
  return (instances ?? []).filter(i => i.status === 'ready').map(i => {
    const n_ctx = tokens(i.metrics?.engine?.max_context) ?? tokens(i.context);
    return { id: i.apiModelId, object: 'model', owned_by: 'strata', meta: { instanceId: i.id, gpus: i.gpus, label: i.label, aliases: i.aliases, ...(n_ctx ? { n_ctx } : {}) } };
  });
}
