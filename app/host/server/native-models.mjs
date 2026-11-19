/** The native model list for GET /v1/models. meta.n_ctx is the context window in tokens, the key serve/server.py uses; it is left out when unknown, never 0 (#351, #3931).
 * #6180: the rows mirror the rest of the engine's /v1/models shape too — a status object, architecture modalities, and one entry per alias
 * (an alias row repeats its instance's entry with the id swapped and alias_of naming the canonical id), so a client that auto-detects
 * through the app sees the same thing as through the engine's private port. */
const tokens = value => Number.isInteger(value) && value > 0 ? value : undefined;

export function nativeModelEntries(instances) {
  const rows = [];
  for (const i of instances ?? []) {
    if (i.status !== 'ready') continue;
    const n_ctx = tokens(i.metrics?.engine?.max_context) ?? tokens(i.context);
    // #5893: also as context_length, the field OpenAI-style clients read; a client that needs the window then has no reason to use the
    // engine's private port, whose key is new at every load
    const entry = {
      id: i.apiModelId,
      object: 'model',
      owned_by: 'strata',
      status: { value: 'loaded' },
      architecture: { input_modalities: i.vision ? ['text', 'image'] : ['text'], output_modalities: ['text'] },
      ...(n_ctx ? { context_length: n_ctx } : {}),
      meta: { instanceId: i.id, gpus: i.gpus, label: i.label, aliases: i.aliases, ...(n_ctx ? { n_ctx } : {}) },
    };
    rows.push(entry);
    // Alias rows mirror the engine: full entries that name their canonical id. The app's default alias is often the model tag itself,
    // so an alias equal to the canonical id is skipped rather than duplicating the row.
    for (const alias of i.aliases ?? []) {
      if (typeof alias !== 'string' || !alias || alias === i.apiModelId) continue;
      rows.push({ ...entry, id: alias, alias_of: i.apiModelId });
    }
  }
  return rows;
}
