import { plainEngineError } from './update-messages.mjs';

/**
 * Settings > Engine: the engines installed on this computer, switched and deleted through Control (`strata components engines | use | remove`),
 * the same code the Terminal runs. This module only asks, and turns Control's answer into plain words; it never touches an engine folder itself.
 * Nothing here downloads: a switch uses what is installed.
 */

const FAMILIES = ['qwen', 'deepseek'];
/** The engine of a Qwen4 model shape (strata_control/engine_library.py SHAPE_PREFIX, shape_engine.SHAPE_RE). */
const SHAPE_FAMILY = /^shape:[a-z][a-z0-9_]{1,21}[a-z0-9]$/;

/** { status, body } for GET /api/engines. Without Control (a source checkout) the list is not available and the page says so. */
export async function listEngines(control) {
  if (typeof control?.engines !== 'function') return { status: 200, body: { available: false, families: [], total_bytes: 0 } };
  try {
    const listing = await control.engines();
    return { status: 200, body: { available: true, families: listing.families ?? [], total_bytes: listing.total_bytes ?? 0, serving: listing.serving === true } };
  } catch (error) {
    return { status: 500, body: { error: "The installed engines couldn't be read.", details: String(error?.message ?? error) } };
  }
}

/**
 * { status, body } for POST /api/engines/use and /api/engines/remove. `modelsRunning()` says whether the app has a model loaded: a switch then
 * answers at once with the one next step (stop it first); a delete of an engine nothing uses is allowed, Control refuses the one in use.
 */
export async function changeEngine(control, action, body, { modelsRunning = () => false } = {}) {
  if (typeof control?.useEngine !== 'function') return { status: 501, body: { error: 'Engines can be switched and deleted in the installed Strata app.' } };
  const family = body?.family, folder = body?.folder;
  if (!(FAMILIES.includes(family) || SHAPE_FAMILY.test(String(family ?? ''))) || typeof folder !== 'string' || !/^[0-9A-Za-z][\w.+-]*$/.test(folder)) {
    return { status: 400, body: { error: 'Choose an engine from the list.' } };
  }
  if (action === 'use' && modelsRunning()) {
    return { status: 409, body: { error: 'A model is running, so nothing was changed. Unload the model in Models, then try again.' } };
  }
  try {
    const result = action === 'use' ? await control.useEngine({ family, folder }) : await control.removeEngine({ family, folder });
    return { status: 200, body: { ok: true, ...result } };
  } catch (error) {
    const plain = plainEngineError(error, action === 'use' ? 'switch' : 'delete');
    return { status: plain.message.startsWith('A model is running') ? 409 : 400, body: { error: plain.message, details: plain.details } };
  }
}
