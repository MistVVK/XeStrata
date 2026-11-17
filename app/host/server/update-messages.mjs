/**
 * Plain words for the app. Whatever the updater says (HTTP codes, hash names, exit codes, file names) stays in `details`, which the UI
 * shows behind the existing "Details" disclosure; the main text says what happened and the one next step (`next`). The Terminal keeps
 * the precise messages: this module is only used on the app side.
 */

export const PHASE_TEXT = {
  feed: 'Looking for updates…',
  download: 'Downloading…',
  verify: 'Checking the download…',
  test: 'Making sure it works…',
  switch: 'Finishing up…',
  done: 'Done.',
};

/** The job line while an update waits for Control's update lock (another update or the app's own check holds it). Wording awaits the owner (lane APP-LIN item 4). */
export const UPDATE_WAITING_FOR_LOCK = 'Waiting for the update check to finish…';

/** True when the failure looks like a computer with no internet (no DNS, no route, refused), not a slow or broken server. */
export function looksOffline(error) {
  const text = [error?.message, error?.cause?.message, error?.cause?.code, error?.code, error].filter(x => typeof x === 'string').join(' ');
  return /ENOTFOUND|EAI_AGAIN|ENETUNREACH|ENETDOWN|EHOSTUNREACH|ECONNREFUSED|getaddrinfo|fetch failed|network is unreachable|name or service not known|temporary failure in name resolution|nodename nor servname|No address associated|URLError/i.test(text);
}
export const OFFLINE_MESSAGE = "No internet connection. Try again when you're online.";

/** { message, details, next } for an error thrown by Control's update calls (or the shell bridge). `subject` is "engine" or "Strata".
 * `retry` names the button of a screen that has no Settings page yet (the first-run card's Continue): the next step then says that button. */
export function plainUpdateError(error, subject = 'engine', retry = undefined) {
  const raw = String(error?.message ?? error ?? '');
  const code = error?.code ?? '';
  const exit = error?.exitCode ?? null;
  const button = `Update ${subject === 'engine' ? 'engine' : 'Strata'}`;
  const keep = 'Your current version still works.';
  let message = `The update didn't finish and nothing was changed. ${keep}`;
  let next = `Press ${button} in Settings to try again.`;
  // Exit 7 is also the update lock (another update or a check holds it): told by its text, and checked first (QAGUI 0213 finding 3).
  if (code === 'update_busy' || /another update is already running/i.test(raw)) {
    message = 'An update is already running.';
    next = `Wait a moment, then press ${button} in Settings.`;
  } else if (code === 'engine_serving' || exit === 7 || /is serving/i.test(raw)) {
    message = 'A model is running, so nothing was changed.';
    next = `Unload the model in Models, then press ${button} in Settings.`;
  } else if (/test start|did not start correctly|did not answer a one-token|did not become ready/i.test(raw)) {
    message = `The new ${subject} didn't start correctly, so nothing was changed. ${keep}`;
  } else if (looksOffline(error) || looksOffline(raw)) {
    message = 'No internet connection.';
    next = "Try again when you're online.";
  } else if (/could not be read|interrupted|timed out|timeout|ECONN|ENOTFOUND|fetch failed|could not reach|server answered|not answer within/i.test(raw)) {
    message = "Couldn't reach the update server.";
    next = `Check your internet connection, then press ${button} in Settings.`;
  } else if (/free space|not enough|no space left|disk is full/i.test(raw)) {
    message = 'There is not enough free disk space for the update.';
    next = `Free some disk space, then press ${button} in Settings.`;
  } else if (/could not be moved into place|could not be unpacked|not accepted after the switch/i.test(raw)) {
    // The files were downloaded and checked; Windows or a virus scanner would not let Strata put them in place (#4853). Not a download problem.
    const long = /longer than Windows allows|path.*too long/i.test(raw);
    message = long
      ? "The engine's folder is too deep for Windows, so it could not be put in place."
      : 'The engine was downloaded and checked, but Windows or a security program would not let Strata put it in place.';
    next = long
      ? `Open Details, then follow the folder advice there, and press ${button} in Settings.`
      : `Wait a minute, then press ${button} in Settings. If it happens again, pause your virus scanner's live scanning for a few minutes while it runs.`;
  } else if (/no engine for this PC|has no engine|no package/i.test(raw)) {
    message = "There isn't an engine for this PC's graphics card yet.";
    next = 'Check for updates again later.';
  } else if (/trusts no update key/i.test(raw)) {
    message = "Updates aren't available in this version of Strata yet.";
    next = 'They will be after the next version.';
  } else if (/signature|SHA-256|does not match|not trusted|expired|older than|replayed|integrity|refused/i.test(raw) || exit === 5) {
    message = `The update didn't pass its safety check, so it was not installed. ${keep}`;
    next = 'Try again later.';
  }
  if (retry) next = next.split(`${button} in Settings`).join(retry).replace(/in Settings/g, '').replace(/ {2,}/g, ' ').replace(/ ([.,])/g, '$1');
  return { message: `${message} ${next}`.trim(), headline: message, next, details: raw || undefined };
}

/** Plain words for a switch or a delete in Settings > Engine: { message, details }. Control's own sentence for "the engine in use" is already plain and is kept. */
export function plainEngineError(error, action = 'switch') {
  const raw = String(error?.message ?? error ?? '');
  const exit = error?.exitCode ?? null;
  let message = action === 'delete' ? "The engine wasn't deleted. Nothing else was changed." : "The engine wasn't switched. Nothing was changed.";
  if (/another update is already running/i.test(raw)) message = 'An update is running. Wait for it to finish, then try again.';
  else if (exit === 7 || /is serving|model is running/i.test(raw)) message = 'A model is running, so nothing was changed. Unload the model in Models, then try again.';
  else if (/engine in use/i.test(raw)) message = 'This is the engine in use. Switch to another one first, then delete it.';
  else if (/running program is using|something is using it/i.test(raw)) message = 'A program is still using this engine, so it was not deleted. Unload your models and try again.';
  else if (/not installed|already been deleted/i.test(raw)) message = 'That engine is already gone. The list has been refreshed.';
  else if (/another update is already running/i.test(raw)) message = 'An update is running. Wait for it to finish, then try again.';
  else if (exit === 5 || /can no longer be used|does not fit|cannot run/i.test(raw)) message = "This engine can't run on this computer, so nothing was changed.";
  return { message, details: raw || undefined };
}

/** An Error carrying the plain message and, behind it, the technical reason. */
export function plainError(error, subject, retry = undefined) {
  const { message, details } = plainUpdateError(error, subject, retry);
  return Object.assign(new Error(message), { details, code: error?.code });
}
