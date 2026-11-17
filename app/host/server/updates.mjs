import { OFFLINE_MESSAGE, looksOffline, plainUpdateError } from './update-messages.mjs';

/** Read-only release checks. Installing/rolling back needs a managed installer. */
const RELEASES = 'https://api.github.com/repos/Niko1221/Strata/releases?per_page=30';
const RELEASE_PREFIX = 'https://github.com/Niko1221/Strata/releases/';

function versionParts(value) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:\+[\w.-]+)?$/.exec(value ?? '');
  return match ? match.slice(1).map(Number) : null;
}
function compare(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i] ? 1 : -1;
  return 0;
}

export async function checkUpdates({ appVersion, engineVersion }, fetcher = fetch) {
  let strata;
  try {
    const response = await fetcher(RELEASES, { headers: { accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(response.status === 403 || response.status === 429 ? 'The release service is busy. Try again later.' : 'Could not reach the Strata release feed. Try again.');
    const releases = await response.json();
    if (!Array.isArray(releases)) throw new Error('The release service returned an invalid response.');
    const candidates = releases.filter(r => !r.draft && !r.prerelease && versionParts(r.tag_name));
    candidates.sort((a, b) => compare(versionParts(b.tag_name), versionParts(a.tag_name)));
    const latest = candidates[0];
    const installed = versionParts(appVersion);
    if (!latest) strata = { status: 'no-releases', message: 'No stable Strata release is published in this feed yet.' };
    else if (!installed) strata = { status: 'unavailable', message: 'This installation has no comparable version number.' };
    else {
      const newer = compare(versionParts(latest.tag_name), installed) > 0;
      // Release content is untrusted. Only link back to the known repository.
      const releaseUrl = typeof latest.html_url === 'string' && latest.html_url.startsWith(RELEASE_PREFIX) ? latest.html_url : undefined;
      strata = { status: newer ? 'available' : 'current', latestVersion: latest.tag_name, releaseUrl, message: newer ? `Strata ${latest.tag_name} is available.` : 'Strata is up to date.' };
    }
  } catch (error) {
    strata = { status: 'error', message: looksOffline(error) ? OFFLINE_MESSAGE : error.name === 'TimeoutError' ? 'The release check timed out. Try again.' : error.message ?? 'Could not check for updates.' };
  }
  return {
    checkedAt: new Date().toISOString(),
    strata: { installedVersion: appVersion, ...strata, canInstall: false },
    // The existing binary downloader has no tested-release compatibility feed
    // or trustworthy installed build ID. Upstream's latest is not an update verdict.
    engine: { installedVersion: engineVersion, status: 'unavailable', canInstall: false, message: 'Engine update checks are not available in this build.' },
  };
}

/** The engine row after a version failed its test start on this computer (owner-approved wording, 11-09). */
export const engineFailedBeforeText = (offered, installed) => `Engine ${offered ?? 'update'} didn't start correctly on this computer, so Strata kept engine ${installed ?? 'in use'}. Press Update engine to try again.`;

/**
 * The packaged app's update check: the signed feed, read by Control (`strata update --check`), plus the shell's own verdict
 * on the app. Same result shape as `checkUpdates`, so the Settings screen needs no change. This is the only place the host
 * reaches the feed, and only when the user presses "Check for updates".
 */
export async function checkComponentUpdates({ control, appUpdater = null, appVersion = null }) {
  const checkedAt = new Date().toISOString();
  let report;
  try {
    report = await control.updateCheck({ appVersion });
  } catch (error) {
    const plain = plainUpdateError(error, 'Strata');
    const message = /update server|aren't available|No internet/.test(plain.headline) ? plain.message : `Couldn't check for updates. Try again in a moment.`;
    return { checkedAt, strata: { installedVersion: appVersion ?? control.version ?? null, status: 'error', message, details: plain.details, canInstall: false }, engine: { installedVersion: null, status: 'error', message, details: plain.details, canInstall: false } };
  }
  const mib = bytes => `${Math.round((bytes ?? 0) / 1_000_000)} MB`;      // decimal megabytes, like the Terminal and the release page
  const engine = report.engine ?? {};
  const installedEngine = engine.installed?.version ?? null;
  const offered = engine.offered?.version ?? null;
  const engineView = { installedVersion: installedEngine, latestVersion: offered ?? undefined, canInstall: engine.status === 'available' || engine.status === 'install' };
  if (engine.status === 'available' && engine.installed_locally) Object.assign(engineView, { status: 'available', message: 'A newer engine is already on this computer. Update switches to it; nothing is downloaded.' });
  else if (engine.status === 'available') Object.assign(engineView, { status: 'available', message: `A new engine is available (${mib(engine.download_bytes)}).` });
  else if (engine.status === 'install') Object.assign(engineView, { status: 'available', message: `This PC needs an engine to run models. Press Update engine to download it (${mib(engine.download_bytes)}).` });
  // A version that failed its test start here is held back by Control for a day (status failed-before). It is not 'up to date': the row says what happened and
  // offers the update again (the app's Update engine passes --force, so the press really tries it again). QAGUI-LIN 0213.
  else if (engine.status === 'failed-before') Object.assign(engineView, { status: 'available', canInstall: true, retry: true, message: engineFailedBeforeText(offered, installedEngine), details: engine.note });
  else if (engine.status === 'current' || engine.status === 'skipped') Object.assign(engineView, { status: 'current', message: 'The engine is up to date.', details: engine.note });
  else if (engine.status === 'wrong-make') Object.assign(engineView, { status: 'unavailable', message: engine.note ?? 'Nothing was changed: the update was for another make of graphics card.', details: (engine.reasons ?? []).join('\n') || undefined });
  else if (engine.status === 'not-in-feed') Object.assign(engineView, { status: 'current', message: 'No engine update is listed right now.' });
  else Object.assign(engineView, { status: 'unavailable', message: "There isn't an engine for this PC's graphics card yet.", details: (engine.reasons ?? []).join('\n') || undefined });
  const control_ = report.control ?? {};
  const app = report.app ?? {};
  let appNew = app.status === 'available' ? app.offered : null;
  let appNotes = app.notes ?? '';
  let appCheckFailed = null;   // the shell's own (Tauri) check threw: before this, the Settings check hid it and showed only Control's text
  let appCannot = null;   // browser mode: the app window's updater is not running, so the app itself cannot be updated from here
  if (appUpdater) {
    try {
      const verdict = await appUpdater.check();
      appNew = verdict.available ? verdict.version : null;
      appNotes = verdict.notes ?? appNotes;
    } catch (error) {
      if (error?.code === 'app-update-unavailable' && appUpdater.unavailable) appCannot = appUpdater.unavailable;
      else appCheckFailed = String(error?.message ?? error ?? 'The app update check failed.');   // the app's own check did not work: say so, raw text behind Details
    }
  }
  const controlNew = control_.status === 'available' ? control_.offered : null;
  const strata = { installedVersion: appVersion ?? control_.installed ?? control.version ?? null, canInstall: !!((appNew && !appCannot) || controlNew) };
  const appCheckLine = "Strata couldn't check for a new version of the app itself. Try again later.";
  if (appNew || controlNew) Object.assign(strata, { status: 'available', latestVersion: appNew ?? controlNew, message: `A new version of Strata is available.${appNotes ? ` ${appNotes}` : ''}${appNew && appCannot ? ` ${appCannot}` : ''}${appCheckFailed ? ` ${appCheckLine}` : ''}` });
  else if (appCheckFailed) Object.assign(strata, { status: 'error', message: appCheckLine });
  else Object.assign(strata, { status: 'current', message: 'Strata is up to date.' });
  if (appCheckFailed) strata.details = appCheckFailed;
  return { checkedAt, feed: report.feed, strata, engine: engineView, serving: report.serving ?? [] };
}

/**
 * What the first-run card shows: this PC's graphics card(s), the engine package the signed feed offers for it and how big it is.
 * One feed read, nothing downloaded. `status`: offer (a package fits), none (nothing fits; `reasons` say why), offline (the feed could not be
 * read), installed (an engine is already here), wrong-make (an engine for another make of card is installed).
 */
export async function engineFirstRun({ control, gpus = [] }) {
  const cards = gpus.map(g => ({ name: g.name ?? 'Graphics card', vramGb: g.vramGb ?? null }));
  let report;
  try {
    report = await control.updateCheck({});
  } catch (error) {
    const plain = plainUpdateError(error, 'Strata');
    return { status: looksOffline(error) || /update server|No internet/.test(plain.headline ?? '') ? 'offline' : 'error', gpus: cards, message: plain.message, details: plain.details };
  }
  const engine = report.engine ?? {};
  const hardware = report.hardware ?? {};
  const base = { gpus: cards, driver: hardware.nvidia_driver ?? null, backend: hardware.backend ?? null };
  if (engine.installed) return { ...base, status: 'installed', installedVersion: engine.installed.version ?? null };
  if (engine.status === 'wrong-make') return { ...base, status: 'wrong-make', message: engine.note ?? null, reasons: engine.reasons ?? [] };
  if (engine.offered && (engine.status === 'install' || engine.status === 'available')) {
    // A newer Strata Control in the same feed comes with the engine (an Arc user who has just installed the app needed 0.1.59.1's fix and had to find Settings > Updates; ARCFIX 0213).
    const ctl = report.control ?? {};
    const control = ctl.status === 'available' && ctl.offered ? { version: ctl.offered, bytes: ctl.download_bytes || 0 } : null;
    return { ...base, status: 'offer', offer: { version: engine.offered.version, bytes: engine.download_bytes || engine.offered.size || 0, variant: engine.offered.variant ?? '', backend: engine.offered.backend ?? null }, control };
  }
  return { ...base, status: 'none', message: engine.note ?? null, reasons: engine.reasons ?? [] };
}
