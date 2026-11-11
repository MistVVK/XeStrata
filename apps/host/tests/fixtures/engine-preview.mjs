/**
 * UI preview only (STRATA_UI_PREVIEW=1): several installed engines for Settings > Engine and an engine update for About, kept in memory.
 * The demo host refuses mutations and simulates the hardware, so this answers the same routes the real host has (/api/engines, /api/engines/use,
 * /api/engines/remove, /api/updates/check, the update-engine job) with believable behaviour: switching changes what is in use, the engine in use
 * cannot be deleted, a switch while a model is loaded answers with the same plain sentence, an update makes the new engine the one in use.
 * Nothing here touches a disk or a process.
 */
const GB = 1e9;

function engine(family, version, bytes, extra = {}) {
  const backend = extra.backend ?? 'cuda';
  const variant = extra.variant ?? (backend === 'cuda' ? 'cuda13' : '');
  const folder = `${version}-windows-x86_64-${backend}${variant ? `-${variant}` : ''}`;
  return { family, folder, version, backend, platform: 'windows-x86_64', variant, bytes, signed: true, fits: true, why_not: null, why_code: null, why_args: [],
    active: false, newest: false, pinned: false, running: false, can_use: true, can_delete: true, delete_blocked: null, blocked_code: null,
    build_id: `preview-${version}`, path: `C:\\Users\\you\\AppData\\Local\\Strata\\components\\engine\\${folder}`, ...extra };
}

export function createEnginePreview(native) {
  const qwen = [engine('qwen', '0.1.55', 1.02 * GB), engine('qwen', '0.1.54', 0.98 * GB), engine('qwen', '0.1.50', 0.91 * GB),
    engine('qwen', '0.1.49', 0.88 * GB, { backend: 'hip', fits: false, why_not: 'x', why_code: 'gpu_make' })];
  const groups = () => [{ family: 'qwen', title: 'Qwen and other models', engines: qwen }];
  const latest = '0.1.56';
  let checked = false;
  const jobs = [];

  function settle(list, activeFolder) {
    const fitting = list.filter(e => e.fits);
    const newest = fitting.reduce((a, b) => (a && a.version.localeCompare(b.version, undefined, { numeric: true }) >= 0 ? a : b), null);
    for (const e of list) {
      e.active = e.folder === activeFolder;
      e.newest = e === newest;
      e.can_use = e.fits && !e.active;
      e.can_delete = !e.active;
      e.blocked_code = e.active ? 'active' : null;
      e.delete_blocked = e.active ? 'This is the engine in use. Switch to another one first.' : null;
    }
  }
  const activeOf = list => list.find(e => e.active)?.folder;
  settle(qwen, qwen[0].folder);
  const syncVersion = () => { native.installed = { ...(native.installed ?? {}), engine: { version: qwen.find(e => e.active)?.version ?? null } }; };
  syncVersion();
  native.capabilities = { ...(native.capabilities ?? {}), componentUpdates: true };

  const total = () => qwen.reduce((sum, e) => sum + e.bytes, 0);
  const loaded = () => (native.instances ?? []).some(i => i.status === 'ready' || i.status === 'loading');

  return {
    /** True when the request was answered here. */
    async handle(path, method, body, json) {
      if (path === '/api/engines' && method === 'GET') return json({ available: true, families: groups(), total_bytes: total(), serving: loaded() }), true;
      if ((path === '/api/engines/use' || path === '/api/engines/remove') && method === 'POST') {
        const list = qwen;
        const row = list.find(e => e.folder === body.folder);
        if (!row) return json({ error: 'That engine is already gone. The list has been refreshed.' }, 400), true;
        await new Promise(r => setTimeout(r, 700));
        if (path.endsWith('/use')) {
          if (loaded()) return json({ error: 'A model is running, so nothing was changed. Unload the model in Models, then try again.' }, 409), true;
          if (!row.fits) return json({ error: "This engine can't run on this computer, so nothing was changed." }, 400), true;
          settle(list, row.folder);
          if (body.family === 'qwen') syncVersion();
          return json({ ok: true, status: 'switched', to: row.version }), true;
        }
        if (row.active) return json({ error: 'This is the engine in use. Switch to another one first, then delete it.' }, 400), true;
        list.splice(list.indexOf(row), 1);
        settle(list, activeOf(list));
        return json({ ok: true, status: 'deleted', version: row.version, freed_bytes: row.bytes }), true;
      }
      // The demo host refuses every change; unloading the demo model is allowed so the switch can be tried after the "a model is running" answer.
      const unload = /^\/api\/native\/instances\/([^/]+)\/unload$/.exec(path);
      if (unload && method === 'POST') {
        native.instances = (native.instances ?? []).filter(i => i.id !== unload[1]);
        if (native.defaultId === unload[1]) native.defaultId = native.instances[0]?.id ?? null;
        return json(native), true;
      }
      if (path === '/api/updates/check' && method === 'GET') {
        checked = true;
        const inUse = qwen.find(e => e.active)?.version ?? null;
        const newer = inUse && inUse.localeCompare(latest, undefined, { numeric: true }) < 0 && !qwen.some(e => e.version === latest);
        return json({
          checkedAt: new Date().toISOString(),
          strata: { installedVersion: '0.2.9', status: 'current', message: 'Strata is up to date.', canInstall: false },
          engine: newer ? { installedVersion: inUse, latestVersion: latest, status: 'available', message: 'A new engine is available (1,030 MB).', canInstall: true }
            : { installedVersion: inUse, status: 'current', message: 'The engine is up to date.', canInstall: false },
        }), true;
      }
      if (path === '/api/native/operations' && method === 'POST' && body.kind === 'update-engine') {
        if (loaded()) return json({ error: 'Unload native models before changing engine files.' }, 400), true;
        const job = { id: `preview-update-${jobs.length}`, kind: 'update-engine', state: 'running', startedAt: new Date().toISOString(), summary: 'Downloading: 12%', error: null,
          result: { install: { step: 'Downloading: 12%', download: { file: `engine ${latest}`, percent: 12 } } } };
        jobs.push(job);
        native.jobs.push(job);
        const step = (delay, fn) => setTimeout(fn, delay).unref();
        step(1500, () => { job.summary = 'Downloading: 64%'; job.result.install.download.percent = 64; });
        step(3000, () => { job.summary = 'Making sure it works…'; job.result = null; });
        step(4500, () => {
          for (const e of qwen) e.active = false;
          qwen.unshift(engine('qwen', latest, 1.03 * GB));
          settle(qwen, qwen[0].folder);
          syncVersion();
          Object.assign(job, { state: 'done', summary: 'The engine is updated and ready.', result: { results: [{ component: 'engine', status: 'updated' }] } });
        });
        return json(job), true;
      }
      return false;
    },
    get checked() { return checked; },
  };
}
