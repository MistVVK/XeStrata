import { useState } from 'react';
import { localeTag, tr } from '../lib/i18n';
import { useNative } from '../lib/native';
import { changeInstalledEngine, engineGroupTitle, engineMake, engineSize, modelsNeedingNewer, useEngineList, type InstalledEngine } from '../lib/engines';
import { ApiError } from '../lib/client';
import { Btn, Dialog, ErrorNotice, Icon } from './ui';

/** Why this computer cannot run an engine: Control gives a code and its values, the page words it (so every language has it). */
function unfitWords(engine: InstalledEngine): string {
  const a = engine.why_args ?? [];
  switch (engine.why_code) {
    case 'os': return tr("This engine was made for a different operating system.");
    case 'gpu_make': return tr("This engine was made for a different kind of graphics card than this computer has.");
    case 'strata': return tr("This engine needs Strata {0} or newer.", [a[0] ?? '']);
    case 'gpu_arch': return tr("This engine doesn't support this computer's graphics card.");
    case 'driver': return tr("This engine needs NVIDIA driver {0} or newer. This computer has {1}.", [a[0] ?? '', a[1] ?? '']);
    case 'files': return tr("Some of this engine's files are missing or changed, so it can't be started.");
    default: return tr("This engine's description is missing or damaged, so it can't be started.");
  }
}

/** Settings > Engine: the engines installed on this computer. Switch to any of them (the newest too) or delete one. Updating is in About. */
export function SettingsEngine() {
  const native = useNative();
  const { list, failed, reload } = useEngineList();
  const [use, setUse] = useState<InstalledEngine | null>(null);
  const [del, setDel] = useState<InstalledEngine | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  const [said, setSaid] = useState('');
  const running = (native?.instances.length ?? 0) > 0;
  const busy = !!native?.jobs.some(j => j.state === 'running') || pending;
  const groups = list?.families ?? [];
  const count = groups.reduce((sum, g) => sum + g.engines.length, 0);

  async function run(action: 'use' | 'remove', engine: InstalledEngine) {
    setPending(true);
    setError(null);
    setSaid('');
    try {
      await changeInstalledEngine(action, engine);
      setSaid(action === 'use' ? tr("Engine {0} is now in use.", [engine.version]) : tr("Engine {0} was deleted.", [engine.version]));
    } catch (e) {
      const body = e instanceof ApiError ? e.body as { details?: string } | undefined : undefined;
      setError({ message: e instanceof Error ? e.message : String(e), detail: body?.details });
    } finally {
      setUse(null);
      setDel(null);
      setPending(false);
      void reload();
    }
  }

  function row(engine: InstalledEngine) {
    const facts = [engineMake(engine.backend), engineSize(engine.bytes, localeTag()), engine.newest ? tr("Newest") : ''].filter(Boolean).join(' · ');
    return <div className="engine-row" key={`${engine.root ?? ''}${engine.folder}`} data-active={engine.active ? 'true' : undefined} data-unfit={engine.fits ? undefined : 'true'}>
      <Icon name="engine" size={20}/>
      <span className="engine-row__text">
        <strong>{engine.version}</strong>
        <small>{facts}</small>
        {engine.fits ? null : <small className="engine-row__why">{unfitWords(engine)}</small>}
      </span>
      {engine.active ? <span className="engine-row__state">{tr("In use")}</span> : <>
        {engine.can_use ? <Btn size="sm" disabled={busy} onClick={() => { setError(null); setSaid(''); setUse(engine); }}>{tr("Use this")}</Btn> : null}
        <Btn size="sm" variant="ghost" disabled={busy || !engine.can_delete} title={engine.blocked_code === 'running' ? tr("A running program is using this engine. Unload your models first.") : undefined} onClick={() => { setError(null); setSaid(''); setDel(engine); }}>{tr("Delete")}</Btn>
      </>}
    </div>;
  }

  const needNewer = use ? modelsNeedingNewer(native, use) : [];
  const useGroup = use ? groups.find(g => g.family === use.family) : undefined;
  return <div className="stack stack--lg native-engine-settings">
    <section className="st-card engine-list">
      <div className="section-toolbar"><h2 className="h3">{tr("Installed engines")}</h2></div>
      <p className="small muted">{tr("Switch to an older engine to test something, then switch back to the newest. Updating to the latest is in About.")}</p>
      {list && !list.available ? <p className="small muted">{tr("Engines can be switched and deleted in the installed Strata app.")}</p> : null}
      {failed ? <p className="small inline-error" role="alert">{tr("The installed engines couldn't be read. Try again in a moment.")}</p> : null}
      {list?.available && count === 0 ? <p className="small muted">{tr("No engine is installed on this computer yet. Update in About downloads the right one.")}</p> : null}
      {groups.map(group => <div key={group.family} className="engine-group">
        {groups.length > 1 ? <h3 className="engine-group__title">{group.title ? tr(group.title) : engineGroupTitle(group, native?.catalog?.families)}</h3> : null}
        {group.engines.map(row)}
      </div>)}
      {count > 0 ? <p className="small muted engine-total">{tr("Engines take {0} on this computer.", [engineSize(list?.total_bytes ?? 0, localeTag())])}</p> : null}
      {count > 0 ? <details className="native-settings-diagnostic engine-details"><summary className="small">{tr("Details")}</summary>{groups.flatMap(g => g.engines).map(e => <p key={`${e.root ?? ''}${e.folder}`} className="small muted">{e.version} · {e.path}{e.build_id ? ` · ${e.build_id}` : ''}</p>)}</details> : null}
      <div role="status" aria-live="polite">{said ? <p className="small">{said}</p> : null}</div>
      {error ? <ErrorNotice message={error.message} detail={error.detail}/> : null}
    </section>

    <Dialog open={!!use} compact title={tr("Use engine {0}?", [use?.version ?? ''])} onClose={() => { if (!pending) setUse(null); }}
      actions={<><Btn disabled={pending} onClick={() => setUse(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={pending || running} onClick={() => use && void run('use', use)}>{pending ? tr("Working…") : tr("Use engine {0}", [use?.version ?? ''])}</Btn></>}>
      {/* a Qwen4 model shape's engine runs only that model (owner 11-09): its own sentence, named like the group's heading */}
      <p>{useGroup?.family.startsWith('shape:') ? tr("{0} runs on engine {1} from its next start. Your models and settings stay as they are. You can switch back at any time.", [engineGroupTitle(useGroup, native?.catalog?.families), use?.version ?? '']) : tr("Models you start from now on run on engine {0}. Your models and settings stay as they are. You can switch back at any time.", [use?.version ?? ''])}</p>
      {needNewer.length ? <p className="small muted">{tr("These models need a newer engine and may not start on it: {0}.", [needNewer.join(', ')])}</p> : null}
      {running ? <p className="small muted" role="status">{tr("A model is running. Unload it in Models first, then switch.")}</p> : null}
    </Dialog>

    <Dialog open={!!del} compact title={tr("Delete engine {0}?", [del?.version ?? ''])} onClose={() => { if (!pending) setDel(null); }}
      actions={<><Btn disabled={pending} onClick={() => setDel(null)}>{tr("Keep engine")}</Btn><Btn variant="danger" disabled={pending} onClick={() => del && void run('remove', del)}>{pending ? tr("Working…") : tr("Delete engine")}</Btn></>}>
      <p>{tr("This frees {0}. Your models and settings are kept.", [engineSize(del?.bytes ?? 0, localeTag())])}</p>
      <p className="small muted">{del?.newest ? tr("The newest engine can be downloaded again with Update in About.") : tr("Strata can only download the newest engine, so a deleted older engine can't be brought back.")}</p>
    </Dialog>
  </div>;
}
