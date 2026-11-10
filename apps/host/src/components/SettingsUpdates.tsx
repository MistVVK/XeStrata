import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Icon } from './ui';
import { useStore } from '../lib/store';
import type { SoftwareUpdate, UpdateCheckResult } from '../lib/types';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { NativeProgress } from './NativeModels';
import { EngineUpdateDialog } from './EngineUpdateDialog';
import { currentUpdateFailure, jobErrorText, updateFinishedAfter } from '../lib/native-job-diagnostics';
import { appVersionLabel } from '../lib/app-version';
import { fetchUpdateCheck, publishUpdateCheck, useInstalledVersions, useLastUpdateCheck } from '../lib/update-check';
import { shownEngineVersion } from '../lib/engine-version';
/** A check from an earlier day shows its date too, because the result is kept between visits. */
function checkedLabel(at: string) {
    const when = new Date(at), sameDay = when.toDateString() === new Date().toDateString();
    return sameDay ? when.toLocaleTimeString(localeTag(), { hour: '2-digit', minute: '2-digit' }) : when.toLocaleString(localeTag(), { dateStyle: 'medium', timeStyle: 'short' });
}
/** Installed versions are real host values. Installation needs a host updater. */
export function SettingsUpdates() {
    const native = useNative();
    const [confirm, setConfirm] = useState('');
    const appVersion = appVersionLabel(useStore(s => s.setup?.app.version), native?.version);
    const engineVersion = useStore(s => s.engine?.engine.version);
    const [checking, setChecking] = useState(false);
    // Opening Settings shows the last check kept on this computer. The network is touched only when "Check for updates" is pressed.
    const versions = useInstalledVersions();
    const last = useLastUpdateCheck(versions);
    const [fresh, setFresh] = useState<UpdateCheckResult | null>(null);
    const [error, setError] = useState('');
    const request = useRef<AbortController | null>(null);
    useEffect(() => () => request.current?.abort(), []);
    const active = native?.jobs.find(j => j.state === 'running');
    const result = checking || error ? null : (fresh ?? last);
    // An update that finished after the check shown makes it out of date: check again, once per update (lane 3 finding 6, 11-09).
    const finished = updateFinishedAfter(native?.jobs ?? [], result?.checkedAt), rechecked = useRef('');
    useEffect(() => { if (finished && !checking && rechecked.current !== finished) { rechecked.current = finished; void check(); } }, [finished, checking]);
    // Update engine with a model loaded: its own dialog (warns naming the model, stops it with the person's yes, offers to start it again).
    const withModel = (native?.instances.length ?? 0) > 0 && !!native?.capabilities.componentUpdates;
    async function operate(kind: string) {
        setError('');
        try {
            await nativeAction('operations', { kind });
            setConfirm('');
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    async function check() {
        request.current?.abort();
        const controller = new AbortController();
        request.current = controller;
        setChecking(true);
        setError('');
        setFresh(null);
        try {
            const data = await fetchUpdateCheck(controller.signal);
            if (!controller.signal.aborted) {
                setFresh(data);
                publishUpdateCheck(data, versions);
            }
        }
        catch (e) {
            if (!controller.signal.aborted)
                setError(e instanceof Error ? e.message : 'Could not check for updates.');
        }
        finally {
            if (!controller.signal.aborted)
                setChecking(false);
        }
    }
    function versionRow(name: string, version: string | undefined, update?: SoftwareUpdate) {
        const labels: Record<SoftwareUpdate['status'], string> = { available: 'Update available', current: 'Up to date', unavailable: 'Check unavailable', error: 'Check failed', 'no-releases': 'No release yet' };
        return <div className="settings-version" data-update={update?.status}>
      <Icon name={name === 'Strata' ? 'layers' : 'engine'} size={20}/>
      <span><strong>{tr(name)}</strong><small>{tr(version ?? 'Version unavailable')}</small>{tr(!checking && update ? <small className="settings-version__message">{tr(update.message)}</small> : null)}{tr(!checking && name === 'Strata' && update?.details ? <details className="native-settings-diagnostic"><summary className="small">{tr("Details")}</summary><p className="small muted">{update.details}</p></details> : null)}</span>
      {tr(!checking && update?.status === 'available' ? (native?.available ? <Btn size="sm" variant="primary" disabled={!!active || (name === 'Engine' && !native.installed?.models.length && !native.capabilities.componentUpdates)} onClick={() => setConfirm(name === 'Strata' ? 'update-strata' : 'update-engine')}>{tr("Update available")}</Btn> : update.releaseUrl ? <a className="st-btn st-btn--secondary st-btn--sm" href={update.releaseUrl} target="_blank" rel="noreferrer" title={tr("Open the release page to update this installation")}>{tr("Update available")}<Icon name="download" size={15}/></a> : <span className="settings-version__label">{tr(labels[update.status])}</span>) : <span className="settings-version__label">{tr(checking ? 'Checking…' : update ? labels[update.status] : version === 'Not installed' ? 'Not installed' : 'Installed')}</span>)}
    </div>;
    }
    return <>
    <div className="settings-versions" aria-busy={checking}>
      {tr(versionRow('Strata', appVersion, result?.strata))}
      {tr(versionRow('Engine', native?.available ? (native.installed?.engine?.version ?? 'Not installed') : shownEngineVersion(native, engineVersion), result?.engine))}
      <div className="settings-update-actions"><Btn size="sm" icon="refresh" disabled={checking} className={checking ? 'settings-update-checking' : undefined} onClick={() => void check()}>{tr(checking ? 'Checking…' : error || result?.strata.status === 'error' ? 'Try again' : result ? 'Check again' : 'Check for updates')}</Btn></div>
      <div className="settings-update-inline" role="status" aria-live="polite">{tr(checking ? 'Checking for new versions…' : error ? <span className="inline-error">{tr(error)}</span> : result ? <><span className="sr-only">{tr(result.strata.message)} {tr(result.engine.message)} </span>{tr("Checked at ")}{tr(checkedLabel(result.checkedAt))}</> : null)}</div>
    </div>
    {tr(active ? <NativeProgress job={active}/> : null)}{tr([currentUpdateFailure(native?.jobs ?? [])].filter((j): j is NativeJob => !!j).map(j => <p role="alert" key={j.id} className="small inline-error">{tr(jobErrorText(j, native?.jobs ?? []))}</p>))}
    <EngineUpdateDialog open={confirm === 'update-engine' && withModel} onClose={() => setConfirm('')}/>
    <Dialog open={!!confirm && !(confirm === 'update-engine' && withModel)} compact title={tr(confirm === 'update-strata' ? 'Update Strata?' : 'Update engine?')} onClose={() => setConfirm('')} actions={<><Btn onClick={() => setConfirm('')}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!!active || (native?.instances.length ?? 0) > 0} onClick={() => void operate(confirm)}>{tr("Update")}</Btn></>}><p>{tr("The native installer updates the installation. Model files and settings are kept. Unload native models before updating.")}</p>{tr((native?.instances.length ?? 0) > 0 ? <p className="small muted">{tr("Open Models and unload the running native models first.")}</p> : null)}</Dialog>
  </>;
}

/** "Strata 0.2.13 · Engine 0.1.59": the app's own version first, the engine's beside it (About). */
export function AboutVersions() {
    const native = useNative();
    const app = appVersionLabel(useStore(s => s.setup?.app.version), native?.version);
    const engineVersion = useStore(s => s.engine?.engine.version);
    const engine = shownEngineVersion(native, engineVersion);
    return <p className="settings-about-version"><strong>{tr("Strata")} {app ?? ''}</strong>{engine ? <span className="muted"> · {tr("Engine")} {engine}</span> : null}</p>;
}
