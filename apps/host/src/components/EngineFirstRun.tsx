import { tr } from '../lib/i18n';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Brand } from './Brand';
import { Bar, Btn, Icon } from './ui';
import { api } from '../lib/client';
import { useDownloadEta } from '../lib/download-eta';
import type { NativeJob } from '../lib/native';
import { megabytes } from '../lib/format';
import { FirstRunLanguage } from './FirstRunLanguage';

/** What the host says about this PC's engine (GET /api/engine/first-run). */
export interface FirstRunInfo {
    status: 'offer' | 'none' | 'offline' | 'error' | 'installed' | 'wrong-make' | 'unavailable';
    gpus: { name: string; vramGb: number | null }[];
    driver?: string | null;
    offer?: { version: string; bytes: number; variant: string; backend: string | null };
    /** A newer Strata Control in the same feed: the Download takes it too (the Arc fix of 0.2.13.1 came this way). */
    control?: { version: string; bytes: number } | null;
    message?: string | null;
    reasons?: string[];
    details?: string;
    freeBytes?: number;
}

export const mb = megabytes;
/** After the size when the download also brings the newer Strata Control (owner-approved wording, 11-09). */
export const WITH_CONTROL = " (with the Strata setup update)";

/**
 * Whether this PC still needs its engine. Reads only Control's local answer (no network): the engine is missing and the engine that came
 * with the app is not being unpacked. `null` while that is not known yet.
 */
export function useEngineNeeded(): boolean | null {
    const [needed, setNeeded] = useState<boolean | null>(null);
    useEffect(() => {
        let live = true;
        const ask = () => api.get<{ engine?: { usable?: boolean } | null; engineSetup?: boolean }>('/api/control/status').then(s => {
            if (!live) return;
            if (s.engineSetup) return setNeeded(null);
            setNeeded(!!s.engine && s.engine.usable === false);
        }, () => { if (live) setNeeded(false); });
        void ask();
        const timer = setInterval(() => void ask(), 3000);
        return () => { live = false; clearInterval(timer); };
    }, []);
    return needed;
}

function gpuLine(info: FirstRunInfo) {
    // a processor's own tiny graphics (no memory of its own) is not "a card"
    const first = info.gpus.find(g => (g.vramGb ?? 0) >= 1) ?? info.gpus[0];
    if (!first) return '';
    const vram = first.vramGb ? `, ${Math.round(first.vramGb)} GB` : '';
    const driver = info.driver ? `, driver ${info.driver}` : '';
    return `${first.name}${vram}${driver}`;
}

/**
 * The first-run card: the engine this PC needs, one button to download it, progress, and the offline way (choose a downloaded file).
 * Nothing is downloaded before a button is pressed.
 */
export function EngineFirstRun({ onDone }: { onDone: () => void }) {
    const [info, setInfo] = useState<FirstRunInfo | null>(null);
    const [job, setJob] = useState<NativeJob | null>(null);
    const [busy, setBusy] = useState(false), [problem, setProblem] = useState('');
    const [finished, setFinished] = useState(false);
    const file = useRef<HTMLInputElement>(null);
    const started = useRef(false);

    const look = useCallback(() => {
        setInfo(null);
        setProblem('');
        api.get<FirstRunInfo>('/api/engine/first-run').then(setInfo, e => setInfo({ status: 'error', gpus: [], message: e instanceof Error ? e.message : String(e) }));
    }, []);
    useEffect(() => { look(); }, [look]);

    // follow the download / install job
    useEffect(() => {
        const timer = setInterval(() => {
            if (!started.current) return;
            api.get<{ job: NativeJob | null }>('/api/engine/job').then(r => {
                setJob(r.job);
                if (r.job?.state === 'done') setFinished(true);
                if (r.job && r.job.state !== 'running') started.current = false;
            }, () => { });
        }, 700);
        return () => clearInterval(timer);
    }, []);

    async function start() {
        setBusy(true);
        setProblem('');
        try {
            started.current = true;
            setJob(await api.post<NativeJob>('/api/native/operations', { kind: 'update-engine', args: { firstRun: true, ...(info?.control ? { withControl: true } : {}) } }));
        }
        catch (e) {
            started.current = false;
            setProblem(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
        }
    }
    async function chosen(files: FileList | null) {
        const picked = files?.[0];
        if (!picked) return;
        setBusy(true);
        setProblem('');
        try {
            const response = await fetch('/api/engine/from-file', { method: 'POST', headers: { 'x-file-name': picked.name }, body: picked });
            const body = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(body?.error ?? `The file could not be used (${response.status}).`);
            started.current = true;
            setJob(body as NativeJob);
        }
        catch (e) {
            setProblem(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
            if (file.current) file.current.value = '';
        }
    }

    const running = job?.state === 'running';
    const failed = !!job && (job.state === 'error' || job.state === 'cancelled');
    const percent = Number(job?.result?.install?.download?.percent ?? 0);
    const bytes = info?.offer?.bytes ?? 0;
    const doneMb = Math.round(mb(bytes) * percent / 100);
    const eta = useDownloadEta(running && bytes ? { done_gb: bytes * percent / 100 / 1e9, remaining_gb: bytes * (100 - percent) / 100 / 1e9 } : undefined, job?.id);
    const cards = (info?.gpus ?? []).filter(g => (g.vramGb ?? 0) >= 1).length;
    const hidden = <input ref={file} type="file" accept=".gz,.tgz,.xz,.zip" hidden onChange={e => void chosen(e.target.files)}/>;
    const pick = (label: string) => <button type="button" className="onboard-continue-anyway" disabled={busy || running} onClick={() => file.current?.click()}>{tr(label)}</button>;

    let body;
    if (finished) {
        body = <><div className="onboard-welcome-copy"><h1 tabIndex={-1} data-initial-focus>{tr("The engine is ready.")}</h1><p>{tr("Next: pick a model.")}</p></div>
            <footer className="onboard-footer onboard-footer--welcome"><Btn variant="primary" onClick={onDone}>{tr("Continue")}<Icon name="chevron" size={16}/></Btn></footer></>;
    }
    else if (running) {
        body = <><div className="onboard-welcome-copy"><h1 tabIndex={-1} data-initial-focus>{tr("Downloading the engine")}</h1>
            <p>{bytes && job?.result?.install?.download ? `${doneMb} of ${mb(bytes)} MB${eta ? ' · ' + eta : ''}` : tr(job?.summary ?? 'Starting…')}</p></div>
            <div className="onboard-error" style={{ maxWidth: 520, margin: '0 auto', width: '100%' }}><Bar value={percent} label="Downloading the engine"/><p className="small muted" style={{ textAlign: 'center' }}>{tr("You can close Strata; it continues where it stopped.")}</p></div></>;
    }
    else if (!info) {
        body = <div className="onboard-welcome-copy"><h1 tabIndex={-1} data-initial-focus>{tr("Welcome to Strata")}</h1><p role="status">{tr("Looking at your graphics card…")}</p></div>;
    }
    else {
        const card = gpuLine(info);
        const found = card ? <p>{tr("Strata found your graphics card:")}<br/><strong>{card}</strong>{cards > 1 ? <><br/><span className="small muted">{tr("{0} cards found. Strata uses the first one for the engine; change it later in Settings.", [cards])}</span></> : null}</p>
            : <p>{tr("Strata did not recognise a supported graphics card.")}</p>;
        let main, actions;
        if (info.status === 'offer' && info.offer) {
            main = <>{found}<p>{tr("To run models on it, Strata needs one download:")}<br/><strong>{tr("The engine for this card")}: {mb(info.offer.bytes + (info.control?.bytes ?? 0))} MB{info.control ? tr(WITH_CONTROL) : null}</strong></p>
                {info.freeBytes !== undefined && info.freeBytes < info.offer.bytes * 2.5 ? <p className="small inline-error">{tr("The engine needs about {0} GB of free space on this drive and only {1} GB is free. Free some space, then continue.", [Math.ceil(info.offer.bytes * 2.5 / 1e9), (info.freeBytes / 1e9).toFixed(1)])}</p> : null}
                <p className="small muted">{tr("It comes from Strata’s own update server and is checked before it is used. Your models and chats are never touched by it.")}</p></>;
            actions = <><Btn variant="primary" disabled={busy} onClick={() => void start()}>{tr(failed ? 'Continue' : 'Download the engine')}</Btn>{pick("I already have the file…")}</>;
        }
        else if (info.status === 'offline') {
            main = <>{found}<p>{tr("Strata could not reach github.com. Check your connection and try again, or download the engine file on another PC and choose it here.")}</p></>;
            actions = <><Btn variant="primary" disabled={busy} onClick={look}>{tr("Try again")}</Btn>{pick("Choose a downloaded file…")}</>;
        }
        else if (info.status === 'wrong-make') {
            main = <>{found}<p>{info.message ?? tr("The engine installed here is for another make of graphics card.")}</p></>;
            actions = <><Btn variant="primary" disabled={busy} onClick={look}>{tr("Try again")}</Btn>{pick("Choose a downloaded file…")}</>;
        }
        else {
            main = <>{found}<p>{tr("Strata can’t pick an engine for this PC from its update list.")}</p>{info.message ? <p className="small muted">{info.message}</p> : null}{(info.reasons ?? []).length ? <ul className="small muted" style={{ textAlign: 'left', display: 'inline-block' }}>{(info.reasons ?? []).map(r => <li key={r}>{r}</li>)}</ul> : null}</>;
            actions = <><Btn variant="primary" disabled={busy} onClick={look}>{tr("Try again")}</Btn>{pick("Choose a downloaded file…")}<button type="button" className="onboard-continue-anyway" onClick={onDone}>{tr("Continue without an engine")}</button></>;
        }
        body = <><div className="onboard-welcome-copy"><h1 tabIndex={-1} data-initial-focus>{tr("Welcome to Strata")}</h1>{main}</div>
            {failed ? <div className="onboard-error" style={{ maxWidth: 560, margin: '0 auto' }}><p role="alert" className="small inline-error">{job?.error ?? job?.summary}</p>
                {job?.details ? <details className="error-detail"><summary className="small">{tr('Details')}</summary><pre className="small muted">{job.details}</pre></details> : null}
                <p className="small muted">{percent >= 100 ? tr("The file was downloaded. Continue tries the install again.") : tr("The download stopped. Continue picks it up where it stopped.")}</p></div> : null}
            {problem ? <div className="onboard-error" style={{ maxWidth: 560, margin: '0 auto' }}><p role="alert" className="small inline-error">{problem}</p></div> : null}
            <footer className="onboard-footer onboard-footer--welcome">{actions}</footer></>;
    }
    return <div className="setup guided-onboarding" data-step="engine">{hidden}<header className="onboard-header"><Brand variant="lockup" height={22}/><FirstRunLanguage/></header>{body}</div>;
}
