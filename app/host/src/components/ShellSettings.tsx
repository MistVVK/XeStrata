import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { useImeKeys } from '../lib/ime-keys';
import { tr } from '../lib/i18n';
import { askNotificationPermission, playDoneSound } from '../lib/attention';
import { setApprovalNotify, setDoneSound, setNumberQuotes, useStore } from '../lib/store';
import { Btn } from './ui';

/** What the host says about the desktop shell (GET /api/shell): null in a browser or while it loads. */
export interface ShellState {
    desktop: boolean;
    tray: boolean;
    port: number;
    defaultPort: number;
    pinnedPort: number | null;
    minimizeToTray: boolean;
    portNotice: { wanted: number; used: number } | null;
}

export function useShellState(): [ShellState | null, (patch: { port?: number | null; tray?: boolean }) => Promise<string | null>] {
    const [shell, setShell] = useState<ShellState | null>(null);
    useEffect(() => {
        let alive = true;
        api.get<ShellState>('/api/shell').then(s => { if (alive) setShell(s); }).catch(() => undefined);
        return () => { alive = false; };
    }, []);
    const save = async (patch: { port?: number | null; tray?: boolean }) => {
        try { setShell(await api.post<ShellState>('/api/shell/prefs', patch)); return null; }
        catch (error) { return error instanceof Error ? error.message : 'The choice was not saved.'; }
    };
    return [shell, save];
}

/** The sound for finished work (#3586): off until switched on, with a button to hear it. */
export function DoneSoundSetting() {
    const on = useStore(s => s.doneSound);
    return <span className="row">
        <Btn size="sm" onClick={() => playDoneSound()}>{tr("Play sound")}</Btn>
        <button className="st-toggle" role="switch" aria-checked={on} aria-label={tr("Sound when work finishes")} onClick={() => setDoneSound(!on)}/>
    </span>;
}

/** A desktop notification when Manual mode waits for an approval (#3477). */
export function ApprovalNotifySetting() {
    const on = useStore(s => s.approvalNotify);
    return <button className="st-toggle" role="switch" aria-checked={on} aria-label={tr("Notify me when Strata needs an answer")}
        onClick={() => { if (!on) void askNotificationPermission(); setApprovalNotify(!on); }}/>;
}

/** Settings > General > Number quotes (#5993): off by default; on, a message that quotes several passages numbers them. */
export function NumberQuotesSetting() {
    const on = useStore(s => s.numberQuotes);
    return <button className="st-toggle" role="switch" aria-checked={on} aria-label={tr("Number quotes")} onClick={() => setNumberQuotes(!on)}/>;
}

/** Keep the app running in the tray when its window is closed (#3477). Desktop app on Windows and macOS only. */
export function TraySetting({ shell, save }: { shell: ShellState; save: (patch: { tray: boolean }) => Promise<string | null> }) {
    return <button className="st-toggle" role="switch" aria-checked={shell.minimizeToTray} aria-label={tr("Minimise to tray")} onClick={() => void save({ tray: !shell.minimizeToTray })}/>;
}

/** The port the app's window and local address use (#3402). Empty lets Strata choose; the choice applies the next time Strata starts. */
export function PortSetting({ shell, save }: { shell: ShellState; save: (patch: { port: number | null }) => Promise<string | null> }) {
    const [text, setText] = useState(shell.pinnedPort ? String(shell.pinnedPort) : ''), [problem, setProblem] = useState(''), ime = useImeKeys();
    useEffect(() => { setText(shell.pinnedPort ? String(shell.pinnedPort) : ''); }, [shell.pinnedPort]);
    const commit = async () => {
        const trimmed = text.trim();
        if (trimmed === (shell.pinnedPort ? String(shell.pinnedPort) : '')) { setProblem(''); return; }
        if (trimmed === '') { setProblem((await save({ port: null })) ?? ''); return; }
        const value = Number(trimmed);
        if (!Number.isInteger(value) || value < 1024 || value > 65535) { setProblem(tr("Choose a port between 1024 and 65535, or leave it empty.")); return; }
        setProblem((await save({ port: value })) ?? '');
    };
    return <span className="stack">
        <span className="row">
            <input className="st-input" inputMode="numeric" size={7} aria-label={tr("Port (empty = automatic)")} placeholder={String(shell.defaultPort)} value={text}
                onChange={e => setText(e.target.value.replace(/[^0-9]/g, '').slice(0, 5))} onBlur={() => void commit()} {...ime.events} onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter') void commit(); }}/>
        </span>
        <span className="xs muted">{tr("Using port {0} now.", [shell.port])}</span>
        {problem ? <span className="xs" role="alert">{problem}</span> : null}
    </span>;
}
