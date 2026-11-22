import {sessionTitle} from '../lib/session-title';
import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { ArtifactTabs, type ExtraPreviewTab } from './ArtifactPanel';
import { FloatingMenu } from './FloatingMenu';
import { Popover } from './Popover';
import { Btn, Dialog, Icon, IconBtn } from './ui';
import { SamplingDrawer } from '../app/SamplingDrawer';
import { downloadSessionLog } from '../lib/session-log';
import { SubagentRow } from './SubagentRow';
import { ChatModelPicker } from './ChatModelPicker';
import { agentCounts } from '../lib/agents';
import { dismissAgent, useStore } from '../lib/store';
import { recordOf, updateHelperRecords, useHelperRecords } from '../lib/helper-records';
import { plainFailureText } from '../lib/plain-errors';
import type { Session, SessionAgent } from '../lib/types';
import type { PreviewArtifact } from '../lib/artifacts';
export function ChatToolbar({ onPauseForUpdate, session, items, selected, onSelect, onClose, onCloseTab, inline, expanded, onExpand, extraTabs = [], previewWidth }: {
    onPauseForUpdate: () => Promise<void>;
    previewWidth?: number;
    extraTabs?: ExtraPreviewTab[];
    session: Session | null;
    items: PreviewArtifact[];
    selected: string | null;
    onSelect: (id: string) => void;
    onClose: () => void;
    onCloseTab: (id: string) => void;
    inline: boolean;
    expanded: boolean;
    onExpand: () => void;
}) {
    const remote = useStore(s => s.remoteModel);
    const [agents, setAgents] = useState(false), [menu, setMenu] = useState(false), [settings, setSettings] = useState(false);
    const [openHelperId, setOpenHelperId] = useState<string | null>(null);
    const [pausing, setPausing] = useState(false);
    const [showArchived, setShowArchived] = useState(false);
    const [archiving, setArchiving] = useState(false);
    const [archiveProblem, setArchiveProblem] = useState('');
    const agentsRef = useRef<HTMLButtonElement>(null), menuRef = useRef<HTMLButtonElement>(null);
    const sessionId = session && !session.showcase ? session.id : undefined;
    const records = useHelperRecords(sessionId);
    const rows = session?.agents ?? [];
    const busy = (a: SessionAgent) => a.status === 'working' || a.status === 'queued';
    const archived = rows.filter(a => !busy(a) && recordOf(records, a.id)?.archived);
    const visible = rows.filter(a => busy(a) || !recordOf(records, a.id)?.archived);
    const finished = visible.filter(a => !busy(a));
    const { working: count, notWorking, workingList } = agentCounts(visible);
    const openHelper = openHelperId ? rows.find(a => a.id === openHelperId) ?? null : null;
    const [now, setNow] = useState(Date.now);
    useEffect(() => { if ((!agents && !openHelper) || !count)
        return; setNow(Date.now()); const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, [agents, openHelper, count, session?.id]);
    useEffect(() => {
        setAgents(false);
        setOpenHelperId(null);
        setShowArchived(false);
        setArchiveProblem('');
    }, [session?.id]);
    async function archiveFinished() {
        if (!sessionId || !finished.length || archiving) return;
        setArchiving(true);
        setArchiveProblem('');
        try { await updateHelperRecords(sessionId, 'archive', finished.map(a => a.id)); }
        catch (error) { setArchiveProblem(plainFailureText(error instanceof Error ? error.message : String(error)).message); }
        finally { setArchiving(false); }
    }
    const renderHelper = (a: SessionAgent) => <SubagentRow key={a.id} agent={a} now={now} sessionId={sessionId} record={recordOf(records, a.id)}
      onOpenWork={() => setOpenHelperId(a.id)}
      onRemove={sessionId ? () => dismissAgent(sessionId, a.id) : undefined}/>;
    const previews = !!selected && inline;
    const title = session ? sessionTitle(session) : tr('New chat');
    const workingNames = workingList.slice(0, 4).map(a => a.name).join(', ') + (workingList.length > 4 ? ` (+${workingList.length - 4})` : '');
    const agentsTitle = count
      ? `${count} working / ${notWorking} not working · ${workingNames}`
      : visible.length
        ? `0 working / ${notWorking} not working · View overview`
        : 'View subagent overview';
    return <header className="chat-toolbar" style={previews && !expanded && previewWidth ? { gridTemplateColumns: `minmax(0,1fr) ${previewWidth}px` } : undefined} data-timeskip={selected?.startsWith('time-skip:') || undefined} data-preview={previews || undefined} data-expanded={expanded && previews || undefined}>
    <div className="chat-toolbar__conversation"><h1 title={title}>{title}</h1><ChatModelPicker /><span className="chat-agents-wrap"><button ref={agentsRef} className="chat-agents" title={tr(agentsTitle)} aria-label={tr(count ? `${count} working, ${notWorking} not working — show overview` : visible.length ? `0 working, ${notWorking} not working — show overview` : 'Subagents — show overview')} aria-haspopup="dialog" aria-expanded={agents} data-working={count > 0 || undefined} onClick={() => setAgents(v => !v)}><Icon name="experts" size={17}/>{tr(count > 0 ? <span className="agent-live-dot"/> : null)}<span className="chat-agents__label">{tr("Subagents")}</span>{visible.length > 0 && <span className="chat-agents__count" aria-label={tr('{0} working, {1} not working', [count, notWorking])}>{count}/{notWorking}</span>}<Icon name="chevron" className={agents ? 'is-open' : ''} size={13}/></button>{count > 0 && !agents && <div className="chat-agents__hover" role="tooltip"><strong>{tr('{0} working / {1} not working', [count, notWorking])}</strong><ul>{workingList.slice(0, 5).map(a => <li key={a.id}><span>{a.name}</span><small>{tr(a.description)}</small></li>)}</ul></div>}</span><span className="spacer"/>
      <Btn size="sm" icon="stop" disabled={pausing} title={tr('Save progress and pause all agents for an update')} onClick={async () => {
        setPausing(true);
        try { await onPauseForUpdate(); } finally { setPausing(false); }
      }}>{tr('Pause for update')}</Btn>
      <button ref={menuRef} className="st-btn st-btn--icon" title={tr("Chat options")} aria-label={tr("Chat options")} aria-haspopup="menu" aria-expanded={menu} onClick={() => setMenu(v => !v)}><Icon name="ellipsis"/></button>{tr(!selected && (items.length || extraTabs.length) ? <IconBtn name="panel" title={tr("Open previews")} onClick={() => onSelect(extraTabs.at(-1)?.id ?? items[0].id)}/> : null)}</div>
    {tr(previews ? <div className="chat-toolbar__previews"><ArtifactTabs extraTabs={extraTabs} onCloseTab={onCloseTab} items={items} selected={selected} onSelect={onSelect}/><div className="preview-window-actions"><div id="preview-file-actions" className="preview-file-actions"/><IconBtn name={expanded ? 'contract' : 'expand'} title={tr(expanded ? 'Exit full screen preview' : 'Full screen preview')} onClick={onExpand}/><IconBtn name="panel" title={tr("Collapse preview sidebar")} onClick={onClose}/></div></div> : null)}
    {agents && <Popover anchor={agentsRef.current} title={visible.length || archived.length ? "Subagents" : "No subagents recorded"}
      detail={visible.length ? <>{count}{tr(" working · ")}{notWorking}{tr(" not working · ")}{visible.length}{tr(" total")}</> : undefined}
      width={460} onClose={() => { setAgents(false); setShowArchived(false); }}>
      {sessionId && (finished.length > 0 || archived.length > 0) && <div className="subagent-overview__actions">
        {finished.length > 0 && <Btn size="sm" disabled={archiving} onClick={() => void archiveFinished()}>{tr('Archive finished')}</Btn>}
        {archived.length > 0 && <Btn size="sm" aria-expanded={showArchived} onClick={() => setShowArchived(v => !v)}>
          {showArchived ? tr('Hide archive') : tr('Show archive ({0})', [archived.length])}
        </Btn>}
      </div>}
      {archiveProblem && <p className="small inline-error" role="alert">{archiveProblem}</p>}
      {visible.length ? visible.map(renderHelper) : archived.length > 0 && <p className="subagent-overview__empty muted">{tr('All helpers are archived.')}</p>}
      {showArchived && archived.length > 0 && <section className="subagent-overview__archive" aria-label={tr('Archived helpers')}>
        <h2>{tr('Archived helpers')}</h2>
        {archived.map(renderHelper)}
      </section>}
    </Popover>}
    {openHelper && <Dialog open title={tr('{0} · work', [openHelper.name])} wide className="subagent-work-dialog" onClose={() => setOpenHelperId(null)} actions={<Btn onClick={() => setOpenHelperId(null)}>{tr('Done')}</Btn>}>
      <SubagentRow agent={openHelper} now={now} sessionId={sessionId} record={recordOf(records, openHelper.id)} panel/>
    </Dialog>}
    {tr(menu ? <FloatingMenu anchor={menuRef.current} title={tr("Chat options")} onClose={() => setMenu(false)}><button role="menuitem" className="menu__item" disabled={!!remote} title={tr(remote ? "API models use their provider’s response settings." : undefined)} onClick={() => { setMenu(false); setSettings(true); }}><Icon name="sliders"/>{tr("Response settings")}</button><button role="menuitem" className="menu__item" disabled={!session} onClick={() => { if (session)
        downloadSessionLog(session); setMenu(false); }}><Icon name="download"/>{tr("Download session log")}</button></FloatingMenu> : null)}
    <SamplingDrawer open={settings} onClose={() => setSettings(false)}/>
  </header>;
}
