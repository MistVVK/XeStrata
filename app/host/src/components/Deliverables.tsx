import { useEffect, useId, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { deliveryFolder, dirOf, fileAction, fileSize, fileTypeLabel, previewOf, rawUrlOf, rootOf, type ActivityGroup, type Deliverable } from '../lib/deliverables';
import { useChatFolder } from '../lib/chat-folder';
import type { PreviewArtifact } from '../lib/artifacts';
import { draftSize, groupFromDraft } from '../lib/draft-view';
import type { ToolDraft } from '../lib/types';
import { pushToast } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { FloatingMenu } from './FloatingMenu';
import { CopyMenuItem } from './CopyBtn';
import { Icon } from './ui';

const problem = (title: string, e: unknown) => pushToast('error', tr(title), plainFailureText(e instanceof Error ? e.message : String(e)).message);

/** A line's numbers ("+225 −0") shown as a pair of small figures; the sentence itself stays plain grey. */
export function Plusminus({ added, removed }: { added: number; removed: number }) {
  return <span className="pm" aria-label={tr('{0} lines added, {1} removed', [added, removed])}><span className="pm__a">+{added}</span><span className="pm__d">−{removed}</span></span>;
}

const showOf = (file: { path: string; name: string }, sessionId: string, onPreview: (a: PreviewArtifact) => void, kind?: string) =>
  previewOf(sessionId, file).then(onPreview, e => {
    // A file with no preview opens in its own app instead.
    if (kind === 'other') return fileAction.open(sessionId, file.path, 'app').catch(err => problem('The file was not opened', err));
    return problem('The file has no preview', e);
  });

/** One file the assistant made or changed: its type, name and size, and Open (in the file's own app or the browser) with a menu for the other things to do with it, among them Open in sidebar. */
export function FileCard({ file, sessionId, onPreview }: { file: Deliverable; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const more = useRef<HTMLButtonElement>(null);
  const [menu, setMenu] = useState(false), [busy, setBusy] = useState(false);
  const run = async (fn: () => Promise<unknown>, failed: string) => { setMenu(false); try { await fn(); } catch (e) { problem(failed, e); } };
  return <div className="fcard" data-kind={file.kind}>
    <span className="fcard__type">{fileTypeLabel(file)}</span>
    <span className="fcard__name"><strong title={file.path}>{file.name}</strong><span>{fileSize(file.size)}</span></span>
    <span className="fcard__split">
      <button type="button" className="fcard__open" disabled={busy} onClick={() => { setBusy(true); void fileAction.open(sessionId, file.path, 'app').catch(err => problem('The file was not opened', err)).finally(() => setBusy(false)); }}>{tr("Open")}</button>
      <button ref={more} type="button" className="fcard__more" aria-haspopup="menu" aria-expanded={menu} aria-label={tr("More ways to open {0}", [file.name])} onClick={() => setMenu(v => !v)}><Icon name="chevron" size={14} className="fcard__chev"/></button>
    </span>
    {tr(menu ? <FloatingMenu anchor={more.current} title={tr("File actions")} width={220} onClose={() => setMenu(false)}>
      <button role="menuitem" className="menu__item" onClick={() => { setMenu(false); void showOf(file, sessionId, onPreview, file.kind); }}><Icon name="panel"/>{tr("Open in sidebar")}</button>
      <button role="menuitem" className="menu__item" onClick={() => void run(() => fileAction.open(sessionId, file.path, 'folder'), 'The folder was not opened')}><Icon name="folder"/>{tr("Show in folder")}</button>
      <CopyMenuItem text={() => fileAction.fullPath(sessionId, file.path)} label="Path" onClose={() => setMenu(false)}>Copy path</CopyMenuItem>
      <button role="menuitem" className="menu__item" onClick={() => void run(async () => { await fileAction.attach(sessionId, file.path); pushToast('info', tr('Attached to your next message'), file.name); }, 'The file was not attached')}><Icon name="attach"/>{tr("Attach as context")}</button>
    </FloatingMenu> : null)}
  </div>;
}

/** "Edited N files": each with its lines added and removed, and the total under See all. */
export function EditedFiles({ files, sessionId, onPreview }: { files: Deliverable[]; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const [all, setAll] = useState(false);
  const textual = files.filter(f => f.lines !== null);
  if (textual.length < 2) return null;
  const shown = all ? textual : textual.slice(0, 3);
  const added = textual.reduce((n, f) => n + f.added, 0), removed = textual.reduce((n, f) => n + f.removed, 0);
  return <div className="edited">
    <div className="edited__head"><b>{tr('Edited {0} files', [textual.length])}</b></div>
    {shown.map(f => <button type="button" className="edited__row" key={f.path} onClick={() => void showOf(f, sessionId, onPreview, f.kind)}><Icon name="file" size={16}/><span>{f.name}</span><Plusminus added={f.added} removed={f.removed}/></button>)}
    <button type="button" className="edited__row edited__row--all" disabled={textual.length <= 3 && !all} onClick={() => setAll(v => !v)}><span>{tr(all ? 'Show fewer' : 'See all')}</span><Plusminus added={added} removed={removed}/></button>
  </div>;
}

/**
 * A run that made many files (379 after one stopped agent run) as one compact block instead of a card per file: where the result was delivered, the folder the
 * chat works in, and the count ("379 files") that opens the full list. The list is closed by default, scrolls inside the block, and every row is a button.
 */
export function FilesSummary({ files, answer, sessionId, onPreview }: { files: Deliverable[]; answer: string; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const [open, setOpen] = useState(false), id = useId(), folder = useChatFolder(sessionId);
  const delivery = deliveryFolder(files, answer);
  const rootPath = folder?.path ?? rootOf(files[0]), rootName = folder?.name ?? (rootPath.split(/[\\/]/).filter(Boolean).at(-1) ?? '');
  const added = files.reduce((n, f) => n + f.added, 0), removed = files.reduce((n, f) => n + f.removed, 0);
  const openFolder = (path: string) => void fileAction.open(sessionId, path, 'app').catch(err => problem('The folder was not opened', err));
  const places: { key: string; label: string; name: string; full: string; open: string }[] = [];
  if (delivery) places.push({ key: 'delivered', label: 'Delivered to', name: delivery, full: rootPath ? `${rootPath}/${delivery}` : delivery, open: delivery });
  if (rootPath) places.push({ key: 'working', label: delivery ? 'Working folder' : 'Delivered to', name: rootName, full: rootPath, open: rootPath });
  return <section className="fsum" aria-label={tr("Files")}>
    {places.length ? <ul className="fsum__places">{places.map(p => <li key={p.key}>
      <Icon name="folder" size={16}/><span className="fsum__label">{tr(p.label)}</span><span className="fsum__name" title={p.full}>{p.name}</span>
      <button type="button" className="fsum__open" aria-label={tr("Open {0}", [p.name])} onClick={() => openFolder(p.open)}>{tr("Open")}</button>
    </li>)}</ul> : null}
    <button type="button" className="fsum__toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen(v => !v)}>
      <Icon name="chevron" size={14} className="fsum__chev"/><span>{tr('{0} files', [files.length.toLocaleString()])}</span>{added || removed ? <Plusminus added={added} removed={removed}/> : null}
    </button>
    {open ? <ul className="fsum__list" id={id}>{files.map(f => { const dir = dirOf(f.path); return <li key={f.path}>
      <button type="button" className="fsum__file" title={f.path} onClick={() => void showOf(f, sessionId, onPreview, f.kind)}>
        <span className="fsum__type">{fileTypeLabel(f)}</span><span className="fsum__fname">{f.name}</span>{dir ? <span className="fsum__dir">{dir}</span> : null}<span className="fsum__size">{fileSize(f.size)}</span>
      </button>
    </li>; })}</ul> : null}
  </section>;
}

/**
 * The saved skills an answer used, as one quiet line under the answer in the same style as the activity line ("Ran 2 commands, read a file"):
 * "Used 2 skills: Weekly report, Brand voice"; opened, each skill with its one-line description.
 */
export function SkillsUsed({ skills }: { skills: { name: string; description: string }[] }) {
  const [open, setOpen] = useState(false);
  if (!skills.length) return null;
  const names = skills.map(s => s.name).join(', ');
  return <div className="act" data-open={open || undefined}>
    <button type="button" className="act__line" aria-expanded={open} onClick={() => setOpen(v => !v)}>
      <Icon name="skills" size={15}/><span>{skills.length === 1 ? tr('Used 1 skill: {0}', [names]) : tr('Used {0} skills: {1}', [skills.length, names])}</span>
      <Icon name="chevron" size={14} className="act__chev"/>
    </button>
    {open ? <ul className="act__files">{skills.map(s => <li key={s.name}><span className="act__skill"><Icon name="skills" size={14}/>{s.name}{s.description ? <span className="small muted">{s.description}</span> : null}</span></li>)}</ul> : null}
  </div>;
}

/** A run of tool use between two pieces of text, as one quiet line: "Ran 2 commands, read monitor.png", with a thumbnail when it touched a picture. */
export function ActivityLine({ group, sessionId, onPreview }: { group: ActivityGroup; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const [open, setOpen] = useState(false);
  const picture = group.files.find(f => f.kind === 'image');
  // The host's sentence may end with the numbers ("… +225 −0"): they are drawn as figures instead.
  const sentence = group.line.replace(/\s*\+\d+\s*[−-]\s*\d+\s*$/, '');
  // While the run is open this is the live step: the host's own words for it when it sends them, else the line it already had.
  const live = !group.closed;
  return <><div className="act" data-open={open || undefined} data-live={live || undefined}>
    <button type="button" className="act__line" aria-expanded={open} disabled={!group.files.length} onClick={() => setOpen(v => !v)}>
      <span>{live && group.live?.label ? group.live.label : sentence}</span>{live && group.live?.bytes ? <span className="act__meta">{draftSize(group.live.bytes)}{group.live.lines && group.live.lines > 1 ? <> {"\u00B7"} {tr("{0} lines", [group.live.lines])}</> : null}</span> : null}{group.added || group.removed ? <Plusminus added={group.added} removed={group.removed}/> : null}
      {group.files.length ? <Icon name="chevron" size={14} className="act__chev"/> : null}
      {!group.closed ? <span className="act__live" aria-hidden="true"/> : null}
    </button>
    {picture ? <img className="act__thumb" src={rawUrlOf(sessionId, picture.path)} alt="" loading="lazy" onClick={() => void showOf(picture, sessionId, onPreview)}/> : null}
    {open ? <ul className="act__files">{group.files.map(f => <li key={f.path}><button type="button" onClick={() => void showOf(f, sessionId, onPreview, f.kind)}><Icon name="file" size={14}/>{f.name}{f.added || f.removed ? <Plusminus added={f.added} removed={f.removed}/> : null}</button></li>)}</ul> : null}
  </div>
    {live && group.live?.output ? <p className="act__output" title={group.live.output}>{group.live.output}</p> : null}
    {live && group.live?.tail !== undefined ? <div className="act__tail" aria-hidden="true">{group.live.tail.split('\n').filter(Boolean).map((line, i) => <span key={i}>{line}</span>)}</div> : null}
  </>;
}

/** The value, at most every `ms` (a file arrives about five times a second; the eye needs two). A cleared value is shown at once. */
function useThrottled<T>(value: T, ms: number, cleared: (v: T) => boolean): T {
  const [shown, setShown] = useState(value), last = useRef(0), timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    clearTimeout(timer.current);
    if (cleared(value)) { setShown(value); return; }
    const wait = Math.max(0, last.current + ms - Date.now());
    timer.current = setTimeout(() => { last.current = Date.now(); setShown(value); }, wait);
    return () => clearTimeout(timer.current);
  }, [value, ms]);
  return shown;
}

/**
 * The files (or commands) the model is writing right now, from `message.drafts`: each is one open activity line with the file's name, its growing size
 * and the last lines of its text. It draws nothing while there is nothing being written, and takes the place of the ordinary card when the call is whole.
 */
export function DraftLines({ drafts, sessionId, onPreview }: { drafts: ToolDraft[] | undefined; sessionId: string; onPreview: (a: PreviewArtifact) => void }) {
  const shown = useThrottled(drafts, 350, v => !v?.length);
  if (!drafts?.length || !shown?.length) return null;
  return <>{shown.map(d => {
    const file = d.target ? d.target.split(/[\\/]/).filter(Boolean).at(-1) : '';
    const verb = tr(d.verb ?? (d.scope === 'shell' ? 'Running a command' : 'Writing a file'));
    return <ActivityLine key={d.id} group={groupFromDraft(d, file ? `${verb} · ${file}` : verb)} sessionId={sessionId} onPreview={onPreview}/>;
  })}</>;
}
