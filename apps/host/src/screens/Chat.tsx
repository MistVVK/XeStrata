import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { UsageDashboard } from '../components/UsageDashboard';
import { FloatingMenu } from '../components/FloatingMenu';
import { lazyScreen } from '../lib/lazy-screen';
import { measureFrame } from '../lib/overlay-place';
import { timeSkipPrompt, timeSkipTabId, type TimeSkipRun } from '../lib/time-skip';
import { ChatToolbar } from '../components/ChatToolbar';
import { ChatTelemetry } from '../components/TelemetryPopover';
import { ArtifactCard } from '../components/ArtifactCard';
import { ChatMedia } from '../components/ChatMedia';
import { closePreviewTab } from '../lib/preview-tabs';
import { endPreviewDeferral, previewsDeferred } from '../lib/recovery';
import { messageArtifacts, openItem, syncArtifacts } from '../lib/preview-merge';
import { createImeGuard } from '../lib/ime-enter';
import { mainCodePreview, setSelectedPreviewId } from '../lib/code-preview';
import { ArtifactPanel } from '../components/ArtifactPanel';
import { PreviewResizeHandle } from '../components/PreviewResizeHandle';
import { DEFAULT_PREVIEW_FRACTION, PREVIEW_SPLIT_KEY, previewWidth } from '../lib/preview-layout';
import { ComposerControls } from '../components/ComposerControls';
import { ChatPhotos } from '../components/ChatPhotos';
import { useChatPhotos, removeChatPhoto } from '../lib/chat-photos';
import { useNative } from '../lib/native';
import { COMPACT_FILES_FROM, mainResult, previewOf } from '../lib/deliverables';
import { noModelNotice } from '../lib/no-model';
import { ReasoningBlock } from '../components/ReasoningBlock';
import { StatusLine } from '../components/StatusLine';
import { GoalBar } from '../components/GoalBar';
import { PlanCard, PlanPanel, QueuedMessages } from '../components/PlanCard';
import { FolderChip } from '../components/FolderChip';
import { DropZone, DROP_EVENT, pastedFiles } from '../components/DropZone';
import { ActivityLine, DraftLines, EditedFiles, FileCard, FilesSummary, SkillsUsed } from '../components/Deliverables';
import { queueMessage, usePlan, usePlanQuiet } from '../lib/plan';
import { plainFailureText } from '../lib/plain-errors';
import { contextRecovery } from '../lib/context-recovery';
import { useLiveStatus } from '../lib/live-status';
import { ApprovalDock, SCOPE } from '../components/ApprovalCard';
import { artifactFromUrl, type PreviewArtifact } from '../lib/artifacts';
import { openExternal } from '../lib/open-external';
import { busyPlan } from '../lib/while-busy';
import { composeQuoted, parseQuoted, type Quote } from '../lib/quotes';
/** Chat — talk to the one loaded model. */
import { memo, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Markdown } from '../components/Markdown';
import { Btn, Icon, IconBtn, ErrorNotice, type IconName } from '../components/ui';
import { composerCap, scrollBehavior } from '../lib/responsive';
import { relTime, cx, duration, tokens, tps } from '../lib/format';
import { addQuote, clearQuotes, removeQuote, createSubagentExample, parkMessage, takeParked, pushToast, setPreviewTabClosed, branchFrom, continueLast, markRead, retryLast, selectSession, rewindTo, sendMessage, stopStreaming, useStore, } from '../lib/store';
import { CopyBtn, CopyMenuItem, MenuCloseContext } from '../components/CopyBtn';
import { KEYS_UP, PILL_DISTANCE, distanceFromBottom, followAfterScroll, wheelLeavesBottom } from '../lib/follow-scroll';
import { getSlashCommands } from '../lib/commands';
import { useLocation } from '../lib/router';
import { composerSends } from '../lib/shortcut-bindings';
import type { ChatMessage, Session, ToolCall } from '../lib/types';
// The Time-skip panel and its progress card are only drawn in a chat that has a Time-skip run: their code loads then (and a few seconds after start, see main.tsx).
const TimeSkipPanel = lazyScreen(() => import('../components/TimeSkipPanel'), (m) => m.TimeSkipPanel);
const TimeSkipWorking = lazyScreen(() => import('../components/TimeSkipWorking'), (m) => m.TimeSkipWorking);
export function ChatScreen() {
    const location = useLocation();
    const sessions = useStore((s) => s.sessions);
    const activeId = useStore((s) => s.activeSession);
    const library = useStore((s) => s.library);
    const streaming = useStore((s) => s.streaming);
    const whileBusy = useStore(s => s.whileBusy);
    const setup = useStore(s => s.setup);
    const setupFlag = useStore(s => s.setupOpen);
    const setupVisible = location.route === 'setup' || setupFlag || !!setup && !setup.complete && !setup.dismissed;
    const session: Session | null = useMemo(() => {
        const wanted = location.param ?? activeId;
        return sessions.find((s) => s.id === wanted) ?? sessions[0] ?? null;
    }, [sessions, activeId, location.param]);
    /* A conversation you are watching is read by definition — including while a
       reply streams into it. Without this, switching away from a session that
       answered you would mark it unread. */
    const messageCount = session?.messages.length ?? 0;
    useEffect(() => {
        if (session && session.id === activeId && !streaming)
            markRead(session.id);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session?.id, messageCount, streaming]);
    const parked = useStore(s => (session ? s.parked[session.id] : undefined));
    const quotes = useStore(s => (session ? s.quotes[session.id] : undefined)) ?? NO_QUOTES;
    // Selecting words in a message offers to quote just them (#4785).
    const [selection, setSelection] = useState<{ text: string; messageId: string; role: 'user' | 'assistant'; x: number; y: number } | null>(null);
    useEffect(() => {
        const read = () => {
            const sel = window.getSelection();
            const el = scroller.current;
            if (!sel || sel.isCollapsed || !sel.rangeCount || !el) { setSelection(null); return; }
            const range = sel.getRangeAt(0);
            const node = range.commonAncestorContainer;
            const host = (node instanceof Element ? node : node.parentElement)?.closest('.st-msg');
            const text = sel.toString().trim();
            if (!host || !el.contains(host) || text.length < 2) { setSelection(null); return; }
            const rect = range.getBoundingClientRect();
            // screen units of this engine, written back in the page's own pixels (lib/overlay-place.ts: Interface size zooms the page)
            const { scale, viewport } = measureFrame();
            setSelection({ text, messageId: host.getAttribute('data-message-id') ?? '', role: host.classList.contains('st-msg--user') ? 'user' : 'assistant', x: Math.min(viewport.width - 96 * scale, Math.max(8, rect.left)) / scale, y: Math.max(8, rect.top - 40 * scale) / scale });
        };
        document.addEventListener('selectionchange', read);
        return () => document.removeEventListener('selectionchange', read);
    }, [session?.id]);
    // Messages parked while an answer was written (Send after, Interrupt) go one by one, each when the chat is quiet again.
    useEffect(() => {
        if (!session || streaming || !parked?.length)
            return;
        const next = takeParked(session.id);
        if (next !== undefined)
            void sendMessage(next);
    }, [streaming, parked?.length, session?.id]);
    const [draft, setDraft] = useState('');
    const draftKey = `strata.chat-draft.${session?.id ?? 'new'}`;
    useEffect(() => {
        setDraft(sessionStorage.getItem(draftKey) ?? '');
        if (session && session.id !== activeId)
            selectSession(session.id);
    }, [draftKey]);
    const updateDraft = (value: string) => { setDraft(value); sessionStorage.setItem(draftKey, value); };
    const splitRef = useRef<HTMLDivElement>(null);
    const [splitWidth, setSplitWidth] = useState(0);
    const [previewFraction, setPreviewFraction] = useState(() => {
        try {
            const saved = localStorage.getItem(PREVIEW_SPLIT_KEY);
            return saved && Number.isFinite(Number(saved)) ? Number(saved) : DEFAULT_PREVIEW_FRACTION;
        }
        catch {
            return DEFAULT_PREVIEW_FRACTION;
        }
    });
    useEffect(() => {
        const timer = setTimeout(() => {
            try {
                localStorage.setItem(PREVIEW_SPLIT_KEY, String(previewFraction));
            }
            catch { }
        }, 180);
        return () => clearTimeout(timer);
    }, [previewFraction]);
    useEffect(() => {
        if (!splitRef.current)
            return;
        const ro = new ResizeObserver(entries => setSplitWidth(entries[0].contentRect.width));
        ro.observe(splitRef.current);
        return () => ro.disconnect();
    }, [session?.id]);
    const [expanded, setExpanded] = useState(false);
    useEffect(() => { setExpanded(false); }, [session?.id]);
    useEffect(() => {
        if (!expanded)
            return;
        const exit = (e: KeyboardEvent) => {
            if (e.key === 'Escape')
                setExpanded(false);
        };
        document.addEventListener('keydown', exit);
        return () => document.removeEventListener('keydown', exit);
    }, [expanded]);
    const [previews, setPreviews] = useState<PreviewArtifact[]>([]);
    const [selectedPreview, setSelectedPreview] = useState<string | null>(null);
    const timeSkipRuns = (session?.timeSkips ?? []).filter(run => session?.messages.some(m => m.timeSkipId === run.id && m.role === 'assistant') && (!run.job || (run.job.files?.length ?? 0) > 0));
    const runTabs = timeSkipRuns.map((run, i) => ({ id: timeSkipTabId(run.id), title: timeSkipRuns.length > 1 ? `Time-skip ${i + 1}` : 'Time-skip', icon: 'branch' as const, controls: 'time-skip-content' })).filter(tab => !session?.closedPreviews?.includes(tab.id));
    const plan = usePlan(session?.id);
    const planTabId = session ? `plan:${session.id}` : '';
    const planTab = plan && plan.state !== 'rejected' && !session?.closedPreviews?.includes(planTabId) ? [{ id: planTabId, title: 'Plan', icon: 'check' as const, controls: 'plan-content' }] : [];
    const planShown = planTab.length > 0 && selectedPreview === planTabId;
    const planRunId = plan?.state === 'running' ? plan.id : '';
    useEffect(() => { if (planRunId && planTabId && !session?.closedPreviews?.includes(planTabId)) setSelectedPreview(planTabId); }, [planRunId]);
    // The chat shows every run of this chat (a running one has no files yet, so no tab); the sidebar tabs only those with something to compare.
    const allTimeSkipRuns = session?.timeSkips ?? [];
    const extraTabs = [...runTabs, ...planTab];
    const selectedRun = timeSkipRuns.find(run => timeSkipTabId(run.id) === selectedPreview);
    const inlinePreview = splitWidth >= (selectedRun || planShown ? 760 : 1120);
    const paneWidth = previewWidth(splitWidth, previewFraction);
    const splitStyle = inlinePreview && selectedPreview && !expanded ? { gridTemplateColumns: `minmax(0,1fr) ${paneWidth}px` } : undefined;
    const resizeHandle = !setupVisible && inlinePreview && selectedPreview && !expanded ? <PreviewResizeHandle container={splitRef} total={splitWidth} fraction={previewFraction} onChange={setPreviewFraction} controls={selectedRun ? 'time-skip-content' : planShown ? 'plan-content' : 'artifact-content'}/> : null;
    const runCount = timeSkipRuns.length;
    const previewCount = session?.messages.reduce((n, m) => n + (m.artifacts?.length ?? 0), 0) ?? 0;
    useEffect(() => { const items = [...new Map((session?.messages.flatMap(m => m.artifacts ?? []) ?? []).map(a => [a.id, a])).values()]; setPreviews(items.filter(a => !session?.closedPreviews?.includes(a.id))); setSelectedPreview(session && previewsDeferred(session.id) || session?.closedPreviews?.includes(session.previewOpen ?? '') ? null : session?.previewOpen?.startsWith('time-skip:') && !timeSkipRuns.some(r => timeSkipTabId(r.id) === session.previewOpen) ? null : session?.previewOpen ?? null); }, [session?.id]);
    useEffect(() => {
        // A message's copy of a file never replaces one that was read later (lib/preview-merge.ts); nothing changed = the same array = no update.
        const items = messageArtifacts(session?.messages);
        if (items.length)
            setPreviews(old => syncArtifacts(old, items, session?.closedPreviews));
    }, [previewCount]);
    const localUrls = useRef<string[]>([]);
    useEffect(() => {
        const id = session?.previewOpen;
        if (id?.startsWith('time-skip:') && timeSkipRuns.some(r => timeSkipTabId(r.id) === id) && !session?.closedPreviews?.includes(id) && !(session && previewsDeferred(session.id)))
            setSelectedPreview(id);
    }, [session?.id, runCount]);
    useEffect(() => () => localUrls.current.forEach(url => URL.revokeObjectURL(url)), []);
    function openPreview(item: PreviewArtifact) {
        endPreviewDeferral();
        if (item.url?.startsWith('blob:'))
            localUrls.current.push(item.url);
        setPreviewTabClosed(session?.id, item.id, false, item.id);
        // The file was read just now: it replaces the tab of the same id when its content differs (it used to be dropped, keeping the first copy).
        setPreviews(old => openItem(old, item));
        setSelectedPreview(item.id);
    }
    /* A run that finishes with a file the sidebar can show opens its main result there by itself (one file, never several). Only a run seen
       finishing here does it: opening an old chat shows nothing new. A file that cannot be read just stays a card with Open. */
    const replyWasStreaming = useRef<Map<string, boolean>>(new Map());
    const lastReply = [...(session?.messages ?? [])].reverse().find(m => m.role === 'assistant');
    useEffect(() => {
        if (!session || !lastReply) return;
        const before = replyWasStreaming.current.get(lastReply.id);
        replyWasStreaming.current.set(lastReply.id, !!lastReply.streaming);
        if (before !== true || lastReply.streaming || lastReply.error) return;
        const main = mainResult(lastReply.deliverables, lastReply.content);
        // No file to open: one whole HTML page in the answer goes to the sidebar too (a snippet keeps its preview button).
        if (!main) { const page = lastReply.blocks?.some(b => b.kind === 'group') ? null : mainCodePreview(lastReply.id, lastReply.content); if (page) openPreview(page); return; }
        const sessionId = session.id;
        void previewOf(sessionId, main).then(item => { if (session.id === sessionId) openPreview(item); }, () => { });
    }, [lastReply?.id, lastReply?.streaming]);
    useEffect(() => { setSelectedPreviewId(setupVisible ? null : selectedPreview); return () => setSelectedPreviewId(null); }, [selectedPreview, setupVisible]);
    function closeTab(id: string) {
        const next = closePreviewTab([...previews.map(p => ({ id: p.id })), ...extraTabs], selectedPreview, id);
        setPreviews(previews.filter(p => p.id !== id));
        setSelectedPreview(next.selected);
        setPreviewTabClosed(session?.id, id, true, next.selected);
        if (!next.items.length) {
            setExpanded(false);
            requestAnimationFrame(() => textarea.current?.focus());
        }
    }
    function selectPreview(id: string) {
        endPreviewDeferral();
        setSelectedPreview(id);
        setPreviewTabClosed(session?.id, id, false, id);
    }
    function openTimeSkip(id: string) { selectPreview(timeSkipTabId(id)); }
    function openLink(url: string) {
        const item = artifactFromUrl(url);
        if (!item)
            return;
        // A web address opens in the person's own browser; only a file (a PDF, picture, table, code or Markdown) is shown in the side panel (#4721).
        if (item.kind === 'website') {
            void openExternal(url);
            return;
        }
        openPreview(item);
    }
    /* The message list is memoised (#4640 family: Out of Memory in a long chat): a streamed token changes one message, so the others must not
       re-render. These handlers keep one identity for the life of the screen and always call the newest functions above. */
    const latestHandlers = useRef({ editPrompt, openPreview, closeTab, openTimeSkip, openLink });
    latestHandlers.current = { editPrompt, openPreview, closeTab, openTimeSkip, openLink };
    const messageHandlers = useMemo(() => ({
        onEditPrompt: (text: string) => latestHandlers.current.editPrompt(text),
        onPreview: (a: PreviewArtifact) => latestHandlers.current.openPreview(a),
        onClosePreview: (id: string) => latestHandlers.current.closeTab(id),
        onOpenTimeSkip: (id: string) => latestHandlers.current.openTimeSkip(id),
        onOpenLink: (url: string) => latestHandlers.current.openLink(url),
    }), []);
    const scroller = useRef<HTMLDivElement>(null);
    const focusMode = useStore((s) => s.focus);
    /** Per-conversation scroll offsets, so returning to one restores the view. */
    const scrollPositions = useRef<Map<string, number>>(new Map());
    /* Scrolling back to reread something is normal; being stranded there while a
       reply arrives is not. The pill only appears once you are far enough up that
       the newest message is genuinely off screen. */
    const [awayFromBottom, setAwayFromBottom] = useState(false);
    /* A reply that finishes while you are reading further up has no way to say so
       — the text simply stops arriving off screen. The pill pulses once, on
       opacity alone, so the cue is noticed without anything moving. */
    const [justFinished, setJustFinished] = useState(false);
    const wasStreaming = useRef(false);
    useEffect(() => {
        if (wasStreaming.current && !streaming) {
            setJustFinished(true);
            const t = setTimeout(() => setJustFinished(false), 700);
            wasStreaming.current = streaming;
            return () => clearTimeout(t);
        }
        wasStreaming.current = streaming;
        return undefined;
    }, [streaming]);
    /** Whether the reader is following the newest message rather than reading back. */
    const pinned = useRef(true);
    /** When the reader last used the wheel, a finger, a key or the scrollbar on the transcript. */
    const readerInput = useRef(0);
    const seenMessages = useRef(0);
    /**
     * Scrolls to the newest message and stays there.
     *
     * `content-visibility: auto` means the transcript's height is an estimate
     * until each turn renders, so a single `scrollTo(scrollHeight)` lands short
     * and the reader ends up above the reply they asked to see. Re-aiming until
     * the height stops growing settles it; the loop is bounded because a
     * transcript that never settles must not spin.
     */
    const scrollToLatest = useCallback((smooth = true) => {
        const el = scroller.current;
        if (!el)
            return;
        pinned.current = true;
        el.scrollTo({ top: el.scrollHeight, behavior: smooth ? scrollBehavior() : 'auto' });
        setAwayFromBottom(false);
    }, []);
    /**
     * Keeps a pinned transcript pinned.
     *
     * With `content-visibility: auto`, an off-screen turn has an *estimated*
     * height until it renders, so the scroll height grows as you approach the
     * bottom and any single scroll lands short. Watching the content box and
     * re-aiming while the reader is pinned is what makes the two compatible —
     * and it fixes the same drift while a long reply streams in.
     */
    useEffect(() => {
        const el = scroller.current;
        // The content, not the scroller's first child — which is the reading
        // progress hairline, and never changes size.
        const inner = el?.querySelector('.chat__inner');
        if (!el || !inner)
            return undefined;
        const ro = new ResizeObserver(() => {
            if (pinned.current && readerHasSelection(el)) {
                // Words are selected: the view stays where they are (#3818); the way back to the newest message is offered.
                pinned.current = false;
                setAwayFromBottom(distanceFromBottom(el) > PILL_DISTANCE);
            }
            if (pinned.current)
                el.scrollTop = el.scrollHeight;
        });
        ro.observe(inner);
        // The scroller itself shrinks when a question appears above the message box (the approval dock): keep the end in view then too.
        ro.observe(el);
        return () => ro.disconnect();
    }, [session?.id, session?.messages.length === 0]);
    const textarea = useRef<HTMLTextAreaElement>(null);
    const remote = useStore(s => s.remoteModel), providers = useStore(s => s.providers);
    const apiModel = providers?.connections.find(c => c.id === remote?.connectionId)?.models.find(m => m.id === remote?.modelId);
    const loaded = apiModel ?? library?.variants.find((v) => v.id === library.loaded) ?? null;
    /**
     * Follow the stream only when the reader is already at the bottom. Scrolling
     * back through a long conversation should not be yanked forward by the next
     * token — which is the single most irritating thing a chat UI can do over a
     * long session.
     */
    useEffect(() => {
        const el = scroller.current;
        if (!el || (session?.showcase && !streaming))
            return;
        // The reader's own scrolling decides (#4014): a wheel notch is about 100 px and used to land inside a 120 px band that pulled
        // the view back on every token. `pinned` is set by the scroller's wheel, touch, key and scroll handlers below.
        const count = session?.messages.length ?? 0;
        const sent = count > seenMessages.current && session?.messages.at(-1)?.role === 'user';
        seenMessages.current = count;
        // Sending a message always brings the reader to it, as it did before; only the answer's growth waits for the reader.
        if (pinned.current && !sent && readerHasSelection(el)) {
            pinned.current = false;
            setAwayFromBottom(distanceFromBottom(el) > PILL_DISTANCE);
        }
        if (pinned.current || sent)
            scrollToLatest();
    }, [session?.messages.length, session?.messages.at(-1)?.content, scrollToLatest]);
    // Scroll position is remembered per conversation, so returning to one puts you
    // back where you were reading rather than at the top or the bottom.
    useEffect(() => {
        const el = scroller.current;
        if (!el || !session)
            return;
        const saved = scrollPositions.current.get(session.id);
        el.scrollTop = saved ?? (session.showcase ? 0 : el.scrollHeight);
        pinned.current = !session.showcase || saved !== undefined;
        // Deliberately keyed on the conversation only: re-running on every message
        // would fight the follow-the-stream effect above.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session?.id]);
    // The box grows with the draft but never past 30% of the window, and scrolls inside after that, so Send and the toolbar stay on screen (QA F08).
    useEffect(() => {
        const fit = () => {
            const el = textarea.current;
            if (!el)
                return;
            el.style.height = 'auto';
            el.style.height = `${Math.min(composerCap(window.innerHeight), el.scrollHeight)}px`;
        };
        fit();
        window.addEventListener('resize', fit);
        return () => window.removeEventListener('resize', fit);
    }, [draft]);
    /** Hands a rewound prompt back to the composer so it can be changed. */
    function editPrompt(text: string) {
        setDraft(text);
        requestAnimationFrame(() => textarea.current?.focus());
    }
    /**
     * Turn navigation, for reading rather than writing.
     *
     * Focus mode is for reading a long conversation, and reading needs a next.
     * `Alt+↑`/`Alt+↓` step between your own turns — the boundaries a reader
     * actually navigates by, rather than by message — and `Cmd+↓` returns to the
     * latest. Only bound while focus mode is on, so the keys stay free elsewhere.
     */
    useEffect(() => {
        if (!focusMode)
            return undefined;
        const onKey = (e: KeyboardEvent) => {
            const el = scroller.current;
            if (!el)
                return;
            const atLatest = () => {
                pinned.current = true;
                el.scrollTo({ top: el.scrollHeight, behavior: scrollBehavior() });
            };
            if ((e.metaKey || e.ctrlKey) && e.key === 'ArrowDown') {
                e.preventDefault();
                atLatest();
                return;
            }
            if (!e.altKey || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown'))
                return;
            e.preventDefault();
            const turns = [...el.querySelectorAll<HTMLElement>('.st-msg--user')];
            if (!turns.length)
                return;
            const top = el.scrollTop;
            /* The tolerance has to exceed TURN_OFFSET, or the key sticks on the turn
               it just moved to: positioning a turn at `scrollTop + 16` leaves it
               satisfying `offsetTop > scrollTop + 8` on the next press. */
            const TURN_OFFSET = 16;
            const seen = TURN_OFFSET + 24;
            const next = e.key === 'ArrowDown'
                ? turns.find((t) => t.offsetTop > top + seen)
                : [...turns].reverse().find((t) => t.offsetTop < top - seen);
            if (next) {
                pinned.current = false;
                el.scrollTo({ top: Math.max(0, next.offsetTop - TURN_OFFSET), behavior: scrollBehavior() });
                setAwayFromBottom(true);
            }
            else if (e.key === 'ArrowDown') {
                atLatest();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [focusMode]);
    const chatNative = useNative(), queuedPhotos = useChatPhotos(session?.id);
    const readyForChat = !!loaded || !!chatNative?.instances.some(i => i.status === 'ready');
    useEffect(() => {
        if (setupVisible || !readyForChat || messageCount)
            return;
        if (scroller.current?.querySelector('.first-chat-start'))
            textarea.current?.focus({ preventScroll: true });
    }, [setupVisible, readyForChat, session?.id, messageCount]);
    function submit() {
        const text = draft.trim();
        if (!text && !queuedPhotos.length)
            return;
        if (streaming) {
            // Typed while a goal runs: it waits in the queue, shown under the conversation. With no goal running there is nothing to queue behind.
            if (!text || !session)
                return;
            const plan = busyPlan(whileBusy);
            const out = quotedMessage(quotes, text);
            // Interrupt: the answer stops where it is (what is written stays) and the message goes as soon as it has stopped (the effect below sends it).
            if (plan === 'interrupt') {
                updateDraft('');
                clearQuotes(session.id);
                parkMessage(session.id, out);
                stopStreaming();
                return;
            }
            updateDraft('');
            void queueMessage(session.id, out).then(ok => { if (ok) { clearQuotes(session.id); return; } if (!ok) { if (plan === 'goal') { clearQuotes(session.id); parkMessage(session.id, out); return; } updateDraft(text); pushToast('info', tr('The assistant is still answering'), tr('Send your message when it has finished.')); } }, e => { updateDraft(text); pushToast('error', tr('The message was not queued'), plainFailureText(e instanceof Error ? e.message : String(e)).message); });
            return;
        }
        const timeSkip = timeSkipPrompt(text);
        if (timeSkip === '') {
            pushToast('info', 'Add a prompt after /time-skip');
            return;
        }
        if (timeSkip === null) {
            // No model to answer: say so plainly (or show the download), and do not start a run.
            const why = noModelNotice(chatNative ?? null, { libraryLoaded: !!loaded });
            if (why) { pushToast('info', tr(why.title), tr(why.message)); return; }
        }
        updateDraft('');
        if (session) clearQuotes(session.id);
        void sendMessage(session ? quotedMessage(quotes, text) : text);
    }
    function onComposerKey(_e: React.KeyboardEvent) { }
    if (!session) {
        return (<>
        <ChatToolbar previewWidth={paneWidth} extraTabs={extraTabs} session={session} items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }} inline={inlinePreview} expanded={expanded} onExpand={() => setExpanded(v => !v)}/>
        <div className="chatsplit" style={splitStyle} data-timeskip={!!selectedRun || undefined} ref={splitRef} data-preview={!!selectedPreview && inlinePreview || undefined} data-expanded={expanded && !!selectedPreview && inlinePreview || undefined}>
         <DropZone className="chatcol">
          <div className="main__body chat" ref={scroller}>
            {tr(awayFromBottom ? (<button type="button" className="jumplatest" data-finished={justFinished || undefined} aria-label={tr("Scroll to the latest message")} title={tr("Scroll to the latest message")} onClick={() => scrollToLatest()}>
                <Icon name="chevron" size={18}/></button>) : null)}
            <div className="chat__inner">
              <UsageDashboard loaded={!!loaded}/>
            </div>
          </div>
          <Composer draft={draft} setDraft={updateDraft} loaded={!!loaded} streaming={streaming} onSubmit={submit} onKey={onComposerKey} session={session} onPreview={openPreview} textareaRef={textarea}/>
         </DropZone>
        <ArtifactPanel items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => setSelectedPreview(null)} inline={inlinePreview}/>
         {tr(resizeHandle)}
        </div>
      </>);
    }
    return (<>
      {tr(selection && session ? <button type="button" className="quote-selection" style={{ left: selection.x, top: selection.y }} onMouseDown={e => e.preventDefault()} onClick={() => { addQuote(session.id, selection.text, selection.messageId, selection.role); window.getSelection()?.removeAllRanges(); setSelection(null); }}>{tr("Quote")}</button> : null)}
      <ChatToolbar previewWidth={paneWidth} extraTabs={extraTabs} session={session} items={previews} onCloseTab={closeTab} selected={setupVisible ? null : selectedPreview} onSelect={selectPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }} inline={inlinePreview} expanded={expanded} onExpand={() => setExpanded(v => !v)}/>

      {/* The composer belongs to the conversation column, not the window: with
                the context panel open, a full-width composer sat underneath the panel
                and drifted out of alignment with the transcript it belongs to. */}
      <div className="chatsplit" style={splitStyle} data-timeskip={!!selectedRun || undefined} ref={splitRef} data-preview={!!selectedPreview && inlinePreview || undefined} data-expanded={expanded && !!selectedPreview && inlinePreview || undefined}>
        <DropZone className="chatcol">
          <div className="main__body chat" ref={scroller} onScroll={(e) => {
            const el = e.currentTarget;
            if (session)
                scrollPositions.current.set(session.id, el.scrollTop);
            // How far you are from the newest message, which decides whether
            // the way back is worth offering.
            // The threshold has to tolerate the height estimate moving under
            // us, or the pill flickers as turns render during a scroll.
            const distance = distanceFromBottom(el);
            pinned.current = followAfterScroll(distance, pinned.current, Date.now() - readerInput.current);
            setAwayFromBottom(distance > PILL_DISTANCE);
        }} onWheel={(e) => {
            readerInput.current = Date.now();
            if (wheelLeavesBottom(e.deltaY) && e.currentTarget.scrollTop > 0)
                pinned.current = false;
        }} onTouchMove={() => { readerInput.current = Date.now(); }} onPointerDown={() => { readerInput.current = Date.now(); }} onKeyDown={(e) => {
            readerInput.current = Date.now();
            if (KEYS_UP.has(e.key) && e.target === e.currentTarget && e.currentTarget.scrollTop > 0)
                pinned.current = false;
        }}>

            <div className="chat__inner">
              {tr(session.messages.length === 0 ? (session.statsAvailable === false ? <p className="small muted">{tr("Editing from an earlier message. Update the prompt below.")}</p> :
            <UsageDashboard loaded={!!loaded}/>) : (session.messages.map((m, i) => (<Message key={m.id} message={m} isLast={i === session.messages.length - 1} onEditPrompt={messageHandlers.onEditPrompt} onPreview={messageHandlers.onPreview} onClosePreview={messageHandlers.onClosePreview} onOpenTimeSkip={messageHandlers.onOpenTimeSkip} timeSkipDone={!!allTimeSkipRuns.find(r => r.id === m.timeSkipId)?.completedAt} timeSkipReal={!!allTimeSkipRuns.find(r => r.id === m.timeSkipId)?.job} timeSkipRun={allTimeSkipRuns.find(r => r.id === m.timeSkipId)} onOpenLink={messageHandlers.onOpenLink}/>))))}
              <QueuedMessages chatId={session.id}/>
            </div>
          </div>
          <div className="jumplatest-row">{tr(awayFromBottom ? (<button type="button" className="jumplatest" data-finished={justFinished || undefined} aria-label={tr("Scroll to the latest message")} title={tr("Scroll to the latest message")} onClick={() => scrollToLatest()}>
                <Icon name="chevron" size={18}/></button>) : null)}</div>
          <Composer draft={draft} setDraft={updateDraft} loaded={!!loaded} streaming={streaming} onSubmit={submit} onKey={onComposerKey} session={session} onPreview={openPreview} textareaRef={textarea}/>
        </DropZone>
        {tr(!setupVisible && selectedRun && session ? <Suspense fallback={null}><TimeSkipPanel key={selectedRun.id} run={selectedRun} sessionId={session.id} inline={inlinePreview} expanded={expanded} onCompare={setExpanded} onPreview={openPreview} onClose={() => { setSelectedPreview(null); setExpanded(false); }}/></Suspense> : null)}
        {tr(!setupVisible && planShown ? <PlanPanel chatId={session?.id}/> : null)}
        <ArtifactPanel items={previews} onCloseTab={closeTab} selected={setupVisible || planShown ? null : selectedPreview} onSelect={selectPreview} onClose={() => setSelectedPreview(null)} inline={inlinePreview}/>
        {tr(resizeHandle)}
      </div>
    </>);
}
interface SlashCommand {
    id: string;
    label: string;
    hint: string;
    icon: IconName;
    run: () => void;
}
function useSlashCommands(clear: () => void, setDraft: (value: string) => void): SlashCommand[] {
    const demo = useStore(s => s.library?.mode === 'simulated');
    return [
        { id: 'time-skip', label: '/time-skip', hint: 'Try a change on a copy, check it, and keep it only if it is better', icon: 'branch', run: () => setDraft('/time-skip ') },
        { id: 'goal', label: '/goal', hint: 'Set a goal and let the assistant keep working until it is done', icon: 'bolt', run: () => setDraft('/goal ') },
        { id: 'plan', label: '/plan', hint: 'Have the assistant write a plan first; nothing runs until you approve it', icon: 'check', run: () => setDraft('/plan ') },
        { id: 'research', label: '/research', hint: 'Search the web, read the pages and answer with sources', icon: 'search', run: () => setDraft('/research ') },
        { id: 'loop', label: '/loop', hint: 'Repeat a message every few minutes while this chat is open', icon: 'refresh', run: () => setDraft('/loop 10m ') },
        { id: 'website', label: '/website', hint: 'Open a website in the preview sidebar', icon: 'link', run: () => { clear(); window.dispatchEvent(new Event('strata:open-website')); } },
        { id: 'preview', label: '/preview', hint: 'Open a local file in the preview sidebar', icon: 'file', run: () => { clear(); window.dispatchEvent(new Event('strata:open-preview')); } },
        ...(demo ? [{ id: 'subagents', label: '/subagents', hint: 'Preview subagent statistics', icon: 'experts' as const, run: () => { clear(); createSubagentExample(); } }] : []),
        ...getSlashCommands().map(command => ({ ...command, run: () => { clear(); command.run(); } }))
    ];
}
function SlashMenu({ query, commands, cursor, onPick, onHover, }: {
    query: string;
    commands: SlashCommand[];
    cursor: number;
    onPick: (c: SlashCommand) => void;
    onHover: (i: number) => void;
}) {
    const matches = commands.filter((c) => c.label.slice(1).startsWith(query.toLowerCase()));
    if (!matches.length) {
        return (<div className="slashmenu">
        <div className="slashmenu__empty small muted">{tr("No command matches \u201C/")}{query}”</div>
      </div>);
    }
    return (<div className="slashmenu" role="listbox" aria-label={tr("Commands")}>
      <div className="slashmenu__label">{tr("Commands")}</div>
      {tr(matches.map((c, i) => (<button key={c.id} type="button" role="option" aria-selected={i === cursor} className="slashmenu__item" data-active={i === cursor} onMouseEnter={() => onHover(i)} onMouseDown={e => e.preventDefault()} onClick={() => onPick(c)}>
          <Icon name={c.icon} size={16}/>
          <span className="slashmenu__name">{tr(c.label)}</span>
          <span className="slashmenu__hint">{tr(c.hint)}</span>
        </button>)))}
    </div>);
}
const NO_PARKED: string[] = [];
/** Whether the reader has words selected inside the transcript: a view that moves under the pointer loses the selection (#3818). */
function readerHasSelection(scroller: HTMLElement): boolean {
    const sel = window.getSelection();
    return !!sel && !sel.isCollapsed && !!sel.anchorNode && scroller.contains(sel.anchorNode) && sel.toString().trim().length > 0;
}
const NO_QUOTES: Quote[] = [];
/** The message that goes out: the quoted passages in their fixed form, then the person's words. A command such as /plan goes out as typed. */
function quotedMessage(quotes: readonly Quote[], text: string): string {
    return quotes.length && !text.startsWith('/') ? composeQuoted(quotes.map(q => q.text), text) : text;
}
function Composer({ draft, setDraft, loaded, streaming, onSubmit, onKey, session, onPreview, textareaRef, }: {
    draft: string;
    setDraft: (v: string) => void;
    loaded: boolean;
    streaming: boolean;
    onSubmit: () => void;
    onKey: (e: React.KeyboardEvent) => void;
    session: Session | null;
    onPreview: (a: PreviewArtifact) => void;
    textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}) {
    const clearDraft = useCallback(() => setDraft(''), [setDraft]);
    const photos = useChatPhotos(session?.id), native = useNative();
    const canChat = loaded || native?.instances.some(i => i.status === 'ready');
    const whileBusy = useStore(s => s.whileBusy), parkedList = useStore(s => (session ? s.parked[session.id] : undefined)) ?? NO_PARKED;
    const quoteList = useStore(s => (session ? s.quotes[session.id] : undefined)) ?? NO_QUOTES;
    const commands = useSlashCommands(clearDraft, setDraft);
    const [cursor, setCursor] = useState(0);
    const [fileError, setFileError] = useState('');
    const [slashOpen, setSlashOpen] = useState(false);
    const [slashDismissed, setSlashDismissed] = useState(false);
    const slashHostRef = useRef<HTMLDivElement>(null);
    const ime = useRef(createImeGuard());
    const slashQuery = slashDismissed ? null : slashOpen ? '' : draft.startsWith('/') && !/\s/.test(draft) ? draft.slice(1) : null;
    const matches = slashQuery === null ? [] : commands.filter((c) => c.label.slice(1).startsWith(slashQuery.toLowerCase()));
    useEffect(() => { setSlashDismissed(false); }, [draft, session?.id]);
    useEffect(() => {
        if (slashQuery === null)
            return;
        const outside = (event: PointerEvent) => {
            if (slashHostRef.current?.contains(event.target as Node) || event.target === textareaRef.current)
                return;
            setSlashOpen(false);
            setSlashDismissed(true);
        };
        document.addEventListener('pointerdown', outside);
        return () => document.removeEventListener('pointerdown', outside);
    }, [slashQuery, textareaRef]);
    useEffect(() => {
        setCursor(0);
    }, [slashQuery]);
    return (<div className="composerwrap">
      <div className="composer">
        <div className="composer__inner">
          <ApprovalDock sessionId={session?.id}/>
          {tr(fileError ? <ErrorNotice message={fileError} action={tr("Dismiss")} onRetry={() => setFileError('')}/> : null)}
          <FolderChip chatId={session?.id} disabled={streaming}/>
          <GoalBar chatId={session?.id}/>
          {tr(quoteList.length ? <div className="quotes">{tr(quoteList.map(q => <div className="quote" key={q.id}><button type="button" className="quote__body" title={tr("Go to the quoted message")} onClick={() => document.querySelector(`[data-message-id="${CSS.escape(q.messageId)}"]`)?.scrollIntoView({ block: 'center', behavior: scrollBehavior() })}><span className="xs muted">{tr(q.role === 'user' ? "Quoting you" : "Quoting the assistant")}</span><span className="quote__text">{q.text}</span></button><IconBtn name="x" size={14} title={tr("Remove this quote")} onClick={() => removeQuote(session!.id, q.id)}/></div>))}</div> : null)}
          {tr(parkedList.length ? <div className="parked" role="status">{tr(parkedList.map((text, i) => <div className="parked__item" key={`${i}-${text}`}><span className="xs muted">{tr("Waiting to send")}</span><span className="parked__text" title={text}>{text}</span><IconBtn name="x" size={14} title={tr("Take back")} onClick={() => { const back = takeParked(session!.id, i); if (back !== undefined && !draft.trim()) setDraft(back); }}/></div>))}</div> : null)}
          <div className="st-composer" style={{ boxShadow: 'var(--st-sh-bar)' }}>
          {tr(photos.length ? <ChatPhotos photos={photos} onRemove={id => removeChatPhoto(id, session?.id)}/> : null)}
          <textarea ref={textareaRef} rows={1} value={draft} aria-label={tr("Message Strata")} placeholder={tr(canChat ? 'Message Strata…' : 'Try /time-skip, or load a model to chat…')} onFocus={() => setSlashDismissed(false)} onBlur={() => ime.current.reset()} onCompositionStart={() => ime.current.compositionStart()} onCompositionEnd={(e) => ime.current.compositionEnd(e.timeStamp)} onPaste={(e) => {
            // A picture on the clipboard (a screenshot, a copied image) is attached like a dropped file; text pastes as usual.
            const files = pastedFiles(e.clipboardData);
            if (!files.length)
                return;
            e.preventDefault();
            window.dispatchEvent(new CustomEvent(DROP_EVENT, { detail: files }));
        }} onChange={(e) => { setSlashOpen(false); setSlashDismissed(false); setDraft(e.target.value); e.currentTarget.style.height = "auto"; e.currentTarget.style.height = `${Math.min(180, e.currentTarget.scrollHeight)}px`; }} onKeyDown={(e) => {
            // A key that belongs to the IME (the Enter that commits a conversion, in the order each engine delivers it) is not ours (#5071).
            if (ime.current.keyDown({ key: e.key, isComposing: e.nativeEvent.isComposing, keyCode: e.nativeEvent.keyCode }, e.timeStamp))
                return;
            // The command menu owns the keyboard while it is open.
            if (slashQuery !== null && e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                setSlashOpen(false);
                setSlashDismissed(true);
                return;
            }
            if (slashQuery !== null && matches.length) {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setCursor((c) => (c + 1) % matches.length);
                    return;
                }
                if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setCursor((c) => (c - 1 + matches.length) % matches.length);
                    return;
                }
                if (e.key === 'Enter' || e.key === 'Tab') {
                    e.preventDefault();
                    setSlashOpen(false);
                    setSlashDismissed(true);
                    matches[cursor].run();
                    return;
                }
            }
            onKey(e);
            if (composerSends(e)) {
                e.preventDefault();
                onSubmit();
            }
        }}/>
          <div className="st-composer__bar">
            <ComposerControls draft={draft} onDraft={setDraft} session={session} onPreview={onPreview} streaming={streaming} onSlash={() => { setSlashDismissed(false); setSlashOpen(true); requestAnimationFrame(() => textareaRef.current?.focus()); }} onError={setFileError} onAttach={(name, text) => { setDraft(`${draft}${draft ? '\n\n' : ''}Attached ${name}\n\n${text}`); requestAnimationFrame(() => textareaRef.current?.focus()); }}/>
            {tr(streaming ? (<Btn variant="secondary" size="sm" icon="stop" onClick={stopStreaming}>{tr("Stop")}</Btn>) : (<Btn variant="primary" size="sm" icon="send" disabled={(!draft.trim() && !photos.length) || draft.trim() === '/' || (!canChat && timeSkipPrompt(draft) === null)} onClick={onSubmit}>{tr("Send")}</Btn>))}
          </div>
        </div>
        {tr(streaming && draft.trim() && whileBusy !== 'wait' ? <p className="xs muted composer__busy-hint" role="status">{tr(whileBusy === 'queue' ? "Enter sends it when the answer is done." : "Enter stops the answer here and sends it now.")}</p> : null)}
        <ChatTelemetry session={session}/>
        </div>
      </div>
      {/* Outside the composer's glass on purpose: a material sampling another
                material blurs an already-blurred backdrop, which Liquid Glass
                prohibits and which doubles the compositor cost for a muddier result.
                The host repeats the composer's padding so the menu keeps its column. */}
      {tr(slashQuery !== null ? (<div className="slashmenu-host" ref={slashHostRef}>
          <SlashMenu query={slashQuery} commands={commands} cursor={cursor} onHover={setCursor} onPick={(c) => {
                setSlashOpen(false);
                setSlashDismissed(true);
                c.run();
            }}/>
        </div>) : null)}
    </div>);
}
const Message = memo(MessageView);
function MessageView({ message: m, isLast, onEditPrompt, onPreview, onClosePreview, onOpenLink, onOpenTimeSkip, timeSkipDone, timeSkipReal, timeSkipRun, }: {
    message: ChatMessage;
    isLast: boolean;
    onEditPrompt: (text: string) => void;
    onPreview: (a: PreviewArtifact) => void;
    onClosePreview: (id: string) => void;
    onOpenLink: (url: string) => void;
    onOpenTimeSkip: (id: string) => void;
    timeSkipDone: boolean;
    timeSkipReal?: boolean;
    timeSkipRun?: TimeSkipRun;
}) {
    const streaming = useStore(s => s.streaming);
    const activeSession = useStore(s => s.activeSession);
    const hostLive = useStore(s => s.connection !== 'offline');
    const liveStatus = useLiveStatus(m.role === 'assistant' && m.streaming ? activeSession : null);
    // A reply that proposed a plan shows the plan card in place of its closing text (which is the plan in prose).
    const storedPlan = usePlanQuiet(m.planId ? activeSession : null);
    const planCard = !!m.planId && storedPlan?.id === m.planId;
    // A reply with activity lines is drawn piece by piece in the order it came; one without is the plain text it always was.
    const grouped = !!m.blocks?.some(b => b.kind === 'group');
    const [sentTime, setSentTime] = useState(() => relTime(m.at));
    const sentQuote = m.role === 'user' ? parseQuoted(m.content) : null;
    if (m.role === 'user') {
        return (<div className="st-msg st-msg--user" data-message-id={m.id} onMouseEnter={() => setSentTime(relTime(m.at))} onFocus={() => setSentTime(relTime(m.at))}>
        <div className="st-bubble">{tr(m.photos?.length ? <ChatPhotos photos={m.photos}/> : null)}{sentQuote ? <><SentQuotes quotes={sentQuote.quotes}/><span className="st-bubble__text">{sentQuote.reply}</span></> : m.markdown ? <Markdown text={m.content} onOpenLink={onOpenLink} imageSession={activeSession}/> : <span className="st-bubble__text">{m.content}</span>}</div>
        <div className="st-msg__meta">
          <time dateTime={m.at} title={tr(new Date(m.at).toLocaleString(localeTag()))}>{tr(sentTime)}</time>
          <span className="st-msg__actions">
            <CopyBtn text={m.content} label="Message" title="Copy message"/>
            <IconBtn name="refresh" title={tr("Rewind to here")} disabled={streaming} onClick={() => {
                const prompt = rewindTo(m.id);
                if (prompt !== null)
                    onEditPrompt(prompt);
            }}/>
            <MessageMenu label={tr("More message actions")}><button role="menuitem" className="menu__item" onClick={() => activeSession && addQuote(activeSession, m.content, m.id, 'user')}>{tr("Quote")}</button><button role="menuitem" className="menu__item" onClick={() => branchFrom(m.id)}><Icon name="branch"/>{tr("Branch from here")}</button></MessageMenu>
          </span>
        </div>
      </div>);
    }
    return (<div className="st-msg st-msg--assistant" data-message-id={m.id} data-error={m.error ? 'true' : undefined}>
      {tr(m.reasoning ? <ReasoningBlock message={m}/> : null)}


      {/* In a reply with activity lines the line is the one live cue for a running or finished tool (and the status line below says what it is doing now); the
          card stays only for what needs the person: a question waiting, a denied or failed call. A card that came and went above the text moved the whole reply. */}
      {tr((m.tools ?? []).filter(t => !grouped || t.state === 'proposed' || t.state === 'denied' || t.state === 'error').map((t) => (<ToolCard key={t.id} tool={t}/>)))}

      {tr(!(m.timeSkipId && timeSkipReal && !timeSkipDone) && (m.content || m.error || !m.streaming || !liveStatus || !!m.drafts?.length) ? <div className="st-bubble">
        {tr(planCard ? <PlanCard chatId={activeSession} planId={m.planId!}/> : null)}
        {tr(m.error ? (<ErrorNotice message={m.error} detail={m.errorDetail} {...contextRecovery(m.errorCode)}/>) : null)}
        {tr(grouped && !planCard && !m.timeSkipId ? m.blocks!.map((b, i) => b.kind === 'group' ? <ActivityLine key={b.id} group={b} sessionId={activeSession ?? ''} onPreview={onPreview}/> : <Markdown key={i} text={b.text} onOpenLink={onOpenLink} imageSession={activeSession} previewId={`${m.id}-b${i}`} onPreview={onPreview} onClosePreview={onClosePreview}/>) : null)}
        {tr(m.content && !planCard && !grouped ? (<Markdown text={m.timeSkipId && !timeSkipReal ? hostLive ? tr('This was a sample run made while the app could not reach its host.') : timeSkipDone ? tr('The result is ready. Switch between Original and Updated, or compare them side by side. This example uses sample outputs and simulated checks.') : tr('Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks.') : m.content} onOpenLink={onOpenLink} imageSession={activeSession} previewId={m.id} onPreview={onPreview} onClosePreview={onClosePreview}/>) : !m.error && m.streaming && !liveStatus ? (<span className="typing" aria-label={tr("Generating")}>
            <i />
            <i />
            <i />
          </span>) : null)}
        {/* a file or command the model is writing right now: its name, growing size and the last lines of it (live data from the host, see ToolDraft) */}
        {tr(m.streaming && m.drafts?.length && !planCard ? <DraftLines drafts={m.drafts} sessionId={activeSession ?? ''} onPreview={onPreview}/> : null)}
        {/* the caret follows the text being written; after an activity line it hung on a row of its own and the reply jumped when the text went on */}
        {tr(m.streaming && m.content && !m.drafts?.length && (!grouped || m.blocks!.at(-1)?.kind === 'text') ? <span className="caret" aria-hidden="true"/> : null)}
      </div> : null)}
      {tr(m.skills?.length && !m.streaming ? <SkillsUsed skills={m.skills}/> : null)}
      {tr(m.deliverables?.length && !m.streaming ? <div className="deliverables">{m.deliverables.length >= COMPACT_FILES_FROM ? <FilesSummary files={m.deliverables} answer={m.content} sessionId={activeSession ?? ''} onPreview={onPreview}/> : <>{m.deliverables.map(f => <FileCard key={f.path} file={f} sessionId={activeSession ?? ''} onPreview={onPreview}/>)}<EditedFiles files={m.deliverables} sessionId={activeSession ?? ''} onPreview={onPreview}/></>}</div> : null)}
      {tr(m.streaming && isLast && !m.timeSkipId ? <StatusLine sessionId={activeSession} startedAt={Date.parse(m.at) || Date.now()} tokens={m.tokens}/> : null)}

      {tr(m.timeSkipId && timeSkipReal && !timeSkipDone && timeSkipRun ? <Suspense fallback={null}><TimeSkipWorking run={timeSkipRun} sessionId={activeSession ?? ''}/></Suspense> : m.timeSkipId && (!timeSkipReal || (timeSkipRun?.job?.files?.length ?? 0) > 0) ? <button className="timeskip-chat-card" onClick={() => onOpenTimeSkip(m.timeSkipId!)}><Icon name="branch" size={20}/><span><strong>{tr("Time-skip")}</strong><span>{tr(timeSkipDone ? 'Result ready · compare versions' : 'Revising · follow progress')}</span></span><Icon name="chevron" size={16} className="timeskip-card-arrow"/></button> : null)}
      {tr(m.media?.length ? <ChatMedia items={m.media} onPreview={onPreview}/> : null)}
      {tr(m.artifacts?.length ? <div className="artifact-cards">{tr(m.artifacts.map(a => <ArtifactCard key={a.id} artifact={a} onPreview={onPreview}/>))}</div> : null)}

      {tr(!m.streaming && !m.timeSkipId && (m.content || m.error) ? (<div className="st-msg__meta">
          <span className="st-msg__actions">
            {tr(isLast ? <>
              <CopyBtn text={m.content} label="Response" title="Copy response"/>
              <IconBtn name="refresh" title={tr("Ask again")} onClick={() => void retryLast(m.id)}/>
            </> : null)}
            <MessageMenu label={tr("More response actions")}>
              {tr(!isLast ? <><CopyMenuItem text={m.content} label="Response">Copy response</CopyMenuItem><button role="menuitem" className="menu__item" onClick={() => void retryLast(m.id)}>{tr("Ask again")}</button></> : null)}
              {tr(isLast ? <button role="menuitem" className="menu__item" onClick={() => void continueLast()}>{tr("Continue")}</button> : null)}
              <button role="menuitem" className="menu__item" onClick={() => activeSession && addQuote(activeSession, m.content, m.id, 'assistant')}>{tr("Quote")}</button>
              <button role="menuitem" className="menu__item" onClick={() => branchFrom(m.id)}>{tr("Branch from here")}</button>
              {tr(m.tokens || m.tps || m.durationMs ? <p className="xs muted response-stats">{tr(m.tokens ? `${tokens(m.tokens)} tokens` : '')}{tr(m.tps ? ` · ${tps(m.tps)} tok/s` : '')}{tr(m.durationMs ? ` · ${duration(m.durationMs)}` : '')}{tr(m.demo ? ' · Example' : '')}</p> : null)}
            </MessageMenu>
          </span>
        </div>) : null)}
    </div>);
}
/** The passages a sent message quoted, one or two lines each; a click shows the rest. */
function SentQuotes({ quotes }: { quotes: string[] }) {
    const [open, setOpen] = useState(false);
    return <div className="sent-quotes" data-open={open || undefined}>{quotes.map((q, i) => <button type="button" key={i} className="sent-quote" aria-expanded={open} title={tr("Show or hide the rest of the quote")} onClick={() => setOpen(v => !v)}><span className="xs muted">{tr("Quoted")}</span><span className="sent-quote__text">{q}</span></button>)}</div>;
}
function MessageMenu({ label, children }: {
    label: string;
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(false);
    const trigger = useRef<HTMLButtonElement>(null);
    return <><button ref={trigger} className="st-btn st-btn--icon" aria-label={tr(label)} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name="ellipsis" size={16}/></button>{tr(open ? <FloatingMenu anchor={trigger.current} title={tr(label)} onClose={() => setOpen(false)}><MenuCloseContext.Provider value={() => setOpen(false)}><div onClick={() => setOpen(false)}>{tr(children)}</div></MenuCloseContext.Provider></FloatingMenu> : null)}</>;
}
/**
 * A tool call, with the grant it needs. Approvals are per action and per turn —
 * the model never gets standing permission.
 */
function ToolCard({ tool: t }: {
    tool: ToolCall;
}) {
    const scope = SCOPE[t.scope];
    const pending = t.state === 'proposed';
    if (!pending && t.state !== 'running')
        return <details className="disclosure tool-summary"><summary><Icon name="chevron" size={14}/>{tr(t.verb ?? t.name)}{t.target ? <><span>·</span><span className="tool-summary__target" title={t.target}>{t.target}</span></> : null}<span>·</span><Icon name={t.state === 'denied' || t.state === 'error' ? 'x' : 'check'} size={14}/>{tr(t.state === 'denied' ? 'Denied' : t.state === 'error' ? 'Failed' : t.live && t.state === 'done' && t.approved ? 'Approved' : 'Done')}</summary><div><code>{t.preview}</code><p className="small muted">{tr(t.result ?? t.effect)}</p></div></details>;
    /* The question and its two buttons are in the dock above the message box (ApprovalDock), so a long reply cannot push them out of view (#4725). */
    if (pending)
        return <p className="approval-wait small muted" role="status"><Icon name="clock" size={14}/>{tr("Waiting for your answer below")}</p>;
    return <section className="toolcard toolcard--compact" data-state={t.state} aria-label={t.name}>
    <details className="toolcard__disclosure">
      <summary><Icon name={t.scope === 'shell' ? 'bolt' : 'tool'} size={17}/><span className="toolcard__title">{tr(t.verb ?? t.name)}</span><span className="toolcard__scope xs muted">{tr(scope.label)}</span><Icon name="chevron" className="toolcard__chevron" size={15}/></summary>
      <div className="toolcard__body"><code className="toolcard__preview">{t.preview}</code><p className="small muted">{tr(t.effect)}</p>{tr(t.result ? <p className="small">{t.result}</p> : null)}</div>
    </details>
    <span className="xs muted toolcard__running" role="status">{tr("Running\u2026")}</span>
  </section>;
}
export { cx };
