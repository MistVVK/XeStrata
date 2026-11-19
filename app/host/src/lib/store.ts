import { moveSessionToWorkspace, workspaceIdForSession } from './session-workspace';
import { shortTitle } from './title-text';
import { DEFAULT_CONTENT_FS, DEFAULT_UI_SCALE, SIZE_DEFAULTS_VERSION, savedSizes } from './default-sizes';
import { detectLanguage } from './language-detect';
import { validWhileBusy, type WhileBusy } from './while-busy';
import { validReplyWidth, type ReplyWidth } from './reply-width';
import { repairSession } from './session-repair';
import { addToQuotes, takeQuoteSources, type Quote } from './quotes';
import { readAsk, type AgentAnswer, type AgentAsk } from './agent-ask';
import { chatFailure, plainFailureText } from './plain-errors';
import { showcaseMessages, SHOWCASE_AGENTS } from './chat-showcase';
import { settleAgents, updateAgent } from './agents';
import { LOOP_USAGE, parseLoop, type LoopCommand } from './loop';
import { LoopScheduler } from './loop-runner';
import { applyStatusFrame, clearStatus } from './live-status';
import { approvalNeeded, questionAsked, workFinished } from './attention';
import { applyGoalFrame } from './goal';
import { tidyAnswer } from './answer-text';
import { appendText, mergeDeliverable, readDeliverable, readGroup, readSkills, upsertGroup } from './deliverables';
import { applyPlanFrame, applyQueueFrame, planApprove, queueMessage } from './plan';
import { forkFields, isForkJob, TIME_SKIP_DURATION, timeSkipPrompt, timeSkipTabId, withTimeSkipResult, type TimeSkipFile, type TimeSkipJob, type TimeSkipRun, type TimeSkipVersion } from './time-skip';
import { PREVIEW_EXAMPLES, type PreviewArtifact } from './artifacts';
import { lifecyclePolicy } from './lifecycle';
import { newestServingState } from './serving-state';
import { photoData, uploadChatPhotos, photoHistorySlots, pendingChatPhotos, takeChatPhotos } from './chat-photos';
import { readHistoryPictures, lostPicturesNote } from './history-pictures';
import { applyNativeEvent, nativeState } from './native';
import { pictureStatus } from './pictures';
import { noModelNotice } from './no-model';
import { taskHandoverMessage, taskHandoverTitle, chatHandoverMessage, chatHandoverTitle } from './task-handover';
import type { Task } from './tasks';
import { postWithFolderFallback } from './folder-pick';
import { mcpToolsPatch } from './mcp-tools-switch';
import { createSessionWriter, exportSessions, readBrowserSessions, HOST_FLAG_KEY, PENDING_KEY, type SaveResult, type StorageLike } from './session-persist';
import { createHttpTransport, hydrateChats, keepLastMomentCopy, WINDOW_ID } from './chat-sync';
import { copyText } from './clipboard';
import { catalogLoaded, loadCatalog, setTranslationLanguage, tr, validLanguage, type Language } from './i18n';
/**
 * Strata app store.
 *
 * A single immutable state object + a tiny subscribe/emit pair, consumed through
 * useSyncExternalStore. Data comes from the engine host over /api; live updates
 * arrive on the event stream. No state library, no dependencies.
 */
import { useSyncExternalStore } from 'react';
import { api, ApiError, onEvent, streamPost, type StreamEvent } from './client';
import type {
  Workspace,
  ChatMessage,
  ToolCall,
  EngineConfig,
  Hardware,
  ModelLibrary,
  RuntimeModels,
  ImageModelLibrary,
  ImageModelChoice,
  ModelProviders,
  ModelConnectionDraft,
  RemoteModelSelection,
  MonitorState,
  ServingState,
  Session,
  ChatMode,
  SetupState,
  Skill,
  WorkspaceFile,
  ToolDraft,
} from './types';
import { navigate, type Route } from './router';
import { estimate, execution, contextTokens } from './experience';

export interface Toast {
  id: string;
  tone: 'info' | 'success' | 'warn' | 'error';
  title: string;
  body?: string;
  leaving?: boolean;
  /** A button under the text. A toast with one stays until it is dismissed. */
  action?: { label: string; run: () => void };
}

export interface AppState {
  language: Language;
  fontFamily: string;
  configDraft: string | null;
  navOpen: boolean;
  ready: boolean;
  connection: 'connecting' | 'live' | 'offline';
  theme: 'dark' | 'light' | 'system';
  contrast: 'standard' | 'increased';
  rail: boolean;
  /** Distraction-free reading: chrome recedes, the transcript narrows. */
  focus: boolean;
  /** Conversation prose size in px, 10–22. Does not touch the UI chrome. */
  contentFontSize: number;
  density: 'comfortable' | 'compact';
  /** How wide the text of an answer may run (#5066). */
  replyWidth: ReplyWidth;
  motion: 'system' | 'full' | 'reduced';
  /** Translucent materials, or flat surfaces for anyone who prefers them. */
  transparency: 'translucent' | 'opaque';
  /** Chrome scale, independent of the conversation's reading size. */
  uiScale: number;
  /** Price of one kWh for the Monitor's cost line (0 = not set) and the currency the person picked instead of the detected one ('' = automatic) (#3628). */
  electricityPrice: number;
  electricityCurrency: string;
  /** Opt-in: a short sound when a task finishes (#3586). */
  doneSound: boolean;
  /** Opt-in: a desktop notification when Manual mode waits for an approval (#3477). */
  approvalNotify: boolean;
  /** #5993: a message that quotes two or more passages numbers them and names who wrote each (off by default). */
  numberQuotes: boolean;
  /** What Enter does while an answer is being written: wait, send after, or interrupt (#4783). */
  whileBusy: WhileBusy;
  /** What a new conversation starts with (Settings → General): mode, reasoning effort, thinking limit. */
  chatDefaults: ChatDefaults;
  /** Governs anonymous usage reporting. Nothing is sent in this build; the
   *  choice is recorded so a version that does offer it starts from it. */
  shareTelemetry: boolean;
  paletteOpen: boolean;
  shortcutsOpen: boolean;
  settingsSection: 'Appearance' | 'Models' | 'Tools' | 'Storage' | 'Engine' | 'About';
  hardware: Hardware | null;
  library: ModelLibrary | null;
  imageModels: ImageModelLibrary | null;
  runtime: RuntimeModels | null;
  providers: ModelProviders | null;
  remoteModel: RemoteModelSelection | null;
  engine: EngineConfig | null;
  serving: ServingState | null;
  monitor: MonitorState | null;
  setup: SetupState | null;
  skills: Skill[];
  files: WorkspaceFile[];
  workspacePath: string;
  workspaces: Workspace[];
  /** Id of the workspace the file tree is scoped to. */
  activeWorkspace: string | null;
  /** First run, and "run setup again", both present the wizard as a modal. */
  setupOpen: boolean;
  sessions: Session[];
  activeSession: string | null;
  streaming: boolean;
  /** Messages the person sent while an answer was being written, waiting to go when it ends (Send after, Interrupt), by chat (#4783). */
  parked: Record<string, string[]>;
  /** The passages quoted into the message being written, by chat, in the order they were quoted (#4785). */
  quotes: Record<string, Quote[]>;
  /** The questions the agent asked the person and is waiting for, by chat (#4244). */
  agentAsks: Record<string, AgentAsk[]>;
  /** Chats with a turn in flight. `streaming` is about the open chat; this is about all of them. */
  liveChats: number;
  toasts: Toast[];
  /** Files the model has been granted read access to, and when. */
  touchedPaths: Record<string, string>;
  /** The last action that failed, kept until it is retried or dismissed.
   *  A toast disappears after five seconds; a failure the reader has not dealt
   *  with should not. */
  actionError: { message: string; label: string; retry: () => void } | null;
  busy: Record<string, boolean>;
  /** Set while the host cannot write some chats (disk full, no permission); null when every chat is saved (#5192).
   *  The chats stay whole in memory and are tried again at every save; `unsavedSessionsJson()` gives them as text.
   *  A host that does not answer is not this: those saves are simply retried. */
  sessionSave: { failedIds: string[]; reason: 'disk-full' | 'permission' | 'other' } | null;
  /** The chat files the host has a name for but cannot read (a hand-edited file, a truncated one, a format from a later release) (#5421).
   *  They are not chats: nothing is in memory for them. The sidebar lists them so the person can delete one. */
  damagedChats: string[];
  /** Chats saved by a newer Strata: listed with a note to update Strata, never offered for deletion (#5421 review). */
  newerChats: string[];
}

const PREFS_KEY = 'strata.prefs.v2';
/** What the model has been allowed to read, kept across restarts. */
const TOUCHED_KEY = 'strata.touched.v1';
/** Kept one release so downgrading does not lose the theme and sidebar state. */
const PREFS_KEY_V1 = 'strata.prefs.v1';

const CONTENT_FS_MIN = 10;
const CONTENT_FS_MAX = 22;

export interface Prefs {
  /** Which default sizes this save was made with (see SIZE_DEFAULTS_VERSION). */
  sizeDefaults?: number;
  language: Language;
  fontFamily: string;
  theme: 'dark' | 'light' | 'system';
  contrast: 'standard' | 'increased';
  rail: boolean;
  focus: boolean;
  contentFontSize: number;
  density: 'comfortable' | 'compact';
  /** How wide the text of an answer may run (#5066). */
  replyWidth: ReplyWidth;
  motion: 'system' | 'full' | 'reduced';
  transparency: 'translucent' | 'opaque';
  uiScale: number;
  electricityPrice: number;
  electricityCurrency: string;
  doneSound: boolean;
  approvalNotify: boolean;
  numberQuotes: boolean;
  /** What Enter does while an answer is being written: wait, send after, or interrupt (#4783). */
  whileBusy: WhileBusy;
  /** What a new conversation starts with: its mode, reasoning effort and thinking limit. */
  chatDefaults: ChatDefaults;
  /** Governs anonymous usage reporting. Nothing is sent in this build; the
   *  choice is recorded so a version that does offer it starts from it. */
  shareTelemetry: boolean;
}

/** What a new conversation starts with, before the person changes it in that chat's own composer.
 *  These are the values a chat with no preference falls back to today, so leaving them untouched
 *  keeps a new chat exactly as it was. */
export interface ChatDefaults {
  mode: ChatMode;
  /** The reasoning effort name the composer shows ('Off' … 'Max'). */
  thinking: string;
  /** The most tokens one reply may spend thinking; 0 = no limit. */
  thinkingBudget: number;
}

export const CHAT_MODE_CHOICES: ChatMode[] = ['manual', 'plan', 'bypass'];
export const CHAT_EFFORT_CHOICES = ['Off', 'Low', 'Medium', 'High', 'Max'];
export const CHAT_BUDGET_CHOICES: number[] = [0, 2048, 8192, 32768];

export const CHAT_DEFAULTS: ChatDefaults = { mode: 'manual', thinking: 'Medium', thinkingBudget: 0 };

/** Keep only what the composer can actually offer: a hand-edited or older stored value must not leave a chat showing a choice that does nothing. */
function validChatDefaults(value: unknown): ChatDefaults {
  const raw = (value ?? {}) as Partial<ChatDefaults>;
  const thinking = typeof raw.thinking === 'string' && CHAT_EFFORT_CHOICES.includes(raw.thinking) ? raw.thinking : CHAT_DEFAULTS.thinking;
  return {
    mode: CHAT_MODE_CHOICES.includes(raw.mode as ChatMode) ? raw.mode as ChatMode : CHAT_DEFAULTS.mode,
    thinking,
    thinkingBudget: CHAT_BUDGET_CHOICES.includes(Number(raw.thinkingBudget)) ? Number(raw.thinkingBudget) : CHAT_DEFAULTS.thinkingBudget,
  };
}

const PREFS_DEFAULTS: Prefs = {
  language: 'en',
  fontFamily: '',
  theme: 'system',
  contrast: 'standard',
  rail: false,
  focus: false,
  contentFontSize: DEFAULT_CONTENT_FS,
  density: 'comfortable',
  replyWidth: 'comfortable',
  motion: 'system',
  transparency: 'translucent',
  uiScale: DEFAULT_UI_SCALE,
  electricityPrice: 0,
  electricityCurrency: '',
  doneSound: false,
  approvalNotify: false,
  numberQuotes: false,
  whileBusy: 'wait',
  chatDefaults: CHAT_DEFAULTS,
  shareTelemetry: false,
};

function uid(prefix = 'id'): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`;
}

/** localStorage, looked up at each call (it may not exist when this file is loaded outside a browser). */
const browserStorage: StorageLike = {
  getItem: key => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
  removeItem: key => localStorage.removeItem(key),
  key: index => localStorage.key(index),
  get length() { return localStorage.length; },
};
/** A chat as the page works with it: a reply that was streaming when the page went away is marked interrupted, not left running. */
function fixSession(raw: Session): Session {
  const session = repairSession(raw);   // a damaged or older chat file must never take the page down (missing date, title, message text)
  return {
    ...session,
    ...(session.showcase && !session.showcaseAgentStates ? {agents:SHOWCASE_AGENTS,showcaseAgentStates:true}:{}),
    ...(!session.showcase && session.agents ? { agents: settleAgents(session.agents) } : {}),
    messages: (session.messages ?? []).map(message => message.streaming ? {
      ...message,
      streaming: false,
      error: message.error ?? 'Response interrupted. Ask again to retry; the partial answer is kept.',
    } : message),
  };
}

/**
 * The chats shown before the host has answered. Once the browser's chats were handed to the host (the flag) the host's files are the only
 * truth and the page starts empty until it has read them (the boot screen is up until then); before that, and when the host cannot be
 * asked, the browser's old copy is shown. It is only ever read.
 */
function loadSessions(): Session[] {
  try {
    if (browserStorage.getItem(HOST_FLAG_KEY) === '1') return [];
    return readBrowserSessions(browserStorage).sessions.map(fixSession);
  } catch {
    return [];
  }
}

/**
 * v2 adds focus, reading size, density and the motion override. A v1 blob (only
 * theme and rail) is read as-is and upgraded on the next write, so the first run
 * after this change keeps the user's theme instead of resetting it.
 */
function loadPrefs(): Prefs {
  for (const key of [PREFS_KEY, PREFS_KEY_V1]) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) continue;
      const parsed = JSON.parse(raw) as Partial<Prefs>;
      return {
        ...PREFS_DEFAULTS,
        ...parsed,
        language: validLanguage(parsed.language) ? parsed.language : 'en',
        fontFamily: typeof parsed.fontFamily === 'string' ? parsed.fontFamily.slice(0,128).replace(/[\x00-\x1f]/g,'') : '',
        ...savedSizes(parsed, CONTENT_FS_MIN, CONTENT_FS_MAX),
        sizeDefaults: SIZE_DEFAULTS_VERSION,
        electricityPrice: Number.isFinite(Number(parsed.electricityPrice)) && Number(parsed.electricityPrice) > 0 && Number(parsed.electricityPrice) < 1000 ? Number(parsed.electricityPrice) : 0,
        electricityCurrency: typeof parsed.electricityCurrency === 'string' && /^[A-Za-z]{3}$/.test(parsed.electricityCurrency) ? parsed.electricityCurrency.toUpperCase() : '',
        doneSound: parsed.doneSound === true,
        approvalNotify: parsed.approvalNotify === true,
        numberQuotes: parsed.numberQuotes === true,
        whileBusy: validWhileBusy(parsed.whileBusy) ? parsed.whileBusy : 'wait',
        replyWidth: validReplyWidth(parsed.replyWidth),
        chatDefaults: validChatDefaults(parsed.chatDefaults),
      };
    } catch {
      /* fall through to the next key */
    }
  }
  // Nothing saved: a first start. It opens in the language of the computer; a choice made on the first screen is saved like any other.
  const tags = typeof navigator === 'undefined' ? [] : navigator.languages?.length ? navigator.languages : [navigator.language];
  return { ...PREFS_DEFAULTS, language: detectLanguage(tags) ?? 'en' };
}

const prefs = loadPrefs();
// Use the saved language for the very first render, before host bootstrap (main.tsx waits for its strings before it draws).
void setTranslationLanguage(prefs.language);

/** The last approval question a notification was shown for, so one question is announced once. */
let lastApprovalNotice = '';

let state: AppState = {
  language: prefs.language,
  fontFamily: prefs.fontFamily,
  configDraft: (() => { try { return sessionStorage.getItem('strata.config-draft.v1'); } catch { return null; } })(),
  navOpen: false,
  ready: false,
  connection: 'connecting',
  shareTelemetry: prefs.shareTelemetry,
  theme: prefs.theme,
  contrast: prefs.contrast,
  rail: prefs.rail,
  focus: prefs.focus,
  contentFontSize: prefs.contentFontSize,
  density: prefs.density,
  replyWidth: prefs.replyWidth,
  motion: prefs.motion,
  transparency: prefs.transparency,
  uiScale: prefs.uiScale,
  electricityPrice: prefs.electricityPrice,
  electricityCurrency: prefs.electricityCurrency,
  doneSound: prefs.doneSound,
  approvalNotify: prefs.approvalNotify,
  numberQuotes: prefs.numberQuotes,
  whileBusy: prefs.whileBusy,
  parked: {},
  quotes: {},
  agentAsks: {},
  chatDefaults: prefs.chatDefaults,
  paletteOpen: false,
  shortcutsOpen: false,
  settingsSection: 'Appearance',
  hardware: null,
  library: null,
  imageModels: null,
  runtime: null,
  providers: null,
  /** The open chat's API model, or null for the local model (the default for every chat). */
  remoteModel: null,
  engine: null,
  serving: null,
  monitor: null,
  setup: null,
  skills: [],
  files: [],
  workspacePath: '',
  workspaces: [],
  activeWorkspace: null,
  setupOpen: false,
  sessions: loadSessions(),
  sessionSave: null,
  damagedChats: [],
  newerChats: [],
  activeSession: null,
  streaming: false,
  liveChats: 0,
  toasts: [],
  touchedPaths: (() => {
    try {
      return JSON.parse(localStorage.getItem(TOUCHED_KEY) ?? '{}') as Record<string, string>;
    } catch {
      return {};
    }
  })(),
  actionError: null,
  busy: {},
};

const listeners = new Set<() => void>();

/** Chats with a turn in flight: the abort handle, and whether the host is running it as an agent turn. */
const liveTurns = new Map<string, { abort: AbortController; agent: boolean }>();

function set(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  const before = state.activeSession;
  const next = typeof patch === 'function' ? patch(state) : patch;
  state = { ...state, ...next };
  // Opening another chat shows that chat's model choice and whether that chat is mid-turn.
  if (state.activeSession !== before) {
    const chosen = state.sessions.find(s => s.id === state.activeSession)?.preferences?.remoteModel ?? null;
    state = { ...state, remoteModel: 'remoteModel' in next ? state.remoteModel : chosen, streaming: !!state.activeSession && liveTurns.has(state.activeSession) };
  }
  for (const l of listeners) l();
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function getState(): AppState {
  return state;
}

export function setConfigDraft(configDraft: string | null) {
  set({ configDraft });
  try {
    if (configDraft === null) sessionStorage.removeItem('strata.config-draft.v1');
    else sessionStorage.setItem('strata.config-draft.v1', configDraft);
  } catch { /* Keep the session draft even when browser storage is unavailable. */ }
}

export function setNavOpen(navOpen: boolean) { set({ navOpen }); }
export function setSettingsSection(settingsSection: AppState['settingsSection']) { set({ settingsSection }); }

/** Choose the open chat's model: an API model, or null for the local one. Other chats keep theirs. */
export function selectRemoteModel(remoteModel: RemoteModelSelection | null) {
  if (state.streaming) return;
  if (remoteModel && !state.providers?.connections.some(c => c.id === remoteModel.connectionId && c.models.some(m => m.id === remoteModel.modelId))) return;
  set({ remoteModel });
  const id = state.activeSession;
  if (id) patchSession(id, s => ({ ...s, preferences: { ...s.preferences, remoteModel } }));
  try { localStorage.removeItem('strata.remote-model.v1'); } catch { /* The old global choice is retired. */ }
}
function applyProviders(providers: ModelProviders) {
  // A chat whose API model was removed keeps its choice: the next send says the model is gone, rather
  // than quietly answering from the local model.
  set({ providers });
}
export async function saveModelConnection(draft: ModelConnectionDraft) {
  applyProviders(await api.post<ModelProviders>('/api/model-providers', draft));
}
export async function removeModelConnection(id: string) {
  applyProviders(await api.del<ModelProviders>(`/api/model-providers/${encodeURIComponent(id)}`));
}

function libraryResponse(data: ModelLibrary): ModelLibrary {
  if (!data || !Array.isArray(data.variants)) throw new Error('The engine returned an incomplete model list. Retry the operation.');
  return data;
}

export function lifecycleStatus(s = state) {
  return lifecyclePolicy({ connected: s.connection === 'live', streaming: !!s.streaming || s.liveChats > 0, stats: s.serving?.stats, busy: s.busy });
}
/** The lifecycle status without subscribing to the whole store: a screen that only needs this must not redraw on every monitor tick (round 8, item 13). */
export function useLifecycle() {
  const connected = useStore(s => s.connection === 'live');
  const streaming = useStore(s => !!s.streaming || s.liveChats > 0);
  const stats = useStore(s => s.serving?.stats);
  const busy = useStore(s => s.busy);
  return lifecyclePolicy({ connected, streaming, stats, busy });
}
export function lifecycleAvailable(): boolean {
  const result = lifecycleStatus();
  if (!result.allowed) pushToast('warn', 'Model change unavailable', result.reason);
  return result.allowed;
}

export function useStore<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(
    subscribe,
    () => selector(state),
    () => selector(state),
  );
}

/* ------------------------------------------------------------------ toasts */

/** Dismissal timers, held so a hovered toast can be paused rather than lost. */
const toastTimers = new Map<string, ReturnType<typeof setTimeout>>();
const TOAST_MS = 5200;

export function pushToast(tone: Toast['tone'], title: string, body?: string, action?: Toast['action']): string {
  const id = uid('toast');
  // Cap the stack: past three, the oldest collapses rather than pushing the
  // rest off-screen. A toast nobody can read is not a notification.
  const next = [...state.toasts, { id, tone, title, body, ...(action ? { action } : {}) }];
  const overflow = next.length > 3 ? next.slice(0, next.length - 3) : [];
  for (const gone of overflow) {
    const t = toastTimers.get(gone.id);
    if (t) clearTimeout(t);
    toastTimers.delete(gone.id);
  }
  const kept = overflow.length ? next.slice(-3) : next;
  set({ toasts: kept });
  if (!action) toastTimers.set(id, setTimeout(() => dismissToast(id), TOAST_MS));
  return id;
}

/**
 * Stops a toast's clock while the pointer is over it, so reading one does not
 * race its own dismissal. Restarting gives a full window rather than the
 * remainder — the intent is "I am reading this", and a resumed half-second is
 * not enough to act on.
 */
export function holdToast(id: string) {
  const t = toastTimers.get(id);
  if (t) clearTimeout(t);
  toastTimers.delete(id);
}

export function releaseToast(id: string) {
  if (toastTimers.has(id)) return;
  if (!state.toasts.some((t) => t.id === id && !t.leaving)) return;
  toastTimers.set(id, setTimeout(() => dismissToast(id), TOAST_MS));
}

export function dismissToast(id: string) {
  const t = toastTimers.get(id);
  if (t) clearTimeout(t);
  toastTimers.delete(id);
  set((s) => ({ toasts: s.toasts.map((x) => (x.id === id ? { ...x, leaving: true } : x)) }));
  setTimeout(() => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })), 340);
}

/* ------------------------------------------------------------------- prefs */

/**
 * Preferences live on <html> as data attributes plus one custom property, so CSS
 * owns presentation and React owns state — the same split the theme already
 * used. Nothing here re-renders a screen: the attributes do the work.
 */
function applyPrefs() {
  const root = document.documentElement;
  void setTranslationLanguage(state.language);
  root.lang = state.language === 'zh' ? 'zh-CN' : state.language;
  root.dir = state.language === 'he' ? 'rtl' : 'ltr';             // #6060: Hebrew reads right to left
  if(state.fontFamily)root.style.setProperty('--st-font',`${JSON.stringify(state.fontFamily)},system-ui,-apple-system,"Segoe UI",sans-serif`);
  else root.style.removeProperty('--st-font');
  root.dataset.theme = state.theme === 'system' ? matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' : state.theme;
  document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute('content', root.dataset.theme === 'dark' ? '#0e1113' : '#f9fafb');
  root.dataset.contrast = state.contrast;
  root.dataset.density = state.density;
  root.dataset.measure = state.replyWidth;
  root.style.setProperty('--st-content-fs', `${state.contentFontSize}px`);
  // The interface size (#3970): the whole page is scaled, so every text, button, menu and dialog follows, not only the parts that were
  // written with the scale token. The token stays at 1 so nothing is scaled twice. The reading size stays separate: someone who needs
  // larger prose does not necessarily want larger buttons.
  root.style.setProperty('--st-ui-scale', '1');
  root.style.setProperty('zoom', state.uiScale === 1 ? '' : String(state.uiScale));

  // "system" means the OS decides, so the attribute is absent rather than set
  // to a third value the stylesheet would have to special-case.
  if (state.motion === 'system') delete root.dataset.motion;
  else root.dataset.motion = state.motion;

  if (state.focus) root.dataset.focus = 'true';
  else delete root.dataset.focus;

  // "translucent" is the default, so the attribute only appears when it is off.
  if (state.transparency === 'opaque') root.dataset.transparency = 'opaque';
  else delete root.dataset.transparency;
}

function savePrefs() {
  const snapshot: Prefs = {
    sizeDefaults: SIZE_DEFAULTS_VERSION,
    language: state.language,
    fontFamily: state.fontFamily,
    theme: state.theme,
    contrast: state.contrast,
    rail: state.rail,
    focus: state.focus,
    contentFontSize: state.contentFontSize,
    density: state.density,
    replyWidth: state.replyWidth,
    motion: state.motion,
    transparency: state.transparency,
    uiScale: state.uiScale,
    electricityPrice: state.electricityPrice,
    electricityCurrency: state.electricityCurrency,
    doneSound: state.doneSound,
    approvalNotify: state.approvalNotify,
    numberQuotes: state.numberQuotes,
    whileBusy: state.whileBusy,
    chatDefaults: state.chatDefaults,
    shareTelemetry: state.shareTelemetry,
  };
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(snapshot));
  } catch {
    /* a full quota must not break the UI */
  }
  applyPrefs();
}

export function setTheme(theme: 'dark' | 'light' | 'system') {
  set({ theme });
  savePrefs();
}

export function toggleRail() {
  set({ rail: !state.rail });
  savePrefs();
}
/** A language whose strings are already here switches at once; the first use of another one waits for its file (a moment), then switches everything together. The last pick wins. */
let languageRequest=0;
export function setLanguage(language:Language){
  if(!validLanguage(language))return;
  const request=++languageRequest;
  const apply=()=>{if(request!==languageRequest)return;setTranslationLanguage(language);set({language});savePrefs();rememberShellLanguage(language);};
  if(catalogLoaded(language))apply();else void loadCatalog(language).then(apply);
}
let shellAnnounced = false;
/** The desktop shell's tray menu follows the app's language; it reads it from a small file the host keeps (no effect in a browser). */
function rememberShellLanguage(language:Language){api.post('/api/shell/prefs',{language}).catch(()=>undefined);}
/** The shell started on another port than it wanted: the chats and settings saved on that port are not here, so say so once (#3402). */
async function announcePortNotice(){
  try{
    const shell=await api.get<{portNotice:{wanted:number;used:number;terminal?:boolean}|null}>('/api/shell');
    /* A model started from the Terminal holds the port (#6278): say that, and that the Terminal's page there needs its own key. */
    if(shell.portNotice?.terminal)pushToast('warn',tr('A model is already running from the Terminal'),tr('A model started with strata launch is using port {0}, so Strata opened on port {1} instead. The Terminal’s page on port {0} asks for its own API key, and this app does not use that page. To load a model here, stop the Terminal’s one first (strata launch --stop). Chats and settings saved on port {0} are not shown here.',[shell.portNotice.wanted,shell.portNotice.used]));
    else if(shell.portNotice)pushToast('warn',tr('Strata could not use its port'),tr('Port {0} is used by another program, so Strata started on port {1}. Chats and settings saved on port {0} are not shown until Strata can use it again. Close the other program and restart Strata.',[shell.portNotice.wanted,shell.portNotice.used]));
  }catch{/* a browser or an older host has no shell */}
}
export function setFontFamily(fontFamily:string){set({fontFamily:fontFamily.trim().slice(0,128).replace(/[\x00-\x1f]/g,'')});savePrefs();}

export function setFocus(focus: boolean) {
  set({ focus });
  savePrefs();
}

export function toggleFocus() {
  setFocus(!state.focus);
}

/**
 * Puts every appearance and reading preference back to its shipped value.
 *
 * Deliberately excludes `shareTelemetry`: a privacy choice is not an appearance
 * setting, and re-enabling reporting as a side effect of resetting one's font
 * size would be exactly the kind of thing this control exists to prevent.
 *
 * Deliberately does **not** touch `rail` or `focus`: those are where you happen
 * to be working, not settings you chose, and collapsing the sidebar on someone
 * who asked to reset their font size would be a surprise. The engine has its own
 * reset, since an expert override and a font size have nothing to do with each
 * other.
 */
export function resetAppearance() {
  set({
    fontFamily: PREFS_DEFAULTS.fontFamily,
    theme: PREFS_DEFAULTS.theme,
    contrast: PREFS_DEFAULTS.contrast,
    contentFontSize: PREFS_DEFAULTS.contentFontSize,
    density: PREFS_DEFAULTS.density,
    replyWidth: PREFS_DEFAULTS.replyWidth,
    motion: PREFS_DEFAULTS.motion,
    transparency: PREFS_DEFAULTS.transparency,
    uiScale: PREFS_DEFAULTS.uiScale,
  });
  savePrefs();
}

/**
 * Records the standing privacy choice.
 *
 * The control was local component state that nothing persisted and nothing
 * read — it snapped back on every reload, so a reader who turned reporting off
 * had no way to know their choice had not been recorded. A privacy control that
 * silently forgets is worse than no control at all.
 */
export function setShareTelemetry(on: boolean) {
  set({ shareTelemetry: on });
  savePrefs();
}

export function setContentFontSize(px: number) {
  const clamped = Math.min(CONTENT_FS_MAX, Math.max(CONTENT_FS_MIN, Math.round(px)));
  set({ contentFontSize: clamped });
  savePrefs();
}

export function setDensity(density: 'comfortable' | 'compact') {
  set({ density });
  savePrefs();
}

export function setReplyWidth(replyWidth: ReplyWidth) {
  set({ replyWidth: validReplyWidth(replyWidth) });
  savePrefs();
}

export function setMotion(motion: 'system' | 'full' | 'reduced') {
  set({ motion });
  savePrefs();
}

export function setTransparency(transparency: 'translucent' | 'opaque') {
  set({ transparency });
  savePrefs();
}

/** The electricity price for the Monitor's cost line: a positive price per kWh; 0 turns the line off. The currency is the detected one unless a three-letter code is given (#3628). */
export function setElectricity(price: number, currency: string) {
  const ok = Number.isFinite(price) && price > 0 && price < 1000;
  set({ electricityPrice: ok ? Math.round(price * 10000) / 10000 : 0, electricityCurrency: /^[A-Za-z]{3}$/.test(currency) ? currency.toUpperCase() : '' });
  savePrefs();
}

export function setWhileBusy(whileBusy: WhileBusy) {
  if (!validWhileBusy(whileBusy)) return;
  set({ whileBusy });
  savePrefs();
}

/** A message sent while an answer is being written waits here; the chat sends it when the answer is done (it is also where Interrupt leaves it). */
export function parkMessage(sessionId: string, text: string) {
  const clean = text.trim();
  if (!clean) return;
  set({ parked: { ...state.parked, [sessionId]: [...(state.parked[sessionId] ?? []), clean] } });
}
function setAsks(sessionId: string, list: AgentAsk[]) {
  const next = { ...state.agentAsks };
  if (list.length) next[sessionId] = list; else delete next[sessionId];
  set({ agentAsks: next });
}
/** The agent asked the person something (a frame of the host), or the question ended without an answer. */
export function addAgentAsk(sessionId: string, frame: { callId?: unknown; questions?: unknown }) {
  const ask = readAsk(frame);
  // A /loop runs without anyone there: the question is skipped at once, so the loop never waits for an answer.
  if (ask && loopScheduler.isRunning(sessionId)) { void answerAgentAsk(sessionId, ask.callId, null).catch(() => {}); return; }
  if (!ask) return;
  const known = (state.agentAsks[sessionId] ?? []).some(a => a.callId === ask.callId);
  setAsks(sessionId, [...(state.agentAsks[sessionId] ?? []).filter(a => a.callId !== ask.callId), ask]);
  // #5617: a question waits for the person like an approval does: the same opt-in notification, once per question
  if (!known) void questionAsked(state.approvalNotify);
}
export function removeAgentAsk(sessionId: string, callId?: string) {
  const rest = callId === undefined ? [] : (state.agentAsks[sessionId] ?? []).filter(a => a.callId !== callId);
  if ((state.agentAsks[sessionId]?.length ?? 0) !== rest.length) setAsks(sessionId, rest);
}
/** Sends the person's answer to the agent, or a skip (answers null). A refusal comes back as an error for the card to say in words. */
export async function answerAgentAsk(sessionId: string, callId: string, answers: AgentAnswer[] | null) {
  await api.post('/api/chat/answer', { sessionId, callId, ...(answers ? { answers } : { skip: true }) });
  removeAgentAsk(sessionId, callId);
}

/** Puts a turn or a selected passage above the message box. */
export function addQuote(sessionId: string, text: string, messageId: string, role: 'user' | 'assistant') {
  const list = state.quotes[sessionId] ?? [];
  const next = addToQuotes(list, { text, messageId, role });
  if (next.length !== list.length) set({ quotes: { ...state.quotes, [sessionId]: next } });
}
export function removeQuote(sessionId: string, id: string) {
  const rest = (state.quotes[sessionId] ?? []).filter(q => q.id !== id);
  const next = { ...state.quotes };
  if (rest.length) next[sessionId] = rest; else delete next[sessionId];
  set({ quotes: next });
}
export function clearQuotes(sessionId: string) {
  if (!state.quotes[sessionId]) return;
  const next = { ...state.quotes };
  delete next[sessionId];
  set({ quotes: next });
}

/** The first parked message of a chat, taken out of the list (to send it, or to give it back to the box). */
export function takeParked(sessionId: string, index = 0): string | undefined {
  const list = state.parked[sessionId] ?? [];
  if (index < 0 || index >= list.length) return undefined;
  const rest = list.filter((_, i) => i !== index);
  const next = { ...state.parked };
  if (rest.length) next[sessionId] = rest; else delete next[sessionId];
  set({ parked: next });
  return list[index];
}

export function setDoneSound(doneSound: boolean) {
  set({ doneSound });
  savePrefs();
}

export function setApprovalNotify(approvalNotify: boolean) {
  set({ approvalNotify });
  savePrefs();
}

/** Settings > General > Number quotes (#5993). */
export function setNumberQuotes(numberQuotes: boolean) {
  set({ numberQuotes: numberQuotes === true });
  savePrefs();
}

/** Change what a new conversation starts with. Existing chats keep what they already have. */
export function setChatDefaults(patch: Partial<ChatDefaults>) {
  set({ chatDefaults: validChatDefaults({ ...state.chatDefaults, ...patch }) });
  savePrefs();
}

export const UI_SCALES = [0.9, 1, 1.1, 1.25, 1.5, 1.75, 2] as const;

export function setUiScale(uiScale: number) {
  set({ uiScale });
  savePrefs();
}

export const CONTENT_FONT_RANGE = { min: CONTENT_FS_MIN, max: CONTENT_FS_MAX };

/* -------------------------------------------------------------- workspaces */

/** Opens the system folder chooser. Returns null when it is dismissed. */
export async function pickWorkspace(): Promise<string | null> {
  const res = await postWithFolderFallback<{ path?: string; cancelled?: boolean }>('/api/workspaces/pick') as { path?: string };
  return res.path ?? null;
}

export async function addWorkspace(path: string) {
  try {
    const ws = await api.post<Workspace>('/api/workspaces', { path });
    set({ workspaces: [...state.workspaces, ws] });
    pushToast('success', 'Workspace added', `Strata can now read files in ${ws.name}.`);
  } catch (err) {
    pushToast('error', "Couldn't add that folder", err instanceof Error ? err.message : String(err));
  }
}

/** The name the sidebar shows for a workspace; its folder is unchanged. */
export async function renameWorkspace(id: string, name: string) {
  try {
    const ws = await api.patch<Workspace>(`/api/workspaces/${id}`, { name });
    set({ workspaces: state.workspaces.map(w => (w.id === id ? { ...w, name: ws.name } : w)) });
  } catch (err) {
    pushToast('error', "Couldn't rename that workspace", err instanceof Error ? err.message : String(err));
  }
}

export async function removeWorkspace(id: string) {
  const ws = state.workspaces.find((w) => w.id === id);
  try {
    await api.del(`/api/workspaces/${id}`);
    set({ workspaces: state.workspaces.filter((w) => w.id !== id) });
    if (state.activeWorkspace === id) set({ activeWorkspace: state.workspaces[0]?.id ?? null });
  } catch (err) {
    pushToast('error', "Couldn't remove that workspace", err instanceof Error ? err.message : String(err));
  }
  void ws;
}

/**
 * Pins a conversation to the top of its workspace.
 *
 * Sessions are currently ordered by recency, which is right until you have
 * thirty of them and two matter. `Session.pinned` has existed in the type since
 * the beginning and was never written to.
 */
export function toggleArchive(id: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, archived: !s.archived } : s));
  persistSessions(sessions);
  set({ sessions });
}

export function togglePin(id: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, pinned: !s.pinned } : s));
  persistSessions(sessions);
  set({ sessions });
}

export function setActiveWorkspace(id: string) {
  set({ activeWorkspace: id });
  void loadWorkspaceFiles();
}

export function openSetup(open: boolean) {
  set({ setupOpen: open });
}

export function setPalette(open: boolean) {
  set({ paletteOpen: open });
}

export function setShortcuts(open: boolean) {
  set({ shortcutsOpen: open });
}

/* --------------------------------------------------------------- transport */

function applyEvent(e: StreamEvent) {
  const data = e.data as Record<string, unknown>;
  switch (e.event) {
    case 'monitor':
      set({ monitor: data as unknown as MonitorState });
      break;
    case 'skills':
      set({ skills: data as unknown as Skill[] });
      break;
    case 'models':
      if (Array.isArray(data.variants)) set({ library: data as unknown as ModelLibrary });
      break;
    case 'runtime':
      set({runtime:data as unknown as RuntimeModels});
      break;
    case 'native':
      applyNativeEvent(data);
      break;
    case 'image-models':
      if (Array.isArray(data.variants)) set({ imageModels: data as unknown as ImageModelLibrary });
      break;
    case 'providers':
      applyProviders(data as unknown as ModelProviders);
      break;
    case 'goal':
      if (typeof data.chatId === 'string') applyGoalFrame(data.chatId, data.goal);
      break;
    case 'chat-wake':
      // The agent answers by itself (a background command finished after its turn ended): that reply streams in as a new message.
      if (typeof data.chatId === 'string' && typeof data.requestId === 'string' && state.sessions.some(s => s.id === data.chatId) && !liveTurns.has(data.chatId)) void sendMessage('', { reattach: data.chatId, requestId: data.requestId, wake: true });
      break;
    case 'plan':
      if (typeof data.chatId === 'string') applyPlanFrame(data.chatId, data.plan);
      break;
    case 'queue':
      if (typeof data.chatId === 'string') applyQueueFrame(data.chatId, data.queue);
      break;
    case 'timeskip':
      applyTimeSkipEvent(data);
      break;
    case 'engine':
      set({ engine: data as unknown as EngineConfig });
      break;
    case 'serving':
      set({ serving: newestServingState(state.serving, data as unknown as ServingState) });
      break;
    case 'setup':
      set({ setup: data as unknown as SetupState });
      break;
    case 'task': {
      // A task finished on its own (the host ran it on schedule): say so, and say when it needs a person's yes or failed.
      const t = data as unknown as { title?: string; status?: string; attention?: string | null; error?: string | null };
      if (t.status === 'error') pushToast('error', 'Task failed', String(t.title ?? ''));
      else if (t.attention) { pushToast('warn', 'Task needs your yes', String(t.title ?? '')); void approvalNeeded(state.approvalNotify, String(t.title ?? '')); }
      else if (t.status === 'done') { pushToast('success', 'Task finished', String(t.title ?? '')); workFinished(state.doneSound, Infinity); }
      break;
    }
    case 'chats':
      // Another window saved or removed a chat (this window's own saves carry its name and are ignored).
      if (typeof data.id === 'string' && data.from !== WINDOW_ID) void applyChatChange(data.id, typeof data.rev === 'number' ? data.rev : null, data.deleted === true);
      break;
    case 'agent':
      if (typeof data.sessionId === 'string') patchSession(data.sessionId, s => ({ ...s, agents: updateAgent(s.agents ?? [], data.agent) }));
      break;
    default:
      break;
  }
}

/** QAGUI 0213: how long one start-up GET may take, how often it is tried, and the pause between tries (a test may shorten them). */
export const bootstrapTiming = { timeoutMs: 8000, tries: 3, backoffMs: 400 };

/** One GET with a time limit, tried again after a timeout, a lost connection or a host error. A refusal (4xx) is final. */
export async function getWithRetry<T>(path: string): Promise<T> {
  const { timeoutMs, tries, backoffMs } = bootstrapTiming;
  let last: unknown;
  for (let attempt = 0; attempt < tries; attempt++) {
    if (attempt) await new Promise(r => setTimeout(r, backoffMs * 2 ** (attempt - 1)));
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        api.get<T>(path, abort.signal),
        new Promise<never>((_, reject) => { timer = setTimeout(() => { abort.abort(); reject(new Error(`No answer from ${path}`)); }, timeoutMs); }),
      ]);
    } catch (err) {
      last = err;
      if (err instanceof ApiError && err.status >= 400 && err.status < 500) throw err;
    } finally { clearTimeout(timer); }
  }
  throw last;
}

let closeEvents: (() => void) | null = null;
export async function bootstrap() {
  applyPrefs();
  try {
    let chatsTimer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([
      loadChatsFromHost(),
      new Promise<never>((_, reject) => { chatsTimer = setTimeout(() => reject(new Error('The chats did not load in time')), bootstrapTiming.timeoutMs); }),
    ]).catch(() => { sessionWriter.enable(); }).finally(() => clearTimeout(chatsTimer));
    // Only what the first screen cannot do without holds up 'live'; the rest is read at the same time and fills in when it arrives.
    const soft = <T,>(path: string) => getWithRetry<T>(path).catch(() => null);
    const late = {
      hardware: soft<Hardware>('/api/hardware'),
      monitor: soft<MonitorState>('/api/monitor'),
      skills: soft<Skill[]>('/api/skills'),
      providers: soft<ModelProviders>('/api/model-providers'),
      imageModels: soft<ImageModelLibrary>('/api/image-models'),
      runtime: soft<RuntimeModels>('/api/runtime'),
    };
    const [setup, library, engine, serving, workspaceState] = await Promise.all([
      getWithRetry<SetupState>('/api/setup'),
      getWithRetry<ModelLibrary>('/api/models'),
      getWithRetry<EngineConfig>('/api/engine'),
      getWithRetry<ServingState>('/api/serving'),
      getWithRetry<{ workspaces: Workspace[] }>('/api/workspaces'),
    ]);
    set({
      ready: true,
      setup,
      library: libraryResponse(library),
      engine,
      serving: newestServingState(state.serving, serving),
      workspacePath: workspaceState.workspaces[0]?.path ?? '',
      workspaces: workspaceState.workspaces,
      activeWorkspace: workspaceState.workspaces.some(w => w.id === state.activeWorkspace) ? state.activeWorkspace : workspaceIdForSession(workspaceState.workspaces, state.sessions.find(s => s.id === state.activeSession)) ?? workspaceState.workspaces[0]?.id ?? null,
      connection: 'live',
    });
    void late.hardware.then(hardware => { if (hardware) set({ hardware }); });
    void late.monitor.then(monitor => { if (monitor) set({ monitor }); });
    void late.skills.then(skills => { if (skills) set({ skills }); });
    void late.providers.then(providers => { if (providers) applyProviders(providers); });
    void late.imageModels.then(imageModels => { if (imageModels) set({ imageModels }); });
    void late.runtime.then(runtime => { if (runtime) set({ runtime }); });
    if (!shellAnnounced) { shellAnnounced = true; void announcePortNotice(); rememberShellLanguage(state.language); }
    closeEvents?.();
    let reconnect = false;
    closeEvents = onEvent(applyEvent, (status) => {
      if (status === 'offline') reconnect = true;
      if (status === 'live' && reconnect) {
        reconnect = false;
        // The restarted host may have new logs, config or readiness. Refresh
        // those facts in place while retaining conversations and unsent drafts.
        // A reply the agent began by itself while the event stream was down was announced to no one: ask for it now.
        void bootstrap().then(() => takeOverTurns());
        return;
      }
      set({ connection: status });
    });
    if (!setup.complete && !setup.dismissed && !window.location.hash.includes('setup')) navigate('setup');
  } catch (err) {
    set({ ready: true, connection: 'offline' });
    pushToast(
      'error',
      'Engine host unreachable',
      err instanceof Error ? err.message : 'Start it with `npm run host` and reload.',
    );
  }
}

/* ------------------------------------------------------------- reads/writes */

export async function refreshHardware() {
  set((s) => ({ busy: { ...s.busy, hardware: true } }));
  try {
    set({ hardware: await api.get<Hardware>('/api/hardware?reprobe=1') });
  } finally {
    set((s) => ({ busy: { ...s.busy, hardware: false } }));
  }
}

function modelFits(id: string) {
  const model = state.library?.variants.find(v => v.id === id);
  if (state.library?.mode === 'real' && model && state.hardware && state.engine && ['too-large', 'unsupported'].includes(estimate(model, state.hardware, execution(Object.fromEntries(state.engine.expert.map(p => [p.id, p.value])), contextTokens(state.engine.beginner.contextLength))).fit)) {
    pushToast('warn', 'Choose a smaller model', 'The estimated memory requirement exceeds this computer.');
    return false;
  }
  return true;
}

export async function selectImageModel(selected: ImageModelChoice) {
  set(s => ({ busy: { ...s.busy, 'image:select': true } }));
  try { set({ imageModels: await api.patch<ImageModelLibrary>('/api/image-models', { selected }) }); }
  finally { set(s => ({ busy: { ...s.busy, 'image:select': false } })); }
}
export async function installImageModel(id: Exclude<ImageModelChoice, 'off'>) {
  set({ imageModels: await api.post<ImageModelLibrary>(`/api/image-models/${id}/install`) });
}
export async function cancelImageModel(id: Exclude<ImageModelChoice, 'off'>) {
  set({ imageModels: await api.post<ImageModelLibrary>(`/api/image-models/${id}/cancel`) });
}
export async function applyImageModel() {
  set({ imageModels: await api.post<ImageModelLibrary>('/api/image-models/apply') });
}

export async function installVariant(id: string) {
  // Download eligibility is checked against disk space by the host. Runtime fit
  // is reviewed separately when the model is loaded.
  if (state.busy[`install:${id}`]) return false;
  set((s) => ({ busy: { ...s.busy, [`install:${id}`]: true } }));
  try {
    const library = libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/install`));
    set({ library });
    const complete = library.variants.find(v => v.id === id)?.state === 'installed';
    pushToast(complete ? 'success' : 'info', complete ? 'Files installed' : library.mode === 'simulated' ? 'Example installation started' : 'Download started', complete ? 'Load the model to start a request.' : 'Progress is shown in Models.');
    return true;
  } catch (err) {
    pushToast('error', 'Download failed', err instanceof Error ? err.message : undefined);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, [`install:${id}`]: false } }));
  }
}

export async function cancelInstall(id: string) {
  try { set({ library: libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/cancel`)) }); } catch (e) { pushToast('error', 'Cancel failed', e instanceof Error ? e.message : String(e)); }
}

export async function loadVariant(id: string) {
  if (!modelFits(id)) return false;
  if (!lifecycleAvailable()) return false;
  set((s) => ({ busy: { ...s.busy, [`load:${id}`]: true }, actionError: null }));
  try {
    const library = libraryResponse(await api.post<ModelLibrary>(`/api/models/${id}/load`));
    set({ library });
    const model = library.variants.find(v => v.id === id);
    if (library.loaded !== id || model?.state !== 'loaded') throw new Error(model?.error ?? 'The model did not finish loading.');
    pushToast('success', library.mode === 'simulated' ? 'Example model ready' : 'Model loaded', library.mode === 'simulated' ? 'Answers are simulated samples, not from a real model.' : 'Ready for a request.');
    return true;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'The model did not load.';
    set({
      actionError: {
        label: 'Could not load model',
        message,
        retry: () => void loadVariant(id),
      },
    });
    pushToast('error', 'Could not load model', message);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, [`load:${id}`]: false } }));
  }
}

export async function unloadModel() {
  if (!lifecycleAvailable()) return false;
  set(s => ({ busy: { ...s.busy, unload: true } }));
  try {
    const previousInstance=state.runtime?.defaultId;
    const library = libraryResponse(await api.post<ModelLibrary>('/api/models/unload'));
    const runtime=await api.get<RuntimeModels>('/api/runtime');
    set({runtime});
    set({ library });
    if (previousInstance?runtime.instances.some(i=>i.id===previousInstance):library.loaded!==null) throw new Error('The engine still reports this model instance as loaded.');
    pushToast('info', 'Model unloaded', library.mode === 'simulated' ? 'Example load state cleared.' : 'Model memory released.');
    return true;
  } catch (err) { pushToast('error', 'Unload failed', err instanceof Error ? err.message : String(err)); return false; }
  finally { set(s => ({ busy: { ...s.busy, unload: false } })); }
}

export async function removeVariant(id: string) {
  if (state.library?.loaded === id && !lifecycleAvailable()) return false;
  const local = !!state.library?.variants.find(v => v.id === id)?.local;
  try { const library = libraryResponse(await api.del<ModelLibrary>(`/api/models/${id}`)); set({ library });
    const remaining = library.variants.find(v => v.id === id);
    if (remaining && remaining.state !== 'available') throw new Error('The model files were not removed.');
    pushToast('success', local ? 'Local model removed' : 'Files removed', local ? 'The original GGUF files are kept.' : library.mode === 'simulated' ? 'Example install state cleared.' : 'Model files removed from disk.'); return true;
  } catch (e) { pushToast('error', 'Remove failed', e instanceof Error ? e.message : String(e)); return false; }
}

export async function waitForInstalled(id: string) {
  const deadline = Date.now() + 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const v = state.library?.variants.find(v => v.id === id);
    if (v && ['installed', 'loaded'].includes(v.state)) return;
    if (!v || ['available', 'error'].includes(v.state)) throw new Error(v?.error ?? 'Download cancelled.');
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw new Error('Download timed out. Try again from Models.');
}

export async function saveEngine(patch: Partial<EngineConfig> | Record<string, unknown>, silent = false, requireIdle = false) {
  if (requireIdle) {
    const status = lifecycleStatus();
    if (!status.allowed) throw new Error(status.reason);
  }
  const engine = await api.patch<EngineConfig>('/api/engine', patch);
  set({ engine, actionError: null });
  if (!silent) pushToast('success', 'Settings saved', 'Load settings take effect on the next load. Sampling defaults affect the next request.');
  return engine;
}

export async function restartEngine() {
  if (state.library?.mode === 'simulated') { pushToast('info', 'Restart unavailable with the example engine', 'Save settings for the next load.'); return false; }
  if (!lifecycleAvailable()) return false;
  set((s) => ({ busy: { ...s.busy, restart: true }, actionError: null }));
  try {
    const engine = await api.post<EngineConfig>('/api/engine/restart');
    set({ engine });
    if (!engine.engine.running) throw new Error(engine.engine.error ?? 'The engine is not running. Load a model before applying settings.');
    pushToast('success', 'Engine restarted', 'Warm-up takes a few seconds.');
    return true;
  } catch (err) {
    // Reported in place as well as in a toast: the toast is a summary, the
    // surface that failed is where the retry belongs.
    set({
      actionError: {
        label: 'Restart failed',
        message: err instanceof Error ? err.message : 'The engine did not come back up.',
        retry: () => void restartEngine(),
      },
    });
    pushToast('error', 'Restart failed', err instanceof Error ? err.message : undefined);
    return false;
  } finally {
    set((s) => ({ busy: { ...s.busy, restart: false } }));
  }
}

export function dismissActionError() {
  set({ actionError: null });
}

export async function resetEngine() {
  set({ engine: await api.post<EngineConfig>('/api/engine/reset') });
  pushToast('info', 'Back to recommended', 'Every value returned to the safe default.');
}

/**
 * Checks an engine config — the saved one by default, or a draft to preview what
 * a change would do before committing to it.
 */
export async function validateEngine(draft?: unknown) {
  return api.post<{
    ok: boolean;
    issues: { level: 'warn' | 'error'; message: string }[];
    predicted?: EngineConfig['predicted'];
  }>('/api/engine/validate', draft ?? {});
}

export async function exportEngine() {
  return api.get<Record<string, unknown>>('/api/engine/export');
}

export async function setCloudflare(body: {action: 'start' | 'stop' | 'prepare'; mode?: 'quick' | 'domain'; hostname?: string; token?: string}) {
  const result = await api.post<ServingState>('/api/serving/tunnel', body);
  set({ serving: newestServingState(state.serving, result) });
}

export async function setLocalSharing(enabled: boolean) {
  const result = await api.post<ServingState>('/api/serving/share', { enabled });
  set({ serving: newestServingState(state.serving, result) });
}

export async function setServingAccess(patch: { otherDevices?: boolean; apiKey?: string | null }) {
  const result = await api.patch<ServingState>('/api/serving', patch);
  set({ serving: newestServingState(state.serving, result) });
}

/** #5661: let apps on this computer use this model's MCP tools through the API (off by default). */
export async function setMcpTools(model: string, on: boolean) {
  const result = await api.patch<ServingState>('/api/serving', mcpToolsPatch(model, on));
  set({ serving: newestServingState(state.serving, result) });
}

export async function pauseServing(paused: boolean) {
  const result = await api.patch<ServingState>('/api/serving', { pauseServing: paused });
  set({ serving: newestServingState(state.serving, result) });
}

export async function cancelRequest(id: string) {
  const result = await api.post<ServingState>(`/api/serving/requests/${id}/cancel`);
  set({ serving: newestServingState(state.serving, result) });
}

export async function completeSetupStep(step: number, choices?: SetupState['choices'], dismissed = false) {
  set({ setup: await api.post<SetupState>('/api/setup/step', { step, choices, dismissed, flowVersion: 2 }) });
}

export async function completeSetup() {
  const setup = await api.post<SetupState>('/api/setup/complete');
  set({ setup });
}

export async function setSkills(skills: Skill[]) {
  set({ skills });
}

/**
 * Re-reads the skill list from the host.
 *
 * For the case where a skill was added to the workspace folder while the app was
 * open — otherwise the only way to see it is to restart.
 */
export async function loadSkills() {
  set({ skills: await api.get<Skill[]>('/api/skills') });
}

/** Re-reads the workspace tree from disk, for the same reason. */
export async function loadWorkspaceFiles() {
  const data = await api.get<{ path: string; entries: WorkspaceFile[] }>(`/api/workspace${state.activeWorkspace ? '?root=' + encodeURIComponent(state.activeWorkspace) : ''}`);
  set({ files: data.entries, workspacePath: data.path });
}

export async function toggleSkill(id: string, enabled: boolean) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, { enabled }) });
}

export async function editSkill(id: string, patch: Pick<Skill, 'name' | 'description' | 'body' | 'triggers' | 'enabled' | 'onlyIn'>) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, patch) });
}

/**
 * Narrows a skill to the workspace you are in, or widens it again.
 *
 * Passing null clears the narrowing. The host resolves `onlyIn` against the
 * workspace the request came from, so the answer depends on where the reader
 * is rather than on a single global switch.
 */
export async function scopeSkill(id: string, workspaceId: string | null) {
  set({ skills: await api.patch<Skill[]>(`/api/skills/${encodeURIComponent(id)}`, { onlyIn: workspaceId }) });
}

/* ---------------------------------------------------------------- sessions */

const chatTransport = createHttpTransport();
let saveToast: string | null = null;

const sessionWriter = createSessionWriter({
  transport: chatTransport,
  enabled: false,   // on once the host's chats have been read (loadChatsFromHost), so an older copy here never replaces a newer file
  // Nothing is trimmed to fit (#5192). A chat the host cannot write keeps its last file and stays whole here; it is reported and
  // tried again. A host that does not answer is retried quietly: that is the app being down, not a full disk.
  onResult: (result: SaveResult) => {
    if (result.ok) {
      try { browserStorage.removeItem(PENDING_KEY); } catch { /* removed at the next start */ }
      if (state.sessionSave) set({ sessionSave: null });
      if (saveToast) { dismissToast(saveToast); saveToast = null; }
      return;
    }
    const real = result.failed.filter(f => f.reason !== 'offline');
    if (!real.length) return;
    const failedIds = sessionWriter.unsaved().map(s => s.id);
    const reason = real[0].reason === 'offline' ? 'other' : real[0].reason;
    const was = state.sessionSave;
    if (!was || was.reason !== reason || was.failedIds.join() !== failedIds.join()) set({ sessionSave: { failedIds, reason } });
    if (!was && !saveToast) {
      saveToast = pushToast('error', 'Some chats could not be saved', tr("Strata's chat storage is full, so the newest changes to {0} chat(s) are only in this window. Delete old chats to make room; they then save by themselves.", [failedIds.length]), {
        label: 'Copy unsaved chats',
        run: () => { const text = unsavedSessionsJson(); if (text) void copyText(text, '', { toast: (tone, title, body) => { pushToast(tone, title, body || undefined); } }); },
      });
    }
  },
});

/** The chats the host could not write, as one JSON text (to keep a copy by hand), or null when all are saved. */
export function unsavedSessionsJson(): string | null {
  const unsaved = sessionWriter.unsaved();
  return unsaved.length ? exportSessions(unsaved) : null;
}

/**
 * Save the chat list. A one-off change (new, rename, delete, pin) is sent now; `soon` is for the streaming
 * path, where a write per token grew the renderer to gigabytes (#3604, #3603): those are coalesced to one a second.
 */
function persistSessions(sessions: Session[], soon = false) {
  sessionWriter.schedule(sessions);
  if (!soon) void sessionWriter.flush();
}

/** Send the newest chat list now: a turn ended, or the page is going away. */
export function flushSessions(): Promise<SaveResult> {
  return sessionWriter.flush();
}

/** The page is going away: send what can still be sent, and keep a last-moment copy of what the host has not confirmed (taken back at the next start). */
function leavingPage() {
  void sessionWriter.flush();
  const unconfirmed = sessionWriter.notOnHost();
  if (unconfirmed.length) keepLastMomentCopy(browserStorage, unconfirmed);
}

if (typeof window !== 'undefined' && typeof document !== 'undefined' && window.addEventListener) {
  window.addEventListener('pagehide', leavingPage);
  document.addEventListener?.('visibilitychange', () => { if (document.visibilityState === 'hidden') leavingPage(); });
}

let chatsRead = false;
/**
 * The first read of the chats at start, and again after the host came back: the host's files become the chats shown (the browser's
 * old chats are handed over first, once), then saving starts. A host that does not answer leaves the page as it is.
 */
export async function loadChatsFromHost() {
  if (chatsRead) { await syncChatsFromHost(); return; }
  const result = await hydrateChats({
    transport: chatTransport, browser: browserStorage, current: state.sessions,
    isLive: id => liveTurns.has(id),
  });
  if (result.offline) {
    // Show the browser's old copy so the person can still read; saves are retried until the host answers.
    if (!state.sessions.length) { try { set({ sessions: readBrowserSessions(browserStorage).sessions.map(fixSession) }); } catch { /* none */ } }
    for (const s of state.sessions) sessionWriter.adopt(s, null);
    sessionWriter.enable();
    return;
  }
  chatsRead = true;
  const sessions = result.sessions.map(fixSession);
  for (const s of sessions) { const rev = result.revs.get(s.id); if (rev !== undefined && !result.recovered.includes(s.id)) sessionWriter.adopt(s, rev); }
  set({ sessions, damagedChats: result.damaged, newerChats: result.newer });
  // Whatever the host does not have yet (a chat it refused during the move, a last-moment copy) is sent now; a refusal raises the notice.
  sessionWriter.schedule(sessions);
  sessionWriter.enable();
  void resumeSessions();
}

/** One chat that another window saved or removed: taken over unless this window is working in it. */
export async function applyChatChange(id: string, rev: number | null, deleted: boolean) {
  const here = state.sessions.find(s => s.id === id);
  if (deleted) {
    // A damaged file is not a session, so it is dropped from its own list: the other window's delete clears the row here too (#5421).
    const damaged = state.damagedChats.includes(id) ? state.damagedChats.filter(d => d !== id) : null;
    if (here && !liveTurns.has(id) && !sessionWriter.dirty(here)) {
      sessionWriter.forget(id);
      set({ sessions: state.sessions.filter(s => s.id !== id), ...(state.activeSession === id ? { activeSession: null } : {}), ...(damaged ? { damagedChats: damaged } : {}) });
    } else if (damaged) set({ damagedChats: damaged });
    return;
  }
  if (rev !== null && sessionWriter.rev(id) >= rev) return;
  if (here && (liveTurns.has(id) || sessionWriter.dirty(here))) return;   // this window's own work wins; its next save keeps the displaced version on the host
  const got = await chatTransport.get(id).catch(() => null);
  if (!got) return;
  const fresh = state.sessions.find(s => s.id === id);
  if (fresh && (liveTurns.has(id) || sessionWriter.dirty(fresh))) return;
  sessionWriter.adopt(got.session, got.rev);
  const next = fresh ? state.sessions.map(s => (s.id === id ? got.session : s)) : [...state.sessions, got.session].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  set({ sessions: next });
}

/** After the host was away: take over what other windows saved or removed meanwhile. */
async function syncChatsFromHost() {
  const list = await chatTransport.list().catch(() => null);
  if (!list) return;
  const onHost = new Set(list.chats.map(c => c.id));
  for (const entry of list.chats) await applyChatChange(entry.id, entry.rev, false);
  for (const s of state.sessions) if (!onHost.has(s.id) && sessionWriter.rev(s.id) > 0) await applyChatChange(s.id, null, true);
  set({ damagedChats: list.damaged ?? [], newerChats: list.newer ?? [] });   // a file deleted in another window leaves the list here (#5421)
  sessionWriter.enable();
}

/** Delete a chat file the host cannot read. The file goes; nothing else changes (#5421). */
export async function deleteDamagedChat(id: string) {
  try {
    await chatTransport.remove(id);
    set({ damagedChats: state.damagedChats.filter(d => d !== id) });
    // dmg09 (coordinator 11-16) says what happened, so it is the toast after the delete, not the question before it
    pushToast('success', tr('{0} was deleted. Your other chats are not affected.', [`${id}.json`]));
  } catch (e) {
    pushToast('error', 'The chat file could not be deleted', e instanceof Error ? e.message : String(e));
  }
}

export function newSession(): Session {
  const session: Session = {
    id: uid('s'),
    title: 'New conversation',
    defaultTitle: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    messages: [],
    workspace: state.workspaces.find(w => w.id === state.activeWorkspace)?.path ?? state.workspacePath,
    tokensIn: 0,
    tokensOut: 0,
    // A new chat starts from what the person set in Settings (#4847; with the stored defaults untouched these are the very values a
    // chat with no preference showed before), on the model the composer shows: the open chat's model, or the choice made before any chat was open.
    preferences: {
      ...state.chatDefaults,
      ...(state.remoteModel ? { remoteModel: state.remoteModel } : {}),
    },
  };
  const sessions = [session, ...state.sessions];
  persistSessions(sessions);
  set({ sessions, activeSession: session.id });
  return session;
}

export function setChatPreferences(id: string | undefined, preferences: NonNullable<Session['preferences']>) {
  const sessionId = id ?? newSession().id;
  patchSession(sessionId, s => ({ ...s, preferences: { ...s.preferences, ...preferences } }));
  // A mode change applies to the very next tool call of a turn that is already running (a goal and its helpers included).
  if (preferences.mode && liveTurns.get(sessionId)?.agent) void api.post('/api/chat/mode', { sessionId, mode: preferences.mode }).catch(() => undefined);
}
export function setPreviewTabClosed(sessionId:string|undefined, artifactId:string, closed:boolean, selected:string|null) {
  if(!sessionId)return;
  patchSession(sessionId,s=>({...s,closedPreviews:closed?[...new Set([...(s.closedPreviews??[]),artifactId])]:(s.closedPreviews??[]).filter(id=>id!==artifactId),previewOpen:selected??undefined}));
}
/** #5548: the preview panel was closed. The chat forgets which preview was open, so the next start does not open the panel again; the tabs
 * stay, and picking one opens and remembers it as before. */
export function setPreviewPanelClosed(sessionId: string | undefined) {
  if (!sessionId) return;
  patchSession(sessionId, s => s.previewOpen === undefined ? s : { ...s, previewOpen: undefined });
}
export function createChatShowcase() {
  const session=newSession();const at=new Date().toISOString();
  patchSession(session.id,s=>({...s,title:'Chat showcase · code, tools & media',showcase:true,statsAvailable:true,tokensIn:28000000,tokensOut:4000000,messages:showcaseMessages(at),agents:SHOWCASE_AGENTS,showcaseAgentStates:true}));
  go('chat',session.id);
}

/** A visible example for checking the subagent interface in the design build. */
export function createSubagentExample() {
  const session = newSession(); const now = Date.now();
  patchSession(session.id, s => ({ ...s, title: 'Subagent preview', showcase: true, showcaseAgentStates: true,
    agents: [
      {id:'example-layout',name:'Layout review',description:'Checking spacing and component alignment.',status:'working',durationMs:39000,tokens:12600,updatedAt:now},
      {id:'example-interactions',name:'Interaction review',description:'Checking menus, controls and keyboard navigation.',status:'working',durationMs:21000,tokens:8400,updatedAt:now},
      {id:'example-copy',name:'Copy review',description:'Labels and descriptions reviewed.',status:'completed',durationMs:18000,tokens:4200,updatedAt:now},
    ], messages: [{id:uid('m'),role:'assistant',at:new Date(now).toISOString(),example:true,content:'Open the subagent count above to inspect this example run. Each row shows status, token usage and elapsed time.'}]
  }));
  go('chat', session.id);
}

export function createPreviewExamples() {
  const session = newSession();
  const at = new Date().toISOString();
  patchSession(session.id, s => ({ ...s, title: 'Design review · six previews', previewOpen:'sample-website', messages: [
    { id:uid('m'), role:'user', at, content:'Open six different output types so I can iterate on the preview design.' },
    { id:uid('m'), role:'assistant', at, demo:true, content:'Here is a design-review workspace for Fieldnotes: a website, PDF, Markdown brief, cover image, TypeScript file and release table.\n\nAll six are loaded in the panel on the right. Switch tabs to explore each view while keeping this conversation in place.', artifacts:PREVIEW_EXAMPLES },
  ] }));
  go('chat', session.id);
}

/** "Continue in a new chat" on a task: a new chat whose first message is the task's findings, what it did and its files (src/lib/task-handover.ts). */
export function continueTaskInNewChat(task: Task) {
  const session = newSession();
  go('chat', session.id);
  void sendMessage(taskHandoverMessage(task), { title: taskHandoverTitle(task), markdown: true });
}

export function selectSession(id: string) {
  // A chat comes back in the workspace it was made in, so its next turn runs in the same folder (#3754).
  const workspace = workspaceIdForSession(state.workspaces, state.sessions.find(s => s.id === id));
  const change = workspace && workspace !== state.activeWorkspace;
  set(change ? { activeSession: id, activeWorkspace: workspace } : { activeSession: id });
  if (change) void loadWorkspaceFiles();
  markRead(id);
}

/** Records that a conversation has been looked at. */
export function markRead(id: string) {
  const at = new Date().toISOString();
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, lastReadAt: at } : s));
  persistSessions(sessions);
  set({ sessions });
}

/**
 * Whether a conversation has moved on since it was last looked at.
 *
 * The active conversation is never unread: you are looking at it, or its reply
 * is arriving in front of you.
 */
export function isUnread(s: Session, activeSession: string | null): boolean {
  if (s.id === activeSession) return false;
  if (!s.lastReadAt) return s.messages.length > 0;
  return s.updatedAt > s.lastReadAt;
}

export function deleteSession(id: string) {
  const sessions = state.sessions.filter((s) => s.id !== id);
  persistSessions(sessions);
  set({ sessions, activeSession: state.activeSession === id ? (sessions[0]?.id ?? null) : state.activeSession });
  // Tell the host too: its agent keeps a Harness session per chat id, and nothing else drops it.
  void api.post('/api/agent/chat/forget', { chatId: id }).catch(() => undefined);
}

/** Move a chat to another workspace (#6051). False when it was not moved (it is working, or it is already there). */
export function moveSession(id: string, workspaceId: string): boolean {
  const result = moveSessionToWorkspace(state.sessions, id, state.workspaces, workspaceId, new Set(liveChatIds()));
  if (result.refused) return false;
  persistSessions(result.sessions);
  const follow = state.activeSession === id && state.activeWorkspace !== workspaceId;
  set(follow ? { sessions: result.sessions, activeWorkspace: workspaceId } : { sessions: result.sessions });
  if (follow) void loadWorkspaceFiles();
  return true;
}

export function renameSession(id: string, title: string) {
  const sessions = state.sessions.map((s) => (s.id === id ? { ...s, title, defaultTitle: false } : s));
  persistSessions(sessions);
  set({ sessions });
}

function patchSession(id: string, fn: (s: Session) => Session) {
  const sessions = state.sessions.map((s) => (s.id === id ? fn(s) : s));
  persistSessions(sessions, true);
  set({ sessions });
}

// The demo clock belongs to the session, so hiding the sidebar or navigating
// away never pauses it. Only start and completion are written to storage.
const timeSkipTimers = new Map<string, ReturnType<typeof setTimeout>>();
function scheduleTimeSkip(sessionId: string, run: TimeSkipRun) {
  const timerId = `${sessionId}:${run.id}`;
  clearTimeout(timeSkipTimers.get(timerId));
  if (run.completedAt || run.job) return;
  timeSkipTimers.set(timerId, setTimeout(() => {
    timeSkipTimers.delete(timerId);
    const session = state.sessions.find(s => s.id === sessionId);
    const current = session?.timeSkips?.find(r => r.id === run.id);
    if (!current || current.startedAt !== run.startedAt || !session?.messages.some(m => m.timeSkipId === run.id && m.role === 'assistant')) return;
    patchSession(sessionId, s => ({ ...s, updatedAt: new Date().toISOString(),
      timeSkips: s.timeSkips?.map(r => r.id === run.id ? { ...r, completedAt: run.startedAt + TIME_SKIP_DURATION } : r),
      messages: s.messages.map(m => m.timeSkipId === run.id && m.role === 'assistant' ? { ...m, content: 'The updated result is ready. Switch between Original and Updated, or compare them side by side. This example uses sample outputs and simulated checks.' } : m),
    }));
  }, Math.max(0, run.startedAt + TIME_SKIP_DURATION - Date.now())));
}

export function replayTimeSkip(sessionId: string, runId: string) {
  const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
  if (!run) return;
  const next = { ...run, startedAt: Date.now(), completedAt: undefined, retainedVersion: undefined };
  patchSession(sessionId, s => ({ ...s, timeSkips: s.timeSkips?.map(r => r.id === runId ? next : r),
    messages: s.messages.map(m => m.timeSkipId === runId && m.role === 'assistant' ? { ...m, content: 'Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks.' } : m),
  }));
  scheduleTimeSkip(sessionId, next);
}

export function setTimeSkipResult(sessionId: string, runId: string, version: TimeSkipVersion) {
  const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
  if (!run) return;
  if (run.job) { void keepTimeSkip(sessionId, runId, version); return; }
  const next = withTimeSkipResult(run, version, Date.now());
  if (next === run) return;
  patchSession(sessionId, s => ({ ...s, timeSkips: s.timeSkips?.map(r => r.id === runId ? next : r) }));
}

/* ------------------------------------------------------- Time-skip, for real
   The host copies the chat's folder, lets the assistant change the copy, checks it and keeps the original untouched until the
   person chooses. The sidebar follows the job's own record; nothing here invents progress. */
const TS_TERMINAL = ['completed', 'cancelled', 'failed', 'interrupted'];
const TS_WORKING = () => tr('Working on a copy of your files. Follow the progress in the sidebar.');
const TS_READY = () => tr('The result is ready. Switch between Original and Updated, or compare them side by side.');

function patchRun(sessionId: string, runId: string, fn: (run: TimeSkipRun) => TimeSkipRun, content?: string) {
  patchSession(sessionId, s => ({ ...s, updatedAt: new Date().toISOString(),
    timeSkips: s.timeSkips?.map(r => r.id === runId ? fn(r) : r),
    messages: content === undefined ? s.messages : s.messages.map(m => m.timeSkipId === runId && m.role === 'assistant' ? { ...m, content } : m),
  }));
}

/** The job's error in a sentence a person can use. */
export function plainTimeSkipError(raw: string | null | undefined): string {
  const text = String(raw ?? '').trim();
  if (!text) return tr('Something went wrong.');
  const afterExit = /\(exit \d+\):\s*([\s\S]*)$/.exec(text);
  const body = (afterExit?.[1] ?? text).replace(/^\w*Error:\s*/, '').trim();
  if (/destination changed|refusing to overwrite/i.test(body)) return tr('Your files changed after this trial started, so nothing was overwritten. Run Time-skip again.');
  if (/too big/i.test(body)) return tr('This folder is too big to try a change on a copy. Open a smaller folder for this chat, then try again.');
  if (/folder of your own/i.test(body)) return tr('Time-skip works on a folder of your own. Choose a workspace folder for this chat, then try again.');
  if (/Load a model|model first/i.test(body)) return tr('Load a model first, then try Time-skip again.');
  if (/needs the Strata app|could not be reached/i.test(body)) return tr('The assistant could not be reached. Try again.');
  return body.length > 300 ? body.slice(0, 299) + '…' : body;
}

/** The host says a Time-skip run moved on (phase, and what it is doing now): the working view shows the newest words. */
function applyTimeSkipEvent(data: Record<string, unknown>) {
  const id = typeof data.id === 'string' ? data.id : '';
  if (!id) return;
  for (const session of state.sessions) {
    const run = session.timeSkips?.find(r => r.job?.id === id);
    if (!run || run.completedAt) continue;
    patchRun(session.id, run.id, r => ({ ...r, job: { ...r.job!, ...(typeof data.phase === 'string' ? { phase: data.phase as TimeSkipJob['phase'] } : {}), activity: data.activity as TimeSkipJob['activity'], ...forkFields(data) } }));
  }
}

function timeSkipEnding(job: TimeSkipJob, files: TimeSkipFile[]): string {
  if (job.phase === 'failed') return tr('Time-skip stopped. {0} Your files were not changed.', [plainTimeSkipError(job.error)]);
  if (job.phase === 'cancelled' || job.phase === 'interrupted') return tr('Time-skip was stopped before it finished. Your files were not changed.');
  if (!files.length) return tr('The assistant did not change anything, so there is nothing to compare. Your files were not changed.');
  return isForkJob(job) ? tr('Both versions are ready. Compare A and B, then keep one.') : TS_READY();
}

async function startTimeSkipReal(prompt: string) {
  const sessionId = state.activeSession ?? newSession().id;
  const run: TimeSkipRun = { id: uid('ts'), prompt, startedAt: Date.now(),
    job: { id: '', phase: 'queued', attempts: [], checkpoints: [], selectedCheckpoint: null, decision: null, needsUserReview: false, error: null } };
  const at = new Date().toISOString();
  patchSession(sessionId, s => ({ ...s,
    defaultTitle: false,
    title: s.messages.length ? s.title : shortTitle(prompt), updatedAt: at,
    // The comparison sidebar opens only when there is an update to compare (when the job ends with changed files), never while it works.
    timeSkips: [...(s.timeSkips ?? []), run],
    messages: [...s.messages,
      { id: uid('m'), role: 'user', at, content: '/time-skip ' + prompt, timeSkipId: run.id },
      { id: uid('m'), role: 'assistant', at, timeSkipId: run.id, content: TS_WORKING() },
    ],
  }));
  go('chat', sessionId);
  markTurn(sessionId, { abort: new AbortController(), agent: false });
  try {
    const started = await api.post<{ run: TimeSkipJob }>('/api/timeskip/start', { prompt, sessionId, workspaceId: state.activeWorkspace, attempts: 1, ...(state.remoteModel ? { remoteModel: state.remoteModel } : {}) });
    patchRun(sessionId, run.id, r => ({ ...r, job: started.run }));
    void followTimeSkip(sessionId, run.id);
  } catch (e) {
    const message = plainTimeSkipError(e instanceof Error ? e.message : String(e));
    patchRun(sessionId, run.id, r => ({ ...r, completedAt: Date.now(), job: { ...r.job!, phase: 'failed', error: message } }), tr('Time-skip could not start. {0}', [message]));
    markTurn(sessionId, null);
  }
}

const followingTimeSkips = new Set<string>();
/** Read the job's record until it ends. Safe to call again for a run already being followed. */
export async function followTimeSkip(sessionId: string, runId: string) {
  const key = sessionId + ':' + runId;
  if (followingTimeSkips.has(key)) return;
  followingTimeSkips.add(key);
  let misses = 0;
  try {
    for (;;) {
      const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
      if (!run?.job?.id || run.completedAt) return;
      let job: TimeSkipJob | null = null;
      try { job = (await api.get<{ run: TimeSkipJob }>('/api/timeskip/status?id=' + encodeURIComponent(run.job.id))).run; misses = 0; }
      catch { misses += 1; }
      if (job && TS_TERMINAL.includes(job.phase)) {
        let files: TimeSkipFile[] = [], extra: Partial<TimeSkipJob> = {};
        // The changed files are what the person came for: ask again a few times before settling for "nothing to compare".
        for (let tryNo = 0; tryNo < 4; tryNo++) {
          try {
            const found = await api.get<{ files: TimeSkipFile[]; applyAttempt: number | null; more: number }>('/api/timeskip/files?id=' + encodeURIComponent(job.id));
            files = found.files; extra = { files: found.files, applyAttempt: found.applyAttempt, moreFiles: found.more };
            break;
          } catch { await new Promise(r => setTimeout(r, 700 * (tryNo + 1))); }
        }
        const finished = { ...job, ...forkFields(job as unknown as Record<string, unknown>) };
        patchRun(sessionId, runId, r => ({ ...r, completedAt: Date.now(), job: { ...r.job!, ...finished, ...extra } }), timeSkipEnding(finished, files));
        // A run that changed files opens the comparison; one that changed nothing, failed or was stopped has no sidebar tab.
        if (files.length > 0 && !['failed', 'cancelled', 'interrupted'].includes(finished.phase)) patchSession(sessionId, s => ({ ...s, previewOpen: timeSkipTabId(runId) }));
        markTurn(sessionId, null);
        return;
      }
      if (job) { const live = { ...job, ...forkFields(job as unknown as Record<string, unknown>) }; patchRun(sessionId, runId, r => ({ ...r, job: { ...r.job!, ...live } })); }
      if (misses >= 6) {
        patchRun(sessionId, runId, r => ({ ...r, completedAt: Date.now(), job: { ...r.job!, phase: 'interrupted' } }), tr('Time-skip lost contact with the app and stopped. Your files were not changed.'));
        markTurn(sessionId, null);
        return;
      }
      await new Promise(r => setTimeout(r, 2000));
    }
  } finally { followingTimeSkips.delete(key); }
}

/** Ask the host to stop a running trial. The saved version stays. */
export async function stopTimeSkip(sessionId: string, runId: string) {
  const id = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId)?.job?.id;
  if (id) await api.post('/api/timeskip/stop', { id }).catch(() => undefined);
}

/** Put the update into the person's files, or put the original back. Both are checked and undoable. */
export async function keepTimeSkip(sessionId: string, runId: string, version: TimeSkipVersion) {
  const run = state.sessions.find(s => s.id === sessionId)?.timeSkips?.find(r => r.id === runId);
  const job = run?.job;
  if (!run || !job?.id || !job.phase || !TS_TERMINAL.includes(job.phase)) return;
  if ((run.retainedVersion ?? 'original') === version && !(isForkJob(job) && run.retainedVersion === undefined)) { patchRun(sessionId, runId, r => ({ ...r, retainedVersion: version })); return; }
  try {
    if (isForkJob(job)) await api.post('/api/timeskip/apply', { id: job.id, which: version === 'original' ? 'a' : 'b' });
    else if (version === 'updated') await api.post('/api/timeskip/apply', { id: job.id, attempt: job.applyAttempt ?? null });
    else await api.post('/api/timeskip/apply', { id: job.id, restore: true });
    patchRun(sessionId, runId, r => ({ ...r, retainedVersion: version }));
    pushToast('success', version === 'updated' ? tr('The update is now in your files') : tr('Your original files are back'));
  } catch (e) {
    pushToast('error', tr('Your files were not changed'), plainTimeSkipError(e instanceof Error ? e.message : String(e)));
  }
}

function startTimeSkipDemo(prompt: string) {
  if (!prompt) { pushToast('info', 'Add a prompt after /time-skip', 'For example: /time-skip Make a launch plan for my app'); return; }
  const sessionId = state.activeSession ?? newSession().id;
  const run: TimeSkipRun = { id: uid('ts'), prompt, startedAt: Date.now() };
  const at = new Date().toISOString();
  patchSession(sessionId, s => ({ ...s,
    defaultTitle: false,
    title: s.messages.length ? s.title : shortTitle(prompt), updatedAt: at,
    timeSkips: [...(s.timeSkips ?? []), run], previewOpen: timeSkipTabId(run.id),
    messages: [...s.messages,
      { id: uid('m'), role: 'user', at, content: `/time-skip ${prompt}`, timeSkipId: run.id },
      { id: uid('m'), role: 'assistant', at, example: true, timeSkipId: run.id, content: 'Revising the saved draft. Follow the result in the sidebar. This example uses sample outputs and simulated checks; it doesn’t run your model.' },
    ],
  }));
  scheduleTimeSkip(sessionId, run);
  go('chat', sessionId);
}

function markTurn(sessionId: string, turn: { abort: AbortController; agent: boolean } | null) {
  if (turn) liveTurns.set(sessionId, turn); else liveTurns.delete(sessionId);
  set({ streaming: !!state.activeSession && liveTurns.has(state.activeSession), liveChats: liveTurns.size });
  document.documentElement.dataset.streaming = liveTurns.size ? 'true' : 'false';
}

/**
 * Stop the open chat's turn. The host is asked first: for an agent turn it cancels exactly this chat's
 * Harness session and answers only once the turn has ended, and the stream then closes by itself with
 * what was produced. For any other turn the host reports nothing to stop and the request is aborted.
 */
export function stopStreaming() {
  const id = state.activeSession;
  const turn = id ? liveTurns.get(id) : undefined;
  if (!id || !turn) return;
  const abort = () => { turn.abort.abort(); };
  void api.post<{ stopped: boolean }>('/api/chat/stop', { sessionId: id }).then(
    result => {
      if (!result.stopped) abort();
      // An agent turn that was told to stop but is slow to end must not leave the chat stuck.
      else setTimeout(() => { if (liveTurns.get(id) === turn) abort(); }, 30000);
    },
    abort,
  );
}

/**
 * Re-runs the last exchange: drops the assistant reply and asks again.
 *
 * Kept in the store rather than the screen because it has to touch the message
 * list, and every attempt to do this from a component ends up duplicating the
 * stream handling that already lives here.
 */
export async function retryLast(messageId?: string) {
  const source = state.sessions.find(s => s.id === state.activeSession);
  if (!source || state.streaming) return;
  const end = messageId ? source.messages.findIndex(m => m.id === messageId) : source.messages.length - 1;
  const user = source.messages.slice(0, end + 1).reverse().find(m => m.role === 'user');
  if (!user) return;
  const prompt = rewindTo(user.id);
  if (prompt !== null) {
    // Retry sends immediately; only Edit should leave the prompt in the composer.
    sessionStorage.setItem(`strata.chat-draft.${state.activeSession}`, '');
    await sendMessage(prompt);
  }
}

/** The open chat's last question and the answer that failed under it (null when the chat does not end that way). */
function failedExchange(): { prompt: string; sessionId: string } | null {
  const sessionId = state.activeSession;
  const chat = state.sessions.find(s => s.id === sessionId);
  if (!sessionId || !chat || state.streaming) return null;
  const reply = chat.messages[chat.messages.length - 1], ask = chat.messages[chat.messages.length - 2];
  return reply?.role === 'assistant' && reply.error && ask?.role === 'user' ? { prompt: ask.content, sessionId } : null;
}
/** Takes the failed question and its error out of the transcript, so asking again does not show the question twice. */
function dropFailedExchange(sessionId: string) {
  patchSession(sessionId, s => ({ ...s, messages: s.messages.slice(0, -2) }));
}

/**
 * The error card of a conversation that no longer fits the model's window: summarise the older messages (the same as typing /compact), then
 * ask the assistant to answer. The refused question is already in the assistant's conversation (it is the newest message, and a summary keeps
 * the newest message as it is), so it is not sent a second time: a large one twice would not fit either.
 */
export async function summarizeAndContinue() {
  if (!failedExchange()) return;
  await sendMessage('/compact');
  await sendMessage('Please answer my last message now.');
}

/**
 * The error card of a message that is too large to send: the host makes it small enough (its text as a file in the chat's folder that the
 * assistant reads in parts, or a shortened preview), and that is sent in its place.
 */
export async function sendOversized(mode: 'file' | 'preview') {
  const failed = failedExchange();
  if (!failed) return;
  let reply: { message?: string };
  try {
    reply = await api.post<{ message?: string }>('/api/chat/oversize', { sessionId: failed.sessionId, mode, text: failed.prompt, nativeModelId: nativeState()?.instances.length ? nativeState()?.defaultId : undefined });
  } catch {
    pushToast('error', 'The message could not be prepared. Try a shorter message.');
    return;
  }
  if (!reply.message) return;
  dropFailedExchange(failed.sessionId);
  await sendMessage(reply.message);
}

/** A new, empty chat, opened. */
export function startNewChat() {
  const session = newSession();
  go('chat', session.id);
}

/**
 * Forks a conversation at a message, leaving the original untouched.
 *
 * The alternative people reach for is duplicating the whole session and
 * deleting messages by hand, which loses the original. Branching keeps both:
 * the new session carries the history up to the chosen turn and diverges there.
 */
export function branchFrom(messageId: string): string | null {
  const sessionId = state.activeSession;
  if (!sessionId) return null;
  const source = state.sessions.find((s) => s.id === sessionId);
  if (!source) return null;
  const at = source.messages.findIndex((m) => m.id === messageId);
  if (at < 0) return null;

  const now = new Date().toISOString();
  const branch: Session = {
    ...source,
    id: uid('s'),
    title: `${source.title} (branch)`,
    defaultTitle: false,
    createdAt: now,
    updatedAt: now,
    messages: source.messages.slice(0, at + 1).map((m) => ({ ...m, id: uid('m') })),
    tokensIn: 0,
    tokensOut: source.messages.slice(0, at + 1).reduce((n, m) => n + (m.tokens ?? 0), 0),
    statsAvailable: false,
  };
  const sessions = [branch, ...state.sessions];
  persistSessions(sessions);
  set({ sessions, activeSession: branch.id });
  for (const run of branch.timeSkips ?? []) scheduleTimeSkip(branch.id, run);
  // The URL wins over `activeSession` when it names a session, so a programmatic
  // switch has to move the route too — otherwise the sidebar shows the branch
  // while the transcript still shows the original, which reads as a broken
  // button rather than a created branch.
  navigate('chat', branch.id);
  return branch.id;
}

/**
 * Asks the model to keep going from where it stopped.
 *
 * Sent as an ordinary user turn rather than a special mode, because that is
 * what it is: a continuation is a request, and the transcript should show it.
 */
export async function continueLast() {
  const sessionId = state.activeSession;
  if (!sessionId || state.streaming) return;
  const session = state.sessions.find((s) => s.id === sessionId);
  const last = session?.messages[session.messages.length - 1];
  if (!last || last.role !== 'assistant' || !last.content.trim()) return;
  await sendMessage('Continue from where you left off. Do not repeat what you have already written.');
}

/**
 * Rewinds the transcript to just before a turn so it can be re-asked with a
 * different prompt. Returns the text so the composer can be filled with it.
 */
export function rewindTo(messageId: string): string | null {
  if (state.streaming) return null;
  const source = state.sessions.find(s => s.id === state.activeSession);
  const at = source?.messages.findIndex(m => m.id === messageId) ?? -1;
  if (!source || at < 0) return null;
  const prompt = source.messages[at].content;
  const branchId = branchFrom(messageId);
  if (!branchId) return null;
  patchSession(branchId, s => ({ ...s, messages: s.messages.slice(0, -1), tokensIn: 0, tokensOut: s.messages.slice(0, -1).reduce((n, m) => n + (m.tokens ?? 0), 0) }));
  sessionStorage.setItem(`strata.chat-draft.${branchId}`, prompt);
  return prompt;
}

export function exportConversation() {
  const session = state.sessions.find(s => s.id === state.activeSession);
  if (!session) return;
  const text = '# ' + session.title + '\n\n' + session.messages.map(m => '## ' + m.role + '\n\n' + (m.role === 'assistant' ? tidyAnswer(m.content) : m.content)).join('\n\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown' }));
  const a = document.createElement('a'); a.href = url; a.download = session.title.replace(/[^a-z0-9 -]/gi, '').slice(0, 60) + '.md';
  document.body.append(a); a.click(); a.remove();
  // Keep the Blob alive until browsers have started consuming the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------------------------------------------------------------- /loop
   The same message, again and again, while the chat is open. The scheduler (count, time, "the answer says it is done") is
   lib/loop-runner.ts; this is the part that talks to the chat. See lib/loop.ts. */

/** A line from the app itself (not the model) in the transcript. It is never sent to the model. */
function addNote(sessionId: string, content: string) {
  const at = new Date().toISOString();
  patchSession(sessionId, s => ({ ...s, updatedAt: at, messages: [...s.messages, { id: uid('m'), role: 'assistant', at, content, local: true }] }));
}

const lastAnswer = (sessionId: string) => [...(state.sessions.find(s => s.id === sessionId)?.messages ?? [])].reverse().find(m => m.role === 'assistant' && !m.local);

const loopScheduler = new LoopScheduler({
  async send(sessionId, { text, shown }) {
    const before = lastAnswer(sessionId)?.id;
    await sendMessage(text, { fromLoop: true, shown, title: shortTitle(shown) });
    const answer = lastAnswer(sessionId);
    if (!answer || answer.id === before || answer.streaming) return { answered: false, text: '', failed: false };
    return { answered: true, text: answer.content ?? '', failed: !!answer.error };
  },
  note: addNote,
  exists: sessionId => state.sessions.some(s => s.id === sessionId),
  ready: sessionId => state.activeSession === sessionId && !state.streaming,
  now: () => Date.now(),
  setTimer: (fn, ms) => setTimeout(fn, ms),
  clearTimer: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
  track: (sessionId, loop) => patchSession(sessionId, s => ({ ...s, loop })),
});

export function loopIsRunning(sessionId: string | null): boolean { return loopScheduler.isRunning(sessionId); }

/* ------------------------------------------------------------- task board */

/** The chats with a turn going, and the chats with a loop going: what the board needs besides the sessions themselves. */
export function liveChatIds(): string[] { return [...liveTurns.keys()]; }
export function loopChatIds(): string[] { return loopScheduler.ids(); }

/** Stop any chat's turn, not only the open one (the board's Stop). Same two-step as stopStreaming. */
export function stopChat(id: string) {
  const turn = liveTurns.get(id);
  if (!turn) return;
  const abort = () => { turn.abort.abort(); };
  void api.post<{ stopped: boolean }>('/api/chat/stop', { sessionId: id }).then(
    result => { if (!result.stopped) abort(); else setTimeout(() => { if (liveTurns.get(id) === turn) abort(); }, 30000); },
    abort,
  );
}

/** Stop a chat's loop from the board. The chat gets the same line /loop stop writes. */
export function stopLoop(id: string) {
  if (loopScheduler.stop(id)) addNote(id, tr('The loop is stopped.'));
}

/** Dismiss a chat's (or its loop's) card: it leaves the board until something newer goes wrong. The chat itself is untouched. */
export function dismissChat(id: string) {
  const sessions = state.sessions.map(s => (s.id === id ? { ...s, ackedAt: new Date().toISOString() } : s));
  persistSessions(sessions);
  set({ sessions });
}

/** Answer a waiting tool card in any chat (the board's Approve and Deny). Only a live card has an approval behind it. */
export function answerChatTool(sessionId: string, messageId: string, toolId: string, approved: boolean) {
  const tool = state.sessions.find(s => s.id === sessionId)?.messages.find(m => m.id === messageId)?.tools?.find(t => t.id === toolId);
  if (tool?.live && tool.state === 'proposed') answerAgentTool(sessionId, messageId, tool, approved);
}

/** "Continue in a new chat" on a chat or a loop that went wrong: a new chat that starts from what happened (src/lib/task-handover.ts). */
export function continueChatInNewChat(sessionId: string) {
  const old = state.sessions.find(s => s.id === sessionId);
  if (!old) return;
  const session = newSession();
  go('chat', session.id);
  void sendMessage(chatHandoverMessage(old), { title: chatHandoverTitle(old), markdown: true });
}

async function runLoopCommand(command: LoopCommand) {
  const sessionId = state.activeSession ?? newSession().id;
  const say = (content: string) => { addNote(sessionId, content); go('chat', sessionId); };
  if (command.kind === 'help') { say(tr(command.problem) + '\n\n' + tr(LOOP_USAGE)); return; }
  if (command.kind === 'status') { say(loopScheduler.describe(sessionId) ?? tr('No loop is running in this chat.') + '\n\n' + tr(LOOP_USAGE)); return; }
  if (command.kind === 'stop') { say(loopScheduler.stop(sessionId) ? tr('The loop is stopped.') : tr('No loop is running in this chat.')); return; }
  go('chat', sessionId);
  await loopScheduler.start(sessionId, command);
}

/** Carry on a paused goal: the host runs its rounds again. No message is sent and no bubble is added. */
export async function resumeGoal() { await sendMessage('', { goalAction: 'resume' }); }

/** Approve a proposed plan and carry it out: the host answers with the message that does the work, which goes out as an ordinary turn. */
export async function approvePlan(sessionId: string, planId: string) {
  const { prompt } = await planApprove(sessionId, planId);
  // A chat in Plan mode only reads; carrying the plan out needs a mode that may act.
  if (state.sessions.find(s => s.id === sessionId)?.preferences?.mode === 'plan') setChatPreferences(sessionId, { mode: 'manual' });
  if (prompt) await sendMessage(prompt, { shown: tr('Approved the plan.') });
}

/** "Start as goal" (and Resume on a paused plan): the host streams the run like any chat turn. */
export async function startPlanGoal(planId: string) { await sendMessage('', { planId }); }

export async function sendMessage(text: string, options: { fromLoop?: boolean; goalAction?: 'resume'; planId?: string; shown?: string; title?: string; markdown?: boolean; reattach?: string; requestId?: string; wake?: boolean } = {}) {
  // `reattach`: a turn the host is still running for this chat (or finished while the page was away) is taken over again after a reload, not started.
  const reattach = options.reattach;
  if (state.streaming && !reattach) {
    // Typed while the assistant works: behind a running goal it waits in the queue; otherwise there is nothing to do but wait.
    const waiting = state.activeSession, line = text.trim();
    if (waiting && line && !options.planId && !options.goalAction) {
      try { if (!(await queueMessage(waiting, line))) pushToast('info', 'The assistant is still answering', 'Send your message when it has finished.'); }
      catch (e) { pushToast('error', 'The message was not queued', plainFailureText(e instanceof Error ? e.message : String(e)).message); }
    }
    return;
  }
  const resume = options.goalAction === 'resume';
  const planStart = options.planId;
  const loopCommand = options.fromLoop || resume || planStart || reattach ? null : parseLoop(text.trim());
  if (loopCommand) { await runLoopCommand(loopCommand); return; }
  const demoPrompt = resume || planStart || reattach ? null : timeSkipPrompt(text.trim());
  if (demoPrompt !== null) {
    if (demoPrompt && (state.remoteModel || nativeState()?.instances.some(i => i.status === 'ready'))) void startTimeSkipReal(demoPrompt); else if (state.connection === 'live' && demoPrompt) pushToast('info', 'Load a model first', 'Time-skip works on a loaded model. Choose one on the Models tab, then try again.');
    else if (state.connection === 'connecting' && demoPrompt) pushToast('info', 'Still starting up', 'Time-skip needs the app to finish starting. Try again in a moment.');
    else startTimeSkipDemo(demoPrompt);
    return;
  }
  // Whatever sent this (the box, a loop, a plan): with no model to answer, say so in plain words and start nothing.
  if (state.connection === 'live' && !reattach) {
    // The first read of the engine's state has not come back yet: do not say "no model" about something not known yet, and do not send.
    if (!nativeState() && !state.remoteModel && !state.library?.loaded) { pushToast('info', 'Still starting up', 'Strata is still checking your models. Try again in a moment.'); return; }
    const why = noModelNotice(nativeState(), { remote: !!state.remoteModel, libraryLoaded: !!state.library?.loaded });
    if (why) { pushToast('info', why.title, why.message); return; }
  }
  const pictureCheck=pictureStatus(nativeState(),state.remoteModel,state.providers);
  if(!reattach&&pendingChatPhotos(state.activeSession).length&&!pictureCheck.ok){pushToast('error','This model cannot read pictures',tr(pictureCheck.message));return;}
  const pictures=reattach?[]:takeChatPhotos(state.activeSession);
  const body=text.trim()||(pictures.length?'Describe this picture.':'');
  if(!body&&!resume&&!planStart&&!reattach)return;

  let sessionId = reattach ?? state.activeSession;
  if (!sessionId) sessionId = newSession().id;

  const quoteSources = takeQuoteSources(body);   // #5993: the numbers of a numbered quote and the messages they came from
  const userMsg: ChatMessage = { id: uid('m'), role: 'user', content: options.shown ?? body, at: new Date().toISOString(),...(pictures.length?{photos:pictures}:{}),...(options.markdown?{markdown:true}:{}),...(quoteSources?{quotes:quoteSources}:{}) };
  // A taken-over turn goes on in the reply the page already has (the last one, if it is the assistant's).
  const lastReply = reattach ? [...(state.sessions.find(s => s.id === sessionId)?.messages ?? [])].reverse()[0] : undefined;
  // A turn the agent started by itself (a background command finished) is a new reply, not the end of the last one.
  const resumedReply = lastReply?.role === 'assistant' && (!options.wake || lastReply.requestId === options.requestId) ? lastReply : undefined;
  let assistantId = resumedReply?.id ?? uid('m');
  const assistantMsg: ChatMessage = {
    id: assistantId,
    role: 'assistant',
    content: '',
    at: new Date().toISOString(),
    streaming: true,
  };

  /* Everything said before this turn, which the model needs in order to have a
     conversation at all.

     It was never sent. The real-engine path builds its prompt as
     `[...history, message]`, and `history` was always empty, so each message
     arrived as the first thing the model had ever heard — no memory of the
     turns above it in the transcript. The demo path hides this, because its
     replies are canned.

     The context meter was reading the same empty array, which is why it sat at
     ~350 tokens however long the conversation grew. */
  const prior = (state.sessions.find((s) => s.id === sessionId)?.messages ?? [])
    .filter(m => !m.timeSkipId && !m.local)
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .filter((m) => m.content.trim())
    .map((m) => ({ role: m.role, content: m.content,photos:m.photos }))
    // Bounded so a long conversation cannot send an unbounded body.
    .slice(-40);

  patchSession(sessionId, (s) => reattach ? ({
    ...s,
    messages: resumedReply
      ? s.messages.map(m => m.id === resumedReply.id ? { ...m, streaming: true } : m)
      : [...s.messages, assistantMsg],
  }) : ({
    ...s,
    defaultTitle: false,
    title: s.messages.length === 0 && !resume && !planStart ? (options.title ?? (shortTitle(body))) : s.title,
    updatedAt: new Date().toISOString(),
    messages: [...s.messages, ...(resume || planStart ? [] : [userMsg]), assistantMsg],
  }));

  const turnAbort = new AbortController();
  markTurn(sessionId, { abort: turnAbort, agent: false });
  // Read by the router to skip view transitions mid-stream, and by CSS to keep
  // the transcript still while tokens arrive.

  const patchAssistant = (fn: (m: ChatMessage) => ChatMessage) =>
    patchSession(sessionId, (s) => ({
      ...s,
      updatedAt: new Date().toISOString(),
      messages: s.messages.map((m) => (m.id === assistantId ? fn(m) : m)),
    }));

  const started = performance.now();

  const imageSlots=photoHistorySlots(prior,pictures.length);
  /* Text and reasoning arrive token by token (90 a second or more). Drawing the whole answer again for each one made the page do more work for every
     word the longer the answer grew (measured: 15.7 s of main-thread time for 24 s of a 40,000-token answer), until the page stopped answering
     and ran out of memory. The pieces are gathered and applied together about ten times a second; any other frame, the end and a failure
     apply what is waiting first, so the order the person sees is the order it was written. */
  const waiting: ['reasoning' | 'delta', string][] = [];
  let waitingTimer: ReturnType<typeof setTimeout> | null = null;
  const applyReasoning = (text: string) => patchAssistant((m) => ({ ...m, reasoning: (m.reasoning ?? '') + text, reasoningStartedAt: m.reasoningStartedAt ?? Date.now() }));
  const applyDelta = (text: string) => patchAssistant((m) => { const content=tidyAnswer(m.content+text); if (m.reasoningStartedAt && m.reasoningMs === undefined) m = { ...m, reasoningMs: Date.now() - m.reasoningStartedAt }; const count=Math.max(1,Math.ceil((content.length+(m.reasoning?.length??0))/4)); return {...m,content,...(m.blocks?{blocks:appendText(m.blocks,text)}:{}),tokens:m.usageSource==='estimated'&&m.tps?m.tokens:count,inputTokens:m.inputTokens??Math.ceil((body.length+prior.reduce((n,p)=>n+p.content.length,0))/4),usageSource:'estimated'}; });
  const flushWaiting = () => {
    if (waitingTimer !== null) { clearTimeout(waitingTimer); waitingTimer = null; }
    for (const [kind, text] of waiting.splice(0)) if (kind === 'delta') applyDelta(text); else applyReasoning(text);
  };
  const queueText = (kind: 'reasoning' | 'delta', text: string) => {
    const last = waiting[waiting.length - 1];
    if (last && last[0] === kind) last[1] += text; else waiting.push([kind, text]);
    waitingTimer ??= setTimeout(flushWaiting, 90);
  };
  try {
    await streamPost(
      reattach ? '/api/chat/attach' : planStart ? '/api/plan/start-goal' : '/api/chat/stream',
      reattach ? { sessionId, ...(options.requestId ? { requestId: options.requestId } : {}) } : {
        sessionId,
        ...(planStart ? { planId: planStart } : {}),
        ...(resume ? { goalAction: 'resume' } : planStart ? {} : { message: options.shown ? body : body }),
        history:await (async()=>{
          /* A picture the browser no longer has (storage cleared, another browser) is left out and named; it used to stop every later message of the chat (GUI-TUI 0.2.17). */
          let missing=false;
          const built=await Promise.all(prior.map(async(m,index)=>{const included=pictureCheck.ok?(m.photos??[]).filter((_,p)=>imageSlots.has(`${index}:${p}`)):[];const omitted=(m.photos??[]).filter(p=>!included.includes(p));const {found,lost}=await readHistoryPictures(included,photoData);if(lost.length)missing=true;const text=m.content+(omitted.length?`\n[Earlier pictures: ${omitted.map(p=>p.name).join(', ')}. Their pixels are outside this model's image context; ask to reattach them if needed.]`:'')+lostPicturesNote(lost);return {role:m.role,content:found.length?[{type:'text',text},...found.map(f=>({type:'image_url',image_url:{url:f.url}}))]:text};}));
          if(missing)pushToast('warn','Some earlier pictures are missing','Pictures from earlier messages are no longer stored in this browser, so the model does not see them in this answer. Attach them again if they matter.');
          return built;
        })(),
        model: state.library?.loaded ?? state.library?.current ?? null,
        remoteModel: state.remoteModel,
        nativeModelId:nativeState()?.instances.length?nativeState()?.defaultId:undefined,
        ...(pictures.length ? { imageIds: await uploadChatPhotos(pictures, sessionId) } : {}),
        useTools:state.sessions.find(s=>s.id===sessionId)?.preferences?.useTools===true,
        // A chat set to plain talk (preferences.agent === false) never runs a tool: the host answers it directly.
        ...(state.sessions.find(s=>s.id===sessionId)?.preferences?.agent===false?{agent:false}:{}),
        thinking:state.sessions.find(s=>s.id===sessionId)?.preferences?.thinking??'Medium',
        thinkingBudget:state.sessions.find(s=>s.id===sessionId)?.preferences?.thinkingBudget??0,
        // The Mode menu's choice for this chat, read when the turn starts; the host applies it to the agent.
        mode:state.sessions.find(s=>s.id===sessionId)?.preferences?.mode??'manual',
        // So a skill narrowed to another workspace does not fire here.
        workspaceId: workspaceIdForSession(state.workspaces, state.sessions.find(s => s.id === sessionId)) ?? state.activeWorkspace,
      },
      (frame) => {
        // A line from the app itself carries `key` (the English sentence): shown in the person's language.
        if (frame.type === 'delta' && typeof frame.key === 'string') frame = { ...frame, text: tr(frame.key, Array.isArray(frame.values) ? frame.values : undefined) };
        const type = frame.type as string;
        if (type === 'delta' || type === 'reasoning') { queueText(type, String(frame.text ?? '')); return; }
        flushWaiting();
        if (type === 'start' && frame.agent === true) { const turn = liveTurns.get(sessionId); if (turn) turn.agent = true; }
        // The host has a turn to take over: the failure the page wrote when it lost the stream no longer applies.
        if (type === 'start' && frame.resumed === true) patchAssistant(m => ({ ...m, error: undefined, errorCode: undefined }));
        if (type === 'queued') {
          patchAssistant(m => ({ ...m, requestId: String(frame.requestId) }));
        } else if (type === 'start') {
          patchAssistant((m) => ({ ...m, demo: frame.demo === true, remoteModel: frame.remoteModel === true, requestId: typeof frame.requestId === 'string' ? frame.requestId : m.requestId, model: String(frame.model), contextMax: Number(frame.contextMax), skillsUsed: Array.isArray(frame.skills) ? frame.skills.map(String) : [] }));
        } else if (type === 'resync') {
          // A taken-over turn: what was said while no page was here. The reply continues from what the page had; where the two disagree the host's text stands.
          const said = String(frame.content ?? ''), thought = String(frame.reasoning ?? '');
          patchAssistant(m => {
            let next = m;
            if (said) next = said.startsWith(next.content) ? { ...next, content: tidyAnswer(said), ...(next.blocks && said.length > next.content.length ? { blocks: appendText(next.blocks, said.slice(next.content.length)) } : {}) } : { ...next, content: tidyAnswer(said), blocks: undefined };
            if (thought && thought.length >= (next.reasoning ?? '').length) next = { ...next, reasoning: thought, reasoningStartedAt: next.reasoningStartedAt ?? Date.now() };
            return next;
          });
        } else if (type === 'status') {
          applyStatusFrame(sessionId, frame as Record<string, unknown>);
          // Manual mode waits for the person: a notification when they have opted in and Strata is in the background (#3477), once per question.
          if (frame.code === 'waiting-approval' && lastApprovalNotice !== `${sessionId}:${String(frame.seq)}`) {
            lastApprovalNotice = `${sessionId}:${String(frame.seq)}`;
            const tool = (frame.args as { tool?: unknown } | undefined)?.tool;
            void approvalNeeded(state.approvalNotify, typeof tool === 'string' ? tool : '');
          }
        } else if (type === 'goal') {
          applyGoalFrame(sessionId, frame.goal);
        } else if (type === 'plan') {
          applyPlanFrame(sessionId, frame.plan);
          // A proposed plan is shown as the plan card in place of this turn's closing text (that text is the plan in prose).
          const planned = frame.plan as { id?: unknown } | null;
          if (frame.event === 'proposed' && planned && typeof planned.id === 'string') { const pid = planned.id; patchAssistant(m => ({ ...m, planId: pid })); }
        } else if (type === 'queued-message') {
          applyQueueFrame(sessionId, frame.queue);
          const sent = frame.message as { text?: unknown } | undefined;
          if (frame.state === 'delivered' && sent && typeof sent.text === 'string') {
            // The host has just sent it into the run: it becomes an ordinary message here, and the answer to it starts a new reply.
            const before = assistantId; assistantId = uid('m');
            const next: ChatMessage = { id: assistantId, role: 'assistant', content: '', at: new Date().toISOString(), streaming: true };
            patchSession(sessionId, sess => ({ ...sess, messages: [...sess.messages.map(m => m.id === before ? { ...m, streaming: false } : m), { id: uid('m'), role: 'user', content: sent.text as string, at: new Date().toISOString(), ...(((q) => (q ? { quotes: q } : {}))(takeQuoteSources(String(sent.text ?? '')))) }, next] }));
          }
        } else if (type === 'queued-release') {
          // The goal ended with messages left: they go out now, in order, as ordinary messages once this turn has finished.
          const left = Array.isArray(frame.messages) ? frame.messages.flatMap(q => (q && typeof (q as { text?: unknown }).text === 'string' ? [String((q as { text: string }).text)] : [])) : [];
          if (left.length) releasedMessages.set(sessionId, [...(releasedMessages.get(sessionId) ?? []), ...left]);
        } else if (type === 'plan-step') {
          /* the whole plan follows in a plan frame; the live status already says which step is running */
        } else if (type === 'reasoning') {
          /* applied through queueText above */
        } else if (type === 'notice') {
          // The host says why this chat is not using the agent. One toast per reason per page load.
          const code = String(frame.code ?? 'notice');
          if (!noticed.has(code)) { noticed.add(code); pushToast('info', 'Agent', plainFailureText(String(frame.message ?? '')).message); }
        } else if (type === 'agent-question') {
          addAgentAsk(sessionId, frame as { callId?: unknown; questions?: unknown });
        } else if (type === 'agent-question-end') {
          removeAgentAsk(sessionId, String(frame.callId ?? ''));
        } else if (type === 'agent-tool-draft') {
          // A tool call (a whole file, say) is being written: keep its name, size and the tail of its text on the reply until the call is whole or lost.
          const d = frame.draft as (ToolDraft & { done?: boolean }) | undefined;
          if (d && typeof d.id === 'string') {
            const { done, ...view } = d;
            patchAssistant(m => {
              const rest = (m.drafts ?? []).filter(x => x.id !== d.id);
              return { ...m, drafts: done ? (rest.length ? rest : undefined) : [...rest, view] };
            });
          }
        } else if (type === 'agent-tool') {
          const tool = frame.tool as ToolCall;
          patchAssistant(m => {
            const existing = m.tools?.find(t => t.id === tool.id);
            const next: ToolCall = { ...existing, ...tool, live: true };
            return { ...m, tools: existing ? m.tools!.map(t => t.id === tool.id ? next : t) : [...(m.tools ?? []), next] };
          });
        } else if (type === 'agent') {
          patchSession(sessionId, s => ({ ...s, agents: updateAgent(s.agents ?? [], frame.agent) }));
        } else if (type === 'delta') {
          /* applied through queueText above */
        } else if (type === 'deliverable') {
          const file = readDeliverable(frame);
          if (file) patchAssistant(m => ({ ...m, deliverables: mergeDeliverable(m.deliverables, file) }));
        } else if (type === 'skills-used') {
          const skills = readSkills(frame);
          if (skills.length) patchAssistant(m => ({ ...m, skills }));
        } else if (type === 'activity-group') {
          const group = readGroup(frame);
          if (group) patchAssistant(m => ({ ...m, blocks: upsertGroup(m.blocks ?? (m.content ? appendText(undefined, m.content) : undefined), group) }));
        } else if (type === 'artifacts' && Array.isArray(frame.artifacts)) {
          patchAssistant(m => ({ ...m, artifacts: frame.artifacts as PreviewArtifact[] }));
        } else if(type==='native-tool'){
          const event=frame.event as {event:string;id:string;name?:string;arguments?:Record<string,unknown>;text?:string;ok?:boolean;ms?:number};
          if(event.id)patchAssistant(m=>{const existing=m.tools?.find(t=>t.id===event.id);const tool={id:event.id,name:event.name??existing?.name??'Connected tool',args:event.arguments??existing?.args??{},preview:'Authorized connected tool',effect:'Access configured in the MCP server',scope:'connected' as const,state:event.event==='result'?event.ok?'done' as const:'error' as const:'running' as const,result:event.text,durationMs:event.ms};return {...m,tools:[...(m.tools??[]).filter(t=>t.id!==tool.id),{...existing,...tool}]};});
        } else if (type === 'tool') {
          const tool = frame.tool as ChatMessage['tools'] extends (infer T)[] | undefined ? T : never;
          patchAssistant((m) => ({ ...m, tools: [...(m.tools ?? []), { ...tool, state: 'proposed' }] }));
        } else if (type === 'stats') {
          patchAssistant((m) => ({
            ...m,
            tokens: typeof frame.tokensOut === 'number' ? frame.tokensOut : m.tokens,
            tps: typeof frame.tps === 'number' ? frame.tps : m.tps,
            promptTps: typeof frame.promptTps === 'number' ? frame.promptTps : m.promptTps,
          }));
        } else if (type === 'context') {
          // What the context holds after each model request of an agent turn: the meter follows the turn instead of waiting for its end.
          const used = Number(frame.contextTokens), max = Number(frame.contextMax);
          if (Number.isFinite(used) && used > 0) patchAssistant((m) => ({ ...m, contextUsed: used, ...(Number.isFinite(max) && max > 0 ? { contextMax: max } : {}) }));
        } else if (type === 'error') {
          clearStatus(sessionId);
          // A sentence the host wrote for the person (it carries its catalog key) is shown in their language as it is; anything else is made plain here.
          // An engine's own line (or a stop it restarts from) becomes a plain sentence; the line itself is kept for Details (chatFailure, 11-16 review).
          const failure = typeof frame.key === 'string' ? null : chatFailure(String(frame.message ?? 'Engine error'));
          const said = typeof frame.key === 'string' ? tr(frame.key, Array.isArray(frame.values) ? frame.values : undefined) : failure!.message;
          const detail = typeof frame.details === 'string' && frame.details ? frame.details : failure && failure.kind !== 'plain' && failure.detail && failure.detail !== said ? failure.detail : undefined;
          patchAssistant((m) => ({ ...m, drafts: undefined, error: said, streaming: false, ...(typeof frame.code === 'string' ? { errorCode: frame.code } : {}), ...(detail ? { errorDetail: detail } : {}) }));
        } else if (type === 'done') {
          clearStatus(sessionId);
          workFinished(state.doneSound, Number(frame.durationMs ?? performance.now() - started));
          patchSession(sessionId, (sess) => ({
            ...sess,
            tokensIn: sess.tokensIn + Number(frame.tokensIn ?? 0),
            tokensOut: sess.tokensOut + Number(frame.tokensOut ?? 0),
            statsAvailable: typeof frame.tokensIn === 'number' && typeof frame.tokensOut === 'number' && sess.messages.filter(m=>m.role==='assistant'&&!m.streaming).every(m=>typeof m.inputTokens==='number'),
          }));
          patchAssistant((m) => ({
            ...m,
            streaming: false,
            drafts: undefined,
            durationMs: Number(frame.durationMs ?? performance.now() - started),
            tokens: typeof frame.tokensOut === 'number' ? frame.tokensOut : undefined,
            ttftMs: typeof frame.ttftMs === 'number' ? frame.ttftMs : undefined,
            inputTokens: typeof frame.tokensIn === 'number' ? frame.tokensIn : undefined,
            contextUsed: typeof frame.contextTokens === 'number' ? frame.contextTokens : undefined,
            usageSource: frame.usageSource === 'reported' || frame.usageSource === 'demo' ? frame.usageSource : 'unavailable',
            tps: typeof frame.tps === 'number' ? frame.tps : m.tps,
            promptTps: typeof frame.promptTps === 'number' ? frame.promptTps : m.promptTps,
          }));
          // The final text is in `state`: write it once now instead of waiting for the coalescing timer.
          flushSessions();
        }
      },
      turnAbort.signal,
    );
  } catch (err) {
    flushWaiting();
    if (reattach && (err as { status?: number } | null)?.status === 404) {
      // Nothing to take over (the turn ended and the host no longer holds it): the reply stays as the page had it.
      patchAssistant((m) => ({ ...m, streaming: false }));
    } else if ((err as Error).name !== 'AbortError') {
      patchAssistant((m) => ({
        ...m,
        streaming: false,
        error: err instanceof Error ? err.message : 'Engine error',
      }));
      pushToast('error', 'Generation failed', err instanceof Error ? err.message : undefined);
    } else {
      patchAssistant((m) => ({ ...m, streaming: false }));
    }
  } finally {
    flushWaiting();
    // A cancelled or failed turn is still worth keeping.
    flushSessions();
    clearStatus(sessionId);
    markTurn(sessionId, null);
    removeAgentAsk(sessionId);
    const next = releasedMessages.get(sessionId)?.shift();
    if (next !== undefined) { if (!releasedMessages.get(sessionId)?.length) releasedMessages.delete(sessionId); setTimeout(() => void sendMessage(next), 0); }
  }
}

/** Messages the host handed back when a goal ended with some still waiting; sent one after another. */
const releasedMessages = new Map<string, string[]>();

/* -------------------------------------------------------------- tool grants */

const noticed = new Set<string>();

/**
 * Approve or deny a tool the agent asked about. The answer goes to the host, which hands it to the
 * Harness approval for exactly this chat and call; the card then follows what the Harness reports
 * (running, done, denied). The card shows the choice straight away and goes back if the host refuses.
 */
function answerAgentTool(sessionId: string, messageId: string, tool: ToolCall, approved: boolean) {
  const setState = (state_: ToolCall['state'], result?: string) => patchSession(sessionId, s => ({
    ...s, messages: s.messages.map(m => m.id === messageId ? { ...m, tools: (m.tools ?? []).map(t => t.id === tool.id ? { ...t, state: state_, ...(result !== undefined ? { result } : {}) } : t) } : m),
  }));
  setState(approved ? 'running' : 'denied');
  api.post('/api/chat/approval', { sessionId, toolId: tool.id, decision: approved ? 'approve' : 'deny' }).catch(error => {
    // #5308: 409 is the host saying nothing is waiting for this answer any more (the turn ended, the assistant restarted): it can never be
    // delivered, so the card ends with the host's sentence instead of asking again; any other failure puts the question back to try again.
    if (error instanceof ApiError && error.status === 409) setState('error', error.message);
    else setState('proposed');
    pushToast('error', 'That answer was not delivered', error instanceof Error ? error.message : undefined);
  });
}

/**
 * Approving a tool in the design build resolves locally: there is no tool
 * runtime behind it yet, so the result is a labelled stand-in. The interaction —
 * what is asked, what is shown, what needs a grant — is the real deliverable.
 */
export function resolveTool(messageId: string, toolId: string, approved: boolean) {
  const sessionId = state.activeSession;
  if (!sessionId) return;
  const tool = state.sessions.find(s => s.id === sessionId)?.messages.find(m => m.id === messageId)?.tools?.find(t => t.id === toolId);
  if (tool?.live) { if (tool.state === 'proposed') answerAgentTool(sessionId, messageId, tool, approved); return; }   // an ended card (#5308) is not answered again

  patchSession(sessionId, (s) => ({
    ...s,
    messages: s.messages.map((m) =>
      m.id === messageId
        ? {
            ...m,
            tools: (m.tools ?? []).map((t) =>
              t.id === toolId
                ? {
                    ...t,
                    state: approved ? 'done' : 'denied',
                    durationMs: undefined,
                    result: approved ? 'Approval recorded. Nothing was executed.' : 'Denied. Nothing ran.',
                  }
                : t,
            ),
          }
        : m,
    ),
  }));

}

/* ------------------------------------------------------------------- routes */

export function go(route: Route, param?: string) {
  navigate(route, param);
  // Focus mode is a reading mode for the conversation. Leaving it to work on
  // something else leaves the mode, rather than stranding a bare shell.
  if (state.focus && route !== 'chat' && route !== 'settings') setFocus(false);
  set({ paletteOpen: false, navOpen: false });
}

/* ---------------------------------------------------------------- bootstrap */

export function bootOnce() {
  if ((globalThis as { __strataBooted?: boolean }).__strataBooted) return;
  (globalThis as { __strataBooted?: boolean }).__strataBooted = true;
  void bootstrap();
}

let sessionsResumed = false;
/**
 * Once, after the chats were first read from the host: what a chat can still have going on. Time-skip runs are followed again, a loop the last
 * window left "running" has ended, a turn the host is still running is taken over (or one that ended while the page was away is shown whole),
 * and tool cards the browser remembers as waiting for a turn that no longer exists are marked interrupted.
 */
async function resumeSessions() {
  if (sessionsResumed) return;
  sessionsResumed = true;
  for (const session of state.sessions) for (const run of session.timeSkips ?? []) { scheduleTimeSkip(session.id, run); if (run.job && !run.completedAt) { markTurn(session.id, { abort: new AbortController(), agent: false }); void followTimeSkip(session.id, run.id); } }
  // A loop lives in the window: one the last window left "running" has ended, and the board says so.
  if (state.sessions.some(s => s.loop?.state === 'running')) {
    const sessions = state.sessions.map(s => s.loop?.state === 'running' ? { ...s, loop: { ...s.loop, state: 'ended' as const, endedWhy: 'closed' as const, endedAt: Date.parse(s.updatedAt) || Date.now() } } : s);
    persistSessions(sessions);
    set({ sessions });
  }
  await takeOverTurns();
  void reconcileAgentTools();
}

/**
 * A reload does not stop an agent turn: the host keeps it going. The page asks which chats have a turn running and takes each one over again
 * (the catch-up, then the rest as it is written, the Stop button and the approvals waiting). A chat whose last reply was cut off by the page
 * going away is asked about too: if its turn ended meanwhile, the host still holds the end of it and shows it whole; if not, nothing changes.
 */
async function takeOverTurns() {
  let running: { chatId: string; requestId?: string; wake?: boolean }[] = [];
  let woken: { chatId: string; requestId: string }[] = [];
  try { const active = await api.get<{ turns: { chatId: string; requestId?: string; wake?: boolean }[]; woken?: { chatId: string; requestId: string }[] }>('/api/chat/active'); running = active.turns; woken = active.woken ?? []; } catch { /* an older host, or none reachable: nothing to take over */ }
  const live = new Set(running.map(t => t.chatId));
  for (const t of running) if (state.sessions.some(s => s.id === t.chatId) && !liveTurns.has(t.chatId)) void sendMessage('', { reattach: t.chatId, ...(t.wake && t.requestId ? { requestId: t.requestId, wake: true } : {}) });
  // Replies the agent began by itself while no page was open (a background command finished), read whole.
  // A chat can have several (two background commands finished): each becomes its own reply, in order, one after the other.
  const wokenByChat = new Map<string, string[]>();
  for (const w of woken) if (state.sessions.some(s => s.id === w.chatId) && !live.has(w.chatId)) wokenByChat.set(w.chatId, [...(wokenByChat.get(w.chatId) ?? []), w.requestId]);
  for (const [chatId, requestIds] of wokenByChat) if (!liveTurns.has(chatId)) void requestIds.reduce<Promise<unknown>>((before, requestId) => before.then(() => sendMessage('', { reattach: chatId, requestId, wake: true })), Promise.resolve());
  // Cut-off replies of chats with no turn running: the most recent few, asked once each.
  const cut = state.sessions.filter(s => !live.has(s.id) && !liveTurns.has(s.id)).slice().sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1)).slice(0, 12).flatMap(s => {
    const last = s.messages[s.messages.length - 1];
    return last?.role === 'assistant' && last.requestId && last.usageSource === undefined && !last.local ? [{ id: s.id, requestId: last.requestId }] : [];
  });
  await Promise.allSettled(cut.map(c => sendMessage('', { reattach: c.id, requestId: c.requestId })));
}

/**
 * After a reload, a tool card the browser remembers as waiting or running may belong to a turn that no
 * longer exists. Cards whose approval the host still holds stay answerable; the rest are marked as
 * interrupted so nothing looks like it is still working.
 */
async function reconcileAgentTools() {
  const open = (t: ToolCall) => t.live && (t.state === 'proposed' || t.state === 'running');
  for (const session of state.sessions) {
    if (liveTurns.has(session.id)) continue;   // a turn taken over again reports its own cards
    if (!session.messages.some(m => m.tools?.some(open))) continue;
    let waiting = new Set<string>();
    try {
      const r = await api.get<{ pending: { callId: string }[] }>(`/api/agent/pending?chatId=${encodeURIComponent(session.id)}`);
      waiting = new Set(r.pending.map(p => p.callId));
    } catch { /* The host is unreachable; leave the cards for the next load. */ continue; }
    patchSession(session.id, sess => ({ ...sess, messages: sess.messages.map(m => m.tools?.some(open) ? { ...m, tools: m.tools.map(t => open(t) && !waiting.has(t.id) ? { ...t, state: 'error' as const, result: 'The page closed before this finished. Check the workspace to see what it did.' } : t) } : m) }));
  }
}


export function setContrast(contrast: 'standard' | 'increased') { set({ contrast }); savePrefs(); }
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.theme === 'system') applyPrefs(); });

export async function loadRuntimeModel(body:unknown){
  const result=await api.post<{state:RuntimeModels;instance:{id:string}}>('/api/runtime/load',body);
  set({runtime:result.state});return result.instance.id;
}
export async function runtimeModelAction(id:string,action:'unload'|'default'){
  set({runtime:await api.post<RuntimeModels>(`/api/runtime/${id}/${action}`)});
}

export async function changeModelDirectory(directory: string, unload: boolean) {
  set(s => ({ busy: { ...s.busy, modelDirectory: true } }));
  try {
    const result = await api.post<{ library: ModelLibrary; imageModels: ImageModelLibrary; runtime: RuntimeModels; hardware: Hardware; setup: SetupState }>('/api/storage/models-directory', { directory, unload });
    set(result);
    return result;
  } finally {
    set(s => ({ busy: { ...s.busy, modelDirectory: false } }));
  }
}

export async function addLocalModel(path: string, name?: string) {
  const result = await api.post<{ library: ModelLibrary; variantId: string }>('/api/local-models', { path, name });
  set({ library: result.library });
  return result.variantId;
}
