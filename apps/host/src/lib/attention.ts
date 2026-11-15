/**
 * Telling the person that Strata needs them or has finished (#3586, #3477). Both are opt-in and off by default.
 *
 * - A short sound when an answer or task that took a while is finished, or when Strata is in the background (public/strata-done-sound.wav,
 *   made for Strata from sine waves, no sample).
 * - A desktop notification when Manual mode waits for an approval and Strata is in the background. The desktop app shows it through its own
 *   shell (the host passes it on, so it also works while the window is hidden in the tray); a browser uses the page's notifications.
 */
import { api } from './client';
import { tr } from './i18n';

/** An answer that came back faster than this does not chime while the person is looking at the window. */
export const QUICK_MS = 8000;
const SOUND_URL = '/strata-done-sound.wav';
const MIN_GAP_MS = 2000;
let lastPlayed = 0;

export interface DoneContext { enabled: boolean; durationMs: number; focused: boolean; now?: number }

/** Whether the finished-work sound should play now: only when asked for, only for work that took a while or when the window is not in front. */
export function shouldPlayDone({ enabled, durationMs, focused, now = Date.now() }: DoneContext): boolean {
  if (!enabled) return false;
  if (now - lastPlayed < MIN_GAP_MS) return false;
  return !focused || durationMs >= QUICK_MS;
}

export const windowInFront = (): boolean => typeof document !== 'undefined' && document.visibilityState === 'visible' && document.hasFocus();

export function playDoneSound(make: (url: string) => { volume: number; play(): Promise<void> | void } = url => new Audio(url)): boolean {
  try {
    const audio = make(SOUND_URL);
    audio.volume = 0.6;
    lastPlayed = Date.now();
    const started = audio.play();
    if (started && typeof (started as Promise<void>).catch === 'function') (started as Promise<void>).catch(() => { /* the browser asked for a click first: no sound this time */ });
    return true;
  } catch {
    return false;
  }
}

/** Called when an answer or a task is finished. */
export function workFinished(enabled: boolean, durationMs: number): void {
  if (shouldPlayDone({ enabled, durationMs, focused: windowInFront() })) playDoneSound();
}

export interface Notice { title: string; body: string }

/** The sentence of the approval notification, in the person's language. */
export function approvalNotice(tool: string): Notice {
  return { title: tr('Strata needs your answer'), body: tr('The assistant wants to use {0}. Open Strata to approve or deny.', [tool || tr('a tool')]) };
}

/** #5617: the agent asked the person a question (ask_user_question): the approval notification's own title, with no body (no new sentence). */
export function questionNotice(): Notice {
  return { title: tr('Strata needs your answer'), body: '' };
}

/** The question notification, under the same opt-in as the approval one ("Notify me when Strata needs an answer") and only when Strata is not in front. */
export function questionAsked(enabled: boolean, focused = windowInFront()): Promise<boolean> {
  if (!enabled || focused) return Promise.resolve(false);
  return showNotice(questionNotice());
}

/** Shows a notification through the desktop shell when there is one, else through the page. Returns whether something was shown. */
export async function showNotice(notice: Notice, deps: { post?: (body: Notice) => Promise<{ shown?: boolean }>; page?: (notice: Notice) => boolean } = {}): Promise<boolean> {
  const post = deps.post ?? (body => api.post<{ shown?: boolean }>('/api/shell/notify', body));
  try {
    const result = await post(notice);
    if (result?.shown) return true;
  } catch { /* no shell here: the page's own notification below */ }
  return (deps.page ?? pageNotification)(notice);
}

function pageNotification(notice: Notice): boolean {
  try {
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return false;
    new Notification(notice.title, { body: notice.body, tag: 'strata-approval' });
    return true;
  } catch {
    return false;
  }
}

/** Manual mode waits for an approval. Only a window that is not in front needs the notification. */
export function approvalNeeded(enabled: boolean, tool: string, focused = windowInFront()): Promise<boolean> {
  if (!enabled || focused) return Promise.resolve(false);
  return showNotice(approvalNotice(tool));
}

/** Turning approval notifications on in a browser asks the page's permission once; the desktop shell needs none. */
export async function askNotificationPermission(): Promise<void> {
  try {
    if (typeof Notification !== 'undefined' && Notification.permission === 'default') await Notification.requestPermission();
  } catch { /* some web views have none */ }
}
