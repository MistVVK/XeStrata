/**
 * When a chat follows the newest token, and when the reader has let go (#4014).
 *
 * The old rule asked "is the reader within 120 px of the bottom?" on every streamed token and pulled the view
 * back if so. One wheel notch is about 100 px, so it always landed inside that band and was undone by the next
 * token; only a scrollbar drag, which jumps further, got away. The rule now follows what the reader did:
 * a wheel, finger or key moving up lets go at once, and the chat follows again when the reader is back at
 * the bottom.
 */
export interface Scroller { scrollHeight: number; scrollTop: number; clientHeight: number }

/** Pixels of slack that still count as "at the bottom" (the same as `atBottom` in reasoning.ts). */
export const FOLLOW_SLACK = 24;
/** Past this far from the bottom the "jump to latest" button is offered. */
export const PILL_DISTANCE = 400;
/** A scroll this soon after the reader's own input was the reader's. */
export const INPUT_WINDOW_MS = 400;

export const distanceFromBottom = (el: Scroller): number => el.scrollHeight - el.scrollTop - el.clientHeight;

/**
 * Whether the chat should keep following after a scroll event.
 * - back at the bottom: follow again;
 * - the reader moved it (a wheel, finger, key or the scrollbar a moment ago) and it is not at the bottom: let go;
 * - far away whatever moved it (a jump): let go;
 * - otherwise (the page grew under a following reader and no input came): keep the state it had.
 */
export function followAfterScroll(distance: number, wasFollowing: boolean, msSinceInput: number): boolean {
  if (distance <= FOLLOW_SLACK) return true;
  if (msSinceInput <= INPUT_WINDOW_MS || distance > PILL_DISTANCE) return false;
  return wasFollowing;
}

/** A wheel turned up (negative delta) or a finger dragging the content down: the reader wants to go back. */
export const wheelLeavesBottom = (deltaY: number): boolean => deltaY < 0;

/** Keys that move a scroller up. */
export const KEYS_UP = new Set(['PageUp', 'ArrowUp', 'Home']);
