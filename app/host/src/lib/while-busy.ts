/** What Enter does when the person sends a message while the assistant is still answering (#4783). The web chat page has the same three. */
export type WhileBusy = 'wait' | 'queue' | 'interject';
export const WHILE_BUSY: readonly WhileBusy[] = ['wait', 'queue', 'interject'];
export const validWhileBusy = (value: unknown): value is WhileBusy => WHILE_BUSY.includes(value as WhileBusy);

/**
 * What to do with the message, as one word:
 *  - 'refuse'    Wait: nothing is sent until the answer is done (the message stays in the box, as always);
 *  - 'goal'      Send after, while a goal runs: the host's own queue delivers it between steps (it asks the host first, so this is only a try);
 *  - 'park'      Send after: the message waits here and goes when the answer is done;
 *  - 'interrupt' Interrupt: the answer stops where it is (what is written stays) and the message goes as soon as it has stopped.
 */
export type BusyPlan = 'refuse' | 'goal' | 'park' | 'interrupt';
export function busyPlan(mode: WhileBusy): BusyPlan {
  return mode === 'queue' ? 'goal' : mode === 'interject' ? 'interrupt' : 'refuse';
}

/**
 * #5806: may the chat on screen send its parked message now? Only when it is also the store's active chat, which is where sendMessage sends:
 * a click in the side bar makes the other chat active before the page shows it, and in that moment the message went to the other chat.
 */
export function parkedReady(viewed: string | null | undefined, active: string | null | undefined, streaming: boolean, parked: number): boolean {
  return !!viewed && viewed === active && !streaming && parked > 0;
}
