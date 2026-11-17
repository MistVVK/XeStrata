/**
 * The Enter that commits an IME conversion must not also send the message (#5071, Japanese input on macOS).
 *
 * The browsers disagree on the order of the events when Enter commits a conversion:
 *   Chrome, Edge (WebView2), Firefox:  keydown(Enter, keyCode 229, isComposing true)  ->  compositionend
 *   Safari (WKWebView):                compositionend  ->  keydown(Enter, keyCode 229, isComposing FALSE)
 *   WebKitGTK (IBus), as reported:     compositionend  ->  keydown(Enter, keyCode 13, isComposing false)
 * `isComposing` alone only covers the first. The rule here covers all three:
 *   1. a key event that says it is composing (isComposing) or that is the IME's own (keyCode 229) belongs to the IME;
 *   2. the first Enter right after a compositionend, in the same tick (a few milliseconds), is the one that committed;
 *   3. every other key is the person's. A second Enter after the conversion is committed sends, exactly once.
 * The flag of rule 2 is armed by compositionend, used up by the next key whatever it is, and expires on its own, so it can never leave
 * the composer unable to send. Times are the events' own `timeStamp`s (one clock).
 */

/** How long after a compositionend an Enter still counts as the one that committed. The three engines deliver it in the same tick. */
export const IME_COMMIT_WINDOW_MS = 50;

export interface ImeKey {
    key: string;
    isComposing?: boolean;
    keyCode?: number;
}

export interface ImeGuard {
    compositionStart(): void;
    compositionEnd(now: number): void;
    /** True when this keydown belongs to the IME: the composer must neither send nor run a menu or shortcut action for it. */
    keyDown(event: ImeKey, now: number): boolean;
    /** The textarea lost the focus: a conversion that was cut short never sends a compositionend. */
    reset(): void;
}

export function createImeGuard(windowMs = IME_COMMIT_WINDOW_MS): ImeGuard {
    let armedAt: number | null = null;
    return {
        compositionStart() {
            armedAt = null;
        },
        compositionEnd(now) {
            armedAt = now;
        },
        keyDown(event, now) {
            const armed = armedAt;
            armedAt = null;                     // the next key of any kind uses the flag up
            if (event.isComposing || event.keyCode === 229)
                return true;
            return armed !== null && event.key === 'Enter' && now - armed >= 0 && now - armed <= windowMs;
        },
        reset() {
            armedAt = null;
        },
    };
}
