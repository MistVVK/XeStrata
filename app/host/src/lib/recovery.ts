/**
 * After a crash of the page's renderer (#5079). The desktop shell loads the page again at `/?recovery=<n>`; the page then starts with the
 * sidebar previews closed, once, so the file that may have caused the crash is not opened again by itself. The chats are the host's and
 * are all still there; only what the sidebar showed is deferred, and only until the person opens a preview themselves.
 */

/** 1..9 from a `recovery=<n>` query, 0 when there is none or it is not a number. */
export function parseRecovery(search: string): number {
    const value = new URLSearchParams(search).get('recovery');
    if (value === null || !/^[1-9]$/.test(value))
        return 0;
    return Number(value);
}

/** The query without the recovery parameter (and with its `?` only when something is left). */
export function withoutRecovery(search: string): string {
    const params = new URLSearchParams(search);
    params.delete('recovery');
    const rest = params.toString();
    return rest ? `?${rest}` : '';
}

let deferred = false;
let deferredSession: string | null = null;

/**
 * Read the shell's recovery mark from the address, remove it (a later reload of the page is an ordinary start), and defer the previews.
 * Called once, before the first render. Returns the recovery number, 0 for an ordinary start.
 */
export function startRecovery(): number {
    let level = 0;
    try {
        level = parseRecovery(window.location.search);
        if (level) {
            const { pathname, search, hash } = window.location;
            window.history.replaceState(window.history.state, '', pathname + withoutRecovery(search) + hash);
        }
    }
    catch { /* no address to read: an ordinary start */ }
    deferred = level > 0;
    deferredSession = null;
    return level;
}

/**
 * True while the sidebar previews stay closed after a recovery: for the chat that was open when the renderer died, until the person
 * opens a preview themselves or goes to another chat. The first chat asked about is that chat.
 */
export function previewsDeferred(sessionId?: string): boolean {
    if (!deferred)
        return false;
    if (sessionId === undefined)
        return true;
    deferredSession ??= sessionId;
    if (deferredSession !== sessionId) {
        deferred = false;
        return false;
    }
    return true;
}

/** The person opened a preview themselves: from now on previews open as usual. */
export function endPreviewDeferral(): void {
    deferred = false;
}
