/**
 * Which copy of a file the sidebar shows (#5180 follow-up: the preview kept a file's first content after the file changed).
 *
 * A preview tab can come from two places: the person (or a finished run) opening a file, which reads it from the host now, and the
 * message the file was made in, which carries the content it had then. They share one id (`file:<path>`). The rules, as pure functions:
 *   - opening a file replaces the tab in place when the content differs, and keeps the very same array when it does not;
 *   - a message's artifact only adds a tab that is missing, or replaces one that is OLDER than the message and different; it never
 *     overwrites a copy that was read later;
 *   - both return the array they were given when nothing changed, so no state update, no render and no effect follows: there is no loop to
 *     start however often they are called (the same artifacts are handed over again whenever a message arrives).
 * The panel reloads when `previewVersion` changes, and only then.
 */
import type { PreviewArtifact } from './artifacts';
import { textRevision } from './preview-render';

/** What identifies the content of a preview: the host's version for a file (modified time and size), else a hash of the text, else the address. */
export function previewVersion(a: Pick<PreviewArtifact, 'version' | 'content' | 'url'>): string {
    if (a.content !== undefined)
        return 'c' + textRevision(a.content);
    return a.version ? 'v' + a.version : 'u' + (a.url ?? '');
}

const sameContent = (a: PreviewArtifact, b: PreviewArtifact) => a.kind === b.kind && a.title === b.title && previewVersion(a) === previewVersion(b);

/** A file was opened (read from the host just now): it takes the place of its tab when it differs. */
export function openItem(old: PreviewArtifact[], item: PreviewArtifact): PreviewArtifact[] {
    const at = old.findIndex(a => a.id === item.id);
    if (at < 0)
        return [...old, item];
    if (sameContent(old[at], item))
        return old;
    const next = old.slice();
    next[at] = item;
    return next;
}

/**
 * The artifacts the messages carry, merged into the tabs. `items` are in message order, each stamped with the time of its message (`at`);
 * `closed` are the ids the person closed.
 */
export function syncArtifacts(old: PreviewArtifact[], items: PreviewArtifact[], closed: readonly string[] | undefined = []): PreviewArtifact[] {
    let next = old;
    const edit = () => { if (next === old) next = old.slice(); };
    for (const item of items) {
        const at = next.findIndex(a => a.id === item.id);
        if (closed?.includes(item.id)) {
            if (at >= 0) { edit(); next.splice(at, 1); }
            continue;
        }
        if (at < 0) {
            edit();
            next.push(item);
            continue;
        }
        const have = next[at];
        // only a message newer than the copy on show, with different content, replaces it
        if (item.at !== undefined && have.at !== undefined && item.at > have.at && !sameContent(have, item)) {
            edit();
            next[at] = item;
        }
    }
    return next;
}

/** The message artifacts of a session, in order and de-duplicated by id (the last of an id wins), each stamped with its message's time. */
export function messageArtifacts(messages: ReadonlyArray<{ at?: string; artifacts?: PreviewArtifact[] }> | undefined): PreviewArtifact[] {
    const byId = new Map<string, PreviewArtifact>();
    for (const m of messages ?? []) {
        const at = m.at ? Date.parse(m.at) : NaN;
        for (const a of m.artifacts ?? [])
            byId.set(a.id, Number.isFinite(at) && a.at === undefined ? { ...a, at } : a);
    }
    return [...byId.values()];
}
