/**
 * Work limits for the Markdown preview in the sidebar (#5180).
 *
 * The preview used to parse and rebuild the whole file on every render of the chat, which is once per group of streamed tokens, and a
 * 60 KB file with many tables and code blocks is tens of thousands of React elements each time. The pieces here are pure text work:
 *   - a revision key of the text, so an unchanged file is the same key however often the chat renders;
 *   - the file cut into slices at block boundaries, each with its own key, so a change in one place re-renders one slice;
 *   - limits: how many slices are drawn at once, when a block is shown as plain text, and when the whole file is.
 * The components that use them are components/PreviewMarkdown.tsx and components/ArtifactPanel.tsx.
 */

/** Slices of about this many characters; a slice is the unit that is rendered, kept and skipped. */
export const PREVIEW_SLICE_CHARS = 6000;
/** Slices drawn at once when a file opens; the rest follow a few at a time, so the first paint is bounded however long the file is. */
export const PREVIEW_FIRST_SLICES = 2;
/** Slices added by each step after that. At most one step is waiting at any time. */
export const PREVIEW_STEP_SLICES = 2;
/** One block (a table or a fenced code block that cannot be cut) above this is shown as plain text, not parsed. */
export const PREVIEW_BLOCK_LIMIT = 40_000;
/** A file above this is shown as plain text (the Source view); the Read view would be thousands of elements per render. */
export const PREVIEW_RENDER_LIMIT = 200_000;

/** 32-bit FNV-1a over the UTF-16 units, in base 36; with the length it is the revision key of a text. Not a security hash. */
export function textRevision(text: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        h ^= text.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return `${text.length.toString(36)}-${h.toString(36)}`;
}

export interface PreviewSlice {
    /** Stable for the same text at the same place: a slice that did not change keeps its key and is not rendered again. */
    key: string;
    text: string;
    /** Too large to parse: shown as plain text. */
    plain: boolean;
}

/** The longest run of lines a `$$` or `\[` block may span; lib/math.ts displayBlock reads at most this many. */
const MATH_LINES = 60;

/**
 * Cut Markdown into slices of about `target` characters, only at a blank line that is outside a fenced code block and outside a
 * display-math block. Blank lines already end a paragraph, a list and a table in components/Markdown.tsx, so the slices drawn one
 * after the other are the same elements as the whole text drawn once.
 */
export function splitPreview(text: string, target = PREVIEW_SLICE_CHARS): PreviewSlice[] {
    const lines = text.split(/\r\n|\r|\n/);
    const slices: PreviewSlice[] = [];
    let start = 0, size = 0, fenced = false, mathEnd = -1;
    const emit = (end: number) => {
        const body = lines.slice(start, end).join('\n');
        if (body.trim()) {
            const plain = body.length > PREVIEW_BLOCK_LIMIT;
            slices.push({ key: `${slices.length}:${textRevision(body)}`, text: body, plain });
        }
        start = end;
        size = 0;
    };
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        size += line.length + 1;
        if (/^\s*```/.test(line)) {
            fenced = !fenced;
            continue;
        }
        if (fenced)
            continue;
        const t = line.trim();
        if (i > mathEnd) {
            const opens = t.startsWith('$$') ? '$$' : t.startsWith('\\[') ? '\\[' : '';
            if (opens && !t.slice(2).includes(opens === '$$' ? '$$' : '\\]'))
                mathEnd = i + MATH_LINES;
        }
        else if (t.includes('$$') || t.includes('\\]'))
            mathEnd = -1;
        // a blank line outside code and math, once the slice is big enough, ends it (the blank line goes with the slice before)
        if (!t && i > mathEnd && size >= target && i + 1 < lines.length)
            emit(i + 1);
    }
    emit(lines.length);
    return slices;
}

/** How a file is shown: drawn as Markdown in slices, or as plain text when it is above the limit. */
export function previewTier(text: string): 'markdown' | 'plain' {
    return text.length > PREVIEW_RENDER_LIMIT ? 'plain' : 'markdown';
}

/**
 * How many slices are drawn after a step. `shown` is the count now; the new text may have fewer or more slices than before. A change
 * in the file never takes back slices already on screen (the page would jump), it only adds what is still missing.
 */
export function slicesAfter(shown: number, total: number, step = PREVIEW_STEP_SLICES): number {
    return Math.min(total, Math.max(shown, 0) + step);
}

/** The count to draw before any step has run for this text: the ones already shown, or the first few. */
export function slicesAtStart(shown: number, total: number): number {
    return Math.min(total, Math.max(shown, PREVIEW_FIRST_SLICES));
}

/**
 * Parsing cache: the slices of the last few texts, found by revision. A chat that renders again with the same file asks for the same
 * text, and the answer is the same array (same identity), so nothing downstream sees a change. `stats.cuts` counts the real work.
 */
const cache = new Map<string, { text: string; slices: PreviewSlice[] }>();
export const previewStats = { cuts: 0 };
export function previewSlices(text: string): PreviewSlice[] {
    const key = textRevision(text);
    const known = cache.get(key);
    if (known && known.text === text) {
        cache.delete(key);
        cache.set(key, known);
        return known.slices;
    }
    previewStats.cuts++;
    const slices = splitPreview(text);
    cache.set(key, { text, slices });
    if (cache.size > 4)
        cache.delete(cache.keys().next().value as string);
    return slices;
}
