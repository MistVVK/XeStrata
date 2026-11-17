/**
 * A chat title cut from the first prompt without cutting a character in half (#4533).
 *
 * `prompt.slice(0, 47)` counts UTF-16 units, so an emoji (two units), a flag (four), a family or a skin-tone sequence (up to eleven) that
 * straddles the cut leaves half a character: a lone surrogate that draws as a replacement box, or a ZWJ with nothing after it. The title is
 * stored and restored as it is, so the broken tail comes back on every start and goes through the font fallback that WebKitGTK crashes in.
 * This cuts at a user-perceived character (grapheme) when the engine has Intl.Segmenter, and at a code point when it has not.
 *
 * The three `prompt.length > 48 ? prompt.slice(0, 47).trimEnd() + '…' : prompt` places in lib/store.ts should call `shortTitle(prompt)`.
 */
const WIDTH = 48;

export function shortTitle(text: string, width = WIDTH): string {
    if (text.length <= width)
        return text;
    const room = width - 1;                     // the ellipsis takes one unit
    let end = 0;
    const Segmenter = (Intl as unknown as { Segmenter?: new (locale?: string, options?: { granularity: string }) => { segment(input: string): Iterable<{ segment: string; index: number }> } }).Segmenter;
    if (Segmenter) {
        for (const part of new Segmenter(undefined, { granularity: 'grapheme' }).segment(text)) {
            if (part.index + part.segment.length > room)
                break;
            end = part.index + part.segment.length;
        }
    }
    else {
        for (const ch of text) {                 // iterates by code point
            if (end + ch.length > room)
                break;
            end += ch.length;
        }
    }
    return text.slice(0, end).trimEnd() + '…';
}
