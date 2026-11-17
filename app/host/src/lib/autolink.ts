/**
 * #6000: an answer that writes a web address as plain text ("see https://example.com/docs") showed it as text: only [text](url) was a link.
 * The plain text of an answer (never a code span or block, which the renderer keeps apart) is split into words and http(s) addresses.
 * An address ends at a space, at CJK punctuation (，。、！？：；「」（） and the other full-width forms) and at < > " ' `; a full stop, comma,
 * colon, semicolon, ! or ? at its end is the sentence's, and so is a closing bracket that has no opening one inside the address. Other
 * schemes (javascript:, file:, data:) are never made links.
 */
export interface TextOrLink { text: string; href?: string }

const ADDRESS = /https?:\/\/[^\s<>"'`　-〿＀-￯]+/gi;
const PAIRS: Record<string, string> = { ')': '(', ']': '[', '}': '{' };

function trimEnd(url: string): string {
  let end = url.length;
  for (;;) {
    const last = url[end - 1];
    if (/[.,:;!?*_~]/.test(last)) { end--; continue; }
    const open = PAIRS[last];
    if (open) {
      const body = url.slice(0, end);
      const opens = body.split(open).length - 1, closes = body.split(last).length - 1;
      if (closes > opens) { end--; continue; }
    }
    return url.slice(0, end);
  }
}

export function splitLinks(text: string): TextOrLink[] {
  const out: TextOrLink[] = [];
  let at = 0;
  for (const match of text.matchAll(ADDRESS)) {
    const start = match.index ?? 0;
    const url = trimEnd(match[0]);
    if (!/^https?:\/\/[^/?#]+/i.test(url) || /^https?:\/\/[.]/i.test(url)) continue;   // a scheme with no host is not an address
    if (start > at) out.push({ text: text.slice(at, start) });
    out.push({ text: url, href: url });
    at = start + url.length;
  }
  if (at < text.length) out.push({ text: text.slice(at) });
  return out.length ? out : [{ text }];
}
