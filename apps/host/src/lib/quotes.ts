/**
 * Quoting a turn or a passage into the next message (#4785). The message that goes out carries the quote in one fixed, visible form, so the
 * model knows what is being answered and the transcript can show the quote apart from the reply:
 *
 *   Quoting:
 *   「the first passage」
 *   「the second passage」
 *   → Reply: what the person wrote
 *
 * With "Number quotes" on (#5993, off by default), a message with two or more quotes numbers them and names who wrote each, so the model can tell
 * them apart and the person can answer one ("[2] yes"):
 *
 *   Quoting:
 *   [1] assistant: 「the first passage」
 *   [2] user: 「the second passage」
 *   → Reply: what the person wrote
 *
 * One quote keeps the form above in both modes. The sources of the numbers ({n, messageId, role}) are kept with the message as data.
 */
export const MAX_QUOTES = 8;
export const MAX_QUOTE_CHARS = 4000;

export interface Quote { id: string; text: string; messageId: string; role: 'user' | 'assistant' }

/** The quote as it is carried: trimmed, bounded, and with the corner brackets of the form turned into other brackets so a quote cannot end itself early. */
export function cleanQuote(text: string): string {
  const flat = String(text ?? '').replace(/\r\n?/g, '\n').replace(/\n{3,}/g, '\n\n').trim().replace(/「/g, '『').replace(/」/g, '』');
  return flat.length > MAX_QUOTE_CHARS ? `${flat.slice(0, MAX_QUOTE_CHARS - 1).trimEnd()}…` : flat;
}

export function composeQuoted(quotes: readonly string[], reply: string): string {
  const list = quotes.map(cleanQuote).filter(Boolean).slice(0, MAX_QUOTES);
  if (!list.length) return reply;
  return `Quoting:\n${list.map(q => `「${q}」`).join('\n')}\n→ Reply: ${reply}`;
}

/** Where a numbered quote came from: kept with the sent message so the transcript can label it and jump to it. */
export interface QuoteSource { n: number; messageId: string; role: 'user' | 'assistant' }

/**
 * The message that goes out with the switch on: two or more quotes are numbered and say who wrote them; one quote (or none) keeps composeQuoted's form.
 * `sources` is what the transcript keeps (numbers follow the quotes that survive cleaning), empty when the plain form was used.
 */
export function composeNumbered(quotes: readonly Quote[], reply: string): { text: string; sources: QuoteSource[] } {
  const list = quotes.map(q => ({ q, text: cleanQuote(q.text) })).filter(x => x.text).slice(0, MAX_QUOTES);
  if (list.length < 2) return { text: composeQuoted(quotes.map(q => q.text), reply), sources: [] };
  const sources = list.map((x, i) => ({ n: i + 1, messageId: x.q.messageId, role: x.q.role }));
  const lines = list.map((x, i) => `[${i + 1}] ${x.q.role}: 「${x.text}」`);
  return { text: `Quoting:\n${lines.join('\n')}\n→ Reply: ${reply}`, sources };
}

/**
 * The quotes and the reply of a message made by composeQuoted or composeNumbered, or null when the message is not one. `numbered` is set for the
 * numbered form: the number and the writer of each quote, in order.
 */
export function parseQuoted(content: string): { quotes: string[]; reply: string; numbered?: { n: number; role: 'user' | 'assistant' }[] } | null {
  const text = String(content ?? '');
  const plain = text.startsWith('Quoting:\n「'), numbered = text.startsWith('Quoting:\n[1] ');
  if (!plain && !numbered) return null;
  const marker = text.indexOf('」\n→ Reply: ');
  if (marker < 0) return null;
  const head = text.slice('Quoting:\n'.length, marker + 1);
  const reply = text.slice(marker + '」\n→ Reply: '.length);
  if (plain) {
    const quotes = [...head.matchAll(/「([^」]*)」/g)].map(m => m[1]);
    return quotes.length ? { quotes, reply } : null;
  }
  const found = [...head.matchAll(/(?:^|\n)\[(\d+)\] (user|assistant): 「([^」]*)」/g)];
  if (!found.length || found.some((m, i) => Number(m[1]) !== i + 1)) return null;
  return { quotes: found.map(m => m[3]), reply, numbered: found.map(m => ({ n: Number(m[1]), role: m[2] as 'user' | 'assistant' })) };
}

/** Where the numbered quotes of the messages being sent came from, until the store has made the message (the text is the key, as sent). */
const pendingSources = new Map<string, QuoteSource[]>();
export function rememberQuoteSources(text: string, sources: readonly QuoteSource[]) { if (sources.length) pendingSources.set(text.trim(), [...sources]); }
export function takeQuoteSources(text: string): QuoteSource[] | undefined { const key = String(text ?? '').trim(), found = pendingSources.get(key); if (found) pendingSources.delete(key); return found; }

/** Adds a quote to a list: an empty one, a repeat and anything past the limit are left out. */
export function addToQuotes(list: readonly Quote[], quote: Omit<Quote, 'id'> & { id?: string }): Quote[] {
  const text = cleanQuote(quote.text);
  if (!text || list.length >= MAX_QUOTES || list.some(q => q.text === text && q.messageId === quote.messageId)) return [...list];
  return [...list, { id: quote.id ?? `q_${Math.random().toString(36).slice(2, 9)}`, text, messageId: quote.messageId, role: quote.role }];
}
