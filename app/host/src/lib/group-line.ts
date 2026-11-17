import { tr, currentLanguage } from './i18n';

/**
 * The host writes an activity group's line in English from its counts ("Ran 2 commands, read a file"; server/agent/deliverables.mjs groupLine) and the saved chat keeps
 * that sentence, so the page reads it back into (kind, count) and says each part in the language of the moment. A part it does not know is left as the host wrote it.
 * The English output is the same sentence as before (the strings below are the host's own words, so English is unchanged).
 */
const KINDS: { kind: string; verb: string; noun: [string, string]; one: string; many: string }[] = [
  { kind: 'command', verb: 'Ran', noun: ['command', 'commands'], one: 'Ran a command', many: 'Ran {0} commands' },
  { kind: 'read', verb: 'Read', noun: ['file', 'files'], one: 'Read a file', many: 'Read {0} files' },
  { kind: 'edit', verb: 'Edited', noun: ['file', 'files'], one: 'Edited a file', many: 'Edited {0} files' },
  { kind: 'search', verb: 'Searched', noun: ['search', 'searches'], one: 'Searched a search', many: 'Searched {0} searches' },
  { kind: 'web', verb: 'Looked up', noun: ['web page', 'web pages'], one: 'Looked up a web page', many: 'Looked up {0} web pages' },
  { kind: 'browser', verb: 'Used the browser for', noun: ['browser step', 'browser steps'], one: 'Used the browser for a browser step', many: 'Used the browser for {0} browser steps' },
  { kind: 'helper', verb: 'Started', noun: ['helper', 'helpers'], one: 'Started a helper', many: 'Started {0} helpers' },
  { kind: 'other', verb: 'Used', noun: ['tool', 'tools'], one: 'Used a tool', many: 'Used {0} tools' },
];
/** Languages whose sentence parts after the first start with a small letter, as in English ("Ran a command, read a file"). */
const SMALL_AFTER_FIRST = new Set(['en', 'es', 'fr', 'pt', 'ru']);
const JOIN: Record<string, string> = { ja: '\u3001', zh: '\uff0c' };

/** One part of the host's sentence as (phrase key, count), or null. Its first letter may be small (the parts after the first). */
export function readPart(part: string): { one: string; many: string; n: number } | null {
  const text = part.trim();
  for (const k of KINDS) {
    const m = new RegExp('^' + k.verb.replace(/^./, c => '[' + c.toUpperCase() + c.toLowerCase() + ']') + ' (a|\\d+) (' + k.noun[0] + '|' + k.noun[1] + ')$').exec(text);
    if (m) return { one: k.one, many: k.many, n: m[1] === 'a' ? 1 : Number(m[1]) };
  }
  return null;
}

export function groupSentence(line: string): string {
  if (!line) return line;
  const parts = line.split(', ');
  const read = parts.map(readPart);
  if (read.some(x => x === null)) return line;                       // a sentence this page does not know: as the host wrote it
  const language = currentLanguage();
  const out = (read as { one: string; many: string; n: number }[]).map((x, i) => {
    const said = x.n === 1 ? tr(x.one) : tr(x.many, [x.n]);
    if (i === 0) return said;
    return SMALL_AFTER_FIRST.has(language) ? said.charAt(0).toLocaleLowerCase(language) + said.slice(1) : said;
  });
  return out.join(JOIN[language] ?? ', ');
}
