import { LANGUAGES, type Language } from './i18n';

/**
 * The first start (nothing saved yet) opens in the language of the computer (#3723): the first of the browser's preferred languages that
 * Strata has. "pt-BR" is Portuguese, "zh-TW" is Chinese (the app has one Chinese, Simplified). Null when none of them is offered.
 */
export function detectLanguage(tags: readonly string[] | null | undefined): Language | null {
  for (const raw of tags ?? []) {
    const base = String(raw).toLowerCase().split(/[-_]/)[0];
    const found = LANGUAGES.find(l => l.value === base);
    if (found) return found.value;
  }
  return null;
}
