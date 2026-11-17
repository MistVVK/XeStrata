/**
 * The computer's regional format, for the Monitor's cost line (#3628): which region and, where the system names it, which currency the
 * person uses. Read once, without PowerShell:
 *   Windows  the regional format (Settings > Time & language > Region > Regional format) from the registry: LocaleName ("en-US"); when it
 *            names no region, the country or region setting (Geo\Name).
 *   macOS    `defaults read -g AppleLocale` ("en_US", or "de_DE@currency=CHF" when the person set another currency).
 *   Linux    LC_ALL, LC_MONETARY or LANG ("de_DE.UTF-8"), and the locale's own currency when `locale -k int_curr_symbol` answers.
 * Nothing here is a guess: no region means no answer, and the page then falls back to its own language list.
 */
import { execFileSync } from 'node:child_process';

const NONE = Object.freeze({ locale: null, region: null, currency: null, source: null });

/** "en_US.UTF-8" / "de_DE@euro" / "zh-Hans-CN" -> { locale: "en-US", region: "US" } (null parts when the tag has none). */
export function parseLocaleTag(text) {
  const tag = String(text ?? '').trim().replace(/[.@].*$/, '').replace(/_/g, '-');
  if (!tag || /^(c|posix)$/i.test(tag)) return { locale: null, region: null };
  const parts = tag.split('-');
  const region = parts.slice(1).find(p => /^[A-Za-z]{2}$/.test(p)) ?? null;
  const language = /^[A-Za-z]{2,3}$/.test(parts[0]) ? parts[0].toLowerCase() : null;
  if (!language) return { locale: null, region: null };
  return { locale: region ? `${language}-${region.toUpperCase()}` : language, region: region ? region.toUpperCase() : null };
}

export function parseRegValue(out, name) {
  const m = new RegExp(`^\\s*${name}\\s+REG_SZ\\s+(.*?)\\s*$`, 'mi').exec(String(out ?? ''));
  return m ? m[1] : null;
}

const currencyCode = value => (/^[A-Z]{3}$/.test(String(value ?? '').trim().toUpperCase()) ? String(value).trim().toUpperCase() : null);

export function detectRegion({ platform = process.platform, env = process.env, run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8', timeout: 4000, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }) } = {}) {
  try {
    if (platform === 'win32') {
      const intl = 'HKCU\\Control Panel\\International';
      const named = parseLocaleTag(parseRegValue(run('reg', ['query', intl, '/v', 'LocaleName']), 'LocaleName'));
      if (named.region) return { ...named, currency: null, source: 'windows' };
      let geo = null;
      try { geo = parseRegValue(run('reg', ['query', `${intl}\\Geo`, '/v', 'Name']), 'Name'); } catch { /* no country setting */ }
      const region = /^[A-Za-z]{2}$/.test(geo ?? '') ? geo.toUpperCase() : null;
      return region ? { locale: named.locale ? `${named.locale}-${region}` : null, region, currency: null, source: 'windows' } : { ...NONE };
    }
    if (platform === 'darwin') {
      const raw = String(run('defaults', ['read', '-g', 'AppleLocale'])).trim();
      const named = parseLocaleTag(raw);
      const currency = currencyCode(/@currency=([A-Za-z]{3})/.exec(raw)?.[1]);
      return named.region || currency ? { ...named, currency, source: 'macos' } : { ...NONE };
    }
    const named = parseLocaleTag(env.LC_ALL || env.LC_MONETARY || env.LANG);
    let currency = null;
    try { currency = currencyCode(/int_curr_symbol="?([A-Za-z]{3})/.exec(run('locale', ['-k', 'int_curr_symbol']))?.[1]); } catch { /* no locale command */ }
    return named.region || currency ? { ...named, currency, source: 'linux' } : { ...NONE };
  } catch {
    return { ...NONE };
  }
}

let cached = null;
/** The answer is cached: the regional format does not change while the host runs. */
export function systemLocale(options) {
  if (options) return detectRegion(options);
  // A test or a demo can name the regional format (the screenshots of the Monitor); a packaged app never reads it.
  if (process.env.STRATA_TEST_LOCALE && process.env.STRATA_PACKAGED !== '1') { const named = parseLocaleTag(process.env.STRATA_TEST_LOCALE); return { ...named, currency: null, source: 'test' }; }
  cached ??= detectRegion();
  return cached;
}
