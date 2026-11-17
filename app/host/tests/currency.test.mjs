// The currency of the Monitor's cost line (#3628): found from the computer's region, formatted for the person's locale, never guessed.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { parseLocaleTag, parseRegValue, detectRegion } from '../server/region.mjs';

const compiled = await build({ stdin: { contents: "export * from './src/lib/currency';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node' });
const norm = text => text.replace(/\s/g, ' ');   // a no-break space becomes a space
const C = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

test('a region maps to its currency, an unknown one to nothing', () => {
  for (const [region, code] of [['DE', 'EUR'], ['SI', 'EUR'], ['BG', 'EUR'], ['US', 'USD'], ['GB', 'GBP'], ['JP', 'JPY'], ['CH', 'CHF'], ['BR', 'BRL'], ['KR', 'KRW'], ['CN', 'CNY'], ['IN', 'INR'], ['de', 'EUR']])
    assert.equal(C.currencyForRegion(region), code, region);
  assert.equal(C.currencyForRegion('ZZ'), null);
  assert.equal(C.currencyForRegion(null), null);
  assert.equal(C.currencyForRegion(''), null);
  for (const code of Object.values(C.REGION_CURRENCY)) assert.match(code, /^[A-Z]{3}$/);
  const known = new Set(Intl.supportedValuesOf('currency'));
  const unknown = [...new Set(Object.values(C.REGION_CURRENCY))].filter(code => !known.has(code));
  assert.ok(unknown.length <= 3, `currencies this runtime does not know: ${unknown}`);
});

test('the region comes out of a language tag, or the tag has none', () => {
  assert.equal(C.regionOfLocale('en-US'), 'US');
  assert.equal(C.regionOfLocale('de_DE.UTF-8'), 'DE');
  assert.equal(C.regionOfLocale('zh-Hans-CN'), 'CN');
  assert.equal(C.regionOfLocale('en'), null);
  assert.equal(C.regionOfLocale(''), null);
  assert.equal(C.regionOfLocale(undefined), null);
});

test('the currency: the override, then the system, then the browser, then the app language, then none', () => {
  const windows = { locale: 'en-US', region: 'US', currency: null, source: 'windows' };
  assert.deepEqual(C.resolveCurrency({ system: windows, language: 'en' }), { currency: 'USD', locale: 'en-US', source: 'system' });
  assert.equal(C.resolveCurrency({ system: { locale: 'de-CH', region: 'CH', currency: 'EUR', source: 'linux' }, language: 'en' }).currency, 'EUR', 'a currency the system names wins over its region');
  assert.equal(C.resolveCurrency({ system: windows, language: 'en', override: 'chf' }).currency, 'CHF', 'the picker overrides');
  assert.equal(C.resolveCurrency({ system: windows, language: 'en', override: 'x' }).currency, 'USD', 'a bad override is ignored');
  assert.deepEqual(C.resolveCurrency({ system: null, browserLanguages: ['en', 'sl-SI'], language: 'en' }), { currency: 'EUR', locale: 'sl-SI', source: 'browser' });
  assert.deepEqual(C.resolveCurrency({ system: null, browserLanguages: [], language: 'ja' }), { currency: 'JPY', locale: 'ja', source: 'language' });
  assert.equal(C.resolveCurrency({ system: null, browserLanguages: [], language: 'zh' }).currency, 'CNY');
  assert.equal(C.resolveCurrency({ system: null, browserLanguages: [], language: 'pt' }).currency, 'BRL');
  assert.deepEqual(C.resolveCurrency({ system: null, browserLanguages: ['en'], language: 'en' }), { currency: null, locale: 'en', source: 'none' }, 'English names no region: no currency, never EUR');
  assert.equal(C.resolveCurrency({ system: { locale: null, region: null, currency: null, source: null }, language: 'en' }).currency, null);
});

test('money is written for the locale with the usual digits of the currency, and a tiny amount is "less than" one unit', () => {
  const us = { currency: 'USD', locale: 'en-US' }, de = { currency: 'EUR', locale: 'de-DE' }, jp = { currency: 'JPY', locale: 'ja-JP' };
  assert.equal(C.formatMoney(1.2345, us), '$1.23');
  assert.equal(C.formatMoney(0.123456, us), '$0.12');
  assert.equal(C.formatMoney(0.01, us), '$0.01');
  assert.equal(C.formatMoney(12.5, us), '$12.50');
  assert.equal(C.formatMoney(0, us), '$0.00');
  assert.equal(C.formatMoney(0.0000339, us), 'less than $0.01', 'not 0.00 and not a run of decimals');
  assert.equal(C.formatMoney(0.0099, us), 'less than $0.01');
  assert.equal(norm(C.formatMoney(1.234, de)), norm('1,23 €'));
  assert.equal(norm(C.formatMoney(0.004, de)), norm('less than 0,01 €'));
  assert.match(C.formatMoney(3.2, jp), /^[¥￥]3$/);
  assert.match(C.formatMoney(0.4, jp), /^less than [¥￥]1$/, 'a currency with no minor unit: below one yen');
  assert.equal(norm(C.formatMoney(0.001, de)), norm('less than 0,01 €'));
});

test('the sentence for a tiny amount comes from the caller, so it can be translated', () => {
  assert.equal(norm(C.formatMoney(0.001, { currency: 'EUR', locale: 'fr-FR', lessThan: shown => `moins de ${shown}` })), norm('moins de 0,01 €'));
  assert.equal(C.minorDigits('USD', 'en-US'), 2);
  assert.equal(C.minorDigits('JPY', 'en-US'), 0);
  assert.equal(C.minorDigits(null, 'en-US'), 2);
});

test('with no currency the number is shown alone, and a broken code or locale never throws', () => {
  assert.equal(C.formatMoney(1.2345, { currency: null, locale: 'en-US' }), '1.23');
  assert.equal(C.formatMoney(0.0123456, { currency: null, locale: 'en-US' }), '0.01');
  assert.equal(C.formatMoney(0.001, { currency: null, locale: 'en-US' }), 'less than 0.01');
  assert.equal(C.formatMoney(0, { currency: null, locale: 'en-US' }), '0.00');
  assert.equal(C.formatMoney(1.5, { currency: 'XXXX', locale: 'en-US' }), '1.50');
  assert.equal(C.formatMoney(1.5, { currency: 'USD', locale: 'not a locale!' }), '1.50');
  assert.equal(C.formatMoney(NaN, { currency: 'USD', locale: 'en-US' }), '—');
});

test('the Settings line names the currency with its symbol, and the picker lists every currency', () => {
  assert.equal(C.currencyLabel('USD', 'en-US'), 'USD ($)');
  assert.equal(C.currencyLabel('EUR', 'de-DE'), 'EUR (€)');
  assert.equal(C.currencyLabel('ZZZ', 'en-US'), 'ZZZ');
  const choices = C.currencyChoices('en-US');
  assert.ok(choices.length > 100);
  assert.ok(choices.some(c => c.value === 'JPY' && /Japanese Yen/.test(c.label)));
});

test('the host reads the regional format: Windows, macOS and Linux, and says nothing when there is none', () => {
  assert.deepEqual(parseLocaleTag('en_US.UTF-8'), { locale: 'en-US', region: 'US' });
  assert.deepEqual(parseLocaleTag('de_DE@euro'), { locale: 'de-DE', region: 'DE' });
  assert.deepEqual(parseLocaleTag('zh-Hans-CN'), { locale: 'zh-CN', region: 'CN' });
  assert.deepEqual(parseLocaleTag('fr'), { locale: 'fr', region: null });
  assert.deepEqual(parseLocaleTag('C'), { locale: null, region: null });
  assert.deepEqual(parseLocaleTag(undefined), { locale: null, region: null });
  assert.equal(parseRegValue('    LocaleName    REG_SZ    en-US\r\n', 'LocaleName'), 'en-US');
  const win = values => (cmd, args) => { const key = args.join(' '); for (const [needle, out] of Object.entries(values)) if (key.includes(needle)) return out; throw new Error('not found'); };
  assert.deepEqual(detectRegion({ platform: 'win32', run: win({ LocaleName: '    LocaleName    REG_SZ    de-DE\r\n' }) }), { locale: 'de-DE', region: 'DE', currency: null, source: 'windows' });
  assert.deepEqual(detectRegion({ platform: 'win32', run: win({ 'Geo /v Name': '    Name    REG_SZ    SI\r\n', LocaleName: '    LocaleName    REG_SZ    en\r\n' }) }), { locale: 'en-SI', region: 'SI', currency: null, source: 'windows' }, 'a regional format with no region uses the country setting');
  assert.deepEqual(detectRegion({ platform: 'win32', run: () => { throw new Error('no registry'); } }), { locale: null, region: null, currency: null, source: null });
  assert.deepEqual(detectRegion({ platform: 'darwin', run: () => 'de_CH@currency=EUR\n' }), { locale: 'de-CH', region: 'CH', currency: 'EUR', source: 'macos' });
  assert.deepEqual(detectRegion({ platform: 'darwin', run: () => 'en_GB\n' }), { locale: 'en-GB', region: 'GB', currency: null, source: 'macos' });
  assert.deepEqual(detectRegion({ platform: 'linux', env: { LANG: 'sl_SI.UTF-8' }, run: () => 'int_curr_symbol="EUR "\n' }), { locale: 'sl-SI', region: 'SI', currency: 'EUR', source: 'linux' });
  assert.deepEqual(detectRegion({ platform: 'linux', env: { LC_ALL: 'C' }, run: () => { throw new Error('no locale command'); } }), { locale: null, region: null, currency: null, source: null });
  assert.equal(detectRegion({ platform: 'linux', env: { LC_MONETARY: 'ja_JP.UTF-8', LANG: 'en_US.UTF-8' }, run: () => { throw new Error('x'); } }).region, 'JP', 'the monetary locale comes first');
});
