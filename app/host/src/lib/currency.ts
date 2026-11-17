/**
 * The currency of the Monitor's cost line (#3628): the person's own, found from the computer's region, never typed and never a guess.
 *
 * Order: the system (the host reads the regional format of Windows, macOS or Linux: its region, or its currency when the system names one),
 * then the browser's language list, then the region of the app's language, and then no currency at all: the number is shown without a
 * symbol rather than in a currency the person does not use. A currency picked in Settings overrides all of this.
 */

/** ISO 3166 region -> ISO 4217 currency. */
const PAIRS = `AD:EUR AE:AED AF:AFN AG:XCD AI:XCD AL:ALL AM:AMD AO:AOA AR:ARS AS:USD AT:EUR AU:AUD AW:AWG AX:EUR AZ:AZN
BA:BAM BB:BBD BD:BDT BE:EUR BF:XOF BG:EUR BH:BHD BI:BIF BJ:XOF BL:EUR BM:BMD BN:BND BO:BOB BQ:USD BR:BRL BS:BSD BT:BTN BV:NOK BW:BWP BY:BYN BZ:BZD
CA:CAD CC:AUD CD:CDF CF:XAF CG:XAF CH:CHF CI:XOF CK:NZD CL:CLP CM:XAF CN:CNY CO:COP CR:CRC CU:CUP CV:CVE CW:XCG CX:AUD CY:EUR CZ:CZK
DE:EUR DJ:DJF DK:DKK DM:XCD DO:DOP DZ:DZD EC:USD EE:EUR EG:EGP EH:MAD ER:ERN ES:EUR ET:ETB FI:EUR FJ:FJD FK:FKP FM:USD FO:DKK FR:EUR
GA:XAF GB:GBP GD:XCD GE:GEL GF:EUR GG:GBP GH:GHS GI:GIP GL:DKK GM:GMD GN:GNF GP:EUR GQ:XAF GR:EUR GS:GBP GT:GTQ GU:USD GW:XOF GY:GYD
HK:HKD HM:AUD HN:HNL HR:EUR HT:HTG HU:HUF ID:IDR IE:EUR IL:ILS IM:GBP IN:INR IO:USD IQ:IQD IR:IRR IS:ISK IT:EUR
JE:GBP JM:JMD JO:JOD JP:JPY KE:KES KG:KGS KH:KHR KI:AUD KM:KMF KN:XCD KP:KPW KR:KRW KW:KWD KY:KYD KZ:KZT
LA:LAK LB:LBP LC:XCD LI:CHF LK:LKR LR:LRD LS:LSL LT:EUR LU:EUR LV:EUR LY:LYD
MA:MAD MC:EUR MD:MDL ME:EUR MF:EUR MG:MGA MH:USD MK:MKD ML:XOF MM:MMK MN:MNT MO:MOP MP:USD MQ:EUR MR:MRU MS:XCD MT:EUR MU:MUR MV:MVR MW:MWK MX:MXN MY:MYR MZ:MZN
NA:NAD NC:XPF NE:XOF NF:AUD NG:NGN NI:NIO NL:EUR NO:NOK NP:NPR NR:AUD NU:NZD NZ:NZD OM:OMR
PA:PAB PE:PEN PF:XPF PG:PGK PH:PHP PK:PKR PL:PLN PM:EUR PR:USD PS:ILS PT:EUR PW:USD PY:PYG QA:QAR RE:EUR RO:RON RS:RSD RU:RUB RW:RWF
SA:SAR SB:SBD SC:SCR SD:SDG SE:SEK SG:SGD SH:SHP SI:EUR SJ:NOK SK:EUR SL:SLE SM:EUR SN:XOF SO:SOS SR:SRD SS:SSP ST:STN SV:USD SX:XCG SY:SYP SZ:SZL
TC:USD TD:XAF TF:EUR TG:XOF TH:THB TJ:TJS TK:NZD TL:USD TM:TMT TN:TND TO:TOP TR:TRY TT:TTD TV:AUD TW:TWD TZ:TZS
UA:UAH UG:UGX UM:USD US:USD UY:UYU UZ:UZS VA:EUR VC:XCD VE:VES VG:USD VI:USD VN:VND VU:VUV WF:XPF WS:WST XK:EUR YE:YER YT:EUR ZA:ZAR ZM:ZMW ZW:ZWG`;

export const REGION_CURRENCY: Readonly<Record<string, string>> = Object.freeze(Object.fromEntries(PAIRS.split(/\s+/).map(pair => pair.split(':') as [string, string])));

/** The region of each app language that has one region of its own; English and Spanish-speaking countries are many, so only the first is a guess. */
export const LANGUAGE_REGION: Readonly<Record<string, string>> = Object.freeze({ zh: 'CN', ja: 'JP', de: 'DE', es: 'ES', ru: 'RU', pt: 'BR', ko: 'KR' });

export const currencyForRegion = (region: string | null | undefined): string | null => (region ? REGION_CURRENCY[region.toUpperCase()] ?? null : null);

/** The region in a language tag ("en-US", "de_DE.UTF-8", "zh-Hans-CN"), or null when the tag names none. */
export function regionOfLocale(tag: string | null | undefined): string | null {
    if (!tag) return null;
    const clean = String(tag).replace(/[.@].*$/, '').replace(/_/g, '-');
    try {
        const region = new Intl.Locale(clean).region;
        return region ?? null;
    } catch {
        const found = /-([A-Za-z]{2})(?:-|$)/.exec(clean);
        return found ? found[1].toUpperCase() : null;
    }
}

/** What the host read from the system (GET /api/locale). */
export interface SystemLocale { locale: string | null; region: string | null; currency: string | null; source: string | null }

export interface ResolvedCurrency { currency: string | null; locale: string; source: 'system' | 'browser' | 'language' | 'none' }

/**
 * The currency to show money in, and the locale to write the number in.
 * `override` is the currency the person chose in Settings (empty: automatic).
 */
export function resolveCurrency({ system, browserLanguages = [], language, override = '' }: { system?: SystemLocale | null; browserLanguages?: readonly string[]; language: string; override?: string }): ResolvedCurrency {
    const appTag = language === 'zh' ? 'zh-CN' : language;
    const locale = system?.locale ?? browserLanguages.find(tag => regionOfLocale(tag)) ?? browserLanguages[0] ?? appTag;
    if (/^[A-Za-z]{3}$/.test(override)) return { currency: override.toUpperCase(), locale, source: 'system' };
    const fromSystem = system?.currency ?? currencyForRegion(system?.region);
    if (fromSystem) return { currency: fromSystem, locale, source: 'system' };
    for (const tag of browserLanguages) {
        const found = currencyForRegion(regionOfLocale(tag));
        if (found) return { currency: found, locale, source: 'browser' };
    }
    const fromLanguage = currencyForRegion(LANGUAGE_REGION[language]);
    if (fromLanguage) return { currency: fromLanguage, locale, source: 'language' };
    return { currency: null, locale, source: 'none' };
}

/** How many decimals this currency normally has (2 for dollars and euros, 0 for yen); 2 when there is no currency. */
export function minorDigits(currency: string | null, locale: string): number {
    if (!currency) return 2;
    try { return new Intl.NumberFormat(locale, { style: 'currency', currency }).resolvedOptions().maximumFractionDigits ?? 2; } catch { return 2; }
}

/**
 * An amount of money, written for the person's locale with the currency's usual digits ("$0.12", "$1.23", "1,23 €", "¥3").
 * An amount below the smallest normal unit (one cent; one yen) is not shown as 0.00 or as a long run of decimals but as
 * "less than $0.01" (`lessThan` writes the sentence in the person's language). With no currency the number is shown alone,
 * and never in a currency that was only guessed.
 */
export function formatMoney(amount: number, { currency, locale, lessThan = shown => `less than ${shown}` }: { currency: string | null; locale: string; lessThan?: (shown: string) => string }): string {
    if (!Number.isFinite(amount) || amount < 0) return '—';
    const digits = minorDigits(currency, locale), unit = 10 ** -digits;
    const write = (value: number): string => {
        try {
            if (currency) return new Intl.NumberFormat(locale, { style: 'currency', currency }).format(value);
        } catch { /* an unknown currency code or locale: the number alone */ }
        try {
            return new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
        } catch {
            return value.toFixed(digits);
        }
    };
    return amount > 0 && amount < unit ? lessThan(write(unit)) : write(amount);
}

/** "USD ($)": the code and its symbol in the person's locale, for the Settings line. */
export function currencyLabel(currency: string, locale: string): string {
    try {
        const symbol = new Intl.NumberFormat(locale, { style: 'currency', currency, currencyDisplay: 'narrowSymbol' }).formatToParts(0).find(part => part.type === 'currency')?.value;
        return symbol && symbol !== currency ? `${currency} (${symbol})` : currency;
    } catch {
        return currency;
    }
}

/** Every currency this web view knows, by name, for the override list. */
export function currencyChoices(locale: string): { value: string; label: string }[] {
    let codes: string[] = [];
    try { codes = (Intl as unknown as { supportedValuesOf?: (key: string) => string[] }).supportedValuesOf?.('currency') ?? []; } catch { /* an older web view */ }
    let names: Intl.DisplayNames | null = null;
    try { names = new Intl.DisplayNames([locale], { type: 'currency' }); } catch { /* no names */ }
    return codes.map(code => ({ value: code, label: `${code} – ${names?.of(code) ?? code}` }));
}
