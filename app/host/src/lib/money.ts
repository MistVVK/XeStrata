import { useEffect, useState } from 'react';
import { tr } from './i18n';
import { api } from './client';
import { formatMoney, resolveCurrency, type ResolvedCurrency, type SystemLocale } from './currency';
import { useStore } from './store';

let asked: Promise<SystemLocale | null> | null = null;

/** The computer's regional format, read once from the host (null in a browser build that has none, or while it loads). */
export function useSystemLocale(): SystemLocale | null {
    const [value, setValue] = useState<SystemLocale | null>(null);
    useEffect(() => {
        let alive = true;
        asked ??= api.get<SystemLocale>('/api/locale').catch(() => null);
        void asked.then(v => { if (alive) setValue(v); });
        return () => { alive = false; };
    }, []);
    return value;
}

/** The person's currency and locale for money (#3628) and a function that writes an amount in them. */
export function useMoney(): ResolvedCurrency & { format: (amount: number) => string; detected: string | null } {
    const system = useSystemLocale(), language = useStore(s => s.language), override = useStore(s => s.electricityCurrency);
    const browserLanguages = typeof navigator === 'undefined' ? [] : [...(navigator.languages ?? [navigator.language])].filter(Boolean);
    const resolved = resolveCurrency({ system, browserLanguages, language, override });
    const detected = override ? resolveCurrency({ system, browserLanguages, language }).currency : resolved.currency;
    return { ...resolved, detected, format: amount => formatMoney(amount, { ...resolved, lessThan: shown => tr('less than {0}', [shown]) }) };
}
