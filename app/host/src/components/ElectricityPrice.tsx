import { useEffect, useMemo, useState } from 'react';
import { currencyChoices, currencyLabel } from '../lib/currency';
import { tr } from '../lib/i18n';
import { useImeKeys } from '../lib/ime-keys';
import { useMoney } from '../lib/money';
import { setElectricity, useStore } from '../lib/store';
import { Btn, Select } from './ui';

/**
 * The price of one kWh for the Monitor's cost line (#3628). The person types only the number: the currency is the one of the computer's
 * region, shown beside it. "Change currency" is a way to pick another one, never a step that has to be taken.
 */
export function ElectricityPrice() {
    const price = useStore(s => s.electricityPrice), override = useStore(s => s.electricityCurrency);
    const money = useMoney(), ime = useImeKeys();
    const [text, setText] = useState(price > 0 ? String(price) : ''), [picking, setPicking] = useState(false);
    useEffect(() => { setText(price > 0 ? String(price) : ''); }, [price]);
    const commit = () => setElectricity(Number(text.replace(',', '.')), override);
    const choices = useMemo(() => currencyChoices(money.locale), [money.locale]);
    return <span className="stack">
        <span className="row">
            <input className="st-input" inputMode="decimal" size={7} aria-label={tr("Price per kWh")} placeholder="0.30" value={text} onChange={e => setText(e.target.value)} onBlur={commit} {...ime.events} onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter') commit(); }}/>
            <span className="small muted">{money.currency ? tr("per kWh, in {0}", [currencyLabel(money.currency, money.locale)]) : tr("per kWh")}</span>
            <Btn size="sm" onClick={() => setPicking(v => !v)}>{tr("Change currency")}</Btn>
        </span>
        {picking ? <Select ariaLabel={tr("Currency")} translateLabels={false} value={override}
            options={[{ value: '', label: tr("Automatic ({0})", [money.detected ?? '—']) }, ...choices]}
            onChange={value => { setElectricity(price, String(value)); setPicking(false); }}/> : null}
    </span>;
}
