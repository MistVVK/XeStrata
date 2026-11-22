import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Btn, Select } from './ui';

const presets = [4, 8, 12, 16, 24, 32];

export function RamReserveSetting() {
    const [saved, setSaved] = useState<number | null>();
    const [choice, setChoice] = useState('auto');
    const [custom, setCustom] = useState('');
    const [pending, setPending] = useState(false);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState(false);
    useEffect(() => {
        let active = true;
        api.get<{ gib: number | null }>('/api/native/ram-reserve').then(({ gib }) => {
            if (!active) return;
            setSaved(gib);
            setChoice(gib === null ? 'auto' : presets.includes(gib) ? String(gib) : 'custom');
            if (gib !== null) setCustom(String(gib));
        }).catch(e => { if (active) setError(e.message); });
        return () => { active = false; };
    }, []);

    const gib = choice === 'auto' ? null : Number(choice === 'custom' ? custom : choice);
    const valid = gib === null || (Number.isSafeInteger(gib) && gib > 0);
    async function save() {
        if (!valid || saved === undefined || pending) return;
        setError('');
        setPending(true);
        try {
            const value = await api.post<{ gib: number | null }>('/api/native/ram-reserve', { gib });
            setSaved(value.gib);
            setNotice(true);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setPending(false);
        }
    }

    return <div className="stack stack--sm">
        {saved !== undefined && <form className="stack stack--sm" onSubmit={e => { e.preventDefault(); void save(); }}>
            <Select ariaLabel={tr("RAM to keep free (GiB) (experimental)")} value={choice} options={[
                    { value: 'auto', label: tr("Automatic") },
                    ...presets.map(n => ({ value: String(n), label: String(n) + ' GiB' })),
                    { value: 'custom', label: tr("Custom") },
            ]} onChange={value => { setChoice(value); setNotice(false); }}/>
            {choice === 'custom' && <label className="input-field">
                <span>{tr("Custom RAM reserve (GiB)")}</span>
                <input className="st-input" type="number" min={1} step={1} required value={custom}
                    onChange={e => { setCustom(e.target.value); setNotice(false); }}/>
            </label>}
            <div className="row">
                <Btn type="submit" size="sm" disabled={pending || !valid || gib === saved}>{tr("Save")}</Btn>
                {notice && <span className="small muted" role="status">{tr("Saved")}</span>}
            </div>
        </form>}
        {error && <p className="small" role="alert">{tr(error)}</p>}
    </div>;
}
