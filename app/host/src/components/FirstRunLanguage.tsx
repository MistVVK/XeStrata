/** The language choice on the first screen, so a person who cannot read English can change it before anything is downloaded (#3723). */
import { tr } from '../lib/i18n';
import { LANGUAGES, type Language } from '../lib/i18n';
import { setLanguage, useStore } from '../lib/store';
import { Select } from './ui';

export function FirstRunLanguage() {
    const language = useStore(s => s.language);
    return <div className="onboard-language"><Select translateLabels={false} ariaLabel={tr("Language")} value={language} options={[...LANGUAGES]} onChange={value => setLanguage(value as Language)}/></div>;
}
