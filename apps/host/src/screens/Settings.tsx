import { tr } from '../lib/i18n';
/** Settings panels share the existing preference controls. */
import { useRef, type ReactNode } from 'react';
import { Btn, Field, Icon, Select, type IconName } from '../components/ui';
import { Brand } from '../components/Brand';
import { SettingsModels } from '../components/SettingsModels';
import { SettingsTools } from '../components/SettingsTools';
import { SettingsStorage } from '../components/SettingsStorage';
import { PrivacyNotes } from '../components/PrivacyNotes';
import { AboutVersions, SettingsUpdates } from '../components/SettingsUpdates';
import { NativeOperations } from '../components/NativeOperations';
import { FontChoice } from '../components/FontChoice';
import { ElectricityPrice } from '../components/ElectricityPrice';
import { ApprovalNotifySetting, DoneSoundSetting, NumberQuotesSetting, PortSetting, TraySetting, useShellState } from '../components/ShellSettings';
import { CompactionSetting } from '../components/CompactionSetting';
import { ChatDefaultsSetting } from '../components/ChatDefaultsSetting';
import { LANGUAGES } from '../lib/i18n';
import { setLanguage, setWhileBusy } from '../lib/store';
import { UserRulesSetting } from '../components/UserRules';
import { CONTENT_FONT_RANGE, setContentFontSize, setDensity, setFocus, setMotion, setShortcuts, setTheme, setTransparency, setUiScale, UI_SCALES, useStore, setSettingsSection } from '../lib/store';
const PANELS = [{ name: 'Appearance', icon: 'sun', sections: ['General', 'Reading & focus'] }, { name: 'Models', icon: 'models', sections: [] }, { name: 'Tools', icon: 'tool', sections: [] }, { name: 'Storage', icon: 'disk', sections: ['Storage'] }, { name: 'Engine', icon: 'engine', sections: [] }, { name: 'About', icon: 'info', sections: ['About', 'Update', 'Privacy'] }] as const;
export function SettingsScreen() {
    const prefs = useStore(s => s);
    const [shell, saveShell] = useShellState();
    const active = prefs.settingsSection;
    const scrollRef = useRef<HTMLDivElement>(null);
    const fields: {
        section: string;
        label: string;
        hint: string;
        aliases?: string;
        control: ReactNode;
    }[] = [
        { section: 'General', label: 'Language', hint: 'Choose the language used throughout Strata.', control: <Select translateLabels={false} ariaLabel={tr("Language")} value={prefs.language} options={[...LANGUAGES]} onChange={value => setLanguage(value as typeof prefs.language)}/> },
        { section: 'General', label: 'While an answer is being written', hint: 'What Enter does when you send a message while the assistant is still answering.', aliases: 'interrupt wait queue interject busy send after stop enter', control: <Select value={prefs.whileBusy} ariaLabel={tr("While an answer is being written")} options={[{ value: 'wait', label: 'Wait for the answer' }, { value: 'queue', label: 'Send after the answer' }, { value: 'interject', label: 'Interrupt the answer' }]} onChange={value => setWhileBusy(value as typeof prefs.whileBusy)}/> },
        { section: 'General', label: 'Sound when work finishes', hint: 'Plays a short sound when an answer or task that took a while is finished, or when Strata is in the background.', aliases: 'audio chime done complete notification', control: <DoneSoundSetting /> },
        { section: 'General', label: 'Notify me when Strata needs an answer', hint: 'In Manual mode, shows a desktop notification when the assistant waits for your approval and Strata is in the background.', aliases: 'approval permission alert notification manual', control: <ApprovalNotifySetting /> },
        ...(shell?.tray ? [{ section: 'General', label: 'Minimise to tray', hint: 'Closing the window keeps Strata running in the tray, next to the clock. Choose Quit in the tray menu to stop it. Applies the next time Strata starts.', aliases: 'tray background close notification area', control: <TraySetting shell={shell} save={saveShell} /> }] : []),
        ...(shell?.desktop ? [{ section: 'General', label: 'App port', hint: 'The port Strata’s window and local address use. Leave it empty to let Strata choose (47831 when it is free). Applies the next time Strata starts.', aliases: 'network address 47831 localhost api', control: <PortSetting shell={shell} save={saveShell} /> }] : []),
        { section: 'General', label: 'Number quotes', hint: 'When one message quotes several passages, each gets a number and says who wrote it, so the assistant can tell them apart. Start a line with [1] or [2] to answer one of them.', aliases: 'quote quoting numbered reference passages', control: <NumberQuotesSetting /> },
        { section: 'General', label: 'Your rules', hint: 'Rules the assistant follows in every chat, for example "Answer in German" or "Keep answers short". New answers use them right away.', aliases: 'instructions system prompt agents.md custom global always', control: <UserRulesSetting /> },
        { section: 'General', label: 'Electricity price', hint: 'Price of one kilowatt-hour. The Monitor then shows what your graphics card has cost since Strata started. Leave it empty to hide the cost.', aliases: 'cost power energy kwh currency money', control: <ElectricityPrice /> },
        { section: 'General', label: 'Summarize older messages', hint: 'How full a chat has to get before the assistant condenses its beginning to make room. Later keeps more of the conversation and leaves less room for one answer; the assistant picks the new number up on its next message.', aliases: 'compact compaction context window threshold summarize tokens history memory', control: <CompactionSetting /> },
        { section: 'General', label: 'New chats', hint: 'What every new conversation starts with: how much it may do on its own, how deeply it thinks, and how long one answer may be. A conversation you are already in keeps its own choice.', aliases: 'default mode permission bypass plan manual thinking reasoning effort budget tokens limit', control: <ChatDefaultsSetting /> },
        { section: 'Reading & focus', label: 'Font', hint: 'Choose an installed font for the interface and conversations.', control: <FontChoice /> },
        { section: 'General', label: 'Keyboard shortcuts', hint: 'Set the key combinations that feel natural to you.', aliases: 'command keys hotkeys', control: <Btn size="sm" onClick={() => setShortcuts(true)}>{tr("Edit shortcuts")}</Btn> },
        { section: 'Reading & focus', label: 'Appearance', hint: 'System follows your computer by default.', aliases: 'theme light dark system automatic', control: <Select value={prefs.theme} ariaLabel={tr("Appearance")} options={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }, { value: 'system', label: 'System' }]} onChange={setTheme}/> },
        { section: 'Reading & focus', label: 'Reading size', hint: 'Conversation text size. Interface controls stay separate.', aliases: 'font text zoom', control: <label className="row"><input aria-label={tr("Reading text size")} type="range" className="st-slider" min={CONTENT_FONT_RANGE.min} max={CONTENT_FONT_RANGE.max} value={prefs.contentFontSize} onChange={e => setContentFontSize(Number(e.target.value))}/><output>{tr(prefs.contentFontSize)}{tr(" px")}</output></label> },
        { section: 'Reading & focus', label: 'Interface size', hint: 'Makes all text, buttons and menus in Strata larger or smaller.', aliases: 'zoom scale font text size bigger smaller 4k 90% 100% 110% 125% 150% 175% 200%', control: <Select value={String(prefs.uiScale)} ariaLabel={tr("Interface size")} options={UI_SCALES.map(v => ({ value: String(v), label: `${Math.round(v * 100)}%` }))} onChange={v => setUiScale(Number(v))}/> },
        { section: 'Reading & focus', label: 'Spacing', hint: 'Comfortable spacing by default.', aliases: 'density comfortable compact', control: <Select value={prefs.density} ariaLabel={tr("Interface spacing")} options={[{ value: 'comfortable', label: 'Comfortable' }, { value: 'compact', label: 'Compact' }]} onChange={setDensity}/> },
        { section: 'Reading & focus', label: 'Motion', hint: 'Reduced motion also disables smooth scrolling.', aliases: 'animation system full reduced', control: <Select value={prefs.motion} ariaLabel={tr("Motion")} options={[{ value: 'system', label: 'Follow system' }, { value: 'full', label: 'Full' }, { value: 'reduced', label: 'Reduced' }]} onChange={setMotion}/> },
        { section: 'Reading & focus', label: 'Transparency', hint: 'Translucent controls or solid surfaces. Text stays opaque.', aliases: 'blur glass liquid opaque translucent', control: <Select value={prefs.transparency} ariaLabel={tr("Transparency")} options={[{ value: 'translucent', label: 'Translucent' }, { value: 'opaque', label: 'Opaque' }]} onChange={setTransparency}/> },
        { section: 'Reading & focus', label: 'Focus reading', hint: 'Hides navigation in chat. Settings stays open above the conversation.', aliases: 'distraction quiet', control: <button className="st-toggle" role="switch" aria-checked={prefs.focus} aria-label={tr("Focus reading")} onClick={() => setFocus(!prefs.focus)}/> },
    ];
    function activate(name: typeof PANELS[number]['name']) {
        setSettingsSection(name);
        scrollRef.current?.scrollTo({ top: 0 });
    }
    return <div className="settings-layout">
    <nav className="settingsindex" role="tablist" aria-label={tr("Settings sections")} onKeyDown={e => {
            if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key))
                return;
            e.preventDefault();
            const index = PANELS.findIndex(p => p.name === active);
            const next = e.key === 'Home' ? 0 : e.key === 'End' ? PANELS.length - 1 : (index + (['ArrowDown', 'ArrowRight'].includes(e.key) ? 1 : -1) + PANELS.length) % PANELS.length;
            activate(PANELS[next].name);
            document.getElementById(`settings-tab-${PANELS[next].name}`)?.focus();
        }}>
      {tr(PANELS.map(panel => <button key={panel.name} role="tab" id={`settings-tab-${panel.name}`} aria-controls={`settings-panel-${panel.name}`} aria-selected={active === panel.name} tabIndex={active === panel.name ? 0 : -1} className="settingsindex__item" onClick={() => activate(panel.name)}><Icon name={panel.icon as IconName} size={17}/>{tr(panel.name)}</button>))}
    </nav>
    <div className="settings-pane" ref={scrollRef}>
      {tr(PANELS.map(panel => <div key={panel.name} role="tabpanel" tabIndex={0} id={`settings-panel-${panel.name}`} aria-labelledby={`settings-tab-${panel.name}`} hidden={active !== panel.name}>
        {tr(panel.name === 'Models' ? <SettingsModels /> : panel.name === 'Tools' ? (active === panel.name ? <SettingsTools /> : null) : panel.name === 'Engine' && active === panel.name ? <div className="stack stack--lg settings-body"><section className="settings-section"><h2 className="h3" style={{ marginBottom: 14 }}>{tr("Engine")}</h2><NativeOperations /></section></div> : active === panel.name ? <div className="stack stack--lg settings-body">
          {tr(panel.sections.map(section => <section className="settings-section" key={section}>
            <h2 className="h3" style={{ marginBottom: 14 }}>{tr(section)}</h2>
            <div className="st-card st-card--flush">
              {tr(section === 'About' ? <div className="settings-about-brand"><Brand variant="lockup" height={28}/><AboutVersions/></div> : null)}
              {tr(section === 'Storage' ? <SettingsStorage /> : null)}
              {tr(section === 'Privacy' ? <PrivacyNotes /> : null)}
              {tr(section === 'Update' ? <SettingsUpdates /> : null)}
              {tr(fields.filter(f => f.section === section).map(f => <Field key={f.label} label={tr(f.label)} hint={tr(f.hint)}>{tr(f.control)}</Field>))}
            </div>
          </section>))}
        </div> : null)}
      </div>))}
    </div>
  </div>;
}
