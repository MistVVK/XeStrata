import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { Icon } from './ui';
import { FloatingMenu } from './FloatingMenu';
import { cx } from '../lib/format';
import { versionNote, type CatalogVersion } from '../lib/native-catalog';
import { nvfp4Version } from '../lib/native-names';
import { fitOn, recommendedVersions, versionFit, versionGroups, type CacheChoice, type FitHardware, type PcPick } from '../lib/native-fit';

/** The name a version goes by in the list: the quantization, with the maker only where the quantization alone would be ambiguous. */
export function versionName(v: CatalogVersion): string {
  switch (v.sourceFamily) {
    case 'swift': return `Swift ${v.model}`;
    case 'unsloth': return `Unsloth ${v.model}`;
    case 'nvfp4': return tr(nvfp4Version(v.model)?.name ?? v.model);
    case 'coder': return `Coder ${v.model}`;
    default: return v.model;
  }
}

/** Who made a version (Details only): the family name of the quantization, a proper name in every language. */
const MAKERS: Record<string, string> = { qwen: 'GSQ-RCO', coder: 'GSQ-RCO', swift: 'Swift', unsloth: 'Unsloth', gyro: 'Gyro', nvfp4: 'NVFP4' };

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;
const whole = (n: number) => `${Math.round(n)} GB`;

/**
 * Where the versions of the chosen base model are listed, each exactly once. At the top: the versions we recommend (the pick for this PC first; no "Recommended" tag since 11-02, the order says it) and the versions already downloaded (tagged "Downloaded"). Below, folded under "More
 * versions": all the others. A row is the name, small tags, a word only where one is needed ("may run slower" for a version that does
 * not fit in memory; a version that fits says nothing: see `versionFit`) and the size. The maker and the
 * memory it needs are behind "Show details" (open from the start in Expert mode). The other base models are not in this list: they
 * appear when picked at the top of the page.
 */
export function VersionPicker({ versions, selectedId, hardware, onSelect, onOwnGguf, downloaded, showRecommended = true, moreLabel, pick, cache, details: detailsOpen = false }: {
  /** The versions of the chosen base model. */
  versions: CatalogVersion[];
  selectedId: string;
  hardware: FitHardware | null | undefined;
  onSelect: (id: string) => void;
  onOwnGguf: () => void;
  /** The versions already on this computer (ready to load): each gets a small "Downloaded" tag. */
  downloaded?: ReadonlySet<string>;
  /** False where the recommended ones are shown elsewhere (the first-run page): the menu then holds only the others. */
  showRecommended?: boolean;
  /** What the button says while the selected version is one of the recommended ones shown elsewhere. */
  moreLabel?: string;
  /** The engine's own pick for this PC (catalog.recommendation). */
  pick?: PcPick | null;
  /** The conversation context and cache format the person chose: the fit words follow them, as the This computer card does. */
  cache?: CacheChoice | null;
  /** Start with the technical detail shown (Expert mode). */
  details?: boolean;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState(detailsOpen);
  useEffect(() => setDetails(detailsOpen), [detailsOpen]);
  const selected = versions.find(v => v.id === selectedId);
  // Where the recommended ones are shown outside the menu, the menu holds only the others.
  const rows = recommendedVersions(versions, pick);
  const apart = new Set(showRecommended ? [] : rows.map(x => x.v.id));
  const recommendedIds = rows.map(x => x.v.id);
  const { top, more } = versionGroups(versions, apart, downloaded, recommendedIds);
  const shown = selected && !apart.has(selected.id) ? selected : undefined;
  // The folded part opens by itself when it holds the chosen version or when there is nothing above it.
  const [moreOpen, setMoreOpen] = useState(false);
  const foldedOpen = moreOpen || !top.length || more.some(v => v.id === selectedId);

  const need = (v: CatalogVersion, technical: boolean) => {
    const size = typeof v.fit_gb === 'number' && Number.isFinite(v.fit_gb) ? v.fit_gb : fitOn(v, hardware, true, cache).memoryGb;       // the files without the lookup table: the figure the label uses
    if (technical) return tr("{0} RAM + VRAM", [gb(size)]);
    return tr("Needs {0} of memory", [whole(size)]);
  };
  const row = (v: CatalogVersion) => {
    const on = v.id === selectedId, have = !!downloaded?.has(v.id);
    const state = versionFit(v, hardware, cache);
    const technical = [MAKERS[v.sourceFamily], need(v, true)].filter(Boolean).join(' · ');
    const note = versionNote(v.model);
    return <button key={v.id} role="menuitemradio" aria-checked={on} className={cx('version-row', on && 'is-on')} onClick={() => { onSelect(v.id); setOpen(false); }}>
      <span className="version-row__check">{tr(on ? <Icon name="check" size={14}/> : null)}</span>
      <span className="version-row__body">
        <span className="version-row__title"><strong>{versionName(v)}</strong>{tr(have ? <span className="version-row__tags"><span className="version-tag version-tag--have">{tr("Downloaded")}</span></span> : null)}</span>
        {tr(state === 'ssd' ? <span className="version-row__fit" data-fit={state}>{tr('may run slower')}</span> : null)}
        {tr(details ? <span className="version-row__tech">{technical}{tr(note ? <><br/>{note}</> : null)}</span> : null)}
      </span>
      <span className="version-row__size">{have ? gb(v.download_gb) : tr("{0} download", [gb(v.download_gb)])}</span>
    </button>;
  };
  // The open menu opens on the chosen version, not at the top of a long list.
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) menu.current?.querySelector('.is-on')?.scrollIntoView({ block: 'nearest' }); }, [open]);
  return <>
    <button ref={trigger} type="button" className="version-trigger" aria-haspopup="menu" aria-expanded={open} aria-label={tr("Model version")} onClick={() => setOpen(v => !v)}>
      <strong>{shown ? versionName(shown) : tr(moreLabel ?? "Choose a version")}</strong>{tr(shown && downloaded?.has(shown.id) ? <span className="version-tag version-tag--have">{tr("Downloaded")}</span> : null)}
      <span className="version-trigger__need muted">{tr(shown ? need(shown, details) : '')}</span>
      <Icon name="chevron" size={14} className="version-trigger__chev"/>
    </button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Model version")} width={520} onClose={() => setOpen(false)}>
      <div className="version-menu" ref={menu}>
        {tr(top.length ? <div className="version-section" role="group" aria-label={tr("Model version")}>{tr(top.map(row))}</div> : null)}
        {tr(more.length ? <div className="version-section" role="group" aria-label={tr("More versions")}>
          {tr(top.length ? <button role="menuitem" className="version-more" aria-expanded={foldedOpen} onClick={() => setMoreOpen(!foldedOpen)}><span>{tr("More versions")} <span className="muted">{more.length}</span></span><Icon name="chevron" size={14} className={cx('version-more__chev', foldedOpen && 'is-open')}/></button> : null)}
          {tr(foldedOpen ? <div className="version-more__rows">{tr(more.map(row))}</div> : null)}
        </div> : null)}
        <div className="version-foot">
          <button role="menuitem" className="version-foot__btn" onClick={() => { setOpen(false); onOwnGguf(); }}><Icon name="folder" size={14}/>{tr("Use your own GGUF…")}</button>
          <button role="menuitem" className="version-foot__btn" aria-pressed={details} onClick={() => setDetails(d => !d)}>{tr(details ? "Hide details" : "Show details")}</button>
        </div>
      </div>
    </FloatingMenu> : null)}
  </>;
}
