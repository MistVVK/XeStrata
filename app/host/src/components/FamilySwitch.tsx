import { useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import type { CatalogModel } from '../lib/native-catalog';
import { FloatingMenu } from './FloatingMenu';
import { Icon } from './ui';

/**
 * The model family in the Models navbar: its icon, its name and a chevron. Opening it lists the families of the catalog (Qwen3.8-Flash-Next,
 * Qwen3.6-35B-A3B, and future ones) to switch between. One family still shows as the same control, so the page looks the same as families arrive.
 */
export function FamilySwitch({ families, selected, onPick }: { families: CatalogModel[]; selected?: string; onPick: (family: CatalogModel) => void }) {
  const shown = families.filter(f => f.sizes.length);
  const missing = families.filter(f => !f.sizes.length && f.available === false);      // a model of the update feed this engine does not run: said, never offered
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  if (!shown.length) return null;
  const current = shown.find(f => f.family === selected) ?? shown[0];
  return <>
    <button ref={trigger} type="button" className="family-trigger" aria-haspopup="menu" aria-expanded={open} aria-label={tr("Model family")} onClick={() => setOpen(v => !v)}>
      <Icon name="models" size={20}/><strong>{tr(current.title)}</strong><Icon name="chevron" size={16} className="family-trigger__chev"/>
    </button>
    {tr(open ? <FloatingMenu anchor={trigger.current} title={tr("Model family")} width={320} onClose={() => setOpen(false)}>
      <FamilyMenuList shown={shown} missing={missing} current={current} onPick={f => { setOpen(false); if (f.family !== current.family) onPick(f); }}/>
    </FloatingMenu> : null)}
  </>;
}

/** The rows of the family menu: the families to pick, then (as notes, not choices) the models of the update feed this engine cannot use, each with the plain reason. */
export function FamilyMenuList({ shown, missing, current, onPick }: { shown: CatalogModel[]; missing: CatalogModel[]; current: CatalogModel; onPick: (family: CatalogModel) => void }) {
  return (
    <div className="family-menu">
      {shown.map(f => <button key={f.family} type="button" role="menuitemradio" aria-checked={f.family === current.family} className="family-menu__item" onClick={() => onPick(f)}>
        <Icon name="models" size={18}/><span><strong>{tr(f.title)}</strong>{f.experimental ? <small>{tr("Experimental")}</small> : null}{f.available === false && f.unavailable_reason ? <small>{f.unavailable_reason}</small> : null}</span>
        {f.family === current.family ? <Icon name="check" size={16} className="family-menu__check"/> : null}
      </button>)}
      {missing.map(f => <div key={f.family} className="family-menu__item" role="note" aria-disabled="true"><Icon name="models" size={18}/><span><strong>{f.title}</strong><small>{f.unavailable_reason}</small></span></div>)}
    </div>
  );
}
