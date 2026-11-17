/** The Copy controls of the app, with one shared behaviour: the control itself shows that the copy worked (#4748). */
import { createContext, useContext } from 'react';
import { tr } from '../lib/i18n';
import { cx } from '../lib/format';
import { COPIED_MENU_MS, useCopy } from '../lib/use-copy';
import { Icon } from './ui';

type Source = string | (() => string | Promise<string>);

/** A round icon button that copies `text`: the icon turns into a check mark for a moment, and a screen reader hears "Copied". */
export function CopyBtn({ text, label, title, disabled, size = 19 }: { text: Source; label: string; title: string; disabled?: boolean; size?: number }) {
    const { copied, copy } = useCopy();
    return <button type="button" className={cx('st-btn st-btn--icon', copied && 'is-copied')} data-copied={copied || undefined} title={tr(copied ? 'Copied' : title)} aria-label={tr(title)} disabled={disabled} onClick={() => void copy(text, label)}>
      <Icon name={copied ? 'check' : 'copy'} size={size}/>
      <span className="sr-only" role="status">{copied ? tr('Copied') : ''}</span>
    </button>;
}

/** What closes the menu a menu item is in; Chat's message menu provides it. */
export const MenuCloseContext = createContext<() => void>(() => {});

/** A menu item that copies `text`: it says "Copied" for a moment, and then the menu closes. */
export function CopyMenuItem({ text, label, children, disabled, onClose }: { text: Source; label: string; children: string; disabled?: boolean; onClose?: () => void }) {
    const { copied, copy } = useCopy(COPIED_MENU_MS);
    const close = useContext(MenuCloseContext);
    return <button role="menuitem" className="menu__item" data-copied={copied || undefined} disabled={disabled} onClick={e => {
        // The menu would close on this click before the person sees that it worked.
        e.stopPropagation();
        void copy(text, label).then(ok => { setTimeout(onClose ?? close, ok ? COPIED_MENU_MS : 0); });
    }}><Icon name={copied ? 'check' : 'copy'}/>{tr(copied ? 'Copied' : children)}</button>;
}
