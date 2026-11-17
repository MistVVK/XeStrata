import { tr } from '../lib/i18n';
import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { measureFrame, placeAnchored } from '../lib/overlay-place';
export function FloatingMenu({ anchor, title, onClose, children, width = 232 }: {
    anchor: HTMLElement | null;
    title: string;
    onClose: () => void;
    children: React.ReactNode;
    width?: number;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const [placed, setPlaced] = useState<{ left: number; top: number; maxWidth?: number; maxHeight?: number }>({ left: 12, top: 12 });
    useLayoutEffect(() => {
        // Placed in visual pixels and written in the page's own pixels (lib/overlay-place.ts): the page may be scaled by "Interface size" (#4653).
        const place = () => {
            const el = ref.current;
            const { scale, viewport } = measureFrame();
            const rect = anchor?.getBoundingClientRect();
            const natural = el ? Math.max(el.getBoundingClientRect().height, el.scrollHeight * scale) : 200 * scale;
            const home = { left: 12, top: 12, right: 12, bottom: 12 };
            const p = placeAnchored({ anchor: rect ?? home, size: { width: width * scale, height: natural }, viewport, scale });
            setPlaced({ left: p.left, top: p.top, maxWidth: p.maxWidth, maxHeight: p.maxHeight });
        };
        place();
        ref.current?.querySelector<HTMLElement>('[role="menuitem"]:not(:disabled), [role="menuitemradio"]:not(:disabled)')?.focus();
        const outside = (e: PointerEvent) => { if (!ref.current?.contains(e.target as Node) && !anchor?.contains(e.target as Node))
            onClose(); };
        document.addEventListener('pointerdown', outside);
        window.addEventListener('resize', place);
        return () => { document.removeEventListener('pointerdown', outside); window.removeEventListener('resize', place); if (anchor?.isConnected)
            anchor.focus({ preventScroll: true }); };
    }, [anchor, width]);
    return createPortal(<div className="menu" role="menu" aria-label={tr(title)} ref={ref} style={{ ...placed, zIndex: anchor?.closest('[role="dialog"]') ? 'var(--layer-dialog-popover)' : 'var(--layer-popover)', width, maxWidth: placed.maxWidth ?? 'calc(100dvw - 24px)', maxHeight: placed.maxHeight ?? 'calc(100dvh - 24px)', overflowY: 'auto' }} onKeyDown={e => {
            const nodes = Array.from(ref.current?.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled), [role="menuitemradio"]:not(:disabled)') ?? []);
            const at = nodes.indexOf(document.activeElement as HTMLElement);
            if (['Escape', 'Tab'].includes(e.key)) {
                e.preventDefault();
                e.stopPropagation();
                onClose();
            }
            if (['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) {
                e.preventDefault();
                nodes[e.key === 'Home' ? 0 : e.key === 'End' ? nodes.length - 1 : (at + (e.key === 'ArrowDown' ? 1 : -1) + nodes.length) % nodes.length]?.focus();
            }
        }}>{tr(children)}</div>, document.body);
}
