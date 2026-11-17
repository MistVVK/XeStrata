import { useEffect, useRef, useState, type ReactNode } from 'react';
import { NATIVE_DRAG_EVENT, NATIVE_DROP_EVENT, dragHasFiles, fetchDropped, readDroppedRefs } from '../lib/native-drop';
import { pushToast } from '../lib/store';
import { tr } from '../lib/i18n';
import { Icon } from './ui';

/** Files dropped on the chat go where the attach button would put them: the composer reads the 'strata:drop-files' event. */
export const DROP_EVENT = 'strata:drop-files';
export { pastedFiles } from '../lib/pasted-files';
const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');

/** A file dropped outside the chat must not take the window to that file: the page ignores it (called once at start). */
export function installDropGuard() {
    const guard = (e: DragEvent) => { if (!e.defaultPrevented && dragHasFiles(e.dataTransfer?.types)) e.preventDefault(); };
    window.addEventListener('dragover', guard);
    window.addEventListener('drop', guard);
}

/** The conversation column, with a calm highlight while something is dragged over it. */
export function DropZone({ children, className }: { children: ReactNode; className?: string }) {
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  // Linux and macOS: the app shell reports the drag and the drop (the page is not given the files there).
  useEffect(() => {
    const drag = (e: Event) => setOver((e as CustomEvent).detail === 'enter');
    const drop = (e: Event) => {
      setOver(false); depth.current = 0;
      const refs = readDroppedRefs((e as CustomEvent).detail);
      if (!refs.length) return;
      void fetchDropped(refs).then(files => {
        if (files.length) window.dispatchEvent(new CustomEvent(DROP_EVENT, { detail: files }));
        else pushToast('error', tr('The file was not added'), tr('Try dropping it again.'));
      });
    };
    window.addEventListener(NATIVE_DRAG_EVENT, drag); window.addEventListener(NATIVE_DROP_EVENT, drop);
    return () => { window.removeEventListener(NATIVE_DRAG_EVENT, drag); window.removeEventListener(NATIVE_DROP_EVENT, drop); };
  }, []);
  return <div className={className} data-dropping={over || undefined}
    onDragEnter={e => { if (!hasFiles(e)) return; e.preventDefault(); depth.current++; setOver(true); }}
    onDragOver={e => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
    onDragLeave={e => { if (!hasFiles(e)) return; depth.current = Math.max(0, depth.current - 1); if (!depth.current) setOver(false); }}
    onDrop={e => { if (!hasFiles(e)) return; e.preventDefault(); depth.current = 0; setOver(false); const files = Array.from(e.dataTransfer.files); if (files.length) window.dispatchEvent(new CustomEvent(DROP_EVENT, { detail: files })); }}>
    {children}
    {over ? <div className="dropzone" aria-hidden="true"><Icon name="attach" size={22}/><strong>{tr("Drop files to add them")}</strong><span>{tr("Pictures and text files")}</span></div> : null}
  </div>;
}
