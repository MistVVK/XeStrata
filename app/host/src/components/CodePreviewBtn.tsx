import { tr } from '../lib/i18n';
import { cx } from '../lib/format';
import { Icon } from './ui';
import { useSelectedPreviewId } from '../lib/code-preview';
import type { PreviewArtifact } from '../lib/artifacts';

/** The preview control of a code block: an icon button like Copy, on its left. It shows both states: closed, and open (filled, "Close preview"). */
export function CodePreviewBtn({ artifact, onPreview, onClose }: { artifact: PreviewArtifact; onPreview: (a: PreviewArtifact) => void; onClose?: (id: string) => void }) {
  const open = useSelectedPreviewId() === artifact.id;
  const label = tr(open ? 'Close preview' : 'Open preview');
  return <button type="button" className={cx('st-btn st-btn--icon', open && 'is-open')} data-open={open || undefined} aria-pressed={open} aria-label={label} title={label}
    onClick={() => (open && onClose ? onClose(artifact.id) : onPreview(artifact))}>
    <Icon name="panel" size={19}/>
  </button>;
}
