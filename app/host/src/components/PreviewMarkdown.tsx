/**
 * The Markdown preview of the sidebar, bounded (#5180). Same elements as <Markdown>, but:
 *   - memoised by the text, so a chat that renders for every group of streamed tokens does not rebuild an unchanged file;
 *   - drawn in slices (lib/preview-render.ts) that are memoised one by one, so a file changed in one place re-renders that slice;
 *   - only a few slices at a time, the rest added by one waiting step that a newer text or leaving the view cancels;
 *   - a block too large to parse, and a file above the limit, shown as plain text.
 */
import { memo, useEffect, useMemo, useState } from 'react';
import { Markdown } from './Markdown';
import { SyntaxCode } from './SyntaxCode';
import { previewSlices, previewTier, slicesAfter, slicesAtStart, textRevision } from '../lib/preview-render';

const Slice = memo(function Slice({ text, plain }: { text: string; plain: boolean }) {
    return plain ? <SyntaxCode text={text}/> : <Markdown text={text}/>;
});

export const PreviewMarkdown = memo(function PreviewMarkdown({ text }: { text: string }) {
    if (previewTier(text) === 'plain')
        return <SyntaxCode text={text}/>;
    return <Sliced text={text}/>;
});

function Sliced({ text }: { text: string }) {
    const slices = useMemo(() => previewSlices(text), [text]);
    const revision = useMemo(() => textRevision(text), [text]);
    const [shown, setShown] = useState(() => slicesAtStart(0, slices.length));
    // one step waiting at most; a new revision or leaving the view clears it, so work for an old text never runs
    useEffect(() => {
        if (shown >= slices.length)
            return;
        const timer = setTimeout(() => setShown(n => slicesAfter(n, slices.length)), 24);
        return () => clearTimeout(timer);
    }, [revision, shown, slices.length]);
    const count = Math.min(Math.max(shown, slicesAtStart(0, slices.length)), slices.length);
    return <>{slices.slice(0, count).map(s => <Slice key={s.key} text={s.text} plain={s.plain}/>)}</>;
}
