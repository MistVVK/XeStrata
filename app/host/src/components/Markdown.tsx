import { tr } from '../lib/i18n';
import { CopyBtn } from './CopyBtn';
/** Safe, dependency-free rendering of the Markdown structures used in answers. */
import { Fragment, type ReactNode } from 'react';
import { CodePreviewBtn } from './CodePreviewBtn';
import { codePreview } from '../lib/code-preview';
import type { PreviewArtifact } from '../lib/artifacts';
import { SyntaxCode } from './SyntaxCode';
import { MathView } from './MathView';
import { displayBlock, hasMath, splitMath } from '../lib/math';
import { imagePlan } from '../lib/markdown-images';
import { splitLinks } from '../lib/autolink';
function inlineMarks(text: string, onOpenLink?: (url: string) => void, imageSession?: string | null): ReactNode {
    return text.split(/(!\[[^\]]*\]\([^\s)]+\)|`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\([^\s)]+\))/g).map((part, i) => {
        const image = part.match(/^!\[([^\]]*)\]\(([^\s)]+)\)$/);
        if (image) {
            const plan = imagePlan(image[2], imageSession);
            if (plan.kind === 'img')
                return <img key={i} src={plan.src} alt={image[1]} className="md-image" loading="lazy" referrerPolicy="no-referrer"/>;
            if (plan.kind === 'link')
                return <a key={i} href={plan.href} target="_blank" rel="noreferrer" onClick={e => { if (onOpenLink) {
                    e.preventDefault();
                    onOpenLink(plan.href);
                } }}>{image[1] || plan.href}</a>;
            return <Fragment key={i}>{image[1] || part}</Fragment>;
        }
        if (part.startsWith('`') && part.endsWith('`'))
            return <code key={i} className="md-code">{part.slice(1, -1)}</code>;
        if (part.startsWith('**') && part.endsWith('**'))
            return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('*') && part.endsWith('*'))
            return <em key={i}>{part.slice(1, -1)}</em>;
        const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (link && /^(https?:\/\/|mailto:|#)/i.test(link[2]))
            return <a key={i} href={link[2]} target="_blank" rel="noreferrer" onClick={e => { if (onOpenLink && /^https?:/i.test(link[2])) {
                e.preventDefault();
                onOpenLink(link[2]);
            } }}>{link[1]}</a>;
        // #6000: a web address written as plain text is a link too (lib/autolink.ts), opened the same way
        return <Fragment key={i}>{splitLinks(part).map((piece, j) => piece.href
            ? <a key={j} href={piece.href} target="_blank" rel="noreferrer" onClick={e => { if (onOpenLink) {
                e.preventDefault();
                onOpenLink(piece.href!);
            } }}>{piece.text}</a>
            : <Fragment key={j}>{piece.text}</Fragment>)}</Fragment>;
    });
}
/** Text with its inline marks and, outside code spans, its math (#3268). Math is cut out first: a * or _ inside a formula is not emphasis. */
function inline(text: string, onOpenLink?: (url: string) => void, imageSession?: string | null): ReactNode {
    if (!hasMath(text))
        return inlineMarks(text, onOpenLink, imageSession);
    return text.split(/(`[^`]+`)/g).map((chunk, c) => {
        if (chunk.length > 1 && chunk.startsWith('`') && chunk.endsWith('`'))
            return <Fragment key={c}>{inlineMarks(chunk, onOpenLink, imageSession)}</Fragment>;
        return <Fragment key={c}>{splitMath(chunk).map((part, p) => part.kind === 'math'
            ? <MathView key={p} tex={part.tex} display={part.display}/>
            : <Fragment key={p}>{inlineMarks(part.text, onOpenLink, imageSession)}</Fragment>)}</Fragment>;
    });
}
/**
 * A tool call the model wrote as plain text that nothing ran (the server could not read it, or named a tool that is not there): a block that starts a
 * line with <tool_call> and holds a <function= or <parameter= tag. It reaches the answer as raw markup, which is never what a person wants to read, so it
 * is shown as one collapsed line that opens to the raw text. Only a block that starts its line counts: the same tag inside a code block (a fence takes
 * its lines first) or in the middle of a sentence stays as written, so a reply that talks about the format still shows it. A block that is not closed
 * yet (the reply is still streaming) runs to the end of the text. `rest` is what follows the closing tag on its line.
 */
export function failedCall(lines: string[], at: number): { raw: string; end: number; rest: string } | null {
    const start = lines[at].match(/^\s*<tool_call>/);
    if (!start)
        return null;
    const out: string[] = [];
    let end = lines.length - 1, rest = '';
    for (let j = at; j < lines.length; j++) {
        const from = j === at ? start[0].length : 0;
        const close = lines[j].indexOf('</tool_call>', from);
        if (close >= 0) {
            out.push(lines[j].slice(0, close + '</tool_call>'.length));
            rest = lines[j].slice(close + '</tool_call>'.length).trim();
            end = j;
            break;
        }
        out.push(lines[j]);
    }
    const raw = out.join('\n');
    return /<function=|<parameter=/.test(raw) ? { raw: raw.trim(), end, rest } : null;
}
/** Does this line continue the paragraph above it, or does it start a block of its own? */
export function continuesParagraph(lines: string[], at: number): boolean {
    const line = lines[at];
    if (!line.trim() || /^\s*```/.test(line) || failedCall(lines, at) || /^(#{1,6})\s+(.+)$/.test(line) || /^\s*[-*+]\s+\S/.test(line) || /^\s*\d+[.)]\s+\S/.test(line) || /^>\s?/.test(line))
        return false;
    if (line.includes('|') && at + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[at + 1]))
        return false;
    return !displayBlock(lines, at);
}
export function Markdown({ text, onOpenLink, previewId, onPreview, onClosePreview, imageSession }: {
    text: string;
    onOpenLink?: (url: string) => void;
    /** Where an HTML or Markdown code block can be shown in the sidebar: the message it belongs to, and the handlers that open and close a preview. */
    previewId?: string;
    onPreview?: (a: PreviewArtifact) => void;
    onClosePreview?: (id: string) => void;
    /** The chat whose folder a picture path in the answer is read from (the host serves only files inside it). */
    imageSession?: string | null;
}) {
    const nodes: ReactNode[] = [];
    // A file written on Windows has \r\n: the \r stays at the end of every line, where `.` does not match it, so a list line was never taken and the
    // list loop below never moved on (an endless loop that filled the page's memory, found in the sidebar preview). Line ends are read as one.
    const lines = text.split(/\r\n|\r|\n/);
    for (let i = 0; i < lines.length;) {
        const line = lines[i];
        if (!line.trim()) {
            i++;
            continue;
        }
        if (/^\s*```/.test(line)) {
            const language = line.trim().slice(3);
            const body: string[] = [];
            const key = i++;
            while (i < lines.length && !/^\s*```/.test(lines[i]))
                body.push(lines[i++]);
            i++;
            const code = body.join('\n');
            const preview = previewId && onPreview ? codePreview(previewId, language, code) : null;
            nodes.push(<div className="st-code" key={key}><div className="st-code__head"><span>{language || tr('code')}</span><span className="st-code__actions">{tr(preview && onPreview ? <CodePreviewBtn artifact={preview} onPreview={onPreview} onClose={onClosePreview}/> : null)}<CopyBtn text={code} label="Code" title="Copy code"/></span></div><SyntaxCode text={code} language={language}/></div>);
            continue;
        }
        const call = failedCall(lines, i);
        if (call) {
            nodes.push(<details className="md-failedcall" key={i}><summary>{tr('A tool call that could not run')}</summary><pre className="md-failedcall__raw">{call.raw}</pre></details>);
            if (call.rest) {
                lines[call.end] = call.rest;
                i = call.end;
            }
            else
                i = call.end + 1;
            continue;
        }
        const block = displayBlock(lines, i);
        if (block) {
            nodes.push(<div className="md-math-block" key={i}><MathView tex={block.tex.trim()} display/></div>);
            i += block.used;
            continue;
        }
        const heading = line.match(/^(#{1,6})\s+(.+)$/);
        if (heading) {
            const Tag = heading[1].length <= 2 ? 'h2' : heading[1].length === 3 ? 'h3' : 'h4';
            nodes.push(<Tag key={i}>{tr(inline(heading[2], onOpenLink, imageSession))}</Tag>);
            i++;
            continue;
        }
        if (/^\s*[-*+]\s+\S/.test(line) || /^\s*\d+[.)]\s+\S/.test(line)) {
            const ordered = /^\s*\d+[.)]/.test(line);
            const items: ReactNode[] = [];
            const key = i;
            const re = ordered ? /^\s*\d+[.)]\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/;
            while (i < lines.length && re.test(lines[i])) {
                items.push(<li key={i}>{tr(inline(lines[i].replace(re, '$1'), onOpenLink))}</li>);
                i++;
            }
            if (i === key) {              // never stand still: a line the list start accepted but the item pattern did not is shown as text and passed
                nodes.push(<p key={key}>{line}</p>);
                i++;
                continue;
            }
            nodes.push(ordered ? <ol key={key}>{tr(items)}</ol> : <ul key={key}>{tr(items)}</ul>);
            continue;
        }
        if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
            const cells = (row: string) => row.trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
            const header = cells(line);
            const rows: string[][] = [];
            const key = i;
            i += 2;
            while (i < lines.length && lines[i].includes('|'))
                rows.push(cells(lines[i++]));
            nodes.push(<div className="md-table-wrap" key={key}><table className="md-table"><thead><tr>{tr(header.map((c, j) => <th key={j}>{tr(inline(c, onOpenLink, imageSession))}</th>))}</tr></thead><tbody>{tr(rows.map((r, j) => <tr key={j}>{tr(r.map((c, k) => <td key={k}>{tr(inline(c, onOpenLink, imageSession))}</td>))}</tr>))}</tbody></table></div>);
            continue;
        }
        if (/^>\s?/.test(line)) {
            nodes.push(<blockquote key={i}>{tr(inline(line.replace(/^>\s?/, ''), onOpenLink))}</blockquote>);
            i++;
            continue;
        }
        // One paragraph: a single newline inside it is a soft break (a space); two spaces or a backslash at the end of a line is a hard break (#3818).
        const key = i, parts: ReactNode[] = [];
        let current = line;
        i++;
        while (true) {
            const hard = /( {2,}|\\)$/.test(current);
            parts.push(<Fragment key={parts.length}>{tr(inline(current.replace(/( {2,}|\\)$/, ''), onOpenLink, imageSession))}</Fragment>);
            if (i >= lines.length || !continuesParagraph(lines, i))
                break;
            parts.push(hard ? <br key={`b${parts.length}`}/> : ' ');
            current = lines[i++].trimStart();
        }
        nodes.push(<p key={key}>{tr(parts)}</p>);
    }
    return <>{tr(nodes)}</>;
}
