import { useEffect, useState } from 'react';
import { parseDecimal, partialDecimal } from '../lib/decimal';

/** A number field that lets the person type a decimal point: it keeps the text while it is typed and tells the page the number once it is one (#3736). */
export function DecimalInput({ value, onChange, ariaLabel, className = 'st-input', placeholder, disabled }: {
  value: number | null | undefined;
  onChange: (value: number | null) => void;
  ariaLabel: string;
  className?: string;
  placeholder?: string;
  disabled?: boolean;
}) {
  const [text, setText] = useState(value === null || value === undefined ? '' : String(value));
  // A change that came from elsewhere (Restore saved, another tab) replaces the text, but never while the typed text already means this number.
  useEffect(() => {
    const shown = text === '' ? null : parseDecimal(text);
    if ((value ?? null) !== shown) setText(value === null || value === undefined ? '' : String(value));
  }, [value]);
  return <input className={className} aria-label={ariaLabel} inputMode="decimal" disabled={disabled} placeholder={placeholder} value={text}
    onChange={e => {
      const next = e.target.value;
      if (!partialDecimal(next)) return;
      setText(next);
      if (next.trim() === '') onChange(null);
      else { const n = parseDecimal(next); if (n !== null) onChange(n); }
    }}
    onBlur={() => { const n = text.trim() === '' ? null : parseDecimal(text); setText(n === null ? '' : String(n)); }}/>;
}
