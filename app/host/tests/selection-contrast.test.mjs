// #5073: selected text could hardly be seen: ::selection only set the accent tint, 1.01 to 1.05:1 against the page.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = name => readFileSync(new URL(`../src/styles/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const lum = hex => {
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
const app = read('app.css'), tokens = read('kit/tokens.css');
const block = (css, selector) => { const i = css.indexOf(`${selector} {`); assert.ok(i >= 0, `${selector} rule`); return css.slice(i, css.indexOf('}', i)); };
const value = (css, name) => { const m = css.match(new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`)); assert.ok(m, name); return m[1]; };

test('selected text uses its own fill and ink tokens, not the accent tint', () => {
  const rule = block(app, '::selection');
  assert.match(rule, /background:\s*var\(--st-select-bg\)/);
  assert.match(rule, /color:\s*var\(--st-select-ink\)/);
  assert.doesNotMatch(rule, /accent-tint/);
});

test('light: the selection shows against the page and the text on it stays readable', () => {
  const light = app.slice(app.indexOf(':root {\n  --st-select-bg'));
  const bg = value(light, '--st-select-bg'), ink = value(light, '--st-select-ink');
  for (const surface of [value(tokens, '--st-surface'), value(tokens, '--st-bg')]) assert.ok(ratio(bg, surface) >= 2, `fill against ${surface}: ${ratio(bg, surface).toFixed(2)}`);
  assert.ok(ratio(ink, bg) >= 4.5, `ink on fill: ${ratio(ink, bg).toFixed(2)}`);
});

test('dark: the selection shows against the page and the text on it stays readable', () => {
  const dark = app.slice(app.indexOf("[data-theme='dark'] {\n  --st-select-bg"));
  const bg = value(dark, '--st-select-bg'), ink = value(dark, '--st-select-ink');
  const darkTokens = tokens.slice(tokens.indexOf('[data-theme="dark"]'));
  for (const surface of [value(darkTokens, '--st-surface'), value(darkTokens, '--st-bg')]) assert.ok(ratio(bg, surface) >= 3, `fill against ${surface}: ${ratio(bg, surface).toFixed(2)}`);
  assert.ok(ratio(ink, bg) >= 4.5, `ink on fill: ${ratio(ink, bg).toFixed(2)}`);
});
