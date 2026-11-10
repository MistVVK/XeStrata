// #4653: the sidebar "More" menu did nothing at "Interface size" 125% on Windows. The page is scaled with CSS `zoom`; WebView2 reports the anchor
// in screen pixels but writes left/top in the page's own pixels, so the menu landed (scale - 1) of its distance too far right and down: off the
// screen at the bottom-left. The placement is now a pure function in screen pixels, divided by the scale at the end.
// Real engine check (Edge 156 = the WebView2 runtime): phaseA-tests/round0213/guidata-shell/more-menu.mjs (14 of 18 cases broken before, 0 after).
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const { code } = await transform(readFileSync(resolve(root, 'src/lib/overlay-place.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const { placeAnchored, measureScale, measureFrame, sane } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const VIEWPORTS = [[1920, 1080], [1536, 864], [1280, 720], [1024, 700], [900, 600], [640, 480], [360, 640]];
const SCALES = [0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];   // the "Interface size" choices (store.ts)
const box = (left, top, width, height) => ({ left, top, right: left + width, bottom: top + height });

/** The way the menu was placed before: the anchor's rectangle written back as CSS values (FloatingMenu.tsx at a3a10b90a). */
function before(anchor, size, vp, scale) {
  const left = Math.max(12, Math.min(anchor.left, vp.width - Math.min(size.width, vp.width - 24) - 12));
  const below = anchor.bottom + 8;
  const top = below + size.height + 12 <= vp.height ? below : Math.max(12, anchor.top - size.height - 6);
  return { left: left * scale, top: top * scale, bottom: top * scale + size.height };   // what the person sees: CSS value times the scale
}

test('the old placement is off the screen at the bottom-left once the page is scaled (the bug)', () => {
  const vp = { width: 1920, height: 1080 };
  const anchor = box(15, 1020, 232, 36);               // the More item at the foot of the sidebar, seen at 125%
  const seen = before(anchor, { width: 232 * 1.25, height: 107 }, vp, 1.25);
  assert.ok(seen.bottom > vp.height, `the menu's bottom edge is at ${seen.bottom} of ${vp.height}`);
  assert.equal(before(box(12, 1020, 232, 36), { width: 232, height: 86 }, vp, 1).bottom <= vp.height, true, 'and fine at 100%');
});

test('everywhere, at every scale, the panel is on the screen and its CSS values are the screen values divided by the scale', () => {
  for (const [w, h] of VIEWPORTS) for (const scale of SCALES) {
    const vp = { width: w, height: h };
    const anchors = [box(12, h - 60, 232, 40), box(12, 8, 232, 40), box(w - 260, h / 2, 40, 40), box(w / 2 - 20, h / 2 - 20, 40, 40), box(w - 30, h - 40, 24, 24)];
    for (const anchor of anchors) for (const menuH of [60, 107, 320, 2000]) {
      const size = { width: 232 * scale, height: menuH * scale };
      const p = placeAnchored({ anchor, size, viewport: vp, scale });
      const tag = `${w}x${h} scale ${scale} anchor ${JSON.stringify(anchor)} menu ${menuH}`;
      // CSS value x scale is where it really is
      assert.ok(Math.abs(p.left * scale - p.visual.left) < 1e-6, tag);
      assert.ok(Math.abs(p.top * scale - p.visual.top) < 1e-6, tag);
      // inside the screen with the margin (the panel shrinks and scrolls when it is taller than the room)
      assert.ok(p.visual.left >= 12 - 1e-6 && p.visual.top >= 12 - 1e-6, tag + ' top-left ' + JSON.stringify(p.visual));
      assert.ok(p.visual.right <= w - 12 + 1e-6 || w < 24 + 232 * scale, tag + ' right ' + JSON.stringify(p.visual));
      assert.ok(p.visual.bottom <= h - 12 + 1e-6, tag + ' bottom ' + JSON.stringify(p.visual));
      // a limit it can be shortened to, in CSS pixels, that keeps it inside
      assert.ok(p.maxHeight * scale <= h - 24 + 1e-6, tag);
      assert.ok(p.maxWidth * scale <= w - 24 + 1e-6, tag);
    }
  }
});

test('the More menu (bottom-left of the window) opens above the item, whole, at every scale of the 1080p window', () => {
  for (const scale of SCALES) {
    const vp = { width: 1920, height: 1080 };
    const anchor = box(12 * scale, 1080 - 12 - 40 * scale, 232 * scale, 40 * scale);
    const p = placeAnchored({ anchor, size: { width: 232 * scale, height: 86 * scale }, viewport: vp, scale });
    assert.equal(p.above, true, 'scale ' + scale);
    assert.ok(p.visual.bottom <= anchor.top, 'does not cover the item');
    assert.ok(Math.abs(p.visual.bottom - p.visual.top - 86 * scale) < 1e-6, 'its whole height is shown');
  }
});

test('below the anchor when it fits; above when only there it fits; the side with more room, shortened, when neither', () => {
  const vp = { width: 1000, height: 700 };
  const mid = placeAnchored({ anchor: box(100, 100, 100, 30), size: { width: 200, height: 150 }, viewport: vp, scale: 1 });
  assert.equal(mid.above, false); assert.equal(mid.top, 138);
  const low = placeAnchored({ anchor: box(100, 640, 100, 30), size: { width: 200, height: 150 }, viewport: vp, scale: 1 });
  assert.equal(low.above, true); assert.equal(low.top, 640 - 6 - 150);
  const tall = placeAnchored({ anchor: box(100, 300, 100, 30), size: { width: 200, height: 900 }, viewport: vp, scale: 1 });
  assert.ok(tall.maxHeight <= 700 - 24 && tall.maxHeight > 250, 'shortened to the larger side: ' + tall.maxHeight);
  assert.ok(tall.visual.top >= 12 && tall.visual.bottom <= 688);
});

test('a short window with no room on either side: the panel covers part of its anchor instead of shrinking to nothing', () => {
  const p = placeAnchored({ anchor: box(100, 80, 100, 40), size: { width: 200, height: 150 }, viewport: { width: 1000, height: 200 }, scale: 1 });
  assert.equal(p.above, false);
  assert.ok(p.maxHeight >= 150, 'its whole height is allowed: ' + p.maxHeight);
  assert.ok(p.visual.top >= 12 && p.visual.bottom <= 188, JSON.stringify(p.visual));
});

test('left aligned with the anchor, pushed back inside when it would run off the right edge', () => {
  const p = placeAnchored({ anchor: box(900, 100, 80, 30), size: { width: 232, height: 100 }, viewport: { width: 1000, height: 700 }, scale: 1 });
  assert.equal(p.visual.right, 1000 - 12);
  const q = placeAnchored({ anchor: box(300, 100, 80, 30), size: { width: 232, height: 100 }, viewport: { width: 1000, height: 700 }, scale: 1 });
  assert.equal(q.left, 300);
});

test('a tooltip beside a sidebar row opens to the right, top aligned, and is kept inside', () => {
  const p = placeAnchored({ anchor: box(10, 400, 240, 32), size: { width: 272 * 1.25, height: 80 * 1.25 }, viewport: { width: 1920, height: 1080 }, scale: 1.25, side: true, gapBelow: 12 });
  assert.equal(p.visual.left, 262);
  assert.equal(p.visual.top, 400);
  const low = placeAnchored({ anchor: box(10, 1060, 240, 32), size: { width: 272, height: 100 }, viewport: { width: 1920, height: 1080 }, scale: 1, side: true, gapBelow: 12 });
  assert.equal(low.visual.bottom, 1080 - 12);
});

test('nonsense numbers never throw and never put the panel outside the screen', () => {
  for (const scale of [0, NaN, -1, Infinity, 100, 0.01]) {
    assert.equal(sane(scale), 1);
    const p = placeAnchored({ anchor: box(0, 0, 0, 0), size: { width: 100, height: 100 }, viewport: { width: 800, height: 600 }, scale });
    assert.ok(Number.isFinite(p.left) && Number.isFinite(p.top));
  }
  const p = placeAnchored({ anchor: box(5, 5, 5, 5), size: { width: NaN, height: NaN }, viewport: { width: NaN, height: 0 }, scale: 1 });
  assert.ok(Number.isFinite(p.left) && Number.isFinite(p.top) && Number.isFinite(p.maxHeight));
});

test('the scale is measured from the page itself, and a failed measure is 1', () => {
  const doc = (width) => ({ createElement: () => ({ style: {}, getBoundingClientRect: () => ({ width }), remove() {} }) });
  const host = { appendChild() {} };
  globalThis.document = doc(125);
  assert.equal(measureScale(host), 1.25);
  globalThis.document = doc(100);
  assert.equal(measureScale(host), 1);
  globalThis.document = doc(0);
  assert.equal(measureScale(host), 1);
  globalThis.document = { createElement: () => { throw new Error('no DOM'); } };
  assert.equal(measureScale(host), 1);
});

test('every anchored panel uses the scale-aware numbers', () => {
  const read = f => readFileSync(resolve(root, 'src', f), 'utf8');
  assert.match(read('components/FloatingMenu.tsx'), /placeAnchored\(/);
  for (const f of ['components/FloatingMenu.tsx', 'components/Popover.tsx', 'components/ui.tsx', 'components/VersionMemoryTooltip.tsx', 'app/Sidebar.tsx', 'screens/Chat.tsx']) {
    assert.match(read(f), /measureFrame\(\)/, f);
    // the window's size comes from the same measure as the rectangles (WebView2 122 counts them differently: QAGUI-STRIX 11-09)
    assert.doesNotMatch(read(f), /window\.inner(Width|Height) - /, f);
  }
  assert.doesNotMatch(read('components/FloatingMenu.tsx'), /calc\(100dvh - \$\{placed\.top/, 'the viewport-unit max height is gone');
});

test('a Select list: as tall as it needs when the window has the room, else all the room above or below, never off the top (owner report, 11-04)', () => {
  // the owner's case: a short window, a long list, the button in the lower half; the old list was capped at 320 px and opened upward by a
  // transform, so at 150% it started above the window
  for (const scale of SCALES) for (const [w, h] of VIEWPORTS) for (const rows of [3, 9, 40, 118]) {
    const natural = (rows * 38 + 12) * scale;
    for (const top of [h * 0.2, h * 0.5, h * 0.8]) {
      const anchor = box(w * 0.6, top, 240, 34 * scale);
      const p = placeAnchored({ anchor, size: { width: 260 * scale, height: natural }, viewport: { width: w, height: h }, scale, gapBelow: 6, gapAbove: 6 });
      const tag = `${w}x${h} scale ${scale} rows ${rows} top ${top}`;
      assert.ok(p.visual.top >= 12 - 1e-6 && p.visual.bottom <= h - 12 + 1e-6, tag + ' ' + JSON.stringify(p.visual));
      const roomBelow = h - anchor.bottom - 6 - 12, roomAbove = anchor.top - 6 - 12;
      if (natural <= Math.max(roomBelow, roomAbove) && Math.max(roomBelow, roomAbove) >= 120)
        assert.ok(Math.abs((p.visual.bottom - p.visual.top) - natural) < 1e-6 || p.maxHeight * scale >= natural - 1e-6, tag + ': it fits, so all of it is shown');
      else if (Math.max(roomBelow, roomAbove) >= 120)
        assert.ok(p.maxHeight * scale >= Math.max(roomBelow, roomAbove) - 1e-6, tag + ': too long, so it uses all the room');
    }
  }
});

test('the Select list is placed by the shared function, with no fixed cap and no upward transform', () => {
  const ui = readFileSync(resolve(root, 'src/components/ui.tsx'), 'utf8');
  assert.match(ui, /placeAnchored\(\{ anchor: r, size: \{ width, height: natural \}/);
  assert.match(ui, /maxHeight: placed\.maxHeight/);
  assert.doesNotMatch(ui, /data-up=\{placed\.up/);
  assert.doesNotMatch(ui, /Math\.min\(320, options\.length/);
});

// QAGUI-STRIX 11-09: on WebView2 122 (Chromium before 128) the More menu opened off the window at 125% and up. That engine reports rectangles in the
// page's own pixels under a zoomed root (a 100 px probe measures 100) while innerWidth/innerHeight stay screen pixels; the current engine reports both
// in screen pixels. A fake page of each kind: the window's size is read from a fixed box measured like any rectangle, so it matches the anchor.
// Real engine check: Edge with --disable-blink-features=StandardizedBrowserZoom behaves as the old one (phaseA-tests/par/appwin/menus.mjs).
function fakeEngine({ zoom, standard, inner }) {
  const unit = standard ? zoom : 1;   // screen pixels per unit the rectangles are reported in
  globalThis.document = { createElement: () => { const el = { style: { cssText: '' }, remove() {} };
    el.getBoundingClientRect = () => el.style.cssText.includes('width:100px') ? { width: 100 * unit, height: unit }
      : { width: standard ? inner.width : inner.width / zoom, height: standard ? inner.height : inner.height / zoom };
    return el; } };
  globalThis.window = { innerWidth: inner.width, innerHeight: inner.height };
  return { toScreen: css => css * zoom, rect: screen => standard ? screen : screen / zoom };
}

test('WebView2 122 and the current engine: the More menu at the foot of the sidebar is inside the window at every Interface size', () => {
  const host = { appendChild() {} };
  try {
    for (const standard of [true, false]) for (const [w, h] of [[1440, 920], [1024, 700], [1920, 1080]]) for (const zoom of [0.9, 1, 1.25, 1.5, 1.75]) {
      const eng = fakeEngine({ zoom, standard, inner: { width: w, height: h } });
      const { scale, viewport } = measureFrame(host);
      // the More item on the screen: 12 px from the left and the bottom, 40 CSS px tall, 232 wide; the menu is 107 CSS px tall
      const item = { left: 12 * zoom, top: h - 52 * zoom, right: 244 * zoom, bottom: h - 12 * zoom };
      const anchor = { left: eng.rect(item.left), top: eng.rect(item.top), right: eng.rect(item.right), bottom: eng.rect(item.bottom) };
      const p = placeAnchored({ anchor, size: { width: 232 * scale, height: 107 * scale }, viewport, scale });
      const top = eng.toScreen(p.top), bottom = top + Math.min(107, p.maxHeight) * zoom;
      const tag = `${standard ? 'current' : 'WebView2 122'} ${w}x${h} at ${Math.round(zoom * 100)}%`;
      assert.ok(top >= 0 && bottom <= h, `${tag}: the menu is painted at ${top.toFixed(0)}..${bottom.toFixed(0)} in a ${h} px window`);
      assert.ok(bottom <= item.top + 1e-6, `${tag}: it does not cover the item`);
      // the window's size taken from innerWidth/innerHeight (before) puts it off the window on the old engine at 125% and up: the bug
      if (!standard && zoom >= 1.25 && w === 1440) {
        const old = placeAnchored({ anchor, size: { width: 232 * scale, height: 107 * scale }, viewport: { width: w, height: h }, scale });
        assert.ok(eng.toScreen(old.top) + 107 * zoom > h, `${tag}: the old measure was off the window`);
      }
    }
  } finally { delete globalThis.document; delete globalThis.window; }
});

