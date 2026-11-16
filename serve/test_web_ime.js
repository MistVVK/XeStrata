// The page's IME rule (serve/web/ime.js, #5071), run in Node: each browser's order of events for an Enter that commits a conversion
// is replayed through the guard and through watch() on a field, the way the composer, the rename field and the answer editor use it.
// VERIFIED BY REPLAY ONLY: the orders are the ones written in ime.js, not captured from a browser in this run.
//   node serve/test_web_ime.js      (serve/test_web_ime.py runs it too and skips when node is missing)
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const I = require(path.join(__dirname, "web", "ime.js"));

// t is the event's timeStamp in ms
const down = (t, key, o = {}) => ({type: "keydown", t, key, keyCode: o.keyCode ?? (key === "Enter" ? 13 : 0), isComposing: !!o.isComposing});
const imeDown = (t, key, isComposing = true) => down(t, key, {keyCode: 229, isComposing});
const start = (t) => ({type: "compositionstart", t});
const end = (t) => ({type: "compositionend", t});

// a field's keydown path reduced to what matters: the guard first, then the field's own Enter and Escape
function replay(events) {
  const g = I.createGuard();
  const out = {enters: 0, escapes: 0, imeKeys: 0};
  for (const e of events) {
    if (e.type === "compositionstart") g.compositionStart();
    else if (e.type === "compositionend") g.compositionEnd(e.t);
    else if (e.type === "blur") g.reset();
    else if (g.keyDown(e, e.t)) out.imeKeys++;
    else if (e.key === "Enter") out.enters++;
    else if (e.key === "Escape") out.escapes++;
  }
  return out;
}

const cases = {
  chrome_edge_firefox_the_composing_keydown_comes_first() {
    const r = replay([start(0), imeDown(100, "Enter"), end(101), down(900, "Enter")]);
    assert.deepEqual(r, {enters: 1, escapes: 0, imeKeys: 1});
  },
  safari_compositionend_first_then_keycode_229_not_composing() {
    // what app.js let through before: isComposing is false here, so the commit sent the message
    const r = replay([start(0), end(100), imeDown(101, "Enter", false), down(900, "Enter")]);
    assert.deepEqual(r, {enters: 1, escapes: 0, imeKeys: 1});
  },
  webkitgtk_compositionend_first_then_a_plain_enter_in_the_same_tick() {
    const r = replay([start(0), end(100), down(101, "Enter"), down(900, "Enter")]);
    assert.deepEqual(r, {enters: 1, escapes: 0, imeKeys: 1});
  },
  latin_input_every_enter_counts() {
    assert.deepEqual(replay([down(10, "Enter"), down(20, "Enter")]), {enters: 2, escapes: 0, imeKeys: 0});
  },
  an_escape_that_cancels_a_conversion_is_the_imes() {
    // Chrome: keydown(Escape, 229, composing) then compositionend; the next Escape is the person's
    const r = replay([start(0), imeDown(100, "Escape"), end(101), down(900, "Escape")]);
    assert.deepEqual(r, {enters: 0, escapes: 1, imeKeys: 1});
  },
  the_flag_expires_and_is_used_up_by_any_key() {
    assert.equal(replay([start(0), end(100), down(100 + I.WINDOW_MS + 1, "Enter")]).enters, 1, "after the window, an Enter is the person's");
    assert.equal(replay([start(0), end(100), down(100 + I.WINDOW_MS, "Enter")]).enters, 0, "at the edge it is still the commit");
    assert.equal(replay([start(0), end(100), down(101, "a"), down(102, "Enter")]).enters, 1, "another key used the flag up");
    assert.equal(replay([start(0), end(100), {type: "blur"}, down(101, "Enter")]).enters, 1, "a blur clears it");
    assert.equal(replay([start(0), end(100), start(101), down(102, "Enter", {}), end(200)]).enters, 1, "a new conversion clears it");
  },
  watch_listens_to_the_fields_own_events() {
    const field = new EventTarget();
    const imeKey = I.watch(field);
    const key = (k, o = {}) => ({key: k, keyCode: o.keyCode ?? 13, isComposing: !!o.isComposing, timeStamp: o.t});
    const at = (type, t) => { const e = new Event(type); Object.defineProperty(e, "timeStamp", {value: t}); field.dispatchEvent(e); };
    at("compositionstart", 0); at("compositionend", 100);
    assert.equal(imeKey(key("Enter", {keyCode: 229, t: 101})), true, "Safari's commit");
    assert.equal(imeKey(key("Enter", {t: 900})), false, "the next Enter sends");
    at("compositionstart", 1000); at("compositionend", 1100); at("blur", 1101);
    assert.equal(imeKey(key("Enter", {t: 1102})), false, "the blur cleared the flag");
    assert.equal(imeKey(key("Enter", {isComposing: true, keyCode: 229, t: 1200})), true);
  },
};

(async () => {
  const name = process.argv[2];
  const run = name ? [[name, cases[name]]] : Object.entries(cases);
  if (name && !cases[name]) throw new Error("unknown test " + name);
  for (const [n, fn] of run) { await fn(); console.log("ok " + n); }
})().catch((e) => { console.error(e); process.exitCode = 1; });
