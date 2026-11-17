// serve/web/ime.js - the Enter that commits an IME conversion (Japanese, Chinese, Korean) belongs to the IME, not to the page (#5071).
// The same rule as the app's composer (apps/host/src/lib/ime-enter.ts; apps/host/tests/ime-fields.test.mjs checks that the two agree).
//
// The browsers disagree on the order of the events when Enter commits a conversion:
//   Chrome, Edge, Firefox:  keydown(Enter, keyCode 229, isComposing true)  ->  compositionend
//   Safari:                 compositionend  ->  keydown(Enter, keyCode 229, isComposing false)
//   WebKitGTK (IBus):       compositionend  ->  keydown(Enter, keyCode 13, isComposing false)
// So: a key that says it is composing, or has keyCode 229, is the IME's; so is the first Enter right after a compositionend (same
// tick, 50 ms). The flag is used up by the next key and expires on its own, so a second Enter always does its job.
//
//   const imeKey = StrataIme.watch(field);
//   field.addEventListener("keydown", (e) => { if (imeKey(e)) return; ... });
"use strict";
(function (root) {
  const WINDOW_MS = 50;

  function createGuard(windowMs = WINDOW_MS) {
    let armedAt = null;
    return {
      compositionStart() { armedAt = null; },
      compositionEnd(now) { armedAt = now; },
      keyDown(e, now) {
        const armed = armedAt;
        armedAt = null;                       // the next key of any kind uses the flag up
        if (e.isComposing || e.keyCode === 229) return true;
        return armed !== null && e.key === "Enter" && now - armed >= 0 && now - armed <= windowMs;
      },
      reset() { armedAt = null; },
    };
  }

  // the field's composition events feed one guard; the returned function says whether a keydown on the field is the IME's
  function watch(field, windowMs = WINDOW_MS) {
    const guard = createGuard(windowMs);
    field.addEventListener("compositionstart", () => guard.compositionStart());
    field.addEventListener("compositionend", (e) => guard.compositionEnd(e.timeStamp));
    field.addEventListener("blur", () => guard.reset());       // a conversion cut short by a click never sends a compositionend
    return (e) => guard.keyDown(e, e.timeStamp);
  }

  const api = {WINDOW_MS, createGuard, watch};
  root.StrataIme = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
