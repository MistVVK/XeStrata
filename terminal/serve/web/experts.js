// serve/web/experts.js - the Experts view (#5723): the rank the adaptive expert cache learned for every (layer, expert) pair, from the
// profile the engine saves when its config has "expert_profile_save" (GET api/expert-heat, serve/expert_heat.py). The tab shows only
// then. Rows are layers, columns experts; darker means a higher rank (used more). The file holds ranks only, so the view claims nothing
// about which experts are in VRAM right now. It asks again every 30 s while the tab is open and the page is visible, and the server
// reads the file only when it changed. English only, like Monitor.
"use strict";
(function (root) {
  const POLL_MS = 30000, ROW_PX = 8;
  let deps = {headers: () => ({})}, timer = null, heat = null, view = null;

  function init(d) { deps = {...deps, ...d}; }
  async function fetchHeat() {
    const r = await fetch("api/expert-heat", {headers: deps.headers()});
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }
  // whether the tab has anything to show: the model saves its expert profile
  async function probe() {
    try { const h = await fetchHeat(); return !!(h && h.configured); } catch (e) { return false; }
  }
  // 1.0 for the first ranked pair down to 0.08 for the last; 0 for a pair the file does not rank
  function shade(rank, ranked) {
    if (!(rank >= 0) || !(ranked > 0)) return 0;
    return 1 - 0.92 * (ranked > 1 ? rank / (ranked - 1) : 0);
  }
  // the (layer, expert) under a point of a grid w x h drawn for layers x experts
  function cellAt(x, y, w, h, layers, experts) {
    if (x < 0 || y < 0 || x >= w || y >= h) return null;
    return {layer: Math.min(layers - 1, Math.floor(y / (h / layers))), expert: Math.min(experts - 1, Math.floor(x / (w / experts)))};
  }
  function savedLine(h, timeStr) {
    const every = h.every_min;
    if (h.saved_at == null) return every > 0 ? `It saves every ${every} minutes and when it stops.` : "";
    const at = timeStr(h.saved_at * 1000);
    return every > 0 ? `Saved by the engine at ${at}. It saves every ${every} minutes and when it stops.` : `Saved by the engine at ${at}.`;
  }
  function tipFor(h, cell) {
    if (!cell) return "";
    const r = h.rank[cell.layer * h.experts + cell.expert];
    return r >= 0 ? `Layer ${cell.layer}, expert ${cell.expert}: rank ${r + 1} of ${h.ranked}` : "";
  }

  function draw() {
    const canvas = view && view.querySelector("#experts-grid");
    if (!canvas || !heat || !heat.rank) { if (canvas) canvas.hidden = true; return; }
    canvas.hidden = false;
    const w = Math.max(200, canvas.parentElement.clientWidth), h = heat.layers * ROW_PX, dpr = root.devicePixelRatio || 1;
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const color = getComputedStyle(document.documentElement).getPropertyValue("--st-accent").trim() || "#4f6bed";
    const cw = w / heat.experts, ch = h / heat.layers;
    ctx.fillStyle = color;
    for (let l = 0; l < heat.layers; l++) {
      for (let e = 0; e < heat.experts; e++) {
        const a = shade(heat.rank[l * heat.experts + e], heat.ranked);
        if (!a) continue;
        ctx.globalAlpha = a;
        ctx.fillRect(e * cw, l * ch, Math.max(1, cw), Math.max(1, ch - 1));
      }
    }
    ctx.globalAlpha = 1;
  }
  function render() {
    if (!view) return;
    view.querySelector("#experts-saved").textContent = heat ? savedLine(heat, deps.timeStr || ((t) => new Date(t).toLocaleTimeString())) : "";
    view.querySelector("#experts-error").textContent = heat && heat.error ? heat.error : "";
    draw();
  }
  async function refresh() {
    try { heat = await fetchHeat(); } catch (e) { heat = {error: String(e.message || e)}; }
    render();
  }
  function start(el) {
    view = el;
    if (!view.dataset.wired) {
      view.dataset.wired = "1";
      const canvas = view.querySelector("#experts-grid"), tip = view.querySelector("#experts-tip");
      canvas.addEventListener("mousemove", (ev) => {
        if (!heat || !heat.rank) return;
        const box = canvas.getBoundingClientRect();
        tip.textContent = tipFor(heat, cellAt(ev.clientX - box.left, ev.clientY - box.top, box.width, box.height, heat.layers, heat.experts));
      });
      canvas.addEventListener("mouseleave", () => { tip.textContent = ""; });
      root.addEventListener("resize", () => { if (timer) draw(); });
    }
    stop();
    refresh();
    timer = setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
  }
  function stop() { if (timer) clearInterval(timer); timer = null; }

  const api = {init, probe, start, stop, shade, cellAt, savedLine, tipFor, POLL_MS};
  root.StrataExperts = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
