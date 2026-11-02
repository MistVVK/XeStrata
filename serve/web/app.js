// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
/* global tr, StrataI18n, StrataMath, StrataPreview, node, panel, msgActions, saveChat, stillWriting, initChats, setChatList,
          clearUndo, syncActions, closePanel, renderSidebar, renderParent, renderUndoBar, setSidebar, sidebarOpen, chatId,
          StrataSegments */
// serve/web/i18n.js, math.js, preview.js, segments.js and chats-ui.js, loaded before this file
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory in binary GB

const store = {
  get(k, d) { try { const v = localStorage.getItem("xestrata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("xestrata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
  del(k) { try { localStorage.removeItem("xestrata." + k); } catch (e) { /* ignore */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", tr("toast.copied", "Copied to clipboard"), "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("xestrata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
  refreshPreviews();                                  // the preview frames carry the page's colours
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("xestrata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
let noteShown = false;
function showTab(name) {
  tab = ["chat", "images", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") {
    $("img-prompt").focus();
    if (health.image_note && !noteShown) {   // once a page load: setup lowered the image model's size for the RAM
      noteShown = true;
      toast("info", tr("img.noteToast", "Image model size lowered"), health.image_note, 12000);
    }
  }
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") { loadConfig(); loadUpdate(); }
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
let apiKey = store.get("apikey", "");
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = apiKey;
function saveApiKey(notify = false) {
  apiKey = $("api-key").value.trim();                // usable even when browser storage is blocked or full
  const saved = store.set("apikey", apiKey);
  if (notify) {
    if (saved) toast("success", tr("about.keySaved", "API key saved"), tr("about.keySavedText", "Kept in this browser only."));
    else toast("warn", tr("about.keyPageOnly", "API key used for this page only"),
               tr("about.keyPageOnlyText", "Browser storage is unavailable. Enter it again after reloading."));
  }
}
$("api-key").oninput = () => saveApiKey();
$("api-key").onchange = () => saveApiKey(true);

let health = {model: "strata", images: false, max_context: 0};
// the texts that depend on what this server offers: the empty chat's line and the attach button's hint
function renderEmptyChat() {
  $("attach-btn").title = health.images ? tr("composer.attachPictures", "Attach a text file or a picture (or drop it here)")
                                        : tr("composer.attach", "Attach a text file (or drop it here)");
  $("chat-empty-sub").textContent = tr("chat.emptyModel", "{model} runs on this PC. Nothing leaves it.", {model: health.model});
}
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    renderEmptyChat();
    $("tab-btn-images").hidden = !health.image_generation;
    $("img-note").hidden = !health.image_note;
    $("img-note").textContent = health.image_note || "";
    imgModels();
    thinkingButtons();
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ the model menu
// A host that swaps models on one GPU (config model_switcher): this page offers the ones served here. Switching
// stops THIS server and starts the other on the same port, so the page waits for /health to name the new model.
let switching = null;
async function loadSwitcher() {
  let sw;
  try { sw = await (await fetch("switcher", {headers: headers()})).json(); } catch (e) { return; }
  const keys = Object.keys(sw.choices || {});
  if (!sw.enabled || sw.error || keys.length < 2) { $("model-switch").hidden = true; return; }
  const sel = $("model-select");
  sel.replaceChildren(...keys.map((k) => {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = sw.choices[k].replace(/ \(Strata SYCL engine\)$/, "");
    o.selected = k === (sw.starting || sw.mode);
    return o;
  }));
  sel.disabled = !!sw.starting || !!switching;
  $("model-switch").hidden = false;
  if (sw.starting && !switching) setPill("reading", tr("switch.loading", "Loading {name}…", {name: sw.choices[sw.starting] || sw.starting}));
}
$("model-select").onchange = async () => {
  const mode = $("model-select").value, name = $("model-select").selectedOptions[0].textContent;
  if (!confirm(tr("switch.confirm", "Load {name}? The current model is unloaded first; it takes about 2 minutes and stops any answer in progress.", {name}))) {
    loadSwitcher();
    return;
  }
  switching = {mode, from: health.model, t0: Date.now()};
  $("model-select").disabled = true;
  setPill("reading", tr("switch.switching", "Switching to {name}… (about 2 minutes)", {name}), true);
  try {
    await fetch("switcher", {method: "POST", headers: headers(true), body: JSON.stringify({mode})});
  } catch (e) { /* this server may already be going down */ }
  const wait = async () => {
    try {
      const h = await (await fetch("health", {cache: "no-store"})).json();
      if (h.model && h.model !== switching.from) { location.reload(); return; }
    } catch (e) { /* down while the models swap */ }
    if (Date.now() - switching.t0 > 6 * 60 * 1000) {
      setPill("error", tr("switch.failed", "{name} did not come up in 6 minutes", {name}), true);
      switching = null;
      return;
    }
    setPill("reading", tr("switch.switchingFor", "Switching to {name}… {s} s", {name, s: Math.round((Date.now() - switching.t0) / 1000)}), true);
    setTimeout(wait, 3000);
  };
  setTimeout(wait, 5000);
};

// ------------------------------------------------------------------ Monitor
// label: in the language of the moment (renderMonitor writes it again after a switch)
const METRICS = [
  {key: "speed", label: () => tr("mon.speed", "Speed"), icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: () => tr("mon.gpuLoad", "GPU load"), icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: () => "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: () => tr("mon.gpuTemp", "GPU temp"), icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: () => tr("mon.power", "Power"), icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: () => "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: () => "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: () => tr("mon.diskRead", "Disk read"), icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}<span id="ml-${m.key}">${esc(m.label())}</span></span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

// gaps: a missing sample breaks the line (the decode speed while an image job has the GPU), else it counts as 0
function spark(id, values, max, gaps = false) {
  const svg = $(id);
  const raw = values || [];
  const v = raw.map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  if (gaps && raw.some((x) => x == null)) {
    let line = "", area = "", run = [];
    const flush = () => {
      if (!run.length) return;
      const d = run.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
      line += d;
      area += `${d}L${run[run.length - 1][0].toFixed(2)},32L${run[0][0].toFixed(2)},32Z`;
      run = [];
    };
    pts.forEach((p, i) => { if (raw[i] == null) flush(); else run.push(p); });
    flush();
    svg.querySelector(".line").setAttribute("d", line);
    svg.querySelector(".area").setAttribute("d", area);
    return;
  }
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
// an image job's step for the header and the state card: "step 3/8", or its stage before and after the sampling
function imageStep(img) {
  return img.stage === "sampling" ? tr("img.step", "step {i}/{n}", {i: img.step, n: img.steps}) : img.label;
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let failureSeen = null;   // issue #10: the engine failure already shown as a toast (its time)
let metricsEpoch = 0;
// After Load or Unload returns, a poll that started earlier can still say the opposite. Hold the POST's state until a later
// poll agrees, or until the first poll that started after the POST arrives. (PR #3550 by sakuumomo)
let metricsHold = null;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
// /metrics live.loading: the model is loading again ("after_image": an image job unloaded it), which a request waits for
let modelLoading = null;
function metricsAgree(live, state) {
  const s = (live && live.state) || "";
  if (state === "unloaded") return s === "unloaded";
  return s !== "" && s !== "unloaded";
}
async function poll() {
  const epoch = ++metricsEpoch;
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", tr("pill.keyNeeded", "API key needed"));
      if (!keyWarned) { keyWarned = true; toast("warn", tr("pill.keyNeeded", "API key needed"), tr("toast.keyAbout", "This server needs a key: add it under About > Settings."), 6000); }
    } else if (r.ok) {
      const m = await r.json();
      if (metricsHold && epoch <= metricsHold.after && !metricsAgree(m.live, metricsHold.state)) {
        setTimeout(poll, 1000);
        return;
      }
      if (metricsHold && (metricsAgree(m.live, metricsHold.state) || epoch > metricsHold.after)) metricsHold = null;
      lastMetrics = m;
      metricsFailures = 0;
      render(lastMetrics);
      const loading = (m.live && m.live.loading) || null;
      if (loading !== modelLoading) { modelLoading = loading; if (busy) setBusy(true); }
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", tr("pill.unreachable", "Server not reachable"));
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text, force = false) {
  if (switching && !force) return;   // while the models swap, the menu owns the pill
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
  // a poll re-sets the sentence every second; refit only when its text changed
  if (text !== lastPillText) { lastPillText = text; fitHeader(); }
}
let lastPillText = null;
// The pill's sentence is the first thing the header row can spare when it runs out of room.
function fitHeader() {
  const header = document.querySelector(".st-header");
  if (!header) return;
  header.classList.remove("st-header--clip");
  if (header.scrollWidth > header.clientWidth) header.classList.add("st-header--clip");
}
window.addEventListener("resize", fitHeader);

// ------------------------------------------------------------------ Load / Unload (POST /load, /unload: the header button)
// Metrics stay "unloaded" until the engine is READY, so a turn or an image job on this page has to paint Loading itself.
// modelOp covers the button's own POST, which returns before the next poll can agree.
let modelOp = null;
let imageBusy = false;
function pageBusy() { return busy != null || imageBusy; }
// live.image (the image engine's job, on the metrics) is the job a refresh can see. imageBusy is this page's POST before
// that job is published. A chat phase already on the metrics wins, because Stop still stops that reply.
function imageLive(live) {
  if (live && live.image) return true;
  const state = live && live.state;
  if (state === "reading" || state === "generating") return false;
  return imageBusy;
}
function paintLive() { if (lastMetrics) render(lastMetrics); }
function turnBlocksUnload(live) {
  const state = (live && live.state) || "";
  return pageBusy() || imageLive(live) || state === "reading" || state === "generating" || (live && live.queued > 0);
}
function setModelButton(text, title, disabled) {
  const btn = $("unload-btn");
  btn.textContent = text; btn.title = title; btn.disabled = disabled;
  fitHeader();
}
function modelButton(live) {
  if (modelOp === "load") return setModelButton(tr("model.loading", "Loading…"), tr("model.loadingTitle", "The model is loading"), true);
  if (modelOp === "unload") return setModelButton(tr("model.unloading", "Unloading…"), tr("model.unloadingTitle", "The model is unloading"), true);
  const state = (live && live.state) || "";
  const metricsBusy = state === "reading" || state === "generating" || (live && live.queued > 0);
  const imageNow = imageLive(live);
  const unload = tr("model.unload", "Unload");
  if (state === "unloaded") {
    // an image job runs on its own engine beside the unloaded text model: Load waits until it has finished
    if (imageNow) return setModelButton(tr("model.load", "Load"), tr("model.imageBusy", "An image is generating. It finishes on its own."), true);
    if (pageBusy()) return setModelButton(tr("model.loading", "Loading…"), tr("model.replyLoading", "A reply is already loading the model"), true);
    return setModelButton(tr("model.load", "Load"), tr("model.loadTitle", "Load the model now. A message also loads it."), false);
  }
  if (imageNow) return setModelButton(unload, tr("model.imageBusy", "An image is generating. It finishes on its own."), true);
  if (!state) return setModelButton("…", tr("model.waiting", "Waiting for the server"), true);
  if (metricsBusy || pageBusy()) return setModelButton(unload, tr("model.stopFirst", "Stop the reply, then unload"), true);
  return setModelButton(unload, tr("model.unloadTitle", "Unload the model and free the GPU"), false);
}
$("unload-btn").onclick = async () => {
  if (modelOp) return;
  const live = (lastMetrics && lastMetrics.live) || {};
  const loading = live.state === "unloaded";
  const busyCopy = (sample, stillLoading) => stillLoading
    ? (imageLive(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.replyLoadingDot", "A reply is already loading the model."))
    : (imageLive(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
  if (loading ? pageBusy() || imageLive(live) : turnBlocksUnload(live)) {
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(live, loading));
    return;
  }
  if (!loading && !confirm(tr("model.confirmUnload", "Unload the model and free the GPU? Load, or the next message, brings it back."))) return;
  // confirm() blocks the poll. Yield so a sample that arrived during the dialog is applied. A confirmed Unload stays an unload.
  if (!loading) {
    modelOp = "unload";
    modelButton(live);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const now = (lastMetrics && lastMetrics.live) || {};
  const loadingNow = loading;
  if (loadingNow ? pageBusy() || imageLive(now) : turnBlocksUnload(now)) {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton(now);
    toast("warn", tr("model.busy", "Model is busy"), busyCopy(now, loadingNow));
    return;
  }
  modelOp = loadingNow ? "load" : "unload";
  modelButton(now);
  if (loadingNow) toast("info", tr("model.loadingToast", "Loading the model"), tr("model.fewMinutes", "This can take a few minutes."));
  const remember = (state) => {
    if (!lastMetrics) lastMetrics = {};
    lastMetrics.live = Object.assign({}, lastMetrics.live, {state, queued: 0});
    metricsHold = {state, after: metricsEpoch};
  };
  try {
    const r = await fetch(loadingNow ? "load" : "unload", {method: "POST", headers: headers(true), body: "{}"});
    let body = {};
    try { body = await r.json(); } catch (e) { /* the connection dropped */ }
    const status = body.status || "";
    const msg = (body.error && body.error.message) ||
                (r.status === 401 ? tr("toast.keyAbout2", "API key needed: add it under About > Settings.") : tr("toast.httpAnswer", "The server answered HTTP {code}.", {code: r.status}));
    const failTitle = loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed");
    if (loadingNow) {
      if (r.ok && status === "loaded") {
        remember("idle");
        toast("success", tr("model.loaded", "Model loaded"), tr("model.gpuInUse", "The GPU is in use."));
      } else {
        toast("error", failTitle, msg);
      }
    } else if (status === "unloaded" || status === "not loaded") {
      remember("unloaded");
      toast(status === "unloaded" ? "success" : "info",
            status === "unloaded" ? tr("model.unloaded", "Model unloaded") : tr("model.alreadyUnloaded", "Already unloaded"),
            tr("model.gpuFree", "The GPU is free. Load, or the next message, brings it back."));
    } else if (r.status === 409 || status === "busy") {
      let sample = (lastMetrics && lastMetrics.live) || {};
      try {
        const mr = await fetch("metrics", {headers: headers()});
        if (mr.ok) {
          const m = await mr.json();
          if (m && m.live) { sample = m.live; lastMetrics = m; }
        }
      } catch (e) { /* the 409 body has no phase */ }
      toast("warn", tr("model.busy", "Model is busy"), imageLive(sample)
        ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
    } else if (status === "unsupported") {
      toast("error", tr("model.unsupported", "Unload is not available"), "");
    } else {
      toast("error", failTitle, msg);
    }
  } catch (e) {
    toast("error", loadingNow ? tr("model.loadFailed", "Load failed") : tr("model.unloadFailed", "Unload failed"), tr("model.noAnswer", "The server did not answer."));
  } finally {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton({});
  }
};

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || []).find((r) => r.kind !== "image");   // the chat's last request (image jobs aside)
  // the header pill. modelOp and a local load win over "N queued": with parallel requests, queued stays up through
  // ensure_loaded(), and metrics stay "unloaded" until the engine is READY. An image job runs beside an unloaded text
  // model (it has its own engine), so only a chat turn here counts as a load.
  const localLoad = live.state === "unloaded" && busy != null;
  if (modelOp === "load") {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (modelOp === "unload") {
    setPill("idle", tr("model.unloading", "Unloading…"));
  } else if (live.image) {
    setPill("generating", tr("img.pill", "Image · {step}", {step: imageStep(live.image)}));
  } else if (localLoad) {
    setPill("idle", tr("model.loading", "Loading…"));
  } else if (imageLive(live)) {
    setPill("reading", tr("pill.image", "Generating image…"));
  } else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? tr("pill.readingPct", "Reading prompt · {pct}%", {pct}) : tr("pill.reading", "Reading prompt"));
  } else if (live.state === "generating") {
    setPill("generating", tr("pill.generating", "Generating · {rate} tok/s", {rate: fmt(live.tok_s, 1)}));
  } else if (live.state === "unloaded") {
    setPill("idle", tr("pill.unloaded", "Unloaded"));
  } else {
    setPill("idle", tr("pill.idle", "Idle"));
  }
  modelButton(live);
  if (!modelOp && !localLoad && live.queued > 0) setPill("queued", tr("pill.queued", "{n} queued", {n: live.queued}));
  // issue #10: an engine that stopped in a request says why once (the server starts it again for the next one)
  const fail = m.engine_failure;
  if (fail && fail.at !== failureSeen) {
    if (failureSeen !== null || Date.now() / 1000 - fail.at < 600) toast("error", tr("engine.stopped", "The engine stopped"), fail.message, 15000);
    failureSeen = fail.at;
  }
  // Intel's compiler 2.27 to 2.30: the engine works around a bug of it (INFO igc_workaround); said once a version
  if (eng.igc_workaround && eng.igc_workaround !== "none" && store.get("igc-told", "") !== eng.igc_workaround) {
    store.set("igc-told", eng.igc_workaround);
    toast("warn", tr("igc.toast", "Intel's compiler slows the decode"), tr("igc.toastText", "IGC {version} has a bug XeStrata works around; the decode of short prompts is up to 10% slower. For full speed, use Fedora 44 (its IGC is 2.32 or later), or install with Podman (Fedora 44 inside).", {version: eng.igc_workaround}), 12000,
          {label: tr("show", "Show"), run: () => showTab("about")});
  }
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, fail);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? tr("mon.justNow", "just now") : s < 3600 ? tr("mon.minAgo", "{n} min ago", {n: fmt(s / 60)})
                                                     : tr("mon.hAgo", "{n} h ago", {n: fmt(s / 3600, 1)});
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? tr("mon.ccSum", "{reused} of {requests} requests reused part of their prompt",
                                            {reused: fmt(c.requests_reused), requests: fmt(c.requests)}) : "";
  const share = c.prompt_tokens ? tr("mon.ccShare", " ({pct}% of all prompt tokens)", {pct: fmt((100 * c.reused_tokens) / c.prompt_tokens)}) : "";
  const event = !c.last_event ? null : c.last_event === "parked"
    ? tr("mon.ccParked", "Parked {n} tokens, {when}", {n: fmt(c.last_tokens), when: since(c.last_at)})
    : tr("mon.ccRestored", "Restored {n} tokens, {when}", {n: fmt(c.last_tokens), when: since(c.last_at)});
  facts($("cc-facts"), [
    [tr("mon.ccLast", "Last request"), c.last_prompt != null ? tr("mon.ccLastValue", "{reused} of {prompt} prompt tokens reused",
                                                                  {reused: fmt(c.last_reused || 0), prompt: fmt(c.last_prompt)}) : null],
    [tr("mon.ccSince", "Reused since start"), c.requests ? tr("mon.ccSinceValue", "{n} tokens{share}", {n: fmt(c.reused_tokens), share}) : null],
    [tr("mon.ccParks", "Parked / restored"), c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}` +
      (c.evictions ? tr("mon.ccEvicted", " · {n} evicted", {n: fmt(c.evictions)}) : "") : null],
    [tr("mon.ccSwitch", "Last switch"), c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? tr("mon.ccNoteOn", "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full.")
    : tr("mon.ccNoteOff", "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).");
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? tr("mon.atRate", " at {rate} tok/s", {rate: fmt(read / (t.prompt_ms / 1000))}) : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? tr("mon.atRate", " at {rate} tok/s", {rate: fmt(t.output_tokens / (t.decode_ms / 1000), 1)}) : "";
  return tr("mon.totals", "Since {since}: {requests} requests · {read} prompt tokens read{prefill} ({reused} reused) · {written} written{decode}",
            {since, requests: fmt(t.requests), read: fmt(read), prefill: pSpeed, reused: fmt(t.reused), written: fmt(t.output_tokens), decode: oSpeed});
}
// the server's phase of a turn (serve/server.py), in the language of the moment; another one as the server says it
function phaseText(phase) {
  if (!phase) return tr("mon.generating", "Generating");
  const tool = /^writing a tool call: (.*)$/.exec(phase);
  if (tool) return tr("mon.phaseTool", "Writing a tool call: {name}", {name: tool[1]});
  const known = {thinking: () => tr("mon.phaseThinking", "Thinking"), answering: () => tr("mon.phaseAnswering", "Answering"),
                 "tool call complete": () => tr("mon.phaseToolDone", "Tool call complete"),
                 "reading the prompt": () => tr("pill.reading", "Reading prompt")};
  return known[phase] ? known[phase]() : phase[0].toUpperCase() + phase.slice(1);
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, fail) {
  for (const m of METRICS) $(`ml-${m.key}`).textContent = m.label();
  // model state
  // A local turn keeps metrics on "unloaded" for the whole engine start.
  const imageNow = imageLive(live);
  const starting = modelOp === "load" || (busy != null && live.state === "unloaded");
  const on = !modelOp && live.queued > 0 ? "queued" : live.image ? "generating" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) {
    const lit = (b.dataset.s === on || (!live.image && b.dataset.s === live.state)) && !(starting && b.dataset.s === "unloaded");
    b.classList.toggle("on", lit);
  }
  const prog = $("state-progress");
  let label = tr("mon.waiting", "Waiting for a request"), detail = "", pct = 0;
  if (live.image) {
    const img = live.image;
    label = img.label + (img.images > 1 ? ` · ${tr("img.nOf", "image {i} of {n}", {i: img.image, n: img.images})}` : "");
    delete prog.dataset.tone;
    pct = 100 * (img.progress || 0);
    detail = [tr("img.step", "step {i}/{n}", {i: img.step, n: img.steps}), img.step_s != null ? `${fmt(img.step_s, 2)} s/step` : null,
              `${fmt(img.elapsed_s, 0)} s`, img.size,
              img.refs ? tr("img.refs", {one: "{n} reference", other: "{n} references"}, {n: img.refs}) : null]
      .filter(Boolean).join(" · ");
  } else if (modelOp === "load" || (busy != null && live.state === "unloaded")) {
    label = tr("mon.loading", "Loading");
    delete prog.dataset.tone;
    detail = tr("mon.loadingDetail", "The model is loading.");
  } else if (imageNow) {
    label = tr("img.generating", "Generating image");
    delete prog.dataset.tone;
    detail = tr("model.imageBusy", "An image is generating. It finishes on its own.");
  } else if (modelOp === "unload") {
    label = tr("mon.unloading", "Unloading");
    delete prog.dataset.tone;
    detail = tr("mon.unloadingDetail", "Freeing the GPU.");
  } else if (live.state === "reading") {
    label = tr("pill.reading", "Reading prompt");
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = tr("mon.readProgress", "{read} / {total} tokens · {pct}%", {read: fmt(live.prompt_read), total: fmt(live.prompt_total), pct: fmt(pct)});
    } else {
      detail = tr("chat.tokens", "{n} tokens", {n: fmt(live.prompt_tokens)});
    }
  } else if (live.state === "generating") {
    label = phaseText(live.phase);
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = tr("mon.genProgress", "{n} tokens · {rate} tok/s", {n: fmt(live.generated), rate: fmt(live.tok_s, 1)});
  } else if (live.state === "unloaded") {
    label = tr("pill.unloaded", "Unloaded");
    delete prog.dataset.tone;
    detail = tr("mon.unloadedDetail", "Load brings it back. A message does too.");
  } else if (last) {
    delete prog.dataset.tone;
    detail = last.decode_tok_s ? tr("mon.lastAt", "last: {n} tokens at {rate} tok/s", {n: fmt(last.output_tokens), rate: fmt(last.decode_tok_s, 1)})
                               : tr("mon.last", "last: {n} tokens", {n: fmt(last.output_tokens)});
  }
  if (fail && live.state === "idle") {   // issue #10: the last engine that stopped, until a request runs again
    label = tr("engine.stoppedAt", "Engine stopped at {time}", {time: new Date(fail.at * 1000).toLocaleTimeString()});
    detail = fail.message;
    for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === "error");
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  if (live.image) {   // an image job: its seconds a step, not a decode rate
    setMetric("speed", live.image.step_s == null ? null : fmt(live.image.step_s, 2), "s/step",
              tr("img.pill", "Image · {step}", {step: imageStep(live.image)}));
  } else {
    const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
    setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
              live.state === "generating" ? tr("mon.decodeNow", "Decode now") : last ? tr("mon.decodeLast", "Decode last request") : tr("mon.decode", "Decode"));
  }
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  if (live.image) setMetric("prefill", null, "t/s", tr("img.job", "Image job"));
  else setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
                 live.state === "reading" ? tr("mon.prefillNow", "Prefill now") : live.state === "generating" ? tr("mon.prefillThis", "Prefill this request")
                 : last ? tr("mon.prefillLast", "Prefill last request") : tr("mon.prefill", "Prefill"));
  spark("sp-speed", h.tok_s, undefined, true);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? tr("mon.expertsCached", "{n} experts cached", {n: fmt(eng.expert_slots)}) : "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  // an idle GPU on the xe driver may report its last temperature, not its present one (Battlemage's hwmon): dimmed
  const stale = Boolean(hw.gpu_temp_may_be_stale) && hw.gpu_temp != null;
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            stale ? tr("mon.tempStale", "GPU idle (the reading may be old)")
            : multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`))
            : hw.gpu_mem_temp == null ? "" : `VRAM ${fmt(hw.gpu_mem_temp)} °C`);
  $("mv-temp").classList.toggle("is-stale", stale);
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? tr("mon.powerLimit", "of {n} W limit", {n: fmt(hw.gpu_power_limit)}) : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : tr("mon.toGpu", "to GPU {rate} MB/s", {rate: fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)}) +
            (hw.gpu_pcie_own_gen && gen && hw.gpu_pcie_own_gen < gen ? ` · ${tr("mon.nowGen", "now Gen{n}", {n: hw.gpu_pcie_own_gen})}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", !st.threads ? ""
            : st.cores ? tr("mon.coresThreads", "{cores} cores · {threads} threads", {cores: st.cores, threads: st.threads})
            : tr("mon.threads", "{threads} threads", {threads: st.threads}));
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : tr("mon.needsPsutil", "needs psutil (setup installs it)"));
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : tr("mon.diskWrite", "write {rate} MB/s", {rate: fmt(hw.disk_write_mb, 1)}));
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">${esc(tr("mon.noRequests", "No requests yet"))}</td></tr>`;
  } else {
    const badge = {stop: ["", tr("tool.done", "Done")], length: ["", tr("drawer.maxTokens", "Max tokens")],
                   cancel: ["st-badge--queued", tr("chat.stopped", "Stopped")], disconnect: ["st-badge--queued", tr("mon.closed", "Closed")],
                   error: ["st-badge--error", tr("tool.error", "Error")], timeout: ["st-badge--queued", tr("mon.timeout", "Timeout")]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      if (r.kind === "image") {   // an image job: its size, references and stages in place of the token columns
        const what = [r.edit ? tr("img.rowEdit", "Edit {size}", {size: r.size}) : tr("img.rowImage", "Image {size}", {size: r.size}),
                      r.n > 1 ? tr("img.rowCount", {one: "{n} image", other: "{n} images"}, {n: r.n}) : null,
                      r.refs ? tr("img.refs", {one: "{n} reference", other: "{n} references"}, {n: r.refs}) : null].filter(Boolean).join(" · ");
        const stages = r.error ? r.error
          : [r.text_encode_ms != null ? tr("img.stageText", "text {s} s", {s: fmt(r.text_encode_ms / 1000, 1)}) : null,
             r.sampling_s != null ? tr("img.stageSampling", "sampling {s} s", {s: fmt(r.sampling_s, 1)}) : null,
             r.vae_decode_s != null ? tr("img.stageVae", "VAE {s} s", {s: fmt(r.vae_decode_s, 1)}) : null].filter(Boolean).join(" · ");
        return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span> <span class="st-badge st-badge--reading">${esc(tr("img.badge", "Image"))}</span></td>
          <td colspan="6" class="req-image${r.error ? " req-image--error" : ""}">${esc(what)}${stages ? ` <span class="muted">· ${esc(stages)}</span>` : ""}</td>
          <td class="num">${fmt(r.seconds != null ? r.seconds : r.duration_s, 1)} s</td></tr>`;
      }
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="${esc(r.projection ? tr("mon.projOn", "experimental speed projection on") : tr("mon.projOff", "experimental speed projection off"))}">${r.projection ? "ESP" : esc(tr("mon.stock", "stock"))}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="${esc(tr("mon.pcieShare", "routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed"))}">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "") +
        // the experts below VRAM: taken from the RAM copy and read from the model files (the SSD), as the engine counts them
        ([r.ram_blobs ? `RAM ${fmt(r.ram_blobs)}` : null,
          r.file_blobs ? `SSD ${fmt(r.file_blobs)}${r.file_mb ? ` (${fmt(r.file_mb / 1000, 1)} GB)` : ""}` : null]
          .filter(Boolean).map((t) => ` <span class="muted" title="${esc(tr("mon.belowVram", "experts taken from the RAM copy and read from the model files"))}">${t}</span>`).join(""));
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? tr("mon.showFewer", "Show fewer") : tr("mon.showAll", "Show all ({n})", {n: kept});
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="${esc(tr("chat.copy", "Copy"))}">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  const vars = {a, b, single: single ? tr("about.projSingle", " (layer {n}'s direction)", {n: single.replace("single", "")}) : ""};
  return mode === "project"
    ? tr("about.projProject", "Projection control vector on layers {a}–{b}{single}. Per chat in Sampling. Its package describes the vector as a refusal-direction projection; measure the speed yourself", vars)
    : tr("about.projAdd", "Additive control vector on layers {a}–{b}{single}. Per chat in Sampling. Its package describes the vector as a refusal-direction projection; measure the speed yourself", vars);
}
// the picture reader that serves: where it runs and how many tokens a picture becomes (an older server: on or off)
function visionText(eng) {
  const v = eng.vision;
  if (v === undefined) return eng.images ? tr("about.on", "on") : tr("img.off", "off");
  if (!v) return tr("img.off", "off");
  if (v.error) return tr("about.visionOff", "off: {error}", {error: v.error});
  const vars = {n: fmt(v.max_tokens), fallback: v.fallback ? tr("about.visionFallback", " (the configured encoder did not start: {why})", {why: v.fallback}) : ""};
  return /GPU/.test(v.encoder || "") ? tr("about.visionGpu", "on the GPU, up to {n} tokens a picture{fallback}", vars)
                                     : tr("about.visionCpu", "on the CPU, up to {n} tokens a picture{fallback}", vars);
}
// one row a model, the default first: its quantizations, steps and whether it edits
function imageModelRows(models) {
  if (!Array.isArray(models)) return [];
  const title = tr("about.imageGen", "Image generation");
  if (!models.length) return [[title, tr("img.off", "off")]];
  return models.map((m, i) => [i ? "" : title,
    `${m.id}${i ? "" : ` ${tr("drawer.thinkDefault", "(default)")}`}: ${[m.quant ? `DiT ${m.quant}` : null,
      m.text_quant ? tr("about.textEncoder", "text encoder {quant}", {quant: m.text_quant}) : null,
      m.steps ? tr("about.steps", "{n} steps", {n: m.steps}) : null, m.edits ? tr("about.edits", "edits images") : null].filter(Boolean).join(", ")}`]);
}
function renderAbout(eng, hw, st) {
  const kv = {int8: tr("about.kv8", "8-bit"), q4_0: tr("about.kv4", "4-bit (Hadamard-rotated)"), fp16: tr("about.kv16", "16-bit")}[eng.kv] || eng.kv;
  const drafts = Math.max(0, (eng.mtp_max || eng.spec) - 1);
  facts($("facts-engine"), [
    [tr("switch.model", "Model"), eng.model],
    [tr("about.engine", "Engine"), eng.version ? `v${eng.version}` : tr("about.fromSource", "built from source")],
    [tr("about.context", "Context"), eng.max_context ? tr("chat.tokens", "{n} tokens", {n: fmt(eng.max_context)}) : null],
    [tr("about.kv", "KV cache"), !kv ? null : eng.kv_resident
      ? tr("about.kvStreamed", "{kv}, streamed: {n} positions per layer in VRAM, the rest in RAM", {kv, n: fmt(eng.kv_resident)})
      : tr("about.kvAll", "{kv}, all in VRAM", {kv})],
    [tr("mon.expertsVram", "Experts in VRAM"), eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    [tr("about.spec", "Speculation"), !eng.spec ? null : eng.lookup ? tr("about.specLookup", "MTP drafts up to {n} tokens, prompt lookup on", {n: drafts})
                                                                     : tr("about.specMtp", "MTP drafts up to {n} tokens", {n: drafts})],
    [tr("about.vision", "Reading images"), visionText(eng)],
    ...imageModelRows(eng.image_models),
    [tr("about.projection", "Experimental speed projection"), projectionText(eng.cvec)],
    [tr("igc.label", "Intel compiler"), eng.igc_workaround && eng.igc_workaround !== "none" ? tr("igc.about", "IGC {version} has a bug (2.27 to 2.30 miscompile some reads for Xe2 and later GPUs); XeStrata works around it, the decode of short prompts up to 10% slower. For full speed, use Fedora 44 (its IGC is 2.32 or later), or install with Podman (Fedora 44 inside).", {version: eng.igc_workaround}) : null],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : tr("about.gpuUnreadable", "not readable (xe driver)")],
    ["CPU", !st.cpu_name ? null : st.threads ? tr("about.cpuThreads", "{name}, {threads} threads", {name: st.cpu_name, threads: st.threads}) : st.cpu_name],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    [tr("about.openaiUrl", "OpenAI base URL"), `${base}/v1`, true],
    [tr("about.anthropicUrl", "Anthropic base URL"), base, true],
    [tr("about.modelName", "Model name"), eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ a newer release (GET/POST /update)
// The server asks GitHub (at most every six hours, or now from the button); this page never leaves this server.
async function loadUpdate(force = false) {
  let u;
  try {
    const r = await fetch("update", force ? {method: "POST", headers: headers(true), body: "{}"} : {headers: headers()});
    if (!r.ok) return false;
    u = await r.json();
  } catch (e) { return false; /* an older server: no check */ }
  renderUpdate(u);
  if (u.newer && store.get("update-told", "") !== u.tag) {   // once for each release
    store.set("update-told", u.tag);
    toast("info", tr("update.out", "XeStrata {tag} is out", {tag: u.tag}), tr("update.outText", "About > Updates says how to update."), 8000,
          {label: tr("show", "Show"), run: () => showTab("about")});
  }
  return true;
}
function renderUpdate(u) {
  const when = u.checked_at ? new Date(u.checked_at * 1000).toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : null;
  facts($("facts-update"), [
    [tr("update.version", "This version"), `xe${u.current}`],
    [tr("update.latest", "Latest release"), u.tag ? `${u.tag} ${u.newer ? tr("update.newer", "(newer)") : tr("update.isLatest", "(up to date)")}` : null],
    [tr("update.checked", "Checked"), when ? `${when}${u.automatic ? "" : ` ${tr("update.autoOffNote", "(automatic checks off)")}`}`
                                           : (u.automatic ? tr("update.notYet", "not yet") : tr("update.autoOff", "automatic checks off"))],
  ]);
  $("update-dot").hidden = !u.newer;
  $("update-how").hidden = !u.newer;
  if (u.newer) {
    $("update-how").innerHTML = `<a href="${esc(u.url)}" target="_blank" rel="noopener noreferrer">${esc(tr("update.notes", "Release notes of {tag}", {tag: u.tag}))}</a>. ` +
                                esc(tr("update.how", "To update: {how}.", {how: u.how}));
  }
  $("update-msg").textContent = u.error || "";
}
$("update-check").onclick = async () => {
  $("update-check").disabled = true;
  $("update-msg").textContent = tr("update.checking", "Checking…");
  if (!await loadUpdate(true)) $("update-msg").textContent = tr("model.noAnswer", "The server did not answer.");
  $("update-check").disabled = false;
};

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", () => tr("mon.mcpConnected", "Connected")],
                   starting: ["st-badge--reading", () => tr("mon.mcpStarting", "Starting")],
                   failed: ["st-badge--error", () => tr("mon.mcpFailed", "Failed")], stopped: ["st-badge--queued", () => tr("chat.stopped", "Stopped")],
                   idle: ["", () => tr("mon.mcpWaiting", "Waiting")]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? tr("mon.mcpSum", "{tools} tools · {ready} of {servers} servers connected",
                                                {tools: fmt(mcpInfo.tools), ready: ready.length, servers: servers.length}) : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? tr("drawer.mcpTools", "{n} tools from {names}; the model calls them when it decides to",
                                                    {n: fmt(mcpInfo.tools), names: ready.map((s) => s.name).join(", ")})
                                               : tr("drawer.mcpNone", "no server is connected yet (see the Monitor)");
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, label] = MCP_STATE[s.status] || ["", () => s.status];
    const text = label();
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${esc(tr("mon.mcpTools", "{n} tools", {n: fmt(s.tools.length)}))}${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (xestrata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.  A key that can also be set
// for every model (scopes: global) gets a second field for the user's settings (settings.json); the model's field
// then overrides it, and "off" there turns an address off for this model alone.  A secret (deepseek_api_key) is set
// for every model only and is write-only: the server says whether it is set, an empty field leaves it, Clear removes it.
let cfgKeys = [], cfgGlobalFile = "settings.json";
// the keys' help in the language of the moment: the English text is serve/runconfig.py's (a key not here shows the server's)
const CFG_HELP = {
  "sampling.temperature": () => tr("about.cfgTemperature", "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, 0.7 not)"),
  "sampling.top_p": () => tr("about.cfgTopP", "Default top_p for requests that send none"),
  "sampling.top_k": () => tr("about.cfgTopK", "Default top_k for requests that send none (1-64)"),
  "sampling.min_p": () => tr("about.cfgMinP", "Default min_p for requests that send none"),
  reasoning_budget_tokens: () => tr("about.cfgBudget", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
  fit_max_tokens: () => tr("about.cfgFit", "Shorten a max_tokens that does not fit the context (on by default); off: answer 400 like llama.cpp"),
  anthropic_thinking: () => tr("about.cfgAnthropic", "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
  effort_position: () => tr("about.cfgEffort", "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
  script_guard: () => tr("about.cfgScript", "Keep other languages' characters out of the answers of requests that name none: ja, en (English and the like: no CJKV), ko, zh-cn or zh-tw; off: as the model writes"),
  aliases: () => tr("about.cfgAliases", "Other model names the server lists and answers to (comma-separated)"),
  idle_unload_s: () => tr("about.cfgIdle", "Unload the model after this many seconds without requests (0 or empty: never)"),
  lazy_load: () => tr("about.cfgLazy", "Start without loading the model; the first request loads it (text only)"),
  engine_silence_s: () => tr("about.cfgSilence", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
  api_monitor: () => tr("about.cfgMonitor", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
  open_browser: () => tr("about.cfgBrowser", "Open the chat page in the browser when the model is ready"),
  vram_reserve_mib: () => tr("about.cfgReserve", "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
};
function cfgInput(k, id, v, placeholder) {
  if (k.kind === "bool" || k.kind === "enum") {
    const opts = k.kind === "bool" ? [["true", tr("about.on", "on")], ["false", tr("img.off", "off")]] : k.choices.map((x) => [x, x]);
    const cur = v == null ? "" : String(v);
    return `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>${esc(placeholder)}</option>` +
      opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
  }
  if (k.kind === "secret")
    return `<input class="st-input" id="${id}" type="password" value="" placeholder="${esc(k.set ? tr("about.cfgSecretSet", "set (type to replace)") : tr("about.cfgSecretUnset", "not set"))}" autocomplete="off">` +
      (k.set ? ` <button class="st-btn st-btn--secondary" type="button" data-cfg-clear="${esc(k.key)}">${esc(tr("about.cfgClear", "Clear"))}</button>` : "");
  const text = v == null ? "" : v === "" ? tr("img.off", "off") : Array.isArray(v) ? v.join(", ") : String(v);
  return `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
    `value="${esc(text)}" placeholder="${esc(placeholder)}" autocomplete="off">`;
}
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  cfgGlobalFile = c.global_file || "settings.json";
  renderConfig(false);
  $("cfg-card").hidden = false;
}
// the form in the language of the moment; again after a language switch (keep: the values typed and not saved stay)
function renderConfig(keep) {
  const ids = cfgKeys.flatMap((k, i) => [`cfg-${i}`, `cfg-g-${i}`]);
  const typed = keep ? ids.map((id) => ($(id) ? $(id).value : null)) : [];
  const every = tr("about.cfgEvery", "Every model ({file})", {file: cfgGlobalFile});
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const scopes = k.scopes || ["model"], id = `cfg-${i}`, gid = `cfg-g-${i}`, help = CFG_HELP[k.key] ? CFG_HELP[k.key]() : k.help;
    const label = (forId, extra) => `<label for="${forId}" title="${esc(help)}">${esc(help)}<code>${esc(k.key)}${extra}</code></label>`;
    if (!scopes.includes("model")) return label(gid, ` &middot; ${esc(every)}`) + cfgInput(k, gid, null, "");
    const row = (extra) => label(id, extra) + cfgInput(k, id, k.value, scopes.includes("global")
      ? tr("about.cfgAsEvery", "as for every model") : tr("about.cfgDefault", "default"));
    if (!scopes.includes("global")) return row("");
    const inUse = k.effective != null && k.effective !== ""
      ? tr("about.cfgInUse", "in use: {value} ({from})", {value: k.effective,
        from: k.from === "model" ? tr("about.cfgThisModel", "this model") : tr("about.cfgEveryModel", "every model")})
      : tr("about.cfgInUseNone", "in use: none");
    return label(gid, ` &middot; ${esc(every)}`) + cfgInput(k, gid, k.global, tr("about.cfgNone", "none")) +
      row(` &middot; ${esc(tr("about.cfgThisModel", "this model"))} &middot; ${esc(inUse)}`);
  }).join("");
  $("cfg-form").querySelectorAll("[data-cfg-clear]").forEach((b) => b.addEventListener("click", () => saveConfig({[b.dataset.cfgClear]: null}, {})));
  typed.forEach((v, i) => { if (v != null && $(ids[i])) $(ids[i]).value = v; });
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "url") return s.toLowerCase() === "off" ? "" : s;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, secret, or names (the server splits them at commas)
}
async function saveConfig(globalSet, set) {
  if (!Object.keys(set).length && !Object.keys(globalSet).length) { $("cfg-msg").textContent = tr("about.cfgNothing", "Nothing changed."); return; }
  try {
    const saved = [];
    for (const [scope, keys] of [["global", globalSet], ["model", set]]) {
      if (!Object.keys(keys).length) continue;
      const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set: keys, scope})});
      const b = await r.json();
      if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
      if (b.changed.length) saved.push(scope === "global" ? tr("about.cfgSavedEvery", "{keys} for every model", {keys: b.changed.join(", ")})
        : tr("about.cfgSavedModel", "{keys}; the earlier file is {file}.bak", {keys: b.changed.join(", "), file: b.file}));
    }
    $("cfg-msg").textContent = saved.length
      ? tr("about.cfgSaved", "Saved ({keys}). Start the model again to use it.", {keys: saved.join("; ")})
      : tr("about.cfgNothing", "Nothing changed.");
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", tr("about.cfgNotSaved", "Not saved"), String(e.message || e), 6000);
  }
  loadConfig();
}
$("cfg-save").addEventListener("click", () => {
  const set = {}, globalSet = {};
  cfgKeys.forEach((k, i) => {
    const m = $(`cfg-${i}`), g = $(`cfg-g-${i}`);
    if (m) {
      const v = configValue(k, m);
      const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
      if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
    }
    if (g) {
      const gv = configValue(k, g);
      if (k.kind === "secret") { if (gv != null) globalSet[k.key] = gv; }   // empty: left as it is
      else if (JSON.stringify(gv) !== JSON.stringify(k.global ?? null)) globalSet[k.key] = gv;
    }
  });
  saveConfig(globalSet, set);
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  // the NUL around a stashed code span is the marker set above
  // eslint-disable-next-line no-control-regex
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
// An ```svg or ```html block that is complete gets a Preview button (serve/web/preview.js): the block's own code in a sandboxed
// frame, built when it is first opened and again when the theme changes. The state lives on the box (data-preview), so it
// resets if the answer is rebuilt. A page (html) can also be run WITH scripts, after a warning, one block at a time.
function pageTheme() {
  const root = document.documentElement;
  return {dark: root.dataset.theme === "dark", ink: getComputedStyle(root).getPropertyValue("--st-ink").trim()};
}
function fillPreview(fr, scripts, blank) {
  const box = fr.closest(".st-code");
  const doc = blank ? null : StrataPreview.previewDoc(fr.dataset.lang, box.querySelector("pre").textContent, scripts, pageTheme(), window);
  const nf = StrataPreview.makeFrame(doc, {scripts, title: fr.title, lang: fr.dataset.lang});
  fr.replaceWith(nf);
  box.classList.toggle("st-code--run", !!(doc && scripts));
  const rb = box.querySelector("[data-preview-scripts]");
  if (rb) {
    rb.setAttribute("aria-pressed", String(!!(doc && scripts)));
    rb.setAttribute("aria-label", doc && scripts ? tr("preview.stopScripts", "Stop scripts") : tr("preview.runScripts", "Run scripts"));
  }
}
function refreshPreviews() { for (const fr of document.querySelectorAll("iframe[data-opened]")) fillPreview(fr, fr.dataset.scripts === "1"); }
function codeBlock(lang, code, complete) {
  const ok = complete && StrataPreview.canPreview(lang || "", code);
  const kind = ok ? StrataPreview.kindOf(lang) : null;
  const label = kind === "svg" ? tr("preview.showSvg", "Preview the picture") : tr("preview.showPage", "Preview the page");
  const head = `<div class="st-code__head"><span>${esc(lang || tr("code.label", "code"))}</span><span class="st-code__btns">` +
    (kind === "page" ? `<button class="st-btn st-btn--icon" data-preview-scripts aria-label="${esc(tr("preview.runScripts", "Run scripts"))}" aria-pressed="false" title="${esc(tr("preview.runScripts", "Run scripts"))}">${icon("warning")}</button>` : "") +
    (ok ? `<button class="st-btn st-btn--icon" data-preview-toggle data-preview-label="${esc(label)}" aria-label="${esc(label)}" title="${esc(label)}" aria-pressed="false">${icon("image")}</button>` : "") +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="${esc(tr("code.copy", "Copy code"))}">${icon("copy")}</button></span></div>`;
  const prev = ok ? `<div class="st-code__preview"><iframe sandbox="" title="${esc(kind === "svg" ? tr("preview.titleSvg", "Picture preview") : tr("preview.titlePage", "Page preview"))}" data-lang="${esc(lang)}"></iframe></div>` : "";
  return `<div class="st-code">${head}<pre><code>${esc(code)}</code></pre>${prev}</div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest, false); break; }  // still streaming: no preview of a half-written block
    html += codeBlock(m[2].trim(), rest.slice(0, end.index), true);
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "", temperature: 1.0, top_p: 0.95, top_k: 20, min_p: "", max: "", seed: "", show: true, esp: true,
                  mcp: true, math: true, chatList: true, chatEdit: true, whileBusy: "wait", segments: true, segmentLimit: "",
                  script: "", scriptContext: true};
const OLD_DEFAULT_TEMPERATURE = 0.6;
// the script guard (serve/script_guard.py): "" follows the page's language until the user picks one - its own mode for
// Japanese, Korean and Chinese (the page's Chinese is Simplified), off for Vietnamese, en for every other
const scriptChoice = (s) => s.script || ({ja: "ja", ko: "ko", zh: "zh-cn", vi: "off"})[StrataI18n.lang] || "en";
function scriptGuard(s) {
  const lang = scriptChoice(s);
  return lang === "off" || s.scriptContext !== false ? lang : {lang, allow_context: false};
}
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
// before the model's own levels (thinking_v 2) "high" was every model's default and saved as it: the default again
if (settings.thinking_v !== 2 && settings.thinking === "high") settings.thinking = "";
// the math, the saved-chats list and the edit controls were off by default before defaults_v 2, and saved so: on once
if (settings.defaults_v !== 2) settings.math = settings.chatList = settings.chatEdit = true;
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);   // the open chat's messages (chats-ui.js: with the saved-chats list on, also that chat's own)
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg, cut, cutAfterTool}
let pending = null;                   // a message parked while an answer is written: {text, images, files, mode}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user" && m.segment && m.segment.k > 1) {
    const d = document.createElement("div");
    d.className = "segment-divider";
    d.textContent = tr("chat.segment", "Segment {k}: the model reads the earlier ones back when it needs them", {k: m.segment.k});
    el.appendChild(d);
  }
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || tr("chat.image", "image"); wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || tr("chat.image", "image")); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `${tr("chat.you", "You")} · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="${esc(tr("chat.copyAnswer", "Copy the answer"))}" title="${esc(tr("chat.copy", "Copy"))}">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  const actions = msgActions(m, i);                 // edit / regenerate / rewind (chats-ui.js), when switched on
  if (actions) el.appendChild(actions);
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", () => tr("tool.writing", "Writing")], running: ["st-badge--generating", () => tr("tool.running", "Running")],
                    done: ["", () => tr("tool.done", "Done")], error: ["st-badge--error", () => tr("tool.error", "Error")],
                    skipped: ["st-badge--queued", () => tr("tool.skipped", "Not run")]};
function toolHtml(t, k) {
  const [cls, labelOf] = TOOL_STATE[t.state] || ["", () => t.state];
  const label = labelOf();
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">${esc(tr("tool.arguments", "Arguments"))}</div><pre class="tool-call__pre">${esc(args || tr("tool.beingWritten", "(being written)"))}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${esc(t.ok ? tr("tool.result", "Result") : tr("tool.error", "Error"))}${t.chars ? ` · ${esc(tr("tool.chars", "{n} characters", {n: fmt(t.chars)}))}` : ""}` +
              `${t.truncated ? esc(tr("tool.cut", ", cut for the model")) : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>` +
    // the images an image tool made, under its block (the server keeps them: /v1/images/files/). Their addresses are
    // strings; an MCP server's picture ({url: "tool-media/..."}, mcp.image_results) is drawn by toolImages below
    imageToolHtml(t, k);
}
// a generate_image call with a layout (MOSS-Image's boxes) also gets a button that opens it in the Images tab
function imageToolHtml(t, k) {
  const urls = (t.images || []).filter((u) => typeof u === "string");
  const a = t.arguments, layout = t.tool === "generate_image" && a && (typeof a === "string" ? a.includes('"layout"') : a.layout);
  const open = layout && t.state === "done" ? `<button type="button" class="st-btn" data-tool-layout="${k}">${esc(tr("img.layoutToImages", "Open the layout in Images"))}</button>` : "";
  return urls.length || open ? `<div class="msg-images tool-images">${urls.map((u) =>
    `<a href="${esc(u.replace(/^\//, ""))}" target="_blank" rel="noopener"><img src="${esc(u.replace(/^\//, ""))}" alt="${esc(t.tool || "image")}"></a>`).join("")}${open}</div>` : "";
}
// Raster tool results are same-origin cached files; credentials stay in request headers.
const mediaBlobs = new Map();
let mediaEpoch = 0;
function clearToolMedia() {
  mediaEpoch++;
  for (const pending of mediaBlobs.values()) pending.then((url) => URL.revokeObjectURL(url), () => {});
  mediaBlobs.clear();
}
window.addEventListener("pagehide", clearToolMedia);
window.addEventListener("pageshow", (event) => { if (event.persisted) hydrateToolImages(document); });
function safeToolMedia(url) { return typeof url === "string" && /^tool-media\/[0-9a-f]{64}\.(png|jpg|webp)$/.test(url); }
function toolImages(t) {
  return (t.images || []).filter((im) => safeToolMedia(im.url)).map((im) =>
    `<figure class="tool-image"><img data-tool-image="${esc(im.url)}" alt="Tool image" loading="lazy">` +
    `<figcaption><a class="st-btn" data-tool-download="${esc(im.url)}" download="${esc(im.name || "generated-image.png")}">Download image</a></figcaption></figure>`).join("");
}
function hydrateToolImages(root) {
  root.querySelectorAll("[data-tool-image]").forEach((img) => {
    const url = img.dataset.toolImage, epoch = mediaEpoch;
    if (!safeToolMedia(url)) return;
    if (!mediaBlobs.has(url)) mediaBlobs.set(url, fetch(url, {headers: headers()}).then((r) => {
      if (!r.ok) throw new Error("Image unavailable");
      return r.blob();
    }).then((b) => URL.createObjectURL(b)).catch((e) => {if (epoch === mediaEpoch) mediaBlobs.delete(url); throw e;}));
    mediaBlobs.get(url).then((blob) => {
      if (epoch !== mediaEpoch || !img.isConnected) return;
      img.src = blob;
      const a = img.closest("figure").querySelector("[data-tool-download]");
      a.href = blob;
    }).catch(() => { img.alt = "Image unavailable; try refreshing"; });
  });
}
// the answer's text with the tool blocks where the model called them
// math (opt-in, serve/web/math.js): formulas become placeholders before the Markdown runs and are drawn after it
function answerMarkdown(text) {
  return settings.math ? StrataMath.markdown(text, markdown) : markdown(text);
}
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return answerMarkdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += answerMarkdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k) + toolImages(t);
  });
  return html + answerMarkdown(m.text.slice(pos)) + sourcesHtml(m.tools);
}
// the pages the web search found for this answer, once each, as links
function sourcesHtml(tools) {
  const seen = new Set(), links = [];
  for (const t of tools) for (const s of t.sources || []) {
    if (seen.has(s.url) || !/^https?:\/\//.test(s.url)) continue;
    seen.add(s.url);
    links.push(`<li><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title || s.url)}</a></li>`);
  }
  return links.length ? `<div class="msg-sources"><span class="muted small">${esc(tr("chatSearch.sources", "Sources"))}</span><ol>${links.join("")}</ol></div>` : "";
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms, images: x.images || [], vision: x.vision === true,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
    if (x.sources) t.sources = x.sources;              // a web search's results: listed under the answer
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? tr("think.now", "Thinking…") :
      m.thinkSecs != null ? tr("think.for", "Thought for {s} s", {s: fmt(m.thinkSecs, 1)}) : tr("think.done", "Thoughts");
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (settings.math) StrataMath.hydrate(bubble);
    hydrateToolImages(bubble);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? tr("chat.stopped", "Stopped") : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  renderSidebar();
  renderParent();
  renderUndoBar();
  scrollDown(true);
}
// Follow the answer while the reader has not left (#4014). Asking "am I within 120 px of the bottom?" on every frame pulled
// the view back after every wheel notch (a notch is about 100 px), so scrolling up during a long answer only worked by
// dragging the scrollbar. Now the reader's own input lets go (wheel up, a finger or a key moving up, the scrollbar), and a
// scroll back to the bottom follows again.
let stickToBottom = true, lastReaderInput = 0;
const FOLLOW_SLACK = 24, FAR_AWAY = 400, INPUT_WINDOW_MS = 400;
function followAfterScroll(distance, wasFollowing, msSinceInput) {
  if (distance <= FOLLOW_SLACK) return true;
  if (msSinceInput <= INPUT_WINDOW_MS || distance > FAR_AWAY) return false;
  return wasFollowing;
}
function scrollDown(force) {
  const s = $("chat-scroll");
  if (force) stickToBottom = true;
  if (stickToBottom) s.scrollTop = s.scrollHeight;
}
(function watchReader() {
  const s = $("chat-scroll");
  const touch = () => { lastReaderInput = Date.now(); };
  s.addEventListener("wheel", (e) => { touch(); if (e.deltaY < 0 && s.scrollTop > 0) stickToBottom = false; }, {passive: true});
  s.addEventListener("touchmove", touch, {passive: true});
  s.addEventListener("pointerdown", touch, {passive: true});
  s.addEventListener("keydown", (e) => {
    touch();
    if (["PageUp", "ArrowUp", "Home"].includes(e.key) && e.target === s && s.scrollTop > 0) stickToBottom = false;
  });
  s.addEventListener("scroll", () => {
    stickToBottom = followAfterScroll(s.scrollHeight - s.scrollTop - s.clientHeight, stickToBottom, Date.now() - lastReaderInput);
  }, {passive: true});
})();

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  // an svg/html block: flip between its code and its sandboxed preview. Closing the preview also turns scripts off.
  const pv = e.target.closest("[data-preview-toggle]");
  if (pv) {
    const box = pv.closest(".st-code");
    const on = box.dataset.preview === "true";
    if (on) delete box.dataset.preview; else box.dataset.preview = "true";
    pv.setAttribute("aria-pressed", String(!on));
    pv.setAttribute("aria-label", on ? pv.dataset.previewLabel : tr("preview.showCode", "Show the code"));
    const fr = box.querySelector("iframe");
    if (on) fillPreview(fr, false, true);
    else if (!fr.dataset.opened) fillPreview(fr, false);
    return;
  }
  const rs = e.target.closest("[data-preview-scripts]");
  if (rs) {
    const fr = rs.closest(".st-code").querySelector("iframe");
    const run = fr.dataset.scripts !== "1";
    if (run && !confirm(tr("preview.confirm", "Run this code with scripts?\n\nIt runs in your browser, cut off from this chat and from the network, but it can still freeze the tab, and it is code the model wrote.\n\nOnly this block, and only until you close the preview."))) return;
    fillPreview(fr, run);
    return;
  }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  const tl = e.target.closest("[data-tool-layout]");
  if (tl) {
    const m = messages[+tl.closest(".st-msg").dataset.i], t = m && m.tools && m.tools[+tl.dataset.toolLayout];
    if (t) imgLayoutFromTool(t);
    return;
  }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

// The messages of the current segment as the API takes them (segments.js): its system note and the exchanges it carries,
// then its messages, each as it was sent before, so the prompt only grows. `partial`: the answer being written goes too,
// as far as it got (a round that called the page's own tools goes on).
function apiMessages(partial) {
  const start = StrataSegments.segmentStart(messages), seg = messages[start] && messages[start].segment;
  const out = seg ? [{role: "system", content: seg.system}, ...seg.carry] : [];
  for (const m of messages.slice(start)) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m) + (m.ctx ? `\n\n${m.ctx}` : "");
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    } else if (partial) {
      out.push(...assistantMessages(m, true));
    }
  }
  return out;
}
// An answer that used tools goes back as the model wrote it: per round the text before the calls, the calls and their
// results (as the model read them), then the rest - so the next question can build on what the tools found. With its
// thinking where the answer keeps it (`keep`, settleTurn; always within the turn being written): then the prompt is the
// one the model wrote, and the conversation cache serves it. An answer written since segments (`v` 2) goes back as it
// streamed, less the blank lines the page put between rounds; an older one, trimmed as before.
function assistantMessages(m, partial) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  const keep = partial || !!m.keep, exact = m.v === 2, reasoning = m.reasoning || "";
  const words = (s) => (exact ? s : s.trim());
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (reasoning && (keep || !m.text)) msg.reasoning_content = reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0, rpos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r), endCall = calls[calls.length - 1];
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    const rat = Math.min(Math.max(rpos, calls[0].rat || 0), reasoning.length);
    const msg = {role: "assistant", content: words(m.text.slice(pos, at)),
                 tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))};
    if (keep && rat > rpos) msg.reasoning_content = reasoning.slice(rpos, rat);
    out.push(msg);
    for (const t of calls) {
      const pictures = t.vision ? (t.images || []).filter((im) => safeToolMedia(im.url)) : [];
      const content = pictures.length ? [{type: "text", text: t.result}, ...pictures.map((im) =>
        ({type: "image_url", image_url: {url: "strata-tool-media:" + im.url.split("/").pop()}}))] : t.result;
      out.push({role: "tool", tool_call_id: t.id, content});
    }
    pos = at + (exact && endCall.csep ? 2 : 0);
    rpos = rat + (exact && endCall.rsep ? 2 : 0);
  }
  const rest = words(m.text.slice(pos)), after = reasoning.slice(rpos);
  if (rest || (keep && after)) {
    const msg = {role: "assistant", content: rest};
    if (keep && after) msg.reasoning_content = after;
    out.push(msg);
  }
  return out;
}

// ---- segments (segments.js): one chat that goes on past the context, with no summary
const PAGE_ROUNDS = 8;                 // rounds of the page's own tools in one turn
// the record of the current segment, or null (a chat from before segments: it goes on as it was until a handoff)
function segmentNow() {
  const i = StrataSegments.segmentStart(messages);
  return (messages[i] && messages[i].segment) || null;
}
// the room an answer keeps: what the settings ask for, else a tenth of the context (2K to 16K)
function answerReserve() {
  const ctx = health.max_context || 0;
  return +settings.max > 0 ? +settings.max : Math.min(16384, Math.max(2048, Math.floor(ctx / 10)));
}
// a segment's tokens: the context less the answer's room and a twentieth for the exchanges a new segment carries, or
// the settings' limit if smaller; 0 when the server has not said its context
function segmentLimit() {
  const ctx = health.max_context || 0;
  if (!ctx) return 0;
  const auto = ctx - answerReserve() - Math.floor(ctx / 20);
  return Math.max(1024, +settings.segmentLimit > 0 ? Math.min(+settings.segmentLimit, auto) : auto);
}
function lastUserIndex() {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === "user") return i;
  return -1;
}
function lastAnswerBefore(i) {
  for (let j = i - 1; j >= 0; j--) if (messages[j].role === "assistant") return messages[j];
  return null;
}
// the record that makes user message i start segment k
function newSegment(k, i, limit) {
  const S = StrataSegments, memory = S.memoryAt(messages, i);
  return {k, limit, system: S.systemNote(k, limit) + (memory ? `\n\nYour memory:\n${memory}` : ""),
          carry: k > 1 ? S.carryFrom(messages, i, userText) : [],
          tools: S.TOOLS.map((t) => t.function.name).filter((n) => k > 1 || !n.startsWith("history_"))};
}
// A message sent: a new chat starts segment 1; one that would take the segment past its limit (the last answer's
// prompt and words, and this message at about two characters a token) starts the next. Else its room line.
function placeSegment(i) {
  if (!settings.segments) return;
  const S = StrataSegments, u = messages[i], limit = segmentLimit();
  if (!limit) return;
  const prev = lastAnswerBefore(i), used = (prev && prev.used) || 0;
  if (i === 0) u.segment = newSegment(1, i, limit);
  else if (used && used + Math.ceil(userText(u).length / 2) > limit) u.segment = newSegment(S.segmentAt(messages, i - 1) + 1, i, limit);
  const seg = segmentNow();
  if (seg && !u.segment && used) u.ctx = S.roomLine(used, seg.limit);
}
// a turn the server refused as too long: its message starts the next segment, or, starting one already, carries one
// exchange fewer; false when there is nothing left to take
function handOffAt(i) {
  if (!settings.segments || i < 0) return false;
  const u = messages[i], limit = segmentLimit();
  if (!limit) return false;
  if (!u.segment) {
    u.segment = newSegment(StrataSegments.segmentAt(messages, i - 1) + 1, i, limit);
    delete u.ctx;
    return true;
  }
  if (u.segment.carry.length > 2) { u.segment.carry = u.segment.carry.slice(-2); return true; }
  return false;
}
// what the next turn needs of an answer that ended: the segment's tokens after it, and whether its thinking goes back
// (segments.js keepThinking: its thinking against what would be read again without it, the answer and its tools'
// results at the answer's own characters a token)
function settleTurn(m, usage) {
  m.v = 2;
  m.used = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
  if (!settings.segments) return;
  const thinking = (usage.completion_tokens_details && usage.completion_tokens_details.reasoning_tokens) || 0;
  const chars = (m.reasoning || "").length + (m.text || "").length;
  const perToken = usage.completion_tokens > 0 && chars > 0 ? chars / usage.completion_tokens : 3;
  const results = (m.tools || []).reduce((n, t) => n + (t.result ? t.result.length : 0), 0);
  const seg = segmentNow();
  m.keep = StrataSegments.keepThinking({thinking, after: (usage.completion_tokens - thinking) + results / perToken,
                                        used: m.used, limit: seg ? seg.limit : segmentLimit()});
}
// a chat that goes on for good needs a store the browser does not clear: asked once a visit, said once
async function askPersist() {
  try {
    if (!navigator.storage || !navigator.storage.persist || await navigator.storage.persisted()) return;
    if (!(await navigator.storage.persist()) && !store.get("persistTold", false)) {
      store.set("persistTold", true);
      toast("warn", tr("store.persistTitle", "Chats may be cleared"), tr("store.persistText", "This browser did not promise to keep this page's data. A long chat's earlier segments live there."), 8000);
    }
  } catch (e) { /* no storage manager here */ }
}

function busyHint() {
  const mode = settings.whileBusy;
  return mode === "queue" ? tr("composer.hintQueue", "Ctrl+Enter sends when the answer is done")
       : mode === "interject" ? tr("composer.hintInterject", "Ctrl+Enter stops the answer here and sends now") : "";
}
// While an answer is written, Send is off ("wait", as always) or parks the message ("queue", "interject": Chat settings)
const parksWhileBusy = () => settings.whileBusy === "queue" || settings.whileBusy === "interject";
// what the chat is waiting for while the model loads (after an image job, every image that took the GPU from it)
const loadingHint = () => modelLoading === "after_image"
  ? tr("composer.loadingAfterImage", "The model is loading again after the image (a minute or two); the answer goes on then")
  : tr("composer.loading", "The model is loading (a minute or two); the answer goes on then");
function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = $("image-btn").disabled = (on && !parksWhileBusy()) || !!panel;
  $("input").disabled = !!panel;
  $("composer-hint").textContent = panel ? tr("composer.finishOpen", "Save or cancel what is open above to continue")
                                         : on && modelLoading ? loadingHint()
                                         : on ? busyHint() : tr("composer.sendHint", "Ctrl+Enter: send");
  if (settings.chatEdit) syncActions();                 // the buttons follow the state: none under the answer being written
  $("s-list").disabled = $("s-edit").disabled = !!busy || !!panel;      // they change the messages on the screen
  renderUndoBar();
  renderPending();
}

// A message sent while an answer is being written (#4004). "queue": it waits and goes when the answer is done. "interject": the
// answer stops where it is (what is written stays, marked interrupted) and the message goes at once; if a tool is running, the
// tool finishes first. Only one message waits. Off by default: with "wait" the page does nothing, as before.
function park(p, own) {
  if (!parksWhileBusy()) return;
  if (pending) { toast("warn", tr("pending.one", "A message is already waiting"), tr("pending.oneText", "Take it back or wait until it is sent.")); return; }
  pending = {...p, mode: settings.whileBusy};
  if (own) { attachments = []; renderAttachments(); $("input").value = ""; autosize(); }
  if (pending.mode === "interject") requestCut();
  setBusy(!!busy);
}
function requestCut() {
  if (!busy || busy.cut) return;
  if ((busy.msg.tools || []).some((t) => t.state === "running")) { busy.cutAfterTool = true; return; }   // let the tool finish
  busy.cut = true;
  stopReply();
}
function renderPending() {
  let el = $("pending");
  if (!pending) { if (el) el.remove(); return; }
  if (!el) {
    el = node("div", "pending");
    el.id = "pending";
    el.setAttribute("role", "status");
    el.innerHTML = "<span class=\"pending__text\"></span><span class=\"pending__actions\"><button type=\"button\" class=\"st-btn st-btn--secondary\" data-pending-now></button>" +
      "<button type=\"button\" class=\"st-btn st-btn--secondary\" data-pending-back></button></span>";
    el.querySelector("[data-pending-now]").onclick = pendingNow;
    el.querySelector("[data-pending-back]").onclick = releasePending;
    $("composer").before(el);
  }
  const quote = (pending.text || (pending.files[0] || pending.images[0] || {}).name || "").replace(/\s+/g, " ").slice(0, 70);
  const text = !busy ? tr("pending.ready", "Not sent yet: “{text}”", {text: quote})
    : pending.mode === "queue" ? tr("pending.queued", "Sends when this answer is done: “{text}”", {text: quote})
    : busy.cutAfterTool ? tr("pending.afterTool", "Waiting for the tool to finish, then sending: “{text}”", {text: quote})
    : tr("pending.cutting", "Stopping the answer, then sending: “{text}”", {text: quote});
  el.querySelector(".pending__text").textContent = text;
  const now = el.querySelector("[data-pending-now]");
  now.textContent = busy ? tr("pending.sendNow", "Send now") : tr("pending.send", "Send");
  now.hidden = !!busy && pending.mode === "interject";            // already on its way
  el.querySelector("[data-pending-back]").textContent = tr("pending.takeBack", "Take back");
}
function pendingNow() {
  if (!pending) return;
  if (busy) { pending.mode = "interject"; requestCut(); setBusy(true); return; }
  const p = pending;
  pending = null;
  setBusy(false);
  send(p);
}
// the waiting message goes back into the message box (a chat switch or a new chat must not lose it)
function releasePending() {
  if (!pending) return;
  const p = pending;
  pending = null;
  if (!$("input").value.trim()) $("input").value = p.text;
  else $("input").value = `${p.text}\n\n${$("input").value}`;
  attachments = [...p.images, ...p.files, ...attachments];
  renderAttachments(); autosize();
  setBusy(!!busy);
}
// after an answer: an interrupted one sends an interjection, a finished one sends a queued message
function settlePending(cut, finished) {
  if (!pending) return;
  if ((pending.mode === "interject" && cut) || (pending.mode === "queue" && finished)) {
    const p = pending;
    pending = null;
    setBusy(false);
    send(p);
  }
}

// the message box (or a parked message, `payload`) becomes a user turn and a reply is asked for
// image: the reply must make an image (the image button): strata_images' force
async function send(payload, image = false) {
  const own = !payload;
  const p = payload || {text: $("input").value.trim(), images: attachments.filter((a) => a.kind !== "file"),
                        files: attachments.filter((a) => a.kind === "file"), image};
  if (!p.text && !p.images.length && !p.files.length) return;
  if (panel) return;                                  // something is open above (an edit): finish it first
  if (busy) { park(p, own); return; }
  clearUndo();
  messages.push({role: "user", text: p.text, images: p.images, files: p.files, time: Date.now()});
  placeSegment(messages.length - 1);
  if (own) {
    attachments = [];
    renderAttachments();
    $("input").value = "";
    autosize();
  }
  return generateReply({image: !!p.image});
}
// an answer for the messages as they are: the last one is a user turn
// A streamed reply the server keeps past its connection (strata_reply): its data lines are counted as they are read, so
// that a connection that breaks (not the page's own abort) is read again from where it stopped, a few times, waiting a
// little longer each time.  A connection that ends before [DONE] broke too (a proxy or a network that closed it); an
// error the stream itself reports is not a broken connection.
const REPLY_TRIES = 6;
const replyId = () => [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
const replyError = (message) => Object.assign(new Error(message), {fromStream: true});
async function readReply(r, rid, signal, onData) {
  let seen = 0, ended = false;
  for (let tries = 0; ;) {
    try {
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const {value, done} = await reader.read();
        if (done && (ended || !rid)) return;
        if (done) throw new TypeError("the connection closed before the reply ended");
        buf += dec.decode(value, {stream: true});
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
          seen++;
          const data = line.slice(5).trim();
          if (data === "[DONE]") ended = true;
          onData(data);
        }
      }
    } catch (e) {
      if (e.name === "AbortError" || e.fromStream || !rid) throw e;
      for (;;) {                                               // the connection broke: the rest, from line `seen`
        if (++tries > REPLY_TRIES) throw e;
        await new Promise((ok) => setTimeout(ok, Math.min(1000 * tries, 5000)));
        if (signal.aborted) throw e;
        try {
          r = await fetch(`v1/strata/replies/${rid}?from=${seen}`, {headers: headers(), signal});
          if (r.ok) break;
          if (r.status === 404) throw e;                       // the server no longer has it (restarted)
        } catch (e2) { if (e2 === e || e2.name === "AbortError") throw e2; }
      }
    }
  }
}
// Stop (and an interjection): the server ends the reply too, then this page's connection is closed
function stopReply() {
  if (busy.rid) fetch(`v1/strata/replies/${busy.rid}/cancel`, {method: "POST", headers: headers(true), body: "{}"}).catch(() => {});
  busy.controller.abort();
}
async function generateReply(opts = {}) {
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m, cut: false, cutAfterTool: false};
  setBusy(true);
  paintLive();

  const body = {model: health.model, messages: [], stream: true};
  const effort = thinkingEffort(settings);
  if (effort) body.reasoning_effort = effort;
  body.script_guard = scriptGuard(settings);
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.temperature > 0 && settings.min_p !== "" && settings.min_p != null) body.min_p = +settings.min_p;
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it
  // and make images when the conversation asks for one: by the model the words choose (auto), by one, or none
  const ci = $("chat-images").value;
  if (health.image_generation && ci !== "off") body.strata_images = ci === "auto" || !ci ? true : {model: ci};
  if (health.image_generation && opts.image)                // the image button: by the model chosen, else the words'
    body.strata_images = {...(ci && ci !== "auto" && ci !== "off" ? {model: ci} : {}), force: true};
  if (health.web_search && $("chat-search").value !== "off") body.web_search_options = {};   // the server's search tools

  let firstAt = null, thinkStart = null, usage = null, frame = 0, overflow = false, handoff = false;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  const seg = segmentNow();                                  // null: a chat from before segments, or segments off
  if (seg) {
    body.tools = StrataSegments.TOOLS.filter((t) => seg.tools.includes(t.function.name));
    body.prompt_cache_key = `${chatId || "chat"}-${seg.k}`;  // the same slot again for this chat (BATCHING.md)
  }
  try {
    // a round that called the page's own tools goes on in another request, with their results
    for (let round = 0; ; round++) {
      body.messages = apiMessages(round > 0);
      const own = [];
      // kept past its connection (a tab put to sleep, a network that dropped): the rest is read again below
      busy.rid = health.kept_replies ? replyId() : null;
      if (busy.rid) body.strata_reply = {id: busy.rid};
      const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                     signal: controller.signal});
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
        if (r.status === 401) msg = tr("toast.keyNeeded", "This server needs an API key: add it under About > Settings.");
        overflow = r.status === 400 && /exceeds the context window/.test(msg);
        throw new Error(msg);
      }
      await readReply(r, busy.rid, controller.signal, (data) => {
          if (data === "[DONE]") return;
          let j;
          try { j = JSON.parse(data); } catch (e) { return; }
          if (j.error) throw replyError(j.error.message || tr("toast.engineError", "the engine reported an error"));
          if (j.usage) usage = j.usage;
          if (j.strata_mcp) {
            onTool(m, j.strata_mcp);
            // an interjection waited for a running tool: it has its result now, so the answer can stop
            if (busy.cutAfterTool && !busy.cut && !(m.tools || []).some((t) => t.state === "running")) { busy.cut = true; stopReply(); }
          }
          const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
          const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
          if (d.reasoning_content) {
            if (!firstAt) firstAt = performance.now();
            if (!thinkStart) thinkStart = performance.now();
            if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) { m.reasoning += "\n\n"; lastTool.rsep = true; }
            m.reasoning += d.reasoning_content;
          }
          if (d.content) {
            if (!firstAt) firstAt = performance.now();
            if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
            if (lastTool && m.text && lastTool.at === m.text.length) { m.text += "\n\n"; lastTool.csep = true; }
            m.text += d.content;
          }
          for (const c of d.tool_calls || []) {                // the page's own tools (segments.js): it runs them
            const o = own[c.index] || (own[c.index] = {id: "", name: "", args: ""});
            if (c.id) o.id = c.id;
            if (c.function && c.function.name) o.name = c.function.name;
            if (c.function && c.function.arguments) o.args += c.function.arguments;
          }
          if (!frame) frame = requestAnimationFrame(paint);
      });
      const calls = own.filter((c) => c && StrataSegments.TOOL_NAMES.has(c.name));
      if (!calls.length || busy.cut || round >= PAGE_ROUNDS) break;
      const room = usage ? usage.prompt_tokens + usage.completion_tokens : 0;
      for (const c of calls) {
        let a = {};
        try { a = JSON.parse(c.args || "{}"); } catch (e) { a = {}; }
        const t = {id: c.id, name: c.name, tool: c.name, arguments: a, round: `page-${round}`, at: m.text.length, rat: m.reasoning.length,
                   page: true, ok: true, state: "done"};
        t.result = StrataSegments.runTool(messages, StrataSegments.segmentStart(messages), c.name, a) +
                   (seg ? `\n${StrataSegments.roomLine(room, seg.limit)}` : "");
        m.tools = m.tools || [];
        m.tools.push(t);
      }
      if (!frame) frame = requestAnimationFrame(paint);
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else if (overflow && handOffAt(lastUserIndex())) handoff = true;
    else { m.error = e.message || String(e); toast("error", tr("toast.requestFailed", "The request failed"), m.error, 6000); }
  }
  const cut = busy.cut;                                      // an interjection ended it: kept as written, marked interrupted
  if (cut) m.interrupted = true;
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (usage) settleTurn(m, usage);
  const ended = cut ? tr("chat.interruptedLower", "interrupted") : tr("chat.stoppedLower", "stopped");
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = tr("chat.tokens", "{n} tokens", {n: fmt(n)}) + (secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : "") + (m.stopped ? ` · ${ended}` : "") +
             (projectionLoaded() ? (settings.esp ? ` · ${tr("chat.projectionOn", "projection on")}` : ` · ${tr("chat.projectionOff", "projection off")}`) : "");
  } else if (m.stopped) {
    m.meta = cut ? tr("chat.interrupted", "Interrupted") : tr("chat.stopped", "Stopped");
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${tr("chat.toolCalls", {one: "{n} tool call", other: "{n} tool calls"}, {n: ran})}`;
  if (m.limit) m.meta = `${m.meta || ""} · ${tr("chat.toolLimit", "tool limit reached ({n} rounds, mcp.max_rounds)", {n: m.limit})}`;
  if (seg && m.used) m.meta = `${m.meta || ""} · ${tr("chat.context", "context {n} of {limit}", {n: fmt(m.used), limit: fmt(seg.limit)})}`;
  busy = null;
  setBusy(false);
  if (handoff) {                                     // the turn did not fit: it starts the next segment and runs again
    if (frame) cancelAnimationFrame(frame);
    messages.splice(messages.indexOf(m), 1);
    saveChat();
    renderChat();
    return generateReply(opts);
  }
  paintLive();
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
  settlePending(cut, !m.stopped && !m.error);
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("image-btn").onclick = () => send(null, true);
$("stop-btn").onclick = () => { if (busy) stopReply(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.isComposing) { e.preventDefault(); send(); }   // Enter alone: a new line
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

// one chat, as before: New chat clears it, with an Undo (with the saved-chats list on, its own New chat button does this job)
$("new-btn").onclick = () => {
  if (busy) { stillWriting(); return; }
  if (panel) { closePanel(); }
  clearUndo();
  releasePending();
  if (!messages.length) return;
  const backup = messages;
  clearToolMedia();
  messages = [];
  saveChat();
  renderChat();
  toast("info", tr("chat.newToast", "New chat"), tr("chat.clearedToast", "The last one was cleared."), 6000, {label: tr("undo", "Undo"), run: () => {
    if (busy || messages.length) return;                         // a new question since: the cleared chat is not put back over it
    messages = backup; saveChat(); renderChat();
  }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", tr("export.nothing", "Nothing to save yet")); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", tr("file.picturesOff", "Pictures are off"), tr("file.picturesOffText", "This model was set up for text only.")); continue; }
      if (f.size > 20e6) { toast("warn", tr("file.pictureBig", "Picture too large"), tr("file.pictureBigText", "{name} is over 20 MB.", {name: f.name})); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || tr("file.pasted", "pasted image"), url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", tr("file.notText", "Not a text file"), health.images ? tr("file.notTextPictures", "{name}: attach text files (code, notes, logs, data) or pictures.", {name: f.name})
                                                              : tr("file.notTextOnly", "{name}: attach text files (code, notes, logs, data).", {name: f.name})); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", tr("file.big", "File too large"), tr("file.bigText", "{name} is over 512 KB.", {name: f.name})); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", tr("file.notText", "Not a text file"), tr("file.binaryText", "{name} looks like a binary file.", {name: f.name})); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", tr("file.remove", "Remove"));
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the chat settings drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
const checked = (id) => $(id).getAttribute("aria-checked") === "true";
const segValue = (id) => { const b = [...$(id).children].find((x) => x.getAttribute("aria-checked") === "true"); return b ? b.dataset.v : null; };
function loadDrawer(s = settings) {
  const level = thinkingChosen(s);
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === level));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed; $("s-minp").value = s.min_p ?? "";
  $("s-math").setAttribute("aria-checked", String(!!s.math));
  const script = scriptChoice(s);
  for (const b of $("s-script").children) b.setAttribute("aria-checked", String(b.dataset.v === script));
  $("s-scriptctx").setAttribute("aria-checked", String(s.scriptContext !== false));
  $("script-ctx-row").hidden = script === "off";
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  $("s-list").setAttribute("aria-checked", String(!!s.chatList));
  $("s-edit").setAttribute("aria-checked", String(!!s.chatEdit));
  $("s-seg").setAttribute("aria-checked", String(s.segments !== false));
  $("s-seglimit").value = s.segmentLimit || "";
  $("s-memory").value = StrataSegments.memoryAt(messages);
  for (const b of $("s-busy").children) b.setAttribute("aria-checked", String(b.dataset.v === (s.whileBusy || "wait")));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {};
  const effort = thinkingEffort(s);
  if (effort) d.reasoning_effort = effort;
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (+s.temperature > 0 && s.min_p !== "" && s.min_p != null) d.min_p = +s.min_p;
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  if (s.script) d.script_guard = scriptGuard(s);      // only a choice of the user's, as the values above
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
// the thinking levels: the model's own (/health's "thinking": GLM-5.3-Flash's low, high and max, a model that only
// turns its thinking on and off its on), an older server's the four of before.  The model's default is sent as no
// reasoning_effort at all, so the model's own default applies (llm-jp-4.1's medium, GLM-5.3-Flash's max)
function thinkingInfo() {
  return health.thinking || {levels: ["none", "low", "medium", "high"], default: "high"};
}
function thinkingChosen(s) {
  const t = thinkingInfo();
  return t.levels.includes(s.thinking) ? s.thinking : t.default;
}
function thinkingEffort(s) {
  const v = thinkingChosen(s);
  return v && v !== thinkingInfo().default && v !== "on" ? v : undefined;
}
function thinkingButtons() {
  const t = thinkingInfo();
  const names = {none: tr("drawer.off", "Off"), low: tr("drawer.low", "Low"), medium: tr("drawer.medium", "Medium"),
                 high: tr("drawer.high", "High"), max: tr("drawer.max", "Max"), on: tr("drawer.on", "On")};
  $("s-thinking").replaceChildren(...t.levels.map((v) => {
    const b = document.createElement("button");
    Object.assign(b, {type: "button", textContent: names[v] || v});
    b.setAttribute("role", "radio");
    b.dataset.v = v;
    b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
    return b;
  }));
  $("s-thinking").closest(".st-field").hidden = !t.levels.length;
  loadDrawer();
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? tr("drawer.greedy", "0 · greedy") : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = segValue("s-thinking");
  const what = {none: tr("drawer.thinkNone", "answers right away"), low: tr("drawer.thinkLow", "short"),
                medium: tr("drawer.thinkMedium", "medium"), high: tr("drawer.thinkHigh", "thorough"),
                max: tr("drawer.thinkMax", "the most thorough"), on: tr("drawer.thinkOn", "thinks first")};
  $("o-thinking").textContent = sel ? what[sel] + (sel === thinkingInfo().default ? ` ${tr("drawer.thinkDefault", "(default)")}` : "") : "";
  const busyMode = segValue("s-busy");
  $("o-busy").textContent = {wait: tr("drawer.busyWaitNote", "Send is off until the answer is done, as before."),
    queue: tr("drawer.busyQueueNote", "Your message waits and goes as soon as the answer is done."),
    interject: tr("drawer.busyInterjectNote", "The answer stops where it is (what is written stays) and your message goes at once. A tool that is running finishes first.")}[busyMode] || "";
  for (const id of ["s-topp", "s-topk", "s-minp"]) $(id).disabled = t === 0;
}
for (const b of $("s-busy").children) b.onclick = () => { for (const x of $("s-busy").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const b of $("s-script").children) b.onclick = () => {
  for (const x of $("s-script").children) x.setAttribute("aria-checked", String(x === b));
  $("script-ctx-row").hidden = b.dataset.v === "off";        // the switch under it is kept as it was while hidden
};
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
for (const id of ["s-math", "s-show", "s-esp", "s-mcp", "s-share", "s-list", "s-edit", "s-scriptctx"]) {
  $(id).onclick = () => $(id).setAttribute("aria-checked", String($(id).getAttribute("aria-checked") !== "true"));
}
// going on past the context is what keeps a long chat going: switching it off asks first
$("s-seg").onclick = () => {
  const on = checked("s-seg");
  if (on && !confirm(tr("drawer.segmentsOff", "Turn off going on past the context?\n\nA chat that fills the context will then stop with an error instead of going on in a new segment. Chats already in segments stay as they are."))) return;
  $("s-seg").setAttribute("aria-checked", String(!on));
};
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const prev = settings;
  // the list and the edit controls change the messages on the screen: not while an answer is written or something is open
  const locked = !!busy || !!panel;
  const sel = segValue("s-thinking");
  settings = {thinking: sel && sel !== thinkingInfo().default ? sel : "", thinking_v: 2, defaults_v: 2, temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, min_p: $("s-minp").value.trim(), max: $("s-max").value.trim(),
              seed: $("s-seed").value.trim(),
              math: checked("s-math"), show: checked("s-show"), esp: checked("s-esp"), mcp: checked("s-mcp"),
              chatList: locked ? !!prev.chatList : checked("s-list"), chatEdit: locked ? !!prev.chatEdit : checked("s-edit"),
              whileBusy: segValue("s-busy") || "wait", segments: checked("s-seg"), segmentLimit: $("s-seglimit").value.trim(),
              // the language's own choice stays "": it follows a later change of the page's language
              script: segValue("s-script") === scriptChoice({}) ? "" : segValue("s-script"), scriptContext: checked("s-scriptctx")};
  store.set("sampling", settings);
  // the chat's memory, edited here: kept on the last message, so a rewind past it takes the edit back too (segments.js)
  const memory = $("s-memory").value;
  if (!locked && messages.length && memory !== StrataSegments.memoryAt(messages)) {
    messages[messages.length - 1].memoryEdit = memory;
    saveChat();
  }
  StrataMath.reset();                                  // math on or off: every answer is drawn again
  if (settings.chatList !== !!prev.chatList) await setChatList(settings.chatList, true);
  if (settings.chatEdit !== !!prev.chatEdit) { clearUndo(); syncActions(); }
  if (settings.whileBusy === "wait" && pending) releasePending();    // "wait" parks nothing: a waiting message goes back to the box
  setBusy(!!busy);
  for (const el of $("chat").querySelectorAll(".st-msg--assistant")) {
    const m = messages[+el.dataset.i];
    updateAssistant(el, m, !!busy && busy.msg === m);  // the element send() holds stays the same while an answer streams
  }
  const share = checked("s-share");
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", tr("drawer.saved", "Chat settings saved"), share ? tr("drawer.savedShared", "Other apps (omp, API clients) use these settings from their next request.")
                                                                         : tr("drawer.savedOwn", "Other apps use their own settings again."));
    } catch (e) {
      toast("error", tr("drawer.savedHere", "Saved here, but not for other apps"), e.message, 6000);
    }
    return;
  }
  toast("success", tr("drawer.saved", "Chat settings saved"), settings.temperature === 0 ? tr("drawer.savedGreedy", "Greedy: the same question gives the same answer.") : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ language (serve/web/i18n.js)
// The user's own choice, kept in this browser; without one ("Automatic"), the browser's language where the page has it,
// else English. /?lang=de shows one for this visit only.
const urlLang = new URLSearchParams(location.search).get("lang");
function browserLanguage() {
  return StrataI18n.choose(urlLang, store.get("lang", ""), navigator.languages || [navigator.language]);
}
// the choices: Automatic first (in the language of the moment), then each language in its own name
function languageChoices() {
  $("lang").innerHTML = `<option value="auto">${esc(tr("settings.languageAuto", "Automatic (browser language)"))}</option>` +
    StrataI18n.LANGUAGES.map(([code, name]) => `<option value="${esc(code)}">${esc(name)}</option>`).join("");
  const saved = store.get("lang", "");
  $("lang").value = urlLang && urlLang === StrataI18n.lang ? StrataI18n.lang
                  : StrataI18n.LANGUAGES.some((l) => l[0] === saved) ? saved : "auto";
}
function showLanguage() {
  languageChoices();
  renderEmptyChat();
  setSidebar(sidebarOpen, false);
  thinkingButtons();   // the level buttons are built in the code, in the language of the moment
  imgModels();         // so are the Images tab's texts and the chat's image choice
  loadUpdate();        // and About's Updates card
  renderConfig(true);  // and its Model settings card
  renderMcp();         // and the Monitor's MCP card
  renderChat();
  setBusy(!!busy);
  if (lastMetrics) render(lastMetrics);
}
$("lang").onchange = async () => {
  const v = $("lang").value;
  if (v === "auto") store.del("lang"); else store.set("lang", v);   // Automatic forgets the choice
  await StrataI18n.setLanguage(v === "auto" ? StrataI18n.choose(null, null, navigator.languages || [navigator.language]) : v);
  showLanguage();
};

// ------------------------------------------------------------------ start
async function start() {
  await StrataI18n.setLanguage(browserLanguage());
  languageChoices();
  renderEmptyChat();
  initChats();
  if (settings.chatList) await setChatList(true, false);       // the saved-chats list is read before the first draw
  if (settings.segments) askPersist();
  setBusy(false);
  renderChat();
  const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
  if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
  thinkingButtons();   // the four of before until /health says the model's own
  loadHealth().then(loadSwitcher).then(loadMcp).then(() => loadUpdate()).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
  showTab(location.hash.slice(1) || "chat");
  poll();
  setInterval(() => { if (!switching) loadSwitcher(); }, 15000);   // a switch started elsewhere shows here too
  requestAnimationFrame(fitHeader);
}
start();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
function readAsDataURL(f) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error || new Error(`${f.name}: could not be read`));
    r.readAsDataURL(f);
  });
}
$("img-size").onchange = () => { $("img-custom").hidden = $("img-size").value !== "custom"; };
// the image model's defaults (/health's image_defaults, what the API gives a request that leaves a field out): the
// fields' placeholders, the steps hidden where they are fixed (Turbo), the transparent background where it has one,
// the negative prompt hidden where the guidance is one token the model reads (HunyuanImage 3.0: guidance_token), and
// the sizes snapped to the model's own (buckets) for the estimate
// the image models (/health's image_models, the default first): a choice where there are several, its defaults and
// whether it edits; one without edits takes one image to start from (img2img) instead of reference images
function imgModel() {
  const ms = health.image_models || [];
  return ms.find(m => m.id === $("img-model").value) || ms[0] || {defaults: health.image_defaults || {}, edits: true};
}
// the chat's image choice: auto (the words choose the model; the default otherwise), off, or one model; kept here
function chatImages() {
  const ms = health.image_models || [], sel = $("chat-images");
  const was = sel.value || store.get("chat_images", "auto");
  const opts = [new Option(ms.length > 1 ? tr("chatImages.auto", "Images: auto") : tr("chatImages.on", "Images: on"), "auto"),
                new Option(tr("chatImages.off", "Images: off"), "off")];
  if (ms.length > 1) opts.push(...ms.map(m => new Option(tr("chatImages.model", "Images: {model}", {model: m.id}), m.id)));
  sel.replaceChildren(...opts);
  sel.value = [...sel.options].some(o => o.value === was) ? was : "auto";
  sel.hidden = $("image-btn").hidden = !health.image_generation;
}
$("chat-images").onchange = () => store.set("chat_images", $("chat-images").value);
// the chat's web search (the server's search_url): on or off, kept here; on sends web_search_options
function chatSearch() {
  const sel = $("chat-search"), was = sel.value || store.get("chat_search", "on");
  sel.replaceChildren(new Option(tr("chatSearch.on", "Search: on"), "on"), new Option(tr("chatSearch.off", "Search: off"), "off"));
  sel.value = was === "off" ? "off" : "on";
  sel.hidden = !health.web_search;
}
$("chat-search").onchange = () => store.set("chat_search", $("chat-search").value);
function imgModels() {
  chatImages();
  chatSearch();
  const ms = health.image_models || [], sel = $("img-model"), was = sel.value;
  sel.replaceChildren(...ms.map(m => new Option(m.id, m.id)));
  if (ms.some(m => m.id === was)) sel.value = was;
  $("img-model-f").hidden = ms.length < 2;
  imgDefaults();
}
$("img-model").onchange = () => {
  if (!imgModel().edits && ($("img-ref").files || []).length > 1) $("img-ref").value = "";
  imgDefaults();
  imgMaskSetup();
};
// the prompt a model starts with (its quality tags, defaults.prompt_start): put in while the field is empty or still
// holds the previous model's
let imgPromptStart = "";
function imgDefaults() {
  const d = imgModel().defaults || {}, edits = imgModel().edits;
  const p = $("img-prompt"), start = d.prompt_start || "";
  if (!p.value.trim() || p.value.trim() === imgPromptStart) p.value = start;
  imgPromptStart = start;
  // an image: an edit's references, or (a model without edits) the one image img2img starts from; its strength
  // (how much of the schedule runs from it) beside it, the model's default as the placeholder
  $("img-ref-l").textContent = edits ? tr("img.edit", "Edit an image (optional)") : tr("img.img2img", "Start from an image (optional, img2img)");
  $("img-ref").multiple = !!edits;
  const refs = ($("img-ref").files || []).length > 0;
  $("img-strength-f").hidden = !refs;
  $("img-steps-f").hidden = !!d.steps_fixed;
  $("img-steps").min = d.steps_min || 1;
  $("img-steps").max = d.steps_max || 100;
  $("img-steps").placeholder = String((refs && edits ? d.edit_steps : d.steps) || "");
  const token = !!d.guidance_token;   // one token, not a second pass with a negative prompt
  $("img-neg-f").hidden = token;
  $("img-cfg-l").textContent = token ? tr("img.guidance", "Guidance") : tr("img.guidanceCfg", "Guidance (CFG)");
  $("img-cfg").placeholder = d.guidance > 1 || token ? String(d.guidance) : tr("img.off", "off");
  $("img-bg-f").hidden = !d.transparency;
  // the samplers the model takes (Anima's, Mage-Flow's without edits), its own first
  const samplers = d.samplers || [];
  $("img-sampler-f").hidden = samplers.length < 2;
  $("img-sampler").replaceChildren(...samplers.map(x => new Option(x, x)));
  if (d.sampler) $("img-sampler").value = d.sampler;
  imgCount();
  imgFormat();
  imgNegHint();
  imgMossSetup();
  imgLoras();
}
// the adapters this image model applies (/health's image_models[].defaults.loras, serve/loras.py's listing): the ones
// it takes, chosen here, with a strength each (the API's -2 to 2, starting from the listing's recommended one) and,
// where the file names them, its trigger words, which a button puts into the prompt.  The choice is kept in this
// browser for each model (xestrata.img_loras.<model>) and comes back with it.  A model whose listing is empty takes no
// adapter (HunyuanImage 3.0, whose config names none: the DiT is not one of TARGETS): the block is hidden and none is sent.
const IMG_LORA_COUNT = 4, IMG_LORA_SCALE = 2;   // serve/loras.py's MAX_LORA, and its SCALE_MIN..SCALE_MAX
let imgLoraSel = [];   // [{name, scale}]: the adapters the next request applies, in the order they were chosen
function imgLoras() {
  const d = imgModel().defaults || {}, list = d.loras || [];
  $("img-lora").hidden = !(d.loras_take ?? list.length);    // an older server says nothing of loras_take
  imgLoraFiles(list);
  const by = new Map(list.map(e => [e.name, e]));
  const kept = list.length ? store.get(`img_loras.${imgModel().id || "default"}`, []) : [];
  const names = new Set();
  imgLoraSel = (Array.isArray(kept) ? kept : []).filter(u => {
    if (!u || typeof u.name !== "string" || !by.has(u.name) || names.has(u.name)) return false;
    names.add(u.name);
    return true;
  }).slice(0, IMG_LORA_COUNT)
    .map(u => ({name: u.name, scale: imgLoraStrength(u.scale, by.get(u.name).scale)}));
  imgLoraDraw(list);
}
// a strength the API takes: the field's number within -2..2 at two decimals, `fallback` (the listing's recommended
// one) when the field is empty or not a number, else the API's default 1
function imgLoraStrength(value, fallback) {
  const v = value == null || value === "" || typeof value === "boolean" ? NaN : Number(value);
  const want = Number.isFinite(v) ? v : Number(fallback);
  const use = Number.isFinite(want) ? want : 1;
  return Math.max(-IMG_LORA_SCALE, Math.min(IMG_LORA_SCALE, Math.round(use * 100) / 100));
}
// the picker (the adapters not chosen yet) and a chip for each chosen one: its name, its strength, its trigger words
// when it has them, and the button that takes it off
function imgLoraDraw(list) {
  const add = $("img-lora-add"), chosen = new Set(imgLoraSel.map(u => u.name));
  const open = list.filter(e => !chosen.has(e.name));
  add.replaceChildren(new Option(tr("img.loraAdd", "Add a LoRA…"), ""), ...open.map(e => new Option(e.name, e.name)));
  add.value = "";
  add.disabled = !open.length || imgLoraSel.length >= IMG_LORA_COUNT;
  $("img-lora-chips").replaceChildren(...imgLoraSel.map(u => {
    const e = list.find(x => x.name === u.name) || {name: u.name};
    const chip = document.createElement("div"), name = document.createElement("span");
    chip.className = "img-lora-chip";
    chip.dataset.name = u.name;
    name.className = "img-lora-chip__name";
    name.textContent = u.name;
    name.title = imgLoraAbout(e);
    const scale = document.createElement("input");
    scale.className = "st-input";
    scale.type = "number";
    scale.min = String(-IMG_LORA_SCALE);
    scale.max = String(IMG_LORA_SCALE);
    scale.step = "0.05";
    scale.value = String(u.scale);
    scale.setAttribute("aria-label", tr("img.strength", "Strength"));
    scale.onchange = () => { u.scale = imgLoraStrength(scale.value, e.scale); scale.value = String(u.scale); imgLoraSave(); };
    chip.append(name, scale);
    if (e.trigger) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "st-btn st-btn--secondary";
      b.textContent = tr("img.loraTrigger", "Trigger");
      b.title = tr("img.loraTriggerTitle", "Put its trigger words into the prompt");
      b.onclick = () => imgLoraTrigger(e.trigger);
      chip.append(b);
    }
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "st-btn st-btn--icon";
    rm.textContent = "×";
    rm.title = tr("img.loraRemove", "Remove this LoRA");
    rm.setAttribute("aria-label", rm.title);
    rm.onclick = () => { imgLoraSel = imgLoraSel.filter(x => x !== u); imgLoraSave(); imgLoraDraw(list); };
    chip.append(rm);
    return chip;
  }));
  const most = imgLoraSel.length >= IMG_LORA_COUNT ? tr("img.loraMost", "Up to {n} LoRAs at once (the API's limit)", {n: IMG_LORA_COUNT}) : "";
  $("img-lora-hint").textContent = most;
  $("img-lora-hint").hidden = !most;
}
// this model's files, each with its delete button
function imgLoraFiles(list) {
  $("img-lora-files").hidden = !list.length;
  $("img-lora-files-sum").textContent = tr("img.loraFiles", "LoRA files on the server ({n})", {n: list.length});
  $("img-lora-file-list").replaceChildren(...list.map((e) => {
    const row = document.createElement("div"), name = document.createElement("span"), rm = document.createElement("button");
    row.className = "img-lora-file";
    name.textContent = e.path;
    name.title = imgLoraAbout(e);
    rm.type = "button";
    rm.className = "st-btn st-btn--secondary";
    rm.textContent = tr("img.loraDelete", "Delete");
    rm.onclick = () => imgLoraRemove(e);
    row.append(name, rm);
    return row;
  }));
}
// a chip's tooltip: the rank and the file's size, and the kinds of weight it changes
function imgLoraAbout(e) {
  const parts = [tr("img.loraAbout", "rank {rank}, {size} MiB", {rank: e.rank || "–", size: e.size_mib == null ? "–" : e.size_mib})];
  if ((e.targets || []).length) parts.push(e.targets.join(", "));
  return parts.join(" · ");
}
// an adapter's trigger words into the prompt, once: after what is written there, comma-separated (the file's own
// words, not a <lora:...> tag: the page names its adapters in the request's field)
function imgLoraTrigger(words) {
  const p = $("img-prompt"), was = p.value.trim();
  if (!was) p.value = words;
  else if (!was.includes(words)) p.value = `${was.replace(/[\s,]+$/, "")}, ${words}`;
  p.focus();
}
function imgLoraSave() {
  store.set(`img_loras.${imgModel().id || "default"}`, imgLoraSel.map(u => ({name: u.name, scale: u.scale})));
}
// the adapters this request asks for, as the API's field (serve/loras.py's "lora": [{"name", "scale"}]); none for a
// model whose block is hidden, or when none is chosen (the model's own weights)
function imgLorasBody() {
  return $("img-lora").hidden ? [] : imgLoraSel.slice(0, IMG_LORA_COUNT).map(u => ({name: u.name, scale: imgLoraStrength(u.scale)}));
}
$("img-lora-add").onchange = () => {
  const list = (imgModel().defaults || {}).loras || [], e = list.find(x => x.name === $("img-lora-add").value);
  if (e && imgLoraSel.length < IMG_LORA_COUNT && !imgLoraSel.some(u => u.name === e.name)) {
    imgLoraSel.push({name: e.name, scale: imgLoraStrength(null, e.scale)});
    imgLoraSave();
  }
  imgLoraDraw(list);
};
// the count: up to max_n, and as many as take budget_s at the speed the server measured (its rates; before it has
// made an image, 4), with the time the images chosen take (the server refuses more, ImageGen.check_n)
function imgEstimate(n) {
  const d = imgModel().defaults || {}, rates = d.rates || {};
  const refs = ($("img-ref").files || []).length > 0, edit = refs && imgModel().edits;
  const pts = rates[edit ? "edit" : "generate"] || rates.generate || rates.edit;
  if (!pts || !pts.length) return null;
  let [w, h] = $("img-size").value === "custom" ? [+$("img-w").value, +$("img-h").value]
    : ($("img-size").value === "auto" ? "1024x1024" : $("img-size").value).split("x").map(Number);
  w = Math.ceil((w || 1024) / 16) * 16;
  h = Math.ceil((h || 1024) / 16) * 16;
  if (d.buckets && d.buckets.length) {   // the model's own sizes: the one whose shape is nearest (the server's choice)
    const r = h / w;
    [w, h] = d.buckets.reduce((a, b) => Math.abs(b[1] / b[0] - r) < Math.abs(a[1] / a[0] - r) ? b : a);
  }
  let steps = d.steps_fixed ? d.steps : (+$("img-steps").value || (edit ? d.edit_steps : d.steps) || 1);
  if (refs) {   // a strength runs part of the schedule (the server's strength_run)
    const s = $("img-strength").value !== "" ? Number($("img-strength").value) : imgStrength(d, steps);
    if (s > 0 && s < 1) steps -= Math.floor(Math.max(steps - steps * s, 0));
  }
  const g = $("img-cfg").value !== "" ? Number($("img-cfg").value) : (d.guidance || 1);
  if (g > 1 && !d.guidance_token && (d.negative_prompt != null || $("img-neg").value.trim())) steps *= 2;
  // as the server's ImageGen.estimate_s: the size measured nearest, a step's time scaled (size_exp above it)
  const mp = w * h / 1e6;
  const [near, stepS, restS] = pts.reduce((a, p) => Math.abs(Math.log(mp / p[0])) < Math.abs(Math.log(mp / a[0])) ? p : a);
  const scale = mp / near;
  return n * (steps * stepS * scale ** (scale > 1 ? (d.size_exp || 1.5) : 1) + restS * scale);
}
// The server's LoRA folder from here (serve/loras.py): a file uploaded (the button, or dropped on the block), or one
// the server fetches from Hugging Face, kept where an image model takes it; and each file of this model, to remove.
// After each, /health is read again for the listings.
function imgLoraStatus(text, kind = "") {
  const el = $("img-lora-status");
  el.textContent = text || "";
  el.hidden = !text;
  el.dataset.kind = kind;
}
async function imgLoraRefresh() {
  try { health = await (await fetch("health", {cache: "no-store"})).json(); } catch (e) { return; }
  imgModels();
}
async function imgLoraError(r) {
  try { return (await r.json()).error.message; } catch (e) { return `HTTP ${r.status}`; }
}
function imgLoraUpload(file) {
  if (!file) return;
  if (!/\.(safetensors|gguf)$/i.test(file.name)) {
    imgLoraStatus(tr("img.loraType", "A LoRA is a .safetensors or .gguf file"), "error");
    return;
  }
  const x = new XMLHttpRequest();                 // XHR, not fetch: its upload reports how far it is
  x.open("POST", `v1/images/loras?name=${encodeURIComponent(file.name)}`);
  for (const [k, v] of Object.entries(headers())) x.setRequestHeader(k, v);
  x.setRequestHeader("Content-Type", "application/octet-stream");
  x.upload.onprogress = (e) => {
    if (e.lengthComputable) imgLoraStatus(tr("img.loraUploading", "Uploading {name}: {pct}%", {name: file.name, pct: Math.round(100 * e.loaded / e.total)}));
  };
  x.onerror = () => imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: tr("pill.unreachable", "Server not reachable")}), "error");
  x.onload = async () => {
    let j = {};
    try { j = JSON.parse(x.responseText); } catch (e) { /* not json */ }
    if (x.status !== 200) {
      imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: (j.error && j.error.message) || `HTTP ${x.status}`}), "error");
      return;
    }
    imgLoraStatus(tr("img.loraAdded", "Added {name}", {name: j.added.name}));
    await imgLoraRefresh();
  };
  imgLoraStatus(tr("img.loraUploading", "Uploading {name}: {pct}%", {name: file.name, pct: 0}));
  x.send(file);
}
async function imgLoraFetch() {
  const url = $("img-lora-url").value.trim();
  if (!url) return;
  let r;
  try {
    r = await fetch("v1/images/loras/fetch", {method: "POST", headers: headers(true), body: JSON.stringify({url})});
  } catch (e) { imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: tr("pill.unreachable", "Server not reachable")}), "error"); return; }
  if (r.status !== 202) { imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: await imgLoraError(r)}), "error"); return; }
  const job = await r.json();
  $("img-lora-url").value = "";
  for (;;) {                                       // the download, until it ends (the server keeps going without us)
    const mb = (n) => fmt((n || 0) / 1048576, 0);
    imgLoraStatus(job.total ? tr("img.loraFetching", "Fetching {name}: {got} of {total} MB", {name: job.name, got: mb(job.bytes), total: mb(job.total)})
                            : tr("img.loraFetchingSize", "Fetching {name}: {got} MB", {name: job.name, got: mb(job.bytes)}));
    await new Promise((ok) => setTimeout(ok, 1000));
    let j;
    try { j = await (await fetch(`v1/images/loras/fetch/${job.id}`, {headers: headers()})).json(); } catch (e) { continue; }
    Object.assign(job, j);
    if (job.state === "done") { imgLoraStatus(tr("img.loraAdded", "Added {name}", {name: job.added.name})); await imgLoraRefresh(); return; }
    if (job.state === "failed") { imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: job.error}), "error"); return; }
  }
}
async function imgLoraRemove(e) {
  if (!confirm(tr("img.loraRemoveConfirm", "Delete {name} from the server's LoRA folder? The file is removed for every image model.", {name: e.path}))) return;
  let r;
  try {
    r = await fetch("v1/images/loras/remove", {method: "POST", headers: headers(true), body: JSON.stringify({path: e.path})});
  } catch (err) { imgLoraStatus(tr("img.loraRemoveFailed", "Not removed: {why}", {why: tr("pill.unreachable", "Server not reachable")}), "error"); return; }
  if (!r.ok) { imgLoraStatus(tr("img.loraRemoveFailed", "Not removed: {why}", {why: await imgLoraError(r)}), "error"); return; }
  imgLoraSel = imgLoraSel.filter((u) => u.name !== e.name);
  imgLoraSave();
  imgLoraStatus(tr("img.loraRemoved", "Removed {name}", {name: e.name}));
  await imgLoraRefresh();
}
$("img-lora-upload").onclick = () => $("img-lora-file").click();
$("img-lora-file").onchange = () => { imgLoraUpload($("img-lora-file").files[0]); $("img-lora-file").value = ""; };
$("img-lora-fetch").onclick = imgLoraFetch;
$("img-lora-url").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); imgLoraFetch(); } };
for (const ev of ["dragenter", "dragover"]) $("img-lora").addEventListener(ev, (e) => {
  if ([...(e.dataTransfer.items || [])].some((i) => i.kind === "file")) { e.preventDefault(); $("img-lora").classList.add("is-drop"); }
});
$("img-lora").addEventListener("dragleave", () => $("img-lora").classList.remove("is-drop"));
$("img-lora").addEventListener("drop", (e) => {
  $("img-lora").classList.remove("is-drop");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) { e.preventDefault(); imgLoraUpload(f); }
});
// the strength a request without one gets: an edit's 1; an img2img's d.strength at most, lowered with few steps so
// that it skips one (the server's img2img_strength)
function imgStrength(d, steps) {
  const s = d.strength != null ? d.strength : 1;
  return !d.img2img || s >= 1 || steps < 2 ? s : Math.min(s, 1 - 1 / steps);
}
function imgStrengthHint() {
  const d = imgModel().defaults || {};
  const steps = d.steps_fixed ? d.steps : (+$("img-steps").value || d.steps || 1);
  $("img-strength").placeholder = String(+imgStrength(d, steps).toFixed(2));
}
function imgCount() {
  imgStrengthHint();
  const d = imgModel().defaults || {}, n = $("img-n"), was = +n.value || 1;
  const maxN = d.max_n || 16, budget = d.budget_s || 600, one = imgEstimate(1);
  const most = one == null ? Math.min(maxN, 4) : Math.max(1, Math.min(maxN, Math.floor(budget / one)));
  if (n.options.length !== most)
    n.replaceChildren(...Array.from({length: most}, (_, i) => new Option(String(i + 1))));
  n.value = String(Math.min(was, most));
  const est = imgEstimate(+n.value);
  $("img-n-est").textContent = est == null ? "" : est < 90 ? tr("img.aboutS", "about {n} s", {n: Math.max(1, Math.round(est))})
    : tr("img.aboutMin", "about {n} min", {n: (est / 60).toFixed(est < 600 ? 1 : 0)});
  $("img-n-f").title = one == null ? tr("img.countFirst", "Up to 4 until the server has made an image and measured its speed")
    : tr("img.countMost", "Up to {n}: the images one request makes in {min} min at most (budget_s)", {n: most, min: Math.round(budget / 60)});
}
for (const id of ["img-size", "img-w", "img-h", "img-steps", "img-cfg", "img-n", "img-strength"])
  $(id).addEventListener("input", imgCount);
$("img-size").addEventListener("change", imgCount);
// whether the negative prompt is used: only with guidance above 1, and some models guide only with one
function imgNegHint() {
  const d = imgModel().defaults || {};
  const g = $("img-cfg").value !== "" ? Number($("img-cfg").value) : (d.guidance || 1);
  const neg = $("img-neg").value.trim() !== "";
  let msg = "";
  if (d.guidance_token) msg = "";   // no negative prompt: the field is hidden
  else if (neg && g <= 1) msg = tr("img.negUnused", "The negative prompt is used only with guidance above 1.");
  else if (!neg && g > 1 && d.negative_prompt == null) msg = tr("img.negNeeded", "This image model guides only with a negative prompt.");
  $("img-neg-hint").textContent = msg;
  $("img-neg-hint").hidden = !msg;
}
// the mask (every model takes one): the first image (an edit's reference, or img2img's) under a canvas of its own
// size, painted where it is to change; sent as OpenAI's mask, a PNG of that size transparent where it was painted
let imgMaskURL = null;
function imgMaskSetup() {
  const d = imgModel().defaults || {};
  const f = ($("img-ref").files || [])[0];
  if (imgMaskURL) URL.revokeObjectURL(imgMaskURL);
  imgMaskURL = null;
  $("img-mask").hidden = !(f && d.mask);
  if ($("img-mask").hidden) { imgLayoutDraw(); return; }
  imgMaskURL = URL.createObjectURL(f);
  const img = $("img-mask-img"), c = $("img-mask-paint");
  img.onload = () => { c.width = img.naturalWidth; c.height = img.naturalHeight; imgLayoutDraw(); };
  img.src = imgMaskURL;
}
(() => {
  const c = $("img-mask-paint");
  let last = null;
  const at = (ev) => {
    const r = c.getBoundingClientRect();
    return [(ev.clientX - r.left) * c.width / r.width, (ev.clientY - r.top) * c.height / r.height, c.width / r.width];
  };
  const stroke = (a, b) => {
    const g = c.getContext("2d");
    g.globalCompositeOperation = $("img-erase").checked ? "destination-out" : "source-over";
    g.strokeStyle = "#e11d48";
    g.lineCap = g.lineJoin = "round";
    g.lineWidth = Number($("img-brush").value) * b[2];   // the brush in screen pixels, at the image's size
    g.beginPath();
    g.moveTo(a[0], a[1]);
    g.lineTo(b[0], b[1]);
    g.stroke();
  };
  c.onpointerdown = (ev) => { c.setPointerCapture(ev.pointerId); last = at(ev); stroke(last, last); };
  c.onpointermove = (ev) => { if (!last) return; const p = at(ev); stroke(last, p); last = p; };
  c.onpointerup = c.onpointercancel = () => { last = null; };
})();
$("img-mask-clear").onclick = () => { const c = $("img-mask-paint"); c.getContext("2d").clearRect(0, 0, c.width, c.height); };
// the mask as a data: URL, or null where nothing is painted
function imgMask() {
  const c = $("img-mask-paint");
  if ($("img-mask").hidden || !c.width) return null;
  const px = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let painted = false;
  for (let i = 3; i < px.length && !painted; i += 4) painted = px[i] > 0;
  if (!painted) return null;
  const m = document.createElement("canvas");
  m.width = c.width;
  m.height = c.height;
  const g = m.getContext("2d");
  g.fillRect(0, 0, m.width, m.height);
  g.globalCompositeOperation = "destination-out";
  g.drawImage(c, 0, 0);
  return m.toDataURL("image/png");
}
// MOSS-Image's own fields, shown where /health's defaults have a layout: the prompt written out by the chat's model
// (enhance), the schedule's shift, the blocks guidance's negative pass skips, skip-layer guidance, and a layout of
// boxes drawn over the image.  A box is kept in shares of the image's sides, so a change of size keeps it in place;
// it goes as the API's layout, [x_min, y_min, x_max, y_max] in the image's pixels from its top-left corner.
const imgLayout = {boxes: [], sel: -1};
const clamp01 = (v) => Math.min(1, Math.max(0, v));
// the size the boxes are counted in: the size chosen; "auto" is 1024 x 1024, or the starting image's shape at about
// as many pixels (sent as that size, so that the boxes and the image agree)
function imgLayoutWH() {
  const v = $("img-size").value;
  if (v === "custom") return [+$("img-w").value || 1024, +$("img-h").value || 1024];
  if (v !== "auto") return v.split("x").map(Number);
  const im = $("img-mask-img");
  if (!imgMaskURL || !im.naturalWidth) return [1024, 1024];
  const s = Math.sqrt(1024 * 1024 / (im.naturalWidth * im.naturalHeight));
  return [Math.max(256, Math.round(im.naturalWidth * s / 16) * 16), Math.max(256, Math.round(im.naturalHeight * s / 16) * 16)];
}
function imgMossSetup() {
  const d = imgModel().defaults || {};
  $("img-moss").hidden = !d.layout;
  if (!d.layout) return;
  const boxes = $("img-layout-on").checked;
  $("img-enhance-f").hidden = !d.enhance || boxes;   // the layout is the prompt already: nothing to write out
  $("img-shift").placeholder = String(d.shift);
  $("img-uskip").placeholder = (d.uncond_skip_layers || []).join(", ") || tr("img.noBlocks", "none");
  $("img-slg-scale").placeholder = String(d.slg_scale || 0);
  $("img-slg-start").placeholder = String(d.slg_start || 0);
  $("img-slg-stop").placeholder = String(d.slg_stop == null ? 1 : d.slg_stop);
  $("img-layout").hidden = !boxes;
  imgLayoutDraw();
  imgLayoutFields();
}
function imgBoxName(label) {
  return label === "text" ? tr("img.boxText", "Text") : label === "character" ? tr("img.boxCharacter", "Character") : tr("img.boxObject", "Object");
}
function imgLayoutDraw() {
  if ($("img-layout").hidden) return;
  const st = $("img-layout-stage"), [w, h] = imgLayoutWH();
  st.style.aspectRatio = `${w} / ${h}`;
  st.style.backgroundImage = imgMaskURL ? `url("${imgMaskURL}")` : "none";
  st.replaceChildren(...imgLayout.boxes.map((b, i) => {
    const e = document.createElement("div"), hd = document.createElement("span");
    e.className = "img-box";
    e.dataset.i = String(i);
    e.dataset.label = b.label;
    e.setAttribute("aria-selected", String(i === imgLayout.sel));
    Object.assign(e.style, {left: `${b.x0 * 100}%`, top: `${b.y0 * 100}%`, width: `${(b.x1 - b.x0) * 100}%`, height: `${(b.y1 - b.y0) * 100}%`,
                            zIndex: String(1 + Math.round((1 - (b.x1 - b.x0) * (b.y1 - b.y0)) * 1000))});   // a smaller box over a larger one
    e.textContent = b.label === "text" && b.text ? `"${b.text}"` : b.caption || imgBoxName(b.label);
    hd.className = "img-box__handle";
    e.append(hd);
    return e;
  }));
}
// the selected box's fields (set when the selection changes, not while they are typed in)
function imgLayoutFields() {
  const b = imgLayout.boxes[imgLayout.sel];
  $("img-layout-item").hidden = !b;
  if (!b) return;
  $("img-box-label").value = b.label;
  $("img-box-caption").value = b.caption;
  $("img-box-text").value = b.text;
  $("img-box-text-f").hidden = b.label !== "text";
}
(() => {
  const st = $("img-layout-stage"), least = 0.02;
  let drag = null;   // {i, mode: "new" | "move" | "size", p: where it began, b: the box as it was}
  st.tabIndex = 0;
  const at = (ev) => {
    const r = st.getBoundingClientRect();
    return [clamp01((ev.clientX - r.left) / r.width), clamp01((ev.clientY - r.top) / r.height)];
  };
  st.onpointerdown = (ev) => {
    const p = at(ev), el = ev.target.closest(".img-box");
    st.setPointerCapture(ev.pointerId);
    if (el) {
      const i = +el.dataset.i;
      drag = {i, mode: ev.target.classList.contains("img-box__handle") ? "size" : "move", p, b: {...imgLayout.boxes[i]}};
    } else {
      imgLayout.boxes.push({x0: p[0], y0: p[1], x1: p[0], y1: p[1], label: "object", caption: "", text: ""});
      drag = {i: imgLayout.boxes.length - 1, mode: "new", p, b: null};
    }
    imgLayout.sel = drag.i;
    imgLayoutDraw();
    imgLayoutFields();
  };
  st.onpointermove = (ev) => {
    if (!drag) return;
    const p = at(ev), b = imgLayout.boxes[drag.i];
    if (drag.mode === "new") {
      [b.x0, b.x1] = [Math.min(drag.p[0], p[0]), Math.max(drag.p[0], p[0])];
      [b.y0, b.y1] = [Math.min(drag.p[1], p[1]), Math.max(drag.p[1], p[1])];
    } else if (drag.mode === "move") {   // kept inside the image
      const o = drag.b, dx = Math.min(1 - o.x1, Math.max(-o.x0, p[0] - drag.p[0])), dy = Math.min(1 - o.y1, Math.max(-o.y0, p[1] - drag.p[1]));
      Object.assign(b, {x0: o.x0 + dx, x1: o.x1 + dx, y0: o.y0 + dy, y1: o.y1 + dy});
    } else {
      b.x1 = Math.max(b.x0 + least, p[0]);
      b.y1 = Math.max(b.y0 + least, p[1]);
    }
    imgLayoutDraw();
  };
  st.onpointerup = st.onpointercancel = () => {
    if (!drag) return;
    const b = imgLayout.boxes[drag.i], made = drag.mode === "new";
    if (made && (b.x1 - b.x0 < least || b.y1 - b.y0 < least)) {   // a tap, not a box: nothing selected
      imgLayout.boxes.splice(drag.i, 1);
      imgLayout.sel = -1;
    }
    drag = null;
    imgLayoutDraw();
    imgLayoutFields();
    if (made && imgLayout.sel >= 0) $("img-box-caption").focus();
  };
  st.onkeydown = (ev) => {
    if ((ev.key === "Delete" || ev.key === "Backspace") && imgLayout.sel >= 0) { ev.preventDefault(); $("img-box-delete").click(); }
  };
})();
const imgBox = () => imgLayout.boxes[imgLayout.sel];
$("img-box-label").onchange = () => { const b = imgBox(); if (!b) return; b.label = $("img-box-label").value; imgLayoutFields(); imgLayoutDraw(); };
$("img-box-caption").oninput = () => { const b = imgBox(); if (b) { b.caption = $("img-box-caption").value; imgLayoutDraw(); } };
$("img-box-text").oninput = () => { const b = imgBox(); if (b) { b.text = $("img-box-text").value; imgLayoutDraw(); } };
$("img-box-delete").onclick = () => {
  if (imgLayout.sel < 0) return;
  imgLayout.boxes.splice(imgLayout.sel, 1);
  imgLayout.sel = -1;
  imgLayoutDraw();
  imgLayoutFields();
};
$("img-layout-clear").onclick = () => { imgLayout.boxes = []; imgLayout.sel = -1; imgLayoutDraw(); imgLayoutFields(); };
$("img-layout-on").onchange = imgMossSetup;
for (const id of ["img-size", "img-w", "img-h"]) $(id).addEventListener("input", imgLayoutDraw);
$("img-size").addEventListener("change", imgLayoutDraw);
// a layout made elsewhere (the prompt enhancer's, a chat's generate_image) into the boxes, at the size it was made
// for: that size chosen, and the boxes on, for a change and another image
function imgLayoutLoad(layout, size, prompt) {
  const [w, h] = String(size || "").split("x").map(Number);
  if (!(w > 0 && h > 0) || !layout || !Array.isArray(layout.items)) return;
  const sel = $("img-size");
  if ([...sel.options].some(o => o.value === `${w}x${h}`)) sel.value = `${w}x${h}`;
  else { sel.value = "custom"; $("img-w").value = String(w); $("img-h").value = String(h); }
  $("img-custom").hidden = sel.value !== "custom";
  imgLayout.boxes = layout.items.filter(it => Array.isArray(it.bbox) && it.bbox.length === 4).map(it => {
    const [x0, y0, x1, y1] = it.bbox.map(Number);
    return {x0: clamp01(Math.min(x0, x1) / w), y0: clamp01(Math.min(y0, y1) / h), x1: clamp01(Math.max(x0, x1) / w), y1: clamp01(Math.max(y0, y1) / h),
            label: ["object", "character", "text"].includes(it.label) ? it.label : "object", caption: String(it.caption || ""), text: String(it.text || "")};
  });
  imgLayout.sel = -1;
  if (prompt != null) $("img-prompt").value = prompt;
  $("img-layout-on").checked = true;
  $("img-enhance").value = "";
  imgCount();
  imgMossSetup();
  $("img-layout").scrollIntoView({block: "nearest"});
}
// a chat's generate_image call with a layout (MOSS-Image's): its prompt, model, size and boxes in the Images tab
function imgLayoutFromTool(t) {
  let a = t.arguments;
  if (typeof a === "string") { try { a = JSON.parse(a); } catch (e) { return; } }
  if (!a || !a.layout) return;
  const ms = health.image_models || [];
  const m = ms.find(x => x.id === a.model && (x.defaults || {}).layout) || ms.find(x => (x.defaults || {}).layout);
  if (!m) return;
  $("img-model").value = m.id;
  imgDefaults();
  const size = /^\d+x\d+$/.test(a.size || "") ? a.size : (String(t.result || "").match(/\((\d+x\d+),/) || [])[1] || "1024x1024";
  showTab("images");
  imgLayoutLoad(a.layout, size, typeof a.prompt === "string" ? a.prompt : null);
}
// MOSS-Image's fields into a request's body: an error's text when a box cannot go (a text box without its words)
function imgMossBody(body) {
  if ($("img-moss").hidden) return null;
  const num = (id, key) => { if ($(id).value !== "") body[key] = Number($(id).value); };
  num("img-shift", "shift");
  num("img-slg-scale", "slg_scale");
  num("img-slg-start", "slg_start");
  num("img-slg-stop", "slg_stop");
  const blocks = (id, key) => {
    const v = $(id).value.trim();
    if (v) body[key] = /^(none|-)$/i.test(v) ? [] : v.split(/[\s,]+/).filter(Boolean).map(Number);
  };
  blocks("img-uskip", "uncond_skip_layers");
  blocks("img-slg", "slg_layers");
  if ($("img-layout-on").checked && imgLayout.boxes.length) {
    const [w, h] = imgLayoutWH(), items = [];
    for (const b of imgLayout.boxes) {
      const words = b.text.trim(), text = b.label === "text";
      if (text && !words) return tr("img.boxNeedsWords", "A text box needs its words.");
      items.push({label: b.label, caption: b.caption.trim() || (text ? words : imgBoxName(b.label)),
                  ...(text ? {text: words} : {}),
                  bbox: [Math.round(b.x0 * w), Math.round(b.y0 * h), Math.round(b.x1 * w), Math.round(b.y1 * h)]});
    }
    body.layout = {items};
    if (body.size === "auto") body.size = `${w}x${h}`;
  } else if ($("img-enhance").value && !$("img-enhance-f").hidden) {
    body.enhance = $("img-enhance").value;
  }
  return null;
}
// what the prompt enhancer did (the response's strata.enhance), under the image: the prompt it wrote, or its layout
// with a button that loads it into the boxes, or why it gave nothing
function imgEnhanced(j, item) {
  const e = j.strata && j.strata.enhance;
  if (!e) return null;
  const div = document.createElement("div"), p = document.createElement("span");
  div.className = "img-enhanced";
  if (e.error) p.textContent = tr("img.enhanceFailed", "The chat's model gave nothing usable ({error}); the prompt was used as it was.", {error: e.error});
  else if (e.made === "layout") p.textContent = tr("img.enhancedLayout", {one: "The chat's model laid the image out in {n} box.", other: "The chat's model laid the image out in {n} boxes."}, {n: (e.layout.items || []).length});
  else p.textContent = tr("img.enhancedCaption", "The chat's model wrote the prompt: {prompt}", {prompt: (item && item.revised_prompt) || e.prompt || ""});
  div.append(p);
  if (!e.error && e.made === "layout" && e.layout) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "st-btn";
    b.textContent = tr("img.layoutEdit", "Edit this layout");
    b.onclick = () => imgLayoutLoad(e.layout, j.size, null);
    div.append(b);
  }
  return div;
}
// whether this browser shows JPEG XL (a 1 x 1 image, decoded); older browsers do not
const imgJxl = new Promise(ok => {
  const i = new Image();
  i.onload = () => ok(i.naturalWidth === 1);
  i.onerror = () => ok(false);
  i.src = "data:image/jxl;base64,/woAEBAJCAABABgASxiLFYIB";
});
let imgJxlOK = false;
imgJxl.then(ok => { imgJxlOK = ok; imgFormat(); });
const IMG_FMT_NAMES = {png: "PNG", jpeg: "JPEG", webp: "WebP", jxl: "JPEG XL"};
// the format: the server's (/health's formats) that this browser shows; the one chosen last, else JPEG XL (lossless)
// where it can, else PNG.  JPEG, WebP and JPEG XL take a quality (JPEG XL's 100: lossless), kept per format; a
// transparent background has no JPEG (no alpha channel)
function imgFormat() {
  const d = imgModel().defaults || {};
  const f = $("img-fmt"), bg = $("img-bg").checked && !$("img-bg-f").hidden;
  const fmts = (d.formats || ["png", "jpeg", "webp"]).filter(x => x !== "jxl" || imgJxlOK);
  const was = store.get("img_fmt", fmts.includes("jxl") ? "jxl" : "png");
  if ([...f.options].map(o => o.value).join() !== fmts.join())
    f.replaceChildren(...fmts.map(x => new Option(IMG_FMT_NAMES[x] || x, x)));
  f.value = fmts.includes(was) ? was : "png";
  for (const o of f.options) o.disabled = bg && o.value === "jpeg";
  if (bg && f.value === "jpeg") f.value = "png";
  $("img-q-f").hidden = f.value === "png";
  $("img-q").value = String(store.get("img_q_" + f.value, f.value === "jxl" ? 100 : 90));
  $("img-q-f").title = f.value === "jxl" ? tr("img.lossless", "100: lossless") : "";
}
$("img-fmt").onchange = () => { store.set("img_fmt", $("img-fmt").value); imgFormat(); };
$("img-q").onchange = () => store.set("img_q_" + $("img-fmt").value, +$("img-q").value);
$("img-bg").onchange = imgFormat;
$("img-partial").checked = store.get("img_partial", false);
$("img-partial").onchange = () => store.set("img_partial", $("img-partial").checked);
// a streamed job (the "show partial images" setting): one figure shows the engine's partial images as they come, and
// the completed events become the body the plain request answers with
async function imgStream(r, prompt) {
  const fig = document.createElement("figure"), img = document.createElement("img"), cap = document.createElement("figcaption");
  img.alt = prompt;
  fig.append(img, cap);
  const j = {data: []};
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // event: lines and ": keep-alive" comments
        let e;
        try { e = JSON.parse(line.slice(5).trim()); } catch (err) { continue; }
        if (e.type === "error") throw new Error((e.error && e.error.message) || tr("img.none", "No image"));
        if (e.type.endsWith(".partial_image")) {
          img.src = `data:image/${e.output_format || "png"};base64,` + e.b64_json;
          cap.textContent = tr("img.partialCap", "Partial image {n}…", {n: e.partial_image_index + 1});
          if (!fig.isConnected) $("img-results").prepend(fig);
        } else if (e.type.endsWith(".completed")) {
          Object.assign(j, {model: e.model, size: e.size, output_format: e.output_format, strata: e.strata});
          j.data.push({b64_json: e.b64_json, seed: e.seed});
        }
      }
    }
  } finally {
    fig.remove();
  }
  if (!j.data.length) throw new Error(tr("img.none", "No image"));
  return j;
}
$("img-neg").oninput = () => { imgNegHint(); imgCount(); };
$("img-cfg").addEventListener("input", imgNegHint);
$("img-ref").onchange = () => { imgDefaults(); imgMaskSetup(); };
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const size = $("img-size").value === "custom" ? `${$("img-w").value}x${$("img-h").value}` : $("img-size").value;
  const refs = Array.from($("img-ref").files || []);
  // "auto" goes as it is: the server makes it 1024 x 1024, a reference's shape, or the model's own choice
  const body = {prompt, size, response_format: "b64_json"};
  if (!$("img-model-f").hidden) body.model = $("img-model").value;
  const loras = imgLorasBody();
  if (loras.length) body.lora = loras;
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  const neg = $("img-neg").value.trim();
  if (neg) body.negative_prompt = neg;
  if ($("img-steps").value !== "" && !$("img-steps-f").hidden) body.steps = Number($("img-steps").value);
  if ($("img-cfg").value !== "") body.true_cfg_scale = Number($("img-cfg").value);
  if ($("img-bg").checked && !$("img-bg-f").hidden) body.background = "transparent";
  if (!$("img-sampler-f").hidden && $("img-sampler").value) body.sampler = $("img-sampler").value;
  if (refs.length && $("img-strength").value !== "") body.strength = Number($("img-strength").value);
  if (+$("img-n").value > 1) body.n = +$("img-n").value;
  body.output_format = $("img-fmt").value;
  if (body.output_format !== "png") body.output_compression = Math.max(0, Math.min(100, +$("img-q").value || 0));
  const bad = imgMossBody(body);
  if (bad) { toast("error", tr("img.none", "No image"), bad, 8000); return; }
  const partial = $("img-partial").checked;
  if (partial) Object.assign(body, {stream: true, partial_images: 2});
  imageBusy = true;
  $("img-go").disabled = true;
  paintLive();
  const t0 = performance.now();
  $("img-hint").textContent = tr("img.generatingHint", "Generating… (a minute or more on a small GPU)");
  try {
    // with an image: an edit (or a model without edits, img2img), in JSON with data: URLs (without an API key the
    // server takes only JSON from a browser page, so that another site's form cannot use it)
    if (refs.length) body.images = await Promise.all(refs.map(async f => ({image_url: await readAsDataURL(f)})));
    const mask = refs.length ? imgMask() : null;
    if (mask) body.mask = {image_url: mask};
    const url = refs.length ? "v1/images/edits" : "v1/images/generations";
    const r = await fetch(url, {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = partial && r.ok ? await imgStream(r, prompt) : await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = `data:image/${j.output_format || "png"};base64,` + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      const made = j.strata && j.strata.made_size ? ` ${tr("img.madeCut", "(made at {size}, middle cut out)", {size: j.strata.made_size})}`
        : j.strata && j.strata.requested_size ? ` ${tr("img.madeNearest", "({size} asked: the model's nearest size)", {size: j.strata.requested_size})}` : "";
      const by = $("img-model-f").hidden ? "" : `${j.model}, `;
      cap.textContent = `${by}${j.size}${made}, ${tr("img.capSeed", "seed {seed}", {seed: d.seed})}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      const note = imgEnhanced(j, d);
      if (note) fig.append(note);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = tr("img.done", "Done. Right-click an image to save it.");
    if (j.strata && j.strata.made_size)
      toast("info", tr("img.madeAt", "Made at {size}", {size: j.strata.made_size}), tr("img.madeCutText", "The image model needs sides that are multiples of 16: the middle {size} was cut out.", {size: j.size}), 8000);
    else if (j.strata && j.strata.requested_size)
      toast("info", tr("img.madeAt", "Made at {size}", {size: j.size}), tr("img.madeNearestText", "This image model makes a fixed set of sizes: {size} is the one whose shape is nearest the {asked} asked for.", {size: j.size, asked: j.strata.requested_size}), 8000);
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", tr("img.none", "No image"), String(e.message || e), 8000);
  } finally {
    imageBusy = false;
    $("img-go").disabled = false;
    $("img-ref").value = "";   // the reference images go with this job only: the next Generate is a new image
    paintLive();
    await loadHealth();        // the speed the job measured (the count's limit and estimate)
    imgMaskSetup();
  }
};
