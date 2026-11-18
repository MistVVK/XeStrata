// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
/* global tr, StrataI18n, StrataMath, StrataMermaid, StrataPreview, node, panel, msgActions, saveChat, stillWriting, initChats, setChatList,
          clearUndo, syncActions, closePanel, renderSidebar, renderParent, renderUndoBar, setSidebar, sidebarOpen, chatId,
          chat, saveChatOf, resumeReplies, chatName, openChat, StrataSegments */
/* exported pending */   // read by chats-ui.js (the history's buttons wait for it)
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
  // closed by hand too (×), before its time is up
  const x = document.createElement("button");
  x.type = "button";
  x.className = "st-btn st-btn--icon st-toast__close";
  x.setAttribute("aria-label", tr("close", "Close"));
  x.title = tr("close", "Close");
  x.innerHTML = `<svg class="st-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6L18 18M18 6L6 18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;
  const timer = setTimeout(() => el.remove(), ms);
  x.onclick = () => { clearTimeout(timer); el.remove(); };
  el.appendChild(x);
  $("toasts").appendChild(el);
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
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
  refreshPreviews();                                  // the preview frames carry the page's colours
  queueMicrotask(redrawCharts);                       // the charts too (after this script set the chat up)
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("xestrata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
let aboutSub = "overview";   // About's own tabs: overview, terms (the models' terms), licenses (the software's)
let noteShown = false;
function showTab(name) {
  const [main, sub] = String(name || "").split("/");
  tab = ["chat", "images", "tools", "video", "translate", "monitor", "about"].includes(main) ? main : "chat";
  if (tab === "about") aboutSub = ["terms", "licenses"].includes(sub) ? sub : "overview";
  for (const b of document.querySelectorAll(".st-header .st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  fitHeader();                                       // on a phone the open tab shows its name: the row's room changes
  for (const v of ["chat", "images", "tools", "video", "translate", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") {
    $("img-prompt").focus();
    if (health.image_note && !noteShown) {   // once a page load: setup lowered the image model's size for the RAM
      noteShown = true;
      toast("info", tr("img.noteToast", "Image model size lowered"), health.image_note, 12000);
    }
  }
  const where = tab === "about" && aboutSub !== "overview" ? `about/${aboutSub}` : tab;   // #about/terms opens that tab
  if (location.hash.slice(1) !== where) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${where}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") { loadConfig(); loadUpdate(); showAboutSub(); }
  if (tab === "video") vidJobs();
  if (tab === "translate") { trnLoad(); $("trn-input").focus(); }
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-header .st-tab")) b.onclick = () => showTab(b.dataset.tab);
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

let health = {model: "xestrata", images: false, max_context: 0};
// the texts that depend on what this server offers: the empty chat's line and the attach button's hint
function renderEmptyChat() {
  $("attach-btn").title = health.images ? tr("composer.attachPictures", "Attach a text file or a picture (or drop it here)")
                                        : tr("composer.attach", "Attach a text file (or drop it here)");
  // with the chat's web search on, the search words go to the search engine and the pages it reads are fetched
  const searching = health.web_search && ($("chat-search").value || store.get("chat_search", "on")) !== "off";
  $("chat-empty-sub").textContent = searching
    ? tr("chat.emptyModelSearch", "{model} runs on this PC. With web search on, the search words go to the search engine and the pages it reads are fetched; nothing else leaves it.", {model: health.model})
    : tr("chat.emptyModel", "{model} runs on this PC. Nothing leaves it.", {model: health.model});
}
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    renderEmptyChat();
    paintCtxRing();                      // the model's context is known now
    $("tab-btn-images").hidden = !health.image_generation;
    $("img-note").hidden = !health.image_note;
    $("img-note").textContent = health.image_note || "";
    imgModels();
    thinkingButtons();
    if (["#images", "#tools"].includes(location.hash) && health.image_generation) showTab(location.hash.slice(1));
    $("tab-btn-translate").hidden = !health.translation;
    if (location.hash === "#translate" && health.translation) showTab("translate");
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
    o.textContent = sw.choices[k].replace(/ \(Strata SYCL engine\)$/, "");   // the label upstream's switch gives
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

function setPill(state, text, force = false, lanes = []) {
  if (switching && !force) return;   // while the models swap, the menu owns the pill
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
  $("pill").title = text;
  // the marks: one for each thing that runs (pillLanes); the sentence stays for a screen reader and the title
  const marks = lanes.map(([name, value]) => `<span class="pill-lane">${icon(name, "st-icon")}${esc(value)}</span>`).join("");
  $("pill").classList.toggle("has-lanes", lanes.length > 0);
  if ($("pill-lanes").innerHTML !== marks) $("pill-lanes").innerHTML = marks;
  // a poll re-sets the sentence every second; refit only when its text changed
  if (text + marks !== lastPillText) { lastPillText = text + marks; fitHeader(); }
}
let lastPillText = null;
// What runs at once, for the pill: the text model's turn (its tok/s, or how much of the prompt is read), the requests
// waiting for it, an image job (its step). Each is [icon, value, its sentence]; none: the pill says Idle or Unloaded.
function pillLanes(live, localLoad) {
  const lanes = [];
  if (modelOp === "load" || localLoad) lanes.push(["chat", "…", tr("model.loading", "Loading…")]);
  else if (modelOp === "unload") lanes.push(["chat", "…", tr("model.unloading", "Unloading…")]);
  else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    lanes.push(["chat", pct != null ? `${pct}%` : "…", pct != null ? tr("pill.readingPct", "Reading prompt · {pct}%", {pct}) : tr("pill.reading", "Reading prompt")]);
  } else if (live.state === "generating") {
    lanes.push(["chat", `${fmt(live.tok_s, 1)} tok/s`, tr("pill.generating", "Generating · {rate} tok/s", {rate: fmt(live.tok_s, 1)})]);
  }
  if (live.queued > 0) lanes.push(["clock", String(live.queued), tr("pill.queued", "{n} queued", {n: live.queued})]);
  if (live.video) {   // a video job (it is the image engine's job too)
    const v = live.video;
    lanes.push(["image", `${v.done}/${v.total}`, tr("vid.pill", "Video · frame {i} of {n}", {i: v.done, n: v.total})]);
  } else if (live.image) {
    const img = live.image;
    lanes.push(["image", img.stage === "sampling" ? `${img.step}/${img.steps}` : "…", tr("img.pill", "Image · {step}", {step: imageStep(img)})]);
  } else if (imageBusy) {
    lanes.push(["image", "…", tr("pill.image", "Generating image…")]);
  }
  return lanes;
}
// the pill's list (a tap on it): each thing that runs on a row, the answers this page writes (a click opens the chat),
// and the Monitor for the rest
let pillLive = null;
function renderPillPop() {
  const pop = $("pill-pop");
  if (pop.hidden || !pillLive) return;
  const {lanes, idle} = pillLive;
  const row = (name, html) => `<div class="pill-pop__row">${icon(name, "st-icon")}<span>${html}</span></div>`;
  let html = lanes.length ? lanes.map(([name, , sentence]) => row(name, esc(sentence) + (name === "image" ? ` <span class="muted">· ${esc(imageWhere())}</span>` : ""))).join("")
                          : row("chat", esc(idle));
  if (replies.size) {
    const items = [...replies.keys()].map((id) => `<li><button type="button" data-pill-chat="${esc(id)}">${esc(chatName(id))}</button></li>`).join("");
    html += row("edit", `<b>${esc(tr("pill.thisPage", "Answers this page is writing"))}</b><ul class="pill-pop__chats">${items}</ul>`);
  }
  html += `<div class="pill-pop__foot"><button type="button" class="st-btn st-btn--secondary" data-pill-monitor>${esc(tr("pill.monitor", "Details in Monitor"))}</button></div>`;
  if (pop.innerHTML !== html) pop.innerHTML = html;
}
// where image jobs run (imageGpu), in words
function imageWhere() {
  const g = imageGpu();
  return g === "separate" ? tr("pill.imageSeparate", "on a GPU of its own, beside the chat")
       : g === "helper" ? tr("pill.imageHelper", "on the text model's second GPU, beside the chat")
       : tr("pill.imageShared", "on the text model's GPU, in its turn");
}
function openPill(open) {
  $("pill-pop").hidden = !open;
  $("pill").setAttribute("aria-expanded", String(open));
  if (open) renderPillPop();
}
$("pill").onclick = (e) => { e.stopPropagation(); openPill($("pill-pop").hidden); };
$("pill-pop").addEventListener("click", (e) => {
  e.stopPropagation();
  const c = e.target.closest("[data-pill-chat]");
  if (c) { openPill(false); showTab("chat"); if (c.dataset.pillChat !== "single") openChat(c.dataset.pillChat); return; }
  if (e.target.closest("[data-pill-monitor]")) { openPill(false); showTab("monitor"); }
});
document.addEventListener("click", () => { if (!$("pill-pop").hidden) openPill(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("pill-pop").hidden) { openPill(false); $("pill").focus(); } });
// The pill's sentence is the first thing the header row can spare when it runs out of room.
// the header row as far as it fits: the pill's text goes first, then the name (XeStrata), before the tabs scroll
function fitHeader() {
  const header = document.querySelector(".st-header");
  if (!header) return;
  const tabs = header.querySelector(".st-tabs");
  const over = () => header.scrollWidth > header.clientWidth || (tabs && tabs.scrollWidth > tabs.clientWidth + 1);
  header.classList.remove("st-header--clip", "st-header--noname");
  if (over()) header.classList.add("st-header--clip");
  if (over()) header.classList.add("st-header--noname");
}
window.addEventListener("resize", fitHeader);

// ------------------------------------------------------------------ Load / Unload (POST /load, /unload: About's button)
// Metrics stay "unloaded" until the engine is READY, so a turn or an image job on this page has to paint Loading itself.
// modelOp covers the button's own POST, which returns before the next poll can agree.
let modelOp = null;
let imageBusy = false;
// where the image jobs run (/metrics engine.image_gpu): "separate" (a card of their own) and "helper" (the text model's
// second card, its helper expert cache closed for the job) beside the chat, "shared" on the text model's card in turn
function imageGpu() { return (lastMetrics && lastMetrics.engine && lastMetrics.engine.image_gpu) || "shared"; }
const imageBeside = () => imageGpu() === "separate" || imageGpu() === "helper";
// an image job holds the text model's Load / Unload, unless it runs on a card of its own: on the text model's card it has
// the GPU's turn, on the helper card the engine's helper
const imageHoldsModel = () => imageGpu() !== "separate";
function pageBusy() { return replies.size > 0 || (imageBusy && imageHoldsModel()); }
// live.image (the image engine's job, on the metrics) is the job a refresh can see. imageBusy is this page's POST before
// that job is published. On a shared card a chat phase already on the metrics wins, because Stop still stops that reply.
function imageLive(live) {
  if (live && live.image) return true;
  const state = live && live.state;
  if (!imageBeside() && (state === "reading" || state === "generating")) return false;
  return imageBusy;
}
const imageBlocksModel = (live) => imageHoldsModel() && imageLive(live);
function paintLive() { if (lastMetrics) render(lastMetrics); }
function turnBlocksUnload(live) {
  const state = (live && live.state) || "";
  return pageBusy() || imageBlocksModel(live) || state === "reading" || state === "generating" || (live && live.queued > 0);
}
function setModelButton(text, title, disabled) {
  const btn = $("unload-btn");
  btn.textContent = text; btn.title = title; btn.disabled = disabled;
}
function modelButton(live) {
  if (modelOp === "load") return setModelButton(tr("model.loading", "Loading…"), tr("model.loadingTitle", "The model is loading"), true);
  if (modelOp === "unload") return setModelButton(tr("model.unloading", "Unloading…"), tr("model.unloadingTitle", "The model is unloading"), true);
  const state = (live && live.state) || "";
  const metricsBusy = state === "reading" || state === "generating" || (live && live.queued > 0);
  const imageNow = imageBlocksModel(live);
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
    ? (imageBlocksModel(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.replyLoadingDot", "A reply is already loading the model."))
    : (imageBlocksModel(sample) ? tr("model.imageBusy", "An image is generating. It finishes on its own.") : tr("model.stopFirstDot", "Stop the reply, then unload."));
  if (loading ? pageBusy() || imageBlocksModel(live) : turnBlocksUnload(live)) {
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
  if (loadingNow ? pageBusy() || imageBlocksModel(now) : turnBlocksUnload(now)) {
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
      toast("warn", tr("model.busy", "Model is busy"), imageBlocksModel(sample)
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
  const localLoad = live.state === "unloaded" && replies.size > 0;
  // everything that runs at once, side by side: the text model's turn, the requests waiting, an image job (pillLanes);
  // the dot's colour is the busiest of them
  const lanes = pillLanes(live, localLoad);
  const idle = live.state === "unloaded" ? tr("pill.unloaded", "Unloaded") : tr("pill.idle", "Idle");
  const state = lanes.some((l) => l[0] === "image") || live.state === "generating" ? "generating"
              : live.state === "reading" ? "reading" : live.queued > 0 ? "queued" : "idle";
  setPill(state, lanes.length ? lanes.map((l) => l[2]).join(" · ") : idle, false, lanes.map((l) => [l[0], l[1]]));
  pillLive = {lanes, idle};
  renderPillPop();
  chatVideoNote(live.video);
  modelButton(live);
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
  // a kernel whose xe driver lacks the fix of CVE-2026-68264 (serve/kernel_check.py); said once a release
  if (m.xe_kernel && store.get("xekernel-told", "") !== m.xe_kernel) {
    store.set("xekernel-told", m.xe_kernel);
    toast("warn", tr("xekernel.toast", "Update the Linux kernel"), tr("xekernel.toastText", "Linux {version} lacks the fix of the xe driver's CVE-2026-68264: a GPU memory bind that is retried can crash the driver. Update the kernel. A container (Podman, incus) runs on the host's kernel, so update the host's.", {version: m.xe_kernel}), 15000,
          {label: tr("show", "Show"), run: () => showTab("about")});
  }
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, fail);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "about") renderAbout(eng, hw, st, m.xe_kernel);
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
// ------------------------------------------------------------------ the GPU tabs (a model on several cards, the images on their own)
// hardware.gpus: every card the server's engines run on, with its roles (server.py gpu_roles) and what it does now (active)
let gpuTab = String(store.get("gpu-tab", "all"));
const ROLES = {text: () => tr("gpu.text", "Text"), image: () => tr("gpu.image", "Image"), helper: () => tr("gpu.helper", "Helper"),
               prefill: () => tr("gpu.prefill", "Prefill"), decode: () => tr("gpu.decode", "Decode")};
const ROLE_TITLES = {helper: () => tr("gpu.helperTitle", "A helper expert cache of the text model; closed while an image job runs on this card")};
const total = (v) => (v.length ? v.reduce((x, y) => x + y, 0) : null);
// every card together: the load's mean, the memory's, power's and link traffic's sum, the hottest temperature
const GPU_FOLD = {util: (v) => (v.length ? total(v) / v.length : null), mem_used: total, mem_total: total, power: total, power_limit: total,
                  pcie_rx_mb: total, temp: (v) => (v.length ? Math.max(...v) : null), mem_temp: (v) => (v.length ? Math.max(...v) : null)};
const GPU_SERIES = ["util", "mem_used", "temp", "power", "pcie_rx_mb"];
function gpuFold(rows) {
  const out = {};
  for (const [k, f] of Object.entries(GPU_FOLD)) out[k] = f(rows.map((r) => r[k]).filter((x) => x != null));
  return out;
}
// the readings and the history the GPU cards show: {g, hist, cards, one (the tab's card), all (the All tab)}
function gpuView(hw, h) {
  const cards = hw.gpus || [];
  if (cards.length < 2) {                      // one card (or an older server): the flat readings
    const g = {};
    for (const k of [...Object.keys(GPU_FOLD), "pcie_gen", "pcie_gen_max", "pcie_own_gen", "pcie_width", "temp_may_be_stale"]) g[k] = hw[`gpu_${k}`];
    const hist = Object.fromEntries(GPU_SERIES.map((k) => [k, h[`gpu_${k}`]]));
    return {g, hist, cards, one: null, all: false};
  }
  const hs = h.gpus || [];
  const one = cards.find((c) => String(c.index) === gpuTab);
  if (one) return {g: one, hist: hs[cards.indexOf(one)] || {}, cards, one, all: false};
  const g = gpuFold(cards);
  const gen = (c) => c.pcie_gen_max || c.pcie_gen || 99;
  const slow = cards.filter((c) => c.pcie_gen_max || c.pcie_gen).sort((x, y) => gen(x) - gen(y))[0] || {};   // the slowest link
  Object.assign(g, {pcie_gen: slow.pcie_gen, pcie_gen_max: slow.pcie_gen_max, pcie_own_gen: slow.pcie_own_gen, pcie_width: slow.pcie_width,
                    temp_may_be_stale: cards.every((c) => c.temp == null || c.temp_may_be_stale)});
  const hist = {};
  for (const k of GPU_SERIES) {
    const n = Math.max(0, ...hs.map((x) => (x[k] || []).length));
    hist[k] = Array.from({length: n}, (_, i) => GPU_FOLD[k](hs.map((x) => (x[k] || [])[i]).filter((v) => v != null)));
  }
  return {g, hist, cards, one: null, all: true};
}
// what a card does right now, for its GPU card's line: the chat's prefill or decode, the image job's step
function gpuDoing(c, live) {
  const on = c.active || [];
  const out = [];
  if (on.includes("text") && live.state === "reading") {
    out.push(live.prompt_total ? tr("pill.readingPct", "Reading prompt · {pct}%", {pct: Math.round((100 * live.prompt_read) / live.prompt_total)})
                               : tr("pill.reading", "Reading prompt"));
  } else if (on.includes("text") && live.state === "generating") {
    out.push(tr("pill.generating", "Generating · {rate} tok/s", {rate: fmt(live.tok_s, 1)}));
  }
  if (on.includes("image") && live.image) out.push(tr("img.pill", "Image · {step}", {step: imageStep(live.image)}));
  return out.join(" · ");
}
function gpuBadges(c) {
  const on = c.active || [];
  const badge = (text, lit, title) => `<span class="st-badge${lit ? " st-badge--generating on" : ""}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
  const out = (c.roles || []).map((r) => badge(ROLES[r] ? ROLES[r]() : r, on.includes(r), ROLE_TITLES[r] ? ROLE_TITLES[r]() : ""));
  if (c.stage != null) out.push(badge(tr("gpu.stage", "Stage {n}", {n: c.stage + 1}), false, tr("gpu.stageTitle", "Its share of the layers")));
  if (c.replica != null) out.push(badge(tr("gpu.replica", "Replica {n}", {n: c.replica + 1}), false, ""));
  return out.join("");
}
let gpuTabsStamp = "";
function renderGpuTabs(cards) {
  const el = $("gpu-tabs");
  el.hidden = cards.length < 2;
  if (cards.length < 2) return;
  const stamp = JSON.stringify([cards.map((c) => [c.index, c.name, c.pci, c.roles, c.active, c.stage, c.replica]), gpuTab, document.documentElement.lang]);
  if (stamp === gpuTabsStamp) return;          // rebuilt only when it changes, so a focused tab keeps its focus
  gpuTabsStamp = stamp;
  const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.gpu : null;
  const tab = (key, label, badges, title) => `<button class="st-tab" role="tab" type="button" data-gpu="${esc(key)}" aria-selected="${gpuTab === key}"${
    title ? ` title="${esc(title)}"` : ""}><span>${esc(label)}</span>${badges}</button>`;
  el.innerHTML = tab("all", tr("gpu.all", "All GPUs"), "", "") +
    cards.map((c) => tab(String(c.index), `GPU ${c.index}`, gpuBadges(c), [c.name, c.pci].filter(Boolean).join(" · "))).join("");
  for (const b of el.querySelectorAll("[data-gpu]")) {
    b.onclick = () => { gpuTab = b.dataset.gpu; store.set("gpu-tab", gpuTab); paintLive(); };
    if (b.dataset.gpu === focused) b.focus({preventScroll: true});
  }
}

// an image job's line on the state card: [label, detail, progress %]
function imageState(img) {
  const label = img.label + (img.images > 1 ? ` · ${tr("img.nOf", "image {i} of {n}", {i: img.image, n: img.images})}` : "");
  const detail = [tr("img.step", "step {i}/{n}", {i: img.step, n: img.steps}), img.step_s != null ? `${fmt(img.step_s, 2)} s/step` : null,
                  `${fmt(img.elapsed_s, 0)} s`, img.size,
                  img.refs ? tr("img.refs", {one: "{n} reference", other: "{n} references"}, {n: img.refs}) : null]
    .filter(Boolean).join(" · ");
  return [label, detail, 100 * (img.progress || 0)];
}

// The run's average speed from the totals: tokens over milliseconds (token-weighted, so a short request does not
// outweigh a long one).  Null when the totals do not hold both numbers yet.
function avgSpeed(ms, tokens) {
  return typeof ms === "number" && ms > 0 && typeof tokens === "number" && tokens > 0 ? tokens / (ms / 1000) : null;
}

function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, fail) {
  for (const m of METRICS) $(`ml-${m.key}`).textContent = m.label();
  // model state. An image job on a card of its own (or the helper card) runs beside the chat: the row under it says
  // its state, and the state above stays the chat's. On the text model's card the job takes the state in its turn.
  // A local turn keeps metrics on "unloaded" for the whole engine start.
  const beside = imageBeside();
  const image = beside ? null : live.image;
  const imageNow = !beside && imageLive(live);
  const starting = modelOp === "load" || (replies.size > 0 && live.state === "unloaded");
  const on = !modelOp && live.queued > 0 ? "queued" : image ? "generating" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) {
    const lit = (b.dataset.s === on || (!image && b.dataset.s === live.state)) && !(starting && b.dataset.s === "unloaded");
    b.classList.toggle("on", lit);
  }
  const prog = $("state-progress");
  let label = tr("mon.waiting", "Waiting for a request"), detail = "", pct = 0;
  if (image) {
    [label, detail, pct] = imageState(image);
    delete prog.dataset.tone;
  } else if (modelOp === "load" || (replies.size > 0 && live.state === "unloaded")) {
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
  const side = beside && (live.image || imageBusy);
  $("state-image").hidden = !side;
  if (side) {
    const [l, d, p] = live.image ? imageState(live.image) : [tr("img.generating", "Generating image"), "", 0];
    $("state-image-label").textContent = `${tr("gpu.image", "Image")} · ${l}`;
    $("state-image-detail").textContent = d;
    $("state-image-bar").style.width = `${p}%`;
  }

  // the eight cards
  // Idle: not the last request's speed (one request is noisy) but the run's average from the server's totals
  // (token-weighted, the same figure the engine's DONE lines and Prometheus report).  Live stays live.
  const avgDecode = avgSpeed(totals && totals.decode_ms, totals && totals.output_tokens);
  const readTokens = totals ? Math.max(0, (totals.prompt_tokens || 0) - (totals.reused || 0)) : 0;
  const avgPrefill = avgSpeed(totals && totals.prompt_ms, readTokens);
  if (image) {   // an image job in the text model's turn: its seconds a step, not a decode rate
    setMetric("speed", live.image.step_s == null ? null : fmt(live.image.step_s, 2), "s/step",
              tr("img.pill", "Image · {step}", {step: imageStep(live.image)}));
  } else {
    const speed = live.state === "generating" ? live.tok_s : avgDecode != null ? avgDecode : last ? last.decode_tok_s : null;
    setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
              live.state === "generating" ? tr("mon.decodeNow", "Decode now") : avgDecode != null ? tr("mon.decodeAvg", "Decode · run average")
              : last ? tr("mon.decodeLast", "Decode last request") : tr("mon.decode", "Decode"));
  }
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : avgPrefill != null ? avgPrefill : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  if (image) setMetric("prefill", null, "t/s", tr("img.job", "Image job"));
  else setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
                 live.state === "reading" ? tr("mon.prefillNow", "Prefill now") : live.state === "generating" ? tr("mon.prefillThis", "Prefill this request")
                 : avgPrefill != null ? tr("mon.prefillAvg", "Prefill · run average")
                 : last ? tr("mon.prefillLast", "Prefill last request") : tr("mon.prefill", "Prefill"));
  spark("sp-speed", h.tok_s, undefined, true);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // the GPU cards: the one card, or the tab chosen above them (every card together, or one of them; issue #112)
  const {g, hist, cards, one, all: allCards} = gpuView(hw, h);
  renderGpuTabs(cards);
  const per = (f) => cards.map((c) => `GPU ${c.index} ${f(c)}`).join(" · ");
  setMetric("gpu", g.util == null ? null : fmt(g.util), "%",
            allCards ? per((c) => (c.util == null ? "–" : `${fmt(c.util)}%`)) : one ? gpuDoing(one, live) || one.name || "" : st.gpu_name || "");
  spark("sp-gpu", hist.util, 100);
  // the expert cache's slots are the text model's first card's
  const firstText = !one || ((one.roles || []).includes("text") && !one.stage);
  setMetric("vram", g.mem_used == null ? null : gb(g.mem_used), g.mem_total ? `/ ${gb(g.mem_total, 0)} GB` : "GB",
            allCards ? per((c) => (c.mem_used == null ? "–" : `${gb(c.mem_used)} GB`))
                : firstText && eng.expert_slots ? tr("mon.expertsCached", "{n} experts cached", {n: fmt(eng.expert_slots)}) : "");
  spark("sp-vram", hist.mem_used, g.mem_total);
  // an idle GPU on the xe driver may report its last temperature, not its present one (Battlemage's hwmon): dimmed
  const stale = Boolean(g.temp_may_be_stale) && g.temp != null;
  setMetric("temp", g.temp == null ? null : fmt(g.temp), "°C",
            stale ? tr("mon.tempStale", "GPU idle (the reading may be old)")
            : allCards ? per((c) => (c.temp == null ? "–" : `${fmt(c.temp)}°`))
            : g.mem_temp == null ? "" : `VRAM ${fmt(g.mem_temp)} °C`);
  $("mv-temp").classList.toggle("is-stale", stale);
  spark("sp-temp", hist.temp, 90);
  setMetric("power", g.power == null ? null : fmt(g.power), "W",
            allCards ? per((c) => (c.power == null ? "–" : `${fmt(c.power)} W`))
            : g.power_limit ? tr("mon.powerLimit", "of {n} W limit", {n: fmt(g.power_limit)}) : "");
  spark("sp-power", hist.power, g.power_limit);
  const link = (c) => c.pcie_gen_max || c.pcie_gen;
  const gen = link(g);
  setMetric("pcie", gen ? `Gen${gen}` : null, g.pcie_width ? `x${g.pcie_width}` : "",
            allCards ? per((c) => (link(c) ? `Gen${link(c)}${c.pcie_width ? ` x${c.pcie_width}` : ""}` : "–"))
            : (g.pcie_rx_mb == null ? "" : tr("mon.toGpu", "to GPU {rate} MB/s", {rate: fmt(g.pcie_rx_mb, g.pcie_rx_mb < 10 ? 1 : 0)})) +
              (g.pcie_own_gen && gen && g.pcie_own_gen < gen ? ` · ${tr("mon.nowGen", "now Gen{n}", {n: g.pcie_own_gen})}` : ""));
  spark("sp-pcie", hist.pcie_rx_mb);
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
  $("temp-text").textContent = g.temp == null ? "–" : `${fmt(g.temp)} °C`;     // the GPU tab's card, or the hottest
  $("temp-bar").style.width = g.temp == null ? "0%" : `${Math.min(100, g.temp)}%`;

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
// How the GPUs share the work (/metrics hardware.gpus: each card's roles, layer split stage and replica, from
// serve/server.py's gpu_roles; engine.image_gpu: where the image jobs run): a sentence, then a row for each card
function gpuPlanRows(eng, hw, st) {
  const title = tr("about.gpuPlan", "GPUs");
  const images = Array.isArray(eng.image_models) && eng.image_models.length > 0, where = eng.image_gpu || "";
  let cards = Array.isArray(hw.gpus) ? hw.gpus : [];
  if (!cards.length) {                       // one card (or an older server): the text model's, the images on it too
    if (!st.gpu_name) return [];
    cards = [{index: 0, name: st.gpu_name, mem_total: hw.gpu_mem_total, roles: ["text", ...(images ? ["image"] : [])]}];
  }
  const has = (r) => cards.filter((c) => (c.roles || []).includes(r));
  const text = has("text"), stages = new Set(text.map((c) => c.stage).filter((x) => x != null)).size;
  const replicas = new Set(text.map((c) => c.replica).filter((x) => x != null)).size;
  const parts = [];
  if (replicas > 1) parts.push(tr("about.gpuPlanReplicas", "the text model runs as {n} copies, the requests shared between them", {n: replicas}));
  else if (has("prefill").length && has("decode").length) parts.push(tr("about.gpuPlanSplitServe", "prefill and decode on different GPUs"));
  else if (stages > 1) parts.push(tr("about.gpuPlanStages", "the text model's layers shared by {n} GPUs", {n: stages}));
  if (images && where === "separate" && !parts.length && text.length === 1) {
    parts.push(tr("about.gpuPlanApart", "text and images on GPUs of their own, side by side"));
  } else if (images) {
    parts.push(where === "separate" ? tr("about.gpuPlanSeparate", "the images on a GPU of their own, beside the text")
             : where === "helper" ? tr("about.gpuPlanHelper", "the images on the text model's helper GPU, lent to them for each image")
             : cards.length === 1 ? tr("about.gpuPlanOneBoth", "text and images on one GPU, in turn")
             : tr("about.gpuPlanShared", "the images on the text model's GPU, in turn"));
  } else if (cards.length === 1) {
    parts.push(tr("about.gpuPlanOneText", "the text model on one GPU"));
  }
  if (!parts.length) parts.push(tr("about.gpuPlanText", "the text model on {n} GPUs", {n: text.length}));
  const sentence = parts.join(tr("about.gpuPlanAnd", "; "));
  const roleName = (c, r) => {
    const name = ROLES[r] ? ROLES[r]() : r;
    if (r !== "text") return r === "helper" && where === "helper" ? tr("about.gpuPlanLent", "{role} (lent to the images)", {role: name}) : name;
    const more = [c.replica != null && replicas > 1 ? tr("gpu.replica", "Replica {n}", {n: c.replica + 1}) : null,
                  c.stage != null && stages > 1 ? tr("gpu.stage", "Stage {n}", {n: c.stage + 1}) : null].filter(Boolean);
    return more.length ? tr("about.gpuPlanRole", "{role} ({more})", {role: name, more: more.join(tr("about.gpuPlanComma", ", "))}) : name;
  };
  return [[title, sentence], ...cards.map((c) => ["", `GPU ${c.index ?? 0} · ${String(c.name || "").replace(/ at [0-9a-f:.]+$/i, "")}` +
    `${c.mem_total ? `, ${gb(c.mem_total, 0)} GB` : ""}: ${(c.roles || []).filter((r) => r !== "text" || !(c.roles || []).some((x) => x === "prefill" || x === "decode"))
      .map((r) => roleName(c, r)).join(tr("about.gpuPlanComma", ", "))}`])];
}
function renderAbout(eng, hw, st, xeKernel) {
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
    ...gpuPlanRows(eng, hw, st),
    [tr("about.vision", "Reading images"), visionText(eng)],
    ...imageModelRows(eng.image_models),
    [tr("about.projection", "Experimental speed projection"), projectionText(eng.cvec)],
    [tr("igc.label", "Intel compiler"), eng.igc_workaround && eng.igc_workaround !== "none" ? tr("igc.about", "IGC {version} has a bug (2.27 to 2.30 miscompile some reads for Xe2 and later GPUs); XeStrata works around it, the decode of short prompts up to 10% slower. For full speed, use Fedora 44 (its IGC is 2.32 or later), or install with Podman (Fedora 44 inside).", {version: eng.igc_workaround}) : null],
    [tr("xekernel.label", "Linux kernel"), xeKernel ? tr("xekernel.about", "Linux {version} lacks the fix of the xe driver's CVE-2026-68264 (a GPU memory bind that is retried can crash the driver). Update the kernel; a container (Podman, incus) runs on the host's kernel.", {version: xeKernel}) : null],
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
// ------------------------------------------------------------------ About > Model terms, Licenses (GET /v1/licenses/*)
for (const b of document.querySelectorAll("#about-tabs [data-about]")) b.onclick = () => showTab(b.dataset.about === "overview" ? "about" : `about/${b.dataset.about}`);
function showAboutSub() {
  for (const b of document.querySelectorAll("#about-tabs [data-about]")) b.setAttribute("aria-selected", String(b.dataset.about === aboutSub));
  for (const k of ["overview", "terms", "licenses"]) $(`about-${k}`).hidden = k !== aboutSub;
  if (aboutSub === "terms") loadTerms();
  if (aboutSub === "licenses") loadLicenses();
}
// one license text, folded: its name and where it is published
function textBlock(t) {
  const link = t.url ? ` · <a href="${esc(t.url)}" target="_blank" rel="noopener noreferrer">${esc(tr("terms.original", "original"))}</a>` : "";
  return t.text != null
    ? `<details><summary>${esc(t.name)}${t.source ? ` (${esc(t.source)})` : ""}</summary><pre>${esc(t.text)}</pre></details>${link ? `<p class="muted small">${link.slice(3)}</p>` : ""}`
    : `<p class="muted small">${esc(t.name)}: ${esc(tr("terms.notHere", "not on this PC yet"))}${link}</p>`;
}
const TERM_ROLES = {model: () => tr("terms.roleModel", "Text model"), mmproj: () => tr("terms.roleMmproj", "Picture reader of the text model"),
  image: () => tr("terms.roleImage", "Image model"), tool: () => tr("terms.roleTool", "Image tool"), lora: () => tr("terms.roleLora", "LoRA"),
  translation: () => tr("terms.roleTranslation", "Translation model"), speed_projection: () => tr("about.projection", "Experimental speed projection")};
let termsTimer = null;
async function loadTerms() {
  let got;
  try {
    const r = await fetch("v1/licenses/models", {headers: headers()});
    if (!r.ok) return;
    got = await r.json();
  } catch (e) { return; }
  const el = $("terms-list");
  el.innerHTML = (got.models || []).map((m) => {
    const role = (TERM_ROLES[m.role] || (() => m.role))();
    if (m.own) {
      return `<div class="st-card"><span class="card-title">${esc(role)} · ${esc(m.name)}${m.model ? ` (${esc(m.model)})` : ""}</span>
        <p class="muted">${esc(tr("terms.own", "A file of your own: its terms are those of whoever made it."))}</p>
        ${m.source ? `<p class="small">${esc(tr("terms.fetchedFrom", "Fetched from"))} <a href="${esc(m.source)}" target="_blank" rel="noopener noreferrer">${esc(m.source)}</a></p>` : ""}</div>`;
    }
    const head = [m.title, m.tool, m.model].filter(Boolean).join(" · ");
    const rows = [
      [tr("terms.repo", "Published as"), m.repo],
      [tr("terms.by", "By"), m.maker || m.by],
      [tr("terms.revision", "Revision"), m.revision ? String(m.revision).slice(0, 12) : null],
      [tr("terms.license", "License"), m.license],
      [tr("terms.cardLicense", "The model card's license field"), m.card_license],
      [tr("terms.agreed", "You agreed to it"), m.agreed],
    ].filter((r) => r[1]);
    return `<div class="st-card"><span class="card-title">${esc(role)}${head ? ` · ${esc(head)}` : ""}</span>
      <dl class="facts">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      ${m.card ? `<p class="small"><a href="${esc(m.card)}" target="_blank" rel="noopener noreferrer">${esc(tr("terms.card", "The model card"))}</a></p>` : ""}
      ${(m.texts || []).map(textBlock).join("") || `<p class="muted small">${esc(tr("terms.noText", "Its publisher gives no text of the license beside the name above."))}</p>`}</div>`;
  }).join("") || `<p class="muted">${esc(tr("terms.none", "No model is set up."))}</p>`;
  clearTimeout(termsTimer);
  if (got.fetching && tab === "about" && aboutSub === "terms") termsTimer = setTimeout(loadTerms, 5000);   // the texts being fetched
}
let licensesLoaded = false;
async function loadLicenses() {
  if (licensesLoaded) return;
  let got;
  try {
    const r = await fetch("v1/licenses/software", {headers: headers()});
    if (!r.ok) return;
    got = await r.json();
  } catch (e) { return; }
  licensesLoaded = true;
  const part = (p) => {
    const rows = [[tr("terms.version", "Version"), p.version], [tr("terms.license", "License"), p.license],
                  [tr("lic.carried", "In this build"), p.bundled === false ? tr("lic.notCarried", "not carried: opened where you installed it, under its maker's terms") : null]]
      .filter((r) => r[1]);
    return `<div class="st-card"><span class="card-title">${esc(p.name)}</span>
      <dl class="facts">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      ${p.source ? `<p class="small"><a href="${esc(p.source)}" target="_blank" rel="noopener noreferrer">${esc(p.source)}</a></p>` : ""}
      ${(p.texts || []).map(textBlock).join("")}</div>`;
  };
  $("lic-list").innerHTML = part(got.xestrata) +
    `<h3>${esc(tr("lic.free", "Free software"))}</h3>` + (got.free || []).map(part).join("") +
    ((got.nonfree || []).length ? `<h3>${esc(tr("lic.nonfree", "Not free software"))}</h3>` + got.nonfree.map(part).join("") : "");
}

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

// ------------------------------------------------------------------ Skills (GET / POST /skills, serve/skills.py)
// Agent Skills' SKILL.md files the user adds here (kept in the server's data folder, the same from every browser) and
// XeStrata's own (the image tools'): each switched on or off; the user's added, edited, imported and deleted.  The
// chat sends "xestrata_skills": true while one of the user's is on: the model sees their names and reads the bodies it
// needs with load_skill.
let skillsInfo = {skills: [], broken: []};
async function loadSkills() {
  try {
    const r = await fetch("skills", {headers: headers()});
    if (!r.ok) return;
    skillsInfo = await r.json();
  } catch (e) { return; /* an older server: no skills */ }
  renderSkills();
}
const userSkillsOn = () => (skillsInfo.skills || []).some((s) => !s.builtin && s.on);
async function skillsPost(body) {
  const r = await fetch("skills", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
  skillsInfo = j;
  renderSkills();
  return j;
}
function skillRow(s) {
  const row = document.createElement("div");
  row.className = "skill-row";
  row.innerHTML = `<div class="skill-row__text"><span class="skill-row__name"></span><span class="muted small"></span>` +
    `<div class="skill-row__acts"></div></div><button class="st-toggle" role="switch"></button>`;
  row.querySelector(".skill-row__name").textContent = s.name;
  row.querySelector(".muted").textContent = s.description;
  const sw = row.querySelector(".st-toggle");
  sw.setAttribute("aria-checked", String(!!s.on));
  sw.setAttribute("aria-label", tr("skills.switch", "Use {name}", {name: s.name}));
  sw.onclick = async () => {
    try { await skillsPost({action: "switch", name: s.name, on: !s.on}); } catch (e) { toast("error", s.name, String(e.message || e), 6000); }
  };
  const acts = row.querySelector(".skill-row__acts");
  const btn = (label, fn) => { const b = document.createElement("button"); b.type = "button"; b.className = "st-btn st-btn--secondary"; b.textContent = label; b.onclick = fn; acts.appendChild(b); };
  btn(s.builtin ? tr("skills.view", "View") : tr("skills.edit", "Edit"), () => openSkill(s.name, s.builtin));
  if (!s.builtin) btn(tr("skills.delete", "Delete"), async () => {
    if (!confirm(tr("skills.deleteAsk", "Delete the skill {name}?", {name: s.name}))) return;
    try { await skillsPost({action: "delete", name: s.name}); } catch (e) { toast("error", s.name, String(e.message || e), 6000); }
  });
  return row;
}
function renderSkills() {
  const all = skillsInfo.skills || [];
  const mine = all.filter((s) => !s.builtin), own = all.filter((s) => s.builtin);
  const list = $("skills-list");
  list.replaceChildren(...mine.map(skillRow));
  for (const b of skillsInfo.broken || []) {
    const e = document.createElement("div");
    e.className = "msg-error small";
    e.textContent = `${b.name}: ${b.error}`;
    list.appendChild(e);
  }
  if (!mine.length && !(skillsInfo.broken || []).length) {
    const e = document.createElement("span");
    e.className = "muted small";
    e.textContent = tr("skills.none", "No skill is added yet.");
    list.appendChild(e);
  }
  $("skills-own-list").replaceChildren(...own.map(skillRow));
  $("skills-own-sum").textContent = tr("skills.ownN", "XeStrata's skills ({n}; they can be switched off)", {n: own.length});
  $("skills-dir").textContent = skillsInfo.dir ? tr("skills.dir", "Kept in {dir}", {dir: skillsInfo.dir}) : "";
}
const SKILL_TEMPLATE = "---\nname: my-skill\ndescription: What it does, and when the model should read it.\n---\n" +
  "Write the instructions here.\n";
async function openSkill(name, builtin) {
  let text = SKILL_TEMPLATE;
  if (name) {
    try {
      const r = await fetch("skills/" + encodeURIComponent(name), {headers: headers()});
      const j = await r.json();
      if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
      text = j.text;
    } catch (e) { toast("error", name, String(e.message || e), 6000); return; }
  }
  const d = document.createElement("dialog");
  d.className = "attach-preview skill-editor";
  d.innerHTML = `<div class="attach-preview__head"><span class="attach-preview__name"></span></div>` +
    `<div class="attach-preview__body"><textarea spellcheck="false"></textarea><p class="msg-error small" hidden></p></div>` +
    `<div class="attach-preview__foot"><button type="button" class="st-btn st-btn--primary" data-save></button> <button type="button" class="st-btn" data-close></button></div>`;
  d.querySelector(".attach-preview__name").textContent = name ? name + (builtin ? " · " + tr("skills.readOnly", "XeStrata's, read only") : "") : tr("skills.add", "Add a skill");
  const ta = d.querySelector("textarea");
  ta.value = text;
  ta.readOnly = !!builtin;
  ta.setAttribute("aria-label", "SKILL.md");
  const save = d.querySelector("[data-save]");
  save.textContent = tr("skills.save", "Save");
  save.hidden = !!builtin;
  d.querySelector("[data-close]").textContent = tr("close", "Close");
  d.querySelector("[data-close]").onclick = () => d.close();
  save.onclick = async () => {
    const err = d.querySelector(".msg-error");
    try {
      await skillsPost({action: "save", text: ta.value, ...(name ? {old: name} : {})});
      d.close();
    } catch (e) { err.hidden = false; err.textContent = String(e.message || e); }
  };
  d.addEventListener("close", () => d.remove());
  document.body.appendChild(d);
  d.showModal();
}
$("skills-add").onclick = () => openSkill(null, false);
$("skills-import").onclick = () => $("skills-file").click();
$("skills-file").onchange = async () => {
  const f = $("skills-file").files[0];
  $("skills-file").value = "";
  if (!f) return;
  try {
    const buf = new Uint8Array(await f.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    const j = await skillsPost({action: "import", filename: f.name, data: btoa(bin)});
    toast("info", tr("skills.title", "Skills"), tr("skills.imported", "Added {name}", {name: j.saved}), 4000);
  } catch (e) { toast("error", tr("skills.import", "Import a file…"), String(e.message || e), 8000); }
};

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
  // the stored empty value has one spelling in every language: a translated word would come back as that word and
  // be saved as a URL, a name or a number (the selects carry their own value, so their labels may be translated)
  const text = v == null ? "" : v === "" ? "off" : Array.isArray(v) ? v.join(", ") : String(v);
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
  if (k.kind === "bool") return s === "true";
  if (k.kind === "enum") return s;                  // script_guard's own value is "off"
  if (s.toLowerCase() === "off") return "";        // the field's spelling of the stored empty value
  if (k.kind === "number") return Number(s);
  return s;                                         // secret or names (the server splits them at commas)
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
  // a finished ```mermaid block: drawn as its chart (mermaid-view.js), its code a click away
  const chart = complete && /^mermaid$/i.test(lang || "");
  const head2 = chart ? head.replace(`<button class="st-btn st-btn--icon" data-code-copy`,
    `<button class="st-btn st-btn--icon" data-chart-code aria-pressed="false" aria-label="${esc(tr("preview.showCode", "Show the code"))}" title="${esc(tr("preview.showCode", "Show the code"))}">${icon("edit")}</button><button class="st-btn st-btn--icon" data-code-copy`) : head;
  return `<div class="st-code"${chart ? " data-mermaid" : ""}>${head2}<pre><code>${esc(code)}</code></pre>${prev}</div>`;
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
                  mcp: true, math: true, chatList: true, chatEdit: true, whileBusy: "queue", segments: true, segmentLimit: "",
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
if (settings.whileBusy === "wait") settings.whileBusy = "queue";   // "wait" (Send off while it answered) is no more
// before the model's own levels (thinking_v 2) "high" was every model's default and saved as it: the default again
if (settings.thinking_v !== 2 && settings.thinking === "high") settings.thinking = "";
// the math, the saved-chats list and the edit controls were off by default before defaults_v 2, and saved so: on once
if (settings.defaults_v !== 2) settings.math = settings.chatList = settings.chatEdit = true;
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);   // the open chat's messages (chats-ui.js: with the saved-chats list on, also that chat's own)
let attachments = [];                 // {name, url}
let busy = null;                      // the open chat's answer being written: {controller, msg, cut, cutAfterTool, rid, home}
// every chat's answer being written, by chat (generateReply's home.id): another chat can be opened or started meanwhile
const replies = new Map();
let imageAlways = store.get("chat_image_always", false) === true;   // the image switch (paintImageSwitch)
// the messages sent while an answer is written, by chat (generateReply's home.id): [{text, images, files, mode}], the first
// sent first; `pending` is the open chat's, null when it has none (the history's buttons wait for it)
const queues = new Map();
let pending = null;
const chatKey = () => (chat ? chat.id : "single");
function queueOf(id = chatKey()) {
  let q = queues.get(id);
  if (!q) { q = []; queues.set(id, q); }
  return q;
}
function syncPending() { pending = queueOf().length ? queueOf() : null; }
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
      for (const f of m.files) wrap.appendChild(attachChip(f, "attach"));
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (!im.url) { wrap.appendChild(attachChip(im, "image")); continue; }
        const img = document.createElement("img");
        img.src = im.url; img.alt = im.name || tr("chat.image", "image"); img.title = img.alt;
        img.className = "msg-image"; img.tabIndex = 0;
        img.onclick = () => previewAttachment({kind: "image", ...im});
        img.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); img.onclick(); } };
        const frame = document.createElement("span");   // where the marks show (paintMarks)
        frame.className = "msg-image-wrap";
        frame.appendChild(img);
        wrap.appendChild(frame);
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
  let toolArgs = t.arguments;
  if (typeof toolArgs === "string") { try { toolArgs = JSON.parse(toolArgs); } catch (e) { toolArgs = null; } }
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
    imageToolHtml(t, k) +
    // train_lora: the LoRA training panel opens here (ltwHydrate mounts it)
    (t.tool === "train_lora" && t.state === "done" ? `<div class="ltw" data-ltw="${k}" data-ltw-requested="${esc((toolArgs || {}).model || "")}"></div>` : "");
}
// a generate_image call with a layout (MOSS-Image's boxes) also gets a button that opens it in the Images tab
function imageToolHtml(t, k) {
  const urls = (t.images || []).filter((u) => typeof u === "string");
  const a = t.arguments, layout = t.tool === "generate_image" && a && (typeof a === "string" ? a.includes('"layout"') : a.layout);
  const open = layout && t.state === "done" ? `<button type="button" class="st-btn" data-tool-layout="${k}">${esc(tr("img.layoutToImages", "Open the layout in Images"))}</button>` : "";
  const src = (u) => (t.copies && t.copies[u]) || u.replace(/^\//, "");   // the chat's own copy (keepToolCopies) first
  return urls.length || open ? `<div class="msg-images tool-images">${urls.map((u) =>
    `<a href="${esc(src(u))}" target="_blank" rel="noopener" data-gen-image="${esc(u)}"><img src="${esc(src(u))}" alt="${esc(t.tool || "image")}"></a>`).join("")}${open}</div>` : "";
}
// The images an answer's tools made, copied into the chat once it is written: the server keeps only its newest files
// (images' keep_files, the MCP media folder), and a saved chat should still show them. Data URLs, as the pictures sent.
async function keepToolCopies(m) {
  const want = (m.tools || []).flatMap((t) => (t.images || []).map((im) => [t, typeof im === "string" ? im : im.url]))
    .filter(([t, u]) => typeof u === "string" && !(t.copies && t.copies[u]) && (u.startsWith("/v1/images/files/") || safeToolMedia(u)));
  for (const [t, u] of want) {
    try {
      const r = await fetch(u.replace(/^\//, ""), {headers: headers()});
      if (!r.ok) continue;
      const blob = await r.blob();
      const url = await new Promise((ok, no) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.onerror = no; fr.readAsDataURL(blob); });
      t.copies = {...t.copies, [u]: url};
    } catch (e) { /* the server's file stays the only one */ }
  }
  return want.length > 0;
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
    `<figure class="tool-image"><img data-tool-image="${esc(im.url)}"${t.copies && t.copies[im.url] ? ` data-copy="${esc(t.copies[im.url])}"` : ""} alt="Tool image" loading="lazy">` +
    `<figcaption><a class="st-btn" data-tool-download="${esc(im.url)}" download="${esc(im.name || "generated-image.png")}">Download image</a></figcaption></figure>`).join("");
}
function hydrateToolImages(root) {
  root.querySelectorAll("[data-tool-image]").forEach((img) => {
    const url = img.dataset.toolImage, epoch = mediaEpoch;
    if (!safeToolMedia(url)) return;
    if (img.dataset.copy) { img.src = img.dataset.copy; img.closest("figure").querySelector("[data-tool-download]").href = img.dataset.copy; return; }
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
// every answer's charts drawn again (the theme's colors)
function redrawCharts() {
  for (const el of document.querySelectorAll("#chat .st-msg--assistant")) {
    const m = messages[+el.dataset.i];
    if (m) StrataMermaid.hydrate(el.querySelector(".st-bubble"), planSteps(m));
  }
}
// the plan's boxes by their ids (combine-tools: a tool call's `step`) -> how each goes: running, done or error
function planSteps(m) {
  const out = new Map();
  for (const t of m.tools || []) {
    let a = t.arguments;
    if (typeof a === "string") { try { a = JSON.parse(a); } catch (e) { a = null; } }
    const id = a && typeof a.step === "string" ? a.step.trim() : "";
    if (!id) continue;
    out.set(id, t.state === "done" ? "done" : t.state === "error" ? "error" : t.state === "skipped" ? "waiting" : "running");
  }
  return out;
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
    StrataMermaid.hydrate(bubble, planSteps(m));
    hydrateToolImages(bubble);
    ltwHydrate(bubble);
    if (!streaming) paintMarks();
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? tr("chat.stopped", "Stopped") : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
// The open chat's context in the message box: the tokens its last answer left in use (prompt and answer: settleTurn's
// `used`) against its segment's limit (segments on), else the model's context; hidden until an answer has said it
function paintCtxRing() {
  const ring = $("ctx-ring");
  let used = 0;
  for (let i = messages.length - 1; i >= 0 && !used; i--) used = (messages[i].role === "assistant" && messages[i].used) || 0;
  const seg = segmentNow(), limit = (seg && seg.limit) || health.max_context || 0;
  ring.hidden = !used || !limit;
  if (ring.hidden) return;
  const frac = Math.min(1, used / limit), c = 2 * Math.PI * 15;
  $("ctx-ring-fill").setAttribute("stroke-dasharray", `${(frac * c).toFixed(2)} ${c.toFixed(2)}`);
  ring.dataset.level = frac >= 0.9 ? "full" : frac >= 0.7 ? "warn" : "";
  $("ctx-ring-pct").textContent = `${Math.round(frac * 100)}%`;
  const text = `${tr("chat.context", "context {n} of {limit}", {n: fmt(used), limit: fmt(limit)})} (${Math.round(frac * 100)}%)`;
  ring.title = text;
  ring.setAttribute("aria-label", text);
}
$("ctx-ring").onclick = () => toast("info", tr("mon.ctx", "Context fill"), $("ctx-ring").title, 4000);   // a phone: no hover
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  paintMarks();
  paintCtxRing();
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
  const cv = e.target.closest("[data-chart-code]");   // a chart: flip between it and its code
  if (cv) {
    const box = cv.closest(".st-code"), on = box.dataset.showCode !== "true";
    if (on) box.dataset.showCode = "true"; else delete box.dataset.showCode;
    cv.setAttribute("aria-pressed", String(on));
    return;
  }
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
  // an image a tool made: its preview (where it can be marked for the next message); a modified click still opens a tab
  const gi = e.target.closest("[data-gen-image]");
  if (gi && e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
    e.preventDefault();
    previewAttachment({kind: "image", url: gi.getAttribute("href"), ref: gi.dataset.genImage, name: gi.dataset.genImage.split("/").pop()});
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
function apiMessages(partial, msgs = messages, live = busy && busy.msg) {
  const start = StrataSegments.segmentStart(msgs), seg = msgs[start] && msgs[start].segment;
  const out = seg ? [{role: "system", content: seg.system}, ...seg.carry] : [];
  for (const m of msgs.slice(start)) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m) + (m.ctx ? `\n\n${m.ctx}` : "");
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (m !== live) {                           // the answer being asked for now is not history yet
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
  // a request's rounds start at 1 again after the page's own tools: group the calls that follow one another, so two
  // rounds with the same number on either side of a page tool stay two assistant turns in the history
  const groups = [];
  for (const t of ran) {
    const last = groups[groups.length - 1];
    if (last && last[0].round === t.round) last.push(t);
    else groups.push([t]);
  }
  for (const calls of groups) {
    const endCall = calls[calls.length - 1];
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
function segmentNow(msgs = messages) {
  const i = StrataSegments.segmentStart(msgs);
  return (msgs[i] && msgs[i].segment) || null;
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
function lastUserIndex(msgs = messages) {
  for (let i = msgs.length - 1; i >= 0; i--) if (msgs[i].role === "user") return i;
  return -1;
}
function lastAnswerBefore(i) {
  for (let j = i - 1; j >= 0; j--) if (messages[j].role === "assistant") return messages[j];
  return null;
}
// the record that makes user message i start segment k
function newSegment(k, i, limit, msgs = messages) {
  const S = StrataSegments, memory = S.memoryAt(msgs, i);
  return {k, limit, system: S.systemNote(k, limit) + (memory ? `\n\nYour memory:\n${memory}` : ""),
          carry: k > 1 ? S.carryFrom(msgs, i, userText) : [],
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
function handOffAt(i, msgs = messages) {
  if (!settings.segments || i < 0) return false;
  const u = msgs[i], limit = segmentLimit();
  if (!limit) return false;
  if (!u.segment) {
    u.segment = newSegment(StrataSegments.segmentAt(msgs, i - 1) + 1, i, limit, msgs);
    delete u.ctx;
    return true;
  }
  if (u.segment.carry.length > 2) { u.segment.carry = u.segment.carry.slice(-2); return true; }
  return false;
}
// what the next turn needs of an answer that ended: the segment's tokens after it, and whether its thinking goes back
// (segments.js keepThinking: its thinking against what would be read again without it, the answer and its tools'
// results at the answer's own characters a token)
function settleTurn(m, usage, msgs = messages) {
  m.v = 2;
  m.used = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
  if (!settings.segments) return;
  const thinking = (usage.completion_tokens_details && usage.completion_tokens_details.reasoning_tokens) || 0;
  const chars = (m.reasoning || "").length + (m.text || "").length;
  const perToken = usage.completion_tokens > 0 && chars > 0 ? chars / usage.completion_tokens : 3;
  const results = (m.tools || []).reduce((n, t) => n + (t.result ? t.result.length : 0), 0);
  const seg = segmentNow(msgs);
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
  return settings.whileBusy === "interject" ? tr("composer.hintInterject", "Ctrl+Enter stops the answer here and sends now")
                                            : tr("composer.hintQueue", "Ctrl+Enter sends when the answer is done");
}
// what the chat is waiting for while the model loads (after an image job, every image that took the GPU from it)
const loadingHint = () => modelLoading === "after_image"
  ? tr("composer.loadingAfterImage", "The model is loading again after the image (a minute or two); the answer goes on then")
  : tr("composer.loading", "The model is loading (a minute or two); the answer goes on then");
function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = !!panel;                    // while it answers, Send queues the message (park)
  paintImageSwitch();
  $("input").disabled = !!panel;
  $("composer-hint").textContent = panel ? tr("composer.finishOpen", "Save or cancel what is open above to continue")
                                         : on && modelLoading ? loadingHint()
                                         : on ? busyHint() : imageAlways ? tr("composer.imageOnHint", "Ctrl+Enter: send (makes an image)")
                                         : tr("composer.sendHint", "Ctrl+Enter: send");
  if (settings.chatEdit) syncActions();                 // the buttons follow the state: none under the answer being written
  $("s-list").disabled = $("s-edit").disabled = !!busy || !!panel;      // they change the messages on the screen
  renderUndoBar();
  renderPending();
}

// A message sent while an answer is being written (#4004) waits in its chat's queue (queueOf). "queue" (the default): it
// goes when the answer is done, the queue one after another. "interject": the answer stops where it is (what is written
// stays, marked interrupted) and the message goes at once; if a tool is running, the tool finishes first. Each waiting
// message can be taken back to the box (cancelled) or sent now (it interrupts).
function park(p, own) {
  const q = queueOf(), mode = settings.whileBusy === "interject" && !q.some((x) => x.mode === "interject") ? "interject" : "queue";
  if (mode === "interject") q.unshift({...p, mode}); else q.push({...p, mode});
  if (own) { attachments = []; renderAttachments(); $("input").value = ""; autosize(); }
  syncPending();
  if (mode === "interject") requestCut();
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
  const q = queueOf();
  if (!q.length) { if (el) el.remove(); return; }
  if (!el) {
    el = node("div", "pending-list");
    el.id = "pending";
    el.setAttribute("role", "status");
    el.addEventListener("click", (e) => {
      const b = e.target.closest("[data-pending-now], [data-pending-back]");
      if (!b) return;
      const i = +b.closest("[data-i]").dataset.i;
      if (b.dataset.pendingNow != null) pendingNow(i); else releasePending(i);
    });
    $("composer").before(el);
  }
  el.innerHTML = q.map((p, i) => {
    const quote = (p.text || (p.files[0] || p.images[0] || {}).name || "").replace(/\s+/g, " ").slice(0, 70);
    const text = !busy ? tr("pending.ready", "Not sent yet: “{text}”", {text: quote})
      : p.mode === "interject" ? (busy.cutAfterTool ? tr("pending.afterTool", "Waiting for the tool to finish, then sending: “{text}”", {text: quote})
                                                    : tr("pending.cutting", "Stopping the answer, then sending: “{text}”", {text: quote}))
      : i === 0 ? tr("pending.queued", "Sends when this answer is done: “{text}”", {text: quote})
      : tr("pending.queuedLater", "Then sends ({n}): “{text}”", {n: i + 1, text: quote});
    const now = !!busy && p.mode === "interject" ? "" : `<button type="button" class="st-btn st-btn--secondary" data-pending-now>${
      esc(busy ? tr("pending.sendNow", "Send now") : tr("pending.send", "Send"))}</button>`;
    return `<div class="pending" data-i="${i}"><span class="pending__text">${esc(text)}</span><span class="pending__actions">${now}` +
           `<button type="button" class="st-btn st-btn--secondary" data-pending-back>${esc(tr("pending.takeBack", "Take back"))}</button></span></div>`;
  }).join("");
}
// a waiting message sent now: while it answers, it goes first and interrupts; else it is sent
function pendingNow(i = 0) {
  const q = queueOf(), [p] = q.splice(i, 1);
  if (!p) return;
  if (busy) {
    const was = q.findIndex((x) => x.mode === "interject");
    if (was >= 0) q[was].mode = "queue";                 // one interrupts: the one asked for last
    q.unshift({...p, mode: "interject"});
    syncPending();
    requestCut();
    setBusy(true);
    return;
  }
  syncPending();
  setBusy(false);
  send(p);
}
// a waiting message back into the message box (cancelled); without `i`, every one of this chat (its chat is cleared)
function releasePending(i) {
  const q = queueOf(), out = i == null ? q.splice(0) : q.splice(i, 1);
  if (!out.length) return;
  syncPending();
  for (const p of out.reverse()) {
    $("input").value = $("input").value.trim() ? `${p.text}\n\n${$("input").value}` : p.text;
    attachments = [...p.images, ...p.files, ...attachments];
  }
  renderAttachments(); autosize();
  setBusy(!!busy);
}
// after an answer: an interrupted one sends an interjection, a finished one the next queued message (Stop keeps them)
function settlePending(cut, finished) {
  const q = queueOf();
  if (!q.length) return;
  if ((q[0].mode === "interject" && cut) || (q[0].mode === "queue" && finished)) {
    const p = q.shift();
    syncPending();
    setBusy(false);
    send(p);
  }
}

// the message box (or a parked message, `payload`) becomes a user turn and a reply is asked for
// while the image switch is on, the reply must make or edit an image (xestrata_images' force)
async function send(payload) {
  const own = !payload;
  const p = payload || {text: $("input").value.trim(), images: attachments.filter((a) => a.kind !== "file"),
                        files: attachments.filter((a) => a.kind === "file"), image: imageAlways};
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
// A streamed reply the server keeps past its connection (xestrata_reply): its data lines are counted as they are read, so
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
          r = await fetch(`v1/xestrata/replies/${rid}?from=${seen}`, {headers: headers(), signal});
          if (r.ok) break;
          if (r.status === 404) throw e;                       // the server no longer has it (restarted)
        } catch (e2) { if (e2 === e || e2.name === "AbortError") throw e2; }
      }
    }
  }
}
// Stop (and an interjection): the server ends the reply too, then this page's connection is closed
function stopReply(b = busy) {
  if (b.rid) fetch(`v1/xestrata/replies/${b.rid}/cancel`, {method: "POST", headers: headers(true), body: "{}"}).catch(() => {});
  b.controller.abort();
}
// a kept reply this page has read to its end (or given up on): the server lets it go (else it keeps it 12 hours)
function replyRead(rid) {
  if (rid) fetch(`v1/xestrata/replies/${rid}/ack`, {method: "POST", headers: headers(true), body: "{}"}).catch(() => {});
}
// The answer goes to the chat it was asked in (`home`: its id, the chat record when the list is on, its messages), on the
// screen or not: another chat can be opened or started while it is written.  `opts.resume`: an answer a closed page left
// unread (its xestrata_reply id kept in the message, rid), read again from the server's start of it.
async function generateReply(opts = {}) {
  const home = opts.home || {id: chat ? chat.id : "single", chat, messages, cache: chatId || "chat"};
  const msgs = home.messages, shown = () => msgs === messages;
  const m = opts.resume || {role: "assistant", text: "", reasoning: "", time: Date.now()};
  if (!opts.resume) msgs.push(m);
  const elOf = () => (shown() ? $("chat").querySelector(`.st-msg[data-i="${msgs.indexOf(m)}"]`) : null);
  if (shown()) renderChat();
  const controller = new AbortController();
  const me = {controller, msg: m, cut: false, cutAfterTool: false, home};
  replies.set(home.id, me);
  if (shown()) busy = me;
  setBusy(!!busy);
  renderSidebar();
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
  if (settings.skills !== false && userSkillsOn()) body.xestrata_skills = true;   // the skills the user added (load_skill)
  if (health.translation) body.xestrata_translate = true;   // translations go to the translation model (its tool)
  // and make images when the conversation asks for one: by the model the words choose (auto), by one, or none
  const ci = $("chat-images").value;
  if (health.image_generation && ci !== "off") body.xestrata_images = ci === "auto" || !ci ? true : {model: ci};
  if (health.image_generation && opts.image)                // the image button: by the model chosen, else the words'
    body.xestrata_images = {...(ci && ci !== "auto" && ci !== "off" ? {model: ci} : {}), force: true};
  // what the user marked on a picture in its preview (previewAttachment): this message's image tools take it, even with
  // images off. marksBody spends the marks; a hand-off retry builds a fresh body after the failed message was dropped,
  // so carry their value over instead of asking the marks again (they are already gone).
  let markedImages = opts.markedImages || null;
  if (health.image_generation && imageMarks && !opts.resume && !markedImages) markedImages = marksBody(body.xestrata_images);
  if (markedImages) body.xestrata_images = markedImages;
  // the chat model's GPU lent to the image calls (the drawer's choice over the server's use_text_gpu)
  if (body.xestrata_images && health.image_text_gpu && settings.textGpu)
    body.xestrata_images = {...(body.xestrata_images === true ? {} : body.xestrata_images), use_text_gpu: settings.textGpu};
  if (health.web_search && $("chat-search").value !== "off") body.web_search_options = {};   // the server's search tools

  let firstAt = null, thinkStart = null, usage = null, frame = 0, overflow = false, handoff = false;
  // the time the server's tools ran (an image made, a search), which tok/s leaves out: the model wrote nothing meanwhile
  let toolMs = 0;
  const toolSince = new Map();
  const paint = () => { frame = 0; const el = elOf(); if (el) { updateAssistant(el, m, true); scrollDown(); } };
  const seg = segmentNow(msgs);                              // null: a chat from before segments, or segments off
  if (seg) {
    body.tools = StrataSegments.TOOLS.filter((t) => seg.tools.includes(t.function.name));
    body.prompt_cache_key = `${home.cache}-${seg.k}`;        // the same slot again for this chat (BATCHING.md)
  }
  try {
    // a round that called the page's own tools goes on in another request, with their results
    for (let round = 0; ; round++) {
      const own = [];
      let r;
      if (round === 0 && opts.resume) {
        // the round the closed page was reading: what it had before that round stays, the round is read from its start
        const base = m.ridBase || {};
        me.rid = m.rid;
        m.text = (m.text || "").slice(0, base.text || 0);
        m.reasoning = (m.reasoning || "").slice(0, base.reasoning || 0);
        if (m.tools) m.tools = m.tools.slice(0, base.tools || 0);
        for (const k of ["error", "stopped", "meta", "limit"]) delete m[k];
        if (!frame) frame = requestAnimationFrame(paint);
        r = await fetch(`v1/xestrata/replies/${me.rid}?from=0`, {headers: headers(), signal: controller.signal});
        if (r.status === 404) throw replyError(tr("chat.replyLost", "The server no longer has this answer (it was restarted, or the answer waited more than 12 hours)."));
      } else {
        body.messages = apiMessages(round > 0, msgs, m);
        // a round after the page's own tools goes on with the same answer: the image button's forced call and the
        // marks hold until an image tool has run in it (sent again after that, every round made one more image;
        // dropped before it, a mask the model had not used yet was lost when it wrote its memory first)
        const imageRan = (m.tools || []).some((t) => ["edit_image", "generate_image"].includes(t.tool || t.name) && t.state !== "writing");
        if (round > 0 && imageRan && body.xestrata_images && typeof body.xestrata_images === "object") {
          const rest = {...body.xestrata_images};
          for (const k of ["force", "edit", "layout"]) delete rest[k];
          body.xestrata_images = Object.keys(rest).length ? rest : true;
        }
        // kept past its connection (a tab put to sleep, a network that dropped, the page closed): the rest is read again
        // below, or, when the page is opened again, from the id kept in the message (resumeReplies)
        const prev = me.rid;
        me.rid = health.kept_replies ? replyId() : null;
        if (me.rid) {
          body.xestrata_reply = {id: me.rid};
          Object.assign(m, {rid: me.rid, ridBase: {text: m.text.length, reasoning: m.reasoning.length, tools: (m.tools || []).length}});
          saveChatOf(home);
        }
        replyRead(prev);                                       // the round before, kept in the message just now
        r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                signal: controller.signal});
      }
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
        if (r.status === 401) msg = tr("toast.keyNeeded", "This server needs an API key: add it under About > Settings.");
        overflow = r.status === 400 && /exceeds the context window/.test(msg);
        throw new Error(msg);
      }
      await readReply(r, me.rid, controller.signal, (data) => {
          if (data === "[DONE]") return;
          let j;
          try { j = JSON.parse(data); } catch (e) { return; }
          if (j.error) throw replyError(j.error.message || tr("toast.engineError", "the engine reported an error"));
          if (j.usage) usage = j.usage;
          if (j.strata_mcp) {
            const x = j.strata_mcp;
            if (x.event === "call") { if (!firstAt) firstAt = performance.now(); toolSince.set(x.id, performance.now()); }
            else if (x.event === "result" && toolSince.has(x.id)) { toolMs += performance.now() - toolSince.get(x.id); toolSince.delete(x.id); }
            onTool(m, x);
            // an interjection waited for a running tool: it has its result now, so the answer can stop
            if (me.cutAfterTool && !me.cut && !(m.tools || []).some((t) => t.state === "running")) { me.cut = true; stopReply(me); }
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
      if (!calls.length || me.cut || round >= PAGE_ROUNDS) break;
      const room = usage ? usage.prompt_tokens + usage.completion_tokens : 0;
      for (const c of calls) {
        let a = {};
        try { a = JSON.parse(c.args || "{}"); } catch (e) { a = {}; }
        const t = {id: c.id, name: c.name, tool: c.name, arguments: a, round: `page-${round}`, at: m.text.length, rat: m.reasoning.length,
                   page: true, ok: true, state: "done"};
        t.result = StrataSegments.runTool(msgs, StrataSegments.segmentStart(msgs), c.name, a) +
                   (seg ? `\n${StrataSegments.roomLine(room, seg.limit)}` : "");
        m.tools = m.tools || [];
        m.tools.push(t);
      }
      if (!frame) frame = requestAnimationFrame(paint);
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else if (overflow && handOffAt(lastUserIndex(msgs), msgs)) handoff = true;
    else { m.error = e.message || String(e); toast("error", tr("toast.requestFailed", "The request failed"), m.error, 6000); }
  }
  const cut = me.cut;                                        // an interjection ended it: kept as written, marked interrupted
  if (cut) m.interrupted = true;
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (usage) settleTurn(m, usage, msgs);
  const ended = cut ? tr("chat.interruptedLower", "interrupted") : tr("chat.stoppedLower", "stopped");
  if (n && firstAt) {
    for (const t0 of toolSince.values()) toolMs += performance.now() - t0;   // a tool still running when it stopped
    // a resumed answer was read in one go (resumeReplies): its speed is not known here
    const secs = opts.resume ? 0 : (performance.now() - firstAt - toolMs) / 1000;
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
  replies.delete(home.id);
  if (busy === me) busy = null;
  setBusy(!!busy);
  delete m.rid;
  delete m.ridBase;
  if (handoff) {                                     // the turn did not fit: it starts the next segment and runs again
    if (frame) cancelAnimationFrame(frame);
    msgs.splice(msgs.indexOf(m), 1);
    saveChatOf(home);
    replyRead(me.rid);
    if (shown()) renderChat();
    const next = {...opts, home, resume: null};
    if (markedImages) next.markedImages = markedImages;
    return generateReply(next);
  }
  paintLive();
  if (frame) cancelAnimationFrame(frame);
  const el = elOf();
  if (el) updateAssistant(el, m, false);
  saveChatOf(home);
  replyRead(me.rid);                                 // kept in the chat now: the server lets it go
  renderSidebar();
  keepToolCopies(m).then((got) => { if (got && msgs.includes(m)) saveChatOf(home); });
  if (shown()) paintCtxRing();
  if (!shown()) {                                    // its queue goes on when the chat is opened again (showChat)
    if (!m.stopped && !m.error && queueOf(home.id).length) queueOf(home.id).go = true;
    return;
  }
  scrollDown();
  settlePending(cut, !m.stopped && !m.error);
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
// the image switch: not a send button - while on, every answer sent makes or edits an image; kept here
function paintImageSwitch() {
  $("image-btn").setAttribute("aria-checked", String(imageAlways));   // (in the + menu)
  const label = imageAlways ? tr("composer.imageLabelOn", "Always image: on") : tr("composer.imageLabelOff", "Always image: off");
  $("image-btn").querySelector(".image-toggle__label").textContent = label;
  $("more-btn").classList.toggle("is-on", imageAlways);                // the + says it while its menu is shut
  $("composer").classList.toggle("image-always", imageAlways);
  if (!busy && !$("input").disabled) $("composer-hint").textContent = imageAlways ? tr("composer.imageOnHint", "Ctrl+Enter: send (makes an image)")
                                                                    : tr("composer.sendHint", "Ctrl+Enter: send");
}
$("image-btn").onclick = () => { imageAlways = !imageAlways; store.set("chat_image_always", imageAlways); paintImageSwitch(); };
paintImageSwitch();
// the + menu: attach, Chat settings, the image switch; it shuts when one is chosen, at a click outside or Escape
function openMore(open) {
  $("more-menu").hidden = !open;
  $("more-btn").setAttribute("aria-expanded", String(open));
  if (open) ($("more-menu").querySelector(".composer-menu__item:not([hidden])") || $("more-btn")).focus();
}
$("more-btn").onclick = (e) => { e.stopPropagation(); openMore($("more-menu").hidden); };
$("more-menu").addEventListener("click", (e) => { e.stopPropagation(); if (e.target.closest(".composer-menu__item")) openMore(false); });
document.addEventListener("click", () => { if (!$("more-menu").hidden) openMore(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("more-menu").hidden) { openMore(false); $("more-btn").focus(); } });
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
  a.download = `xestrata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};
$("drawer-export").onclick = () => $("export-btn").onclick();

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("video/") && health.video && health.video.available) { chatVideo(f); continue; }
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
// an attachment as a chip: its name cut short to the room there is (the whole name on hover), and a button that previews
// it while the page has it (a picture's data, a file's text; a chat kept by name only has neither)
function attachChip(a, kind) {
  const c = document.createElement("span"), name = a.name || tr("chat.image", "image");
  c.className = "chip";
  c.title = name;
  c.innerHTML = icon(kind, "st-icon st-icon--sm");
  const can = a.url || typeof a.text === "string";
  const n = document.createElement(can ? "button" : "span");
  n.className = "chip__name";
  n.textContent = name;
  if (can) {
    n.type = "button";
    n.setAttribute("aria-label", tr("file.show", "Show {name}", {name}));
    n.onclick = () => previewAttachment({kind: a.url ? "image" : "file", ...a});
  }
  c.appendChild(n);
  return c;
}
// a picture at its size (to the window's), or a file's text, over the page; Escape, the button or a click outside closes it.
// A picture can be marked for the next message's image tools (imageMarks): a mask painted over the part edit_image
// changes, MOSS-Image's boxes that generate_image lays the new image out by.  `a.ref`: what the server knows it by (a
// /v1/images/files/ link of an image made here), else its data: URL.
let imageMarks = null;   // {name, url, ref, w, h, mask: a data: URL or null, boxes: [{x0, y0, x1, y1, label, caption, text}]}
const canMask = () => !!health.image_generation;
const canBox = () => (health.image_models || []).some((m) => (m.defaults || {}).layout);
function previewAttachment(a) {
  let d = $("attach-preview");
  if (!d) {
    d = document.createElement("dialog");
    d.id = "attach-preview";
    d.className = "attach-preview";
    d.innerHTML = `<div class="attach-preview__head"><span class="attach-preview__name"></span><span class="attach-preview__modes" role="group"></span>` +
      `<a class="st-btn st-btn--icon" data-preview-download>${icon("download")}</a>` +
      `<button type="button" class="st-btn st-btn--icon" data-preview-close><svg class="st-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 6L18 18M18 6L6 18" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button></div>` +
      `<div class="attach-preview__tools"></div><div class="attach-preview__body"></div><div class="attach-preview__foot"></div>`;
    d.querySelector("[data-preview-close]").onclick = () => d.close();
    d.addEventListener("click", (e) => { if (e.target === d) d.close(); });   // the backdrop: outside the box
    d.addEventListener("close", () => {
      for (const k of ["body", "tools", "foot", "modes"]) d.querySelector(`.attach-preview__${k}`).innerHTML = "";
      if (d._fit) { d._fit.disconnect(); d._fit = null; }
      const dl = d.querySelector("[data-preview-download]");
      if (dl.href.startsWith("blob:")) URL.revokeObjectURL(dl.href);
    });
    document.body.appendChild(d);
  }
  const name = a.name || tr("chat.image", "image");
  d.querySelector(".attach-preview__name").textContent = name;
  d.querySelector("[data-preview-close]").setAttribute("aria-label", tr("close", "Close"));
  // the picture or the file, saved as it is (its name)
  const dl = d.querySelector("[data-preview-download]");
  dl.href = a.kind === "image" ? a.url : URL.createObjectURL(new Blob([a.text || ""], {type: "text/plain"}));
  dl.download = name;
  dl.title = tr("preview.download", "Download");
  dl.setAttribute("aria-label", tr("preview.download", "Download"));
  d.classList.toggle("is-image", a.kind === "image");
  const body = d.querySelector(".attach-preview__body");
  if (a.kind !== "image") {
    const pre = document.createElement("pre");
    pre.textContent = a.text;
    body.appendChild(pre);
    d.showModal();
    return;
  }
  // the picture, with the paint and the boxes over it at its own size (the screen's pixels are counted back into its own)
  const ref = a.ref || a.url;
  const was = imageMarks && imageMarks.ref === ref ? imageMarks : null;
  const st = {mode: "view", boxes: was ? was.boxes.map((x) => ({...x})) : [], sel: -1};
  const wrap = document.createElement("div"), img = document.createElement("img"), paint = document.createElement("canvas"), stage = document.createElement("div");
  wrap.className = "attach-preview__stage";
  img.src = a.url; img.alt = name;
  paint.className = "attach-preview__paint";
  stage.className = "attach-preview__boxes";
  wrap.append(img, paint, stage);
  body.appendChild(wrap);
  const tools = d.querySelector(".attach-preview__tools"), foot = d.querySelector(".attach-preview__foot"), modes = d.querySelector(".attach-preview__modes");
  const btn = (text, cls = "st-btn") => { const b = document.createElement("button"); b.type = "button"; b.className = cls; b.textContent = text; return b; };
  // the picture as large as the box's room lets it, never larger than itself: the head, tools and foot keep their rows,
  // and a change of their height (the tools of another mode, the window) fits it again
  const fit = () => {
    if (!img.naturalWidth) return;
    const k = Math.min(1, (body.clientWidth - 16) / img.naturalWidth, (body.clientHeight - 16) / img.naturalHeight);
    img.style.width = `${Math.max(1, Math.floor(img.naturalWidth * k))}px`;
    img.style.height = `${Math.max(1, Math.floor(img.naturalHeight * k))}px`;
  };
  if (window.ResizeObserver) { d._fit = new ResizeObserver(fit); d._fit.observe(body); }
  img.onload = () => {
    fit();
    paint.width = img.naturalWidth; paint.height = img.naturalHeight;
    if (was && was.mask) { const m = new Image(); m.onload = () => paintFromMask(m); m.src = was.mask; }
    drawBoxes();
  };
  // the mask as sent (opaque, transparent where painted) back into paint
  const paintFromMask = (m) => {
    const g = paint.getContext("2d");
    g.fillStyle = "#e11d48"; g.fillRect(0, 0, paint.width, paint.height);
    g.globalCompositeOperation = "destination-in";
    const inv = document.createElement("canvas");
    inv.width = paint.width; inv.height = paint.height;
    const ig = inv.getContext("2d");
    ig.fillRect(0, 0, inv.width, inv.height);
    ig.globalCompositeOperation = "destination-out";
    ig.drawImage(m, 0, 0, inv.width, inv.height);
    g.drawImage(inv, 0, 0);
    g.globalCompositeOperation = "source-over";
  };
  const painted = () => {
    if (!paint.width) return false;
    const px = paint.getContext("2d").getImageData(0, 0, paint.width, paint.height).data;
    for (let i = 3; i < px.length; i += 4) if (px[i] > 0) return true;
    return false;
  };
  const maskURL = () => {             // as imgMask: opaque, transparent where it was painted
    const m = document.createElement("canvas");
    m.width = paint.width; m.height = paint.height;
    const g = m.getContext("2d");
    g.fillRect(0, 0, m.width, m.height);
    g.globalCompositeOperation = "destination-out";
    g.drawImage(paint, 0, 0);
    return m.toDataURL("image/png");
  };
  // the mask's tools: the brush (in screen pixels), erase, clear
  const brush = document.createElement("input"), erase = document.createElement("input");
  brush.type = "range"; brush.min = "4"; brush.max = "120"; brush.value = "32"; brush.setAttribute("aria-label", tr("img.brushSize", "Brush size"));
  erase.type = "checkbox";
  (() => {
    let last = null;
    const at = (ev) => { const r = paint.getBoundingClientRect(); return [(ev.clientX - r.left) * paint.width / r.width, (ev.clientY - r.top) * paint.height / r.height, paint.width / r.width]; };
    const stroke = (p, q) => {
      const g = paint.getContext("2d");
      g.globalCompositeOperation = erase.checked ? "destination-out" : "source-over";
      g.strokeStyle = "#e11d48"; g.lineCap = g.lineJoin = "round";
      g.lineWidth = Number(brush.value) * q[2];
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    };
    paint.onpointerdown = (ev) => { if (st.mode !== "mask") return; paint.setPointerCapture(ev.pointerId); last = at(ev); stroke(last, last); };
    paint.onpointermove = (ev) => { if (!last) return; const q = at(ev); stroke(last, q); last = q; };
    paint.onpointerup = paint.onpointercancel = () => { last = null; };
  })();
  // the boxes: drawn, moved and sized as the Images tab's (in shares of the picture's sides)
  const label = document.createElement("select"), caption = document.createElement("input"), words = document.createElement("input");
  for (const v of ["object", "text", "character"]) label.add(new Option(imgBoxName(v), v));
  caption.className = words.className = label.className = "st-input";
  caption.placeholder = tr("img.boxCaption", "What it looks like"); words.placeholder = tr("img.boxWords", "The exact words");
  const boxFields = document.createElement("div");
  boxFields.className = "attach-preview__box";
  const del = btn(tr("img.boxDelete", "Delete the box"));
  boxFields.append(label, caption, words, del);
  const drawBoxes = () => {
    stage.replaceChildren(...st.boxes.map((b, i) => {
      const e = document.createElement("div"), hd = document.createElement("span");
      e.className = "img-box"; e.dataset.i = String(i); e.dataset.label = b.label;
      e.setAttribute("aria-selected", String(i === st.sel));
      Object.assign(e.style, {left: `${b.x0 * 100}%`, top: `${b.y0 * 100}%`, width: `${(b.x1 - b.x0) * 100}%`, height: `${(b.y1 - b.y0) * 100}%`,
                              zIndex: String(1 + Math.round((1 - (b.x1 - b.x0) * (b.y1 - b.y0)) * 1000))});
      e.textContent = b.label === "text" && b.text ? `"${b.text}"` : b.caption || imgBoxName(b.label);
      hd.className = "img-box__handle";
      e.append(hd);
      return e;
    }));
  };
  // the selected box's fields, always in their row (off with no box, the words off but for a text box): a box drawn
  // does not make the row grow and the picture move
  const fields = () => {
    const b = st.boxes[st.sel];
    for (const x of [label, caption, del]) x.disabled = !b;
    words.disabled = !b || b.label !== "text";
    label.value = b ? b.label : "object"; caption.value = b ? b.caption : ""; words.value = b ? b.text : "";
  };
  label.onchange = () => { const b = st.boxes[st.sel]; if (b) { b.label = label.value; words.disabled = b.label !== "text"; drawBoxes(); } };
  caption.oninput = () => { const b = st.boxes[st.sel]; if (b) { b.caption = caption.value; drawBoxes(); } };
  words.oninput = () => { const b = st.boxes[st.sel]; if (b) { b.text = words.value; drawBoxes(); } };
  del.onclick = () => { if (st.sel >= 0) { st.boxes.splice(st.sel, 1); st.sel = -1; drawBoxes(); fields(); } };
  (() => {
    const least = 0.02;
    let drag = null;
    const at = (ev) => { const r = stage.getBoundingClientRect(); return [clamp01((ev.clientX - r.left) / r.width), clamp01((ev.clientY - r.top) / r.height)]; };
    stage.onpointerdown = (ev) => {
      if (st.mode !== "boxes") return;
      const q = at(ev), el = ev.target.closest(".img-box");
      stage.setPointerCapture(ev.pointerId);
      if (el) drag = {i: +el.dataset.i, mode: ev.target.classList.contains("img-box__handle") ? "size" : "move", p: q, b: {...st.boxes[+el.dataset.i]}};
      else { st.boxes.push({x0: q[0], y0: q[1], x1: q[0], y1: q[1], label: "object", caption: "", text: ""}); drag = {i: st.boxes.length - 1, mode: "new", p: q}; }
      st.sel = drag.i; drawBoxes(); fields();
    };
    stage.onpointermove = (ev) => {
      if (!drag) return;
      const q = at(ev), b = st.boxes[drag.i];
      if (drag.mode === "new") {
        [b.x0, b.x1] = [Math.min(drag.p[0], q[0]), Math.max(drag.p[0], q[0])];
        [b.y0, b.y1] = [Math.min(drag.p[1], q[1]), Math.max(drag.p[1], q[1])];
      } else if (drag.mode === "move") {
        const o = drag.b, dx = Math.min(1 - o.x1, Math.max(-o.x0, q[0] - drag.p[0])), dy = Math.min(1 - o.y1, Math.max(-o.y0, q[1] - drag.p[1]));
        Object.assign(b, {x0: o.x0 + dx, x1: o.x1 + dx, y0: o.y0 + dy, y1: o.y1 + dy});
      } else { b.x1 = Math.max(b.x0 + least, q[0]); b.y1 = Math.max(b.y0 + least, q[1]); }
      drawBoxes();
    };
    stage.onpointerup = stage.onpointercancel = () => {
      if (!drag) return;
      const b = st.boxes[drag.i], made = drag.mode === "new";
      if (made && (b.x1 - b.x0 < least || b.y1 - b.y0 < least)) { st.boxes.splice(drag.i, 1); st.sel = -1; }
      drag = null; drawBoxes(); fields();
      if (made && st.sel >= 0) caption.focus();
    };
  })();
  // the modes: view, the mask, the boxes (where the image models take them)
  const setMode = (m) => {
    st.mode = m;
    wrap.dataset.mode = m;
    for (const b of modes.children) b.setAttribute("aria-pressed", String(b.dataset.mode === m));
    tools.replaceChildren();
    if (m === "mask") {
      const lb = document.createElement("label"), le = document.createElement("label"), clear = btn(tr("img.clear", "Clear"));
      lb.className = le.className = "attach-preview__tool";
      lb.append(tr("img.brush", "Brush"), brush);
      le.append(erase, tr("img.erase", "Erase"));
      clear.onclick = () => paint.getContext("2d").clearRect(0, 0, paint.width, paint.height);
      tools.append(lb, le, clear);
    } else if (m === "boxes") {
      const hint = document.createElement("span");
      hint.className = "muted small";
      hint.textContent = tr("img.layoutHint", "Drag on the canvas to draw a box; drag a box to move it, its corner to change its size. Select a box to say what goes in it.");
      tools.append(hint, boxFields);
      fields();
    }
  };
  if (canMask()) {
    for (const [m, text] of [["view", tr("preview.view", "View")], ["mask", tr("preview.mask", "Mask")], ["boxes", tr("preview.boxes", "Boxes")]]) {
      const b = btn(text, "st-btn attach-preview__mode");
      b.dataset.mode = m;
      b.onclick = () => {
        if (m === "boxes" && !canBox()) { toast("error", tr("preview.noMoss", "Boxes need MOSS-Image, and no MOSS-Image model is installed here.")); return; }
        setMode(m);
      };
      modes.appendChild(b);
    }
    const use = btn(tr("preview.use", "Use in the next message"), "st-btn st-btn--primary"), drop = btn(tr("preview.drop", "Remove the marks"));
    drop.hidden = !was;
    use.onclick = () => {
      const boxes = st.boxes.filter((b) => b.x1 > b.x0 && b.y1 > b.y0);
      if (boxes.some((b) => b.label === "text" && !b.text.trim())) { toast("warn", tr("img.boxNeedsWords", "A text box needs its words.")); return; }
      const mask = painted() ? maskURL() : null;
      imageMarks = mask || boxes.length ? {name, url: a.url, ref, w: img.naturalWidth, h: img.naturalHeight, mask, boxes,
                                           overlay: marksOverlay(paint, boxes, mask != null)} : null;
      renderAttachments();
      paintMarks();
      d.close();
      $("input").focus();
    };
    drop.onclick = () => { imageMarks = null; renderAttachments(); paintMarks(); d.close(); };
    foot.append(drop, use);
  }
  setMode(was && was.boxes.length && !was.mask ? "boxes" : was ? "mask" : "view");
  d.showModal();
}
// what the marks on a picture are, beside it: its chip in the message box, or over it in the chat (paintMarks)
function marksBadge() {
  const b = document.createElement("span");
  b.className = "marks-badge";
  b.textContent = [imageMarks.mask ? tr("preview.mask", "Mask") : "", imageMarks.boxes.length ? tr("preview.boxes", "Boxes") : ""].filter(Boolean).join(" · ");
  b.title = tr("preview.use", "Use in the next message");
  return b;
}
// the marks as a picture to lay over the image's thumbnail: the paint (half see-through, as the preview shows it) and the
// boxes' outlines, at the image's size
function marksOverlay(paint, boxes, painted) {
  const c = document.createElement("canvas");
  c.width = paint.width; c.height = paint.height;
  if (!c.width) return null;
  const g = c.getContext("2d");
  if (painted) { g.globalAlpha = 0.55; g.drawImage(paint, 0, 0); g.globalAlpha = 1; }
  g.lineWidth = Math.max(3, Math.round(c.width / 150));
  for (const b of boxes) {
    g.strokeStyle = b.label === "text" ? "#e11d48" : b.label === "character" ? "#16a34a" : "#2563eb";   // the boxes' colours
    g.strokeRect(b.x0 * c.width, b.y0 * c.height, (b.x1 - b.x0) * c.width, (b.y1 - b.y0) * c.height);
  }
  return c.toDataURL("image/png");
}
// the marks over a thumbnail: their picture laid on it (marksOverlay), or none
function setMarksOverlay(el, overlay) {
  let o = el.querySelector(":scope > .marks-overlay");
  if (!overlay) { if (o) o.remove(); return; }
  if (!o) { o = document.createElement("img"); o.className = "marks-overlay"; o.alt = ""; el.appendChild(o); }
  if (o.src !== overlay) o.src = overlay;
}
// the pictures in the chat the marks are on (a picture sent, an image a tool made): their frame marked, the label and the
// marks themselves over it
function paintMarks() {
  const label = imageMarks ? marksBadge().textContent : "";
  for (const el of document.querySelectorAll("#chat .msg-image-wrap, #chat [data-gen-image]")) {
    const ref = el.dataset.genImage || (el.querySelector("img:not(.marks-overlay)") || {}).src;
    const on = !!imageMarks && ref === imageMarks.ref;
    el.classList.toggle("is-marked", on);
    if (on) el.dataset.marks = label; else delete el.dataset.marks;
    setMarksOverlay(el, on ? imageMarks.overlay : null);
  }
}
// the marks into a request's xestrata_images (then they are spent): edit's image and mask, layout's boxes at the image's
// shape at about 1024 x 1024 pixels (as the Images tab's auto size)
function marksBody(want) {
  if (!imageMarks) return want;
  const m = imageMarks, out = {...(want && typeof want === "object" ? want : {})};
  // the picture itself where the page has it (an image a tool made, copied into the chat: keepToolCopies), not its
  // /v1/images/files/ link - the server keeps those only so long (a restart, or 64 newer ones, lets them go)
  const image = typeof m.url === "string" && m.url.startsWith("data:") ? m.url : m.ref;
  if (m.mask) out.edit = {image, mask: {image_url: m.mask}};
  if (m.boxes.length) {
    const k = Math.sqrt(1024 * 1024 / (m.w * m.h));
    const w = Math.max(256, Math.round(m.w * k / 16) * 16), h = Math.max(256, Math.round(m.h * k / 16) * 16);
    // the boxes are MOSS-Image's: its model, chosen or not (another model chosen gives way to it)
    const moss = (health.image_models || []).filter((x) => (x.defaults || {}).layout).map((x) => x.id);
    if (moss.length && !moss.includes(out.model)) out.model = moss[0];
    out.layout = {size: `${w}x${h}`, image, items: m.boxes.map((b) => ({label: b.label, caption: b.caption.trim() || (b.label === "text" ? b.text.trim() : imgBoxName(b.label)),
                                                                 ...(b.label === "text" ? {text: b.text.trim()} : {}), box: [b.x0, b.y0, b.x1, b.y1]}))};
  }
  imageMarks = null;
  renderAttachments();
  paintMarks();
  return out;
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = attachChip(a, a.kind === "file" ? "attach" : "image");
    if (imageMarks && a.url === imageMarks.ref) {     // the marked picture: a thumbnail with its marks, and their label
      const t = document.createElement("span"), im = document.createElement("img");
      t.className = "chip__thumb";
      im.src = a.url; im.alt = "";
      t.appendChild(im);
      setMarksOverlay(t, imageMarks.overlay);
      c.querySelector(".st-icon").replaceWith(t);
      c.querySelector(".chip__name").after(marksBadge());
    }
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
  if (open) { loadDrawer(); loadShared(); loadMcp(); loadSkills(); }
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
  $("s-skills").setAttribute("aria-checked", String(s.skills !== false));
  $("s-textgpu").value = s.textGpu || "";
  $("textgpu-row").hidden = !health.image_text_gpu;
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  $("s-list").setAttribute("aria-checked", String(!!s.chatList));
  $("s-edit").setAttribute("aria-checked", String(!!s.chatEdit));
  $("s-seg").setAttribute("aria-checked", String(s.segments !== false));
  $("s-seglimit").value = s.segmentLimit || "";
  $("s-memory").value = StrataSegments.memoryAt(messages);
  for (const b of $("s-busy").children) b.setAttribute("aria-checked", String(b.dataset.v === (s.whileBusy === "interject" ? "interject" : "queue")));
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
  $("o-busy").textContent = {queue: tr("drawer.busyQueueNote", "Your message waits and goes as soon as the answer is done."),
    interject: tr("drawer.busyInterjectNote", "The answer stops where it is (what is written stays) and your message goes at once. A tool that is running finishes first.")}[busyMode] || "";
  for (const id of ["s-topp", "s-topk", "s-minp"]) $(id).disabled = t === 0;
}
for (const b of $("s-busy").children) b.onclick = () => { for (const x of $("s-busy").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const b of $("s-script").children) b.onclick = () => {
  for (const x of $("s-script").children) x.setAttribute("aria-checked", String(x === b));
  $("script-ctx-row").hidden = b.dataset.v === "off";        // the switch under it is kept as it was while hidden
};
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
for (const id of ["s-math", "s-show", "s-esp", "s-mcp", "s-skills", "s-share", "s-list", "s-edit", "s-scriptctx"]) {
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
              skills: checked("s-skills"), textGpu: $("s-textgpu").value,
              chatList: locked ? !!prev.chatList : checked("s-list"), chatEdit: locked ? !!prev.chatEdit : checked("s-edit"),
              whileBusy: segValue("s-busy") || "queue", segments: checked("s-seg"), segmentLimit: $("s-seglimit").value.trim(),
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
// the page's text size (About > Settings): the size tokens' scale (tokens.css --st-text-scale), kept in this browser
// and set before the first paint (index.html)
function textSizeChoices() {
  let saved = "1";
  try { saved = localStorage.getItem("xestrata.textScale") || "1"; } catch (e) { /* the default */ }
  const sizes = [["0.9", tr("settings.textSmall", "Small")], ["1", tr("settings.textNormal", "Standard")],
                 ["1.15", tr("settings.textLarge", "Large")], ["1.3", tr("settings.textLarger", "Larger")]];
  $("text-size").replaceChildren(...sizes.map(([v, name]) => new Option(name, v)));
  $("text-size").value = sizes.some((t) => t[0] === saved) ? saved : "1";
}
$("text-size").onchange = () => {
  const v = $("text-size").value;
  document.documentElement.style.setProperty("--st-text-scale", v);
  try { if (v === "1") localStorage.removeItem("xestrata.textScale"); else localStorage.setItem("xestrata.textScale", v); } catch (e) { /* this visit only */ }
  fitHeader();
  fitChatImages();
};
function showLanguage() {
  languageChoices();
  textSizeChoices();
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
  textSizeChoices();
  renderEmptyChat();
  initChats();
  if (settings.chatList) await setChatList(true, false);       // the saved-chats list is read before the first draw
  if (settings.segments) askPersist();
  setBusy(false);
  renderChat();
  const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... fills the box (a shortcut)
  if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
  thinkingButtons();   // the four of before until /health says the model's own
  loadHealth().then(() => resumeReplies()).then(loadSwitcher).then(loadMcp).then(loadSkills).then(() => loadUpdate()).then(() => {
    // upstream #4221: the question is filled in and the box focused; nothing is sent until the person presses Send
    if (startQuestion) { $("input").value = startQuestion; $("input").focus(); }
  });
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
// the size: a preset puts its sides into the width and the height, and moving those chooses the preset they match (else
// Custom); "auto" hides them
function imgSizeShow() {
  const v = $("img-size").value;
  if (/^\d+x\d+$/.test(v)) [$("img-w").value, $("img-h").value] = v.split("x");
  $("img-custom").hidden = v === "auto";
  $("img-swap").disabled = v === "auto";
  imgSliders();
}
function imgSizeFromWH() {
  const sel = $("img-size"), v = `${$("img-w").value}x${$("img-h").value}`;
  sel.value = [...sel.options].some(o => o.value === v) ? v : "custom";
}
$("img-size").onchange = imgSizeShow;
for (const id of ["img-w", "img-h"]) $(id).addEventListener("input", imgSizeFromWH);
$("img-swap").onclick = () => {
  [$("img-w").value, $("img-h").value] = [$("img-h").value, $("img-w").value];
  imgSizeFromWH();
  imgCount();
  imgLayoutDraw();
};
// the sliders beside the numbers (data-for): the number, or the count's select, is what a request reads, and the slider
// moves it; an empty number is the model's default (its placeholder), which the slider shows
const imgRanges = [...document.querySelectorAll(".img-page .img-sl__range")];
function imgSliders() {
  for (const r of imgRanges) {
    const f = $(r.dataset.for);
    if (f.tagName === "SELECT") {
      Object.assign(r, {min: "1", max: String(Math.max(1, f.options.length)), step: "1", disabled: f.options.length < 2});
      r.value = f.value;
      continue;
    }
    Object.assign(r, {min: f.min, max: f.max, step: r.dataset.step || f.step || "1"});
    const v = parseFloat(f.value !== "" ? f.value : f.placeholder);
    r.value = String(Number.isFinite(v) ? v : f.min);
  }
  for (const r of imgRanges) r.style.setProperty("--fill", `${r.max > r.min ? (r.value - r.min) / (r.max - r.min) * 100 : 0}%`);
}
for (const r of imgRanges) r.addEventListener("input", () => {
  const f = $(r.dataset.for);
  f.value = r.value;
  f.dispatchEvent(new Event("input", {bubbles: true}));
  if (f.tagName === "SELECT") f.dispatchEvent(new Event("change", {bubbles: true}));
});
for (const f of ["img-form", "tool-form", "vid-form"]) $(f).addEventListener("input", (e) => { if (!imgRanges.includes(e.target)) imgSliders(); });
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
  $("chat-images-box").hidden = $("image-btn").hidden = !health.image_generation;
  fitChatImages();
}
// the choice as wide as the option chosen (not the longest one, as a select is by itself): its box scrolls the rest
function fitChatImages() {
  const sel = $("chat-images"), o = sel.options[sel.selectedIndex];
  if (!o) return;
  const cs = getComputedStyle(sel), g = (fitChatImages.c || (fitChatImages.c = document.createElement("canvas"))).getContext("2d");
  g.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  const pad = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) + parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
  sel.style.width = `${Math.ceil(g.measureText(o.text).width + pad + 24)}px`;   // 24: the arrow
}
$("chat-images").onchange = () => { store.set("chat_images", $("chat-images").value); fitChatImages(); };
window.addEventListener("resize", fitChatImages);   // its padding follows the width (app.css)
// the chat's web search (the server's search_url): on or off, kept here; on sends web_search_options
function chatSearch() {
  const sel = $("chat-search"), was = sel.value || store.get("chat_search", "on");
  sel.replaceChildren(new Option(tr("chatSearch.on", "Search: on"), "on"), new Option(tr("chatSearch.off", "Search: off"), "off"));
  sel.value = was === "off" ? "off" : "on";
  $("search-row").hidden = !health.web_search;        // in Chat settings (the message box keeps its row short)
  renderEmptyChat();
}
$("chat-search").onchange = () => { store.set("chat_search", $("chat-search").value); renderEmptyChat(); };
function imgModels() {
  chatImages();
  chatSearch();
  const ms = health.image_models || [], sel = $("img-model"), was = sel.value;
  sel.replaceChildren(...ms.map(m => new Option(m.id, m.id)));
  if (ms.some(m => m.id === was)) sel.value = was;
  $("img-model-f").hidden = ms.length < 2;
  imgDefaults();
  toolsSetup();
  videoSetup();
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
  const one = ($("img-ref").files || []).length === 1;   // extend takes the one image it grows
  $("img-ext-f").hidden = !one;
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
  imgGuideSetup();
  imgCtrlSetup();
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
// MOSS-Image's 4-step LoRA needs more steps for larger images (the server chooses the same defaults).
function imgDefaultSteps(d) {
  if (!d.moss_turbo) return d.steps || 1;
  let [w, h] = $("img-size").value === "custom" ? [+$("img-w").value, +$("img-h").value]
    : ($("img-size").value === "auto" ? "1024x1024" : $("img-size").value).split("x").map(Number);
  w = Math.ceil((w || 1024) / 16) * 16;
  h = Math.ceil((h || 1024) / 16) * 16;
  return w * h <= 1024 ** 2 ? 4 : w * h <= 1536 ** 2 ? 6 : 8;
}
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
  let steps = d.steps_fixed ? d.steps : (+$("img-steps").value || (edit ? d.edit_steps : imgDefaultSteps(d)) || 1);
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
  let misses = 0;
  for (;;) {                                       // the download, until it ends (the server keeps going without us)
    const mb = (n) => fmt((n || 0) / 1048576, 0);
    imgLoraStatus(job.total ? tr("img.loraFetching", "Fetching {name}: {got} of {total} MB", {name: job.name, got: mb(job.bytes), total: mb(job.total)})
                            : tr("img.loraFetchingSize", "Fetching {name}: {got} MB", {name: job.name, got: mb(job.bytes)}));
    await new Promise((ok) => setTimeout(ok, 1000));
    let poll = null, j = null;
    try { poll = await fetch(`v1/images/loras/fetch/${job.id}`, {headers: headers()}); } catch (e) { /* the server may be down */ }
    if (poll && !poll.ok) {                        // a restart dropped FETCHES: "no such fetch", never a state
      imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: await imgLoraError(poll)}), "error");
      return;
    }
    try { if (poll) j = await poll.json(); } catch (e) { /* a partial answer: try again */ }
    if (!j) {
      if (++misses > 5) {                          // a server that stays away is not going to answer either
        imgLoraStatus(tr("img.loraFailed", "Not added: {why}", {why: tr("pill.unreachable", "Server not reachable")}), "error");
        return;
      }
      continue;
    }
    misses = 0;
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
  const steps = d.steps_fixed ? d.steps : (+$("img-steps").value || imgDefaultSteps(d));
  $("img-strength").placeholder = String(+imgStrength(d, steps).toFixed(2));
}
function imgCount() {
  imgStrengthHint();
  const d = imgModel().defaults || {}, n = $("img-n"), was = +n.value || 1;
  if (d.moss_turbo) $("img-steps").placeholder = String(imgDefaultSteps(d));
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
  imgSliders();
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
  $("img-crop").checked = d.crop_edit !== false;
  $("img-refine-f").hidden = !(d.refine && ($("img-ref").files || []).length > 1);
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
// the words' mask (POST /v1/images/segmentations: SAM 3, or Florence-2 where its license is not agreed to) painted
// into the canvas over what is there, the brush free to change it after
async function imgRegionMask() {
  const f = ($("img-ref").files || [])[0], words = $("img-region").value.trim();
  if (!f || !words) return;
  $("img-region-go").disabled = true;
  try {
    const r = await fetch("v1/images/segmentations", {method: "POST", headers: headers(true),
      body: JSON.stringify({image: await readAsDataURL(f), prompt: words})});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    const c = $("img-mask-paint"), g = c.getContext("2d");
    const px = g.getImageData(0, 0, c.width, c.height);
    for (const d of j.data) {
      if (!d.mask) continue;
      const im = new Image();
      await new Promise((ok, no) => { im.onload = ok; im.onerror = no; im.src = "data:image/png;base64," + d.mask; });
      const t = document.createElement("canvas");
      t.width = c.width; t.height = c.height;
      t.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
      const m = t.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      for (let i = 0; i < m.length; i += 4)
        if (m[i] > 127) { px.data[i] = 0xe1; px.data[i + 1] = 0x1d; px.data[i + 2] = 0x48; px.data[i + 3] = 255; }
    }
    g.putImageData(px, 0, 0);
    $("img-hint").textContent = j.data.length ? tr("img.regionFound", "{n} found by {model}", {n: j.data.length, model: j.model})
      : tr("img.regionNone", "Not found: try other words");
    $("img-sam-license").hidden = !(j.xestrata && /license/.test(j.xestrata.fallback || ""));
  } catch (e) {
    toast("error", tr("img.regionGo", "Make mask"), String(e.message || e), 8000);
  } finally {
    $("img-region-go").disabled = false;
  }
}
$("img-region-go").onclick = imgRegionMask;
$("img-region").onkeydown = (ev) => { if (ev.key === "Enter") { ev.preventDefault(); imgRegionMask(); } };
// SAM 3's license (Meta's SAM License, fetched to show): agreed here, the server fetches its weights and finds parts
// with it; until then Florence-2 (MIT) does
$("img-sam-license").onclick = async (ev) => {
  ev.preventDefault();
  try {
    const r = await fetch("v1/images/licenses/sam3", {headers: headers()});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    if (j.status !== "ready") { toast("info", tr("img.samLicense", "SAM 3's license…"), tr("img.samFetching", "Fetching the license's text: try again in a moment."), 6000); return; }
    const d = document.createElement("dialog");
    d.className = "attach-preview";
    d.innerHTML = `<div class="attach-preview__head"><span class="attach-preview__name"></span></div><div class="attach-preview__body"><pre></pre></div>` +
      `<div class="attach-preview__foot"><button type="button" class="st-btn" data-agree></button> <button type="button" class="st-btn" data-close></button></div>`;
    d.querySelector(".attach-preview__name").textContent = tr("img.samTitle", "SAM 3.1: the SAM License (Meta's)");
    d.querySelector("pre").textContent = j.text;
    d.querySelector("[data-agree]").textContent = j.accepted ? tr("img.samAgreed", "Agreed") : tr("img.samAgree", "I agree");
    d.querySelector("[data-agree]").disabled = !!j.accepted;
    d.querySelector("[data-close]").textContent = tr("close", "Close");
    d.querySelector("[data-close]").onclick = () => d.close();
    d.querySelector("[data-agree]").onclick = async () => {
      const a = await fetch("v1/images/licenses/sam3", {method: "POST", headers: headers(true), body: JSON.stringify({accept: true})});
      if (a.ok) { $("img-sam-license").hidden = true; toast("info", tr("img.samTitle", "SAM 3.1: the SAM License (Meta's)"), tr("img.samDone", "Agreed: the next mask is found by SAM 3 (its weights, 1.7 GB, are fetched first)."), 8000); }
      d.close();
    };
    d.addEventListener("close", () => d.remove());
    document.body.appendChild(d);
    d.showModal();
  } catch (e) {
    toast("error", tr("img.samLicense", "SAM 3's license…"), String(e.message || e), 8000);
  }
};
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
// the sampling guidance the model takes (its defaults' guidance_methods): each method's box and its main parameter
const IMG_GUIDE = {detail_daemon: "detail_amount", skip_layer_guidance: "scale", apg: "eta", cfg_norm: "strength",
                   cfg_zero_star: null};
function imgGuideSetup() {
  const g = (imgModel().defaults || {}).guidance_methods || {};
  $("img-guide").hidden = !Object.keys(g).length;
  for (const [name, param] of Object.entries(IMG_GUIDE)) {
    $(`img-g-${name}-f`).hidden = !g[name];
    if (g[name] && param) $(`img-g-${name}-v`).placeholder = String(g[name].params[param].default);
  }
}
function imgGuideBody(body) {
  if ($("img-guide").hidden) return;
  const out = [];
  for (const [name, param] of Object.entries(IMG_GUIDE)) {
    if ($(`img-g-${name}-f`).hidden || !$(`img-g-${name}`).checked) continue;
    const v = param ? $(`img-g-${name}-v`).value : "";
    out.push(v !== "" ? {name, [param]: Number(v)} : name);
  }
  if (out.length) body.guidance = out;
}
// the composition from a picture (image_controls.py: the model's defaults' control, its mode and types): the kind,
// the picture, ControlNet's strength and its share of the steps (a reference image's model has neither), the map
// looked at before sending (POST /v1/images/controls) and sent as it is
let imgCtrlMap = null;
function imgCtrlSetup() {
  const c = (imgModel().defaults || {}).control;
  $("img-ctrl").hidden = !c;
  if (!c) return;
  for (const o of $("img-ctrl-type").options) o.hidden = !!o.value && o.value !== "none" && !c.types.includes(o.value);
  if ($("img-ctrl-type").selectedOptions[0] && $("img-ctrl-type").selectedOptions[0].hidden) $("img-ctrl-type").value = "";
  imgCtrlKind();
}
// the kind's way into the model (/health's control.modes): ControlNet's strength and span, MOSS-Image's adapter's
// strength and end (no start: not trained), a reference image's neither
function imgCtrlMode() {
  const c = (imgModel().defaults || {}).control || {}, k = $("img-ctrl-type").value;
  return (c.modes || {})[k === "none" ? "canny" : k] || c.mode;
}
function imgCtrlKind() {
  const k = $("img-ctrl-type").value, mode = imgCtrlMode();
  $("img-ctrl-strength-f").hidden = !["controlnet", "moss", "lora"].includes(mode);
  $("img-ctrl-span-f").hidden = mode !== "controlnet" && mode !== "moss";
  $("img-ctrl-start").hidden = mode === "moss";
  $("img-ctrl-hint").hidden = mode !== "reference";
  $("img-ctrl-hint").textContent = mode === "reference" ? tr("img.ctrlRefHint", "This model has no ControlNet: the map goes in as one more reference image, and the prompt asks to follow it.") : "";
  $("img-ctrl-body").hidden = !k;
  $("img-ctrl-canny-f").hidden = k !== "canny";
  $("img-ctrl-pose-f").hidden = k !== "pose";
  $("img-ctrl-drawing-f").hidden = mode !== "controlnet" || !["lineart", "lineart_coarse"].includes(k);
  $("img-ctrl-regions-f").hidden = mode !== "moss" || k !== "color";
  $("img-ctrl-preview").hidden = k === "none";
  imgCtrlReset();
}
function imgCtrlReset() {
  imgCtrlMap = null;
  $("img-ctrl-map").hidden = true;
  $("img-ctrl-use").checked = false;
  $("img-ctrl-use").disabled = true;
}
// the control's fields (the API's), the picture as a data: URL
async function imgCtrlFields() {
  const k = $("img-ctrl-type").value, f = ($("img-ctrl-file").files || [])[0];
  if (!k || !f) return null;
  const c = {type: k, image: await readAsDataURL(f)};
  const num = (id, key) => { if ($(id).value !== "") c[key] = Number($(id).value); };
  num("img-ctrl-res", "resolution");
  if (k === "canny") { num("img-ctrl-low", "low_threshold"); num("img-ctrl-high", "high_threshold"); }
  if (k === "pose") Object.assign(c, {detect_body: $("img-ctrl-body-on").checked, detect_hands: $("img-ctrl-hands").checked, detect_face: $("img-ctrl-face").checked});
  if (!$("img-ctrl-drawing-f").hidden && $("img-ctrl-drawing").checked) c.lineart_input = "drawing";
  return c;
}
async function imgCtrlBody(body) {
  if ($("img-ctrl").hidden || !$("img-ctrl-type").value) return null;
  const c = await imgCtrlFields();
  if (!c) return tr("img.ctrlNoPicture", "Choose the picture to take the composition from (or turn the control off).");
  // the map shown, as it is: MOSS-Image's color and depth keep their kind (preprocess false), the others are "none"
  if (imgCtrlMap && $("img-ctrl-use").checked) {
    delete c.lineart_input;
    Object.assign(c, ["moss", "lora"].includes(imgCtrlMode()) ? {preprocess: false, image: imgCtrlMap} : {type: "none", image: imgCtrlMap});
  }
  if (!$("img-ctrl-regions-f").hidden) {
    if ($("img-ctrl-color-map").checked) c.preprocess = false;
    const lines = $("img-ctrl-regions").value.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
    if (lines.length) {
      const regions = lines.map(x => x.match(/^(#[0-9a-fA-F]{6})\s*:\s*(\S.*)$/));
      if (regions.some(x => !x)) return tr("img.ctrlRegionsInvalid", "Each region needs a #rrggbb color, a colon, and a description.");
      c.regions = regions.map(x => ({color: x[1], prompt: x[2]}));
    }
  }
  if (!$("img-ctrl-strength-f").hidden && $("img-ctrl-strength").value !== "") c.strength = Number($("img-ctrl-strength").value);
  if (!$("img-ctrl-span-f").hidden) {
    if ($("img-ctrl-start").value !== "" && !$("img-ctrl-start").hidden) c.start = Number($("img-ctrl-start").value);
    if ($("img-ctrl-end").value !== "") c.end = Number($("img-ctrl-end").value);
  }
  body.control = c;
  return null;
}
$("img-ctrl-type").onchange = imgCtrlKind;
$("img-ctrl-file").onchange = imgCtrlReset;
for (const id of ["img-ctrl-res", "img-ctrl-low", "img-ctrl-high", "img-ctrl-body-on", "img-ctrl-hands", "img-ctrl-face", "img-ctrl-drawing"]) $(id).addEventListener("change", imgCtrlReset);
$("img-ctrl-preview").onclick = async () => {
  const c = await imgCtrlFields();
  if (!c) { toast("error", tr("img.ctrlMap", "The control map"), tr("img.ctrlNoPicture", "Choose the picture to take the composition from (or turn the control off)."), 6000); return; }
  const b = $("img-ctrl-preview");
  b.disabled = true;
  try {
    const r = await fetch("v1/images/controls", {method: "POST", headers: headers(true), body: JSON.stringify(c)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    imgCtrlMap = "data:image/png;base64," + j.data[0].b64_json;
    $("img-ctrl-map").src = imgCtrlMap;
    $("img-ctrl-map").hidden = false;
    $("img-ctrl-use").disabled = false;
    $("img-ctrl-use").checked = true;
  } catch (e) {
    toast("error", tr("img.ctrlMap", "The control map"), String(e.message || e), 8000);
  } finally {
    b.disabled = false;
  }
};
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
  imgSizeShow();
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
// what the prompt enhancer did (the response's xestrata.enhance), under the image: the prompt it wrote, or its layout
// with a button that loads it into the boxes, or why it gave nothing
function imgEnhanced(j, item) {
  const e = j.xestrata && j.xestrata.enhance;
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
// a streamed job (the "show partial images" setting): the large picture shows the engine's partial images as they come,
// and the completed events become the body the plain request answers with
async function imgStream(r) {
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
          imgPartial(`data:image/${e.output_format || "png"};base64,` + e.b64_json,
                     tr("img.partialCap", "Partial image {n}…", {n: e.partial_image_index + 1}));
        } else if (e.type.endsWith(".completed")) {
          Object.assign(j, {model: e.model, size: e.size, output_format: e.output_format, xestrata: e.xestrata});
          j.data.push({b64_json: e.b64_json, seed: e.seed});
        }
      }
    }
  } finally {
    imgPartial(null);
  }
  if (!j.data.length) throw new Error(tr("img.none", "No image"));
  return j;
}
// the pictures made on this page, newest first: the one chosen large with its save, its generation info (the request in
// the API's names) and the buttons that use it; the others as a row of thumbnails below it
const imgShots = [];   // [{url, name, seed, info, note}]
let imgSel = -1;
function imgShow(i) {
  imgSel = i;
  const s = imgShots[i], main = $("img-main");
  main.hidden = !s;
  $("img-empty").hidden = !!s;
  $("img-actions").hidden = $("img-send").hidden = $("img-info-box").hidden = !s;
  $("img-results").hidden = imgShots.length < 2;
  for (const [k, t] of [...$("img-results").children].entries()) t.setAttribute("aria-current", String(k === i));
  if (!s) { main.removeAttribute("src"); return; }
  main.src = s.url;
  main.alt = s.prompt;
  Object.assign($("img-save"), {href: s.url, download: s.name});
  $("img-use-seed").disabled = s.seed == null;
  $("img-info").replaceChildren(s.info, ...(s.note ? [s.note] : []));
}
// a partial image in the large picture while the job runs; null puts back the one chosen
function imgPartial(url, caption) {
  const cap = $("img-view-cap");
  cap.hidden = !url;
  if (!url) { imgShow(imgSel); return; }
  cap.textContent = caption;
  $("img-main").src = url;
  $("img-main").hidden = false;
  $("img-empty").hidden = true;
}
function imgAdd(shot) {
  imgShots.unshift(shot);
  const b = document.createElement("button"), im = document.createElement("img");
  b.type = "button";
  b.className = "img-thumb";
  b.setAttribute("role", "listitem");
  im.className = "img-thumb__img";
  im.src = shot.url;
  im.alt = shot.prompt;
  b.append(im);
  b.onclick = () => imgShow([...$("img-results").children].indexOf(b));
  $("img-results").prepend(b);
}
// an image's info in the API's names: the request that made it (its seed the image's, n one: sent again, it makes this
// image again) and what the answer says of it; the copy button copies that request as JSON
function imgInfo(url, body, j, d, seconds) {
  const req = {...body, seed: d.seed};
  for (const k of ["n", "stream", "partial_images"]) delete req[k];
  const made = j.xestrata && j.xestrata.crop && !j.xestrata.crop.whole && j.xestrata.made_size ? tr("img.madeCrop", "(drawn at {size} around the painted part)", {size: j.xestrata.made_size})
    : j.xestrata && j.xestrata.crop ? ""
    : j.xestrata && j.xestrata.made_size ? tr("img.madeCut", "(made at {size}, middle cut out)", {size: j.xestrata.made_size})
    : j.xestrata && j.xestrata.requested_size ? tr("img.madeNearest", "({size} asked: the model's nearest size)", {size: j.xestrata.requested_size}) : "";
  return {req, info: imgInfoText(url, req, [`model: ${j.model}`, `size: ${j.size} ${made}`.trim(), `seconds: ${seconds}`])};
}
// "POST <endpoint>", the request's fields as key: value (an image as its kind, not its data), then the answer's lines
function imgInfoText(url, req, answer) {
  const val = (k, v) => typeof v === "string" && v.startsWith("data:") ? `(${v.slice(5, v.indexOf(";"))})`
    : k === "images" ? `(${v.length})` : k === "mask" ? "(image/png)" : typeof v === "string" ? v : JSON.stringify(v);
  return [`POST /${url}`, ...Object.entries(req).filter(([k]) => k !== "response_format").map(([k, v]) => `${k}: ${val(k, v)}`),
          "", ...answer].join("\n");
}
$("img-main").onclick = () => {
  if (!$("img-view-cap").hidden) return;   // a partial image
  $("img-full-img").src = $("img-main").src;
  $("img-full").showModal();
};
$("img-full-close").onclick = () => $("img-full").close();
$("img-full").addEventListener("click", (e) => { if (e.target !== $("img-full-close")) $("img-full").close(); });
$("img-copy").onclick = () => copyText(JSON.stringify(imgShots[imgSel].req, null, 2), $("img-copy"));
let imgLastSeed = null;
$("img-use-seed").onclick = () => { $("img-seed").value = String(imgShots[imgSel].seed); imgSliders(); };
$("img-seed-reuse").onclick = () => { if (imgLastSeed != null) $("img-seed").value = String(imgLastSeed); };
$("img-seed-random").onclick = () => { $("img-seed").value = ""; };
// the picture chosen as the next job's image to start from (an edit's reference, or img2img's)
async function imgUseAsRef(url, name) {
  const blob = await (await fetch(url)).blob();
  const dt = new DataTransfer();
  dt.items.add(new File([blob], name, {type: blob.type}));
  $("img-ref").files = dt.files;
  $("img-ref").dispatchEvent(new Event("change"));
  $("img-sec-ref").open = true;
  showTab("images");
  $("img-sec-ref").scrollIntoView({block: "nearest", behavior: "smooth"});
}
$("img-to-ref").onclick = () => imgUseAsRef(imgShots[imgSel].url, imgShots[imgSel].name);
// the picture into the Tools tab, with the tool chosen there
for (const b of document.querySelectorAll("[data-send-tool]"))
  b.onclick = () => { const s = imgShots[imgSel]; toolOpen(b.dataset.sendTool, s.url, s.name, s.req && s.req.prompt); };
// the sections' open or closed, kept here; on a phone the negative prompt (while empty) and the info start closed
const imgPhone = matchMedia("(max-width: 640px)");
for (const id of ["img-neg-f", "img-sec-ref", "img-lora", "img-guide", "img-sec-out", "img-moss", "img-info-box"]) {
  const d = $(id), was = store.get("img_open_" + id, null);
  d.open = was != null ? was : !(imgPhone.matches && id === "img-info-box") && !(imgPhone.matches && id === "img-neg-f");
  d.addEventListener("toggle", () => store.set("img_open_" + id, d.open));
}
// a phone's software keyboard over a text field of this tab: Generate, held at the bottom, steps out of its way
if (window.visualViewport) {
  const typing = () => {
    const a = document.activeElement;
    const field = a && $("img-form").contains(a) && (a.tagName === "TEXTAREA" || (a.tagName === "INPUT" && !["range", "checkbox", "file"].includes(a.type)));
    $("img-form").classList.toggle("is-typing", !!field && visualViewport.height < window.innerHeight * 0.8);
  };
  visualViewport.addEventListener("resize", typing);
  $("img-form").addEventListener("focusout", () => setTimeout(typing, 100));
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
  imgGuideBody(body);
  const bad = imgMossBody(body) || await imgCtrlBody(body);
  if (bad) { toast("error", tr("img.none", "No image"), bad, 8000); return; }
  const partial = $("img-partial").checked;
  if (partial) Object.assign(body, {stream: true, partial_images: 2});
  imageBusy = true;
  imgAbort = new AbortController();
  let t0 = performance.now();
  imgGoBusy(t0);
  paintLive();
  $("img-hint").textContent = tr("img.generatingHint", "Generating… (a minute or more on a small GPU)");
  try {
    // with an image: an edit (or a model without edits, img2img), in JSON with data: URLs (without an API key the
    // server takes only JSON from a browser page, so that another site's form cannot use it)
    if (refs.length) body.images = await Promise.all(refs.map(async f => ({image_url: await readAsDataURL(f)})));
    const mask = refs.length ? imgMask() : null;
    if (mask) {
      body.mask = {image_url: mask};
      body.crop_edit = $("img-crop").checked;
      if ($("img-color").value !== "off") body.color_match = $("img-color").value;
      if (!$("img-refine-f").hidden && $("img-refine").checked) body.refine = true;
    }
    if (!mask && refs.length === 1 && $("img-ext").value) {
      body.extend = {aspect: $("img-ext").value};
      delete body.size;   // the size is the grown image's
    }
    const url = refs.length ? "v1/images/edits" : "v1/images/generations";
    // Interrupt aborts the request; the server cancels a job whose client left, after the step it is on
    let r, j;
    for (;;) {
      r = await fetch(url, {method: "POST", headers: headers(true), body: JSON.stringify(body), signal: imgAbort.signal});
      j = partial && r.ok ? await imgStream(r) : await r.json();
      if (r.ok || !j.error || j.error.code !== "weights_downloading") break;
      await auxWait(j.error, imgAbort.signal, (p, words) => { $("img-hint").textContent = words; });
      $("img-hint").textContent = tr("img.generatingHint", "Generating… (a minute or more on a small GPU)");
      t0 = performance.now();
      imgGoBusy(t0);
    }
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    const seconds = Math.round((performance.now() - t0) / 1000);
    for (const d of [...j.data].reverse()) {
      const fmt = j.output_format || "png";
      imgAdd({url: `data:image/${fmt};base64,` + d.b64_json, name: `xestrata-${d.seed}.${fmt === "jpeg" ? "jpg" : fmt}`, seed: d.seed,
              prompt, ...imgInfo(url, body, j, d, seconds), note: imgEnhanced(j, d)});
    }
    imgLastSeed = j.data[0].seed;
    $("img-seed-reuse").disabled = false;
    imgShow(0);
    $("img-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
    if (imgPhone.matches) $("img-view").scrollIntoView({block: "start", behavior: "smooth"});
    if (j.xestrata && j.xestrata.made_size && !j.xestrata.crop)
      toast("info", tr("img.madeAt", "Made at {size}", {size: j.xestrata.made_size}), tr("img.madeCutText", "The image model needs sides that are multiples of 16: the middle {size} was cut out.", {size: j.size}), 8000);
    else if (j.xestrata && j.xestrata.requested_size)
      toast("info", tr("img.madeAt", "Made at {size}", {size: j.size}), tr("img.madeNearestText", "This image model makes a fixed set of sizes: {size} is the one whose shape is nearest the {asked} asked for.", {size: j.size, asked: j.xestrata.requested_size}), 8000);
  } catch (e) {
    if (imgAbort.signal.aborted) $("img-hint").textContent = tr("img.interrupted", "Interrupted.");
    else {
      $("img-hint").textContent = "";
      toast("error", tr("img.none", "No image"), String(e.message || e), 8000);
    }
  } finally {
    imageBusy = false;
    imgAbort = null;
    imgGoBusy(null);
    $("img-ref").value = "";   // the reference images go with this job only: the next Generate is a new image
    paintLive();
    await loadHealth();        // the speed the job measured (the count's limit and estimate)
    imgMaskSetup();
  }
};
// Generate, and while a job of this page runs, Interrupt in its place with the seconds so far
let imgAbort = null, imgTimer = 0;
function imgGoBusy(t0) {
  clearInterval(imgTimer);
  $("img-go").classList.toggle("is-busy", t0 != null);
  $("img-go-time").textContent = "";
  if (t0 != null)
    imgTimer = setInterval(() => { $("img-go-time").textContent = tr("img.elapsed", "{n} s", {n: Math.round((performance.now() - t0) / 1000)}); }, 1000);
}
// Enter in a field while a job runs (the form's implicit submission, a click of no pointer on the button) does not stop it
$("img-go").onclick = (ev) => {
  if (!imgAbort) return;
  ev.preventDefault();
  if (ev.detail > 0 || document.activeElement === $("img-go")) imgAbort.abort();
};

// A tool's weights downloading on the server (503 weights_downloading: serve/image_aux.py): wait for them through the
// metrics' image_aux (polled every second), show(share, words) the progress, and return once they are done for the
// caller to ask again; a download that fails, or one the server does not run (a license to agree to), throws.
async function auxWait(err, signal, show) {
  const downloads = err.downloads || [], names = new Set(downloads.map(d => d.name)), before = lastMetrics;
  if (!downloads.length || downloads.some(d => d.status !== "downloading")) throw new Error(err.message || "weights_downloading");
  for (;;) {
    const m = lastMetrics, rows = m && m !== before ? (m.image_aux || []).filter(d => names.has(d.name)) : null;
    const bad = rows && rows.find(d => d.status === "failed");
    if (bad) throw new Error(tr("tool.downloadFailed", "Downloading {name} failed: {why}", {name: bad.name, why: bad.error || ""}));
    if (rows && !rows.length) return;
    const now = rows || downloads, got = now.reduce((a, d) => a + (d.bytes || 0), 0), all = now.reduce((a, d) => a + (d.total || 0), 0);
    show(all ? got / all : 0, tr("tool.fetching", "Downloading the tool's weights ({name}): {got} of {total} MiB. It runs once they are here.",
                                 {name: now.map(d => d.name).join(", "), got: Math.round(got / 1048576), total: Math.round(all / 1048576)}));
    await new Promise((ok, no) => {
      const stop = () => { clearTimeout(t); no(new DOMException("aborted", "AbortError")); };
      const t = setTimeout(() => { signal.removeEventListener("abort", stop); ok(); }, 1000);
      signal.addEventListener("abort", stop, {once: true});
    });
  }
}

// ------------------------------------------------------------------ the Tools tab: the image tools on their own
// Each tool is its API's request: the upscale /v1/images/upscale (model, scale, extra_params: color_correction_method,
// redraw, redraw_strength, redraw_model), the parts' fix the edits' fix ({parts, strength}), the parts found
// /v1/images/segmentations (prompt, task, model, score_threshold, mask_threshold), the background removed
// /v1/images/remove-background (model, operating_resolution, refine_foreground, mask_only, background_color, mask_grow,
// mask_blur, output_format).  The image is one of this browser's files or one the Images tab sends (toolOpen).
const TOOL_NAMES = {upscale: "upscale_image", fix: "fix_image", segment: "find_region", rembg: "remove_background"};
let toolKind = store.get("tool_kind", "upscale"), toolSrc = null, toolAbort = null, toolTimer = 0, toolLast = null;
// the tools some image model offers (/health's defaults.tools): the tab, and each tool's button
function toolsSetup() {
  const ms = health.image_models || [];
  const has = (k) => ms.some(m => ((m.defaults || {}).tools || []).includes(TOOL_NAMES[k]));
  const any = Object.keys(TOOL_NAMES).filter(has);
  $("tab-btn-tools").hidden = !health.image_generation || !any.length;
  for (const b of $("tool-kind").children) b.hidden = !has(b.dataset.tool);
  for (const b of document.querySelectorAll("[data-send-tool]")) b.hidden = !has(b.dataset.sendTool);
  if (any.length && !any.includes(toolKind)) toolKind = any[0];
  // the image model that draws (an upscale's redraw, the parts' fix): one that offers the tool, a choice of several
  const sel = $("tool-model"), was = sel.value;
  const draw = ms.filter(m => ((m.defaults || {}).tools || []).includes(TOOL_NAMES[toolKind === "fix" ? "fix" : "upscale"]));
  sel.replaceChildren(...draw.map(m => new Option(m.id, m.id)));
  if (draw.some(m => m.id === was)) sel.value = was;
  toolShow();
}
// the tool's name (its button) and what it does
const toolRunText = (k) => k === "fix" ? tr("tool.fix", "Fix parts") : k === "segment" ? tr("tool.segment", "Find parts")
  : k === "rembg" ? tr("tool.rembg", "Remove background") : tr("tool.upscale", "Upscale");
const toolAboutText = (k) => k === "fix" ? tr("tool.aboutFix", "Finds faces, hands or people and draws each again, finer. The image keeps its size.")
  : k === "segment" ? tr("tool.aboutSegment", "Finds the parts the words name. A mask found here can be the mask of an edit in the Images tab.")
  : k === "rembg" ? tr("tool.aboutRembg", "Cuts the subject out (BiRefNet): a PNG with a transparent background, or its mask.")
  : tr("tool.aboutUpscale", "Makes the image larger on the GPU; with the redraw, the image model draws the larger picture again for detail.");
// the fields of the tool chosen, the prompt's label (or none), and the button's words
function toolShow() {
  const k = toolKind;
  for (const b of $("tool-kind").children) b.setAttribute("aria-checked", String(b.dataset.tool === k));
  for (const f of document.querySelectorAll("#tool-form .tool-fields")) f.hidden = !f.dataset.tool.split(" ").includes(k);
  $("tool-go-l").textContent = toolRunText(k);
  $("tool-about").textContent = toolAboutText(k);
  const redraw = $("tool-up-redraw").checked;
  $("tool-up-strength-f").hidden = !redraw;
  $("tool-prompt-f").hidden = k === "rembg" || (k === "upscale" && !redraw);
  $("tool-prompt-l").textContent = k === "segment" ? tr("tool.promptParts", "The parts to find (a few words: hair, the sky)")
    : k === "fix" ? tr("tool.promptFix", "Prompt (optional: each part's own words otherwise)")
    : tr("tool.promptRedraw", "What the picture shows (the redraw's prompt)");
  $("tool-model-f").hidden = $("tool-model").options.length < 2 || (k === "upscale" && !redraw);
  $("tool-seed-f").hidden = k === "upscale" && !redraw;
  $("tool-up-about").textContent = tr("img.upMethodTitle", "How the image is made larger: Auto (RealESRGAN when it is drawn again, SeedVR2 when not), Lanczos (a plain resize), RealESRGAN (sharper) or SeedVR2 (restores detail in one step; sharp: for anime and blurred pictures). The weights are downloaded the first time");
  toolUpSize();
}
for (const b of $("tool-kind").children)
  b.onclick = () => { toolKind = b.dataset.tool; store.set("tool_kind", toolKind); toolResult(null); toolsSetup(); };
$("tool-up-redraw").onchange = toolShow;
// the upscale's size: the sides times the scale, multiples of 16, at most 4096 (the server's UPSCALE_MAX_SIDE)
function toolUpSize() {
  const el = $("tool-up-size");
  if (!toolSrc || toolKind !== "upscale") { el.textContent = ""; return; }
  const sc = +$("tool-up-scale").value || 2, [w, h] = [toolSrc.w, toolSrc.h].map(x => Math.max(16, Math.round(x * sc / 16) * 16));
  el.textContent = tr("tool.upSize", "{w} × {h} → {W} × {H}", {w: toolSrc.w, h: toolSrc.h, W: w, H: h}) +
    (Math.max(w, h) > 4096 ? " " + tr("tool.upTooLarge", "(past 4096 a side: a smaller scale)") : "");
}
$("tool-up-scale").addEventListener("input", toolUpSize);
// the image the tool reads: a file of this browser's, one dropped on the tab, or one the Images tab sends
async function toolSetSource(url, name) {
  const im = new Image();
  await new Promise((ok, no) => { im.onload = ok; im.onerror = () => no(new Error(tr("tool.unreadable", "The browser cannot read this image"))); im.src = url; });
  toolSrc = {url, name: name || "image.png", w: im.naturalWidth, h: im.naturalHeight};
  toolLast = null;
  $("tool-before").src = url;
  $("tool-stage").hidden = false;
  $("tool-empty").hidden = true;
  toolResult(null);
  toolUpSize();
}
async function toolFile(f) {
  if (!f) return;
  try { await toolSetSource(await readAsDataURL(f), f.name); } catch (e) { toast("error", tr("tab.tools", "Tools"), String(e.message || e), 6000); }
}
$("tool-file").onchange = () => toolFile($("tool-file").files[0]);
for (const ev of ["dragenter", "dragover"]) $("tool-form").addEventListener(ev, (e) => {
  if ([...(e.dataTransfer.items || [])].some((i) => i.kind === "file")) e.preventDefault();
});
$("tool-form").addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) { e.preventDefault(); toolFile(f); }
});
async function toolOpen(kind, url, name, prompt) {
  toolKind = kind;
  store.set("tool_kind", kind);
  $("tool-file").value = "";
  if (prompt && kind !== "segment") $("tool-prompt").value = prompt;
  showTab("tools");
  toolsSetup();
  await toolSetSource(url, name);
}
// the request of the tool chosen, on the image: [the endpoint, the body]
function toolRequest() {
  const k = toolKind, num = (id) => $(id).value === "" ? undefined : Number($(id).value);
  const prompt = $("tool-prompt").value.trim(), seed = num("tool-seed"), model = $("tool-model-f").hidden ? undefined : $("tool-model").value;
  const body = {image: toolSrc.url};
  if (k === "upscale") {
    Object.assign(body, {model: $("tool-up-model").value, scale: num("tool-up-scale") ?? 2});
    const extra = {};
    if ($("tool-up-color").value) extra.color_correction_method = $("tool-up-color").value;
    if ($("tool-up-redraw").checked) {
      Object.assign(extra, {redraw: true, ...(num("tool-up-strength") != null && {redraw_strength: num("tool-up-strength")}), ...(model && {redraw_model: model})});
      Object.assign(body, {prompt, ...(seed != null && {seed})});
    }
    if (Object.keys(extra).length) body.extra_params = extra;
    return ["v1/images/upscale", {...body, response_format: "b64_json"}];
  }
  if (k === "fix") {
    const parts = [...document.querySelectorAll("[name=tool-part]:checked")].map(x => x.value);
    const req = {images: [{image_url: toolSrc.url}], fix: {parts, ...(num("tool-fix-strength") != null && {strength: num("tool-fix-strength")})}};
    return ["v1/images/edits", {...req, ...(prompt && {prompt}), ...(model && {model}), ...(seed != null && {seed}), response_format: "b64_json"}];
  }
  if (k === "segment") {
    const m = $("tool-seg-model").value;
    return ["v1/images/segmentations", {...body, prompt, task: $("tool-seg-task").value, ...(m && {model: m}),
            ...(num("tool-seg-score") != null && {score_threshold: num("tool-seg-score")}),
            ...(num("tool-seg-mask") != null && {mask_threshold: num("tool-seg-mask")})}];
  }
  return ["v1/images/remove-background", {...body, model: $("tool-bg-model").value,
          ...($("tool-bg-res").value.trim() && {operating_resolution: $("tool-bg-res").value.trim()}),
          refine_foreground: $("tool-bg-refine").checked, mask_only: $("tool-bg-mask").checked,
          ...($("tool-bg-color-on").checked && {background_color: $("tool-bg-color").value}),
          ...(num("tool-bg-grow") != null && {mask_grow: num("tool-bg-grow")}), ...(num("tool-bg-blur") != null && {mask_blur: num("tool-bg-blur")}),
          output_format: $("tool-bg-fmt").value, response_format: "b64_json"}];
}
// the result: the new picture over the image (a divider to compare them), or the parts found drawn over it
function toolResult(r) {
  toolLast = r;
  const after = $("tool-after"), ov = $("tool-overlay");
  after.hidden = ov.hidden = $("tool-split").hidden = $("tool-divider").hidden = true;
  $("tool-found").hidden = $("tool-to-mask").hidden = true;
  $("tool-actions").hidden = $("tool-info-box").hidden = !r;
  if (!r) return;
  $("tool-info").textContent = r.info;
  if (r.url) {
    after.src = r.url;
    after.hidden = $("tool-split").hidden = $("tool-divider").hidden = false;
    toolSplit();
    Object.assign($("tool-save"), {href: r.url, download: r.name});
    $("tool-save").hidden = $("tool-to-ref").hidden = false;
    return;
  }
  $("tool-save").hidden = $("tool-to-ref").hidden = true;
  toolFound(r.found);
}
function toolSplit() {
  $("tool-stage").style.setProperty("--split", `${$("tool-split").value}%`);
  $("tool-split").style.setProperty("--fill", `${$("tool-split").value}%`);
}
$("tool-split").addEventListener("input", toolSplit);
// the parts found: each drawn in its color over the image (its mask, or its box), and listed with its score; the ones
// ticked make the mask "Edit these parts" sends to the Images tab
const TOOL_COLORS = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2"];
async function toolFound(found) {
  const list = $("tool-found");
  list.hidden = false;
  list.replaceChildren(...(found.items.length ? found.items.map((it, i) => {
    const row = document.createElement("label"), cb = document.createElement("input"), sw = document.createElement("span"), t = document.createElement("span");
    row.className = "tool-found__row img-check";
    cb.type = "checkbox"; cb.checked = true; cb.onchange = () => toolDrawFound(found);
    sw.className = "tool-found__swatch"; sw.style.background = TOOL_COLORS[i % TOOL_COLORS.length];
    t.textContent = `${it.label || found.prompt} · ${it.score.toFixed(2)}`;
    row.append(cb, sw, t);
    it.check = cb;
    return row;
  }) : [Object.assign(document.createElement("span"), {className: "composer-hint", textContent: tr("img.regionNone", "Not found: try other words")})]));
  for (const it of found.items) {
    if (!it.mask) continue;
    it.img = new Image();
    await new Promise((ok, no) => { it.img.onload = ok; it.img.onerror = no; it.img.src = "data:image/png;base64," + it.mask; });
  }
  $("tool-to-mask").hidden = !found.items.length || !toolMaskable();
  toolDrawFound(found);
}
const toolMaskable = () => !!(imgModel().defaults || {}).mask;
// the ticked parts as one mask at the image's size: white where they are (their masks, or their boxes)
function toolMaskCanvas(found, color) {
  const c = document.createElement("canvas");
  c.width = found.width; c.height = found.height;
  const g = c.getContext("2d");
  found.items.forEach((it, i) => {
    if (!it.check.checked) return;
    const col = color || TOOL_COLORS[i % TOOL_COLORS.length];
    if (it.img) {
      const t = document.createElement("canvas");
      t.width = c.width; t.height = c.height;
      const tg = t.getContext("2d");
      tg.drawImage(it.img, 0, 0, c.width, c.height);
      tg.globalCompositeOperation = "source-in";   // the mask PNG is white on black: keep its white as the color
      const d = tg.getImageData(0, 0, c.width, c.height);
      for (let p = 0; p < d.data.length; p += 4) d.data[p + 3] = d.data[p] > 127 ? 255 : 0;
      tg.putImageData(d, 0, 0);
      tg.fillStyle = col;
      tg.fillRect(0, 0, c.width, c.height);
      g.drawImage(t, 0, 0);
    } else if (it.box) {
      g.fillStyle = col;
      g.fillRect(it.box[0], it.box[1], it.box[2] - it.box[0], it.box[3] - it.box[1]);
    }
  });
  return c;
}
function toolDrawFound(found) {
  const ov = $("tool-overlay");
  ov.width = found.width; ov.height = found.height;
  const g = ov.getContext("2d");
  g.clearRect(0, 0, ov.width, ov.height);
  g.globalAlpha = 0.45;
  g.drawImage(toolMaskCanvas(found), 0, 0);
  g.globalAlpha = 1;
  g.lineWidth = Math.max(2, ov.width / 300);
  found.items.forEach((it, i) => {
    if (!it.check.checked || !it.box) return;
    g.strokeStyle = TOOL_COLORS[i % TOOL_COLORS.length];
    g.strokeRect(it.box[0], it.box[1], it.box[2] - it.box[0], it.box[3] - it.box[1]);
  });
  ov.hidden = false;
}
// "Edit these parts": the image to the Images tab as its reference, the parts ticked painted into its mask
$("tool-to-mask").onclick = async () => {
  const found = toolLast && toolLast.found, src = toolSrc;
  if (!found) return;
  const m = toolMaskCanvas(found, "#e11d48");
  await imgUseAsRef(src.url, src.name);
  const c = $("img-mask-paint"), img = $("img-mask-img");
  const paint = () => { c.getContext("2d").drawImage(m, 0, 0, c.width, c.height); };
  if (img.complete && img.naturalWidth && c.width) paint(); else img.addEventListener("load", () => setTimeout(paint, 0), {once: true});
};
$("tool-to-ref").onclick = () => toolLast && imgUseAsRef(toolLast.url, toolLast.name);
$("tool-copy").onclick = () => toolLast && copyText(JSON.stringify(toolLast.req, null, 2), $("tool-copy"));
$("tool-sam-license").onclick = (ev) => $("img-sam-license").onclick(ev);
// running: the button is Interrupt with the seconds so far; an image job's progress (the metrics' live.image) beside
function toolBusy(t0) {
  clearInterval(toolTimer);
  $("tool-go").classList.toggle("is-busy", t0 != null);
  $("tool-go-time").textContent = "";
  $("tool-progress").hidden = true;
  $("tool-cap").hidden = true;
  if (t0 == null) return;
  toolTimer = setInterval(() => {
    const s = (performance.now() - t0) / 1000, live = lastMetrics && lastMetrics.live && lastMetrics.live.image;
    $("tool-go-time").textContent = tr("img.elapsed", "{n} s", {n: Math.round(s)});
    if (!live) return;
    const p = live.progress || 0;
    $("tool-progress").hidden = false;
    $("tool-bar").style.width = `${Math.round(100 * p)}%`;
    $("tool-cap").hidden = false;
    $("tool-cap").textContent = imageState(live)[0] + (p > 0.05 ? " · " + tr("tool.left", "about {n} s left", {n: Math.round(s * (1 - p) / p)}) : "");
  }, 500);
}
$("tool-go").onclick = (ev) => {
  if (!toolAbort) return;
  ev.preventDefault();
  if (ev.detail > 0 || document.activeElement === $("tool-go")) toolAbort.abort();
};
$("tool-form").onsubmit = async (ev) => {
  ev.preventDefault();
  if (toolAbort) return;
  if (!toolSrc) { toast("warn", tr("tab.tools", "Tools"), tr("tool.noImage", "Choose an image first."), 5000); return; }
  const [url, body] = toolRequest();
  if (!$("tool-prompt-f").hidden && toolKind !== "fix" && !body.prompt) { $("tool-prompt").focus(); return; }
  if (toolKind === "fix" && !body.fix.parts.length) { toast("warn", tr("tool.fix", "Fix parts"), tr("tool.noParts", "Tick a part to find."), 5000); return; }
  toolAbort = new AbortController();
  imageBusy = true;
  paintLive();
  let t0 = performance.now();
  toolBusy(t0);
  $("tool-hint").textContent = "";
  try {
    let r, j;
    for (;;) {
      r = await fetch(url, {method: "POST", headers: headers(true), body: JSON.stringify(body), signal: toolAbort.signal});
      j = await r.json();
      if (r.ok || !j.error || j.error.code !== "weights_downloading") break;
      await auxWait(j.error, toolAbort.signal, (p, words) => {
        $("tool-progress").hidden = false;
        $("tool-bar").style.width = `${Math.round(100 * p)}%`;
        $("tool-cap").hidden = false;
        $("tool-cap").textContent = words;
      });
      t0 = performance.now();
      toolBusy(t0);
    }
    if (!r.ok) throw new Error((j.error || {}).message || r.statusText);
    const seconds = Math.round((performance.now() - t0) / 1000);
    if (url.endsWith("segmentations")) {
      const items = j.data.map(d => ({score: d.score, box: d.box, mask: d.mask, label: d.label}));
      toolResult({req: body, info: imgInfoText(url, body, [`model: ${j.model}`, `task: ${j.task}`, `found: ${items.length}`, `seconds: ${seconds}`]),
                  found: {items, width: j.width, height: j.height, prompt: body.prompt}});
      $("tool-sam-license").hidden = !(j.xestrata && /license/.test(j.xestrata.fallback || ""));
    } else {
      const fmt = j.output_format || "png", d = j.data[0];
      const fix = j.xestrata && j.xestrata.fix;   // the parts' fix's regions found and drawn again
      const answer = [`model: ${j.model}`, j.size ? `size: ${j.size}` : j.width ? `size: ${j.width}x${j.height}` : "", d.seed != null ? `seed: ${d.seed}` : "",
                      fix ? `regions: ${(fix.regions || []).map(x => `${x.part} ${x.score.toFixed(2)}`).join(", ") || 0}` : "", `seconds: ${seconds}`].filter(Boolean);
      const req = d.seed != null && body.seed == null && (body.prompt || body.fix) ? {...body, seed: d.seed} : body;
      toolResult({url: `data:image/${fmt};base64,` + d.b64_json, name: `xestrata-${toolKind}.${fmt === "jpeg" ? "jpg" : fmt}`, req, info: imgInfoText(url, req, answer)});
    }
    $("tool-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
  } catch (e) {
    if (toolAbort.signal.aborted) $("tool-hint").textContent = tr("img.interrupted", "Interrupted.");
    else toast("error", toolRunText(toolKind), String(e.message || e), 10000);
  } finally {
    toolAbort = null;
    imageBusy = false;
    toolBusy(null);
    paintLive();
  }
};

// ------------------------------------------------------------------ Video
// The video tools on a video of this browser's.  The upscale is /v1/videos/sync for a short video (the file in the
// answer) and a /v1/videos job for a longer one: it runs on the server, the page may close, and the jobs list (GET
// /v1/videos: this page's and the API's) shows it.  The source plays from the browser's file, a result from the server
// (its job's id is the access, as an image file's link).  A job's progress is its record's; the frames of the one
// running are the metrics' live.video.
const VID_SYNC_S = 20;   // seconds: a video this short is upscaled in the answer, a longer one as a job
let vidSrc = null, vidAbort = null, vidTimer = 0, vidLast = null, vidJobsTimer = 0, vidJob = null;
// the segmentation: the tool chosen, the answer shown ({answer, fps, mine, hidden, gone, urls}), the clicks given (each
// {time, target: "new<n>" or an object's id, x, y, label} or {time, target, box}), the click mode, the new objects
let vidKind = store.get("vid_kind", "upscale"), vidSeg = null, vidClicks = [], vidMode = "", vidNewN = 1, vidDrag = null;
function videoSetup() {
  const has = (t) => (health.image_models || []).some(m => ((m.defaults || {}).tools || []).includes(t));
  const ups = has("upscale_image"), seg = has("segment_video");
  $("tab-btn-video").hidden = !health.image_generation || !health.video || !(ups || seg);
  for (const b of $("vid-kind").children) b.hidden = !(b.dataset.vid === "upscale" ? ups : seg);
  if (!(vidKind === "segment" ? seg : ups)) vidKind = ups ? "upscale" : "segment";
  vidKindSet(vidKind);
  const off = health.video && !health.video.available;
  $("vid-go").disabled = !!off;
  if (off) $("vid-hint").textContent = tr("vid.noFfmpeg", "The video tools need ffmpeg on the server: xestrata --setup says how to install it.");
}
// a duration as m:ss (or h:mm:ss)
function vidClock(s) {
  s = Math.max(0, Math.round(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
// the size it will be: the sides times the scale, even, at most 4096 a side
function vidUpSize() {
  const el = $("vid-up-size");
  if (!vidSrc || !vidSrc.w) { el.textContent = ""; return; }
  const sc = +$("vid-up-scale").value || 2, [w, h] = [vidSrc.w, vidSrc.h].map(x => Math.round(x * sc / 2) * 2);
  el.textContent = tr("tool.upSize", "{w} × {h} → {W} × {H}", {w: vidSrc.w, h: vidSrc.h, W: w, H: h}) +
    (Math.max(w, h) > 4096 ? " " + tr("tool.upTooLarge", "(past 4096 a side: a smaller scale)") : "");
}
$("vid-up-scale").addEventListener("input", vidUpSize);
// what is known of the source: the browser's (size, length, file size), then the server's (frame rate, frames, VFR)
function vidSrcInfo(server) {
  if (!vidSrc) { $("vid-src-info").textContent = ""; return; }
  const parts = [];
  if (vidSrc.w) parts.push(`${vidSrc.w} × ${vidSrc.h}`);
  if (vidSrc.duration) parts.push(vidClock(vidSrc.duration));
  parts.push(`${fmt(vidSrc.file.size / 1048576, 1)} MiB`);
  if (server) {
    parts.push(tr("vid.fps", "{n} fps", {n: fmt(server.fps, 2)}), tr("vid.frames", "{n} frames", {n: server.frames}));
    if (server.vfr) parts.push(tr("vid.vfr", "a varying frame rate, read at its average"));
  }
  if (vidSrc.unplayable) parts.push(tr("vid.unplayable", "this browser cannot play it; the server still reads it"));
  $("vid-src-info").textContent = parts.join(" · ");
}
function vidSetSource(f) {
  if (!f) return;
  vidClicks = [];
  vidSegShow(null);
  if (vidSrc) URL.revokeObjectURL(vidSrc.url);
  vidSrc = {file: f, url: URL.createObjectURL(f), w: 0, h: 0, duration: 0, unplayable: false};
  const v = $("vid-src");
  v.onloadedmetadata = () => { vidSrc.w = v.videoWidth; vidSrc.h = v.videoHeight; vidSrc.duration = v.duration; vidSrcInfo(); vidUpSize(); };
  v.onerror = () => { vidSrc.unplayable = true; vidSrcInfo(); };
  v.src = vidSrc.url;
  vidResult(null);
  vidShow("source");
  vidSrcInfo();
  vidUpSize();
}
$("vid-file").onchange = () => vidSetSource($("vid-file").files[0]);
for (const ev of ["dragenter", "dragover"]) $("vid-form").addEventListener(ev, (e) => {
  if ([...(e.dataTransfer.items || [])].some((i) => i.kind === "file")) e.preventDefault();
});
$("vid-form").addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) { e.preventDefault(); vidSetSource(f); }
});
// the player: the source, the result, or both side by side; with both, the result follows the source
function vidShow(which) {
  const have = {source: !!vidSrc, result: !!(vidLast && vidLast.url)};
  if (which === "both" && !(have.source && have.result)) which = have.result ? "result" : "source";
  if (!have[which] && which !== "both") which = have.source ? "source" : "result";
  $("vid-view").dataset.show = which;
  for (const b of $("vid-show").children) {
    b.setAttribute("aria-checked", String(b.dataset.show === which));
    b.hidden = b.dataset.show === "both" ? !(have.source && have.result) : !have[b.dataset.show];
  }
  $("vid-src").hidden = !have.source || which === "result";
  $("vid-out").hidden = !have.result || which === "source";
  $("vid-empty").hidden = have.source || have.result;
  $("vid-bar").hidden = !(have.source || have.result);
  if (which !== "both") $("vid-out").muted = false;
  vidSegDraw();
}
for (const b of $("vid-show").children) b.onclick = () => vidShow(b.dataset.show);
const vidBoth = () => $("vid-view").dataset.show === "both";
$("vid-src").addEventListener("play", () => { if (vidBoth()) { $("vid-out").muted = true; $("vid-out").currentTime = $("vid-src").currentTime; $("vid-out").play(); } });
$("vid-src").addEventListener("pause", () => { if (vidBoth()) $("vid-out").pause(); });
$("vid-src").addEventListener("seeked", () => { if (vidBoth()) $("vid-out").currentTime = $("vid-src").currentTime; });
// a frame back or on (the result's rate where there is one, else the source's as the server read it, else 30)
function vidStep(n) {
  const fps = (vidLast && vidLast.fps) || (vidSeg && vidSeg.fps) || 30;
  for (const v of [$("vid-src"), $("vid-out")]) {
    if (v.hidden || !v.src) continue;
    v.pause();
    v.currentTime = Math.max(0, v.currentTime + n / fps);
  }
}
$("vid-prev").onclick = () => vidStep(-1);
$("vid-next").onclick = () => vidStep(1);
// the result: its file to play and save, the request and the server's figures
function vidResult(r) {
  if (vidLast && vidLast.blob) URL.revokeObjectURL(vidLast.url);
  vidLast = r;
  $("vid-save").hidden = $("vid-copy").hidden = $("vid-info-box").hidden = !r;
  if (!r) { $("vid-out").removeAttribute("src"); $("vid-out").load(); vidShow("source"); return; }
  $("vid-out").src = r.url;
  Object.assign($("vid-save"), {href: r.url, download: r.name});
  $("vid-info").textContent = r.info;
  vidShow(vidSrc && r.mine ? "both" : "result");
}
$("vid-copy").onclick = () => vidLast && copyText(JSON.stringify(vidLast.req, null, 2), $("vid-copy"));
// the request's fields (/v1/videos' form): the model, the scale, the seed, extra_params
function vidRequest() {
  const num = (id) => $(id).value === "" ? undefined : Number($(id).value);
  const req = {model: $("vid-up-model").value, scale: num("vid-up-scale") ?? 2}, extra = {};
  if (num("vid-seed") != null) req.seed = num("vid-seed");
  if ($("vid-up-color").value) extra.color_correction_method = $("vid-up-color").value;
  if (num("vid-chunk") != null) extra.chunk_frames = num("vid-chunk");
  if (num("vid-overlap") != null) extra.temporal_overlap = num("vid-overlap");
  if (Object.keys(extra).length) req.extra_params = extra;
  return req;
}
function vidForm(req, field = "input_references") {
  const fd = new FormData();
  for (const [k, v] of Object.entries(req)) fd.append(k, typeof v === "object" ? JSON.stringify(v) : String(v));
  fd.append(field, vidSrc.file, vidSrc.file.name);
  return fd;
}
// a POST of the form with its upload's progress (fetch tells none): {status, headers(name), body: a Blob}
function vidPost(url, fd, signal, uploaded) {
  return new Promise((ok, no) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    for (const [k, v] of Object.entries(headers(false))) x.setRequestHeader(k, v);
    x.setRequestHeader("X-XeStrata-Upload", "1");   // the server's mark of its own page's form (_foreign_page)
    x.responseType = "blob";
    x.upload.onprogress = (e) => { if (e.lengthComputable) uploaded(e.loaded / e.total); };
    x.onload = () => ok({status: x.status, header: (n) => x.getResponseHeader(n), body: x.response});
    x.onerror = () => no(new Error(tr("vid.netError", "The connection to the server failed")));
    x.onabort = () => no(new DOMException("aborted", "AbortError"));
    signal.addEventListener("abort", () => x.abort());
    x.send(fd);
  });
}
async function vidErrorBody(r) {
  try { return JSON.parse(await r.body.text()).error || {}; } catch { return {}; /* not JSON */ }
}
async function vidError(r, e) {
  e = e || await vidErrorBody(r);
  const ra = r.header("Retry-After");
  return (e.message || `HTTP ${r.status}`) + (r.status === 503 && ra ? " " + tr("vid.retry", "(try again in about {n} s)", {n: ra}) : "");
}
// running: the button is Interrupt with the seconds so far; the upload's share, then the frames done
function vidBusy(t0) {
  clearInterval(vidTimer);
  $("vid-go").classList.toggle("is-busy", t0 != null);
  $("vid-go-time").textContent = "";
  $("vid-progress").hidden = true;
  $("vid-cap").textContent = "";
  if (t0 == null) return;
  vidTimer = setInterval(() => {
    const s = (performance.now() - t0) / 1000, run = vidAbort;
    $("vid-go-time").textContent = tr("img.elapsed", "{n} s", {n: Math.round(s)});
    let p = null, cap = "";
    if (run && run.aux) {
      [p, cap] = run.aux;
    } else if (run && run.upload < 1) {
      p = run.upload;
      cap = tr("vid.uploading", "Sending the video · {pct}%", {pct: Math.round(100 * p)});
    } else {
      const v = (lastMetrics && lastMetrics.live && lastMetrics.live.video) || (vidJob && vidJob.rec && {done: vidJob.rec.frames_done || 0, total: (vidJob.rec.source || {}).frames || 0});
      if (v && v.total) {
        p = Math.min(1, v.done / v.total);
        cap = tr("vid.frameOf", "Frame {i} of {n}", {i: v.done, n: v.total}) + (v.left_s ? " · " + tr("tool.left", "about {n} s left", {n: v.left_s}) : "");
      }
    }
    $("vid-progress").hidden = p == null;
    if (p != null) $("vid-bar-fill").style.width = `${Math.round(100 * p)}%`;
    $("vid-cap").textContent = cap;
  }, 500);
}
$("vid-go").onclick = (ev) => {
  if (!vidAbort) return;
  ev.preventDefault();
  if (ev.detail > 0 || document.activeElement === $("vid-go")) vidAbort.ctl.abort();
};
const vidInfo = (req, path, answer) => [`POST /${path}`, ...Object.entries(req).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`),
                                        `${path.includes("segment") ? "video" : "input_references"}: (${vidSrc ? vidSrc.file.name : "?"})`, "", ...answer].join("\n");
$("vid-form").onsubmit = async (ev) => {
  ev.preventDefault();
  if (vidAbort) return;
  if (!vidSrc) { toast("warn", tr("tab.video", "Video"), tr("vid.noVideo", "Choose a video first."), 5000); return; }
  const seg = vidKind === "segment", req = seg ? vidSegRequest() : vidRequest(), v = health.video || {};
  if (seg && !req.prompt && !(req.clicks || []).length) {
    toast("warn", tr("vid.segment", "Find objects"), tr("vid.segNothing", "Give the words of what to find, or click on the video."), 6000);
    return;
  }
  const sync = vidSrc.duration > 0 && vidSrc.duration <= VID_SYNC_S && vidSrc.file.size <= (v.sync_max_mib || 512) * 1048576;
  const path = seg ? "v1/videos/segmentations" : sync ? "v1/videos/sync" : "v1/videos";
  if (seg && !sync) req.job = true;
  if (!sync && v.max_upload_mib && vidSrc.file.size > v.max_upload_mib * 1048576) {
    toast("error", tr("tool.upscale", "Upscale"), tr("vid.tooLarge", "The video is larger than the server takes ({n} MiB: video.max_upload_mib).", {n: v.max_upload_mib}), 10000);
    return;
  }
  vidAbort = {ctl: new AbortController(), upload: 0};
  imageBusy = true;
  paintLive();
  let t0 = performance.now();
  vidBusy(t0);
  $("vid-hint").textContent = "";
  vidResult(null);
  try {
    let r;
    for (;;) {
      r = await vidPost(path, vidForm(req, seg ? "video" : "input_references"), vidAbort.ctl.signal, (p) => { vidAbort.upload = p; });
      vidAbort.upload = 1;
      if (r.status === 200) break;
      const e = await vidErrorBody(r);
      if (e.code !== "weights_downloading") throw new Error(await vidError(r, e));
      await auxWait(e, vidAbort.ctl.signal, (p, words) => { vidAbort.aux = [p, words]; });
      Object.assign(vidAbort, {aux: null, upload: 0});
      t0 = performance.now();
      vidBusy(t0);
    }
    if (seg && sync) {
      const seconds = Math.round((performance.now() - t0) / 1000);
      vidSegAnswer(JSON.parse(await r.body.text()), req, path, seconds);
      $("vid-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
    } else if (sync) {
      const seconds = Math.round((performance.now() - t0) / 1000), fps = Number(r.header("X-Video-Fps")) || 0;
      const answer = [`model: ${r.header("X-Model")}`, `frames: ${r.header("X-Video-Frames")}`, `fps: ${fmt(fps, 3)}`,
                      r.header("X-Video-Vfr") ? `vfr: ${r.header("X-Video-Vfr")}` : "", `seconds: ${seconds}`].filter(Boolean);
      const ext = (r.body.type || "video/mp4").includes("webm") ? "webm" : (r.body.type || "").includes("matroska") ? "mkv" : "mp4";
      vidResult({url: URL.createObjectURL(r.body), blob: true, name: `xestrata-upscale.${ext}`, fps, req, mine: true, info: vidInfo(req, path, answer)});
      $("vid-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
    } else {
      const rec = JSON.parse(await r.body.text());
      vidSrcInfo(rec.source);
      vidJob = {id: rec.id, rec, req};
      vidJobs();
      await vidFollow(t0);
    }
  } catch (e) {
    if (vidAbort.ctl.signal.aborted) $("vid-hint").textContent = tr("img.interrupted", "Interrupted.");
    else toast("error", seg ? tr("vid.segment", "Find objects") : tr("tool.upscale", "Upscale"), String(e.message || e), 12000);
  } finally {
    vidAbort = null;
    vidJob = null;
    imageBusy = false;
    vidBusy(null);
    paintLive();
    vidJobs();
  }
};
// this page's job, followed to its end: its record every 2 s; Interrupt deletes it on the server
async function vidFollow(t0) {
  const id = vidJob.id;
  for (;;) {
    if (vidAbort.ctl.signal.aborted) {
      await fetch(`v1/videos/${id}`, {method: "DELETE", headers: headers()}).catch(() => {});
      return;
    }
    await new Promise((ok) => setTimeout(ok, 2000));
    const r = await fetch(`v1/videos/${id}`, {headers: headers()});
    if (!r.ok) throw new Error(tr("vid.lost", "The job is gone from the server."));
    const rec = vidJob.rec = await r.json();
    if (rec.status === "completed") {
      vidOpenJob(rec, vidJob.req, Math.round((performance.now() - t0) / 1000));
      return;
    }
    if (rec.status === "failed") throw new Error((rec.error && rec.error.message) || rec.status);
  }
}
// a completed job's result in the player: its file from the server (a segmentation's: its answer, and its overlay video)
async function vidOpenJob(rec, req, seconds) {
  const res = rec.result || {}, up = res.upscale || {};
  if (rec.kind === "segmentation") {
    const r = await fetch(`v1/videos/${rec.id}/content`, {headers: headers()});
    if (!r.ok) { toast("error", tr("vid.segment", "Find objects"), `HTTP ${r.status}`, 8000); return; }
    vidSegAnswer(await r.json(), req, "v1/videos/segmentations", seconds != null ? seconds : res.seconds, rec);
    if (seconds != null) $("vid-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
    return;
  }
  const answer = [`id: ${rec.id}`, `model: ${rec.model}`, `size: ${rec.size}`, res.frames != null ? `frames: ${res.frames}` : "",
                  res.fps ? `fps: ${fmt(res.fps, 3)}` : "", res.vfr ? "vfr: read at its average frame rate" : "",
                  up.method ? `upscaler: ${up.method}${up.variant ? " " + up.variant : ""}` : "", up.chunk ? `chunk: ${up.chunk}` : "",
                  res.seed != null ? `seed: ${res.seed}` : "", `seconds: ${seconds != null ? seconds : res.seconds}`].filter(Boolean);
  vidResult({url: `v1/videos/${rec.id}/content`, name: rec.file_name || `${rec.id}.mp4`, fps: res.fps, req: req || {model: rec.model, size: rec.size},
             mine: !!req, info: (req ? vidInfo(req, "v1/videos", answer) : answer.join("\n"))});
  if (seconds != null) $("vid-hint").textContent = tr("img.doneIn", "Done in {n} s.", {n: seconds});
}
// the jobs on the server, newest first: each one's state, progress and time left, and its result to open, save or
// delete; asked again every 3 s while one runs and the tab is open
async function vidJobs() {
  clearTimeout(vidJobsTimer);
  if (tab !== "video" || $("tab-btn-video").hidden) return;
  let list;
  try {
    const r = await fetch("v1/videos?limit=20", {headers: headers()});
    if (!r.ok) return;
    list = (await r.json()).data || [];
  } catch { return; }
  const now = Date.now() / 1000;
  const rows = list.map((rec) => {
    const row = document.createElement("div"), what = document.createElement("span"), acts = document.createElement("span");
    row.className = "vid-job";
    acts.className = "vid-job__acts";
    const state = {queued: tr("vid.queued", "Waiting"), in_progress: tr("vid.running", "Running"), completed: tr("vid.done", "Done"),
                   failed: tr("vid.failed", "Failed")}[rec.status] || rec.status;
    let detail = `${state} · ${rec.model} · ${rec.size}`;
    if (rec.status === "in_progress" && rec.progress > 0 && rec.started_at)
      detail += ` · ${rec.progress}% · ` + tr("tool.left", "about {n} s left", {n: Math.round((now - rec.started_at) * (100 - rec.progress) / rec.progress)});
    if (rec.status === "failed" && rec.error) detail += ` · ${rec.error.message}`;
    what.textContent = `${new Date(rec.created_at * 1000).toLocaleString()} · ${detail}`;
    const btn = (label, run) => { const b = document.createElement("button"); b.type = "button"; b.className = "st-btn st-btn--secondary"; b.textContent = label; b.onclick = run; return b; };
    if (rec.status === "completed") {
      acts.append(btn(tr("vid.open", "Open"), () => vidOpenJob(rec, null, null)));
      const a = Object.assign(document.createElement("a"), {className: "st-btn st-btn--secondary", href: `v1/videos/${rec.id}/content`,
                                                             download: rec.file_name || "", textContent: tr("img.save", "Save")});
      acts.append(a);
    }
    const live = rec.status === "queued" || rec.status === "in_progress";
    acts.append(btn(live ? tr("vid.cancel", "Cancel") : tr("vid.delete", "Delete"), async () => {
      await fetch(`v1/videos/${rec.id}`, {method: "DELETE", headers: headers()}).catch(() => {});
      vidJobs();
    }));
    row.append(what, acts);
    return row;
  });
  $("vid-jobs").replaceChildren(...(rows.length ? rows : [Object.assign(document.createElement("span"), {className: "composer-hint", textContent: tr("vid.noJobs", "No jobs yet.")})]));
  if (list.some((r) => r.status === "queued" || r.status === "in_progress")) vidJobsTimer = setTimeout(vidJobs, 3000);
}
// the chat's note while a video job runs on the server (live.video): its requests answered between the job's chunks,
// or not until it ends where the text model made way for it
function chatVideoNote(v) {
  const el = $("chat-video-note");
  el.hidden = !v;
  if (!v) return;
  el.textContent = v.chat === "refused"
    ? tr("vid.chatRefused", "A video job has the GPU (the model made way for it): the chat answers again in about {n} s.", {n: v.left_s || "?"})
    : tr("vid.chatBetween", "A video job is running: the chat answers between its chunks, a little later than usual.");
}

// ------------------------------------------------------------------ Video: finding objects
// The segmentation (/v1/videos/segmentations, SAM 3.1): words, and points and boxes clicked on the paused source (at
// its time; the server takes the frame there), each making a new object or fixing one found before.  Its answer's masks
// (vLLM-Omni's COCO RLE) are drawn over the source as it plays, each object shown or hidden; its videos (the overlay,
// the masks) come as files.  A frame and its objects' mask go to the Images tab's edit.
const VID_COLORS = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
                    [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]];   // the engine's overlay's, by id
const vidColor = (id) => VID_COLORS[id % VID_COLORS.length];
function vidKindSet(k) {
  vidKind = k;
  store.set("vid_kind", k);
  for (const b of $("vid-kind").children) b.setAttribute("aria-checked", String(b.dataset.vid === k));
  for (const el of $("vid-form").querySelectorAll("[data-vid]")) if (el.parentElement !== $("vid-kind")) el.hidden = el.dataset.vid !== k;
  const key = k === "segment" ? "vid.segment" : "tool.upscale";
  $("vid-go-label").dataset.i18n = key;
  $("vid-go-label").textContent = k === "segment" ? tr("vid.segment", "Find objects") : tr("tool.upscale", "Upscale");
  vidModeSet(k === "segment" ? vidMode : "");
  vidSegDraw();
}
for (const b of $("vid-kind").children) b.onclick = () => vidKindSet(b.dataset.vid);
function vidModeSet(m) {
  vidMode = m;
  for (const b of $("vid-seg-mode").children) b.setAttribute("aria-checked", String(b.dataset.mode === m));
  $("vid-canvas").classList.toggle("is-clicking", vidKind === "segment" && m !== "");
  vidSegDraw();
}
for (const b of $("vid-seg-mode").children) b.onclick = () => vidModeSet(b.dataset.mode);
// the clicks' targets: the new objects so far, then the objects of the answer shown
function vidTargets() {
  const sel = $("vid-seg-target"), keep = sel.value;
  const opts = [];
  for (let n = 1; n <= vidNewN; n++) opts.push([`new${n}`, tr("vid.segNewN", "New object {n}", {n})]);
  for (const o of (vidSeg && vidSeg.answer.objects) || []) opts.push([String(o.id), tr("vid.segObj", "Object {id}", {id: o.id}) + (o.prompt ? ` (${o.prompt})` : "")]);
  sel.replaceChildren(...opts.map(([v, t]) => Object.assign(document.createElement("option"), {value: v, textContent: t})));
  sel.value = opts.some(([v]) => v === keep) ? keep : opts[0][0];
}
$("vid-seg-new").onclick = () => { vidNewN += 1; vidTargets(); $("vid-seg-target").value = `new${vidNewN}`; };
vidTargets();
function vidClickList() {
  const chips = vidClicks.map((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "st-btn st-btn--secondary";
    const what = c.box ? tr("vid.segBox", "Box") : c.label ? tr("vid.segIn", "Point: in") : tr("vid.segOut", "Point: out");
    const to = c.target.startsWith("new") ? tr("vid.segNewN", "New object {n}", {n: c.target.slice(3)}) : tr("vid.segObj", "Object {id}", {id: c.target});
    b.textContent = `${fmt(c.time, 2)} s · ${what} → ${to} ×`;
    b.title = tr("vid.segRemove", "Remove");
    b.onclick = () => { vidClicks.splice(i, 1); vidClickList(); vidSegDraw(); };
    return b;
  });
  $("vid-seg-clicks").replaceChildren(...chips);
}
// the request's fields: the words (the first as prompt, the rest as prompts), the clicks grouped by their time and
// target (one object each), the videos asked for
function vidSegRequest() {
  const words = $("vid-seg-words").value.split(",").map(w => w.trim()).filter(Boolean);
  const req = {model: "sam3.1"};
  if (words.length) req.prompt = words[0];
  if (words.length > 1) req.prompts = words.slice(1);
  // the server reads int(time * fps + 1e-3): three decimals turn one frame in three of a 30 fps source into the frame
  // before (k/fps = 0.0333... sent as 0.033). Six keep the frame the page shows, and group clicks on the same frame.
  const clickTime = (t) => Number(t.toFixed(6));
  const groups = new Map();
  for (const c of vidClicks) {
    const key = `${clickTime(c.time)}|${c.target}`;
    if (!groups.has(key)) groups.set(key, {time: clickTime(c.time), object: c.target.startsWith("new") ? null : Number(c.target), points: []});
    const g = groups.get(key);
    if (c.box) g.box = c.box.map(Math.round);
    else g.points.push([Math.round(c.x), Math.round(c.y), c.label]);
  }
  if (groups.size) req.clicks = [...groups.values()].map((g) => { if (!g.points.length) delete g.points; return g; });
  const outs = [];
  if ($("vid-seg-overlay").checked) outs.push("overlay");
  if ($("vid-seg-mask").checked) outs.push("mask");
  if (outs.length) req.outputs = outs;
  if ($("vid-seg-masks").checked) req.mask_video = true;
  return req;
}
// the canvas over the source: placed on the picture the video shows (its letterbox left out), at its pixels
function vidCanvasPlace() {
  const v = $("vid-src"), c = $("vid-canvas");
  if (!v.videoWidth || v.hidden) { c.hidden = true; return false; }
  const ew = v.clientWidth, eh = v.clientHeight, k = Math.min(ew / v.videoWidth, eh / v.videoHeight);
  const w = v.videoWidth * k, h = v.videoHeight * k;
  if (c.width !== v.videoWidth || c.height !== v.videoHeight) { c.width = v.videoWidth; c.height = v.videoHeight; }
  Object.assign(c.style, {left: `${v.offsetLeft + (ew - w) / 2}px`, top: `${v.offsetTop + (eh - h) / 2}px`, width: `${w}px`, height: `${h}px`});
  c.hidden = false;
  return true;
}
// the frame the source shows: its time at the answer's rate (the server's frame of a click at that time)
const vidFrameAt = (t, fps) => Math.floor(t * fps + 1e-3);
// an object's mask on a frame from its RLE (the runs down the columns, the first the background's) into `img`'s pixels
function vidPaintRle(img, counts, W, H, rgb, a) {
  let p = 0;
  for (let i = 0; i < counts.length; i++) {
    const n = counts[i];
    if (i % 2) {
      for (let q = p; q < p + n; q++) {
        const o = ((q % H) * W + Math.floor(q / H)) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = a;
      }
    }
    p += n;
  }
}
// the objects shown on the source's frame, then the clicks of this time
function vidSegDraw() {
  const c = $("vid-canvas"), v = $("vid-src");
  const want = vidKind === "segment" && ((vidSeg && vidSeg.mine) || vidClicks.length || vidMode !== "");
  if (!want || !vidCanvasPlace()) { c.hidden = true; return; }
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  if (vidSeg && vidSeg.mine) {
    const A = vidSeg.answer, f = A.frames[Math.min(A.frames.length - 1, vidFrameAt(v.currentTime, vidSeg.fps))];
    if (f && f.instances.length) {
      const img = g.createImageData(A.width, A.height);
      for (const it of f.instances)
        if (!vidSeg.hidden.has(it.id) && !vidSeg.gone.has(it.id)) vidPaintRle(img, it.mask.counts, A.width, A.height, vidColor(it.id), 120);
      g.putImageData(img, 0, 0);
    }
  }
  const r = Math.max(4, c.width / 120);
  g.lineWidth = Math.max(2, c.width / 400);
  for (const k of [...vidClicks, ...(vidDrag ? [vidDrag] : [])]) {
    if (Math.abs(k.time - v.currentTime) > 1e-3) continue;
    g.strokeStyle = g.fillStyle = k.box ? "#facc15" : k.label ? "#22c55e" : "#ef4444";
    if (k.box) { g.strokeRect(k.box[0], k.box[1], k.box[2] - k.box[0], k.box[3] - k.box[1]); continue; }
    g.beginPath(); g.arc(k.x, k.y, r, 0, 2 * Math.PI); g.fill();
  }
}
for (const ev of ["timeupdate", "seeked", "loadedmetadata", "pause"]) $("vid-src").addEventListener(ev, vidSegDraw);
window.addEventListener("resize", vidSegDraw);
// while it plays, each frame shown
$("vid-src").addEventListener("play", () => {
  const v = $("vid-src");
  const tick = () => { if (v.paused || v.ended) return; vidSegDraw(); v.requestVideoFrameCallback ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick); };
  tick();
});
// a click (a point) or a drag (a box) on the paused source, at the video's pixels
function vidCanvasXY(e) {
  const c = $("vid-canvas"), b = c.getBoundingClientRect();
  return [(e.clientX - b.left) * c.width / b.width, (e.clientY - b.top) * c.height / b.height];
}
$("vid-canvas").addEventListener("pointerdown", (e) => {
  if (vidKind !== "segment" || !vidMode) return;
  const v = $("vid-src");
  if (!v.paused) { v.pause(); toast("info", tr("vid.segment", "Find objects"), tr("vid.segPaused", "Paused: click on this frame."), 3000); return; }
  e.preventDefault();
  const [x, y] = vidCanvasXY(e), target = $("vid-seg-target").value || "new1", time = v.currentTime;
  if (vidMode !== "box") {
    vidClicks.push({time, target, x, y, label: Number(vidMode)});
    vidClickList(); vidSegDraw();
    return;
  }
  vidDrag = {time, target, box: [x, y, x, y], x0: x, y0: y};
  $("vid-canvas").setPointerCapture(e.pointerId);
});
$("vid-canvas").addEventListener("pointermove", (e) => {
  if (!vidDrag) return;
  const [x, y] = vidCanvasXY(e);
  vidDrag.box = [Math.min(vidDrag.x0, x), Math.min(vidDrag.y0, y), Math.max(vidDrag.x0, x), Math.max(vidDrag.y0, y)];
  vidSegDraw();
});
$("vid-canvas").addEventListener("pointerup", () => {
  if (!vidDrag) return;
  const {time, target, box} = vidDrag;
  vidDrag = null;
  if (box[2] - box[0] >= 4 && box[3] - box[1] >= 4) {
    for (let i = vidClicks.length - 1; i >= 0; i--)   // one box an object on a frame: a new one takes its place
      if (vidClicks[i].box && vidClicks[i].target === target && Math.abs(vidClicks[i].time - time) < 1e-3) vidClicks.splice(i, 1);
    vidClicks.push({time, target, box});
  }
  vidClickList(); vidSegDraw();
});
// an answer shown: its objects listed, its masks over the source (when it is this page's video), its videos as files
function vidSegShow(seg) {
  if (vidSeg) for (const u of vidSeg.urls) URL.revokeObjectURL(u);
  vidSeg = seg;
  $("vid-objects").hidden = $("vid-seg-bar").hidden = !seg;
  vidTargets();
  if (!seg) { $("vid-objects").replaceChildren(); $("vid-seg-files").replaceChildren(); vidSegDraw(); return; }
  const rows = seg.answer.objects.map((o) => {
    const row = document.createElement("label"), sw = document.createElement("span"), box = document.createElement("input"), del = document.createElement("button");
    row.className = "vid-obj";
    sw.className = "vid-obj__swatch";
    sw.style.background = `rgb(${vidColor(o.id).join(",")})`;
    box.type = "checkbox";
    box.checked = true;
    box.title = tr("vid.segShowObj", "Show");
    box.onchange = () => { if (box.checked) seg.hidden.delete(o.id); else seg.hidden.add(o.id); vidSegDraw(); };
    del.type = "button";
    del.className = "st-btn st-btn--icon";
    del.textContent = "×";
    del.title = tr("vid.segRemove", "Remove");
    del.onclick = (ev) => {
      ev.preventDefault();
      if (seg.gone.has(o.id)) seg.gone.delete(o.id); else seg.gone.add(o.id);
      row.classList.toggle("is-gone", seg.gone.has(o.id));
      vidSegDraw();
    };
    const what = o.prompt || tr("vid.segClicked", "clicked");
    row.append(box, sw, document.createTextNode(`#${o.id} ${what}`), del);
    row.ondblclick = () => { $("vid-seg-target").value = String(o.id); };
    return row;
  });
  $("vid-objects").replaceChildren(...(rows.length ? rows : [Object.assign(document.createElement("span"), {className: "composer-hint", textContent: tr("vid.segNone", "Nothing found: try other words, or click on it.")})]));
  $("vid-seg-files").replaceChildren(...seg.files.map(([label, href, name]) => Object.assign(document.createElement("a"),
    {className: "st-btn st-btn--secondary", href, download: name, textContent: label})));
  const editable = toolMaskable() && seg.mine;
  $("vid-seg-edit").hidden = $("vid-seg-edit-bg").hidden = !editable;
  vidSegDraw();
}
// a segmentation's answer (the synchronous one, or a job's content with its record) shown
function vidSegAnswer(answer, req, path, seconds, rec) {
  const urls = [], files = [];
  const blobOf = (b64, type) => { const u = URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), ch => ch.charCodeAt(0))], {type})); urls.push(u); return u; };
  let overlay = null;
  if (rec) {
    files.push([tr("vid.segFileJson", "Masks (JSON)"), `v1/videos/${rec.id}/content`, `${rec.id}.json`]);
    const fl = rec.files || {};
    if (fl.overlay) overlay = `v1/videos/${rec.id}/content?variant=overlay`;
    if (fl.mask) files.push([tr("vid.segFileMask", "Mask video"), `v1/videos/${rec.id}/content?variant=mask`, fl.mask]);
    for (const o of answer.objects) if (fl[`object_${o.id}`]) files.push([tr("vid.segFileObj", "Object {id}'s mask", {id: o.id}), `v1/videos/${rec.id}/content?variant=object_${o.id}`, fl[`object_${o.id}`]]);
  } else {
    if (answer.overlay_video) overlay = blobOf(answer.overlay_video, "video/mp4");
    if (answer.mask_union_video) files.push([tr("vid.segFileMask", "Mask video"), blobOf(answer.mask_union_video, "video/mp4"), "xestrata-masks.mp4"]);
    for (const o of answer.objects) if (o.mask_video) files.push([tr("vid.segFileObj", "Object {id}'s mask", {id: o.id}), blobOf(o.mask_video, "video/mp4"), `xestrata-object-${o.id}.mp4`]);
    delete answer.overlay_video; delete answer.mask_union_video;
    for (const o of answer.objects) delete o.mask_video;
    files.unshift([tr("vid.segFileJson", "Masks (JSON)"), (() => { const u = URL.createObjectURL(new Blob([JSON.stringify(answer)], {type: "application/json"})); urls.push(u); return u; })(), "xestrata-segmentation.json"]);
  }
  const mine = !!req && !!vidSrc;
  const info = [`objects: ${answer.objects.map(o => `#${o.id} ${o.prompt || "click " + o.click}`).join(", ") || "none"}`,
                `frames: ${answer.num_frames}`, `fps: ${fmt(answer.fps, 3)}`, answer.vfr ? `vfr: ${answer.vfr}` : "",
                seconds != null ? `seconds: ${seconds}` : ""].filter(Boolean);
  if (overlay) vidResult({url: overlay, name: rec ? `${rec.id}-overlay.mp4` : "xestrata-overlay.mp4", fps: answer.fps, req: req || {model: "sam3.1"}, mine,
                          info: req ? vidInfo(req, path, info) : info.join("\n")});
  else { vidResult(null); $("vid-info-box").hidden = false; $("vid-info").textContent = req ? vidInfo(req, path, info) : info.join("\n"); }
  vidSegShow({answer, fps: answer.fps, mine, hidden: new Set(), gone: new Set(), urls, files});
  if (mine) vidShow("source");
}
// the source's frame shown, with the objects kept (or all but them) as the mask, to the Images tab's edit
async function vidSegEdit(background) {
  const v = $("vid-src"), A = vidSeg && vidSeg.answer;
  if (!A || !v.videoWidth) return;
  v.pause();
  const pic = document.createElement("canvas");
  pic.width = v.videoWidth; pic.height = v.videoHeight;
  pic.getContext("2d").drawImage(v, 0, 0);
  const m = document.createElement("canvas");
  m.width = A.width; m.height = A.height;
  const g = m.getContext("2d"), img = g.createImageData(A.width, A.height);
  const f = A.frames[Math.min(A.frames.length - 1, vidFrameAt(v.currentTime, vidSeg.fps))];
  for (const it of (f && f.instances) || []) if (!vidSeg.gone.has(it.id) && !vidSeg.hidden.has(it.id)) vidPaintRle(img, it.mask.counts, A.width, A.height, [225, 29, 72], 255);
  if (background) for (let p = 3; p < img.data.length; p += 4) { const on = img.data[p] === 0; img.data[p - 3] = 225; img.data[p - 2] = 29; img.data[p - 1] = 72; img.data[p] = on ? 255 : 0; }
  g.putImageData(img, 0, 0);
  await imgUseAsRef(pic.toDataURL("image/png"), `frame-${vidFrameAt(v.currentTime, vidSeg.fps)}.png`);
  const c = $("img-mask-paint"), el = $("img-mask-img");
  const paint = () => { c.getContext("2d").drawImage(m, 0, 0, c.width, c.height); };
  if (el.complete && el.naturalWidth && c.width) paint(); else el.addEventListener("load", () => setTimeout(paint, 0), {once: true});
}
$("vid-seg-edit").onclick = () => vidSegEdit(false);
$("vid-seg-edit-bg").onclick = () => vidSegEdit(true);

// a video attached in the chat: sent to the server first (POST /v1/videos/files: the chat's tools take it from there,
// upscale_video), then attached as a text of its link and what the server read of it, which the model sees
async function chatVideo(f) {
  const v = health.video || {};
  if (v.max_upload_mib && f.size > v.max_upload_mib * 1048576) {
    toast("error", tr("vid.attach", "Video"), tr("vid.tooLarge", "The video is larger than the server takes ({n} MiB: video.max_upload_mib).", {n: v.max_upload_mib}), 10000);
    return;
  }
  const fd = new FormData();
  fd.append("file", f, f.name);
  toast("info", tr("vid.attach", "Video"), tr("vid.sending", "Sending {name} to the server…", {name: f.name}), 4000);
  try {
    const r = await vidPost("v1/videos/files", fd, new AbortController().signal, () => {});
    if (r.status !== 200) throw new Error(await vidError(r));
    const j = JSON.parse(await r.body.text());
    const text = `Video: ${j.url} (${j.width}x${j.height}, ${fmt(j.fps, 2)} fps, ${j.frames} frames, ${fmt(j.duration, 1)} s` +
      (j.vfr ? ", a varying frame rate" : "") + ")";
    attachments.push({kind: "file", name: f.name, text});
    renderAttachments();
  } catch (e) {
    toast("error", tr("vid.attach", "Video"), String(e.message || e), 10000);
  }
}


// ------------------------------------------------------------------ the translation page
// The translation model's (health.translation): the text translated through /v1/chat/completions, its settings in
// chat_template_kwargs, the translation streamed into the right pane.  The languages are the model's (/v1/models),
// named in the page's language by the browser (Intl.DisplayNames).
let trnModel = null, trnAbort = null, trnStarted = null;
const trnTag = (code) => ({"zh-hans": "zh-Hans", "zh-hant": "zh-Hant"})[code] || code;
function trnName(code) {
  try { return new Intl.DisplayNames([StrataI18n.lang], {type: "language"}).of(trnTag(code)) || code; } catch (e) { return code; }
}
async function trnLoad() {
  if (trnModel || !health.translation) return;
  let m = null;
  try {
    const j = await (await fetch("v1/models", {headers: headers()})).json();
    m = (j.data || []).find((x) => x.translation);
  } catch (e) { /* tried again when the tab is opened again */ }
  if (!m) return;
  trnModel = m.id;
  const codes = m.translation.languages || [];
  const opts = codes.map((c) => [c, trnName(c)]).sort((a, b) => a[1].localeCompare(b[1], StrataI18n.lang));
  const html = opts.map(([c, n]) => `<option value="${esc(c)}">${esc(n)}</option>`).join("");
  $("trn-source").insertAdjacentHTML("beforeend", html);
  $("trn-target").innerHTML = html;
  // the target: the one chosen last, else the page's language (English where the page is Japanese: text in Japanese
  // most often goes out)
  const page = StrataI18n.lang === "zh" ? "zh-hans" : StrataI18n.lang;
  const fallback = page === "ja" ? "en" : codes.includes(page) ? page : "ja";
  const last = store.get("trn_target", "");
  $("trn-target").value = codes.includes(last) ? last : fallback;
  const src = store.get("trn_source", "");
  $("trn-source").value = codes.includes(src) ? src : "";
  $("trn-form").value = store.get("trn_form", "");
  trnFormShown();
}
function trnFormShown() { $("trn-form-field").hidden = $("trn-target").value !== "ja"; }
$("trn-target").onchange = () => { store.set("trn_target", $("trn-target").value); trnFormShown(); };
$("trn-source").onchange = () => store.set("trn_source", $("trn-source").value);
$("trn-form").onchange = () => store.set("trn_form", $("trn-form").value);
$("trn-swap").onclick = () => {
  const s = $("trn-source").value, t = $("trn-target").value, out = $("trn-output").textContent;
  if (s) { $("trn-target").value = s; $("trn-source").value = t; }
  else $("trn-source").value = t;                      // a detected source: the target becomes the source
  if (out) { $("trn-input").value = out; $("trn-output").textContent = ""; $("trn-copy").disabled = true; }
  store.set("trn_target", $("trn-target").value);
  store.set("trn_source", $("trn-source").value);
  trnFormShown();
};
$("trn-copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("trn-output").textContent);
    toast("success", tr("trn.copied", "Translation copied"));
  } catch (e) { toast("warn", tr("trn.copyFailed", "Could not copy"), String(e.message || e)); }
};
// the glossary's lines, "source => translation" each
function trnGlossary() {
  const out = [];
  for (const line of $("trn-glossary").value.split("\n")) {
    const i = line.indexOf("=>");
    if (i < 0) continue;
    const a = line.slice(0, i).trim(), b = line.slice(i + 2).trim();
    if (a && b) out.push({source: a, target: b});
  }
  return out;
}
function trnBusy(on) {
  $("trn-go").classList.toggle("is-busy", on);
  trnStarted = on ? performance.now() : null;
  if (!on) return;
  const tick = () => {
    if (trnStarted == null) return;
    $("trn-status").textContent = tr("img.elapsed", "{n} s", {n: Math.round((performance.now() - trnStarted) / 1000)});
    setTimeout(tick, 500);
  };
  tick();
}
async function trnRun() {
  const text = $("trn-input").value;
  if (!text.trim() || !trnModel) return;
  const kw = {target_lang: $("trn-target").value};
  if ($("trn-source").value) kw.source_lang = $("trn-source").value;
  if (kw.target_lang === "ja" && $("trn-form").value) kw.form = $("trn-form").value;
  if ($("trn-context").value.trim()) kw.context = $("trn-context").value;
  const gl = trnGlossary();
  if (gl.length) kw.glossary = gl;
  trnAbort = new AbortController();
  trnBusy(true);
  $("trn-output").textContent = "";
  $("trn-copy").disabled = true;
  let got = "";
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), signal: trnAbort.signal,
      body: JSON.stringify({model: trnModel, stream: true, messages: [{role: "user", content: text}],
                            chat_template_kwargs: kw})});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { const j = await r.json(); msg = (j.error && (j.error.message || j.error)) || j.message || msg; } catch (e) { /* the status */ }
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:") || line === "data: [DONE]") continue;
        let e;
        try { e = JSON.parse(line.slice(5).trim()); } catch (err) { continue; }
        const c = e.choices && e.choices[0];
        if (c && c.delta && c.delta.content) { got += c.delta.content; $("trn-output").textContent = got; }
        if (c && c.finish_reason === "error") throw new Error(tr("trn.failed", "The translation stopped"));
      }
    }
    $("trn-status").textContent = got ? tr("trn.done", "Done in {n} s", {n: ((performance.now() - trnStarted) / 1000).toFixed(1)}) : "";
  } catch (e) {
    if (e.name !== "AbortError") toast("error", tr("trn.error", "Translation failed"), String(e.message || e));
    $("trn-status").textContent = "";
  } finally {
    trnBusy(false);
    trnAbort = null;
    $("trn-copy").disabled = !got;
  }
}
$("tr-form").onsubmit = (ev) => {
  ev.preventDefault();
  if (trnAbort) trnAbort.abort(); else trnRun();
};
$("trn-input").onkeydown = (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); if (!trnAbort) trnRun(); }
};

// ------------------------------------------------------------------ LoRA training (serve/lora_train.py)
// The chat's train_lora tool opens this panel (data-ltw under the tool block); the Images view's "Train a LoRA"
// button opens the same panel in a dialog.  One job a panel: the setup (images, kind, name, trigger, model,
// captions, settings), the run (stage, step, loss, the GPU's give-and-take, cancel), and the versions (a sample
// picture each, pick one, register it as a LoRA; the unselected ones can go).
const ltwMounted = new WeakSet();
function ltwHydrate(root) {
  root.querySelectorAll("[data-ltw]").forEach((el) => {
    if (ltwMounted.has(el)) return;
    ltwMounted.add(el);
    ltwPanel(el);
  });
}
const LTW_ACTIVE = ["queued", "preprocessing", "training", "sampling", "captioning"];
async function ltwFetch(path, opts) {
  const r = await fetch(path.replace(/^\//, ""), opts);
  let j = null;
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((j && j.error && (j.error.message || j.error)) || `HTTP ${r.status}`);
  return j;
}
function ltwTrigger() {
  const letters = "bcdfghjklmnpqrstvwxyz";
  return [...crypto.getRandomValues(new Uint8Array(8))].map((n) => letters[n % letters.length]).join("");
}
function ltwPanel(el) {
  const st = {models: [], job: null, timer: null, view: "", picked: null, regName: "", styleWords: null, trigger: ltwTrigger()};
  let edits = Promise.resolve();
  const base = () => st.job ? `v1/lora/jobs/${st.job.id}` : "";
  const stop = () => { if (st.timer) { clearInterval(st.timer); st.timer = null; } };
  const err = (e) => toast("error", tr("ltw.title", "LoRA training"), String(e.message || e), 6000);
  async function loadModels() {
    try { st.models = (await ltwFetch("v1/lora/models", {headers: headers()})).models || []; }
    catch (e) { st.models = []; }
  }
  async function refresh() {
    if (!st.job) return;
    try {
      st.job = await ltwFetch(base(), {headers: headers()});
      render();
      if (!LTW_ACTIVE.includes(st.job.status)) stop();
    } catch (e) { stop(); err(e); }
  }
  function poll() { stop(); st.timer = setInterval(refresh, 2000); }
  function fields() {
    return {model: el.querySelector("[data-ltw-model]").value, kind: el.querySelector("[data-ltw-kind]").value,
            name: el.querySelector("[data-ltw-name]").value.trim(), trigger: el.querySelector("[data-ltw-trigger]").value.trim()};
  }
  async function addFiles(files) {
    const pics = [...files].filter((f) => /^image\/(png|jpeg|webp|bmp)/.test(f.type));
    if (!pics.length) return;
    try {
      if (!st.job) {
        const f = fields();
        if (!f.name || !f.trigger) { err(new Error(tr("ltw.needFields", "Give the LoRA a name and a trigger word first."))); return; }
        const fd = new FormData();
        for (const [k, v] of Object.entries(f)) fd.append(k, v);
        pics.forEach((p) => fd.append("images", p, p.name));
        st.job = await ltwFetch("v1/lora/jobs", {method: "POST", headers: {...headers(), "X-XeStrata-Upload": "1"}, body: fd});
      } else {
        const fd = new FormData();
        pics.forEach((p) => fd.append("images", p, p.name));
        st.job = await ltwFetch(`${base()}/images`, {method: "POST", headers: {...headers(), "X-XeStrata-Upload": "1"}, body: fd});
      }
      render();
    } catch (e) { err(e); }
  }
  async function chatPics() {
    const urls = [...new Set([...document.querySelectorAll("#view-chat img")]
      .map((i) => i.getAttribute("src") || "").filter((u) => u.includes("/v1/images/files/")))];
    if (!urls.length) { toast("warn", tr("ltw.title", "LoRA training"), tr("ltw.noChatPics", "This chat has no images to pick.")); return; }
    const box = el.querySelector("[data-ltw-chatpics]");
    box.hidden = false;
    box.innerHTML = urls.map((u, i) =>
      `<label class="ltw-pick"><input type="checkbox" data-ltw-pick="${i}"><img src="${esc(u)}" alt=""></label>`).join("") +
      `<div class="ltw-row"><button type="button" class="st-btn" data-ltw-addpicked>${esc(tr("ltw.addPicked", "Add the picked pictures"))}</button></div>`;
    box.querySelector("[data-ltw-addpicked]").onclick = async () => {
      const picked = [...box.querySelectorAll("[data-ltw-pick]")].filter((c) => c.checked).map((c) => urls[+c.dataset.ltwPick]);
      const files = [];
      for (const u of picked) {
        try {
          const b = await (await fetch(u.replace(/^\//, ""), {headers: headers()})).blob();
          files.push(new File([b], u.split("/").pop(), {type: b.type || "image/png"}));
        } catch (e) { /* the server's file is gone */ }
      }
      box.hidden = true;
      if (files.length) await addFiles(files);
    };
  }
  async function autoCaption() {
    try {
      const words = el.querySelector("[data-ltw-style-words]");
      const styleWords = words ? words.value.trim() : "";
      await saveCaptions();
      st.job = await ltwFetch(`${base()}/caption`, {method: "POST", headers: headers(true), body: JSON.stringify({style_words: styleWords})});
      render(); poll();
    }
    catch (e) { err(e); }
  }
  async function saveCaptions() {
    const caps = {};
    el.querySelectorAll("[data-ltw-cap]").forEach((t) => {
      const note = el.querySelector(`[data-ltw-note="${CSS.escape(t.dataset.ltwCap)}"]`);
      caps[t.dataset.ltwCap] = {caption: t.value, note: note.value};
    });
    edits = edits.catch(() => {}).then(async () => {
      st.job = await ltwFetch(`${base()}/captions`, {method: "PUT", headers: headers(true), body: JSON.stringify(caps)});
    });
    await edits;
    el.querySelectorAll("[data-ltw-cap], [data-ltw-note]").forEach((t) => {
      const file = t.dataset.ltwCap || t.dataset.ltwNote;
      const saved = caps[file] && caps[file][t.dataset.ltwCap ? "caption" : "note"];
      if (t.value === saved) t.defaultValue = saved;
    });
  }
  function settings() {
    const g = (k) => { const i = el.querySelector(`[data-ltw-set="${k}"]`); return i ? i.value : ""; };
    const out = {};
    for (const k of ["steps", "save_every", "max_size", "rank", "lr", "seed", "dropout", "accum", "base", "method",
                     "train_blocks", "only_if_contains", "ignore_if_contains", "sample_seed"]) {
      const v = g(k);
      if (v !== "" || k === "train_blocks") out[k] = v;
    }
    const sp = g("sample_prompts");
    if (sp.trim()) out.sample_prompts = sp.split("\n").map((s) => s.trim()).filter(Boolean);
    return out;
  }
  async function start() {
    try { await saveCaptions(); st.job = await ltwFetch(`${base()}/start`, {method: "POST", headers: headers(true), body: JSON.stringify(settings())}); render(); poll(); }
    catch (e) { err(e); }
  }
  async function act(action, body) {
    try { st.job = await ltwFetch(`${base()}/${action}`, {method: "POST", headers: headers(true), body: JSON.stringify(body || {})}); render(); if (LTW_ACTIVE.includes(st.job.status)) poll(); }
    catch (e) { err(e); }
  }
  async function register() {
    const step = st.picked != null ? st.picked : (st.job.versions.at(-1) || {}).step;
    const name = (el.querySelector("[data-ltw-regname]") || {}).value || st.job.name;
    try { st.job = await ltwFetch(`${base()}/register`, {method: "POST", headers: headers(true), body: JSON.stringify({step, name})}); render(); toast("ok", tr("ltw.title", "LoRA training"), tr("ltw.registeredToast", "The LoRA is in the model's list; the chat's images can use it now.")); }
    catch (e) { err(e); }
  }
  async function deleteUnselected() {
    const keep = st.picked != null ? st.picked : (st.job.versions.at(-1) || {}).step;
    try {
      for (const v of st.job.versions) {
        if (v.step !== keep && !v.registered) await ltwFetch(`${base()}/versions/${v.step}`, {method: "DELETE", headers: headers()});
      }
      await refresh();
    } catch (e) { err(e); }
  }
  function setupHtml() {
    const j = st.job;
    const models = st.models.filter((m) => m.can_train);
    const requested = el.dataset.ltwRequested || "";
    const selected = j ? j.model : ((models.find((m) => m.model === requested || (m.requested_models || []).includes(requested)) || models[0] || {}).model || "");
    const modelOpts = (st.models.length ? st.models : [{model: "", name: tr("ltw.noModels", "no trainable model here"), can_train: false, why: ""}])
      .map((m) => `<option value="${esc(m.model)}"${selected === m.model ? " selected" : ""}${m.can_train ? "" : " disabled"}>${esc(m.name || m.model)}${m.can_train ? "" : " — " + esc(m.why || "")}</option>`).join("");
    const kind = j ? j.kind : "character";
    const preset = ((models.find((m) => m.model === selected) || models[0] || {}).presets || {})[kind] || {};
    const imgs = j ? j.images.map((im) =>
      `<div class="ltw-img"><img src="${base()}/images/${esc(im.file)}" alt="${esc(im.file)}">` +
      `<div><textarea class="st-input" rows="2" data-ltw-cap="${esc(im.file)}" placeholder="${esc(tr("ltw.captionPh", "trigger, description"))}">${esc(im.caption || "")}</textarea>` +
      `<textarea class="st-input" rows="2" data-ltw-note="${esc(im.file)}" placeholder="${esc(tr("ltw.notePh", "Image note for captioning (optional)"))}">${esc(im.note || "")}</textarea></div>` +
      `<button type="button" class="st-btn st-btn--icon" data-ltw-delimg="${esc(im.file)}" title="${esc(tr("ltw.remove", "Remove"))}">×</button></div>`).join("") : "";
    return `<div class="ltw-head">${esc(tr("ltw.title", "LoRA training"))}</div>
      <div class="ltw-grid">
        <label class="st-field"><span>${esc(tr("ltw.model", "Model"))}</span><select class="st-input" data-ltw-model${j ? " disabled" : ""}>${modelOpts}</select></label>
        <label class="st-field"><span>${esc(tr("ltw.kind", "Kind"))}</span><select class="st-input" data-ltw-kind${j ? " disabled" : ""}>
          <option value="character"${kind === "character" ? " selected" : ""}>${esc(tr("ltw.character", "Character"))}</option>
          <option value="style"${kind === "style" ? " selected" : ""}>${esc(tr("ltw.style", "Style"))}</option>
          <option value="other"${kind === "other" ? " selected" : ""}>${esc(tr("ltw.other", "Other"))}</option></select></label>
        <label class="st-field"><span>${esc(tr("ltw.name", "Name"))}</span><input class="st-input" data-ltw-name value="${esc(j ? j.name : "")}"${j ? " disabled" : ""}></label>
        <label class="st-field"><span>${esc(tr("ltw.trigger", "Trigger word"))}</span><input class="st-input" data-ltw-trigger value="${esc(j ? j.trigger : st.trigger)}"${j ? " disabled" : ""}></label>
      </div>
      <div class="muted small">${esc(tr("ltw.triggerHint", "Use a short invented word, not a known name: known words can change the subject. Before training, compare one or two images with and without the word, no LoRA, using the same seed."))}</div>
      <div class="ltw-drop" data-ltw-drop tabindex="0">${esc(tr("ltw.drop", "Drop the training pictures here (several at once)"))}
        <div class="ltw-row">
          <button type="button" class="st-btn st-btn--secondary" data-ltw-choose>${esc(tr("ltw.choose", "Choose files…"))}</button>
          <button type="button" class="st-btn st-btn--secondary" data-ltw-fromchat>${esc(tr("ltw.fromChat", "Pick this chat's images"))}</button>
          <input type="file" data-ltw-file accept="image/png,image/jpeg,image/webp,image/bmp" multiple hidden>
        </div></div>
      <div class="ltw-chatpics" data-ltw-chatpics hidden></div>
      ${j ? `<div class="ltw-imgs">${imgs}</div>
      ${j.kind === "style" ? `<label class="st-field"><span>${esc(tr("ltw.styleWords", "Style names and features to learn (optional, comma-separated)"))}</span><input class="st-input" data-ltw-style-words value="${esc(st.styleWords == null ? (j.style_words || "") : st.styleWords)}"${j.status !== "draft" ? " disabled" : ""}></label>` : ""}
      <div class="ltw-row">
        <button type="button" class="st-btn st-btn--secondary" data-ltw-autocap${j.status !== "draft" ? " disabled" : ""}>${esc(tr("ltw.autoCaption", "Caption automatically"))}</button>
        <span class="muted small">${esc(tr("ltw.captionHint", "One line each: the trigger word, then the description. The lora-caption rules: a character's own look stays with the trigger; a style captions the content only."))}</span>
      </div>
      <details class="st-collapse ltw-settings"><summary>${esc(tr("ltw.settings", "Settings"))} <span class="muted small">${esc(tr("ltw.presetLine", "{n} steps, a version every {k}", {n: preset.steps || 3000, k: preset.save_every || 250}))}</span></summary>
        <div class="st-collapse__body ltw-grid">
          <label class="st-field"><span>${esc(tr("ltw.steps", "Most steps (n)"))}</span><input class="st-input" type="number" min="1" data-ltw-set="steps" placeholder="${preset.steps || 3000}"></label>
          <label class="st-field"><span>${esc(tr("ltw.saveEvery", "Save a version every (k)"))}</span><input class="st-input" type="number" min="1" data-ltw-set="save_every" placeholder="${preset.save_every || 250}"></label>
          <label class="st-field"><span>${esc(tr("ltw.maxSize", "Maximum training image size"))}</span><select class="st-input" data-ltw-set="max_size"><option value="1024">1024</option><option value="768">768</option><option value="512">512</option></select></label>
          <label class="st-field"><span>${esc(tr("ltw.rank", "Rank"))}</span><input class="st-input" type="number" min="1" data-ltw-set="rank" placeholder="${preset.rank || 32}"></label>
          <label class="st-field"><span>${esc(tr("ltw.lr", "Learning rate"))}</span><input class="st-input" data-ltw-set="lr" placeholder="${preset.lr || 1e-4}"></label>
          <label class="st-field"><span>${esc(tr("ltw.seed", "Seed"))}</span><input class="st-input" type="number" data-ltw-set="seed" placeholder="${preset.seed != null ? preset.seed : 42}"></label>
          <label class="st-field"><span>${esc(tr("ltw.dropout", "Caption dropout"))}</span><input class="st-input" data-ltw-set="dropout" placeholder="${preset.dropout != null ? preset.dropout : ""}"></label>
          ${preset.accum != null ? `<label class="st-field"><span>${esc(tr("ltw.accum", "Gradient accumulation"))}</span><input class="st-input" type="number" min="1" data-ltw-set="accum" placeholder="${preset.accum}"></label>` : ""}
          ${preset.base ? `<label class="st-field"><span>${esc(tr("ltw.base", "Base precision"))}</span><select class="st-input" data-ltw-set="base"><option value="">${esc(preset.base)}</option><option value="bf16">bf16</option><option value="int8">int8</option></select></label>` : ""}
          ${preset.method ? `<label class="st-field"><span>${esc(tr("ltw.method", "Method"))}</span><select class="st-input" data-ltw-set="method"><option value="">${esc(preset.method)}</option><option value="lora">lora</option><option value="lokr">lokr</option><option value="dora">dora</option></select></label>` : ""}
          <label class="st-field"><span>${esc(tr("ltw.trainBlocks", "Train blocks (e.g. 0-5,10)"))}</span><input class="st-input" data-ltw-set="train_blocks" value="${esc(preset.train_blocks || "")}"></label>
          <label class="st-field"><span>${esc(tr("ltw.onlyIf", "Only names containing"))}</span><input class="st-input" data-ltw-set="only_if_contains"></label>
          <label class="st-field"><span>${esc(tr("ltw.ignoreIf", "Ignore names containing"))}</span><input class="st-input" data-ltw-set="ignore_if_contains"></label>
          <label class="st-field"><span>${esc(tr("ltw.sampleSeed", "Sample seed"))}</span><input class="st-input" type="number" data-ltw-set="sample_seed" placeholder="7"></label>
          <label class="st-field ltw-wide"><span>${esc(tr("ltw.samplePrompts", "Sample prompts (one a line; {trigger} is the trigger word)"))}</span>
            <textarea class="st-input" rows="2" data-ltw-set="sample_prompts"></textarea></label>
        </div></details>
      <div class="ltw-row">
        <button type="button" class="st-btn" data-ltw-start${j.status !== "draft" || !j.images.length ? " disabled" : ""}>${esc(tr("ltw.start", "Start the training"))}</button>
        <button type="button" class="st-btn st-btn--secondary" data-ltw-deljob>${esc(tr("ltw.deleteJob", "Delete this job"))}</button>
        ${j.status === "captioning" ? `<span class="st-badge st-badge--generating">${esc(tr("ltw.captioning", "Writing captions…"))}</span>` : ""}
      </div>` : ""}`;
  }
  function runHtml() {
    const j = st.job, p = j.progress || {}, pl = j.place || {};
    const pct = p.total ? Math.min(100, Math.round(100 * (p.step || 0) / p.total)) : 0;
    const stage = {queued: tr("ltw.queued", "Waiting"), preprocessing: tr("ltw.preprocess", "Preparing the pictures (VAE and text)"),
                   training: tr("ltw.training", "Training"), sampling: tr("ltw.sampling", "Making the sample pictures")}[j.status] || j.status;
    const mode = {run: ["st-badge--generating", tr("ltw.modeRun", "running full")],
                  shrunk: ["st-badge--reading", tr("ltw.modeShrunk", "slowed (sharing the GPU)")],
                  paused: ["st-badge--queued", tr("ltw.modePaused", "paused")]}[pl.mode] || ["", pl.mode || ""];
    const eta = p.eta_s != null ? tr("ltw.eta", "about {t} left", {t: p.eta_s > 3600 ? `${(p.eta_s / 3600).toFixed(1)} h` : `${Math.ceil(p.eta_s / 60)} min`}) : "";
    return `<div class="ltw-head">${esc(tr("ltw.title", "LoRA training"))} · ${esc(j.name)} <span class="muted small">${esc(j.model)}</span></div>
      <div class="ltw-row"><span class="st-badge ${mode[0]}">${esc(stage)}${pl.mode ? " · " + esc(mode[1]) : ""}</span>
        ${pl.name ? `<span class="muted small">${esc(pl.name)}${pl.budget_mib != null ? ` · ${fmt(pl.budget_mib / 1024, 1)} GiB` : ""}</span>` : ""}</div>
      ${pl.reason ? `<div class="muted small">${esc(pl.reason)}</div>` : ""}
      <div class="ltw-bar"><div class="ltw-bar__fill" style="width:${pct}%"></div></div>
      <div class="ltw-row muted small">
        <span>${esc(tr("ltw.stepLine", "step {i} of {n}", {i: p.step || 0, n: p.total || (j.settings || {}).steps || "?"}))}</span>
        ${p.loss != null ? `<span>${esc(tr("ltw.loss", "loss"))} ${fmt(p.loss, 5)}</span>` : ""}
        ${p.s_per_step ? `<span>${fmt(p.s_per_step, 2)} s/step</span>` : ""}
        ${eta ? `<span>${esc(eta)}</span>` : ""}
      </div>
      <div class="ltw-row"><button type="button" class="st-btn st-btn--secondary" data-ltw-cancel>${esc(tr("ltw.cancel", "Cancel"))}</button>
        <span class="muted small">${esc(tr("ltw.shareHint", "The chat stays usable; image jobs go first. The versions saved so far stay on a cancel."))}</span></div>`;
  }
  function versionsHtml() {
    const j = st.job;
    const note = {awaiting_choice: tr("ltw.awaiting", "Pick a version and register it as a LoRA (the last one by default)."),
                  cancelled: tr("ltw.cancelledNote", "Cancelled. The versions saved so far are here to pick."),
                  interrupted: tr("ltw.interruptedNote", "The server stopped mid-job. Resume it, or pick a version saved so far."),
                  completed: tr("ltw.completedNote", "Registered. The LoRA is in the model's list.")}[j.status] || "";
    const canContinue = ["awaiting_choice", "completed"].includes(j.status) && j.state_bytes && j.settings.steps < 100000;
    const cards = j.versions.map((v) =>
      `<label class="ltw-ver${v.registered ? " ltw-ver--reg" : ""}">
        <input type="radio" name="ltw-ver-${esc(j.id)}" data-ltw-ver="${v.step}"${st.picked === v.step || (st.picked == null && v === j.versions.at(-1)) ? " checked" : ""}${v.registered ? " disabled" : ""}>
        <div class="ltw-ver__step">${esc(tr("ltw.stepN", "step {n}", {n: v.step}))}${v.registered ? ` · ${esc(tr("ltw.registered", "registered"))}` : ""}</div>
        <div class="ltw-ver__pics">${(v.samples || []).map((s) => `<img src="${base()}/samples/${esc(s)}" alt="${esc(tr("ltw.sample", "sample"))}" loading="lazy">`).join("") || `<span class="muted small">${esc(tr("ltw.noSample", "no sample yet"))}</span>`}</div>
      </label>`).join("");
    return `<div class="ltw-head">${esc(tr("ltw.title", "LoRA training"))} · ${esc(j.name)}</div>
      ${j.error ? `<div class="msg-error">${esc(j.error.message || "")}</div>` : ""}
      <div class="muted small">${esc(note)}</div>
      <div class="ltw-vers">${cards}</div>
      ${canContinue ? `<div class="ltw-row"><label class="st-field"><span>${esc(tr("ltw.continueTotal", "New total steps"))}</span>
        <input class="st-input" type="number" min="${j.settings.steps + 1}" max="100000" value="${Math.min(100000, j.settings.steps + j.settings.save_every)}" data-ltw-continue-total></label>
        <button type="button" class="st-btn st-btn--secondary" data-ltw-continue>${esc(tr("ltw.continue", "Continue training"))}</button>
        <span class="muted small">${esc(tr("ltw.continueState", "The latest continuation state uses {size} MiB; earlier versions do not have a separate state.", {size: fmt(j.state_bytes / 1048576, 1)}))}</span></div>` : ""}
      <div class="ltw-row">
        <input class="st-input" data-ltw-regname placeholder="${esc(j.name)}" aria-label="${esc(tr("ltw.name", "Name"))}">
        <button type="button" class="st-btn" data-ltw-register${j.registered || !j.versions.length ? " disabled" : ""}>${esc(tr("ltw.register", "Register the picked version"))}</button>
        ${["interrupted", "cancelled", "failed"].includes(j.status) && j.settings ? `<button type="button" class="st-btn st-btn--secondary" data-ltw-resume>${esc(tr("ltw.resume", "Resume"))}</button>` : ""}
        <button type="button" class="st-btn st-btn--secondary" data-ltw-delunsel${j.registered ? " disabled" : ""}>${esc(tr("ltw.deleteUnselected", "Delete the unselected versions"))}</button>
        <button type="button" class="st-btn st-btn--secondary" data-ltw-deljob>${esc(tr("ltw.deleteJob", "Delete this job"))}</button>
      </div>`;
  }
  function render() {
    const j = st.job;
    const view = !j || j.status === "draft" || j.status === "captioning" ? "setup"
      : LTW_ACTIVE.includes(j.status) ? "run" : "versions";
    const keep = [];
    if (view === "setup" && st.view === "setup" && st.renderedJob === (j && j.id)) {
      el.querySelectorAll("[data-ltw-set], [data-ltw-cap], [data-ltw-note]").forEach((input) => {
        const key = input.dataset.ltwSet != null ? "ltwSet" : input.dataset.ltwCap != null ? "ltwCap" : "ltwNote";
        if (key !== "ltwSet" && input.value === input.defaultValue) return;
        keep.push({key, name: input.dataset[key], value: input.value, focus: input === document.activeElement,
                   start: input.selectionStart, end: input.selectionEnd});
      });
    }
    st.view = view;
    st.renderedJob = j && j.id;
    el.innerHTML = view === "setup" ? setupHtml() : view === "run" ? runHtml() : versionsHtml();
    for (const saved of keep) {
      const input = [...el.querySelectorAll("[data-ltw-set], [data-ltw-cap], [data-ltw-note]")]
        .find((n) => n.dataset[saved.key] === saved.name);
      if (!input) continue;
      input.value = saved.value;
      if (saved.focus) {
        input.focus();
        if (saved.start != null) input.setSelectionRange(saved.start, saved.end);
      }
    }
    const q = (s) => el.querySelector(s);
    const on = (s, ev, fn) => { const n = q(s); if (n) n[ev] = fn; };
    if (view === "setup") {
      const drop = q("[data-ltw-drop]"), file = q("[data-ltw-file]");
      drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("ltw-drop--over"); };
      drop.ondragleave = () => drop.classList.remove("ltw-drop--over");
      drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("ltw-drop--over"); addFiles(e.dataTransfer.files); };
      on("[data-ltw-choose]", "onclick", () => file.click());
      file.onchange = () => { addFiles(file.files); file.value = ""; };
      on("[data-ltw-fromchat]", "onclick", chatPics);
      on("[data-ltw-trigger]", "oninput", (e) => { st.trigger = e.target.value; });
      on("[data-ltw-style-words]", "oninput", (e) => { st.styleWords = e.target.value; });
      on("[data-ltw-autocap]", "onclick", autoCaption);
      on("[data-ltw-start]", "onclick", start);
      el.querySelectorAll("[data-ltw-delimg]").forEach((b) => {
        b.onclick = async () => {
          try { st.job = await ltwFetch(`${base()}/images/${b.dataset.ltwDelimg}`, {method: "DELETE", headers: headers()}); render(); }
          catch (e) { err(e); }
        };
      });
      el.querySelectorAll("[data-ltw-cap], [data-ltw-note]").forEach((t) => { t.onchange = () => saveCaptions().catch(err); });
    } else if (view === "run") {
      on("[data-ltw-cancel]", "onclick", () => act("cancel"));
    } else {
      el.querySelectorAll("[data-ltw-ver]").forEach((r) => { r.onchange = () => { st.picked = +r.dataset.ltwVer; }; });
      on("[data-ltw-register]", "onclick", register);
      on("[data-ltw-resume]", "onclick", () => act("resume"));
      on("[data-ltw-continue]", "onclick", () => {
        st.picked = null;
        act("resume", {steps: Number(q("[data-ltw-continue-total]").value)});
      });
      on("[data-ltw-delunsel]", "onclick", deleteUnselected);
    }
    on("[data-ltw-deljob]", "onclick", async () => {
      try { stop(); await ltwFetch(base(), {method: "DELETE", headers: headers()}); st.job = null; st.picked = null; st.styleWords = null; st.trigger = ltwTrigger(); render(); }
      catch (e) { err(e); }
    });
  }
  loadModels().then(render);
  return {stop};
}
// The Images view's button: the same panel in a dialog
(function () {
  const btn = $("img-lora-train");
  if (!btn) return;
  btn.onclick = () => {
    const d = $("ltw-dialog");
    const box = d.querySelector("[data-ltw-dialog]");
    if (!ltwMounted.has(box)) { ltwMounted.add(box); ltwPanel(box); }
    d.showModal();
  };
  $("ltw-dialog-close").onclick = () => $("ltw-dialog").close();
})();
