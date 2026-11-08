// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// serve/web/mermaid-view.js - a ```mermaid block in an answer drawn as its chart, once the block is written (while it
// streams it stays code). mermaid 12.3.0 is vendored in third_party/main/mermaid (MIT, see PROVENANCE.json there),
// which the server hands out as web/vendor/mermaid/: no network, no CDN, nothing loaded until an answer holds a chart.
//
// What a chart may do: nothing but draw. securityLevel "strict" turns off its HTML labels and click handlers (the
// labels are text), the SVG mermaid makes is parsed into nodes here (no innerHTML of it into the page) and a chart that
// cannot be drawn stays as its code. The colors are the page's (tokens.css), light or dark, and a chart is drawn again
// when the theme changes.
//
// The image tools' plan (serve/skills/combine-tools): a box's id (s1, s2, ...) is the step a tool call names; the box is
// colored by how its call goes (waiting, running, done, failed) and marked with a sign as well, not by color alone.
/* global module, globalThis, tr */
"use strict";
(function (root) {
  const URL = "web/vendor/mermaid/dist/mermaid.min.js";
  const MARK = {waiting: "○", running: "▶", done: "✓", error: "✗"};
  let loading = null, theme = "", seq = 0;
  const drawn = new Map();   // the chart's code and theme -> its SVG's text (or null: it cannot be drawn)

  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  async function ready() {
    if (!loading) loading = root.StrataMath.script(URL).catch((e) => { loading = null; throw e; });
    await loading;
    const now = document.documentElement.dataset.theme || "light";
    if (now !== theme) {
      theme = now;
      root.mermaid.initialize({
        startOnLoad: false, securityLevel: "strict", theme: "base", htmlLabels: false,
        flowchart: {htmlLabels: false, useMaxWidth: true},
        fontFamily: cssVar("--st-font"),
        themeVariables: {
          darkMode: now === "dark", background: cssVar("--st-surface"), fontFamily: cssVar("--st-font"),
          primaryColor: cssVar("--st-surface-2"), primaryTextColor: cssVar("--st-ink"), primaryBorderColor: cssVar("--st-line"),
          secondaryColor: cssVar("--st-accent-tint"), tertiaryColor: cssVar("--st-surface"),
          lineColor: cssVar("--st-ink-muted"), textColor: cssVar("--st-ink"), mainBkg: cssVar("--st-surface-2"),
          nodeBorder: cssVar("--st-line"), clusterBkg: cssVar("--st-surface"), edgeLabelBackground: cssVar("--st-surface"),
        },
      });
    }
    return now;
  }
  async function svgOf(code) {
    const key = `${theme}\n${code}`;
    if (drawn.has(key)) return drawn.get(key);
    let svg = null;
    try {
      ({svg} = await root.mermaid.render(`xs-mermaid-${++seq}`, code));
    } catch (_) {
      svg = null;
      for (const junk of document.querySelectorAll(`[id^="dxs-mermaid-"]`)) junk.remove();   // its error box
    }
    drawn.set(key, svg);
    return svg;
  }
  // the SVG as nodes (DOMParser: no script runs; mermaid's strict output has none)
  function nodesOf(svg) {
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    const el = doc.documentElement;
    return el && el.nodeName.toLowerCase() === "svg" ? document.importNode(el, true) : null;
  }
  // a box of the chart by its id: mermaid names a flowchart's node "<chart>-flowchart-<id>-<n>"
  function stepOf(g) {
    const m = /-flowchart-(.+)-\d+$/.exec(g.id || "");
    return m ? m[1] : null;
  }
  function paint(svg, steps) {
    for (const g of svg.querySelectorAll("g.node")) {
      const id = stepOf(g);
      g.querySelector(".xs-step-mark")?.remove();
      if (!id || !steps || !steps.size) { delete g.dataset.step; continue; }
      const state = steps.get(id) || (/^s\d+$/.test(id) ? "waiting" : null);
      if (!state) { delete g.dataset.step; continue; }
      g.dataset.step = state;
      const shape = g.querySelector("rect, polygon, path, circle, ellipse");
      if (!shape || typeof shape.getBBox !== "function") continue;
      let box;
      try { box = shape.getBBox(); } catch (_) { continue; }
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "xs-step-mark");
      t.setAttribute("x", String(box.x + box.width - 4));
      t.setAttribute("y", String(box.y + 12));
      t.setAttribute("text-anchor", "end");
      t.textContent = MARK[state] || "";
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = state;
      t.appendChild(title);
      g.appendChild(t);
    }
  }
  // every finished ```mermaid block in `container` drawn (codeBlock marks them data-mermaid), its boxes colored by the
  // steps' states (`steps`: Map of a box's id -> waiting | running | done | error)
  async function hydrate(container, steps) {
    const boxes = [...container.querySelectorAll(".st-code[data-mermaid]")];
    if (!boxes.length) return;
    let now;
    try { now = await ready(); } catch (_) { return; }   // no script: the code stays
    for (const box of boxes) {
      if (!box.isConnected) continue;
      const code = box.querySelector("pre").textContent;
      const svg = await svgOf(code);
      if (!box.isConnected) continue;
      let view = box.querySelector(".st-mermaid");
      if (!svg) { view?.remove(); box.classList.remove("st-code--chart"); continue; }
      if (!view || view.dataset.theme !== now || view.dataset.code !== code) {
        const el = nodesOf(svg);
        if (!el) continue;
        el.removeAttribute("height");
        view?.remove();
        view = document.createElement("div");
        view.className = "st-mermaid";
        view.dataset.theme = now;
        view.dataset.code = code;
        view.setAttribute("role", "img");
        view.setAttribute("aria-label", tr("chart.label", "A chart of the plan"));
        view.appendChild(el);
        box.querySelector("pre").before(view);
        box.classList.add("st-code--chart");
      }
      paint(view.querySelector("svg"), steps);
    }
  }
  const api = {hydrate, stepOf};
  root.StrataMermaid = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
