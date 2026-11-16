// serve/web/history.js - what the chat history does to a list of messages. Pure functions: no page, no storage, no clock of
// their own, so they can be tested alone (serve/test_web_history.js). app.js draws the controls and keeps the state.
//
// One design for five proposals (see docs/DETAILS.md, "The web page"):
//   #2615 (uppet)         edit your question, correct an answer, regenerate, go back, undo
//   #1787 (Scotty-Hudson)  saved chats in a list, kept in this browser (IndexedDB), renamed, deleted, searched
//   #3823 (xvcezrqpbxtpzblrt) "Rewind to here" asks: edit and send again HERE, or BRANCH into a new chat
//   #3550 / #4004           the header's Load / Unload; sending while an answer is being written (app.js)
// A message list is an array of {role, text, time, images, files, reasoning, tools, meta, ...}; an attachment that was read
// in this page has its contents (image url, file text), one that came back from the browser's store has its name only.
"use strict";
(function (root) {
  const TITLE_MAX = 60;

  const newChatId = (now, rnd) => `c${now.toString(36)}${Math.floor(rnd * 1e6).toString(36)}`;
  function newChat(now = Date.now(), rnd = Math.random()) {
    return {id: newChatId(now, rnd), title: "", time: now, messages: []};
  }
  // the name in the list: what the user called it, else the first line of their first question, else the caller's word for "new"
  function chatTitle(c, untitled = "New chat") {
    if (c.title) return c.title;
    const first = (c.messages.find((m) => m.role === "user") || {}).text || "";
    return first.split("\n")[0].trim().slice(0, TITLE_MAX) || untitled;
  }

  // pictures and files are kept as their names only: a data URL would fill the browser's storage
  // a picture keeps its small preview (thumb) when it has one; records saved before that still clean to names alone
  // (#5998 review) only a preview this page makes is kept: a JPEG data URL of at most THUMB_MAX_CHARS, so a stored chat stays small
  const THUMB_SIDE = 480, THUMB_QUALITY = 0.72, THUMB_MAX_CHARS = 200000, THUMB_WAIT_MS = 5000;
  const validThumb = (t) => typeof t === "string" && t.length <= THUMB_MAX_CHARS && /^data:image\/jpeg;base64,[A-Za-z0-9+/]+=*$/.test(t);
  const cleanMessages = (list) => list.map((m) => ({...m, images: (m.images || []).map((i) => (validThumb(i.thumb) ? {name: i.name, thumb: i.thumb} : {name: i.name})),
                                                    files: (m.files || []).map((f) => ({name: f.name}))}));
  // The preview of a picture (long side THUMB_SIDE px, JPEG): null when the browser cannot decode or draw it, when the canvas has no
  // 2D context, when export throws or is blocked, when the result is too big, or after THUMB_WAIT_MS. It never rejects, and nothing
  // waits for it: the attachment with its original goes in at once and gets the preview when it comes. `deps` are the page's
  // Image and canvas (makeImage(), makeCanvas()), so the rule runs in Node too.
  function makeThumb(dataUrl, deps, waitMs = THUMB_WAIT_MS) {
    return new Promise((resolve) => {
      let settled = false;
      const done = (v) => { if (!settled) { settled = true; clearTimeout(timer); resolve(v); } };
      const timer = setTimeout(() => done(null), waitMs);
      try {
        const im = deps.makeImage();
        im.onerror = () => done(null);
        im.onload = () => {
          try {
            const w = im.naturalWidth || im.width, h = im.naturalHeight || im.height;
            if (!(w > 0 && h > 0)) return done(null);
            const s = Math.min(1, THUMB_SIDE / Math.max(w, h));
            const c = deps.makeCanvas();
            c.width = Math.max(1, Math.round(w * s)); c.height = Math.max(1, Math.round(h * s));
            const ctx = c.getContext("2d");
            if (!ctx) return done(null);
            ctx.drawImage(im, 0, 0, c.width, c.height);
            const out = c.toDataURL("image/jpeg", THUMB_QUALITY);
            done(validThumb(out) ? out : null);
          } catch (e) { done(null); }
        };
        im.src = dataUrl;
      } catch (e) { done(null); }
    });
  }
  const cleanChat = (c) => ({...c, messages: cleanMessages(c.messages)});

  // "today" | "yesterday" | "week" | "month" | "older", by calendar days back from now
  function groupKey(t, now = Date.now()) {
    const days = Math.floor((now - t) / 86400000);
    if (!days) return "today";
    if (days === 1) return "yesterday";
    if (days <= 7) return "week";
    if (days <= 30) return "month";
    return "older";
  }
  // the list for a search box: titles and message text, newest first
  function filterChats(chats, query, untitled) {
    const q = String(query || "").trim().toLowerCase();
    return chats.filter((c) => !q || chatTitle(c, untitled).toLowerCase().includes(q) ||
                               c.messages.some((m) => (m.text || "").toLowerCase().includes(q)))
                .sort((a, b) => b.time - a.time);
  }

  // Can a user turn be sent again as it was? A picture or file that came back from the browser's store has only its name.
  const lostAttachments = (m) => ((m.images || []).filter((x) => !x.url).length + (m.files || []).filter((x) => typeof x.text !== "string").length);
  // what goes back into the message box when a user turn is edited: its text and the attachments that still have contents
  function draftOf(m) {
    return {text: m.text || "",
            images: (m.images || []).filter((x) => x.url).map((x) => ({kind: "image", name: x.name, url: x.url})),
            files: (m.files || []).filter((x) => typeof x.text === "string").map((x) => ({kind: "file", name: x.name, text: x.text})),
            lost: lostAttachments(m)};
  }

  // "Rewind to here" on message i. A user turn goes back into the box and leaves the list; an answer stays and the list ends
  // with it. {keep, draft, hidden}: the messages that stay, the draft for the box (null for an answer), how many are hidden.
  function rewind(messages, i) {
    const m = messages[i];
    if (!m) return null;
    if (m.role === "user") return {keep: messages.slice(0, i), draft: draftOf(m), hidden: messages.length - i};
    return {keep: messages.slice(0, i + 1), draft: null, hidden: messages.length - i - 1};
  }
  // Regenerate the answer at i (or the last user turn): the list up to and including the user turn before it
  function regenerate(messages, i) {
    let u = i;
    while (u >= 0 && messages[u] && messages[u].role !== "user") u--;
    if (u < 0 || !messages[u]) return null;
    return {keep: messages.slice(0, u + 1), hidden: messages.length - u - 1};
  }
  // Edit the answer at i: its text replaced, its thinking, tool calls, error and stats dropped, everything after it hidden
  function editAnswer(messages, i, text, meta = "Edited") {
    const m = messages[i];
    if (!m || m.role !== "assistant") return null;
    return {keep: [...messages.slice(0, i), {role: "assistant", text, time: m.time, meta}], hidden: messages.length - i - 1};
  }
  // A branch: a new chat that starts with a copy of the first `end` messages. The original is not touched. The title says
  // where it came from and stays: the first question does not rename it (the first-message case of #3823).
  function branch(chat, messages, end, title, now = Date.now(), rnd = Math.random()) {
    const copy = JSON.parse(JSON.stringify(messages.slice(0, end)));
    return {...newChat(now, rnd), title, parent: {id: chat.id, title: chatTitle(chat)}, messages: copy};
  }

  // Bring a saved chat back (the other half of the composer's "Save this chat as Markdown"): the export's
  // `## You` / `## <model>` headings become messages (app.js, the export button - one heading per turn).
  //
  // Two things in a saved file look like a turn and are not:
  //   - a heading inside a fenced block is code (C's `##` token paste, a quoted outline in a sample);
  //   - a heading the text itself carries, like the `## Summary` most answers start a section with, is a section of
  //     that turn. So the answer's heading is the name the file repeats (the model name, written on every answer and
  //     on nothing else); a `## ` line that is neither `You` nor that name stays in the text.
  // If the name ties with headings inside a question (a one-question file), the careful reading finds no answer at all;
  // then the file is read the simple way, every `## ` line a turn, which is no worse than before. Residual shape,
  // stated plainly: a question that itself starts a line with `## You` is still read as a new turn - the export
  // writes no marker that tells the two apart.
  //
  // Thinking and tool calls were exported as <details> blocks, which carry nothing the store keeps (it keeps
  // attachments' names only), so they are dropped - but only the export's own two kinds, named in their <summary>.
  // A message may hold a <details> of its own (an HTML sample in a fence, an answer explaining the tag) and that is
  // content, not a trace. No timestamps in the file: the whole chat takes the time it came back. Returns the chat, or
  // null when the file holds no turns.
  const headingOf = (line) => { const h = line.match(/^## (.+)$/); return h ? h[1].trim() : null; };
  // the two blocks the export writes for itself (app.js, "Save this chat as Markdown")
  const EXPORTED_BLOCK = /^<details>\s*<summary>(?:Thinking|Tool\b[^<]*)<\/summary>/;
  // drop those blocks and keep every other line; `open` counts the block's own <details> still open, so a block that
  // holds a nested one is dropped whole and a line that merely mentions the tag does not close it early
  function dropExportedBlocks(text) {
    const kept = [];
    let open = 0;
    for (const line of String(text || "").split("\n")) {
      const t = line.trim();
      if (open) {
        if (t.startsWith("<details")) open += 1;
        if (t.startsWith("</details>") && --open <= 0) open = 0;
        continue;
      }
      if (EXPORTED_BLOCK.test(t)) { open = t.includes("</details>") ? 0 : 1; continue; }
      kept.push(line);
    }
    return kept.join("\n");
  }
  const fenceOf = (line) => { const f = line.match(/^ {0,3}(`{3,}|~{3,})/); return f ? f[1] : null; };
  const closesFence = (line, open) => {
    const m = line.match(/^ {0,3}(`{3,}|~{3,})[ \t]*$/);
    return !!m && m[1][0] === open[0] && m[1].length >= open.length;
  };
  function parseChatMarkdown(text, now = Date.now(), rnd = Math.random()) {
    const lines = String(text || "").split("\n");
    // the name on the answers: the heading the file repeats most (a tie keeps the one that comes first)
    const counts = new Map();
    let open = null;
    for (const line of lines) {
      const f = fenceOf(line);
      if (open) { if (f && closesFence(line, open)) open = null; continue; }
      if (f) { open = f; continue; }
      const name = headingOf(line);
      if (name && name !== "You") counts.set(name, (counts.get(name) || 0) + 1);
    }
    let model = null, most = 0;
    for (const [name, n] of counts) if (n > most) { model = name; most = n; }

    const read = (isTurnHeading) => {
      const messages = [];
      let role = null, body = [], fence = null;
      const flush = () => {
        const t = dropExportedBlocks(body.join("\n")).trim();
        if (role && t) messages.push({role, text: t, time: now});
        body = [];
      };
      for (const line of lines) {
        const f = fenceOf(line);
        if (fence) { body.push(line); if (f && closesFence(line, fence)) fence = null; continue; }
        if (f) { fence = f; body.push(line); continue; }
        const name = headingOf(line);
        if (name && isTurnHeading(name)) { flush(); role = name === "You" ? "user" : "assistant"; continue; }
        body.push(line);
      }
      flush();
      return messages;
    };

    const named = read((name) => name === "You" || (model !== null && name === model));
    const whole = named.some((m) => m.role === "assistant") && named.some((m) => m.role === "user");
    const messages = whole ? named : read(() => true);
    if (!messages.length) return null;
    return {...newChat(now, rnd), messages};
  }

  const api = {TITLE_MAX, newChat, chatTitle, cleanMessages, cleanChat, groupKey, filterChats, lostAttachments, draftOf, rewind,
               regenerate, editAnswer, branch, THUMB_MAX_CHARS, validThumb, makeThumb, parseChatMarkdown};
  root.StrataHistory = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
