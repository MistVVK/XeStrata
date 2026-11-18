// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// serve/web/history.js - what the chat history does to a list of messages. Pure functions: no page, no storage, no clock of
// their own, so they can be tested alone. app.js draws the controls and keeps the state.
//
// One design for five proposals (see docs/DETAILS.md, "The web page"):
//   #2615 (uppet)         edit your question, correct an answer, regenerate, go back, undo
//   #1787 (Scotty-Hudson)  saved chats in a list, kept in this browser (IndexedDB), renamed, deleted, searched
//   #3823 (xvcezrqpbxtpzblrt) "Rewind to here" asks: edit and send again HERE, or BRANCH into a new chat
//   #3550 / #4004           the header's Load / Unload; sending while an answer is being written (app.js)
// A message list is an array of {role, text, time, images, files, reasoning, tools, meta, ...}; an attachment that was read
// in this page has its contents (image url, file text), one that came back from the browser's store has its name only.
/* global module, globalThis */   // module: Node, when a script loads this file
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

  // localStorage (the list off) holds about 5 MB for the whole page: pictures and files are kept as their names only.
  // A tool image's copy is a data URL as big as the picture: drop it too, or a few image tools fill the store and
  // every later save fails silently (store.set returns false).
  const cleanMessages = (list) => list.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                                    files: (m.files || []).map((f) => ({name: f.name})),
                                                    ...(m.tools && m.tools.some((t) => t.copies)
                                                      ? {tools: m.tools.map((t) => { const c = {...t}; delete c.copies; return c; })} : {})}));
  // IndexedDB (the saved-chats list) keeps a chat whole: the pictures and files sent and the copies of the images the
  // tools made (keepToolCopies), so a chat read again shows them and sends them to the model as before
  const cleanChat = (c) => ({...c, messages: c.messages.slice()});

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

  const api = {TITLE_MAX, newChat, chatTitle, cleanMessages, cleanChat, groupKey, filterChats, lostAttachments, draftOf, rewind,
               regenerate, editAnswer, branch};
  root.StrataHistory = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
