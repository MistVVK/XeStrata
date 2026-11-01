// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// serve/web/segments.js - one chat that goes on for good, in segments, as Mainplane's sessions do (no compaction). Pure
// functions: no page, no storage, no clock of their own, so they can be tested alone. app.js draws and keeps the state.
//
// Within a segment the prompt only grows: the earlier turns go back as they were written, so the server's conversation
// cache holds their start and each turn reads only what it added. A message that would take the segment past its limit
// starts the next one instead of being summarized: the new segment's prompt is a system note (the limit, the tools), the
// chat's memory, the last exchanges word for word and the message. The earlier segments stay whole in the chat, and the
// model reads what it needs of them back with history_search and history_read; what must carry on, it writes to the
// memory (memory_write), which every segment starts with.
//
// State lives in the messages, so it survives the browser's store, a reload, a rewind and a branch alike:
//   a user message with `segment` {k, system, carry} starts segment k (carry: the exchanges copied into its prompt);
//   an assistant message's `keep` says whether its thinking goes back (decided once, when it ended: keepThinking);
//   a user message's `ctx` is the line on the room left, sent with it and kept so it is sent the same each time;
//   the memory is the last memory_write call's text, or a user's `memoryEdit` on a message, whichever is later.
/* global module, globalThis */   // module: Node, when a script loads this file
"use strict";
(function (root) {
  const CARRY = 2;              // exchanges copied word for word into a new segment
  const HIT_CHARS = 160;        // text around a search hit
  const HITS = 12;              // search hits returned
  const READ_CHARS = 8000;      // the longest read; a longer one keeps its head
  const ROOM_SHARE = 0.5;       // more than this share of the segment left: an answer's thinking always goes back

  // where the current segment starts (0 before any handoff)
  function segmentStart(messages) {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].segment) return i;
    return 0;
  }
  // the segment number of message i (1 for a chat that has never been handed off)
  function segmentAt(messages, i) {
    for (let j = Math.min(i, messages.length - 1); j >= 0; j--) if (messages[j].segment) return messages[j].segment.k;
    return 1;
  }

  // the memory as of the messages before `end`
  function memoryAt(messages, end = messages.length) {
    let text = "";
    for (let i = 0; i < end; i++) {
      const m = messages[i];
      for (const t of m.tools || []) {
        if (t.name === "memory_write" && t.state === "done" && t.arguments && typeof t.arguments.text === "string") text = t.arguments.text;
      }
      if (typeof m.memoryEdit === "string") text = m.memoryEdit;
    }
    return text;
  }

  // a message's words as the model read them: the question, or the answer with its tools' calls and results
  function plainText(m) {
    if (m.role === "user") return m.text || "";
    let s = m.text || "";
    for (const t of m.tools || []) {
      if (t.result != null) s += `\n[${t.name}] ${JSON.stringify(t.arguments || {})}\n${t.result}`;
    }
    return s;
  }

  // up to HITS places of `query` (a case-insensitive substring, words of any language) in the messages before `end`
  function search(messages, end, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return [];
    const hits = [];
    for (let i = 0; i < end && hits.length < HITS; i++) {
      const text = plainText(messages[i]), low = text.toLowerCase();
      for (let at = low.indexOf(q); at >= 0 && hits.length < HITS; at = low.indexOf(q, at + q.length)) {
        const from = Math.max(0, at - HIT_CHARS / 2);
        hits.push({message: i, segment: segmentAt(messages, i), role: messages[i].role, offset: at,
                   text: text.slice(from, at + q.length + HIT_CHARS / 2)});
      }
    }
    return hits;
  }

  // message i's words from `offset`, at most `length` characters (READ_CHARS at most: a longer read keeps its head)
  function read(messages, end, i, offset = 0, length = READ_CHARS) {
    if (!Number.isInteger(i) || i < 0 || i >= end) return {error: `message must be 0 to ${end - 1}`};
    const text = plainText(messages[i]), from = Math.max(0, Math.floor(offset) || 0);
    const n = Math.min(Math.max(1, Math.floor(length) || READ_CHARS), READ_CHARS);
    return {message: i, segment: segmentAt(messages, i), role: messages[i].role, offset: from, total: text.length,
            text: text.slice(from, from + n), cut: from + n < text.length};
  }

  // Whether an answer's thinking goes back with it, decided once when it ends: kept, the next turn reads only what it
  // adds; dropped, it reads again what came after the thinking (the answer, the tools' calls and results: `after`
  // tokens) and the segment keeps `thinking` tokens of room. So: kept while the segment has room to spare, or where the
  // read again would be longer than the thinking it saves.
  function keepThinking({thinking, after, used, limit}) {
    if (!(thinking > 0)) return false;
    if (limit > 0 && limit - used > ROOM_SHARE * limit) return true;
    return after >= thinking;
  }

  // the room line sent at the end of a message
  const roomLine = (used, limit) => `[context ${used} of ${limit} tokens used in this segment]`;

  // the system note of segment k
  function systemNote(k, limit) {
    return [
      `This conversation goes on across segments, with no summaries. This is segment ${k}. A segment holds at most ${limit} ` +
        "tokens; when the next message would not fit, a new segment starts with this note, your memory and the last exchanges.",
      k > 1 ? "The earlier segments are kept whole: find what you need in them with history_search and read it with " +
        "history_read before you answer from memory." : "",
      "Write what must carry on (facts about the user, decisions, open tasks) to your memory with memory_write; it replaces " +
        "the whole memory, and every segment starts with it.",
    ].filter(Boolean).join("\n");
  }

  // the tools the page runs (OpenAI function tools)
  const fn = (name, description, properties, required) =>
    ({type: "function", function: {name, description, parameters: {type: "object", properties, required}}});
  const TOOLS = [
    fn("history_search", "Search the earlier segments of this conversation for a word or phrase (case-insensitive). " +
       "Returns the places found: the message number, its segment and role, and the text around it.",
       {query: {type: "string", description: "the words to find"}}, ["query"]),
    fn("history_read", "Read a message of the earlier segments word for word, from a character offset.",
       {message: {type: "integer", description: "the message number from history_search"},
        offset: {type: "integer", description: "the first character, 0 by default"},
        length: {type: "integer", description: `characters to read, at most ${READ_CHARS}`}}, ["message"]),
    fn("memory_write", "Replace your memory: the notes every segment of this conversation starts with. Keep it short.",
       {text: {type: "string", description: "the whole memory"}}, ["text"]),
  ];
  const TOOL_NAMES = new Set(TOOLS.map((t) => t.function.name));

  // run one of TOOLS over the messages before the current segment (`end`); a result string for the model
  function runTool(messages, end, name, args) {
    const a = args && typeof args === "object" ? args : {};
    if (name === "history_search") {
      const hits = search(messages, end, a.query);
      return hits.length ? JSON.stringify(hits) : "no match in the earlier segments";
    }
    if (name === "history_read") return JSON.stringify(read(messages, end, a.message, a.offset, a.length));
    if (name === "memory_write") return typeof a.text === "string" ? "memory saved" : "error: text must be a string";
    return `error: no tool ${name}`;
  }

  // the exchanges copied into a new segment: the last CARRY questions and answers before `at`, as API messages without
  // thinking or tool rounds (the answer's words only)
  function carryFrom(messages, at, userText) {
    const out = [];
    let exchanges = 0;
    for (let i = at - 1; i >= 0 && exchanges < CARRY; i--) {
      const m = messages[i];
      if (m.role === "assistant") out.unshift({role: "assistant", content: m.text || ""});
      else if (m.role === "user") { out.unshift({role: "user", content: userText(m)}); exchanges++; }
    }
    while (out.length && out[0].role !== "user") out.shift();
    return out;
  }

  const api = {segmentStart, segmentAt, memoryAt, search, read, keepThinking, roomLine, systemNote, TOOLS, TOOL_NAMES,
               runTool, carryFrom, CARRY, READ_CHARS};
  root.StrataSegments = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
