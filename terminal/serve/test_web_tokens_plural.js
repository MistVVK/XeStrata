// The token count under an answer on the web page is a plural: Russian says 1 токен, 2 токена, 5 токенов, 21 токен.
//   node serve/test_web_tokens_plural.js      (serve/test_web_tokens_plural.py runs it too and skips when node is missing)
// It reads the shipped Russian catalog and the call in app.js, so a catalog or a call that loses the plural fails here.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const web = path.join(__dirname, "web");
globalThis.document = {documentElement: {lang: "en"}, querySelectorAll: () => []};
globalThis.fetch = async (url) => {
  const m = /web\/locales\/([\w-]+)\.json$/.exec(url);
  return {ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(web, "locales", `${m[1]}.json`), "utf8"))};
};
const i18n = require(path.join(web, "i18n.js"));

// the English text and the variables app.js passes for the token count
const src = fs.readFileSync(path.join(web, "app.js"), "utf8");
const call = /tr\("chat\.tokens",\s*(\{(?:[^{}"]|"[^"]*")*\}|"[^"]*"),\s*\{([^}]*)\}\)/.exec(src);   // a plural's texts hold braces ("{n} token")
assert.ok(call, "app.js shows the token count through tr(\"chat.tokens\", ...)");
const english = Function(`return (${call[1]});`)();
assert.equal(typeof english, "object", "the token count is a plural in the code ({one, other})");
assert.match(call[2], /\bcount:\s*n\b/, "the form is chosen by the number itself, not by its formatted text (\"1,001\")");

(async () => {
  assert.equal(await i18n.setLanguage("ru"), "ru");
  const shown = (n, text = String(n)) => i18n.tr("chat.tokens", english, {n: text, count: n});
  assert.equal(shown(1), "1 токен");
  assert.equal(shown(2), "2 токена");
  assert.equal(shown(5), "5 токенов");
  assert.equal(shown(21), "21 токен");
  assert.equal(shown(1001, "1 001"), "1 001 токен", "a formatted count still picks the form by the number");
  assert.equal(await i18n.setLanguage("en"), "en");
  assert.equal(shown(1), "1 token");
  assert.equal(shown(1234, "1,234"), "1,234 tokens");
  console.log("ok: web token count plural");
})().catch((e) => { console.error(e); process.exit(1); });
