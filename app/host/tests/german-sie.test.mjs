// German addresses the reader as "Sie" everywhere (owner, 11-16 22:0x). Every German catalog of the app and of the engine's web page is
// read: no du-pronoun (du, dich, dir, dein…) and no sentence that starts with a common du-imperative ("Öffne …", "Wähle …"). The allow-list
// holds the texts that are not addressed to the reader: what the person sends to the model (example prompts, the Continue message) and
// the app's own first-person status lines.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const APP = join(here, '..', 'src', 'locales');
const WEB = join(here, '..', '..', '..', 'terminal', 'serve', 'web', 'locales', 'de.json');

/** Not addressed to the reader. Prompts the person sends to the model (du to the assistant is natural): s0386 (the Continue message),
 * s0500, s0637, s0711, s0793, s0801, s0817, s0850, s0892-s0894, s1002, s1238, s1653, s1804, s2772 (the quoted example only). The app's
 * own status lines in the first person: s2934, s2938, s2944, s2945, s2947, s2948, s2987, s8342. "Gehe zu" (s0810) is the menu label
 * German software uses for "Go to". */
export const NOT_TO_THE_READER = new Set(['s0386', 's0500', 's0637', 's0711', 's0793', 's0801', 's0810', 's0817', 's0850', 's0892', 's0893',
  's0894', 's1002', 's1238', 's1653', 's1804', 's2772', 's2934', 's2938', 's2944', 's2945', 's2947', 's2948', 's2987', 's8342']);

const PRONOUN = /(?<![A-Za-zÄÖÜäöüß])(du|Du|dich|Dich|dir|Dir|dein|Dein|deine|Deine|deinen|Deinen|deinem|Deinem|deiner|Deiner|deines|Deines)(?![A-Za-zÄÖÜäöüß])/;
const VERBS = ['Öffne', 'Wähle', 'Lade', 'Klicke', 'Gib', 'Starte', 'Prüfe', 'Füge', 'Sende', 'Lösche', 'Speichere', 'Schließe', 'Nimm', 'Versuche',
  'Installiere', 'Aktualisiere', 'Trage', 'Schreibe', 'Gehe', 'Drücke', 'Tippe', 'Erstelle', 'Ändere', 'Kopiere', 'Verwende', 'Nutze', 'Wechsle',
  'Melde', 'Schalte', 'Setze', 'Warte', 'Behalte', 'Ziehe', 'Lies', 'Sieh', 'Beende', 'Entferne', 'Aktiviere', 'Deaktiviere', 'Mach', 'Mache',
  'Hole', 'Bestätige', 'Erlaube', 'Verbinde', 'Wiederhole', 'Folge', 'Achte', 'Denke', 'Hilf', 'Beschreibe', 'Erkläre', 'Bereite', 'Repariere',
  'Verbessere', 'Fasse', 'Übersetze', 'Finde', 'Zeige', 'Plane', 'Lerne', 'Blicke', 'Verwandle', 'Erhöhe'];
const IMPERATIVE = new RegExp(`(?:^|[.!?:;]\\s+|[„"«(]\\s*|\\n)(${VERBS.join('|')})(?=[\\s,])(?!\\s+Sie(?![A-Za-zäöü]))`);

export function duForm(text) {
  const t = String(text);
  const m = PRONOUN.exec(t) ?? IMPERATIVE.exec(t);
  return m ? m[1] ?? m[0] : null;
}

function germanCatalogs() {
  const out = [];
  for (const f of readdirSync(APP)) if (/^(?:.+-)?de\.json$/.test(f)) out.push([f, JSON.parse(readFileSync(join(APP, f), 'utf8'))]);
  out.push(['web/de.json', JSON.parse(readFileSync(WEB, 'utf8'))]);
  return out;
}

test('the du check', () => {
  assert.equal(duForm('Öffne die Einstellungen.'), 'Öffne');
  assert.equal(duForm('Öffnen Sie die Einstellungen.'), null);
  assert.equal(duForm('Wähle Sie aus.'), null, 'a verb before Sie is not du');
  assert.equal(duForm('Lade das Modell.'), 'Lade');
  assert.equal(duForm('Das ist dein Modell.'), 'dein');
  assert.equal(duForm('Dubai, Duden, dirigieren'), null);
  assert.equal(duForm('Der Ordner wird gelesen. Laden Sie ihn neu.'), null);
  assert.equal(duForm('Fertig. Klicke auf Weiter.'), 'Klicke');
});

test('every German string addresses the reader as Sie', () => {
  const found = [];
  for (const [file, cat] of germanCatalogs()) {
    for (const [k, v] of Object.entries(cat)) {
      if (NOT_TO_THE_READER.has(k)) continue;
      for (const text of typeof v === 'string' ? [v] : Object.values(v ?? {})) {
        const hit = typeof text === 'string' ? duForm(text) : null;
        if (hit) found.push(`${file} ${k} (${hit}): ${text.slice(0, 90)}`);
      }
    }
  }
  assert.deepEqual(found, []);
});
