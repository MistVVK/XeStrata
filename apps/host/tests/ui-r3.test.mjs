/** Owner's GUI feedback round 3 (2026-11-01), items 1 to 14 that are UI: each pinned where the source can show it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const read = path => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const models = read('src/components/NativeModels.tsx');
/** Load a src/lib module (with its imports) the way the other tests do. */
async function lib(file) {
  const built = await build({ entryPoints: [fileURLToPath(new URL(`../${file}`, import.meta.url))], bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' }, logLevel: 'silent' });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
}

test('1: the Models page carries no DFlash2 paragraph, no per-card line and no picture-reader sentence; the Drafting dropdown stays', () => {
  assert.ok(!/MTP is the default\. DFlash2/.test(models), 'the DFlash2 paragraph is gone');
  assert.ok(!/dflash2Support/.test(models), 'the per-card line is gone');
  assert.ok(!/engine installed here does not include the picture reader/.test(models), 'the picture-reader sentence is gone');
  assert.match(models, /<Select ariaLabel=\{tr\("Drafting"\)\}/, 'the dropdown stays');
});

test('2: pictures are on by default; a missing reader says nothing wordy and saves nothing', () => {
  const host = read('server/native-control.mjs');
  // The row is shown for every model that can read pictures; without the reader it is off, disabled and carries a tooltip, no paragraph.
  assert.match(models, /const noReaderNow = !!selectedSize\?\.imagesCapable && !selectedSize\.images/);
  assert.match(models, /const noReader = noReaderSeen\.current\.value;/, 'a job that runs never changes what the row shows');
  assert.match(models, /selectedSize\?\.images \|\| noReader \? <><div className="native-image-reading"><label title=\{noReader \? tr\("Needs an engine update"\) : undefined\}>/);
  assert.match(models, /checked=\{!noReader && vision !== 'no'\} disabled=\{noReader \|\| !!active\}/);
  assert.match(models, /else \{\s*setVision\('yes'\);|setVision\('yes'\);/, 'a model that is not downloaded starts with pictures on');
  // A load without a reader used to save "off" (ui_vision_enabled false), and the model stayed off after the engine got the reader.
  assert.match(models, /\.\.\.\(noReader \? \{ visionUnavailable: true \} : \{\}\)/);
  assert.match(host, /const noReader=args\.visionUnavailable===true;/);
  assert.match(host, /const visionPreferences=args\.vision!==undefined&&!noReader\?/, 'nothing is saved when the reader is missing');
  assert.match(host, /runtimeVision\(config,noReader\?'no':args\.vision\)/);
  // An old "off" (no marker) that sits on a model without the picture files is not a choice of the person: it starts on.
  assert.match(readFileSync(new URL('../src/lib/pictures-choice.ts', import.meta.url), 'utf8'), /installed\.visionEnabled === false && \(installed\.visionPrepared !== false \|\| installed\.visionChosen === true\)\) return 'no'/);   // since 0.2.14 an off with the marker (ui_vision_chosen) holds too: tests/pictures-hold.test.mjs
});

test('6: the conversation context opens at 128K for every configuration, or the model largest size when that is smaller; a warning, never a refusal', async () => {
  const { defaultContext, openingContext, contextIsTight, DEFAULT_CONTEXT } = await lib('src/lib/default-context.ts');
  assert.equal(DEFAULT_CONTEXT, 131072);
  assert.equal(defaultContext([8192, 32768, 65536, 131072, 262144]), 131072);
  assert.equal(defaultContext([8192, 32768, 65536]), 65536, 'a model whose largest size is under 128K opens at its largest');
  assert.equal(defaultContext([8192, 32768, 65536, 131072, 262144, 524288, 1048576]), 131072);
  assert.equal(defaultContext(undefined), 131072);
  assert.equal(openingContext(32768, [8192, 32768, 131072]), 131072, 'a saved 32K is the old default, not a choice');
  assert.equal(openingContext(65536, [8192, 32768, 65536, 131072]), 65536, 'a size that was chosen is kept');
  assert.equal(openingContext(undefined, [8192, 131072]), 131072);
  // What it costs: the conversation memory of 128K is 1.9 GB (int8) against 0.5 GB at 32K, so it is tight on cards of about 8 GB and below.
  assert.equal(contextIsTight(6, 131072), true);
  assert.equal(contextIsTight(8, 131072), true);
  assert.equal(contextIsTight(12, 131072), false, 'the 12 GB card of the owner is fine');
  assert.equal(contextIsTight(24, 131072), false);
  assert.equal(contextIsTight(8, 32768), false, '32K is the baseline: never a warning');
  assert.equal(contextIsTight(0, 131072), false, 'no card figure: no claim');
  assert.equal(contextIsTight(null, 131072), false);
  // The page opens with the default, ignores the engine recommended context, warns inline and never lowers by itself.
  assert.match(models, /useKept\('models\.context', DEFAULT_CONTEXT\)/);
  assert.ok(!/rec\.context/.test(models), 'the engine recommendation is not what the page opens with');
  assert.match(models, /setContext\(defaultContext\(first\?\.multi_file \? deepseekContexts\(\) : native\.catalog\.contexts\)\)/, 'DeepSeek too, from its own list');
  assert.match(models, /contextIsTight\(fitHardware\.vram_gb, context, kv, selectedSize\?\.context_fit\) \? <p className="small muted native-spill-note"><Icon name="warning"/);
  assert.ok(!/setContext\([^)]*contextIsTight/.test(models), 'the warning never changes the context');
  assert.match(models, /cache=\{\{ context, kv, drafter: fitDrafter \}\}/, 'the version list follows the chosen context, as the card does');
  const guided = read('src/components/GuidedSetup.tsx');
  assert.match(guided, /const context = pickedContext \?\? defaultContext\(contextList\)/, 'first-run opens at 128K on every card too');
  assert.ok(!/OLD_DEFAULT_CONTEXT|contextIsTight\(topCardGb, DEFAULT_CONTEXT/.test(guided), 'no fallback to 32K on a tight card');
  assert.match(guided, /contextIsTight\(topCardGb, context, 'int8', version\?\.context_fit\) \? <p className="small muted native-spill-note">/, 'the same inline warning, nothing lowered');
});

test('7 and 8: Model files has no explanatory sentence and holds "Other local models..." beside Storage settings', () => {
  const card = models.slice(models.indexOf('<h2 className="h4">{tr("Model files")}</h2>'), models.indexOf('</aside>'));
  assert.ok(!/Strata prepares all shards/.test(models), 'the sentence is gone');
  assert.ok(!/<p className="small muted">/.test(card), 'the card holds no paragraph');
  assert.match(card, /<div className="model-files__actions"><Btn[\s\S]*?>\{tr\("Storage settings"\)\}<\/Btn><Btn[^>]*onClick=\{onOtherModels\}[^>]*>\{tr\("Other local models\\u2026"\)\}<\/Btn><\/div><\/section>/, 'inside the card, after Storage settings');
  assert.ok(!/<\/section><Btn[^>]*other-models/.test(models), 'not a separate control under the card');
});

test('12: with no subagents the dropdown is one row, "No subagents recorded" and the X, with no icon and no explanation', () => {
  const bar = read('src/components/ChatToolbar.tsx');
  assert.match(bar, /title=\{tr\(session\?\.agents\?\.length \? "Subagents" : "No subagents recorded"\)\}/, 'the empty state is the title');
  assert.ok(!/subagents-empty|Subagents that run in this conversation/.test(bar), 'no icon, no explanatory text');
  assert.ok(!/subagents-empty/.test(read('src/styles/annotation-fixes.css')), 'no dead rule');
  assert.match(read('src/styles/app.css'), /\.telepop__head:last-child \{ align-items: center; padding-bottom: 0; margin-bottom: 0; border-bottom: 0; \}/, 'no empty body under the title');
});

test('14: the Board and Schedule help lines sit in the Tasks header after the title, smaller, and wrap rather than being cut', () => {
  const tasks = read('src/screens/Tasks.tsx'), board = read('src/components/TaskBoard.tsx'), css = read('src/styles/annotation-fixes.css');
  assert.match(tasks, /titleDetail=\{currentTask \? undefined : view === 'board' \? <span className="tasks-note"><Icon name="info" size=\{14\}\/>\{tr\('Cards move on their own as things happen\./);
  assert.match(tasks, /<span className="tasks-note"><Icon name="clock" size=\{14\}\/>[\s\S]*Runs while Strata\u2019s host is open\. Runs saved instructions using the loaded model\./);
  assert.ok(!/board-note|task-runtime/.test(tasks + board + read('src/styles/board.css') + read('src/styles/app.css')), 'not repeated in the body');
  assert.match(css, /\.main__title-detail:has\(\.tasks-note\) \{ white-space: normal; overflow: visible;/, 'wraps instead of being cut at narrow widths');
  assert.match(css, /\.main__head:has\(\.tasks-note\) \{ height: auto; \}/, 'the header grows to hold it');
});

test('13: Load model has no pop-up: the size and the warnings are in the page, a disk that is too small stops it there', () => {
  assert.ok(!/<Dialog open=\{!!plan\}/.test(models), 'the download and prepare dialog is gone');
  assert.ok(!/Download and prepare|Apply settings and load|Apply & load/.test(models), 'none of its words are left');
  // The press is the yes: the plan is read (Control checks the free disk), a short disk is said inline and nothing starts, otherwise the install starts at once.
  const load = models.slice(models.indexOf('async function load()'), models.indexOf('/* Unload / Use in chat'));
  assert.match(load, /api\.post<Record<string, any>>\('\/api\/native\/plan', downloadValues\)/);
  assert.match(load, /if \(planned\?\.plan\?\.disk_short\) \{\s*setPlan\(planned\);\s*return;\s*\}/, 'safety: not enough disk, nothing starts');
  assert.match(load, /kind: 'install', args: downloadValues/, 'otherwise it starts at once');
  assert.match(models, /\{tr\(plan\?\.plan\?\.disk_short \? <ErrorNotice message=\{plan\.plan\.disk_short\}\/> : null\)\}/, 'the disk message is in the page');
  assert.match(models, /active && drafterGb \? <p className="small muted native-spill-note"><Icon name="info" size=\{16\}\/>\{tr\(nowDrafter === 'mtp' \? \(splitCards && drafting === 'auto' \? "Auto uses the all-cards split engine with the MTP draft layer: about \{0\} GB to download, once\." : "Choosing MTP downloads its drafter: about \{0\} GB, once\."\) : "Choosing DFlash2 downloads its drafter/, 'the extra download is said in the page while it runs');
  assert.match(models, /className="native-download-size small muted">\{tr\("\{0\} download", \[sizeGb\(selectedSize\.download_gb\)\]\)\}/, 'the download size sits beside the button');
  // The warnings for a version that does not fit stay inline (item 5): may run slower under the version list.
  assert.match(models, /fitState === 'ssd' \? <p className="small muted native-spill-note"><Icon name="disk" size=\{16\}\/>/);
  assert.ok(!/fitState === 'no'|Doesn't fit/.test(models), "no Doesn't fit under the version list any more");
});

test('9: a chat that is working shows it on its sidebar row, and the model chip says which chat when it is elsewhere', () => {
  const bar = read('src/app/Sidebar.tsx'), css = read('src/styles/liquid.css');
  assert.match(bar, /const working = new Set\(liveCount \? liveChatIds\(\) : \[\]\)/, 'the same list of running turns the board reads');
  assert.match(bar, /data-working=\{working\.has\(s\.id\) \|\| undefined\}/);
  assert.match(bar, /working\.has\(s\.id\) \? <span className="navsession__working" role="img" aria-label=\{tr\("Working"\)\}/, 'a labelled cue, not only colour');
  assert.match(bar, /!working\.has\(s\.id\) && isUnread\(s, activeSession\) \? <span className="navsession__unread"/, 'a working chat is not also shown as unread');
  // GUI-FB2 item 6: the ring comes BEFORE the chat's name
  assert.ok(bar.indexOf('navsession__working') < bar.indexOf('<span className="navsession__title">'), 'the ring is left of the name');
  assert.match(bar, /\{workingLine \?\? modelQuant\}/, 'the model chip names the working chat');
  assert.match(bar, /!\(location\.route === 'chat' && id === activeSession\)/, 'no line for the chat that is in front of the person');
  // a stroked SVG circle (round caps), not a bordered box: it stays round at fractional sizes and turning angles
  assert.match(bar, /<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle className="navsession__track" cx="8" cy="8" r="6"\/>/);
  assert.match(css, /\.ws__session \.navsession__working svg \{[^}]*animation: st-spin 1\.1s linear infinite;/, 'one turn, existing keyframes');
  assert.match(css, /\.navsession__arc \{[^}]*stroke-linecap: round;/);
  assert.doesNotMatch(css, /\.ws__session \.navsession__working \{[^}]*border-radius/, 'no bordered box');
  assert.match(css, /prefers-reduced-motion: reduce\) \{\s*\.ws__session \.navsession__working svg \{\s*animation: none;/, 'reduced motion holds it still');
  assert.match(css, /html\[data-motion='reduced'\] \.ws__session \.navsession__working svg \{\s*animation: none;/);
});

test('10 (visual): the reply stays still while it streams, one live cue per step, a slot for the host to feed live output, a slower preview panel', async () => {
  const chat = read('src/screens/Chat.tsx'), css = read('src/styles/annotation-fixes.css'), app = read('src/styles/app.css'), liquid = read('src/styles/liquid.css');
  // Stop and Send are one height, so the message box no longer moves 5px when a reply starts and ends.
  assert.match(css, /\.st-composer__bar > \.st-btn:not\(\.st-btn--icon\) \{ height: 32px; min-height: 32px; box-sizing: border-box; \}/);
  // In a reply with activity lines the line is the one live cue for a tool; the running card above the text is gone.
  assert.match(chat, /filter\(t => !grouped \|\| t\.state === 'proposed' \|\| t\.state === 'denied' \|\| t\.state === 'error'\)/);
  // The caret follows text, never an activity line.
  assert.match(chat, /m\.streaming && m\.content && !m\.drafts\?\.length && \(!grouped \|\| m\.blocks!\.at\(-1\)\?\.kind === 'text'\) \? <span className="caret"/);
  // A slot for what the host knows while a run is open (AGENT-LOST feeds it): read, shown, dropped at close.
  const { readGroup } = await lib('src/lib/deliverables.ts');
  const open = readGroup({ id: 'g', closed: false, line: 'Writing pagoda.html', live: { label: 'Writing pagoda.html', output: 'a\nb\n412 lines written\n' } });
  assert.deepEqual(open.live, { label: 'Writing pagoda.html', output: '412 lines written' }, 'the last line only, one fixed row');
  assert.equal(readGroup({ id: 'g', closed: true, line: 'Wrote a file', live: { label: 'x', output: 'y' } }).live, undefined, 'a closed group has no live part');
  assert.equal(readGroup({ id: 'g', closed: false, line: 'Ran a command' }).live, undefined, 'without data it draws as before');
  const deliver = read('src/components/Deliverables.tsx');
  assert.match(deliver, /live && group\.live\?\.output \? <p className="act__output"/);
  assert.match(app, /\.act\[data-live\] \.act__line \{ color: var\(--st-ink\); \}/);
  // The sidebar preview opens and closes in 460 ms (the motion scale's own step, existing easings); reduced motion makes it 1 ms.
  assert.match(liquid, /animation: st-panel-in var\(--st-t-choreo\) var\(--st-ease-enter\) both;/);
  assert.match(liquid, /animation: st-panel-out var\(--st-t-choreo\) var\(--st-ease-exit\) both;/);
  assert.match(app, /--st-t-choreo: 460ms;/);
  assert.match(app, /@media \(prefers-reduced-motion: reduce\) \{\s*:root \{[^}]*--st-t-choreo: 1ms;/);
  assert.match(app, /\[data-motion='reduced'\] \{[^}]*--st-t-choreo: 1ms;/);
  assert.match(read('src/components/ArtifactPanel.tsx'), /setTimeout\(\(\) => setLeaving\(false\), 520\)/, 'the leaving shell stays for the whole 460 ms');
});

test('the new words are in all nine languages with the same slots', () => {
  const en = JSON.parse(read('src/locales/r3-en.json'));
  assert.ok(Object.keys(en).length >= 5);
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const cat = JSON.parse(read(`src/locales/r3-${lang}.json`));
    assert.deepEqual(Object.keys(cat), Object.keys(en), lang);
    for (const [id, text] of Object.entries(en)) {
      assert.ok(cat[id] && cat[id] !== text, `${lang} ${id} is translated`);
      assert.deepEqual((cat[id].match(/\{\d+\}/g) ?? []).sort(), (text.match(/\{\d+\}/g) ?? []).sort(), `${lang} ${id} slots`);
    }
  }
});

test('a context picked after this update sticks, 32K included; only an old saved 32K without the marker moves once', async () => {
  const { openingContext } = await lib('src/lib/default-context.ts');
  const list = [8192, 32768, 65536, 131072];
  assert.equal(openingContext(32768, list), 131072, 'an old saved 32K has no marker: it moves to the new default');
  assert.equal(openingContext(32768, list, true), 32768, 'a 32K that was picked has the marker: it stays');
  assert.equal(openingContext(65536, list), 65536);
  const host = read('server/native-control.mjs');
  assert.match(host, /\.\.\.\(args\.contextChosen===true\?\{ui_context_chosen:true\}:\{\}\)/, 'a load saves the marker');
  assert.match(host, /rememberSetting\(record\(id\)\.path,'ui_context_chosen',true\)/, 'an install saves it too');
  assert.match(host, /contextChosen:cfg\.ui_context_chosen===true/, 'and it is read back');
  assert.match(models, /openingContext\(installed\.context, contextList, installed\.contextChosen === true, selectedSize\.multi_file \? \[OLD_DEEPSEEK_CONTEXT\] : undefined\)/);
  assert.match(models, /\.\.\.\(contextPicked \? \{ contextChosen: true \} : \{\}\)/, 'a pick is sent with the install');
  assert.match(models, /\.\.\.\(contextPicked && !installed\.contextChosen \? \{ contextChosen: true \} : \{\}\)/, 'and with a direct load, so a picked 32K that equals the saved one is marked too');
  assert.match(read('src/components/GuidedSetup.tsx'), /contextTouched \? \{ contextChosen: true \}/);
});

test('the tune dialog also runs without pictures and saves nothing when the engine has no reader', () => {
  assert.match(models, /vision=\{selectedSize\.images \? vision : 'no'\} visionUnavailable=\{noReader\}/);
  assert.match(read('src/components/NativeTune.tsx'), /\.\.\.\(visionUnavailable \? \{ visionUnavailable: true \} : \{\}\)/);
  const host = read('server/native-control.mjs');
  assert.match(host, /vision:current\.config\.vision\?args\.vision:'no',\.\.\.\(args\.visionUnavailable===true\?\{visionUnavailable:true\}:\{\}\)/);
  assert.match(host, /args\.contextChosen===true\|\|JSON\.stringify\(cfg\[USER_KEY\]/, 'a load that carries the context marker writes it to the saved settings');
});

test('owner 11-01 (second decision): 128K on every card, in first-run setup, and for DeepSeek where its pack allows it', async () => {
  const { defaultContext, openingContext, OLD_DEEPSEEK_CONTEXT } = await lib('src/lib/default-context.ts');
  const { deepseekContexts } = await lib('src/lib/deepseek-context.ts');
  assert.equal(defaultContext(deepseekContexts()), 131072, 'DeepSeek without a pack limit: 128K');
  assert.equal(defaultContext(deepseekContexts(65536)), 65536, 'a pack that stops at 64K opens at its largest');
  assert.equal(defaultContext(deepseekContexts(1048576)), 131072, 'a pack that reads 1M still opens at 128K');
  assert.equal(openingContext(8192, deepseekContexts(), false, [OLD_DEEPSEEK_CONTEXT]), 131072, 'an unmarked saved 8K is the old DeepSeek default');
  assert.equal(openingContext(8192, deepseekContexts(), true, [OLD_DEEPSEEK_CONTEXT]), 8192, 'an 8K that was picked stays');
  assert.equal(openingContext(8192, [8192, 32768, 131072]), 8192, 'for any other model an 8K is a choice (Terminal setups for small cards write it)');
  const guided = read('src/components/GuidedSetup.tsx');
  assert.ok(!/fixedContext/.test(guided) && !/\[8192\]/.test(guided), 'DeepSeek is no longer fixed at 8K in first-run setup');
  assert.match(guided, /options=\{contextList\.map\(n =>/);
});

test('10 (live file write): message.drafts is drawn through the open activity line: file, growing size, the tail only if it reads well, no jumps', async () => {
  const { draftSize, draftTail, groupFromDraft } = await lib('src/lib/draft-view.ts');
  assert.equal(draftSize(0), '0 KB'); assert.equal(draftSize(80), '80 B'); assert.equal(draftSize(7800), '7.8 KB'); assert.equal(draftSize(18720), '19 KB', 'the same thousands as the cut sentence');
  const code = ['<!doctype html>', '<html lang="en">', '<head><meta charset="utf-8"></head>', '<script type="module">', 'const scene = new THREE.Scene();'].join('\n');
  assert.deepEqual(draftTail(code), ['<head><meta charset="utf-8"></head>', '<script type="module">', 'const scene = new THREE.Scene();'], 'the last three lines');
  assert.deepEqual(draftTail(''), [], 'nothing yet');
  assert.deepEqual(draftTail('\u00e9\u0001\u0002 \u0003\u0004\u0005\u0006\u0007\u0008\u000b\u000c\u000e\u000f'), [], 'a wall of control characters is not shown');
  assert.deepEqual(draftTail('{"a":1,"b":2,"c":3,"d":4,"e":5}'), [], 'a line without words is not shown');
  const draft = { id: 't1', name: 'write_file', target: 'pagoda.html', bytes: 7800, lines: 165, preview: code, scope: 'write', verb: 'Writing' };
  const g = groupFromDraft(draft, 'Writing · pagoda.html');
  assert.equal(g.closed, false); assert.equal(g.live.bytes, 7800); assert.equal(g.live.lines, 165); assert.match(g.live.tail, /THREE\.Scene/);
  assert.equal(groupFromDraft({ ...draft, scope: 'edit' }, 'x').live.tail, undefined, 'a tool that is not a write or a command has no text box');
  assert.equal(groupFromDraft({ ...draft, preview: '' }, 'x').live.tail, '', 'the box is kept from the start (empty) so it does not appear later and push the text down');
  const deliver = read('src/components/Deliverables.tsx'), chat = read('src/screens/Chat.tsx'), app = read('src/styles/app.css');
  assert.match(deliver, /useThrottled\(drafts, 350,/, 'about five frames a second are drawn about three times a second');
  assert.match(deliver, /<div className="act__tail" aria-hidden="true">/);
  assert.match(chat, /m\.streaming && m\.drafts\?\.length && !planCard \? <DraftLines drafts=\{m\.drafts\}/);
  assert.match(chat, /\|\| !!m\.drafts\?\.length\) \? <div className="st-bubble">/, 'the reply shows before any text, as soon as a file starts');
  assert.match(chat, /!m\.drafts\?\.length && \(!grouped/, 'no caret while a file is being written');
  assert.match(app, /\.act__tail \{[^}]*height: calc\(3 \* 1\.45 \* var\(--st-fs-xs\) \+ 16px\);[^}]*overflow: hidden;/, 'a box that is always three lines high');
  assert.ok(!/animation|transition/.test(app.slice(app.indexOf('.act__meta {'), app.indexOf('.act__output {'))), 'nothing in the write line animates, so reduced motion has nothing to hold still');
});
