// The Drafting choice in the app (MTP / DFlash2 / Off / Auto): the Models screen row, the setting the host saves, what the host asks Control before
// a load, the download plan line, and the words in all nine languages. The table of which card gets which drafter lives in Python
// (strata_control/drafting.py, tests/terminal/test_drafting.py); the app never reads it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { NATIVE_FIELDS, APP_OWNED_FIELDS, validateNativeSettings, withoutDraftLayer } from '../server/native-settings.mjs';
import { createControl } from '../server/control-runtime.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('Drafting is a setting of the model with four choices, saved by the app itself', () => {
  const field = NATIVE_FIELDS.find(f => f.id === 'drafting');
  assert.ok(field);
  assert.deepEqual(field.choices, ['auto', 'mtp', 'dflash2', 'off']);
  assert.equal(field.type, 'enum');
  assert.match(field.hint, /Auto \(recommended\)/);
  assert.match(field.hint, /Qwen3\.8-Flash-Next/);
  assert.match(field.hint, /one at a time/);
  assert.ok(APP_OWNED_FIELDS.includes('drafting'), 'the engine\'s own source never lists it: the app applies it');
  for (const ok of ['auto', 'mtp', 'dflash2', 'off']) assert.deepEqual(validateNativeSettings({ drafting: ok }), { drafting: ok });
  assert.deepEqual(validateNativeSettings({ drafting: null }), { drafting: null }, 'cleared = Auto');
  assert.throws(() => validateNativeSettings({ drafting: 'fastest' }), /Drafting: choose a supported option/);
});

test('Off leaves the draft layer out of every start, the same as "Draft layer (MTP)" off', () => {
  assert.deepEqual(withoutDraftLayer(['--pack', 'p', '--mtp', '/m', '--spec', '4', '--spec-min-p', '0.5', '--batch-mtp']), ['--pack', 'p', '--spec', '4']);
  const source = read('../server/native-control.mjs');
  assert.match(source, /\(cfg\.mtp===false\|\|cfg\.drafting==='off'\)&&Array\.isArray\(cfg\.args\)\)cfg\.args=withoutDraftLayer\(cfg\.args\)/);
});

test('the Models screen offers the choice next to context, DFlash2 only for Qwen3.8-Flash-Next, and carries it to the plan and the load', () => {
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /draftingOptions\(selectedSize\.sourceFamily \?\? selectedFamily\?\.family, native\?\.capabilities\?\.dflash2\)/);
  assert.match(screen, /draftingWords: Record<string, string> = \{ auto: 'Auto \(recommended\)', mtp: 'MTP', dflash2: 'DFlash2', off: 'Off' \}/);
  const row = screen.slice(screen.indexOf('<div className="native-choice__row">'), screen.indexOf('{roleCards.length > 1 ? <GpuRoles'));
  assert.ok(row.indexOf('Conversation context') < row.indexOf('tr("Drafting")') && row.indexOf('tr("Drafting")') < row.indexOf('Computer use'), 'in the same row, between context and computer use');
  assert.match(row, /<Select ariaLabel=\{tr\("Drafting"\)\}/, 'the existing Select, no new component');
  assert.match(screen, /setDrafting\(installed\.drafting \?\? 'auto'\)/, 'a model shows what it was saved with');
  assert.match(screen, /const downloadValues = \{ \.\.\.values, \.\.\.\(draftingChoices \? \{ drafting \} : \{\}\)/);
  assert.match(screen, /\.\.\.\(draftingChoices \? \{ values: \{ drafting \} \} : \{\}\)/);
  assert.match(screen, /Choosing MTP downloads its drafter: about \{0\} GB, once\./);
  assert.match(screen, /Choosing DFlash2 downloads its drafter: about \{0\} GB, once\./);
  assert.match(screen, /resolveDrafter\(drafting, selectedSize\?\.sourceFamily \?\? selectedFamily\?\.family, native\?\.capabilities\?\.dflash2, splitCards\)/, 'the fit uses the drafter that is chosen, and the split word counts only with two or more cards');
  assert.match(screen, /loadedSelected\?\.metrics\?\.engine\?\.drafting_note/, 'what the engine said is shown');
});

test('the host asks Control before a load, and Control decides (the app never reads the table)', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async (args, options) => {
    calls.push({ args, options });
    return args[1] === 'plan'
      ? { code: 0, json: { ok: true, needed: true, gb: 1.2, line: 'Drafting: DFlash2' }, text: '', stderr: '' }
      : { code: 0, json: { ok: true, event: 'result', pack: 'C:/d/dflash2/x.pack' }, text: '', stderr: '' };
  } });
  const plan = await control.draftingPlan({ mode: 'dflash2', family: 'gyro', dataDir: 'C:/d' });
  assert.deepEqual(calls[0].args, ['drafting', 'plan', '--mode', 'dflash2', '--family', 'gyro', '--data-dir', 'C:/d', '--output', 'json']);
  assert.equal(plan.needed, true);
  const events = [];
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'dflash2', onEvent: e => events.push(e) });
  assert.deepEqual(calls[1].args, ['drafting', 'fetch', '--config', 'C:/w/cfg.json', '--mode', 'dflash2', '--if-needed', '--output', 'jsonl']);
  assert.equal(calls[1].options.acquire, true, 'only this explicit step may reach the network');
  assert.equal(calls[0].options.acquire, undefined, 'planning never does');
  const failing = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async () => ({ code: 5, json: { ok: false, error: { code: 'drafting_failed', message: 'The DFlash2 download did not finish.' } }, text: '', stderr: '' }) });
  await assert.rejects(failing.draftingFetch({ configPath: 'x' }), /DFlash2 download did not finish/);
});

test('a load fetches the drafter first, never stops for it, and says what ran', () => {
  const source = read('../server/native-control.mjs');
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /rememberSetting\(configFile,'drafting',asked===null\|\|asked==='auto'\?null:asked\)/, 'the choice is kept for the next load');
  assert.match(prepare, /drafts\?\.needed!==true\)return/, 'only what Control says is needed is downloaded: DFlash2 for Auto and DFlash2, nothing for MTP and Off');
  assert.match(prepare, /control\.draftingFetch\(/);
  assert.match(prepare, /catch\(error\)\{if\(job\.state==='running'\)\{job\.drafterNote=error\.message;/, 'a failed download leaves the MTP and the job says so');
  assert.match(source, /await prepareDrafter\(job,args,cfg,configFile,drafts,\{noSplit\}\)/);
  assert.match(source, /drafts\?\.mtp_needed===true&&\(args\.values\?\.drafting!==undefined\|\|\(drafts\.split===true\|\|args\.firstLoad===true\)&&[\s\S]*?\{[\s\S]*?await installMtpLayer/, 'a switch to the MTP fetches its layer once, from the Models screen own Load (and a plain Auto load when the plan says split, or the first load after a setup)');
  assert.match(source, /async function noteDrafting\(job,instance,key\)/);
  assert.match(source, /instance\.status='ready';instance\.phase='idle';await noteDrafting\(job,instance,key\)/);
  assert.match(source, /typeof choice==='string'&&typeof control\.draftingPlan==='function'/, 'the plan lists the drafter Control says is needed');
  assert.match(source, /drafting:cfg\.drafting\?\?'auto'/, 'the installed model reports its saved choice');
});

test('every sentence of the Drafting choice is in the catalog in all nine languages, with the same slots', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
  const catalogs = Object.fromEntries(langs.map(l => [l, { ...JSON.parse(read(`../src/locales/${l}.json`)), ...JSON.parse(read(`../src/locales/df2-${l}.json`)) }]));
  const idOf = text => Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
  const sentences = ['Drafting', 'Auto (recommended)', 'MTP', 'DFlash2',
    "Drafting: MTP (DFlash2 didn't fit)", "Drafting: MTP (DFlash2 isn't available in this mode)", 'Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)',
    "Drafting: MTP (the DFlash2 drafter isn't downloaded yet)",
    'Choosing DFlash2 downloads its drafter: about {0} GB, once.', 'Choosing MTP downloads its drafter: about {0} GB, once.', 'Drafting: off (no drafter is downloaded yet)', "Drafting: DFlash2 (the MTP drafter isn't downloaded yet)",
    'The DFlash2 drafter could not be downloaded, so the model starts without a drafter. Loading the model…', 'Downloading the MTP drafter…', 'The MTP drafter could not be downloaded.',
    'Downloading the DFlash2 drafter…', 'Packing the DFlash2 drafter for the engine…', 'The DFlash2 drafter is ready.',
    'The DFlash2 drafter could not be downloaded, so the model starts with the MTP. Loading the model…', 'Loading the model…',
    // the DFlash2 option (df2-<lang>.json): the paragraph under the dropdown, the lines about the card, the two new notes
    "Drafting: MTP (DFlash2 isn't available on this graphics card)", "There is no internet connection, so the DFlash2 drafter can't be downloaded now.",
    'On graphics cards like yours, DFlash2 was faster than MTP on some kinds of text and slower on others in our tests.',
    'On graphics cards like yours, DFlash2 was slower than MTP in our tests.', 'DFlash2 has not been tested on your graphics card yet.', 'On your Mac, DFlash2 was faster than MTP on this model in our tests.', 'DFlash2 is off by default on this graphics card for now; choose it in Drafting to use it.', 'SSD: not needed'];
  const slots = t => (t.match(/\{\d+\}/g) ?? []).sort().join();
  for (const text of sentences) {
    const id = idOf(text);
    assert.ok(id, `"${text.slice(0, 60)}" is in the English catalog`);
    for (const l of langs) {
      const t = catalogs[l][id];
      assert.ok(typeof t === 'string' && t.trim(), `${id} in ${l}`);
      assert.equal(slots(t), slots(text), `${id} in ${l} has the same slots`);
      if (l !== 'en' && text.length > 25) assert.notEqual(t, text, `${id} is translated in ${l}`);
    }
  }
});

test('the Drafting dropdown exists only where there is something to choose: Qwen3.8-Flash-Next (qwen, gyro, unsloth) on a computer whose engine can run DFlash2', async () => {
  const { transform } = await import('esbuild');
  const code = (await transform(read('../src/lib/drafting.ts'), { loader: 'ts', format: 'esm' })).code;
  const { draftingChoices } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const all = ['auto', 'mtp', 'dflash2', 'off'];
  // the engine here cannot run it (Metal, Intel, no card), or Control has not answered yet: no dropdown for anyone
  for (const family of ['qwen', 'gyro', 'unsloth', 'coder', 'swift', 'deepseek41', undefined]) {
    assert.equal(draftingChoices(family, false), null, `${family}, not served`);
    assert.equal(draftingChoices(family, undefined), null, `${family}, Control has not answered`);
    assert.equal(draftingChoices(family, null), null);
  }
  // it can: only the models it is for. DeepSeek has no MTP (its DSpark drafter is chosen by the engine); the other families have the MTP and nothing to pick against it.
  assert.deepEqual(draftingChoices('qwen', true), all);
  assert.deepEqual(draftingChoices('gyro', true), all);
  assert.deepEqual(draftingChoices('unsloth', true), all, "Unsloth's quantizations are the same weights (DFlash2 measured on UD-Q4_K_XL)");
  for (const family of ['coder', 'swift', 'nvfp4', 'qwen36', 'deepseek41', undefined]) assert.equal(draftingChoices(family, true), null, `${family}, served`);
  // the app learns it from Control, once, and false until Control answers
  const host = read('../server/native-control.mjs');
  assert.match(host, /dflash2:dflash2Served===true,dflash2Support,dflash2Selq4,dflash2Split\}/);
  assert.match(host, /dflash2Served=p\.serves===true/);
  assert.match(host, /function probeDrafting\(\)\{if\(!packaged\|\|dflash2Served!==null/);
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /\{tr\(draftingChoices \? <div className="native-choice__col">/, 'no choices: no column');
});

test('Settings > Native engine settings hides the MTP and Drafting rows for DeepSeek and the Drafting row where it has no choice', () => {
  const byId = Object.fromEntries(NATIVE_FIELDS.map(f => [f.id, f]));
  for (const id of ['mtp', 'drafting', 'batchMtp']) assert.equal(byId[id].notForModel, 'deepseek', id);
  assert.equal(byId.ds_experts.forModel, 'deepseek');
  const settings = read('../src/components/NativeSettings.tsx');
  assert.match(settings, /\(!f\.notForModel \|\| !model\.toLowerCase\(\)\.includes\(f\.notForModel\)\) && \(f\.id !== 'drafting' \|\| draftingOffered\)/);
});

test('the DFlash2 drafter is fetched only by the Models screen own Load: a restart, a tuning run or any other load never downloads', () => {
  const source = read('../server/native-control.mjs');
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /asked===undefined\|\|drafts\?\.needed!==true/, 'no explicit choice in this job: return before any fetch');
  const loadScreen = read('../src/components/NativeModels.tsx');
  assert.match(loadScreen, /\.\.\.\(draftingChoices \? \{ values: \{ drafting \} \} : \{\}\)/, 'the screen passes the choice only when it showed the dropdown');
  const tune = source.slice(source.indexOf("const loaded=await loadEngine(job,{model:args.model,multiple:false"), source.indexOf("const loaded=await loadEngine(job,{model:args.model,multiple:false") + 140);
  assert.doesNotMatch(tune, /values:/, 'the tuning run loads without a choice');
});

test('the paragraph under the dropdown is gone (owner, 11-01: people who care know what to do); the catalog words carry no verdict', () => {
  const screen = read('../src/components/NativeModels.tsx');
  for (const part of ['7.8% faster than MTP on 19 short prompts', '1.1% faster on long thinking', 'faster on 21 of 23 prompts', 'the answers were identical']) assert.ok(!screen.includes(part), part);
  const en = JSON.parse(read('../src/locales/df2-en.json'));
  for (const text of Object.values(en)) for (const word of ['recommend', 'best ', 'better', 'should']) assert.ok(!text.toLowerCase().includes(word), `${word} in "${text.slice(0, 50)}"`);
});

test('one drafter per run: Auto is DFlash2 where the engine has it, the same word in Python and in the app', async () => {
  const { transform } = await import('esbuild');
  const code = (await transform(read('../src/lib/drafting.ts'), { loader: 'ts', format: 'esm' })).code;
  const ts = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const host = await import('../server/native-settings.mjs');
  for (const [choice, family, served] of [['auto', 'qwen', true], [undefined, 'gyro', true], ['auto', 'unsloth', true], ['auto', 'swift', true], ['auto', 'nvfp4', true], ['auto', 'qwen', false], ['auto', 'coder', true], ['mtp', 'qwen', true], ['dflash2', 'qwen', true], ['off', 'qwen', true], ['dflash2', 'coder', true]]) {
    assert.equal(ts.resolveDrafter(choice, family, served), host.resolveDrafter(choice, family, served), `${choice} ${family} ${served}`);
  }
  assert.equal(ts.resolveDrafter('auto', 'qwen', true), 'dflash2');
  assert.equal(ts.resolveDrafter('auto', 'qwen', null), 'mtp', 'Control has not answered: the MTP');
  assert.equal(host.resolveDrafter('auto', 'Qwen3.8-Flash-Next-iq3_s', true), 'dflash2', 'a run config model name works too');
  // the names setup.py writes (f"{family name}-{model.lower()}"): Unsloth's quantizations are DFlash2's; Swift, Coder and NVFP4 are not
  for (const name of ['qwen3.8-flash-next-unsloth-ud-q4_k_xl', 'qwen3.8-flash-next-unsloth-ud-iq4_xs', 'qwen3.8-flash-next-unsloth-q8_0', 'qwen3.8-flash-next-gyro-gyro-s']) assert.equal(host.resolveDrafter('auto', name, true), 'dflash2', name);
  for (const name of ['qwen3.8-flash-next-coder-iq1_m', 'qwen3.8-flash-next-nvfp4-nvfp4', 'swift-1.5-iq2_xs', 'qwen3.8-flash-next-unsloth', 'qwen3.6-35b-a3b-ud-q4_k_xl']) assert.equal(host.resolveDrafter('auto', name, true), 'mtp', name);
  assert.equal(ts.resolveDrafter('auto', 'unsloth', true), 'dflash2');
  assert.equal(host.resolveDrafter('auto', 'unsloth', true), 'dflash2');
});

test('a new install fetches only the drafter it will use; the fit and the slot estimates use the drafter chosen', async () => {
  const source = read('../server/native-control.mjs');
  assert.match(source, /'gguf_dir','drafter'\]/);
  assert.match(source, /resolveDrafter\(args\.drafting,args\.family,dflash2Served,dflash2Split\);if\(d==='dflash2'\)args=\{\.\.\.args,drafter:'dflash2'\}/, 'DFlash2: the installer leaves the MTP layer out');
  const { chooseParallel, MTP_VRAM_GB, DFLASH2_VRAM_GB } = await import('../server/native-parallel.mjs');
  assert.deepEqual([MTP_VRAM_GB, DFLASH2_VRAM_GB], [1.08, 0.89]);
  const config = { args: ['--mtp', '/m', '--max-context', '32768'], model_name: 'qwen3.8-flash-next-iq3_s' };
  const hardware = { gpus: [{ index: 0, vram_gb: 24 }] };
  const solo = chooseParallel({ config, hardware, drafter: 'dflash2' });
  assert.equal(solo.slots, 1);
  assert.match(solo.note, /one request at a time/);
  assert.notEqual(chooseParallel({ config, hardware, drafter: 'mtp' }).slots, 1, 'the MTP keeps the slots Strata fills in');
  assert.equal(chooseParallel({ config, hardware, drafter: 'dflash2', values: { parallel: 4 } }).slots, 4, 'a number the person chose stays (the start then loads the MTP and says so)');
  const two = { config: { ...config, gpu: [0, 1] }, hardware: { gpus: [{ index: 0, vram_gb: 24 }, { index: 1, vram_gb: 24 }] } };
  const split = chooseParallel({ ...two, drafter: 'dflash2' });
  assert.equal(split.slots, 1, 'a layer split runs DFlash2 alone, one request at a time (owner 11-02)');
  assert.match(split.note, /one request at a time/);
  assert.ok(chooseParallel({ ...two, drafter: 'mtp' }).slots > 1, 'the MTP on a layer split keeps the slots Strata fills in');
  const fit = read('../src/lib/native-fit.ts');
  assert.match(fit, /DRAFTER_VRAM_GB: Record<string, number> = \{ mtp: MTP_VRAM_GB, dflash2: DFLASH2_VRAM_GB, dflash2_selq4: DFLASH2_SELQ4_VRAM_GB, off: 0 \}/);
});

test('a failed MTP download says one short line, not the installer text, and the Load goes on', () => {
  const source = read('../server/native-control.mjs');
  const fn = source.slice(source.indexOf('async function installMtpLayer'), source.indexOf('async function prepareDrafter'));
  assert.match(fn, /job\.drafterNote='The MTP drafter could not be downloaded\.'/);
  assert.doesNotMatch(fn, /error\.message/);
});

test('the drafter memory figures in the app are the ones strata_control/drafting.py holds (DRAFTER_MIB), and the page uses the true one', async () => {
  const py = read('../../../strata_control/drafting.py');
  const table = JSON.parse(py.match(/^DRAFTER_MIB = (\{[^}]*\})/m)[1]);
  const gb = mib => Math.floor(mib / 10 + 0.5) / 100;          // the same rounding as DRAFTER_GB (half up to 10 MiB)
  assert.deepEqual(Object.keys(table).sort(), ['dflash2', 'dflash2_selq4', 'mtp', 'off']);
  const parallel = await import('../server/native-parallel.mjs');
  assert.equal(parallel.MTP_VRAM_GB, gb(table.mtp));
  assert.equal(parallel.DFLASH2_VRAM_GB, gb(table.dflash2));
  assert.equal(parallel.DFLASH2_SELQ4_VRAM_GB, gb(table.dflash2_selq4));
  assert.deepEqual([table.dflash2, table.dflash2_selq4, table.mtp], [890, 715, 1082]);
  const fit = read('../src/lib/native-fit.ts');
  const num = name => Number(fit.match(new RegExp(`export const ${name} = ([0-9.]+);`))[1]);
  assert.equal(num('MTP_VRAM_GB'), gb(table.mtp));
  assert.equal(num('DFLASH2_VRAM_GB'), gb(table.dflash2));
  assert.equal(num('DFLASH2_SELQ4_VRAM_GB'), gb(table.dflash2_selq4));
  assert.match(fit, /DRAFTER_VRAM_GB: Record<string, number> = \{ mtp: MTP_VRAM_GB, dflash2: DFLASH2_VRAM_GB, dflash2_selq4: DFLASH2_SELQ4_VRAM_GB, off: 0 \}/);
  // Control's word reaches the page, and the page hands the fit the drafter it really loads
  const control = read('../server/native-control.mjs');
  assert.match(control, /dflash2Selq4=p\.serves===true&&p\.selq4===true/);
  assert.match(control, /dflash2Support,dflash2Selq4,dflash2Split\}/);
  assert.match(read('../src/lib/native.ts'), /dflash2Selq4\?:boolean/);
  const page = read('../src/components/NativeModels.tsx');
  assert.match(page, /const fitDrafter = nowDrafter === 'dflash2' && native\?\.capabilities\?\.dflash2Selq4 === true \? 'dflash2_selq4' : nowDrafter;/);
  assert.match(page, /drafter: fitDrafter/);
  assert.doesNotMatch(page, /drafter: nowDrafter/);
});

test('where the all-cards split engine takes the start, Auto is the MTP in the label, the fit, the install and a plain Auto load (Control plan says so)', async () => {
  const { transform } = await import('esbuild');
  const code = (await transform(read('../src/lib/drafting.ts'), { loader: 'ts', format: 'esm' })).code;
  const ts = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const host = await import('../server/native-settings.mjs');
  for (const [choice, family, served, split, want] of [
    ['auto', 'qwen', true, true, 'mtp'], [undefined, 'unsloth', true, true, 'mtp'], ['auto', 'gyro', true, true, 'mtp'],
    ['auto', 'qwen', true, false, 'dflash2'], ['auto', 'qwen', true, undefined, 'dflash2'], ['auto', 'qwen', true, null, 'dflash2'],
    ['dflash2', 'qwen', true, true, 'dflash2'], ['mtp', 'qwen', true, true, 'mtp'], ['off', 'qwen', true, true, 'off'],
    ['auto', 'qwen', false, true, 'mtp'], ['auto', 'swift', true, true, 'mtp'],
  ]) {
    assert.equal(ts.resolveDrafter(choice, family, served, split), want, `ts ${choice} ${family} ${served} ${split}`);
    assert.equal(host.resolveDrafter(choice, family, served, split), want, `host ${choice} ${family} ${served} ${split}`);
  }
  const source = read('../server/native-control.mjs');
  assert.match(source, /dflash2Split=p\.split===true/, 'the probe reads the plan word');
  assert.match(source, /drafts\?\.mtp_needed===true&&\(args\.values\?\.drafting!==undefined\|\|\(drafts\.split===true\|\|args\.firstLoad===true\)&&\(config\.drafting==null\|\|String\(config\.drafting\)\.toLowerCase\(\)==='auto'\)\)/,
    'a plain Auto load fetches the draft layer when the plan says split (or it is the first load after a setup); a saved explicit choice is left alone');
  assert.match(read('../src/lib/native.ts'), /dflash2Split\?:boolean/);
});

test('the MTP fetch the app makes names the folder of the model own files, so the installer never downloads the model again', () => {
  const source = read('../server/native-control.mjs');
  assert.match(source, /async function installMtpLayer\(job,args,config\)\{[\s\S]*?const native=\(\(\)=>\{const a=\(config\.args\?\?\[\]\)\.map\(String\),i=a\.indexOf\('--native'\)[\s\S]*?install\(job,\{family:fam\.family,model:size\.model,drafter:'mtp',vision:args\.vision\?\?\(config\.vision\?'yes':'no'\),\.\.\.\(native\?\{gguf_dir:dirname\(native\)\}:\{\}\)\}\)/);
});

test('0.2.10.1: the app plans and fetches like the Terminal (no --no-split), and Auto on a computer where Control says split is the MTP in the label and the fit', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async args => { calls.push({ args }); return { code: 0, json: { ok: true, event: 'result', serves: true }, text: '', stderr: '' }; } });
  await control.draftingPlan({ mode: 'auto' });
  await control.draftingPlan({ configPath: 'C:/w/cfg.json', mode: 'auto', family: 'unsloth', dataDir: 'C:/d' });
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'auto' });
  assert.equal(calls.length, 3);
  for (const c of calls) assert.ok(!c.args.includes('--no-split'), c.args.join(' '));
  const { transform } = await import('esbuild');
  const code = (await transform(read('../src/lib/drafting.ts'), { loader: 'ts', format: 'esm' })).code;
  const ts = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const host = await import('../server/native-settings.mjs');
  for (const split of [false, undefined, null]) {
    assert.equal(ts.resolveDrafter('auto', 'unsloth', true, split), 'dflash2');
    assert.equal(host.resolveDrafter('auto', 'unsloth', true, split), 'dflash2');
  }
  // Control's word split:true (capabilities.dflash2Split, read from `drafting plan` at start-up) makes Auto the MTP; a person's own DFlash2 stays
  assert.equal(ts.resolveDrafter('auto', 'unsloth', true, true), 'mtp');
  assert.equal(host.resolveDrafter('auto', 'unsloth', true, true), 'mtp');
  assert.equal(ts.resolveDrafter('dflash2', 'unsloth', true, true), 'dflash2');
  assert.equal(host.resolveDrafter('dflash2', 'unsloth', true, true), 'dflash2');
  assert.equal(ts.resolveDrafter('off', 'unsloth', true, true), 'off');
  assert.match(read('../server/native-control.mjs'), /dflash2Split=p\.split===true/, 'the probe reads the split word of Control');
  assert.doesNotMatch(read('../server/control-runtime.mjs'), /'--no-split', '--output', 'json'\]|'--no-split', '--if-needed'/);
});
