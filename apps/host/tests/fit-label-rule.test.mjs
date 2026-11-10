/**
 * The version list's fit word and Control's (strata_control/fit_label.py) are one rule (owner 11-08): the weight files without the lookup table against
 * RAM + the graphics memory that counts. QAGUI-STRIX 11-09: the app counted a unified-memory chip's shared memory (the RAM again) as graphics memory, so
 * Q8_0 on Strix Halo said nothing while the catalog and the Terminal said "may run slower". This test runs the Python rule and the app's on the same
 * computers and sizes and needs them to agree.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const repo = resolve(root, '../..');
const { code } = await transform(readFileSync(resolve(root, 'src/lib/native-fit.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const { cardRoomGb, fitOn, versionFit } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const PCS = {
  // Strix Halo, Windows (QAGUI-STRIX 11-09): 63.6 GB RAM, a 64 GB BIOS carve-out + 31.8 GB shared = 95.8 GB the GPU can reach
  strix: { ram: 63.6, cards: [{ name: 'AMD Radeon 8060S', uma: true, dedicated_gb: 64, shared_gb: 31.8, vram_gb: 95.8 }] },
  // an APU whose report has no carve-out
  apu: { ram: 32, cards: [{ name: 'iGPU', uma: true, vram_gb: 16 }] },
  // RTX 5070 12 GB, 64 GB RAM (the owner PC)
  rtx12: { ram: 63.9, cards: [{ name: 'RTX 5070', vram_gb: 11.9 }] },
  // two cards: the larger counts
  two: { ram: 32, cards: [{ name: 'A', vram_gb: 8 }, { name: 'B', vram_gb: 24 }] },
  nocard: { ram: 64, cards: [] },
};
// sizes of the files without the table (Q8_0 133.8 = 188.2 - 54.4) and edges around each PC's RAM + card room
const SIZES = [27.6, 44.7, 54.4, 64.0, 75.8, 75.9, 96.0, 127.6, 127.7, 133.8];

function python() {
  for (const exe of process.platform === 'win32' ? ['python', 'py'] : ['python3', 'python']) {
    const r = spawnSync(exe, ['--version'], { encoding: 'utf8' });
    if (r.status === 0) return exe;
  }
  return null;
}

test('the app\'s fit word is Control\'s (fit_label.py) on Strix Halo, an APU, a 12 GB card, two cards and no card', { skip: !python() && 'no python here' }, () => {
  const exe = python();
  const script = 'import json,sys; sys.path.insert(0, sys.argv[1]); from strata_control import fit_label as f; pcs=json.loads(sys.argv[2]); sizes=json.loads(sys.argv[3]); '
    + 'print(json.dumps({k: [f.label(s, p["ram"], f.card_room_gb(p["cards"])) for s in sizes] for k, p in pcs.items()}))';
  const r = spawnSync(exe, ['-I', '-B', '-c', script, repo, JSON.stringify(PCS), JSON.stringify(SIZES)], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  const control = JSON.parse(r.stdout);
  for (const [name, pc] of Object.entries(PCS)) {
    const hw = { ram_gb: pc.ram, vram_gb: cardRoomGb(pc.cards) };
    const app = SIZES.map(size => versionFit({ fit_gb: size, memory_gb: size }, hw) === 'fits' ? 'fits' : 'may run slower');
    assert.deepEqual(app, control[name], name);
  }
  assert.equal(control.strix[SIZES.indexOf(133.8)], 'may run slower', 'Strix Q8_0: may run slower');
  assert.equal(control.strix[SIZES.indexOf(127.6)], 'fits', 'up to RAM + carve-out it fits');
});

test('a unified-memory chip counts only its carve-out; a card its memory; the largest card counts', () => {
  assert.equal(cardRoomGb(PCS.strix.cards), 64);
  assert.equal(cardRoomGb(PCS.apu.cards), 0);
  assert.equal(cardRoomGb(PCS.rtx12.cards), 11.9);
  assert.equal(cardRoomGb(PCS.two.cards), 24);
  assert.equal(cardRoomGb([]), 0);
  assert.equal(cardRoomGb(undefined), 0);
});

test('the Models page and the first-run page give the label the card room, not the shared pool', () => {
  const read = f => readFileSync(resolve(root, 'src', f), 'utf8');
  const models = read('components/NativeModels.tsx');
  assert.match(models, /cardRoomGb\(pcGpu \? \[pcGpu\] : \[\]\)/);
  assert.match(models, /versionFit\(selectedSize, labelHardware,/);
  assert.match(models, /hardware=\{labelHardware\}/);
  assert.match(read('components/GuidedSetup.tsx'), /vram_gb: cardRoomGb\(/);
});

test('the This computer card follows the same rule: on Strix Halo Q8_0 now says part is read from the SSD, as the label does', () => {
  const q8 = { model: 'Q8_0', experts_gb: 128.3, fit_gb: 133.8 };
  const now = fitOn(q8, { ram_gb: 63.6, vram_gb: cardRoomGb(PCS.strix.cards) }, true, { context: 131072, kv: 'int8' });
  const before = fitOn(q8, { ram_gb: 63.6, vram_gb: 95.8 }, true, { context: 131072, kv: 'int8' });
  assert.ok(now.spillGb > 0, 'some of it from the SSD: ' + JSON.stringify(now));
  assert.equal(before.spillGb, 0, 'the shared pool counted twice said it all fits');
  assert.ok(now.vramUseGb + now.ramUseGb <= 63.6 + 64, 'never more than the memory there is');
  const models = readFileSync(resolve(root, 'src/components/NativeModels.tsx'), 'utf8');
  assert.match(models, /fit=\{selectedSize \? fitOn\(selectedSize, labelHardware,/);
  assert.match(models, /vram_gb: cardRoomGb\(\[card \?\? pcGpu\]/);
});
