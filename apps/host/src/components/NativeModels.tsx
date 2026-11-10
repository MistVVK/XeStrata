import { tr } from '../lib/i18n';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Screen } from './layout';
import { VersionPicker } from './VersionPicker';
import { FamilySwitch } from './FamilySwitch';
import { PcComputer } from './PcComputer';
import { cacheGb, cardRoomGb, fitOn, memoryNeeded, versionFit } from '../lib/native-fit';
import { contextIsTight, defaultContext, DEFAULT_CONTEXT, OLD_DEEPSEEK_CONTEXT, openingContext } from '../lib/default-context';
import { GpuRoles } from './GpuRoles';
import { fromLegacy, normalize as normalizeRoles, order as roleOrder, recommend as recommendRoles, rows as roleRows, sameRoles, vramOn, type RoleCard, type Roles } from '../lib/gpu-roles';
import { Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
import { installProgress, installStage, stageText, useDownloadEta } from '../lib/download-eta';
import { useKept } from '../lib/kept-state';
import { DownloadParts } from './DownloadParts';
import { EngineWarning } from './EngineWarning';
import { EngineUpdateCard } from './EngineUpdateCard';
import { EngineUpdateDialog } from './EngineUpdateDialog';
import { sizeGb } from '../lib/format';
import { NativeDeleteDialog } from './NativeDelete';
import { plainFailureText } from '../lib/plain-errors';
import { api } from '../lib/client';
import { go, setSettingsSection, selectRemoteModel, pushToast } from '../lib/store';
import { installedListPending, nativeAction, useNative, type NativeJob } from '../lib/native';
import { tuneEffective } from '../lib/native-tune';
import { tunedFields } from '../lib/tune-values';
import { shownVision, steadyNoReader } from '../lib/pictures-choice';
import { draftingChoices as draftingOptions, resolveDrafter } from '../lib/drafting';
import { deepseekContexts, deepseekContextNote } from '../lib/deepseek-context';
import { NativeSettings } from './NativeSettings';
import { NativeSettingsFile } from './NativeSettingsFile';
import { RunImageModel } from './RunImageModel';
import { IMAGE_GENERATION_VISIBLE } from '../lib/features';
import { useStore } from '../lib/store';
import { NativePresets } from './NativePresets';
import { NativeTune } from './NativeTune';
import { NativeTuneBadge } from './NativeTuneBadge';
import { nativeCatalogModels, catalogVersionLabel, fetchesPictures } from '../lib/native-catalog';
export function NativeModels({ onOtherModels, onReady, onImport, onConnectEngine }: {
    onOtherModels: () => void;
    onImport?: () => void;
    onConnectEngine?: () => void;
    onReady?: () => void;
}) {
    const native = useNative(), imageLibrary = useStore(s => s.imageModels), hardware = useStore(s => s.hardware);
    // The choices are kept while the page is closed (Models -> Monitor -> Models used to start them over and a person lost track of what was downloading).
    const [family, setFamily] = useKept('models.family', ''), [size, setSize] = useKept('models.size', ''), [context, setContext] = useKept('models.context', DEFAULT_CONTEXT), [contextPicked, setContextPicked] = useKept('models.contextPicked', false), [vision, setVision] = useKept('models.vision', 'yes');
    const manual = useStore(s => s.setup?.choices?.onboarding?.manual ?? false);
    const [expert, setExpert] = useKept('models.expert', manual), [multiple, setMultiple] = useKept('models.multiple', false), [roles, setRoles] = useKept<Roles | null>('models.roles', null), [kv, setKv] = useKept('models.kv', 'int8'), [lowRam, setLowRam] = useKept('models.lowRam', 'auto'), [split, setSplit] = useKept('models.split', 'auto');
    const [resourceProfile, setResourceProfile] = useKept<'work' | 'performance'>('models.profile', 'work');
    const [drafting, setDrafting] = useState('auto');
    const [pending, setPending] = useState(false), [error, setError] = useState(''), [plan, setPlan] = useState<Record<string, any> | null>(null), [drafterGb, setDrafterGb] = useState<number | null>(null), [loadedError, setLoadedError] = useState(''), [tuneOpen, setTuneOpen] = useState(false), [jobId, setJobId] = useKept('models.job', ''), [manage, setManage] = useState(false);
    const [toastedJob, setToastedJob] = useKept('models.toasted', ''), [appliedSig, setAppliedSig] = useKept('models.applied', '');
    const presetPending = useRef(false), visionPicks = useRef(new Map<string, string>());   // Read pictures per version: what the person set, or what the version opened with; later reports of the host never overwrite it
    const [deleting, setDeleting] = useState<string | null>(null);
    const families = nativeCatalogModels(native?.catalog?.families ?? []);
    const expertSection = useRef<HTMLElement>(null);
    // The card is inserted above the fold and the browser's scroll anchoring
    // immediately compensates by scrolling down, which cancelled the smooth
    // scroll started in the same frame: the page stopped 121px in and clipped
    // the cards above. Scroll after the layout has settled, and never animate
    // when motion is reduced.
    useEffect(() => {
        if (!expert)
            return;
        const frame = requestAnimationFrame(() => expertSection.current?.scrollIntoView({
            block: 'start',
            behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        }));
        return () => cancelAnimationFrame(frame);
    }, [expert]);
    const selectedFamily = families.find(f => f.family === family || f.sizes.some(size => size.sourceFamily === family)) ?? families[0], selectedSize = selectedFamily?.sizes.find(s => s.id === size) ?? selectedFamily?.sizes[0];
    const installed = native?.installed?.models.find(m => m.model === selectedSize?.id), listPending = installedListPending(native);
    // The sizes this model offers for the conversation context; it opens at 128K (or its largest size when that is smaller).
    const contextList = selectedSize?.multi_file ? deepseekContexts(installed?.contextMax) : native?.catalog?.contexts ?? [8192, 32768, 65536, 131072, 262144];
    // The model that is running is shown as running, with Unload: not offered for loading again.
    const downloaded = native?.installed?.models ?? [], allSizes = families.flatMap(f => f.sizes);
    // This computer for the fit figures: the biggest usable graphics card, or the ones chosen under Expert settings.
    const engineVendor = native?.hardware?.engine_vendor ?? null, roleCards: RoleCard[] = roleRows(native?.hardware?.gpus as RoleCard[] | undefined, engineVendor);
    const pcGpus = (native?.hardware?.gpus ?? []).filter(g => g.usable !== false), pcGpu = [...pcGpus].sort((x, y) => y.vram_gb - x.vram_gb)[0] ?? null;
    const pcRam = native?.hardware?.ram_gb || hardware?.memory.totalGb || null, pcVram = pcGpu?.vram_gb ?? 0;
    // What the models that stay loaded (Run multiple models) already take: the fit figures are for what is left.
    const sizeOf = (id: string) => allSizes.find(s => s.id === id);
    const keptModels = expert && multiple ? (native?.instances ?? []).filter(i => i.status !== 'error' && i.model !== selectedSize?.id) : [];
    const keptUse = keptModels.map(i => { const s = sizeOf(i.model); const card = pcGpus.find(g => g.index === i.gpus[0]); const f = s && pcRam ? fitOn(s, { ram_gb: pcRam, vram_gb: card?.vram_gb ?? pcVram }) : null; return { gpus: i.gpus, ram: f?.ramUseGb ?? 0, vram: f?.vramUseGb ?? 0 }; });
    const keptRam = keptUse.reduce((n, u) => n + u.ram, 0);
    const keptVramOn = (cards: number[]) => keptUse.filter(u => u.gpus.some(g => cards.includes(g))).reduce((n, u) => n + u.vram, 0);
    // The models that are loaded now, with what each takes of graphics memory, RAM and SSD (the app's estimate from the same rule the fit uses: the engine does not report this split).
    const loadedModels = (native?.instances ?? []).filter(i => i.status === 'ready' || i.status === 'loading').flatMap(i => { const s = sizeOf(i.model); if (!s || !pcRam) return []; const card = pcGpus.find(g => g.index === i.gpus[0]); return [{ id: i.model, label: `${i.label} · ${s.model}`, fit: fitOn(s, { ram_gb: pcRam, vram_gb: cardRoomGb([card ?? pcGpu].filter(c => !!c)) }, true, { context: i.context ?? context, kv }) }]; });
    // The versions that are really on this computer: a config whose files are all there (every part of a model in parts).
    const readyIds = useMemo(() => new Set(downloaded.filter(m => m.ready).map(m => m.model)), [downloaded]);
    // What a running job works on, by its name in the list: "Qwen3.8-Flash-Next · Gyro · Smallest".
    const jobLabel = (j: NativeJob) => { const v = allSizes.find(s => s.id === j.model || (s.model === j.model && (!j.family || s.sourceFamily === j.family))); const f = v && families.find(x => x.sizes.includes(v)); return v ? [f?.title, catalogVersionLabel(v)].filter(Boolean).join(' · ') : ''; };
    const loadedSelected = native?.instances.find(i => i.model === selectedSize?.id && i.status === 'ready');
    // What the running model was really started with is what the page shows: after Load the highlight must still be the profile that was applied.
    useEffect(() => { if (loadedSelected?.resourceProfile)
        setResourceProfile(loadedSelected.resourceProfile); }, [loadedSelected?.id, loadedSelected?.resourceProfile]);
    // After a tuning, Expert mode shows the values the tuning set (the active preset's), not the ones the model was prepared with.
    const [tunedNow, setTunedNow] = useState<{ name: string; fields: ReturnType<typeof tunedFields> } | null>(null);
    useEffect(() => {
        if (!selectedSize || !installed?.ready) { setTunedNow(null); return; }
        let alive = true;
        tuneEffective(selectedSize.id, resourceProfile, selectedSize.images ? vision : 'no').then(t => { if (alive) setTunedNow(t.applied && t.preset ? { name: t.preset.name, fields: tunedFields(t.changed) } : null); }).catch(() => { if (alive) setTunedNow(null); });
        return () => { alive = false; };
    }, [selectedSize?.id, installed?.ready, resourceProfile, vision, native?.jobs.filter(j => j.state === 'done').length]);
    useEffect(() => { if (tunedNow?.fields.kv) setKv(tunedNow.fields.kv as typeof kv); if (tunedNow?.fields.lowRam) setLowRam(tunedNow.fields.lowRam as typeof lowRam); }, [tunedNow]);
    const active = native?.jobs.find(j => j.state === 'running');
    // Catalog groups share quantization names. A remembered job must still belong
    // to the selected version; another family's failure stays in diagnostics.
    const selectedJob = (j: NativeJob) => !!selectedSize && (j.model === selectedSize.id ||
        (j.model === selectedSize.model && j.family === selectedSize.sourceFamily));
    const job = native?.jobs.find(j => j.id === jobId && selectedJob(j)) ??
        native?.jobs.filter(j => ['load', 'install'].includes(j.kind) && selectedJob(j)).at(-1);
    // This model can read pictures but the installed engine has no picture reader: pictures run off for now, nothing is saved, and the row says nothing wordy.
    // While a job runs the host is busy and its picture-reader probe can drop out for a moment: the checkbox then keeps what it showed, so the person's choice never flickers.
    const noReaderNow = !!selectedSize?.imagesCapable && !selectedSize.images, noReaderSeen = useRef({ id: '', value: false });
    noReaderSeen.current = steadyNoReader(noReaderNow, !!active, noReaderSeen.current, selectedSize?.id ?? '');
    const noReader = noReaderSeen.current.value;
    // Graphics cards (strata_control/gpu_fit.py: the one rule of the app, the Terminal and setup): the recommended option is pre-selected, the person may pick another
    // A card a loaded model already uses (a second model on it usually does not fit): said on its row, and not the pre-selected one while another is free.
    const usedBy = (cards: number[]) => (native?.instances ?? []).filter(i => i.status !== 'error' && i.model !== selectedSize?.id && i.gpus.some(g => cards.includes(g))).map(i => i.label);
    // Main / Helper / Off per card (src/lib/gpu-roles.ts, strata_control/gpu_roles.py): the recommendation until the person changes a role; a loaded model shows what it was started with.
    // With Run multiple models, a card another loaded model already uses starts Off for this one (while a free card exists).
    const busy = (i: number) => expert && multiple && usedBy([i]).length > 0, freeCards = roleCards.filter(c => !busy(c.index));
    const recommended = freeCards.length && freeCards.length < roleCards.length ? { ...recommendRoles(freeCards, engineVendor), roles: { ...Object.fromEntries(roleCards.map(c => [c.index, 'off'])), ...recommendRoles(freeCards, engineVendor).roles } as Roles } : recommendRoles(roleCards, engineVendor);
    const loadedRoles = loadedSelected && roleCards.length > 1 ? fromLegacy(loadedSelected.gpus, roleCards, 'as_given') : null;
    const effectiveRoles: Roles = roleCards.length > 1 ? (roles ? normalizeRoles(roles, roleCards) : loadedRoles ?? recommended.roles) : recommended.roles;
    const customRoles = !sameRoles(effectiveRoles, recommended.roles);
    const changeRoles = (next: Roles) => setRoles(sameRoles(next, recommended.roles) ? null : next);
    const cardsGpus: number[] = roleCards.length > 1 ? roleOrder(effectiveRoles, roleCards) : [];
    const chosenCards = cardsGpus;
    const fitHardware = { ram_gb: pcRam ? Math.max(0, pcRam - keptRam) : pcRam, vram_gb: Math.max(0, (roleCards.length > 1 ? vramOn(effectiveRoles, roleCards) : pcVram) - keptVramOn(chosenCards)) };
    // The fit label's own figures (strata_control/fit_label.py): a unified-memory chip counts only its carve-out beside the RAM (lib/native-fit.ts cardRoomGb).
    const labelHardware = { ram_gb: fitHardware.ram_gb, vram_gb: roleCards.length > 1 ? fitHardware.vram_gb : Math.max(0, cardRoomGb(pcGpu ? [pcGpu] : []) - keptVramOn(chosenCards)) };
    // How the chosen version sits on this computer (the same rule as the version list): the page says a word only for "may run slower".
    const fitState = selectedSize ? versionFit(selectedSize, labelHardware, { context, kv }) : 'unknown';
    // A layer split written for another number of cards no longer fits: back to automatic.
    useEffect(() => { if (cardsGpus.length <= 1 && split !== 'auto') setSplit('auto'); }, [cardsGpus.length]);
    // Drafting (which drafter guesses ahead): a choice exists only for Qwen3.8-Flash-Next (the qwen and gyro families) on a computer whose engine can run DFlash2; every other model has no dropdown.
    const draftingChoices = selectedSize ? draftingOptions(selectedSize.sourceFamily ?? selectedFamily?.family, native?.capabilities?.dflash2) : null;
    // Control's plan says the all-cards split engine takes an Auto start on this computer (it runs the MTP draft layer); it needs two or more cards, so one card chosen in the roles is not it.
    const splitCards = native?.capabilities?.dflash2Split === true && cardsGpus.length !== 1;
    // The drafter this run loads (one per run): the fit and the plan use its graphics memory (DFlash2 0.89 GB, MTP 1.08 GB).
    const nowDrafter = draftingChoices ? resolveDrafter(drafting, selectedSize?.sourceFamily ?? selectedFamily?.family, native?.capabilities?.dflash2, splitCards) : 'mtp';
    // DFlash2 with the converted pack (the selector as Q4_0 rows) takes 175 MiB less graphics memory: the fit says the true number (Control's `selq4`, strata_control/drafting.py vram_gb).
    const fitDrafter = nowDrafter === 'dflash2' && native?.capabilities?.dflash2Selq4 === true ? 'dflash2_selq4' : nowDrafter;
    const draftingWords: Record<string, string> = { auto: 'Auto (recommended)', mtp: 'MTP', dflash2: 'DFlash2', off: 'Off' };
    const values = useMemo(() => ({ family: selectedSize?.sourceFamily, model: selectedSize?.model, context, vision: selectedSize?.images ? vision : 'no', kv, low_ram: lowRam, resourceProfile, ...(cardsGpus.length > 1 ? { gpus: cardsGpus.join(',') } : cardsGpus.length === 1 ? { gpu: cardsGpus[0] } : {}), loadAfter: true, multiple: expert && multiple }), [selectedSize?.sourceFamily, selectedSize?.model, context, vision, kv, lowRam, resourceProfile, cardsGpus.join(','), expert, multiple]);
    const downloadValues = { ...values, ...(draftingChoices ? { drafting } : {}), vision: selectedSize?.images && vision !== 'no' ? 'yes' : 'no', loadVision: selectedSize?.images ? vision : 'no', ...(noReader ? { visionUnavailable: true } : {}), ...(contextPicked ? { contextChosen: true } : {}) };
    useEffect(() => { if (!native?.catalog || family)
        return; const current = native.instances.find(i => i.id === native.defaultId); const rec = native.catalog.recommendation ?? { family: null, model: null }; const loadedFamily = families.find(f => f.sizes.some(s => s.id === current?.model)); const have = new Set((native.installed?.models ?? []).filter(m => m.ready).map(m => m.model)); const installedFamily = families.find(f => f.sizes.some(s => have.has(s.id))); const f = loadedFamily ?? installedFamily ?? families.find(f => f.family === rec.family || f.sizes.some(s => s.sourceFamily === rec.family)) ?? families[0]; if (f) {
        setFamily(f.family);
        const first = f.sizes.find(s => s.id === current?.model) ?? f.sizes.find(s => have.has(s.id)) ?? f.sizes.find(s => s.model === rec.model) ?? f.sizes[0];
        setSize(first?.id ?? '');
        // 128K for every configuration, DeepSeek included where its pack allows (the engine's own recommendation is not what the app opens with).
        setContext(defaultContext(first?.multi_file ? deepseekContexts() : native.catalog.contexts));
    } }, [native?.catalog]);
    // The version that is selected: a model already on this PC shows the settings it was prepared with (so loading it never asks to download);
    // a model that is not yet downloaded starts with Read pictures ON (owner rule: pictures on by default). This runs when the SELECTION or the
    // installed settings change, not when the page opens again: coming back from another tab keeps what was chosen.
    useEffect(() => {
        if (presetPending.current) {
            presetPending.current = false;
            return;
        }
        if (!selectedSize)
            return;
        const sig = installed ? ['have', installed.model, installed.ready, installed.context, installed.contextChosen, installed.kv, installed.lowRamMode, installed.drafting, installed.resourceProfile].join('|') : `new|${selectedSize.id}`;
        if (sig === appliedSig)
            return;
        setAppliedSig(sig);
        if (installed) {
            // The context it was prepared with when someone chose it; a model that holds the old default (32K, DeepSeek's 8K) without the marker opens at the new default.
            if (installed.context)
                setContext(openingContext(installed.context, contextList, installed.contextChosen === true, selectedSize.multi_file ? [OLD_DEEPSEEK_CONTEXT] : undefined));
            // Off only when the person turned it off (a saved choice on a model that has the picture files); a model prepared without them (images 'off', or an old "off" saved while the engine had no reader) starts with it ON.
            // The first time this version is seen as installed it opens from its saved settings; from then on that value is the person's (a host report that changes while a job runs never changes it).
            const known = visionPicks.current.get(selectedSize.id), value = shownVision(known, installed);
            if (known === undefined)
                visionPicks.current.set(selectedSize.id, value);
            setVision(value);
            setKv(installed.kv ?? 'int8');
            setLowRam(installed.lowRamMode ?? 'auto');
            setDrafting(installed.drafting ?? 'auto');
            // A model never tuned has no saved choice: keep the page's own (Leave room for my work) rather than flipping to Full performance.
            if (installed.resourceProfile && !loadedSelected?.resourceProfile)
                setResourceProfile(installed.resourceProfile);
        }
        else {
            setDrafting('auto');    // a model that is not installed has no saved choice: Auto (the page never carries the last model's Drafting over to another one)
            setVision(shownVision(visionPicks.current.get(selectedSize.id), undefined));
            if (!contextPicked)    // a context the person picked stays; otherwise 128K (DeepSeek too, where its pack allows)
                setContext(defaultContext(contextList));
        }
    }, [selectedSize?.id, installed?.model, installed?.ready, installed?.context, installed?.contextChosen, installed?.kv, installed?.lowRamMode, installed?.resourceProfile]);
    const pickVision = (next: string) => { if (selectedSize) visionPicks.current.set(selectedSize.id, next); setVision(next); };
    useEffect(() => { if (jobId && job?.state === 'done' && job.id !== toastedJob && (job.kind === 'load' || job.kind === 'install')) {
        setToastedJob(job.id);
        selectRemoteModel(null);
        pushToast('success', 'Model ready', 'Start a new chat when you’re ready.');
    } }, [job?.state]);
    // The picture files came without a load: a model that is loaded right now still runs without pictures until it is loaded again.
    useEffect(() => { if (jobId && job?.state === 'done' && job.id !== toastedJob && job.kind === 'pictures') {
        setToastedJob(job.id);
        pushToast('success', 'Picture files ready', loadedSelected && !loadedSelected.vision ? 'Load the model again to read pictures.' : 'Picture reading is ready.');
    } }, [job?.state]);
    async function perform(action: () => Promise<unknown>) { setPending(true); setError(''); try {
        await action();
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    /* Load model: no pop-up. The press itself is the yes. The plan is read first (Control checks the free disk space); a disk that is too small is said
       here, in the page, and nothing starts. Otherwise the download or the apply-and-load starts at once, with its size and notes in the page. */
    // The model is on this computer and only its picture files are missing: the Load fetches those files itself (about 0.9 GB, once) and goes on; it does not run the installer again.
    const picturesByLoad = fetchesPictures(selectedSize, native?.packaged);
    async function load() { if (!selectedSize)
        return; setPlan(null); setDrafterGb(null); await perform(async () => { if (installed?.ready && context === installed.context && (!selectedSize?.images || vision === 'no' || picturesByLoad || installed.images && installed.images !== 'off') && (!installed.kv || kv === installed.kv || kv === tunedNow?.fields.kv) && (lowRam === (installed.lowRamMode ?? 'auto') || lowRam === tunedNow?.fields.lowRam)) {
        const result = await nativeAction<NativeJob>('operations', { kind: 'load', args: { model: selectedSize.id, resourceProfile, vision: selectedSize?.images ? vision : 'no', ...(noReader ? { visionUnavailable: true } : {}), ...(contextPicked && !installed.contextChosen ? { contextChosen: true } : {}), multiple: expert && multiple, ...(draftingChoices ? { values: { drafting } } : {}), ...(roleCards.length > 1 ? (roles && customRoles ? { gpuRoles: effectiveRoles, layerSplit: expert && cardsGpus.length > 1 ? split : 'auto' } : cardsGpus.length ? { gpus: cardsGpus, gpusFromRecommendation: true, layerSplit: expert && cardsGpus.length > 1 ? split : 'auto' } : {}) : {}) } });
        setJobId(result.id);
    }
    else {
        const planned = await api.post<Record<string, any>>('/api/native/plan', downloadValues);
        if (planned?.plan?.disk_short) {
            setPlan(planned);
            return;
        }
        setDrafterGb(planned?.drafting?.needed ? Number(planned.drafting.gb) || null : null);
        const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: downloadValues });
        setJobId(result.id);
    } }); }
    /* Unload / Use in chat: a refusal (a chat is still using the model) is said here, beside the button, not far below the page. */
    async function loadedAction(path: string) { setLoadedError(''); setPending(true); try {
        await nativeAction(path);
    }
    catch (e) {
        setLoadedError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setPending(false);
    } }
    // The drafter note belongs to the download that is running; the not-enough-disk message belongs to the version it was said for.
    useEffect(() => { if (!active) setDrafterGb(null); }, [!!active]);
    useEffect(() => { setPlan(null); }, [selectedSize?.id]);
    return <Screen plainHeader title={<span className="sr-only">{tr("Models")}</span>} titleDetail={<FamilySwitch families={families} selected={selectedFamily?.family} onPick={f => { setFamily(f.family); setSize((f.sizes.find(s => native?.installed?.models.some(m => m.model === s.id && m.ready)) ?? f.sizes[0])?.id ?? ''); setVision('yes'); setError(''); }}/>} actions={<Btn icon="sliders" aria-pressed={expert} aria-expanded={expert} aria-controls="native-expert-settings" onClick={() => { setExpert(v => !v); setMultiple(false); }}>{tr(expert ? 'Close expert mode' : 'Expert mode')}</Btn>}>
  <div className="native-model-layout native-model-layout--fit">
   <div className="stack">
    {tr(native?.instances.length && expert ? <section className="st-card native-running"><h2 className="h4">{tr(native.instances.length === 1 ? 'Loaded model' : 'Loaded models')}</h2>{tr(native.instances.map(i => <div key={i.id}><span><strong>{tr(i.label)}</strong><small>{tr(i.gpus.length ? i.gpus.map((g, n) => `GPU ${g + 1}${i.gpus.length > 1 && n === i.gpus.length - 1 ? ' (Main)' : ''}`).join(' + ') : 'Automatic GPU')} · {tr(i.status === 'ready' ? i.phase : i.status)}{tr(native.defaultId === i.id ? ' · Chat default' : '')}</small>{tr(i.error ? <small className="inline-error">{tr(plainFailureText(i.error).message)}</small> : null)}</span>{tr(expert && multiple && native.defaultId !== i.id ? <Btn size="sm" variant="ghost" onClick={() => void loadedAction(`instances/${i.id}/default`)}>{tr("Use in chat")}</Btn> : null)}<Btn size="sm" disabled={pending || !!active} onClick={() => void loadedAction(`instances/${i.id}/unload`)}>{tr("Unload")}</Btn></div>))}{tr(loadedError ? <ErrorNotice message={loadedError} next={tr("Stop or answer the chat that is still using this model, then try again.")}/> : null)}</section> : null)}
    <section className="st-card native-choice">{tr(selectedFamily && (!selectedSize || selectedSize.needsNewerEngine) ? <><h2 className="h3">{tr("This model needs a newer engine.")}</h2><p className="small muted">{tr("Update your engine or import a local GGUF for this model.")}</p><Btn onClick={() => { setSettingsSection('Engine'); go('settings'); }}>{tr("Update engine")}</Btn><Btn variant="ghost" onClick={onImport ?? onOtherModels}>{tr("Use a local GGUF\u2026")}</Btn></> : null)}{tr(!selectedFamily && !listPending ? <><h2 className="h3">{tr("Set up Strata")}</h2><p className="small muted">{tr("Connect your local installation to choose an optimized model.")}</p><Btn variant="primary" onClick={onConnectEngine ?? (() => { setSettingsSection('Engine'); go('settings'); })}>{tr("Connect engine")}</Btn>{tr(native?.error ? <p className="small inline-error">{tr(native.error)}</p> : null)}<Btn variant="ghost" onClick={onOtherModels}>{tr("Other local models")}</Btn></> : null)}
     {tr(selectedSize ? <>
      <div className="native-choice__head"><span className="native-choice__label">{tr("Version")}</span><span className="native-choice__tools"><NativePresets values={values} onApply={v => { const f = families.find(f => f.family === v.family || f.sizes.some(s => s.sourceFamily === v.family)); if (!f)
        return; presetPending.current = true; setFamily(f.family); setSize(f.sizes.find(s => s.model === v.model && (s.sourceFamily === v.family || f.family === v.family))?.id ?? ''); setContext(v.context ?? 32768); setVision(v.vision ?? 'no'); setKv(v.kv ?? 'int8'); setLowRam(v.low_ram ?? 'auto'); }}/><Btn icon="models" size="sm" variant="ghost" onClick={() => setManage(true)}>{tr("Manage models")}</Btn></span></div>
      <div className="native-version"><VersionPicker pick={native?.catalog?.recommendation} versions={selectedFamily?.sizes ?? []} details={expert} selectedId={selectedSize.id} hardware={labelHardware} cache={{ context, kv, drafter: fitDrafter }} downloaded={readyIds} onSelect={id => { const f = families.find(f => f.sizes.some(s => s.id === id)); if (f) { setFamily(f.family); setSize(id); setError(''); } }} onOwnGguf={() => (onImport ?? onOtherModels)()}/></div>
      {tr(fitState === 'ssd' ? <p className="small muted native-spill-note"><Icon name="disk" size={16}/>{tr("may run slower")}</p> : null)}
      {tr(selectedSize?.multi_file && deepseekContextNote(context) ? <p className="small muted native-spill-note"><Icon name="info" size={16}/>{tr(deepseekContextNote(context))}</p> : null)}
      <div className="native-choice__row">
       <div className="native-choice__col"><span className="native-choice__label">{tr("Conversation context")}</span><Select ariaLabel={tr("Conversation context")} value={String(context)} options={contextChoices(contextList, context).map(n => ({ value: String(n), label: n % 1024 === 0 ? `${n / 1024}K` : `${n.toLocaleString()} tokens` }))} onChange={n => { setContextPicked(true); setContext(Number(n)); }}/></div>
       {tr(draftingChoices ? <div className="native-choice__col"><span className="native-choice__label">{tr("Drafting")}</span><Select ariaLabel={tr("Drafting")} value={drafting} disabled={!!active} options={draftingChoices.map(v => ({ value: v, label: tr(draftingWords[v]) }))} onChange={v => setDrafting(String(v))}/></div> : null)}
       <div className="native-choice__col"><span className="native-choice__label">{tr("Computer use")}</span><div className="native-resource-choice"><button aria-pressed={resourceProfile === 'work'} data-selected={resourceProfile === 'work'} onClick={() => setResourceProfile('work')}><Icon name="computer" size={20}/><span><strong>{tr("Leave room for my work")}</strong><small>{tr("Keeps 5-8 GB of memory free for other programs.")}</small></span>{tr(resourceProfile === 'work' ? <Icon name="check" size={15}/> : null)}</button><button aria-pressed={resourceProfile === 'performance'} data-selected={resourceProfile === 'performance'} onClick={() => setResourceProfile('performance')}><Icon name="bolt" size={20}/><span><strong>{tr("Full performance")}</strong><small>{tr("Give Strata priority")}</small></span>{tr(resourceProfile === 'performance' ? <Icon name="check" size={15}/> : null)}</button></div></div>
      </div>
      {tr(contextIsTight(fitHardware.vram_gb, context, kv) ? <p className="small muted native-spill-note"><Icon name="warning" size={16}/>{tr("This much context takes about {0} GB of graphics memory and leaves little room for the model on this card. Strata will try it; if it does not start, choose a shorter context.", [cacheGb(context, kv).toFixed(1)])}</p> : null)}
      {tr(loadedSelected?.metrics?.engine?.drafting_note ? <p className="small muted native-spill-note">{tr(String(loadedSelected.metrics.engine.drafting_note))}</p> : null)}
      {roleCards.length > 1 ? <GpuRoles cards={roleCards} roles={effectiveRoles} custom={customRoles} recommendedRoles={recommended.roles} vendor={engineVendor} disabled={!!active} onChange={changeRoles} onBack={() => setRoles(null)} memoryGb={selectedSize ? memoryNeeded(selectedSize) || null : null} ramGb={pcRam} spillGb={selectedSize ? fitOn(selectedSize, fitHardware).spillGb : 0} usedBy={i => usedBy([i])}/> : null}
      {tr(roleCards.length > 1 && (native?.hardware?.gpus ?? []).length > roleCards.length ? <p className="small muted native-cards__note">{tr('Not used: {0} (another make of graphics card than this engine drives).', [(native?.hardware?.gpus ?? []).filter(g => !roleCards.some(r => r.index === g.index && r.vendor === g.vendor)).map(g => g.name).join(', ')])}</p> : null)}
      {tr(selectedSize?.images || noReader ? <><div className="native-image-reading"><label title={noReader ? tr("Needs an engine update") : undefined}><input type="checkbox" checked={!noReader && vision !== 'no'} disabled={noReader || !!active} onChange={e => pickVision(e.target.checked ? 'yes' : 'no')}/><span><strong>{tr("Read pictures")}</strong></span></label>{tr(expert && vision !== 'no' && selectedSize?.sourceFamily !== 'deepseek41' ? <Select ariaLabel={tr("Image encoder device")} value={vision === 'yes' ? 'yes' : vision} options={[{ value: 'yes', label: 'Automatic' }, { value: 'gpu', label: 'GPU' }, { value: 'cpu', label: 'CPU' }]} onChange={pickVision}/> : null)}</div>{tr(vision !== 'no' && !noReader && !active && loadedSelected?.picturesNote ? <p className="small muted native-vision-note" role="status">{tr(loadedSelected.picturesNote)}</p> : null)}</> : null)}
     </> : null)}
     <EngineWarning native={native}/>
     <EngineUpdateCard native={native} busy={!!active}/>
     <EngineUpdateDialog open={false} onClose={() => {}}/>
     {tr(active ? <NativeProgress job={active} label={jobLabel(active)}/> : null)}
     {tr(active && drafterGb ? <p className="small muted native-spill-note"><Icon name="info" size={16}/>{tr(nowDrafter === 'mtp' ? (splitCards && drafting === 'auto' ? "Auto uses the all-cards split engine with the MTP draft layer: about {0} GB to download, once." : "Choosing MTP downloads its drafter: about {0} GB, once.") : "Choosing DFlash2 downloads its drafter: about {0} GB, once.", [drafterGb])}</p> : null)}
     {tr(job?.state === 'error' ? <ErrorNotice message={job.error ?? job.summary} action={tr("Retry")} onRetry={() => void load()}/> : null)}
     {tr(plan?.plan?.disk_short ? <ErrorNotice message={plan.plan.disk_short}/> : null)}
     {tr(error ? <ErrorNotice message={error}/> : null)}
     {selectedSize && installed?.ready ? <NativeTuneBadge model={selectedSize.id} resourceProfile={resourceProfile} vision={selectedSize.images ? vision : 'no'}/> : null}
     <div className="native-model-actions">{tr(!loadedSelected && !listPending && !installed?.ready && selectedSize?.download_gb && !active ? <span className="native-download-size small muted">{tr("{0} download", [sizeGb(selectedSize.download_gb)])}</span> : null)}{tr(loadedSelected ? <><Btn disabled={pending || !!active} onClick={() => void perform(() => nativeAction(`instances/${loadedSelected.id}/unload`))}>{tr("Unload")}</Btn>{tr(onReady ? <Btn variant="primary" onClick={onReady}>{tr("Start chatting")}</Btn> : <span className="native-ready">{tr("Ready")}</span>)}</> : onReady && native?.instances.some(i => i.status === 'ready') ? <Btn variant="primary" onClick={onReady}>{tr("Start chatting")}</Btn> : <><Btn icon="gauge" disabled={listPending || !installed?.ready || !native?.available || !!active || !selectedSize || selectedSize.needsNewerEngine} onClick={() => setTuneOpen(true)}>{tr("Tune model")}</Btn><Btn variant="primary" icon="models" disabled={pending || listPending || !native?.available || !!active || !selectedSize || !!selectedSize.needsNewerEngine} onClick={() => void load()}>{tr(pending || listPending ? 'Checking…' : installed?.ready ? 'Load model' : 'Download & load')}</Btn></>)}</div>
    </section>
    {tr(expert && IMAGE_GENERATION_VISIBLE ? <><details className="native-generation"><summary>{tr("Image generation ")}<span className="muted">{tr("Optional")}</span></summary><RunImageModel library={imageLibrary} disabled={pending || !!active} onChange={() => { }}/></details></> : null)}
    {tr(expert ? <section id="native-expert-settings" ref={expertSection} className="st-card native-execution"><h2 className="h4">{tr("Expert settings")}</h2>{tr(tunedNow ? <p className="small muted native-tuned-note">{tr("Showing the values of your tuning '{0}'.", [tunedNow.name])}</p> : null)}<label className="native-toggle"><span><strong>{tr("Run multiple models")}</strong><small>{tr("Loading another model keeps the others loaded.")}</small></span><button className="st-toggle" role="switch" aria-label={tr("Run multiple models")} aria-checked={multiple} onClick={() => setMultiple(v => !v)}/></label>{tr(cardsGpus.length > 1 ? <label className="input-field"><span>{tr("Layer split")}</span><input className="st-input" value={split} placeholder={tr("auto or 16,32")} onChange={e => setSplit(e.target.value)}/></label> : null)}<div className="native-field-grid"><label className="input-field"><span>{tr("Context memory")}</span><Select ariaLabel={tr("Native KV format")} value={kv} options={KV_CHOICES.map(c => ({ value: c.value, label: c.label, title: c.hint }))} onChange={setKv}/>{tr(KV_CHOICES.find(c => c.value === kv)?.hint ? <small className="muted native-kv-hint">{tr(KV_CHOICES.find(c => c.value === kv)!.hint)}</small> : null)}</label><label className="input-field"><span>{tr("Low-RAM mode")}</span><Select ariaLabel={tr("Native low RAM mode")} value={lowRam} options={['auto', 'resident', 'mmap'].map(v => ({ value: v, label: v === 'auto' ? 'Automatic' : v === 'resident' ? 'Keep experts in RAM' : 'Read experts from disk' }))} onChange={setLowRam}/></label></div>{tr(installed ? <><NativeSettingsFile model={installed.model}/><NativeSettings model={installed.model}/></> : <p className="small muted">{tr("More engine parameters become available after preparing this version.")}</p>)}</section> : null)}
   </div>
   <aside className="stack stack--lg"><PcComputer gpuName={pcGpu?.name} ramGb={pcRam} vramGb={pcVram} vramAllGb={pcGpus.reduce((n, g) => n + g.vram_gb, 0) || undefined} loaded={loadedModels} selectedLoaded={!!selectedSize && loadedModels.some(m => m.id === selectedSize.id)} diskGb={selectedSize?.multi_file ? selectedSize.disk_gb ?? selectedSize.download_gb : null} inUse={keptModels.length ? { ramGb: Math.round(keptRam), vramGb: Math.round(keptVramOn(chosenCards)) } : null} fit={selectedSize ? fitOn(selectedSize, labelHardware, true, { context, kv }) : null}/><section className="st-card"><h2 className="h4">{tr("Model files")}</h2><code className="native-path">{installed?.dataFolder ?? native?.downloadDirectory ?? native?.installed?.data_folder ?? 'Choose an installation'}</code><div className="model-files__actions"><Btn size="sm" variant="ghost" onClick={() => { setSettingsSection('Storage'); go('settings'); }}>{tr("Storage settings")}</Btn><Btn size="sm" variant="ghost" onClick={onOtherModels} className="other-models">{tr("Other local models\u2026")}</Btn></div></section></aside>
  </div>
  
  <NativeDeleteDialog id={deleting} onClose={() => setDeleting(null)}/>
  <Dialog open={manage} compact title={tr("Manage models")} onClose={() => setManage(false)} actions={<Btn onClick={() => setManage(false)}>{tr("Done")}</Btn>}><div className="native-manage">
   {tr(downloaded.some(m => m.ready || m.diskGb) ? <div className="native-manage-list" role="list">{tr(downloaded.filter(m => m.ready || m.diskGb).map(m => { const v = allSizes.find(x => x.id === m.model), loaded = native?.instances.some(i => i.model === m.model && i.status === 'ready'); return <div className="native-manage-row" role="listitem" key={m.model} data-selected={selectedSize?.id === m.model || undefined}>
     <span className="native-manage-name"><strong>{tr(v ? `${catalogVersionLabel(v)}` : m.model_name ?? m.model)}</strong><small>{tr(v ? v.model : m.model)}{tr(m.diskGb ? ` · ${sizeGb(m.diskGb)}` : '')}</small></span>
     <span className={loaded ? 'st-pill st-pill--accent' : 'st-pill'}>{tr(loaded ? 'Loaded' : 'Downloaded')}</span>
     <span className="native-manage-actions"><Btn size="sm" disabled={!!active || !v} onClick={() => { if (v) setSize(v.id); setManage(false); }}>{tr(selectedSize?.id === m.model ? 'Selected' : 'Use version')}</Btn><Btn size="sm" variant="danger" disabled={!!active} onClick={() => setDeleting(m.model)}>{tr("Delete")}</Btn></span>
    </div>; }))}</div> : <p className="small muted native-manage-empty">{tr("No downloaded models")}</p>)}
   <div className="native-manage-foot"><Btn variant="ghost" icon="folder" onClick={() => { setManage(false); (onImport ?? onOtherModels)(); }}>{tr("Use a local GGUF…")}</Btn></div>
  </div></Dialog>
 {selectedSize ? <NativeTune open={tuneOpen} onClose={() => setTuneOpen(false)} model={selectedSize.id} resourceProfile={resourceProfile} vision={selectedSize.images ? vision : 'no'} visionUnavailable={noReader}/> : null}
 </Screen>;
}
/** setup's "5/7: downloading …" as a sentence a person can read. */
/** The listed context sizes, plus the model's own when it is a size set by hand in its run config (the select would show nothing, and a load would silently pick another). */
export function contextChoices(listed: number[], current: number): number[] {
    return current > 0 && !listed.includes(current) ? [...listed, current].sort((a, b) => a - b) : listed;
}
function installStepLabel(step?: string): string {
    const n = Number(step?.match(/^(\d+)\//)?.[1]);
    if (!n)
        return step ? 'Getting things ready…' : 'Downloading and preparing model files…';
    return n <= 2 ? 'Checking your computer…' : n <= 4 ? 'Getting things ready…' : n === 5 ? 'Downloading your model…' : n === 6 ? 'Preparing your model for Strata…' : 'Finishing up…';
}
/** The formats of the conversation cache and what each costs, in one line (the memory figures are from docs/DETAILS.md: Q4 half of 8-bit, K8/V4 23% less). */
const KV_CHOICES = [
    { value: 'int8', label: 'INT8', hint: 'INT8 is the standard: good quality and a medium amount of memory.' },
    { value: 'q4_0', label: 'Q4', hint: 'Q4 uses about half the memory of INT8, but is less precise on very long documents.' },
    { value: 'k8v4', label: 'K8 / V4', hint: 'K8 / V4 uses about a quarter less memory than INT8 and is more precise than Q4. Prompts are 2 to 5% slower.' },
];
/**
 * What a running job is doing, and nothing else: one line with the step, one small line with the numbers when there are any, a thin bar, and Stop as a
 * plain word. A soft ring breathes beside the step (transform and opacity only; still under reduced motion). The bar fills with the real percent, or
 * glides while no percent is known.
 */
export function NativeProgress({ job, label }: {
    job: NativeJob;
    label?: string;
}) {
    const download = installProgress(job.result?.install), detailsText = [job.details, job.result?.install?.engine?.details].filter(Boolean).join('\n');
    const eta = useDownloadEta(job.kind === 'install' ? download : undefined, job.id);
    const [stopping, setStopping] = useState(false), [error, setError] = useState('');
    async function stop() { setStopping(true); setError(''); try {
        await nativeAction(`operations/${job.id}/cancel`);
    }
    catch (e) {
        setError(e instanceof Error ? e.message : String(e));
    }
    finally {
        setStopping(false);
    } }
    const running = job.state === 'running', stoppable = ['install', 'load', 'pictures', 'autotune', 'recheck'].includes(job.kind);
    const kindTitle = job.kind === 'install' ? (label ? tr('Preparing {0}', [label]) : 'Preparing your model') : job.kind === 'load' ? 'Loading model' : job.kind === 'pictures' ? 'Downloading the picture files' : job.kind === 'autotune' || job.kind === 'recheck' ? 'Tuning your model' : 'Engine operation';
    const step = job.kind === 'install' ? (job.result?.install?.label ?? (download && /^5\//.test(job.result?.install?.step ?? '') ? stageText(installStage(job.result?.install)) : installStepLabel(job.result?.install?.step))) : job.summary;
    const percent = download && Number.isFinite(download.percent) ? Math.max(0, Math.min(100, Number(download.percent))) : null;
    const numbers = [percent !== null ? `${Math.round(percent)}%` : '', download && Number.isFinite(download.done_gb) && Number.isFinite(download.total_gb) ? `${download.done_gb!.toFixed(1)} / ${download.total_gb!.toFixed(1)} GB` : '', eta].filter(Boolean).join(' · ');
    const words = tr(step || kindTitle);
    return <section className="st-card native-progress" aria-live="polite" data-state={job.state}>
        <div className="native-progress__row">
            <span className="native-progress__ring" aria-hidden="true"/>
            <div className="native-progress__text"><p className="native-progress__step">{words}</p>{numbers ? <p className="native-progress__numbers">{numbers}</p> : null}</div>
            {stoppable && running ? <button type="button" className="native-progress__stop" disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop')}</button> : null}
        </div>
        {running ? <div className="native-progress__bar" role="progressbar" aria-label={words} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} data-indeterminate={percent === null || undefined}><i style={percent !== null ? { transform: `scaleX(${percent / 100})` } : undefined}/></div> : null}
        <DownloadParts parts={job.result?.install?.parts}/>
        {detailsText ? <details><summary className="small">{tr("Details")}</summary><pre className="small muted error-detail-text">{detailsText}</pre></details> : null}
        {error ? <ErrorNotice message={error}/> : null}
    </section>;
}
