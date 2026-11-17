import { tr } from '../lib/i18n';
import { GuidedSetup } from '../components/GuidedSetup';
import { useNative } from '../lib/native';
import { useStore } from '../lib/store';
import { useEngineUnpacking } from '../lib/engine-setup';
import { EngineFirstRun, useEngineNeeded } from '../components/EngineFirstRun';
import { useEffect, useState } from 'react';
/** Every setup entry uses the same guided onboarding. */
export function SetupScreen() {
    const native = useNative(), setup = useStore(s => s.setup);
    const needsEngine = useEngineNeeded(), [engineStepDone, setEngineStepDone] = useState(false), [cardShown, setCardShown] = useState(false);
    useEffect(() => { if (needsEngine) setCardShown(true); }, [needsEngine]);
    const unpacking = useEngineUnpacking(!native || !setup);
    // No engine on this PC yet: the first-run card comes before the model steps (it works before the native snapshot has connected).
    if (!engineStepDone && !setup?.complete && (needsEngine || cardShown))
        return <EngineFirstRun onDone={() => setEngineStepDone(true)}/>;
    if (!native || !setup)
        return <div className="setup guided-onboarding"><p className="onboard-loading" role="status">{tr("Getting your setup ready\u2026")}</p>{unpacking ? <p className="onboard-loading small" role="status">{tr("Unpacking the engine that came with the app. This takes about a minute the first time.")}</p> : null}</div>;
    return <GuidedSetup />;
}
