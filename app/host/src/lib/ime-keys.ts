/**
 * The #5071 rule for the app's other text fields: a key that belongs to the IME (the Enter that commits a Japanese, Chinese or
 * Korean conversion, in the order each engine delivers it) does not run the field's action. The composer (screens/Chat.tsx)
 * uses the same guard (lib/ime-enter.ts).
 *
 *   const ime = useImeKeys();
 *   <input {...ime.events} onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter') save(); }}/>
 *
 * One guard per component: only the focused field composes, so fields of one component can share it.
 */
import { useRef } from 'react';
import type { CompositionEvent, KeyboardEvent } from 'react';
import { createImeGuard } from './ime-enter';

export interface ImeKeys {
    /** Spread on the field: they tell the guard when a conversion starts and ends. */
    events: {
        onCompositionStart: () => void;
        onCompositionEnd: (e: CompositionEvent<Element>) => void;
    };
    /** True when this keydown is the IME's: the handler returns before it looks at the key. */
    owns: (e: KeyboardEvent<Element>) => boolean;
}

export function imeKeys(): ImeKeys {
    const guard = createImeGuard();
    return {
        events: {
            onCompositionStart: () => guard.compositionStart(),
            onCompositionEnd: e => guard.compositionEnd(e.timeStamp),
        },
        owns: e => guard.keyDown({ key: e.key, isComposing: e.nativeEvent.isComposing, keyCode: e.nativeEvent.keyCode }, e.timeStamp),
    };
}

export function useImeKeys(): ImeKeys {
    const ref = useRef<ImeKeys | null>(null);
    ref.current ??= imeKeys();
    return ref.current;
}
