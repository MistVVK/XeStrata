import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/kit/tokens.css';
import './styles/kit/components.css';
import './styles/app.css';
import { App } from './app/App';
import { installDropGuard } from './components/DropZone';
import { startRecovery } from './lib/recovery';
import { translationsReady } from './lib/i18n';
import { loadHighlighter } from './components/SyntaxCode';

const el = document.getElementById('root');
if (!el) throw new Error('#root missing');

installDropGuard();
startRecovery();

// The saved language's strings are fetched while the rest of the code loads; the first screen is drawn once they are in (English needs no wait).
// Code colouring (the Prism grammars, 130 kB) is not needed to show the first screen: it is fetched a few seconds later, when the browser is idle,
// so the first code block of the session is already coloured when it appears. A block that appears sooner fetches it itself.
window.setTimeout(() => {
  const idle: (run: () => void) => void = 'requestIdleCallback' in window ? (run) => window.requestIdleCallback(run, { timeout: 4000 }) : (run) => window.setTimeout(run, 0);
  idle(() => { void loadHighlighter(); void import('./components/TimeSkipPanel'); void import('./components/TimeSkipWorking'); });
}, 3000);

void translationsReady().then(() => {
  createRoot(el).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});

import './styles/annotation-fixes.css';
import './styles/monitor.css';
import './styles/time-skip.css';

import './styles/model-run.css';

import './styles/onboarding.css';

import './styles/board.css';

import './styles/gpu-roles.css';

import './styles/engine-tab.css';

// Loaded last: the shared depth, material and interaction treatment. See the
// header of the file for the order of the layers and why it sits here.
import './styles/liquid.css';
