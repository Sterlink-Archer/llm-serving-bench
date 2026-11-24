import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import * as Tooltip from '@radix-ui/react-tooltip';
import { RouterProvider, createHashHistory, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import './index.css';
import { ThemeProvider } from './lib/theme';
import { Layout } from './layout';
import { Overview } from './pages/overview';
import { Journey } from './pages/journey';
import { Cost } from './pages/cost';
import { Goodput } from './pages/goodput';
import { Latency } from './pages/latency';
import { RunDetail, Runs } from './pages/runs';
import { Compare, Mtp, SourcesPage, Spend, Startup } from './pages/misc';
import { Paired } from './pages/paired';
import { Quality } from './pages/quality';

const root = createRootRoute({ component: Layout });
const r = (path: string, component: () => React.JSX.Element) => createRoute({ getParentRoute: () => root, path, component });
const routeTree = root.addChildren([
  r('/', Overview), r('/journey', Journey), r('/paired', Paired), r('/quality', Quality), r('/cost', Cost), r('/goodput', Goodput), r('/latency', Latency), r('/runs', Runs),
  createRoute({ getParentRoute: () => root, path: '/run/$runId', component: RunDetail }),
  r('/spend', Spend), r('/startup', Startup), r('/mtp', Mtp), r('/compare', Compare), r('/sources', SourcesPage),
]);
// hash history: the site works from file:// and any static server with no rewrites
const router = createRouter({ routeTree, history: createHashHistory(), defaultPreload: false, scrollRestoration: true });
declare module '@tanstack/react-router' { interface Register { router: typeof router } }

const el = document.getElementById('root')!;
if (!window.BENCH) {
  el.textContent = 'data.js missing: run python3 -I dashboard/build.py';
} else {
  createRoot(el).render(
    <StrictMode>
      <ThemeProvider>
        <Tooltip.Provider delayDuration={150}>
          <RouterProvider router={router} />
        </Tooltip.Provider>
      </ThemeProvider>
    </StrictMode>,
  );
}
