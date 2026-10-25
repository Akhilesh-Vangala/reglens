# RegLens web console

`public/` is the console FastAPI serves at `/` and `/static`: plain HTML, CSS and JavaScript with no build step, so it
runs under a strict Content-Security-Policy (no inline script). `e2e/` drives it in Chromium against the real server.

| Path | What it is |
| --- | --- |
| `public/index.html` | Seven tabs: copilot, deadline calculator, checker, citation graph, Q&A pipeline, results, architecture |
| `public/app.js` | Hash router with per-tab URL state, SSE copilot client, recorded-run replay, calculator, checker, graph |
| `public/app.css` | Design tokens for light and dark themes, components, responsive rules |
| `public/theme.js` | Applies a saved theme before first paint |
| `e2e/` | Playwright flows plus an axe WCAG 2 A/AA scan of every tab in both themes |

```bash
cd e2e && npm ci && npx playwright install chromium && npx playwright test
```
