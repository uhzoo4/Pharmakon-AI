# VulnSentry AI — playwright-cli E2E test plan

## Why this project specifically needs it
An SSE-driven dashboard has a failure mode unit tests don't catch: the
backend scan logic can be perfectly correct while the frontend silently
stops rendering new events (a dropped EventSource connection, a React
state update that doesn't re-render, a severity color that resolves to
`undefined`). That's only visible by actually driving a browser through
a real scan.

## Smoke test: scan lifecycle
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli snapshot
# trigger a scan the way a real user would
playwright-cli click <scan-button-ref>
# poll — new SSE events should change the snapshot over time
playwright-cli snapshot
playwright-cli snapshot
playwright-cli close
```

## Checking the live-region / accessibility fix from the design notes
Straight from `references/element-attributes.md`'s pattern — confirm the
`aria-live` region actually exists and updates, not just that it's present
in the source:
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli click <scan-button-ref>
playwright-cli eval "document.querySelector('[aria-live]')?.textContent"
# re-run after a beat to confirm the text actually changes as results stream in
playwright-cli eval "document.querySelector('[aria-live]')?.textContent"
```

## Network inspection during a live scan
Per `references/request-mocking.md` and the SKILL.md's DevTools section —
useful both for confirming the SSE stream is open and for testing how the
UI behaves if Gemini's explanation call is slow or fails:
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli click <scan-button-ref>
playwright-cli requests
# simulate a slow/failed explanation call to check the UI's fallback state
playwright-cli route "**/api/explain**" --status=504
```

## Severity color regression (ties back to the design-context findings)
Since the design notes call for severity to be legible via shape/icon +
text, not color alone — a concrete check rather than just a visual eyeball:
```bash
playwright-cli eval "[...document.querySelectorAll('[data-severity]')].map(el => ({ severity: el.dataset.severity, hasIcon: !!el.querySelector('svg, [data-icon]'), hasText: !!el.textContent.trim() }))"
```
Flags any severity badge that's relying on color alone.

## Honest caveat
Same as the Icarus Papers plan — real commands from the real skill,
not verified against your actual running app. The element refs (`<scan-
button-ref>`) and selectors (`[data-severity]`, `[aria-live]`) are
placeholders matching what the design notes assume your markup looks
like; swap in the real ones from `playwright-cli snapshot` against your
actual dev server.
