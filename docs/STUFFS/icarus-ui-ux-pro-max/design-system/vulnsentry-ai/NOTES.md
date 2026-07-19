# Calibration notes — VulnSentry AI

Generated with:
```
search.py "cybersecurity platform network scanner realtime dashboard" --design-system -p "VulnSentry AI" --density 7
```

## Where the auto-run missed
Both the auto `--design-system` run and a direct `--domain color` lookup
for `"Cybersecurity Platform"` return the exact same palette: matrix green
`#00FF41` on pure black with a red alert accent. That's not a fluke — it's
the database's single dedicated "Cybersecurity Platform" entry, and it's
a "hacker terminal" cliché (Matrix/terminal green), not the dark-navy/teal
identity you actually built. `MASTER.md` keeps this raw output for
reference, but its Color section isn't the right fit here.

The **Pattern** section (App Store Style Landing: screenshots carousel,
download CTAs) also doesn't apply — VulnSentry is a scanner dashboard, not
a mobile app-store listing. Skip that section too.

## Closer match, assembled from the same real database
No single entry in the shipped 160-palette set is dark-navy-background +
teal-accent. The closest real components, combined:
- **Structure** from the "Financial Dashboard" entry: background `#020617`,
  primary/card `#0F172A`, secondary `#1E293B` — genuinely dark navy, not a
  pure-black terminal
- **Accent** from the "Real Estate/Property" entry's trust-teal:
  `#14B8A6` (or `#2DD4BF` for a lighter variant) in place of that entry's
  green

This is **my assembly of two real entries**, not a single verbatim tool
result — flagging that distinction explicitly so it doesn't read as more
authoritative than it is. Worth comparing directly against whatever
tokens are already in your Tailwind config.

## Supplementary files (targeted searches, same visit)
- **`chart-reference.md`** — Real-Time Streaming and Anomaly Detection
  chart types both map directly onto SSE-driven live scan results;
  Anomaly Detection's accessibility note (shape marker + text annotation,
  not color-only) is directly relevant to a severity-coded scanner.
- **`react-stack-reference.md`** — general React 18 performance/a11y
  notes. The most relevant one for an SSE feed: dynamic content needs
  `aria-live` regions so screen readers announce new scan results as they
  stream in, not just silent DOM updates.

## Honest caveat
As with the other two projects, treat the assembled palette as a grounded
starting point to diff against your actual `tailwind.config` / CSS
variables — not a claim that this is the exact scheme already shipped in
VulnSentry's frontend.
