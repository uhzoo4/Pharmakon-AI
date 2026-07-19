# Taste Skill dial reading — VulnSentry AI

```
DESIGN_VARIANCE: 4   (1 = Perfect Symmetry, 10 = Artsy Chaos)
MOTION_INTENSITY: 5  (1 = Static, 10 = Cinematic / Physics)
VISUAL_DENSITY: 7    (1 = Art Gallery / Airy, 10 = Cockpit / Packed Data)
```

## Why these numbers
No single row in taste-skill's Dial Inference table (§1.A) is "security
dashboard," so this blends the two closest:
- `"trust-first / public-sector / regulated / accessibility-critical"` →
  VARIANCE 3-4, MOTION 2-3, DENSITY 4-5 — matches the trust/legibility
  requirement from the Impeccable PRODUCT.md draft almost exactly
- A live-scanning dashboard reasonably pushes DENSITY above that range
  (real scan results, severity data, multiple concurrent findings) and
  MOTION slightly above it too — but only for legitimate purposes: new
  results streaming in, severity state changing — not decorative motion

This lands at 4/5/7: low variance (predictable, scannable layout), a bit
more motion than a pure trust-first site (justified by the SSE stream
needing to visually announce new data), high density (it's a real scan
dashboard, not a marketing page).

## What this actually gates, per the skill's own rules
- `DESIGN_VARIANCE: 4` sits right at the ANTI-CENTER BIAS threshold
  (rule fires above 4) — meaning this is intentionally on the "regular,
  centered layout is fine" side of that line, which is the right call
  for a dashboard that needs to be scanned quickly, not explored.
- `VISUAL_DENSITY: 7` crosses the skill's `>7`... — actually sits just
  under it. If real usage pushes this to 8 (many concurrent findings,
  dense tables), the skill's rule kicks in: **generic card containers are
  banned above DENSITY 7** — metrics should breathe in a plain layout
  rather than every finding getting boxed in its own card. Worth
  reassessing once real scan-result volume is known.
- `MOTION_INTENSITY: 5` clears the "motion claimed, motion shown" and
  `prefers-reduced-motion` thresholds (both fire above 3-4) — same
  practical requirement as Icarus Papers, lower stakes given less motion
  overall, but not exempt.

## One useful gut-check
This is a case where taste-skill's dials and the earlier ui-ux-pro-max
color findings agree from two independent tools: both pushed away from
the "cybersecurity = neon hacker terminal" default and toward something
calmer and more legible. Two unrelated tools converging on the same
correction is a reasonably strong signal, not just one opinion.
