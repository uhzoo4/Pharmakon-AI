# Calibration notes — Icarus Papers

Generated with:
```
search.py "portfolio storytelling mythology cinematic editorial dark" --design-system -p "Icarus Papers" --motion 9 --variance 8
```

## Where the auto-run missed
The reasoning engine matched "portfolio storytelling" to its generic
**Portfolio** product type and picked **Bento Grid** — a light-background
(`#FAFAFA`), card-grid style. That's a reasonable default for a typical
portfolio site, but it's the wrong read for a dark, cinematic, myth-driven
scroll narrative. `MASTER.md` still contains this raw first-pass output
so you can see exactly what the automated reasoning does with these
keywords — but don't use its Style/Color section for this project.

## Closer match, from the same real database
A direct, more targeted style search (`--domain style "cinematic dark
mythology editorial film"`) surfaced two entries that fit far better:

**Parallax Storytelling** (pattern)
- Scroll-driven, layered, progressive disclosure, cinematic transitions
- Listed at **GSAP ScrollTrigger 10/10** compatibility — this is the
  pattern this database considers a direct match for your actual stack
- Honest tradeoff the entry itself flags: Performance ❌ Poor,
  Accessibility ❌ Poor (motion) — worth deliberately deciding your
  reduced-motion fallback rather than skipping it

**Modern Dark (Cinema Mobile)** (style/color)
- Gradient background `#0a0a0f → #020203`, elevated surface `#0a0a0c`,
  indigo glow accent `#5E6AD2`, hairline borders at 8% white opacity
- Mood keywords: dark mode, cinematic, ambient light, glow, atmospheric,
  premium, layered — much closer to an Icarus-myth tone than the
  Bento Grid default

Treat these two as the recommended override for this project's Style/Color
section; everything else in `MASTER.md` (typography, checklist) still
applies.

## Supplementary files (targeted searches, same visit)
- **`gsap-motion-reference.md`** — scroll-pin/scrub timeline, SplitText
  headline stagger, and a directly relevant note: pin-based
  ScrollTrigger sections need a deterministic height and a
  `ScrollTrigger.refresh()` call after images/fonts load — worth checking
  against the pinned-scene position bug from your last debugging pass.
- **`threejs-stack-reference.md`** — four Three.js rules, including one
  that reads like it could be exactly your mesh-blending bug:
  **`material.opacity` only tweens correctly if `transparent: true` is set
  on the material before the tween starts.** If blending looked broken or
  inconsistent across scenes, check every material transitioning opacity
  has that flag set ahead of time.

## Honest caveat
The palette/typography above are grounded, real database entries — not a
restatement of whatever's already coded into your eight scene files. Cross-
check before treating any hex value as final.
