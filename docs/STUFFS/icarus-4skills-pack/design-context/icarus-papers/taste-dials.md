# Taste Skill dial reading — Icarus Papers

Using taste-skill's actual three-dial framework (`skills/taste-skill/SKILL.md`
§1) rather than inventing a new one.

```
DESIGN_VARIANCE: 9   (1 = Perfect Symmetry, 10 = Artsy Chaos)
MOTION_INTENSITY: 9  (1 = Static, 10 = Cinematic / Physics)
VISUAL_DENSITY: 3    (1 = Art Gallery / Airy, 10 = Cockpit / Packed Data)
```

## Why these numbers
The skill's own Dial Inference table (§1.A) has two rows that both apply
here and both point the same direction:
- `"playful / wild / Dribbble / Awwwards / experimental / agency"` →
  VARIANCE 9-10, MOTION 8-10, DENSITY 3-4
- `"landing page / portfolio / marketing site (default)"` → VARIANCE 7-9,
  MOTION 6-8, DENSITY 3-5

Icarus Papers sits at the top of both ranges rather than the middle —
Awwwards-level ambition plus genuinely heavy GSAP/Three.js scene work — so
9/9/3 instead of a more conservative 7/6/4.

## What this actually gates, per the skill's own rules
- `DESIGN_VARIANCE: 9` triggers the skill's **ANTI-CENTER BIAS** rule
  (>4 = centered hero sections are avoided by default; the skill pushes
  toward split-screen, asymmetric whitespace, or scroll-pinned structures
  instead) — worth checking your hero scene against this on purpose,
  not by accident.
- `MOTION_INTENSITY: 9` crosses the skill's own "motion claimed, motion
  shown" threshold (>4): the rule is explicit that a page claiming this
  dial has to actually move on entry, on scroll, and on hover — and
  that half-built motion (cut-off ScrollTriggers, jumpy enters, missing
  cleanup) is treated as a defect, not a rough edge to fix later.
- Both `MOTION_INTENSITY: 9` and the general `>3` threshold mean
  `prefers-reduced-motion` is non-negotiable per the skill's a11y rule —
  same flag as the Impeccable PRODUCT.md draft in this folder, from an
  independent source.

## One useful gut-check
The skill frames `VISUAL_DENSITY` low-end as "Art Gallery" — worth holding
onto deliberately. A mythological narrative piece has real gravity toward
adding more (more particles, more UI chrome, more indicators of scroll
progress); the 3 is doing real work by keeping each scene reading as a
gallery piece rather than a dashboard of what's happening technically.
