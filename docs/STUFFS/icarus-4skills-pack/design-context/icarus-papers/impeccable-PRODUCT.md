# Product

> Draft hypothesis, not a finished PRODUCT.md. Impeccable's own init.md is
> explicit that this file should come from a codebase crawl + a real
> interview (Steps 2-3), and warns against synthesizing it from a prompt
> alone. I don't have your actual repo open, so treat this as the Step-2
> "register hypothesis" — run `/impeccable init` for real inside the
> project and use this only as a starting answer to confirm or correct.

## Register

brand

## Users

Visitors arriving from a portfolio link, Awwwards-style showcase, or direct
share — evaluating craft and technical range in the first 10-20 seconds of
scroll. Likely other developers/designers, not a general consumer audience.

## Product Purpose

A cinematic, scroll-driven retelling of the Icarus myth built to demonstrate
technical and visual craft (GSAP + Three.js + Lenis) as a portfolio
centerpiece — not a product with a conversion goal, a narrative experience
with an implicit "look what I can build" goal.

## Brand Personality

Mythic, precise, atmospheric. Three words: cinematic / mythological /
exacting. Emotional goal: awe building to tension building to catharsis
across the 8 scenes, mirroring the myth's own arc (ascent → warning →
hubris → fall).

## Anti-references

Generic scroll-jacking marketing sites (SaaS product tours with parallax
bolted on for its own sake); stock-photo "Greek mythology" clip-art
aesthetics; motion for motion's sake — every scene transition should read
as a story beat, not a demo of what ScrollTrigger can do.

## Design Principles

- Every scroll interaction is a narrative beat, not a decoration — if you
  can't name what story point a transition serves, cut it
- Tonal consistency across all 8 scenes takes priority over any single
  scene being individually flashier
- Motion is expensive: each new effect should earn its performance cost
  against the mobile/mid-tier-device experience, not just desktop
- The myth's emotional arc (ascent, warning, hubris, fall) should be
  legible in the pacing even with the sound off and no narration text

## Accessibility & Inclusion

Motion-heavy by design (pinned ScrollTrigger sections, particle systems) —
`prefers-reduced-motion` needs a real fallback path, not just disabled
easing, since several scenes rely on scroll-position-driven state that
has no meaning as a static frame. Worth deciding this deliberately rather
than late.
