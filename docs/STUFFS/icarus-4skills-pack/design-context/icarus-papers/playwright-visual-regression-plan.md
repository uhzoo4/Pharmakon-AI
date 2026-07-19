# Icarus Papers — playwright-cli visual regression plan

Not a "theme" file like the others — Playwright doesn't have an aesthetic
opinion. This is a concrete starting plan for using the real
`playwright-cli` skill against your 8-scene scroll experience, using
commands straight from `skills/playwright-cli/SKILL.md` and
`references/tracing.md` / `references/video-recording.md`.

## Why this project specifically needs it
Scroll-pinned GSAP ScrollTrigger sections are exactly the kind of thing
that silently breaks: a pin height becomes non-deterministic after a copy
change, a font swap shifts layout before `ScrollTrigger.refresh()` fires,
a Three.js scene fails to init on a slower GPU and scene 4 just never
appears. None of that throws a JS error you'd notice in normal dev — it
just quietly looks wrong. That's a screenshot-diffing problem, not a
console-log problem.

## Baseline capture (run once, commit the output)
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli resize 1440 900
# scroll to each scene boundary and snapshot — repeat per scene
playwright-cli eval "window.scrollTo(0, document.body.scrollHeight * (1/8))"
playwright-cli screenshot --filename=baseline/scene-01.png --hires
# ...repeat for scenes 2-8 at their scroll fractions
playwright-cli close
```

## Regression check (after any scene/shader/GSAP change)
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli resize 1440 900
playwright-cli eval "window.scrollTo(0, document.body.scrollHeight * (1/8))"
playwright-cli screenshot --filename=current/scene-01.png --hires
# ...repeat per scene, then diff current/ against baseline/ with your
# image-diff tool of choice (pixelmatch, odiff, etc. — not part of this skill)
```

## Catching the specific pin/refresh bug class
Straight from `references/tracing.md`'s pattern — trace a full scroll pass
and inspect for layout thrash or dropped frames around pin boundaries:
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli tracing-start
playwright-cli eval "window.scrollTo(0, document.body.scrollHeight)"
playwright-cli tracing-stop
```

## Recording a full scroll-through as video (for the portfolio itself, or for review)
Per `references/video-recording.md`'s workflow:
```bash
playwright-cli open http://localhost:<your-dev-port>
playwright-cli video-start scroll-through.webm
playwright-cli video-chapter "Scene 1 — Ascent" --duration=3000
playwright-cli eval "window.scrollTo(0, document.body.scrollHeight * (1/8))"
# ...chapter + scroll per scene
playwright-cli video-stop
```

## Honest caveat
None of this has actually been run — I don't have your dev server
reachable from this sandbox. These are real commands from the real skill,
adapted to your project's structure as I understand it (8 scenes,
scroll-fraction navigation), not a verified working script. Swap in your
actual dev port and confirm the scroll-fraction math matches how your
scenes are actually laid out before trusting the diffs.
