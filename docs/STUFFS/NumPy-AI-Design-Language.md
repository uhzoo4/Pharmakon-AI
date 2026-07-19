# NumPy AI — Design Language Specification

**Version 1.0 — July 2026**
**Sources synthesized:** *theme.md* (personal aesthetic profile, "Intelligent Darkness"), the *NumPy AI* product brief, and *The Pharmakon Spec v3.1* (project reference, execution style)

---

## Synthesis Note — How This Document Was Built

These three sources were not averaged. They were read as one design DNA with a recurring argument running underneath them, and one real conflict that had to be resolved before anything else could be built.

**The argument, stated once, that all three sources are making:** restraint is not the absence of intensity — it's intensity under control. *theme.md* wants calm and dangerous in the same object. The Pharmakon Spec wants a machine "trained on the darkest literature humanity ever produced" governed by finite-difference proofs accurate to 1e-5. The NumPy AI brief wants precision, calm, intelligence, and depth "without visual noise." Every source pairs an extreme with a discipline. That pairing — not darkness, not minimalism, not rigor alone — is the actual fingerprint.

**Three recurring patterns, found independently in all three documents:**
- **Process as identity.** The Pharmakon Spec is structured as numbered phases with validation gates. The NumPy AI brief asks for progressive disclosure and predictive workflows. *theme.md* singles out "architecture diagrams" as something genuinely enjoyed, not tolerated. NumPy AI should look like it was *built in phases*, because it was, and the interface should let you see that.
- **Complexity respected, not hidden.** The Pharmakon Spec's entire thesis is that nothing is hidden — every gradient is shown. *theme.md* wants a desktop that reads "someone who builds systems lives here." The brief calls for "invisible complexity." These sound opposed. They aren't — see contradiction #2 below.
- **Motion with mass.** All three sources reject decoration for its own sake and describe good motion as physical: "elegant inertia," "the Jacobian spins with it," "weight over speed."

**Contradictions found, and how they were resolved — not split the difference, but a decision:**

1. **Genre clash.** The Pharmakon Spec's surface — skulls, blood, "grimoire," "weapon," cream paper, hand-lettered display type — directly contradicts *theme.md*'s explicit instruction: *"Not horror. Not edgy. Not neon hacker."* Resolution: the horror iconography is discarded entirely. What survives is the *seriousness* underneath it — the idea that a technical document can be treated with the gravity of scripture, that correctness is sacred, that a spec is worth staining with coffee. NumPy AI keeps the reverence and drops the genre. Where Pharmakon reaches for a skull, NumPy AI reaches for an exact diagram. That substitution — gravity without gore — is used throughout this document.
2. **"Invisible complexity" vs. "expose every gradient."** Resolved as one mechanism: **progressive disclosure.** The surface is quiet by default. The machinery is one interaction away, always, for anyone who goes looking. Nothing is hidden; nothing is forced on you either. This single rule reconciles all three sources and becomes a governing law for the rest of this spec (see §17).
3. **Maximalist voice vs. minimalist restraint.** Pharmakon narrates in accumulating, escalating prose ("Print it. Stain it with coffee and tears."). *theme.md* wants "minimal text," "few buttons," quiet confidence. Resolution: NumPy AI's intensity is expressed through *precision of a single sentence*, not volume of sentences. One exact line does what Pharmakon does in a paragraph.
4. **Warm ritual-paper texture vs. cold nocturnal palette.** Pharmakon's cream-and-red-ink presentation is rejected outright in favor of *theme.md*'s near-black surface system, which is kept close to verbatim — it was already precise.

What follows is the resulting system: something none of the three source documents would have produced alone.

---

# Part I — Foundation

## 1. Design Manifesto

Darkness is a canvas, not a mood.
Complexity is never hidden — only quieted, one interaction from view.
Every animation is a sentence. If it doesn't say something, cut it.
We do not decorate a surface. We disclose what's underneath it, on request.
Precision is the only luxury that doesn't depreciate.
NumPy AI does not perform intelligence. It demonstrates it, and shows its work.
Nothing exists in this system because "most products do it." Everything exists because it was decided.

## 2. Core Philosophy

NumPy AI is built on one generative tension: **structure emerging from the dark.** Not chaos for atmosphere, and not order for its own sake — the interface's entire visual grammar is the process of something precise assembling itself out of a void. Three pillars carry this:

- **Restraint as the primary luxury.** Every screen should look like something was removed from it, not added to it. This is inherited from the Kingsman/Leica/Nothing reference points in *theme.md*: expense reads as omission, not ornament.
- **Exposure on demand.** Nothing is a black box. Everything that computes, reasons, or transforms is inspectable — but only when asked. This is the resolved form of Pharmakon's "nothing is hidden."
- **Weight.** Every surface, panel, and transition behaves as if it has mass and inertia. Nothing in this product is weightless, bouncy, or free — that's the line that separates an instrument from a toy.

## 3. Emotional Design Goals

A person using NumPy AI should feel:

- **Quiet competence** — the sense of being in a well-run control room, not a chat window.
- **Focused momentum** — things move forward at a measured pace; nothing stalls, nothing rushes.
- **Earned trust** — trust built by visible correctness (the model shows its reasoning when asked), not by friendly affect.
- **Unhurried precision** — calm is a function of confidence, not slowness.

Explicitly *not* the goal: excitement, cuteness, playfulness, novelty, or the "delight" register common to consumer SaaS. Those are legitimate goals for other products. They are not this one.

## 4. Brand Personality

| NumPy AI is | NumPy AI is not |
|---|---|
| Exact | Cute |
| Restrained | Chatty |
| Nocturnal | Bright/energetic |
| Architectural | Decorative |
| Unhurried, but fast | Instant-feeling / flashy |
| Quietly confident | Enthusiastic |
| Transparent on request | Mysterious for effect |
| Serious about craft | Playful about craft |

## 5. Visual Identity

The fingerprint, in one paragraph a designer could recognize with the logo removed: a layered near-black surface system (never a flat single black), cool titanium and steel-blue structural accents used with total discipline, mono-spaced type used deliberately to signal "this is the machine talking," and one recurring geometric signature — **the Gnomon** — a single angled tick mark, borrowed from sundial instruments, that means "here is the precise point" wherever it appears: active navigation, focus states, the cursor in reasoning traces, the phase marker in loading sequences. One mark, one meaning, reused everywhere instead of inventing a new affordance for each. That repetition, not any single screen, is what makes it recognizable.

---

# Part II — Color & Type

## 6. Color System

Four to six named colors, not a swatch book. Every hue that exists in this system exists because nothing else could do that job.

| Token | Name | Hex | Role |
|---|---|---|---|
| `surface-0` | Void | `#050505` | App background |
| `surface-1` | Base | `#0A0A0A` | Primary content canvas |
| `surface-2` | Raised | `#101214` | Cards, panels |
| `surface-3` | Overlay | `#16181D` | Modals, popovers, glass |
| `surface-navy` | Deep Navy | `#0B1220` | Hero / marketing surfaces only |
| `text-primary` | Moon White | `#EDEEF0` | Primary text |
| `text-secondary` | Titanium | `#9DA3AC` | Secondary text, labels |
| `text-tertiary` | Slate | `#6B7078` | Large/decorative text only |
| `border-hairline` | — | `rgba(255,255,255,0.08)` | 1px separators |
| `border-visible` | Gunmetal | `#3A3D42` | Emphasized borders |
| `accent-steel` | Steel Blue | `#5C7A94` | Primary interactive accent |
| `accent-steel-bright` | — | `#7FA0B8` | Hover/active steel |
| `accent-ice` | Ice Blue | `#8ECAE6` | Focus, selection, the Gnomon mark |
| `semantic-amber` | Amber | `#D9A441` | Caution only |
| `semantic-red` | Soft Red | `#C9605A` | Destructive / error only |

**There is no green in this system, and that is a decision, not an oversight.** *theme.md* is explicit: "never rainbow," accent colors used "only in tiny amounts." Adding a success-green would be the single most common shortcut in SaaS design — and it would violate the palette's own law. Success in NumPy AI is communicated with `accent-ice` plus a glyph, never with a new hue. If a future contributor wants to introduce a color, it must first prove no existing token can carry that meaning (see §43).

## 7. Typography System

**Sans (interface text):** Geist, primary. Inter as the web-safe fallback stack. IBM Plex Sans reserved specifically for dense tabular contexts where its numeral set outperforms Geist's. SF Pro is used *only* when rendering natively inside an Apple system context — never chosen, only inherited, so the brand face stays consistent everywhere it has a choice.

**Mono (machine voice):** Geist Mono, primary — it is the literal typographic signal that "the model," not a person, is speaking: reasoning traces, code, telemetry, timestamps, tabular figures. JetBrains Mono is the fallback. Berkeley Mono is reserved for a CLI/terminal-adjacent surface only, if one ever exists — it is not a general UI font.

**Display:** no separate display family is introduced. Display treatment is Geist at its lightest variable weight (100–300), tracked +2% to +4% at large sizes, set in sentence case. Uppercase is reserved for small tracked eyebrow labels only (10–12px) — never full headlines. This is the "elegant, wide, thin, premium" instruction from *theme.md* executed as a weight-and-tracking rule rather than a new typeface, keeping the system to two families total.

| Tier | Size / Line-height | Weight | Family | Tracking |
|---|---|---|---|---|
| Caption | 12 / 16 | 400 | Geist | 0 |
| Body Small | 14 / 20 | 400 | Geist | 0 |
| Body | 16 / 24 | 400 | Geist | 0 |
| Body Large | 18 / 28 | 400–500 | Geist | 0 |
| Subhead | 21 / 28 | 500 | Geist | 0 |
| Title | 24 / 32 | 500 | Geist | 0 |
| Display S | 32 / 40 | 300 | Geist | +2% |
| Display M | 40 / 48 | 200 | Geist | +2% |
| Display L | 56 / 64 | 200 | Geist | +3% |
| Display XL | 72 / 80 | 100 | Geist | +4% |
| Data / Mono | 14 / 20 | 400 | Geist Mono | 0, tabular figures on |

---

# Part III — Space & Structure

## 8. Layout System

Content breathes. Reading columns cap at 680–760px regardless of viewport, so prose never sprawls edge to edge. Workspace and dashboard views are allowed to use the full canvas width — they are instruments, not articles. Layouts default to asymmetry over dead-centered compositions; a centered layout is a choice made for a specific reason (an empty state, a single focal action), not a default.

## 9. Grid System

A 12-column grid at desktop, sharing its base unit with the spacing scale (§10) so grid and spacing are never two competing systems. Gutters: 24px desktop, 16px tablet, 12px mobile. Breakpoints: 480 / 768 / 1024 / 1440 / 1920.

## 10. Spacing Scale

Base unit 4px, using Tailwind-compatible naming (token number ≈ px ÷ 4) so it maps cleanly into implementation without a translation layer.

| Token | Value |
|---|---|
| `space-1` | 4px |
| `space-2` | 8px |
| `space-3` | 12px |
| `space-4` | 16px |
| `space-6` | 24px |
| `space-8` | 32px |
| `space-12` | 48px |
| `space-16` | 64px |
| `space-24` | 96px |
| `space-32` | 128px |
| `space-48` | 192px |
| `space-64` | 256px |

Fine steps at the small end (4/8/12) give control inside dense, data-heavy panels. Aggressive jumps at the large end (96/128/192+) produce the "huge cards, large whitespace" instruction from *theme.md* — the scale is asymmetric on purpose.

## 11. Radius Scale

| Token | Value | Use |
|---|---|---|
| `radius-sm` | 6px | Buttons, inputs, chips |
| `radius-md` | 12px | Cards, panels |
| `radius-lg` | 20px | Modals, hero surfaces |
| `radius-full` | 9999px | Circular only — avatars, status dots, icon-only buttons |

Nothing is a pill except things that are actually circular. Pill-shaped buttons and badges are the fastest way to make a precision instrument look like a mobile game, and are excluded outright (see §44).

---

# Part IV — Depth & Surface

## 12. Shadow Philosophy

Shadows here don't exist to look pretty; they exist to report elevation truthfully. Every shadow is neutral or cool-tinted — never the default warm/brown drop-shadow — low opacity (8–16%), large soft blur radius, minimal offset, as if lit by one soft source high and slightly behind the viewer. Raised elements additionally carry a 1px rim-light along their top edge rather than relying on shadow alone: elevation is told through light arriving, not darkness pooling underneath. This is a direct, literal execution of *theme.md*'s "rim light, moon light" lighting language.

## 13. Glass System

Frosted glass is reserved for **transient** surfaces only — command palette, tooltips, context menus, short-lived modals. Persistent chrome (the left rail, primary navigation, the workspace canvas) stays fully opaque, because legibility and GPU cost both lose to blur over time. Where used: 8–16px backdrop blur, 60–80% surface opacity, always paired with a 1px hairline border, because blur without a defined edge reads as smudge rather than glass on a near-black background. "Subtle. No jelly." — *theme.md*'s own words are the acceptance test.

## 14. Surface Hierarchy

| Level | Token | Role |
|---|---|---|
| 0 | `surface-0` | App background — the void everything sits inside |
| 1 | `surface-1` | Primary content canvas |
| 2 | `surface-2` | Cards, panels — one lightness step up, optional rim-light |
| 3 | `surface-3` | Popovers, modals, glass overlays |

Elevation is always communicated by moving one step in this table — never by jumping straight to gray, and never by a color-tint shift. `surface-navy` is the one deliberate exception, reserved for hero/marketing contexts where a cooler, more saturated backdrop is earned by the occasion.

---

# Part V — Components, Navigation & Heuristics

## 15. Component Principles

Every interactive component ships with all five of its states designed together, never one at a time: resting, hover, active/pressed, focus, disabled. Hover states lighten the surface one elevation step and add rim-light — they do not scale, bounce, or tilt the element. A component that needs more than one accent color to read correctly is a component that's trying to do two jobs; split it instead.

## 16. Navigation Philosophy

Primary navigation lives in a persistent left rail, not a top bar. A top bar reads as a website; a left rail reads as an instrument panel — closer to the mission-control feeling *theme.md* describes. The active item is marked with the Gnomon tick, not a filled pill background — the generic "solid blue pill" nav-active state is one of the most templated moves in SaaS UI and is deliberately avoided here. Depth beyond the rail (e.g., inside a workspace) uses breadcrumbs, because a workspace's history really is a sequence — unlike most numbered UI, this one is earned (see §21's rationale).

## 17. Information Architecture

**Governing law: progressive disclosure.** Every view has a default state that shows conclusions, and an expanded state, one interaction away, that shows process. Nothing auto-expands without the user asking. Nothing is buried more than two levels deep. This single rule is how the "invisible complexity" vs. "expose every gradient" contradiction (see Synthesis Note) is resolved in every part of the product, not just the AI's reasoning traces.

## 18. UX Heuristics

Recognition over recall — persistent context, no modal amnesia when a panel closes and reopens. Reversibility — every destructive action has an undo path, treated with the same visual seriousness as the action it reverses (build and destroy get matched styling, never a scary one and a boring one). One primary action per view. Silence is a valid, and often correct, response — not every action needs a toast to confirm it happened.

## 19. Accessibility Standards

- Body text against any surface token must clear 4.5:1 contrast; `text-tertiary` (Slate, `#6B7078`) is restricted to large or non-essential text, where the 3:1 minimum applies — it is not approved for body copy.
- Focus is never removed. Visible focus = 2px `accent-ice` ring, 2px offset, on every focusable element without exception.
- No state is communicated by color alone — destructive actions pair `semantic-red` with an icon and a label, not a red background by itself.
- `prefers-reduced-motion` cuts all durations by roughly 70% and removes ambient/parallax motion (Drift, §20) entirely, replacing entrances with a simple opacity cross-fade.
- The system is dark-first, but a real light-mode surface stack is defined, not an inverted dark mode — inversion breaks the elevation logic in §14.
- Full keyboard operability of the left rail and command palette is a launch requirement, not a backlog item.

---

# Part VI — Motion

## 20. Motion Language

Motion reports state; it never decorates. If a transition can't answer "what changed, and why," it's cut. Per *theme.md*'s explicit rejections — no bounce, no cartoon, no rubber, no overshoot — every curve in this system is custom and monotonic, named for what it represents rather than borrowed generic easing terminology:

| Curve | Bezier | Use | Rationale |
|---|---|---|---|
| **Approach** | `cubic-bezier(0.16, 1, 0.3, 1)` | Entrances | Fast start, long soft settle — like a craft docking, never an overshoot-and-correct |
| **Departure** | `cubic-bezier(0.7, 0, 0.84, 0)` | Exits | The mirror of Approach — accelerates away cleanly |
| **Hold** | linear, 100ms | Press/tap feedback | Near-instant acknowledgment; no curve needed at this duration |
| **Drift** | `cubic-bezier(0.37, 0, 0.63, 1)` | Ambient/idle motion | Extremely slow (8–20s loops), for fog/particle backgrounds only — never on interactive elements |

## 21. Animation Timing

Duration scales with the physical size of what's moving — small things move fast, large surfaces move slow, because that's what mass actually looks like.

| Token | Duration | Use |
|---|---|---|
| `dur-instant` | 100ms | Press/tap feedback (Hold) |
| `dur-fast` | 180ms | Hover, micro-interactions |
| `dur-base` | 320ms | Dropdowns, tooltips, small component transitions |
| `dur-slow` | 480ms | Panel and modal open/close |
| `dur-cinematic` | 720ms | First-load reveals, hero moments — used at most once per view |

Numbered/staged progress indicators (§25) are the one place this system uses literal sequence markers — justified because AI reasoning genuinely happens in stages, not because numbering looks technical.

---

# Part VII — Imagery

## 22. Iconography

Monochrome linework only, 1.5px stroke on a 24px grid. No filled or duotone icon sets. The single semantic rule: **stroke = idle, fill = selected/active** — one dimension of meaning, applied consistently, rather than a color-coded icon library. No skeuomorphism.

## 23. Illustration Style

No mascots, no isometric blob-people, no gradient-mesh SaaS art. Where illustration is needed — onboarding, marketing, empty states — the reference set is literal and already defined by *theme.md*'s own wallpaper list: night-lit architecture, fog, empty hallways, long-exposure light trails, monoliths. Rendered as restrained duotone/grain graphics in the `surface-navy`/`accent-steel` range. Never literal photography of people.

---

# Part VIII — States

## 24. Empty States

An empty state is an invitation, not an apology — no sad-face illustrations, no "Nothing here yet! 🎉." NumPy AI shows the instrument at rest: a quiet diagram of what will populate the space, with exactly one next action labeled in plain language.

## 25. Loading States

Skeleton layouts for anything with a known shape — never a generic spinner where the final layout is already known. For AI generation specifically, this is the signature moment: instead of a spinner, a labeled phase sequence (e.g., *Reading → Reasoning → Drafting → Verifying*) marked with the Gnomon tick as it advances. This turns "the model is thinking" from a black box into something demonstrated — the resolved, de-occulted descendant of Pharmakon's phase structure and "nothing is hidden" ethos.

## 26. AI Interaction Patterns

The AI has no avatar, face, or mascot — its presence is the mono-typeface itself. Reasoning traces render in `Geist Mono`, dimmed by default, expandable on request (§17's law, applied to the model directly). Responses stream at a measured, even pace; no artificial staggering for "liveliness." The system is allowed to pause visibly at real decision points rather than typing at a uniform rate — a pause that means something, per §20.

## 27. Prompt Workspace UX

The input area is an instrument panel, not a search bar: generous height, visible (not hidden-until-hover) affordances for attachments, context, and tools. No placeholder text that vanishes without a trace on focus — context stays visible so the user never loses track of what they were about to ask.

## 28. Chat Experience

Conversation history is a ship's log, not a messaging app: sessions in the left rail are titled from their content, never "New chat 14," with timestamps set in small mono caption type. Message content avoids chat "bubbles" — flush-left flowing text with sender distinguished by type weight and color reads as document/instrument rather than SMS, matching the calmer register from §3.

## 29. Workspace Philosophy

The canvas where NumPy AI builds things is spatial and persistent — it remembers where the user left panels and artifacts, per the "human memory support, fast recognition over recall" heuristic from the brief. Generated artifacts dock into a fixed panel alongside the conversation rather than replacing it, so context is never lost mid-task.

## 30. Dashboard Principles

Default to the fewest metrics that answer one real question — not maximum density. Figures use tabular mono numerals. `accent-steel` is spent on the single number that matters most in a given view, not spread across every card equally. Dashboards never auto-refresh in a way that shifts layout under the user's cursor.

---

# Part IX — Feedback Systems

## 31. Error Design

Errors state exactly what happened and the one next action, in the interface's own voice — never an apology, never blame directed at the user. `semantic-red` appears only at the precise point of consequence, never as an ambient warning wash across a whole panel. Destructive confirmations are calm and exact, not alarm screens — the resolved poison/cure duality from the Synthesis Note: building and destroying get equally deliberate, unshouted treatment.

## 32. Success Design

Success is quiet: an `accent-ice` check glyph and a state change, nothing more. No confetti, no toast for minor confirmations, no green (§6). Success shouldn't require the user's attention — only their trust that it happened.

## 33. Notification Philosophy

Notifications are opt-in interruptions, ranked by whether they require action. Anything that doesn't require action lands in a persistent, low-priority log rather than a transient toast. Never more than one toast visible at a time — queue additional ones instead of stacking.

---

# Part X — Responsiveness & Engineering

## 34. Responsive Strategy

Below the tablet breakpoint, the left rail collapses into a command-first bottom pattern rather than a hamburger drawer. Data tables reflow into stacked key-value cards rather than horizontal scroll. Motion durations shorten roughly 20% on mobile — a smaller viewport reads as a smaller object, and smaller objects move faster (§21).

## 35. Performance Constraints

Backdrop blur (§13) is limited to two simultaneous surfaces on screen at once — a hard budget, not a guideline. Animations only ever touch `transform` and `opacity`; anything that would otherwise animate blur or shadow instead cross-fades between two pre-rendered states. Lists and tables beyond ~50 rows are virtualized. `prefers-reduced-data` disables ambient Drift motion and grain textures entirely.

## 36. Engineering Rules

Every color, spacing, radius, and motion value ships as a token — never a hard-coded value inside a component. A component needing a value outside the token set is a signal to propose a new token (§43), not to override inline. This is the design-system translation of the Pharmakon Spec's own governing instinct: nothing ships unvalidated against the system it's built from.

---

# Part XI — Implementation

## 37. Design Tokens (Reference Index)

All tokens defined in §6, §10, §11, §20–21, consolidated:

`surface-0/1/2/3/navy` · `text-primary/secondary/tertiary` · `border-hairline/visible` · `accent-steel/steel-bright/ice` · `semantic-amber/red` · `space-1…64` · `radius-sm/md/lg/full` · `dur-instant/fast/base/slow/cinematic` · `ease-approach/departure/hold/drift`

## 38. CSS Variables

```css
:root {
  /* Surfaces */
  --np-surface-0: #050505;
  --np-surface-1: #0A0A0A;
  --np-surface-2: #101214;
  --np-surface-3: #16181D;
  --np-surface-navy: #0B1220;

  /* Text */
  --np-text-primary: #EDEEF0;
  --np-text-secondary: #9DA3AC;
  --np-text-tertiary: #6B7078;

  /* Borders */
  --np-border-hairline: rgba(255, 255, 255, 0.08);
  --np-border-visible: #3A3D42;

  /* Accents */
  --np-accent-steel: #5C7A94;
  --np-accent-steel-bright: #7FA0B8;
  --np-accent-ice: #8ECAE6;

  /* Semantic */
  --np-semantic-amber: #D9A441;
  --np-semantic-red: #C9605A;

  /* Spacing */
  --np-space-1: 4px;  --np-space-2: 8px;  --np-space-3: 12px;
  --np-space-4: 16px; --np-space-6: 24px; --np-space-8: 32px;
  --np-space-12: 48px; --np-space-16: 64px; --np-space-24: 96px;
  --np-space-32: 128px; --np-space-48: 192px; --np-space-64: 256px;

  /* Radius */
  --np-radius-sm: 6px;
  --np-radius-md: 12px;
  --np-radius-lg: 20px;
  --np-radius-full: 9999px;

  /* Motion */
  --np-dur-instant: 100ms;
  --np-dur-fast: 180ms;
  --np-dur-base: 320ms;
  --np-dur-slow: 480ms;
  --np-dur-cinematic: 720ms;

  --np-ease-approach: cubic-bezier(0.16, 1, 0.3, 1);
  --np-ease-departure: cubic-bezier(0.7, 0, 0.84, 0);
  --np-ease-drift: cubic-bezier(0.37, 0, 0.63, 1);
}
```

## 39. Tailwind Mapping

```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        'np-surface': {
          0: '#050505', 1: '#0A0A0A', 2: '#101214', 3: '#16181D',
          navy: '#0B1220',
        },
        'np-text': {
          primary: '#EDEEF0', secondary: '#9DA3AC', tertiary: '#6B7078',
        },
        'np-border': {
          hairline: 'rgba(255,255,255,0.08)', visible: '#3A3D42',
        },
        'np-accent': {
          steel: '#5C7A94', 'steel-bright': '#7FA0B8', ice: '#8ECAE6',
        },
        'np-semantic': {
          amber: '#D9A441', red: '#C9605A',
        },
      },
      spacing: {
        '1': '4px', '2': '8px', '3': '12px', '4': '16px', '6': '24px',
        '8': '32px', '12': '48px', '16': '64px', '24': '96px',
        '32': '128px', '48': '192px', '64': '256px',
      },
      borderRadius: {
        'np-sm': '6px', 'np-md': '12px', 'np-lg': '20px',
      },
      transitionDuration: {
        'np-instant': '100ms', 'np-fast': '180ms', 'np-base': '320ms',
        'np-slow': '480ms', 'np-cinematic': '720ms',
      },
      transitionTimingFunction: {
        'np-approach': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'np-departure': 'cubic-bezier(0.7, 0, 0.84, 0)',
        'np-drift': 'cubic-bezier(0.37, 0, 0.63, 1)',
      },
    },
  },
};
```

## 40. Component Library Architecture

Component tiers are named after array dimensionality — a naming convention that is unmistakably NumPy AI's own, not borrowed from Brad Frost's atomic design vocabulary, though it maps onto it directly:

| NumPy AI tier | Dimensionality metaphor | Atomic-design equivalent | Examples |
|---|---|---|---|
| **Scalar** | A single value | Atom | Button, input, badge, icon |
| **Vector** | An ordered sequence of scalars | Molecule | Search bar, card header, form field group |
| **Matrix** | Rows and columns of vectors | Organism | Data table, panel, side rail |
| **Tensor** | A composed structure of matrices | Template / page | Full workspace layout, dashboard |

## 41. File Structure

```
/design-system
  /tokens
    colors.json
    spacing.json
    motion.json
    radius.json
  /scalars       (buttons, inputs, icons, badges)
  /vectors       (search, card-header, field-groups)
  /matrices      (tables, panels, rails)
  /tensors       (workspace, dashboard, chat layouts)
  /patterns      (empty-states, loading, error, notification)
  /docs
    this-file.md
```

## 42. Naming Convention

Components: `PascalCase` (`PromptPanel`, `PhaseIndicator`). Token/CSS variables: `kebab-case`, namespaced `--np-` to avoid collision with host-app tokens. Boolean props: `is`/`has` prefixes (`isActive`, `hasFocus`). File name matches component name exactly — no index-file indirection that hides what a folder contains.

## 43. Future Expansion Rules

Before adding anything new, it must pass a test:

- **New color:** does it serve a meaning no existing token serves? If not, reuse an existing token — this is what keeps the "never rainbow" rule real over time, not just at launch.
- **New motion curve:** is it distinguishable in purpose from Approach / Departure / Hold / Drift? A fifth curve needs a fifth *job*, not just a fifth feeling.
- **New illustration:** does it match the established night-architecture reference set (§23)? If it needs a new visual world to make sense, it's probably for a different brand.

## 44. Anti-Patterns (Never Allowed)

- Bounce, spring, or elastic easing of any kind
- Pill-shaped buttons, badges, or nav items (circular only for genuinely circular things)
- Purple-to-blue "AI startup" gradients
- Confetti, emoji bursts, or celebratory overlays
- Glassmorphism used decoratively rather than to signal a transient surface
- Neon glow / bloom on interactive elements (bloom is reserved for illustrative imagery only, §23)
- Cartoon mascots or anthropomorphized AI avatars
- More than one accent hue carrying meaning in a single view
- More than two simultaneous blurred surfaces on screen
- Sparkle (✨) or "magic" iconography standing in for an explanation of what the AI actually did

## 45. Design Constitution

**Article I.** Darkness is a canvas. It is never used as a substitute for a decision.

**Article II.** Nothing is hidden. Everything is quiet by default. These are the same rule, not two rules.

**Article III.** Every motion answers "what changed" before it is allowed to ship.

**Article IV.** Every color exists on purpose. If a token isn't doing a job no other token can do, it is removed, not kept "just in case."

**Article V.** Restraint is spent, not withheld — the boldest choice in this system is that there is almost never more than one bold choice on screen at once.

**Article VI.** The Gnomon marks the precise point. It does not decorate; it locates.

**Article VII.** A contributor who wants to add something new must be able to state, in one sentence, what job it does that nothing else in this document already does.

**Article VIII.** This document is amendable. It is not optional.
