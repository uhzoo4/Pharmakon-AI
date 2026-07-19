# Icarus ArchOS — why this folder is mostly empty

All four tools in this zip (Impeccable, Huashu-Design, Taste Skill,
Playwright) are built for **web frontends** — CSS, React/Vue/Svelte
components, browser-rendered pages. Icarus ArchOS is a window manager
config (Hyprland, GTK, waybar, a boot animation) with no browser DOM
anywhere in the loop. None of them apply directly, and I'd rather say
that plainly than force a translation that reads as more useful than it
is.

## If you want to borrow the concepts anyway
Loose, honest mappings — worth treating as vocabulary, not instructions:

- **Taste Skill's three dials** translate the most cleanly of anything
  here, because they're really about *how much* rather than *which
  framework*: `DESIGN_VARIANCE` ≈ how asymmetric/eccentric your rice
  looks (waybar module arrangement, workspace layout) vs. a clean
  symmetric grid; `MOTION_INTENSITY` ≈ Hyprland's `animations {}` block —
  bezier curve choice and speed for window open/close/workspace-switch;
  `VISUAL_DENSITY` ≈ how packed waybar/rofi are with modules vs. how much
  they stay out of the way.
- **Impeccable's brand-vs-product register split** doesn't map cleanly —
  an OS shell is closer to "product" (design serves daily use) than
  "brand," but the skill's actual rule engine (React/Tailwind detectors,
  DOM-based hooks) has literally nothing to detect in a `hyprland.conf`.
- **Huashu-Design and Playwright** don't translate at all — one produces
  HTML deliverables, the other drives a browser. Neither has anything to
  point at here.

## What would actually help instead
The ui-ux-pro-max color/typography tokens from the first zip are still
the closer starting point for this project — not because that tool is
OS-aware either (it isn't), but because color-token output is at least
medium-independent in a way "React component patterns" isn't. Beyond
that, the real source of truth for Linux desktop theming is the Linux
ecosystem itself (Hyprland's own animation docs, GTK theming, Papirus/
Tela icon themes) rather than any of these web-design skills.
