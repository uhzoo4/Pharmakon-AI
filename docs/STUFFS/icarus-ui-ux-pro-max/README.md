# UI/UX Pro Max — packaged for Zeaki's projects

## First, a correction worth knowing
`ui-ux-pro-max` isn't a skill built into this chat interface — it doesn't
appear anywhere in Claude.ai's own skill set. It's a real, separate,
open-source project by **Next Level Builder** (MIT license), built to run
as a **Claude Code** skill (also works with Cursor, Windsurf, and others).
Source: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill

Everything in `skill/` below is the actual, unmodified skill — cloned
from the upstream repo at commit `12b486b` / tag `v2.10.2` (2026-07-06),
not recreated from memory. `design-system/` is new: three starter design
systems generated *by that real tool*, one per project you're actively
building, plus my own notes on where the tool's automated guess was a
good fit and where it wasn't.

## What's in this zip

```
skill/                    ← the real skill. Copy this folder to
├── SKILL.md                .claude/skills/ui-ux-pro-max/ in any project
├── scripts/                 and Claude Code will auto-load it.
│   ├── search.py          (rename the top folder to "ui-ux-pro-max"
│   ├── core.py             when you copy it in)
│   └── design_system.py
└── data/
    ├── *.csv              ← the searchable databases (see inventory below)
    └── stacks/*.csv        ← per-framework guideline sets

design-system/             ← pre-generated starter systems for your projects
├── icarus-archos/
│   ├── MASTER.md            (raw tool output)
│   └── NOTES.md              (what to use, what to ignore, why)
├── icarus-papers/
│   ├── MASTER.md
│   ├── NOTES.md
│   ├── gsap-motion-reference.md
│   └── threejs-stack-reference.md
└── vulnsentry-ai/
    ├── MASTER.md
    ├── NOTES.md
    ├── chart-reference.md
    └── react-stack-reference.md
```

**Read each project's `NOTES.md` before its `MASTER.md`.** The automated
`--design-system` command is a keyword-matching reasoning engine, not
magic — for two of the three projects it picked a *plausible but wrong*
style/color match, and the notes explain exactly what to use instead
(still pulled from the same real database, just a more targeted query).

## How to actually install it (Claude Code)

```bash
# Option A — this exact snapshot, no network needed
cp -r skill ~/your-project/.claude/skills/ui-ux-pro-max

# Option B — the live upstream project (gets updates, has a CLI + docs)
npm install -g ui-ux-pro-max-cli
uipro init --ai claude --global
```
Then in a Claude Code session: `python3 .claude/skills/ui-ux-pro-max/scripts/search.py "<query>" --design-system -p "Project Name"` — or just describe what you're building; the SKILL.md's own trigger conditions are broad enough that Claude Code will reach for it on its own.

This chat interface (claude.ai) can't load third-party Claude Code skills directly — there's no local `.claude/skills/` folder here to install into. This zip is meant for your local Claude Code / Cursor / Windsurf setup.

## Full component inventory (measured directly from this snapshot)

| File | Domain flag | Real entries |
|---|---|---|
| `colors.csv` | `--domain color` | 160 product-specific palettes |
| `products.csv` | `--domain product` | 161 product types with reasoning rules |
| `ui-reasoning.csv` | *(used internally by `--design-system`)* | 161 rules |
| `google-fonts.csv` | `--domain google-fonts` | 1,923 individual fonts |
| `styles.csv` | `--domain style` | 84 UI styles |
| `icons.csv` | `--domain icons` | 104 icon-system configs |
| `typography.csv` | `--domain typography` | 73 font pairings |
| `ux-guidelines.csv` | `--domain ux` | 98 UX rules |
| `react-performance.csv` | `--domain react` | 44 React/Next.js perf rules |
| `app-interface.csv` | `--domain web` | 29 native app-interface (iOS/Android) rules |
| `landing.csv` | `--domain landing` | 34 landing-page structures |
| `charts.csv` | `--domain chart` | 25 chart-type recommendations |
| `motion.csv` | `--domain gsap` | 16 GSAP animation skeletons by intensity tier |
| `data/stacks/*.csv` | `--stack <name>` | 16 stacks with real data (below) |

**Stacks with actual data in this snapshot:** react, nextjs, vue, svelte,
astro, swiftui, react-native, flutter, nuxtjs, nuxt-ui, html-tailwind,
shadcn, jetpack-compose, threejs, angular, laravel.

**Known gap:** the CLI also *accepts* `--stack javafx|wpf|winui|avalonia|uno|uwp`
(desktop-native frameworks) but this snapshot ships no CSV for any of
them — passing one throws `Stack file not found`. There's also no
Linux-desktop/Hyprland/GTK stack at all (this tool's whole domain is
web + mobile-app UI, not window-manager theming), and its `icons` domain
only covers mobile icon-library configs (Phosphor/react-native) — not
useful for Linux desktop icon themes. Noted in `icarus-archos/NOTES.md`.

There are also two large unwired files, `design.csv` (1,774 rows) and
`draft.csv` (1,777 rows), sitting in `data/` — I checked `core.py`
directly and neither is connected to any `--domain` or `--stack` flag,
so they're upstream staging/internal files, not part of the searchable
surface. Left them out of `skill/data/` here since they add ~200KB of
dead weight; nothing in `SKILL.md`'s documented usage reads from them, so
removing them doesn't change any documented behavior. If you want the
byte-for-byte upstream folder instead, use Option B above.

## A note on safety
Before running anything, I checked all three scripts
(`search.py`/`core.py`/`design_system.py`, ~1,700 lines total) for
network calls, `subprocess`/`os.system`, and `eval`/`exec` — there are
none. It's a local CLI that only reads its own bundled CSVs and writes
Markdown/JSON to a folder you specify. Worth re-checking yourself if you
pull a newer version later, same as you'd want for any script that isn't
yours.

## License
MIT, © 2024 Next Level Builder. Full text: upstream repo's `LICENSE`.
This is a redistribution of an already-public, MIT-licensed tool (its own
install docs are literally `git clone` / `npm install`), not a claim of
authorship.
