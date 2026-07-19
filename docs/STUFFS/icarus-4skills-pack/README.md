# Impeccable · Huashu-Design · Taste Skill · Playwright — packaged for Zeaki's projects

## Same correction as last time
None of these four are built into this chat interface either. All four
are real, separate, open-source **Claude Code** skills (three design
skills + one official testing tool), cloned from their actual upstream
repos — not recreated from memory. Commits/versions below are exact.

| Skill | Author | License | Stars (as reported, growing fast) | Source |
|---|---|---|---|---|
| Impeccable | Paul Bakaus (`pbakaus`) | Apache 2.0 | ~44k+ | github.com/pbakaus/impeccable |
| Huashu-Design (华叔设计) | 花叔 / Hua Shu (`alchaincyf`) | MIT (since 2026-05-14) | ~14k+ | github.com/alchaincyf/huashu-design |
| Taste Skill | Leon Lin (`Leonxlnx`) | MIT | ~59k+ | github.com/Leonxlnx/taste-skill |
| playwright-cli | Microsoft | Apache 2.0 | official | github.com/microsoft/playwright-cli |

Star counts move fast and vary by source/date — treat them as "very
popular," not exact figures.

## What's in this zip

```
skills/
├── impeccable/          ← real, pre-compiled Claude Code build (not the
│                            template source — see "Why not skill/" below)
├── huashu-design/        ← SKILL.md + 26 references + export scripts +
│                            JSX assets + demos (audio trimmed, see below)
├── taste-skill/           ← all 13 sub-skills + the laziness research
└── playwright-cli/        ← official Microsoft skill: SKILL.md + 10 refs

design-context/            ← my notes applying each tool to your actual
├── icarus-papers/            projects, using each tool's REAL format —
├── vulnsentry-ai/             not generic advice. Read each file's own
└── icarus-archos/             "honest caveat" section; none of this was
                                verified against your live repos.

README.md                  ← this file
```

## Install (all four are Claude Code skills/plugins, not claude.ai)

**Impeccable**
```bash
cp -r skills/impeccable/skills/impeccable your-project/.claude/skills/impeccable
cp -r skills/impeccable/agents your-project/.claude/agents
cp skills/impeccable/settings.json your-project/.claude/settings.json  # wires the auto-run-on-edit hook — merge, don't overwrite, if you already have one
# then inside Claude Code: /impeccable init
```
*Why not the repo's `skill/` folder:* that one's `SKILL.src.md` still has
unresolved `{{scripts_path}}` / `{{model}}` template placeholders — it's
the multi-provider build source, not something you can drop in directly.
What's zipped here is Impeccable's own already-compiled `.claude/`
folder (their own dogfooded build), confirmed to have zero unresolved
placeholders before I packaged it.

**Huashu-Design**
```bash
cp -r skills/huashu-design your-project/.claude/skills/huashu-design
# then: "make a keynote for X" / "build an iOS prototype for Y" / etc.
```
Six background-music files (~27MB, `bgm-*.mp3`) and the `sfx/` folder
(~600KB) were left out — they're stock audio beds for the video-export
feature, not skill logic. The video/animation/PPTX export pipeline still
works without them; if you want the actual music beds, `git clone
https://github.com/alchaincyf/huashu-design` gets the full repo.
**Note:** `SKILL.md` and most references are Chinese-primary (the author
is a Chinese-speaking developer); the agent is documented as bilingual
and works fine with English tasks. A community English mirror exists at
`github.com/pcornelissen/huashu-design-en` if you'd rather read the
skill's own instructions in English — I packaged the primary source here
since the mirror's translation isn't independently verified.

**Taste Skill**
```bash
cp -r skills/taste-skill/skills/taste-skill your-project/.claude/skills/taste-skill
# or grab any of the other 12 the same way — brutalist-skill, minimalist-skill,
# redesign-skill, brandkit, etc. — see the inventory table below
```
The upstream repo's own installer (`npx skills add Leonxlnx/taste-skill`)
handles multi-agent installs (Cursor/Codex/etc.) if you're not
Claude-Code-only; not run here since it needs to write into your actual
project.

**playwright-cli**
```bash
npm install -g @playwright/cli@latest
cp -r skills/playwright-cli your-project/.claude/skills/playwright-cli
# then: "use playwright-cli to open <url> and take a screenshot"
```
This needs the real `@playwright/cli` npm package installed to actually
drive a browser — the skill folder alone is just the instructions Claude
reads, not the browser automation engine itself. Not installed/run in
this sandbox (no display, and nothing of yours reachable to test against).

## Full component inventory (measured directly, not quoted from marketing)

| Skill | Core doc | Reference/sub-skill count | Notable extras |
|---|---|---|---|
| Impeccable | `SKILL.md`, v3.9.1 | 28 reference files (one per command: audit, polish, critique, distill, animate, colorize, typeset, layout, harden, onboard, live, …) | 2 subagents (`manual-edit-applier`, `asset-producer`); a PostToolUse hook that auto-runs a 45-rule detector after every UI file edit; a `palette.mjs` OKLCH color-seeding script |
| Huashu-Design | `SKILL.md` (~56K, Chinese) | 26 reference files | 20 named design philosophies (`design-styles.md`, 3 "schools": Bold/Neutral/Quiet) × web + PPT = effectively 40 style variants; JSX device-frame components (iOS/Android/macOS/browser); an HTML→PPTX/PDF export toolchain; 24 pre-built showcase examples |
| Taste Skill | 13 separate `SKILL.md` files | — | `design-taste-frontend` (v2, the flagship), `-v1` (pinned legacy), `high-end-visual-design`, `redesign-existing-projects`, `minimalist-ui`(folder: `minimalist-skill`), `industrial-brutalist-ui` (folder: `brutalist-skill`), `full-output-enforcement` (`output-skill`), `stitch-design-taste` (`stitch-skill`), `gpt-taste` (`gpt-tasteskill`), `image-to-code`, `brandkit`, `imagegen-frontend-web`, `imagegen-frontend-mobile`; plus a `research/laziness/` folder studying *why* LLMs ship lazy/truncated output, with root-cause and remediation docs |
| playwright-cli | `SKILL.md` | 10 reference files | Full CLI surface: click/type/fill/drag/upload/dialogs, network mocking (`route`), storage (cookies/localStorage/sessionStorage), tracing, video recording with chapters, an interactive `show --annotate` mode for live human UI-review sessions |

## A note on safety
Same due-diligence pass as the ui-ux-pro-max zip: scanned every script in
Huashu-Design's `scripts/` and Taste Skill's `research/`+`skills/` for
network calls, `eval`/`exec`, and `child_process`/`subprocess` before
packaging.
- **Huashu-Design**: `scripts/fetch_images.py`, `render-video.js`, and
  `tts-doubao.mjs` do call out to external services (an image-gen API, a
  headless-Playwright video renderer, and ByteDance's Doubao
  text-to-speech API) — this is documented, expected behavior for the
  export pipeline, and each needs its own API key you'd supply. None of
  it ran in this sandbox.
- **playwright-cli** is Microsoft's own signed, official package — by
  design it launches real browsers and can execute arbitrary JS via
  `run-code`. That's the entire point of the tool, not a red flag, but
  it's real capability worth being deliberate about (e.g. don't point it
  at a session with credentials you don't want touched, same as you'd
  treat any browser automation tool).
- **Impeccable** and **Taste Skill**'s packaged contents are pure
  markdown instructions + a few local file-writing/OKLCH-math scripts —
  no network calls in what's zipped here.

## design-context/ — how it's different from generic advice
Each file names which real tool feature it's using (Impeccable's exact
`PRODUCT.md` template from `reference/init.md`; Taste Skill's actual
`DESIGN_VARIANCE`/`MOTION_INTENSITY`/`VISUAL_DENSITY` dials and inference
table; Huashu-Design's real 20-philosophy library, cross-checked to find
the one specific style — **Cosmic Retro-Futurism** — that's a structural
fit for a flight-and-fall myth, not just a "dark and moody" guess;
`playwright-cli`'s actual command syntax). Every file also says plainly
what wasn't verified — none of this was run against your live repos from
this sandbox, and Impeccable's own docs are explicit that PRODUCT.md
should come from a real interview, not a one-shot guess. Treat everything
here as a first draft to run through each tool's real flow, not a
finished answer.
