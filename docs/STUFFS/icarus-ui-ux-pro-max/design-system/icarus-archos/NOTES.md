# Calibration notes — Icarus ArchOS

Generated with:
```
search.py "developer platform operating system dark minimal AI-native tool" --design-system -p "Icarus ArchOS" --variance 6 --density 6
```

## What's usable as-is
- The resolved style — **Modern Dark (Cinema Mobile)** — landed on a dark
  slate/navy palette with a technical, premium, "developer tool" mood
  description. That's a genuine directional match for a systems-level,
  dark-first OS identity.
- The color tokens (`MASTER.md`) are real entries from the shipped
  `colors.csv`/`styles.csv` database, not invented — a legitimate starting
  palette to hold up against whatever's already committed in your
  Hyprland/waybar/GTK config files.

## What to ignore
- The **Pattern** section (Minimal Single Column: hero → CTA → footer) is a
  marketing-landing-page pattern. It's a byproduct of this tool being built
  for web/app UI — it doesn't apply to a window manager and should be
  skipped entirely for this project.
- There is no Linux desktop / Hyprland / GTK stack in this tool (16 shipped
  stacks are all web or mobile-app frameworks — see the root `README.md`).
  So there's no `--stack` output for this project; only the color/type/motion
  tokens are transferable, and they need manual translation into:
  - Hyprland `hyprland.conf` colors (`col.active_border`, etc.)
  - GTK CSS / `gtk.css` custom theme
  - waybar / rofi color config
  - terminal emulator theme (alacritty.toml / kitty.conf)
- I tried the `icons` domain for a Linux-appropriate icon-theme
  recommendation and it only returned React Native mobile icon-library
  configs (Phosphor icon imports) — not applicable, so I left it out rather
  than force a bad fit. For icon theming, the Linux ecosystem itself
  (Papirus, Tela, candy-icons, etc.) is the better source.

## Honest caveat
I don't have the literal hex values / exact token names you've already
committed for "Intelligent Darkness" — only the name and general direction
from past conversations. Treat everything in `MASTER.md` as a grounded
*candidate* palette to diff against your actual theme files, not a
restatement of a palette I already know is correct.
