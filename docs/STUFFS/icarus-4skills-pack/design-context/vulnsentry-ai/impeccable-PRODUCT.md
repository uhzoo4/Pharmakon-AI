# Product

> Draft hypothesis, not a finished PRODUCT.md — same caveat as the Icarus
> Papers version. Run `/impeccable init` for real inside the repo; this is
> a starting answer for Step 3's interview, not a replacement for it.

## Register

product

## Users

You, running local network scans during hackathon judging/demo and personal
use. Secondary: anyone judging or reviewing the CodeQuest submission who
needs to understand scan results at a glance without security expertise —
the Gemini plain-language explanations exist specifically for this reader.

## Product Purpose

A local network port scanner that turns raw scan output into plain-language
vulnerability explanations via Gemini, streamed to the frontend over SSE
as the scan runs. Success = a judge or a non-security-expert user can look
at the dashboard mid-scan and understand what's wrong and how bad it is,
without reading a CVE database.

## Brand Personality

Three words: precise / alert / legible. Not a "hacker terminal" cosplay —
the Gemini-explanation feature only works if the UI reads as trustworthy
and clear under pressure (mid-scan, mid-demo), not as a Matrix-poster
aesthetic that a judge has to decode.

## Anti-references

Matrix-green-on-black "hacker terminal" cliché (glitch text, scanlines,
neon everywhere) — it's the default a lot of security-tool UIs reach for,
and it actively works against the "plain-language for non-experts" goal.
Also avoid dense, unlabeled data-table dumps — severity needs to be
readable as a glance, not a spreadsheet.

## Design Principles

- Severity is legible without reading — color + icon + text label together,
  never color alone (matches the accessibility note below)
- Live/streaming state (SSE) needs its own visual language: a scan in
  progress should look different from a completed scan, not just "the
  same list with more rows"
- Trustworthy over cool — every visual choice should ask "does this help
  a judge trust the explanation" before "does this look impressive"
- Plain-language explanations are the actual product; the UI's job is to
  not get in their way

## Accessibility & Inclusion

Severity indicators need a shape/icon or text label, not color alone
(red/green/amber colorblind conflicts are exactly the kind of thing a
security tool can't afford to get wrong). Live-updating scan results
need an `aria-live` region so the streaming updates are announced, not
silently inserted into the DOM.
