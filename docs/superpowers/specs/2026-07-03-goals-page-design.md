# /goals — a guided life-visioning exercise for Karin

**Date:** 2026-07-03
**Route:** alex-loftus.com/goals (unlisted, noindex — same policy as /sti and /klist)
**Origin:** WhatsApp chat 2026-07-03. Karin: happy and fulfilled, but recurring "where am I
even headed" spirals; every June wobble was a symptom of that one disease. Alex: "sit down and
think about what you want your life to look like 5, 10, 20, and 40 years from now." Karin:
"o i know this, i just dont see that happening anymore." → The exercise isn't the problem;
the blank page is. This page removes the sitting-down part.

## What it is

A fully client-side, single-file guided version of the 5/10/20/40-year visioning exercise.
One small concrete question per screen, everything skippable, autosaved to localStorage,
ending with an assembled "map" document she can print, download, or copy — plus a
note-to-self she writes for the next time the fog shows up.

## Design principles

1. **Concrete beats abstract.** "Describe an ordinary Tuesday in 2031" is answerable;
   "what are your goals" is a rabbit hole. Detail scales DOWN as the horizon grows
   (5 questions at 5y → 2 questions at 40y) because far futures go soft and pretending
   otherwise produces dread.
2. **One question at a time.** Never show the mountain. Progress is a thin timeline
   (now · +5 · +10 · +20 · +40 · threads · map) — the page's only graphic.
3. **Sketch, not contract.** Guessing allowed, skipping allowed (the primary button reads
   "skip for now" when the box is empty, "next" once she types). Maps are for redrawing.
4. **Private by construction.** Zero network calls. The privacy promise in the intro
   ("nothing is sent anywhere — not even to me") is literally true. No backend, no
   analytics, no fonts fetched.
5. **Close the loop.** The exercise ends by (a) showing her everything she wrote,
   (b) extracting 2–3 threads, (c) one inch-sized action within a month, and (d) a
   note-to-self for the next fog. On a return visit after completion, that note is the
   first thing the landing screen shows.

## Structure (18 prompts + 5 interstitials)

- **Landing** — title in her own words ("where am I even headed?"), subtitle "a map-drawing
  kit, built for Karin", short note from Alex (voice: warm, direct, no wellness-speak;
  thesis: "fog is scary; a hand-drawn map is not"). Begin / Continue (+ saved-note epigraph
  when returning after completion).
- **Now** (3) — most alive; never want to lose; name what the fog attaches to.
- **Five years — 2031** (5, "ordinary Tuesday") — wake up where; who's around; the work
  part; the evening; proud-of from that year's vantage.
- **Ten years — 2036** (3) — what life contains that it doesn't today; known for;
  stopped doing on purpose.
- **Twenty years — 2046** (2) — walk into your home, what does it feel like; the body of work.
- **Forty years — 2066** (2) — rocking-chair answer; what's still on the list even then.
- **Threads** (3) — first shows everything she wrote, then: name 2–3 threads; one small
  move within a month; write the note for the next fog.
- **Your map** — typeset document (question in small italic, answer beneath; unanswered
  omitted). Actions: Print / save PDF (print CSS shows only the document), Download .txt,
  Copy all. Any section tap jumps back to that question. Start-over uses a two-tap inline
  confirm (no browser dialogs).

Horizon years are computed from a `baseYear` stored at first visit (2026 → 2031/2036/2046/2066)
so questions and document stay consistent across resumed sessions.

## Approaches considered

- **A. One-question wizard (chosen)** — lowest activation energy; matches the promise.
- **B. Single long worksheet (klist-style)** — simpler, but shows the whole mountain;
  recreates exactly the wall she bounces off.
- **C. Chat-style fake conversation** — gimmicky, more code, a simulated Alex is weird.
- **Backend "send to Alex"** — rejected. Her map is hers; surveillance would poison the
  honesty the exercise needs. She can share the exported copy herself if she wants.

## Technical

- `public/goals/index.html`, self-contained vanilla HTML/CSS/JS (house pattern; klist/sti
  precedent). Rewrite `{ source: "/goals", destination: "/goals/index.html" }` in
  `next.config.ts`. `<meta name="robots" content="noindex, nofollow">`.
- House Tufte style: `#fffff8` paper, Palatino/Georgia serif stack (the et-book @font-face
  on /sti points at missing files, so no font files), ~34rem measure, ink reserved for
  meaning, klist-style buttons. Mobile-first — she'll open it from WhatsApp on a phone.
- State `localStorage["goals-v1"]`: `{ v, baseYear, cursor, answers: {qid: string},
  startedAt, updatedAt }`. Stable string question IDs (`now.alive`, `y5.wake`, …) so
  content edits never orphan answers. Autosave on input; corrupted state → fresh start;
  localStorage unavailable → in-memory state + gentle "couldn't save on this device" notice.
- Card transitions 200ms fade/translate, disabled under `prefers-reduced-motion`.
  Cmd/Ctrl+Enter advances. Autogrowing textareas. Textarea autofocus only on
  pointer-fine devices (no surprise keyboards on phones).
- No history-API wiring: on-page Back covers navigation; leaving the page loses nothing
  (autosave).

## Testing

`pnpm build` must pass. Full walk-through in Chrome at localhost:3000/goals: answer,
skip, reload mid-flow (resume), threads screen shows prior answers, map assembly,
download/copy, print CSS, start-over confirm, mobile viewport (390px), return-visit
epigraph. No network requests after page load (DevTools check).
