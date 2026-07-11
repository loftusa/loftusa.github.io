# /houses weight sliders — "What matters to you"

**Date:** 2026-07-11 · **Status:** approved by Alex (in-session) · **Owner file:** `public/houses/index.html` only

## Goal

Turn the personal scout board into a general instrument: remove the "Where should Alex live?" framing and add
a slider panel that re-weights the fit score live in the browser, so moving a slider changes which houses
appear on the map and board. No pipeline changes — `data.js` already ships per-listing 0–10 component scores
and `meta.fit_weights`.

## Non-goals

- No changes to `refresh.py`, `rate.py`, CI, or `data.js` shape (a parallel workstream owns those files).
- Gio section untouched (his fixed rubric stays).
- No cross-device persistence of slider state (localStorage only).

## Header

- `<h1>` → **"Bay Area rental scout"**.
- Subtitle → neutral copy: real, currently-listed rentals, mapped and LLM-rated daily; set what matters to
  you below and the board re-ranks. Stats strip unchanged.
- "Top picks" section blurb neutralized (see Re-ranking).

## The instrument (new panel between header and map)

- **Allocation bar**: one slim stacked bar showing the *normalized* weight split; segments use the exact
  driver hues the map uses (`--commute --nice --aesthetic --nature --value --gym` + quiet/social). Animates
  on drag (CSS width transition).
- **7 sliders** (`<input type=range>` 0–100), one per component, hue-tinted, live % readout
  (normalized share, not raw). Order: commute, nice area, photos, nature, value, quiet/social, gym.
- **Max price slider**: range = data min→max rounded to $50s; default = max (uncapped); live $ readout.
- **Top N select**: 10 / 20 / 35 / all; default 20.
- **Reset ↺**: restores shipped `meta.fit_weights.alex` (scaled to raw 0–100) + uncapped price + N=20.
- **Persistence**: `localStorage["housesWeights.v1"]` = `{w: {commute,nice,aesthetic,nature,value,soft,gym}, cap, topN}`.
  Absent/corrupt → defaults. Saved on every change.

## Math (client mirrors server exactly)

- Raw slider values `r_i` → effective weights `ŵ_i = r_i / Σr`; if `Σr = 0`, equal weights (1/7).
- Per listing: `soft = quiet` if bucket=apt else `social`; `gym` score missing → 5.0 neutral;
  `fit = Σ ŵ_i · s_i`, `+0.8` if `loved`, capped at 10, rounded to 0.1.
- **Parity invariant**: with sliders at shipped defaults, client fit === shipped `x.fit` for every listing.
  Enforced by an automated node test (`tests/houses/parity.test.mjs`) run against the live `data.js` before
  any push; the fit computation is factored into pure functions so the test evals them directly.
- Driver hue = `argmax ŵ_i·(s_i − mean_i)` (≤0.02 margin → neutral clay), recomputed against **live**
  weights and live field means of the visible candidate set — map encoding always honest to the sliders.

## Re-ranking (what re-renders on input)

- **Candidate set** = all listings minus `gone` minus hidden (✕). Recompute fit for all; sort desc → RANK.
- **Top N is a global rank cutoff**: price cap filters the candidate set, then top N by fit defines "the
  board". The existing Area/Type/Contacted chips filter *within* the board (grid only, as today) and do not
  change who makes the N — rank numbers stay stable across chip toggles.
- **Map + card grid**: show the board (map ignores chips, as today). Rank badges renumber; coxcomb petals
  and "strongest vs field" popup lines use live weights.
- **Leaderboard**: ALL candidates, re-sorted, with a hairline rule after position N (the visibility cutoff);
  rows past the rule slightly faded. Gone rows stay at the bottom, struck through, as today.
- **Top picks** → **"Top 3 right now"**: top 3 under current sliders. The daily rater's pipeline `pick`
  becomes a small ★ badge on cards ("rater's pick") instead of owning the section.
- Debounce re-render ~80ms on slider input; markers rebuilt via the existing `drawMarkers` path.

## Edge cases

- Σ raw = 0 → equal weights (no NaN).
- Price cap below cheapest listing → empty grid/map with the existing "Nothing matches" empty state.
- N ≥ candidate count → everything shows, no cutoff rule.
- Pinned carry-forward rows missing `scores.gym` → neutral 5 (existing convention).
- Old localStorage schema / unknown keys → ignored, defaults used.

## Testing

- Node test file `tests/houses/parity.test.mjs` (no framework, plain asserts): default-weight parity vs
  shipped fits; renormalization (Σ=0, single-slider-max); soft mapping apt/room; gym-missing neutral;
  top-N + price-cap filter counts; loved bonus + cap.
- `node --check` on extracted inline script (existing convention) before push.
- Manual browser verification of slider → map/list re-rank before push (verification-before-completion).
