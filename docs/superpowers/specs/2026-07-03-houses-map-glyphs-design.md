# /houses map glyphs — dots that explain their ranking

**Date:** 2026-07-03 · **Status:** approved (Alex picked D + C from rendered mockups of five encodings)

## Goal

The map markers currently encode rank (number), fit (size), and type (hue) — nothing about *why*
a listing ranks where it does. Ship the two encodings Alex selected from the live-data mockup:

- **D — dominant-driver hue** (far zoom): each dot is filled with the color of the component that
  most lifts the listing above the field average; type moves to the stroke.
- **C — coxcomb petals** (zoom ≥ 13): dots swap to six-wedge Nightingale roses, wedge radius ∝ raw
  score, so the full score profile becomes the dot's shape where there's room (micro/macro reading).

Rejected in the same review: B contribution donut (rings look uniform when scores are high),
E small-multiples strip (not picked; possible follow-up).

## Encoding rules

- Components, fixed order and colors (matching the card bars): commute `--commute`, nice `--nice`,
  aesthetic `--aesthetic`, nature `--nature`, value `--value`, soft (= quiet `--quiet` for apt,
  social `--social` for room).
- **Driver** = argmax over components of `w_i · (s_i − mean_i)`, means computed per audience
  (Alex's listings vs Gio's) over currently shown non-gone listings. Missing scores default 5.
  If the best lift ≤ 0.02, fill with neutral clay `#a39c8d` ("nothing above average").
- Weights: Alex `{commute:.26, nice:.17, aesthetic:.16, nature:.15, value:.13, soft:.13}`,
  Gio `{commute:.34, aesthetic:.20, nice:.16, soft:.16, value:.14, nature:0}` (nature excluded
  from Gio's driver argmax).
- **Far-zoom marker (zoom < 13):** fill = driver color; stroke 2.5px = audience/type (apt teal,
  room orange, Gio indigo); size = fit (unchanged); rank number stays white; pick gold ring,
  reached green ring, gone fade all unchanged.
- **Near-zoom marker (zoom ≥ 13):** 34px coxcomb SVG — six 60° wedges from 12 o'clock in fixed
  component order, radius 4→17px ∝ score, hairline paper-colored gaps; center disc = type hue
  with rank; pick/reached rings and gone fade applied to the wrapper. Swap happens on `zoomend`
  when crossing the threshold (setIcon on every registered marker, both audiences).
- **Popup** (both audiences) gains one line: "strongest vs field: <driver name>" in driver color.
- **Legend** rewritten: component color chips + "fill = what most lifts it vs the field ·
  outline = type · zoom in for full score profiles". Leaderboard dots stay type-colored (v1).

## Single source of truth for weights

`refresh.py --build` now:
- factors Alex's inline fit arithmetic into `alex_fit(scores, bucket, dual_commute)` (same terms,
  same order — existing fit values must not change), alongside the existing `gio_fit`;
- defines `ALEX_W` next to `GIO_W` and computes both fits from the dicts;
- emits `meta.fit_weights = {"alex": {...}, "gio": {...}}` in data.js.

The frontend reads `D.meta.fit_weights` and falls back to hardcoded constants (needed until the
next full run regenerates data.js, and for the pinned-listing carry-forward case).

## Testing

- `test_refresh_pipeline.py`: `alex_fit` pinned with a hand-computed case + apt/room soft swap;
  fit values byte-identical through `do_build` (existing invariance/build tests must stay green
  untouched); `meta.fit_weights` present, each audience's weights summing to 1.0 and matching the
  module dicts.
- Frontend: no JS test infra — verify via local preview (live data + synthetic zoom cases) in
  Chrome, then live after deploy: driver colors distribute across ≥3 components, zoom swap fires
  both directions, popups show the driver line, no console errors.

## Out of scope (v1)

Small-multiples strip (E), leaderboard driver swatches, colorblind palette rework
(nature-green vs value-teal proximity is pre-existing), contribution donut.
