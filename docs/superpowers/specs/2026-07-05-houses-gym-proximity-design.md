# /houses gym-proximity scoring

**Date:** 2026-07-05 · **Status:** approved (Alex: weight 0.08, his board only; Gio display-only)

## Goal

"Gyms are important to me" — add walk-distance-to-nearest-gym as a deterministic 8th scoring
component on Alex's board, then re-run the full refresh so the live set reflects it.

## Data

- Gyms from OSM Overpass: `leisure=fitness_centre` + `leisure=sports_centre` with
  `sport~climbing|fitness`, nodes + ways (`out center`), bbox (37.55,-122.60,38.05,-122.10).
  Probe 2026-07-05: 372 gyms, 180 in SF proper.
- `fetch_gyms()` at pull time: on success (≥100 results) writes the snapshot to
  `public/houses/refresh/gyms.json` and returns it; on failure falls back to the committed
  snapshot (warn), else `[]`. The workflow's commit step also stages `gyms.json`, so the
  snapshot self-refreshes daily. A seed snapshot is committed with this change.

## Scoring

- Pull: every shortlisted row (Alex + Gio) gets `gym_min` = walk minutes to nearest gym
  (straight-line mi × 1.3 × 20, same model as Gio's office walk). No gyms data → None.
- Build: `gym_score(m) = clamp(10 − max(0, m − 4)/2.0, 0, 10)` — 10 within a 4-min walk,
  ~5 at 14 min, 0 at 24+; `gym_score(None) = 5.0` (neutral: missing data, pinned old rows).
- `ALEX_W` becomes `{nice .16, nature .14, soft .12, value .12, commute .24, aesthetic .14,
  gym .08}` (sum 1.0); `alex_fit` gains the gym term. `GIO_W` gains `gym: 0` — Gio's
  ranking unchanged; his listings still carry `gym_min` + a display `scores.gym`.
- `scores.gym` is set deterministically at build (like `scores.commute`); the LLM rubric
  is untouched.

## Frontend

8th component everywhere: card bars + score key (`SC`/`SCNAME`, new `--gym` color),
driver-hue candidates + legend (weight-0 audiences skip it automatically), coxcomb wedges
(angle generalizes to 360/n), "gym Nm" in card metalines and popups on both boards,
footer sentence. `FW` fallback updated to the new weights.

## Tests

Updated pinned fits (new weights change the expected values — recomputed by hand),
`gym_score` curve incl. None-neutral, nearest-gym walk math, `fetch_gyms` fallback chain
(Overpass dead → snapshot → []), weights-sum + meta emission.

## Done

Suite green → push → dispatch full run → live board re-ranked with gym scores visible,
`gyms.json` committed by the run.
