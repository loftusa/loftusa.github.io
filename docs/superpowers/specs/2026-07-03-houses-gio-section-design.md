# /houses "For Gio" section — walkable to OpenAI HQ

**Date:** 2026-07-03 · **Status:** approved (design approved by Alex in-session)

## Goal

Alex's friend Gio works at OpenAI's SF headquarters and wants to live within a short
walk of it. Near-office inventory mostly exceeds the board's price caps ($3,100 apt /
$2,100 room), so the board gets a new, separately-priced, separately-rated section:
listings walkable to the office, with price / type / walk-time toggles.

## Verified facts

- Office: **1455 Third St, San Francisco 94158** (Mission Bay campus, 1455 + 1515 Third
  St — the ex-Uber buildings). Geocoded via OSM Nominatim: **37.7700, -122.3888**
  ("Uber Headquarters Building 1, 1455, 3rd Street, Mission Bay").
- Craigslist sapi items carry per-listing `lat`/`lon` (parsed in `parse_item`), so exact
  walk distance is computable with the existing `haversine_mi`.

## Decisions (from brainstorm with Alex)

1. **Both** rooms and apartments/studios, with a type toggle.
2. **LLM-rate Gio's listings** with a Gio-specific rubric (+~$0.15/day; total ≈ $0.48/day).
3. Price ceiling **$6,000 apt / $3,500 room** — show everything walkable, luxury towers included.
4. Approach: **parallel sub-pipeline sharing the existing files/workflow** (no YAML changes,
   Alex's `listings` array untouched). Map integration included (office anchor + Gio dots).

## Architecture

Same three scripts, same GitHub Actions workflow. Gio candidates ride through the
existing files tagged by audience:

```
--pull   Alex pull (unchanged) ──┐
         Gio pull (office-centered, wider prices) ──┴─> shortlist.json  (rows tagged aud: alex|gio; ids L##/G##)
rate.py  batches grouped by aud; RUBRICS[aud]        ─> ratings.json    (same strict schema, same fail-loud coverage)
--build  aud=alex -> data.listings (byte-identical behavior)
         aud=gio  -> data.gio = {office, listings, meta}
--sweep  prunes dead listings in BOTH arrays
```

### Pull (refresh.py)

- Constants: `GIO_LAT, GIO_LON = 37.7700, -122.3888`; `GIO_QUERIES = [("apa", 1400, 6000, "apt"),
  ("roo", 700, 3500, "room")]`; pull radius 2 mi; `GIO_MAX = 24` shortlist slots.
- `pull_raw` is parameterized over (centers, queries) — default args preserve existing behavior.
- Per listing: `walk_min = haversine_mi(office) × 1.3 (route factor) × 20 (min/mi)`; keep
  `walk_min <= 32`, price ≥ 700, has photos + lat/lon.
- Selection: sort by walk_min ascending; per-bucket cap so neither rooms nor apts exceed
  ~60% once ≥8 selected; take up to 24. Scrape galleries/bodies with the existing
  `scrape_page`. Ids `G01…`; `aud: "gio"` on every row (Alex rows default `alex` when absent).
- **Failure isolation:** the whole Gio pull is wrapped; on error, `pull_stats.json` gets
  `gio_pull_ok: false`, shortlist carries zero G rows, and Alex's pipeline proceeds
  untouched. Thin Gio inventory (even 0–4 rows) is a valid result, not a failure —
  no MIN gate for Gio.

### Rate (rate.py)

- `RUBRICS = {"alex": RUBRIC, "gio": GIO_RUBRIC}`; batches never mix audiences;
  `rate_batch(client, batch, montages, rubric)`.
- `listing_text` appends the computed walk line for gio rows ("~9 min walk to OpenAI HQ").
- GIO_RUBRIC gist: rating for Gio, who works AT OpenAI HQ (1455 Third St, Mission Bay);
  #1 criterion is the short walk (already computed and shown — sanity-check text claims
  against it); `nature` = waterfront/parks/greenery (bay trail, Mission Bay parks);
  `social` = friendly / young-professional housemate vibe, generic-good (NO
  founder-networking agenda, solo studio scores low but is not penalized overall);
  `value` = price vs. what you get for near-office SF; `aesthetic` unchanged (photo
  quality, weighted into rank); `commute` = walkability judgment (deterministically
  overridden downstream). Same 7-dim schema, same int-1..10 validation, same
  full-coverage fail-loud (coverage assert spans L + G ids).

### Build (refresh.py)

- Split rated rows by aud. Alex path unchanged — regression-locked by test.
- Gio proximity score: `prox = max(0, min(10, 10 − max(0, walk_min − 8) / 2.4))`
  (10 at ≤8 min, ~0 at 32).
- Gio fit: `0.34·prox + 0.20·aesthetic + 0.16·nice + 0.14·value + 0.16·soft`
  (soft = quiet for apt, social for room), rounded to 0.1, cap 10.
  `scores.commute` displays `round(prox, 1)`.
- Same drops as Alex's (live=false, commercial, fit ≤ 2), dedupe by URL within gio.
- `data.gio = { office: {name, addr, lat, lon}, listings: [...], meta: {generated,
  n_scouted, n_shortlist, n_shown, price_min/max/med} }`. Gio listing fields mirror
  Alex's minus drive/ferry/loved/pick/pinned, plus `walk_mi`, `walk_min`.
- **Carry-forward:** if the Gio pull failed (`gio_pull_ok: false`) or shortlist has no G
  rows while the previous data.js has a gio object, carry the previous `gio` forward
  verbatim (stale `generated` date shows honestly). Alex's build result never depends
  on Gio's.
- No pinning / reach-out for Gio listings (reach-out tracking is Alex's).

### Sweep (refresh.py)

- Checks `listings` + `gio.listings`. Gio prune = positive-dead only, no pinning.
- Guard: if a single sweep would remove >60% of current Gio listings, skip Gio changes
  (scrape problem more likely than mass delisting). Alex's existing `< 10` guard unchanged.

### Frontend (index.html)

- New section between "All scouted listings" and "Neighborhood scorecards":
  `<h2 id="gio">For Gio — walkable to OpenAI HQ</h2>` (+ note naming the office and
  rating basis; deep-linkable as `/houses#gio`).
- Chips (reuse `chips()`): price ≤$2k / ≤$3k / ≤$4.5k / all(≤$6k); type All/Apartments/Rooms;
  walk ≤10 / ≤20 / ≤30 min. Sort select: Best fit (default) / Walk time ↑ / Price ↑ / Best photos.
- `gioCard(x)`: existing card idiom (gallery nav via shared IMAGES/IDX — G ids can't
  collide with L/P ids), bucket badge, price, Gio-rank fit, metaline shows
  **walk N min to OpenAI** where Alex's SF/Berkeley line sits, `glyphRow(x.scores)`,
  rationale, "View listing ↗". No reach-out button, no pick star.
- Map: `anchor(office)` "OpenAI HQ" ◆; Gio listings as indigo (`--gio`) rank-numbered
  dots (own registry, drawn like `drawMarkers`) with a compact popup variant (photo
  gallery, price, walk min, rationale); legend line. Card-hover ↔ marker highlight and
  click-to-fly wired like Alex's.
- Stats strip gains "For Gio: N walkable". Whole section renders a graceful empty state
  when `D.gio` is missing (old data.js) or has zero listings.

## Testing (TDD, backend/tests/test_refresh_pipeline.py)

1. Walk math: office itself → walk_min 0/prox 10; known offsets straddle the ≤32 keep-gate.
2. `select_gio`: walk-sorted, bucket balance, 24-cap, no-photo/no-geo rows dropped.
3. Gio fit formula: extremes and a hand-computed mid case; apt uses quiet, room uses social.
4. Regression: alex-only shortlist+ratings → build output identical to pre-change
   (listings content + ordering), gio absent-but-tolerated.
5. Carry-forward: gio_pull_ok false + previous data.js gio → verbatim carry.
6. Sweep: dead G pruned, live kept; >60% dead → gio untouched; Alex guard behavior intact.
7. rate.py pure parts: audience batch-grouping never mixes; gio listing_text contains walk line.

## Cost & ops

+2 sapi calls, ~24 scrapes, ~5 rating batches ≈ +$0.15/day → ~$0.48/day total.
No workflow YAML changes; bot commits and Vercel deploy paths unchanged.

## Done criteria

Full backend suite green → push master → dispatch `houses-refresh` full run → live
alex-loftus.com/houses#gio shows LLM-rated Gio listings; toggles filter correctly;
walk times sane against the office pin; Alex's board unchanged aside from the new section.

## Out of scope (v1)

Reach-out tracking for Gio, a separate /houses/gio page, transit/bike times,
per-building walking routes (route factor 1.3 is the estimate), Gio-specific
neighborhood cards.
