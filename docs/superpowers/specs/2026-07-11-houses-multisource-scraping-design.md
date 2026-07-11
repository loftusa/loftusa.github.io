# Houses multi-source scraping — design (2026-07-11)

## What / why

The /houses board was Craigslist-only. Craigslist's `roo` (rooms) coverage is
good, but managed apartment complexes (the "apt" bucket's professionally-run
end) mostly never post there. This adds **Rent.com** as a second listing
source, feeding the same shortlist -> LLM-rate -> build flow. Craigslist stays
the backbone: if every new source fails, the run's output is behaviorally
identical to before.

## Feasibility (verified 2026-07-11)

Probed from a residential IP and then re-verified **from a GitHub Actions
runner** via a temporary `scratch-probe.yml` workflow (removed after use):

| Source        | From Actions DC IP | Verdict |
|---------------|--------------------|---------|
| Rent.com      | HTTP 200, 524KB page, `__NEXT_DATA__` 177KB present; `i.rent.com` photo CDN also 200 | **added** |
| SpareRoom US  | HTTP 200 but only ~11 offered listings Bay-wide, and the search HTML carries no per-listing lat/lon (neighborhood + zip only) | skipped — the map requires coords; honoring "no geocoder" would drop every row, and per-listing page fetches for ~a handful of rooms isn't worth it |
| ApartmentList | not probed | skipped — same managed-complex segment Rent.com covers, 3.3MB messier ld+json parse; revisit only if Rent.com coverage proves insufficient |
| Zillow, Trulia, HotPads, Zumper, PadMapper, Apartments.com, Roomies | hard-blocked (PerimeterX / DataDome / FingerprintJS) | do not attempt |

## Rent.com fetcher

- One GET per city page per run (browser UA, no auth, ~1.2s spacing):
  San Francisco, Berkeley, Oakland, Sausalito — each with the
  `/max-price-3000` filter baked into the URL (matches Alex's apt ceiling).
  ~30 listings/page, ~90-120 raw rows/run.
- Parse: `<script id="__NEXT_DATA__">` JSON ->
  `props.pageProps.pageData.location.listingSearch.listings`. No HTML scraping.
- Each listing maps to the same normalized row shape Craigslist rows use
  (`pid/price/beds/bucket/hood/lat/lon/url/imgs/title/...`), plus `src: "rent"`
  (Craigslist rows now carry `src: "cl"`). Photos come from
  `https://i.rent.com/t_3x2_fixed_webp_lg/<photo-id>` (webp; Pillow handles it).
- `hood` = city name (the blob has only numeric hood ids); the street address
  goes into the title and body so the LLM rater sees it. City names hit the
  existing `priors()` rules (san francisco / berkeley / oakland / sausalito).
- Rows without lat/lon, price, or photos are dropped at parse time (the map
  needs coords; the montage needs photos). All 30/30 observed rows had both.
- `body` is synthesized (beds/baths/sqft/units/amenities + office phone) so
  rating and `extract_contact()` work unchanged — complex office phone becomes
  the card's contact number.
- Inventory is managed complexes only -> enriches the `apt` bucket; `room`
  stays Craigslist-only. The Gio section also stays Craigslist-only (the
  max-price-3000 filter is Alex's budget, not Gio's).

## Cross-source dedupe

Existing dedupe is URL-based and can't catch the same unit listed on both
sites. New pass before shortlisting: a non-Craigslist row within ~50m
(haversine) AND ~3% price of any Craigslist row is dropped, keeping the
Craigslist row (established id scheme + contact flow). Rows lacking coords on
either side never match (and coordless new-source rows are already dropped).

## Failure modes

- Each extra source is wrapped in try/except (`pull_extra_sources`); a printed
  warning, never a dead run. Craigslist failing hard still kills the run
  (unchanged fail-loud contract).
- Zero extra sources reachable -> same listings, same ranking, same
  `meta.source` string as today.
- Sweeps: `check_live()` applies its generic dead-signal check to rent.com
  URLs; a 200 marketing page never matches, so rent rows are pruned only on
  404/410 — acceptable (full daily refresh re-pulls anyway).
- Shortlist cap (55) and per-hood cap (3) are unchanged and rent rows flow
  through them, so at most ~12 rent rows can enter the shortlist and the
  LLM rating cost stays ~= today's (~$0.46-0.50/full run).

## Data shape changes (additive only)

- Shortlist rows + data.js listings carry `src` ("cl" | "rent"; absent rows
  default to "cl" for backward compat).
- `meta.sources` = per-source shown counts, e.g. `{"cl": 30, "rent": 8}`.
- `meta.source` string becomes "Craigslist (live API) + Rent.com, refreshed
  ..." when rent rows are shown; unchanged otherwise.
- `pull_stats.json` gains `n_extra_raw` / `n_extra_kept` per source.
