# /houses daily refresh pipeline

Regenerates the data behind **alex-loftus.com/houses** from live Craigslist
listings. Run daily by the Anthropic cloud routine `houses-daily-refresh`
(the agent clones this repo, runs the two steps below, and pushes `data.js`;
Vercel then auto-deploys).

## Steps

```bash
python3 public/houses/refresh/refresh.py --pull    # -> refresh/shortlist.json (+ pull_stats.json)
#   ... rating agent reads shortlist.json, writes refresh/ratings.json ...
python3 public/houses/refresh/refresh.py --build   # -> public/houses/data.js
```

- **`--pull`** hits Craigslist's `sapi` JSON API across SF / East Bay / North Bay
  (apartments $1.3–3.1k + rooms $0.8–2.1k), dedupes, filters to Alex's commute
  zone/budget, attaches neighborhood **vibe** + **dual-anchor commute** priors
  (SF *and* downtown Berkeley / FAR Labs), selects ~50 diverse candidates, and
  scrapes each one's **photo gallery + posting body text**. It **exits non-zero**
  if Craigslist returns too little data (blocked/rate-limited datacenter IP), so a
  bad pull never ships a degraded page.
- **`--build`** merges `shortlist.json` with the agent's `ratings.json`
  (per-listing 1–10 scores + `fit`/`live`/`commercial`), applies the dual-anchor
  commute model, drops dead/commercial/low-fit, dedupes by URL, ranks by fit,
  and builds neighborhood cards. Exits non-zero if too few listings survive.

## ratings.json schema (written by the rating agent)

Array of objects, one per shortlist `id`:

```json
{"id": "L01", "nature": 7, "quiet": 8, "nice": 9, "social": 3,
 "value": 6, "commute": 7, "fit": 8, "why": "one line",
 "live": true, "commercial": false}
```

`social` = networking potential with Alex's target crowd (tech founders/CEOs,
AI-lab & elite-university people) — **not** artists/creative-collective houses.
See the routine prompt for the full rubric.

## Notes

- `shortlist.json`, `ratings.json`, `pull_stats.json` are regenerated every run
  and are git-ignored. Only `public/houses/data.js` is committed/deployed.
- The output schema must match what `public/houses/index.html` reads
  (`listings[].{scores,fit,rationale,drive_sf,drive_berk,ferry_sf,loved,pick,imgs}`,
  `neighborhoods[].{name,region,scores}`, `meta.{n_scouted,n_shown,generated,source}`).
