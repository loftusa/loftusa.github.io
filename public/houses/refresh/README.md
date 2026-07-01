# /houses refresh pipeline

Keeps **alex-loftus.com/houses** fresh from live Craigslist data. Runs on
GitHub Actions (`.github/workflows/houses-refresh.yml`) — Craigslist is
reachable from Actions runners (verified: sapi/pages/images all 200).
Pushing `data.js` auto-deploys via Vercel; `deploy.yml` is paths-filtered so
these commits don't redeploy the Fly backend.

## Modes

```bash
python3 public/houses/refresh/refresh.py --pull    # Craigslist -> refresh/shortlist.json
python3 public/houses/refresh/rate.py              # Claude API rates every listing -> ratings.json
python3 public/houses/refresh/refresh.py --build   # merge + rank -> public/houses/data.js
python3 public/houses/refresh/refresh.py --sweep   # keyless: prune dead listings from data.js
```

- **full** (daily, 7am PT): `--pull` scrapes the sapi JSON API across SF /
  East Bay / North Bay, filters to Alex's zone/budget, attaches neighborhood
  vibe + dual-anchor (SF **and** Berkeley) commute priors, selects ~55 diverse
  candidates, scrapes photo galleries + posting bodies. `rate.py` tiles each
  listing's photos into a montage and rates batches via the Messages API
  (structured JSON output, full-coverage validation, fails loud; needs
  `ANTHROPIC_API_KEY`; `RATE_MODEL` env overrides the model — default
  claude-sonnet-4-6, ~$0.30/run). `--build` merges ratings with the commute
  model, drops dead/commercial/low-fit, dedupes, ranks, and **pins
  reached-out listings** (fetched from the Fly backend) so contacted places
  never silently vanish — delisted ones are kept, flagged `gone`.
- **sweep** (every 6h, free): re-checks each shown listing's page; prunes
  positively-dead ones (fetch failures count as alive), keeps contacted ones
  with `gone: true`, aborts on suspicious mass-death.

Both fail loud rather than ever shipping a degraded/empty board. Without the
`ANTHROPIC_API_KEY` repo secret, the daily full run degrades to a sweep.

## ratings.json schema (one object per shortlist id)

```json
{"id": "L01", "nature": 7, "quiet": 8, "nice": 9, "social": 3, "value": 6,
 "commute": 7, "aesthetic": 8, "fit": 8, "why": "one line",
 "live": true, "commercial": false}
```

`social` = networking with Alex's target crowd (tech founders/CEOs, AI-lab &
elite-university people) — **not** artist/creative-collective houses.
`aesthetic` = how good the place looks in its photos (weighted 0.16 into fit
so ugly-photo listings can't rank high on looks alone). Full rubric: `RUBRIC`
in `rate.py`.

## Tests

Pure-function regressions (priors incl. the marina/Marin substring bug,
berk_drive, cscore, contact extraction) live in
`backend/tests/test_refresh_pipeline.py` and run with the backend suite:
`uv run pytest backend/tests/`.

## Notes

- `shortlist.json`, `ratings.json`, `pull_stats.json` are regenerated per run
  and git-ignored; only `public/houses/data.js` is committed/deployed.
- Output schema must match what `public/houses/index.html` reads
  (`listings[].{scores{...aesthetic},fit,rationale,drive_sf,drive_berk,ferry_sf,loved,pick,pinned,gone,imgs,contact_email,contact_phone}`,
  `neighborhoods[]`, `meta.{n_scouted,n_shown,generated,price_med,move_by}`).
- Reach-out state lives in the Fly backend (`/houses/reached-out`, craigslist-
  URL-validated + per-IP rate-limited) with a localStorage mirror in the page.
