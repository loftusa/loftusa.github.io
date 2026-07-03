# Jobs Pro — concierge personal boards (v1 runbook)

Paid layer on top of /jobs: a customer gets their own scored board — their
profile, their fit scores/offer probabilities, their prep decks — as a
password-gated static page at `alex-loftus.com/jobs/u/<slug>/`.

## New customer → live board

1. **Intake → profile.** Write `users/<slug>/profile.md` from their intake
   (same rubric structure as `profile.md`: role-shape dimensions, what each
   1–10 means for THEM, hard constraints). Optional: `users/<slug>/prep_bank.json`
   (same shape as `prep_bank.json`: `star`, `hard_qs`, `pitch_core`,
   `negotiation_anchors`, `hard_rules`) — without it they still get scores and
   prep questions, just no STAR-story mapping.
   Everything under `users/` is gitignored — never commit customer data.

2. **Fresh pull** (if today's hasn't run yet):
   `cd public/jobs/refresh && python3 refresh.py --pull`

3. **Build the board:**
   ```
   ANTHROPIC_API_KEY=... uv run make_user.py \
     --slug <slug> --profile users/<slug>/profile.md \
     --password <SECRET> [--prep-bank users/<slug>/prep_bank.json]
   ```
   This pulls its own full-JD snapshot (the daily shortlist only has
   new/changed roles) and scores the LANE roles against THEIR profile via
   rate.py — all Anthropic roles plus non-Anthropic titles matching
   refresh.py's PREFILTER keywords (~500 of ~1,550). Widen PREFILTER if a
   customer's lane isn't covered; the public sales copy promises 'your
   target lane (~500 roles)', so keep that honest.
   and writes `public/jobs/u/<slug>/{index.html,data.enc.js}`. It also saves
   `users/<slug>.ratings.json` for cheap re-emits.

4. **Ship:** `git add public/jobs/u/<slug> && git commit && git push`.

5. **Deliver:** send the URL and the password via separate channels. Page is
   noindexed; the payload is AES-256-GCM, decrypted in their browser.

## Cost per run

One full rate.py pass over the whole board (~500+ roles): the same cost as the
main board's day-1 scoring run — roughly 50–60 score batches + up to 12 prep
batches on `RATE_MODEL` (default claude-sonnet-4-6). A few dollars, not cents.
Budget one full pass per customer per refresh.

## Refreshing a customer's board (manual in v1)

Re-run step 3 after any daily pull — it re-pulls, re-scores everything against
their profile, and overwrites `public/jobs/u/<slug>/`. Then commit/push.
There is no per-user diffing in v1, so every refresh is a full-cost scoring
run; refresh weekly or on request, not daily.

Password rotation / re-emit without re-scoring (no API cost):
```
uv run make_user.py --slug <slug> --profile users/<slug>/profile.md \
  --skip-rate --ratings users/<slug>.ratings.json   # --ratings is REQUIRED here
# password: prefer JOBS_USER_PASSWORD=<NEWSECRET> env over --password (argv
# lands in shell history / ps)
```

## Removing a customer

`git rm -r public/jobs/u/<slug>` + push. Their local copy stops updating but
already-downloaded pages keep decrypting — rotation only helps future visits.
