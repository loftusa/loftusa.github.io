# /klist — interactive partner-preferences checklist

**Date:** 2026-07-02 · **Status:** approved (design approved in conversation; user said "build")

## Goal

An interactive version of the paper kink-checklist Alex shares with potential partners, at
**alex-loftus.com/klist**. Fillers complete it in the browser, submit it privately to Alex, and can
print / save a PDF of their filled sheet. Alex reads submissions on a password-gated admin page.

## Decisions (from brainstorming)

- **Content:** faithful digitization of the provided sheet image — same sections, items, and
  colour-code rating scheme. Item labels kept verbatim (including dated ones like "Gypsy").
- **Submissions:** stored in the existing Fly.io FastAPI backend's SQLite DB (on the Fly volume —
  never in this public repo). No email delivery.
- **Admin access:** password-gated web page at `/klist/admin` reusing the backend's existing
  `require_bearer` / `LOG_ACCESS_TOKEN` gate.
- **Discoverability:** unlisted — `noindex,nofollow` meta, not linked from site nav. The blank form
  itself is public content (it lives in this public repo); only submissions are private.
- **Print:** works standalone — a filler can print without submitting, and can print after
  submitting. Admin page is printable per-submission the same way.
- **No picture upload** (the paper sheet's "Picture" box is dropped — image storage is out of scope).

## Architecture

Three pieces, following the site's established patterns:

1. **Form page** — `public/klist/index.html`, self-contained vanilla HTML/CSS/JS (like `/sti`,
   `/jobs`). Served via a `next.config.ts` rewrite `/klist → /klist/index.html`.
2. **Backend router** — `backend/app/routers/klist.py` on the existing Fly app:
   - `POST /klist/submissions` — public, rate-limited like other public endpoints; body
     `{name, answers}`; stores a row; returns `{ok, id}`.
   - `GET /klist/submissions` — bearer-gated (`require_bearer`); returns all submissions as JSON.
   New table `klist_submissions` (SQLAlchemy model + Alembic migration):
   `id, ts (ISO string), name (nullable), payload (JSON), ip, created_at`.
3. **Admin page** — `public/klist/admin.html`, rewrite `/klist/admin → /klist/admin.html`.
   Prompts for the token once (sessionStorage), fetches the gated GET, renders each submission
   read-only in the same checklist layout, with a print button per submission.

Data flow: form page → `fetch POST https://<fly-app>/klist/submissions` (CORS already allows the
domain) → SQLite on the Fly volume → admin page GET with `Authorization: Bearer <token>`.

## Form content (transcribed from the sheet)

- **Profile** (text inputs): Age, Gender, Location, Orientation, Role, Contact.
- **Physical Features** (plain checkboxes): Tall, Short, Muscular, Fat, Chubby, Thin, Skinny,
  Petite, White, Black, Asian, Hispanic, Arabic, Gypsy, Indian, Native american, Hairy, Shaved,
  Trimmed, Bearded, Big boobs, Small boobs, Big cock, Small cock, Big ass, Small ass.
- **Group sex** (rated): MFM, FMF, MMM, FFF, Orgy, Partner swap, Gangbang, Bukkake.
- **General¹** (rated): Blowjobs, Cunnilingus, Handjobs, Fingering, Boobjobs, Hair pulling, Toys,
  Tickling, Face fucking, Face sitting, Vaginal fisting, Vaginal creampie, Facials, Rough sex,
  Swallowing, Blindfolds.
- **Kinks/Fetishes²** (rated): Incest, Impregnation, Pregnancy, Lactation, Exhibitionism,
  Voyeurism, Roleplay, Petplay, Dirty talking, Feet, Armpits, Public play, Hidden public play,
  Public sex, Hidden public sex, Ageplay, Taking pictures/videos, Phone sex.
- **BDSM** (rated): Daddy/little, Dominant/submissive, Master/slave, Owner/pet, Femdom, Maledom,
  Power exchange, Humiliation, Degradation, Name-calling, Discipline, Light bondage, Heavy bondage,
  Shibari, Predicament bondage, Encasement, Caging, Chastity, Forced orgasms, Orgasm denial,
  Suspension, Spanking, Servitude, Masturbation instructions, Sensation play, Impact play,
  Breath play, Choking, E-stim, Human furniture, Gags, Begging, Teasing, Sounding, Collars,
  Leashes, Cock worship, Pussy worship, Feet worship, Ass worship, Body worship,
  Sensory deprivation.
- **Ass play** (rated): Anal sex, Anal fisting, Anal fingering, Anal creampie, Rimming, Pegging,
  Anal toys, Enemas.
- **Clothing** (rated): Clothed sex, Crossdressing, Forced clothing, Diapers, Lingerie,
  Locking clothes, Stockings, Heels, Leather, Latex, Uniforms, Zentai, Socks, Costumes.
- **Pain** (rated): Mild pain, Strong pain, Nipple clamps, Genital slapping, Hard spanking,
  Caning, Wax play, Biting, Scratching, Clothespins, Flogging, Breast torture.
- **Other kinks** (3+ free-text rows, each ratable).
- **Other limits** (free text): Hard, Soft.
- **Footnotes** (displayed): ¹ Common sexual practices, or things not exclusively/mostly associated
  with BDSM. ² Things that one likes (kinks) or needs (fetishes) and do not clearly belong elsewhere.

### Rating scheme (the sheet's colour code)

Six ratings: **Favourite** (cyan), **Yes** (green), **Maybe** (yellow), **As a fantasy** (purple),
**No** (red), **Does not apply / do not care** (blank, default). Plus an orthogonal
**Give / Receive / Both** modifier on every rated item (the sheet's half-fill / quarter-fill codes).

Interaction: tapping an item's swatch cycles Favourite → Yes → Maybe → Fantasy → No → blank.
Once an item has a rating, a small G/R/B tri-toggle appears beside it (tap cycles G → R → B → none).
A legend at the top explains both, mirroring the sheet's Colour code box.

## Submission payload

```json
{
  "name": "optional display name (from Contact/Profile)",
  "profile": {"age": "", "gender": "", "location": "", "orientation": "", "role": "", "contact": ""},
  "features": ["Tall", "Hairy"],
  "ratings": {"General/Blowjobs": {"r": "favourite", "m": "give"}},
  "other_kinks": [{"label": "…", "r": "yes", "m": null}],
  "limits": {"hard": "…", "soft": "…"}
}
```

Only non-blank ratings are sent. Server validates shape loosely (dict with expected top-level keys,
size cap ~64 KB) and stores the payload verbatim as JSON — the form is the schema owner.

## Error handling

- POST failures (network/5xx): form keeps state, shows a retry message; answers also persist to
  `localStorage` on every change so nothing is lost on reload.
- Backend: 422 on malformed body, 413-style rejection (400) on oversized payload, existing
  rate-limiting pattern on the public POST. Admin GET: 401 wrong/missing token, 500 if token unset.
- Admin page: 401 → re-prompt for password.

## Print

A print stylesheet (`@media print`) renders the filled form compactly (multi-column, colour swatches
with `print-color-adjust: exact`, inputs shown as text). "Print / save PDF" buttons on both pages
call `window.print()`. Unrated items print with empty swatches so the sheet reads like the paper
original.

## Testing

- **Backend (pytest, TDD):** POST stores a row and returns id; GET without/with wrong token → 401;
  GET with token returns stored submissions; oversized/malformed payload rejected.
- **Frontend:** manual verification via `pnpm dev` + browser (fill, cycle ratings, submit against
  local backend, print preview, admin render). `pnpm build` must pass.

## Out of scope (YAGNI)

Picture upload, filler accounts/editing past submissions, comparing two submissions side-by-side,
email notifications, i18n.
