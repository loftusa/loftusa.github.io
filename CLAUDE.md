# loftusa.github.io

Personal website for Alex Loftus — **Next.js (App Router) on Vercel** + a **FastAPI chat API on Fly.io**
(`backend/`). Migrated from Jekyll to Next.js on 2026-07-01; the old `bundle`/`jekyll`/`_config.yml`
workflow is dead — ignore any doc or habit that says otherwise.

## Working in this repo

This is the **public** repository for **alex-loftus.com** and contains **only** the public website.
Clone it, edit, and `git push` / `git pull` on `master` normally.

**For AI assistants & contributors:** keep this repo public-clean. Do **not** add, copy, or merge
private or experimental work into it. If a local clone you are working in contains a `red-teaming/`
directory, that is a *separate, private* clone — never push that content here. (`/red-teaming/` on the
live site is served by **proxy** from a separate private Vercel project; its source is never in this
repo — see the rewrite in `next.config.ts`.)

## Run & build

Package manager is **pnpm**.

```bash
pnpm install
pnpm dev            # Next.js dev server → http://localhost:3000
pnpm build          # production build (same build Vercel runs)
pnpm start          # serve the production build locally
# Backend (only when working on the chat API):
cd backend && uvicorn app.main:app --port 8000 --reload
```

## Deploy

- **Frontend → Vercel.** Project `aol-frontend` is **git-connected** to this repo: a push to `master`
  auto-deploys to production (alex-loftus.com) in ~15s, no manual step. Manual fallback:
  `pnpm dlx vercel@latest deploy --prod`.
- **Backend → Fly, via GitHub Actions.** `.github/workflows/deploy.yml` runs on push to `main`/`master`:
  `flyctl deploy --remote-only` for app `llm-resume-restless-thunder-9259`, then health + `/chat`
  smoke tests. So a `master` push ships BOTH the site (Vercel) and the backend (Fly).
- GitHub Pages is bypassed (apex DNS → Vercel). Ignore it.

## Structure

```
app/                 Next.js App Router. Root layout + (site) route group (SiteHeader/Footer).
                     Routes: / (home tabs), /cv, /publications, /year-archive, /posts/[...slug],
                     /account (GitHub OAuth), /feed.xml, /api/auth/*, /api/api-token.
content/             Markdown SOURCE the app renders: home-about.md, publications.md, posts/*.md.
lib/content.ts       Markdown → HTML pipeline (unified/remark/rehype: gfm, math+katex, highlight,
                     slug+autolink); gray-matter frontmatter. Runs at build time (static).
components/          React components (SiteHeader, HomeTabs, ChatWidget, PdfEmbed, PuzzleForm, …).
public/              Served verbatim: assets/{css,js,data,fonts}, files/ (cv.pdf, papers), images/,
                     favicon.svg, and the assembled static pages below.
public/_networks/    4 assembled D3 coauthorship pages (from _pages/coauthorship*.html).
public/_perfumes/    2 assembled perfume-atlas pages (from _pages/perfumes*.html).
public/houses/       Self-contained Leaflet rental scout.  public/talkmap/  Leaflet talk map.
public/klist/        Partner preferences checklist (/klist, unlisted+noindex like /sti) + its
                     bearer-gated viewer (/klist/admin; password = backend LOG_ACCESS_TOKEN).
                     Submissions POST to the Fly backend (klist_submissions table on the volume,
                     never in this repo). index.html and admin.html share item data by hand-sync.
next.config.ts       trailingSlash + rewrites (clean URLs for the public/ static pages + the
                     /red-teaming proxy) + redirects (old Jekyll URL parity).
scripts/             Build/util scripts (see below).
backend/             FastAPI chat API, deployed to Fly. See backend/README.md.
experiments/         Python library the backend imports (fold logic, RAG, resume/system_prompt).
_pages/ _posts/      LEGACY Jekyll. Unused by Next EXCEPT the sources the build scripts read
                     (_pages/coauthorship*.html, _pages/perfumes*.html). content/ is canonical for
                     everything else — editing _posts/ does NOT change the live blog.
```

## Scripts (`scripts/`)

- `build_networks_html.mjs` — assemble `public/_networks/*.html` from `_pages/coauthorship*.html`.
  Re-run after editing those sources: `node scripts/build_networks_html.mjs`.
- `build_perfumes_html.mjs` — assemble `public/_perfumes/{atlas,analyses}.html` from
  `_pages/perfumes*.html` (wraps each body in the fullscreen / bare shell). Re-run after editing
  those sources: `node scripts/build_perfumes_html.mjs`.
- `fetch_scholar.py` / `update_scholar_cron.sh` — Google Scholar sync. ⚠ STALE: writes the old
  `_pages/about.md`, NOT `content/publications.md`, so it no longer updates the live site. Fix the
  target before relying on it. Run via `uv run scripts/fetch_scholar.py`.

## Adding a static sub-page (the pattern)

1. Put the self-contained page under `public/<name>/`; shared assets under `public/assets/`.
2. Add a rewrite in `next.config.ts`: `{ source: "/<name>", destination: "/<name>/index.html" }`.
3. If it's assembled from a `_pages/*.html` source, add `scripts/build_<name>_html.mjs`
   (mirror `build_perfumes_html.mjs`) and commit the generated `public/` output.
4. `pnpm build` to verify. Old Jekyll top-level folders (no `public/` + rewrite) will 404.

## Interactive maps

- `/networks/` — coauthorship graph. `_pages/coauthorship*.html` → `public/_networks/` via
  `build_networks_html.mjs`.
- `/perfumes/` (+ `/perfumes/analyses/`) — perfume similarity atlas, 24k bottles by smell.
  `_pages/perfumes*.html` → `public/_perfumes/` via `build_perfumes_html.mjs`; renderer
  `public/assets/js/perfumes-atlas.js`; data `public/assets/data/perfumes-{atlas,neighbors}.json`
  (built by `perfumes/build_atlas.py`). See `perfumes/README.md`.

## Backend API (backend/)

- `backend/app/` — FastAPI package, Cerebras `zai-glm-4.7`, ChromaDB RAG, **SQLite on the Fly volume**
  (durable rate-limits + cost caps + event-sourced crowd-edit logs). Imports the pure fold logic +
  RAG module from `experiments/`. Full architecture + cutover runbook: `backend/README.md`.
- Fly.io app: `llm-resume-restless-thunder-9259` (build = `backend/Dockerfile`).
- Deploy: `fly deploy` (entrypoint runs `alembic upgrade head`), or auto via the Actions workflow on
  a `master` push.
- Endpoints: `POST /chat` (streaming), `POST /reset`, `GET /health`, `/coauthorship/*` +
  `/affiliations/*` crowd edits.
- Auth: NextAuth (GitHub) in the frontend; `/api/api-token` mints a short-lived JWT the backend verifies.
- $2 USD cost cap per conversation (durable across restarts).

## Package management

- **JS**: pnpm (`pnpm install`, `pnpm dlx …`)
- **Python** (backend + scripts): `uv` (`uv run <script>.py`)
