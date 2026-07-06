# Front-page project previews ("Projects, live")

**Date:** 2026-07-05
**Status:** Approved (brainstorm 2026-07-05, session "wbsite-frontpage")

## Goal

The front page links to /houses, /jobs, and /networks but gives no sense of what they are. Add a strip of three miniature, genuinely interactive previews — Tufte-style small multiples fed by the apps' real data — so a visitor gets a feel for each app without leaving the home page.

Approaches considered and rejected: scaled live iframes (1.5 MB+ page weight, unreadable at 30% scale, scroll traps, networks onboarding modal appears in miniature) and static screenshot cards (not interactive, which was the point).

## Placement & layout

- Inside the **About tab**, between the intro paragraph ("Hi, I'm Alex Loftus…") and the "fun projects" list.
- Small-caps section label: **Projects, live**.
- Three cards side-by-side within the 44rem column (~13.5rem each); stack vertically under 640px.
- Card anatomy: small-caps title → interactive mini (~11rem tall) → one-line stat footer from real meta → "open →" link. The whole card navigates to its page.
- Styling uses existing `app/tokens.css` variables only (paper-raised surface, hairline rules, oxblood accent, Newsreader/Fraunces). No new fonts, colors, or dependencies.

## Data flow: build-time extraction

New script `scripts/build_previews.mjs` (same pattern as `scripts/build_networks_html.mjs`), run as an npm `prebuild` step so every Vercel build regenerates it.

**Inputs** (committed artifacts, always present):
- `public/houses/data.js` — `window.HOUSES_DATA` (~106 KB)
- `public/jobs/data.js` — `window.JOBS_DATA` (~1.3 MB)
- `public/assets/data/coauthorship.json` (~320 KB; 132 nodes, 368 links)

**Output:** `lib/previews.json` (~8 KB, generated and gitignored — the committed data artifacts stay the single source of truth). The script runs as both `prebuild` and `predev` so a fresh clone's `next dev` never hits a missing import.
- `houses`: top ~25 listings by fit — `{lat, lon, fit, price, hood}` — plus meta `{n_scouted, price_min, price_max, price_med, generated}`.
- `jobs`: 6 newest open roles — `{company, title, comp, date}` — plus open-count per lab and meta `{open, companies, generated}`.
- `networks`: all nodes `{id, name, initials, group}` and links `{source, target}` (communities preserved for color).

The home page server component imports `previews.json` statically and passes slices as props. Zero client-side fetches; the big data files are never loaded by the front page.

**Freshness:** houses/jobs refresh actions commit new data.js → push → Vercel rebuild → previews regenerate. No new pipeline.

**Failure mode:** the script parses each input and asserts the shapes it needs (fields present, non-empty arrays, numbers where expected). Any drift **fails the build loudly**. These are committed artifacts that should always be valid; a silent stale/empty preview is worse than a red build.

## Components

`components/previews/` — one shared card shell + three client components, plain SVG, **no new npm dependencies** (force layout is a ~50-line hand-rolled simulation, not D3).

1. **HousesMini** — listings as dots in lat/lon space (no basemap): hairline frame, two commute anchors as ▲, dots sized/colored by fit using the same categorical logic as the real map. Hover → tooltip (neighborhood · price · fit). Footer: "1,122 scouted · $800–$2,856 · med $1,682".
2. **JobsMini** — 6 newest roles: company pill (lab colors from the jobs page), truncated title, comp when present. Below, a spark-bar row of open roles per lab. Footer: "1,552 open · 8 labs · refreshed daily".
3. **NetworksMini** — live force layout of the real co-authorship graph, nodes colored by community, draggable; hover → name label. Footer: "132 researchers · 368 co-authorships".

Footer numbers come from meta, never hard-coded.

**Interaction rules:**
- Desktop: hover for tooltips; drag for graph nodes (drag/tooltip handlers stop propagation so they don't trigger navigation); any other click navigates.
- Touch: tap shows tooltip; navigation only via the explicit "open →" link.
- `prefers-reduced-motion`: force sim renders pre-settled; no animation anywhere.
- Minis assert non-empty props at render.

## Content insertion

`content/home-about.md` splits into `home-about-intro.md` (through the intro paragraph) and `home-about-rest.md` (fun-projects list onward). The About tab renders intro → `<ProjectPreviews/>` → rest. No MDX machinery.

## Testing

Software mode: TDD per superpowers.

- **Extraction script:** real tests — fixture data.js/json → expected slice; malformed/missing fields → throws. This is the highest-drift-risk unit.
- **Components:** render tests with fixture props (non-empty output, correct counts); assert-on-empty behavior.
- **Visual verification:** build + inspect the strip at desktop and mobile widths before handoff.

## Decisions made (flag if wrong later)

- Jobs mini shows *newest* roles, not top-fit (feels alive; fit is personal).
- Houses mini has no coastline/basemap — pure dot-field (Tufte; avoids a basemap rabbit hole).
- `previews.json` is generated at build and gitignored, not committed (single source of truth is the data artifacts).
- v2 candidate (out of scope): click-to-expand live iframe lightbox per card.
