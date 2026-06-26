# Co-authorship network (`/networks/`)

Builds the data behind the interactive graph at
[alex-loftus.com/networks](https://alex-loftus.com/networks/).

Each researcher's data is **cross-referenced against both Semantic Scholar (S2) and OpenAlex**:
identity is validated against both indices, papers are unioned for coverage, and every node/edge is
tagged with which source(s) attest it (shown in the page's tooltips + legend).

## Run

Two steps: resolve identities **once** (hand-verified), then build the graph from the pins.

```bash
cd experiments/coauthorship
uv run build_seeds.py        # resolve names -> S2 + OpenAlex authors, cross-check, writes seeds.json
#   ... then hand-verify seeds.json (see below) ...
uv run build_graph.py        # union papers from pinned IDs -> ../../assets/data/coauthorship.json
```

Raw API responses are cached under `raw/` so re-runs are instant and offline. Set `S2_API_KEY` in
the env for higher S2 rate limits (otherwise the shared pool 429s and the scripts just back off);
OpenAlex uses the polite pool (`mailto`). Re-running `build_seeds.py` **preserves** an entry's
hand-verified S2 side and only re-computes the OpenAlex side + crosscheck.

The page (`_pages/coauthorship.html` + `assets/js/coauthorship-network.js`) loads that static
JSON — no live API calls at view time.

## Verifying `seeds.json`

`build_seeds.py` auto-picks each person's **dominant** profile in **each** index (both fragment one
author across several profiles, and OpenAlex in particular sometimes over-merges a same-named
stranger). It writes everything you need to check the picks — `affiliation`, `orcid`, paper count,
`sample_titles` / `oa_titles` (their top-cited papers, the strongest "right person?" signal), the
runner-ups in `alternatives` / `oa_alternatives` — plus a **`crosscheck`** block comparing the two:

- `crosscheck.agree == false` → S2 and OpenAlex share no ORCID and no top-cited paper, so they may
  have landed on **different people**. Scrutinise these first.
- `crosscheck.orcid_conflict == true` → both picks have an ORCID and they differ: one is wrong.

For each entry:
- Do `affiliation` / `sample_titles` / `oa_titles` match the person you mean? Click `url` / `oa_url`.
- If **wrong**: replace `s2_id` / `oa_id` (+ url) — often the right profile is already in
  `alternatives` / `oa_alternatives`, else paste the id from their profile page.
- To fold a fragmented profile in, add its id to `merge_ids` (S2) / `oa_merge_ids` (OpenAlex).
- Set `"verified": true` (S2) / `"oa_verified": true` (OpenAlex) once checked. **`oa_verified`
  pins the OpenAlex id** so re-runs don't re-auto-pick it, and force-trusts it for the union.
- `s2_id: null` (no S2 profile, e.g. no publications) still appears in the graph as an **open
  (hollow, dashed) circle** — a distinct colour from people who *have* papers but no co-author shown
  here. Such people are also listed under `unresolved` and carry no edges or paper popup. `oa_id:
  null` just means no OpenAlex coverage for that person (S2-only). A `null` s2_id with
  `"verified": true` is a hand-checked "no academic profile" — `build_seeds.py` preserves it instead
  of re-auto-picking a same-named stranger. Symmetrically, a `null` oa_id with `"oa_verified": true`
  is a hand-checked "no OpenAlex profile" (the dominant OpenAlex pick was a homonym — e.g.
  `daniel brown` the geographer, `cat khor` the pharmacology researcher); re-runs keep the OA side
  empty instead of re-adopting the stranger. Regression: `tests/test_seed_preserve.py`.
- **`"exclude_titles": [...]`** — case-insensitive substring blacklist (matched on the normalized
  title) for dropping off-topic papers a same-named stranger contributed to an *otherwise-correct*
  profile (S2/OpenAlex author disambiguation merges homonyms). Applied before the union, so excluded
  papers affect neither edges nor the popup. e.g. `devin crowley` drops a civil-engineering
  "rammed Earth" paper while keeping his real neurodata + physics work.
- **`"oa_reject": true`** — force-*withhold* the OpenAlex side even when `crosscheck.agree`. For a
  profile whose OpenAlex id over-merges a homonym you can't fix by re-pinning (e.g. `jeremy howard`,
  whose OpenAlex blob injects a 1983 cheetah-biology paper + tourism articles). Keeps that person
  S2-only.

### The safety gate (why some people are S2-only)

`build_graph.py` only unions a person's OpenAlex papers when the OpenAlex pick is **trusted** —
i.e. `crosscheck.agree` (shared ORCID or shared top paper) **or** you set `oa_verified`. An
unverified, disagreeing OpenAlex pick is *withheld* (that person stays S2-only, and the build prints
who). This keeps an over-merged homonym — e.g. a same-named geographer's 91 papers — out of the
graph. To bring a withheld person's OpenAlex coverage in: fix their `oa_id` to the right profile
(`agree` flips true on rebuild) or set `oa_verified: true`.

## What the page does

The ~49 listed researchers are **anchors** and are **all shown from hop 1** (the roster is the point
of the page). A "Reach" slider reveals how they connect — i.e. which *outside* people fill in between:

- **1 hop** — everyone listed; lines drawn only for direct co-authorships *between listed people*.
- **2 hops** — plus the outside co-author who bridges two listed people.
- **3 / 4 hops** — plus chains through two / three intermediate people.

A listed person with no co-authorship visible at the current hop is anchored (via a never-drawn
"ghost" link) to its nearest fellow listed person — or, for the no-paper people, to the network hub —
so the roster stays one tidy cluster instead of scattering. The hover popup lists a person's papers
**most-cited first**, with each paper's citation count, de-duplicated across S2/OpenAlex + version
variants.

Each node carries `minhop` = the length of the shortest listed-pair path that first brings it
in; the UI shows everything with `minhop ≤ slider`. Clicking a name in the side list raises the
slider just enough to reveal that person, then centres on them. Listed people with no traceable
path to the rest are pinned as a row of isolated avatars below the network.

A **"Shortest path" panel** in the sidebar takes two listed people — type a name into a slot
(type-ahead over the roster) or arm a slot and click anyone in the graph or the side list — and
lights the minimum-hop chain between them. BFS at view time over genuine shared-paper edges:
the drawn links plus the bridge-route edges (real co-authorships pre-computed for isolated
people), never the ghost anchor links. Including bridge edges keeps the finder consistent with
clicking an isolated person — both trace the same chain. The chain is drawn in warm ink with
everything else dimmed, connectors beyond the current Reach are revealed, and a hop-count badge
+ breadcrumb appears in the sidebar. Pairs in separate components (e.g. one endpoint has no
indexed papers) report "no co-authorship path" honestly.
Tests in `tests/`: `uv run test_shortest_path_data.py` (BFS sanity over the built JSON) and
`uv run test_shortest_path_e2e.py` (headless Playwright pass over the built site — run
`bundle exec jekyll build` first).

Each node also carries `shared_papers` = the number of papers it co-authored with at least one
*other node shown here*. This is what the hover tooltip reports — deliberately **not** a raw
profile total, because a merged secondary profile can fold in same-named strangers (which would
inflate a total but not the in-network count, since strangers' papers share no one in this set).
Listed people also carry `papers` (their own titles, newest first, deduped across both sources);
hovering them pops up that list.

### Provenance (cross-referencing in the UI)

Every node and edge carries `sources`: `"both"` (in both S2 and OpenAlex), `"s2"`, or `"oa"`.
- **Solid** edges are corroborated by both indices; **dashed** edges appear in only one.
- The hover tooltip names the source(s); the legend keys the solid/dashed distinction.
- `meta.links_by_source` reports the split (e.g. `both`/`s2`/`oa` edge counts) and `meta.n_oa_trusted`
  how many anchors contributed OpenAlex coverage.

## How the data is built

1. **Resolve** each name to a **hand-verified** S2 author id **and** OpenAlex author id via
   `seeds.json` — no fuzzy search at build time, nothing guessed. `build_seeds.py` proposes both
   picks and a `crosscheck`; you confirm. People with no S2 profile are pinned to `null` and listed
   under `unresolved`.
2. **Fetch + union** papers from **both** sources for each trusted pin (see the safety gate above),
   dedup by DOI → normalized title, union the author lists, and build a **name-keyed** co-authorship
   graph (so fragmented/duplicate profiles, cross-source id namespaces, and `"Last, First"` spellings
   all collapse to one node). Each paper/edge/node records which `sources` attest it. Papers with >
   `MAX_AUTHORS` authors only add list↔list edges, not edges through outsiders (kills mega-paper cliques).
3. **Hop reveal** — for every pair of listed people, take shortest paths up to `K_MAX` hops; the
   union of those path nodes/edges is the shipped graph, each tagged with its `minhop`.
4. **Communities** — degree-corrected spectral clustering (Priebe/NeuroData): a regularized Laplacian
   spectral embedding (R-DAD), row-normalized onto the unit sphere, then a Gaussian mixture with
   `k = 3` named groups → node colour; **`spring_layout`** seeds the force layout. Junk intermediaries
   (single-token / common-name collisions) are dropped first. (Plain adjacency embedding folds
   EleutherAI into the Bau lab on this graph; the degree-normalized Laplacian keeps the three labs apart.)

Because the graph keeps only outside people who sit on a path *between two anchors*, an over-merged
OpenAlex homonym's stray papers are mostly self-pruned (they reach no second anchor) — so the safety
gate above plus this pruning together keep cross-source coverage from polluting the network.

## Self-service corrections (`overrides.json`)

The auto-built graph has real errors about real people. Anyone can fix their corner of it **from the
website** — there's an "✎ Suggest a correction" toggle in the sidebar. Editing is **open & instant**:
clicking a node or edge opens an inline editor, and a save shows immediately and persists. Capabilities:
rename a paper (everywhere), delete a paper, move a person to another group, edit/delete an edge, add a
missing co-authorship, remove yourself from the graph (opt-out), and fix a display name / photo /
profile link.

How it survives the nightly rebuild — three layers, one durable contract:

```
Browser edit ──POST event──▶ Fly API (experiments/chat_api.py)
   page load ◀─GET /networks/overlay─ corrections.jsonl (Fly volume, append-only)
              (merged overlay shown to everyone within seconds)
   nightly:  merge_corrections.py ─fold events─▶ overrides.json (git)
             build_graph.py ─apply_overrides()─▶ coauthorship.json (git) ─▶ commit
```

- **`overrides.json`** — the git-tracked correction contract (keys: `node_label`, `node_community`,
  `node_url`, `node_photo`, `remove_nodes`, `paper_rename`, `remove_papers`, `add_papers`,
  `remove_edges`). `overrides.py::apply_overrides()` replays it onto the **finished** graph as the last
  build step (so it never perturbs the graph/hop/clustering algorithms), and the page's
  `applyOverlay()` is the JS mirror — **keep the two in sync**.
- **`merge_corrections.py`** — pulls the merged overlay from the Fly API into `overrides.json`.
  `uv run merge_corrections.py` (live, **no token needed** — it reads the open overlay endpoint) or
  `--from file.jsonl` / `--dry-run` (offline) or `--raw` (fold the Bearer-protected raw log; needs
  `COAUTHOR_TOKEN` = the API's `LOG_ACCESS_TOKEN`). Folding is last-write-wins per key with ts-aware
  undo (`overrides.py::fold_events`).
- **Backend** (`experiments/chat_api.py`): `POST /networks/corrections` (open, rate-limited),
  `GET /networks/overlay` (open, merged), `GET /networks/corrections` (Bearer, raw audit log),
  `DELETE /networks/corrections?ts=<ISO>` (Bearer, admin revert).

**Reverting a bad/vandal edit:** delete its event from the API log — that fixes the live overlay AND
every future bake in one call:

```bash
curl -X DELETE -H "Authorization: Bearer $LOG_ACCESS_TOKEN" \
     "https://llm-resume-restless-thunder-9259.fly.dev/networks/corrections?ts=<event ts>"
```

(Find the `ts` in the raw log via the Bearer `GET`.) A git revert of `overrides.json` alone is **not**
durable — the nightly merge regenerates that file wholesale from the event log. Opt-out
(`remove_nodes`) is honoured immediately and stickily. Curated build-time identity fixes
(`NAME_ALIAS`, `COLLISION_STOP`, `EDGE_DROP`) stay in `build_graph.py` — they act *mid-build* on the
homonym/fragment problem, which a post-build override can't express.

Tests: `tests/test_overrides.py` (apply, per override type), `tests/test_merge_corrections.py` (fold:
LWW + idempotency + undo), `experiments/tests/test_corrections_api.py` (endpoints).

```bash
cd experiments/coauthorship && uv run --with pytest --with click --with httpx pytest tests/
```

**To activate the live flow (one-time ops):**
1. Deploy the chat API so the new endpoints exist: `cd experiments && fly deploy`.
2. Extend the daily routine (`trig_017nGSZYzfesXXwuSWJc2gwL`) to run `uv run merge_corrections.py`
   **before** `uv run build_graph.py`, and stage **both** `overrides.json` and `coauthorship.json` in
   its commit. No secret needed (the default merge reads the open overlay endpoint).
   `merge_corrections.py` is fail-safe: if the API is unreachable it warns and leaves
   `overrides.json` untouched, so the build never breaks.

## Avatars / photos

Nodes are **monogram avatars** (initials on the community colour) by default. To use real
photos, drop image files in `assets/images/coauthors/` named `<slug>.jpg` (or `.png`/`.webp`),
where `slug` is the lowercase name with spaces → hyphens (e.g. `can-rager.jpg`), and re-run the
build — `photo_url()` picks them up and the UI fills the node (falling back to the monogram if an
image fails to load). *(There is no WhatsApp/contacts integration; photos must be supplied as files.)*

## Knobs (top of `build_graph.py`)

| constant | meaning |
|---|---|
| `K_MAX` | longest listed-pair path traced (= max slider hops) |
| `PATHS_PER_PAIR` | shortest paths kept per pair |
| `MAX_AUTHORS` | papers bigger than this only add list↔list edges |
| `COMMON_STOP` | common names dropped to avoid collision false bridges |
| `COLLISION_STOP` | connectors whose single profile conflates two real people (verified) — dropped |
| `NAME_ALIAS` | fold a verified non-anchor fragment into its twin by name (e.g. `c priebe`→`carey e priebe`) |
| `EDGE_DROP` | anchor pairs with no real co-authorship (verified) — removed (and never re-added) |
| `COMMUNITY_LABELS` | anchor person → (fixed community id/colour, legend label) for the 3 named groups |
| `COMMUNITY_FORCE` | editorial override: force a person into another anchor's community (e.g. `kola ayonrinde`, `jesse hoogland` → Bau interp group, not EleutherAI) |
| `EMBED_DIM` | Laplacian spectral embedding dimension fed to the GMM (pinned; robust over 4–8) |

## Caveats

- Anchor identity is pinned + verified in `seeds.json`; **outside** intermediaries are still keyed
  by normalised name, so a stoplist (`COMMON_STOP`) guards against collisions but exotic ones slip.
- We pin each person's **dominant** profile per index (precision over recall): a few papers living
  only in a fragmented secondary profile are missed unless you add that id to `merge_ids` /
  `oa_merge_ids`.
- OpenAlex coverage is **gated** on `crosscheck.agree`/`oa_verified` (see the safety gate). People
  whose OpenAlex pick disagrees and isn't hand-verified are S2-only — the build prints the list.
- **Edge weights are fractional co-authorship counts** (suggested by Stella Biderman; cf. Newman
  2001's 1/(n−1) variant): each distinct paper contributes `1/n_authors` to every co-author pair
  on it, so a two-person paper binds ~15× tighter than a 30-author one and mega-collab papers
  can't dominate the layout, clustering, or analyses. Every link also keeps an integer `n_papers`
  for human-readable display. Cross-source paper dedup joins on DOI, else normalized title; an
  audit pass then collapses remaining **version variants** per edge (`fractional_weight()` —
  preprint/published with different subtitles, "...Open-Weight..." retitles) so each distinct
  paper counts once. An independent cross-check (OpenAlex/ORCID/DBLP/arXiv) also seeded `NAME_ALIAS`,
  `COLLISION_STOP`, `EDGE_DROP`, and the `can rager`/`chris wendler` `merge_ids`. Note: S2 author
  `7689277` ("Ronak Mehta") conflates two researchers and can't be split within S2; its edges are
  nonetheless correct.
- `COMMUNITY_LABELS` uses S2 name spellings (e.g. `j vogelstein`) to pin each named group's colour and
  legend label; clusters are keyed to whichever GMM component each anchor lands in. The build asserts
  the three anchors separate into three clusters, so a data shift that collapses them fails loudly.
- Reflects Semantic Scholar **and** OpenAlex coverage at build time.

## Affiliations tab (`/networks/affiliations/`)

The co-authorship graph structurally excludes chat members with no publications. The fix is the
**shared-affiliation bipartite graph** (people ↔ orgs) on its own tab, where employment,
communities, and programs count as much as papers. Its source of truth is `affiliations.json` —
hand-reviewed, NOT derived from S2/OpenAlex (whose affiliation strings are often homonym artifacts).

```
affiliations.json ──build_affiliations.py──▶ assets/data/affiliations.json
  (source of truth,    CANON: org-string variants -> one canonical (label, type) —
   per-person entries)  the org identity layer, analogue of seeds.json. Joins people to
                        coauthorship.json for label/community colour. Deterministic.
_pages/coauthorship-affiliations.html + assets/js/coauthorship-affiliations.js   the view
assets/css/coauthorship-page.css   chrome shared with /networks/ (analyses page diverges)
_includes/coauthorship-tabs.html   the map|affiliations|analyses nav (all three pages)
```

Two views: **people & orgs** (bipartite force layout; org label size = member count; org colour =
type; filter by type, single-member orgs hidden by default with a count) and **people only** (the
projection: ties weighted by shared orgs, lab 3 / program & company 2 / community & university 1,
**nesting-discounted** via `PARENT` — a pair sharing a lab AND the university it sits in counts
only the lab, since the degree usually came with the lab; e.g. NeuroData's 10 members all also
hold JHU degrees).
Click anything for the sidebar detail — every person row links its public source. Deep link:
`#p=<person id>`. Tests: `uv run tests/test_affiliations_data.py` (determinism, link/projection
invariants, and that the Bau-lab/NeuroData org-string variants actually merged). After editing
`affiliations.json` or `CANON`, re-run `uv run build_affiliations.py`.

- Built initially by a parallel web sweep (one research agent per person + an adversarial
  source-check pass); every entry carries a `source` URL and `verified: false` until reviewed.
  Sweep result (2026-06-10): 46/48 people with sourced entries (228 entries, all http-sourced);
  the 2 stubs are `daniel brown` and `cat khor`, whose public footprints are homonym-only —
  fill those by hand via the CLI.
- Schema: `{name: {entries: [{org, type, role, years, current, source, evidence, verified}],
  city, identity_confident, notes, reviewed}}` with `type ∈ lab | university | company | community
  | program`.
- **Review/fill tool**: `uv run fill_affiliations.py` — walks people stub-first (accept/edit/
  delete/add per entry, saves after every person); `--stats` prints coverage; `--person "name"`
  jumps to one person; `--all` revisits reviewed people.

### Off-map hop layer (`build_hops.py` → `assets/data/affiliations-hops.json`)

The reach slider's *steps* mode reveals people OUTSIDE the map who verifiably shared a room
with a member — hollow dashed nodes, the careers-map analogue of the papers map's hop reveal.
The iron rule: a hop link needs a **co-event**, never a shared label alone (LinkedIn data is
off-limits; this stack replaces it with legitimate sources).

- `openalex` (no network — mines the `raw/oa_works_v2_*` cache): an outside co-author who
  listed the same mapped institution as a member on the same paper. Room + moment + artifact.
- `github` (`uv run crawl_github_hops.py`, public API via `gh auth token`): members resolve to
  handles by name search, accepted only with profile corroboration (mapped-org match in
  company/bio, or homepage match) → `hop_sources/github_handles.json` for hand review. Two
  co-events, merged one-link-per-(person, room): **co-contribution** (member and outsider both
  committed to one of the org's top-starred repos — the stronger signal, catches people who
  never set public membership) and **public org membership** (≤200 public members — `microsoft`
  at ~4500 is noise). Bot accounts filtered. Closed companies (openai, anthropic, scaleapi…)
  honestly yield nothing here — their work is in private repos; the OpenAlex leg covers them.
- Identity: outside people fold by name (middle initials dropped) across OA ids and sources;
  anyone matching a member name is excluded. Fan-out capped at top-10 per (member, room).
- A hop person shows when a vouching member is strictly inside the slider's reach, so widening
  the reach walks member→member steps and fringes each step with the people they'd know.
- Hop nodes are clickable: detail panel shows the rooms, vouching members, co-event evidence,
  and a profile link; **"add them to the map →"** prefills the join form with their vouched
  rooms (the recruitment funnel). The reveal stays anchored to the last selected person, and a
  successful join drops the person's hop node client-side (build_hops re-applies server-side).
- Extra sources drop into `hop_sources/*.json` (same schema, `src` tag) and merge on rebuild.

### Guest finder ("search for a person near this graph")

The search bar over the careers map tries ANYONE on the map temporarily. People already here
(members/guests) match first; otherwise OpenAlex's CORS-open API resolves the name
(`/autocomplete/authors`, exact-name match outranks works count, disambiguation hints shown),
and the author's `affiliations` record is placed as a dashed **guest node** wired through the
same org nodes as everyone else — projection ties to members form automatically, so the reach
slider immediately answers "what are they up to relative to us". Institutions map to existing
orgs via `meta.inst2org` in affiliations-hops.json (the JS mirror of build_hops's resolver);
unmatched ones mint ghost orgs (hidden until 2 members unless "show single-member orgs").
Guests are loud about provenance ("guest preview · OpenAlex, unverified" — OpenAlex affiliation
records contain extraction artifacts) and have two exits: **dismiss**, or **add to the map →**
which hands the prefilled entries to the normal join flow. Nothing persists without the join.

The finder is on BOTH maps. The papers map draws members and their outside coauthors already,
so its locals cover both (clicking auto-bumps the hop slider to reveal a hidden coauthor);
strangers hand off to the careers map's guest engine via `?guest=<OpenAlex id>`. Between locals
and OpenAlex sits **`guest-index.json`** (built by build_hops.py): the extended social graph —
every outside coauthor in the works cache ranked by papers shared with members (top 300), with
resolver-canonicalized entries baked in, so the people most likely to be searched for place
instantly and still merge the live OpenAlex record when reachable.

## Page URLs

Everything lives under **/networks/** (papers map at `/networks/`, careers at
`/networks/affiliations/`, analyses at `<base>analyses/`); the old `/coauthorship/…` URLs
redirect (jekyll-redirect-from). **Clean person URLs**: `/networks/<first-last>/` (e.g.
`/networks/can-rager/`) — static stub pages generated by `build_affiliations.py` (one per
person, og: tags for link previews) that forward to the person's seat; unknown slugs are caught
by a router in the 404 page (covers brand-new joiners whose stub isn't committed yet — the
nightly bot's staging list doesn't yet include `networks/`, so joiner stubs ship on the next
human commit; the 404 router covers the gap).

## The person registry (adding networks later)

`roster.json` + `registry.py` are the single source of truth for WHO is on the maps
(`core` = Alex's hand list; `self_joined` = machine-owned by merge_affiliations.py).
**Contract for every network over this node set, including future ones (chat-interaction,
"how we met", …): derive membership from the registry**, via
`registry.reconcile_membership(records)` — roster people with no data in your network get an
honest empty state (the affiliation build ships them as empty seats; the papers build asserts
roster ⊆ seeds). That is what makes "add a person once, they appear everywhere" true by
construction: one `aff_join` event → roster + seeds stub at the next bake → every network and
every panel includes them automatically.

## Self-service membership (edit my row / join the map / your seat)

The affiliation pages are self-service. Events POST to the Fly API and fold into an overlay;
the hand-curated `affiliations.json` is NEVER machine-written:

```
browser ──POST /affiliations/corrections──▶ Fly API (affiliation_events.jsonl, append-only)
  page load ◀─GET /affiliations/overlay── folded events (live preview for everyone, 4s timeout)
  nightly:  merge_affiliations.py ─▶ affiliation_overrides.json (git) + self_joined seed stubs
            build_affiliations.py load_src() applies the overlay IN MEMORY over affiliations.json
```

- Event types: `aff_entry_set/remove` (own-row CRUD, keyed by whitespace-folded org string),
  `aff_city`, `aff_join` (≤10 entries; creates a `self_joined` seeds stub + roster entry so
  build_graph mints a hollow node), `aff_confirm` (info-check freshness signal). Fold/apply live
  in `affiliation_events.py` (pure, LWW, ts-aware undo — mirror of overrides.py).
- The map page applies the overlay at load AND optimistically after each save (no reload — the
  node visibly re-wires); new raw org strings render as their own nodes until you add a CANON
  line (the build prints them). Self-edits ship with `source: "self-reported via site (date)"`,
  `verified: false`.
- **Revert** (vandal or mistake): find the ts in the Bearer raw log, `DELETE
  /affiliations/corrections?ts=…` — the next nightly regenerates everything without it,
  including removing the seeds stub. **Promotion**: copy an overlay edit you endorse into
  `affiliations.json` by hand, then DELETE its events (else they LWW-shadow future hand edits).
- The roster moved to `roster.json` (`core` = hand list; `self_joined` = machine-owned);
  build_seeds.py reads both.
- Identity: `network-identity.js` greets first visits on the /networks/* pages ("who are
  you?" → quick info check → `aff_confirm`), stores `{id,label}` in localStorage, and defaults
  the analyses page to that person's seat (`?p=…#your-seat`). Skippable, never nags.
- Tests: `tests/test_affiliation_events.py` (fold/apply/stub sync + the vandal round-trip),
  `experiments/tests/test_affiliations_api.py` (endpoints), `tests/test_affiliations_edit_e2e.py`
  (Playwright, page.route-mocked API). ⚠️ API endpoints go live only after `cd experiments &&
  fly deploy` (Alex-gated); pre-deploy the frontend degrades gracefully.

## Career analyses (`/networks/affiliations/analyses/`)

Seven panels over the affiliation network — six methods plus `your-seat` (the People capstone: a per-person digest of all six, deep-linkable via `?p=<id>#your-seat`) — orthogonal in method to the paper page's twelve:
same-rooms-no-paper (two-graph edge overlap — the flagship), eras (Lexis cohort swimlanes),
the-pipeline (first-order Markov over org-type sequences), where-we-are-now (cross-sectional
census from `current` flags + city metros), range (Hill-number type-diversity), embassies
(bipartite degree heavy tail). Same CONTRACT.md rules; panel files live in their own namespace:

```
experiments/coauthorship/analyses-affiliations/<slug>.py  -> assets/data/analyses-affiliations/<slug>.json
assets/js/analyses-affiliations/<slug>.js                 (IIFE, registers into AnalysesRegistry)
```

The analyses **shell is now config-driven**: each analyses page sets `window.ANALYSES_CONFIG`
({methods, sharedPath, panelDataDir, minimap}) in a plain script before `shell.js?v=2`; defaults
reproduce the papers page. The careers page's minimap reuses the co-authorship layout for the
same 48 people (`restrictToShared` + projection edges from `analyses-affiliations/shared.json`,
emitted by `build_affiliations.py`). Site nav is two-level (networks × views) via
`_includes/coauthorship-tabs.html` + `_data/network_tabs.yml` — a future network is one yml line
plus its two pages. Tests: `tests/test_affiliations_data.py` (incl. `test_panels_fresh`, which
re-runs all six panels so their headline asserts re-fire after any data edit) and
`tests/test_nav_e2e.py` (Playwright: nav pills, view-preserving links, every panel on both
analyses pages activates). After editing affiliation data: `uv run build_affiliations.py`, then
re-run the six panel scripts.

## Analyses page (`/networks/analyses/`)

Twelve graph-statistics panels over the same network, each doubling as a plain-language method
explainer for ML readers. Architecture (everything under `analyses/`):

```
analyses/_prep.py      shared derivation: re-keys per-paper records from the raw cache with the SAME
                       vendored identity logic as build_graph.py (keep-in-sync header), then emits
                       _derived/{papers,yearly,layers,tfidf}.json (committed, never published) and
                       ../../assets/data/analyses/shared.json (browser lookup). Gate: re-derived link
                       weights must match the shipped graph >=95% (currently 100%).
analyses/<slug>.py     one deterministic compute script per panel (PEP-723; uv run <slug>.py)
                       -> assets/data/analyses/<slug>.json  ({slug,title,headline,data}, <300KB)
assets/js/analyses/    shell.js (registry/toolbar/minimap/tooltip; orchestrator-owned)
                       + one IIFE render module per panel (registers into AnalysesRegistry)
analyses/CONTRACT.md   the binding contract for panels (envelope, JS pattern, voice, audience)
_pages/coauthorship-analyses.html   the shell page (toolbar sections, prose slots, all script tags)
```

**To add a panel**: pick a slug; add it to `METHODS` in `shell.js` and a `<script>` tag in the page;
write `analyses/<slug>.py` (envelope per CONTRACT.md) and `assets/js/analyses/<slug>.js`
(registration per CONTRACT.md); run the .py twice (byte-identical), `node --check` the .js.
A panel that fails to load degrades to a per-panel notice; others are unaffected.

**Audience note**: the people on the map read these pages (see CONTRACT.md "Audience & voice") —
findings name people as compliments/curiosities, database artifacts are never framed as a person's
fault, and method citations live in the collapsible "For the curious" footnotes (several methods
were invented by people on the map: Priebe, Vogelstein, Bridgeford, Mehta).
