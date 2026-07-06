# Front-Page Project Previews ("Projects, live") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strip of three miniature interactive previews (houses map, jobs ticker, co-authorship graph) to the About tab of alex-loftus.com's front page, fed by build-time slices of the apps' real data.

**Architecture:** A build-time script (`scripts/build_previews.mjs`) parses the three committed data artifacts and emits a small gitignored `lib/previews.json`; the home page server component reads it with `fs` (same pattern as `readContent`) and passes slices into three small components. Pure logic (extraction, spring relaxation) lives in `.mjs` modules tested with Node's built-in `node --test`; components are verified by the type-checked `next build` plus visual inspection.

**Tech Stack:** Next.js 15 App Router, React server + client components, CSS Modules, plain SVG. Node 22, pnpm. **Zero new npm dependencies** (test runner is `node --test`, built in).

**Spec:** `docs/superpowers/specs/2026-07-05-frontpage-previews-design.md`

## Global Constraints

- No new npm dependencies, runtime or dev. Force layout is hand-rolled; tests use `node:test` + `node:assert/strict`.
- All styling uses existing `app/tokens.css` variables (`--paper-raised`, `--rule`, `--accent`, `--muted`, `--ink-soft`, `--radius`, `--shadow-soft`, `--text-xs`, `--text-sm`, `--font-serif-text`); mono stat lines use `ui-monospace, "SF Mono", Menlo, monospace` directly (no mono token exists).
- Extraction fails the build **loudly** on any shape drift (missing fields, non-finite numbers, unresolvable link endpoints).
- `lib/previews.json` is generated + gitignored, never committed.
- pnpm does NOT auto-run `prebuild`/`predev` hooks — extraction runs inline: `"build": "node scripts/build_previews.mjs && next build"`, same for `dev`.
- Section copy: strip label **"Projects, live"**; card titles **Houses / Jobs / Networks**; footer stats always from meta, never hard-coded.
- Work on branch `worktree-frontpage-previews` in this worktree. Commit per task. **Do not push** (background-session git push is gated; Alex pushes himself).
- Run every command from the worktree root: `/Users/alex/loftusa.github.io/.claude/worktrees/frontpage-previews`.

## Verified data facts (do not re-derive)

- `public/houses/data.js` → `window.HOUSES_DATA = {meta, listings}`. meta: `n_scouted:1122, price_min:800, price_max:2856, price_med:1682, generated:"2026-07-05", fit_weights:{alex:{nice,nature,soft,value,commute,aesthetic}}`. Listings (44): `{id, price, pdisp:"$1,950", beds, bucket:"apt", hood, lat, lon, fit:8.4, scores:{nature,quiet,nice,social,value,commute,aesthetic}, gone?}`.
- Houses "driver" (dot color) logic in `public/houses/index.html:541-575`: component that most lifts a listing above the field — argmax over `weight[k] * (score[k] - field_mean[k])`, k ∈ {commute, nice, aesthetic, nature, value, soft}, where soft = quiet for `bucket==="apt"` else social. Driver colors (index.html:15): nature `#4f8a55`, quiet `#5f7189`, nice `#7b6091`, social `#bd8a3a`, value `#3f867e`, commute `#8a8378`, aesthetic `#b0654d`.
- Houses commute anchors (index.html:629-630): Downtown SF `37.7935,-122.3970`; Berkeley · FAR Labs `37.8703,-122.2680`.
- `public/jobs/data.js` → `window.JOBS_DATA = {meta, jobs}`. meta: `open:1552, date:"2026-07-05", companies:[8 labs]`. Jobs: `{company, group:"anthropic"|"interp"|"frontier", title, comp:"$350–500K"|null…, published, first_seen, closed:bool, …}`. "Newest" ordering (index.html:326): `(b.published||b.first_seen||'').localeCompare(a.published||a.first_seen||'')`. Company pill colors (index.html:12,93): anthropic `var(--accent)`, interp `#5f7d6e`, frontier `#6b7a94`.
- `public/assets/data/coauthorship.json`: `{nodes(132), links(368), communities:[{id:0,label:"EleutherAI"},{id:1,label:"David Bau"},{id:2,label:"Joshua Vogelstein"}], meta:{n_nodes:132, n_links:368}}`. Nodes: `{id, label, community, x, y}` — **all 132 have finite x,y in [-1.3, 1.3]** (verified). Links: `{source:id, target:id}` — all endpoints resolve (verified).
- Front page: `app/(site)/page.tsx` reads markdown via `readContent()` + `renderMarkdown()` from `@/lib/content`, renders `<HomeTabs aboutHtml pubsHtml posts>`. About panel (`components/HomeTabs.tsx:63-68`): `<ChatWidget/>` then `<div className="prose" dangerouslySetInnerHTML={{__html: aboutHtml}}/>`.
- `content/home-about.md`: paragraph 1 = intro ("Hi, I'm **Alex Loftus**…Johns Hopkins University."); rest starts at "I've been fortunate…".
- tsconfig path alias: `"@/*": ["./*"]`. Repo has no test framework and no `test` script.

## File structure

```
scripts/build_previews.mjs        # extraction: pure functions + CLI (create)
scripts/build_previews.test.mjs   # node:test suite (create)
lib/force-sim.mjs                 # spring relaxation for networks mini (create)
lib/force-sim.test.mjs            # node:test suite (create)
lib/force-sim.d.ts                # types for TS consumers (create)
lib/previews.json                 # GENERATED, gitignored
components/previews/types.ts      # PreviewsData interfaces (create)
components/previews/ProjectPreviews.tsx        # strip + card shell, server (create)
components/previews/ProjectPreviews.module.css # all strip styling (create)
components/previews/HousesMini.tsx             # client: svg dot map + hover caption (create)
components/previews/JobsMini.tsx               # server: ticker + lab spark-bars (create)
components/previews/NetworksMini.tsx           # client: draggable graph (create)
content/home-about-intro.md       # split from home-about.md (create)
content/home-about-rest.md        # split from home-about.md (create)
content/home-about.md             # DELETE
app/(site)/page.tsx               # read previews.json, pass to HomeTabs (modify)
components/HomeTabs.tsx           # intro/rest htmls + previews slot (modify)
package.json                      # build/dev run extraction; add test script (modify)
.gitignore                        # add lib/previews.json (modify)
CLAUDE.md                         # note the generated-file build step (modify)
```

---

### Task 1: Extraction pure functions (TDD)

**Files:**
- Create: `scripts/build_previews.mjs`
- Test: `scripts/build_previews.test.mjs`

**Interfaces:**
- Produces (used by Task 2's CLI and mirrored by Task 5's `types.ts`):
  - `parseWindowData(src: string, varName: string): object`
  - `extractHouses(data): {meta:{n_scouted,price_min,price_max,price_med,generated}, anchors:[{lat,lon,label}×2], listings:[{lat,lon,fit,pdisp,hood,driver}]}` (≤25 listings, fit-desc)
  - `extractJobs(data): {meta:{open,n_labs,generated}, latest:[{company,group,title,comp,date}×6], byLab:[{company,n}]}`
  - `extractNetworks(data): {meta:{n_nodes,n_links}, communities:[{id,label}], nodes:[{label,community,x,y}], links:[[si,ti]]}`
  - `buildPreviews({housesSrc, jobsSrc, networksJson}): {houses, jobs, networks}`

- [ ] **Step 1: Write the failing test**

Create `scripts/build_previews.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseWindowData, extractHouses, extractJobs, extractNetworks, buildPreviews,
} from "./build_previews.mjs";

// ---------- fixtures ----------
const listing = (over = {}) => ({
  id: "L1", price: 1500, pdisp: "$1,500", beds: 1, bucket: "apt",
  hood: "mission", lat: 37.76, lon: -122.42, fit: 7.0,
  scores: { nature: 5, quiet: 5, nice: 5, social: 5, value: 5, commute: 5, aesthetic: 5 },
  ...over,
});
const HOUSES = {
  meta: {
    n_scouted: 100, price_min: 800, price_max: 2900, price_med: 1700, generated: "2026-07-05",
    fit_weights: { alex: { nice: 0.17, nature: 0.15, soft: 0.13, value: 0.13, commute: 0.26, aesthetic: 0.16 } },
  },
  listings: [
    listing({ id: "L1", fit: 7.0 }),
    listing({ id: "L2", fit: 9.1, hood: "marina", scores: { nature: 9, quiet: 5, nice: 5, social: 5, value: 5, commute: 5, aesthetic: 5 } }),
    listing({ id: "L3", fit: 8.0, gone: true }),
    listing({ id: "L4", fit: 5.5, bucket: "room", scores: { nature: 5, quiet: 5, nice: 5, social: 9, value: 5, commute: 5, aesthetic: 5 } }),
  ],
};
const job = (over = {}) => ({
  company: "Anthropic", group: "anthropic", title: "Research Engineer",
  comp: "$300–405K", published: "2026-07-01", first_seen: "2026-07-01", closed: false,
  ...over,
});
const JOBS = {
  meta: { open: 3, date: "2026-07-05", companies: ["Anthropic", "OpenAI"] },
  jobs: [
    job({ title: "Old", published: "2026-06-01" }),
    job({ title: "Newest", company: "OpenAI", group: "frontier", published: "2026-07-04", comp: null }),
    job({ title: "Closed", published: "2026-07-05", closed: true }),
    job({ title: "Mid", published: "2026-07-02" }),
  ],
};
const NET = {
  meta: { n_nodes: 3, n_links: 2 },
  communities: [{ id: 0, label: "EleutherAI" }, { id: 1, label: "David Bau" }],
  nodes: [
    { id: "a", label: "Ada", community: 0, x: -1.0, y: 0.5 },
    { id: "b", label: "Bo", community: 1, x: 0.2, y: -0.3 },
    { id: "c", label: "Cy", community: 1, x: 1.1, y: 1.2 },
  ],
  links: [{ source: "a", target: "b" }, { source: "b", target: "c" }],
};

// ---------- parseWindowData ----------
test("parseWindowData round-trips the JSON payload", () => {
  const src = `// header comment\nwindow.FOO_DATA = {"a": 1, "b": [2]};\n`;
  assert.deepEqual(parseWindowData(src, "FOO_DATA"), { a: 1, b: [2] });
});
test("parseWindowData throws when the marker is missing", () => {
  assert.throws(() => parseWindowData("var x = 1;", "FOO_DATA"), /FOO_DATA/);
});

// ---------- extractHouses ----------
test("extractHouses sorts by fit desc and drops gone listings", () => {
  const h = extractHouses(HOUSES);
  assert.deepEqual(h.listings.map((l) => l.fit), [9.1, 7.0, 5.5]);
  assert.equal(h.listings.some((l) => l.hood === ""), false);
});
test("extractHouses computes drivers: lifted component wins; soft maps by bucket", () => {
  const h = extractHouses(HOUSES);
  assert.equal(h.listings[0].driver, "nature");   // L2: nature 9 vs field
  assert.equal(h.listings[2].driver, "social");   // L4: room bucket, social 9
});
test("extractHouses keeps meta numbers and both anchors", () => {
  const h = extractHouses(HOUSES);
  assert.equal(h.meta.n_scouted, 100);
  assert.equal(h.anchors.length, 2);
  assert.equal(h.anchors[0].label, "Downtown SF");
});
test("extractHouses throws on empty listings and missing weights", () => {
  assert.throws(() => extractHouses({ meta: HOUSES.meta, listings: [] }), /listings/);
  assert.throws(() => extractHouses({ meta: { ...HOUSES.meta, fit_weights: {} }, listings: HOUSES.listings }), /fit_weights/);
});
test("extractHouses throws on non-finite coordinates", () => {
  const bad = { meta: HOUSES.meta, listings: [listing({ lat: null })] };
  assert.throws(() => extractHouses(bad), /lat/);
});

// ---------- extractJobs ----------
test("extractJobs filters closed and orders newest-first like the jobs page", () => {
  const j = extractJobs(JOBS);
  assert.deepEqual(j.latest.map((x) => x.title), ["Newest", "Mid", "Old"]);
});
test("extractJobs keeps comp null when absent and counts open roles per lab", () => {
  const j = extractJobs(JOBS);
  assert.equal(j.latest[0].comp, null);
  assert.deepEqual(j.byLab, [{ company: "Anthropic", n: 2 }, { company: "OpenAI", n: 1 }]);
  assert.equal(j.meta.n_labs, 2);
});
test("extractJobs throws when meta.companies is missing", () => {
  assert.throws(() => extractJobs({ meta: { open: 1 }, jobs: JOBS.jobs }), /companies/);
});

// ---------- extractNetworks ----------
test("extractNetworks maps links to node-index pairs and keeps communities", () => {
  const n = extractNetworks(NET);
  assert.deepEqual(n.links, [[0, 1], [1, 2]]);
  assert.equal(n.nodes[0].label, "Ada");
  assert.deepEqual(n.communities[1], { id: 1, label: "David Bau" });
});
test("extractNetworks throws on unresolvable link endpoints and non-finite positions", () => {
  assert.throws(() => extractNetworks({ ...NET, links: [{ source: "a", target: "zzz" }] }), /zzz/);
  const badNodes = [{ ...NET.nodes[0], x: undefined }, ...NET.nodes.slice(1)];
  assert.throws(() => extractNetworks({ ...NET, nodes: badNodes }), /x of/);
});

// ---------- buildPreviews ----------
test("buildPreviews assembles all three sections from raw sources", () => {
  const out = buildPreviews({
    housesSrc: `window.HOUSES_DATA = ${JSON.stringify(HOUSES)};`,
    jobsSrc: `window.JOBS_DATA = ${JSON.stringify(JOBS)};`,
    networksJson: JSON.stringify(NET),
  });
  assert.equal(out.houses.listings.length, 3);
  assert.equal(out.jobs.latest.length, 3);
  assert.equal(out.networks.nodes.length, 3);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test scripts/build_previews.test.mjs`
Expected: FAIL — `Cannot find module ... build_previews.mjs`

- [ ] **Step 3: Write the implementation**

Create `scripts/build_previews.mjs`:

```js
#!/usr/bin/env node
// Build lib/previews.json — small data slices for the front-page "Projects,
// live" strip — from the committed artifacts behind /houses, /jobs, /networks.
// Runs at the start of `pnpm build` / `pnpm dev`. Fails loudly on shape drift:
// these inputs are committed artifacts that should always be valid.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

export function parseWindowData(src, varName) {
  const marker = `window.${varName} = `;
  const i = src.indexOf(marker);
  if (i < 0) throw new Error(`window.${varName} assignment not found`);
  let body = src.slice(i + marker.length).trim();
  if (body.endsWith(";")) body = body.slice(0, -1);
  return JSON.parse(body);
}

function num(v, what) {
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`${what} is not a finite number: ${v}`);
  }
  return v;
}

// Mirrors the "driver" logic in public/houses/index.html:541-575 — the score
// component that most lifts a listing above the field, for the alex audience.
const DRIVER_KEYS = ["commute", "nice", "aesthetic", "nature", "value", "soft"];

function compScores(x) {
  const s = x.scores || {};
  return {
    commute: s.commute ?? 5, nice: s.nice ?? 5, aesthetic: s.aesthetic ?? 5,
    nature: s.nature ?? 5, value: s.value ?? 5,
    soft: x.bucket === "apt" ? (s.quiet ?? 5) : (s.social ?? 5),
  };
}

export function extractHouses(data) {
  const { meta, listings } = data;
  if (!Array.isArray(listings) || listings.length === 0) throw new Error("houses: listings empty");
  const weights = meta?.fit_weights?.alex;
  if (!weights || Object.keys(weights).length === 0) throw new Error("houses: meta.fit_weights.alex missing");
  const live = listings.filter((x) => !x.gone);
  if (live.length === 0) throw new Error("houses: all listings gone");
  const scored = live.map(compScores);
  const means = {};
  for (const k of DRIVER_KEYS) {
    means[k] = scored.reduce((t, r) => t + r[k], 0) / scored.length;
  }
  const driver = (x) => {
    const sc = compScores(x);
    let bestKey = "commute";
    let bestVal = -Infinity;
    for (const k of DRIVER_KEYS) {
      const v = (weights[k] ?? 0) * (sc[k] - means[k]);
      if (v > bestVal) { bestVal = v; bestKey = k; }
    }
    if (bestKey === "soft") return x.bucket === "apt" ? "quiet" : "social";
    return bestKey;
  };
  const top = [...live].sort((a, b) => b.fit - a.fit).slice(0, 25).map((x) => ({
    lat: num(x.lat, `houses lat of ${x.id}`),
    lon: num(x.lon, `houses lon of ${x.id}`),
    fit: num(x.fit, `houses fit of ${x.id}`),
    pdisp: String(x.pdisp ?? `$${x.price}`),
    hood: String(x.hood ?? "").trim() || "unknown",
    driver: driver(x),
  }));
  return {
    meta: {
      n_scouted: num(meta.n_scouted, "houses n_scouted"),
      price_min: num(meta.price_min, "houses price_min"),
      price_max: num(meta.price_max, "houses price_max"),
      price_med: num(meta.price_med, "houses price_med"),
      generated: String(meta.generated ?? ""),
    },
    // Coordinates from public/houses/index.html:629-630
    anchors: [
      { lat: 37.7935, lon: -122.397, label: "Downtown SF" },
      { lat: 37.8703, lon: -122.268, label: "Berkeley · FAR Labs" },
    ],
    listings: top,
  };
}

export function extractJobs(data) {
  const { meta, jobs } = data;
  if (!Array.isArray(jobs) || jobs.length === 0) throw new Error("jobs: jobs empty");
  const companies = meta?.companies;
  if (!Array.isArray(companies) || companies.length === 0) throw new Error("jobs: meta.companies missing");
  const open = jobs.filter((j) => !j.closed);
  if (open.length === 0) throw new Error("jobs: no open jobs");
  // Same "newest" ordering as public/jobs/index.html:326
  const latest = [...open]
    .sort((a, b) => (b.published || b.first_seen || "").localeCompare(a.published || a.first_seen || ""))
    .slice(0, 6)
    .map((j) => ({
      company: String(j.company),
      group: String(j.group || "frontier"),
      title: String(j.title),
      comp: j.comp ? String(j.comp) : null,
      date: String(j.published || j.first_seen || ""),
    }));
  return {
    meta: {
      open: num(meta.open, "jobs meta.open"),
      n_labs: companies.length,
      generated: String(meta.date || meta.generated || ""),
    },
    latest,
    byLab: companies.map((c) => ({ company: c, n: open.filter((j) => j.company === c).length })),
  };
}

export function extractNetworks(data) {
  const { nodes, links, communities, meta } = data;
  if (!Array.isArray(nodes) || nodes.length === 0) throw new Error("networks: nodes empty");
  if (!Array.isArray(links) || links.length === 0) throw new Error("networks: links empty");
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  return {
    meta: { n_nodes: num(meta.n_nodes, "networks n_nodes"), n_links: num(meta.n_links, "networks n_links") },
    communities: (communities ?? []).map((c) => ({ id: c.id, label: String(c.label) })),
    nodes: nodes.map((n) => ({
      label: String(n.label),
      community: num(n.community ?? 0, `community of ${n.id}`),
      x: num(n.x, `x of ${n.id}`),
      y: num(n.y, `y of ${n.id}`),
    })),
    links: links.map((l) => {
      const s = index.get(l.source);
      const t = index.get(l.target);
      if (s === undefined || t === undefined) {
        throw new Error(`networks: link endpoint not found: ${l.source} -> ${l.target}`);
      }
      return [s, t];
    }),
  };
}

export function buildPreviews({ housesSrc, jobsSrc, networksJson }) {
  return {
    houses: extractHouses(parseWindowData(housesSrc, "HOUSES_DATA")),
    jobs: extractJobs(parseWindowData(jobsSrc, "JOBS_DATA")),
    networks: extractNetworks(JSON.parse(networksJson)),
  };
}

// ---------------- CLI ----------------
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");
  const out = buildPreviews({
    housesSrc: read("public/houses/data.js"),
    jobsSrc: read("public/jobs/data.js"),
    networksJson: read("public/assets/data/coauthorship.json"),
  });
  const json = JSON.stringify(out);
  if (json.length > 60_000) throw new Error(`previews.json unexpectedly large: ${json.length} bytes`);
  fs.writeFileSync(path.join(ROOT, "lib", "previews.json"), json);
  console.log(
    `lib/previews.json written: ${json.length} bytes — ` +
    `${out.houses.listings.length} listings, ${out.jobs.latest.length} jobs, ` +
    `${out.networks.nodes.length} nodes / ${out.networks.links.length} links`,
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test scripts/build_previews.test.mjs`
Expected: all tests pass (14 pass, 0 fail)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_previews.mjs scripts/build_previews.test.mjs
git commit -m "feat: previews extraction — data slices for front-page strip"
```

---

### Task 2: CLI wiring — package.json, .gitignore, generate for real

**Files:**
- Modify: `package.json` (scripts block)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `scripts/build_previews.mjs` CLI (Task 1)
- Produces: `lib/previews.json` on every `pnpm build` / `pnpm dev`; `pnpm test` runs all `.test.mjs` suites. Task 8 relies on `pnpm test` and `pnpm build`.

- [ ] **Step 1: Update package.json scripts**

Replace the scripts block (keep everything else):

```json
"scripts": {
  "dev": "node scripts/build_previews.mjs && next dev",
  "build": "node scripts/build_previews.mjs && next build",
  "start": "next start",
  "lint": "next lint",
  "test": "node --test scripts/ lib/"
},
```

(Inline invocation, not `prebuild`/`predev`: pnpm does not run npm pre/post hooks by default, and Vercel builds with pnpm.)

- [ ] **Step 2: Gitignore the generated file**

Append to `.gitignore`:

```
# generated by scripts/build_previews.mjs (build/dev)
lib/previews.json
```

- [ ] **Step 3: Generate against the real data and sanity-check**

Run: `node scripts/build_previews.mjs`
Expected output like: `lib/previews.json written: ~18000 bytes — 25 listings, 6 jobs, 132 nodes / 368 links`

Run: `node -e "const p=require('./lib/previews.json'); console.log(p.jobs.latest[0], p.houses.listings[0], p.networks.communities)"`
Expected: a real newest job, the top-fit listing with a `driver` key, three named communities.

Run: `pnpm test`
Expected: Task 1 suite passes (lib/ has no tests yet — `node --test` accepts the empty dir because force-sim tests land there in Task 3; if it errors on no-files-found, temporarily use `node --test scripts/` and restore in Task 3).

Run: `git status --short` — `lib/previews.json` must NOT appear (gitignored).

- [ ] **Step 4: Commit**

```bash
git add package.json .gitignore
git commit -m "build: generate lib/previews.json in build/dev; add pnpm test"
```

---

### Task 3: Spring relaxation module (TDD)

**Files:**
- Create: `lib/force-sim.mjs`
- Create: `lib/force-sim.d.ts`
- Test: `lib/force-sim.test.mjs`

**Interfaces:**
- Produces (consumed by `NetworksMini.tsx`, Task 7):
  - `relax(nodes, links, opts?)` — one or more relaxation iterations, mutates nodes in place. `nodes: {x,y,x0,y0,pinned?}[]` (x0,y0 = home position), `links: [si,ti][]` index pairs, `opts: {iterations?, spring?, home?}`.
  - `settled(nodes, eps?): boolean` — true when every unpinned node is within `eps` of rest (used to stop the rAF loop).

Model: the graph is an elastic sheet anchored at home positions — each link tries to preserve its *original* offset vector, each node is pulled back to its home. Deterministic (no RNG), no repulsion needed since homes prevent collapse. Dragging pins a node elsewhere; its neighborhood follows; on release everything relaxes home.

- [ ] **Step 1: Write the failing test**

Create `lib/force-sim.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { relax, settled } from "./force-sim.mjs";

const mk = (x, y, over = {}) => ({ x, y, x0: x, y0: y, ...over });

test("a graph at rest stays at rest", () => {
  const nodes = [mk(0, 0), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 5 });
  assert.equal(nodes[0].x, 0);
  assert.equal(nodes[1].x, 10);
  assert.ok(settled(nodes));
});

test("pinned nodes never move", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 30, y: 40 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 3 });
  assert.equal(nodes[0].x, 30);
  assert.equal(nodes[0].y, 40);
});

test("neighbors follow a displaced pinned node", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 20 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 10 });
  assert.ok(nodes[1].x > 10, `expected neighbor pulled right, got ${nodes[1].x}`);
});

test("after unpinning, everything relaxes back home", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 20, y: 15 }), mk(10, 0)];
  relax(nodes, [[0, 1]], { iterations: 10 });
  nodes[0].pinned = false;
  relax(nodes, [[0, 1]], { iterations: 400 });
  assert.ok(Math.abs(nodes[0].x - 0) < 0.01, `node0.x=${nodes[0].x}`);
  assert.ok(Math.abs(nodes[1].x - 10) < 0.01, `node1.x=${nodes[1].x}`);
  assert.ok(settled(nodes, 0.02));
});

test("positions stay finite under a large displacement", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 1e4 }), mk(1, 0), mk(2, 0)];
  relax(nodes, [[0, 1], [1, 2]], { iterations: 200 });
  for (const n of nodes) {
    assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y));
  }
});

test("settled respects eps and ignores pinned nodes", () => {
  const nodes = [mk(0, 0, { pinned: true, x: 100 }), mk(10, 0)];
  assert.ok(settled(nodes, 0.01), "pinned displacement should not count");
  nodes[1].x = 10.5;
  assert.ok(!settled(nodes, 0.01));
  assert.ok(settled(nodes, 1));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test lib/force-sim.test.mjs`
Expected: FAIL — `Cannot find module ... force-sim.mjs`

- [ ] **Step 3: Write the implementation**

Create `lib/force-sim.mjs`:

```js
// Minimal deterministic spring relaxation for the front-page networks mini.
// The graph behaves like an elastic sheet anchored at home positions (x0, y0):
// each link tries to preserve its original offset vector, each unpinned node
// is pulled back toward home. No RNG, no repulsion (homes prevent collapse).
export function relax(nodes, links, { iterations = 1, spring = 0.06, home = 0.1 } = {}) {
  const n = nodes.length;
  const fx = new Float64Array(n);
  const fy = new Float64Array(n);
  for (let it = 0; it < iterations; it++) {
    fx.fill(0);
    fy.fill(0);
    for (const [s, t] of links) {
      const a = nodes[s];
      const b = nodes[t];
      // deviation of the current offset from the original offset
      const dx = (b.x - a.x) - (b.x0 - a.x0);
      const dy = (b.y - a.y) - (b.y0 - a.y0);
      fx[s] += spring * dx; fy[s] += spring * dy;
      fx[t] -= spring * dx; fy[t] -= spring * dy;
    }
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (node.pinned) continue;
      node.x += fx[i] + home * (node.x0 - node.x);
      node.y += fy[i] + home * (node.y0 - node.y);
    }
  }
}

export function settled(nodes, eps = 0.05) {
  return nodes.every(
    (n) => n.pinned || (Math.abs(n.x - n.x0) < eps && Math.abs(n.y - n.y0) < eps),
  );
}
```

Create `lib/force-sim.d.ts`:

```ts
export type SimNode = { x: number; y: number; x0: number; y0: number; pinned?: boolean };
export function relax(
  nodes: SimNode[],
  links: Array<[number, number]>,
  opts?: { iterations?: number; spring?: number; home?: number },
): void;
export function settled(nodes: SimNode[], eps?: number): boolean;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm test`
Expected: both suites pass (Task 1's 14 + these 6).

- [ ] **Step 5: Commit**

```bash
git add lib/force-sim.mjs lib/force-sim.d.ts lib/force-sim.test.mjs
git commit -m "feat: deterministic spring relaxation for networks mini"
```

---

### Task 4: Content split + HomeTabs previews slot

**Files:**
- Create: `content/home-about-intro.md`, `content/home-about-rest.md`
- Delete: `content/home-about.md`
- Modify: `app/(site)/page.tsx`
- Modify: `components/HomeTabs.tsx`

**Interfaces:**
- Produces: `HomeTabs` props change from `aboutHtml: string` to `aboutIntroHtml: string; aboutRestHtml: string; previews?: React.ReactNode`. Task 8 passes `previews={<ProjectPreviews data={...}/>}` from `page.tsx`; this task passes nothing (slot renders empty) so the page is unchanged.

- [ ] **Step 1: Split the markdown**

`content/home-about-intro.md` = exactly the first paragraph of `content/home-about.md` (from `Hi, I'm **Alex Loftus**.` through `…Johns Hopkins University.`), one trailing newline.

`content/home-about-rest.md` = everything from `I've been fortunate to work with…` to the end of the file, unchanged.

```bash
git rm content/home-about.md
```

Verify no other reader of the old file exists:
Run: `grep -rn "home-about" app components lib --include="*.ts*"`
Expected: only `app/(site)/page.tsx` (about to be edited).

- [ ] **Step 2: Update page.tsx**

In `app/(site)/page.tsx`, replace the about rendering:

```tsx
export default async function HomePage() {
  const [aboutIntroHtml, aboutRestHtml, pubsHtml] = await Promise.all([
    renderMarkdown(readContent("home-about-intro.md")),
    renderMarkdown(readContent("home-about-rest.md")),
    renderMarkdown(readContent("publications.md")),
  ]);
  const posts = getAllPostsMeta()
    .slice(0, 8)
    .map((p) => ({ permalink: p.permalink, title: p.title, date: p.date, excerpt: p.excerpt }));

  return (
    <main className={`page-main ${styles.home}`}>
      <AuthorHeader />
      <HomeTabs
        aboutIntroHtml={aboutIntroHtml}
        aboutRestHtml={aboutRestHtml}
        pubsHtml={pubsHtml}
        posts={posts}
      />
    </main>
  );
}
```

- [ ] **Step 3: Update HomeTabs**

In `components/HomeTabs.tsx`, change the props:

```tsx
export default function HomeTabs({
  aboutIntroHtml,
  aboutRestHtml,
  previews,
  pubsHtml,
  posts,
}: {
  aboutIntroHtml: string;
  aboutRestHtml: string;
  previews?: React.ReactNode;
  pubsHtml: string;
  posts: PostMeta[];
}) {
```

and the about panel (currently lines 63-68):

```tsx
{tab === "about" && (
  <div className={styles.panel}>
    <ChatWidget />
    <div className="prose" dangerouslySetInnerHTML={{ __html: aboutIntroHtml }} />
    {previews}
    <div className="prose" dangerouslySetInnerHTML={{ __html: aboutRestHtml }} />
  </div>
)}
```

(`React.ReactNode` needs `import type { ReactNode } from "react"` or the `React.` namespace — the file already imports from `react`, so add `ReactNode` to that import and use `previews?: ReactNode`.)

- [ ] **Step 4: Build and eyeball**

Run: `pnpm build`
Expected: compiles clean, no type errors.

Run: `pnpm dev` (background), open http://localhost:3000 — About tab must look byte-identical to production (intro, then projects list, no visible seam between the two prose blocks). Stop the dev server.

- [ ] **Step 5: Commit**

```bash
git add content/home-about-intro.md content/home-about-rest.md app/\(site\)/page.tsx components/HomeTabs.tsx
git commit -m "refactor: split home-about.md; add previews slot to HomeTabs"
```

---

### Task 5: Strip shell, styles, types + JobsMini

**Files:**
- Create: `components/previews/types.ts`
- Create: `components/previews/ProjectPreviews.tsx`
- Create: `components/previews/ProjectPreviews.module.css`
- Create: `components/previews/JobsMini.tsx`
- Modify: `app/(site)/page.tsx` (read previews.json, pass the strip)

**Interfaces:**
- Consumes: `lib/previews.json` (Task 2), HomeTabs `previews` slot (Task 4).
- Produces: `ProjectPreviews({data: PreviewsData})`; `types.ts` exports `PreviewsData, HousesPreview, JobsPreview, NetworksPreview`. Tasks 6-7 fill the two placeholder mini slots with `<HousesMini data={data.houses}/>` / `<NetworksMini data={data.networks}/>` and use the shared CSS classes `svgWrap`, `caption`.

- [ ] **Step 1: Types**

Create `components/previews/types.ts` (mirrors Task 1 extraction output exactly):

```ts
export type HousesPreview = {
  meta: { n_scouted: number; price_min: number; price_max: number; price_med: number; generated: string };
  anchors: { lat: number; lon: number; label: string }[];
  listings: { lat: number; lon: number; fit: number; pdisp: string; hood: string; driver: string }[];
};

export type JobsPreview = {
  meta: { open: number; n_labs: number; generated: string };
  latest: { company: string; group: string; title: string; comp: string | null; date: string }[];
  byLab: { company: string; n: number }[];
};

export type NetworksPreview = {
  meta: { n_nodes: number; n_links: number };
  communities: { id: number; label: string }[];
  nodes: { label: string; community: number; x: number; y: number }[];
  links: [number, number][];
};

export type PreviewsData = {
  houses: HousesPreview;
  jobs: JobsPreview;
  networks: NetworksPreview;
};
```

- [ ] **Step 2: Strip + card shell**

Create `components/previews/ProjectPreviews.tsx`:

```tsx
import Link from "next/link";
import type { ReactNode } from "react";
import JobsMini from "./JobsMini";
import type { PreviewsData } from "./types";
import styles from "./ProjectPreviews.module.css";

const fmt = (n: number) => n.toLocaleString("en-US");

function Card({ href, title, foot, children }: {
  href: string; title: string; foot: string; children: ReactNode;
}) {
  return (
    <article className={styles.card}>
      <h3 className={styles.cardTitle}>{title}</h3>
      <div className={styles.mini}>{children}</div>
      <p className={styles.foot}>{foot}</p>
      <Link href={href} className={styles.open}>open →</Link>
    </article>
  );
}

export default function ProjectPreviews({ data }: { data: PreviewsData }) {
  const { houses, jobs, networks } = data;
  return (
    <section className={styles.strip} aria-label="Live previews of Alex's project pages">
      <h2 className={styles.label}>Projects, live</h2>
      <div className={styles.grid}>
        <Card
          href="/houses/"
          title="Houses"
          foot={`${fmt(houses.meta.n_scouted)} scouted · $${fmt(houses.meta.price_min)}–$${fmt(houses.meta.price_max)} · med $${fmt(houses.meta.price_med)}`}
        >
          {/* HousesMini lands in Task 6 */}
          <div />
        </Card>
        <Card
          href="/jobs/"
          title="Jobs"
          foot={`${fmt(jobs.meta.open)} open · ${jobs.meta.n_labs} labs · refreshed daily`}
        >
          <JobsMini data={jobs} />
        </Card>
        <Card
          href="/networks/"
          title="Networks"
          foot={`${networks.meta.n_nodes} researchers · ${networks.meta.n_links} co-authorships`}
        >
          {/* NetworksMini lands in Task 7 */}
          <div />
        </Card>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: JobsMini**

Create `components/previews/JobsMini.tsx` (server component — no interactivity needed; every datum is visible in the row):

```tsx
import type { JobsPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Same group hues as public/jobs/index.html:12,93
const GROUP_COLORS: Record<string, string> = {
  anthropic: "var(--accent)",
  interp: "#5f7d6e",
  frontier: "#6b7a94",
};

export default function JobsMini({ data }: { data: JobsPreview }) {
  if (data.latest.length === 0 || data.byLab.length === 0) {
    throw new Error("JobsMini: empty preview data");
  }
  const max = Math.max(...data.byLab.map((l) => l.n));
  return (
    <div className={styles.jobsMini}>
      <ul className={styles.jobsList}>
        {data.latest.map((j, i) => (
          <li key={i} className={styles.jobRow}>
            <span
              className={styles.jobCo}
              style={{ background: GROUP_COLORS[j.group] ?? GROUP_COLORS.frontier }}
            >
              {j.company}
            </span>
            <span className={styles.jobTitle}>{j.title}</span>
            {j.comp && <span className={styles.jobComp}>{j.comp}</span>}
          </li>
        ))}
      </ul>
      <div className={styles.labBars} aria-label={`Open roles per lab across ${data.byLab.length} labs`}>
        {data.byLab.map((l) => (
          <div key={l.company} className={styles.labBarCol} title={`${l.company}: ${l.n} open`}>
            <div
              className={styles.labBar}
              style={{ height: `${Math.max(8, Math.round((l.n / max) * 100))}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Styles**

Create `components/previews/ProjectPreviews.module.css`:

```css
/* ---- strip ---- */
.strip { margin: 2.1rem 0 2.4rem; }
.label {
  font-family: var(--font-serif-text);
  font-size: var(--text-sm);
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.35rem;
  margin: 0 0 0.9rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.9rem;
}
@media (max-width: 640px) {
  .grid { grid-template-columns: 1fr; }
}

/* ---- card ---- */
.card {
  position: relative; /* containing block for the stretched .open link */
  display: flex;
  flex-direction: column;
  background: var(--paper-raised);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 0.65rem 0.7rem 0.55rem;
  transition: box-shadow 140ms ease, border-color 140ms ease;
}
.card:hover { box-shadow: var(--shadow-soft); border-color: var(--rule-soft); }
.cardTitle {
  font-family: var(--font-serif-text);
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin: 0 0 0.45rem;
}
.mini { height: 10.5rem; }
.foot {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.68rem;
  color: var(--muted);
  margin: 0.45rem 0 0;
  border-top: 1px solid var(--rule-soft);
  padding-top: 0.4rem;
}
.open {
  font-family: var(--font-serif-text);
  font-size: var(--text-xs);
  color: var(--accent);
  text-decoration: none;
  margin-top: 0.15rem;
}
/* stretched link: the whole card navigates */
.open::after { content: ""; position: absolute; inset: 0; }

/* ---- interactive layers sit above the stretched link; only marked
       elements catch the pointer, everything else falls through ---- */
.svgWrap {
  position: relative;
  z-index: 1;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.svgWrap svg { flex: 1; width: 100%; min-height: 0; }
.caption {
  font-family: var(--font-serif-text);
  font-style: italic;
  font-size: 0.72rem;
  color: var(--muted);
  margin: 0.25rem 0 0;
  min-height: 1.05rem; /* reserve the line: no layout shift on hover */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ---- jobs mini (non-interactive: clicks fall through to the card link) ---- */
.jobsMini {
  display: flex;
  flex-direction: column;
  height: 100%;
  pointer-events: none;
}
.jobsList { list-style: none; margin: 0; padding: 0; flex: 1; }
.jobRow {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  font-size: 0.72rem;
  line-height: 1.15;
  padding: 0.14rem 0;
}
.jobRow + .jobRow { border-top: 1px solid var(--rule-soft); }
.jobCo {
  flex: none;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.58rem;
  font-weight: 600;
  color: #fff;
  border-radius: 8px;
  padding: 0.05rem 0.45rem;
}
.jobTitle {
  flex: 1;
  min-width: 0;
  font-family: var(--font-serif-text);
  color: var(--ink-soft);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.jobComp {
  flex: none;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.62rem;
  color: var(--muted);
}
.labBars {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 1.5rem;
  margin-top: 0.35rem;
  border-bottom: 1px solid var(--rule);
}
.labBarCol { flex: 1; display: flex; align-items: flex-end; height: 100%; }
.labBar { width: 100%; background: var(--rule); }
.labBarCol:first-child .labBar { background: var(--accent-soft); } /* Anthropic first in meta.companies */

/* ---- houses mini ---- */
.dot { pointer-events: all; cursor: pointer; }
.anchorMark {
  font-size: 7px;
  fill: var(--ink);
  text-anchor: middle;
  pointer-events: none;
}

/* ---- networks mini ---- */
.netLink { stroke: var(--muted); stroke-opacity: 0.16; stroke-width: 0.6; }
.netNode { pointer-events: all; cursor: grab; }
.netNodeDragging { cursor: grabbing; }
```

- [ ] **Step 5: Wire into the page**

In `app/(site)/page.tsx` add imports and the fs read (same pattern as `readContent`):

```tsx
import ProjectPreviews from "@/components/previews/ProjectPreviews";
import type { PreviewsData } from "@/components/previews/types";

function readPreviews(): PreviewsData {
  const raw = fs.readFileSync(path.join(process.cwd(), "lib", "previews.json"), "utf8");
  return JSON.parse(raw) as PreviewsData;
}
```

and inside `HomePage`, pass the slot:

```tsx
const previews = readPreviews();
…
<HomeTabs
  aboutIntroHtml={aboutIntroHtml}
  aboutRestHtml={aboutRestHtml}
  previews={<ProjectPreviews data={previews} />}
  pubsHtml={pubsHtml}
  posts={posts}
/>
```

- [ ] **Step 6: Build and eyeball**

Run: `pnpm build`
Expected: clean. Then `pnpm dev`, open http://localhost:3000: strip appears after the intro paragraph with three cards; Jobs card shows 6 real rows + 8 bars; Houses/Networks cards are empty boxes (expected until Tasks 6-7). Whole-card click navigates to /jobs/. Stop dev server.

- [ ] **Step 7: Commit**

```bash
git add components/previews app/\(site\)/page.tsx
git commit -m "feat: Projects-live strip shell + jobs mini on front page"
```

---

### Task 6: HousesMini

**Files:**
- Create: `components/previews/HousesMini.tsx`
- Modify: `components/previews/ProjectPreviews.tsx` (fill the placeholder)

**Interfaces:**
- Consumes: `HousesPreview` (Task 5 types), CSS classes `svgWrap`, `caption`, `dot`, `anchorMark` (Task 5).
- Produces: `<HousesMini data={houses}/>`.

- [ ] **Step 1: Implement**

Create `components/previews/HousesMini.tsx`:

```tsx
"use client";

import { useMemo, useState } from "react";
import type { HousesPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Same driver hues as public/houses/index.html:15
const DRIVER_COLORS: Record<string, string> = {
  nature: "#4f8a55", quiet: "#5f7189", nice: "#7b6091", social: "#bd8a3a",
  value: "#3f867e", commute: "#8a8378", aesthetic: "#b0654d",
};

const W = 200;
const H = 128;
const PAD = 10;

export default function HousesMini({ data }: { data: HousesPreview }) {
  if (data.listings.length === 0) throw new Error("HousesMini: empty listings");
  const [hover, setHover] = useState<number | null>(null);

  const { sx, sy } = useMemo(() => {
    const pts = [...data.listings, ...data.anchors];
    const lats = pts.map((p) => p.lat);
    const lons = pts.map((p) => p.lon);
    const latMin = Math.min(...lats);
    const latMax = Math.max(...lats);
    const lonMin = Math.min(...lons);
    const lonMax = Math.max(...lons);
    return {
      sx: (lon: number) => PAD + ((lon - lonMin) / (lonMax - lonMin || 1)) * (W - 2 * PAD),
      // north up: larger latitude -> smaller y
      sy: (lat: number) => H - PAD - ((lat - latMin) / (latMax - latMin || 1)) * (H - 2 * PAD),
    };
  }, [data]);

  const hovered = hover != null ? data.listings[hover] : null;
  return (
    <div className={styles.svgWrap}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dot map of shortlisted Bay Area rentals">
        {data.anchors.map((a) => (
          <text key={a.label} x={sx(a.lon)} y={sy(a.lat)} className={styles.anchorMark}>▲</text>
        ))}
        {data.listings.map((l, i) => (
          <circle
            key={i}
            cx={sx(l.lon)}
            cy={sy(l.lat)}
            r={1.8 + (l.fit / 10) * 3.2}
            fill={DRIVER_COLORS[l.driver] ?? DRIVER_COLORS.commute}
            fillOpacity={hover === i ? 1 : 0.72}
            className={styles.dot}
            onPointerEnter={() => setHover(i)}
            onPointerLeave={() => setHover(null)}
          />
        ))}
      </svg>
      <p className={styles.caption}>
        {hovered
          ? `${hovered.hood} · ${hovered.pdisp} · fit ${hovered.fit}`
          : "the rental shortlist, mapped — hover a dot"}
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Replace the placeholder**

In `components/previews/ProjectPreviews.tsx`: add `import HousesMini from "./HousesMini";` and replace the Houses card's `{/* HousesMini lands in Task 6 */}<div />` with `<HousesMini data={houses} />`.

- [ ] **Step 3: Build and eyeball**

Run: `pnpm build` — clean. `pnpm dev` → http://localhost:3000: dot cloud with two ▲ anchors; hovering a dot fills the caption with e.g. `marina / cow hollow · $1,950 · fit 8.4`; clicking empty map space navigates to /houses/. Stop dev server.

- [ ] **Step 4: Commit**

```bash
git add components/previews/HousesMini.tsx components/previews/ProjectPreviews.tsx
git commit -m "feat: houses mini — svg dot map with hover caption"
```

---

### Task 7: NetworksMini

**Files:**
- Create: `components/previews/NetworksMini.tsx`
- Modify: `components/previews/ProjectPreviews.tsx` (fill the placeholder)

**Interfaces:**
- Consumes: `NetworksPreview` (Task 5), `relax`/`settled` from `@/lib/force-sim.mjs` (Task 3), CSS classes `svgWrap`, `caption`, `netLink`, `netNode`, `netNodeDragging` (Task 5).
- Produces: `<NetworksMini data={networks}/>`.

Behavior: nodes render at the precomputed layout (scaled to the viewBox) — zero CPU at idle. Dragging a node pins it to the pointer and runs a rAF loop calling `relax()`, so its neighborhood follows elastically; on release the loop continues until `settled()`, easing everything home. With `prefers-reduced-motion: reduce`, drags are ignored (static picture, hover captions still work).

- [ ] **Step 1: Implement**

Create `components/previews/NetworksMini.tsx`:

```tsx
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { relax, settled, type SimNode } from "@/lib/force-sim.mjs";
import type { NetworksPreview } from "./types";
import styles from "./ProjectPreviews.module.css";

// Community -> site categorical tokens (0 EleutherAI, 1 David Bau, 2 Vogelstein)
const COMMUNITY_COLORS = ["var(--cat-blue)", "var(--cat-sienna)", "var(--cat-sage)"];

const W = 200;
const H = 132;
const PAD = 8;

function scaleNodes(data: NetworksPreview): SimNode[] {
  const xs = data.nodes.map((n) => n.x);
  const ys = data.nodes.map((n) => n.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  return data.nodes.map((n) => {
    const x = PAD + ((n.x - xMin) / (xMax - xMin || 1)) * (W - 2 * PAD);
    const y = PAD + ((n.y - yMin) / (yMax - yMin || 1)) * (H - 2 * PAD);
    return { x, y, x0: x, y0: y };
  });
}

export default function NetworksMini({ data }: { data: NetworksPreview }) {
  if (data.nodes.length === 0) throw new Error("NetworksMini: empty nodes");
  const nodesRef = useRef<SimNode[]>([]);
  const [, setTick] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const dragRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const reducedRef = useRef(false);

  // Initialize once (and if data identity ever changes)
  useMemo(() => {
    nodesRef.current = scaleNodes(data);
  }, [data]);

  useEffect(() => {
    reducedRef.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  function loop() {
    relax(nodesRef.current, data.links, { iterations: 2 });
    setTick((t) => t + 1);
    const done = dragRef.current == null && settled(nodesRef.current, 0.15);
    rafRef.current = done ? null : requestAnimationFrame(loop);
    if (done) {
      // snap exactly home so idle state is byte-identical to first render
      for (const n of nodesRef.current) { n.x = n.x0; n.y = n.y0; }
      setTick((t) => t + 1);
    }
  }

  function toLocal(e: React.PointerEvent): { x: number; y: number } {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * W,
      y: ((e.clientY - rect.top) / rect.height) * H,
    };
  }

  function onNodeDown(i: number, e: React.PointerEvent) {
    if (reducedRef.current) return;
    e.preventDefault();
    dragRef.current = i;
    nodesRef.current[i].pinned = true;
    (e.target as Element).setPointerCapture(e.pointerId);
    if (rafRef.current == null) rafRef.current = requestAnimationFrame(loop);
  }
  function onNodeMove(e: React.PointerEvent) {
    const i = dragRef.current;
    if (i == null) return;
    const p = toLocal(e);
    nodesRef.current[i].x = Math.max(2, Math.min(W - 2, p.x));
    nodesRef.current[i].y = Math.max(2, Math.min(H - 2, p.y));
  }
  function onNodeUp() {
    const i = dragRef.current;
    if (i == null) return;
    nodesRef.current[i].pinned = false;
    dragRef.current = null;
  }

  const nodes = nodesRef.current;
  const hovered = hover != null ? data.nodes[hover] : null;
  const hoveredGroup = hovered ? data.communities.find((c) => c.id === hovered.community)?.label : null;

  return (
    <div className={styles.svgWrap}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Co-authorship network of researchers Alex has worked with"
      >
        {data.links.map(([s, t], i) => (
          <line
            key={i}
            x1={nodes[s].x} y1={nodes[s].y}
            x2={nodes[t].x} y2={nodes[t].y}
            className={styles.netLink}
          />
        ))}
        {nodes.map((n, i) => (
          <circle
            key={i}
            cx={n.x} cy={n.y} r={hover === i ? 3.6 : 2.4}
            fill={COMMUNITY_COLORS[data.nodes[i].community % COMMUNITY_COLORS.length]}
            fillOpacity={0.85}
            className={dragRef.current === i ? `${styles.netNode} ${styles.netNodeDragging}` : styles.netNode}
            onPointerDown={(e) => onNodeDown(i, e)}
            onPointerMove={onNodeMove}
            onPointerUp={onNodeUp}
            onPointerEnter={() => setHover(i)}
            onPointerLeave={() => setHover(null)}
          />
        ))}
      </svg>
      <p className={styles.caption}>
        {hovered
          ? `${hovered.label}${hoveredGroup ? ` · ${hoveredGroup}` : ""}`
          : "co-authors, linked by papers — drag a node"}
      </p>
    </div>
  );
}
```

Note on the `.mjs` import: TypeScript resolves `@/lib/force-sim.mjs` against `lib/force-sim.d.ts` (Task 3). If `tsc` complains about the explicit `.mjs` specifier, rename the declaration to `lib/force-sim.d.mts` — that is the declaration counterpart for `.mjs` files.

- [ ] **Step 2: Replace the placeholder**

In `components/previews/ProjectPreviews.tsx`: add `import NetworksMini from "./NetworksMini";` and replace the Networks card's `{/* NetworksMini lands in Task 7 */}<div />` with `<NetworksMini data={networks} />`.

- [ ] **Step 3: Build and eyeball**

Run: `pnpm build` — clean. `pnpm dev` → http://localhost:3000: 132 dots in three community colors over faint links; hover shows "Name · group" in the caption; dragging a node pulls its neighborhood and everything springs back on release; clicking empty space navigates to /networks/. Verify idle CPU is flat after release (no perpetual rAF). Stop dev server.

- [ ] **Step 4: Commit**

```bash
git add components/previews/NetworksMini.tsx components/previews/ProjectPreviews.tsx
git commit -m "feat: networks mini — draggable co-authorship graph"
```

---

### Task 8: Full verification + docs

**Files:**
- Modify: `CLAUDE.md` (repo root — build-step note)
- Modify (if drift found): `docs/superpowers/specs/2026-07-05-frontpage-previews-design.md`

- [ ] **Step 1: Test suite + lint + build**

```bash
pnpm test && pnpm lint && pnpm build
```
Expected: all suites pass, lint clean (pre-existing warnings acceptable), build clean.

- [ ] **Step 2: Visual verification, desktop + mobile**

`pnpm dev`, then in the browser check http://localhost:3000:
- Desktop (~1280px): strip sits between intro and "I've been fortunate…"; three cards aligned; footers show real meta numbers; hover captions work on houses + networks; drag works on networks; whole-card click navigates for all three; "open →" links work.
- Mobile (~375px viewport): cards stack vertically; nothing overflows; text legible.
- Screenshot both states for Alex.
Stop the dev server when done.

- [ ] **Step 3: Update repo docs**

Add to the repo-root `CLAUDE.md` (site architecture/behaviors section):

```markdown
- Front page shows a "Projects, live" strip (components/previews/) — mini previews of /houses, /jobs, /networks. `pnpm build`/`pnpm dev` first run scripts/build_previews.mjs, which slices public/houses/data.js + public/jobs/data.js + public/assets/data/coauthorship.json into gitignored lib/previews.json. Shape drift in those artifacts fails the build on purpose. Tests: `pnpm test` (node --test).
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: record previews build step; verification pass"
```

- [ ] **Step 5: Hand off**

Do NOT push (background-session push is gated). Report to Alex with screenshots and the exact push command he can run himself, e.g. `! git -C ~/loftusa.github.io/.claude/worktrees/frontpage-previews push -u origin worktree-frontpage-previews`.
