# Houses Weight Sliders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live weight sliders on /houses that re-rank and re-filter the board (map, grid, leaderboard, picks) client-side, with the fit math factored into a pure, node-testable module.

**Architecture:** A new pure-function file `public/houses/fitmath.js` (no DOM) mirrors the server fit formula and board ranking; `index.html` gains a slider panel whose state drives every render path through one `recomputeAll()` entry point. `data.js` already ships per-listing component scores and `meta.fit_weights` — zero pipeline changes.

**Tech Stack:** Vanilla JS (inline script + one new script file), Leaflet (existing), plain-assert node test run with `node`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-11-houses-weight-sliders-design.md`.
- ONLY touch: `public/houses/index.html`, `public/houses/fitmath.js` (new), `tests/houses/parity.test.mjs` (new), spec/plan docs, memory. A parallel agent owns `refresh.py`/`rate.py`/workflow/backend tests — never edit those.
- Component keys (canonical order): `["commute","nice","aesthetic","nature","value","soft","gym"]`.
- `soft` = `scores.quiet` for `bucket==="apt"`, `scores.social` for `"room"`. Missing any component score → 5.0 neutral.
- Fit: weighted sum, `+0.8` if `loved`, cap 10, round to 0.1 — mirror the exact cap/round order from `refresh.py`'s `alex_fit` call site (read it first; parity test enforces).
- Parity invariant: default weights reproduce shipped `x.fit` for every listing whose `scores` include `gym` (pinned pre-gym carry-forwards lack it and legitimately drift; assert those are all `pinned`).
- localStorage key: `housesWeights.v1` = `{w:{<7 keys>: raw 0-100}, cap:number|null, topN:number|"all"}`.
- Top-N is a global rank cutoff after price-cap filter; chips filter within the board; map ignores chips (as today).
- Candidates = not `gone`, not hidden (✕). Gone rows stay at leaderboard bottom struck-through.
- The map legend was rewritten by another session (`e419af1`) — grep current text before editing anything near it.
- Verify with `node --check` on the extracted inline script + `node tests/houses/parity.test.mjs` before every commit; browser-verify before push. Push to master only after all pass.

---

### Task 1: FitMath pure module + node tests

**Files:**
- Create: `public/houses/fitmath.js`
- Test: `tests/houses/parity.test.mjs`
- Read first: `public/houses/refresh/refresh.py` (the `alex_fit` definition and its call site in `do_build` — copy the exact rounding/cap order), `public/houses/data.js` (confirm `window.HOUSES_DATA` shape and `meta.fit_weights.alex`).

**Interfaces:**
- Produces (used by Tasks 2–3): global `FitMath` (browser) / `module.exports` (node) with:
  - `COMP_KEYS: string[]` — the 7 canonical keys.
  - `compScores(x) -> {commute,nice,aesthetic,nature,value,soft,gym}` — extracts per-listing 0–10 scores with soft mapping and 5.0 defaults.
  - `normWeights(raw) -> {k: number}` — raw (any non-negative numbers) → normalized to sum 1; all-zero → equal 1/7.
  - `fitOf(x, w) -> number` — w already normalized; returns final displayed fit (bonus/cap/round applied).
  - `rankBoard(candidates, w, cap, topN) -> {ranked: x[], boardIds: Set<string>, fits: Map<string,number>}` — `ranked` sorted by fit desc (ties: price asc); cap `null` = uncapped; topN `"all"` = everything.
  - `fieldMeans(candidates) -> {k: mean}` and `driverOf(x, w, means) -> string|null` — argmax `w[k]*(score[k]-means[k])`, margin ≤ 0.02 → `null`.

- [ ] **Step 1: Read the server formula.** `grep -n "def alex_fit" -A 12 public/houses/refresh/refresh.py` and `grep -n "alex_fit(" public/houses/refresh/refresh.py` — note the exact expression that applies loved bonus, cap, round.

- [ ] **Step 2: Write the failing test** at `tests/houses/parity.test.mjs`:

```js
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

const FitMath = require("../../public/houses/fitmath.js");
const ctx = { window: {} };
vm.createContext(ctx);
vm.runInContext(readFileSync(new URL("../../public/houses/data.js", import.meta.url), "utf8"), ctx);
const D = ctx.window.HOUSES_DATA;
assert.ok(D.listings.length > 0, "data.js loaded");

const W = FitMath.normWeights(D.meta.fit_weights.alex);

// 1) Parity: default weights reproduce shipped fits (rows rated under current weights)
let checked = 0;
for (const x of D.listings) {
  if (!("gym" in x.scores)) { assert.ok(x.pinned, `${x.id} missing gym but not pinned`); continue; }
  assert.equal(FitMath.fitOf(x, W), x.fit, `parity ${x.id}`);
  checked++;
}
assert.ok(checked >= D.listings.length - 4, "parity covered nearly all listings");

// 2) normWeights: all-zero -> equal; sums to 1
const eq = FitMath.normWeights(Object.fromEntries(FitMath.COMP_KEYS.map(k => [k, 0])));
for (const k of FitMath.COMP_KEYS) assert.ok(Math.abs(eq[k] - 1 / 7) < 1e-9);
const one = FitMath.normWeights({ commute: 50, nice: 0, aesthetic: 0, nature: 0, value: 0, soft: 0, gym: 0 });
assert.equal(one.commute, 1);

// 3) soft mapping + missing-score neutrality
const apt = { bucket: "apt", scores: { quiet: 9, social: 1 } };
const room = { bucket: "room", scores: { quiet: 9, social: 1 } };
assert.equal(FitMath.compScores(apt).soft, 9);
assert.equal(FitMath.compScores(room).soft, 1);
assert.equal(FitMath.compScores(apt).gym, 5);

// 4) loved bonus + cap
const hot = { bucket: "apt", loved: true,
  scores: { quiet: 10, social: 0, nice: 10, nature: 10, value: 10, commute: 10, aesthetic: 10, gym: 10 } };
assert.equal(FitMath.fitOf(hot, W), 10, "capped at 10");

// 5) rankBoard: cap filter, topN cutoff, "all"
const cands = D.listings.filter(x => !x.gone);
const rb = FitMath.rankBoard(cands, W, null, 5);
assert.equal(rb.boardIds.size, Math.min(5, cands.length));
assert.equal(rb.ranked.length, cands.length);
const fits = rb.ranked.map(x => rb.fits.get(x.id));
for (let i = 1; i < fits.length; i++) assert.ok(fits[i - 1] >= fits[i], "sorted desc");
const cheap = FitMath.rankBoard(cands, W, 1500, "all");
assert.ok([...cheap.boardIds].every(id => cands.find(x => x.id === id).price <= 1500));
const none = FitMath.rankBoard(cands, W, 1, "all");
assert.equal(none.boardIds.size, 0, "cap below min price -> empty board");

// 6) driverOf: obvious winner + neutral margin
const means = FitMath.fieldMeans(cands);
const flat = { bucket: "apt", scores: Object.fromEntries(
  FitMath.COMP_KEYS.map(k => [k === "soft" ? "quiet" : k, means[k]])) };
assert.equal(FitMath.driverOf(flat, W, means), null, "at-the-mean listing is neutral");

console.log(`parity.test.mjs OK — ${checked} parity rows, ${cands.length} candidates`);
```

- [ ] **Step 3: Run to verify it fails.** `node tests/houses/parity.test.mjs` — Expected: FAIL, `Cannot find module '.../fitmath.js'`.

- [ ] **Step 4: Implement `public/houses/fitmath.js`:**

```js
/* Pure fit math for /houses — no DOM. Mirrors alex_fit in refresh/refresh.py.
   Loaded by index.html (window.FitMath) and by tests/houses/parity.test.mjs (module.exports). */
(function () {
  const COMP_KEYS = ["commute", "nice", "aesthetic", "nature", "value", "soft", "gym"];

  function compScores(x) {
    const s = x.scores || {};
    const soft = x.bucket === "apt" ? s.quiet : s.social;
    return {
      commute: s.commute ?? 5, nice: s.nice ?? 5, aesthetic: s.aesthetic ?? 5,
      nature: s.nature ?? 5, value: s.value ?? 5, soft: soft ?? 5, gym: s.gym ?? 5,
    };
  }

  function normWeights(raw) {
    let tot = 0;
    for (const k of COMP_KEYS) tot += Math.max(0, raw[k] ?? 0);
    const out = {};
    for (const k of COMP_KEYS) out[k] = tot > 0 ? Math.max(0, raw[k] ?? 0) / tot : 1 / COMP_KEYS.length;
    return out;
  }

  function fitOf(x, w) {
    const s = compScores(x);
    let f = 0;
    for (const k of COMP_KEYS) f += (w[k] ?? 0) * s[k];
    if (x.loved) f += 0.8;                    // match refresh.py call-site order exactly
    return Math.min(10, Math.round(f * 10) / 10);
  }

  function rankBoard(candidates, w, cap, topN) {
    const passing = candidates.filter(x => cap == null || x.price <= cap);
    const fits = new Map(passing.map(x => [x.id, fitOf(x, w)]));
    const ranked = [...passing].sort((a, b) =>
      (fits.get(b.id) - fits.get(a.id)) || (a.price - b.price));
    const n = topN === "all" ? ranked.length : Math.min(topN, ranked.length);
    return { ranked, boardIds: new Set(ranked.slice(0, n).map(x => x.id)), fits };
  }

  function fieldMeans(candidates) {
    const m = Object.fromEntries(COMP_KEYS.map(k => [k, 0]));
    for (const x of candidates) { const s = compScores(x); for (const k of COMP_KEYS) m[k] += s[k]; }
    const n = Math.max(1, candidates.length);
    for (const k of COMP_KEYS) m[k] /= n;
    return m;
  }

  function driverOf(x, w, means) {
    const s = compScores(x);
    let best = null, bv = -Infinity;
    for (const k of COMP_KEYS) {
      const v = (w[k] ?? 0) * (s[k] - means[k]);
      if (v > bv) { bv = v; best = k; }
    }
    return bv > 0.02 ? best : null;
  }

  const FitMath = { COMP_KEYS, compScores, normWeights, fitOf, rankBoard, fieldMeans, driverOf };
  if (typeof module !== "undefined" && module.exports) module.exports = FitMath;
  if (typeof window !== "undefined") window.FitMath = FitMath;
})();
```

If Step 1 shows a different bonus/cap/round order (e.g. round before cap), match it exactly — the parity assertion on the loved listing (PL06 area) will catch a mismatch.

- [ ] **Step 5: Run to verify it passes.** `node tests/houses/parity.test.mjs` — Expected: `parity.test.mjs OK — …`. If parity fails only on `pinned` rows lacking `gym`, that's the skip path working; anything else, fix fitOf, don't loosen the test.

- [ ] **Step 6: Commit.** `git add public/houses/fitmath.js tests/houses/parity.test.mjs && git commit -m "feat(houses): FitMath pure module with fit parity tests"`

---

### Task 2: Slider panel UI + persisted state (no re-rank wiring yet)

**Files:**
- Modify: `public/houses/index.html` — masthead area (~line 228), CSS block, top of the inline script.

**Interfaces:**
- Consumes: `FitMath.COMP_KEYS`, `FitMath.normWeights`; `D.meta.fit_weights.alex`, `D.meta.price_min/price_max`.
- Produces (Task 3 relies on): global `WSTATE = {w, cap, topN}`; `defaultRawW()`; `onWeightsChanged(cb)` registering a debounced (80 ms) change callback; `renderWeightPanel()`; constant `WKEY = "housesWeights.v1"`. Panel DOM ids: `#wpanel`, `#wbar`, `#wsliders`, `#wprice`, `#wtopn`, `#wreset`.

- [ ] **Step 1: Load fitmath.js before the inline script.** Find the `<script src="data.js">` tag (grep `data.js`); add on the next line:

```html
<script src="fitmath.js?v=1"></script>
```

(match the existing cache-busting query style if data.js uses one; grep first).

- [ ] **Step 2: Add panel markup** directly after the closing `</header>` of the masthead (before the `<h2>The map` heading):

```html
<section id="wpanel" class="ui">
  <div class="wtop"><span class="wtitle">What matters to you</span>
    <button id="wreset" title="Restore default weights">reset ↺</button></div>
  <div id="wbar"></div>
  <div id="wsliders"></div>
  <div class="wrow2">
    <label class="wprice-lab">max price <span id="wpriceval" class="num"></span>
      <input type="range" id="wprice"></label>
    <label>show top <select id="wtopn">
      <option value="10">10</option><option value="20" selected>20</option>
      <option value="35">35</option><option value="all">all</option></select></label>
  </div>
</section>
```

- [ ] **Step 3: Add CSS** (inside the existing `<style>`, near the `.controls` rules; reuse the page's palette vars):

```css
#wpanel{margin:14px 0 6px;padding:12px 14px;border:1px solid var(--line,#e4ddd3);border-radius:8px;background:#fdfbf7}
.wtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.wtitle{font-weight:600;font-size:13px}
#wreset{font:11px -apple-system,sans-serif;border:none;background:none;color:var(--muted);cursor:pointer}
#wreset:hover{color:var(--ink,#222)}
#wbar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:10px}
#wbar i{transition:width .25s ease}
#wsliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:2px 18px}
.wsl{display:grid;grid-template-columns:64px 1fr 34px;align-items:center;gap:8px;font-size:11.5px;color:var(--muted)}
.wsl input[type=range]{width:100%;height:3px}
.wsl .wpct{text-align:right;font-variant-numeric:tabular-nums}
.wrow2{display:flex;gap:22px;align-items:center;margin-top:10px;font-size:11.5px;color:var(--muted)}
.wrow2 input[type=range]{width:130px;vertical-align:middle}
@media(max-width:860px){#wsliders{grid-template-columns:1fr}}
```

- [ ] **Step 4: Add state + rendering JS** near the top of the inline script (after `D`/`GIO` are read, before render calls). Component label/hue table reuses the existing CSS custom properties (grep `--gym` to confirm names; quiet/social share the soft slider — tint it with `--quiet`):

```js
/* ---- weight sliders ---- */
const WKEY = "housesWeights.v1";
const WCOMP = [
  ["commute","commute","--commute"], ["nice","nice area","--nice"],
  ["aesthetic","photos","--aesthetic"], ["nature","nature","--nature"],
  ["value","value","--value"], ["soft","quiet/social","--quiet"], ["gym","gym","--gym"]];
const defaultRawW = () => Object.fromEntries(FitMath.COMP_KEYS.map(k =>
  [k, Math.round((D.meta.fit_weights?.alex?.[k] ?? 1/7) * 100)]));
let WSTATE = { w: defaultRawW(), cap: null, topN: 20 };
try {
  const s = JSON.parse(localStorage.getItem(WKEY) || "null");
  if (s && s.w && FitMath.COMP_KEYS.every(k => typeof s.w[k] === "number"))
    WSTATE = { w: s.w, cap: s.cap ?? null, topN: s.topN ?? 20 };
} catch (e) {}
const wSave = () => localStorage.setItem(WKEY, JSON.stringify(WSTATE));

let wCb = null, wTimer = null;
function onWeightsChanged(cb) { wCb = cb; }
function wFire() { clearTimeout(wTimer); wTimer = setTimeout(() => { wSave(); wCb && wCb(); }, 80); }

const PMIN = Math.floor((D.meta.price_min ?? 500) / 50) * 50;
const PMAX = Math.ceil((D.meta.price_max ?? 4000) / 50) * 50;

function renderWeightPanel() {
  const nw = FitMath.normWeights(WSTATE.w);
  document.getElementById("wbar").innerHTML = WCOMP.map(([k,,v]) =>
    `<i style="width:${(nw[k]*100).toFixed(1)}%;background:${css(v)}"></i>`).join("");
  document.querySelectorAll("#wsliders .wpct").forEach(el =>
    el.textContent = Math.round(nw[el.dataset.k]*100) + "%");
  document.getElementById("wpriceval").textContent = WSTATE.cap == null ? "any" : money(WSTATE.cap);
}

(function initWeightPanel() {
  document.getElementById("wsliders").innerHTML = WCOMP.map(([k,label,v]) =>
    `<label class="wsl"><span>${label}</span>
      <input type="range" min="0" max="100" value="${WSTATE.w[k]}" data-k="${k}" style="accent-color:${css(v)}">
      <span class="wpct" data-k="${k}"></span></label>`).join("");
  document.querySelectorAll("#wsliders input").forEach(inp =>
    inp.oninput = () => { WSTATE.w[inp.dataset.k] = +inp.value; renderWeightPanel(); wFire(); });
  const pr = document.getElementById("wprice");
  pr.min = PMIN; pr.max = PMAX; pr.step = 50; pr.value = WSTATE.cap ?? PMAX;
  pr.oninput = () => { WSTATE.cap = +pr.value >= PMAX ? null : +pr.value; renderWeightPanel(); wFire(); };
  const tn = document.getElementById("wtopn");
  tn.value = String(WSTATE.topN);
  tn.onchange = () => { WSTATE.topN = tn.value === "all" ? "all" : +tn.value; wFire(); };
  document.getElementById("wreset").onclick = () => {
    WSTATE = { w: defaultRawW(), cap: null, topN: 20 };
    document.querySelectorAll("#wsliders input").forEach(i => i.value = WSTATE.w[i.dataset.k]);
    pr.value = PMAX; tn.value = "20";
    renderWeightPanel(); wFire();
  };
  renderWeightPanel();
})();
```

(`css()` and `money()` already exist in the page — grep to confirm names before relying on them.)

- [ ] **Step 5: Syntax check.** Extract inline script and `node --check` it (existing convention), plus `node tests/houses/parity.test.mjs` still green. Open the page locally (`python3 -m http.server` in `public/houses` or via file://) and confirm the panel renders, sliders move the bar, reset works, reload restores state. No re-ranking yet — expected.

- [ ] **Step 6: Commit.** `git add public/houses/index.html && git commit -m "feat(houses): weight slider panel with persisted state"`

---

### Task 3: Wire live re-ranking through every render path

**Files:**
- Modify: `public/houses/index.html` — the fit/rank/driver plumbing: `RANK` construction, `filtered()`, `renderGrid`, `renderLeaderboard`, `renderPicks`, `drawMarkers` call sites, `driver()` / `DRIVER_MEANS` / coxcomb code, popup builders, `refreshReachedUI()`.
- Read first: grep for `RANK`, `driver(`, `DRIVER_MEANS`, `drawMarkers(`, `renderPicks`, `pick` badge usage, and the legend block (rewritten in `e419af1`) to anchor edits on current text.

**Interfaces:**
- Consumes: `WSTATE`, `onWeightsChanged`, `renderWeightPanel`, and all `FitMath.*` from Tasks 1–2.
- Produces: globals the rest of the page reads — `LIVE = {fits: Map, rank: {id:n}, boardIds: Set, means, nw}` refreshed by `recomputeAll()`; every fit/rank/driver read goes through `LIVE`.

- [ ] **Step 1: Build the recompute core.** Add after the panel code:

```js
/* ---- live ranking (single source of truth for fit/rank/board) ---- */
let LIVE = null;
function recomputeAll() {
  const nw = FitMath.normWeights(WSTATE.w);
  const cands = D.listings.filter(x => !x.gone && !isHidden(x));
  const { ranked, boardIds, fits } = FitMath.rankBoard(cands, nw, WSTATE.cap, WSTATE.topN);
  const rank = {}; ranked.forEach((x, i) => rank[x.id] = i + 1);
  LIVE = { fits, rank, boardIds, ranked, means: FitMath.fieldMeans(cands), nw };
}
const liveFit  = x => LIVE.fits.get(x.id) ?? x.fit;   // gone rows keep shipped fit
const liveRank = x => LIVE.rank[x.id] ?? "–";
const onBoard  = x => LIVE.boardIds.has(x.id);
```

- [ ] **Step 2: Replace static reads.** Using grep results, systematically:
  - Every `x.fit` display / sort in Alex-board code → `liveFit(x)`. Every `RANK[x.id]` → `liveRank(x)` (keep the old `RANK` for Gio if shared — Gio paths (`GRANK`, gio weights) stay untouched).
  - `filtered()` gains `&& onBoard(x)` (chips then filter within the board, per spec).
  - `drawMarkers(...)` call sites pass `D.listings.filter(x => onBoard(x))` (gone rows excluded from the map by construction since they're not candidates).
  - `driver(x)` for Alex listings → `FitMath.driverOf(x, LIVE.nw, LIVE.means)` mapped to the existing hue table; coxcomb petal radii keep using raw scores (unchanged) but petal ORDER/keys stay as-is; the "strongest vs field" popup line uses `driverOf` + `LIVE.nw`.
  - `renderPicks()` → top 3 of `LIVE.ranked` (heading copy in Task 4); pipeline `x.pick` renders as a small `★ rater's pick` badge on cards instead of gating the section.
  - `renderLeaderboard()` → rows = `[...LIVE.ranked, ...goneRows]` where `goneRows = D.listings.filter(x => x.gone && !isHidden(x))` (struck-through at bottom, as today); insert `<div class="lbcut"></div>` after row N when `WSTATE.topN !== "all"` and fade rows past it:

```css
.lbcut{border-top:1px dashed var(--line,#d8d0c4);margin:4px 0}
.lbrow.offboard{opacity:.45}
```

- [ ] **Step 3: Single entry point.** Make `refreshReachedUI()` (existing full-rerender helper) call `recomputeAll()` first; register `onWeightsChanged(refreshReachedUI)`; replace the initial render sequence's first call with `recomputeAll()` before any render. Hide ✕ toggles already call `refreshReachedUI()` so hidden listings automatically free board slots.

- [ ] **Step 4: Verify empty state.** With price slider at minimum, grid should show the existing `Nothing matches these filters.` empty div, map shows only anchors, leaderboard shows only gone rows. Fix anything that throws on `boardIds.size === 0`.

- [ ] **Step 5: Syntax + tests + browser check.** `node --check` extract; `node tests/houses/parity.test.mjs`; then in the browser: (a) defaults → board identical to shipped top-20 order, (b) drag gym to 100 → gym-close listings enter the map, dots swap, (c) rank badges renumber, (d) popup "strongest" line changes with weights, (e) reset restores, (f) reload persists.

- [ ] **Step 6: Commit.** `git commit -am "feat(houses): sliders drive live re-ranking of map, grid, leaderboard, picks"`

---

### Task 4: Copy pass — neutral header + explainers

**Files:**
- Modify: `public/houses/index.html` masthead (~line 230), map `h2note`, picks `<h2>`/`picknote`, footer fit-explainer sentence (grep `gym` in the footer block).

- [ ] **Step 1: Replace masthead:**

```html
<h1>Bay Area rental scout</h1>
<p class="sub">Real, currently-listed Bay Area rentals — scraped, mapped, photographed, and LLM-rated
  daily. Set what matters to you below and the whole board re-ranks: the map, the ranking, and the
  top picks all follow your sliders.</p>
```

- [ ] **Step 2: Picks + map copy.** `<h2>Top picks</h2>` → `<h2>Top 3 right now</h2>`; replace the Alex-specific `picknote` JS string (grep `My ${picks.length} favorites`) with: `` `The ${n} best fits under your current sliders. ★ marks the daily rater's own pick.` ``. In the map `h2note`, replace the sentence about fixed rank with "…numbered by rank under <b>your sliders</b> (1 = best); only the top N you've chosen appear —the full ranking stays in the list on the right."

- [ ] **Step 3: Check for other "Alex" copy** — `grep -n -i "alex" public/houses/index.html` — neutralize board copy but LEAVE: the Gio section (his rubric mentions Alex's caps), the reach-out draft message (it's genuinely from Alex), and the footer credit. Judgment: copy that describes *the board* goes neutral; copy that *is Alex speaking* stays.

- [ ] **Step 4: Verify + commit.** `node --check`; browser skim; `git commit -am "feat(houses): neutral masthead + slider-aware copy"`

---

### Task 5: Final verification, rebase, push, cleanup

- [ ] **Step 1: Full checks.** `node tests/houses/parity.test.mjs` && `node --check` extract && browser pass of the Task 3 Step 5 checklist once more.
- [ ] **Step 2: Rebase.** `git pull --rebase origin master` (data.js refreshes 4×/day and the multisource agent may have landed — if conflicts touch files we don't own, stop and reconcile carefully). **Re-run the parity test after rebase** — data.js may have changed.
- [ ] **Step 3: Push.** `git push origin sliders:master`. Verify with `git show origin/master:public/houses/fitmath.js | head -5`.
- [ ] **Step 4: Browser-verify production** once Vercel deploys (real browser, not curl — Vercel checkpoint trap).
- [ ] **Step 5: Cleanup + memory.** ExitWorktree(keep) → `git worktree remove --force .claude/worktrees/sliders && git branch -D sliders && git pull --rebase --autostash origin master`. Update `houses-daily-refresh-routine.md` memory (sliders feature bullet: fitmath.js, parity test, WKEY, top-N semantics) and add the desired-behavior bullet to the repo CLAUDE.md if one exists for this page.

## Self-Review

- **Spec coverage:** header (T4), instrument (T2), math+parity (T1), re-ranking incl. cutoff rule/picks/drivers (T3), edge cases (T1 tests + T3 Step 4), persistence (T2), testing (T1/T3/T5). Gio untouched (T3 Step 2 explicitly guards). ✓
- **Placeholder scan:** none; all code shown. Steps that say "grep first" are anchoring instructions for an actively-drifting file, with the replacement content given. ✓
- **Type consistency:** `FitMath.rankBoard` returns `{ranked, boardIds, fits}` consumed identically in T3; `WSTATE.topN` is `number|"all"` handled in both T2 (select) and T1 (rankBoard). `liveFit/liveRank/onBoard` defined T3 Step 1, used T3 Step 2. ✓
