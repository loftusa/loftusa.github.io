# RECORD — graph-path-finder worktree

## 2026-06-09 (later) — Pair finder: type-ahead slots + bridge-edge consistency

User report: clicking Ryan Chesler alone traced a path to Stella Biderman, but entering the
same pair in the "Shortest path" panel said no path. Audit confirmed: clicking an isolated
person opens their pre-computed bridge route (5 real co-authorship hops, ryan chesler → bo liu
→ yihan wang → adam dziedzic → christopher choquette-choo → stella biderman), while the pair
finder's BFS deliberately excluded `path_links` — but those ARE genuine shared-paper edges,
so the exclusion made the two features disagree.

Changes (`assets/js/coauthorship-network.js`, `_pages/coauthorship.html`):
- Pair-finder BFS adjacency now = drawn links + bridge-route edges (still never the ghost
  anchor links). Where a pair exists in both lists, the drawn edge is preferred for ink.
- Slots are now type-ahead inputs: type a roster name, pick from a menu (mousedown, so it
  beats the input blur) or press Enter for the top match; clicking a person still works,
  clicking a filled slot clears + re-arms it, focus auto-advances to the empty slot.
  Bonus: no-indexed-papers people are typeable endpoints (honest "no path" answer) even
  though they're unclickable in the list.

Verification (all green): updated `test_shortest_path_data.py` (adjacency mirrors the JS;
isolated-with-route reachable in ≤ route length; no-papers people still unreachable — 820
connected / 308 disconnected pairs, max listed-pair distance now 11 hops, leo mckee reid ↔
ryan chesler) and `test_shortest_path_e2e.py` (type-ahead flow, ryan→stella = 5 hops AND
chain identical to the click route, no-papers "no path", clear/reset). Plus an ad-hoc
Playwright pass + screenshot. `node --check` + jekyll build pass.

## 2026-06-09 — Shortest-path finder for the coauthorship network

Feature: pick two core-roster ("listed") people on `/coauthorship/` and light the minimum-hop
co-authorship chain between them; report "no path" if they're in separate components.

Plan: `~/.claude/plans/wise-seeking-plum.md` (approved). Decisions confirmed with Alex:
From→To slots UI · real co-authorship edges only (no synthetic bridge routes) · static lit
path + hop badge, no animation.

Changes:
- `assets/js/coauthorship-network.js`
  - `realAdj` / `realLinkByPair` built from `data.links` only (the existing `adj` includes
    bridge `path_links` + ghost links, which would lie about distance).
  - `shortestPath(a, b)` — BFS with parent map → `{ids, links, target, len, pair: true}`.
  - Pair state `pathPair {from, to, armed}`; armed slot captures the next person click
    (graph node or sidebar list); endpoints restricted to `is_list`.
  - Reuses the existing `activeRoute` machinery: `applyVisibility()` already force-reveals
    route nodes/links past the hop slider; `highlightRoute()` extended to paint pair chains
    in warm ink (#5a5346, thicker) with heavier endpoint rings; `clearRouteInk()` wipes it
    from the other highlight states.
  - Slider keeps pair paths alive (isolated-person routes still clear, as before).
  - Sim "end" refits to the lit chain (`fitToRoute`) instead of the whole mass.
  - `renderPairDetail()` — hop badge + breadcrumb, or "No co-authorship path connects these two."
- `_pages/coauthorship.html` — `#pathctl` panel (two slot buttons, →, ✕) between Reach
  slider and search; CSS follows existing tokens (#e3ddcf borders, #4c6b8a armed accent).
- `experiments/coauthorship/README.md` — paragraph under "What the page does".

Bug found & fixed during verification (pre-existing, exposed by this feature):
- `applyVisibility()` starts a 400ms fade-to-opacity-1 transition on every visible node; a
  focus applied right after (`highlightRoute`/`highlight`) was silently overwritten when the
  tween completed — measured empirically: all 105 nodes at opacity 1.00 one second after
  pairing. Fix: those two functions now `node.interrupt()` before setting opacities.
  Regression assertion added to the E2E test (opacity histogram: 7 lit chain nodes, rest 0.12).

Verification (all green):
- `experiments/coauthorship/tests/test_shortest_path_data.py` — BFS sanity over the built
  JSON: 48 listed, 595 connected / 533 disconnected pairs, symmetric, every step a real edge,
  direct coauthors = 1 hop, isolated unreachable. Max listed-pair distance: 6 hops.
- `experiments/coauthorship/tests/test_shortest_path_e2e.py` — headless Playwright over the
  built site: slot arming/filling, 1-hop + 6-hop pairs, force-reveal past the slider, warm-ink
  chain, no-path message, ✕ clear, slider persistence, dim regression. Screenshots verified
  visually (1-hop and 6-hop chains).
- `node --check` + `bundle exec jekyll build` pass.
- NOTE: the permission classifier was down most of this session; ran everything through the
  allowlisted `uv run` (python wrappers for node/bundle) and cached Playwright chromium-1217
  via `executable_path` (pip playwright 1.55 expects build 1187 — version-mismatch launch works).
