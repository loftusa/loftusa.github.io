# /houses Map Glyphs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Scaled lean: three tasks, frontend-heavy, spec has every encoding rule — `docs/superpowers/specs/2026-07-03-houses-map-glyphs-design.md`.

**Goal:** Driver-hue dots (far zoom) + coxcomb-petal dots (zoom ≥ 13) on the /houses map, weights emitted from the build.

**Architecture:** `refresh.py` gains `ALEX_W`/`alex_fit` (pure refactor of the inline fit) and emits `meta.fit_weights`; `index.html` computes per-audience means + drivers client-side (meta weights, hardcoded fallback), recolors far-zoom markers, and swaps all markers to coxcomb SVGs across the zoom-13 threshold.

**Tech:** stdlib Python + pytest; vanilla JS/Leaflet divIcons (conic SVG wedges from the approved mockup).

---

### Task 1: `alex_fit` + `meta.fit_weights` (TDD)

**Files:** Modify `public/houses/refresh/refresh.py`; test `backend/tests/test_refresh_pipeline.py`.

- [ ] Failing tests: `alex_fit` pinned case — scores {nice 8, nature 9, quiet 9, social 4, value 9, aesthetic 8}, apt, dual 8.0 → **8.4** (0.17·8+0.15·9+0.13·9+0.13·9+0.26·8+0.16·8); room variant uses social (→ 7.7); `meta.fit_weights.alex == refresh.ALEX_W`, `.gio == refresh.GIO_W`, each sums to 1.0 (via existing `_run_build` fixture).
- [ ] Implement: `ALEX_W = {"nice":.17, "nature":.15, "soft":.13, "value":.13, "commute":.26, "aesthetic":.16}`; `def alex_fit(scores, bucket, dual_commute)` returning `round(min(10.0, Σ terms), 1)` with terms in the existing literal order; `do_build` calls it (loved bonus stays at the call site: `fit += 0.8` before cap — NOTE: cap currently applied inside round(min(...)); keep exact current semantics: compute base via alex_fit WITHOUT cap? Current code: fit = Σ; if loved: fit += 0.8; x["fit"] = round(min(10.0, fit), 1). So `alex_fit` must return the UNROUNDED, UNCAPPED sum; keep rounding/cap/loved at the call site so values stay byte-identical.) `meta["fit_weights"] = {"alex": ALEX_W, "gio": GIO_W}`.
- [ ] All tests green; commit.

### Task 2: frontend glyphs

**Files:** Modify `public/houses/index.html`.

- [ ] Driver machinery after the GIO block: component table (reusing CSS vars), `W = D.meta.fit_weights || fallback`, per-audience means over non-gone listings, `driver(x)` → `{key,label,color}` or neutral `#a39c8d` when best lift ≤ 0.02.
- [ ] `markerIcon(x)` / `gioMarkerIcon(x)` become zoom-aware: far = existing circle with `background: driverColor`, `border:2.5px solid` type hue (apt/room/Gio indigo); near (zoom ≥ 13) = 34px coxcomb SVG from the mockup (fixed wedge order from 12 o'clock, radius 4+s/10·13, paper gaps, center disc = type hue + rank). Pick/reached/gone treatments preserved in both modes.
- [ ] `map.on("zoomend")` swap: track `glyphMode`; when crossing 13, `setIcon` every marker in `markerById` (icon factories re-read current reached/gone state).
- [ ] Popup line (both `popupInner` and `gioPopup`): `strongest vs field: <b style="color:...">Nature</b>`.
- [ ] Legend rewrite per spec. `node --check` the extracted inline script.
- [ ] Commit.

### Task 3: verify + ship

- [ ] Local preview (http.server + Chrome): driver colors spread across ≥3 components, both zoom directions swap, popups correct, no console errors; screenshot far + near zoom.
- [ ] Full backend suite; restore chroma test side-effects if dirtied.
- [ ] Rebase on fresh origin/master, push `map-glyphs:master`.
- [ ] Dispatch `houses-refresh` full run; after it lands verify live `data.js` has `meta.fit_weights` and the live map uses it (JS check), plus visual pass.
- [ ] Memory + close-out.
