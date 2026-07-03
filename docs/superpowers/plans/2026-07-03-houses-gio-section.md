# /houses "For Gio" Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a daily-refreshed, LLM-rated "For Gio" section to alex-loftus.com/houses showing listings walkable to OpenAI HQ (1455 Third St, Mission Bay), with price/type/walk toggles.

**Architecture:** Parallel sub-pipeline sharing the existing files and workflow. Gio candidates ride through `shortlist.json` tagged `aud: "gio"` with `G##` ids; `rate.py` swaps rubric per audience-pure batch; `--build` writes a separate `gio` object into `data.js`; `--sweep` prunes dead Gio listings. Alex's `listings` array behavior is regression-locked. Spec: `docs/superpowers/specs/2026-07-03-houses-gio-section-design.md`.

**Tech Stack:** Python 3 stdlib pipeline scripts (`public/houses/refresh/`), Anthropic SDK + Pillow (rate.py), vanilla JS + Leaflet (index.html), pytest (`backend/tests/`).

## Global Constraints

- Office anchor: `GIO_LAT, GIO_LON = 37.7700, -122.3888` (Nominatim-verified, "Uber Headquarters Building 1, 1455 3rd Street").
- Price caps: apartments $1,400–6,000; rooms $700–3,500.
- Walk model: `walk_min = haversine_mi × 1.3 × 20`, keep `walk_min <= 32`.
- Gio fit: `0.34·prox + 0.20·aesthetic + 0.16·nice + 0.14·value + 0.16·soft` (soft = quiet for apt / social for room); `prox = clamp(10 − max(0, walk_min − 8)/2.4, 0, 10)`.
- A Gio-pull failure must never break Alex's pipeline; zero fresh Gio rows → carry the previous `gio` object forward.
- Ids: Gio `G01…` (Alex `L##`, pinned `P##` — namespaces must stay disjoint).
- Run tests from the repo root: `cd backend && uv run pytest tests/test_refresh_pipeline.py -v` (backend venv; uv 0.4.23 quirks — install extras via `uv pip`).
- Every `.py` edit is auto-formatted by the ruff+black PostToolUse hook; don't hand-format.
- Commit after each green task; final push goes to `master` (worktree branch `gio-section`).

---

### Task 1: Pull side — office constants, parameterized `pull_raw`, `select_gio`, `do_pull` wiring

**Files:**
- Modify: `public/houses/refresh/refresh.py` (constants near line 54; `pull_raw` line 192; `do_pull` line 470)
- Test: `backend/tests/test_refresh_pipeline.py` (append)

**Interfaces:**
- Produces: `GIO_LAT/GIO_LON/GIO_OFFICE/GIO_CENTERS/GIO_QUERIES/GIO_WALK_MAX_MIN/GIO_MAX/GIO_BUCKET_CAP`; `gio_walk_min(lat, lon) -> int`; `select_gio(rows) -> list[dict]` (each row gains `walk_mi: float`, `walk_min: int`, `id: "G##"`, `aud: "gio"`); `pull_raw(centers=CENTERS, queries=QUERIES)`; shortlist rows now all carry `aud`; `pull_stats.json` gains `gio_pull_ok: bool`, `n_gio_raw: int`, `n_gio: int`.

- [ ] **Step 1: Write the failing tests** (append to `backend/tests/test_refresh_pipeline.py`)

```python
# ---- Gio section: walk math + selection (spec 2026-07-03-houses-gio-section)


def _gio_row(i, dlat=0.0, dlon=0.0, bucket="apt", price=3000, nimg=3):
    return {
        "pid": 9000 + i,
        "price": price,
        "pdisp": None,
        "beds": 1,
        "bucket": bucket,
        "hood": "mission bay",
        "lat": refresh.GIO_LAT + dlat,
        "lon": refresh.GIO_LON + dlon,
        "title": f"gio listing {i}",
        "url": f"https://www.craigslist.org/view/d/g/{9000 + i}",
        "img": "https://images.craigslist.org/a_b_c_600x450.jpg",
        "nimg": nimg,
    }


def test_gio_walk_min_zero_at_office():
    assert refresh.gio_walk_min(refresh.GIO_LAT, refresh.GIO_LON) == 0


def test_gio_walk_min_matches_haversine_model():
    lat, lon = refresh.GIO_LAT + 0.01, refresh.GIO_LON  # ~0.69 straight mi north
    mi = refresh.haversine_mi(lat, lon, refresh.GIO_LAT, refresh.GIO_LON)
    assert refresh.gio_walk_min(lat, lon) == round(mi * 1.3 * 20)
    assert 17 <= refresh.gio_walk_min(lat, lon) <= 19


def test_select_gio_filters_and_sorts():
    rows = [
        _gio_row(1, dlat=0.001),            # ~2 min
        _gio_row(2, dlat=0.015),            # ~27 min
        _gio_row(3, dlat=0.030),            # ~54 min -> dropped (too far)
        _gio_row(4, dlat=0.002, nimg=0),    # dropped (no photos)
        dict(_gio_row(5, dlat=0.002), lat=None),  # dropped (no geo)
        _gio_row(6, dlat=0.002, price=500),       # dropped (< $700)
    ]
    sel = refresh.select_gio(rows)
    assert [r["pid"] for r in sel] == [9001, 9002]
    assert sel[0]["walk_min"] <= sel[1]["walk_min"]
    assert all(r["aud"] == "gio" for r in sel)
    assert [r["id"] for r in sel] == ["G01", "G02"]
    assert all(0 < r["walk_mi"] < 2 for r in sel)


def test_select_gio_bucket_cap_and_max():
    rows = [_gio_row(i, dlat=0.0002 * i, bucket="apt") for i in range(1, 31)]
    rows += [_gio_row(40 + i, dlat=0.0002 * i, bucket="room") for i in range(1, 6)]
    sel = refresh.select_gio(rows)
    n_apt = sum(1 for r in sel if r["bucket"] == "apt")
    n_room = sum(1 for r in sel if r["bucket"] == "room")
    assert len(sel) <= refresh.GIO_MAX
    assert n_apt <= refresh.GIO_BUCKET_CAP  # rooms not crowded out
    assert n_room == 5


def test_select_gio_single_bucket_not_starved():
    rows = [_gio_row(i, dlat=0.0002 * i, bucket="apt") for i in range(1, 31)]
    sel = refresh.select_gio(rows)
    assert len(sel) == refresh.GIO_BUCKET_CAP  # cap, not stalled at 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v -k gio`
Expected: FAIL / AttributeError — `houses_refresh` has no attribute `GIO_LAT` / `gio_walk_min` / `select_gio`.

- [ ] **Step 3: Implement**

In `refresh.py`, after the `DT_LAT, DT_LON` block (line ~56):

```python
# OpenAI HQ (1455 Third St, Mission Bay) — anchor for the "For Gio" section.
# Geocoded via OSM Nominatim ("Uber Headquarters Building 1, 1455, 3rd Street").
GIO_LAT, GIO_LON = 37.7700, -122.3888
GIO_OFFICE = {
    "name": "OpenAI HQ",
    "addr": "1455 Third St, Mission Bay",
    "lat": GIO_LAT,
    "lon": GIO_LON,
}
GIO_CENTERS = [("gio_openai", GIO_LAT, GIO_LON, 2)]
# (category, min_price, max_price, bucket) — Gio's budget, NOT Alex's
GIO_QUERIES = [("apa", 1400, 6000, "apt"), ("roo", 700, 3500, "room")]
GIO_WALK_FACTOR = 1.3 * 20  # straight-line mi -> minutes (route factor x 20 min/mi)
GIO_WALK_MAX_MIN = 32
GIO_MAX = 24  # shortlist slots for the Gio section
GIO_BUCKET_CAP = 16  # neither bucket may fill more than 2/3 of the section
```

Change the `pull_raw` signature (line ~192) — body unchanged, loop reads the params:

```python
def pull_raw(centers=CENTERS, queries=QUERIES):
    seen = {}
    for cname, lat, lon, dist in centers:
        for cat, pmin, pmax, bucket in queries:
```

After `select_shortlist` (line ~435), add:

```python
def gio_walk_min(lat, lon):
    """Estimated walking minutes from (lat, lon) to the OpenAI office."""
    return round(haversine_mi(lat, lon, GIO_LAT, GIO_LON) * GIO_WALK_FACTOR)


def select_gio(rows):
    """Walkable-to-OpenAI candidates: filter, walk-sort, bucket-cap, assign G-ids."""
    keep = []
    for r in rows:
        if not (r["lat"] and r["lon"]) or r["nimg"] == 0 or r["price"] < 700:
            continue
        wm = gio_walk_min(r["lat"], r["lon"])
        if wm > GIO_WALK_MAX_MIN:
            continue
        r["walk_mi"] = round(haversine_mi(r["lat"], r["lon"], GIO_LAT, GIO_LON), 2)
        r["walk_min"] = wm
        keep.append(r)
    keep.sort(key=lambda r: r["walk_min"])
    sel, nbucket = [], collections.Counter()
    for r in keep:
        if len(sel) >= GIO_MAX:
            break
        if nbucket[r["bucket"]] >= GIO_BUCKET_CAP:
            continue
        sel.append(r)
        nbucket[r["bucket"]] += 1
    for i, r in enumerate(sel):
        r["id"] = f"G{i + 1:02d}"
        r["aud"] = "gio"
    return sel
```

In `do_pull` (line ~470): after the `for i, r in enumerate(sel): r["id"] = f"L{i + 1:02d}"` loop, insert the Gio pull + tag Alex rows; extend the scrape and writes:

```python
    for r in sel:
        r["aud"] = "alex"
    # Gio section: office-centered pull. Failure must never block Alex's pipeline.
    gio_sel, gio_ok, n_gio_raw = [], True, 0
    try:
        gio_rows = pull_raw(GIO_CENTERS, GIO_QUERIES)
        n_gio_raw = len(gio_rows)
        gio_sel = select_gio(gio_rows)
    except Exception as e:
        gio_ok = False
        print(f"warn: gio pull failed ({e}) — continuing without a Gio refresh", file=sys.stderr)
    print(f"scraping {len(sel) + len(gio_sel)} listing pages (photos + body)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        scraped = dict(ex.map(scrape_page, sel + gio_sel))
    n_dead = n_body = 0
    for r in sel + gio_sel:
```

(the existing per-row gallery/body loop body is unchanged; it now iterates `sel + gio_sel`). Update the two writes and the summary prints at the end of `do_pull`:

```python
    json.dump(sel + gio_sel, open(SHORTLIST, "w"), indent=1)
    json.dump(
        {
            "n_raw": len(rows),
            "n_kept": len(keep),
            "n_shortlist": len(sel),
            "gio_pull_ok": gio_ok,
            "n_gio_raw": n_gio_raw,
            "n_gio": len(gio_sel),
            "pulled": datetime.date.today().isoformat(),
        },
        open(PULL_STATS, "w"),
    )
    print(
        f"wrote {SHORTLIST}: {len(sel)} candidates + {len(gio_sel)} gio "
        f"(raw={len(rows)} kept={len(keep)}) | bodies={n_body} dead={n_dead}"
    )
    print(f"gio: {'ok' if gio_ok else 'PULL FAILED'} — {len(gio_sel)} walkable (raw={n_gio_raw})")
    print("by region:", dict(collections.Counter(r["region"] for r in sel)))
    print("by bucket:", dict(collections.Counter(r["bucket"] for r in sel)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v`
Expected: all existing tests + 5 new gio tests PASS.

- [ ] **Step 5: Amend the spec's bucket-balance wording to the final rule**

In `docs/superpowers/specs/2026-07-03-houses-gio-section-design.md`, replace the selection sentence with: "Selection: sort by walk_min ascending; per-bucket cap of 16 (~2/3 of 24 — cannot starve when one bucket has no inventory); take up to 24."

- [ ] **Step 6: Commit**

```bash
git add public/houses/refresh/refresh.py backend/tests/test_refresh_pipeline.py docs/superpowers/specs/2026-07-03-houses-gio-section-design.md
git commit -m "houses: gio pull side — office-centered query, walk filter, G-id selection"
```

---

### Task 2: Build side — `gio_fit`, `build_gio`, `do_build` audience split + carry-forward

**Files:**
- Modify: `public/houses/refresh/refresh.py` (`do_build` line ~861; helpers before it)
- Test: `backend/tests/test_refresh_pipeline.py` (append)

**Interfaces:**
- Consumes: shortlist rows with `aud`, `walk_mi`, `walk_min` (Task 1); `gs()`, `extract_contact()`, `load_data_js()`, `write_data_js()` (existing).
- Produces: `gio_fit(scores, bucket, walk_min) -> (fit: float, prox: float)`; `build_gio(gio_rows, ratings, stats) -> dict | None`; `data.js` top-level `gio: {office, listings, meta}`; carry-forward of previous `gio` when `build_gio` returns None.

- [ ] **Step 1: Write the failing tests**

```python
# ---- Gio build: fit formula, assembly, alex-invariance, carry-forward


def test_gio_fit_formula():
    scores = {"nature": 5, "quiet": 7, "nice": 8, "social": 4, "value": 6, "aesthetic": 9}
    fit, prox = refresh.gio_fit(scores, "apt", 8)
    assert prox == 10.0
    assert fit == 8.4  # .34*10 + .20*9 + .16*8 + .14*6 + .16*7
    fit_room, _ = refresh.gio_fit(scores, "room", 8)
    assert fit_room == 8.0  # soft uses social(4) instead of quiet(7)
    _, prox32 = refresh.gio_fit(scores, "apt", 32)
    assert prox32 == 0.0
    _, prox20 = refresh.gio_fit(scores, "apt", 20)
    assert prox20 == 5.0


def _rating(lid, fit=7):
    return {
        "id": lid, "nature": 5, "quiet": 6, "nice": 7, "social": 5, "value": 6,
        "commute": 8, "aesthetic": 7, "fit": fit, "why": "fine",
        "live": True, "commercial": False,
    }


def _alex_row(i, hood="mission"):
    return {
        "id": f"L{i:02d}", "aud": "alex", "pid": i, "price": 1500 + 10 * i,
        "pdisp": None, "beds": 1, "bucket": "room" if i % 2 else "apt",
        "hood": hood, "lat": 37.76, "lon": -122.42,
        "url": f"https://www.craigslist.org/view/d/a/{1000 + i}",
        "img": "https://images.craigslist.org/a_b_c_600x450.jpg",
        "imgs": ["https://images.craigslist.org/a_b_c_600x450.jpg"], "nimg": 1,
        "title": f"alex listing {i}", "body": "", "region": "SF", "drive_min": 11,
    }


def _gio_short_row(i, walk_min=10):
    r = _gio_row(i, dlat=0.001 * i)
    r.update(
        id=f"G{i:02d}", aud="gio", walk_mi=round(walk_min / 26, 2), walk_min=walk_min,
        imgs=[r["img"]], body="text me at 415-555-0142", beds=0,
    )
    return r


def _run_build(tmp_path, monkeypatch, shortlist, ratings, stats, prev_data=None):
    sl, rt, ps, dj = (tmp_path / n for n in ("s.json", "r.json", "p.json", "data.js"))
    import json as _json

    sl.write_text(_json.dumps(shortlist))
    rt.write_text(_json.dumps(ratings))
    ps.write_text(_json.dumps(stats))
    if prev_data is not None:
        dj.write_text("window.HOUSES_DATA = " + _json.dumps(prev_data) + ";\n")
    for attr, p in [("SHORTLIST", sl), ("RATINGS", rt), ("PULL_STATS", ps), ("DATA_JS", dj)]:
        monkeypatch.setattr(refresh, attr, str(p))
    monkeypatch.setattr(refresh, "fetch_reached_urls", lambda: set())
    refresh.do_build()
    return refresh.load_data_js()


def test_build_gio_section_and_alex_invariance(tmp_path, monkeypatch):
    alex = [_alex_row(i) for i in range(1, 17)]
    gio = [_gio_short_row(1, walk_min=5), _gio_short_row(2, walk_min=25)]
    ratings = [_rating(r["id"]) for r in alex + gio]
    stats = {"n_kept": 100, "n_shortlist": 16, "gio_pull_ok": True, "n_gio_raw": 40, "n_gio": 2}
    d_both = _run_build(tmp_path / "a", monkeypatch, alex + gio, ratings, stats)
    (tmp_path / "b").mkdir()
    d_alex = _run_build(
        tmp_path / "b", monkeypatch, alex, [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16},
    )
    assert d_both["listings"] == d_alex["listings"]  # Gio rows never perturb Alex's board
    assert "gio" not in d_alex
    g = d_both["gio"]
    assert [x["id"] for x in g["listings"]] == ["G01", "G02"]  # closer walk -> higher fit
    assert g["listings"][0]["fit"] > g["listings"][1]["fit"]
    assert g["listings"][0]["scores"]["commute"] == 10.0
    assert g["listings"][0]["contact_phone"] == "(415) 555-0142"
    assert g["office"]["lat"] == refresh.GIO_LAT
    assert g["meta"]["n_shown"] == 2 and g["meta"]["n_scouted"] == 40
    assert not any(x["id"].startswith("G") for x in d_both["listings"])


def test_build_gio_carry_forward_on_failed_pull(tmp_path, monkeypatch):
    alex = [_alex_row(i) for i in range(1, 17)]
    prev_gio = {
        "office": dict(refresh.GIO_OFFICE),
        "listings": [{"id": "G01", "url": "https://x", "fit": 7.0, "price": 3000}],
        "meta": {"generated": "2026-07-02", "n_shown": 1},
    }
    prev = {"meta": {}, "listings": [], "neighborhoods": [], "gio": prev_gio}
    d = _run_build(
        tmp_path, monkeypatch, alex, [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16, "gio_pull_ok": False, "n_gio": 0},
        prev_data=prev,
    )
    assert d["gio"] == prev_gio  # carried forward verbatim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v -k "gio_fit or build_gio"`
Expected: AttributeError — no `gio_fit` / `build_gio`.

- [ ] **Step 3: Implement**

In `refresh.py` before `do_build`:

```python
GIO_W = {"prox": 0.34, "aesthetic": 0.20, "nice": 0.16, "value": 0.14, "soft": 0.16}


def gio_fit(scores, bucket, walk_min):
    """(fit, prox) — deterministic Gio ranking: walk time first, then looks/value."""
    prox = max(0.0, min(10.0, 10 - max(0, walk_min - 8) / 2.4))
    soft = gs(scores, "quiet") if bucket == "apt" else gs(scores, "social")
    fit = (
        GIO_W["prox"] * prox
        + GIO_W["aesthetic"] * gs(scores, "aesthetic")
        + GIO_W["nice"] * gs(scores, "nice")
        + GIO_W["value"] * gs(scores, "value")
        + GIO_W["soft"] * soft
    )
    return round(min(10.0, fit), 1), round(prox, 1)


def build_gio(gio_rows, ratings, stats):
    """Assemble the data.gio object. Returns None when there is nothing fresh
    (pull failed / zero G rows) so the caller carries the previous section forward."""
    if not gio_rows:
        return None
    listings = []
    for r in gio_rows:
        rt = ratings.get(r["id"])
        if rt is None or rt.get("live") is False or rt.get("commercial") is True:
            continue
        if (rt.get("fit") or 0) <= 2:
            continue
        src = rt.get("scores", rt)
        scores = {
            k: src.get(k)
            for k in ["nature", "quiet", "nice", "social", "value", "commute", "aesthetic"]
        }
        fit, prox = gio_fit(scores, r["bucket"], r["walk_min"])
        scores["commute"] = prox
        gallery = r.get("imgs") or ([r["img"]] if r.get("img") else [])
        email, phone = extract_contact(r.get("body", ""))
        listings.append(
            {
                "id": r["id"],
                "price": r["price"],
                "pdisp": r.get("pdisp") or f"${r['price']:,}",
                "beds": r.get("beds"),
                "bucket": r["bucket"],
                "hood": r["hood"],
                "lat": r["lat"],
                "lon": r["lon"],
                "url": r["url"],
                "img": gallery[0] if gallery else r.get("img"),
                "imgs": gallery,
                "nimg": len(gallery),
                "title": r.get("title", ""),
                "walk_mi": r["walk_mi"],
                "walk_min": r["walk_min"],
                "fit": fit,
                "scores": scores,
                "rationale": rt.get("rationale") or rt.get("why") or "",
                "contact_email": email,
                "contact_phone": phone,
            }
        )
    by_url = {}
    for x in sorted(listings, key=lambda x: -(x["fit"] or 0)):
        by_url.setdefault(x["url"], x)
    listings = sorted(by_url.values(), key=lambda x: -(x["fit"] or 0))
    prices = [x["price"] for x in listings]
    return {
        "office": dict(GIO_OFFICE),
        "listings": listings,
        "meta": {
            "generated": datetime.date.today().isoformat(),
            "n_scouted": stats.get("n_gio_raw", len(gio_rows)),
            "n_shortlist": stats.get("n_gio", len(gio_rows)),
            "n_shown": len(listings),
            "price_min": min(prices) if prices else None,
            "price_max": max(prices) if prices else None,
            "price_med": int(statistics.median(prices)) if prices else None,
        },
    }
```

In `do_build`, change the shortlist load (line ~869) to split by audience:

```python
    rows_all = json.load(open(SHORTLIST))
    sel = {r["id"]: r for r in rows_all if r.get("aud", "alex") == "alex"}
    gio_rows = [r for r in rows_all if r.get("aud") == "gio"]
```

After the existing `stats` load and before `meta = {...}`, add:

```python
    gio = build_gio(gio_rows, ratings, stats)
    if gio is None:
        gio = (load_data_js() or {}).get("gio")
        if gio:
            print("gio: no fresh data — carrying previous section forward")
```

And attach it after the `data = {...}` literal:

```python
    if gio:
        data["gio"] = gio
```

Extend the final summary print:

```python
    print(
        f"wrote {DATA_JS}: shown={len(listings)} picks={len(picks)} "
        f"neighborhoods={len(neighborhoods)} gio={len((gio or {}).get('listings', []))}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v`
Expected: all PASS (existing + Task 1 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add public/houses/refresh/refresh.py backend/tests/test_refresh_pipeline.py
git commit -m "houses: gio build side — walk-first fit, data.gio object, carry-forward"
```

---

### Task 3: Sweep covers Gio listings

**Files:**
- Modify: `public/houses/refresh/refresh.py` (`do_sweep` line ~818)
- Test: `backend/tests/test_refresh_pipeline.py` (append)

**Interfaces:**
- Consumes: `data.gio.listings` (Task 2), `check_live`, `fetch_reached_urls`.
- Produces: sweep prunes positive-dead Gio listings; skips Gio changes when >60% flagged dead at once; Alex guard (`len(kept) < 10`) untouched.

- [ ] **Step 1: Write the failing tests**

```python
def _sweep_fixture(tmp_path, monkeypatch, n_alex=12, gio_urls=()):
    import json as _json

    listings = [
        dict(_alex_row(i), fit=6.0, scores={}, pick=False)
        for i in range(1, n_alex + 1)
    ]
    gio_listings = [
        {"id": f"G{j:02d}", "url": u, "fit": 7.0, "price": 3000, "hood": "mission bay"}
        for j, u in enumerate(gio_urls, 1)
    ]
    data = {
        "meta": {"n_shown": n_alex},
        "listings": listings,
        "neighborhoods": [],
        "searchlinks": [],
        "gio": {
            "office": dict(refresh.GIO_OFFICE),
            "listings": gio_listings,
            "meta": {"generated": "2026-07-03", "n_shown": len(gio_listings)},
        },
    }
    dj = tmp_path / "data.js"
    dj.write_text("window.HOUSES_DATA = " + _json.dumps(data) + ";\n")
    monkeypatch.setattr(refresh, "DATA_JS", str(dj))
    monkeypatch.setattr(refresh, "fetch_reached_urls", lambda: set())
    monkeypatch.setattr(refresh, "check_live", lambda url: not url.endswith("dead"))
    return dj


def test_sweep_prunes_dead_gio_listing(tmp_path, monkeypatch):
    _sweep_fixture(
        tmp_path, monkeypatch,
        gio_urls=["https://g/1", "https://g/2dead", "https://g/3", "https://g/4", "https://g/5"],
    )
    refresh.do_sweep()
    d = refresh.load_data_js()
    assert [x["id"] for x in d["gio"]["listings"]] == ["G01", "G03", "G04", "G05"]
    assert d["gio"]["meta"]["n_shown"] == 4
    assert len(d["listings"]) == 12  # alex untouched


def test_sweep_gio_mass_death_guard(tmp_path, monkeypatch):
    dj = _sweep_fixture(
        tmp_path, monkeypatch,
        gio_urls=["https://g/1dead", "https://g/2dead", "https://g/3dead",
                  "https://g/4dead", "https://g/5"],
    )
    before = dj.read_text()
    refresh.do_sweep()
    assert dj.read_text() == before  # 80% dead at once -> scrape problem, no write
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v -k sweep`
Expected: FAIL — gio listings unpruned (first test) since do_sweep ignores `gio`.

- [ ] **Step 3: Implement** — in `do_sweep`, after `listings = data["listings"]`:

```python
    gio = data.get("gio") or {}
    gio_listings = gio.get("listings") or []
```

Change the liveness check to cover both arrays:

```python
    with ThreadPoolExecutor(max_workers=4) as ex:
        alive = dict(
            ex.map(lambda x: (x["url"], check_live(x["url"])), listings + gio_listings)
        )
```

After the Alex kept/pruned loop and before the `len(kept) < 10` guard, add:

```python
    # Gio: prune positive-dead only. A >60% single-sweep wipe is more likely a
    # scrape problem than reality — skip Gio changes entirely in that case.
    g_kept = [x for x in gio_listings if alive.get(x["url"]) is not False]
    n_gdead = len(gio_listings) - len(g_kept)
    if n_gdead and len(g_kept) < 0.4 * len(gio_listings):
        print(
            f"gio sweep: {n_gdead}/{len(gio_listings)} flagged dead at once — "
            f"likely a scrape problem, keeping gio unchanged."
        )
        g_kept, n_gdead = gio_listings, 0
```

Change the no-op early return and apply Gio changes before `write_data_js`:

```python
    if not pruned and not newly_gone and not n_gdead:
        print(f"sweep: all {len(kept)} listings still live; no changes.")
        return
```

```python
    if n_gdead:
        gio["listings"] = g_kept
        gio.setdefault("meta", {})["n_shown"] = len(g_kept)
        data["gio"] = gio
```

Extend the final print:

```python
    print(
        f"sweep: pruned {len(pruned)} dead {pruned}, flagged {newly_gone} contacted-as-gone, "
        f"pruned {n_gdead} gio, kept {len(kept)}."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_refresh_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add public/houses/refresh/refresh.py backend/tests/test_refresh_pipeline.py
git commit -m "houses: sweep prunes dead gio listings (with mass-death guard)"
```

---

### Task 4: rate.py — Gio rubric, audience-pure batches, walk line

**Files:**
- Modify: `public/houses/refresh/rate.py`
- Test: Create `backend/tests/test_rate_batching.py`

**Interfaces:**
- Consumes: shortlist rows with `aud` / `walk_min` (Task 1).
- Produces: `GIO_RUBRIC: str`; `RUBRICS = {"alex": RUBRIC, "gio": GIO_RUBRIC}`; `group_batches(sel, batch_size) -> list[tuple[str, list[dict]]]`; `rate_batch(client, batch, montages, rubric)`; `listing_text(r)` shows the walk line for gio rows. ratings.json shape unchanged.

- [ ] **Step 1: Write the failing tests** (new file `backend/tests/test_rate_batching.py`)

```python
"""rate.py pure parts: audience grouping + gio listing text.

rate.py imports anthropic + PIL at module level; skip cleanly where absent.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("PIL")

_RATE = Path(__file__).resolve().parents[2] / "public" / "houses" / "refresh" / "rate.py"
_spec = importlib.util.spec_from_file_location("houses_rate", _RATE)
rate = importlib.util.module_from_spec(_spec)
sys.modules["houses_rate"] = rate
_spec.loader.exec_module(rate)


def _row(lid, aud):
    return {
        "id": lid, "aud": aud, "price": 2000, "hood": "mission bay", "region": "SF",
        "bucket": "room", "title": "t", "body": "", "walk_min": 9,
    }


def test_group_batches_never_mixes_audiences():
    sel = [_row(f"L{i:02d}", "alex") for i in range(1, 8)]
    sel += [_row(f"G{i:02d}", "gio") for i in range(1, 4)]
    batches = rate.group_batches(sel, 5)
    assert [(aud, len(b)) for aud, b in batches] == [("alex", 5), ("alex", 2), ("gio", 3)]
    for aud, b in batches:
        assert all(r.get("aud", "alex") == aud for r in b)
    got = [r["id"] for _, b in batches for r in b]
    assert got == [r["id"] for r in sel]  # nothing dropped, stable order


def test_rubrics_exist_and_differ():
    assert set(rate.RUBRICS) == {"alex", "gio"}
    assert "OpenAI" in rate.RUBRICS["gio"] and "FAR Labs" not in rate.RUBRICS["gio"]
    assert "networking" in rate.RUBRICS["alex"]


def test_listing_text_gio_walk_line():
    txt = rate.listing_text(_row("G01", "gio"))
    assert "9 min walk to OpenAI HQ" in txt
    txt_alex = rate.listing_text(_row("L01", "alex"))
    assert "walk to OpenAI" not in txt_alex and "(SF)" in txt_alex
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_rate_batching.py -v` (if anthropic/PIL missing locally: `uv pip install anthropic pillow` into the backend venv first — test-env only, not a pyproject change)
Expected: AttributeError — no `group_batches` / `RUBRICS`.

- [ ] **Step 3: Implement**

Add after the existing `RUBRIC` in `rate.py`:

```python
GIO_RUBRIC = """You are rating SF rental listings for GIO, for the "For Gio" section of the live board at alex-loftus.com/houses.

WHO IT'S FOR — Gio: works AT OpenAI's headquarters (1455 Third St, Mission Bay, San Francisco). His #1 criterion is a SHORT WALK to that office — each listing below states its computed walk time; treat it as ground truth and sanity-check any location claims in the text against it. Budget is flexible (rooms up to ~$3,500, apartments/studios up to ~$6,000) but he still wants his money's worth. He wants a place that looks genuinely nice; friendly housemates are a plus when it's a shared place — young-professional vibe welcome; he has NO networking agenda (he already works at OpenAI).

Each listing below has a photo MONTAGE (a grid of ALL its photos) and its posting text. LOOK AT EVERY PHOTO in each montage before scoring that listing.

Score each dimension as an integer 1-10, literally as defined:
- nature — greenery/water nearby: bay waterfront, Mission Bay parks, street trees (text + what the photos show). 10 = right on a park/waterfront; 1 = bare concrete canyon.
- quiet — residential calm; low traffic/noise; active construction (common in Mission Bay) counts against. 10 = quiet; 1 = above a bar / loud arterial / construction next door.
- nice — how desirable/safe/well-kept the area & unit are, per the photos. 10 = clearly nice & good shape; 1 = rough/run-down.
- social — housemate/vibe signal for a shared place: friendly, sociable, young-professional households HIGH (7-10); generic shared place with no signal MID (4-6); a solo studio/1BR with no housemates LOW (1-3) on this dimension — but that must NOT drag your overall read down (the final ranking handles it).
- value — price vs. what you get (space/condition/location per the photos) for near-office San Francisco. 10 = underpriced for what it is; 1 = overpriced.
- commute — your judgment of the walk to OpenAI HQ given the stated walk time and the text (a deterministic model recomputes this downstream; score it honestly anyway). 10 = a few minutes' stroll; 1 = not realistically walkable.
- aesthetic — HOW GOOD THE PLACE LOOKS IN ITS PHOTOS, judged ONLY from the montage. Attractive, well-lit, tasteful, clean, good finishes/light/views? 10 = genuinely beautiful, magazine-quality photos of an attractive space. 5 = ordinary/plain but fine. 1-3 = ugly, dark, cluttered, grimy, low-effort/blurry photos, or a clearly unappealing space (or no usable photos). This is weighted into the ranking — do not give high aesthetic to a place whose pictures are bad.
Then:
- fit — overall 1-10 holistic fit for Gio (a deterministic model recomputes the final ranking; give your honest overall anyway). Short walk + nice-looking place = high fit; a gorgeous place 30+ minutes away is NOT high fit.
- why — one line in Gio's terms, referencing what the photos/listing show and the walk time.
- live — false if the post looks dead/expired/duplicate or like a scam (too cheap for the area, generic copy, off-platform payment).
- commercial — true if it's an office/retail/parking/commercial space, not a home.

Rate EVERY listing id you are given, once each."""

RUBRICS = {"alex": RUBRIC, "gio": GIO_RUBRIC}
```

Replace `listing_text`:

```python
def listing_text(r):
    body = (r.get("body") or "")[:1500]
    if r.get("aud") == "gio":
        place = f"~{r.get('walk_min', '?')} min walk to OpenAI HQ (1455 Third St) — {r.get('hood')}"
    else:
        place = f"{r.get('hood')} ({r.get('region')})"
    return (
        f"### Listing {r['id']} — ${r['price']:,}/mo — {place} — "
        f"{'room in shared home' if r.get('bucket') == 'room' else 'apartment/studio'}\n"
        f"Title: {r.get('title')}\nPosting text: {body or '(no body scraped)'}"
    )
```

Add `group_batches` and thread the rubric through `rate_batch`:

```python
def group_batches(sel, batch_size):
    """[(aud, rows)] — batches never mix audiences; stable order within each."""
    out = []
    for aud in ("alex", "gio"):
        rows = [r for r in sel if r.get("aud", "alex") == aud]
        out += [(aud, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]
    return out
```

```python
def rate_batch(client, batch, montages, rubric):
    content = [{"type": "text", "text": rubric}]
```

In `main()`, replace the batch construction and loop head:

```python
    batches = group_batches(sel, args.batch_size)
    for bi, (aud, batch) in enumerate(batches):
```

and the call site inside the retry loop:

```python
                obj, usage = rate_batch(client, todo, montages, RUBRICS[aud])
```

Also update the startup print to show the split:

```python
    n_gio = sum(1 for r in sel if r.get("aud") == "gio")
    print(
        f"rating {len(sel)} listings ({len(sel) - n_gio} alex + {n_gio} gio) "
        f"with {MODEL} (batch={args.batch_size})"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/test_rate_batching.py tests/test_refresh_pipeline.py -v`
Expected: all PASS (or the new file SKIPPED if anthropic/PIL truly unavailable — install then).

- [ ] **Step 5: Commit**

```bash
git add public/houses/refresh/rate.py backend/tests/test_rate_batching.py
git commit -m "houses: gio rubric + audience-pure rating batches"
```

---

### Task 5: Frontend — section, chips, cards, map layer

**Files:**
- Modify: `public/houses/index.html`

**Interfaces:**
- Consumes: `data.js` `gio: {office, listings, meta}` (Task 2); existing `chips()`, `glyphRow()`, `anchor()`, `wireGalleries()`, `highlightMarker()`, `flyToListing()`, `markerById`, `IMAGES/IDX`, `fitColor()`, `money()`, `bedsLabel()`.
- Produces: `#gio` section with price/type/walk chips + sort; Gio markers registered in the shared `markerById` (unique G-ids) so hover/fly/popup wiring is shared; OpenAI ◆ anchor; legend + stats entries; graceful empty state when `D.gio` is absent.

- [ ] **Step 1: CSS + HTML skeleton**

Add `--gio:#5a5f9e;` to the `:root` block (after `--pick:#977a1c;`). Insert between the `#grid` div and the "Neighborhood scorecards" `<h2>`:

```html
  <h2 id="gio">For Gio — walkable to OpenAI HQ</h2>
  <p class="h2note">Gio works at OpenAI's Mission Bay HQ (1455 Third St — the ◆ on the map). Everything here is a
    ≤30-minute walk from the office, refreshed and rated daily with Gio's own rubric: walk time first, then how
    good the place looks, then value. Prices run higher than Alex's caps on purpose — toggle below.</p>
  <div class="controls ui" id="gioControls">
    <div class="cgroup"><span class="lbl">Max price</span><span id="gPrice"></span></div>
    <div class="cgroup"><span class="lbl">Type</span><span id="gBkt"></span></div>
    <div class="cgroup"><span class="lbl">Walk</span><span id="gWalk"></span></div>
    <div class="cgroup"><span class="lbl">Sort</span>
      <select id="gioSort">
        <option value="fit">Best fit</option>
        <option value="walk">Walk time ↑</option>
        <option value="price">Price ↑</option>
        <option value="aesthetic">Best photos</option>
      </select></div>
    <div class="cgroup"><span id="gioCount" class="lbl"></span></div>
  </div>
  <div class="grid" id="gioGrid"></div>
```

- [ ] **Step 2: Gio JS block** — insert after the `renderGrid()` function definition:

```js
/* ---- For Gio: walkable to OpenAI HQ ---- */
const GIO = (D.gio && Array.isArray(D.gio.listings)) ? D.gio : {office:null, listings:[], meta:null};
const GRANK = {};
[...GIO.listings].sort((a,b)=>(b.fit||0)-(a.fit||0)).forEach((x,i)=>{ GRANK[x.id]=i+1; });
const gstate = {price:"All", bucket:"All", walk:"All", sort:"fit"};
const GPRICES=[["2000","≤$2k"],["3000","≤$3k"],["4500","≤$4.5k"],["All","all ≤$6k"]];
const GWALKS=[["10","≤10 min"],["20","≤20 min"],["All","≤30 min"]];
function gioCard(x){
  const imgs=(x.imgs&&x.imgs.length)?x.imgs:[x.img||FALLBACK];
  IMAGES[x.id]=imgs; IDX[x.id]=0; const n=imgs.length;
  const nav=n>1?`
      <button class="gnav gprev" data-id="${x.id}" data-dir="-1" aria-label="Previous photo" title="Previous photo">‹</button>
      <button class="gnav gnext" data-id="${x.id}" data-dir="1" aria-label="Next photo" title="Next photo">›</button>
      <span class="gcount" data-id="${x.id}">1/${n}</span>`:"";
  return `<article class="card ${x.gone?'gone':''}" data-id="${x.id}">
    <div class="ph" data-id="${x.id}">
      <img class="gimg" loading="lazy" src="${imgs[0]}" onerror="this.src='${FALLBACK}'" alt="">
      <span class="badge ${x.bucket==='apt'?'b-apt':'b-room'}">${x.bucket==='apt'?'Apartment':'Room'}</span>
      ${x.gone?'<span class="gone-badge">no longer listed</span>':''}${nav}
    </div>
    <div class="body">
      <div class="row1">
        <span class="price num">${money(x.price)}<small>/mo</small></span>
        <span class="fit num"><span class="rk" title="rank in Gio's section">#${GRANK[x.id]}</span><b style="color:${fitColor(x.fit)}">${x.fit==null?'–':x.fit}</b><span class="s">FIT</span></span>
      </div>
      <div class="metaline">${bedsLabel(x.beds)} · ${x.hood||"Mission Bay area"}<br>
        <span class="drv"><b class="num">${x.walk_min} min walk</b> to OpenAI HQ · <span class="num">${x.walk_mi} mi</span></span></div>
      ${glyphRow(x.scores)}
      <p class="rat">${x.rationale||""}</p>
      <div class="actions"><a class="view" href="${x.url}" target="_blank" rel="noopener">View listing ↗</a></div>
    </div>
  </article>`;
}
function gioFiltered(){
  let a=GIO.listings.filter(x=>
    (gstate.price==="All"||x.price<=+gstate.price) &&
    (gstate.bucket==="All"||x.bucket===gstate.bucket) &&
    (gstate.walk==="All"||x.walk_min<=+gstate.walk));
  const s=gstate.sort;
  a.sort((p,q)=> s==="price"? p.price-q.price : s==="walk"? p.walk_min-q.walk_min
      : s==="aesthetic"? (q.scores.aesthetic||0)-(p.scores.aesthetic||0) : (q.fit||0)-(p.fit||0));
  return a;
}
function renderGio(){
  if(!GIO.listings.length){
    document.getElementById("gioControls").style.display="none";
    document.getElementById("gioGrid").innerHTML =
      `<div class="empty">No walkable listings right now — this section refreshes daily at 7am PT.</div>`;
    return;
  }
  chips(document.getElementById("gPrice"),GPRICES,gstate.price,v=>{gstate.price=v;renderGio();});
  chips(document.getElementById("gBkt"),BUCKETS,gstate.bucket,v=>{gstate.bucket=v;renderGio();});
  chips(document.getElementById("gWalk"),GWALKS,gstate.walk,v=>{gstate.walk=v;renderGio();});
  const a=gioFiltered();
  document.getElementById("gioGrid").innerHTML=a.map(gioCard).join("")||`<div class="empty">Nothing matches these filters.</div>`;
  document.getElementById("gioCount").textContent=a.length+" shown";
  wireGalleries(document.getElementById("gioGrid"));
  document.querySelectorAll("#gioGrid .card").forEach(c=>{
    c.onmouseenter=()=>highlightMarker(c.dataset.id,true);
    c.onmouseleave=()=>highlightMarker(c.dataset.id,false);
    c.onclick=e=>{ if(e.target.closest("a,.gnav,.gcount"))return; flyToListing(c.dataset.id); };
  });
}
document.getElementById("gioSort").onchange=e=>{gstate.sort=e.target.value;renderGio();};
```

- [ ] **Step 3: Map layer** — after the two `anchor(...)` calls add:

```js
if(GIO.office) anchor(GIO.office.lat,GIO.office.lon,"OpenAI HQ").addTo(map);
```

After `drawMarkers` add:

```js
function gioPopup(x){
  const imgs=IMAGES[x.id]||[x.img||FALLBACK]; const i=IDX[x.id]||0; const n=imgs.length;
  const nav=n>1?`<button class="gnav gprev" data-id="${x.id}" data-dir="-1" aria-label="Previous photo">‹</button>
    <button class="gnav gnext" data-id="${x.id}" data-dir="1" aria-label="Next photo">›</button>
    <span class="gcount" data-id="${x.id}">${i+1}/${n}</span>`:"";
  return `<div class="popup">
    <div class="ph pop-ph" data-id="${x.id}"><img class="gimg" src="${imgs[i]}" onerror="this.src='${FALLBACK}'">${nav}</div>
    <div class="pp num">G#${GRANK[x.id]} · ${money(x.price)} <span style="font-size:12px;color:#777">${bedsLabel(x.beds)}</span>
      <b style="float:right;color:${fitColor(x.fit)}">${x.fit==null?'–':x.fit}</b></div>
    <div class="ph2">${x.hood||"Mission Bay area"}<br><b class="num">${x.walk_min} min walk</b> to OpenAI HQ</div>
    <div style="font-size:12px;margin:5px 0 4px">${x.rationale||""}</div>
    <div class="actions"><a href="${x.url}" target="_blank" rel="noopener">View listing ↗</a></div></div>`;
}
function gioMarkerIcon(x){
  const size=Math.round(18+(x.fit||5)*1.4);
  const fs=GRANK[x.id]>9?10:12;
  return L.divIcon({className:"rankmark-wrap",iconSize:[size,size],iconAnchor:[size/2,size/2],
    html:`<div class="rankmark ${x.gone?'gone-mk':''}" style="width:${size}px;height:${size}px;background:${css('--gio')};box-shadow:0 1px 3px rgba(0,0,0,.4);font-size:${fs}px">${GRANK[x.id]}</div>`});
}
function drawGioMarkers(list){
  list.slice().sort((a,b)=>(a.fit||0)-(b.fit||0)).forEach(x=>{
    if(x.lat==null||x.lon==null) return;
    const mk=L.marker([x.lat,x.lon],{icon:gioMarkerIcon(x),riseOnHover:true});
    mk.bindPopup(()=>gioPopup(x),{maxWidth:236,minWidth:220});
    mk.addTo(map); markerById[x.id]=mk;
  });
}
```

In `refreshReachedUI`, after `drawMarkers(D.listings);` add `drawGioMarkers(GIO.listings);`.
In the legend HTML, before the anchors line, add:

```js
    ${GIO.listings.length?`<span class="dot" style="background:${css('--gio')}"></span>Gio · walkable to OpenAI<br>`:""}
```

(convert the legend `d.innerHTML=` string to a template literal for the interpolation).
In the stats strip array, insert before the "Move by" entry:

```js
  ...(GIO.listings.length?[["For Gio", GIO.listings.length+" <small>walkable to OpenAI</small>"]]:[]),
```

In the footer template, append after the "Data mode" sentence:

```js
  ${GIO.listings.length?`<b>For Gio:</b> the “walkable to OpenAI” section is pulled the same way but rated with Gio's own rubric — walk time to 1455 Third St first, then photo quality and value.`:""}
```

At the bottom init block, after `drawMarkers(D.listings);` add `drawGioMarkers(GIO.listings);` and after `renderGrid();` add `renderGio();`.

- [ ] **Step 4: Verify JS parses + section renders**

```bash
cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section
python3 -c "
import re, pathlib
h = pathlib.Path('public/houses/index.html').read_text()
blocks = re.findall(r'<script(?![^>]*src)[^>]*>(.*?)</script>', h, re.S)
assert len(blocks) == 1, f'expected 1 inline script, got {len(blocks)}'
pathlib.Path('$CLAUDE_JOB_DIR/tmp/houses_inline.js').write_text(blocks[0])
print('extracted', len(blocks[0]), 'chars')"
node --check "$CLAUDE_JOB_DIR/tmp/houses_inline.js" && echo "JS OK"
```

Expected: `JS OK`. Then a render smoke test against the LIVE data.js (which has no `gio` yet — must show the empty state, proving backward compat).

- [ ] **Step 5: Commit**

```bash
git add public/houses/index.html
git commit -m "houses: For Gio section — chips, walk-first cards, OpenAI map layer"
```

---

### Task 6: Ship — full suite, push, CI full run, live verification

**Files:**
- Modify: none new (memory + push only)

- [ ] **Step 1: Full test suite**

Run: `cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section/backend && uv run pytest tests/ -v`
Expected: everything green (houses pipeline + rate batching + existing API tests).

- [ ] **Step 2: Rebase onto latest master and push**

```bash
cd /Users/alex/loftusa.github.io/.claude/worktrees/gio-section
git fetch origin master && git rebase origin/master
git push origin gio-section:master
```

- [ ] **Step 3: Dispatch a FULL refresh and watch it**

```bash
gh workflow run houses-refresh.yml --repo loftusa/loftusa.github.io -f mode=full
sleep 10 && gh run list --repo loftusa/loftusa.github.io --workflow houses-refresh.yml --limit 1
gh run watch <run-id> --repo loftusa/loftusa.github.io --exit-status
```

Expected: run succeeds; logs show `gio: ok — N walkable`, rating counts include gio batches, build prints `gio=N`.

- [ ] **Step 4: Verify live**

```bash
curl -s https://alex-loftus.com/houses/data.js | node -e "
let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{global.window={};eval(s);
const g=window.HOUSES_DATA.gio; console.log('gio listings:',g.listings.length,
'| walk range:',Math.min(...g.listings.map(x=>x.walk_min)),'-',Math.max(...g.listings.map(x=>x.walk_min)),
'| all rated:',g.listings.every(x=>Number.isInteger(x.scores.aesthetic)));});"
```

Expected: nonzero listings, walk range within 1–32, all rated true. Then eyeball https://alex-loftus.com/houses#gio.

- [ ] **Step 5: Update memory + wrap up**

Update `houses-daily-refresh-routine.md` (pipeline now includes the Gio sub-pipeline; cost ~$0.48/day) and report to Alex.

---

## Self-Review Notes

- Spec coverage: pull (T1), rate (T4), build+carry-forward (T2), sweep+guard (T3), frontend incl. map/stats/legend/empty state (T5), tests 1–7 of the spec map to T1–T4 test code, done-criteria (T6). Gio contact extraction covered in T2 (build_gio calls `extract_contact`; test asserts the phone).
- Type consistency: `select_gio` emits `walk_mi/walk_min/id/aud`; `build_gio` consumes exactly those; frontend consumes `walk_mi/walk_min/fit/scores/imgs` as produced. `group_batches` returns `(aud, rows)` tuples consumed by `main`'s loop. `GIO_CENTERS` (not CENTER) used in both definition and `do_pull`.
- No placeholders; every step has full code/commands.
