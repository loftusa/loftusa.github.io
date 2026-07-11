"""Regression tests for the /houses refresh pipeline's pure functions.

refresh.py lives under public/houses/refresh/ (it ships with the static page),
not in a package — import it by path. Network-touching parts (--pull/--build)
are exercised in production; these tests pin the scoring/parsing logic that an
unattended daily run depends on.
"""

import importlib.util
import sys
from pathlib import Path

_REFRESH = (
    Path(__file__).resolve().parents[2] / "public" / "houses" / "refresh" / "refresh.py"
)
_spec = importlib.util.spec_from_file_location("houses_refresh", _REFRESH)
refresh = importlib.util.module_from_spec(_spec)
sys.modules["houses_refresh"] = refresh
_spec.loader.exec_module(refresh)


# ---- priors(): neighborhood -> (region, drive_min, nature, quiet, nice, social)


def test_marina_is_sf_not_marin():
    # Regression: "marina" contains the substring "marin"; the NorthBay rule
    # once captured Fort Mason / Cow Hollow listings as NorthBay @ 35 min.
    for hood in ("marina / cow hollow", "Marina District", "cow hollow"):
        region, drive, *_ = refresh.priors(hood)
        assert region == "SF", hood
        assert drive == 13, hood


def test_marin_county_still_northbay():
    assert refresh.priors("marin county")[0] == "NorthBay"
    assert refresh.priors("mill valley")[:2] == ("NorthBay", 28)
    assert refresh.priors("sausalito")[:2] == ("NorthBay", 23)


def test_richmond_disambiguation():
    assert refresh.priors("richmond / seacliff")[0] == "SF"
    assert refresh.priors("inner richmond")[0] == "SF"
    assert refresh.priors("richmond")[0] == "EastBay"  # the East Bay city


def test_unknown_hood_gets_default():
    assert refresh.priors("nowhereville") == refresh.DEFAULT_PRIOR
    assert refresh.priors(None) == refresh.DEFAULT_PRIOR


# ---- berk_drive(): hood/region -> off-peak minutes to downtown Berkeley


def test_berk_drive_spot_values():
    assert refresh.berk_drive("berkeley", "EastBay") == 6
    assert refresh.berk_drive("sausalito", "NorthBay") == 38
    assert refresh.berk_drive("marina / cow hollow", "SF") == 29  # central-SF default
    assert refresh.berk_drive("mill valley", "NorthBay") == 42


def test_berk_drive_mill_valley_not_caught_by_mills():
    # "mills" (Mills College area) must not substring-match "mill valley".
    assert refresh.berk_drive("mill valley", "NorthBay") != 18


# ---- cscore(): minutes -> 0..10


def test_cscore_bounds_and_monotonicity():
    assert refresh.cscore(0) == 10.0
    assert refresh.cscore(18) == 10.0  # <=18 min is perfect
    vals = [refresh.cscore(m) for m in range(18, 80, 5)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # non-increasing
    assert all(0.0 <= v <= 10.0 for v in vals)


# ---- extract_contact(): listing body -> (email, phone)


def test_extract_contact_email_and_phone():
    email, phone = refresh.extract_contact(
        "Contact me at jane.doe@gmail.com or call 510-555-0142 for a showing"
    )
    assert email == "jane.doe@gmail.com"
    assert phone == "(510) 555-0142"


def test_extract_contact_skips_craigslist_relay():
    email, _ = refresh.extract_contact("reply via abc-123@reply.craigslist.org only")
    assert email is None


def test_extract_contact_no_false_phone_from_specs():
    # prices, sqft, years, ZIP+4 must not parse as phone numbers
    _, phone = refresh.extract_contact(
        "Built in 1998, 2200 sqft, $2,100/mo, San Francisco CA 94123-1234"
    )
    assert phone is None


def test_extract_contact_empty_body():
    assert refresh.extract_contact("") == (None, None)
    assert refresh.extract_contact(None) == (None, None)


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
        _gio_row(1, dlat=0.001),  # ~2 min
        _gio_row(2, dlat=0.015),  # ~27 min
        _gio_row(3, dlat=0.030),  # ~54 min -> dropped (too far)
        _gio_row(4, dlat=0.002, nimg=0),  # dropped (no photos)
        dict(_gio_row(5, dlat=0.002), lat=None),  # dropped (no geo)
        _gio_row(6, dlat=0.002, price=500),  # dropped (< $700)
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


# ---- Gio build: fit formula, assembly, alex-invariance, carry-forward


def test_gio_fit_formula():
    scores = {
        "nature": 5,
        "quiet": 7,
        "nice": 8,
        "social": 4,
        "value": 6,
        "aesthetic": 9,
    }
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
        "id": lid,
        "nature": 5,
        "quiet": 6,
        "nice": 7,
        "social": 5,
        "value": 6,
        "commute": 8,
        "aesthetic": 7,
        "fit": fit,
        "why": "fine",
        "live": True,
        "commercial": False,
    }


def _alex_row(i, hood="mission"):
    return {
        "id": f"L{i:02d}",
        "aud": "alex",
        "pid": i,
        "price": 1500 + 10 * i,
        "pdisp": None,
        "beds": 1,
        "bucket": "room" if i % 2 else "apt",
        "hood": hood,
        "lat": 37.76,
        "lon": -122.42,
        "url": f"https://www.craigslist.org/view/d/a/{1000 + i}",
        "img": "https://images.craigslist.org/a_b_c_600x450.jpg",
        "imgs": ["https://images.craigslist.org/a_b_c_600x450.jpg"],
        "nimg": 1,
        "title": f"alex listing {i}",
        "body": "",
        "region": "SF",
        "drive_min": 11,
    }


def _gio_short_row(i, walk_min=10):
    r = _gio_row(i, dlat=0.001 * i)
    r.update(
        id=f"G{i:02d}",
        aud="gio",
        walk_mi=round(walk_min / 26, 2),
        walk_min=walk_min,
        imgs=[r["img"]],
        body="text me at 415-555-0142",
        beds=0,
    )
    return r


def _run_build(tmp_path, monkeypatch, shortlist, ratings, stats, prev_data=None):
    import json as _json

    tmp_path.mkdir(parents=True, exist_ok=True)
    sl, rt, ps, dj = (tmp_path / n for n in ("s.json", "r.json", "p.json", "data.js"))
    sl.write_text(_json.dumps(shortlist))
    rt.write_text(_json.dumps(ratings))
    ps.write_text(_json.dumps(stats))
    if prev_data is not None:
        dj.write_text("window.HOUSES_DATA = " + _json.dumps(prev_data) + ";\n")
    for attr, p in [
        ("SHORTLIST", sl),
        ("RATINGS", rt),
        ("PULL_STATS", ps),
        ("DATA_JS", dj),
    ]:
        monkeypatch.setattr(refresh, attr, str(p))
    monkeypatch.setattr(refresh, "fetch_reached_urls", lambda: set())
    refresh.do_build()
    return refresh.load_data_js()


def test_build_gio_section_and_alex_invariance(tmp_path, monkeypatch):
    alex = [_alex_row(i) for i in range(1, 17)]
    gio = [_gio_short_row(1, walk_min=5), _gio_short_row(2, walk_min=25)]
    ratings = [_rating(r["id"]) for r in alex + gio]
    stats = {
        "n_kept": 100,
        "n_shortlist": 16,
        "gio_pull_ok": True,
        "n_gio_raw": 40,
        "n_gio": 2,
    }
    d_both = _run_build(tmp_path / "a", monkeypatch, alex + gio, ratings, stats)
    d_alex = _run_build(
        tmp_path / "b",
        monkeypatch,
        alex,
        [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16},
    )
    assert (
        d_both["listings"] == d_alex["listings"]
    )  # Gio rows never perturb Alex's board
    assert "gio" not in d_alex
    g = d_both["gio"]
    assert [x["id"] for x in g["listings"]] == [
        "G01",
        "G02",
    ]  # closer walk -> higher fit
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
        tmp_path,
        monkeypatch,
        alex,
        [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16, "gio_pull_ok": False, "n_gio": 0},
        prev_data=prev,
    )
    assert d["gio"] == prev_gio  # carried forward verbatim


# ---- Gio sweep: prune dead, mass-death guard


def _sweep_fixture(tmp_path, monkeypatch, n_alex=12, gio_urls=()):
    import json as _json

    listings = [
        dict(_alex_row(i), fit=6.0, scores={}, pick=False) for i in range(1, n_alex + 1)
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
        tmp_path,
        monkeypatch,
        gio_urls=[
            "https://g/1",
            "https://g/2dead",
            "https://g/3",
            "https://g/4",
            "https://g/5",
        ],
    )
    refresh.do_sweep()
    d = refresh.load_data_js()
    assert [x["id"] for x in d["gio"]["listings"]] == ["G01", "G03", "G04", "G05"]
    assert d["gio"]["meta"]["n_shown"] == 4
    assert len(d["listings"]) == 12  # alex untouched


def test_sweep_gio_mass_death_guard(tmp_path, monkeypatch):
    dj = _sweep_fixture(
        tmp_path,
        monkeypatch,
        gio_urls=[
            "https://g/1dead",
            "https://g/2dead",
            "https://g/3dead",
            "https://g/4dead",
            "https://g/5",
        ],
    )
    before = dj.read_text()
    refresh.do_sweep()
    assert dj.read_text() == before  # 80% dead at once -> scrape problem, no write


# ---- fit weights: alex_fit refactor must not move anyone's fit; build emits weights


def test_alex_fit_formula():
    scores = {
        "nice": 8,
        "nature": 9,
        "quiet": 9,
        "social": 4,
        "value": 9,
        "aesthetic": 8,
    }
    # apt uses quiet for soft: .16*8+.14*9+.12*9+.12*9+.24*8+.14*8+.08*6 = 8.22
    assert round(min(10.0, refresh.alex_fit(scores, "apt", 8.0, 6.0)), 1) == 8.2
    # room swaps soft to social(4): 8.22 - .12*9 + .12*4 = 7.62
    assert round(min(10.0, refresh.alex_fit(scores, "room", 8.0, 6.0)), 1) == 7.6


def test_gym_score_curve():
    assert refresh.gym_score(0) == 10.0
    assert refresh.gym_score(4) == 10.0  # within a 4-min walk is perfect
    assert refresh.gym_score(14) == 5.0
    assert refresh.gym_score(24) == 0.0
    assert refresh.gym_score(None) == 5.0  # unknown -> neutral, never punitive
    vals = [refresh.gym_score(m) for m in range(0, 30, 2)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_nearest_gym_min():
    gyms = [[refresh.GIO_LAT, refresh.GIO_LON], [37.9, -122.5]]
    assert refresh.nearest_gym_min(refresh.GIO_LAT, refresh.GIO_LON, gyms) == 0
    wm = refresh.nearest_gym_min(refresh.GIO_LAT + 0.01, refresh.GIO_LON, gyms)
    assert wm == refresh.gio_walk_min(refresh.GIO_LAT + 0.01, refresh.GIO_LON)
    assert refresh.nearest_gym_min(37.77, -122.4, []) is None
    assert refresh.nearest_gym_min(None, None, gyms) is None


def test_fetch_gyms_fallback_chain(tmp_path, monkeypatch):
    import json as _json

    snap = tmp_path / "gyms.json"
    monkeypatch.setattr(refresh, "GYMS_JSON", str(snap))

    def boom():
        raise RuntimeError("overpass down")

    # overpass dead + snapshot present -> snapshot
    snap.write_text(_json.dumps([[37.77, -122.4]]))
    monkeypatch.setattr(refresh, "_overpass_gyms", boom)
    assert refresh.fetch_gyms() == [[37.77, -122.4]]
    # overpass dead + no snapshot -> [] (neutral scoring)
    snap.unlink()
    assert refresh.fetch_gyms() == []
    # overpass healthy -> returns AND refreshes the snapshot
    pts = [[37.7 + i * 1e-4, -122.4] for i in range(150)]
    monkeypatch.setattr(refresh, "_overpass_gyms", lambda: pts)
    assert refresh.fetch_gyms() == pts
    assert _json.loads(snap.read_text()) == pts


def test_build_emits_fit_weights_and_pins_fit_values(tmp_path, monkeypatch):
    alex = [_alex_row(i) for i in range(1, 17)]
    d = _run_build(
        tmp_path,
        monkeypatch,
        alex,
        [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16},
    )
    fw = d["meta"]["fit_weights"]
    assert fw["alex"] == refresh.ALEX_W
    assert fw["gio"] == refresh.GIO_W
    assert abs(sum(refresh.ALEX_W.values()) - 1.0) < 1e-9
    assert abs(sum(refresh.GIO_W.values()) - 1.0) < 1e-9
    # hood "mission" (SF): dual = round(.65*cscore(29)+.35*cscore(20), 1) = 8.1;
    # no gym_min on the rows -> gym neutral 5.0. _rating scores -> apt 6.6, room 6.5.
    assert {x["fit"] for x in d["listings"]} == {6.6, 6.5}
    assert all(x["scores"]["gym"] == 5.0 for x in d["listings"])


# ---- multi-source: Rent.com parsing, cross-source dedupe, failure tolerance
#      (spec 2026-07-11-houses-multisource-scraping-design)

_RENT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "rent_next_data_sample.json"
)


def _rent_fixture_listings():
    import json as _json

    d = _json.loads(_RENT_FIXTURE.read_text())
    # the same nesting pull_rent() navigates in the live __NEXT_DATA__ blob
    return d["props"]["pageProps"]["pageData"]["location"]["listingSearch"]["listings"]


def test_parse_rent_listing_normalizes_real_blob():
    ls = _rent_fixture_listings()
    r = refresh.parse_rent_listing(ls[0])  # Lakewood Apartments (real data)
    assert r["pid"] == "lc6069711"
    assert r["src"] == "rent" and r["bucket"] == "apt"
    assert r["price"] == 2998 and r["pdisp"] == "$2,998+"
    assert r["beds"] == 1  # cheapest bedCountData entry is the 1BR
    assert r["hood"] == "san francisco"
    assert abs(r["lat"] - 37.716523) < 1e-6 and abs(r["lon"] - (-122.4982)) < 1e-6
    assert r["url"].startswith("https://www.rent.com/apartment/")
    assert r["imgs"] and all(u.startswith("https://i.rent.com/") for u in r["imgs"])
    assert r["img"] == r["imgs"][0] and r["nimg"] == len(r["imgs"])
    assert "Lakewood" in r["title"] and "John Muir Dr" in r["title"]
    # synthesized body must surface the leasing-office phone to extract_contact
    email, phone = refresh.extract_contact(r["body"])
    assert phone == "(628) 222-1909" and email is None
    assert r["scrape_status"] == "src"  # do_pull must not page-scrape this row


def test_parse_rent_listing_drops_unusable_rows():
    by_id = {x["id"]: x for x in _rent_fixture_listings()}
    assert refresh.parse_rent_listing(by_id["lcNOGEO"]) is None  # map needs coords
    assert (
        refresh.parse_rent_listing(by_id["lcNOPHOTO"]) is None
    )  # montage needs photos
    assert refresh.parse_rent_listing(by_id["lcNOPRICE"]) is None
    # all normal rows parse
    good = [x for k, x in by_id.items() if not k.startswith("lcNO")]
    assert all(refresh.parse_rent_listing(x) is not None for x in good)


def test_parse_rent_listing_city_becomes_hood_with_working_priors():
    by_id = {x["id"]: x for x in _rent_fixture_listings()}
    r = refresh.parse_rent_listing(by_id["lc5932591"])  # Berkeley listing
    assert r["hood"] == "berkeley"
    assert refresh.priors(r["hood"])[0] == "EastBay"


def _cl_anchor(price=2000, lat=37.78, lon=-122.42):
    return {
        "pid": 1,
        "src": "cl",
        "price": price,
        "lat": lat,
        "lon": lon,
        "url": "https://www.craigslist.org/view/d/x/1",
    }


def _rent_near(price=2000, dlat=0.0, dlon=0.0):
    return {
        "pid": "lc1",
        "src": "rent",
        "price": price,
        "lat": 37.78 + dlat,
        "lon": -122.42 + dlon,
        "url": "https://www.rent.com/apartment/x-lc1",
    }


def test_dedupe_cross_source_drops_same_unit_keeps_cl():
    cl = [_cl_anchor()]
    out = refresh.dedupe_cross_source(cl, [_rent_near(price=2040)])  # 2% off, same spot
    assert out == cl  # rent copy dropped, craigslist row kept


def test_dedupe_cross_source_keeps_distinct_rows():
    cl = [_cl_anchor()]
    far = _rent_near(dlat=0.01)  # ~0.7 mi away
    pricey = _rent_near(price=2300)  # same spot, 15% price gap
    out = refresh.dedupe_cross_source(cl, [far, pricey])
    assert len(out) == 3


def test_dedupe_cross_source_coordless_rows_pass_through():
    coordless = dict(_rent_near(), lat=None, lon=None)
    out = refresh.dedupe_cross_source([_cl_anchor()], [coordless])
    assert coordless in out  # select_shortlist drops it later; dedupe never crashes


def test_pull_extra_sources_source_failure_never_raises():
    def boom():
        raise RuntimeError("blocked by WAF")

    def fine():
        return [_rent_near()]

    rows = refresh.pull_extra_sources([("bad", boom), ("rent", fine)])
    assert rows == [_rent_near()]  # bad source warned + skipped, good one kept
    assert refresh.pull_extra_sources([("bad", boom)]) == []


# ---- build: src carried into data.js, per-source meta counts, source string


def _rent_build_row(i):
    return dict(
        _alex_row(i, hood="berkeley"),
        src="rent",
        bucket="apt",
        region="EastBay",
        url=f"https://www.rent.com/apartment/complex-{i}-lc{i}",
        body="Managed apartment complex (via Rent.com). Leasing office: (628) 222-1909.",
    )


def test_build_carries_src_and_per_source_meta(tmp_path, monkeypatch):
    alex = [_alex_row(i) for i in range(1, 17)]
    rent = [_rent_build_row(i) for i in range(17, 20)]
    d = _run_build(
        tmp_path,
        monkeypatch,
        alex + rent,
        [_rating(r["id"]) for r in alex + rent],
        {"n_kept": 100, "n_shortlist": 19},
    )
    by_src = {}
    for x in d["listings"]:
        by_src[x["src"]] = by_src.get(x["src"], 0) + 1
    assert by_src == {"cl": 16, "rent": 3}
    assert d["meta"]["sources"] == by_src
    assert d["meta"]["source"].startswith(
        "Craigslist (live API) + Rent.com, refreshed "
    )
    rent_shown = [x for x in d["listings"] if x["src"] == "rent"]
    assert all(x["contact_phone"] == "(628) 222-1909" for x in rent_shown)


def test_build_craigslist_only_output_unchanged(tmp_path, monkeypatch):
    # zero extra sources reachable -> meta.source string identical to before
    alex = [_alex_row(i) for i in range(1, 17)]  # no src key at all (old shape)
    d = _run_build(
        tmp_path,
        monkeypatch,
        alex,
        [_rating(r["id"]) for r in alex],
        {"n_kept": 100, "n_shortlist": 16},
    )
    import datetime as _dt

    today = _dt.date.today().isoformat()
    assert d["meta"]["source"] == f"Craigslist (live API), refreshed {today}"
    assert d["meta"]["sources"] == {"cl": 16}
    assert all(x["src"] == "cl" for x in d["listings"])


def test_build_stamps_first_seen_new_vs_carried(tmp_path, monkeypatch):
    """first_seen: new URLs stamp today; URLs already on the board carry their
    date forward (None when the previous board predates the field), so the UI
    can flag same-day arrivals without a false 'new' flood on rollout."""
    import datetime as _dt

    alex = [_alex_row(i) for i in range(1, 17)]
    ratings = [_rating(r["id"]) for r in alex]
    stats = {"n_kept": 100, "n_shortlist": 16}
    today = _dt.date.today().isoformat()

    # bootstrap: no previous data.js at all -> everything is genuinely new
    d1 = _run_build(tmp_path / "a", monkeypatch, alex, ratings, stats)
    assert all(x["first_seen"] == today for x in d1["listings"])

    # carried board: known url keeps its date; known url without the field
    # stays None (old, date unknown); unseen urls stamp today
    prev = {
        "meta": {"generated": "2026-07-01"},
        "listings": [
            {"url": alex[0]["url"], "first_seen": "2026-07-01"},
            {"url": alex[1]["url"]},
        ],
    }
    d2 = _run_build(tmp_path / "b", monkeypatch, alex, ratings, stats, prev_data=prev)
    by_url = {x["url"]: x for x in d2["listings"]}
    assert by_url[alex[0]["url"]]["first_seen"] == "2026-07-01"
    assert by_url[alex[1]["url"]]["first_seen"] is None
    fresh = [
        x for x in d2["listings"] if x["url"] not in (alex[0]["url"], alex[1]["url"])
    ]
    assert fresh and all(x["first_seen"] == today for x in fresh)
