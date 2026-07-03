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
