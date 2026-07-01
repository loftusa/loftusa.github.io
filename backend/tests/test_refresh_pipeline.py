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
