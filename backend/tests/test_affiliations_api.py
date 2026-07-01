"""Endpoint tests for the affiliation self-service API — ported from the pre-migration
experiments/tests/test_affiliations_api.py. Behavioral assertions preserved; the rate-limit test
patches config.CORRECTION_RATE (read by the router at call time) instead of the old module global.
"""
import json

from backend.app import config
from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-token"}


def post(body):
    return client.post("/affiliations/corrections", json=body)


def set_entry(person="can rager", org="Goodfire", typ="company", **kw):
    return post(
        {
            "type": "aff_entry_set",
            "payload": {"person": person, "org": org, "type": typ, **kw},
        }
    )


def raw_rows():
    r = client.get("/affiliations/corrections", headers=AUTH)
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def test_valid_entry_set_appended():
    r = set_entry(role="Research Engineer", years="2026–", current=True)
    assert r.status_code == 200 and r.json()["ok"]
    rows = raw_rows()
    assert len(rows) == 1 and rows[0]["type"] == "aff_entry_set" and "ts" in rows[0]


def test_unknown_type_rejected():
    assert post({"type": "aff_drop_table", "payload": {}}).status_code == 422


def test_missing_keys_rejected():
    r = post({"type": "aff_entry_set", "payload": {"person": "can rager"}})
    assert r.status_code == 422 and "org" in r.json()["detail"]


def test_bad_entry_type_rejected():
    r = set_entry(typ="cult")
    assert r.status_code == 422 and "type" in r.json()["detail"]


def test_field_caps_rejected():
    assert set_entry(org="x" * 201).status_code == 422
    assert (
        post(
            {"type": "aff_city", "payload": {"person": "can rager", "city": "y" * 81}}
        ).status_code
        == 422
    )


def test_join_entry_count_capped():
    entries = [{"org": f"Org {i}", "type": "company"} for i in range(11)]
    r = post(
        {"type": "aff_join", "payload": {"name": "New Person", "entries": entries}}
    )
    assert r.status_code == 422


def test_join_collision_with_roster_409():
    r = post({"type": "aff_join", "payload": {"name": "Can RAGER", "entries": []}})
    assert r.status_code == 409


def test_join_collision_with_pending_join_409():
    assert (
        post({"type": "aff_join", "payload": {"name": "Brand New"}}).status_code == 200
    )
    assert (
        post({"type": "aff_join", "payload": {"name": "brand  new"}}).status_code == 409
    )


def test_confirm_and_city_roundtrip_via_overlay():
    assert (
        post({"type": "aff_confirm", "payload": {"person": "can rager"}}).status_code
        == 200
    )
    assert (
        post(
            {"type": "aff_city", "payload": {"person": "can rager", "city": "Boston"}}
        ).status_code
        == 200
    )
    ov = client.get("/affiliations/overlay").json()
    assert "can rager" in ov["confirmed"] and ov["city"]["can rager"] == "Boston"


def test_overlay_folds_set_then_remove():
    set_entry(org="Goodfire")
    post(
        {
            "type": "aff_entry_remove",
            "payload": {"person": "can rager", "org": "goodfire"},
        }
    )
    ov = client.get("/affiliations/overlay").json()
    assert "can rager" not in ov["entry_set"]
    assert ov["entry_remove"]["can rager"] == ["goodfire"]


def test_oversized_event_413():
    r = post(
        {
            "type": "aff_city",
            "payload": {"person": "can rager", "city": "Boston", "junk": "A" * 8000},
        }
    )
    assert r.status_code == 413


def test_export_requires_bearer():
    assert client.get("/affiliations/corrections").status_code == 401
    set_entry()
    r = client.get("/affiliations/corrections", headers=AUTH)
    assert r.status_code == 200 and '"aff_entry_set"' in r.text


def test_delete_reverts_durably():
    set_entry()
    ts = raw_rows()[0]["ts"]
    assert client.delete(f"/affiliations/corrections?ts={ts}").status_code == 401
    r = client.delete(f"/affiliations/corrections?ts={ts}", headers=AUTH)
    assert r.status_code == 200 and r.json()["removed"] == 1
    assert client.get("/affiliations/overlay").json()["entry_set"] == {}
    assert (
        client.delete(f"/affiliations/corrections?ts={ts}", headers=AUTH).status_code
        == 404
    )


def test_rate_limit_429(monkeypatch):
    monkeypatch.setattr(config, "CORRECTION_RATE", (2, 600))
    assert set_entry().status_code == 200
    assert set_entry(org="Other Org").status_code == 200
    assert set_entry(org="Third Org").status_code == 429
